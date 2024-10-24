import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from einops import rearrange


class Transpose(nn.Module):
    """
    Module to transpose tensor dimensions.
    """
    def __init__(self, *dims, contiguous: bool = False):
        """
        Initializes the Transpose module.

        Args:
            *dims: Dimensions to transpose.
            contiguous (bool, optional): If True, returns a contiguous tensor. Defaults to False.
        """
        super(Transpose, self).__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to transpose tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transposed tensor.
        """
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    """
    Module to flatten and project the encoder output for prediction.
    """
    def __init__(self, n_vars: int, nf: int, target_window: int, head_dropout: float = 0.0):
        """
        Initializes the FlattenHead module.

        Args:
            n_vars (int): Number of variables.
            nf (int): Number of features.
            target_window (int): Length of the prediction window.
            head_dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super(FlattenHead, self).__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to flatten and project the tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, n_vars, d_model, patch_num].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, n_vars, target_window].
        """
        x = self.flatten(x)  # Shape: [batch_size, n_vars, d_model * patch_num]
        x = self.linear(x)   # Shape: [batch_size, n_vars, target_window]
        x = self.dropout(x)
        return x


class TimeSeriesTransformerModel(nn.Module):
    """
    Time Series Transformer Model for Forecasting Tasks.

    Paper Reference: https://arxiv.org/abs/2406.16964
    """
    def __init__(self, configs: dict, patch_len: int = 16, stride: int = 8):
        """
        Initializes the TimeSeriesTransformerModel.

        Args:
            configs (dict): Configuration dictionary containing model parameters.
            patch_len (int, optional): Length of each patch for patch embedding. Defaults to 16.
            stride (int, optional): Stride for patch embedding. Defaults to 8.
        """
        super(TimeSeriesTransformerModel, self).__init__()

        # Configuration parameters
        self.seq_len = configs.get('seq_len', 96)
        self.pred_len = configs.get('pred_len', 24)
        self.patch_size = patch_len
        self.stride = stride
        self.d_model = configs.get('d_model', 512)
        self.factor = configs.get('factor', 5)
        self.n_heads = configs.get('n_heads', 8)
        self.d_ff = configs.get('d_ff', 2048)
        self.activation = configs.get('activation', 'gelu')
        self.dropout = configs.get('dropout', 0.1)

        # Calculate the number of patches
        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 2

        # Layers
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.in_layer = nn.Linear(self.patch_size, self.d_model)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model,
                        self.n_heads
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(1)
            ],
            norm_layer=nn.LayerNorm(self.d_model)
        )

        self.out_layer = nn.Linear(self.d_model * self.patch_num, self.pred_len)

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor = None,
                x_dec: torch.Tensor = None, x_mark_dec: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x_enc (torch.Tensor): Encoder input tensor of shape [batch_size, seq_len, n_vars].
            x_mark_enc (torch.Tensor, optional): Encoder time features. Defaults to None.
            x_dec (torch.Tensor, optional): Decoder input tensor. Defaults to None.
            x_mark_dec (torch.Tensor, optional): Decoder time features. Defaults to None.

        Returns:
            torch.Tensor: Forecasted tensor of shape [batch_size, pred_len, n_vars].
        """
        # Normalize the input
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = (x_enc - means) / torch.sqrt(x_enc.var(dim=1, keepdim=True, unbiased=False) + 1e-5)

        B, _, C = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)  # Shape: [batch_size, n_vars, seq_len]

        # Apply padding and extract patches
        x_enc = self.padding_patch_layer(x_enc)
        x_enc = x_enc.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # Shape: [B, C, patch_num, patch_size]

        # Linear projection of patches
        enc_out = self.in_layer(x_enc)  # Shape: [B, C, patch_num, d_model]

        # Rearrange for Transformer encoder
        enc_out = rearrange(enc_out, 'b c m l -> (b c) m l')  # Shape: [B*C, patch_num, d_model]

        # Encode
        enc_out, _ = self.encoder(enc_out)  # Shape: [B*C, patch_num, d_model]

        # Reshape back to [B, C, d_model, patch_num]
        enc_out = rearrange(enc_out, '(b c) m l -> b c l m', b=B, c=C)  # Shape: [B, C, d_model, patch_num]

        # Flatten and project to prediction
        dec_out = self.out_layer(enc_out.view(B, C, -1))  # Shape: [B, C, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # Shape: [B, pred_len, C]

        # Denormalize the output
        dec_out = dec_out * torch.sqrt(x_enc.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        dec_out = dec_out + means.detach()

        return dec_out
