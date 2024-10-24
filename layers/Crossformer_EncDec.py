import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from layers.SelfAttention_Family import TwoStageAttentionLayer


class SegMerging(nn.Module):
    """
    Segment Merging Module for aggregating segments within a window.

    This module merges multiple segments within a specified window size by concatenating them,
    applying layer normalization, and transforming them through a linear layer.

    Args:
        d_model (int): Dimension of the model.
        win_size (int): Window size for merging segments.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
    """

    def __init__(self, d_model: int, win_size: int, norm_layer: nn.Module = nn.LayerNorm):
        super(SegMerging, self).__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for segment merging.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Time_Steps, Segment_Num, D_Model].

        Returns:
            torch.Tensor: Merged tensor of shape [Batch, Time_Steps, D_Model].
        """
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = (self.win_size - seg_num % self.win_size) % self.win_size

        if pad_num > 0:
            # Repeat the last segment to pad the segment dimension
            padding = x[:, :, -pad_num:, :].repeat(1, 1, 1, 1)
            x = torch.cat((x, padding), dim=2)  # [B, ts_d, seg_num + pad_num, d_model]

        # Reshape to merge segments within each window
        x = rearrange(
            x, 'b ts_d (w seg) d -> b ts_d w (seg d)',
            w=self.win_size
        )  # [B, ts_d, W, seg * d_model]

        # Apply normalization and linear transformation
        x = self.norm(x)
        x = self.linear_trans(x)  # [B, ts_d, W, d_model]

        # Aggregate merged segments by averaging over the window dimension
        x = x.mean(dim=2)  # [B, ts_d, d_model]

        return x


class ScaleBlock(nn.Module):
    """
    Scale Block consisting of segment merging and multiple Two-Stage Attention Layers.

    This block optionally merges segments based on the window size and applies a stack of
    TwoStageAttentionLayers for encoding.

    Args:
        configs: Configuration object containing model parameters.
        win_size (int): Window size for merging segments.
        d_model (int): Dimension of the model.
        n_heads (int): Number of attention heads.
        d_ff (int): Dimension of the feed-forward network.
        depth (int): Number of attention layers.
        dropout (float): Dropout probability.
        seg_num (int, optional): Number of segments. Defaults to 10.
        factor (int, optional): Factor for attention. Defaults to 10.
    """

    def __init__(
        self,
        configs,
        win_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        depth: int,
        dropout: float,
        seg_num: int = 10,
        factor: int = 10
    ):
        super(ScaleBlock, self).__init__()

        self.merge_layer = SegMerging(d_model, win_size) if win_size > 1 else None

        self.encode_layers = nn.ModuleList([
            TwoStageAttentionLayer(
                configs=configs,
                seg_num=seg_num,
                factor=factor,
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(depth)
        ])

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        tau: torch.Tensor = None,
        delta: torch.Tensor = None
    ) -> tuple:
        """
        Forward pass for the ScaleBlock.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Time_Steps, Segment_Num, D_Model].
            attn_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            tau (torch.Tensor, optional): Additional parameter for attention. Defaults to None.
            delta (torch.Tensor, optional): Additional parameter for attention. Defaults to None.

        Returns:
            tuple: Tuple containing the encoded tensor and attention weights (if any).
        """
        if self.merge_layer:
            x = self.merge_layer(x)

        for layer in self.encode_layers:
            x = layer(x, attn_mask=attn_mask, tau=tau, delta=delta)

        return x, None  # Assuming TwoStageAttentionLayer returns only x


class Encoder(nn.Module):
    """
    Transformer Encoder composed of multiple ScaleBlocks.

    Args:
        attn_layers (list of nn.Module): List of ScaleBlock instances.
    """

    def __init__(self, attn_layers: list):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList(attn_layers)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass for the Encoder.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Time_Steps, Segment_Num, D_Model].

        Returns:
            tuple: Tuple containing a list of encoded tensors and attention weights (if any).
        """
        encode_x = [x]

        for block in self.encode_blocks:
            x, _ = block(x)
            encode_x.append(x)

        return encode_x, None  # Assuming attention weights are not collected


class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer with Self-Attention, Cross-Attention, and MLP.

    This layer performs self-attention, cross-attention with encoder outputs, followed by a
    feed-forward network (MLP) with residual connections and layer normalization.

    Args:
        self_attention (nn.Module): Self-attention mechanism.
        cross_attention (nn.Module): Cross-attention mechanism.
        seg_len (int): Length of the segment.
        d_model (int): Dimension of the model.
        d_ff (int, optional): Dimension of the feed-forward network. Defaults to 4 * d_model.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """

    def __init__(
        self,
        self_attention: nn.Module,
        cross_attention: nn.Module,
        seg_len: int,
        d_model: int,
        d_ff: int = None,
        dropout: float = 0.1
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor
    ) -> tuple:
        """
        Forward pass of the DecoderLayer.

        Args:
            x (torch.Tensor): Decoder input tensor of shape [Batch, Time_Steps, Segment_Num, D_Model].
            cross (torch.Tensor): Encoder output tensor for cross-attention.

        Returns:
            tuple: Tuple containing the decoded tensor and layer predictions.
        """
        # Self-Attention with Residual Connection
        self_attn_out, _ = self.self_attention(x)
        x = x + self.dropout(self_attn_out)
        x = self.norm1(x)

        # Reshape for Cross-Attention
        batch_size = x.shape[0]
        x = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')

        cross = rearrange(cross, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')

        # Cross-Attention with Residual Connection
        cross_attn_out, _ = self.cross_attention(x, cross, cross)
        x = x + self.dropout(cross_attn_out)

        # MLP with Residual Connection and Layer Normalization
        y = self.norm1(x)
        y = self.mlp(y)
        x = self.norm2(x + self.dropout(y))

        # Reshape back to original dimensions
        x = rearrange(x, '(b ts_d) seg_num d_model -> b ts_d seg_num d_model', b=batch_size)

        # Prediction Layer
        layer_predict = self.linear_pred(x)
        layer_predict = rearrange(layer_predict, 'b ts_d seg_num seg_len -> b (ts_d seg_num) seg_len')

        return x, layer_predict


class Decoder(nn.Module):
    """
    Transformer Decoder composed of multiple DecoderLayers.

    This decoder aggregates predictions from each decoder layer and combines them to produce the final output.

    Args:
        layers (list of nn.Module): List of DecoderLayer instances.
    """

    def __init__(self, layers: list):
        super(Decoder, self).__init__()
        self.decode_layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, cross: list) -> torch.Tensor:
        """
        Forward pass for the Decoder.

        Args:
            x (torch.Tensor): Decoder input tensor of shape [Batch, Time_Steps, Segment_Num, D_Model].
            cross (list of torch.Tensor): List of encoder output tensors for each decoder layer.

        Returns:
            torch.Tensor: Final prediction tensor of shape [Batch, (Time_Steps * Segment_Num), Seg_Len].
        """
        final_predict = 0
        for i, layer in enumerate(self.decode_layers):
            cross_enc = cross[i]
            x, layer_predict = layer(x, cross_enc)
            final_predict += layer_predict

        # Rearrange to combine Time_Steps and Segment_Num dimensions
        final_predict = rearrange(
            final_predict,
            'b (ts_d seg_num) seg_len -> b (seg_num seg_len) ts_d',
            ts_d=x.shape[1]
        )

        return final_predict
