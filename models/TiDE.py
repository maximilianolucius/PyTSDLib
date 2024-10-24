import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    """Layer Normalization with optional bias."""

    def __init__(self, ndim: int, bias: bool = True):
        """
        Initializes the LayerNorm module.

        Args:
            ndim (int): Number of features.
            bias (bool, optional): If True, includes a bias term. Defaults to True.
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Applies layer normalization.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class ResBlock(nn.Module):
    """Residual Block with two linear layers and layer normalization."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1, bias: bool = True):
        """
        Initializes the ResBlock module.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Dimension of the output features.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            bias (bool, optional): If True, includes bias in linear layers. Defaults to True.
        """
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.fc3 = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = LayerNorm(output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, *, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, *, output_dim].
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + self.fc3(x)
        out = self.ln(out)
        return out


class TiDEModel(nn.Module):
    """
    Time Series Forecasting and Imputation Model based on TiDE Architecture.

    Paper Reference: https://arxiv.org/pdf/2304.08424.pdf
    """

    def __init__(self, configs: dict, bias: bool = True, feature_encode_dim: int = 2):
        """
        Initializes the TiDEModel.

        Args:
            configs (dict): Configuration dictionary containing model parameters.
            bias (bool, optional): If True, includes bias in linear layers. Defaults to True.
            feature_encode_dim (int, optional): Dimension for feature encoding. Defaults to 2.
        """
        super(TiDEModel, self).__init__()
        self.task_name = configs.get('task_name', 'forecast')
        self.seq_len = configs.get('seq_len', 96)
        self.label_len = configs.get('label_len', 48)
        self.pred_len = configs.get('pred_len', 24)
        self.hidden_dim = configs.get('d_model', 512)
        self.res_hidden = configs.get('d_model', 512)
        self.encoder_num = configs.get('e_layers', 2)
        self.decoder_num = configs.get('d_layers', 2)
        self.freq = configs.get('freq', 'h')
        self.feature_encode_dim = feature_encode_dim
        self.decode_dim = configs.get('c_out', 1)
        self.temporalDecoderHidden = configs.get('d_ff', 2048)
        dropout = configs.get('dropout', 0.1)

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        self.feature_dim = freq_map.get(self.freq, 1)

        flatten_dim = self.seq_len + (self.seq_len + self.pred_len) * self.feature_encode_dim

        # Feature Encoder
        self.feature_encoder = ResBlock(
            input_dim=self.feature_dim,
            hidden_dim=self.res_hidden,
            output_dim=self.feature_encode_dim,
            dropout=dropout,
            bias=bias
        )

        # Encoders
        enc_layers = [ResBlock(flatten_dim, self.res_hidden, self.hidden_dim, dropout, bias)]
        enc_layers += [ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias) for _ in range(self.encoder_num - 1)]
        self.encoders = nn.Sequential(*enc_layers)

        # Decoders and Projections based on task
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            decoder_layers = [ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias) for _ in range(self.decoder_num - 1)]
            decoder_layers.append(ResBlock(self.hidden_dim, self.res_hidden, self.decode_dim * self.pred_len, dropout, bias))
            self.decoders = nn.Sequential(*decoder_layers)
            self.temporalDecoder = ResBlock(
                input_dim=self.decode_dim + self.feature_encode_dim,
                hidden_dim=self.temporalDecoderHidden,
                output_dim=1,
                dropout=dropout,
                bias=bias
            )
            self.residual_proj = nn.Linear(self.seq_len, self.pred_len, bias=bias)

        elif self.task_name in ['imputation', 'anomaly_detection']:
            decoder_layers = [ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias) for _ in range(self.decoder_num - 1)]
            decoder_layers.append(ResBlock(self.hidden_dim, self.res_hidden, self.decode_dim * self.seq_len, dropout, bias))
            self.decoders = nn.Sequential(*decoder_layers)
            self.temporalDecoder = ResBlock(
                input_dim=self.decode_dim + self.feature_encode_dim,
                hidden_dim=self.temporalDecoderHidden,
                output_dim=1,
                dropout=dropout,
                bias=bias
            )
            self.residual_proj = nn.Linear(self.seq_len, self.seq_len, bias=bias)

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout_layer = nn.Dropout(dropout)
            self.projection = nn.Linear(
                in_features=self.enc_in * self.seq_len,
                out_features=configs.get('num_class', 5)
            )

    def forecast(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, batch_y_mark: torch.Tensor) -> torch.Tensor:
        """
        Performs forecasting.

        Args:
            x_enc (torch.Tensor): Encoder input tensor of shape [batch_size, seq_len, enc_in].
            x_mark_enc (torch.Tensor): Encoder time features.
            x_dec (torch.Tensor): Decoder input tensor.
            batch_y_mark (torch.Tensor): Batch time features for prediction.

        Returns:
            torch.Tensor: Forecasted tensor of shape [batch_size, pred_len, enc_in].
        """
        # Normalization
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = (x_enc - means) / torch.sqrt(x_enc.var(dim=1, keepdim=True, unbiased=False) + 1e-5)

        # Feature Encoding
        feature = self.feature_encoder(batch_y_mark)

        # Concatenate and encode
        hidden = self.encoders(torch.cat([x_enc, feature.view(feature.size(0), -1)], dim=-1))

        # Decode
        decoded = self.decoders(hidden).view(hidden.size(0), self.pred_len, self.decode_dim)

        # Temporal Decoding and Residual Projection
        dec_out = self.temporalDecoder(torch.cat([feature[:, self.seq_len:], decoded], dim=-1)).squeeze(-1) + self.residual_proj(x_enc)

        # Denormalization
        dec_out = dec_out * stdev[:, 0].unsqueeze(1).repeat(1, self.pred_len)
        dec_out = dec_out + means[:, 0].unsqueeze(1).repeat(1, self.pred_len)
        return dec_out

    def imputation(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, batch_y_mark: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Performs data imputation.

        Args:
            x_enc (torch.Tensor): Encoder input tensor of shape [batch_size, seq_len, enc_in].
            x_mark_enc (torch.Tensor): Encoder time features.
            x_dec (torch.Tensor): Decoder input tensor.
            batch_y_mark (torch.Tensor): Batch time features for imputation.
            mask (torch.Tensor): Mask tensor indicating missing values.

        Returns:
            torch.Tensor: Imputed tensor of shape [batch_size, seq_len, enc_in].
        """
        # Normalization
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = (x_enc - means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc = x_enc.masked_fill(mask == 0, 0)

        # Feature Encoding
        feature = self.feature_encoder(x_mark_enc)

        # Concatenate and encode
        hidden = self.encoders(torch.cat([x_enc, feature.view(feature.size(0), -1)], dim=-1))

        # Decode
        decoded = self.decoders(hidden).view(hidden.size(0), self.seq_len, self.decode_dim)

        # Temporal Decoding and Residual Projection
        dec_out = self.temporalDecoder(torch.cat([feature[:, :self.seq_len], decoded], dim=-1)).squeeze(-1) + self.residual_proj(x_enc)

        # Denormalization
        dec_out = dec_out * stdev[:, 0].unsqueeze(1).repeat(1, self.seq_len)
        dec_out = dec_out + means[:, 0].unsqueeze(1).repeat(1, self.seq_len)
        return dec_out

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, batch_y_mark: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the TiDEModel.

        Args:
            x_enc (torch.Tensor): Encoder input tensor of shape [batch_size, seq_len, enc_in].
            x_mark_enc (torch.Tensor): Encoder time features.
            x_dec (torch.Tensor): Decoder input tensor.
            batch_y_mark (torch.Tensor): Batch time features for prediction or imputation.
            mask (torch.Tensor, optional): Mask tensor for imputation. Defaults to None.

        Returns:
            torch.Tensor: Output tensor corresponding to the task.
        """
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            if batch_y_mark is None:
                batch_y_mark = torch.zeros(
                    (x_enc.size(0), self.seq_len + self.pred_len, self.feature_dim),
                    device=x_enc.device
                ).detach()
            else:
                batch_y_mark = torch.cat([x_mark_enc, batch_y_mark[:, -self.pred_len:, :]], dim=1)
            dec_out = torch.stack([
                self.forecast(x_enc[:, :, feature], x_mark_enc, x_dec, batch_y_mark)
                for feature in range(x_enc.size(-1))
            ], dim=-1)
            return dec_out  # Shape: [batch_size, pred_len, enc_in]

        if self.task_name == 'imputation':
            dec_out = torch.stack([
                self.imputation(x_enc[:, :, feature], x_mark_enc, x_dec, batch_y_mark, mask)
                for feature in range(x_enc.size(-1))
            ], dim=-1)
            return dec_out  # Shape: [batch_size, seq_len, enc_in]

        if self.task_name in ['anomaly_detection', 'classification']:
            raise NotImplementedError(f"Task '{self.task_name}' is temporarily not supported.")

        raise ValueError(f"Unsupported task: {self.task_name}")
