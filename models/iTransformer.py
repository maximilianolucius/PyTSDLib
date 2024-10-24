# https://arxiv.org/abs/2310.06625
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class TimeSeriesTransformer(nn.Module):
    """
    Time Series Transformer Model for Various Tasks.

    Supported Tasks:
    - Long-term Forecasting
    - Short-term Forecasting
    - Imputation
    - Anomaly Detection
    - Classification

    Paper Reference: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs: dict):
        """
        Initializes the TimeSeriesTransformer model.

        Args:
            configs (dict): Configuration dictionary containing model parameters.
        """
        super(TimeSeriesTransformer, self).__init__()

        # Task configuration
        self.task_name = configs.get('task_name', 'forecast')
        self.seq_len = configs.get('seq_len', 24)
        self.pred_len = configs.get('pred_len', 12)
        self.d_model = configs.get('d_model', 512)
        self.embed = configs.get('embed', 'fixed')
        self.freq = configs.get('freq', 'h')
        self.dropout = configs.get('dropout', 0.1)
        self.factor = configs.get('factor', 5)
        self.n_heads = configs.get('n_heads', 8)
        self.e_layers = configs.get('e_layers', 3)
        self.d_ff = configs.get('d_ff', 2048)
        self.activation = configs.get('activation', 'gelu')
        self.enc_in = configs.get('enc_in', 10)
        self.num_class = configs.get('num_class', 5)

        # Embedding layer
        self.enc_embedding = DataEmbedding_inverted(
            self.seq_len, self.d_model, self.embed, self.freq, self.dropout
        )

        # Encoder setup
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
                ) for _ in range(self.e_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model)
        )

        # Projection layers based on task
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.projection = nn.Linear(self.d_model, self.pred_len, bias=True)
        elif self.task_name in ['imputation', 'anomaly_detection']:
            self.projection = nn.Linear(self.d_model, self.seq_len, bias=True)
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout_layer = nn.Dropout(self.dropout)
            self.projection = nn.Linear(self.d_model * self.enc_in, self.num_class)

    def _normalize(self, x: torch.Tensor) -> tuple:
        """
        Normalizes the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, channels).

        Returns:
            tuple: Normalized tensor, means, and standard deviations.
        """
        means = x.mean(dim=1, keepdim=True).detach()
        x_norm = x - means
        stdev = torch.sqrt(x_norm.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm /= stdev
        return x_norm, means, stdev

    def _denormalize(self, x: torch.Tensor, means: torch.Tensor, stdev: torch.Tensor, pred_len: int) -> torch.Tensor:
        """
        Denormalizes the output tensor.

        Args:
            x (torch.Tensor): Output tensor of shape (batch_size, pred_len, channels).
            means (torch.Tensor): Means used for normalization.
            stdev (torch.Tensor): Standard deviations used for normalization.
            pred_len (int): Prediction length.

        Returns:
            torch.Tensor: Denormalized tensor.
        """
        stdev_expanded = stdev[:, 0, :].unsqueeze(1).repeat(1, pred_len, 1)
        means_expanded = means[:, 0, :].unsqueeze(1).repeat(1, pred_len, 1)
        x = x * stdev_expanded + means_expanded
        return x

    def _encode(self, x: torch.Tensor, x_mark: torch.Tensor = None) -> torch.Tensor:
        """
        Encodes the input tensor using the Transformer encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, channels).
            x_mark (torch.Tensor, optional): Time features. Defaults to None.

        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, seq_len, d_model).
        """
        enc_out = self.enc_embedding(x, x_mark)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return enc_out

    def forecast(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, x_mark_dec: torch.Tensor) -> torch.Tensor:
        """
        Performs forecasting.

        Args:
            x_enc (torch.Tensor): Encoder input tensor.
            x_mark_enc (torch.Tensor): Encoder time features.
            x_dec (torch.Tensor): Decoder input tensor.
            x_mark_dec (torch.Tensor): Decoder time features.

        Returns:
            torch.Tensor: Forecasted tensor.
        """
        x_norm, means, stdev = self._normalize(x_enc)
        enc_out = self._encode(x_norm, x_mark_enc)
        dec_out = self.projection(enc_out)
        dec_out = dec_out.permute(0, 2, 1)[:, :, :self.enc_in]
        dec_out = self._denormalize(dec_out, means, stdev, self.pred_len)
        return dec_out

    def imputation(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, x_mark_dec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Performs data imputation.

        Args:
            x_enc (torch.Tensor): Encoder input tensor.
            x_mark_enc (torch.Tensor): Encoder time features.
            x_dec (torch.Tensor): Decoder input tensor.
            x_mark_dec (torch.Tensor): Decoder time features.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Imputed tensor.
        """
        x_norm, means, stdev = self._normalize(x_enc)
        enc_out = self._encode(x_norm, x_mark_enc)
        dec_out = self.projection(enc_out)
        dec_out = dec_out.permute(0, 2, 1)[:, :, :self.enc_in]
        dec_out = self._denormalize(dec_out, means, stdev, self.seq_len)
        return dec_out

    def anomaly_detection(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Detects anomalies in the input tensor.

        Args:
            x_enc (torch.Tensor): Encoder input tensor.

        Returns:
            torch.Tensor: Anomaly scores or related tensor.
        """
        x_norm, means, stdev = self._normalize(x_enc)
        enc_out = self._encode(x_norm, None)
        dec_out = self.projection(enc_out)
        dec_out = dec_out.permute(0, 2, 1)[:, :, :self.enc_in]
        dec_out = self._denormalize(dec_out, means, stdev, self.seq_len)
        return dec_out

    def classification(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor) -> torch.Tensor:
        """
        Performs classification.

        Args:
            x_enc (torch.Tensor): Encoder input tensor.
            x_mark_enc (torch.Tensor): Encoder time features.

        Returns:
            torch.Tensor: Classification logits.
        """
        enc_out = self._encode(x_enc, x_mark_enc)
        activated = self.act(enc_out)
        dropped = self.dropout_layer(activated)
        flattened = dropped.view(dropped.size(0), -1)
        logits = self.projection(flattened)
        return logits

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor = None,
        x_dec: torch.Tensor = None,
        x_mark_dec: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x_enc (torch.Tensor): Encoder input tensor of shape (batch_size, seq_len, channels).
            x_mark_enc (torch.Tensor, optional): Encoder time features. Defaults to None.
            x_dec (torch.Tensor, optional): Decoder input tensor. Defaults to None.
            x_mark_dec (torch.Tensor, optional): Decoder time features. Defaults to None.
            mask (torch.Tensor, optional): Mask tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor corresponding to the task.
        """
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # Shape: [B, pred_len, D]

        elif self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # Shape: [B, seq_len, D]

        elif self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # Shape: [B, seq_len, D]

        elif self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # Shape: [B, num_class]

        else:
            raise ValueError(f"Unsupported task: {self.task_name}")
