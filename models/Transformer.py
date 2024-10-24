import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


class TransformerModel(nn.Module):
    def __init__(self, configs):
        """
        Initializes the TransformerModel.

        Args:
            configs: Configuration object containing model parameters.
        """
        super(TransformerModel, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model

        # Input Embedding for Encoder
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # Encoder Layers
        encoder_layers = [
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                    configs.d_model,
                    configs.n_heads
                ),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            ) for _ in range(configs.e_layers)
        ]
        self.encoder = Encoder(encoder_layers, norm_layer=nn.LayerNorm(configs.d_model))

        # Task-specific Layers
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # Input Embedding for Decoder
            self.dec_embedding = DataEmbedding(
                configs.dec_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout
            )
            # Decoder Layers
            decoder_layers = [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model,
                        configs.n_heads
                    ),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.d_layers)
            ]
            self.decoder = Decoder(
                decoder_layers,
                norm_layer=nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out)
            )
        elif self.task_name in ['imputation', 'anomaly_detection']:
            self.projection = nn.Linear(configs.d_model, configs.c_out)
        elif self.task_name == 'classification':
            self.activation = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)
        else:
            self.projection = None

    def forecast(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor,
                 x_mark_dec: torch.Tensor) -> torch.Tensor:
        """
        Performs forecasting by passing through encoder and decoder.

        Args:
            x_enc (torch.Tensor): Encoder input of shape [Batch, Seq_Len, Features].
            x_mark_enc (torch.Tensor): Encoder time features.
            x_dec (torch.Tensor): Decoder input of shape [Batch, Dec_Len, Features].
            x_mark_dec (torch.Tensor): Decoder time features.

        Returns:
            torch.Tensor: Forecasted output of shape [Batch, Pred_Len, Features].
        """
        # Encode
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out)

        # Decode
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out)
        return dec_out[:, -self.pred_len:, :]

    def imputation(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, *args) -> torch.Tensor:
        """
        Performs data imputation by passing through encoder and projection.

        Args:
            x_enc (torch.Tensor): Encoder input of shape [Batch, Seq_Len, Features].
            x_mark_enc (torch.Tensor): Encoder time features.
            *args: Additional arguments (unused).

        Returns:
            torch.Tensor: Imputed data of shape [Batch, Seq_Len, Features].
        """
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out)
        return self.projection(enc_out)

    def anomaly_detection(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Performs anomaly detection by passing through encoder and projection.

        Args:
            x_enc (torch.Tensor): Encoder input of shape [Batch, Seq_Len, Features].

        Returns:
            torch.Tensor: Anomaly scores of shape [Batch, Seq_Len, Features].
        """
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out)
        return self.projection(enc_out)

    def classification(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor) -> torch.Tensor:
        """
        Performs classification by passing through encoder, activation, dropout, and projection.

        Args:
            x_enc (torch.Tensor): Encoder input of shape [Batch, Seq_Len, Features].
            x_mark_enc (torch.Tensor): Encoder time features.

        Returns:
            torch.Tensor: Class logits of shape [Batch, Num_Classes].
        """
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out)

        # Apply activation and dropout
        output = self.activation(enc_out)
        output = self.dropout(output)

        # Mask padding if necessary
        if x_mark_enc is not None:
            output = output * x_mark_enc.unsqueeze(-1)

        # Flatten for classification
        output = output.view(output.size(0), -1)
        return self.projection(output)

    def forward(self,
                x_enc: torch.Tensor,
                x_mark_enc: torch.Tensor,
                x_dec: torch.Tensor = None,
                x_mark_dec: torch.Tensor = None,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the TransformerModel.

        Args:
            x_enc (torch.Tensor): Encoder input of shape [Batch, Seq_Len, Features].
            x_mark_enc (torch.Tensor): Encoder time features.
            x_dec (torch.Tensor, optional): Decoder input for forecasting tasks. Defaults to None.
            x_mark_dec (torch.Tensor, optional): Decoder time features for forecasting tasks. Defaults to None.
            mask (torch.Tensor, optional): Mask for imputation tasks. Defaults to None.

        Returns:
            torch.Tensor: Output tensor based on the task.
        """
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        elif self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        elif self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        else:
            raise ValueError(f"Unsupported task: {self.task_name}")
