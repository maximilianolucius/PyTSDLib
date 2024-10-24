# https://openreview.net/pdf?id=I55UqU-M11y
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, MyLayerNorm, SeriesDecomp
from layers.Embed import DataEmbedding_wo_pos



class AutoformerModel(nn.Module):
    """
    Autoformer: Efficient and interpretable transformer for time series forecasting with O(LlogL) complexity.
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self, configs):
        super(AutoformerModel, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # Time series decomposition layer
        self.decomp = SeriesDecomp(configs.moving_avg)

        # Embedding layer for input data
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Encoder with AutoCorrelation layers
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=MyLayerNorm(configs.d_model)
        )

        # Decoder for forecasting tasks
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                            configs.d_model, configs.n_heads
                        ),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                            configs.d_model, configs.n_heads
                        ),
                        configs.d_model,
                        configs.c_out,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    ) for _ in range(configs.d_layers)
                ],
                norm_layer=MyLayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out)
            )

        # Task-specific projection layers
        if self.task_name in ['imputation', 'anomaly_detection']:
            self.projection = nn.Linear(configs.d_model, configs.c_out)

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecast task: Uses encoder-decoder structure for long-term or short-term forecasting.
        """
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)

        seasonal_init, trend_init = self.decomp(x_enc)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out)

        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, trend=trend_init)

        return trend_part + seasonal_part

    def imputation(self, x_enc, x_mark_enc):
        """
        Imputation task: Directly projects the encoder output for missing value imputation.
        """
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out)
        return self.projection(enc_out)

    def anomaly_detection(self, x_enc):
        """
        Anomaly detection task: Encodes the input and projects it to detect anomalies.
        """
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out)
        return self.projection(enc_out)

    def classification(self, x_enc, x_mark_enc):
        """
        Classification task: Uses encoder output for classification by applying non-linearity and dropout.
        """
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out)

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)

        return self.projection(output)

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward function: Executes the specific task based on the task name.
        """
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]
        elif self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc)
        elif self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        elif self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
