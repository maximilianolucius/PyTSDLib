import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from layers.Crossformer_EncDec import ScaleBlock, Encoder, Decoder, DecoderLayer
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention, TwoStageAttentionLayer
from models.PatchTST import FlattenHead
from math import ceil

class CrossformerModel(nn.Module):
    """
    Crossformer: A model that utilizes segmentation and attention mechanisms for time series tasks.
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """
    def __init__(self, configs):
        super(CrossformerModel, self).__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = 12  # Segment length
        self.win_size = 2   # Window size for scaling blocks
        self.task_name = configs.task_name

        # Padding to handle segment length mismatch with sequence length
        self.pad_in_len = ceil(configs.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(configs.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (configs.e_layers - 1)))
        self.head_nf = configs.d_model * self.out_seg_num

        # Embedding layers for input values and positional encoding
        self.enc_value_embedding = PatchEmbedding(configs.d_model, self.seg_len, self.seg_len, self.pad_in_len - configs.seq_len, 0)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, configs.enc_in, self.in_seg_num, configs.d_model))
        self.pre_norm = nn.LayerNorm(configs.d_model)

        # Encoder with scaled attention blocks
        self.encoder = Encoder([
            ScaleBlock(configs, 1 if l == 0 else self.win_size, configs.d_model, configs.n_heads, configs.d_ff,
                        1, configs.dropout, self.in_seg_num if l == 0 else ceil(self.in_seg_num / self.win_size ** l),
                        configs.factor) for l in range(configs.e_layers)
        ])

        # Decoder positional embedding
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, configs.enc_in, (self.pad_out_len // self.seg_len), configs.d_model))

        # Decoder with two-stage attention and full attention
        self.decoder = Decoder([
            DecoderLayer(
                TwoStageAttentionLayer(configs, (self.pad_out_len // self.seg_len), configs.factor, configs.d_model, configs.n_heads,
                                       configs.d_ff, configs.dropout),
                AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                               configs.d_model, configs.n_heads),
                self.seg_len, configs.d_model, configs.d_ff, dropout=configs.dropout)
            for l in range(configs.e_layers + 1)
        ])

        # Task-specific heads for different use cases
        if self.task_name in ['imputation', 'anomaly_detection']:
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len, head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecasting task: Encoder-decoder structure for long-term or short-term forecasting.
        """
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)

        enc_out, attns = self.encoder(x_enc)

        # Decoder input
        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc.shape[0])
        dec_out = self.decoder(dec_in, enc_out)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        Imputation task: Directly outputs imputed values from encoder output.
        """
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)

        enc_out, attns = self.encoder(x_enc)

        dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)
        return dec_out

    def anomaly_detection(self, x_enc):
        """
        Anomaly detection task: Encodes input and applies anomaly detection head.
        """
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)

        enc_out, attns = self.encoder(x_enc)

        dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """
        Classification task: Uses encoder output for classification, followed by projection layer.
        """
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)

        enc_out, attns = self.encoder(x_enc)
        output = self.flatten(enc_out[-1].permute(0, 1, 3, 2))
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass: Executes the appropriate task (forecasting, imputation, anomaly detection, or classification).
        """
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]
        elif self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        elif self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        elif self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
