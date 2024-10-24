# https://arxiv.org/abs/2308.11200.pdf
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesModel(nn.Module):
    """
    Time Series Forecasting and Analysis Model.

    Supported Tasks:
    - Long-term Forecasting
    - Short-term Forecasting
    - Classification

    Paper Reference: https://arxiv.org/abs/2308.11200.pdf
    """

    def __init__(self, configs: dict):
        """
        Initializes the TimeSeriesModel.

        Args:
            configs (dict): Configuration dictionary containing model parameters.
        """
        super(TimeSeriesModel, self).__init__()

        # Configuration parameters
        self.seq_len = configs.get('seq_len', 96)
        self.enc_in = configs.get('enc_in', 10)
        self.d_model = configs.get('d_model', 512)
        self.dropout_rate = configs.get('dropout', 0.1)
        self.task_name = configs.get('task_name', 'forecast')

        # Determine prediction length based on task
        if self.task_name in ['classification', 'anomaly_detection', 'imputation']:
            self.pred_len = self.seq_len
        else:
            self.pred_len = configs.get('pred_len', 24)

        self.label_len = configs.get('label_len', 48)
        self.seg_len = configs.get('seg_len', 12)
        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        # Embedding layers
        self.value_embedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )

        # Recurrent layer (GRU)
        self.rnn = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=False
        )

        # Positional and Channel Embeddings for Classification
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))

        # Prediction layers
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model, self.seg_len)
        )

        if self.task_name == 'classification':
            self.activation = F.gelu
            self.dropout = nn.Dropout(self.dropout_rate)
            self.projection = nn.Linear(
                self.enc_in * self.seq_len, configs.get('num_class', 5)
            )

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, enc_in].

        Returns:
            torch.Tensor: Encoded tensor of shape [batch_size, enc_in, pred_len].
        """
        batch_size = x.size(0)

        # Normalization: subtract the last time step and permute
        seq_last = x[:, -1:, :].detach()
        x = (x - seq_last).permute(0, 2, 1)  # Shape: [batch_size, enc_in, seq_len]

        # Segment the sequence and apply value embedding
        x = x.reshape(-1, self.seg_num_x, self.seg_len)  # Shape: [batch_size * enc_in, seg_num_x, seg_len]
        x = self.value_embedding(x)  # Shape: [batch_size * enc_in, seg_num_x, d_model]

        # Encode with GRU
        _, hn = self.rnn(x)  # hn shape: [1, batch_size * enc_in, d_model]

        # Create positional and channel embeddings
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),  # Shape: [enc_in, seg_num_y, d_model//2]
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)  # Shape: [enc_in, seg_num_y, d_model//2]
        ], dim=-1).view(-1, 1, self.d_model)  # Shape: [batch_size * enc_in * seg_num_y, 1, d_model]

        # Apply GRU with positional embeddings
        _, hy = self.rnn(
            pos_emb,
            hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)
        )  # hy shape: [1, batch_size * enc_in * seg_num_y, d_model]

        # Predict the output
        y = self.predict(hy)  # Shape: [1, batch_size * enc_in * seg_num_y, seg_len]
        y = y.view(-1, self.enc_in, self.pred_len)  # Shape: [batch_size, enc_in, pred_len]

        # Denormalize and permute to original shape
        y = y.permute(0, 2, 1) + seq_last  # Shape: [batch_size, pred_len, enc_in]
        return y

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Performs forecasting.

        Args:
            x_enc (torch.Tensor): Encoder input tensor of shape [batch_size, seq_len, enc_in].

        Returns:
            torch.Tensor: Forecasted tensor of shape [batch_size, pred_len, enc_in].
        """
        return self.encoder(x_enc)

    def imputation(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Performs data imputation.

        Args:
            x_enc (torch.Tensor): Encoder input tensor of shape [batch_size, seq_len, enc_in].

        Returns:
            torch.Tensor: Imputed tensor of shape [batch_size, pred_len, enc_in].
        """
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Detects anomalies in the input sequence.

        Args:
            x_enc (torch.Tensor): Encoder input tensor of shape [batch_size, seq_len, enc_in].

        Returns:
            torch.Tensor: Anomaly scores tensor of shape [batch_size, pred_len, enc_in].
        """
        return self.encoder(x_enc)

    def classification(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Performs classification based on the encoded sequence.

        Args:
            x_enc (torch.Tensor): Encoder input tensor of shape [batch_size, seq_len, enc_in].

        Returns:
            torch.Tensor: Classification logits of shape [batch_size, num_class].
        """
        enc_out = self.encoder(x_enc)  # Shape: [batch_size, pred_len, enc_in]
        output = enc_out.reshape(enc_out.size(0), -1)  # Shape: [batch_size, pred_len * enc_in]
        output = self.projection(output)  # Shape: [batch_size, num_class]
        return output

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
            x_enc (torch.Tensor): Encoder input tensor of shape [batch_size, seq_len, enc_in].
            x_mark_enc (torch.Tensor, optional): Encoder time features. Defaults to None.
            x_dec (torch.Tensor, optional): Decoder input tensor. Defaults to None.
            x_mark_dec (torch.Tensor, optional): Decoder time features. Defaults to None.
            mask (torch.Tensor, optional): Mask tensor for imputation. Defaults to None.

        Returns:
            torch.Tensor: Output tensor corresponding to the task.
        """
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc)  # Shape: [batch_size, pred_len, enc_in]
            return dec_out[:, -self.pred_len:, :]  # Shape: [batch_size, pred_len, enc_in]

        elif self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)  # Shape: [batch_size, pred_len, enc_in]
            return dec_out

        elif self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)  # Shape: [batch_size, pred_len, enc_in]
            return dec_out

        elif self.task_name == 'classification':
            dec_out = self.classification(x_enc)  # Shape: [batch_size, num_class]
            return dec_out

        else:
            raise ValueError(f"Unsupported task: {self.task_name}")
