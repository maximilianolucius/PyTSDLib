import torch
import torch.nn as nn
from layers.Autoformer_EncDec import SeriesDecomp


class TimeSeriesModel(nn.Module):
    """
    Time Series Prediction Model based on Autoformer Architecture.

    Supports multiple tasks:
    - Long-term Forecasting
    - Short-term Forecasting
    - Imputation
    - Anomaly Detection
    - Classification

    Paper Reference: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs: dict, individual: bool = False):
        """
        Initializes the TimeSeriesModel.

        Args:
            configs (dict): Configuration dictionary containing model parameters.
            individual (bool, optional): If True, uses separate linear layers for each channel.
                                         If False, shares linear layers across channels. Defaults to False.
        """
        super(TimeSeriesModel, self).__init__()

        # Task configuration
        self.task_name = configs.get('task_name', 'forecast')
        self.seq_len = configs.get('seq_len', 24)
        self.pred_len = configs.get('pred_len', configs.get('seq_len', 24)) if self.task_name not in \
                ['classification', 'anomaly_detection', 'imputation'] else configs.get('seq_len', 24)

        self.num_classes = configs.get('num_class', 0)

        # Series decomposition using Autoformer's method
        self.decomposition = SeriesDecomp(configs.get('moving_avg', 25))

        self.individual = individual
        self.enc_in = configs.get('enc_in', 1)  # Number of input channels

        # Initialize linear layers for seasonal and trend components
        self._init_linear_layers()

        # Classification projection layer, if required
        if self.task_name == 'classification':
            self.projection = nn.Linear(self.enc_in * self.pred_len, self.num_classes)


def _init_linear_layers(self):
    """
    Initializes the linear layers for seasonal and trend components.
    Supports both individual and shared configurations.
    """
    if self.individual:
        # Separate linear layers for each channel
        self.linear_seasonal = nn.ModuleList([
            nn.Linear(self.seq_len, self.pred_len) for _ in range(self.enc_in)
        ])
        self.linear_trend = nn.ModuleList([
            nn.Linear(self.seq_len, self.pred_len) for _ in range(self.enc_in)
        ])

        # Initialize weights to average the input
        avg_weight = (1 / self.seq_len) * torch.ones(self.pred_len, self.seq_len)
        for layer in self.linear_seasonal:
            layer.weight = nn.Parameter(avg_weight.clone())
            layer.bias.data.zero_()
        for layer in self.linear_trend:
            layer.weight = nn.Parameter(avg_weight.clone())
            layer.bias.data.zero_()
    else:
        # Shared linear layers across all channels
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

        # Initialize weights to average the input
        avg_weight = (1 / self.seq_len) * torch.ones(self.pred_len, self.seq_len)
        self.linear_seasonal.weight = nn.Parameter(avg_weight)
        self.linear_trend.weight = nn.Parameter(avg_weight)
        self.linear_seasonal.bias.data.zero_()
        self.linear_trend.bias.data.zero_()


def _encode(self, x: torch.Tensor) -> torch.Tensor:
    """
    Encodes the input series by decomposing into seasonal and trend components
    and applying linear transformations.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, channels).

    Returns:
        torch.Tensor: Encoded tensor of shape (batch_size, pred_len, channels).
    """
    # Decompose the input series
    seasonal, trend = self.decomposition(x)  # Both have shape (batch_size, channels, seq_len)

    # Permute to shape (batch_size, channels, seq_len)
    seasonal = seasonal.permute(0, 2, 1)
    trend = trend.permute(0, 2, 1)

    if self.individual:
        # Initialize output tensors
        batch_size, channels, _ = seasonal.size()
        seasonal_out = torch.empty(batch_size, channels, self.pred_len, device=x.device, dtype=x.dtype)
        trend_out = torch.empty(batch_size, channels, self.pred_len, device=x.device, dtype=x.dtype)

        # Apply individual linear layers per channel
        for i in range(channels):
            seasonal_out[:, i, :] = self.linear_seasonal[i](seasonal[:, i, :])
            trend_out[:, i, :] = self.linear_trend[i](trend[:, i, :])
    else:
        # Apply shared linear layers
        seasonal_out = self.linear_seasonal(seasonal)  # Shape: (batch_size, channels, pred_len)
        trend_out = self.linear_trend(trend)  # Shape: (batch_size, channels, pred_len)

    # Combine seasonal and trend components
    encoded = seasonal_out + trend_out  # Shape: (batch_size, channels, pred_len)

    # Permute to (batch_size, pred_len, channels) for consistency
    return encoded.permute(0, 2, 1)


def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
    """
    Performs forecasting by encoding the input series.

    Args:
        x_enc (torch.Tensor): Encoder input tensor.

    Returns:
        torch.Tensor: Forecasted tensor.
    """
    return self._encode(x_enc)


def imputation(self, x_enc: torch.Tensor) -> torch.Tensor:
    """
    Performs data imputation by encoding the input series.

    Args:
        x_enc (torch.Tensor): Encoder input tensor.

    Returns:
        torch.Tensor: Imputed tensor.
    """
    return self._encode(x_enc)


def anomaly_detection(self, x_enc: torch.Tensor) -> torch.Tensor:
    """
    Detects anomalies by encoding the input series.

    Args:
        x_enc (torch.Tensor): Encoder input tensor.

    Returns:
        torch.Tensor: Anomaly scores or related tensor.
    """
    return self._encode(x_enc)


def classification(self, x_enc: torch.Tensor) -> torch.Tensor:
    """
    Performs classification by encoding the input series and projecting to class logits.

    Args:
        x_enc (torch.Tensor): Encoder input tensor.

    Returns:
        torch.Tensor: Classification logits.
    """
    encoded = self._encode(x_enc)  # Shape: (batch_size, pred_len, channels)
    flattened = encoded.view(encoded.size(0), -1)  # Shape: (batch_size, pred_len * channels)
    logits = self.projection(flattened)  # Shape: (batch_size, num_classes)
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
    Forward pass of the model. Routes the input to the appropriate task-specific method.

    Args:
        x_enc (torch.Tensor): Encoder input tensor of shape (batch_size, seq_len, channels).
        x_mark_enc (torch.Tensor, optional): Encoder time features. Defaults to None.
        x_dec (torch.Tensor, optional): Decoder input tensor. Defaults to None.
        x_mark_dec (torch.Tensor, optional): Decoder time features. Defaults to None.
        mask (torch.Tensor, optional): Mask tensor for attention mechanisms. Defaults to None.

    Returns:
        torch.Tensor: Output tensor corresponding to the task.
    """
    if self.task_name in ['long_term_forecast', 'short_term_forecast']:
        encoded = self.forecast(x_enc)  # Shape: (batch_size, pred_len, channels)
        return encoded  # Returning the full prediction; slicing can be done outside if needed

    elif self.task_name == 'imputation':
        return self.imputation(x_enc)  # Shape: (batch_size, pred_len, channels)

    elif self.task_name == 'anomaly_detection':
        return self.anomaly_detection(x_enc)  # Shape: (batch_size, pred_len, channels)

    elif self.task_name == 'classification':
        return self.classification(x_enc)  # Shape: (batch_size, num_classes)

    else:
        raise ValueError(f"Unsupported task: {self.task_name}")
