import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    Residual Block with Temporal and Channel Transformations.
    Applies transformations along the temporal and channel dimensions with residual connections.
    """
    def __init__(self, seq_len: int, enc_in: int, d_model: int, dropout: float):
        """
        Initializes the ResBlock.

        Args:
            seq_len (int): Length of the input sequence.
            enc_in (int): Number of input channels/features.
            d_model (int): Dimension of the model.
            dropout (float): Dropout probability.
        """
        super(ResBlock, self).__init__()
        self.temporal = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, seq_len),
            nn.Dropout(dropout)
        )
        self.channel = nn.Sequential(
            nn.Linear(enc_in, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, enc_in),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Length, Channels].

        Returns:
            torch.Tensor: Output tensor of shape [Batch, Length, Channels].
        """
        # Apply temporal transformation with residual connection
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        # Apply channel transformation with residual connection
        x = x + self.channel(x)
        return x


class ForecastModel(nn.Module):
    """
    Forecasting Model using Stacked Residual Blocks.
    Supports long-term and short-term forecasting tasks.
    """
    def __init__(self, configs):
        """
        Initializes the ForecastModel.

        Args:
            configs: Configuration object containing model parameters.
        """
        super(ForecastModel, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers
        self.dropout = configs.dropout

        # Initialize stacked Residual Blocks
        self.res_blocks = nn.Sequential(*[
            ResBlock(seq_len=self.seq_len, enc_in=self.enc_in, d_model=self.d_model, dropout=self.dropout)
            for _ in range(self.e_layers)
        ])

        # Projection layer to map sequence length to prediction length
        self.projection = nn.Linear(self.seq_len, self.pred_len)

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Performs the forecasting operation.

        Args:
            x_enc (torch.Tensor): Encoded input tensor of shape [Batch, Length, Channels].

        Returns:
            torch.Tensor: Forecasted output tensor of shape [Batch, Pred_Length, Channels].
        """
        # Pass through residual blocks
        x = self.res_blocks(x_enc)
        # Apply projection on the temporal dimension
        x = self.projection(x.transpose(1, 2)).transpose(1, 2)
        return x

    def forward(self, x_enc: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass of the ForecastModel.

        Args:
            x_enc (torch.Tensor): Encoded input tensor of shape [Batch, Length, Channels].
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: Forecasted output tensor of shape [Batch, Pred_Length, Channels].

        Raises:
            ValueError: If the task_name is not supported.
        """
        if self.task_name in {'long_term_forecast', 'short_term_forecast'}:
            forecast_output = self.forecast(x_enc)
            # Return the last pred_len time steps
            return forecast_output[:, -self.pred_len:, :]
        else:
            raise ValueError('Only forecast tasks are implemented.')
