import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from einops import rearrange, reduce, repeat
import math
from scipy.fftpack import next_fast_len


class Transform:
    """
    Applies a series of transformations to input tensors for data augmentation.
    Transformations include scaling, shifting, and jittering.
    """

    def __init__(self, sigma: float):
        """
        Initializes the Transform with a specified sigma for noise scaling.

        Args:
            sigma (float): Standard deviation for noise in transformations.
        """
        self.sigma = sigma

    @torch.no_grad()
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies scaling, shifting, and jittering transformations to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        return self.jitter(self.shift(self.scale(x)))

    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds Gaussian noise to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Jittered tensor.
        """
        noise = torch.randn_like(x) * self.sigma
        return x + noise

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scales the input tensor by a random factor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Scaled tensor.
        """
        scale_factors = torch.randn(x.size(-1), device=x.device) * self.sigma + 1
        return x * scale_factors

    def shift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shifts the input tensor by adding Gaussian noise.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Shifted tensor.
        """
        shift = torch.randn(x.size(-1), device=x.device) * self.sigma
        return x + shift


def conv1d_fft(f: torch.Tensor, g: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Performs 1D convolution using the Fast Fourier Transform (FFT).

    Args:
        f (torch.Tensor): Input tensor.
        g (torch.Tensor): Kernel tensor.
        dim (int, optional): Dimension along which to perform convolution. Defaults to -1.

    Returns:
        torch.Tensor: Convolved tensor.
    """
    N = f.size(dim)
    M = g.size(dim)
    fast_len = next_fast_len(N + M - 1)

    # Compute FFTs
    F_f = fft.rfft(f, n=fast_len, dim=dim)
    F_g = fft.rfft(g, n=fast_len, dim=dim)

    # Element-wise multiplication in frequency domain
    F_fg = F_f * F_g.conj()

    # Inverse FFT to get convolution result
    out = fft.irfft(F_fg, n=fast_len, dim=dim)

    # Circular shift to align the convolution result
    out = out.roll(shifts=-1, dims=dim)

    # Select the valid part of the convolution
    idx = torch.arange(fast_len - N, fast_len, device=out.device)
    out = out.index_select(dim, idx)

    return out


class ExponentialSmoothing(nn.Module):
    """
    Applies exponential smoothing to input values, optionally incorporating auxiliary values.
    """

    def __init__(self, dim: int, nhead: int, dropout: float = 0.1, aux: bool = False):
        """
        Initializes the ExponentialSmoothing module.

        Args:
            dim (int): Dimension of the input features.
            nhead (int): Number of attention heads.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            aux (bool, optional): Whether to use auxiliary values. Defaults to False.
        """
        super().__init__()
        self._smoothing_weight = nn.Parameter(torch.randn(nhead, 1))
        self.v0 = nn.Parameter(torch.randn(1, 1, nhead, dim))
        self.dropout = nn.Dropout(dropout)
        if aux:
            self.aux_dropout = nn.Dropout(dropout)

    @property
    def weight(self) -> torch.Tensor:
        """
        Sigmoid activation applied to the smoothing weight parameter.

        Returns:
            torch.Tensor: Smoothed weight.
        """
        return torch.sigmoid(self._smoothing_weight)

    def forward(self, values: torch.Tensor, aux_values: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for exponential smoothing.

        Args:
            values (torch.Tensor): Input tensor of shape (batch, time, heads, dim).
            aux_values (torch.Tensor, optional): Auxiliary input tensor. Defaults to None.

        Returns:
            torch.Tensor: Smoothed output tensor.
        """
        b, t, h, d = values.shape
        init_weight, weight = self.get_exponential_weight(t)

        # Apply dropout and perform convolution via FFT
        convolved = conv1d_fft(self.dropout(values), weight, dim=1)

        # Initialize with v0 and add convolved values
        output = init_weight * self.v0 + convolved

        if aux_values is not None:
            # Adjust weights for auxiliary values
            aux_weight = weight / (1 - self.weight) * self.weight
            aux_output = conv1d_fft(self.aux_dropout(aux_values), aux_weight, dim=1)
            output += aux_output

        return output

    def get_exponential_weight(self, T: int) -> tuple:
        """
        Generates exponential weights for smoothing.

        Args:
            T (int): Time dimension length.

        Returns:
            tuple: Initial weights and convolution weights.
        """
        # Create a tensor [0, 1, ..., T-1]
        powers = torch.arange(T, dtype=torch.float, device=self.weight.device)

        # Compute (1 - alpha) * alpha^t for convolution weights
        weight = (1 - self.weight) * (self.weight ** torch.flip(powers, dims=(0,)))

        # Compute alpha^t for initial weights
        init_weight = self.weight ** (powers + 1)

        # Reshape weights for convolution
        init_weight = rearrange(init_weight, 'h t -> 1 t h 1')
        weight = rearrange(weight, 'h t -> 1 t h 1')

        return init_weight, weight


class Feedforward(nn.Module):
    """
    Implements a feedforward neural network with customizable activation.
    """

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1, activation: str = 'relu'):
        """
        Initializes the Feedforward module.

        Args:
            d_model (int): Input feature dimension.
            dim_feedforward (int): Hidden layer dimension.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            activation (str, optional): Activation function name. Defaults to 'relu'.
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = getattr(F, activation, F.relu)  # Default to ReLU if activation not found

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feedforward network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after feedforward operations.
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class GrowthLayer(nn.Module):
    """
    Captures growth trends in the input data using exponential smoothing.
    """

    def __init__(self, d_model: int, nhead: int, d_head: int = None, dropout: float = 0.1):
        """
        Initializes the GrowthLayer.

        Args:
            d_model (int): Input feature dimension.
            nhead (int): Number of attention heads.
            d_head (int, optional): Dimension per head. Defaults to d_model // nhead.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.d_head = d_head or (d_model // nhead)
        self.d_model = d_model
        self.nhead = nhead

        self.z0 = nn.Parameter(torch.randn(self.nhead, self.d_head))
        self.in_proj = nn.Linear(self.d_model, self.d_head * self.nhead)
        self.es = ExponentialSmoothing(self.d_head, self.nhead, dropout=dropout)
        self.out_proj = nn.Linear(self.d_head * self.nhead, self.d_model)

        assert self.d_head * self.nhead == self.d_model, "d_model must be divisible by nhead"

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GrowthLayer.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor with captured growth trends.
        """
        b, t, d = inputs.shape
        # Project inputs and reshape for multi-head processing
        values = self.in_proj(inputs).view(b, t, self.nhead, self.d_head)

        # Initialize with z0 and compute differences
        z0_expanded = repeat(self.z0, 'h d -> b 1 h d', b=b)
        values = torch.cat([z0_expanded, values], dim=1)
        values = values[:, 1:] - values[:, :-1]

        # Apply exponential smoothing
        smoothed = self.es(values)

        # Concatenate the initial value v0
        v0_expanded = repeat(self.es.v0, '1 1 h d -> b 1 h d', b=b)
        smoothed = torch.cat([v0_expanded, smoothed], dim=1)

        # Reshape and project back to original dimension
        smoothed = rearrange(smoothed, 'b t h d -> b t (h d)')
        return self.out_proj(smoothed)


class FourierLayer(nn.Module):
    """
    Captures seasonal patterns in the input data using Fourier transforms.
    """

    def __init__(self, d_model: int, pred_len: int, k: int = None, low_freq: int = 1):
        """
        Initializes the FourierLayer.

        Args:
            d_model (int): Input feature dimension.
            pred_len (int): Prediction horizon length.
            k (int, optional): Number of top frequencies to keep. Defaults to None.
            low_freq (int, optional): Starting frequency index. Defaults to 1.
        """
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.k = k
        self.low_freq = low_freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FourierLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, dim).

        Returns:
            torch.Tensor: Extrapolated tensor capturing seasonal patterns.
        """
        b, t, d = x.shape
        # Compute real FFT along the time dimension
        x_freq = fft.rfft(x, dim=1)

        # Select frequencies excluding the first low_freq frequencies
        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = fft.rfftfreq(t)[self.low_freq:]

        # Keep top-k frequencies based on amplitude
        x_freq, index_tuple = self.topk_freq(x_freq)
        f = f[index_tuple[1]]  # Select frequencies corresponding to top-k indices
        f = repeat(f, 'f -> b f d', b=b, d=d)
        f = rearrange(f, 'b f d -> b f () d').to(x_freq.device)

        # Extrapolate using the selected frequencies
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq: torch.Tensor, f: torch.Tensor, t: int) -> torch.Tensor:
        """
        Extrapolates the time series data based on frequency components.

        Args:
            x_freq (torch.Tensor): Selected frequency components.
            f (torch.Tensor): Corresponding frequencies.
            t (int): Original time dimension length.

        Returns:
            torch.Tensor: Extrapolated time series.
        """
        # Mirror the frequency components for inverse FFT
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)

        # Create time indices for extrapolation
        t_total = t + self.pred_len
        t_val = torch.arange(t_total, dtype=torch.float, device=x_freq.device).view(1, 1, t_total, 1)

        # Compute amplitude and phase
        amp = rearrange(x_freq.abs() / t, 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')

        # Reconstruct the time series using cosine with the given phase and frequency
        x_time = amp * torch.cos(2 * math.pi * f * t_val + phase)

        # Sum across frequency components to get the final time series
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq: torch.Tensor) -> tuple:
        """
        Selects the top-k frequency components based on amplitude.

        Args:
            x_freq (torch.Tensor): Frequency components.

        Returns:
            tuple: Selected frequency components and their indices.
        """
        if self.k is None:
            self.k = x_freq.size(1)  # If k is not specified, keep all frequencies

        # Compute top-k indices based on amplitude
        values, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)

        # Generate indices for advanced indexing
        batch_indices = torch.arange(x_freq.size(0), device=x_freq.device).unsqueeze(1).expand(-1, self.k)
        feature_indices = torch.arange(x_freq.size(2), device=x_freq.device).unsqueeze(0).expand(x_freq.size(0), -1)
        index_tuple = (batch_indices, indices, feature_indices)

        # Select top-k frequency components
        x_freq_selected = x_freq[index_tuple]

        return x_freq_selected, index_tuple


class LevelLayer(nn.Module):
    """
    Models the level component of a time series, incorporating growth and seasonal effects.
    """

    def __init__(self, d_model: int, c_out: int, dropout: float = 0.1):
        """
        Initializes the LevelLayer.

        Args:
            d_model (int): Input feature dimension.
            c_out (int): Number of output channels.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.d_model = d_model
        self.c_out = c_out

        self.es = ExponentialSmoothing(dim=1, nhead=self.c_out, dropout=dropout, aux=True)
        self.growth_pred = nn.Linear(self.d_model, self.c_out)
        self.season_pred = nn.Linear(self.d_model, self.c_out)

    def forward(self, level: torch.Tensor, growth: torch.Tensor, season: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LevelLayer.

        Args:
            level (torch.Tensor): Current level tensor.
            growth (torch.Tensor): Growth component tensor.
            season (torch.Tensor): Seasonal component tensor.

        Returns:
            torch.Tensor: Updated level tensor.
        """
        b, t, _ = level.shape

        # Predict growth and seasonal components
        growth = self.growth_pred(growth).view(b, t, self.c_out, 1)
        season = self.season_pred(season).view(b, t, self.c_out, 1)

        # Reshape tensors for exponential smoothing
        growth = growth.view(b, t, self.c_out, 1)
        season = season.view(b, t, self.c_out, 1)
        level = level.view(b, t, self.c_out, 1)

        # Apply exponential smoothing to update the level
        out = self.es(level - season, aux_values=growth)

        # Reshape back to original dimensions
        out = rearrange(out, 'b t h d -> b t (h d)')
        return out


class EncoderLayer(nn.Module):
    """
    Represents a single layer in the Encoder, combining growth, seasonal, and level components.
    """

    def __init__(
            self,
            d_model: int,
            nhead: int,
            c_out: int,
            seq_len: int,
            pred_len: int,
            k: int,
            dim_feedforward: int = None,
            dropout: float = 0.1,
            activation: str = 'relu',
            layer_norm_eps: float = 1e-5
    ):
        """
        Initializes the EncoderLayer.

        Args:
            d_model (int): Input feature dimension.
            nhead (int): Number of attention heads.
            c_out (int): Number of output channels.
            seq_len (int): Length of the input sequence.
            pred_len (int): Length of the prediction horizon.
            k (int): Number of top frequencies to keep in FourierLayer.
            dim_feedforward (int, optional): Dimension of the feedforward network. Defaults to 4 * d_model.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            activation (str, optional): Activation function name. Defaults to 'relu'.
            layer_norm_eps (float, optional): Epsilon value for layer normalization. Defaults to 1e-5.
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.c_out = c_out
        self.seq_len = seq_len
        self.pred_len = pred_len
        dim_feedforward = dim_feedforward or 4 * d_model

        # Initialize sub-layers
        self.growth_layer = GrowthLayer(d_model, nhead, dropout=dropout)
        self.seasonal_layer = FourierLayer(d_model, pred_len, k=k)
        self.level_layer = LevelLayer(d_model, c_out, dropout=dropout)

        self.ff = Feedforward(d_model, dim_feedforward, dropout=dropout, activation=activation)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, res: torch.Tensor, level: torch.Tensor, attn_mask: torch.Tensor = None) -> tuple:
        """
        Forward pass for the EncoderLayer.

        Args:
            res (torch.Tensor): Residual tensor from the previous layer.
            level (torch.Tensor): Level tensor from the LevelLayer.
            attn_mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            tuple: Updated residual, level, growth, and seasonal tensors.
        """
        # Capture seasonal patterns
        season = self._season_block(res)
        res = res - season[:, :-self.pred_len]

        # Capture growth trends
        growth = self._growth_block(res)
        res = self.norm1(res - growth[:, 1:])

        # Apply feedforward network and layer normalization
        res = self.norm2(res + self.ff(res))

        # Update the level with the latest growth and seasonal components
        level = self.level_layer(level, growth[:, :-1], season[:, :-self.pred_len])
        return res, level, growth, season

    def _growth_block(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input tensor through the growth layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Growth component tensor after dropout.
        """
        x = self.growth_layer(x)
        return self.dropout1(x)

    def _season_block(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input tensor through the seasonal (Fourier) layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Seasonal component tensor after dropout.
        """
        x = self.seasonal_layer(x)
        return self.dropout2(x)


class Encoder(nn.Module):
    """
    Stacks multiple EncoderLayer modules to form the Encoder part of the model.
    """

    def __init__(self, layers: list):
        """
        Initializes the Encoder with a list of EncoderLayer modules.

        Args:
            layers (list): List of EncoderLayer instances.
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, res: torch.Tensor, level: torch.Tensor, attn_mask: torch.Tensor = None) -> tuple:
        """
        Forward pass for the Encoder.

        Args:
            res (torch.Tensor): Residual tensor from the input.
            level (torch.Tensor): Level tensor from the LevelLayer.
            attn_mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            tuple: Final level tensor, list of growth tensors, and list of seasonal tensors.
        """
        growths = []
        seasons = []
        for layer in self.layers:
            res, level, growth, season = layer(res, level, attn_mask=attn_mask)
            growths.append(growth)
            seasons.append(season)

        return level, growths, seasons


class DampingLayer(nn.Module):
    """
    Applies damping to the growth component to control its influence over the prediction horizon.
    """

    def __init__(self, pred_len: int, nhead: int, dropout: float = 0.1):
        """
        Initializes the DampingLayer.

        Args:
            pred_len (int): Length of the prediction horizon.
            nhead (int): Number of attention heads.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.pred_len = pred_len
        self.nhead = nhead
        self._damping_factor = nn.Parameter(torch.randn(1, nhead))
        self.dropout = nn.Dropout(dropout)

    @property
    def damping_factor(self) -> torch.Tensor:
        """
        Sigmoid activation applied to the damping factor parameter.

        Returns:
            torch.Tensor: Damping factors.
        """
        return torch.sigmoid(self._damping_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DampingLayer.

        Args:
            x (torch.Tensor): Growth tensor of shape (batch, time, heads, dim).

        Returns:
            torch.Tensor: Damped growth tensor for the prediction horizon.
        """
        # Repeat the last growth state across the prediction horizon
        x = repeat(x, 'b 1 d -> b t d', t=self.pred_len)
        b, t, d = x.shape

        # Compute damping factors for each time step in the prediction horizon
        powers = torch.arange(1, self.pred_len + 1, device=self.damping_factor.device).float().view(t, 1)
        damping_factors = self.damping_factor ** powers
        damping_factors = damping_factors.cumsum(dim=0)

        # Reshape and apply damping
        x = x.view(b, t, self.nhead, -1)
        x = self.dropout(x) * damping_factors.unsqueeze(-1)
        return x.view(b, t, d)


class DecoderLayer(nn.Module):
    """
    Represents a single layer in the Decoder, applying damping to growth and handling seasonal components.
    """

    def __init__(self, d_model: int, nhead: int, c_out: int, pred_len: int, dropout: float = 0.1):
        """
        Initializes the DecoderLayer.

        Args:
            d_model (int): Input feature dimension.
            nhead (int): Number of attention heads.
            c_out (int): Number of output channels.
            pred_len (int): Length of the prediction horizon.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.c_out = c_out
        self.pred_len = pred_len

        self.growth_damping = DampingLayer(pred_len, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, growth: torch.Tensor, season: torch.Tensor) -> tuple:
        """
        Forward pass for the DecoderLayer.

        Args:
            growth (torch.Tensor): List of growth tensors from each EncoderLayer.
            season (torch.Tensor): List of seasonal tensors from each EncoderLayer.

        Returns:
            tuple: Growth and seasonal horizons for prediction.
        """
        # Apply damping to the latest growth component
        growth_horizon = self.growth_damping(growth[:, -1:])
        growth_horizon = self.dropout1(growth_horizon)

        # Select the latest seasonal components for the prediction horizon
        seasonal_horizon = season[:, -self.pred_len:]
        return growth_horizon, seasonal_horizon


class Decoder(nn.Module):
    """
    Stacks multiple DecoderLayer modules to form the Decoder part of the model.
    """

    def __init__(self, layers: list):
        """
        Initializes the Decoder with a list of DecoderLayer modules.

        Args:
            layers (list): List of DecoderLayer instances.
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)
        if len(layers) == 0:
            raise ValueError("Decoder must contain at least one DecoderLayer.")

        # Assuming all layers have the same configuration
        self.d_model = layers[0].d_model
        self.c_out = layers[0].c_out
        self.pred_len = layers[0].pred_len
        self.nhead = layers[0].nhead

        # Prediction heads for growth and seasonal components
        self.pred = nn.Linear(self.d_model, self.c_out)

    def forward(self, growths: list, seasons: list) -> tuple:
        """
        Forward pass for the Decoder.

        Args:
            growths (list): List of growth tensors from each EncoderLayer.
            seasons (list): List of seasonal tensors from each EncoderLayer.

        Returns:
            tuple: Predicted growth and seasonal components.
        """
        growth_repr = []
        season_repr = []

        for idx, layer in enumerate(self.layers):
            # Apply each DecoderLayer to the corresponding growth and seasonal components
            growth_horizon, season_horizon = layer(growths[idx], seasons[idx])
            growth_repr.append(growth_horizon)
            season_repr.append(season_horizon)

        # Aggregate representations by summing across layers
        growth_repr = sum(growth_repr)
        season_repr = sum(season_repr)

        # Project to output dimensions
        return self.pred(growth_repr), self.pred(season_repr)
