import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Tuple


class MyLayerNorm(nn.Module):
    """
    Special designed LayerNorm for the seasonal part.
    """

    def __init__(self, channels: int):
        super(MyLayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1, keepdim=True).repeat(1, x.shape[1], 1)
        return x_hat - bias


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series.
    """

    def __init__(self, kernel_size: int, stride: int):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Padding on both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block.
    """

    def __init__(self, kernel_size: int):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class SeriesDecompMulti(nn.Module):
    """
    Multiple Series decomposition block from FEDformer.
    """

    def __init__(self, kernel_sizes: list):
        super(SeriesDecompMulti, self).__init__()
        self.series_decomp = nn.ModuleList([SeriesDecomp(kernel) for kernel in kernel_sizes])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        moving_mean = []
        res = []
        for func in self.series_decomp:
            sea, moving_avg = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with two phases:
    (1) Period-based dependencies discovery
    (2) Time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """

    def __init__(self, mask_flag: bool = True, factor: int = 1, scale: Any = None,
                 attention_dropout: float = 0.1, output_attention: bool = False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        top_k = max(1, int(self.factor * math.log(length)))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1).indices
        weights = torch.stack([mean_value[:, idx] for idx in index], dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()

        for i in range(top_k):
            pattern = torch.roll(tmp_values, shifts=-int(index[i]), dims=-1)
            delays_agg += pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)

        return delays_agg

    def time_delay_agg_inference(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch, head, channel, length = values.shape
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).to(values.device)
        top_k = max(1, int(self.factor * math.log(length)))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()

        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg += pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)

        return delays_agg

    def time_delay_agg_full(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """
        Standard version of Autocorrelation.
        """
        batch, head, channel, length = values.shape
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).to(values.device)
        top_k = max(1, int(self.factor * math.log(length)))
        weights, delay = torch.topk(corr, top_k, dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()

        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg += pattern * tmp_corr[..., i].unsqueeze(-1)

        return delays_agg

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: Any = None) -> Tuple[torch.Tensor, Any]:
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float().to(values.device)
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # Period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # Time delay aggregation
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return V.contiguous(), corr.permute(0, 3, 1, 2)
        else:
            return V.contiguous(), None


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation: nn.Module, d_model: int, n_heads: int,
                 d_keys: int = None, d_values: int = None):
        super(AutoCorrelationLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: Any = None) -> Tuple[torch.Tensor, Any]:
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture.
    """

    def __init__(self, attention: nn.Module, d_model: int, d_ff: int = None,
                 moving_avg: int = 25, dropout: float = 0.1, activation: str = "relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, attn_mask: Any = None) -> Tuple[torch.Tensor, Any]:
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.permute(0, 2, 1))))
        y = self.dropout(self.conv2(y).permute(0, 2, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder.
    """

    def __init__(self, attn_layers: list, conv_layers: list = None, norm_layer: nn.Module = None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, attn_mask: Any = None) -> Tuple[torch.Tensor, list]:
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture.
    """

    def __init__(self, self_attention: nn.Module, cross_attention: nn.Module, d_model: int, c_out: int,
                 d_ff: int = None, moving_avg: int = 25, dropout: float = 0.1, activation: str = "relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.decomp3 = SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1,
                                    padding=1, padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, cross: torch.Tensor,
                x_mask: Any = None, cross_mask: Any = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, _ = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, _ = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.permute(0, 2, 1))))
        y = self.dropout(self.conv2(y).permute(0, 2, 1))
        x, _ = self.decomp3(x + y)

        residual_trend = self.decomp1.moving_avg(x) + self.decomp2.moving_avg(x) + self.decomp3.moving_avg(x)
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer decoder.
    """

    def __init__(self, layers: list, norm_layer: nn.Module = None, projection: nn.Module = None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x: torch.Tensor, cross: torch.Tensor,
                x_mask: Any = None, cross_mask: Any = None,
                trend: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            if trend is not None:
                trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend
