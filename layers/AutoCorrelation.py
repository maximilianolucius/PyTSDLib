import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism for capturing period-based dependencies and aggregating time delays.
    This block can seamlessly replace self-attention mechanisms.
    """

    def __init__(self, mask_flag: bool = True, factor: int = 1, scale: float = None,
                 attention_dropout: float = 0.1, output_attention: bool = False):
        """
        Initializes the AutoCorrelation module.

        Args:
            mask_flag (bool): Whether to apply masking.
            factor (int): Factor to determine top-k selection.
            scale (float, optional): Scaling factor. Defaults to None.
            attention_dropout (float): Dropout rate for attention weights.
            output_attention (bool): Whether to output attention scores.
        """
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _get_top_k(self, corr: torch.Tensor, length: int) -> int:
        """
        Calculates the top-k value based on the factor and sequence length.

        Args:
            corr (torch.Tensor): Correlation tensor.
            length (int): Length of the sequence.

        Returns:
            int: Top-k value.
        """
        return max(1, int(self.factor * math.log(length)))

    def time_delay_agg_training(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """
        Aggregates values based on top-k correlations during training.

        Args:
            values (torch.Tensor): Value tensor of shape [B, C, H, L].
            corr (torch.Tensor): Correlation tensor of shape [B, H, C, L].

        Returns:
            torch.Tensor: Aggregated tensor of shape [B, C, H, L].
        """
        _, _, _, length = values.shape
        top_k = self._get_top_k(corr, length)
        mean_corr = corr.mean(dim=(1, 2))  # Shape: [B, L]
        topk_indices = mean_corr.topk(top_k, dim=-1).indices  # Shape: [B, top_k]
        weights = torch.stack([mean_corr[i, topk_indices[i]] for i in range(top_k)], dim=-1)  # [B, top_k]
        tmp_corr = F.softmax(weights, dim=-1).unsqueeze(1).unsqueeze(2).unsqueeze(3)  # [B, 1, 1, top_k]
        delays_agg = torch.sum(values * tmp_corr, dim=-1)  # [B, C, H, L]
        return delays_agg

    def time_delay_agg_inference(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """
        Aggregates values based on top-k correlations during inference.

        Args:
            values (torch.Tensor): Value tensor of shape [B, C, H, L].
            corr (torch.Tensor): Correlation tensor of shape [B, H, C, L].

        Returns:
            torch.Tensor: Aggregated tensor of shape [B, C, H, L].
        """
        B, H, C, L = corr.shape
        device = corr.device
        top_k = self._get_top_k(corr, L)
        mean_corr = corr.mean(dim=(1, 2))  # Shape: [B, L]
        topk_values, topk_indices = mean_corr.topk(top_k, dim=-1)  # [B, top_k]

        tmp_corr = F.softmax(topk_values, dim=-1).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, top_k]
        delays_agg = torch.zeros_like(values).float()

        for i in range(top_k):
            delay = topk_indices[:, i].unsqueeze(-1).unsqueeze(-1).repeat(1, C, H)
            pattern = torch.gather(values, dim=-1, index=delay.unsqueeze(1).unsqueeze(2).repeat(1, H, C, 1))
            delays_agg += pattern.squeeze(-2) * tmp_corr[:, :, :, i].unsqueeze(-1)

        return delays_agg

    def time_delay_agg_full(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """
        Standard aggregation based on full correlations.

        Args:
            values (torch.Tensor): Value tensor of shape [B, C, H, L].
            corr (torch.Tensor): Correlation tensor of shape [B, H, C, L].

        Returns:
            torch.Tensor: Aggregated tensor of shape [B, C, H, L].
        """
        _, _, _, L = values.shape
        top_k = self._get_top_k(corr, L)
        weights, delay = corr.topk(top_k, dim=-1)  # [B, H, C, top_k]
        tmp_corr = F.softmax(weights, dim=-1).unsqueeze(-1)  # [B, H, C, top_k, 1]

        init_index = torch.arange(L, device=corr.device).view(1, 1, 1, 1, L).repeat(corr.size(0), corr.size(1), corr.size(2), top_k, 1)
        tmp_delay = init_index + delay.unsqueeze(-1)  # [B, H, C, top_k, L]
        pattern = torch.gather(values, dim=-1, index=tmp_delay)  # [B, H, C, top_k, L]
        delays_agg = (pattern * tmp_corr).sum(dim=3)  # [B, H, C, L]
        return delays_agg

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: torch.Tensor = None) -> tuple:
        """
        Forward pass for the AutoCorrelation mechanism.

        Args:
            queries (torch.Tensor): Query tensor of shape [B, L, H, E].
            keys (torch.Tensor): Key tensor of shape [B, S, H, E].
            values (torch.Tensor): Value tensor of shape [B, S, H, D].
            attn_mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            tuple: Tuple containing the aggregated values and attention scores (if enabled).
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        if L > S:
            padding = torch.zeros(B, L - S, H, D, device=values.device)
            values = torch.cat([values, padding], dim=1)
            keys = torch.cat([keys, padding], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # Period-based dependencies using FFT
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1), dim=-1)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), dim=-1)

        # Time delay aggregation
        values = values.permute(0, 2, 3, 1).contiguous()  # [B, H, C, L]
        if self.training:
            V = self.time_delay_agg_training(values, corr)
        else:
            V = self.time_delay_agg_inference(values, corr)

        V = V.permute(0, 3, 1, 2).contiguous()  # [B, L, H, C]

        if self.output_attention:
            return V, corr.permute(0, 2, 1, 3).contiguous()
        return V, None


class AutoCorrelationLayer(nn.Module):
    """
    AutoCorrelation Layer integrating the AutoCorrelation mechanism with projection layers.
    """

    def __init__(self, correlation: nn.Module, d_model: int, n_heads: int,
                 d_keys: int = None, d_values: int = None):
        """
        Initializes the AutoCorrelationLayer.

        Args:
            correlation (nn.Module): AutoCorrelation module instance.
            d_model (int): Dimension of the model.
            n_heads (int): Number of attention heads.
            d_keys (int, optional): Dimension of keys. Defaults to d_model // n_heads.
            d_values (int, optional): Dimension of values. Defaults to d_model // n_heads.
        """
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
                attn_mask: torch.Tensor = None) -> tuple:
        """
        Forward pass for the AutoCorrelationLayer.

        Args:
            queries (torch.Tensor): Query tensor of shape [B, L, D].
            keys (torch.Tensor): Key tensor of shape [B, S, D].
            values (torch.Tensor): Value tensor of shape [B, S, D].
            attn_mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            tuple: Tuple containing the output tensor and attention scores (if enabled).
        """
        B, L, D = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        device = queries.device

        # Project inputs
        queries = self.query_projection(queries).view(B, L, H, -1)  # [B, L, H, d_keys]
        keys = self.key_projection(keys).view(B, S, H, -1)        # [B, S, H, d_keys]
        values = self.value_projection(values).view(B, S, H, -1)  # [B, S, H, d_values]

        # Apply AutoCorrelation
        out, attn = self.inner_correlation(queries, keys, values, attn_mask)

        # Reshape and project output
        out = out.view(B, L, -1)  # [B, L, H * d_values]
        out = self.out_projection(out)  # [B, L, D]

        return out, attn
