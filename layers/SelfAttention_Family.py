import math
import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat


class DSAttention(nn.Module):
    """
    De-stationary Attention Mechanism.

    This attention mechanism rescales the pre-softmax scores with learned de-stationary factors.
    """

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        """
        Initializes the DSAttention module.

        Args:
            mask_flag (bool): Whether to apply attention masking.
            factor (int): Scaling factor (unused in DSAttention but kept for compatibility).
            scale (float or None): Scaling factor for the attention scores. If None, defaults to 1/sqrt(E).
            attention_dropout (float): Dropout rate for attention probabilities.
            output_attention (bool): Whether to return attention weights alongside the output.
        """
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        Forward pass for DSAttention.

        Args:
            queries (torch.Tensor): Query tensor of shape (B, L, H, E).
            keys (torch.Tensor): Key tensor of shape (B, S, H, E).
            values (torch.Tensor): Value tensor of shape (B, S, H, D).
            attn_mask (torch.Tensor or None): Attention mask tensor.
            tau (torch.Tensor or None): Learnable scaling factor tensor.
            delta (torch.Tensor or None): Learnable bias tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention.
            torch.Tensor or None: Attention weights if output_attention is True.
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)  # Shape: (B, 1, 1, 1)
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)  # Shape: (B, 1, 1, S)

        # Compute attention scores with de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, float('-inf'))

        # Apply softmax to get attention probabilities
        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        # Compute the weighted sum of values
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    """
    Standard Full Attention Mechanism.

    Computes attention scores over all key-value pairs.
    """

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        """
        Initializes the FullAttention module.

        Args:
            mask_flag (bool): Whether to apply attention masking.
            factor (int): Scaling factor (unused in FullAttention but kept for compatibility).
            scale (float or None): Scaling factor for the attention scores. If None, defaults to 1/sqrt(E).
            attention_dropout (float): Dropout rate for attention probabilities.
            output_attention (bool): Whether to return attention weights alongside the output.
        """
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        Forward pass for FullAttention.

        Args:
            queries (torch.Tensor): Query tensor of shape (B, L, H, E).
            keys (torch.Tensor): Key tensor of shape (B, S, H, E).
            values (torch.Tensor): Value tensor of shape (B, S, H, D).
            attn_mask (torch.Tensor or None): Attention mask tensor.
            tau (torch.Tensor or None): Learnable scaling factor tensor (unused in FullAttention).
            delta (torch.Tensor or None): Learnable bias tensor (unused in FullAttention).

        Returns:
            torch.Tensor: Output tensor after applying attention.
            torch.Tensor or None: Attention weights if output_attention is True.
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        # Compute attention scores
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, float('-inf'))

        # Apply softmax to get attention probabilities
        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        # Compute the weighted sum of values
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    """
    ProbSparse Attention Mechanism.

    An efficient attention mechanism that approximates full attention by focusing on the most relevant queries.
    """

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        """
        Initializes the ProbAttention module.

        Args:
            mask_flag (bool): Whether to apply attention masking.
            factor (int): Scaling factor for the number of sampled keys.
            scale (float or None): Scaling factor for the attention scores. If None, defaults to 1/sqrt(D).
            attention_dropout (float): Dropout rate for attention probabilities.
            output_attention (bool): Whether to return attention weights alongside the output.
        """
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        Sample Q-K interactions to identify top queries.

        Args:
            Q (torch.Tensor): Queries of shape (B, H, L_Q, D).
            K (torch.Tensor): Keys of shape (B, H, L_K, D).
            sample_k (int): Number of keys to sample.
            n_top (int): Number of top queries to select.

        Returns:
            torch.Tensor: Top attention scores.
            torch.Tensor: Indices of top queries.
        """
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Sample K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(
            -2)  # Shape: (B, H, L_Q, sample_k)

        # Compute sparsity measurement
        M = Q_K_sample.max(dim=-1).values - Q_K_sample.mean(dim=-1)
        M_top = M.topk(n_top, dim=-1, sorted=False).indices  # Shape: (B, H, n_top)

        # Gather top queries
        Q_reduce = Q[:, :, M_top, :]  # Shape: (B, H, n_top, D)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # Shape: (B, H, n_top, L_K)

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        """
        Initialize the context tensor.

        Args:
            V (torch.Tensor): Values tensor of shape (B, H, L_V, D).
            L_Q (int): Length of the query sequence.

        Returns:
            torch.Tensor: Initial context tensor.
        """
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # Average pooling over V
            V_mean = V.mean(dim=-2)  # Shape: (B, H, D)
            context = V_mean.unsqueeze(-2).expand(B, H, L_Q, D).clone()
        else:
            # Cumulative sum for masked attention
            assert L_Q == L_V, "For masked attention, L_Q must equal L_V."
            context = V.cumsum(dim=-2)
        return context

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        """
        Update the context tensor with the computed attention.

        Args:
            context_in (torch.Tensor): Current context tensor.
            V (torch.Tensor): Values tensor of shape (B, H, L_V, D).
            scores (torch.Tensor): Attention scores for top queries.
            index (torch.Tensor): Indices of top queries.
            L_Q (int): Length of the query sequence.
            attn_mask (torch.Tensor or None): Attention mask.

        Returns:
            torch.Tensor: Updated context tensor.
            torch.Tensor or None: Updated attention weights if output_attention is True.
        """
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, float('-inf'))

        # Compute attention probabilities
        attn = torch.softmax(scores, dim=-1)

        # Update the context
        context_in.scatter_(
            2, index.unsqueeze(-1).expand_as(torch.matmul(attn, V)),
            torch.matmul(attn, V)
        )

        if self.output_attention:
            # Initialize attention weights with uniform distribution
            attns = torch.ones(B, H, L_Q, L_V, device=attn.device) / L_V
            attns.scatter_(2, index.unsqueeze(-1).expand_as(attn), attn)
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        Forward pass for ProbAttention.

        Args:
            queries (torch.Tensor): Query tensor of shape (B, L_Q, H, D).
            keys (torch.Tensor): Key tensor of shape (B, L_K, H, D).
            values (torch.Tensor): Value tensor of shape (B, L_V, H, D).
            attn_mask (torch.Tensor or None): Attention mask tensor.
            tau (torch.Tensor or None): Learnable scaling factor tensor (unused in ProbAttention).
            delta (torch.Tensor or None): Learnable bias tensor (unused in ProbAttention).

        Returns:
            torch.Tensor: Output context tensor after applying attention.
            torch.Tensor or None: Attention weights if output_attention is True.
        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        # Transpose for multi-head attention
        queries = queries.transpose(2, 1)  # Shape: (B, H, L_Q, D)
        keys = keys.transpose(2, 1)  # Shape: (B, H, L_K, D)
        values = values.transpose(2, 1)  # Shape: (B, H, L_V, D)

        # Determine sampling parameters
        U_part = min(self.factor * math.ceil(math.log(L_K)), L_K)
        u = min(self.factor * math.ceil(math.log(L_Q)), L_Q)

        # Sample Q-K interactions and get top queries
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # Scale the scores
        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # Initialize context
        context = self._get_initial_context(values, L_Q)

        # Update context with top queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    """
    Attention Layer combining query, key, value projections with an attention mechanism.
    """

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        """
        Initializes the AttentionLayer.

        Args:
            attention (nn.Module): Attention mechanism module (e.g., FullAttention, ProbAttention).
            d_model (int): Dimensionality of the input features.
            n_heads (int): Number of attention heads.
            d_keys (int or None): Dimensionality of the keys. Defaults to d_model // n_heads.
            d_values (int or None): Dimensionality of the values. Defaults to d_model // n_heads.
        """
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        Forward pass for AttentionLayer.

        Args:
            queries (torch.Tensor): Query tensor of shape (B, L, d_model).
            keys (torch.Tensor): Key tensor of shape (B, S, d_model).
            values (torch.Tensor): Value tensor of shape (B, S, d_model).
            attn_mask (torch.Tensor or None): Attention mask tensor.
            tau (torch.Tensor or None): Learnable scaling factor tensor.
            delta (torch.Tensor or None): Learnable bias tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention.
            torch.Tensor or None: Attention weights if the inner attention mechanism outputs them.
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project inputs to multi-head dimensions
        queries = self.query_projection(queries).view(B, L, H, -1)  # Shape: (B, L, H, d_keys)
        keys = self.key_projection(keys).view(B, S, H, -1)  # Shape: (B, S, H, d_keys)
        values = self.value_projection(values).view(B, S, H, -1)  # Shape: (B, S, H, d_values)

        # Apply attention mechanism
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )

        # Reshape and project back to d_model
        out = out.view(B, L, -1)  # Shape: (B, L, H * d_values)
        out = self.out_projection(out)  # Shape: (B, L, d_model)

        return out, attn


class ReformerLayer(nn.Module):
    """
    Reformer Layer using Locality-Sensitive Hashing (LSH) based Self-Attention.
    """

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, causal=False, bucket_size=4,
                 n_hashes=4):
        """
        Initializes the ReformerLayer.

        Args:
            attention (nn.Module): Attention mechanism module.
            d_model (int): Dimensionality of the input features.
            n_heads (int): Number of attention heads.
            d_keys (int or None): Dimensionality of the keys.
            d_values (int or None): Dimensionality of the values.
            causal (bool): Whether to apply causal masking.
            bucket_size (int): Size of the buckets for LSH.
            n_hashes (int): Number of hash rounds for LSH.
        """
        super(ReformerLayer, self).__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        """
        Adjusts the length of queries to fit the bucket size requirements.

        Args:
            queries (torch.Tensor): Query tensor of shape (B, N, C).

        Returns:
            torch.Tensor: Adjusted query tensor.
        """
        B, N, C = queries.shape
        required_length = self.bucket_size * 2
        if N % required_length == 0:
            return queries
        else:
            fill_len = required_length - (N % required_length)
            padding = torch.zeros(B, fill_len, C, device=queries.device)
            return torch.cat([queries, padding], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        """
        Forward pass for ReformerLayer.

        Args:
            queries (torch.Tensor): Query tensor of shape (B, N, d_model).
            keys (torch.Tensor): Key tensor of shape (B, N, d_model).
            values (torch.Tensor): Value tensor of shape (B, N, d_model).
            attn_mask (torch.Tensor or None): Attention mask tensor.
            tau (torch.Tensor or None): Learnable scaling factor tensor.
            delta (torch.Tensor or None): Learnable bias tensor.

        Returns:
            torch.Tensor: Output tensor after applying Reformer-based attention.
            None: Reformer does not output attention weights.
        """
        B, N, C = queries.shape
        queries = self.fit_length(queries)[:, :N, :]  # Adjust length and trim back
        out = self.attn(queries)  # Shape: (B, N, d_model)
        return out, None


class TwoStageAttentionLayer(nn.Module):
    """
    Two-Stage Attention (TSA) Layer.

    Combines cross-time and cross-dimension attention mechanisms to capture dependencies in both temporal and feature dimensions.
    """

    def __init__(self, configs, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        """
        Initializes the TwoStageAttentionLayer.

        Args:
            configs (object): Configuration object containing hyperparameters.
            seg_num (int): Number of segments.
            factor (int): Routing factor for the router.
            d_model (int): Dimensionality of the model.
            n_heads (int): Number of attention heads.
            d_ff (int or None): Dimensionality of the feed-forward network. Defaults to 4 * d_model.
            dropout (float): Dropout rate.
        """
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        # Initialize attention mechanisms for time and dimension
        self.time_attention = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
            d_model, n_heads
        )
        self.dim_sender = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
            d_model, n_heads
        )
        self.dim_receiver = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
            d_model, n_heads
        )

        # Learnable router parameters
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        # Dropout layers
        self.dropout = nn.Dropout(dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        # Feed-forward networks
        self.MLP1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        Forward pass for TwoStageAttentionLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, ts_d, seg_num, d_model).
            attn_mask (torch.Tensor or None): Attention mask tensor.
            tau (torch.Tensor or None): Learnable scaling factor tensor.
            delta (torch.Tensor or None): Learnable bias tensor.

        Returns:
            torch.Tensor: Output tensor after applying two-stage attention.
        """
        batch = x.shape[0]

        # ---- Cross Time Stage ----
        # Reshape to combine batch and time dimensions for time attention
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')

        # Apply time attention
        time_out, attn = self.time_attention(time_in, time_in, time_in, attn_mask=None)

        # Residual connection and layer normalization
        dim_in = time_in + self.dropout(time_out)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # ---- Cross Dimension Stage ----
        # Reshape to separate segments and apply dimension attention
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)

        # Expand router parameters for the batch
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)

        # Apply dimension sender attention
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None)

        # Apply dimension receiver attention
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None)

        # Residual connection and layer normalization
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        # Reshape back to original dimensions
        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out

