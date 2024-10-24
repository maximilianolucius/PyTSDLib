import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Embed import DataEmbedding


def get_mask(input_size, window_size, inner_size):
    """
    Generate the attention mask for PAM-Naive.

    Args:
        input_size (int): Size of the input sequence.
        window_size (list of int): Window sizes for each pyramid layer.
        inner_size (int): Size defining the intra-scale attention window.

    Returns:
        mask (torch.BoolTensor): Attention mask of shape (seq_length, seq_length).
        all_size (list of int): Sizes of each pyramid layer.
    """
    all_size = [input_size]
    for w in window_size:
        layer_size = math.floor(all_size[-1] / w)
        all_size.append(layer_size)

    seq_length = sum(all_size)
    mask = torch.zeros(seq_length, seq_length, dtype=torch.float32)

    inner_window = inner_size // 2

    # Intra-scale masking
    for layer_idx, size in enumerate(all_size):
        start = sum(all_size[:layer_idx])
        end = start + size
        for i in range(start, end):
            left = max(i - inner_window, start)
            right = min(i + inner_window + 1, end)
            mask[i, left:right] = 1

    # Inter-scale masking
    for layer_idx in range(1, len(all_size)):
        current_start = sum(all_size[:layer_idx])
        prev_size = all_size[layer_idx - 1]
        current_size = all_size[layer_idx]
        for i in range(current_start, current_start + current_size):
            relative_idx = i - current_start
            left = current_start - prev_size + relative_idx * window_size[layer_idx - 1]
            right = current_start - prev_size + (relative_idx + 1) * window_size[layer_idx - 1]
            if i == current_start + current_size - 1:
                right = current_start
            mask[i, left:right] = 1
            mask[left:right, i] = 1

    mask = (~mask.bool()).to(torch.bool)

    return mask, all_size


def refer_points(all_sizes, window_size):
    """
    Compute indices to gather features from pyramid sequences in PAM.

    Args:
        all_sizes (list of int): Sizes of each pyramid layer.
        window_size (list of int): Window sizes for each pyramid layer.

    Returns:
        indexes (torch.LongTensor): Indices tensor of shape (1, input_size, num_layers, 1).
    """
    input_size = all_sizes[0]
    num_layers = len(all_sizes)
    indexes = torch.zeros(input_size, num_layers, dtype=torch.long)

    for i in range(input_size):
        indexes[i, 0] = i
        previous_index = i
        for j in range(1, num_layers):
            start = sum(all_sizes[:j])
            prev_start = sum(all_sizes[:j - 1])
            inner_layer_idx = previous_index - prev_start
            gathered_idx = min(inner_layer_idx // window_size[j - 1], all_sizes[j] - 1)
            previous_index = start + gathered_idx
            indexes[i, j] = previous_index

    indexes = indexes.unsqueeze(0).unsqueeze(-1)  # Shape: (1, input_size, num_layers, 1)
    return indexes


class RegularMask:
    """
    Wrapper for attention mask to include batch dimension.
    """

    def __init__(self, mask):
        self._mask = mask.unsqueeze(1)  # Shape: (1, 1, seq_length, seq_length)

    @property
    def mask(self):
        return self._mask


class PositionwiseFeedForward(nn.Module):
    """
    Two-layer position-wise feed-forward network with optional pre-normalization.
    """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super(PositionwiseFeedForward, self).__init__()
        self.normalize_before = normalize_before
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.linear1 = nn.Linear(d_in, d_hid)
        self.linear2 = nn.Linear(d_hid, d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    """
    Encoder layer consisting of self-attention and position-wise feed-forward network.
    """

    def __init__(self, d_model, d_inner, n_head, dropout=0.1, normalize_before=True):
        super(EncoderLayer, self).__init__()
        self.self_attn = AttentionLayer(
            FullAttention(mask_flag=True, factor=0, attention_dropout=dropout, output_attention=False),
            d_model, n_head
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before
        )

    def forward(self, enc_input, attn_mask=None):
        mask = RegularMask(attn_mask)
        enc_output, _ = self.self_attn(enc_input, enc_input, enc_input, attn_mask=mask)
        enc_output = self.feed_forward(enc_output)
        return enc_output


class ConvLayer(nn.Module):
    """
    Convolutional layer with downsampling, batch normalization, and ELU activation.
    """

    def __init__(self, in_channels, kernel_size):
        super(ConvLayer, self).__init__()
        self.down_conv = nn.Conv1d(in_channels=in_channels,
                                   out_channels=in_channels,
                                   kernel_size=kernel_size,
                                   stride=kernel_size)
        self.batch_norm = nn.BatchNorm1d(in_channels)
        self.activation = nn.ELU()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, seq_length).

        Returns:
            torch.Tensor: Output tensor after convolution, normalization, and activation.
        """
        x = self.down_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


class BottleneckConstruct(nn.Module):
    """
    Bottleneck structure with multiple convolutional layers and linear transformations.
    """

    def __init__(self, d_model, window_size, d_inner):
        super(BottleneckConstruct, self).__init__()
        self.down_linear = Linear(d_model, d_inner)
        self.up_linear = Linear(d_inner, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        if isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_inner, ws) for ws in window_size
            ])
        else:
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_inner, window_size) for _ in range(3)
            ])

    def forward(self, enc_input):
        """
        Args:
            enc_input (torch.Tensor): Input tensor of shape (batch, seq_length, d_model).

        Returns:
            torch.Tensor: Output tensor after bottleneck processing.
        """
        # Down project
        x = self.down_linear(enc_input).transpose(1, 2)  # Shape: (batch, d_inner, seq_length)

        # Apply each convolutional layer
        conv_outputs = [conv(x) for conv in self.conv_layers]

        # Concatenate along the channel dimension
        x = torch.cat(conv_outputs, dim=1)  # Shape: (batch, d_inner * num_conv_layers, new_seq_length)
        x = x.transpose(1, 2)  # Shape: (batch, new_seq_length, d_inner * num_conv_layers)

        # Up project
        x = self.up_linear(x)  # Shape: (batch, new_seq_length, d_model)

        # Concatenate with original input
        x = torch.cat([enc_input, x], dim=1)  # Shape: (batch, seq_length + new_seq_length, d_model)

        # Normalize
        x = self.layer_norm(x)
        return x


class Encoder(nn.Module):
    """
    Encoder model with self-attention mechanism and convolutional bottleneck.
    """

    def __init__(self, configs, window_size, inner_size):
        super(Encoder, self).__init__()
        d_bottleneck = configs.d_model // 4

        # Generate attention mask and pyramid sizes
        self.mask, self.all_size = get_mask(configs.seq_len, window_size, inner_size)

        # Compute indices for gathering features from pyramid layers
        self.indexes = refer_points(self.all_size, window_size).to(torch.long)

        # Initialize encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=configs.d_model,
                d_inner=configs.d_ff,
                n_head=configs.n_heads,
                dropout=configs.dropout,
                normalize_before=False
            ) for _ in range(configs.e_layers)
        ])

        # Embedding layer for input data
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.dropout
        )

        # Convolutional bottleneck layers
        self.conv_layers = BottleneckConstruct(
            d_model=configs.d_model,
            window_size=window_size,
            d_inner=d_bottleneck
        )

    def forward(self, x_enc, x_mark_enc):
        """
        Forward pass of the encoder.

        Args:
            x_enc (torch.Tensor): Input tensor of shape (batch, seq_length, enc_in).
            x_mark_enc (torch.Tensor): Additional input tensor for embeddings (e.g., time features).

        Returns:
            torch.Tensor: Encoded output tensor of shape (batch, input_size, d_model).
        """
        # Apply embedding to input
        seq_enc = self.enc_embedding(x_enc, x_mark_enc)  # Shape: (batch, seq_length, d_model)

        # Repeat mask for the batch and move to device
        mask = self.mask.unsqueeze(0).expand(seq_enc.size(0), -1, -1).to(x_enc.device)

        # Apply convolutional bottleneck
        seq_enc = self.conv_layers(seq_enc)  # Shape depends on bottleneck structure

        # Pass through each encoder layer
        for layer in self.layers:
            seq_enc = layer(seq_enc, attn_mask=mask)

        # Gather features from pyramid layers using precomputed indices
        indexes = self.indexes.expand(seq_enc.size(0), -1, -1, seq_enc.size(2))
        indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))  # Shape: (batch, total_size, d_model)
        all_enc = torch.gather(seq_enc, 1, indexes)

        # Reshape to the original input size
        seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)  # Shape: (batch, input_size, d_model)

        return seq_enc
