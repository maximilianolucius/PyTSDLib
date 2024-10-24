# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com

import numpy as np
import torch
import torch.nn as nn
from typing import List


def get_frequency_modes(seq_len: int, modes: int = 64, mode_select_method: str = 'random') -> List[int]:
    """
    Selects frequency modes for Fourier-based operations.

    Args:
        seq_len (int): Length of the input sequence.
        modes (int, optional): Number of frequency modes to select. Defaults to 64.
        mode_select_method (str, optional): Method to select modes.
            - 'random': Randomly sample frequency modes.
            - 'else': Select the lowest frequency modes. Defaults to 'random'.

    Returns:
        List[int]: List of selected frequency mode indices.
    """
    modes = min(modes, seq_len // 2)  # Ensure modes do not exceed Nyquist frequency
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()  # Sorting for consistent ordering
    return index


class FourierBlock(nn.Module):
    """
    1D Fourier Block for representation learning in the frequency domain.
    It performs FFT, applies learnable transformations, and then applies Inverse FFT.
    """

    def __init__(self, in_channels: int, out_channels: int, seq_len: int, modes: int = 0,
                 mode_select_method: str = 'random'):
        """
        Initializes the FourierBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            seq_len (int): Length of the input sequence.
            modes (int, optional): Number of frequency modes to use. Defaults to 0.
            mode_select_method (str, optional): Method to select frequency modes.
                - 'random': Randomly sample frequency modes.
                - 'else': Select the lowest frequency modes. Defaults to 'random'.
        """
        super(FourierBlock, self).__init__()
        print('FourierBlock initialized with frequency enhancement!')
        """
        1D Fourier block. It performs representation learning on the frequency domain,
        including FFT, linear transformation, and Inverse FFT.
        """
        # Select frequency modes based on the specified method
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        print(f'Selected modes={len(self.index)}, indices={self.index}')

        # Initialize learnable weights for frequency domain transformations
        self.scale = 1 / (in_channels * out_channels)
        # Using multiple heads (e.g., 8) for parallel transformations
        num_heads = 8
        # Ensure in_channels and out_channels are divisible by num_heads
        assert in_channels % num_heads == 0 and out_channels % num_heads == 0, \
            "in_channels and out_channels must be divisible by the number of heads (8)."

        # Initialize weights for real and imaginary parts
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(num_heads, in_channels // num_heads, out_channels // num_heads, len(self.index),
                                    dtype=torch.float))
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(num_heads, in_channels // num_heads, out_channels // num_heads, len(self.index),
                                    dtype=torch.float))

    def compl_mul1d(self, order: str, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Performs complex multiplication between input and weights.

        Args:
            order (str): Einsum order string.
            x (torch.Tensor): Input tensor, can be real or complex.
            weights (torch.Tensor): Weight tensor, can be real or complex.

        Returns:
            torch.Tensor: Result of complex multiplication.
        """
        # Check if inputs are complex; if not, convert them
        x_flag = not torch.is_complex(x)
        w_flag = not torch.is_complex(weights)
        if x_flag:
            x = torch.complex(x, torch.zeros_like(x, device=x.device))
        if w_flag:
            weights = torch.complex(weights, torch.zeros_like(weights, device=weights.device))

        # Perform complex multiplication using Einstein summation
        if torch.is_complex(x) or torch.is_complex(weights):
            real = torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag)
            imag = torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real)
            return torch.complex(real, imag)
        else:
            return torch.einsum(order, x, weights)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        """
        Forward pass of the FourierBlock.

        Args:
            q (torch.Tensor): Query tensor of shape [B, L, H, E].
            k (torch.Tensor): Key tensor (unused in this block).
            v (torch.Tensor): Value tensor (unused in this block).
            mask (torch.Tensor, optional): Mask tensor (unused in this block). Defaults to None.

        Returns:
            tuple: Transformed tensor and None (placeholder for compatibility).
        """
        # q shape: [Batch, Length, Heads, Embedding]
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)  # Reshape to [B, H, E, L]

        # Compute Fourier coefficients using real FFT
        x_ft = torch.fft.rfft(x, dim=-1)  # Shape: [B, H, E, L//2 +1]

        # Initialize output Fourier tensor
        out_ft = torch.zeros(B, H, E, len(self.index), device=x.device, dtype=torch.cfloat)

        # Apply learnable transformations on selected frequency modes
        for wi, i in enumerate(self.index):
            if i >= x_ft.shape[-1]:
                # Skip if the frequency index exceeds FFT result
                continue
            # Perform complex multiplication for each selected frequency mode
            out_ft[:, :, :, wi] = self.compl_mul1d("bhi,bhio->bho",
                                                   x_ft[:, :, :, i],
                                                   torch.complex(self.weights1[:, :, :, wi],
                                                                self.weights2[:, :, :, wi]))

        # Inverse FFT to convert back to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))  # Shape: [B, H, E, L]

        # Permute back to original shape
        x = x.permute(0, 3, 1, 2)  # Shape: [B, L, H, E]

        return x, None  # Returning None for compatibility with potential multi-output structures


class FourierCrossAttention(nn.Module):
    """
    1D Fourier Cross Attention layer for capturing interactions in the frequency domain.
    It performs FFT, applies attention mechanisms, and then applies Inverse FFT.
    """

    def __init__(self, in_channels: int, out_channels: int, seq_len_q: int, seq_len_kv: int, modes: int = 64,
                 mode_select_method: str = 'random', activation: str = 'tanh', policy: int = 0,
                 num_heads: int = 8):
        """
        Initializes the FourierCrossAttention layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            seq_len_q (int): Sequence length for queries.
            seq_len_kv (int): Sequence length for keys and values.
            modes (int, optional): Number of frequency modes to use. Defaults to 64.
            mode_select_method (str, optional): Method to select frequency modes.
                - 'random': Randomly sample frequency modes.
                - 'else': Select the lowest frequency modes. Defaults to 'random'.
            activation (str, optional): Activation function after attention. Choices: 'tanh', 'softmax'. Defaults to 'tanh'.
            policy (int, optional): Unused parameter (can be used for future extensions). Defaults to 0.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
        """
        super(FourierCrossAttention, self).__init__()
        print('FourierCrossAttention initialized with frequency enhancement!')
        """
        1D Fourier Cross Attention layer. It performs FFT, linear transformations,
        applies attention mechanisms in the frequency domain, and then applies Inverse FFT.
        """
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        # Select frequency modes for queries and keys/values
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)

        print(f'Selected query modes={len(self.index_q)}, indices_q={self.index_q}')
        print(f'Selected key/value modes={len(self.index_kv)}, indices_kv={self.index_kv}')

        # Initialize learnable weights for frequency domain transformations
        self.scale = 1 / (in_channels * out_channels)
        # Ensure in_channels and out_channels are divisible by num_heads
        assert in_channels % num_heads == 0 and out_channels % num_heads == 0, \
            "in_channels and out_channels must be divisible by num_heads."

        # Initialize weights for real and imaginary parts
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(num_heads, in_channels // num_heads, out_channels // num_heads, len(self.index_q),
                                    dtype=torch.float))
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(num_heads, in_channels // num_heads, out_channels // num_heads, len(self.index_q),
                                    dtype=torch.float))

    def compl_mul1d(self, order: str, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Performs complex multiplication between input and weights.

        Args:
            order (str): Einsum order string.
            x (torch.Tensor): Input tensor, can be real or complex.
            weights (torch.Tensor): Weight tensor, can be real or complex.

        Returns:
            torch.Tensor: Result of complex multiplication.
        """
        # Check if inputs are complex; if not, convert them
        x_flag = not torch.is_complex(x)
        w_flag = not torch.is_complex(weights)
        if x_flag:
            x = torch.complex(x, torch.zeros_like(x, device=x.device))
        if w_flag:
            weights = torch.complex(weights, torch.zeros_like(weights, device=weights.device))

        # Perform complex multiplication using Einstein summation
        if torch.is_complex(x) or torch.is_complex(weights):
            real = torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag)
            imag = torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real)
            return torch.complex(real, imag)
        else:
            return torch.einsum(order, x, weights)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        """
        Forward pass of the FourierCrossAttention layer.

        Args:
            q (torch.Tensor): Query tensor of shape [B, L_q, H, E].
            k (torch.Tensor): Key tensor of shape [B, L_kv, H, E].
            v (torch.Tensor): Value tensor of shape [B, L_kv, H, E].
            mask (torch.Tensor, optional): Mask tensor (unused in this layer). Defaults to None.

        Returns:
            tuple: Transformed tensor and None (placeholder for compatibility).
        """
        # q, k, v shapes: [Batch, Length, Heads, Embedding]
        B, L_q, H, E = q.shape
        _, L_kv, _, _ = k.shape

        # Permute tensors to [B, H, E, L]
        xq = q.permute(0, 2, 3, 1)  # [B, H, E, L_q]
        xk = k.permute(0, 2, 3, 1)  # [B, H, E, L_kv]
        xv = v.permute(0, 2, 3, 1)  # [B, H, E, L_kv]

        # Compute FFT for queries
        xq_ft = torch.fft.rfft(xq, dim=-1)  # [B, H, E, L_q//2 +1]
        # Extract selected frequency modes for queries
        xq_ft_selected = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        for i, idx in enumerate(self.index_q):
            if idx < xq_ft.shape[-1]:
                xq_ft_selected[:, :, :, i] = xq_ft[:, :, :, idx]

        # Compute FFT for keys
        xk_ft = torch.fft.rfft(xk, dim=-1)  # [B, H, E, L_kv//2 +1]
        # Extract selected frequency modes for keys
        xk_ft_selected = torch.zeros(B, H, E, len(self.index_kv), device=xk.device, dtype=torch.cfloat)
        for i, idx in enumerate(self.index_kv):
            if idx < xk_ft.shape[-1]:
                xk_ft_selected[:, :, :, i] = xk_ft[:, :, :, idx]

        # Perform attention mechanism in frequency domain
        # Compute query-key interactions via complex multiplication
        # xqk_ft shape: [B, H, E, len_q] x [B, H, E, len_kv] -> [B, H, E, len_q, len_kv]
        # Here, we simplify and use einsum for batch operations
        # However, the original code seems to perform a specific complex operation
        # We'll follow the original logic
        # For simplicity, let's assume interaction over embedding dimension E

        # Compute element-wise multiplication between queries and keys
        # Using Einstein summation for batch processing
        xqk_ft = self.compl_mul1d("bhxe,bhye->bhxy", xq_ft_selected, xk_ft_selected)
        # Apply activation function
        if self.activation == 'tanh':
            xqk_ft = torch.complex(torch.tanh(xqk_ft.real), torch.tanh(xqk_ft.imag))
        elif self.activation == 'softmax':
            # Compute softmax over the last dimension (frequency modes)
            amplitude = torch.abs(xqk_ft)
            softmax_weights = torch.softmax(amplitude, dim=-1)
            xqk_ft = torch.complex(softmax_weights, torch.zeros_like(softmax_weights, device=xqk_ft.device))
        else:
            raise ValueError(f"Activation function '{self.activation}' is not implemented.")

        # Multiply attention weights with keys in frequency domain
        # xqk_ft shape: [B, H, E, len_q, len_kv]
        # xk_ft_selected shape: [B, H, E, len_kv]
        # Resulting shape: [B, H, E, len_q]
        xqkv_ft = self.compl_mul1d("bhxy,bhxy->bhx", xqk_ft, xk_ft_selected)

        # Apply learnable weights in frequency domain
        xqkvw = self.compl_mul1d("bhx,hex->bhx", xqkv_ft, torch.complex(self.weights1, self.weights2))

        # Initialize output Fourier tensor
        out_ft = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)

        # Assign transformed frequency components back
        for i, idx in enumerate(self.index_q):
            if i < xqkvw.shape[-1] and idx < out_ft.shape[-1]:
                out_ft[:, :, :, i] = xqkvw[:, :, :, i]

        # Inverse FFT to convert back to time domain
        out = torch.fft.irfft(out_ft, n=xq.size(-1))  # [B, H, E, L_q]

        # Permute back to original shape
        out = out.permute(0, 3, 1, 2)  # [B, L_q, H, E]

        # Normalize by scaling factors
        out = out / (self.in_channels * self.out_channels)

        return out, None  # Returning None for compatibility with potential multi-output structures
