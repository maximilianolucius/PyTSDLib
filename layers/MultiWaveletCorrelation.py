# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com

import math
from functools import partial
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import eval_legendre
from sympy import Poly, legendre, Symbol, chebyshevt
from torch import Tensor
from torch import nn


def legendre_derivative(k: int, x: float) -> float:
    """
    Computes the derivative of the Legendre polynomial of degree k at point x.

    Args:
        k (int): Degree of the Legendre polynomial.
        x (float): Point at which to evaluate the derivative.

    Returns:
        float: The derivative value.
    """
    def _legendre(k: int, x: float) -> float:
        return (2 * k + 1) * eval_legendre(k, x)

    derivative = 0.0
    for i in np.arange(k - 1, -1, -2):
        derivative += _legendre(i, x)
    return derivative


def phi_(phi_coeff: np.ndarray, x: np.ndarray, lb: float = 0.0, ub: float = 1.0) -> np.ndarray:
    """
    Evaluates the polynomial with given coefficients at points x,
    and masks out values outside the interval [lb, ub].

    Args:
        phi_coeff (np.ndarray): Coefficients of the polynomial.
        x (np.ndarray): Points at which to evaluate the polynomial.
        lb (float, optional): Lower bound of the valid interval. Defaults to 0.0.
        ub (float, optional): Upper bound of the valid interval. Defaults to 1.0.

    Returns:
        np.ndarray: Evaluated and masked polynomial values.
    """
    mask = np.logical_or(x < lb, x > ub).astype(np.float32)
    poly = np.poly1d(np.flip(phi_coeff))
    return poly(x) * (1 - mask)


def get_phi_psi(k: int, base: str = 'legendre') -> Tuple[List[np.poly1d], List[partial], List[partial]]:
    """
    Generates phi and psi polynomials based on the specified base ('legendre' or 'chebyshev').

    Args:
        k (int): Number of polynomials.
        base (str, optional): Type of polynomial basis. Choices: 'legendre', 'chebyshev'. Defaults to 'legendre'.

    Returns:
        Tuple[List[np.poly1d], List[partial], List[partial]]: Lists of phi, psi1, and psi2 polynomials.
    """
    x = Symbol('x')
    phi = []
    psi1 = []
    psi2 = []

    if base == 'legendre':
        phi_coeff = np.zeros((k, k))
        phi_2x_coeff = np.zeros((k, k))

        # Generate phi and phi_2x coefficients using Legendre polynomials
        for ki in range(k):
            # Phi coefficients for Legendre
            leg_poly = legendre(ki, 2 * x - 1)
            poly = Poly(leg_poly, x).all_coeffs()
            phi_coeff[ki, :ki + 1] = np.flip(np.sqrt(2 * ki + 1) * np.array(poly, dtype=np.float64))

            # Phi coefficients for 2x scaling
            leg_poly_2x = legendre(ki, 4 * x - 1)
            poly_2x = Poly(leg_poly_2x, x).all_coeffs()
            phi_2x_coeff[ki, :ki + 1] = np.flip(np.sqrt(2) * np.sqrt(2 * ki + 1) * np.array(poly_2x, dtype=np.float64))

        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))

        # Orthogonalize psi1 and psi2 against phi
        for ki in range(k):
            psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
            psi2_coeff[ki, :] = phi_2x_coeff[ki, :]

            # Orthogonalize against phi
            for i in range(k):
                a = phi_2x_coeff[ki, :ki + 1]
                b = phi_coeff[i, :i + 1]
                prod = np.convolve(a, b)
                prod[np.abs(prod) < 1e-8] = 0
                proj = (prod * (1 / (np.arange(len(prod)) + 1)) * (0.5 ** (1 + np.arange(len(prod))))).sum()
                psi1_coeff[ki, :] -= proj * phi_coeff[i, :]
                psi2_coeff[ki, :] -= proj * phi_coeff[i, :]

            # Orthogonalize against previous psi1 and psi2
            for j in range(ki):
                a = phi_2x_coeff[ki, :ki + 1]
                b = psi1_coeff[j, :]
                prod = np.convolve(a, b)
                prod[np.abs(prod) < 1e-8] = 0
                proj = (prod * (1 / (np.arange(len(prod)) + 1)) * (0.5 ** (1 + np.arange(len(prod))))).sum()
                psi1_coeff[ki, :] -= proj * psi1_coeff[j, :]
                psi2_coeff[ki, :] -= proj * psi2_coeff[j, :]

            # Normalize psi1 and psi2
            norm1 = (np.convolve(psi1_coeff[ki, :], psi1_coeff[ki, :]) * (1 / (np.arange(len(psi1_coeff[ki, :])) + 1)) * (0.5 ** (1 + np.arange(len(psi1_coeff[ki, :]))))).sum()
            norm2 = (np.convolve(psi2_coeff[ki, :], psi2_coeff[ki, :]) * (1 / (np.arange(len(psi2_coeff[ki, :])) + 1)) * (1 - 0.5 ** (1 + np.arange(len(psi2_coeff[ki, :]))))).sum()
            norm = np.sqrt(norm1 + norm2)
            psi1_coeff[ki, :] /= norm
            psi2_coeff[ki, :] /= norm

            # Zero out negligible coefficients
            psi1_coeff[np.abs(psi1_coeff) < 1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff) < 1e-8] = 0

            # Define psi1 and psi2 polynomials with appropriate masks
            psi1.append(partial(phi_, psi1_coeff[ki, :], lb=0, ub=0.5))
            psi2.append(partial(phi_, psi2_coeff[ki, :], lb=0.5, ub=1))

        phi = [np.poly1d(np.flip(phi_coeff[i, :])) for i in range(k)]

    elif base == 'chebyshev':
        phi_coeff = np.zeros((k, k))
        phi_2x_coeff = np.zeros((k, k))

        # Generate phi and phi_2x coefficients using Chebyshev polynomials
        for ki in range(k):
            if ki == 0:
                phi_coeff[ki, :ki + 1] = np.sqrt(2 / np.pi)
                phi_2x_coeff[ki, :ki + 1] = np.sqrt(2 / np.pi) * np.sqrt(2)
            else:
                cheb_poly = chebyshevt(ki, 2 * x - 1)
                poly = Poly(cheb_poly, x).all_coeffs()
                phi_coeff[ki, :ki + 1] = np.flip(2 / np.sqrt(np.pi) * np.array(poly, dtype=np.float64))

                cheb_poly_2x = chebyshevt(ki, 4 * x - 1)
                poly_2x = Poly(cheb_poly_2x, x).all_coeffs()
                phi_2x_coeff[ki, :ki + 1] = np.flip(np.sqrt(2) * 2 / np.sqrt(np.pi) * np.array(poly_2x, dtype=np.float64))

        phi = [partial(phi_, phi_coeff[i, :]) for i in range(k)]

        # Compute roots for Chebyshev polynomials
        k_use = 2 * k
        roots = Poly(chebyshevt(k_use, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots], dtype=np.float64)
        wm = np.pi / k_use / 2

        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))

        # Orthogonalize psi1 and psi2 against phi
        for ki in range(k):
            psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
            psi2_coeff[ki, :] = phi_2x_coeff[ki, :]

            # Orthogonalize against phi
            for i in range(k):
                proj = (wm * phi[i](x_m) * np.sqrt(2) * phi[ki](2 * x_m)).sum()
                psi1_coeff[ki, :] -= proj * phi_coeff[i, :]
                psi2_coeff[ki, :] -= proj * phi_coeff[i, :]

            # Orthogonalize against previous psi1 and psi2
            for j in range(ki):
                proj = (wm * psi1[j](x_m) * np.sqrt(2) * phi[ki](2 * x_m)).sum()
                psi1_coeff[ki, :] -= proj * psi1_coeff[j, :]
                psi2_coeff[ki, :] -= proj * psi2_coeff[j, :]

            # Normalize psi1 and psi2
            norm1 = (wm * psi1_coeff[ki, :](x_m) * psi1_coeff[ki, :](x_m)).sum()
            norm2 = (wm * psi2_coeff[ki, :](x_m) * psi2_coeff[ki, :](x_m)).sum()
            norm = np.sqrt(norm1 + norm2)
            psi1_coeff[ki, :] /= norm
            psi2_coeff[ki, :] /= norm

            # Zero out negligible coefficients
            psi1_coeff[np.abs(psi1_coeff) < 1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff) < 1e-8] = 0

            # Define psi1 and psi2 polynomials with appropriate masks
            psi1.append(partial(phi_, psi1_coeff[ki, :], lb=0, ub=0.5 + 1e-16))
            psi2.append(partial(phi_, psi2_coeff[ki, :], lb=0.5 + 1e-16, ub=1))

    else:
        raise ValueError("Base must be either 'legendre' or 'chebyshev'.")

    return phi, psi1, psi2


def get_filter(base: str, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs filter matrices for multiwavelet transforms based on the specified base.

    Args:
        base (str): Type of polynomial basis. Choices: 'legendre', 'chebyshev'.
        k (int): Number of wavelet scales.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Filter matrices H0, H1, G0, G1, PHI0, PHI1.
    """
    def psi(x: callable, psi1: List[partial], psi2: List[partial], i: int, inp: np.ndarray) -> float:
        """
        Combines psi1 and psi2 with masking based on input.

        Args:
            psi1 (List[partial]): List of psi1 partial functions.
            psi2 (List[partial]): List of psi2 partial functions.
            i (int): Index of the current wavelet.
            inp (np.ndarray): Input array.

        Returns:
            float: Combined psi values.
        """
        mask = (inp <= 0.5).astype(float)
        return psi1[i](inp) * mask + psi2[i](inp) * (1 - mask)

    if base not in ['legendre', 'chebyshev']:
        raise ValueError("Base must be either 'legendre' or 'chebyshev'.")

    phi, psi1, psi2 = get_phi_psi(k, base)
    H0, H1, G0, G1, PHI0, PHI1 = np.zeros((k, k)), np.zeros((k, k)), np.zeros((k, k)), np.zeros((k, k)), np.eye(k), np.eye(k)

    if base == 'legendre':
        roots = Poly(legendre(k, 2 * Symbol('x') - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots], dtype=np.float64)
        wm = 1 / k / legendre_derivative(k, 2 * x_m - 1) / eval_legendre(k - 1, 2 * x_m - 1)

        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki](x_m / 2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m / 2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki]((x_m + 1) / 2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m + 1) / 2) * phi[kpi](x_m)).sum()

        PHI0 = np.eye(k)
        PHI1 = np.eye(k)

    elif base == 'chebyshev':
        k_use = 2 * k
        roots = Poly(chebyshevt(k_use, 2 * Symbol('x') - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots], dtype=np.float64)
        wm = np.pi / k_use / 2

        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki](x_m / 2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m / 2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki]((x_m + 1) / 2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m + 1) / 2) * phi[kpi](x_m)).sum()

                PHI0[ki, kpi] = (wm * phi[ki](2 * x_m) * phi[kpi](2 * x_m)).sum() * 2
                PHI1[ki, kpi] = (wm * phi[ki](2 * x_m - 1) * phi[kpi](2 * x_m - 1)).sum() * 2

        PHI0[np.abs(PHI0) < 1e-8] = 0
        PHI1[np.abs(PHI1) < 1e-8] = 0

    # Zero out negligible values to ensure sparsity
    H0[np.abs(H0) < 1e-8] = 0
    H1[np.abs(H1) < 1e-8] = 0
    G0[np.abs(G0) < 1e-8] = 0
    G1[np.abs(G1) < 1e-8] = 0

    return H0, H1, G0, G1, PHI0, PHI1


class SparseKernelFT1d(nn.Module):
    """
    Sparse Kernel Fourier Transform for 1D data.
    Applies sparse frequency domain transformations using learnable weights.
    """

    def __init__(self, k: int, alpha: int, c: int = 1, **kwargs):
        """
        Initializes the SparseKernelFT1d module.

        Args:
            k (int): Number of wavelet scales.
            alpha (int): Number of frequency modes.
            c (int, optional): Number of channels. Defaults to 1.
            **kwargs: Additional keyword arguments.
        """
        super(SparseKernelFT1d, self).__init__()
        self.k = k
        self.alpha = alpha
        self.c = c

        # Initialize learnable weights for real and imaginary parts
        self.scale = 1 / (c * k * c * k)
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(c * k, c * k, self.alpha, dtype=torch.float32))
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(c * k, c * k, self.alpha, dtype=torch.float32))

    @staticmethod
    def compl_mul1d(order: str, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Performs complex multiplication using Einstein summation.

        Args:
            order (str): Einsum equation.
            x (torch.Tensor): Input tensor (can be real or complex).
            weights (torch.Tensor): Weight tensor (can be real or complex).

        Returns:
            torch.Tensor: Result of complex multiplication.
        """
        # Convert to complex if necessary
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x, device=x.device))
        if not torch.is_complex(weights):
            weights = torch.complex(weights, torch.zeros_like(weights, device=weights.device))

        # Perform complex multiplication
        real = torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag)
        imag = torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real)
        return torch.complex(real, imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SparseKernelFT1d.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, c, k].

        Returns:
            torch.Tensor: Transformed tensor of shape [B, N, c, k].
        """
        B, N, c, k = x.shape  # [Batch, Length, Channels, Scales]

        # Reshape and permute for FFT
        x = x.view(B, N, -1)  # [B, N, c*k]
        x = x.permute(0, 2, 1)  # [B, c*k, N]

        # Compute FFT
        x_fft = torch.fft.rfft(x, dim=-1)  # [B, c*k, N//2 +1]

        # Determine number of frequency modes to use
        l = min(self.alpha, x_fft.shape[-1])

        # Initialize output in frequency domain
        out_ft = torch.zeros(B, self.c * self.k, x_fft.shape[-1], device=x.device, dtype=torch.cfloat)

        # Apply sparse frequency transformations
        out_ft[:, :, :l] = self.compl_mul1d("bix,iox->box", x_fft[:, :, :l],
                                           torch.complex(self.weights1[:, :, :l], self.weights2[:, :, :l]))

        # Inverse FFT to convert back to time domain
        x = torch.fft.irfft(out_ft, n=N, dim=-1)  # [B, c*k, N]

        # Reshape back to original dimensions
        x = x.permute(0, 2, 1).view(B, N, c, k)  # [B, N, c, k]

        return x


class MWT_CZ1d(nn.Module):
    """
    MultiWavelet Transform with Cross Correlation for 1D data.
    Applies multiwavelet transformations and learnable frequency domain operations.
    """

    def __init__(self, k: int = 3, alpha: int = 64, c: int = 1, L: int = 0, base: str = 'legendre', **kwargs):
        """
        Initializes the MWT_CZ1d module.

        Args:
            k (int, optional): Number of wavelet scales. Defaults to 3.
            alpha (int, optional): Number of frequency modes. Defaults to 64.
            c (int, optional): Number of channels. Defaults to 1.
            L (int, optional): Number of levels. Defaults to 0.
            base (str, optional): Type of polynomial basis. Choices: 'legendre', 'chebyshev'. Defaults to 'legendre'.
            **kwargs: Additional keyword arguments.
        """
        super(MWT_CZ1d, self).__init__()
        self.k = k
        self.alpha = alpha
        self.c = c
        self.L = L

        # Get filter matrices based on the base
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)

        # Compute transformed filters
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        # Zero out negligible values
        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0

        # Initialize attention mechanisms
        self.attn1 = FourierCrossAttentionW(
            in_channels=c,
            out_channels=c,
            seq_len_q=k,
            seq_len_kv=k,
            modes=alpha,
            activation='tanh',
            mode_select_method='random'
        )
        self.attn2 = FourierCrossAttentionW(
            in_channels=c,
            out_channels=c,
            seq_len_q=k,
            seq_len_kv=k,
            modes=alpha,
            activation='tanh',
            mode_select_method='random'
        )
        self.attn3 = FourierCrossAttentionW(
            in_channels=c,
            out_channels=c,
            seq_len_q=k,
            seq_len_kv=k,
            modes=alpha,
            activation='tanh',
            mode_select_method='random'
        )
        self.attn4 = FourierCrossAttentionW(
            in_channels=c,
            out_channels=c,
            seq_len_q=k,
            seq_len_kv=k,
            modes=alpha,
            activation='tanh',
            mode_select_method='random'
        )

        # Linear transformation
        self.T0 = nn.Linear(k, k)

        # Register buffers for filter matrices
        self.register_buffer('ec_s', torch.tensor(np.concatenate((H0.T, H1.T), axis=0), dtype=torch.float32))
        self.register_buffer('ec_d', torch.tensor(np.concatenate((G0.T, G1.T), axis=0), dtype=torch.float32))
        self.register_buffer('rc_e', torch.tensor(np.concatenate((H0r, G0r), axis=0), dtype=torch.float32))
        self.register_buffer('rc_o', torch.tensor(np.concatenate((H1r, G1r), axis=0), dtype=torch.float32))

        # Initialize linear layers for input transformations
        self.Lk = nn.Linear(c * k, c * k)
        self.Lq = nn.Linear(c * k, c * k)
        self.Lv = nn.Linear(c * k, c * k)
        self.out = nn.Linear(c * k, c)

    def wavelet_transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs wavelet decomposition on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, c, k].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Detail coefficients and approximation coefficients.
        """
        # Concatenate even and odd indices
        xa = torch.cat([x[:, ::2, :, :], x[:, 1::2, :, :]], dim=-1)  # [B, N/2, c, 2k]

        # Apply filter matrices
        d = torch.matmul(xa, self.ec_d)  # Detail coefficients
        s = torch.matmul(xa, self.ec_s)  # Approximation coefficients

        return d, s

    def even_odd(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the signal from even and odd components.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, c, 2k].

        Returns:
            torch.Tensor: Reconstructed tensor of shape [B, 2*N, c, k].
        """
        B, N, c, ich = x.shape  # [B, N, c, 2k]
        assert ich == 2 * self.k, "Input channels must be twice the number of wavelet scales."

        # Apply reconstruction filters
        x_e = torch.matmul(x, self.rc_e)  # Even components
        x_o = torch.matmul(x, self.rc_o)  # Odd components

        # Initialize reconstructed tensor
        x_recon = torch.zeros(B, N * 2, c, self.k, device=x.device)
        x_recon[..., ::2, :, :] = x_e
        x_recon[..., 1::2, :, :] = x_o

        return x_recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MWT_CZ1d.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, c, k].

        Returns:
            torch.Tensor: Output tensor after multiwavelet transform.
        """
        B, N, c, k = x.shape

        # Ensure the sequence length is a power of two by padding if necessary
        ns = math.floor(np.log2(N))
        nl = 2 ** math.ceil(np.log2(N))
        extra_x = x[:, :nl - N, :, :] if nl > N else torch.zeros_like(x[:, :0, :, :], device=x.device)
        x = torch.cat([x, extra_x], dim=1)  # [B, nl, c, k]

        Ud = []
        Us = []

        # Wavelet decomposition
        for _ in range(ns - self.L):
            d, x = self.wavelet_transform(x)
            Ud.append(self.attn1(d) + self.attn2(x))
            Us.append(self.attn3(d))

        # Apply final linear transformation at the coarsest scale
        x = self.attn4(x)

        # Wavelet reconstruction
        for i in reversed(range(ns - self.L)):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), dim=-1)
            x = self.even_odd(x)

        # Trim to original sequence length
        x = x[:, :N, :, :]

        # Final linear projection
        x = self.out(x.view(B, N, -1))  # [B, N, c]

        return x


class FourierCrossAttentionW(nn.Module):
    """
    Fourier Cross Attention Wrapper for 1D data.
    Applies cross attention mechanisms in the frequency domain with optional activation.
    """

    def __init__(self, in_channels: int, out_channels: int, seq_len_q: int, seq_len_kv: int, modes: int = 16,
                 activation: str = 'tanh', mode_select_method: str = 'random'):
        """
        Initializes the FourierCrossAttentionW module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            seq_len_q (int): Sequence length for queries.
            seq_len_kv (int): Sequence length for keys and values.
            modes (int, optional): Number of frequency modes. Defaults to 16.
            activation (str, optional): Activation function. Choices: 'tanh', 'softmax'. Defaults to 'tanh'.
            mode_select_method (str, optional): Method to select frequency modes. Defaults to 'random'.
        """
        super(FourierCrossAttentionW, self).__init__()
        print('Fourier Cross Attention Wrapper initialized with frequency correlation!')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.activation = activation
        self.mode_select_method = mode_select_method

    @staticmethod
    def compl_mul1d(order: str, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Performs complex multiplication using Einstein summation.

        Args:
            order (str): Einsum equation.
            x (torch.Tensor): Input tensor (can be real or complex).
            weights (torch.Tensor): Weight tensor (can be real or complex).

        Returns:
            torch.Tensor: Result of complex multiplication.
        """
        # Convert to complex if necessary
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x, device=x.device))
        if not torch.is_complex(weights):
            weights = torch.complex(weights, torch.zeros_like(weights, device=weights.device))

        # Perform complex multiplication
        real = torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag)
        imag = torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real)
        return torch.complex(real, imag)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, None]:
        """
        Forward pass for FourierCrossAttentionW.

        Args:
            q (torch.Tensor): Query tensor of shape [B, L, E, H].
            k (torch.Tensor): Key tensor of shape [B, S, E, H].
            v (torch.Tensor): Value tensor of shape [B, S, E, H].
            mask (torch.Tensor, optional): Mask tensor. Defaults to None.

        Returns:
            Tuple[torch.Tensor, None]: Transformed tensor and None (placeholder for compatibility).
        """
        B, L, E, H = q.shape
        _, S, _, _ = k.shape

        # Permute to [B, H, E, L] and [B, H, E, S]
        xq = q.permute(0, 3, 2, 1)  # [B, H, E, L]
        xk = k.permute(0, 3, 2, 1)  # [B, H, E, S]
        xv = v.permute(0, 3, 2, 1)  # [B, H, E, S]

        # Select frequency modes for queries and keys/values
        index_q = list(range(0, min(L // 2, self.modes)))
        index_kv = list(range(0, min(S // 2, self.modes)))

        # Compute FFT for queries and keys
        xq_ft = torch.fft.rfft(xq, dim=-1)  # [B, H, E, L//2 +1]
        xk_ft = torch.fft.rfft(xk, dim=-1)  # [B, H, E, S//2 +1]

        # Select the specified frequency modes
        xq_ft_selected = xq_ft[:, :, :, index_q]  # [B, H, E, len(index_q)]
        xk_ft_selected = xk_ft[:, :, :, index_kv]  # [B, H, E, len(index_kv)]

        # Perform cross attention in the frequency domain
        xqk_ft = self.compl_mul1d("bhex,bhey->bhxy", xq_ft_selected, xk_ft_selected)  # [B, H, E, len_q, len_kv]

        # Apply activation function
        if self.activation == 'tanh':
            xqk_ft = torch.complex(torch.tanh(xqk_ft.real), torch.tanh(xqk_ft.imag))
        elif self.activation == 'softmax':
            amplitude = torch.abs(xqk_ft)
            softmax_weights = torch.softmax(amplitude, dim=-1)
            xqk_ft = torch.complex(softmax_weights, torch.zeros_like(softmax_weights, device=xqk_ft.device))
        else:
            raise ValueError(f"Activation function '{self.activation}' is not implemented.")

        # Multiply attention weights with keys in frequency domain
        xqkv_ft = self.compl_mul1d("bhxy,bhey->bhex", xqk_ft, xk_ft_selected)  # [B, H, E, len_q]

        # Assign transformed frequency components back
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(index_q):
            if i < xqkv_ft.shape[-1] and j < out_ft.shape[-1]:
                out_ft[:, :, :, j] = xqkv_ft[:, :, :, i]

        # Inverse FFT to convert back to time domain
        out = torch.fft.irfft(out_ft / (self.in_channels * self.out_channels), n=L, dim=-1)  # [B, H, E, L]

        # Permute back to original shape [B, L, H, E]
        out = out.permute(0, 3, 1, 2)

        return out, None  # Returning None for compatibility


class MultiWaveletTransform(nn.Module):
    """
    1D MultiWavelet Transform block.
    Applies multiwavelet transformations to the input data.
    """

    def __init__(self, ich: int = 1, k: int = 8, alpha: int = 16, c: int = 128,
                 nCZ: int = 1, L: int = 0, base: str = 'legendre', attention_dropout: float = 0.1):
        """
        Initializes the MultiWaveletTransform module.

        Args:
            ich (int, optional): Number of input channels. Defaults to 1.
            k (int, optional): Number of wavelet scales. Defaults to 8.
            alpha (int, optional): Number of frequency modes. Defaults to 16.
            c (int, optional): Number of channels. Defaults to 128.
            nCZ (int, optional): Number of Cross Zeta layers. Defaults to 1.
            L (int, optional): Number of levels. Defaults to 0.
            base (str, optional): Type of polynomial basis. Choices: 'legendre', 'chebyshev'. Defaults to 'legendre'.
            attention_dropout (float, optional): Dropout rate for attention mechanisms. Defaults to 0.1.
        """
        super(MultiWaveletTransform, self).__init__()
        print('MultiWaveletTransform initialized with base:', base)
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ

        # Initialize linear layers for input transformation
        self.Lk0 = nn.Linear(ich, c * k)
        self.Lk1 = nn.Linear(c * k, ich)
        self.ich = ich

        # Initialize MultiWavelet Cross Zeta layers
        self.MWT_CZ = nn.ModuleList([MWT_CZ1d(k, alpha, L, c, base) for _ in range(nCZ)])

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, attn_mask: torch.Tensor = None) -> Tuple[torch.Tensor, None]:
        """
        Forward pass for MultiWaveletTransform.

        Args:
            queries (torch.Tensor): Query tensor of shape [B, L, H, E].
            keys (torch.Tensor): Key tensor of shape [B, S, H, E].
            values (torch.Tensor): Value tensor of shape [B, S, H, E].
            attn_mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            Tuple[torch.Tensor, None]: Transformed tensor and None (placeholder for compatibility).
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # Adjust sequence lengths by padding or trimming
        if L > S:
            zeros = torch.zeros_like(queries[:, :L - S, :, :], dtype=queries.dtype, device=queries.device)
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # Reshape for linear transformation
        values = values.view(B, L, -1)

        # Apply linear transformation
        V = self.Lk0(values).view(B, L, self.c, -1)

        # Apply MultiWavelet Cross Zeta layers with ReLU activation except for the last layer
        for i in range(self.nCZ):
            V = self.MWT_CZ[i](V)
            if i < self.nCZ - 1:
                V = F.relu(V)

        # Final linear transformation
        V = self.Lk1(V.view(B, L, -1))
        V = V.view(B, L, -1, D)

        return V.contiguous(), None  # Returning None for compatibility


class MWT_CZ1d(nn.Module):
    """
    MultiWavelet Transform Cross Zeta for 1D data.
    Applies sparse frequency domain transformations and wavelet operations.
    """

    def __init__(self, k: int = 3, alpha: int = 64, c: int = 1, L: int = 0, base: str = 'legendre', **kwargs):
        """
        Initializes the MWT_CZ1d module.

        Args:
            k (int, optional): Number of wavelet scales. Defaults to 3.
            alpha (int, optional): Number of frequency modes. Defaults to 64.
            c (int, optional): Number of channels. Defaults to 1.
            L (int, optional): Number of levels. Defaults to 0.
            base (str, optional): Type of polynomial basis. Choices: 'legendre', 'chebyshev'. Defaults to 'legendre'.
            **kwargs: Additional keyword arguments.
        """
        super(MWT_CZ1d, self).__init__()
        self.k = k
        self.alpha = alpha
        self.c = c
        self.L = L

        # Get filter matrices based on the base
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        # Zero out negligible values
        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0

        # Initialize sparse kernel Fourier transforms
        self.A = SparseKernelFT1d(k, alpha, c)
        self.B = SparseKernelFT1d(k, alpha, c)
        self.C = SparseKernelFT1d(k, alpha, c)

        # Linear transformation
        self.T0 = nn.Linear(k, k)

        # Register buffers for filter matrices
        self.register_buffer('ec_s', torch.tensor(np.concatenate((H0.T, H1.T), axis=0), dtype=torch.float32))
        self.register_buffer('ec_d', torch.tensor(np.concatenate((G0.T, G1.T), axis=0), dtype=torch.float32))
        self.register_buffer('rc_e', torch.tensor(np.concatenate((H0r, G0r), axis=0), dtype=torch.float32))
        self.register_buffer('rc_o', torch.tensor(np.concatenate((H1r, G1r), axis=0), dtype=torch.float32))

    def wavelet_transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs wavelet decomposition on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, c, k].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Detail coefficients and approximation coefficients.
        """
        xa = torch.cat([x[:, ::2, :, :], x[:, 1::2, :, :]], dim=-1)  # [B, N/2, c, 2k]
        d = torch.matmul(xa, self.ec_d)  # Detail coefficients
        s = torch.matmul(xa, self.ec_s)  # Approximation coefficients
        return d, s

    def even_odd(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the signal from even and odd components.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, c, 2k].

        Returns:
            torch.Tensor: Reconstructed tensor of shape [B, 2*N, c, k].
        """
        B, N, c, ich = x.shape  # [B, N, c, 2k]
        assert ich == 2 * self.k, "Input channels must be twice the number of wavelet scales."

        # Apply reconstruction filters
        x_e = torch.matmul(x, self.rc_e)  # Even components
        x_o = torch.matmul(x, self.rc_o)  # Odd components

        # Initialize reconstructed tensor
        x_recon = torch.zeros(B, N * 2, c, self.k, device=x.device)
        x_recon[..., ::2, :, :] = x_e
        x_recon[..., 1::2, :, :] = x_o

        return x_recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MWT_CZ1d.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, c, k].

        Returns:
            torch.Tensor: Output tensor after multiwavelet transform.
        """
        B, N, c, k = x.shape

        # Ensure the sequence length is a power of two by padding if necessary
        ns = math.floor(np.log2(N))
        nl = 2 ** math.ceil(np.log2(N))
        extra_x = x[:, :nl - N, :, :] if nl > N else torch.zeros_like(x[:, :0, :, :], device=x.device)
        x = torch.cat([x, extra_x], dim=1)  # [B, nl, c, k]

        Ud = []
        Us = []

        # Wavelet decomposition and sparse frequency transformations
        for _ in range(ns - self.L):
            d, x = self.wavelet_transform(x)
            Ud.append(self.A(d) + self.B(x))
            Us.append(self.C(d))

        # Apply linear transformation at the coarsest scale
        x = self.T0(x)

        # Wavelet reconstruction
        for i in reversed(range(ns - self.L)):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), dim=-1)
            x = self.even_odd(x)

        # Trim to original sequence length
        x = x[:, :N, :, :]

        return x


class FourierCrossAttentionW(nn.Module):
    """
    Fourier Cross Attention Wrapper for 1D data.
    Applies cross attention mechanisms in the frequency domain with optional activation.
    """

    def __init__(self, in_channels: int, out_channels: int, seq_len_q: int, seq_len_kv: int, modes: int = 16,
                 activation: str = 'tanh', mode_select_method: str = 'random'):
        """
        Initializes the FourierCrossAttentionW module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            seq_len_q (int): Sequence length for queries.
            seq_len_kv (int): Sequence length for keys and values.
            modes (int, optional): Number of frequency modes. Defaults to 16.
            activation (str, optional): Activation function. Choices: 'tanh', 'softmax'. Defaults to 'tanh'.
            mode_select_method (str, optional): Method to select frequency modes. Defaults to 'random'.
        """
        super(FourierCrossAttentionW, self).__init__()
        print('Fourier Cross Attention Wrapper initialized with frequency correlation!')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.activation = activation
        self.mode_select_method = mode_select_method

    @staticmethod
    def compl_mul1d(order: str, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Performs complex multiplication using Einstein summation.

        Args:
            order (str): Einsum equation.
            x (torch.Tensor): Input tensor (can be real or complex).
            weights (torch.Tensor): Weight tensor (can be real or complex).

        Returns:
            torch.Tensor: Result of complex multiplication.
        """
        # Convert to complex if necessary
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x, device=x.device))
        if not torch.is_complex(weights):
            weights = torch.complex(weights, torch.zeros_like(weights, device=weights.device))

        # Perform complex multiplication
        real = torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag)
        imag = torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real)
        return torch.complex(real, imag)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, None]:
        """
        Forward pass for FourierCrossAttentionW.

        Args:
            q (torch.Tensor): Query tensor of shape [B, L, E, H].
            k (torch.Tensor): Key tensor of shape [B, S, E, H].
            v (torch.Tensor): Value tensor of shape [B, S, E, H].
            mask (torch.Tensor, optional): Mask tensor. Defaults to None.

        Returns:
            Tuple[torch.Tensor, None]: Transformed tensor and None (placeholder for compatibility).
        """
        B, L, E, H = q.shape
        _, S, _, _ = k.shape

        # Permute to [B, H, E, L] and [B, H, E, S]
        xq = q.permute(0, 3, 2, 1)  # [B, H, E, L]
        xk = k.permute(0, 3, 2, 1)  # [B, H, E, S]
        xv = v.permute(0, 3, 2, 1)  # [B, H, E, S]

        # Select frequency modes for queries and keys/values
        index_q = list(range(0, min(L // 2, self.modes)))
        index_kv = list(range(0, min(S // 2, self.modes)))

        # Compute FFT for queries and keys
        xq_ft = torch.fft.rfft(xq, dim=-1)  # [B, H, E, L//2 +1]
        xk_ft = torch.fft.rfft(xk, dim=-1)  # [B, H, E, S//2 +1]

        # Select the specified frequency modes
        xq_ft_selected = xq_ft[:, :, :, index_q]  # [B, H, E, len(index_q)]
        xk_ft_selected = xk_ft[:, :, :, index_kv]  # [B, H, E, len(index_kv)]

        # Perform cross attention in the frequency domain
        xqk_ft = self.compl_mul1d("bhex,bhey->bhxy", xq_ft_selected, xk_ft_selected)  # [B, H, E, len_q, len_kv]

        # Apply activation function
        if self.activation == 'tanh':
            xqk_ft = torch.complex(torch.tanh(xqk_ft.real), torch.tanh(xqk_ft.imag))
        elif self.activation == 'softmax':
            amplitude = torch.abs(xqk_ft)
            softmax_weights = torch.softmax(amplitude, dim=-1)
            xqk_ft = torch.complex(softmax_weights, torch.zeros_like(softmax_weights, device=xqk_ft.device))
        else:
            raise ValueError(f"Activation function '{self.activation}' is not implemented.")

        # Multiply attention weights with keys in frequency domain
        xqkv_ft = self.compl_mul1d("bhxy,bhey->bhex", xqk_ft, xk_ft_selected)  # [B, H, E, len_q]

        # Initialize output Fourier tensor
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)

        # Assign transformed frequency components back
        for i, j in enumerate(index_q):
            if i < xqkv_ft.shape[-1] and j < out_ft.shape[-1]:
                out_ft[:, :, :, j] = xqkv_ft[:, :, :, i]

        # Inverse FFT to convert back to time domain
        out = torch.fft.irfft(out_ft / (self.in_channels * self.out_channels), n=L, dim=-1)  # [B, H, E, L]

        # Permute back to original shape [B, L, H, E]
        out = out.permute(0, 3, 1, 2)

        return out, None  # Returning None for compatibility


