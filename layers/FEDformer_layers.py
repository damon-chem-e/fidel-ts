"""
FEDformer Layers: Frequency Enhanced Decomposed Transformer Components

This module implements the core components of FEDformer:
1. Series Decomposition (Seasonal-Trend)
2. Fourier Block (Frequency-Enhanced Self-Attention)
3. Fourier Cross Attention (Frequency-Enhanced Cross-Attention)
4. MultiWavelet Transform and Cross Attention
5. FEDformer Encoder/Decoder Layers

Reference: Zhou et al., "FEDformer: Frequency Enhanced Decomposed Transformer
for Long-term Series Forecasting" (ICML 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple
from torch import Tensor


# =============================================================================
# Frequency Mode Selection
# =============================================================================

def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    Select frequency modes for sparse frequency-domain operations.
    
    Args:
        seq_len: Sequence length (determines available frequencies)
        modes: Number of modes to select
        mode_select_method: 'random' samples randomly, 'low' takes lowest frequencies
        
    Returns:
        list: Sorted indices of selected frequency modes
    """
    # Maximum number of modes is half the sequence length (Nyquist)
    modes = min(modes, seq_len // 2)
    
    if mode_select_method == 'random':
        # Randomly sample frequency indices
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        # Take lowest frequency modes (most energy typically)
        index = list(range(0, modes))
    
    # Sort indices for consistent ordering
    index.sort()
    return index


# =============================================================================
# Series Decomposition Components
# =============================================================================

class moving_avg(nn.Module):
    """
    Moving average block to extract trend from time series.
    
    Uses symmetric padding to avoid boundary effects.
    """
    
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, L, C]
        # Symmetric padding on both ends
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        
        # Apply pooling: permute to [B, C, L] for AvgPool1d
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block: separates time series into seasonal and trend.
    
    Trend is extracted via moving average, seasonal is the residual.
    """
    
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        # Extract trend via moving average
        moving_mean = self.moving_avg(x)
        # Seasonal is the residual
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Multi-scale series decomposition: combines multiple moving average scales.
    
    Uses learned weights to combine different scale decompositions.
    """
    
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        # Create moving average for each kernel size
        self.moving_avg = nn.ModuleList([
            moving_avg(kernel, stride=1) for kernel in kernel_size
        ])
        # Learnable weights for combining scales
        self.layer = nn.Linear(1, len(kernel_size))

    def forward(self, x):
        # Compute moving average at each scale
        moving_mean = []
        for func in self.moving_avg:
            moving_avg_result = func(x)
            moving_mean.append(moving_avg_result.unsqueeze(-1))
        
        # Combine with learned weights
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(
            moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), 
            dim=-1
        )
        
        res = x - moving_mean
        return res, moving_mean


class my_Layernorm(nn.Module):
    """
    Special layernorm for seasonal component (removes mean bias).
    """
    
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        # Remove mean bias (important for seasonal component)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


# =============================================================================
# Fourier Block (Self-Attention Replacement)
# =============================================================================

class FourierBlock(nn.Module):
    """
    Frequency Enhanced Block for self-attention.
    
    Performs representation learning in the frequency domain using FFT,
    achieving O(N log N) complexity instead of O(N²) for standard attention.
    
    Args:
        in_channels: Input channel dimension (d_model // n_heads)
        out_channels: Output channel dimension
        seq_len: Sequence length (for determining frequency modes)
        modes: Number of frequency modes to keep
        mode_select_method: 'random' or 'low' frequency selection
    """
    
    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        
        # Select frequency modes to operate on
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        
        # Learnable complex weights for frequency transformation
        # Shape: [n_heads, in_channels//n_heads, out_channels//n_heads, n_modes]
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        """
        Complex multiplication in frequency domain.
        
        Args:
            input: [batch, in_channel, x]
            weights: [in_channel, out_channel, x]
            
        Returns:
            [batch, out_channel, x]
        """
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # Input: q, k, v all have shape [B, L, H, E]
        B, L, H, E = q.shape
        
        # Permute to [B, H, E, L] for FFT over time dimension
        x = q.permute(0, 2, 3, 1)
        
        # FFT to frequency domain
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Sparse frequency operation on selected modes only
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            if i < x_ft.shape[-1]:
                out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        
        # Inverse FFT back to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        
        return (x, None)


# =============================================================================
# Fourier Cross Attention
# =============================================================================

class FourierCrossAttention(nn.Module):
    """
    Frequency Enhanced Cross Attention for encoder-decoder interaction.
    
    Performs cross-attention in the frequency domain between query and key/value
    sequences of potentially different lengths.
    
    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        seq_len_q: Query sequence length
        seq_len_kv: Key/Value sequence length
        modes: Number of frequency modes
        mode_select_method: 'random' or 'low' frequency selection
        activation: 'tanh' or 'softmax' for attention weights
    """
    
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, 
                 modes=64, mode_select_method='random', activation='tanh'):
        super(FourierCrossAttention, self).__init__()
        
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Select frequency modes for query and key/value
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)

        # Learnable weights for frequency transformation
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index_q), dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        """Complex multiplication."""
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # Input shapes: [B, L, H, E]
        B, L, H, E = q.shape
        
        # Permute to [B, H, E, L] for FFT
        xq = q.permute(0, 2, 3, 1)
        xk = k.permute(0, 2, 3, 1)
        # Note: xv not used directly - cross attention uses xk for values

        # FFT to frequency domain
        xq_ft = torch.fft.rfft(xq, dim=-1)
        xk_ft = torch.fft.rfft(xk, dim=-1)

        # Extract selected modes for query
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            if j < xq_ft.shape[-1]:
                xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        
        # Extract selected modes for key
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_kv):
            if j < xk_ft.shape[-1]:
                xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        # Cross-attention in frequency domain
        xqk_ft = torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_)
        
        # Apply activation
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise ValueError(f'{self.activation} activation not implemented')
        
        # Aggregate values
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        
        # Place back into full frequency spectrum
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            if j < out_ft.shape[-1]:
                out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        
        # Inverse FFT to time domain
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
        
        return (out, None)


# =============================================================================
# AutoCorrelation Layer (Wrapper for Fourier operations)
# =============================================================================

class AutoCorrelationLayer(nn.Module):
    """
    Wrapper layer that applies Fourier-based attention with Q/K/V projections.
    
    Matches the interface expected by FEDformer's encoder/decoder layers.
    """
    
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super(AutoCorrelationLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project Q, K, V and reshape for multi-head
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # Apply Fourier-based correlation
        out, attn = self.inner_correlation(queries, keys, values, attn_mask)

        # Reshape and project output
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


# =============================================================================
# FEDformer Encoder Layer
# =============================================================================

class FEDformerEncoderLayer(nn.Module):
    """
    FEDformer encoder layer with progressive decomposition.
    
    Structure:
    1. Frequency-enhanced attention + residual
    2. Decomposition (extract and discard trend)
    3. Feed-forward network
    4. Decomposition (extract and discard trend)
    """
    
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(FEDformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        
        # Decomposition layers
        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # 1. Attention with residual
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        
        # 2. First decomposition (discard trend, keep seasonal)
        x, _ = self.decomp1(x)
        
        # 3. Feed-forward network
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        # 4. Second decomposition
        res, _ = self.decomp2(x + y)
        
        return res, attn


# =============================================================================
# FEDformer Decoder Layer
# =============================================================================

class FEDformerDecoderLayer(nn.Module):
    """
    FEDformer decoder layer with progressive decomposition and trend accumulation.
    
    Structure:
    1. Self-attention + decomposition → trend1
    2. Cross-attention + decomposition → trend2
    3. FFN + decomposition → trend3
    4. Accumulate trends
    """
    
    def __init__(self, self_attention, cross_attention, d_model, c_out, 
                 d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(FEDformerDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        
        # Decomposition layers
        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
            self.decomp3 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)
            self.decomp3 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        # Trend projection to output channels
        self.projection = nn.Conv1d(
            in_channels=d_model, out_channels=c_out, 
            kernel_size=3, stride=1, padding=1,
            padding_mode='circular', bias=False
        )
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # 1. Self-attention + decomposition
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)
        
        # 2. Cross-attention + decomposition
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)
        
        # 3. FFN + decomposition
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        # 4. Accumulate and project trends
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        
        return x, residual_trend


# =============================================================================
# FEDformer Encoder
# =============================================================================

class FEDformerEncoder(nn.Module):
    """
    FEDformer encoder: stacks multiple encoder layers.
    """
    
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(FEDformerEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


# =============================================================================
# FEDformer Decoder
# =============================================================================

class FEDformerDecoder(nn.Module):
    """
    FEDformer decoder: stacks decoder layers and accumulates trends.
    """
    
    def __init__(self, layers, norm_layer=None, projection=None):
        super(FEDformerDecoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        # Accumulate trends through all layers
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
            
        return x, trend


# =============================================================================
# Wavelet Filter Utilities
# =============================================================================

def legendreDer(k, x):
    """Compute derivative of Legendre polynomial."""
    from scipy.special import eval_legendre
    
    def _legendre(k, x):
        return (2 * k + 1) * eval_legendre(k, x)
    
    out = 0
    for i in np.arange(k - 1, -1, -2):
        out += _legendre(i, x)
    return out


def phi_(phi_c, x, lb=0, ub=1):
    """Evaluate polynomial basis function with support constraints."""
    mask = np.logical_or(x < lb, x > ub) * 1.0
    return np.polynomial.polynomial.Polynomial(phi_c)(x) * (1 - mask)


def get_phi_psi(k, base):
    """
    Compute wavelet basis functions (phi and psi).
    
    Args:
        k: Number of wavelet coefficients
        base: 'legendre' or 'chebyshev'
        
    Returns:
        phi, psi1, psi2: Basis functions
    """
    from functools import partial
    from sympy import Poly, legendre, Symbol, chebyshevt
    
    x = Symbol('x')
    phi_coeff = np.zeros((k, k))
    phi_2x_coeff = np.zeros((k, k))
    
    if base == 'legendre':
        for ki in range(k):
            coeff_ = Poly(legendre(ki, 2 * x - 1), x).all_coeffs()
            phi_coeff[ki, :ki + 1] = np.flip(np.sqrt(2 * ki + 1) * np.array(coeff_).astype(np.float64))
            coeff_ = Poly(legendre(ki, 4 * x - 1), x).all_coeffs()
            phi_2x_coeff[ki, :ki + 1] = np.flip(np.sqrt(2) * np.sqrt(2 * ki + 1) * np.array(coeff_).astype(np.float64))

        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))
        
        for ki in range(k):
            psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
            for i in range(k):
                a = phi_2x_coeff[ki, :ki + 1]
                b = phi_coeff[i, :i + 1]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_) < 1e-8] = 0
                proj_ = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()
                psi1_coeff[ki, :] -= proj_ * phi_coeff[i, :]
                psi2_coeff[ki, :] -= proj_ * phi_coeff[i, :]
            
            for j in range(ki):
                a = phi_2x_coeff[ki, :ki + 1]
                b = psi1_coeff[j, :]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_) < 1e-8] = 0
                proj_ = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()
                psi1_coeff[ki, :] -= proj_ * psi1_coeff[j, :]
                psi2_coeff[ki, :] -= proj_ * psi2_coeff[j, :]

            a = psi1_coeff[ki, :]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_) < 1e-8] = 0
            norm1 = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()

            a = psi2_coeff[ki, :]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_) < 1e-8] = 0
            norm2 = (prod_ * 1 / (np.arange(len(prod_)) + 1) * (1 - np.power(0.5, 1 + np.arange(len(prod_))))).sum()
            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki, :] /= norm_
            psi2_coeff[ki, :] /= norm_
            psi1_coeff[np.abs(psi1_coeff) < 1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff) < 1e-8] = 0

        phi = [np.poly1d(np.flip(phi_coeff[i, :])) for i in range(k)]
        psi1 = [np.poly1d(np.flip(psi1_coeff[i, :])) for i in range(k)]
        psi2 = [np.poly1d(np.flip(psi2_coeff[i, :])) for i in range(k)]

    elif base == 'chebyshev':
        for ki in range(k):
            if ki == 0:
                phi_coeff[ki, :ki + 1] = np.sqrt(2 / np.pi)
                phi_2x_coeff[ki, :ki + 1] = np.sqrt(2 / np.pi) * np.sqrt(2)
            else:
                coeff_ = Poly(chebyshevt(ki, 2 * x - 1), x).all_coeffs()
                phi_coeff[ki, :ki + 1] = np.flip(2 / np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))
                coeff_ = Poly(chebyshevt(ki, 4 * x - 1), x).all_coeffs()
                phi_2x_coeff[ki, :ki + 1] = np.flip(np.sqrt(2) * 2 / np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))

        phi = [partial(phi_, phi_coeff[i, :]) for i in range(k)]

        x = Symbol('x')
        kUse = 2 * k
        roots = Poly(chebyshevt(kUse, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        wm = np.pi / kUse / 2

        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))

        psi1 = [[] for _ in range(k)]
        psi2 = [[] for _ in range(k)]

        for ki in range(k):
            psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
            for i in range(k):
                proj_ = (wm * phi[i](x_m) * np.sqrt(2) * phi[ki](2 * x_m)).sum()
                psi1_coeff[ki, :] -= proj_ * phi_coeff[i, :]
                psi2_coeff[ki, :] -= proj_ * phi_coeff[i, :]

            for j in range(ki):
                proj_ = (wm * psi1[j](x_m) * np.sqrt(2) * phi[ki](2 * x_m)).sum()
                psi1_coeff[ki, :] -= proj_ * psi1_coeff[j, :]
                psi2_coeff[ki, :] -= proj_ * psi2_coeff[j, :]

            psi1[ki] = partial(phi_, psi1_coeff[ki, :], lb=0, ub=0.5)
            psi2[ki] = partial(phi_, psi2_coeff[ki, :], lb=0.5, ub=1)

            norm1 = (wm * psi1[ki](x_m) * psi1[ki](x_m)).sum()
            norm2 = (wm * psi2[ki](x_m) * psi2[ki](x_m)).sum()

            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki, :] /= norm_
            psi2_coeff[ki, :] /= norm_
            psi1_coeff[np.abs(psi1_coeff) < 1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff) < 1e-8] = 0

            psi1[ki] = partial(phi_, psi1_coeff[ki, :], lb=0, ub=0.5 + 1e-16)
            psi2[ki] = partial(phi_, psi2_coeff[ki, :], lb=0.5 + 1e-16, ub=1)

    return phi, psi1, psi2


def get_filter(base, k):
    """
    Compute wavelet filter matrices.
    
    Args:
        base: 'legendre' or 'chebyshev'
        k: Number of wavelet coefficients
        
    Returns:
        H0, H1, G0, G1, PHI0, PHI1: Filter matrices
    """
    from sympy import Poly, legendre, Symbol, chebyshevt
    from scipy.special import eval_legendre
    
    def psi(psi1, psi2, i, inp):
        mask = (inp <= 0.5) * 1.0
        return psi1[i](inp) * mask + psi2[i](inp) * (1 - mask)

    if base not in ['legendre', 'chebyshev']:
        raise Exception('Base not supported')

    x = Symbol('x')
    H0 = np.zeros((k, k))
    H1 = np.zeros((k, k))
    G0 = np.zeros((k, k))
    G1 = np.zeros((k, k))
    PHI0 = np.zeros((k, k))
    PHI1 = np.zeros((k, k))
    
    phi, psi1, psi2 = get_phi_psi(k, base)
    
    if base == 'legendre':
        roots = Poly(legendre(k, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        wm = 1 / k / legendreDer(k, 2 * x_m - 1) / eval_legendre(k - 1, 2 * x_m - 1)

        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki](x_m / 2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m / 2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki]((x_m + 1) / 2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m + 1) / 2) * phi[kpi](x_m)).sum()

        PHI0 = np.eye(k)
        PHI1 = np.eye(k)

    elif base == 'chebyshev':
        x = Symbol('x')
        kUse = 2 * k
        roots = Poly(chebyshevt(kUse, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        wm = np.pi / kUse / 2

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

    H0[np.abs(H0) < 1e-8] = 0
    H1[np.abs(H1) < 1e-8] = 0
    G0[np.abs(G0) < 1e-8] = 0
    G1[np.abs(G1) < 1e-8] = 0

    return H0, H1, G0, G1, PHI0, PHI1


# =============================================================================
# Wavelet Fourier Cross Attention (used inside MultiWaveletCross)
# =============================================================================

class FourierCrossAttentionW(nn.Module):
    """
    Fourier cross attention used within MultiWavelet decomposition.
    
    Operates on wavelet-decomposed signals in frequency domain.
    """
    
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, 
                 modes=16, activation='tanh', mode_select_method='random'):
        super(FourierCrossAttentionW, self).__init__()
        print('cross fourier correlation used!')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes
        self.activation = activation

    def forward(self, q, k, v, mask):
        # Input shape: [B, L, E, H] (different from standard Fourier attention)
        B, L, E, H = q.shape

        xq = q.permute(0, 3, 2, 1)  # [B, H, E, L]
        xk = k.permute(0, 3, 2, 1)
        xv = v.permute(0, 3, 2, 1)
        
        # Dynamically select modes based on input length
        self.index_q = list(range(0, min(int(L // 2), self.modes1)))
        self.index_k_v = list(range(0, min(int(xv.shape[3] // 2), self.modes1)))

        # Compute Fourier coefficients for query
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]

        # Compute Fourier coefficients for key
        xk_ft_ = torch.zeros(B, H, E, len(self.index_k_v), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_k_v):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]
        
        # Cross attention in frequency domain
        xqk_ft = torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_)
        
        # Apply activation
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} activation function is not implemented'.format(self.activation))
        
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = xqkv_ft
        
        # Place back into full spectrum
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]

        # Inverse FFT to time domain
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1)).permute(0, 3, 2, 1)
        # Output shape: [B, L, H, E]
        return (out, None)


# =============================================================================
# Sparse Kernel FT for Wavelet Transform
# =============================================================================

class sparseKernelFT1d(nn.Module):
    """
    Sparse kernel in Fourier domain for 1D wavelet transform.
    
    Performs learnable frequency-domain operations on wavelet coefficients.
    """
    
    def __init__(self, k, alpha, c=1, nl=1, initializer=None, **kwargs):
        super(sparseKernelFT1d, self).__init__()
        self.modes1 = alpha
        self.scale = (1 / (c * k * c * k))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(c * k, c * k, self.modes1, dtype=torch.cfloat)
        )
        self.weights1.requires_grad = True
        self.k = k

    def compl_mul1d(self, x, weights):
        """Complex multiplication: (batch, in, x), (in, out, x) -> (batch, out, x)"""
        return torch.einsum("bix,iox->box", x, weights)

    def forward(self, x):
        B, N, c, k = x.shape  # (B, N, c, k)
        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)
        x_fft = torch.fft.rfft(x)
        
        # Multiply relevant Fourier modes
        num_modes = min(self.modes1, N // 2 + 1)
        out_ft = torch.zeros(B, c * k, N // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :num_modes] = self.compl_mul1d(x_fft[:, :, :num_modes], self.weights1[:, :, :num_modes])
        
        x = torch.fft.irfft(out_ft, n=N)
        x = x.permute(0, 2, 1).view(B, N, c, k)
        return x


# =============================================================================
# MWT_CZ1d: Multi-Wavelet Transform 1D Component
# =============================================================================

class MWT_CZ1d(nn.Module):
    """
    Multi-Wavelet Transform 1D component.
    
    Performs hierarchical wavelet decomposition with learnable frequency operations.
    """
    
    def __init__(self, k=3, alpha=64, L=0, c=1, base='legendre', initializer=None, **kwargs):
        super(MWT_CZ1d, self).__init__()
        self.k = k
        self.L = L
        
        # Get wavelet filters
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        # Learnable sparse kernels
        self.A = sparseKernelFT1d(k, alpha, c)
        self.B = sparseKernelFT1d(k, alpha, c)
        self.C = sparseKernelFT1d(k, alpha, c)
        self.T0 = nn.Linear(k, k)

        # Register filter buffers
        self.register_buffer('ec_s', torch.Tensor(np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(np.concatenate((G0.T, G1.T), axis=0)))
        self.register_buffer('rc_e', torch.Tensor(np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(np.concatenate((H1r, G1r), axis=0)))

    def forward(self, x):
        B, N, c, k = x.shape
        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        
        # Pad to power of 2
        extra_x = x[:, 0:nl - N, :, :]
        x = torch.cat([x, extra_x], 1)
        
        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])
        
        # Decompose
        for i in range(ns - self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        
        # Coarsest scale transform
        x = self.T0(x)

        # Reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)
        
        x = x[:, :N, :, :]
        return x

    def wavelet_transform(self, x):
        """Forward wavelet transform step."""
        xa = torch.cat([x[:, ::2, :, :], x[:, 1::2, :, :]], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        """Inverse wavelet transform step (even-odd interleaving)."""
        B, N, c, ich = x.shape
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)
        x = torch.zeros(B, N * 2, c, self.k, device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x


# =============================================================================
# MultiWavelet Transform (Self-Attention Replacement)
# =============================================================================

class MultiWaveletTransform(nn.Module):
    """
    1D Multi-Wavelet Transform block for self-attention replacement.
    
    Uses multi-wavelet decomposition for efficient sequence representation.
    
    Args:
        ich: Input channel dimension (d_model)
        k: Number of wavelet coefficients (default: 8)
        alpha: Frequency resolution (default: 16)
        c: Number of channels for wavelet operation (default: 128)
        nCZ: Number of MWT_CZ layers (default: 1)
        L: Wavelet decomposition level (default: 0)
        base: Wavelet base: 'legendre' or 'chebyshev'
        attention_dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(self, ich=1, k=8, alpha=16, c=128, nCZ=1, L=0, 
                 base='legendre', attention_dropout=0.1):
        super(MultiWaveletTransform, self).__init__()
        print('base', base)
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ
        self.Lk0 = nn.Linear(ich, c * k)
        self.Lk1 = nn.Linear(c * k, ich)
        self.ich = ich
        self.MWT_CZ = nn.ModuleList(MWT_CZ1d(k, alpha, L, c, base) for i in range(nCZ))

    def forward(self, queries, keys, values, attn_mask):
        # Input shape: [B, L, H, E]
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        # Pad/truncate to match lengths
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]
        
        # Reshape values
        values = values.view(B, L, -1)

        # Project to wavelet space
        V = self.Lk0(values).view(B, L, self.c, -1)
        
        # Apply MWT_CZ layers
        for i in range(self.nCZ):
            V = self.MWT_CZ[i](V)
            if i < self.nCZ - 1:
                V = F.relu(V)

        # Project back
        V = self.Lk1(V.view(B, L, -1))
        V = V.view(B, L, -1, D)
        
        return (V.contiguous(), None)


# =============================================================================
# MultiWavelet Cross Attention
# =============================================================================

class MultiWaveletCross(nn.Module):
    """
    1D Multi-Wavelet Cross Attention layer.
    
    Performs cross-attention using multi-wavelet decomposition for
    encoder-decoder interaction.
    
    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        seq_len_q: Query sequence length
        seq_len_kv: Key/Value sequence length
        modes: Number of frequency modes
        c: Wavelet channels (default: 64)
        k: Number of wavelet coefficients (default: 8)
        ich: Inner channel dimension (default: 512)
        L: Wavelet decomposition level (default: 0)
        base: Wavelet base: 'legendre' or 'chebyshev'
        mode_select_method: Mode selection method (default: 'random')
        activation: Activation for cross attention (default: 'tanh')
    """
    
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes, 
                 c=64, k=8, ich=512, L=0, base='legendre',
                 mode_select_method='random', initializer=None, activation='tanh', **kwargs):
        super(MultiWaveletCross, self).__init__()
        print('base', base)

        self.c = c
        self.k = k
        self.L = L
        
        # Get wavelet filters
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        # Cross attention layers for different wavelet levels
        self.attn1 = FourierCrossAttentionW(
            in_channels=in_channels, out_channels=out_channels, 
            seq_len_q=seq_len_q, seq_len_kv=seq_len_kv, 
            modes=modes, activation=activation,
            mode_select_method=mode_select_method
        )
        self.attn2 = FourierCrossAttentionW(
            in_channels=in_channels, out_channels=out_channels,
            seq_len_q=seq_len_q, seq_len_kv=seq_len_kv,
            modes=modes, activation=activation,
            mode_select_method=mode_select_method
        )
        self.attn3 = FourierCrossAttentionW(
            in_channels=in_channels, out_channels=out_channels,
            seq_len_q=seq_len_q, seq_len_kv=seq_len_kv,
            modes=modes, activation=activation,
            mode_select_method=mode_select_method
        )
        self.attn4 = FourierCrossAttentionW(
            in_channels=in_channels, out_channels=out_channels,
            seq_len_q=seq_len_q, seq_len_kv=seq_len_kv,
            modes=modes, activation=activation,
            mode_select_method=mode_select_method
        )
        
        self.T0 = nn.Linear(k, k)
        
        # Register filter buffers
        self.register_buffer('ec_s', torch.Tensor(np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(np.concatenate((G0.T, G1.T), axis=0)))
        self.register_buffer('rc_e', torch.Tensor(np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(np.concatenate((H1r, G1r), axis=0)))

        # Projections
        self.Lk = nn.Linear(ich, c * k)
        self.Lq = nn.Linear(ich, c * k)
        self.Lv = nn.Linear(ich, c * k)
        self.out = nn.Linear(c * k, ich)
        self.modes1 = modes

    def forward(self, q, k, v, mask=None):
        # Input shape: [B, N, H, E]
        B, N, H, E = q.shape
        _, S, _, _ = k.shape

        # Reshape and project
        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)
        v = v.view(v.shape[0], v.shape[1], -1)
        
        q = self.Lq(q).view(q.shape[0], q.shape[1], self.c, self.k)
        k = self.Lk(k).view(k.shape[0], k.shape[1], self.c, self.k)
        v = self.Lv(v).view(v.shape[0], v.shape[1], self.c, self.k)

        # Pad/truncate
        if N > S:
            zeros = torch.zeros_like(q[:, :(N - S), :]).float()
            v = torch.cat([v, zeros], dim=1)
            k = torch.cat([k, zeros], dim=1)
        else:
            v = v[:, :N, :, :]
            k = k[:, :N, :, :]

        # Pad to power of 2
        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_q = q[:, 0:nl - N, :, :]
        extra_k = k[:, 0:nl - N, :, :]
        extra_v = v[:, 0:nl - N, :, :]
        q = torch.cat([q, extra_q], 1)
        k = torch.cat([k, extra_k], 1)
        v = torch.cat([v, extra_v], 1)

        # Initialize lists for wavelet decomposition
        Ud_q = torch.jit.annotate(List[Tuple[Tensor, Tensor]], [])
        Ud_k = torch.jit.annotate(List[Tuple[Tensor, Tensor]], [])
        Ud_v = torch.jit.annotate(List[Tuple[Tensor, Tensor]], [])
        Us_q = torch.jit.annotate(List[Tensor], [])
        Us_k = torch.jit.annotate(List[Tensor], [])
        Us_v = torch.jit.annotate(List[Tensor], [])
        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

        # Decompose q, k, v
        for i in range(ns - self.L):
            d, q = self.wavelet_transform(q)
            Ud_q += [(d, q)]
            Us_q += [d]
        for i in range(ns - self.L):
            d, k = self.wavelet_transform(k)
            Ud_k += [(d, k)]
            Us_k += [d]
        for i in range(ns - self.L):
            d, v = self.wavelet_transform(v)
            Ud_v += [(d, v)]
            Us_v += [d]

        # Cross attention at each level
        for i in range(ns - self.L):
            dk, sk = Ud_k[i], Us_k[i]
            dq, sq = Ud_q[i], Us_q[i]
            dv, sv = Ud_v[i], Us_v[i]
            Ud += [self.attn1(dq[0], dk[0], dv[0], mask)[0] + self.attn2(dq[1], dk[1], dv[1], mask)[0]]
            Us += [self.attn3(sq, sk, sv, mask)[0]]
        
        v = self.attn4(q, k, v, mask)[0]

        # Reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            v = v + Us[i]
            v = torch.cat((v, Ud[i]), -1)
            v = self.evenOdd(v)
        
        v = self.out(v[:, :N, :, :].contiguous().view(B, N, -1))
        return (v.contiguous(), None)

    def wavelet_transform(self, x):
        """Forward wavelet transform step."""
        xa = torch.cat([x[:, ::2, :, :], x[:, 1::2, :, :]], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        """Inverse wavelet transform step."""
        B, N, c, ich = x.shape
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)
        x = torch.zeros(B, N * 2, c, self.k, device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x

