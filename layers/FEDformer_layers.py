"""
FEDformer Layers: Frequency Enhanced Decomposed Transformer Components

This module implements the core components of FEDformer:
1. Series Decomposition (Seasonal-Trend)
2. Fourier Block (Frequency-Enhanced Self-Attention)
3. Fourier Cross Attention (Frequency-Enhanced Cross-Attention)
4. FEDformer Encoder/Decoder Layers

Reference: Zhou et al., "FEDformer: Frequency Enhanced Decomposed Transformer
for Long-term Series Forecasting" (ICML 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


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
        xv = v.permute(0, 2, 3, 1)

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

