"""
Utility layers for LeRet model.

Provides common operations used by LeRet including:
- Transpose helper module
- Activation function getter
- Moving average for series decomposition
- Series decomposition block
- Positional encoding variants

Many utilities are reused from PatchTST_layers.py with minor adaptations.
"""

__all__ = [
    'Transpose', 
    'get_activation_fn', 
    'moving_avg', 
    'series_decomp',
    'PositionalEncoding',
    'SinCosPosEncoding', 
    'Coord2dPosEncoding',
    'Coord1dPosEncoding',
    'positional_encoding'
]

import torch
from torch import nn
import math


class Transpose(nn.Module):
    """
    Transpose layer for dimension reordering.
    
    Useful for inserting transposes in nn.Sequential.
    
    Args:
        *dims: Dimensions to transpose
        contiguous: Whether to make result contiguous
    """
    
    def __init__(self, *dims, contiguous: bool = False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


def get_activation_fn(activation: str):
    """
    Get activation function by name.
    
    Args:
        activation: Activation name or callable
        
    Returns:
        Activation module
        
    Raises:
        ValueError: If activation name not recognized
    """
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    elif activation.lower() == "silu" or activation.lower() == "swish":
        return nn.SiLU()
    raise ValueError(
        f'{activation} is not available. '
        f'You can use "relu", "gelu", "silu", or a callable'
    )


# =============================================================================
# Series Decomposition
# =============================================================================

class moving_avg(nn.Module):
    """
    Moving average block to extract trend from time series.
    
    Uses average pooling with symmetric padding to compute
    a smoothed version of the input.
    
    Args:
        kernel_size: Size of the averaging window
        stride: Stride for average pooling (typically 1)
    """
    
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute moving average.
        
        Args:
            x: Input tensor [batch, seq_len, channels]
            
        Returns:
            Smoothed tensor [batch, seq_len, channels]
        """
        # Symmetric padding by replicating boundary values
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        
        # Apply average pooling (transpose for channel-last format)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block for trend-residual separation.
    
    Decomposes input into trend and residual (seasonal) components
    using moving average filtering.
    
    Args:
        kernel_size: Size of the moving average window
    """
    
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
    
    def forward(self, x: torch.Tensor):
        """
        Decompose series into trend and residual.
        
        Args:
            x: Input tensor [batch, seq_len, channels]
            
        Returns:
            Tuple of:
                - residual: x - trend [batch, seq_len, channels]
                - trend: moving average [batch, seq_len, channels]
        """
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean


# =============================================================================
# Positional Encodings
# =============================================================================

def pv(message: str, verbose: bool = False):
    """Print message if verbose mode is enabled."""
    if verbose:
        print(message)


def PositionalEncoding(q_len: int, d_model: int, normalize: bool = True) -> torch.Tensor:
    """
    Standard sinusoidal positional encoding.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    
    Args:
        q_len: Sequence length
        d_model: Model dimension
        normalize: Whether to normalize encoding
        
    Returns:
        Positional encoding tensor [q_len, d_model]
    """
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    
    return pe


# Alias for sinusoidal encoding
SinCosPosEncoding = PositionalEncoding


def Coord2dPosEncoding(
    q_len: int, 
    d_model: int, 
    exponential: bool = False, 
    normalize: bool = True,
    eps: float = 1e-3, 
    verbose: bool = False
) -> torch.Tensor:
    """
    2D coordinate-based positional encoding.
    
    Creates a 2D grid of positions that can be exponentially scaled.
    
    Args:
        q_len: Sequence length
        d_model: Model dimension
        exponential: Whether to use exponential scaling
        normalize: Whether to normalize encoding
        eps: Tolerance for mean centering
        verbose: Print optimization progress
        
    Returns:
        Positional encoding tensor [q_len, d_model]
    """
    x = 0.5 if exponential else 1
    
    # Iteratively find exponent that centers the mean
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * \
              (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        
        if abs(cpe.mean()) <= eps:
            break
        elif cpe.mean() > eps:
            x += 0.001
        else:
            x -= 0.001
    
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    
    return cpe


def Coord1dPosEncoding(
    q_len: int, 
    exponential: bool = False, 
    normalize: bool = True
) -> torch.Tensor:
    """
    1D coordinate-based positional encoding.
    
    Creates a simple linear or exponential ramp encoding.
    
    Args:
        q_len: Sequence length
        exponential: Whether to use sqrt scaling
        normalize: Whether to normalize encoding
        
    Returns:
        Positional encoding tensor [q_len, 1]
    """
    power = 0.5 if exponential else 1
    cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** power) - 1
    
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    
    return cpe


def positional_encoding(
    pe: str, 
    learn_pe: bool, 
    q_len: int, 
    d_model: int
) -> nn.Parameter:
    """
    Create positional encoding as a learnable or fixed parameter.
    
    Args:
        pe: Encoding type - one of:
            - None: Random uniform initialization (fixed)
            - 'zero': Single column, uniform init
            - 'zeros': Full matrix, uniform init
            - 'normal'/'gauss': Single column, normal init
            - 'uniform': Single column, uniform init
            - 'lin1d': Linear 1D coordinate
            - 'exp1d': Exponential 1D coordinate
            - 'lin2d': Linear 2D coordinate
            - 'exp2d': Exponential 2D coordinate
            - 'sincos': Sinusoidal positional encoding
        learn_pe: Whether encoding should be learnable
        q_len: Sequence length
        d_model: Model dimension
        
    Returns:
        Positional encoding parameter [q_len, d_model] or [q_len, 1]
    """
    if pe is None:
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos':
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(
            f"{pe} is not a valid positional encoding type. "
            f"Available: 'gauss'/'normal', 'zeros', 'zero', 'uniform', "
            f"'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None."
        )
    
    return nn.Parameter(W_pos, requires_grad=learn_pe)

