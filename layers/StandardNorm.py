"""
StandardNorm (RevIN) Layer for TimeCMA.

This module implements the reversible instance normalization layer from TimeCMA.
This is functionally identical to RevIN but includes the `non_norm` option
from the original TimeCMA implementation.

Reference:
    TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting
    via Cross-Modality Alignment (AAAI 2025)
"""

import torch
import torch.nn as nn


class Normalize(nn.Module):
    """
    Reversible Instance Normalization for time series data.
    
    Normalizes input data and stores statistics for later denormalization.
    This is the exact implementation from TimeCMA's StandardNorm.py.
    
    Args:
        num_features: Number of features/channels in the input
        eps: Small value added for numerical stability
        affine: If True, learn affine parameters (gamma, beta)
        subtract_last: If True, subtract last value instead of mean
        non_norm: If True, skip normalization entirely (passthrough)
    
    Example:
        >>> norm = Normalize(num_features=7, affine=False)
        >>> x_norm = norm(x, 'norm')      # Normalize and store stats
        >>> x_recon = norm(x_norm, 'denorm')  # Denormalize
        >>> assert torch.allclose(x, x_recon)
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
        subtract_last: bool = False,
        non_norm: bool = False
    ):
        super(Normalize, self).__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        
        # Initialize learnable affine parameters if requested
        if self.affine:
            self._init_params()
    
    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Apply normalization or denormalization.
        
        Args:
            x: Input tensor [B, L, N] where B=batch, L=seq_len, N=channels
            mode: Either 'norm' to normalize or 'denorm' to denormalize
        
        Returns:
            Normalized or denormalized tensor
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Unknown mode: {mode}")
        return x
    
    def _init_params(self):
        """Initialize learnable affine parameters: gamma (weight) and beta (bias)."""
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
    
    def _get_statistics(self, x: torch.Tensor):
        """
        Compute and store normalization statistics.
        
        For [B, L, N] input, computes statistics over dimension L (sequence length).
        """
        # Reduce over all dimensions except first (batch) and last (features)
        dim2reduce = tuple(range(1, x.ndim - 1))
        
        if self.subtract_last:
            # Use last time step as reference
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            # Use mean as reference
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        
        # Always compute standard deviation
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization using stored statistics."""
        # Skip normalization if non_norm is True
        if self.non_norm:
            return x
        
        # Subtract mean or last value
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        
        # Divide by standard deviation
        x = x / self.stdev
        
        # Apply learnable affine transformation if enabled
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        
        return x
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse normalization using stored statistics."""
        # Skip denormalization if non_norm is True
        if self.non_norm:
            return x
        
        # Reverse affine transformation
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        
        # Multiply by standard deviation
        x = x * self.stdev
        
        # Add back mean or last value
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        
        return x

