"""
RevIN-style Instance Normalization for time series.

This module provides reversible instance normalization that:
- Normalizes input at forward pass using per-sample statistics
- Denormalizes output using stored statistics
- Optionally applies learnable affine transformation

This is critical for Time-LLM to handle varying scales across
different time series while maintaining the ability to produce
outputs in the original scale.

Reference:
    Kim et al., "Reversible Instance Normalization for Accurate 
    Time-Series Forecasting against Distribution Shift" (ICLR 2022)
"""

import torch
import torch.nn as nn


class Normalize(nn.Module):
    """
    Reversible Instance Normalization for time series.
    
    Normalizes input using per-sample statistics (mean, std) computed
    over the time dimension. Statistics are stored for later denormalization.
    
    Args:
        num_features: Number of channels/features (C dimension)
        eps: Small constant for numerical stability in std computation
        affine: If True, apply learnable scale and shift after normalization
    
    Example:
        >>> norm = Normalize(num_features=7)
        >>> x = torch.randn(2, 96, 7)  # [B, T, C]
        >>> x_norm = norm(x, 'norm')
        >>> # ... model processing ...
        >>> x_out = norm(output, 'denorm')  # Back to original scale
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        """
        Initialize Normalize layer.
        
        Args:
            num_features: Number of channels/features to normalize
            eps: Small constant for numerical stability
            affine: If True, learn scale and shift parameters
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        # Learnable affine parameters (optional)
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        
        # Statistics stored per forward pass (not learned)
        # These are set during 'norm' and used during 'denorm'
        self.mean = None
        self.std = None
    
    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Normalize or denormalize input tensor.
        
        Args:
            x: Input tensor with shape [B, T, C]
            mode: 'norm' to normalize, 'denorm' to reverse normalization
        
        Returns:
            Normalized or denormalized tensor with same shape [B, T, C]
        
        Raises:
            ValueError: If mode is not 'norm' or 'denorm'
            RuntimeError: If 'denorm' is called before 'norm'
        """
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'norm' or 'denorm'.")
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute and apply normalization.
        
        Computes per-sample statistics over the time dimension and
        stores them for later denormalization.
        
        Args:
            x: Input tensor [B, T, C]
        
        Returns:
            Normalized tensor [B, T, C]
        """
        # Compute per-sample statistics over time dimension
        # mean: [B, 1, C], std: [B, 1, C]
        self.mean = x.mean(dim=1, keepdim=True)
        self.std = x.std(dim=1, keepdim=True) + self.eps
        
        # Normalize: (x - mean) / std
        x_norm = (x - self.mean) / self.std
        
        # Apply learnable affine transformation if enabled
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
        
        return x_norm
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverse normalization using stored statistics.
        
        Args:
            x: Normalized tensor [B, T, C]
        
        Returns:
            Denormalized tensor [B, T, C] in original scale
        
        Raises:
            RuntimeError: If called before _normalize (no stored stats)
        """
        if self.mean is None or self.std is None:
            raise RuntimeError(
                "Must call forward with mode='norm' before 'denorm'. "
                "No statistics are stored for denormalization."
            )
        
        # Remove learnable affine if enabled
        if self.affine:
            x = (x - self.affine_bias) / self.affine_weight
        
        # Denormalize: x * std + mean
        return x * self.std + self.mean

