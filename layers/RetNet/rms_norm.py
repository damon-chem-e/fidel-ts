"""
Root Mean Square Layer Normalization for RetNet.

RMSNorm is more efficient than LayerNorm as it doesn't require
mean computation, only root mean square normalization.

Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Normalizes inputs by their root mean square, optionally with
    learnable scale parameters. More efficient than LayerNorm as
    it doesn't compute mean for centering.
    
    Formula:
        y = x / sqrt(mean(x^2) + eps) * weight (if elementwise_affine)
    
    Args:
        dim: Feature dimension to normalize over
        eps: Small constant for numerical stability (default: 1e-6)
        elementwise_affine: If True, learn per-element scale (default: True)
    """
    
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        # Learnable scale parameter (no bias in RMSNorm)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute RMS normalization.
        
        Args:
            x: Input tensor [..., dim]
            
        Returns:
            Normalized tensor [..., dim]
        """
        # Compute root mean square along last dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [..., dim]
            
        Returns:
            RMS-normalized tensor [..., dim]
        """
        # Normalize
        output = self._norm(x.float()).type_as(x)
        
        # Apply learnable scale if enabled
        if self.weight is not None:
            output = output * self.weight
        
        return output
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

