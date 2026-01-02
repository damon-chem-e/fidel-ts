"""
Feedforward networks for RetNet.

Implements Gated Linear Unit (GLU) variants used in RetNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    """
    Gated Linear Unit feedforward network.
    
    Uses a gated activation where one linear projection acts as
    a gate for another, similar to SwiGLU in LLaMA.
    
    Architecture:
        gate = activation(W_gate @ x)
        hidden = W_up @ x  
        output = W_down @ (gate * hidden)
    
    Args:
        embed_dim: Input/output dimension
        ffn_dim: Hidden dimension (typically 4x embed_dim)
        activation_fn: Activation function name ('gelu', 'swish')
        dropout: Dropout rate after activation (default: 0.0)
        activation_dropout: Dropout rate in activation (default: 0.0)
    """
    
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        activation_fn: str = "gelu",
        dropout: float = 0.0,
        activation_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        
        # Select activation function
        if activation_fn == "swish" or activation_fn == "silu":
            self.activation_fn = F.silu
        elif activation_fn == "gelu":
            self.activation_fn = F.gelu
        elif activation_fn == "relu":
            self.activation_fn = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation_fn}")
        
        # Gated linear projections
        # fc1 is the "gate" projection
        self.fc1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        # fc2 is the "up" projection (gets gated)
        self.fc2 = nn.Linear(embed_dim, ffn_dim, bias=False)
        # fc3 is the "down" projection
        self.fc3 = nn.Linear(ffn_dim, embed_dim, bias=False)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(activation_dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GLU.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            
        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        # Compute gate and hidden projections
        gate = self.activation_fn(self.fc1(x))
        hidden = self.fc2(x)
        
        # Apply gating with dropout
        gated = gate * hidden
        gated = self.activation_dropout(gated)
        
        # Project back to embed_dim
        output = self.fc3(gated)
        output = self.dropout(output)
        
        return output
    
    def extra_repr(self) -> str:
        return f'embed_dim={self.embed_dim}, ffn_dim={self.ffn_dim}'

