"""
Layers for ZhangHanBest model.

Contains:
- ResidualProjection: Projects text embeddings to time series representation space
- PredictionHead: Maps fused representation to predictions
"""

import torch
import torch.nn as nn


class ResidualProjection(nn.Module):
    """
    Residual block to project text representation to time series representation space.
    
    Architecture (per Zhang et al. 2025):
    - Main path: text_dim -> hidden_dim -> ts_rep_dim (two-layer MLP)
    - Residual path: text_dim -> ts_rep_dim (single linear, if dims differ)
    - Output: main_path + residual_path
    - Optional: LayerNorm, dropout after residual addition
    
    Default hidden_dim=2048 matches paper: 768 -> 2048 -> 512 + residual 768 -> 512.
    """
    
    def __init__(self, text_dim, ts_rep_dim, hidden_dim=2048,
                 use_layer_norm=True, activation='gelu', dropout=0.1):
        """
        Initialize residual projection block.
        
        Args:
            text_dim: Dimension of input text embeddings (e.g., 768 for BERT)
            ts_rep_dim: Dimension of time series representation space (e.g., 512)
            hidden_dim: Hidden layer dimension in main MLP path (default 2048 per paper)
            use_layer_norm: Whether to use LayerNorm after residual addition
            activation: Activation function for hidden layer ('gelu', 'relu', or None)
            dropout: Dropout rate after residual addition
        """
        super().__init__()
        
        # Validate and convert dimensions to integers (handle cases where configs might pass tuples/other types)
        self.text_dim = int(text_dim) if text_dim is not None else 768
        self.ts_rep_dim = int(ts_rep_dim) if ts_rep_dim is not None else 512
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else 2048
        
        # Validate dimensions are positive integers
        if self.text_dim <= 0 or self.ts_rep_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError(
                f"ResidualProjection requires positive integer dimensions. "
                f"Got text_dim={text_dim} -> {self.text_dim}, "
                f"ts_rep_dim={ts_rep_dim} -> {self.ts_rep_dim}, "
                f"hidden_dim={hidden_dim} -> {self.hidden_dim}"
            )
        
        # Build activation for hidden layer
        if activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        else:
            act_fn = nn.Identity()
        
        # Main projection path: text_dim -> hidden_dim -> ts_rep_dim (two-layer MLP)
        self.main_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),   # 768 -> 2048
            act_fn,                                      # GELU activation
            nn.Linear(self.hidden_dim, self.ts_rep_dim)  # 2048 -> 512
        )
        
        # Residual projection: text_dim -> ts_rep_dim (single linear if dims differ)
        if self.text_dim != self.ts_rep_dim:
            self.residual_proj = nn.Linear(self.text_dim, self.ts_rep_dim)  # 768 -> 512
        else:
            self.residual_proj = None
        
        # Optional components after residual addition
        self.layer_norm = nn.LayerNorm(ts_rep_dim) if use_layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, text_repr):
        """
        Forward pass.
        
        Args:
            text_repr: Text representation [B, text_dim]
        
        Returns:
            projected_repr: [B, ts_rep_dim] - projected to TS representation space
        """
        # Main path: text_dim -> hidden_dim -> ts_rep_dim
        out = self.main_proj(text_repr)  # [B, ts_rep_dim]
        
        # Residual path: text_dim -> ts_rep_dim
        if self.residual_proj is not None:
            residual = self.residual_proj(text_repr)  # [B, ts_rep_dim]
        else:
            residual = text_repr  # [B, text_dim] = [B, ts_rep_dim] (same dim)
        
        # Add residual connection
        out = out + residual
        
        # Optional normalization and dropout after residual
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out


class PredictionHead(nn.Module):
    """
    Prediction head that maps fused representation to predictions.
    
    Input: Aggregated fused representation [B, d_model]
    Output: Predictions [B, pred_len, C]
    """
    
    def __init__(self, d_model, pred_len, n_variates, 
                 use_mlp=True, hidden_dim=None, dropout=0.1):
        """
        Initialize prediction head.
        
        Args:
            d_model: Dimension of fused representation
            pred_len: Prediction horizon length
            n_variates: Number of channels/variates
            use_mlp: Whether to use MLP instead of single linear layer
            hidden_dim: Hidden dimension for MLP (if use_mlp=True)
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.n_variates = n_variates
        
        output_dim = pred_len * n_variates  # Total output dimensions
        
        if use_mlp and hidden_dim is not None:
            # Multi-layer prediction head
            self.head = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            # Simple linear projection
            self.head = nn.Linear(d_model, output_dim)
    
    def forward(self, fused_repr):
        """
        Forward pass.
        
        Args:
            fused_repr: [B, d_model] - aggregated fused representation
            
        Returns:
            predictions: [B, pred_len, n_variates] where n_variates = number of channels
        """
        # [B, d_model] -> [B, pred_len * n_variates]
        pred_flat = self.head(fused_repr)  # [B, pred_len * n_variates]
        
        # Reshape to [B, pred_len, n_variates]
        pred = pred_flat.view(-1, self.pred_len, self.n_variates)
        
        return pred

