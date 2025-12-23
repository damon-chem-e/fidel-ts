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
    
    Architecture (per paper):
    - Linear projection: text_dim -> ts_rep_dim
    - Residual connection (always used, with projection if dims differ)
    - Optional: LayerNorm, activation, dropout
    
    Note: Residual connection always used (as per paper recommendation).
    """
    
    def __init__(self, text_dim, ts_rep_dim, 
                 use_layer_norm=True, activation='gelu', dropout=0.1):
        """
        Initialize residual projection block.
        
        Args:
            text_dim: Dimension of input text embeddings
            ts_rep_dim: Dimension of time series representation space
            use_layer_norm: Whether to use LayerNorm
            activation: Activation function ('gelu', 'relu', or None)
            dropout: Dropout rate
        """
        super().__init__()
        self.text_dim = text_dim
        self.ts_rep_dim = ts_rep_dim
        
        # Main projection
        self.projection = nn.Linear(text_dim, ts_rep_dim)
        
        # Residual projection (if dims differ, need to project residual too)
        if text_dim != ts_rep_dim:
            self.residual_proj = nn.Linear(text_dim, ts_rep_dim)
        else:
            self.residual_proj = None
        
        # Optional components
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(ts_rep_dim)
        else:
            self.layer_norm = None
            
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = None
            
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, text_repr):
        """
        Forward pass.
        
        Args:
            text_repr: Text representation [B, text_dim]
        
        Returns:
            projected_repr: [B, ts_rep_dim] - projected to TS representation space
        """
        # Project input
        out = self.projection(text_repr)  # [B, ts_rep_dim]
        
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(text_repr)  # [B, ts_rep_dim]
        else:
            residual = text_repr  # [B, text_dim] = [B, ts_rep_dim] (same dim)
        
        out = out + residual  # Residual connection (always used)
        
        # Optional normalization, activation, dropout
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        
        if self.activation is not None:
            out = self.activation(out)
        
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

