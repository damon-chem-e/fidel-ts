"""
MM-TSFlib Layers for Fidel-TS.

Implements the text-to-prediction projection and pooling mechanisms
from MM-TSFlib's late fusion approach.

Components:
    - TextToPredsProjection: MLP that projects text embeddings to prediction space
    - TextPooling: Various pooling strategies for aggregating token embeddings
    - normalize_embeddings: Instance normalization for embeddings

Reference:
    MM-TSFlib: https://github.com/AdityaLab/MM-TSFlib
    Time-MMD: https://github.com/AdityaLab/Time-MMD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


class TextToPredsProjection(nn.Module):
    """
    MLP that projects text embeddings to prediction space.
    
    Following MM-TSFlib architecture:
        d_llm → d_llm/reduction_factor → pred_len
    
    This maps LLM token embeddings to a dimension matching predictions,
    allowing direct prediction-level ensemble.
    
    Args:
        d_llm: Dimension of input LLM embeddings (e.g., 768 for BERT)
        pred_len: Prediction horizon length (output dimension)
        reduction_factor: Hidden layer size = d_llm / reduction (default: 8)
        dropout: Dropout rate (default: 0.3)
    
    Input:
        text_emb: [B, L, d_llm] - Token-level embeddings
        
    Output:
        text_proj: [B, L, pred_len] - Projected to prediction dimension
    """
    
    def __init__(
        self,
        d_llm: int,
        pred_len: int,
        reduction_factor: int = 8,
        dropout: float = 0.3
    ):
        """
        Initialize text-to-predictions projection MLP.
        
        Args:
            d_llm: Dimension of input LLM embeddings
            pred_len: Prediction horizon length (output dimension)
            reduction_factor: Hidden dim = d_llm / reduction_factor
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.d_llm = d_llm
        self.pred_len = pred_len
        
        # Compute hidden dimension with minimum bound
        hidden_dim = max(d_llm // reduction_factor, pred_len)
        
        # Two-layer MLP: d_llm -> hidden -> pred_len
        # Following MM-TSFlib: Linear -> ReLU -> Dropout -> Linear
        self.layers = nn.Sequential(
            nn.Linear(d_llm, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len)
        )
    
    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Project text embeddings to prediction dimension.
        
        Args:
            text_emb: [B, L, d_llm] or [B, d_llm] (if already pooled)
        
        Returns:
            text_proj: [B, L, pred_len] or [B, pred_len]
        """
        return self.layers(text_emb)


class TextPooling(nn.Module):
    """
    Pooling layer to aggregate token-level embeddings into a single vector.
    
    Supports multiple pooling strategies from MM-TSFlib:
        - 'avg': Global average pooling
        - 'max': Global max pooling
        - 'min': Global min pooling (via negated max)
        - 'attention': Attention-weighted pooling using TS predictions
    
    Args:
        pool_type: Pooling strategy ('avg', 'max', 'min', 'attention')
    
    Input:
        text_emb: [B, L, D] - Token-level embeddings
        ts_preds: [B, pred_len, C] - TS predictions (only for attention pooling)
        
    Output:
        pooled: [B, D, 1] - Pooled embeddings (with trailing dim for broadcast)
    """
    
    POOL_TYPES = {'avg', 'max', 'min', 'attention'}
    
    def __init__(self, pool_type: Literal['avg', 'max', 'min', 'attention'] = 'avg'):
        """
        Initialize pooling layer.
        
        Args:
            pool_type: Pooling strategy to use
        """
        super().__init__()
        
        # Validate pool type
        if pool_type not in self.POOL_TYPES:
            raise ValueError(
                f"pool_type must be one of {self.POOL_TYPES}, got '{pool_type}'"
            )
        
        self.pool_type = pool_type
    
    def forward(
        self,
        text_emb: torch.Tensor,
        ts_preds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool token embeddings into single vector.
        
        Args:
            text_emb: [B, L, D] - Token-level embeddings
            ts_preds: [B, pred_len, C] - TS predictions (required for attention)
        
        Returns:
            pooled: [B, D, 1] - Pooled embeddings with trailing dimension
        """
        # text_emb shape: [B, L, D]
        
        if self.pool_type == 'avg':
            # Global average pooling: [B, L, D] -> [B, D]
            pooled = F.adaptive_avg_pool1d(
                text_emb.transpose(1, 2),  # [B, D, L]
                1  # Output length
            ).squeeze(2)  # [B, D]
            
        elif self.pool_type == 'max':
            # Global max pooling: [B, L, D] -> [B, D]
            pooled = F.adaptive_max_pool1d(
                text_emb.transpose(1, 2),  # [B, D, L]
                1
            ).squeeze(2)  # [B, D]
            
        elif self.pool_type == 'min':
            # Global min pooling via negated max: [B, L, D] -> [B, D]
            pooled = -F.adaptive_max_pool1d(
                -text_emb.transpose(1, 2),  # [B, D, L]
                1
            ).squeeze(2)  # [B, D]
            
        elif self.pool_type == 'attention':
            # Attention-weighted pooling using TS predictions as query
            if ts_preds is None:
                raise ValueError(
                    "ts_preds required for attention pooling. "
                    "Pass time series predictions to compute attention weights."
                )
            
            # Normalize embeddings for stable attention computation
            # text_emb: [B, L, D], ts_preds: [B, pred_len, C]
            text_norm = F.normalize(text_emb, p=2, dim=2)  # [B, L, D]
            ts_norm = F.normalize(ts_preds, p=2, dim=1)    # [B, pred_len, C]
            
            # Compute attention scores
            # Note: D should equal pred_len after MLP projection
            # [B, L, D] @ [B, D, C] -> [B, L, C]
            attention_scores = torch.bmm(text_norm, ts_norm)
            attention_weights = F.softmax(attention_scores, dim=1)  # [B, L, C]
            
            # Weighted sum: average attention over C dimension for simplicity
            attention_weights_avg = attention_weights.mean(dim=2, keepdim=True)  # [B, L, 1]
            pooled = (text_emb * attention_weights_avg).sum(dim=1)  # [B, D]
        
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")
        
        # Add trailing dimension for broadcast compatibility with [B, pred_len, C]
        return pooled.unsqueeze(-1)  # [B, D, 1]


def normalize_embeddings(emb: torch.Tensor) -> torch.Tensor:
    """
    Instance normalization for embeddings (from MM-TSFlib).
    
    Normalizes each sample independently:
        emb_norm = (emb - mean) / std
    
    This is the 'norm' function from MM-TSFlib exp_long_term_forecasting.py.
    
    Args:
        emb: Input tensor [B, D, 1] or [B, D]
    
    Returns:
        emb_norm: Normalized tensor (same shape as input)
    """
    # Handle both [B, D, 1] and [B, D] inputs
    squeeze_output = emb.dim() == 2
    if squeeze_output:
        emb = emb.unsqueeze(-1)
    
    # Compute mean and std over D dimension (dim=1)
    # Detach mean to match MM-TSFlib behavior
    mean = emb.mean(dim=1, keepdim=True).detach()
    var = torch.var(emb, dim=1, keepdim=True, unbiased=False) + 1e-5
    std = torch.sqrt(var)
    
    # Normalize: (x - mean) / std
    emb_norm = (emb - mean) / std
    
    # Restore original shape if needed
    if squeeze_output:
        emb_norm = emb_norm.squeeze(-1)
    
    return emb_norm
