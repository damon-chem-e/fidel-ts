"""
Text embedding aggregation methods.

Supports:
- CLS token: Extract first token embedding (default, current behavior)
- Average pooling: Average over all non-padding tokens
- None: Return full sequence (preserve sequence dimension)
"""

import torch
from typing import Tuple, Optional


def aggregate_cls_token(last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Extract CLS token embedding (first token).
    
    Args:
        last_hidden_state: [B, seq_len, hidden_dim] token embeddings
        attention_mask: Optional [B, seq_len] mask (unused, kept for API consistency)
    
    Returns:
        embeddings: [B, hidden_dim] CLS token embeddings
    """
    return last_hidden_state[:, 0, :]


def aggregate_average_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Average pooling over tokens (excluding padding).
    
    Args:
        last_hidden_state: [B, seq_len, hidden_dim] token embeddings
        attention_mask: [B, seq_len] mask (1 for real tokens, 0 for padding)
    
    Returns:
        embeddings: [B, hidden_dim] averaged token embeddings
    """
    # Mask out padding tokens
    masked_embeddings = last_hidden_state * attention_mask.unsqueeze(-1)  # [B, seq_len, hidden_dim]
    
    # Sum over sequence dimension
    sum_embeddings = masked_embeddings.sum(dim=1)  # [B, hidden_dim]
    
    # Count non-padding tokens
    sum_mask = attention_mask.sum(dim=1, keepdim=True)  # [B, 1]
    
    # Avoid division by zero
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    
    # Average
    embeddings = sum_embeddings / sum_mask  # [B, hidden_dim]
    
    return embeddings


def aggregate_none(last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Return full sequence (no aggregation).
    
    Args:
        last_hidden_state: [B, seq_len, hidden_dim] token embeddings
        attention_mask: Optional [B, seq_len] mask (returned as-is for downstream use)
    
    Returns:
        embeddings: [B, seq_len, hidden_dim] full sequence embeddings
        attention_mask: [B, seq_len] attention mask (if provided)
    """
    return last_hidden_state, attention_mask


# Registry of aggregation methods
AGGREGATION_METHODS = {
    'cls': aggregate_cls_token,
    'average': aggregate_average_pooling,
    'none': aggregate_none,
}


def get_aggregation_function(method: str):
    """
    Get aggregation function by name.
    
    Args:
        method: 'cls', 'average', or 'none'
    
    Returns:
        Aggregation function
    
    Raises:
        ValueError: If method is not recognized
    """
    if method not in AGGREGATION_METHODS:
        raise ValueError(
            f"Unknown aggregation method: {method}. "
            f"Must be one of: {list(AGGREGATION_METHODS.keys())}"
        )
    return AGGREGATION_METHODS[method]

