"""
TimeCMA Cross-Modal Alignment Layers.

This module implements the cross-modal attention mechanism from TimeCMA
(Time Series Cross-Modality Alignment) from AAAI 2025.

The cross-modal attention aligns time series embeddings with LLM prompt
embeddings to retrieve 'disentangled and robust' representations.

Reference:
    TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting
    via Cross-Modality Alignment (AAAI 2025)
"""

from typing import Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np


# =============================================================================
# Utility Classes
# =============================================================================

class Transpose(nn.Module):
    """Transpose dimensions of a tensor."""
    
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous
    
    def forward(self, x: Tensor) -> Tensor:
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


def get_activation_fn(activation):
    """Get activation function by name."""
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(
        f'{activation} is not available. You can use "relu", "gelu", or a callable'
    )


# =============================================================================
# Cross-Modal Attention Components
# =============================================================================

class CrossModal(nn.Module):
    """
    Cross-Modal Alignment Layer for TimeCMA.
    
    Aligns time series embeddings (Q) with LLM prompt embeddings (K, V)
    using multi-head attention with optional residual attention.
    
    Args:
        d_model: Model dimension (typically num_nodes/channels)
        n_heads: Number of attention heads
        d_k: Key dimension per head (default: d_model // n_heads)
        d_v: Value dimension per head (default: d_model // n_heads)
        d_ff: Feed-forward hidden dimension
        norm: Normalization type ('LayerNorm' or 'BatchNorm')
        attn_dropout: Dropout rate for attention weights
        dropout: Dropout rate for other components
        activation: Activation function ('gelu' or 'relu')
        res_attention: Whether to use residual attention
        n_layers: Number of encoder layers
        pre_norm: Whether to apply normalization before attention
        store_attn: Whether to store attention weights
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: Optional[int] = None,
        norm: str = 'LayerNorm',
        attn_dropout: float = 0.,
        dropout: float = 0.,
        activation: str = 'gelu',
        res_attention: bool = False,
        n_layers: int = 1,
        pre_norm: bool = False,
        store_attn: bool = False
    ):
        super().__init__()
        
        # Build encoder layers
        self.layers = nn.ModuleList([
            TSTEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                activation=activation,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn
            )
            for _ in range(n_layers)
        ])
        self.res_attention = res_attention
    
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass for cross-modal attention.
        
        Args:
            q: Query tensor (time series features) [B, C, N]
            k: Key tensor (LLM features) [B, E, N]
            v: Value tensor (LLM features) [B, E, N]
            key_padding_mask: Optional padding mask
            attn_mask: Optional attention mask
        
        Returns:
            Cross-modality aligned features [B, C, N]
        """
        scores = None
        
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(
                    q, k, v,
                    prev=scores,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask
                )
            return output
        else:
            for mod in self.layers:
                output = mod(
                    q, k, v,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask
                )
            return output


class TSTEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer for TimeCMA Cross-Modal Attention.
    
    Implements a transformer encoder layer with:
    1. Multi-head cross-attention (Q from TS, KV from LLM)
    2. Position-wise feed-forward network
    3. Residual connections and layer normalization
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_k: Key dimension per head
        d_v: Value dimension per head
        d_ff: Feed-forward hidden dimension
        store_attn: Whether to store attention weights
        norm: Normalization type
        attn_dropout: Dropout for attention
        dropout: Dropout for other components
        bias: Whether to use bias in linear layers
        activation: Activation function
        res_attention: Whether to use residual attention
        pre_norm: Whether to apply normalization before attention
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        store_attn: bool = False,
        norm: str = 'LayerNorm',
        attn_dropout: float = 0.,
        dropout: float = 0.,
        bias: bool = True,
        activation: str = "gelu",
        res_attention: bool = False,
        pre_norm: bool = False
    ):
        super().__init__()
        
        # Validate dimensions
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        
        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention
        )
        
        # Add & Norm (post-attention)
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(d_model),
                Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(d_model)
        
        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias)
        )
        
        # Add & Norm (post-FFN)
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(d_model),
                Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(d_model)
        
        self.pre_norm = pre_norm
        self.store_attn = store_attn
    
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass for encoder layer.
        
        Args:
            q: Query tensor [B, seq_len, d_model]
            k: Key tensor [B, seq_len, d_model]
            v: Value tensor [B, seq_len, d_model]
            prev: Previous attention scores for residual attention
            key_padding_mask: Padding mask
            attn_mask: Attention mask
        
        Returns:
            Output tensor, and optionally attention scores
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            q = self.norm_attn(q)
            k = self.norm_attn(k)
            v = self.norm_attn(v)
        
        # Multi-Head attention
        if self.res_attention:
            q2, attn, scores = self.self_attn(
                q, k, v, prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask
            )
        else:
            q2, attn = self.self_attn(
                q, k, v,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask
            )
        
        if self.store_attn:
            self.attn = attn
        
        # Add & Norm
        q = q + self.dropout_attn(q2)  # Residual connection with dropout
        if not self.pre_norm:
            q = self.norm_attn(q)
        
        # Feed-forward sublayer
        if self.pre_norm:
            q = self.norm_ffn(q)
        
        # Position-wise Feed-Forward
        q2 = self.ff(q)
        
        # Add & Norm
        q = q + self.dropout_ffn(q2)  # Residual connection with dropout
        if not self.pre_norm:
            q = self.norm_ffn(q)
        
        if self.res_attention:
            return q, scores
        else:
            return q


class _MultiheadAttention(nn.Module):
    """
    Multi-Head Attention for TimeCMA.
    
    Implements multi-head attention with separate Q, K, V projections
    and optional residual attention from previous layers.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_k: Key dimension per head
        d_v: Value dimension per head
        res_attention: Whether to use residual attention
        attn_dropout: Dropout for attention weights
        proj_dropout: Dropout for output projection
        qkv_bias: Whether to use bias in QKV projections
        lsa: Whether to use learnable scale attention
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        res_attention: bool = False,
        attn_dropout: float = 0.,
        proj_dropout: float = 0.,
        qkv_bias: bool = True,
        lsa: bool = False
    ):
        super().__init__()
        
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        
        # Q, K, V projections
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)
        
        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(
            d_model=d_model,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            res_attention=self.res_attention,
            lsa=lsa
        )
        
        # Project output
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model),
            nn.Dropout(proj_dropout)
        )
    
    def forward(
        self,
        Q: Tensor,
        K: Optional[Tensor] = None,
        V: Optional[Tensor] = None,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None
    ):
        """
        Forward pass for multi-head attention.
        
        Args:
            Q: Query tensor [B, q_len, d_model]
            K: Key tensor [B, k_len, d_model] (default: Q)
            V: Value tensor [B, v_len, d_model] (default: Q)
            prev: Previous attention scores for residual attention
            key_padding_mask: Padding mask [B, k_len]
            attn_mask: Attention mask [q_len, k_len]
        
        Returns:
            Output tensor and attention weights (and scores if res_attention)
        """
        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q
        
        # Linear projections (+ split into multiple heads)
        # q_s: [B, n_heads, q_len, d_k]
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        # k_s: [B, n_heads, d_k, k_len] (transposed for matmul)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        # v_s: [B, n_heads, v_len, d_v]
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        # Apply Scaled Dot-Product Attention
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_s, k_s, v_s,
                prev=prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask
            )
        else:
            output, attn_weights = self.sdp_attn(
                q_s, k_s, v_s,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask
            )
        
        # Reshape back to [B, q_len, n_heads * d_v]
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)
        
        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.
    
    Implements the attention mechanism from "Attention is All You Need"
    with optional residual attention from Realformer and learnable scale
    attention from Vision Transformer for Small-Size Datasets.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        attn_dropout: Dropout for attention weights
        res_attention: Whether to use residual attention
        lsa: Whether to use learnable scale attention
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float = 0.,
        res_attention: bool = False,
        lsa: bool = False
    ):
        super().__init__()
        
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(
            torch.tensor(head_dim ** -0.5),
            requires_grad=lsa
        )
        self.lsa = lsa
    
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None
    ):
        """
        Forward pass for scaled dot-product attention.
        
        Input shapes:
            q: [B, n_heads, q_len, d_k]
            k: [B, n_heads, d_k, k_len]
            v: [B, n_heads, v_len, d_v]
            prev: [B, n_heads, q_len, k_len]
            key_padding_mask: [B, k_len]
            attn_mask: [1, k_len, k_len]
        
        Output shapes:
            output: [B, n_heads, q_len, d_v]
            attn: [B, n_heads, q_len, k_len]
            scores: [B, n_heads, q_len, k_len]
        """
        # Scaled MatMul (q, k) - similarity scores
        attn_scores = torch.matmul(q, k) * self.scale
        
        # Add pre-softmax attention scores from previous layer (optional)
        if prev is not None:
            attn_scores = attn_scores + prev
        
        # Attention mask (optional)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
        
        # Key padding mask (optional)
        if key_padding_mask is not None:
            attn_scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                -np.inf
            )
        
        # Normalize attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute output values
        output = torch.matmul(attn_weights, v)
        
        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights

