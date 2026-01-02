"""
Time Series - Language Integration module for LeRet.

Implements bidirectional cross-attention between time series patches
and language embeddings from the fidel-ts embedder infrastructure.

The integration flow:
1. ts2text: Language embeddings (Q) attend to TS patches (K, V)
2. text2ts: TS patches (Q) attend to enriched language (K, V)

This allows the model to incorporate textual domain knowledge into
the time series representation for enhanced forecasting.
"""

from typing import Optional, Tuple
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.LeRet_layers import Transpose, get_activation_fn


class LanguageIntegrator(nn.Module):
    """
    Bidirectional cross-attention between time series and language embeddings.
    
    Integrates language knowledge into time series representations using
    the fidel-ts embedder infrastructure for text embedding generation.
    
    Architecture:
    1. Project language embeddings: language_dim -> d_model
    2. ts2text: Cross-attention where language queries TS patches
    3. text2ts: Cross-attention where TS patches query language
    
    Args:
        c_in: Number of input channels
        patch_num: Number of patches
        patch_len: Length of each patch
        d_model: Model dimension
        n_heads: Number of attention heads
        language_dim: Language embedding dimension (e.g., 768 for BERT)
        d_k: Key dimension per head (optional)
        d_v: Value dimension per head (optional)
        d_ff: FFN dimension
        norm: Normalization type ('BatchNorm' or 'LayerNorm')
        attn_dropout: Attention dropout rate
        dropout: General dropout rate
        pre_norm: Use pre-normalization
        activation: Activation function name
        res_attention: Use residual attention
        n_layers: Number of cross-attention layers
        store_attn: Store attention weights
    """
    
    def __init__(
        self,
        c_in: int,
        patch_num: int,
        patch_len: int,
        d_model: int = 128,
        n_heads: int = 8,
        language_dim: int = 768,  # BERT-base dimension
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = 'BatchNorm',
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        pre_norm: bool = False,
        activation: str = "gelu",
        res_attention: bool = True,
        n_layers: int = 1,
        store_attn: bool = False,
        max_seq_len: int = 1024,
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.language_dim = language_dim
        self.dropout = nn.Dropout(dropout)
        
        # =================================================================
        # Language embedding projection: language_dim -> d_model
        # =================================================================
        self.text_linear = nn.Linear(language_dim, d_model)
        
        # =================================================================
        # Cross-attention modules
        # =================================================================
        # ts2text: Language (Q) attends to TS patches (K, V)
        self.ts2text = CrossEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=activation,
            res_attention=res_attention,
            n_layers=n_layers,
            store_attn=store_attn
        )
        
        # text2ts: TS patches (Q) attend to enriched language (K, V)
        self.text2ts = CrossEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=activation,
            res_attention=res_attention,
            n_layers=n_layers,
            store_attn=store_attn
        )
    
    def forward(
        self, 
        x: Tensor, 
        language_embeddings: Optional[Tensor] = None
    ) -> Tensor:
        """
        Integrate language knowledge into time series representation.
        
        Args:
            x: Encoded TS patches [batch, n_vars, d_model, patch_num]
            language_embeddings: Language embeddings [text_num, language_dim]
                                If None, returns x unchanged (no language integration)
        
        Returns:
            Language-enhanced representation [batch, n_vars, d_model, patch_num]
        """
        # Skip integration if no language embeddings provided
        if language_embeddings is None:
            return x
        
        bs, n_vars = x.shape[0], x.shape[1]
        
        # =================================================================
        # Project language embeddings to model dimension
        # =================================================================
        # language_embeddings: [text_num, language_dim] -> [text_num, d_model]
        lang_proj = self.text_linear(language_embeddings)
        
        # Expand for batch and channels: [bs*n_vars, text_num, d_model]
        lang_proj = lang_proj.unsqueeze(0).expand(bs * n_vars, -1, -1)
        lang_proj = self.dropout(lang_proj)
        
        # =================================================================
        # Reshape time series patches
        # =================================================================
        # x: [B, C, d_model, patch_num] -> [B, C, patch_num, d_model]
        x = x.permute(0, 1, 3, 2)
        # x: [B, C, patch_num, d_model] -> [B*C, patch_num, d_model]
        x = torch.reshape(x, (bs * n_vars, x.shape[2], x.shape[3]))
        
        # =================================================================
        # Cross-attention: Language -> TS (ts2text)
        # =================================================================
        # Language queries, TS patches are keys/values
        # lang_enhanced: [B*C, text_num, d_model]
        lang_enhanced = self.ts2text(q=lang_proj, k=x, v=x)
        
        # =================================================================
        # Cross-attention: TS -> Language (text2ts)
        # =================================================================
        # TS patches query, enriched language are keys/values
        # z: [B*C, patch_num, d_model]
        z = self.text2ts(q=x, k=lang_enhanced, v=lang_enhanced)
        
        # =================================================================
        # Reshape back to original format
        # =================================================================
        # z: [B*C, patch_num, d_model] -> [B, C, patch_num, d_model]
        z = torch.reshape(z, (bs, n_vars, z.shape[-2], z.shape[-1]))
        # z: [B, C, patch_num, d_model] -> [B, C, d_model, patch_num]
        z = z.permute(0, 1, 3, 2)
        
        return z


class CrossEncoder(nn.Module):
    """
    Stack of cross-attention layers.
    
    Each layer performs multi-head cross-attention followed by
    feedforward network with residual connections.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_k: Key dimension per head
        d_v: Value dimension per head
        d_ff: FFN dimension
        norm: Normalization type
        attn_dropout: Attention dropout
        dropout: General dropout
        activation: Activation function
        res_attention: Use residual attention
        n_layers: Number of layers
        pre_norm: Use pre-normalization
        store_attn: Store attention weights
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = 'BatchNorm',
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        res_attention: bool = False,
        n_layers: int = 1,
        pre_norm: bool = False,
        store_attn: bool = False
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            CrossEncoderLayer(
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
            ) for _ in range(n_layers)
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
        Forward pass through cross-encoder layers.
        
        Args:
            q: Query tensor [batch, q_len, d_model]
            k: Key tensor [batch, kv_len, d_model]
            v: Value tensor [batch, kv_len, d_model]
            key_padding_mask: Optional key padding mask
            attn_mask: Optional attention mask
        
        Returns:
            Output tensor [batch, q_len, d_model]
        """
        scores = None
        output = q
        
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(
                    output, k, v,
                    prev=scores,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask
                )
        else:
            for mod in self.layers:
                output = mod(
                    output, k, v,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask
                )
        
        return output


class CrossEncoderLayer(nn.Module):
    """
    Single cross-attention layer with feedforward network.
    
    Architecture:
    1. Multi-head cross-attention (Q from one source, K/V from another)
    2. Add & Norm
    3. Feedforward network
    4. Add & Norm
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_k: Key dimension per head
        d_v: Value dimension per head
        d_ff: FFN dimension
        store_attn: Store attention weights
        norm: Normalization type
        attn_dropout: Attention dropout
        dropout: General dropout
        bias: Use bias in linear layers
        activation: Activation function
        res_attention: Use residual attention
        pre_norm: Use pre-normalization
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        store_attn: bool = False,
        norm: str = 'BatchNorm',
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        bias: bool = True,
        activation: str = "gelu",
        res_attention: bool = False,
        pre_norm: bool = False
    ):
        super().__init__()
        
        # Validate d_model divisibility
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        
        # =================================================================
        # Multi-Head Cross-Attention
        # =================================================================
        self.res_attention = res_attention
        self.cross_attn = MultiheadCrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention
        )
        
        # =================================================================
        # Add & Norm (attention)
        # =================================================================
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(d_model),
                Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(d_model)
        
        # =================================================================
        # Feedforward Network
        # =================================================================
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias)
        )
        
        # =================================================================
        # Add & Norm (FFN)
        # =================================================================
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
    ):
        """
        Forward pass through cross-attention layer.
        
        Args:
            q: Query tensor [batch, q_len, d_model]
            k: Key tensor [batch, kv_len, d_model]
            v: Value tensor [batch, kv_len, d_model]
            prev: Previous attention scores (for residual attention)
            key_padding_mask: Key padding mask
            attn_mask: Attention mask
        
        Returns:
            If res_attention: (output, scores) tuple
            Else: output tensor
        """
        # =================================================================
        # Cross-Attention sublayer
        # =================================================================
        if self.pre_norm:
            q = self.norm_attn(q)
            k = self.norm_attn(k)
            v = self.norm_attn(v)
        
        # Multi-head cross-attention
        if self.res_attention:
            q2, attn, scores = self.cross_attn(
                q, k, v, prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask
            )
        else:
            q2, attn = self.cross_attn(
                q, k, v,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask
            )
        
        if self.store_attn:
            self.attn = attn
        
        # Add & Norm
        q = q + self.dropout_attn(q2)
        if not self.pre_norm:
            q = self.norm_attn(q)
        
        # =================================================================
        # Feedforward sublayer
        # =================================================================
        if self.pre_norm:
            q = self.norm_ffn(q)
        
        q2 = self.ff(q)
        
        # Add & Norm
        q = q + self.dropout_ffn(q2)
        if not self.pre_norm:
            q = self.norm_ffn(q)
        
        if self.res_attention:
            return q, scores
        else:
            return q


class MultiheadCrossAttention(nn.Module):
    """
    Multi-head cross-attention mechanism.
    
    Computes attention with separate Q, K, V projections where
    Q comes from one source and K, V from another.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_k: Key dimension per head
        d_v: Value dimension per head
        res_attention: Use residual attention
        attn_dropout: Attention dropout
        proj_dropout: Projection dropout
        qkv_bias: Use bias in QKV projections
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        res_attention: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qkv_bias: bool = True
    ):
        super().__init__()
        
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        
        # Separate projections for Q, K, V
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)
        
        # Scaled dot-product attention
        self.res_attention = res_attention
        self.sdp_attn = ScaledDotProductAttention(
            d_model=d_model,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            res_attention=res_attention
        )
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model),
            nn.Dropout(proj_dropout)
        )
    
    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None
    ):
        """
        Forward pass through multi-head cross-attention.
        
        Args:
            Q: Query tensor [batch, q_len, d_model]
            K: Key tensor [batch, kv_len, d_model]
            V: Value tensor [batch, kv_len, d_model]
            prev: Previous attention scores
            key_padding_mask: Key padding mask
            attn_mask: Attention mask
        
        Returns:
            If res_attention: (output, attn_weights, attn_scores)
            Else: (output, attn_weights)
        """
        bs = Q.size(0)
        
        # Linear projections + split into heads
        # q_s: [bs, n_heads, q_len, d_k]
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        # k_s: [bs, n_heads, d_k, kv_len] (transposed for matmul)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        # v_s: [bs, n_heads, kv_len, d_v]
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        # Scaled dot-product attention
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
        
        # Concatenate heads and project
        # output: [bs, n_heads, q_len, d_v] -> [bs, q_len, n_heads * d_v]
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)
        
        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention with optional residual attention.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    Args:
        d_model: Model dimension
        n_heads: Number of heads (for scaling)
        attn_dropout: Attention dropout rate
        res_attention: Use residual attention (add prev scores)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float = 0.0,
        res_attention: bool = False
    ):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = head_dim ** -0.5
    
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
        Compute scaled dot-product attention.
        
        Args:
            q: Query [bs, n_heads, q_len, d_k]
            k: Key [bs, n_heads, d_k, kv_len] (pre-transposed)
            v: Value [bs, n_heads, kv_len, d_v]
            prev: Previous attention scores for residual attention
            key_padding_mask: [bs, kv_len] mask
            attn_mask: [q_len, kv_len] mask
        
        Returns:
            If res_attention: (output, attn_weights, attn_scores)
            Else: (output, attn_weights)
        """
        # Compute attention scores: [bs, n_heads, q_len, kv_len]
        attn_scores = torch.matmul(q, k) * self.scale
        
        # Add previous scores for residual attention
        if prev is not None:
            attn_scores = attn_scores + prev
        
        # Apply attention mask (additive)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_scores = attn_scores + attn_mask
        
        # Apply key padding mask
        if key_padding_mask is not None:
            attn_scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Softmax normalization
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute output: [bs, n_heads, q_len, d_v]
        output = torch.matmul(attn_weights, v)
        
        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights

