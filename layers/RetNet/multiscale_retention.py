"""
Multi-Scale Retention mechanism for RetNet.

Implements the core retention operation that replaces self-attention
in RetNet. Currently implements parallel mode only.

Reference: Sun et al., "Retentive Network: A Successor to Transformer 
for Large Language Models" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .rms_norm import RMSNorm


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate pairs of dimensions for rotary position encoding.
    
    Rearranges tensor so that adjacent pairs are negated and swapped:
    [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
    
    Args:
        x: Input tensor [..., dim] where dim is even
        
    Returns:
        Rotated tensor [..., dim]
    """
    x1 = x[:, :, :, ::2]   # Even indices
    x2 = x[:, :, :, 1::2]  # Odd indices
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # Flatten last two dims


def theta_shift(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position encoding via theta shift.
    
    Implements: x_rotated = x * cos + rotate(x) * sin
    
    Args:
        x: Input tensor [batch, heads, seq_len, head_dim]
        sin: Sine component [seq_len, head_dim]
        cos: Cosine component [seq_len, head_dim]
        
    Returns:
        Position-encoded tensor [batch, heads, seq_len, head_dim]
    """
    return (x * cos) + (rotate_every_two(x) * sin)


class MultiScaleRetention(nn.Module):
    """
    Multi-Scale Retention mechanism (Parallel Mode).
    
    Replaces self-attention with O(n) complexity retention using exponentially
    decaying attention weights. Each head has a different decay rate, enabling
    multi-scale temporal modeling.
    
    This implementation uses PARALLEL MODE only for training efficiency.
    
    Retention Formula:
        Retention(X) = (QK^T ⊙ D) V
        where D_nm = γ^(n-m) for n >= m, else 0 (exponential decay mask)
    
    Args:
        args: RetNetConfig with model hyperparameters
        embed_dim: Input embedding dimension
        value_dim: Value dimension (can differ from embed_dim)
        num_heads: Number of retention heads
        gate_fn: Gating activation function ('swish' or 'gelu')
    
    Note on Future Extension (Chunkwise Recurrent Mode):
        For very long sequences, chunk-recurrent mode can reduce memory from O(n²)
        to O(n/c * c²) = O(nc) where c is chunk size. To implement:
        
        1. Add `chunkwise_recurrent` and `recurrent_chunk_size` to config
        2. Modify RetNetRelPos to compute chunk-based decay masks (see original
           retnet.py lines 37-57 for inner_mask, cross_decay, query_inner_decay)
        3. Implement `chunk_recurrent_forward()` method following original
           multiscale_retention.py lines 114-165
        4. Update forward() to dispatch based on config.chunkwise_recurrent
    """
    
    def __init__(
        self,
        args,
        embed_dim: int,
        value_dim: int,
        num_heads: int,
        gate_fn: str = "swish",
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        
        # Per-head dimensions
        self.head_dim = value_dim // num_heads
        self.key_dim = embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        
        # Gating activation
        if gate_fn == "swish" or gate_fn == "silu":
            self.gate_fn = F.silu
        elif gate_fn == "gelu":
            self.gate_fn = F.gelu
        else:
            self.gate_fn = F.silu  # Default to swish
        
        # =================================================================
        # Projection layers (no bias, following original)
        # =================================================================
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, value_dim, bias=False)
        self.g_proj = nn.Linear(embed_dim, value_dim, bias=False)  # Gating projection
        self.out_proj = nn.Linear(value_dim, embed_dim, bias=False)
        
        # Group normalization per head (applied before output projection)
        self.group_norm = RMSNorm(
            self.head_dim, 
            eps=args.layernorm_eps, 
            elementwise_affine=False
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters with scaled Xavier for stable training."""
        # Scaled initialization for retention (2^-2.5 gain)
        gain = 2 ** -2.5
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def parallel_forward(
        self, 
        qr: torch.Tensor, 
        kr: torch.Tensor, 
        v: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute retention in parallel mode.
        
        Similar to attention but with exponential decay mask instead of
        causal mask.
        
        Args:
            qr: Rotary-encoded queries [batch, heads, seq_len, key_dim]
            kr: Rotary-encoded keys [batch, heads, seq_len, key_dim]
            v: Values [batch, seq_len, value_dim]
            mask: Exponential decay mask [heads, seq_len, seq_len]
            
        Returns:
            output: Retained values [batch, seq_len, heads, head_dim]
        """
        bsz, tgt_len, _ = v.size()
        
        # Reshape values for multi-head: [B, L, V] -> [B, H, L, D_h]
        vr = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention-like scores: [B, H, L, L]
        qk_mat = qr @ kr.transpose(-1, -2)
        
        # Apply exponential decay mask
        qk_mat = qk_mat * mask
        
        # Normalize for numerical stability
        # This is the key difference from softmax - we use absolute sum normalization
        qk_mat = qk_mat / qk_mat.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1)
        
        # Apply to values: [B, H, L, D_h] -> [B, L, H, D_h]
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2)
        
        return output
    
    def forward(
        self,
        x: torch.Tensor,
        rel_pos: Tuple,
        chunkwise_recurrent: bool = False,
        incremental_state: Optional[dict] = None
    ) -> torch.Tensor:
        """
        Forward pass through Multi-Scale Retention.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            rel_pos: Tuple of ((sin, cos), decay_mask) from RetNetRelPos
            chunkwise_recurrent: If True, use chunk-recurrent mode (NOT IMPLEMENTED)
            incremental_state: For recurrent inference (NOT IMPLEMENTED)
            
        Returns:
            output: Retained tensor [batch, seq_len, embed_dim]
            
        Raises:
            NotImplementedError: If chunkwise_recurrent=True or incremental_state provided
        """
        # =================================================================
        # Validate mode (parallel only in this implementation)
        # =================================================================
        if chunkwise_recurrent:
            raise NotImplementedError(
                "Chunkwise recurrent mode not implemented. "
                "See class docstring for extension instructions."
            )
        if incremental_state is not None:
            raise NotImplementedError(
                "Recurrent inference mode not implemented. "
                "See class docstring for extension instructions."
            )
        
        bsz, tgt_len, _ = x.size()
        (sin, cos), mask = rel_pos
        
        # =================================================================
        # Project Q, K, V, G
        # =================================================================
        q = self.q_proj(x)
        k = self.k_proj(x) * self.scaling  # Scale keys
        v = self.v_proj(x)
        g = self.g_proj(x)
        
        # =================================================================
        # Reshape for multi-head: [B, L, E] -> [B, H, L, D_k]
        # =================================================================
        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        
        # =================================================================
        # Apply rotary position encoding
        # =================================================================
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)
        
        # =================================================================
        # Compute retention (parallel mode)
        # =================================================================
        output = self.parallel_forward(qr, kr, v, mask)
        
        # =================================================================
        # Apply group norm per head, then reshape
        # =================================================================
        # output: [B, L, H, D_h] -> apply norm -> [B, L, H * D_h]
        output = self.group_norm(output)
        output = output.reshape(bsz, tgt_len, self.head_dim * self.num_heads)
        
        # =================================================================
        # Gated output
        # =================================================================
        output = self.gate_fn(g) * output
        
        # =================================================================
        # Output projection
        # =================================================================
        output = self.out_proj(output)
        
        return output

