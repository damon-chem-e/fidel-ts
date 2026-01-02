"""
RetNet Decoder for LeRet model.

Implements the stack of retention layers (decoder-only architecture)
for time series encoding. Simplified version without MoE support.

Reference: Sun et al., "Retentive Network: A Successor to Transformer 
for Large Language Models" (2023)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .rms_norm import RMSNorm
from .feedforward import GLU
from .multiscale_retention import MultiScaleRetention


class RetNetRelPos(nn.Module):
    """
    Relative Position Encoding for RetNet.
    
    Computes position-dependent decay masks and rotary position encodings
    for the retention mechanism. Each head has a different decay rate
    enabling multi-scale temporal modeling.
    
    The decay rates are computed as: γ_h = 1 - 2^(-5 - h) for head h
    This gives slower decay (longer memory) for early heads.
    
    Args:
        args: RetNetConfig with decoder_embed_dim and decoder_retention_heads
    """
    
    def __init__(self, args):
        super().__init__()
        
        # =================================================================
        # Compute rotary position encoding angles
        # =================================================================
        # angle = 1 / (10000 ^ (2i / d)) for position encoding
        head_dim = args.decoder_embed_dim // args.decoder_retention_heads
        angle = 1.0 / (10000 ** torch.linspace(0, 1, head_dim // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        
        # =================================================================
        # Compute per-head decay rates
        # =================================================================
        # decay_h = log(1 - 2^(-5-h)) gives exponential decay
        # Early heads have slower decay (longer memory span)
        decay = torch.log(1 - 2 ** (-5 - torch.arange(args.decoder_retention_heads, dtype=torch.float)))
        
        # Register as buffers (not parameters)
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        
        # Store chunk size for potential future extension
        self.recurrent_chunk_size = getattr(args, 'recurrent_chunk_size', 512)
    
    def forward(
        self, 
        slen: int, 
        activate_recurrent: bool = False, 
        chunkwise_recurrent: bool = False
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Compute position encodings and decay mask.
        
        Args:
            slen: Sequence length
            activate_recurrent: Use recurrent mode (NOT IMPLEMENTED)
            chunkwise_recurrent: Use chunk-recurrent mode (NOT IMPLEMENTED)
            
        Returns:
            Tuple of:
                - (sin, cos): Rotary position encoding components
                    - sin: [seq_len, head_dim]
                    - cos: [seq_len, head_dim]
                - mask: Exponential decay attention mask [heads, seq_len, seq_len]
                
        Raises:
            NotImplementedError: If activate_recurrent or chunkwise_recurrent is True
        """
        # =================================================================
        # Validate mode (parallel only)
        # =================================================================
        if activate_recurrent:
            raise NotImplementedError(
                "Recurrent mode not implemented. Use parallel mode for training."
            )
        if chunkwise_recurrent:
            raise NotImplementedError(
                "Chunkwise recurrent mode not implemented. "
                "See implementation plan for extension instructions."
            )
        
        # =================================================================
        # Compute rotary position encodings
        # =================================================================
        # index: [0, 1, 2, ..., slen-1]
        index = torch.arange(slen).to(self.decay)
        
        # sin/cos: [seq_len, head_dim]
        sin = torch.sin(index[:, None] * self.angle[None, :])
        cos = torch.cos(index[:, None] * self.angle[None, :])
        
        # =================================================================
        # Compute exponential decay mask (parallel mode)
        # =================================================================
        # Create lower triangular mask: mask[i,j] = 1 if i >= j, else 0
        mask = torch.tril(torch.ones(slen, slen).to(self.decay))
        
        # Compute position differences: diff[i,j] = i - j (for i >= j)
        # Set to inf for positions above diagonal (will become 0 after exp)
        diff = index[:, None] - index[None, :]
        diff = torch.masked_fill(diff, ~mask.bool(), float("inf"))
        
        # Apply per-head exponential decay: mask[h,i,j] = exp(decay[h] * (i-j))
        # decay is negative, so this gives values in (0, 1] that decrease with distance
        mask = torch.exp(diff * self.decay[:, None, None])
        
        # Handle NaN from exp(-inf) -> 0
        mask = torch.nan_to_num(mask)
        
        # Normalize by sqrt of row sum for numerical stability
        mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
        
        return ((sin, cos), mask)


class DecoderLayer(nn.Module):
    """
    Single RetNet decoder layer.
    
    Consists of:
    1. Multi-Scale Retention (replaces self-attention)
    2. Gated Linear Unit feedforward network
    
    With residual connections and RMS normalization.
    
    Args:
        args: RetNetConfig with model hyperparameters
        depth: Layer index (for drop path rate scheduling)
    """
    
    def __init__(self, args, depth: int):
        super().__init__()
        self.args = args
        self.embed_dim = args.decoder_embed_dim
        
        # Dropout
        self.dropout_module = nn.Dropout(args.dropout)
        
        # =================================================================
        # Stochastic depth (drop path)
        # =================================================================
        if args.drop_path_rate > 0:
            # Linear increase in drop probability with depth
            drop_path_prob = np.linspace(0, args.drop_path_rate, args.decoder_layers)[depth]
            self.drop_path = DropPath(drop_path_prob)
        else:
            self.drop_path = None
        
        # =================================================================
        # Multi-Scale Retention
        # =================================================================
        self.retention = MultiScaleRetention(
            args,
            self.embed_dim,
            args.decoder_value_embed_dim,
            args.decoder_retention_heads,
        )
        
        # Pre-LN vs Post-LN
        self.normalize_before = args.decoder_normalize_before
        
        # RMS Normalization layers
        self.retention_layer_norm = RMSNorm(self.embed_dim, eps=args.layernorm_eps)
        self.final_layer_norm = RMSNorm(self.embed_dim, eps=args.layernorm_eps)
        
        # =================================================================
        # Feedforward Network (GLU)
        # =================================================================
        self.ffn_dim = args.decoder_ffn_embed_dim
        self.ffn = GLU(
            self.embed_dim,
            self.ffn_dim,
            args.activation_fn,
            args.dropout,
            args.activation_dropout,
        )
        
        # =================================================================
        # DeepNorm scaling (if enabled)
        # =================================================================
        if args.deepnorm:
            self.alpha = math.pow(2.0 * args.decoder_layers, 0.25)
        else:
            self.alpha = 1.0
    
    def residual_connection(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Apply residual connection with optional DeepNorm scaling."""
        return residual * self.alpha + x
    
    def forward(
        self,
        x: torch.Tensor,
        incremental_state: Optional[dict] = None,
        chunkwise_recurrent: bool = False,
        retention_rel_pos: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Forward pass through decoder layer.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            incremental_state: For recurrent inference (NOT IMPLEMENTED)
            chunkwise_recurrent: Use chunk-recurrent mode (NOT IMPLEMENTED)
            retention_rel_pos: Position encodings from RetNetRelPos
            
        Returns:
            Tuple of (output, aux_loss) where aux_loss is always None
            (MoE aux_loss not supported in this implementation)
        """
        # =================================================================
        # Retention sublayer
        # =================================================================
        residual = x
        
        # Pre-norm
        if self.normalize_before:
            x = self.retention_layer_norm(x)
        
        # Multi-Scale Retention
        x = self.retention(
            x,
            rel_pos=retention_rel_pos,
            chunkwise_recurrent=chunkwise_recurrent,
            incremental_state=incremental_state,
        )
        x = self.dropout_module(x)
        
        # Drop path (stochastic depth)
        if self.drop_path is not None:
            x = self.drop_path(x)
        
        # Residual connection
        x = self.residual_connection(x, residual)
        
        # Post-norm
        if not self.normalize_before:
            x = self.retention_layer_norm(x)
        
        # =================================================================
        # FFN sublayer
        # =================================================================
        residual = x
        
        # Pre-norm
        if self.normalize_before:
            x = self.final_layer_norm(x)
        
        # GLU feedforward
        x = self.ffn(x)
        
        # Drop path
        if self.drop_path is not None:
            x = self.drop_path(x)
        
        # Residual connection
        x = self.residual_connection(x, residual)
        
        # Post-norm
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        
        return x, None  # No aux_loss (MoE not supported)


class DropPath(nn.Module):
    """
    Stochastic Depth (Drop Path) regularization.
    
    Randomly drops entire residual branches during training.
    
    Args:
        drop_prob: Probability of dropping the path
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        # Create random tensor of shape [batch, 1, 1, ...] for broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        
        # Scale to maintain expected value
        output = x.div(keep_prob) * random_tensor
        return output


class RetNetDecoder(nn.Module):
    """
    RetNet Decoder - stack of retention layers.
    
    A decoder-only architecture using Multi-Scale Retention instead of
    self-attention. Designed for time series encoding in LeRet.
    
    Args:
        args: RetNetConfig with model hyperparameters
        embed_tokens: Optional embedding layer (not used for time series)
        output_projection: Optional output projection layer
    """
    
    def __init__(
        self,
        args,
        embed_tokens: Optional[nn.Module] = None,
        output_projection: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.args = args
        
        self.dropout_module = nn.Dropout(args.dropout)
        
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)
        
        # Token embedding (optional, not used for time series)
        self.embed_tokens = embed_tokens
        
        # Output projection (optional)
        if output_projection is None and not args.no_output_layer and args.vocab_size > 0:
            self.output_projection = self._build_output_projection(args)
        else:
            self.output_projection = output_projection
        
        # Embedding layer norm (optional)
        if args.layernorm_embedding:
            self.layernorm_embedding = RMSNorm(embed_dim, eps=args.layernorm_eps)
        else:
            self.layernorm_embedding = None
        
        # =================================================================
        # Stack of decoder layers
        # =================================================================
        self.layers = nn.ModuleList([
            DecoderLayer(args, depth=i)
            for i in range(args.decoder_layers)
        ])
        self.num_layers = len(self.layers)
        
        # Final layer norm (if using pre-norm)
        if args.decoder_normalize_before:
            self.layer_norm = RMSNorm(embed_dim, eps=args.layernorm_eps)
        else:
            self.layer_norm = None
        
        # =================================================================
        # Relative position encoding
        # =================================================================
        self.retnet_rel_pos = RetNetRelPos(args)
        
        # Store chunk settings (for future extension)
        self.chunkwise_recurrent = args.chunkwise_recurrent
        self.recurrent_chunk_size = args.recurrent_chunk_size
        
        # =================================================================
        # DeepNorm initialization (if enabled)
        # =================================================================
        if args.deepnorm:
            init_scale = math.pow(8.0 * args.decoder_layers, 0.25)
            for name, p in self.named_parameters():
                if "fc1" in name or "fc2" in name or "fc3" in name or "out_proj" in name or "v_proj" in name:
                    p.data.div_(init_scale)
    
    def _build_output_projection(self, args) -> nn.Module:
        """Build output projection layer for language modeling."""
        if self.embed_tokens is not None and args.share_decoder_input_output_embed:
            output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            output_projection.weight = self.embed_tokens.weight
        else:
            output_projection = nn.Linear(
                args.decoder_embed_dim, args.vocab_size, bias=False
            )
            nn.init.normal_(
                output_projection.weight, mean=0, std=args.decoder_embed_dim ** -0.5
            )
        return output_projection
    
    def forward(
        self,
        prev_output_tokens: torch.Tensor,
        incremental_state: Optional[Dict[int, dict]] = None,
        features_only: bool = False,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through RetNet decoder.
        
        Args:
            prev_output_tokens: Input tensor [batch, seq_len, embed_dim]
                For time series, this is the projected patch embeddings
            incremental_state: For recurrent inference (NOT IMPLEMENTED)
            features_only: If True, skip output projection
            return_all_hiddens: If True, return all layer outputs
            token_embeddings: Alternative to prev_output_tokens (not used)
            
        Returns:
            Tuple of:
                - output: [batch, seq_len, embed_dim] if features_only else [batch, seq_len, vocab_size]
                - extra: Dict with 'inner_states', 'l_aux', 'attn'
        """
        # =================================================================
        # For time series, input is already embedded patches
        # Skip token embedding - use input directly
        # =================================================================
        x = prev_output_tokens
        
        # Check for first step in recurrent mode
        is_first_step = self._is_first_step(incremental_state)
        
        # =================================================================
        # Handle padding for chunk-recurrent mode (NOT IMPLEMENTED)
        # =================================================================
        slen = prev_output_tokens.size(1)
        if self.chunkwise_recurrent and slen % self.recurrent_chunk_size != 0:
            # Pad to chunk boundary
            padding_len = self.recurrent_chunk_size - slen % self.recurrent_chunk_size
            slen = slen + padding_len
            x = F.pad(x, (0, 0, 0, padding_len))
        
        # =================================================================
        # Compute relative position encoding
        # =================================================================
        retention_rel_pos = self.retnet_rel_pos(
            slen,
            activate_recurrent=(incremental_state is not None and not is_first_step),
            chunkwise_recurrent=self.chunkwise_recurrent
        )
        
        # =================================================================
        # Pass through decoder layers
        # =================================================================
        inner_states = [x]
        l_aux = []
        
        for idx, layer in enumerate(self.layers):
            # Prepare incremental state for layer
            if incremental_state is not None:
                if is_first_step and idx not in incremental_state:
                    incremental_state[idx] = {}
                elif idx not in incremental_state:
                    incremental_state[idx] = {}
            
            x, l_aux_i = layer(
                x,
                incremental_state=incremental_state[idx] if incremental_state is not None else None,
                retention_rel_pos=retention_rel_pos,
                chunkwise_recurrent=self.chunkwise_recurrent,
            )
            l_aux.append(l_aux_i)
            inner_states.append(x)
        
        # =================================================================
        # Remove padding if added for chunk-recurrent
        # =================================================================
        if self.chunkwise_recurrent and prev_output_tokens.size(1) % self.recurrent_chunk_size != 0:
            x = x[:, :prev_output_tokens.size(1), :]
        
        # =================================================================
        # Final layer norm
        # =================================================================
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        # =================================================================
        # Output projection (for language modeling, skip for time series)
        # =================================================================
        if not features_only and self.output_projection is not None:
            x = self.output_projection(x)
        
        return x, {
            "inner_states": inner_states,
            "l_aux": l_aux,
            "attn": None,
        }
    
    def _is_first_step(self, incremental_state: Optional[dict]) -> bool:
        """Check if this is the first step in recurrent inference."""
        if incremental_state is None:
            return False
        return incremental_state.get("is_first_step", False)

