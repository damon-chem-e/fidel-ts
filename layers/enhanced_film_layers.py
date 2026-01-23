"""
Enhanced FiLM Layers for lynx_film_enhanced Model

This module implements the core building blocks for the lynx_film_enhanced architecture,
which provides a unified framework for text-modulated time series forecasting with
configurable:
- Pathway type: "shared" (Option 9) or "parallel" (Option 10)
- Gate type: "global", "channel", or "conditional"  
- Mixing type: "interpolative" or "additive"

Classes:
    GateModule: Computes mixing gate α based on gate_type
    PathwayMixer: Mixes two pathway outputs based on mixing_type
    EnhancedEncoderLayerShared: Encoder layer with shared backbone, separate LayerNorms
    EnhancedEncoderLayerParallel: Encoder layer with frozen unimodal + learned text pathways
    EnhancedEncoderFilm: Encoder wrapper that manages layers and FiLM generators

Reference:
    See docs/planning/learned_text_regularization_options.md Part III for detailed design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.FiLM_layers import FiLMGenerator


class GateModule(nn.Module):
    """
    Computes mixing gate α based on gate_type.
    
    The gate determines how much weight to give to the text-modulated pathway
    versus the base pathway. Values are in [0, 1] after sigmoid.
    
    Gate Types:
        - "global": Single scalar gate for all samples/channels (1 param)
        - "channel": Per-channel static gate (C params, lazy-initialized if C unknown)
        - "conditional": Attention-based gate conditioned on x and T (sample-adaptive)
          Uses self-attention to pool temporal/sequence dimensions before computing gate.
    
    Args:
        gate_type: One of "global", "channel", or "conditional"
        num_channels: Number of channels (optional for "channel" gate - can be inferred)
        d_model: Model dimension (unused currently, kept for API compatibility)
        text_dim: Text embedding dimension (for "conditional" gate attention pooling)
        hidden_dim: Hidden dimension for conditional gate's attention and MLP layers.
                    Default: 32. Controls expressiveness vs parameter count.
                    Recommended: 32-64 for most cases, increase for complex datasets.
        init_bias: Initialization bias for gate logits (0 = balanced, + = favor text)
    """
    
    def __init__(
        self, 
        gate_type: str, 
        num_channels: int = None, 
        d_model: int = 256,
        text_dim: int = 256, 
        hidden_dim: int = 32, 
        init_bias: float = 0.0
    ):
        super().__init__()
        self.gate_type = gate_type
        self.num_channels = num_channels
        self.init_bias = init_bias
        self.hidden_dim = hidden_dim
        self.text_dim = text_dim
        self._channel_gate_initialized = False
        
        if gate_type == "global":
            # Single scalar gate - broadcasts to all samples/channels
            self.gate_raw = nn.Parameter(torch.tensor(init_bias))
            
        elif gate_type == "channel":
            # Per-channel static gate
            # If num_channels is provided, initialize now; otherwise lazy-init on first forward
            if num_channels is not None:
                self.gate_raw = nn.Parameter(torch.full((num_channels,), init_bias))
                self._channel_gate_initialized = True
            else:
                # Will be initialized on first forward pass
                self.register_parameter('gate_raw', None)
            
        elif gate_type == "conditional":
            # ═══════════════════════════════════════════════════════════════════
            # CONDITIONAL GATE WITH ATTENTION-BASED POOLING
            # ═══════════════════════════════════════════════════════════════════
            # Instead of simple mean pooling, we use self-attention to aggregate
            # temporal information, then project to a scalar per channel.
            #
            # Time series path: x [B, seq_len, C] -> self-attend -> project -> [B, C]
            # Text path: T [B, C, L, text_dim] -> self-attend -> project -> [B, C]
            # Combined: [B, C, 2] -> MLP -> [B, C] gate values
            # ═══════════════════════════════════════════════════════════════════
            
            # Attention pooling for time series (along temporal dimension)
            # We use a learned query to attend over the sequence
            self.ts_query = nn.Parameter(torch.randn(1, 1, hidden_dim))  # [1, 1, hidden_dim]
            self.ts_key_proj = nn.Linear(1, hidden_dim)  # Project scalar per timestep to hidden_dim
            self.ts_value_proj = nn.Linear(1, hidden_dim)
            self.ts_output_proj = nn.Linear(hidden_dim, 1)  # Project back to scalar
            
            # Attention pooling for text (along sequence dimension L)
            self.text_query = nn.Parameter(torch.randn(1, 1, hidden_dim))  # [1, 1, hidden_dim]
            self.text_key_proj = nn.Linear(text_dim, hidden_dim)
            self.text_value_proj = nn.Linear(text_dim, hidden_dim)
            self.text_output_proj = nn.Linear(hidden_dim, 1)
            
            # Final MLP to combine pooled representations
            # Input: [x_pooled, T_pooled] per channel -> 2 scalars
            self.gate_mlp = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            # Initialize bias to favor balanced mixing initially
            self.gate_mlp[-1].bias.data.fill_(init_bias)
            
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}. Must be 'global', 'channel', or 'conditional'")
    
    def forward(self, x: torch.Tensor = None, text_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Compute gate values.
        
        Args:
            x: Time series input [B, seq_len, C] (for conditional gate)
            text_emb: Text embeddings [B, C, L, text_dim] (for conditional gate)
        
        Returns:
            alpha: Gate values in [0, 1]
                - "global": scalar (broadcasts naturally)
                - "channel": [C] tensor
                - "conditional": [B, C] tensor
        """
        if self.gate_type == "global":
            # Return scalar gate value
            return torch.sigmoid(self.gate_raw)
        
        elif self.gate_type == "channel":
            # Lazy initialization of channel gate if not already done
            if not self._channel_gate_initialized:
                if x is None:
                    raise ValueError("Channel gate requires x input for lazy initialization")
                # Infer num_channels from x: [B, seq_len, C]
                num_channels = x.shape[-1]
                device = x.device
                # Create and initialize the parameter
                self.gate_raw = nn.Parameter(
                    torch.full((num_channels,), self.init_bias, device=device)
                )
                self.num_channels = num_channels
                self._channel_gate_initialized = True
            
            # Return per-channel gate values [C]
            return torch.sigmoid(self.gate_raw)
        
        elif self.gate_type == "conditional":
            # Validate inputs for conditional gate
            if x is None or text_emb is None:
                raise ValueError("Conditional gate requires both x and text_emb")
            
            # x: [B, seq_len, C]
            B, seq_len, C = x.shape
            
            # ═══════════════════════════════════════════════════════════════════
            # ATTENTION-BASED POOLING FOR TIME SERIES
            # ═══════════════════════════════════════════════════════════════════
            # x: [B, seq_len, C] -> process per channel
            # Reshape to [B*C, seq_len, 1] to process each channel independently
            x_reshaped = x.permute(0, 2, 1).reshape(B * C, seq_len, 1)  # [B*C, seq_len, 1]
            
            # Project to key/value: [B*C, seq_len, hidden_dim]
            ts_keys = self.ts_key_proj(x_reshaped)  # [B*C, seq_len, hidden_dim]
            ts_values = self.ts_value_proj(x_reshaped)  # [B*C, seq_len, hidden_dim]
            
            # Expand query for batch: [1, 1, hidden_dim] -> [B*C, 1, hidden_dim]
            ts_query = self.ts_query.expand(B * C, -1, -1)
            
            # Attention: [B*C, 1, hidden_dim] @ [B*C, hidden_dim, seq_len] -> [B*C, 1, seq_len]
            attn_scores = torch.bmm(ts_query, ts_keys.transpose(1, 2)) / (self.hidden_dim ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)  # [B*C, 1, seq_len]
            
            # Weighted sum: [B*C, 1, seq_len] @ [B*C, seq_len, hidden_dim] -> [B*C, 1, hidden_dim]
            ts_attended = torch.bmm(attn_weights, ts_values)  # [B*C, 1, hidden_dim]
            
            # Project to scalar and reshape: [B*C, 1, hidden_dim] -> [B*C, 1] -> [B, C]
            x_pool = self.ts_output_proj(ts_attended).squeeze(-1).reshape(B, C)  # [B, C]
            
            # ═══════════════════════════════════════════════════════════════════
            # ATTENTION-BASED POOLING FOR TEXT
            # ═══════════════════════════════════════════════════════════════════
            # text_emb: [B, C, L, text_dim] -> process per channel
            _, _, L, text_dim = text_emb.shape
            
            # Reshape to [B*C, L, text_dim]
            text_reshaped = text_emb.reshape(B * C, L, text_dim)
            
            # Project to key/value: [B*C, L, hidden_dim]
            text_keys = self.text_key_proj(text_reshaped)  # [B*C, L, hidden_dim]
            text_values = self.text_value_proj(text_reshaped)  # [B*C, L, hidden_dim]
            
            # Expand query for batch: [1, 1, hidden_dim] -> [B*C, 1, hidden_dim]
            text_query = self.text_query.expand(B * C, -1, -1)
            
            # Attention: [B*C, 1, hidden_dim] @ [B*C, hidden_dim, L] -> [B*C, 1, L]
            text_attn_scores = torch.bmm(text_query, text_keys.transpose(1, 2)) / (self.hidden_dim ** 0.5)
            text_attn_weights = F.softmax(text_attn_scores, dim=-1)  # [B*C, 1, L]
            
            # Weighted sum: [B*C, 1, L] @ [B*C, L, hidden_dim] -> [B*C, 1, hidden_dim]
            text_attended = torch.bmm(text_attn_weights, text_values)  # [B*C, 1, hidden_dim]
            
            # Project to scalar and reshape: [B*C, 1, hidden_dim] -> [B*C, 1] -> [B, C]
            T_pool = self.text_output_proj(text_attended).squeeze(-1).reshape(B, C)  # [B, C]
            
            # ═══════════════════════════════════════════════════════════════════
            # COMBINE AND COMPUTE GATE
            # ═══════════════════════════════════════════════════════════════════
            # Concatenate: [B, C, 2]
            gate_input = torch.stack([x_pool, T_pool], dim=-1)
            
            # Apply MLP per channel: [B, C, 2] -> [B, C, 1] -> [B, C]
            alpha_logit = self.gate_mlp(gate_input).squeeze(-1)
            
            return torch.sigmoid(alpha_logit)  # [B, C]
        
        else:
            raise ValueError(f"Unknown gate_type: {self.gate_type}")


class PathwayMixer(nn.Module):
    """
    Mixes two pathway outputs based on mixing_type.
    
    Mixing Types:
        - "interpolative": z = α * z_text + (1 - α) * z_base (convex combination)
        - "additive": z = z_base + α * z_text (residual/additive)
    
    The interpolative mixing keeps the output bounded between the two pathways,
    while additive mixing allows the output to go beyond the convex hull.
    
    Args:
        mixing_type: One of "interpolative" or "additive"
    """
    
    def __init__(self, mixing_type: str):
        super().__init__()
        if mixing_type not in ("interpolative", "additive"):
            raise ValueError(f"Unknown mixing_type: {mixing_type}. Must be 'interpolative' or 'additive'")
        self.mixing_type = mixing_type
    
    def forward(
        self, 
        z_base: torch.Tensor, 
        z_text: torch.Tensor, 
        alpha: torch.Tensor
    ) -> torch.Tensor:
        """
        Mix two pathway outputs.
        
        Args:
            z_base: Base pathway output [B, C, d_model]
            z_text: Text pathway output [B, C, d_model]
            alpha: Gate value(s) - scalar, [C], or [B, C]
        
        Returns:
            z_mixed: Mixed output [B, C, d_model]
        """
        # Expand alpha for broadcasting if needed
        if alpha.dim() == 0:
            # Scalar - broadcasts naturally to [B, C, d]
            pass
        elif alpha.dim() == 1:
            # [C] -> [1, C, 1] for broadcasting to [B, C, d]
            alpha = alpha.unsqueeze(0).unsqueeze(-1)
        elif alpha.dim() == 2:
            # [B, C] -> [B, C, 1] for broadcasting to [B, C, d]
            alpha = alpha.unsqueeze(-1)
        
        if self.mixing_type == "interpolative":
            # Convex combination: z = α * z_text + (1 - α) * z_base
            return alpha * z_text + (1 - alpha) * z_base
        elif self.mixing_type == "additive":
            # Residual/additive: z = z_base + α * z_text
            return z_base + alpha * z_text
        else:
            raise ValueError(f"Unknown mixing_type: {self.mixing_type}")


class EnhancedEncoderLayerShared(nn.Module):
    """
    Encoder layer with shared FFN/Attention and separate LayerNorm pathways.
    
    This implements the "shared" pathway_type (Option 9):
    - Shared Attention and FFN weights (both learned)
    - Separate LayerNorm pathways: base (learned affine) vs text (FiLM modulated)
    - Gated mixing between pathways
    
    Architecture Flow:
        1. Attention(x) + residual
        2. LayerNorm (no affine)
        3. Split into base path (learned γ,β) and text path (FiLM γ,β)
        4. Gate and mix pathways
        5. FFN + residual
        6. LayerNorm (no affine)
        7. Split, gate, and mix again
    
    Design Note - elementwise_affine=False:
    =======================================
    We use LayerNorm with `elementwise_affine=False` which means the LayerNorm
    only performs mean-centering and variance-scaling WITHOUT its own learnable
    γ (scale) and β (shift) parameters.
    
    Instead, we provide SEPARATE affine parameters for each pathway:
    - Base pathway: Learned γ_base, β_base (standard affine transform)
    - Text pathway: FiLM-generated (1+γ_FiLM), β_FiLM (text-conditioned)
    
    This differs from FiLM_layers.py's EncoderLayerFilm which uses:
        LayerNorm (WITH affine) -> then FiLM modulation on top
    
    The key difference is architectural:
    - EncoderLayerFilm: LN(x) = γ_LN * norm(x) + β_LN, then FiLM: (1+γ)·LN(x) + β
      This means FiLM modulates the ALREADY-AFFINE-TRANSFORMED output.
    - EnhancedEncoderLayerShared: norm(x), then EITHER γ_base·norm(x)+β_base 
      OR (1+γ_FiLM)·norm(x)+β_FiLM, then gate-mix the two.
      This gives TRUE parallel pathways with independent affine transforms.
    
    The latter design allows the base pathway to learn its own optimal affine
    transform, while the text pathway learns a different text-conditioned one.
    
    Args:
        attention: Attention layer module
        d_model: Model dimension
        d_ff: Feed-forward dimension
        text_dim: Text embedding dimension
        text_seq_len: Length of text sequence for FiLMGenerator
        gate_module: GateModule instance for mixing
        mixer: PathwayMixer instance
        dropout: Dropout rate
        activation: Activation function ("relu" or "gelu")
    """
    
    def __init__(
        self, 
        attention,
        d_model: int, 
        d_ff: int, 
        text_dim: int, 
        text_seq_len: int,
        gate_module: GateModule,
        mixer: PathwayMixer,
        dropout: float = 0.1,
        activation: str = "relu",
        text_film_per_channel: bool = True
    ):
        super().__init__()
        
        # Store dimensions for reference
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Shared attention block
        self.attention = attention
        
        # Shared FFN block
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # LayerNorms (no affine - we apply our own scale/shift via base or FiLM)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        
        # Base pathway affine parameters (post-attention)
        self.gamma_base_1 = nn.Parameter(torch.ones(d_model))
        self.beta_base_1 = nn.Parameter(torch.zeros(d_model))
        
        # Base pathway affine parameters (post-FFN)
        self.gamma_base_2 = nn.Parameter(torch.ones(d_model))
        self.beta_base_2 = nn.Parameter(torch.zeros(d_model))
        
        # Text pathway: FiLM generator produces gamma1, beta1, gamma2, beta2
        self.film_gen = FiLMGenerator(
            text_dim=text_dim, 
            output_dim=d_model, 
            seq_len=text_seq_len,
            hidden_dim=d_model,
            per_channel=text_film_per_channel
        )
        
        # Gate and mixer (shared references - not owned by this layer)
        self.gate = gate_module
        self.mixer = mixer
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        text_emb: torch.Tensor, 
        x_raw: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        apply_film: bool = True
    ) -> tuple:
        """
        Forward pass through the encoder layer.
        
        Args:
            x: Input tensor [B, C, d_model]
            text_emb: Text embeddings [B, C, L, text_dim]
            x_raw: Raw time series [B, seq_len, C] (for conditional gate)
            attn_mask: Optional attention mask
        
        Returns:
            tuple: (output [B, C, d_model], alpha gate value)
        """
        # ═══════════════════════════════════════════════════════════════
        # ATTENTION BLOCK
        # ═══════════════════════════════════════════════════════════════
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask)
        h = x + self.dropout(attn_out)
        
        # LayerNorm (no affine)
        h_norm = self.norm1(h)
        
        # ═══════════════════════════════════════════════════════════════
        # GENERATE FiLM PARAMETERS (or disable when requested)
        # ═══════════════════════════════════════════════════════════════
        if apply_film:
            gamma1, beta1, gamma2, beta2 = self.film_gen(text_emb)
        else:
            gamma1 = beta1 = gamma2 = beta2 = x.new_zeros(x.shape)
        
        # ═══════════════════════════════════════════════════════════════
        # PATHWAY SPLIT 1 (POST-ATTENTION)
        # ═══════════════════════════════════════════════════════════════
        # Base pathway: learned affine transform
        z_base_1 = self.gamma_base_1 * h_norm + self.beta_base_1
        
        # Text pathway: FiLM modulation (1 + gamma for multiplicative around identity)
        z_text_1 = (1 + gamma1) * h_norm + beta1
        
        # Compute gate value (or force base-only mixing when FiLM is disabled)
        if apply_film:
            alpha = self.gate(x_raw, text_emb)
        else:
            if self.gate.gate_type == "global":
                alpha = x.new_tensor(0.0)
            elif self.gate.gate_type == "channel":
                alpha = x.new_zeros(x.shape[1])
            else:
                alpha = x.new_zeros(x.shape[0], x.shape[1])
        
        # Mix pathways
        z_mix_1 = self.mixer(z_base_1, z_text_1, alpha)
        
        # ═══════════════════════════════════════════════════════════════
        # FFN BLOCK
        # ═══════════════════════════════════════════════════════════════
        f = self.ffn(z_mix_1)
        f = z_mix_1 + f  # Residual connection
        
        # LayerNorm (no affine)
        f_norm = self.norm2(f)
        
        # ═══════════════════════════════════════════════════════════════
        # PATHWAY SPLIT 2 (POST-FFN)
        # ═══════════════════════════════════════════════════════════════
        # Base pathway: learned affine transform
        z_base_2 = self.gamma_base_2 * f_norm + self.beta_base_2
        
        # Text pathway: FiLM modulation
        z_text_2 = (1 + gamma2) * f_norm + beta2
        
        # Mix pathways (reuse same alpha for both splits)
        z_mix_2 = self.mixer(z_base_2, z_text_2, alpha)
        
        return z_mix_2, alpha


class EnhancedEncoderLayerParallel(nn.Module):
    """
    Encoder layer with parallel frozen unimodal and learned text pathways.
    
    This implements the "parallel" pathway_type (Option 10):
    - Frozen unimodal pathway: Entire pre-trained layer (Attention + FFN + LayerNorm)
    - Learned text pathway: New Attention + FFN + FiLM-modulated LayerNorm
    - Gated mixing between pathway outputs
    
    This requires a pre-trained unimodal checkpoint to provide the frozen layer.
    The text pathway learns to complement/correct the frozen predictions.
    
    Args:
        frozen_layer: Pre-trained encoder layer (will be frozen)
        d_model: Model dimension
        d_ff: Feed-forward dimension
        text_dim: Text embedding dimension
        text_seq_len: Length of text sequence for FiLMGenerator
        gate_module: GateModule instance for mixing
        mixer: PathwayMixer instance
        n_heads: Number of attention heads
        dropout: Dropout rate
        activation: Activation function ("relu" or "gelu")
    """
    
    def __init__(
        self, 
        frozen_layer,
        d_model: int, 
        d_ff: int, 
        text_dim: int,
        text_seq_len: int,
        gate_module: GateModule,
        mixer: PathwayMixer,
        n_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "relu",
        text_film_per_channel: bool = True
    ):
        super().__init__()
        
        # Store dimensions for reference
        self.d_model = d_model
        self.d_ff = d_ff
        
        # ═══════════════════════════════════════════════════════════════
        # FROZEN UNIMODAL PATHWAY (entire layer from checkpoint)
        # ═══════════════════════════════════════════════════════════════
        self.frozen_layer = frozen_layer
        # Freeze all parameters
        for param in self.frozen_layer.parameters():
            param.requires_grad = False
        
        # ═══════════════════════════════════════════════════════════════
        # LEARNED TEXT PATHWAY (new attention + FFN + FiLM)
        # ═══════════════════════════════════════════════════════════════
        # New attention block for text pathway
        self.text_attention = AttentionLayer(
            FullAttention(False, attention_dropout=dropout),
            d_model, n_heads
        )
        
        # New FFN block for text pathway
        self.text_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # LayerNorms for text pathway (no affine - FiLM provides scale/shift)
        self.text_norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.text_norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        
        # FiLM generator for text pathway
        self.film_gen = FiLMGenerator(
            text_dim=text_dim,
            output_dim=d_model,
            seq_len=text_seq_len,
            hidden_dim=d_model,
            per_channel=text_film_per_channel
        )
        
        # Gate and mixer
        self.gate = gate_module
        self.mixer = mixer
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        text_emb: torch.Tensor, 
        x_raw: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        apply_film: bool = True
    ) -> tuple:
        """
        Forward pass through the parallel encoder layer.
        
        Args:
            x: Input tensor [B, C, d_model]
            text_emb: Text embeddings [B, C, L, text_dim]
            x_raw: Raw time series [B, seq_len, C] (for conditional gate)
            attn_mask: Optional attention mask
        
        Returns:
            tuple: (output [B, C, d_model], alpha gate value)
        """
        # ═══════════════════════════════════════════════════════════════
        # FROZEN UNIMODAL PATHWAY
        # ═══════════════════════════════════════════════════════════════
        with torch.no_grad():
            # Pass through frozen layer
            # The frozen layer is a standard EncoderLayer, expecting (x, attn_mask)
            z_uni, _ = self.frozen_layer(x, attn_mask=attn_mask)
        
        # ═══════════════════════════════════════════════════════════════
        # LEARNED TEXT PATHWAY
        # ═══════════════════════════════════════════════════════════════
        # Generate FiLM parameters (or disable when requested)
        if apply_film:
            gamma1, beta1, gamma2, beta2 = self.film_gen(text_emb)
        else:
            gamma1 = beta1 = gamma2 = beta2 = x.new_zeros(x.shape)
        
        # Attention block
        attn_out, _ = self.text_attention(x, x, x, attn_mask=attn_mask)
        h = x + self.dropout(attn_out)
        h_norm = self.text_norm1(h)
        
        # FiLM modulation 1
        z_film_1 = (1 + gamma1) * h_norm + beta1
        
        # FFN block
        f = self.text_ffn(z_film_1)
        f = z_film_1 + f  # Residual
        f_norm = self.text_norm2(f)
        
        # FiLM modulation 2
        z_text = (1 + gamma2) * f_norm + beta2
        
        # ═══════════════════════════════════════════════════════════════
        # MIXING
        # ═══════════════════════════════════════════════════════════════
        if apply_film:
            alpha = self.gate(x_raw, text_emb)
        else:
            if self.gate.gate_type == "global":
                alpha = x.new_tensor(0.0)
            elif self.gate.gate_type == "channel":
                alpha = x.new_zeros(x.shape[1])
            else:
                alpha = x.new_zeros(x.shape[0], x.shape[1])
        z = self.mixer(z_uni, z_text, alpha)
        
        return z, alpha


class EnhancedEncoderFilm(nn.Module):
    """
    Enhanced Encoder with FiLM modulation and pathway mixing.
    
    This encoder wraps multiple EnhancedEncoderLayer instances (shared or parallel)
    and handles the overall encoding process.
    
    Args:
        layers: List of encoder layers (EnhancedEncoderLayerShared or EnhancedEncoderLayerParallel)
        norm_layer: Optional final normalization layer
    """
    
    def __init__(self, layers: list, norm_layer=None, film_last_n=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.film_last_n = film_last_n

    def _should_apply_film(self, layer_idx: int) -> bool:
        """
        Decide whether to apply FiLM on the given layer index.

        Args:
            layer_idx: Index of the current encoder layer.

        Returns:
            bool: True if FiLM should be applied on this layer.
        """
        if self.film_last_n is None:
            return True
        if self.film_last_n <= 0:
            return False
        return layer_idx >= len(self.layers) - int(self.film_last_n)
    
    def forward(
        self, 
        x: torch.Tensor, 
        text_emb: torch.Tensor,
        x_raw: torch.Tensor = None,
        attn_mask: torch.Tensor = None
    ) -> tuple:
        """
        Forward pass through all encoder layers.
        
        Args:
            x: Input tensor [B, C, d_model]
            text_emb: Text embeddings [B, C, L, text_dim]
            x_raw: Raw time series [B, seq_len, C] (for conditional gates)
            attn_mask: Optional attention mask
        
        Returns:
            tuple: (output [B, C, d_model], list of alpha gate values per layer)
        """
        alphas = []
        
        for idx, layer in enumerate(self.layers):
            apply_film = self._should_apply_film(idx)
            x, alpha = layer(
                x,
                text_emb,
                x_raw=x_raw,
                attn_mask=attn_mask,
                apply_film=apply_film
            )
            alphas.append(alpha)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x, alphas
