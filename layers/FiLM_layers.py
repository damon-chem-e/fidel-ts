import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMGenerator(nn.Module):
    """
    Generates FiLM (Feature-wise Linear Modulation) parameters from text embeddings.
    
    FiLM Modulation Semantics
    =========================
    FiLM modulates features as: y = gamma * x + beta
    
    This generator produces (gamma, beta) pairs for transformer layers:
    - gamma1, beta1: Post-attention normalization modulation
    - gamma2, beta2: Post-FFN normalization modulation
    
    Key Design: GLOBAL Text Conditioning
    =====================================
    The generator FLATTENS ALL text timesteps into a single vector:
        [B, C, L, text_dim] -> [B, C, L * text_dim]
    
    Then produces ONE set of modulation parameters per channel:
        [B, C, L * text_dim] -> MLP -> [B, C, output_dim * 4]
    
    This means:
    - ALL text timesteps contribute to a SINGLE (gamma, beta) per channel
    - There is NO timestamp-specific modulation
    - All predictions receive the SAME modulation derived from aggregate text
    
    The intuition is that the text provides GLOBAL context (e.g., "storm approaching")
    that uniformly affects how the model processes the time series.
    
    CRITICAL: Fixed Input Dimension
    ===============================
    The input Linear layer has FIXED input_dim = seq_len * text_dim.
    The seq_len MUST match the actual text input length:
    - For timestamp_semantics='t_about': seq_len = ceil(pred_len / hetero_stride) (y_hetero)
    - For timestamp_semantics='t_known': seq_len = ceil(seq_len / hetero_stride) (x_hetero)
    
    A dimension mismatch will cause a runtime error!
    """
    
    def __init__(self, text_dim, output_dim, seq_len=1, hidden_dim=None):
        """
        Args:
            text_dim: Dimension of text embeddings (D in [B, C, L, D])
            output_dim: Dimension of modulation parameters (d_model)
            seq_len: Length of text sequence (L) that will be flattened.
                     MUST match actual input text length at runtime!
                     - t_about: ceil(pred_len / hetero_stride) 
                     - t_known: ceil(seq_len / hetero_stride)
            hidden_dim: Hidden dimension of MLP (defaults to input_dim)
        """
        super().__init__()
        # Store seq_len for debugging dimension mismatches
        self.expected_seq_len = seq_len
        self.text_dim = text_dim
        
        # Flatten input: [B, C, L, text_dim] -> [B, C, L * text_dim]
        input_dim = seq_len * text_dim
        hidden_dim = hidden_dim or input_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            # Output 4 sets of parameters: gamma1, beta1, gamma2, beta2
            nn.Linear(hidden_dim, output_dim * 4) 
        )
        
    def forward(self, x):
        """
        Generate FiLM parameters from text embeddings.
        
        Args:
            x: Text embeddings [B, C, L, text_dim] or [B, C, L*text_dim] if pre-flattened
               - B: batch size
               - C: number of channels (must match time series channels)
               - L: number of text timesteps (MUST equal self.expected_seq_len)
               - text_dim: embedding dimension (MUST equal self.text_dim)
        
        Returns:
            gamma1, beta1, gamma2, beta2: Each [B, C, output_dim]
            These are used by EncoderLayerFilm for modulation.
        """
        # Flatten text sequence: [B, C, L, D] -> [B, C, L*D]
        if x.dim() == 4:
            B, C, L, D = x.shape
            x = x.view(B, C, L * D)
            
        # Generate modulation parameters: [B, C, L*D] -> [B, C, output_dim * 4]
        params = self.net(x)
        
        # Split into 4 tensors, each [B, C, output_dim]
        gamma1, beta1, gamma2, beta2 = torch.chunk(params, 4, dim=-1)
        return gamma1, beta1, gamma2, beta2

class EncoderLayerFilm(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayerFilm, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, gamma1, beta1, gamma2, beta2, attn_mask=None, tau=None, delta=None):
        # x: [B, N, E]
        
        # 1. Self Attention Block
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        # FiLM Modulation 1 (Post-Attention Norm)
        x = (1 + gamma1) * x + beta1
        
        # 2. Feed Forward Block
        y = x
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        
        x = self.norm2(x + y)
        
        # FiLM Modulation 2 (Post-FFN Norm)
        x = (1 + gamma2) * x + beta2
        
        return x, attn

class EncoderFilm(nn.Module):
    def __init__(self, attn_layers, film_generators, conv_layers=None, norm_layer=None):
        super(EncoderFilm, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.film_generators = nn.ModuleList(film_generators) # One generator per layer
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, text_emb, attn_mask=None, tau=None, delta=None):
        # x: [B, N, E]
        # text_emb: [B, C, L, text_dim] (C=N)
        
        attns = []
        if self.conv_layers is not None:
            # Conv layers logic (usually not used in iTransformer, but keeping for compatibility)
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                
                # Generate FiLM params for this layer
                gamma1, beta1, gamma2, beta2 = self.film_generators[i](text_emb)
                
                x, attn = attn_layer(x, gamma1, beta1, gamma2, beta2, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            
            # Last layer
            gamma1, beta1, gamma2, beta2 = self.film_generators[-1](text_emb)
            x, attn = self.attn_layers[-1](x, gamma1, beta1, gamma2, beta2, tau=tau, delta=None)
            attns.append(attn)
        else:
            for i, attn_layer in enumerate(self.attn_layers):
                # Generate FiLM params for this layer
                gamma1, beta1, gamma2, beta2 = self.film_generators[i](text_emb)
                
                x, attn = attn_layer(x, gamma1, beta1, gamma2, beta2, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
