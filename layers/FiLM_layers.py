import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMGenerator(nn.Module):
    def __init__(self, text_dim, output_dim, seq_len=1, hidden_dim=None):
        """
        Args:
            text_dim: Dimension of text embeddings
            output_dim: Dimension of modulation parameters (d_model)
            seq_len: Length of text sequence (L) to flatten
            hidden_dim: Hidden dimension of MLP
        """
        super().__init__()
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
        # x: [B, C, L, text_dim] or [B, C, L*text_dim] if pre-flattened
        if x.dim() == 4:
            B, C, L, D = x.shape
            x = x.view(B, C, L * D)
            
        params = self.net(x) # [B, C, output_dim * 4]
        
        # Split into 4 tensors
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
