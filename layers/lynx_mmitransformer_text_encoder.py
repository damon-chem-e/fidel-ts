"""
Text Encoder for lynx_mmitransformer

Embeds temporal news into time-agnostic global text variates.
Similar to DataEmbedding_inverted but for news data.
"""

import torch
import torch.nn as nn


class NewsEmbedding(nn.Module):
    """
    Embeds temporal news into time-agnostic representations.
    
    Similar to DataEmbedding_inverted, this takes temporal news [B, L, N, text_dim]
    and embeds it to time-agnostic variates [B, K, output_dim].
    
    Process:
    1. Aggregate news items per time step using full transformer
    2. Embed temporal sequence to time-agnostic representation (linear projection)
    3. Project directly to K global text variates without intermediate compression
    """
    
    def __init__(self, text_dim, output_dim, num_text_variates=1, 
                 seq_len=None, dropout=0.1, num_heads=4, num_layers=2):
        """
        Args:
            text_dim: Dimension of input text embeddings
            output_dim: Output dimension (d_model + M)
            num_text_variates: Number of global text variates K (default 1)
            seq_len: Temporal sequence length (required for temporal embedding)
            dropout: Dropout rate
            num_heads: Number of attention heads for news aggregation transformer
            num_layers: Number of transformer layers for news aggregation
        """
        super(NewsEmbedding, self).__init__()
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.num_text_variates = num_text_variates
        
        if seq_len is None:
            raise ValueError("seq_len must be provided for temporal embedding")
        
        # Step 1: Full transformer to aggregate news items per time step
        # Input: [B*L, N, text_dim] -> Output: [B*L, 1, text_dim] or [B*L, N, text_dim]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=num_heads,
            dim_feedforward=text_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.news_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(text_dim)
        )
        
        # Learnable query for aggregation (one per time step)
        # After transformer, we'll use this to aggregate all news items
        self.aggregation_query = nn.Parameter(torch.randn(1, 1, text_dim))
        
        # Step 2: Temporal embedding and direct projection to K variates
        # Input: [B, L, text_dim] -> Output: [B, K, output_dim]
        # No intermediate compression to [B, text_dim] - project directly from [B, L, text_dim]
        # Use learned projection that considers all L timesteps at once
        if num_text_variates == 1:
            # Project from [B, L, text_dim] directly to [B, 1, output_dim]
            self.variate_projection = nn.Linear(seq_len * text_dim, output_dim)
        else:
            # Create K different projections (multi-head style)
            # Each projection takes [B, L, text_dim] -> [B, output_dim]
            self.variate_projections = nn.ModuleList([
                nn.Linear(seq_len * text_dim, output_dim) 
                for _ in range(num_text_variates)
            ])
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, news_emb, news_mask=None):
        """
        Args:
            news_emb: [B, L, N, text_dim] - Temporal news embeddings
            news_mask: [B, L, N] - Mask for padded news items (optional)
            
        Returns:
            [B, K, output_dim] - Time-agnostic global text variates
        """
        B, L, N, D = news_emb.shape
        
        # Step 1: Aggregate news items per time step using full transformer
        # [B, L, N, text_dim] -> [B*L, N, text_dim]
        news_flat = news_emb.reshape(B * L, N, D)
        
        # Apply transformer to aggregate news items
        if news_mask is not None:
            # Create padding mask for transformer
            mask_flat = news_mask.reshape(B * L, N).bool()
            # Transformer expects True for positions to ignore
            padding_mask = ~mask_flat
            aggregated = self.news_transformer(news_flat, src_key_padding_mask=padding_mask)
        else:
            aggregated = self.news_transformer(news_flat)  # [B*L, N, text_dim]
        
        # Use learnable query to aggregate all news items into single representation
        # Expand query to match batch: [1, 1, text_dim] -> [B*L, 1, text_dim]
        query = self.aggregation_query.expand(B * L, -1, -1)  # [B*L, 1, text_dim]
        
        # Cross-attention: query attends to aggregated news
        # Use simple attention mechanism
        attn_weights = torch.softmax(
            torch.bmm(query, aggregated.transpose(1, 2)) / (D ** 0.5), 
            dim=-1
        )  # [B*L, 1, N]
        aggregated = torch.bmm(attn_weights, aggregated)  # [B*L, 1, text_dim]
        aggregated = aggregated.squeeze(1)  # [B*L, text_dim]
        
        # Reshape back: [B*L, text_dim] -> [B, L, text_dim]
        aggregated = aggregated.reshape(B, L, D)
        
        # Step 2: Project directly from [B, L, text_dim] to [B, K, output_dim]
        # No intermediate compression - consider all L timesteps when creating K variates
        # Flatten temporal dimension: [B, L, text_dim] -> [B, L*text_dim]
        aggregated_flat = aggregated.reshape(B, -1)  # [B, L*text_dim]
        
        if self.num_text_variates == 1:
            output = self.variate_projection(aggregated_flat)  # [B, output_dim]
            output = output.unsqueeze(1)  # [B, 1, output_dim]
        else:
            # Apply K different projections directly from flattened temporal representation
            outputs = []
            for proj in self.variate_projections:
                outputs.append(proj(aggregated_flat))  # [B, output_dim]
            output = torch.stack(outputs, dim=1)  # [B, K, output_dim]
        
        output = self.dropout(output)
        
        return output  # [B, K, output_dim]
