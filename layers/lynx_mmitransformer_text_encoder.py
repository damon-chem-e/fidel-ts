"""
Text Encoder for lynx_mmitransformer

Embeds temporal news into time-agnostic global text variates.
Similar to DataEmbedding_inverted but for news data.
"""

import torch
import torch.nn as nn


class NewsTransformer(nn.Module):
    """
    Transformer that processes news items (N dimension) per time step.
    Input: [B, L, N, D] -> Output: [B, L, D]
    """
    
    def __init__(self, text_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(NewsTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=num_heads,
            dim_feedforward=text_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(text_dim)
        )
    
    def forward(self, news_emb, news_mask=None):
        """
        Args:
            news_emb: [B, L, N, D] - News embeddings per time step
            news_mask: [B, L, N] - Mask for padded news items (optional)
        Returns:
            [B, L, D] - Aggregated news per time step
        """
        B, L, N, D = news_emb.shape
        
        # Reshape to process each time step independently
        news_flat = news_emb.reshape(B * L, N, D)  # [B*L, N, D]
        
        # Apply transformer
        if news_mask is not None:
            mask_flat = news_mask.reshape(B * L, N).bool()
            padding_mask = ~mask_flat
            aggregated = self.transformer(news_flat, src_key_padding_mask=padding_mask)
        else:
            aggregated = self.transformer(news_flat)  # [B*L, N, D]
        
        # Aggregate all news items per time step (mean pooling)
        aggregated = aggregated.mean(dim=1)  # [B*L, D]
        
        # Reshape back
        aggregated = aggregated.reshape(B, L, D)  # [B, L, D]
        
        return aggregated


class TemporalTransformer(nn.Module):
    """
    Transformer that processes temporal sequence (L dimension).
    Input: [B, L, D] -> Output: [B, L, D]
    """
    
    def __init__(self, text_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(TemporalTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=num_heads,
            dim_feedforward=text_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(text_dim)
        )
    
    def forward(self, temporal_emb):
        """
        Args:
            temporal_emb: [B, L, D] - Temporal sequence
        Returns:
            [B, L, D] - Processed temporal sequence
        """
        return self.transformer(temporal_emb)


class HierarchicalAggregator(nn.Module):
    """
    Hierarchical aggregation: first across news items (N), then across time (L).
    Option B - Default approach.
    """
    
    def __init__(self, text_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(HierarchicalAggregator, self).__init__()
        self.news_transformer = NewsTransformer(text_dim, num_heads, num_layers, dropout)
        self.temporal_transformer = TemporalTransformer(text_dim, num_heads, num_layers, dropout)
    
    def forward(self, news_emb, news_mask=None):
        """
        Args:
            news_emb: [B, L, N, D] - Temporal news embeddings
            news_mask: [B, L, N] - Mask for padded news items (optional)
        Returns:
            [B, L, D] - Processed temporal sequence
        """
        # Step 1: Aggregate news items per time step
        aggregated = self.news_transformer(news_emb, news_mask)  # [B, L, D]
        
        # Step 2: Process temporal sequence
        output = self.temporal_transformer(aggregated)  # [B, L, D]
        
        return output


class FlatAggregator(nn.Module):
    """
    Flat aggregation: process all L·N positions simultaneously.
    Option A - Good when L or N is small.
    """
    
    def __init__(self, text_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(FlatAggregator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=num_heads,
            dim_feedforward=text_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(text_dim)
        )
    
    def forward(self, news_emb, news_mask=None):
        """
        Args:
            news_emb: [B, L, N, D] - Temporal news embeddings
            news_mask: [B, L, N] - Mask for padded news items (optional)
        Returns:
            [B, L, D] - Aggregated representation (mean pooled over N)
        """
        B, L, N, D = news_emb.shape
        
        # Reshape to [B, L·N, D]
        news_flat = news_emb.reshape(B, L * N, D)
        
        # Apply transformer across all L·N positions
        if news_mask is not None:
            mask_flat = news_mask.reshape(B, L * N).bool()
            padding_mask = ~mask_flat
            aggregated = self.transformer(news_flat, src_key_padding_mask=padding_mask)
        else:
            aggregated = self.transformer(news_flat)  # [B, L·N, D]
        
        # Reshape back and mean pool over N dimension
        aggregated = aggregated.reshape(B, L, N, D)  # [B, L, N, D]
        aggregated = aggregated.mean(dim=2)  # [B, L, D]
        
        return aggregated


class MultiQueryAggregator(nn.Module):
    """
    Uses K learnable queries to aggregate temporal sequence into K variates.
    Each query learns to extract different information.
    """
    
    def __init__(self, text_dim, num_variates=1):
        super(MultiQueryAggregator, self).__init__()
        self.num_variates = num_variates
        
        # K learnable queries, each of dimension text_dim
        self.queries = nn.Parameter(torch.randn(num_variates, 1, text_dim))
    
    def forward(self, temporal_emb):
        """
        Args:
            temporal_emb: [B, L, D] - Temporal sequence (from aggregator)
        Returns:
            [B, K, D] - K different aggregated representations
        """
        B, L, D = temporal_emb.shape
        K = self.num_variates
        
        # Expand queries to match batch: [K, 1, D] -> [B, K, 1, D]
        queries = self.queries.unsqueeze(0).expand(B, -1, -1, -1)  # [B, K, 1, D]
        queries = queries.reshape(B * K, 1, D)  # [B*K, 1, D]
        
        # Expand temporal_emb for each query: [B, L, D] -> [B*K, L, D]
        temporal_expanded = temporal_emb.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, L, D]
        temporal_expanded = temporal_expanded.reshape(B * K, L, D)  # [B*K, L, D]
        
        # Scaled dot-product attention for each query
        attn_scores = torch.bmm(queries, temporal_expanded.transpose(1, 2)) / (D ** 0.5)  # [B*K, 1, L]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B*K, 1, L]
        
        # Weighted aggregation
        aggregated = torch.bmm(attn_weights, temporal_expanded)  # [B*K, 1, D]
        aggregated = aggregated.squeeze(1)  # [B*K, D]
        
        # Reshape back: [B*K, D] -> [B, K, D]
        aggregated = aggregated.reshape(B, K, D)
        
        return aggregated


class VariateProjector(nn.Module):
    """
    Projects K variates from text_dim to output_dim.
    Uses separate projection for each variate.
    """
    
    def __init__(self, text_dim, output_dim, num_variates=1):
        super(VariateProjector, self).__init__()
        self.num_variates = num_variates
        
        # K separate projections
        if num_variates == 1:
            self.projections = nn.Linear(text_dim, output_dim)
        else:
            self.projections = nn.ModuleList([
                nn.Linear(text_dim, output_dim)
                for _ in range(num_variates)
            ])
    
    def forward(self, variates):
        """
        Args:
            variates: [B, K, text_dim] - K variates
        Returns:
            [B, K, output_dim] - Projected variates
        """
        if self.num_variates == 1:
            output = self.projections(variates)  # [B, 1, output_dim]
        else:
            outputs = []
            for k, proj in enumerate(self.projections):
                outputs.append(proj(variates[:, k, :]))  # [B, output_dim]
            output = torch.stack(outputs, dim=1)  # [B, K, output_dim]
        
        return output


class NewsEmbedding(nn.Module):
    """
    Embeds temporal news into time-agnostic representations.
    
    Similar to DataEmbedding_inverted, this takes temporal news [B, L, N, text_dim]
    and embeds it to time-agnostic variates [B, K, output_dim].
    
    Process:
    1. Aggregate news items (hierarchical or flat)
    2. Use K queries to extract K different representations
    3. Project each to output_dim
    """
    
    def __init__(self, text_dim, output_dim, num_text_variates=1, 
                 seq_len=None, dropout=0.1, num_heads=4, num_layers=2,
                 aggregation_type='hierarchical'):
        """
        Args:
            text_dim: Dimension of input text embeddings
            output_dim: Output dimension (d_model + M)
            num_text_variates: Number of global text variates K (default 1)
            seq_len: Temporal sequence length (required for temporal embedding)
            dropout: Dropout rate
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            aggregation_type: 'hierarchical' (default) or 'flat'
        """
        super(NewsEmbedding, self).__init__()
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.num_text_variates = num_text_variates
        
        if seq_len is None:
            raise ValueError("seq_len must be provided")
        
        # Step 1: News aggregation (hierarchical or flat)
        if aggregation_type == 'hierarchical':
            self.aggregator = HierarchicalAggregator(
                text_dim, num_heads, num_layers, dropout
            )
        elif aggregation_type == 'flat':
            self.aggregator = FlatAggregator(
                text_dim, num_heads, num_layers, dropout
            )
        else:
            raise ValueError(f"Unknown aggregation_type: {aggregation_type}")
        
        # Step 2: Multi-query aggregation
        self.query_aggregator = MultiQueryAggregator(
            text_dim, num_text_variates
        )
        
        # Step 3: Variate projection
        self.projector = VariateProjector(
            text_dim, output_dim, num_text_variates
        )
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, news_emb, news_mask=None):
        """
        Args:
            news_emb: [B, L, N, text_dim] - Temporal news embeddings
            news_mask: [B, L, N] - Mask for padded news items (optional)
            
        Returns:
            [B, K, output_dim] - Time-agnostic global text variates
        """
        # Step 1: Aggregate news items
        temporal_emb = self.aggregator(news_emb, news_mask)  # [B, L, text_dim]
        
        # Step 2: Use K queries to extract K different representations
        variates = self.query_aggregator(temporal_emb)  # [B, K, text_dim]
        
        # Step 3: Project each variate to output_dim
        output = self.projector(variates)  # [B, K, output_dim]
        
        output = self.dropout(output)
        
        return output  # [B, K, output_dim]
