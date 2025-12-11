"""
Text Encoder for lynx_mmitransformer

Embeds temporal news into time-agnostic global text variates.
Similar to DataEmbedding_inverted but for news data.

Note: When N=1 (concatenated articles per time step), NewsTransformer is essentially
skipped and we proceed directly to temporal attention. If sequence dimension is available,
text self-attention can be applied within each article's token sequence.
"""

import torch
import torch.nn as nn
import warnings


class NewsTransformer(nn.Module):
    """
    Transformer that processes news items (N dimension) per time step.
    Supports both 4D input [B, L, N, D] and 5D input [B, L, N, seq_len, D].
    When N=1, this essentially becomes a pass-through (no aggregation needed).
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
            news_emb: [B, L, N, D] or [B, L, N, seq_len, D] - News embeddings per time step
            news_mask: [B, L, N] - Mask for padded news items (optional)
        Returns:
            [B, L, D] or [B, L, seq_len, D] - Aggregated news per time step
        """
        if len(news_emb.shape) == 4:
            # Standard case: [B, L, N, D]
            B, L, N, D = news_emb.shape
            
            # If N=1, no aggregation needed - just reshape
            if N == 1:
                return news_emb.squeeze(2)  # [B, L, D]
            
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
        
        elif len(news_emb.shape) == 5:
            # Sequence dimension case: [B, L, N, seq_len, D]
            B, L, N, seq_len, D = news_emb.shape
            
            # If N=1, no aggregation needed - just reshape
            if N == 1:
                return news_emb.squeeze(2)  # [B, L, seq_len, D]
            
            # Reshape to process each time step independently
            # [B, L, N, seq_len, D] -> [B*L, N, seq_len, D]
            news_flat = news_emb.reshape(B * L, N, seq_len, D)
            
            # For now, mean pool over sequence dimension first, then aggregate news items
            # This could be enhanced to do proper 2D attention
            news_flat = news_flat.mean(dim=2)  # [B*L, N, D]
            
            # Apply transformer across news items
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
        
        else:
            raise ValueError(f"Unexpected news_emb shape: {news_emb.shape}")


class TextSelfAttention(nn.Module):
    """
    Self-attention over token sequence dimension within each time step.
    Input: [B, L, seq_len, D] -> Output: [B, L, seq_len, D]
    Only used when sequence dimension is available.
    """
    
    def __init__(self, text_dim, num_heads=4, num_layers=1, dropout=0.1):
        super(TextSelfAttention, self).__init__()
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
    
    def forward(self, text_emb, text_mask=None):
        """
        Args:
            text_emb: [B, L, seq_len, D] - Text sequence per time step
            text_mask: [B, L, seq_len] - Mask for padded tokens (optional)
        Returns:
            [B, L, seq_len, D] - Processed text sequence
        """
        B, L, seq_len, D = text_emb.shape
        
        # Reshape to process each time step independently
        text_flat = text_emb.reshape(B * L, seq_len, D)  # [B*L, seq_len, D]
        
        # Apply transformer
        if text_mask is not None:
            mask_flat = text_mask.reshape(B * L, seq_len).bool()
            padding_mask = ~mask_flat
            output = self.transformer(text_flat, src_key_padding_mask=padding_mask)
        else:
            output = self.transformer(text_flat)  # [B*L, seq_len, D]
        
        # Reshape back
        output = output.reshape(B, L, seq_len, D)  # [B, L, seq_len, D]
        
        return output


class TemporalTransformer(nn.Module):
    """
    Transformer that processes temporal sequence (L dimension).
    Input: [B, L, D] -> Output: [B, L, D]
    Also supports [B, L, seq_len, D] by processing across L for each seq_len position.
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
            temporal_emb: [B, L, D] or [B, L, seq_len, D] - Temporal sequence
        Returns:
            [B, L, D] or [B, L, seq_len, D] - Processed temporal sequence
        """
        if len(temporal_emb.shape) == 3:
            # Standard: [B, L, D]
            return self.transformer(temporal_emb)
        elif len(temporal_emb.shape) == 4:
            # Sequence dimension: [B, L, seq_len, D]
            # Process temporal dimension (L) for each sequence position
            B, L, seq_len, D = temporal_emb.shape
            # Reshape: [B, L, seq_len, D] -> [B*seq_len, L, D]
            temporal_reshaped = temporal_emb.permute(0, 2, 1, 3).reshape(B * seq_len, L, D)
            # Apply transformer across L dimension
            output = self.transformer(temporal_reshaped)  # [B*seq_len, L, D]
            # Reshape back: [B*seq_len, L, D] -> [B, L, seq_len, D]
            output = output.reshape(B, seq_len, L, D).permute(0, 2, 1, 3)
            return output
        else:
            raise ValueError(f"Unexpected temporal_emb shape: {temporal_emb.shape}")


class AlternatingBlock(nn.Module):
    """
    Alternating block: temporal transformer layers followed by text self-attention layers.
    Toto-style architecture.
    """
    
    def __init__(self, text_dim, num_temporal_layers=2, num_text_layers=2, 
                 num_heads=4, dropout=0.1, use_text_sequence=False):
        super(AlternatingBlock, self).__init__()
        self.use_text_sequence = use_text_sequence
        
        # Temporal transformer layers
        self.temporal_layers = nn.ModuleList([
            TemporalTransformer(text_dim, num_heads, 1, dropout)
            for _ in range(num_temporal_layers)
        ])
        
        # Text self-attention layers (only if sequence dimension available)
        if use_text_sequence:
            self.text_layers = nn.ModuleList([
                TextSelfAttention(text_dim, num_heads, 1, dropout)
                for _ in range(num_text_layers)
            ])
        else:
            self.text_layers = None
    
    def forward(self, temporal_emb, text_mask=None):
        """
        Args:
            temporal_emb: [B, L, D] or [B, L, seq_len, D] - Temporal sequence
            text_mask: [B, L, seq_len] or [B, L, N, seq_len] - Mask for padded tokens (optional)
        Returns:
            [B, L, D] or [B, L, seq_len, D] - Processed sequence
        """
        # Apply temporal transformer layers
        # TemporalTransformer now handles both 3D and 4D inputs
        for layer in self.temporal_layers:
            temporal_emb = layer(temporal_emb)
        
        # Apply text self-attention layers (if available)
        if self.use_text_sequence and self.text_layers is not None:
            if len(temporal_emb.shape) == 4:
                # [B, L, seq_len, D]
                # Handle text_mask shape
                if text_mask is not None and len(text_mask.shape) == 4:
                    # [B, L, N, seq_len] -> mean pool over N or use first
                    if text_mask.shape[2] == 1:
                        text_mask = text_mask.squeeze(2)  # [B, L, seq_len]
                    else:
                        # Use first news item's mask or mean
                        text_mask = text_mask[:, :, 0, :]  # [B, L, seq_len]
                
                for layer in self.text_layers:
                    temporal_emb = layer(temporal_emb, text_mask)  # [B, L, seq_len, D]
            else:
                # Sequence dimension not available - skip with warning
                warnings.warn(
                    "Text self-attention layers requested but sequence dimension not available. "
                    "Skipping text self-attention layers.",
                    UserWarning
                )
        
        return temporal_emb


class HierarchicalAggregator(nn.Module):
    """
    Hierarchical aggregation: first across news items (N), then alternating blocks.
    Option B - Default approach.
    """
    
    def __init__(self, text_dim, num_heads=4, num_layers=2, dropout=0.1,
                 num_blocks=1, num_temporal_layers_per_block=2, num_text_layers_per_block=2,
                 use_text_sequence=False):
        super(HierarchicalAggregator, self).__init__()
        self.use_text_sequence = use_text_sequence
        
        # News aggregation (only one step, only when N>1)
        self.news_transformer = NewsTransformer(text_dim, num_heads, num_layers, dropout)
        
        # Alternating blocks
        self.blocks = nn.ModuleList([
            AlternatingBlock(
                text_dim, 
                num_temporal_layers_per_block, 
                num_text_layers_per_block,
                num_heads, 
                dropout,
                use_text_sequence
            )
            for _ in range(num_blocks)
        ])
    
    def forward(self, news_emb, news_mask=None, text_mask=None):
        """
        Args:
            news_emb: [B, L, N, D] or [B, L, N, seq_len, D] - Temporal news embeddings
            news_mask: [B, L, N] - Mask for padded news items (optional)
            text_mask: [B, L, N, seq_len] or [B, L, seq_len] - Mask for padded tokens (optional)
        Returns:
            [B, L, D] or [B, L, seq_len, D] - Processed temporal sequence
        """
        # Step 1: Aggregate news items per time step (only when N>1)
        aggregated = self.news_transformer(news_emb, news_mask)
        
        # Step 2: Apply alternating blocks
        for block in self.blocks:
            aggregated = block(aggregated, text_mask)
        
        return aggregated


class FlatAggregator(nn.Module):
    """
    Flat aggregation: process all L·N positions simultaneously.
    Option A - Good when L or N is small.
    Supports multiple layers.
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
            news_emb: [B, L, N, D] or [B, L, N, seq_len, D] - Temporal news embeddings
            news_mask: [B, L, N] - Mask for padded news items (optional)
        Returns:
            [B, L, D] - Aggregated representation (mean pooled over N and seq_len if present)
        """
        if len(news_emb.shape) == 4:
            # Standard: [B, L, N, D]
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
        
        elif len(news_emb.shape) == 5:
            # Sequence dimension: [B, L, N, seq_len, D]
            B, L, N, seq_len, D = news_emb.shape
            
            # Reshape to [B, L·N·seq_len, D]
            news_flat = news_emb.reshape(B, L * N * seq_len, D)
            
            # Apply transformer across all L·N·seq_len positions
            aggregated = self.transformer(news_flat)  # [B, L·N·seq_len, D]
            
            # Reshape back and mean pool over N and seq_len dimensions
            aggregated = aggregated.reshape(B, L, N, seq_len, D)  # [B, L, N, seq_len, D]
            aggregated = aggregated.mean(dim=2).mean(dim=2)  # [B, L, D]
            
            return aggregated
        
        else:
            raise ValueError(f"Unexpected news_emb shape: {news_emb.shape}")


class MultiQueryAggregator(nn.Module):
    """
    Uses K learnable queries to aggregate temporal sequence into K variates.
    Each query learns to extract different information.
    Supports both 3D [B, L, D] and 4D [B, L, seq_len, D] inputs.
    """
    
    def __init__(self, text_dim, num_variates=1):
        super(MultiQueryAggregator, self).__init__()
        self.num_variates = num_variates
        
        # K learnable queries, each of dimension text_dim
        self.queries = nn.Parameter(torch.randn(num_variates, 1, text_dim))
    
    def forward(self, temporal_emb):
        """
        Args:
            temporal_emb: [B, L, D] or [B, L, seq_len, D] - Temporal sequence
        Returns:
            [B, K, D] - K different aggregated representations
        """
        if len(temporal_emb.shape) == 3:
            # Standard: [B, L, D]
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
        
        elif len(temporal_emb.shape) == 4:
            # Sequence dimension: [B, L, seq_len, D]
            # Mean pool over sequence dimension first, then aggregate
            temporal_pooled = temporal_emb.mean(dim=2)  # [B, L, D]
            return self.forward(temporal_pooled)  # [B, K, D]
        
        else:
            raise ValueError(f"Unexpected temporal_emb shape: {temporal_emb.shape}")


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
    or [B, L, N, seq_len, text_dim] and embeds it to time-agnostic variates [B, K, output_dim].
    
    Process:
    1. Aggregate news items (hierarchical or flat)
    2. Use K queries to extract K different representations
    3. Project each to output_dim
    
    Note: When sequence dimension is available (5D input), text self-attention can be applied.
    When sequence dimension is unavailable (4D input), text self-attention is skipped with warning.
    """
    
    def __init__(self, text_dim, output_dim, num_text_variates=1, 
                 seq_len=None, dropout=0.1, num_heads=4, num_layers=2,
                 aggregation_type='hierarchical', use_text_sequence=False,
                 num_blocks=1, num_temporal_layers_per_block=2, num_text_layers_per_block=2):
        """
        Args:
            text_dim: Dimension of input text embeddings
            output_dim: Output dimension (d_model + M)
            num_text_variates: Number of global text variates K (default 1)
            seq_len: Temporal sequence length (required for temporal embedding)
            dropout: Dropout rate
            num_heads: Number of attention heads
            num_layers: Number of transformer layers (for news aggregation or flat)
            aggregation_type: 'hierarchical' (default) or 'flat'
            use_text_sequence: Whether to expect sequence dimension in input (explicit config)
            num_blocks: Number of alternating blocks (for hierarchical)
            num_temporal_layers_per_block: Temporal transformer layers per block
            num_text_layers_per_block: Text self-attention layers per block
        """
        super(NewsEmbedding, self).__init__()
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.num_text_variates = num_text_variates
        self.use_text_sequence = use_text_sequence
        
        if seq_len is None:
            raise ValueError("seq_len must be provided")
        
        # Step 1: News aggregation (hierarchical or flat)
        if aggregation_type == 'hierarchical':
            self.aggregator = HierarchicalAggregator(
                text_dim, num_heads, num_layers, dropout,
                num_blocks, num_temporal_layers_per_block, num_text_layers_per_block,
                use_text_sequence
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
    
    def forward(self, news_emb, news_mask=None, text_mask=None):
        """
        Args:
            news_emb: [B, L, N, text_dim] or [B, L, N, seq_len, text_dim] - Temporal news embeddings
            news_mask: [B, L, N] - Mask for padded news items (optional)
            text_mask: [B, L, N, seq_len] or [B, L, seq_len] - Mask for padded tokens (optional)
            
        Returns:
            [B, K, output_dim] - Time-agnostic global text variates
        """
        # Detect input shape
        input_dim = len(news_emb.shape)
        
        if input_dim == 4:
            # Standard: [B, L, N, text_dim]
            if self.use_text_sequence:
                warnings.warn(
                    "use_text_sequence=True but input is 4D (no sequence dimension). "
                    "Text self-attention layers will be skipped.",
                    UserWarning
                )
        elif input_dim == 5:
            # Sequence dimension: [B, L, N, seq_len, text_dim]
            if not self.use_text_sequence:
                warnings.warn(
                    "Input is 5D (has sequence dimension) but use_text_sequence=False. "
                    "Sequence dimension will be mean-pooled. Set use_text_sequence=True to use text self-attention.",
                    UserWarning
                )
        else:
            raise ValueError(f"Unexpected news_emb shape: {news_emb.shape}. Expected 4D or 5D.")
        
        # Step 1: Aggregate news items
        temporal_emb = self.aggregator(news_emb, news_mask, text_mask)
        
        # Step 2: Use K queries to extract K different representations
        variates = self.query_aggregator(temporal_emb)  # [B, K, text_dim]
        
        # Step 3: Project each variate to output_dim
        output = self.projector(variates)  # [B, K, output_dim]
        
        output = self.dropout(output)
        
        return output  # [B, K, output_dim]
