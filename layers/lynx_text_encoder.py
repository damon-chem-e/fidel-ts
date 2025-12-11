"""Text encoder without token sequence suppor (no sequence length dimension for text)."""

import torch
import torch.nn as nn


class TextItemsTransformer(nn.Module):
    """Attention over news items (N) per time step for 4D inputs [B, L, N, D]."""

    def __init__(self, text_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(TextItemsTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=num_heads,
            dim_feedforward=text_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(text_dim)
        )

    def forward(self, news_emb, news_mask=None):
        """
        Args:
            news_emb: [B, L, N, D]
            news_mask: [B, L, N] or None
        Returns:
            [B, L, D]
        """
        if len(news_emb.shape) != 4:
            raise ValueError(f"Expected 4D news_emb, got {news_emb.shape}")

        B, L, N, D = news_emb.shape
        if N == 1:
            return news_emb.squeeze(2)

        news_flat = news_emb.reshape(B * L, N, D)
        if news_mask is not None:
            mask_flat = news_mask.reshape(B * L, N).bool()
            aggregated = self.transformer(
                news_flat, src_key_padding_mask=~mask_flat
            )
        else:
            aggregated = self.transformer(news_flat)

        aggregated = aggregated.mean(dim=1)
        return aggregated.reshape(B, L, D)


class TemporalTransformer(nn.Module):
    """Transformer over temporal dimension L for 3D inputs [B, L, D]."""

    def __init__(self, text_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(TemporalTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=num_heads,
            dim_feedforward=text_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(text_dim)
        )

    def forward(self, temporal_emb):
        if len(temporal_emb.shape) != 3:
            raise ValueError(f"Expected 3D temporal_emb, got {temporal_emb.shape}")
        return self.transformer(temporal_emb)


class HierarchicalAggregator(nn.Module):
    """Option B: aggregate across N, then a stack of temporal transformers (4D only)."""

    def __init__(
        self,
        text_dim,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        num_blocks=1,
        num_temporal_layers_per_block=2,
    ):
        super(HierarchicalAggregator, self).__init__()
        self.text_items_transformer = TextItemsTransformer(text_dim, num_heads, num_layers, dropout)
        self.temporal_blocks = nn.ModuleList(
            [
                TemporalTransformer(text_dim, num_heads, num_temporal_layers_per_block, dropout)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, news_emb, news_mask=None):
        aggregated = self.text_items_transformer(news_emb, news_mask)
        for block in self.temporal_blocks:
            aggregated = block(aggregated)
        return aggregated


class FlatAggregator(nn.Module):
    """Option A: flatten L·N, attend jointly, mean-pool over N (4D only)."""

    def __init__(self, text_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(FlatAggregator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=num_heads,
            dim_feedforward=text_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(text_dim)
        )

    def forward(self, news_emb, news_mask=None):
        if len(news_emb.shape) != 4:
            raise ValueError(f"Expected 4D news_emb, got {news_emb.shape}")

        B, L, N, D = news_emb.shape
        news_flat = news_emb.reshape(B, L * N, D)
        if news_mask is not None:
            mask_flat = news_mask.reshape(B, L * N).bool()
            transformed = self.transformer(
                news_flat, src_key_padding_mask=~mask_flat
            )
        else:
            transformed = self.transformer(news_flat)

        transformed = transformed.reshape(B, L, N, D).mean(dim=2)
        return transformed


class MultiQueryAggregator(nn.Module):
    """K learnable queries attend over L to produce K variates (3D input)."""

    def __init__(self, text_dim, num_variates=1):
        super(MultiQueryAggregator, self).__init__()
        self.num_variates = num_variates
        self.queries = nn.Parameter(torch.randn(num_variates, 1, text_dim))

    def forward(self, temporal_emb):
        if len(temporal_emb.shape) != 3:
            raise ValueError(f"Expected 3D temporal_emb, got {temporal_emb.shape}")

        B, L, D = temporal_emb.shape
        K = self.num_variates
        queries = self.queries.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * K, 1, D)
        temporal_expanded = (
            temporal_emb.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, L, D)
        )
        attn_scores = torch.bmm(queries, temporal_expanded.transpose(1, 2)) / (
            D**0.5
        )
        attn_weights = torch.softmax(attn_scores, dim=-1)
        aggregated = torch.bmm(attn_weights, temporal_expanded).squeeze(1)
        return aggregated.reshape(B, K, D)


class VariateProjector(nn.Module):
    """Separate projection per variate to output_dim."""

    def __init__(self, text_dim, output_dim, num_variates=1):
        super(VariateProjector, self).__init__()
        self.num_variates = num_variates
        if num_variates == 1:
            self.projections = nn.Linear(text_dim, output_dim)
        else:
            self.projections = nn.ModuleList(
                [nn.Linear(text_dim, output_dim) for _ in range(num_variates)]
            )

    def forward(self, variates):
        if self.num_variates == 1:
            return self.projections(variates)
        outputs = [proj(variates[:, k, :]) for k, proj in enumerate(self.projections)]
        return torch.stack(outputs, dim=1)


class TextEmbedding(nn.Module):
    """Embeds 4D news [B, L, N, D] to K variates [B, K, output_dim] (no token seq)."""

    def __init__(
        self,
        text_dim,
        output_dim,
        num_text_variates=1,
        dropout=0.1,
        num_heads=4,
        num_layers=2,
        aggregation_type="hierarchical",
        num_blocks=1,
        num_temporal_layers_per_block=2,
    ):
        super(TextEmbedding, self).__init__()
        if aggregation_type == "hierarchical":
            self.aggregator = HierarchicalAggregator(
                text_dim,
                num_heads,
                num_layers,
                dropout,
                num_blocks,
                num_temporal_layers_per_block,
            )
        elif aggregation_type == "flat":
            self.aggregator = FlatAggregator(
                text_dim, num_heads, num_layers, dropout
            )
        else:
            raise ValueError(f"Unknown aggregation_type: {aggregation_type}")

        self.query_aggregator = MultiQueryAggregator(text_dim, num_text_variates)
        self.projector = VariateProjector(text_dim, output_dim, num_text_variates)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, news_emb, news_mask=None):
        if len(news_emb.shape) != 4:
            raise ValueError(
                f"TextEmbedding expects 4D input [B, L, N, D]; got {news_emb.shape}"
            )
        temporal_emb = self.aggregator(news_emb, news_mask)  # [B, L, D]
        variates = self.query_aggregator(temporal_emb)  # [B, K, D]
        output = self.projector(variates)  # [B, K, output_dim]
        return self.dropout(output)