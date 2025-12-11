"""Text encoder with token sequence support (5D input)."""

import torch
import torch.nn as nn


class TokenSelfAttention(nn.Module):
    """Self-attention over seq_len per article/time."""

    def __init__(self, text_dim, num_heads=4, num_layers=1, dropout=0.1):
        super(TokenSelfAttention, self).__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=num_heads,
            dim_feedforward=text_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(text_dim))

    def forward(self, tokens, token_mask=None):
        """
        tokens: [B, L, N, seq_len, D]
        token_mask: [B, L, N, seq_len] or None
        """
        B, L, N, seq_len, D = tokens.shape
        flat = tokens.reshape(B * L * N, seq_len, D)
        mask = None
        if token_mask is not None:
            mask = ~token_mask.reshape(B * L * N, seq_len).bool()
        out = self.encoder(flat, src_key_padding_mask=mask)
        return out.reshape(B, L, N, seq_len, D)


class TokenAggregator(nn.Module):
    """K queries attend over seq_len per article to get K heads."""

    def __init__(self, text_dim, num_variates):
        super(TokenAggregator, self).__init__()
        self.num_variates = num_variates
        self.queries = nn.Parameter(torch.randn(num_variates, 1, text_dim))

    def forward(self, token_emb, token_mask=None):
        """
        token_emb: [B, L, N, seq_len, D]
        token_mask: [B, L, N, seq_len] or None
        returns: [B, L, N, K, D]
        """
        B, L, N, seq_len, D = token_emb.shape
        K = self.num_variates
        emb_flat = token_emb.reshape(B * L * N, seq_len, D)
        queries = self.queries.unsqueeze(0).expand(B * L * N, -1, -1, -1).reshape(B * L * N * K, 1, D)
        emb_exp = emb_flat.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * L * N * K, seq_len, D)
        scores = torch.bmm(queries, emb_exp.transpose(1, 2)) / (D**0.5)
        if token_mask is not None:
            mask = ~token_mask.unsqueeze(3).expand(-1, -1, -1, K, -1).reshape(B * L * N * K, seq_len).bool()
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        agg = torch.bmm(weights, emb_exp).reshape(B, L, N, K, D)
        return agg


class ArticlePooler(nn.Module):
    """Attention over articles N dimension to condense to [B, L, K, D]."""

    def __init__(self, text_dim, num_heads=4, num_layers=1, dropout=0.1):
        super(ArticlePooler, self).__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=num_heads,
            dim_feedforward=text_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(text_dim))

    def forward(self, text_heads, news_mask=None):
        """
        text_heads: [B, L, N, K, D]
        news_mask: [B, L, N] or None
        returns: [B, L, K, D]
        """
        B, L, N, K, D = text_heads.shape
        # Reshape to process each (L, K) combination independently
        # [B, L, N, K, D] -> [B*L*K, N, D]
        heads_flat = text_heads.permute(0, 1, 3, 2, 4).reshape(B * L * K, N, D)
        
        # Create padding mask if news_mask provided
        padding_mask = None
        if news_mask is not None:
            # news_mask: [B, L, N] -> expand to [B*L*K, N]
            padding_mask = ~news_mask.unsqueeze(2).expand(-1, -1, K, -1).reshape(B * L * K, N).bool()
        
        # Apply transformer attention over N dimension
        attended = self.encoder(heads_flat, src_key_padding_mask=padding_mask)  # [B*L*K, N, D]
        
        # Mean pool over N dimension (after attention)
        pooled = attended.mean(dim=1)  # [B*L*K, D]
        
        # Reshape back: [B*L*K, D] -> [B, L, K, D]
        return pooled.reshape(B, L, K, D)


class TemporalTransformerSeq(nn.Module):
    """Temporal transformer over L; default per-head weights."""

    def __init__(
        self,
        text_dim,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        per_head=True,
        num_variates=1,
    ):
        super(TemporalTransformerSeq, self).__init__()
        self.per_head = per_head
        if per_head:
            self.shared = None
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=text_dim,
                nhead=num_heads,
                dim_feedforward=text_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.shared = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(text_dim))
        self.layers = None
        if per_head:
            layer = nn.TransformerEncoderLayer(
                d_model=text_dim,
                nhead=num_heads,
                dim_feedforward=text_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoder(
                        layer, num_layers=num_layers, norm=nn.LayerNorm(text_dim)
                    )
                    for _ in range(num_variates)
                ]
            )

    def forward(self, seq_emb, time_mask=None):
        """
        seq_emb: [B, L, K, D]
        time_mask: [B, L] or None
        returns: [B, L, K, D]
        """
        B, L, K, D = seq_emb.shape
        padding_mask = None
        if time_mask is not None:
            padding_mask = ~time_mask.bool()
        if self.per_head:
            outs = []
            for k in range(K):
                out = self.layers[k](seq_emb[:, :, k, :], src_key_padding_mask=padding_mask)
                outs.append(out.unsqueeze(2))
            return torch.cat(outs, dim=2)
        seq_flat = seq_emb.reshape(B * K, L, D)
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).expand(-1, K, -1).reshape(B * K, L)
        out = self.shared(seq_flat, src_key_padding_mask=padding_mask)
        return out.reshape(B, K, L, D).permute(0, 2, 1, 3)


class TemporalAggregator(nn.Module):
    """K queries attend over L to produce [B, K, D]."""

    def __init__(self, text_dim, num_variates):
        super(TemporalAggregator, self).__init__()
        self.num_variates = num_variates
        self.queries = nn.Parameter(torch.randn(num_variates, 1, text_dim))

    def forward(self, temporal_emb, time_mask=None):
        """
        temporal_emb: [B, L, K, D]
        time_mask: [B, L] or None
        returns: [B, K, D]
        """
        B, L, K, D = temporal_emb.shape
        queries = self.queries.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * K, 1, D)
        temp_flat = temporal_emb.permute(0, 2, 1, 3).reshape(B * K, L, D)
        scores = torch.bmm(queries, temp_flat.transpose(1, 2)) / (D**0.5)
        if time_mask is not None:
            mask = ~time_mask.unsqueeze(1).expand(-1, K, -1).reshape(B * K, L).bool()
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        agg = torch.bmm(weights, temp_flat).reshape(B, K, D)
        return agg


class VariateProjectorSeq(nn.Module):
    """Separate projection per variate to output_dim."""

    def __init__(self, text_dim, output_dim, num_variates=1, dropout=0.1):
        super(VariateProjectorSeq, self).__init__()
        self.num_variates = num_variates
        if num_variates == 1:
            self.projections = nn.Linear(text_dim, output_dim)
        else:
            self.projections = nn.ModuleList(
                [nn.Linear(text_dim, output_dim) for _ in range(num_variates)]
            )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, variates):
        if self.num_variates == 1:
            return self.dropout(self.projections(variates))
        outputs = [proj(variates[:, k, :]) for k, proj in enumerate(self.projections)]
        return self.dropout(torch.stack(outputs, dim=1))


class TextEmbeddingSeq(nn.Module):
    """Token-aware encoder: tokens -> K heads -> temporal -> [B, K, output_dim]."""

    def __init__(
        self,
        text_dim,
        output_dim,
        num_text_variates=1,
        token_num_heads=4,
        token_num_layers=1,
        article_num_heads=4,
        article_num_layers=1,
        temporal_num_heads=4,
        temporal_num_layers=2,
        temporal_num_blocks=2,
        temporal_per_head=True,
        dropout=0.1,
    ):
        super(TextEmbeddingSeq, self).__init__()
        self.token_attn = TokenSelfAttention(text_dim, token_num_heads, token_num_layers, dropout)
        self.token_agg = TokenAggregator(text_dim, num_text_variates)
        self.article_pool = ArticlePooler(text_dim, article_num_heads, article_num_layers, dropout)
        # Stack of temporal transformers
        self.temporal_blocks = nn.ModuleList([
            TemporalTransformerSeq(
                text_dim,
                temporal_num_heads,
                temporal_num_layers,
                dropout,
                per_head=temporal_per_head,
                num_variates=num_text_variates,
            )
            for _ in range(temporal_num_blocks)
        ])
        self.temporal_agg = TemporalAggregator(text_dim, num_text_variates)
        self.projector = VariateProjectorSeq(text_dim, output_dim, num_text_variates, dropout)

    def forward(self, news_emb, news_mask=None, text_mask=None):
        """
        news_emb: [B, L, N, seq_len, D]
        news_mask: [B, L, N] or None
        text_mask: [B, L, N, seq_len] or None
        """
        if len(news_emb.shape) != 5:
            raise ValueError(f"TextEmbeddingSeq expects 5D input; got {news_emb.shape}")

        token_out = self.token_attn(news_emb, text_mask)  # [B,L,N,seq_len,D]
        token_heads = self.token_agg(token_out, text_mask)  # [B,L,N,K,D]
        pooled = self.article_pool(token_heads, news_mask)  # [B,L,K,D]

        time_mask = None
        if news_mask is not None:
            time_mask = (news_mask.sum(dim=2) > 0).float()

        # Apply stack of temporal transformers
        temporal_out = pooled  # [B,L,K,D]
        for temporal_block in self.temporal_blocks:
            temporal_out = temporal_block(temporal_out, time_mask)  # [B,L,K,D]
        
        agg = self.temporal_agg(temporal_out, time_mask)  # [B,K,D]
        return self.projector(agg)  # [B,K,output_dim]

