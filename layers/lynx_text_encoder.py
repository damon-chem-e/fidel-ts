"""
Lynx text encoder implementations for FiLM-based models.
"""

from typing import Optional

import numpy as np
import torch
from torch import nn
from einops import rearrange


def _positional_encoding(q_len: int, d_model: int) -> nn.Parameter:
    """
    Create learnable positional encodings.
    
    Args:
        q_len: Sequence length for positional encoding.
        d_model: Embedding dimension.
        
    Returns:
        Learnable positional encoding parameter with shape [q_len, 1, d_model].
    """
    # Initialize positional encoding with small uniform values
    weight = torch.empty((q_len, d_model))
    nn.init.uniform_(weight, -0.02, 0.02)
    # Add singleton dimension for broadcasting and return as parameter
    return nn.Parameter(weight.unsqueeze(1), requires_grad=True)


class LynxTextEncoder(nn.Module):
    """
    Text encoder with configurable cross, self, or MLP pathways.
    """

    def __init__(
        self,
        cross_layer: int,
        self_layer: int,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
        pred_len: int,
        stride: int,
        encoder_type: str = "cross",
        mlp_hidden_dim: Optional[int] = None,
        mlp_dropout: float = 0.0
    ):
        """
        Initialize the encoder with configurable pathway type.
        """
        super().__init__()
        # Store core configuration for later use
        self.pred_len = pred_len
        self.stride = stride
        self.num_heads = num_heads
        # Validate and store the encoder type
        self.encoder_type = self._validate_encoder_type(encoder_type)
        # Build the requested encoder modules
        self.cross_encoder = self._build_cross_encoder(cross_layer, embedding_dim, num_heads, dropout)
        self.self_encoder = self._build_self_encoder(self_layer, embedding_dim, num_heads, dropout)
        self.mlp_encoder = self._build_mlp_encoder(embedding_dim, mlp_hidden_dim, mlp_dropout)
        # Initialize positional encoding and dropout
        self.W_pos = _positional_encoding(int(np.ceil(pred_len / stride)), embedding_dim)
        self.dropout_layer = nn.Dropout(dropout)
        # Initialize cross-attention weights when present
        self._init_cross_weights()

    def _validate_encoder_type(self, encoder_type: str) -> str:
        """
        Validate the encoder type and return the normalized value.
        """
        # Ensure the encoder type is supported
        if encoder_type not in ("cross", "self", "mlp"):
            raise ValueError(f"Invalid encoder_type: {encoder_type!r}. Must be 'cross', 'self', or 'mlp'.")
        return encoder_type

    def _build_cross_encoder(self, cross_layer: int, embedding_dim: int, num_heads: int, dropout: float):
        """
        Build the cross-attention encoder if requested.
        """
        # Skip building if cross-attention is not selected
        if self.encoder_type != "cross" or cross_layer <= 0:
            return None
        # Create a TransformerDecoder for cross-attention
        cross_encoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embedding_dim * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        encoder_norm = nn.LayerNorm(embedding_dim, eps=1e-5)
        return nn.TransformerDecoder(cross_encoder_layer, cross_layer, norm=encoder_norm)

    def _build_self_encoder(self, self_layer: int, embedding_dim: int, num_heads: int, dropout: float):
        """
        Build the self-attention encoder if requested.
        """
        # Skip building if self-attention is not selected
        if self.encoder_type != "self":
            return None
        # Validate the layer count for self-attention
        if self_layer < 1:
            raise ValueError("self_layer must be >= 1 when encoder_type='self'.")
        # Create a TransformerEncoder for self-attention
        self_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embedding_dim * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self_norm_layer = nn.LayerNorm(embedding_dim, eps=1e-5)
        return nn.TransformerEncoder(self_encoder_layer, self_layer, norm=self_norm_layer)

    def _build_mlp_encoder(self, embedding_dim: int, mlp_hidden_dim: Optional[int], mlp_dropout: float):
        """
        Build the shallow MLP encoder if requested.
        """
        # Skip building if MLP is not selected
        if self.encoder_type != "mlp":
            return None
        # Default hidden dim to embedding dim when not provided
        mlp_hidden_dim = mlp_hidden_dim or embedding_dim
        # Build a small MLP for per-timestep text encoding
        return nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, embedding_dim)
        )

    def _init_cross_weights(self) -> None:
        """
        Initialize cross-attention weights when cross-attention is enabled.
        """
        # Skip if cross-attention is not active
        if self.cross_encoder is None:
            return
        # Apply Xavier initialization to weight matrices
        for param in self.cross_encoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def _pool_news_embeddings(self, news_emb: torch.Tensor, news_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool news embeddings with masking to avoid padded tokens.
        """
        # Convert mask to float weights (1 for valid tokens, 0 for padding)
        valid_mask = (~news_mask).float()
        # Sum embeddings over the news dimension
        summed = (news_emb * valid_mask.unsqueeze(-1)).sum(dim=2)
        # Count valid tokens per timestep (avoid divide-by-zero)
        counts = valid_mask.sum(dim=2).clamp(min=1.0)
        # Compute masked mean pooling
        return summed / counts.unsqueeze(-1)

    def _apply_positional_encoding(self, text_emb: torch.Tensor, B: int, L: int, C: int) -> torch.Tensor:
        """
        Apply positional encoding with variable-length support.
        """
        # Rearrange for positional encoding application
        x = rearrange(text_emb, 'b l c d -> (b c) l d', b=B, c=C).contiguous()
        # Prepare positional encoding for broadcasting
        W_pos_permuted = self.W_pos.permute(1, 0, 2)
        L_pos = W_pos_permuted.shape[1]
        # Match positional encoding length to input length
        if L == L_pos:
            x = x + W_pos_permuted
        elif L < L_pos:
            x = x + W_pos_permuted[:, :L, :]
        else:
            repeats = int(np.ceil(L / L_pos))
            W_pos_repeated = W_pos_permuted.repeat(1, repeats, 1)
            x = x + W_pos_repeated[:, :L, :]
        # Apply dropout after positional encoding
        x = self.dropout_layer(x)
        # Restore original shape
        return rearrange(x, '(b c) l d -> b l c d', b=B, c=C)

    def _encode_with_self(self, news_emb: torch.Tensor, description_emb: torch.Tensor) -> torch.Tensor:
        """
        Encode text using self-attention over news embeddings only.
        """
        # Extract dimensions for reshaping
        B, L, C, D = description_emb.shape
        # Reshape news embeddings for encoder
        news_emb = news_emb.contiguous().view(B * L, news_emb.shape[2], D)
        # Build padding mask for self-attention
        news_mask = (news_emb.sum(dim=-1) == 0).contiguous()
        # Apply self-attention over news tokens
        encoded = self.self_encoder(news_emb, src_key_padding_mask=news_mask)
        # Pool news tokens to a single vector per timestep
        pooled = self._pool_news_embeddings(encoded.view(B, L, -1, D), news_mask.view(B, L, -1))
        # Repeat pooled embeddings across channels
        text_emb = pooled.unsqueeze(2).repeat(1, 1, C, 1)
        # Apply positional encoding and return
        return self._apply_positional_encoding(text_emb, B, L, C)

    def _encode_with_mlp(self, news_emb: torch.Tensor, description_emb: torch.Tensor) -> torch.Tensor:
        """
        Encode text using a shallow MLP over pooled news embeddings.
        """
        # Extract dimensions for reshaping
        B, L, C, D = description_emb.shape
        # Reshape news embeddings for pooling
        news_emb = news_emb.contiguous().view(B * L, news_emb.shape[2], D)
        # Build padding mask for pooling
        news_mask = (news_emb.sum(dim=-1) == 0).contiguous()
        # Pool news tokens to a single vector per timestep
        pooled = self._pool_news_embeddings(news_emb.view(B, L, -1, D), news_mask.view(B, L, -1))
        # Apply MLP to pooled embeddings
        pooled = self.mlp_encoder(pooled)
        # Repeat pooled embeddings across channels
        text_emb = pooled.unsqueeze(2).repeat(1, 1, C, 1)
        # Apply positional encoding and return
        return self._apply_positional_encoding(text_emb, B, L, C)

    def _encode_with_cross(self, news_emb: torch.Tensor, description_emb: torch.Tensor) -> torch.Tensor:
        """
        Encode text using cross-attention from channel descriptions to news.
        """
        # Extract dimensions for reshaping
        B, L, C, D = description_emb.shape
        # Reshape the news embeddings for cross-attention
        news_emb = news_emb.contiguous().view(B * L, news_emb.shape[2], D)
        # Build padding mask for cross-attention
        news_mask = (news_emb.sum(dim=-1) == 0).contiguous()
        # Reshape the description embeddings for cross-attention
        text_emb = description_emb.contiguous().view(B * L, description_emb.shape[2], D)
        # Build the expanded mask for TransformerDecoder
        memory_mask = self._build_memory_mask(news_mask, text_emb.shape[0], text_emb.shape[1])
        # Apply cross-attention if enabled
        if self.cross_encoder is not None:
            text_emb = self.cross_encoder(tgt=text_emb, memory=news_emb, memory_mask=memory_mask)
        # Restore original shape
        text_emb = text_emb.view(B, L, C, D)
        # Apply positional encoding and return
        return self._apply_positional_encoding(text_emb, B, L, C)

    def _build_memory_mask(self, news_mask: torch.Tensor, batch_size: int, tgt_len: int) -> torch.Tensor:
        """
        Build a kernel-compatible memory mask for cross-attention.
        """
        # Determine the source length from the mask
        src_len = news_mask.shape[1]
        # Expand mask to [batch * heads, tgt_len, src_len]
        expanded = (
            news_mask
            .unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, self.num_heads, tgt_len, src_len)
            .reshape(batch_size * self.num_heads, tgt_len, src_len)
            .contiguous()
        )
        # Convert bool mask to additive float mask
        memory_mask = torch.zeros(expanded.shape, dtype=torch.float32, device=expanded.device)
        memory_mask.masked_fill_(expanded, float('-inf'))
        return memory_mask

    def forward(self, news_emb: torch.Tensor, description_emb: torch.Tensor) -> torch.Tensor:
        """
        Encode text embeddings according to the configured encoder type.
        """
        # Route to the appropriate encoder branch
        if self.encoder_type == "self":
            return self._encode_with_self(news_emb, description_emb)
        if self.encoder_type == "mlp":
            return self._encode_with_mlp(news_emb, description_emb)
        # Default to cross-attention when not self or MLP
        return self._encode_with_cross(news_emb, description_emb)
