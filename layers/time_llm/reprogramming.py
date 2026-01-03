"""
Reprogramming Layer - Maps time series patches to LLM vocabulary space.

This is the KEY innovation of Time-LLM. The reprogramming layer uses
cross-attention to translate time series patch embeddings into the
LLM's embedding space by attending to word embeddings.

Architecture:
    Query: Time series patch embeddings [B, num_patches, d_model]
    Key/Value: LLM word embeddings (mapped) [num_tokens, d_llm]
    Output: Reprogrammed embeddings in LLM space [B, num_patches, d_llm]

The cross-attention enables the frozen LLM to process time series by
expressing patches as weighted combinations of word embeddings.
"""

import math
import torch
import torch.nn as nn


class ReprogrammingLayer(nn.Module):
    """
    Cross-attention layer that maps time series patches to LLM vocabulary space.
    
    The reprogramming enables frozen LLMs to process time series by translating
    patch embeddings into the LLM's embedding space via cross-attention with
    word embeddings.
    
    Args:
        d_model: Dimension of time series patch embeddings
        n_heads: Number of attention heads
        d_keys: Dimension of keys/queries per head (defaults to d_model // n_heads)
        d_llm: Dimension of LLM embeddings (hidden size)
        attention_dropout: Dropout rate for attention weights
    
    Example:
        >>> reprogram = ReprogrammingLayer(d_model=32, n_heads=8, d_keys=32, d_llm=768)
        >>> patches = torch.randn(2, 12, 32)  # [B, num_patches, d_model]
        >>> word_embeds = torch.randn(1000, 768)  # [num_tokens, d_llm]
        >>> reprogrammed = reprogram(patches, word_embeds, word_embeds)
        >>> # reprogrammed: [2, 12, 768]
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_keys: int = None, 
        d_llm: int = None, 
        attention_dropout: float = 0.1
    ):
        """
        Initialize ReprogrammingLayer.
        
        Args:
            d_model: Dimension of time series patch embeddings
            n_heads: Number of attention heads
            d_keys: Dimension of keys per head (defaults to d_model // n_heads)
            d_llm: Dimension of LLM embeddings (hidden size)
            attention_dropout: Dropout rate for attention weights
        """
        super().__init__()
        
        # Default d_keys if not specified
        d_keys = d_keys or (d_model // n_heads)
        
        # Project time series patches to query space
        # Input: [B, L, d_model] -> Output: [B, L, d_keys * n_heads]
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        
        # Project LLM word embeddings to key/value space
        # Input: [S, d_llm] -> Output: [S, d_keys * n_heads]
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        
        # Output projection back to LLM space
        # Input: [B, L, d_keys * n_heads] -> Output: [B, L, d_llm]
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self, 
        target_embedding: torch.Tensor, 
        source_embedding: torch.Tensor, 
        value_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Reprogram time series patches using LLM word embeddings.
        
        Args:
            target_embedding: Time series patch embeddings [B, L, d_model]
            source_embedding: LLM word embeddings for keys [S, d_llm]
            value_embedding: LLM word embeddings for values [S, d_llm]
        
        Returns:
            Reprogrammed embeddings in LLM space [B, L, d_llm]
        """
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        # Project to multi-head attention space
        # target: [B, L, d_model] -> [B, L, H, d_keys]
        target = self.query_projection(target_embedding).view(B, L, H, -1)
        
        # source/value: [S, d_llm] -> [S, H, d_keys]
        source = self.key_projection(source_embedding).view(S, H, -1)
        value = self.value_projection(value_embedding).view(S, H, -1)

        # Compute reprogramming via cross-attention
        out = self._reprogramming(target, source, value)
        
        # Reshape and project output: [B, L, H, d_keys] -> [B, L, d_llm]
        out = out.reshape(B, L, -1)
        return self.out_projection(out)

    def _reprogramming(
        self, 
        target: torch.Tensor, 
        source: torch.Tensor, 
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Core reprogramming via scaled dot-product cross-attention.
        
        Computes attention between time series patch queries and
        word embedding keys, then applies attention to values.
        
        Args:
            target: Query tensor [B, L, H, E] where E = d_keys
            source: Key tensor [S, H, E]
            value: Value tensor [S, H, E]
        
        Returns:
            Attention output [B, L, H, E]
        """
        B, L, H, E = target.shape
        
        # Scaling factor for stable gradients
        scale = 1. / math.sqrt(E)
        
        # Compute attention scores: Q @ K^T
        # target: [B, L, H, E], source: [S, H, E]
        # einsum "blhe,she->bhls" computes [B, H, L, S] attention matrix
        scores = torch.einsum("blhe,she->bhls", target, source)
        
        # Apply softmax with scaling for attention weights
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        
        # Apply attention to values
        # A: [B, H, L, S], value: [S, H, E]
        # einsum "bhls,she->blhe" computes [B, L, H, E] output
        return torch.einsum("bhls,she->blhe", A, value)

