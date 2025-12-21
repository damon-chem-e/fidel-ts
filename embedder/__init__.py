"""
Embedder module for centralized text embedding functionality.

This module provides:
- EmbeddingModelRegistry: Shared model/tokenizer registry
- TextEmbedder: Main embedding interface
- Aggregation methods: CLS token, average pooling, no pooling
- Metadata management: Track embedding configurations
- Cache management: Hash-based cache directories with metadata
"""

from .registry import EmbeddingModelRegistry
from .embedder import TextEmbedder
from .aggregation import (
    aggregate_cls_token,
    aggregate_average_pooling,
    aggregate_none,
    get_aggregation_function,
    AGGREGATION_METHODS
)
from .metadata import EmbeddingMetadata
from .cache_manager import EmbeddingCacheManager
from .fidel_ts_path_resolver import FidelTSPathResolver
from .fidel_ts_embedder import FidelTSEmbeddingLoader

__all__ = [
    'EmbeddingModelRegistry',
    'TextEmbedder',
    'aggregate_cls_token',
    'aggregate_average_pooling',
    'aggregate_none',
    'get_aggregation_function',
    'AGGREGATION_METHODS',
    'EmbeddingMetadata',
    'EmbeddingCacheManager',
    'FidelTSPathResolver',
    'FidelTSEmbeddingLoader',
]

