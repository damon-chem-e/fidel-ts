"""
Embedder module for centralized text and LLM embedding functionality.

This module provides:

Text Embeddings (BERT-style):
- EmbeddingModelRegistry: Shared model/tokenizer registry
- TextEmbedder: Main embedding interface
- Aggregation methods: CLS token, average pooling, no pooling
- Metadata management: Track embedding configurations
- Cache management: Hash-based cache directories with metadata

LLM Embeddings (GPT-style):
- LLMRegistry: Singleton registry for LLM models (GPT-2, Qwen, LLaMA)
- LLMEmbedder: End-to-end embedding generator
- PromptBuilder: Time series to prompt conversion
- LLMEmbeddingCache: Cache manager for LLM embeddings
- LLM utilities: Memory estimation, quantization support
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

# LLM Embedding components
from .llm_registry import LLMRegistry
from .llm_cache import LLMEmbeddingCache, LLMEmbeddingMetadata
from .prompt_builder import TSPromptBuilder, PromptTemplate, TimeCMATemplate
from .llm_embedder import LLMEmbedder
from .llm_utils import (
    estimate_model_memory,
    get_quantization_config,
    load_model_for_embedding,
    get_optimal_batch_size,
    get_gpu_memory_info,
    MODEL_SPECS,
)

__all__ = [
    # Text Embeddings
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
    # LLM Embeddings
    'LLMRegistry',
    'LLMEmbeddingCache',
    'LLMEmbeddingMetadata',
    'TSPromptBuilder',
    'PromptTemplate',
    'TimeCMATemplate',
    'LLMEmbedder',
    'estimate_model_memory',
    'get_quantization_config',
    'load_model_for_embedding',
    'get_optimal_batch_size',
    'get_gpu_memory_info',
    'MODEL_SPECS',
]

