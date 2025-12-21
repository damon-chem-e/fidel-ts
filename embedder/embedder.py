"""
Main text embedding functionality.

Unified interface for embedding text with support for:
- Multiple aggregation methods (CLS, average, none)
- Caching with metadata
- Batch processing
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path

from .registry import EmbeddingModelRegistry
from .aggregation import get_aggregation_function
from .metadata import EmbeddingMetadata
from .cache_manager import EmbeddingCacheManager


class TextEmbedder:
    """
    Unified text embedder with caching and multiple aggregation methods.
    """
    
    def __init__(self,
                 model_name: str = 'bert-base-uncased',
                 aggregation_method: str = 'cls',
                 device: str = 'cpu',
                 hf_cache_dir: str = './HF_cache/',
                 max_length: int = 512,
                 batch_size: int = 32,
                 cache_root: Optional[str] = None,
                 cache_path: Optional[str] = None,
                 force_reembed: bool = False):
        """
        Initialize text embedder.
        
        Args:
            model_name: HuggingFace model name
            aggregation_method: 'cls', 'average', or 'none'
            device: Device for model ('cpu', 'cuda:0', etc.)
            hf_cache_dir: HuggingFace cache directory
            max_length: Maximum tokenization length
            batch_size: Batch size for embedding computation
            cache_root: Root directory for embedding cache (if None, caching disabled)
            cache_path: Dataset-specific path within cache_root
            force_reembed: If True, recompute embeddings even if cache exists
        """
        self.model_name = model_name
        self.aggregation_method = aggregation_method
        self.device = device
        self.hf_cache_dir = hf_cache_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.force_reembed = force_reembed
        
        # Get model and tokenizer from registry
        self.model = EmbeddingModelRegistry.get_model(model_name, device, hf_cache_dir)
        self.tokenizer = EmbeddingModelRegistry.get_tokenizer(model_name, hf_cache_dir)
        
        # Get aggregation function
        self.aggregate_fn = get_aggregation_function(aggregation_method)
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        
        # Cache manager
        if cache_root is not None:
            self.cache_manager = EmbeddingCacheManager(cache_root, cache_path or '')
        else:
            self.cache_manager = None
        
        # Determine sequence_length for aggregation='none'
        self.sequence_length = None
        if aggregation_method == 'none':
            self.sequence_length = max_length  # All sequences padded to max_length
    
    def create_metadata(self, **extra_config) -> EmbeddingMetadata:
        """
        Create metadata object for current configuration.
        
        Args:
            **extra_config: Additional configuration parameters to include in metadata
        
        Returns:
            EmbeddingMetadata object
        """
        return EmbeddingMetadata(
            tokenizer_name=self.model_name,
            model_name=self.model_name,
            aggregation_method=self.aggregation_method,
            embedding_dim=self.embedding_dim,
            sequence_length=self.sequence_length,
            max_length=self.max_length,
            device=self.device,
            hf_cache_dir=self.hf_cache_dir,
            **extra_config
        )
    
    def embed_texts(self, 
                   texts: List[str],
                   text_keys: Optional[List[str]] = None,
                   return_metadata: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, EmbeddingMetadata]]:
        """
        Embed list of texts.
        
        Args:
            texts: List of text strings to embed
            text_keys: Optional list of keys for caching (e.g., timestamps). 
                      If None, uses indices as keys
            return_metadata: If True, also return metadata
        
        Returns:
            embeddings: numpy array of embeddings
                - If aggregation='cls' or 'average': [N, embedding_dim]
                - If aggregation='none': [N, sequence_length, embedding_dim]
            metadata: EmbeddingMetadata (if return_metadata=True)
        
        Note:
            For aggregation='none', attention masks are not currently returned.
            This may need to be extended based on use cases.
        """
        if text_keys is None:
            text_keys = [str(i) for i in range(len(texts))]
        
        # Check cache
        if self.cache_manager and not self.force_reembed:
            metadata = self.create_metadata()
            cache_dir = self.cache_manager.find_existing_cache(metadata)
            if cache_dir is not None:
                try:
                    # Load from cache
                    cached_embeddings = self.cache_manager.load_embeddings(cache_dir)
                    
                    # Try to match texts to cached embeddings by keys
                    # This assumes cached_embeddings is a dict mapping keys to embeddings
                    if isinstance(cached_embeddings, dict):
                        # Check if all keys are present
                        if all(key in cached_embeddings for key in text_keys):
                            # Convert to array in correct order
                            emb_list = [cached_embeddings[key] for key in text_keys]
                            result = np.stack(emb_list, axis=0)
                            
                            if return_metadata:
                                return result, metadata
                            return result
                except Exception as e:
                    print(f"[ warning ] Failed to load from cache: {e}. Recomputing embeddings.")
        
        # Compute embeddings
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state  # [B, seq_len, hidden_dim]
            
            # Aggregate
            if self.aggregation_method == 'none':
                embeddings, _ = self.aggregate_fn(last_hidden_state, attention_mask)
                all_embeddings.append(embeddings.cpu().numpy())
            else:
                embeddings = self.aggregate_fn(last_hidden_state, attention_mask)
                all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate
        result = np.concatenate(all_embeddings, axis=0)
        
        # Save to cache if enabled
        if self.cache_manager:
            metadata = self.create_metadata()
            cache_dir = self.cache_manager.find_existing_cache(metadata)
            if cache_dir is None:
                cache_dir = self.cache_manager.create_cache_dir(metadata)
            
            # Save embeddings as dict with keys
            embeddings_dict = {key: result[i] for i, key in enumerate(text_keys)}
            self.cache_manager.save_embeddings(embeddings_dict, cache_dir, format='pkl')
        
        if return_metadata:
            metadata = self.create_metadata()
            return result, metadata
        return result
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed single text.
        
        Args:
            text: Text string to embed
        
        Returns:
            embedding: numpy array of shape [embedding_dim] or [seq_len, embedding_dim]
                depending on aggregation method
        """
        result = self.embed_texts([text])
        if self.aggregation_method == 'none':
            return result[0]  # [seq_len, embedding_dim]
        else:
            return result[0]  # [embedding_dim]
    
    def embed_text_dict(self, 
                       text_dict: Dict[str, str],
                       return_metadata: bool = False) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], EmbeddingMetadata]]:
        """
        Embed dictionary of texts (key -> text mapping).
        
        Useful for TimeMMD datasets where embeddings are keyed by timestamps.
        
        Args:
            text_dict: Dictionary mapping keys (e.g., timestamps) to text strings
            return_metadata: If True, also return metadata
        
        Returns:
            embeddings_dict: Dictionary mapping keys to embedding arrays
            metadata: EmbeddingMetadata (if return_metadata=True)
        """
        keys = list(text_dict.keys())
        texts = [text_dict[key] for key in keys]
        
        # Embed all texts
        if return_metadata:
            embeddings_array, metadata = self.embed_texts(texts, text_keys=keys, return_metadata=True)
            # Convert array to dict
            if embeddings_array.ndim == 2:
                # [N, embedding_dim] -> dict of [embedding_dim] arrays
                embeddings_dict = {key: embeddings_array[i] for i, key in enumerate(keys)}
            elif embeddings_array.ndim == 3:
                # [N, seq_len, embedding_dim] -> dict of [seq_len, embedding_dim] arrays
                embeddings_dict = {key: embeddings_array[i] for i, key in enumerate(keys)}
            else:
                raise ValueError(f"Unexpected embeddings_array shape: {embeddings_array.shape}")
            return embeddings_dict, metadata
        else:
            embeddings_array = self.embed_texts(texts, text_keys=keys, return_metadata=False)
            # Convert array to dict
            if embeddings_array.ndim == 2:
                # [N, embedding_dim] -> dict of [embedding_dim] arrays
                embeddings_dict = {key: embeddings_array[i] for i, key in enumerate(keys)}
            elif embeddings_array.ndim == 3:
                # [N, seq_len, embedding_dim] -> dict of [seq_len, embedding_dim] arrays
                embeddings_dict = {key: embeddings_array[i] for i, key in enumerate(keys)}
            else:
                raise ValueError(f"Unexpected embeddings_array shape: {embeddings_array.shape}")
            return embeddings_dict

