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

from .registry import EmbeddingModelRegistry
from .aggregation import get_aggregation_function
from .metadata import EmbeddingMetadata
from .cache_manager import EmbeddingCacheManager


class TextEmbedder:
    """
    Unified text embedder with caching and multiple aggregation methods.
    
    Uses lazy loading: model and tokenizer are only loaded when actually needed
    for computing embeddings. This enables CPU-only cache lookups without loading
    GPU-intensive models.
    """
    
    # Known embedding dimensions for common models (enables GPU-free cache lookup)
    # Add new models here as needed to avoid model loading overhead
    # Source: HuggingFace model cards / config.json files
    EMBEDDING_DIM_LOOKUP = {
        # BERT family
        'bert-base-uncased': 768,
        'bert-base-cased': 768,
        'bert-large-uncased': 1024,
        'bert-large-cased': 1024,
        # RoBERTa family
        'roberta-base': 768,
        'roberta-large': 1024,
        # DistilBERT (smaller, faster BERT)
        'distilbert-base-uncased': 768,
        'distilbert-base-cased': 768,
        # ALBERT (parameter-efficient BERT)
        'albert-base-v2': 768,
        'albert-large-v2': 1024,
        'albert-xlarge-v2': 2048,
        'albert-xxlarge-v2': 4096,
        # Sentence transformers (if used)
        'sentence-transformers/all-MiniLM-L6-v2': 384,
        'sentence-transformers/all-mpnet-base-v2': 768,
    }
    
    def __init__(self,
                 model_name: str = 'bert-base-uncased',
                 aggregation_method: str = 'cls',
                 device: str = 'cpu',
                 hf_cache_dir: str = './HF_cache/',
                 max_length: int = 512,
                 batch_size: int = 32,
                 cache_root: Optional[str] = None,
                 cache_path: Optional[str] = None,
                 force_reembed: bool = False,
                 console: Optional[Any] = None):
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
            console: Optional Rich Console instance for progress bar display
        
        Note:
            Model and tokenizer are lazy-loaded (only when needed for computation).
            This enables CPU-only cache lookups without loading GPU-intensive models.
        """
        self.model_name = model_name
        self.aggregation_method = aggregation_method
        self.device = device
        self.hf_cache_dir = hf_cache_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.force_reembed = force_reembed
        self.console = console  # Rich Console for progress bars
        
        # Lazy-loaded attributes (None until needed)
        self._model = None
        self._tokenizer = None
        
        # Get aggregation function (doesn't require model)
        self.aggregate_fn = get_aggregation_function(aggregation_method)
        
        # Get embedding dimension from lookup table (no model loading)
        self.embedding_dim = self._get_embedding_dim_without_model(model_name)
        
        # Flag to track if we need to load model to get dimension
        self._needs_model_for_dim = (self.embedding_dim is None)
        
        # Warn if model not in lookup table
        if self._needs_model_for_dim:
            import warnings
            warnings.warn(
                f"Model '{model_name}' not in EMBEDDING_DIM_LOOKUP. "
                f"Model will be loaded to determine embedding dimension. "
                f"Consider adding this model to EMBEDDING_DIM_LOOKUP for faster cache checks."
            )
        
        # Cache manager
        if cache_root is not None:
            self.cache_manager = EmbeddingCacheManager(cache_root, cache_path or '')
        else:
            self.cache_manager = None
        
        # Determine sequence_length for aggregation='none'
        self.sequence_length = None
        if aggregation_method == 'none':
            self.sequence_length = max_length  # All sequences padded to max_length
    
    @staticmethod
    def _get_embedding_dim_without_model(model_name: str) -> Optional[int]:
        """
        Get embedding dimension from lookup table (no model loading).
        
        Args:
            model_name: HuggingFace model name
        
        Returns:
            Embedding dimension if known, None otherwise
        """
        return TextEmbedder.EMBEDDING_DIM_LOOKUP.get(model_name)
    
    @property
    def model(self):
        """
        Lazy-load model only when needed.
        
        Returns:
            Loaded model instance
        """
        if self._model is None:
            # Load model from registry
            self._model = EmbeddingModelRegistry.get_model(
                self.model_name, self.device, self.hf_cache_dir
            )
            
            # If we didn't have embedding_dim from lookup, get it now
            if self._needs_model_for_dim:
                self.embedding_dim = self._model.config.hidden_size
                self._needs_model_for_dim = False
        
        return self._model
    
    @property
    def tokenizer(self):
        """
        Lazy-load tokenizer only when needed.
        
        Returns:
            Loaded tokenizer instance
        """
        if self._tokenizer is None:
            # Load tokenizer from registry
            self._tokenizer = EmbeddingModelRegistry.get_tokenizer(
                self.model_name, self.hf_cache_dir
            )
        
        return self._tokenizer
    
    def create_metadata(self, **extra_config) -> EmbeddingMetadata:
        """
        Create metadata object for current configuration.
        
        Args:
            **extra_config: Additional configuration parameters to include in metadata
        
        Returns:
            EmbeddingMetadata object
        
        Note:
            If embedding_dim is not known from lookup table, this will trigger
            model loading to get the dimension.
        """
        # If embedding_dim still not set (unknown model), load model to get it
        if self.embedding_dim is None:
            _ = self.model  # Trigger lazy load to get embedding_dim
        
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
    
    def _tokenize_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a batch of texts.
        
        Args:
            texts: List of text strings to tokenize
        
        Returns:
            tuple: (input_ids, attention_mask) tensors on self.device
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        return input_ids, attention_mask
    
    def _compute_embeddings_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
        """
        Compute embeddings for a batch of tokenized texts.
        
        Args:
            input_ids: Token IDs tensor [B, seq_len]
            attention_mask: Attention mask tensor [B, seq_len]
        
        Returns:
            embeddings: numpy array of aggregated embeddings
                - If aggregation='cls' or 'average': [B, embedding_dim]
                - If aggregation='none': [B, seq_len, embedding_dim]
        """
        # Get embeddings from model
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state  # [B, seq_len, hidden_dim]
        
        # Aggregate based on method
        if self.aggregation_method == 'none':
            embeddings, _ = self.aggregate_fn(last_hidden_state, attention_mask)
        else:
            embeddings = self.aggregate_fn(last_hidden_state, attention_mask)
        
        return embeddings.cpu().numpy()
    
    def _load_from_cache(self, text_keys: List[str], metadata: EmbeddingMetadata) -> Optional[np.ndarray]:
        """
        Attempt to load embeddings from cache.
        
        Args:
            text_keys: List of keys to look up in cache
            metadata: Metadata to match cache
        
        Returns:
            embeddings array if all keys found in cache, None otherwise
        """
        # Return None with reason tracking for diagnostics
        if not self.cache_manager:
            self._cache_miss_reason = "No cache_manager configured (cache_root not provided)"
            return None
        
        if self.force_reembed:
            self._cache_miss_reason = "force_reembed=True, skipping cache"
            return None
        
        cache_dir = self.cache_manager.find_existing_cache(metadata)
        if cache_dir is None:
            # Provide detailed reason for cache miss
            expected_hash = metadata.compute_hash()
            self._cache_miss_reason = (
                f"No matching cache found. Looking for embeddings_{expected_hash[:16]} in {self.cache_manager.cache_base}. "
                f"Config: model={metadata.model_name}, aggregation={metadata.aggregation_method}, "
                f"dim={metadata.embedding_dim}, max_len={metadata.max_length}"
            )
            return None
        
        try:
            # Load from cache
            cached_embeddings = self.cache_manager.load_embeddings(cache_dir)
            
            # Try to match texts to cached embeddings by keys
            if isinstance(cached_embeddings, dict):
                # Check if all keys are present
                if all(key in cached_embeddings for key in text_keys):
                    # Convert to array in correct order
                    emb_list = [cached_embeddings[key] for key in text_keys]
                    result = np.stack(emb_list, axis=0)
                    self._cache_miss_reason = None  # Clear any previous reason
                    return result
                else:
                    # Some keys missing
                    missing_keys = [k for k in text_keys if k not in cached_embeddings]
                    self._cache_miss_reason = (
                        f"Cache found at {cache_dir} but {len(missing_keys)}/{len(text_keys)} keys missing. "
                        f"First missing: {missing_keys[:3]}"
                    )
        except Exception as e:
            self._cache_miss_reason = f"Failed to load from cache {cache_dir}: {e}"
            print(f"[ warning ] {self._cache_miss_reason}. Recomputing embeddings.")
        
        return None
    
    def _check_gpu_available(self) -> bool:
        """
        Check if GPU is available for embedding computation.
        
        Returns:
            True if device is 'cpu' or CUDA is available for cuda devices, False otherwise
        """
        # CPU always available
        if self.device == 'cpu':
            return True
        
        # For CUDA devices, check availability
        if self.device.startswith('cuda') or self.device.startswith('gpu'):
            try:
                return torch.cuda.is_available()
            except Exception:
                return False
        
        # Unknown device type - assume available and let it fail naturally
        return True
    
    def _save_to_cache(self, embeddings: np.ndarray, text_keys: List[str], metadata: EmbeddingMetadata):
        """
        Save embeddings to cache.
        
        Args:
            embeddings: Embeddings array [N, ...]
            text_keys: List of keys corresponding to embeddings
            metadata: Metadata to save with cache
        """
        if not self.cache_manager:
            return
        
        cache_dir = self.cache_manager.find_existing_cache(metadata)
        if cache_dir is None:
            cache_dir = self.cache_manager.create_cache_dir(metadata)
        
        # Save embeddings as dict with keys
        embeddings_dict = {key: embeddings[i] for i, key in enumerate(text_keys)}
        self.cache_manager.save_embeddings(embeddings_dict, cache_dir, format='pkl')
    
    def embed_texts(self, 
                   texts: List[str],
                   text_keys: Optional[List[str]] = None,
                   return_metadata: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, EmbeddingMetadata]]:
        """
        Embed list of texts.
        
        Args:
            texts: List of text strings to embed (empty strings will be embedded as zero vectors)
            text_keys: Optional list of keys for caching (e.g., timestamps). 
                      If None, uses indices as keys
            return_metadata: If True, also return metadata
        
        Returns:
            embeddings: numpy array of embeddings
                - If aggregation='cls' or 'average': [N, embedding_dim]
                - If aggregation='none': [N, sequence_length, embedding_dim]
            metadata: EmbeddingMetadata (if return_metadata=True)
        
        Note:
            Empty strings are embedded normally (will produce non-zero embeddings).
            Missing embeddings (keys not in cache) will trigger recomputation.
        """
        if text_keys is None:
            text_keys = [str(i) for i in range(len(texts))]
        
        if len(texts) != len(text_keys):
            raise ValueError(f"Number of texts ({len(texts)}) must match number of keys ({len(text_keys)})")
        
        metadata = self.create_metadata()
        
        # Try to load from cache
        cached_result = self._load_from_cache(text_keys, metadata)
        if cached_result is not None:
            if return_metadata:
                return cached_result, metadata
            return cached_result
        
        # Cache miss - check if we can compute embeddings
        # If device is CUDA but no GPU available, fail with clear error
        if self.device != 'cpu' and not self._check_gpu_available():
            cache_reason = getattr(self, '_cache_miss_reason', 'Unknown reason')
            raise RuntimeError(
                f"Embeddings not found in cache and no CUDA GPU available to compute them.\n"
                f"Cache miss reason: {cache_reason}\n\n"
                f"Solutions:\n"
                f"  1. Generate embeddings on a GPU node first:\n"
                f"     python -m cli.embeddings generate <config>\n"
                f"  2. Or set device='cpu' to compute on CPU (slow but works)\n"
                f"  3. Or use pre-computed embeddings with use_old_pkl=True"
            )
        
        # Compute embeddings
        all_embeddings = []
        
        # Calculate number of batches
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        # Use Rich Progress if console is available, otherwise simple iteration
        if self.console is not None:
            from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
            
            progress_columns = (
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("[progress.completed]{task.completed}/{task.total} batches"),
                TimeElapsedColumn(),
            )
            
            with Progress(*progress_columns, console=self.console, transient=False) as progress:
                task = progress.add_task("Computing embeddings", total=num_batches)
                
                # Process in batches
                for i in range(0, len(texts), self.batch_size):
                    batch_texts = texts[i:i+self.batch_size]
                    
                    # Tokenize batch
                    input_ids, attention_mask = self._tokenize_batch(batch_texts)
                    
                    # Compute embeddings
                    batch_embeddings = self._compute_embeddings_batch(input_ids, attention_mask)
                    all_embeddings.append(batch_embeddings)
                    
                    progress.update(task, advance=1)
        else:
            # Fallback: simple iteration without progress bar
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i+self.batch_size]
                
                # Tokenize batch
                input_ids, attention_mask = self._tokenize_batch(batch_texts)
                
                # Compute embeddings
                batch_embeddings = self._compute_embeddings_batch(input_ids, attention_mask)
                all_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        result = np.concatenate(all_embeddings, axis=0)
        
        # Save to cache
        self._save_to_cache(result, text_keys, metadata)
        
        if return_metadata:
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
    
    def _format_embeddings_for_time_mmd(self, embeddings_array: np.ndarray, keys: List[str]) -> Dict[str, np.ndarray]:
        """
        Format embeddings array to TimeMMD expected format.
        
        TimeMMD expects: {timestamp_str: np.ndarray(shape=(1, bert_dim))} for CLS/average
        or {timestamp_str: np.ndarray(shape=(seq_len, bert_dim))} for 'none' aggregation.
        
        Args:
            embeddings_array: Embeddings array from embed_texts
                - [N, embedding_dim] for CLS/average
                - [N, seq_len, embedding_dim] for 'none'
            keys: List of keys corresponding to embeddings
        
        Returns:
            Dictionary mapping keys to formatted embedding arrays
        """
        embeddings_dict = {}
        
        if embeddings_array.ndim == 2:
            # [N, embedding_dim] -> dict of [embedding_dim] arrays
            # Reshape to (1, embedding_dim) for TimeMMD compatibility
            for i, key in enumerate(keys):
                embeddings_dict[key] = embeddings_array[i].reshape(1, -1).astype(np.float32)
        elif embeddings_array.ndim == 3:
            # [N, seq_len, embedding_dim] -> dict of [seq_len, embedding_dim] arrays
            for i, key in enumerate(keys):
                embeddings_dict[key] = embeddings_array[i].astype(np.float32)
        else:
            raise ValueError(f"Unexpected embeddings_array shape: {embeddings_array.shape}")
        
        return embeddings_dict
    
    def embed_text_dict(self, 
                       text_dict: Dict[str, str],
                       return_metadata: bool = False,
                       format_for_time_mmd: bool = True) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], EmbeddingMetadata]]:
        """
        Embed dictionary of texts (key -> text mapping).
        
        Useful for TimeMMD datasets where embeddings are keyed by timestamps.
        
        Args:
            text_dict: Dictionary mapping keys (e.g., timestamps) to text strings
            return_metadata: If True, also return metadata
            format_for_time_mmd: If True, format embeddings as (1, embedding_dim) for CLS/average
                                to match TimeMMD expected format. If False, returns raw shapes.
        
        Returns:
            embeddings_dict: Dictionary mapping keys to embedding arrays
                - If format_for_time_mmd=True and aggregation='cls'/'average': {key: [1, embedding_dim]}
                - If format_for_time_mmd=True and aggregation='none': {key: [seq_len, embedding_dim]}
                - If format_for_time_mmd=False: {key: [embedding_dim]} or {key: [seq_len, embedding_dim]}
            metadata: EmbeddingMetadata (if return_metadata=True)
        """
        keys = list(text_dict.keys())
        texts = [text_dict[key] for key in keys]
        
        # Embed all texts
        if return_metadata:
            embeddings_array, metadata = self.embed_texts(texts, text_keys=keys, return_metadata=True)
        else:
            embeddings_array = self.embed_texts(texts, text_keys=keys, return_metadata=False)
            metadata = None
        
        # Format embeddings
        if format_for_time_mmd:
            embeddings_dict = self._format_embeddings_for_time_mmd(embeddings_array, keys)
        else:
            # Return raw shapes (no reshaping)
            if embeddings_array.ndim == 2:
                embeddings_dict = {key: embeddings_array[i] for i, key in enumerate(keys)}
            elif embeddings_array.ndim == 3:
                embeddings_dict = {key: embeddings_array[i] for i, key in enumerate(keys)}
            else:
                raise ValueError(f"Unexpected embeddings_array shape: {embeddings_array.shape}")
        
        if return_metadata:
            return embeddings_dict, metadata
        return embeddings_dict

