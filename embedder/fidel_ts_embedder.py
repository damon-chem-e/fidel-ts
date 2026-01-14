"""
Fidel-TS embedding loader for handling embeddings in Fidel-TS datasets.

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

This module handles text embeddings for Fidel-TS datasets (Bear_room, Jena, NYC, etc.).
It supports two embedding sources:
  1. NEW CACHE SYSTEM: Hash-based cache directories with metadata validation
  2. OLD PKL FILES: Legacy .pkl files (explicit request only via use_old_embeddings=True)

Key Design Decisions:
---------------------
1. NO CPU FALLBACK FOR EMBEDDING COMPUTATION
   - BERT/transformer embeddings are GPU-intensive operations
   - Running on CPU is impractically slow (10-100x slower)
   - If GPU is unavailable and embeddings need computing → FAIL with clear error
   - This forces users to either: (a) use GPU, or (b) use pre-computed embeddings

2. GPU-FREE CACHE LOOKUP
   - Finding existing cache directories does NOT require loading the model
   - We use a lookup table for embedding dimensions (KNOWN_MODELS)
   - This enables tensor_cache generation on CPU-only nodes when embeddings exist

3. EXPLICIT SYSTEM SELECTION
   - No automatic fallback between old and new embedding systems
   - User must explicitly choose via use_old_embeddings parameter
   - Prevents silent data corruption from mismatched embedding formats

================================================================================
TYPICAL USAGE PATTERNS
================================================================================

Pattern 1: GPU node with embeddings not cached
  - _find_cache_dir() returns None (no model loading needed)
  - _compute_and_cache() is called
  - _init_embedder() loads BERT on GPU
  - Embeddings computed and saved to cache
  - REQUIRES GPU

Pattern 2: CPU node with embeddings already cached  
  - _find_cache_dir() returns cache path (no model loading needed)
  - _load_from_cache() loads pre-computed embeddings from disk
  - NO GPU REQUIRED

Pattern 3: CPU node with embeddings NOT cached
  - _find_cache_dir() returns None
  - _compute_and_cache() is called
  - _init_embedder() detects no GPU → RAISES ERROR
  - User must either use GPU or pre-compute embeddings elsewhere

================================================================================
"""

import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

from .fidel_ts_path_resolver import FidelTSPathResolver
from .embedder import TextEmbedder
from .cache_manager import EmbeddingCacheManager


def flatten_nested_text_data(data: Dict[str, Any], separator: str = ' ') -> Dict[str, str]:
    """
    Flatten nested JSON structures into flat timestamp -> text string mapping.
    
    Handles both flat and nested JSON structures by converting nested dicts to
    DataFrames and concatenating all text fields per timestamp. This preserves
    all information by joining text values from different locations/time periods.
    
    Args:
        data: Dictionary loaded from JSON file. Can be:
            - Flat: {timestamp: text_string}
            - 2-level nested: {timestamp: {location: text_string}}
            - 3-level nested: {timestamp: {location: {time_period: text_string}}}
            - Mixed structures
        separator: String separator to use when concatenating multiple text values.
                   Default is single space ' '.
    
    Returns:
        Flat dictionary mapping timestamps (as strings) to concatenated text strings.
        Each timestamp maps to a single string containing all text values joined.
    
    Example:
        Input (nested):
        {
            "20170101": {
                "brooklyn": {
                    "daily": "Weather text 1",
                    "Morning": "Weather text 2"
                },
                "queens": {
                    "daily": "Weather text 3"
                }
            }
        }
        
        Output (flat):
        {
            "20170101": "Weather text 1 Weather text 2 Weather text 3"
        }
    """
    if not data:
        return {}
    
    # Check if data is already flat (all values are strings)
    # If all values are strings, return as-is (but ensure keys are strings)
    if all(isinstance(v, str) for v in data.values()):
        return {str(k): str(v) for k, v in data.items()}
    
    # Nested structure detected - convert to DataFrame and concatenate
    try:
        # Convert nested dict to DataFrame with timestamps as index
        # This handles arbitrary nesting levels: nested keys become columns
        df = pd.DataFrame.from_dict(data, orient='index')
        
        # Concatenate all columns (text fields) into a single string per row
        # Convert all columns to strings, then join with separator
        df_text = df.astype(str).apply(separator.join, axis=1)
        
        # Convert back to dictionary: {timestamp_str: concatenated_text_string}
        result = df_text.to_dict()
        
        # Ensure all keys are strings
        return {str(k): str(v) for k, v in result.items()}
    
    except Exception as e:
        raise ValueError(
            f"Failed to flatten nested text data. "
            f"Expected dict structure with string or dict values. Error: {e}"
        )


class FidelTSEmbeddingLoader:
    """
    Handles embedding loading/computation for Fidel-TS datasets.
    
    Uses hardcoded, subdataset-specific path resolution via FidelTSPathResolver.
    Supports loading from old .pkl files (if explicitly requested) or new cache system.
    """
    
    def __init__(self,
                 dataset_name: str,
                 hetero_info: Dict[str, Any],
                 base_data_path: str,
                 embed_model_name: str = 'bert-base-uncased',
                 aggregation_method: str = 'cls',
                 device: str = 'cpu',
                 hf_cache_dir: str = './HF_cache/',
                 force_reembed: bool = False,
                 use_old_embeddings: bool = False,
                 console=None):
        """
        Initialize Fidel-TS embedding loader.
        
        Args:
            dataset_name: Name of Fidel-TS dataset (e.g., 'Bear_room')
            hetero_info: Config dict with root_path, formatter, static_path, etc.
            base_data_path: Base path to data directory (e.g., './data' or '/nfs/.../data')
            embed_model_name: HuggingFace model name for text embedding
            aggregation_method: Aggregation method ('cls', 'average', 'none')
            device: Device for embedding model
            hf_cache_dir: HuggingFace cache directory
            force_reembed: If True, recompute embeddings even if cache exists
            use_old_embeddings: If True, load from old .pkl files (explicit request)
        """
        self.dataset_name = dataset_name
        self.hetero_info = hetero_info
        self.base_data_path = base_data_path
        self.embed_model_name = embed_model_name
        self.aggregation_method = aggregation_method
        self.device = device
        self.hf_cache_dir = hf_cache_dir
        self.force_reembed = force_reembed
        self.use_old_embeddings = use_old_embeddings
        self.console = console  # Rich Console for progress bars
        
        # Initialize path resolver
        self.path_resolver = FidelTSPathResolver(dataset_name, hetero_info, base_data_path)
        self.paths = self.path_resolver.resolve_paths()
        
        # Initialize embedder (lazy - only if needed)
        self.embedder = None
        self.cache_manager = None
    
    def _get_embedding_dim_for_model(self, model_name: str) -> int:
        """
        Get embedding dimension for a model WITHOUT loading model weights.
        
        This is a critical optimization that enables GPU-free cache lookup.
        
        Why this exists:
        ----------------
        To find existing embedding caches, we need to compute a metadata hash.
        The hash includes the embedding dimension. Traditionally, getting the
        embedding dimension requires loading the model (which needs GPU for
        large models like BERT).
        
        Our solution: Maintain a lookup table of known embedding dimensions.
        For 99% of use cases (BERT, RoBERTa, etc.), we can get the dimension
        instantly without touching the model weights.
        
        Fallback behavior:
        ------------------
        For unknown models, we attempt to load just the config file (not weights).
        HuggingFace model configs are small JSON files that contain hidden_size.
        This is still much faster than loading full model weights.
        
        Args:
            model_name: HuggingFace model name (e.g., 'bert-base-uncased')
        
        Returns:
            Embedding dimension (hidden_size from model config)
        
        Raises:
            ValueError: If dimension cannot be determined for unknown model
        """
        # =========================================================================
        # KNOWN MODELS LOOKUP TABLE
        # =========================================================================
        # This table enables GPU-free cache lookup for common embedding models.
        # Add new models here as needed to avoid config loading overhead.
        #
        # Source: HuggingFace model cards / config.json files
        # =========================================================================
        KNOWN_MODELS = {
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
        
        if model_name in KNOWN_MODELS:
            return KNOWN_MODELS[model_name]
        
        # Fallback: Load only the config file (not model weights)
        # This is still fast because config.json is small (~1KB)
        try:
            from transformers import AutoConfig
            import os
            os.makedirs(self.hf_cache_dir, exist_ok=True)
            config = AutoConfig.from_pretrained(model_name, cache_dir=self.hf_cache_dir)
            return config.hidden_size
        except Exception as e:
            raise ValueError(
                f"Could not determine embedding dimension for model '{model_name}'. "
                f"Add it to KNOWN_MODELS in _get_embedding_dim_for_model() or ensure "
                f"the model config is accessible. Error: {e}"
            )
    
    def _create_metadata_without_model(self) -> 'EmbeddingMetadata':
        """
        Create embedding metadata WITHOUT loading the model.
        
        This is the key method that enables GPU-free cache lookup. It creates
        metadata that can be used to compute cache hashes and find existing
        cache directories, all without loading BERT weights.
        
        Why this matters:
        -----------------
        Traditional flow:
          Load BERT (needs GPU) → Get config → Create metadata → Compute hash
        
        Our optimized flow:
          Lookup embedding_dim → Create metadata → Compute hash (no GPU!)
        
        This enables the following workflow:
        1. Generate embeddings on GPU cluster (saves to cache)
        2. Generate tensor_cache on CPU-only node (loads from cache)
        
        The second step doesn't need GPU because:
        - Finding cache: Uses this method (no model loading)
        - Loading embeddings: Just reads .pkl files from disk
        - Tensor cache gen: Pure numpy/CPU operations
        
        Implementation notes:
        --------------------
        - embedding_dim comes from lookup table (see _get_embedding_dim_for_model)
        - sequence_length is only relevant for aggregation='none'
        - max_length is hardcoded to 512 (standard BERT max)
        
        Returns:
            EmbeddingMetadata object with all fields needed for hash computation
        """
        # Get embedding dimension from lookup table (no model loading!)
        embedding_dim = self._get_embedding_dim_for_model(self.embed_model_name)
        
        # sequence_length only matters for aggregation='none' (full token embeddings)
        # For 'cls' and 'average', output is always [embedding_dim] regardless of input length
        sequence_length = None
        if self.aggregation_method == 'none':
            sequence_length = 512  # Matches max_length for padding consistency
        
        # Create metadata directly without going through TextEmbedder
        from .metadata import EmbeddingMetadata
        return EmbeddingMetadata(
            tokenizer_name=self.embed_model_name,
            model_name=self.embed_model_name,
            aggregation_method=self.aggregation_method,
            embedding_dim=embedding_dim,
            sequence_length=sequence_length,
            max_length=512,
            device=self.device,
            hf_cache_dir=self.hf_cache_dir
        )
    
    def _init_embedder(self):
        """
        Initialize TextEmbedder for computing new embeddings.
        
        IMPORTANT: This method is ONLY called when embeddings need to be computed.
        If embeddings are already cached, this method is never invoked.
        
        GPU Requirement:
        ----------------
        Embedding computation with BERT/transformers is a GPU-intensive operation.
        Running on CPU is impractically slow (10-100x slower), making it unsuitable
        for production use. Therefore:
        
        - If GPU device requested but CUDA unavailable → RAISE ERROR (no CPU fallback)
        - If CPU device explicitly requested → Allow (user knows what they're doing)
        
        This design forces users to either:
        1. Run embedding computation on a GPU node
        2. Use pre-computed embeddings from cache (no GPU needed for loading)
        
        The rationale is that silently falling back to CPU would result in:
        - Hours of computation instead of minutes
        - Poor user experience with no clear indication of the problem
        - Potential cluster resource waste
        
        Raises:
            RuntimeError: If CUDA device requested but not available
        """
        if self.embedder is None:
            import torch
            
            # =========================================================================
            # GPU AVAILABILITY CHECK - NO CPU FALLBACK FOR EMBEDDING COMPUTATION
            # =========================================================================
            # Embedding computation is a GPU-intensive process. We explicitly DO NOT
            # fall back to CPU because:
            #   1. CPU embedding is 10-100x slower (impractical for real datasets)
            #   2. Silent fallback would surprise users with multi-hour waits
            #   3. Better to fail fast with clear guidance
            #
            # If embeddings are already cached, this code path is never reached.
            # Use --cpu-only flag in CLI only when embeddings are pre-computed.
            # =========================================================================
            
            if self.device.startswith('cuda') and not torch.cuda.is_available():
                raise RuntimeError(
                    f"\n"
                    f"╔══════════════════════════════════════════════════════════════════════╗\n"
                    f"║  GPU REQUIRED FOR EMBEDDING COMPUTATION                              ║\n"
                    f"╠══════════════════════════════════════════════════════════════════════╣\n"
                    f"║  Requested device: {self.device:<50} ║\n"
                    f"║  CUDA available: False                                               ║\n"
                    f"║                                                                      ║\n"
                    f"║  Embedding computation with BERT/transformers requires a GPU.        ║\n"
                    f"║  Running on CPU is 10-100x slower and not supported.                 ║\n"
                    f"║                                                                      ║\n"
                    f"║  SOLUTIONS:                                                          ║\n"
                    f"║  1. Run on a GPU node (recommended)                                  ║\n"
                    f"║  2. Pre-compute embeddings on GPU, then use --cpu-only flag          ║\n"
                    f"║  3. Check if embeddings already exist in cache                       ║\n"
                    f"║                                                                      ║\n"
                    f"║  Cache location: {str(self.paths.get('cache_base', 'N/A'))[:52]:<52} ║\n"
                    f"╚══════════════════════════════════════════════════════════════════════╝\n"
                )
            
            # Initialize the embedder with the requested device
            self.embedder = TextEmbedder(
                model_name=self.embed_model_name,
                aggregation_method=self.aggregation_method,
                device=self.device,
                hf_cache_dir=self.hf_cache_dir,
                max_length=512,
                batch_size=32,
                cache_root=None,  # Not using TextEmbedder's cache for Fidel-TS
                cache_path=None,
                force_reembed=self.force_reembed,
                console=self.console
            )
    
    def load_embeddings(self) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Load embeddings for Fidel-TS dataset.
        
        Behavior:
        - If use_old_embeddings=True: Load from old .pkl files ONLY
        - Else if cache exists and not force_reembed: Load from new cache
        - Else if text files available: Compute and save to cache
        - Else: Error (no source available)
        
        NO FALLBACK between old and new systems.
        
        Returns:
            tuple: (dynamic_embeddings, static_embeddings)
                - dynamic_embeddings: Dictionary mapping timestamps to embedding arrays
                - static_embeddings: Dictionary of static embeddings (or None)
        
        Raises:
            ValueError: If no embedding source is available
            FileNotFoundError: If explicitly requested files don't exist
        """
        if self.use_old_embeddings:
            # Explicitly requested old embeddings - load them
            return self._load_old_embeddings()
        
        # Try new cache system
        cache_dir = self._find_cache_dir()
        if cache_dir and not self.force_reembed:
            return self._load_from_cache(cache_dir)
        
        # Compute from text
        if self._has_text_source():
            return self._compute_and_cache()
        
        raise ValueError(
            f"No embedding source available for {self.dataset_name}. "
            f"Set use_old_embeddings=True to use old .pkl files, or provide text source for re-embedding."
        )
    
    def _load_old_embeddings(self) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Load embeddings from old .pkl files (explicitly requested).
        
        Validates that static embeddings contain required keys (channel_info).
        If validation fails, provides clear error with guidance.
        
        Returns:
            tuple: (dynamic_embeddings, static_embeddings)
        """
        # Load dynamic embeddings
        if 'old_embedding_path' in self.paths:
            # Single file (Bear_room, Jena_Atmospheric_Physics)
            old_path = self.paths['old_embedding_path']
            if not old_path.exists():
                raise FileNotFoundError(
                    f"Old embedding file not found: {old_path}. "
                    f"Set use_old_embeddings=False to use new cache system."
                )
            dynamic_embeddings = joblib.load(old_path)
        elif 'old_embedding_paths' in self.paths:
            # Multiple files (year-based datasets)
            old_paths = self.paths['old_embedding_paths']
            dynamic_embeddings = {}
            for old_path in old_paths:
                if not old_path.exists():
                    print(f"[ warning ] Old embedding file not found: {old_path}, skipping")
                    continue
                file_embeddings = joblib.load(old_path)
                dynamic_embeddings.update(file_embeddings)  # Merge dictionaries
        else:
            raise ValueError(f"No old embedding paths found in resolved paths")
        
        # Load static embeddings
        static_path = self.paths['old_static_path']
        if static_path.exists():
            static_embeddings = joblib.load(static_path)
        else:
            print(f"[ warning ] Old static embeddings file not found: {static_path}")
            static_embeddings = None
        
        # Validate static embeddings have required keys
        # channel_info is required for Fidel-TS datasets with hetero_info
        if static_embeddings is not None and 'channel_info' not in static_embeddings:
            raise ValueError(
                f"Old static embeddings file is missing 'channel_info' key: {static_path}. "
                f"The old .pkl file may be corrupted or was created with a buggy version. "
                f"Set use_old_embeddings=False to use the new cache system which will "
                f"recompute embeddings from the text source (static_info.json)."
            )
        
        return dynamic_embeddings, static_embeddings
    
    def has_cached_embeddings(self) -> bool:
        """
        Check if embeddings are available in cache WITHOUT loading the model.
        
        This is a lightweight check that can be performed on CPU-only nodes
        to determine if embedding computation will be required. It enables
        the CLI to validate --cpu-only mode before attempting initialization.
        
        How it works:
        -------------
        1. Creates metadata using lookup table (no model loading)
        2. Searches for matching cache directory
        3. Returns True if valid cache found, False otherwise
        
        Use case:
        ---------
        Before initializing Data_Provider with --cpu-only flag, the CLI can
        call this method to verify embeddings are cached. If not cached,
        the CLI can fail early with a helpful message instead of failing
        deep in the embedding loading code.
        
        Returns:
            True if embeddings are cached and valid, False otherwise
        """
        # Also check for old embeddings if that's what's configured
        if self.use_old_embeddings:
            return self._has_old_embeddings()
        
        # Check new cache system
        cache_dir = self._find_cache_dir()
        return cache_dir is not None
    
    def _has_old_embeddings(self) -> bool:
        """
        Check if old .pkl embedding files exist.
        
        Returns:
            True if old embedding files exist
        """
        if 'old_embedding_path' in self.paths:
            return self.paths['old_embedding_path'].exists()
        elif 'old_embedding_paths' in self.paths:
            return any(p.exists() for p in self.paths['old_embedding_paths'])
        return False
    
    def _find_cache_dir(self) -> Optional[Path]:
        """
        Find existing cache directory matching current embedding configuration.
        
        IMPORTANT: This method does NOT load the embedding model.
        
        How it works:
        -------------
        1. Creates metadata using _create_metadata_without_model()
           - Uses lookup table for embedding dimensions (no model loading)
           - This is the key optimization that enables CPU-only cache lookup
        
        2. Searches cache_base for directories matching the metadata hash
           - Compares model name, aggregation method, embedding dim, etc.
           - Returns path if match found, None otherwise
        
        Why this matters:
        -----------------
        Traditional approach: Load BERT → Get embedding_dim → Compute hash → Find cache
        Our approach: Lookup embedding_dim → Compute hash → Find cache (no BERT loading!)
        
        This enables tensor cache generation on CPU-only nodes when embeddings
        are already computed, because we never need to load the GPU-hungry model
        just to find out if the cache exists.
        
        Returns:
            Path to cache directory if found, None otherwise
        """
        cache_base = self.paths['cache_base']
        
        # Create metadata without loading the model
        # Uses lookup table for known models (BERT, RoBERTa, etc.)
        metadata = self._create_metadata_without_model()
        
        # Use static method to find cache
        return EmbeddingCacheManager.find_fidel_ts_cache(cache_base, metadata)
    
    def _load_from_cache(self, cache_dir: Path) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Load embeddings from new cache directory.
        
        Validates that static embeddings contain required keys (channel_info).
        If validation fails and text source is available, triggers recomputation.
        
        Args:
            cache_dir: Path to cache directory
        
        Returns:
            tuple: (dynamic_embeddings, static_embeddings)
        
        Raises:
            ValueError: If cache is invalid and no text source available for recomputation
        """
        cached_data = EmbeddingCacheManager.load_fidel_ts_embeddings(cache_dir)
        dynamic_embeddings = cached_data['dynamic']
        static_embeddings = cached_data['static']
        
        # Validate static embeddings have required keys
        # channel_info is required for Fidel-TS datasets with hetero_info
        if static_embeddings is not None and 'channel_info' not in static_embeddings:
            print(f'[ warning ] Cached static embeddings missing "channel_info" key. Cache may be outdated.')
            print(f'[ info ] Attempting to recompute embeddings from text source...')
            
            # Try to recompute from text (force=True to overwrite invalid cache)
            if self._has_text_source():
                return self._compute_and_cache(force=True)
            else:
                raise ValueError(
                    f"Cached static embeddings are missing 'channel_info' and no text source is available "
                    f"for recomputation. Delete the cache directory and ensure text source files exist: {cache_dir}"
                )
        
        return dynamic_embeddings, static_embeddings
    
    def _has_text_source(self) -> bool:
        """Check if text source files are available for re-embedding."""
        if 'old_text_path' in self.paths:
            return self.paths['old_text_path'].exists()
        elif 'old_text_paths' in self.paths:
            return any(p.exists() for p in self.paths['old_text_paths'])
        return False
    
    def _compute_and_cache(self, force: bool = False) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Compute embeddings from text files and save to cache.
        
        Args:
            force: If True, overwrite existing cache directory
        
        Returns:
            tuple: (dynamic_embeddings, static_embeddings)
        """
        print(f'[ info ] Computing embeddings for {self.dataset_name} from text files...')
        
        self._init_embedder()
        
        # Load text data
        text_data = self._load_text_data()
        
        # Compute dynamic embeddings
        dynamic_embeddings = self._compute_dynamic_embeddings(text_data)
        
        # Compute static embeddings (if static text available)
        static_embeddings = self._compute_static_embeddings()
        
        # Save to cache (use force if explicitly requested or if self.force_reembed is set)
        self._save_to_cache(dynamic_embeddings, static_embeddings, force=force or self.force_reembed)
        
        return dynamic_embeddings, static_embeddings
    
    def _load_text_data(self) -> Dict[str, str]:
        """
        Load text data from JSON files and flatten nested structures.
        
        Handles both flat and nested JSON structures. For nested structures
        (e.g., {timestamp: {location: {time_period: text}}}), flattens by
        converting to DataFrame and concatenating all text values per timestamp.
        This preserves all information while producing a flat dict for embedding.
        
        Returns:
            Dictionary mapping timestamps (as strings) to text strings.
            For nested structures, text values are concatenated with spaces.
        """
        import json
        
        text_data = {}
        
        if 'old_text_path' in self.paths:
            # Single file
            text_path = self.paths['old_text_path']
            if text_path.exists():
                with open(text_path, 'r') as f:
                    data = json.load(f)
                    # Flatten nested structure if needed
                    flattened = flatten_nested_text_data(data)
                    text_data.update(flattened)
        elif 'old_text_paths' in self.paths:
            # Multiple files - load and merge them
            for text_path in self.paths['old_text_paths']:
                if text_path.exists():
                    with open(text_path, 'r') as f:
                        data = json.load(f)
                        # Flatten nested structure if needed
                        flattened = flatten_nested_text_data(data)
                        text_data.update(flattened)
        
        return text_data
    
    def _compute_dynamic_embeddings(self, text_data: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Compute dynamic embeddings from text data.
        
        Args:
            text_data: Dictionary mapping timestamps to text strings
        
        Returns:
            Dictionary mapping timestamps to embedding arrays
        """
        if not text_data:
            raise ValueError("No text data available for embedding computation")
        
        # Use TextEmbedder to embed all texts
        embeddings_dict = self.embedder.embed_text_dict(
            text_data,
            format_for_time_mmd=False  # Don't reshape - keep raw shapes
        )
        
        return embeddings_dict
    
    def _load_static_text(self) -> Optional[Dict[str, Any]]:
        """
        Load static text data from JSON file.
        
        Loads the original static text JSON file containing general_info,
        channel_info, and downtime_prompt as text strings.
        
        Returns:
            Dictionary with keys: 'general_info', 'channel_info', 'downtime_prompt'
            or None if file doesn't exist
        """
        import json
        
        static_text_path = self.paths.get('static_text_path')
        if not static_text_path:
            print(f'[ warning ] No static_text_path found in resolved paths')
            print(f'[ debug ] Available paths: {list(self.paths.keys())}')
            return None
        
        print(f'[ debug ] Checking for static text at: {static_text_path}')
        if not static_text_path.exists():
            print(f'[ warning ] Static text JSON file not found: {static_text_path}')
            return None
        
        try:
            with open(static_text_path, 'r') as f:
                static_text = json.load(f)
            print(f'[ info ] Loaded static text from: {static_text_path}')
            return static_text
        except Exception as e:
            print(f'[ warning ] Failed to load static text from {static_text_path}: {e}')
            return None
    
    def _compute_static_embeddings(self) -> Optional[Dict[str, Any]]:
        """
        Compute static embeddings from static text JSON file.
        
        Loads static text (general_info, channel_info, downtime_prompt) and
        embeds each using TextEmbedder with current aggregation method.
        This ensures embeddings match the current aggregation configuration.
        
        Returns:
            Dictionary with keys:
                - 'general_info': np.ndarray (shape depends on aggregation)
                - 'channel_info': Dict[str, np.ndarray] mapping channel IDs to embeddings
                - 'downtime_prompt': np.ndarray
            or None if static text not available
        """
        # Load static text data
        static_text = self._load_static_text()
        if not static_text:
            return None
        
        # Initialize embedder if not already initialized
        self._init_embedder()
        
        # Collect all static texts and their keys for batch embedding
        static_texts_list = []
        text_keys_list = []
        keys_to_type = {}  # Map to track which key belongs to which type ('general_info', 'downtime_prompt', or 'channel_info')
        
        # Collect general_info
        if 'general_info' in static_text:
            general_text = static_text['general_info']
            if isinstance(general_text, str) and general_text.strip():
                static_texts_list.append(general_text)
                text_keys_list.append('general_info')
                keys_to_type['general_info'] = 'general_info'
        
        # Collect downtime_prompt
        if 'downtime_prompt' in static_text:
            downtime_text = static_text['downtime_prompt']
            if isinstance(downtime_text, str) and downtime_text.strip():
                static_texts_list.append(downtime_text)
                text_keys_list.append('downtime_prompt')
                keys_to_type['downtime_prompt'] = 'downtime_prompt'
        
        # Collect channel_info texts
        # Handles both flat (channel_id -> string) and nested (channel_id -> {measurement_type: string}) structures
        if 'channel_info' in static_text:
            channel_info_text = static_text['channel_info']
            if isinstance(channel_info_text, dict):
                for channel_id, channel_text in channel_info_text.items():
                    # Handle nested dict structure (e.g., Bear_room has {measurement_type: description})
                    # Flatten by concatenating all values with separator
                    if isinstance(channel_text, dict):
                        # Concatenate all measurement descriptions into a single string
                        flattened_text = ' '.join(str(v) for v in channel_text.values() if v)
                        if flattened_text.strip():
                            static_texts_list.append(flattened_text)
                            key = f'channel_info_{channel_id}'
                            text_keys_list.append(key)
                            keys_to_type[key] = 'channel_info'
                    elif isinstance(channel_text, str) and channel_text.strip():
                        # Simple string case (e.g., NYC)
                        static_texts_list.append(channel_text)
                        key = f'channel_info_{channel_id}'
                        text_keys_list.append(key)
                        keys_to_type[key] = 'channel_info'
            else:
                print(f'[ warning ] channel_info is not a dictionary, skipping')
        
        if not static_texts_list:
            print(f'[ warning ] No valid static text fields found, returning None')
            return None
        
        # Embed all static texts in a single batch (single progress bar)
        embeddings_array = self.embedder.embed_texts(static_texts_list, text_keys=text_keys_list)
        
        # Map embeddings back to the correct structure
        static_embeddings = {}
        
        for i, key in enumerate(text_keys_list):
            emb = embeddings_array[i]
            text_type = keys_to_type[key]
            
            if text_type == 'general_info':
                static_embeddings['general_info'] = emb
            elif text_type == 'downtime_prompt':
                static_embeddings['downtime_prompt'] = emb
            elif text_type == 'channel_info':
                # Extract channel_id from key
                channel_id = key.replace('channel_info_', '')
                if 'channel_info' not in static_embeddings:
                    static_embeddings['channel_info'] = {}
                static_embeddings['channel_info'][channel_id] = emb
        
        print(f'[ info ] Computed static embeddings with aggregation method: {self.aggregation_method}')
        return static_embeddings
    
    def _save_to_cache(self, dynamic_embeddings: Dict[str, np.ndarray], 
                       static_embeddings: Optional[Dict[str, np.ndarray]],
                       force: bool = False):
        """
        Save embeddings to cache directory.
        
        Args:
            dynamic_embeddings: Dictionary mapping timestamps to embedding arrays
            static_embeddings: Dictionary of static embeddings (or None)
            force: If True, overwrite existing cache directory
        """
        cache_base = self.paths['cache_base']
        
        # Get metadata
        metadata = self.embedder.create_metadata()
        
        # Create cache directory using static method
        cache_dir = EmbeddingCacheManager.create_fidel_ts_cache_dir(cache_base, metadata, force=force)
        
        # Save embeddings using static method
        EmbeddingCacheManager.save_fidel_ts_embeddings(
            dynamic_embeddings,
            cache_dir,
            static_embeddings=static_embeddings
        )
        
        print(f'[ info ] Saved embeddings to cache: {cache_dir}')

