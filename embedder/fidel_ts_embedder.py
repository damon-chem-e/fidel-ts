"""
Fidel-TS embedding loader for handling embeddings in Fidel-TS datasets.

Loads/computes embeddings for Fidel-TS datasets using the new hash-based cache system.
Supports loading from old .pkl files (explicitly requested) or new cache system.
NO fallback between systems - explicit requests only.
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
    
    def _init_embedder(self):
        """Initialize TextEmbedder if not already initialized."""
        if self.embedder is None:
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
    
    def _find_cache_dir(self) -> Optional[Path]:
        """Find existing cache directory matching current metadata."""
        cache_base = self.paths['cache_base']
        
        # Initialize embedder to get metadata
        self._init_embedder()
        metadata = self.embedder.create_metadata()
        
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
            
            # Try to recompute from text
            if self._has_text_source():
                return self._compute_and_cache()
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
    
    def _compute_and_cache(self) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Compute embeddings from text files and save to cache.
        
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
        
        # Save to cache
        self._save_to_cache(dynamic_embeddings, static_embeddings)
        
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
                       static_embeddings: Optional[Dict[str, np.ndarray]]):
        """Save embeddings to cache directory."""
        cache_base = self.paths['cache_base']
        
        # Get metadata
        metadata = self.embedder.create_metadata()
        
        # Create cache directory using static method
        cache_dir = EmbeddingCacheManager.create_fidel_ts_cache_dir(cache_base, metadata, force=False)
        
        # Save embeddings using static method
        EmbeddingCacheManager.save_fidel_ts_embeddings(
            dynamic_embeddings,
            cache_dir,
            static_embeddings=static_embeddings
        )
        
        print(f'[ info ] Saved embeddings to cache: {cache_dir}')

