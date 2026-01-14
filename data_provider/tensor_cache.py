"""
Tensor Cache for Amortized Data Loading (V2 - Indexed/Deduplicated Format)

================================================================================
OVERVIEW
================================================================================

This module provides pre-computation of all CPU-intensive dataloader operations,
storing results as memory-mapped numpy arrays for ultra-fast training data loading.

V2 introduces INDEX-BASED DEDUPLICATION to dramatically reduce disk usage:
- Sliding window samples share 99%+ of their data (embeddings, time series)
- Instead of storing data per-sample, we store UNIQUE data once in shared tables
- Per-sample arrays store INDICES into the shared tables
- Result: ~100x reduction in disk usage (408 GB → ~4 GB for Bear_room)

================================================================================
CACHE STRUCTURE (V2 - INDEXED FORMAT)
================================================================================

tensor_cache/{hash}/
├── metadata.json              # Version, config, shapes, format info
├── shared/                    # Shared data tables (deduplicated)
│   ├── timeseries.npy         # (N_unique, n_features) - raw time series
│   ├── timestamps.npy         # (N_unique,) - timestamp values
│   ├── embeddings.npy         # (N_unique, embed_dim) - text embeddings
│   ├── hetero_time.npy        # (N_unique, n_time_features) - hetero time features
│   ├── entity_general.npy     # (N_entities, embed_dim) - static general
│   ├── entity_channel.npy     # (N_entities, embed_dim) - static channel
│   └── index_mappings.json    # {timestamp: idx}, {entity_id: idx}
├── train/
│   ├── progress.json          # Checkpointing state (deleted on completion)
│   ├── sample_ids.npy         # (N,) str
│   ├── entity_indices.npy     # (N,) int16
│   ├── x_indices.npy          # (N, input_len) int32
│   ├── y_indices.npy          # (N, output_len) int32
│   ├── x_time_features.npy    # (N, input_len, n_tf)
│   └── y_time_features.npy    # (N, output_len, n_tf)
├── val/
└── test/

================================================================================
CHECKPOINTING & RESUMABILITY
================================================================================

Generation can be interrupted and resumed:
- progress.json tracks completed entities and write position
- Arrays are flushed after each entity
- On restart, completed entities are skipped
- progress.json is deleted on successful completion

================================================================================
BACKWARD COMPATIBILITY
================================================================================

V1 (direct) caches are still supported for reading. The format is detected
from metadata.json. New caches are always created in V2 (indexed) format.

================================================================================
"""

import os
import json
import hashlib
import logging
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, TYPE_CHECKING
from datetime import datetime
from tqdm import tqdm
import warnings

# Rich progress bar imports (optional, graceful fallback to tqdm)
try:
    from rich.progress import (
        Progress, BarColumn, TextColumn, TimeElapsedColumn,
        TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn
    )
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

if TYPE_CHECKING:
    from data_provider.data_factory import Data_Provider

logger = logging.getLogger(__name__)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_config_hash(config: dict) -> str:
    """
    Compute hash of config parameters that affect cache validity.

    Cache must be regenerated if ANY of these change:
    - input_len / output_len
    - hetero_stride
    - normalize settings
    - split settings
    - truncate_train_for_purge
    - Entity filtering settings
    """
    relevant_keys = [
        'input_len', 'output_len', 'hetero_stride',
        'scale', 'split', 'truncate_train_for_purge',
        'downsample', 'hetero_type', 'data_name',
        'missing_value_strategy'
    ]

    relevant_config = {}
    for key in relevant_keys:
        if key in config:
            relevant_config[key] = config[key]

    config_str = json.dumps(relevant_config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def _save_progress_atomic(path: Path, data: dict):
    """
    Atomically save progress file using write-to-temp + rename.
    
    This ensures crash safety - if we die during write, the old file remains.
    """
    temp_path = path.with_suffix('.progress.tmp')
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)
    temp_path.rename(path)  # Atomic on POSIX


def _load_progress(path: Path) -> Optional[dict]:
    """Load progress file, returning None if corrupted or missing."""
    if not path.exists():
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Corrupted progress file {path}: {e}")
        return None


# =============================================================================
# METADATA
# =============================================================================

class TensorCacheMetadata:
    """
    Metadata for tensor cache validation and info.
    
    V2 adds:
    - cache_format: 'indexed' (new) or 'direct' (legacy)
    - shared_shapes: shapes of shared tables
    - index_mappings_info: info about timestamp/entity mappings
    """

    VERSION = "2.0.0"
    
    # Supported formats
    FORMAT_DIRECT = "direct"    # V1: stores data directly per sample
    FORMAT_INDEXED = "indexed"  # V2: stores indices into shared tables

    def __init__(
        self,
        config_hash: str,
        data_config: dict,
        shapes: dict,
        dtypes: dict,
        cache_format: str = FORMAT_INDEXED,
        shared_shapes: Optional[dict] = None,
        scaler_params: Optional[dict] = None,
        entity_info: Optional[dict] = None,
        created_at: Optional[str] = None,
        version: Optional[str] = None
    ):
        self.version = version or self.VERSION
        self.config_hash = config_hash
        self.data_config = data_config
        self.shapes = shapes
        self.dtypes = dtypes
        self.cache_format = cache_format
        self.shared_shapes = shared_shapes or {}
        self.scaler_params = scaler_params or {}
        self.entity_info = entity_info or {}
        self.created_at = created_at or datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            'version': self.version,
            'config_hash': self.config_hash,
            'created_at': self.created_at,
            'cache_format': self.cache_format,
            'data_config': self.data_config,
            'shapes': self.shapes,
            'shared_shapes': self.shared_shapes,
            'dtypes': self.dtypes,
            'scaler_params': self.scaler_params,
            'entity_info': self.entity_info
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TensorCacheMetadata':
        return cls(
            config_hash=data['config_hash'],
            data_config=data['data_config'],
            shapes=data['shapes'],
            dtypes=data['dtypes'],
            cache_format=data.get('cache_format', cls.FORMAT_DIRECT),  # Default to direct for old caches
            shared_shapes=data.get('shared_shapes', {}),
            scaler_params=data.get('scaler_params', {}),
            entity_info=data.get('entity_info', {}),
            created_at=data.get('created_at'),
            version=data.get('version', '1.0.0')
        )

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> 'TensorCacheMetadata':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @property
    def is_indexed(self) -> bool:
        """Check if this cache uses the indexed (deduplicated) format."""
        return self.cache_format == self.FORMAT_INDEXED


# =============================================================================
# V2 GENERATOR (INDEXED FORMAT WITH CHECKPOINTING)
# =============================================================================

class TensorCacheGenerator:
    """
    Generates pre-computed tensor cache for fast training data loading.
    
    V2 Features:
    - INDEX-BASED DEDUPLICATION: ~100x disk space reduction
    - ENTITY-LEVEL CHECKPOINTING: Resume interrupted generation
    - NESTED PROGRESS BARS: Better visibility into generation progress
    
    How deduplication works:
    1. First pass: Collect all unique timestamps and their data
    2. Build shared tables with deduplicated data
    3. Second pass: For each sample, store indices into shared tables
    
    The shared tables are stored in shared/ directory and referenced by all splits.
    This means train/val/test can share the same embedding data.
    """

    # Per-sample array specs (V2 indexed format)
    INDEXED_ARRAY_SPECS = {
        'sample_ids': {'dtype': 'U64'},      # (N,) - sample identifiers
        'entity_indices': {'dtype': 'int16'},  # (N,) - index into entity tables
        'x_indices': {'dtype': 'int32'},     # (N, input_len) - indices into shared tables
        'y_indices': {'dtype': 'int32'},     # (N, output_len) - indices into shared tables
        'x_time_features': {'dtype': 'float32'},  # (N, input_len, n_tf) - not deduplicated
        'y_time_features': {'dtype': 'float32'},  # (N, output_len, n_tf) - not deduplicated
    }
    
    # Shared table specs (stored once, referenced by indices)
    SHARED_TABLE_SPECS = {
        'timeseries': {'dtype': 'float32'},      # (N_unique, n_features)
        'timestamps': {'dtype': 'int64'},        # (N_unique,)
        'embeddings': {'dtype': 'float32'},      # (N_unique, embed_dim)
        'hetero_time': {'dtype': 'float32'},     # (N_unique, n_time_features)
        'entity_general': {'dtype': 'float32'},  # (N_entities, embed_dim)
        'entity_channel': {'dtype': 'float32'},  # (N_entities, embed_dim)
    }
    
    # Legacy V1 format specs (for backward compatibility)
    LEGACY_ARRAY_SPECS = {
        'sample_ids': {'index': 0, 'dtype': 'U64'},
        'seq_x': {'index': 1, 'dtype': 'float32'},
        'seq_y': {'index': 2, 'dtype': 'float32'},
        'x_time': {'index': 3, 'dtype': 'int64'},
        'y_time': {'index': 4, 'dtype': 'int64'},
        'hetero_x': {'index': 5, 'dtype': 'float32'},
        'hetero_y': {'index': 6, 'dtype': 'float32'},
        'hetero_x_time': {'index': 7, 'dtype': 'float32'},
        'hetero_y_time': {'index': 8, 'dtype': 'float32'},
        'hetero_general': {'index': 9, 'dtype': 'float32'},
        'hetero_channel': {'index': 10, 'dtype': 'float32'},
        'x_time_features': {'index': 11, 'dtype': 'float32'},
        'y_time_features': {'index': 12, 'dtype': 'float32'},
    }

    def __init__(
        self,
        data_provider: 'Data_Provider',
        cache_dir: Union[str, Path],
        config: dict,
        chunk_size: int = 10000,
        verbose: bool = True,
        console: Optional[Any] = None
    ):
        """
        Initialize tensor cache generator.

        Args:
            data_provider: Data_Provider instance with datasets configured
            cache_dir: Directory to store cache files
            config: Cache config dict from centralized build_cache_config() function
            chunk_size: Number of samples to process at once (memory management)
            verbose: Whether to show progress bars
            console: Optional Rich Console for enhanced progress display
        """
        self.data_provider = data_provider
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.console = console
        self.config_hash = compute_config_hash(config)

    def generate(self, flags: List[str] = None) -> Path:
        """
        Generate cache for specified data splits using indexed format.

        The generation process:
        1. Build shared tables (first pass over all data to collect unique values)
        2. Generate per-split index arrays (with checkpointing)
        3. Write metadata

        Args:
            flags: List of splits to generate ('train', 'val', 'test')

        Returns:
            Path to cache directory
        """
        if flags is None:
            flags = ['train', 'val', 'test']

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        shared_dir = self.cache_dir / 'shared'
        shared_dir.mkdir(exist_ok=True)

        # Step 1: Build shared tables (collect all unique data)
        logger.info("Building shared tables (collecting unique timestamps)...")
        shared_tables, index_mappings = self._build_shared_tables(flags)
        
        # Save shared tables
        shared_shapes = {}
        for name, data in shared_tables.items():
            if data is not None and len(data) > 0:
                filepath = shared_dir / f"{name}.npy"
                np.save(filepath, data)
                shared_shapes[name] = list(data.shape)
                logger.info(f"Saved shared table {name}: {data.shape}")
        
        # Save index mappings
        with open(shared_dir / 'index_mappings.json', 'w') as f:
            json.dump(index_mappings, f, indent=2)

        # Step 2: Generate per-split index arrays
        shapes = {}
        entity_info = {'entity_ids': [], 'samples_per_entity': {}}

        for flag in flags:
            logger.info(f"Generating index arrays for {flag} split...")
            split_shapes, split_entity_info = self._generate_split_indexed(
                flag, index_mappings
            )
            shapes[flag] = split_shapes

            # Merge entity info
            for eid in split_entity_info.get('entity_ids', []):
                if eid not in entity_info['entity_ids']:
                    entity_info['entity_ids'].append(eid)
            entity_info['samples_per_entity'].update(
                split_entity_info.get('samples_per_entity', {})
            )

        # Step 3: Save metadata
        metadata = TensorCacheMetadata(
            config_hash=self.config_hash,
            data_config=self.config,
            shapes=shapes,
            dtypes={name: spec['dtype'] for name, spec in self.INDEXED_ARRAY_SPECS.items()},
            cache_format=TensorCacheMetadata.FORMAT_INDEXED,
            shared_shapes=shared_shapes,
            entity_info=entity_info
        )
        metadata.save(self.cache_dir / 'metadata.json')

        logger.info(f"Cache generated at: {self.cache_dir}")
        return self.cache_dir

    def _build_shared_tables(
        self,
        flags: List[str]
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        """
        Build shared tables by collecting all unique timestamps across all splits.
        
        This is the first pass over the data. We iterate through all samples
        and collect unique (timestamp, data) pairs. The data includes:
        - Time series values (seq_x/seq_y values at that timestamp)
        - Embeddings (hetero_x/hetero_y embeddings at that timestamp)
        - Hetero time features
        
        Also collects entity-level static data (hetero_general, hetero_channel).
        
        Returns:
            Tuple of (shared_tables dict, index_mappings dict)
        """
        # Collectors for unique data
        timestamp_to_idx = {}  # timestamp -> index in shared tables
        entity_to_idx = {}     # entity_id -> index in entity tables
        
        # Lists to accumulate data (will convert to arrays)
        timeseries_list = []
        timestamps_list = []
        embeddings_list = []
        hetero_time_list = []
        entity_general_list = []
        entity_channel_list = []
        
        # Track shapes from first sample
        embed_dim = None
        n_features = None
        n_hetero_time_features = None
        
        # Iterate through all splits to collect unique data
        for flag in flags:
            datasets = self.data_provider.get_datasets(flag)
            if not datasets:
                continue
                
            for entity_id, dataset in datasets.items():
                if len(dataset) == 0:
                    continue
                
                # Register entity if new
                if entity_id not in entity_to_idx:
                    # Get first sample to extract entity-level data
                    sample = dataset[0]
                    entity_to_idx[entity_id] = len(entity_general_list)
                    
                    # hetero_general (index 9) and hetero_channel (index 10)
                    hetero_general = sample[9]
                    hetero_channel = sample[10]
                    
                    if hetero_general is not None:
                        entity_general_list.append(np.asarray(hetero_general))
                        if embed_dim is None:
                            embed_dim = entity_general_list[-1].shape[-1]
                    else:
                        entity_general_list.append(None)
                        
                    if hetero_channel is not None:
                        entity_channel_list.append(np.asarray(hetero_channel))
                    else:
                        entity_channel_list.append(None)
                
                # Sample a few samples to collect unique timestamps
                # We process in chunks to avoid memory issues
                for sample_idx in range(len(dataset)):
                    sample = dataset[sample_idx]
                    
                    # Extract timestamp arrays
                    x_time = sample[3]  # (input_len,) int64
                    y_time = sample[4]  # (output_len,) int64
                    
                    # Extract data arrays
                    seq_x = sample[1]   # (input_len, n_features)
                    seq_y = sample[2]   # (output_len, n_features)
                    hetero_x = sample[5]  # (input_len, embed_dim) or None
                    hetero_y = sample[6]  # (output_len, embed_dim) or None
                    hetero_x_time = sample[7]  # (input_len, n_hetero_tf) or None
                    hetero_y_time = sample[8]  # (output_len, n_hetero_tf) or None
                    
                    # Infer shapes from first valid data
                    if n_features is None and seq_x is not None:
                        seq_x_arr = np.asarray(seq_x)
                        if seq_x_arr.ndim >= 1:
                            n_features = seq_x_arr.shape[-1] if seq_x_arr.ndim > 1 else 1
                    
                    if n_hetero_time_features is None and hetero_x_time is not None:
                        htx_arr = np.asarray(hetero_x_time)
                        if htx_arr.ndim >= 1 and htx_arr.size > 0:
                            n_hetero_time_features = htx_arr.shape[-1] if htx_arr.ndim > 1 else 1
                    
                    if embed_dim is None and hetero_x is not None:
                        hx_arr = np.asarray(hetero_x)
                        if hx_arr.ndim >= 1 and hx_arr.size > 0:
                            embed_dim = hx_arr.shape[-1] if hx_arr.ndim > 1 else hx_arr.shape[0]
                    
                    # Process input window timestamps
                    if x_time is not None:
                        x_time_arr = np.asarray(x_time).flatten()
                        seq_x_arr = np.asarray(seq_x) if seq_x is not None else None
                        hetero_x_arr = np.asarray(hetero_x) if hetero_x is not None else None
                        hetero_x_time_arr = np.asarray(hetero_x_time) if hetero_x_time is not None else None
                        
                        for i, ts in enumerate(x_time_arr):
                            ts_key = int(ts)
                            if ts_key not in timestamp_to_idx:
                                timestamp_to_idx[ts_key] = len(timestamps_list)
                                timestamps_list.append(ts_key)
                                
                                # Time series value at this timestamp
                                if seq_x_arr is not None and i < len(seq_x_arr):
                                    ts_val = seq_x_arr[i] if seq_x_arr.ndim > 1 else seq_x_arr[i:i+1]
                                    timeseries_list.append(np.asarray(ts_val).flatten())
                                else:
                                    timeseries_list.append(np.zeros(n_features or 1, dtype=np.float32))
                                
                                # Embedding at this timestamp
                                if hetero_x_arr is not None and hetero_x_arr.size > 0 and i < len(hetero_x_arr):
                                    emb = hetero_x_arr[i] if hetero_x_arr.ndim > 1 else hetero_x_arr
                                    embeddings_list.append(np.asarray(emb).flatten())
                                else:
                                    embeddings_list.append(np.zeros(embed_dim or 768, dtype=np.float32))
                                
                                # Hetero time features at this timestamp
                                if hetero_x_time_arr is not None and hetero_x_time_arr.size > 0 and i < len(hetero_x_time_arr):
                                    htf = hetero_x_time_arr[i] if hetero_x_time_arr.ndim > 1 else hetero_x_time_arr
                                    hetero_time_list.append(np.asarray(htf).flatten())
                                else:
                                    hetero_time_list.append(np.zeros(n_hetero_time_features or 1, dtype=np.float32))
                    
                    # Process output window timestamps (same logic)
                    if y_time is not None:
                        y_time_arr = np.asarray(y_time).flatten()
                        seq_y_arr = np.asarray(seq_y) if seq_y is not None else None
                        hetero_y_arr = np.asarray(hetero_y) if hetero_y is not None else None
                        hetero_y_time_arr = np.asarray(hetero_y_time) if hetero_y_time is not None else None
                        
                        for i, ts in enumerate(y_time_arr):
                            ts_key = int(ts)
                            if ts_key not in timestamp_to_idx:
                                timestamp_to_idx[ts_key] = len(timestamps_list)
                                timestamps_list.append(ts_key)
                                
                                if seq_y_arr is not None and i < len(seq_y_arr):
                                    ts_val = seq_y_arr[i] if seq_y_arr.ndim > 1 else seq_y_arr[i:i+1]
                                    timeseries_list.append(np.asarray(ts_val).flatten())
                                else:
                                    timeseries_list.append(np.zeros(n_features or 1, dtype=np.float32))
                                
                                if hetero_y_arr is not None and hetero_y_arr.size > 0 and i < len(hetero_y_arr):
                                    emb = hetero_y_arr[i] if hetero_y_arr.ndim > 1 else hetero_y_arr
                                    embeddings_list.append(np.asarray(emb).flatten())
                                else:
                                    embeddings_list.append(np.zeros(embed_dim or 768, dtype=np.float32))
                                
                                if hetero_y_time_arr is not None and hetero_y_time_arr.size > 0 and i < len(hetero_y_time_arr):
                                    htf = hetero_y_time_arr[i] if hetero_y_time_arr.ndim > 1 else hetero_y_time_arr
                                    hetero_time_list.append(np.asarray(htf).flatten())
                                else:
                                    hetero_time_list.append(np.zeros(n_hetero_time_features or 1, dtype=np.float32))
        
        # Convert lists to arrays
        shared_tables = {}
        
        if timestamps_list:
            shared_tables['timestamps'] = np.array(timestamps_list, dtype=np.int64)
        
        if timeseries_list:
            shared_tables['timeseries'] = np.stack(timeseries_list).astype(np.float32)
        
        if embeddings_list:
            shared_tables['embeddings'] = np.stack(embeddings_list).astype(np.float32)
        
        if hetero_time_list:
            shared_tables['hetero_time'] = np.stack(hetero_time_list).astype(np.float32)
        
        # Entity tables
        if entity_general_list:
            valid_generals = [g for g in entity_general_list if g is not None]
            if valid_generals:
                shared_tables['entity_general'] = np.stack(valid_generals).astype(np.float32)
        
        if entity_channel_list:
            valid_channels = [c for c in entity_channel_list if c is not None]
            if valid_channels:
                shared_tables['entity_channel'] = np.stack(valid_channels).astype(np.float32)
        
        # Index mappings
        index_mappings = {
            'timestamp_to_idx': {str(k): v for k, v in timestamp_to_idx.items()},
            'entity_to_idx': entity_to_idx
        }
        
        logger.info(f"Built shared tables: {len(timestamps_list)} unique timestamps, {len(entity_to_idx)} entities")
        
        return shared_tables, index_mappings

    def _generate_split_indexed(
        self,
        flag: str,
        index_mappings: dict
    ) -> Tuple[dict, dict]:
        """
        Generate per-split index arrays with checkpointing support.
        
        For each sample, stores indices into the shared tables instead of
        the actual data. This enables ~100x disk space reduction.
        
        Supports resumption via progress.json - if generation is interrupted,
        completed entities are skipped on restart.
        """
        split_dir = self.cache_dir / flag
        split_dir.mkdir(exist_ok=True)
        progress_file = split_dir / 'progress.json'
        
        # Get datasets
        datasets = self.data_provider.get_datasets(flag)
        if not datasets:
            logger.warning(f"No datasets found for {flag} split")
            return {}, {}
        
        # Calculate total samples
        total_samples = sum(len(ds) for ds in datasets.values())
        logger.info(f"Total samples for {flag}: {total_samples:,}")
        
        if total_samples == 0:
            return {}, {}
        
        # Get sample shape from first dataset for array sizing
        first_dataset = next(iter(datasets.values()))
        first_sample = first_dataset[0]
        
        # Determine array shapes
        input_len = len(np.asarray(first_sample[3]).flatten())  # x_time
        output_len = len(np.asarray(first_sample[4]).flatten())  # y_time
        
        # Time features shape
        x_tf = first_sample[11]
        y_tf = first_sample[12]
        n_x_tf = np.asarray(x_tf).shape[-1] if x_tf is not None and np.asarray(x_tf).size > 0 else 0
        n_y_tf = np.asarray(y_tf).shape[-1] if y_tf is not None and np.asarray(y_tf).size > 0 else 0
        
        # Check for existing progress (resumption)
        progress = _load_progress(progress_file)
        completed_entities = set()
        current_idx = 0
        
        if progress and progress.get('config_hash') == self.config_hash:
            completed_entities = set(progress.get('completed_entities', []))
            current_idx = progress.get('current_write_position', 0)
            if completed_entities:
                logger.info(f"Resuming from checkpoint: {len(completed_entities)}/{len(datasets)} entities done")
        
        # Create or open arrays
        array_shapes = {
            'sample_ids': (total_samples,),
            'entity_indices': (total_samples,),
            'x_indices': (total_samples, input_len),
            'y_indices': (total_samples, output_len),
        }
        if n_x_tf > 0:
            array_shapes['x_time_features'] = (total_samples, input_len, n_x_tf)
        if n_y_tf > 0:
            array_shapes['y_time_features'] = (total_samples, output_len, n_y_tf)
        
        arrays = self._create_or_open_arrays(split_dir, array_shapes, completed_entities)
        
        # Initialize progress tracking
        entity_info = {'entity_ids': [], 'samples_per_entity': {}}
        timestamp_to_idx = {int(k): v for k, v in index_mappings['timestamp_to_idx'].items()}
        entity_to_idx = index_mappings['entity_to_idx']
        
        # Process with progress display
        if self.console is not None and RICH_AVAILABLE and self.verbose:
            shapes = self._process_indexed_with_rich(
                flag, datasets, arrays, entity_info, total_samples,
                timestamp_to_idx, entity_to_idx, completed_entities,
                current_idx, progress_file
            )
        else:
            shapes = self._process_indexed_with_tqdm(
                flag, datasets, arrays, entity_info,
                timestamp_to_idx, entity_to_idx, completed_entities,
                current_idx, progress_file
            )
        
        # Remove progress file on success
        if progress_file.exists():
            progress_file.unlink()
        
        return shapes, entity_info

    def _create_or_open_arrays(
        self,
        split_dir: Path,
        shapes: Dict[str, tuple],
        completed_entities: set
    ) -> Dict[str, np.memmap]:
        """Create new arrays or open existing ones for resumption."""
        arrays = {}
        
        for name, shape in shapes.items():
            dtype = self.INDEXED_ARRAY_SPECS.get(name, {}).get('dtype', 'float32')
            filepath = split_dir / f"{name}.npy"
            
            if filepath.exists() and completed_entities:
                # Open existing for resumption
                arrays[name] = np.lib.format.open_memmap(
                    str(filepath), dtype=dtype, mode='r+', shape=shape
                )
            else:
                # Create new
                arrays[name] = np.lib.format.open_memmap(
                    str(filepath), dtype=dtype, mode='w+', shape=shape
                )
        
        return arrays

    def _process_indexed_with_rich(
        self,
        flag: str,
        datasets: Dict[str, Any],
        arrays: Dict[str, np.memmap],
        entity_info: dict,
        total_samples: int,
        timestamp_to_idx: dict,
        entity_to_idx: dict,
        completed_entities: set,
        start_idx: int,
        progress_file: Path
    ) -> dict:
        """Process with Rich progress bars and checkpointing."""
        current_idx = start_idx
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
            refresh_per_second=10
        ) as progress:
            
            entity_task = progress.add_task(f"[cyan]Entities ({flag})", total=len(datasets))
            samples_task = progress.add_task("[dim]  └─ waiting...[/dim]", total=100)
            total_task = progress.add_task("[green]Total samples", total=total_samples)
            
            # Update total progress for already completed
            progress.update(total_task, completed=start_idx)
            
            for entity_id, dataset in datasets.items():
                entity_samples = len(dataset)
                
                # Skip completed entities
                if entity_id in completed_entities:
                    progress.update(entity_task, advance=1)
                    continue
                
                if entity_samples == 0:
                    progress.update(entity_task, advance=1)
                    continue
                
                entity_info['entity_ids'].append(entity_id)
                entity_info['samples_per_entity'][entity_id] = entity_samples
                
                progress.update(samples_task, description=f"[yellow]  └─ {entity_id}[/yellow]",
                               completed=0, total=entity_samples)
                
                entity_idx = entity_to_idx.get(entity_id, 0)
                
                # Process in chunks
                for chunk_start in range(0, entity_samples, self.chunk_size):
                    chunk_end = min(chunk_start + self.chunk_size, entity_samples)
                    chunk_size = chunk_end - chunk_start
                    
                    # Process chunk
                    for i in range(chunk_start, chunk_end):
                        sample = dataset[i]
                        write_idx = current_idx + i
                        
                        # Sample ID
                        arrays['sample_ids'][write_idx] = str(sample[0])
                        
                        # Entity index
                        arrays['entity_indices'][write_idx] = entity_idx
                        
                        # Build x_indices from timestamps
                        x_time = np.asarray(sample[3]).flatten()
                        x_indices = np.array([timestamp_to_idx.get(int(ts), 0) for ts in x_time], dtype=np.int32)
                        arrays['x_indices'][write_idx] = x_indices
                        
                        # Build y_indices from timestamps
                        y_time = np.asarray(sample[4]).flatten()
                        y_indices = np.array([timestamp_to_idx.get(int(ts), 0) for ts in y_time], dtype=np.int32)
                        arrays['y_indices'][write_idx] = y_indices
                        
                        # Time features (not deduplicated)
                        if 'x_time_features' in arrays and sample[11] is not None:
                            arrays['x_time_features'][write_idx] = np.asarray(sample[11])
                        if 'y_time_features' in arrays and sample[12] is not None:
                            arrays['y_time_features'][write_idx] = np.asarray(sample[12])
                    
                    progress.update(samples_task, advance=chunk_size)
                    progress.update(total_task, advance=chunk_size)
                
                # Checkpoint after entity
                current_idx += entity_samples
                for arr in arrays.values():
                    if hasattr(arr, 'flush'):
                        arr.flush()
                
                completed_entities.add(entity_id)
                _save_progress_atomic(progress_file, {
                    'config_hash': self.config_hash,
                    'completed_entities': list(completed_entities),
                    'current_write_position': current_idx,
                    'last_updated': datetime.now().isoformat(),
                    'status': 'in_progress'
                })
                
                progress.update(entity_task, advance=1)
            
            progress.update(samples_task, description="[dim]  └─ complete[/dim]", visible=False)
        
        return {name: list(arr.shape) for name, arr in arrays.items()}

    def _process_indexed_with_tqdm(
        self,
        flag: str,
        datasets: Dict[str, Any],
        arrays: Dict[str, np.memmap],
        entity_info: dict,
        timestamp_to_idx: dict,
        entity_to_idx: dict,
        completed_entities: set,
        start_idx: int,
        progress_file: Path
    ) -> dict:
        """Process with tqdm and checkpointing (fallback)."""
        current_idx = start_idx
        
        for entity_id, dataset in tqdm(datasets.items(), desc=f"Entities ({flag})", disable=not self.verbose):
            entity_samples = len(dataset)
            
            if entity_id in completed_entities or entity_samples == 0:
                continue
            
            entity_info['entity_ids'].append(entity_id)
            entity_info['samples_per_entity'][entity_id] = entity_samples
            entity_idx = entity_to_idx.get(entity_id, 0)
            
            for i in range(entity_samples):
                sample = dataset[i]
                write_idx = current_idx + i
                
                arrays['sample_ids'][write_idx] = str(sample[0])
                arrays['entity_indices'][write_idx] = entity_idx
                
                x_time = np.asarray(sample[3]).flatten()
                arrays['x_indices'][write_idx] = np.array(
                    [timestamp_to_idx.get(int(ts), 0) for ts in x_time], dtype=np.int32
                )
                
                y_time = np.asarray(sample[4]).flatten()
                arrays['y_indices'][write_idx] = np.array(
                    [timestamp_to_idx.get(int(ts), 0) for ts in y_time], dtype=np.int32
                )
                
                if 'x_time_features' in arrays and sample[11] is not None:
                    arrays['x_time_features'][write_idx] = np.asarray(sample[11])
                if 'y_time_features' in arrays and sample[12] is not None:
                    arrays['y_time_features'][write_idx] = np.asarray(sample[12])
            
            # Checkpoint
            current_idx += entity_samples
            for arr in arrays.values():
                if hasattr(arr, 'flush'):
                    arr.flush()
            
            completed_entities.add(entity_id)
            _save_progress_atomic(progress_file, {
                'config_hash': self.config_hash,
                'completed_entities': list(completed_entities),
                'current_write_position': current_idx,
                'last_updated': datetime.now().isoformat(),
                'status': 'in_progress'
            })
        
        return {name: list(arr.shape) for name, arr in arrays.items()}


# =============================================================================
# DATASET (SUPPORTS BOTH V1 AND V2 FORMATS)
# =============================================================================

class TensorCacheDataset(Dataset):
    """
    Ultra-fast dataset that loads from pre-computed tensor cache.
    
    Supports both V1 (direct) and V2 (indexed) cache formats.
    Format is auto-detected from metadata.json.
    
    V2 (indexed) format:
    - Loads shared tables into RAM at init time (~2-4 GB)
    - __getitem__ performs index lookups into shared tables
    - ~100x smaller disk footprint
    
    V1 (direct) format (legacy):
    - Memory-maps arrays directly
    - __getitem__ is direct array access
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        flag: str = 'train',
        preload_to_ram: bool = False
    ):
        """
        Initialize tensor cache dataset.

        Args:
            cache_dir: Path to tensor cache directory
            flag: Data split ('train', 'val', 'test')
            preload_to_ram: If True, load all data to RAM (for V1 format)
        """
        self.cache_dir = Path(cache_dir)
        self.split_dir = self.cache_dir / flag
        self.flag = flag
        self.preload_to_ram = preload_to_ram

        # Load metadata and detect format
        self.metadata = TensorCacheMetadata.load(self.cache_dir / 'metadata.json')
        
        if self.metadata.is_indexed:
            self._init_indexed_format()
        else:
            self._init_legacy_format()
        
        logger.info(
            f"TensorCacheDataset initialized: {flag}, "
            f"{self.n_samples:,} samples, "
            f"format={self.metadata.cache_format}"
        )

    def _init_indexed_format(self):
        """Initialize for V2 indexed format."""
        # Load shared tables into RAM (they're small after deduplication)
        shared_dir = self.cache_dir / 'shared'
        self.shared = {}
        
        for name in TensorCacheGenerator.SHARED_TABLE_SPECS.keys():
            filepath = shared_dir / f"{name}.npy"
            if filepath.exists():
                self.shared[name] = np.load(filepath, mmap_mode=None)  # Load to RAM
        
        # Load per-split index arrays (memory-mapped)
        self.arrays = {}
        for name in TensorCacheGenerator.INDEXED_ARRAY_SPECS.keys():
            filepath = self.split_dir / f"{name}.npy"
            if filepath.exists():
                self.arrays[name] = np.load(filepath, mmap_mode='r', allow_pickle=True)
        
        # Get sample count
        if 'x_indices' in self.arrays:
            self.n_samples = self.arrays['x_indices'].shape[0]
        elif 'sample_ids' in self.arrays:
            self.n_samples = self.arrays['sample_ids'].shape[0]
        else:
            self.n_samples = 0

    def _init_legacy_format(self):
        """Initialize for V1 direct format (backward compatibility)."""
        self.shared = None
        self.arrays = {}
        
        for name in TensorCacheGenerator.LEGACY_ARRAY_SPECS.keys():
            filepath = self.split_dir / f"{name}.npy"
            if filepath.exists():
                if self.preload_to_ram:
                    self.arrays[name] = np.load(filepath, mmap_mode=None, allow_pickle=True)
                else:
                    self.arrays[name] = np.load(filepath, mmap_mode='r', allow_pickle=True)
        
        first_array = next(iter(self.arrays.values()))
        self.n_samples = first_array.shape[0]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> tuple:
        """
        Get sample by index.
        
        For V2 (indexed) format: Performs index lookups into shared tables.
        For V1 (direct) format: Direct array access.
        """
        if self.metadata.is_indexed:
            return self._getitem_indexed(index)
        else:
            return self._getitem_legacy(index)

    def _getitem_indexed(self, index: int) -> tuple:
        """Get sample using V2 indexed format - lookup into shared tables."""
        # Get indices
        x_idx = self.arrays['x_indices'][index]      # (input_len,)
        y_idx = self.arrays['y_indices'][index]      # (output_len,)
        entity_idx = self.arrays['entity_indices'][index]
        
        # Lookup time series
        seq_x = self.shared['timeseries'][x_idx] if 'timeseries' in self.shared else None
        seq_y = self.shared['timeseries'][y_idx] if 'timeseries' in self.shared else None
        
        # Lookup timestamps
        x_time = self.shared['timestamps'][x_idx] if 'timestamps' in self.shared else None
        y_time = self.shared['timestamps'][y_idx] if 'timestamps' in self.shared else None
        
        # Lookup embeddings
        hetero_x = self.shared['embeddings'][x_idx] if 'embeddings' in self.shared else None
        hetero_y = self.shared['embeddings'][y_idx] if 'embeddings' in self.shared else None
        
        # Lookup hetero time features
        hetero_x_time = self.shared['hetero_time'][x_idx] if 'hetero_time' in self.shared else None
        hetero_y_time = self.shared['hetero_time'][y_idx] if 'hetero_time' in self.shared else None
        
        # Lookup entity-level static embeddings
        hetero_general = self.shared['entity_general'][entity_idx] if 'entity_general' in self.shared else None
        hetero_channel = self.shared['entity_channel'][entity_idx] if 'entity_channel' in self.shared else None
        
        # Per-sample data (not deduplicated)
        sample_id = self.arrays.get('sample_ids', [''])[index] if 'sample_ids' in self.arrays else ''
        x_time_features = self.arrays['x_time_features'][index] if 'x_time_features' in self.arrays else None
        y_time_features = self.arrays['y_time_features'][index] if 'y_time_features' in self.arrays else None
        
        return (
            sample_id,
            seq_x,
            seq_y,
            x_time,
            y_time,
            hetero_x,
            hetero_y,
            hetero_x_time,
            hetero_y_time,
            hetero_general,
            hetero_channel,
            x_time_features,
            y_time_features,
        )

    def _getitem_legacy(self, index: int) -> tuple:
        """Get sample using V1 direct format (backward compatibility)."""
        return (
            self._get_item('sample_ids', index, default=''),
            self._get_item('seq_x', index),
            self._get_item('seq_y', index),
            self._get_item('x_time', index),
            self._get_item('y_time', index),
            self._get_item('hetero_x', index),
            self._get_item('hetero_y', index),
            self._get_item('hetero_x_time', index),
            self._get_item('hetero_y_time', index),
            self._get_item('hetero_general', index),
            self._get_item('hetero_channel', index),
            self._get_item('x_time_features', index),
            self._get_item('y_time_features', index),
        )

    def _get_item(self, name: str, index: int, default=None):
        """Get item from array or return default (for legacy format)."""
        if name in self.arrays:
            return self.arrays[name][index]
        return default if default is not None else np.zeros((1,), dtype=np.float32)

    def get_scaler_params(self) -> Optional[dict]:
        """Get scaler parameters for inverse transform."""
        return self.metadata.scaler_params


# =============================================================================
# VALIDATION AND UTILITIES
# =============================================================================

def validate_cache(cache_dir: Union[str, Path], config: dict) -> Tuple[bool, str]:
    """
    Validate that a cache exists and matches the given config.
    
    Also checks for incomplete caches (progress.json exists).
    """
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        return False, f"Cache directory does not exist: {cache_dir}"

    metadata_path = cache_dir / 'metadata.json'
    if not metadata_path.exists():
        return False, f"Metadata file not found: {metadata_path}"

    try:
        metadata = TensorCacheMetadata.load(metadata_path)
    except Exception as e:
        return False, f"Failed to load metadata: {e}"

    # Check for incomplete cache (progress file exists)
    for flag in ['train', 'val', 'test']:
        progress_file = cache_dir / flag / 'progress.json'
        if progress_file.exists():
            progress = _load_progress(progress_file)
            if progress and progress.get('status') == 'in_progress':
                completed = len(progress.get('completed_entities', []))
                return False, (
                    f"Cache generation incomplete for {flag} split "
                    f"({completed} entities done). Run 'generate' to resume."
                )

    # Check config hash
    current_hash = compute_config_hash(config)
    if metadata.config_hash != current_hash:
        return False, (
            f"Config hash mismatch (cache may be stale). "
            f"Expected: {current_hash}, Got: {metadata.config_hash}"
        )

    # Check that required files exist
    if metadata.is_indexed:
        # V2 format: check shared tables exist
        shared_dir = cache_dir / 'shared'
        if not shared_dir.exists():
            return False, "Missing shared/ directory for indexed format"
    else:
        # V1 format: check split directories have data
        for flag in ['train', 'val', 'test']:
            split_dir = cache_dir / flag
            if split_dir.exists():
                seq_x_file = split_dir / 'seq_x.npy'
                if not seq_x_file.exists():
                    return False, f"Missing seq_x.npy in {flag} split"

    return True, "Cache is valid"


def get_tensor_cache_dataloader(
    cache_dir: Union[str, Path],
    flag: str,
    batch_size: int,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    shuffle: bool = None,
    preload_to_ram: bool = False
) -> torch.utils.data.DataLoader:
    """
    Create an optimized DataLoader from tensor cache.
    
    Works with both V1 (direct) and V2 (indexed) formats.
    """
    if shuffle is None:
        shuffle = (flag == 'train')

    dataset = TensorCacheDataset(
        cache_dir=cache_dir,
        flag=flag,
        preload_to_ram=preload_to_ram
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=(flag == 'train')
    )
