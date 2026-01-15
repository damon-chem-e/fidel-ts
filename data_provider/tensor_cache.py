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
ARCHITECTURE
================================================================================

The module is organized into these components:

1. CONFIGURATION & HASHING
   - compute_config_hash(): Determines cache validity based on config params
   
2. CHECKPOINTING
   - _save_progress_atomic(): Crash-safe progress file writing
   - _load_progress(): Progress file loading with corruption handling
   
3. METADATA
   - TensorCacheMetadata: Cache metadata, versioning, format detection

4. SAMPLE DATA EXTRACTION
   - SampleDataExtractor: Extracts and validates data from dataset samples
   - Handles shape inference, None values, array conversion
   
5. SHARED TABLE BUILDING  
   - SharedTableBuilder: Collects unique timestamps across all data
   - Deduplicates time series, embeddings, and entity-level data
   
6. INDEX ARRAY GENERATION
   - IndexArrayWriter: Writes per-sample index arrays with checkpointing
   - Supports resumption from interrupted generation
   
7. CACHE GENERATION
   - TensorCacheGenerator: Orchestrates the full generation pipeline
   
8. CACHE READING
   - TensorCacheDataset: Ultra-fast dataset for training
   - Supports both V1 (direct) and V2 (indexed) formats

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
WHY DEDUPLICATION WORKS
================================================================================

Time series forecasting uses SLIDING WINDOWS:
- Sample i:   timestamps [t, t+1, ..., t+input_len-1] → predict [t+input_len, ...]
- Sample i+1: timestamps [t+1, t+2, ..., t+input_len] → predict [t+input_len+1, ...]

Adjacent samples share (input_len - 1) / input_len ≈ 99.65% of their data.

OLD APPROACH (V1 - direct):
- Store full arrays per sample: hetero_x[i] = embeddings for sample i
- 726k samples × 288 timestamps × 768 dims × 4 bytes = 267 GB (just input embeddings!)

NEW APPROACH (V2 - indexed):
- Store unique embeddings once: shared_embeddings[726k unique, 768 dims] = 2.2 GB
- Store indices per sample: x_indices[726k samples, 288] = 0.8 GB
- Total: ~3 GB instead of ~400 GB

================================================================================
CHECKPOINTING & RESUMABILITY
================================================================================

Generation can be interrupted and resumed:
- progress.json tracks completed entities and write position
- Arrays are flushed after each entity
- On restart, completed entities are skipped
- progress.json is deleted on successful completion

================================================================================
"""

import json
import hashlib
import logging
import numpy as np
import torch
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, TYPE_CHECKING
from datetime import datetime
from tqdm import tqdm

# BEGIN DEBUG
import psutil
import os
import gc

def _debug_memory(label: str, force_gc: bool = False, collector: Optional[Any] = None) -> None:
    """
    Print current memory usage with a label.
    
    Enable via environment variable: TENSOR_CACHE_DEBUG_MEMORY=1
    
    Args:
        label: Description of current operation/phase
        force_gc: If True, run gc.collect() before measuring memory
        collector: Optional SharedTableCollector to report sizes
    """
    if os.environ.get('TENSOR_CACHE_DEBUG_MEMORY', '0') != '1':
        return
    
    if force_gc:
        gc.collect()
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_gb = mem_info.rss / (1024 ** 3)
    vms_gb = mem_info.vms / (1024 ** 3)
    
    msg = f"[DEBUG MEM] {label}: RSS={rss_gb:.2f}GB, VMS={vms_gb:.2f}GB"
    
    if collector is not None:
        msg += (f" | Collector: timestamps={len(collector.timestamps)}, "
                f"embeddings={len(collector.embeddings)}, "
                f"timeseries={len(collector.timeseries)}, "
                f"hetero_time={len(collector.hetero_time)}")
    
    print(msg)

_DEBUG_GETITEM_COUNT = 0
_DEBUG_GETITEM_LIMIT = 5  # Only print first N __getitem__ calls
# END DEBUG

# Rich progress bar imports (optional, graceful fallback to tqdm)
try:
    from rich.progress import (
        Progress, BarColumn, TextColumn, TimeElapsedColumn,
        TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn
    )
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

if TYPE_CHECKING:
    from data_provider.data_factory import Data_Provider

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: CONFIGURATION & HASHING
# =============================================================================
# These functions determine when a cache is valid vs needs regeneration.
# The hash is computed from config parameters that affect the output data.
# =============================================================================

# Keys that affect cache validity - if ANY of these change, cache is stale
CACHE_RELEVANT_KEYS = [
    'input_len',                  # Length of input window (affects array shapes)
    'output_len',                 # Length of output window (affects array shapes)
    'hetero_stride',              # Embedding stride (affects which timestamps have data)
    'scale',                      # Normalization setting (affects values)
    'split',                      # Train/val/test split ratios
    'split_info',                 # Alternate split specification used by some configs
    'truncate_train_for_purge',   # Data truncation setting
    'downsample',                 # Downsampling factor
    'hetero_type',                # Type of heterogeneous data
    'data_name',                  # Dataset identifier
    'timemmd_text_output',        # 'text' vs 'embedding' changes hetero payload + shapes
    'missing_value_strategy'      # How missing values are handled
]


def compute_config_hash(config: dict) -> str:
    """
    Compute a deterministic hash of config parameters that affect cache validity.
    
    WHY THIS EXISTS:
    - Cache must be regenerated if data-affecting config changes
    - Hash provides a quick validity check without comparing all data
    - Only includes relevant keys to avoid false invalidation
    
    Args:
        config: Experiment configuration dictionary
        
    Returns:
        16-character hex hash string (truncated SHA-256)
    """
    # Extract only the keys that affect cache validity
    relevant_config = {
        key: config[key] 
        for key in CACHE_RELEVANT_KEYS 
        if key in config
    }
    
    # Deterministic JSON serialization (sorted keys, consistent formatting)
    config_str = json.dumps(relevant_config, sort_keys=True, default=str)
    
    # SHA-256 truncated to 16 chars (64 bits of entropy - sufficient for collision avoidance)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# =============================================================================
# SECTION 2: CHECKPOINTING UTILITIES
# =============================================================================
# These functions enable crash-safe progress tracking and resumption.
# Critical for long-running cache generation jobs that may be interrupted.
# =============================================================================

def _save_progress_atomic(path: Path, data: dict) -> None:
    """
    Atomically save progress file using write-to-temp + rename pattern.
    
    WHY ATOMIC WRITES:
    - If process dies during write, we get a corrupted half-written file
    - Atomic rename ensures we either have old file or new file, never partial
    - On POSIX systems, rename() is atomic within same filesystem
    
    PATTERN:
    1. Write to temporary file (progress.progress.tmp)
    2. Rename temp to target (atomic operation)
    3. If crash during step 1: old file remains intact
    4. If crash during step 2: rename either completes or doesn't
    
    Args:
        path: Target path for progress file
        data: Progress state dictionary to save
    """
    temp_path = path.with_suffix('.progress.tmp')
    
    # Step 1: Write to temp file (may be interrupted)
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Step 2: Atomic rename (either completes fully or not at all)
    temp_path.rename(path)


def _load_progress(path: Path) -> Optional[dict]:
    """
    Load progress file with corruption handling.
    
    WHY CORRUPTION HANDLING:
    - Progress files may be corrupted if process died during non-atomic operations
    - Better to return None (restart from beginning) than crash
    - Logs warning so user knows what happened
    
    Args:
        path: Path to progress.json file
        
    Returns:
        Progress dictionary if valid, None if missing or corrupted
    """
    if not path.exists():
        return None
        
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Corrupted progress file {path}: {e}")
        return None


# =============================================================================
# SECTION 3: METADATA
# =============================================================================
# Metadata tracks cache version, format, shapes, and validation info.
# Supports both V1 (direct) and V2 (indexed) formats.
# =============================================================================

class TensorCacheMetadata:
    """
    Metadata container for tensor cache validation and information.
    
    RESPONSIBILITIES:
    - Track cache version for compatibility checking
    - Store config hash for staleness detection
    - Record array shapes for pre-allocation
    - Identify cache format (indexed vs direct)
    - Store num_news_items for embedding shape handling
    
    VERSIONING:
    - VERSION = "2.1.0": Current version with indexed format and downtime handling
    - VERSION = "2.0.0": Indexed format without explicit num_news_items
    - VERSION = "1.0.0": Legacy direct format (still readable)
    
    FORMATS:
    - FORMAT_INDEXED: V2 format with shared tables + indices (~100x smaller)
    - FORMAT_DIRECT: V1 format with per-sample arrays (legacy)
    
    NUM_NEWS_ITEMS:
    - N=1: Embeddings stored as (D,) per timestamp - only text embedding
    - N=2: Embeddings stored as (2, D) per timestamp - text embedding + downtime indicator
    - Determined during cache generation based on whether training data has downtime
    """

    VERSION = "2.1.0"
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
        version: Optional[str] = None,
        num_news_items: int = 1
    ):
        """
        Initialize metadata.
        
        Args:
            config_hash: Hash of cache-relevant config parameters
            data_config: Full config dict for reference
            shapes: Per-split array shapes {split: {array_name: shape}}
            dtypes: Array data types {array_name: dtype_string}
            cache_format: 'indexed' (V2) or 'direct' (V1)
            shared_shapes: Shapes of shared tables (V2 only)
            scaler_params: Normalization parameters for inverse transform
            entity_info: Entity IDs and sample counts
            created_at: ISO timestamp of cache creation
            version: Metadata version string
            num_news_items: Number of news items per timestamp (1=embedding only, 2=embedding+downtime)
        """
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
        self.num_news_items = num_news_items

    def to_dict(self) -> dict:
        """Serialize metadata to dictionary for JSON storage."""
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
            'entity_info': self.entity_info,
            'num_news_items': self.num_news_items
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TensorCacheMetadata':
        """Deserialize metadata from dictionary."""
        return cls(
            config_hash=data['config_hash'],
            data_config=data['data_config'],
            shapes=data['shapes'],
            dtypes=data['dtypes'],
            # Default to 'direct' for old caches without format field
            cache_format=data.get('cache_format', cls.FORMAT_DIRECT),
            shared_shapes=data.get('shared_shapes', {}),
            scaler_params=data.get('scaler_params', {}),
            entity_info=data.get('entity_info', {}),
            created_at=data.get('created_at'),
            version=data.get('version', '1.0.0'),
            # Default to 1 for old caches without num_news_items field
            # Old caches stored flattened embeddings (effectively N=1)
            num_news_items=data.get('num_news_items', 1)
        )

    def save(self, path: Path) -> None:
        """Save metadata to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> 'TensorCacheMetadata':
        """Load metadata from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @property
    def is_indexed(self) -> bool:
        """Check if this cache uses the indexed (deduplicated) format."""
        return self.cache_format == self.FORMAT_INDEXED


# =============================================================================
# SECTION 4: SAMPLE DATA EXTRACTION
# =============================================================================
# These classes handle extracting data from dataset samples and inferring shapes.
# The Universal_Dataset returns 13-element tuples with specific indices.
# =============================================================================

# Sample tuple indices (matches Universal_Dataset.__getitem__ output)
SAMPLE_IDX_SAMPLE_ID = 0        # str: unique sample identifier
SAMPLE_IDX_SEQ_X = 1            # (input_len, n_features): input time series
SAMPLE_IDX_SEQ_Y = 2            # (output_len, n_features): output time series
SAMPLE_IDX_X_TIME = 3           # (input_len,): input timestamps (int64)
SAMPLE_IDX_Y_TIME = 4           # (output_len,): output timestamps (int64)
SAMPLE_IDX_HETERO_X = 5         # (input_len, embed_dim): input embeddings
SAMPLE_IDX_HETERO_Y = 6         # (output_len, embed_dim): output embeddings
SAMPLE_IDX_HETERO_X_TIME = 7    # (input_len, n_tf): input hetero time features
SAMPLE_IDX_HETERO_Y_TIME = 8    # (output_len, n_tf): output hetero time features
SAMPLE_IDX_HETERO_GENERAL = 9   # (embed_dim,): entity-level general embedding
SAMPLE_IDX_HETERO_CHANNEL = 10  # (embed_dim,): entity-level channel embedding
SAMPLE_IDX_X_TIME_FEATURES = 11 # (input_len, n_tf): input time features
SAMPLE_IDX_Y_TIME_FEATURES = 12 # (output_len, n_tf): output time features


@dataclass
class InferredShapes:
    """
    Container for shapes inferred from sample data.
    
    WHY INFERENCE:
    - Shapes vary by dataset (different n_features, embed_dim, etc.)
    - We need to know shapes before allocating arrays
    - Infer from first valid sample rather than hardcoding
    """
    n_features: Optional[int] = None          # Number of time series features
    embed_dim: Optional[int] = None           # Embedding dimension (e.g., 768)
    n_hetero_time_features: Optional[int] = None  # Hetero time feature count
    input_len: Optional[int] = None           # Input window length
    output_len: Optional[int] = None          # Output window length
    n_x_time_features: Optional[int] = None   # Per-sample x time feature count
    n_y_time_features: Optional[int] = None   # Per-sample y time feature count


def _safe_array(data: Any) -> Optional[np.ndarray]:
    """
    Safely convert data to numpy array, handling None and empty cases.
    
    Args:
        data: Input data (array, list, scalar, or None)
        
    Returns:
        numpy array or None if data is None/empty
    """
    if data is None:
        return None
    arr = np.asarray(data)
    if arr.size == 0:
        return None
    return arr


def _infer_dim(arr: Optional[np.ndarray], axis: int = -1) -> Optional[int]:
    """
    Infer dimension size from array along specified axis.
    
    For embedding inference:
    - 1D array (768,) means static embedding with dim=768 (axis=-1 returns 768)
    - 2D array (360, 768) means per-timestep embeddings with dim=768 (axis=-1 returns 768)
    
    Args:
        arr: Input array or None
        axis: Axis to get size from (default: last axis)
        
    Returns:
        Size along axis, or None if array is None/invalid
    """
    if arr is None or arr.ndim == 0:
        return None
    if arr.ndim == 1:
        # For 1D arrays, the embedding dimension is the array length
        return arr.shape[0]
    return arr.shape[axis]


def infer_shapes_from_sample(sample: tuple) -> InferredShapes:
    """
    Infer all shapes from a single sample tuple.
    
    WHY THIS FUNCTION:
    - Need shapes to allocate arrays before processing all samples
    - Different datasets have different dimensions
    - Extract all shape info from one sample to avoid repeated checks
    
    Args:
        sample: 13-element tuple from dataset.__getitem__
        
    Returns:
        InferredShapes with all discoverable dimensions
    """
    shapes = InferredShapes()
    
    # Time arrays - always present, give us window lengths
    x_time = _safe_array(sample[SAMPLE_IDX_X_TIME])
    y_time = _safe_array(sample[SAMPLE_IDX_Y_TIME])
    if x_time is not None:
        shapes.input_len = len(x_time.flatten())
    if y_time is not None:
        shapes.output_len = len(y_time.flatten())
    
    # Time series - gives us n_features
    seq_x = _safe_array(sample[SAMPLE_IDX_SEQ_X])
    if seq_x is not None:
        shapes.n_features = _infer_dim(seq_x, axis=-1)
    
    # Embeddings - gives us embed_dim
    hetero_x = _safe_array(sample[SAMPLE_IDX_HETERO_X])
    hetero_general = _safe_array(sample[SAMPLE_IDX_HETERO_GENERAL])
    if hetero_x is not None:
        shapes.embed_dim = _infer_dim(hetero_x, axis=-1)
    elif hetero_general is not None:
        shapes.embed_dim = _infer_dim(hetero_general, axis=-1)
    
    # Hetero time features
    hetero_x_time = _safe_array(sample[SAMPLE_IDX_HETERO_X_TIME])
    if hetero_x_time is not None:
        shapes.n_hetero_time_features = _infer_dim(hetero_x_time, axis=-1)
    
    # Per-sample time features
    x_tf = _safe_array(sample[SAMPLE_IDX_X_TIME_FEATURES])
    y_tf = _safe_array(sample[SAMPLE_IDX_Y_TIME_FEATURES])
    if x_tf is not None:
        shapes.n_x_time_features = _infer_dim(x_tf, axis=-1)
    if y_tf is not None:
        shapes.n_y_time_features = _infer_dim(y_tf, axis=-1)
    
    return shapes


# =============================================================================
# SECTION 5: SHARED TABLE BUILDING
# =============================================================================
# These classes collect unique data across all samples for deduplication.
# The key insight: adjacent samples share 99%+ of their timestamps.
# =============================================================================

@dataclass
class SharedTableCollector:
    """
    Collector for building shared (deduplicated) tables.
    
    DESIGN PATTERN:
    - Uses dict for O(1) duplicate checking (timestamp_to_idx)
    - Uses lists for O(1) append (converted to arrays at end)
    - Processes all splits to ensure shared tables cover all data
    
    MEMORY CONSIDERATION:
    - Lists grow during collection phase
    - Converted to numpy arrays once at the end
    - Final arrays are much smaller than per-sample storage
    
    DOWNTIME HANDLING:
    - The original dataloader produces embeddings with shape (L, N, D) where N=2:
      - [:, 0, :] = actual text embedding
      - [:, 1, :] = downtime indicator (or zeros if no downtime)
    - We detect whether training data has any non-zero downtime indicators
    - If no downtime in training: store N=1 (only embeddings), saving 50% memory
    - If downtime in training: store N=2 (embeddings + downtime), matching original
    - Rationale: If no downtime in training, model won't learn to use it anyway
    """
    # Mapping from timestamp to index in shared tables
    timestamp_to_idx: Dict[int, int] = field(default_factory=dict)
    
    # Mapping from entity_id to index in entity tables  
    entity_to_idx: Dict[str, int] = field(default_factory=dict)
    
    # Data lists (will become numpy arrays)
    timestamps: List[int] = field(default_factory=list)
    timeseries: List[np.ndarray] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)
    hetero_time: List[np.ndarray] = field(default_factory=list)
    entity_general: List[Optional[np.ndarray]] = field(default_factory=list)
    entity_channel: List[Optional[np.ndarray]] = field(default_factory=list)
    
    # Inferred shapes (updated as we see data)
    shapes: InferredShapes = field(default_factory=InferredShapes)
    
    # Downtime handling: num_news_items determines embedding storage format
    # N=1: store only embedding (no downtime), shape (L, D)
    # N=2: store embedding + downtime indicator, shape (L, 2, D)
    num_news_items: int = 1  # Default to N=1; set to 2 if downtime detected in training


def _detect_downtime_in_training(
    data_provider,
    max_samples_to_check: int = 1000
) -> bool:
    """
    Detect if training data has any non-zero downtime indicators.
    
    RATIONALE:
    The original dataloader produces embeddings with shape (L, N, D) where N=2:
    - [:, 0, :] = actual text embedding
    - [:, 1, :] = downtime indicator (non-zero if sensor was down, zeros otherwise)
    
    If training data has NO non-zero downtime indicators, the model won't learn
    how to use them anyway. In this case, we can store N=1 (just embeddings)
    which saves 50% memory and matches model expectations.
    
    If training data HAS non-zero downtime indicators, we store N=2 to preserve
    this signal for the model to learn from.
    
    Args:
        data_provider: Data provider with get_datasets() method
        max_samples_to_check: Maximum samples to scan (for efficiency)
        
    Returns:
        True if any non-zero downtime indicators found in training data
    """
    datasets = data_provider.get_datasets('train')
    if not datasets:
        return False
    
    samples_checked = 0
    
    for entity_id, dataset in datasets.items():
        if len(dataset) == 0:
            continue
        
        # Check a subset of samples from each entity
        samples_per_entity = min(len(dataset), max_samples_to_check // max(len(datasets), 1))
        
        for sample_idx in range(0, len(dataset), max(1, len(dataset) // samples_per_entity)):
            if samples_checked >= max_samples_to_check:
                break
                
            sample = dataset[sample_idx]
            
            # Check hetero_x (input embeddings)
            hetero_x = _safe_array(sample[SAMPLE_IDX_HETERO_X])
            if hetero_x is not None and hetero_x.ndim == 3:
                # Shape: (L, N, D) where N should be 2 for embeddings + downtime
                if hetero_x.shape[1] >= 2:
                    downtime_indicator = hetero_x[:, 1, :]  # Second item is downtime
                    if np.any(downtime_indicator != 0):
                        return True
            
            # Check hetero_y (output embeddings)
            hetero_y = _safe_array(sample[SAMPLE_IDX_HETERO_Y])
            if hetero_y is not None and hetero_y.ndim == 3:
                if hetero_y.shape[1] >= 2:
                    downtime_indicator = hetero_y[:, 1, :]
                    if np.any(downtime_indicator != 0):
                        return True
            
            samples_checked += 1
        
        if samples_checked >= max_samples_to_check:
            break
    
    return False


def _register_timestamp_data(
    collector: SharedTableCollector,
    timestamp: int,
    ts_value: Optional[np.ndarray],
    embedding: Optional[np.ndarray],
    hetero_time_feat: Optional[np.ndarray]
) -> None:
    """
    Register data for a timestamp if not already seen.
    
    WHY DEDUPLICATION HAPPENS HERE:
    - Check if timestamp already in mapping (O(1) dict lookup)
    - If new: assign next index, append data to lists
    - If seen: skip (data already stored)
    
    EMBEDDING HANDLING:
    - If collector.num_news_items == 1: store embedding as (D,) - just the embedding
    - If collector.num_news_items == 2: store embedding as (2, D) - embedding + downtime
    - This is determined by _detect_downtime_in_training() before collection starts
    
    Args:
        collector: SharedTableCollector to update
        timestamp: Unix timestamp (int64)
        ts_value: Time series value at this timestamp
        embedding: Text embedding at this timestamp (may be (D,), (N, D), or full per-timestep)
        hetero_time_feat: Hetero time features at this timestamp
    """
    ts_key = int(timestamp)
    
    # Skip if already registered
    if ts_key in collector.timestamp_to_idx:
        return
    
    # Assign next available index
    idx = len(collector.timestamps)
    collector.timestamp_to_idx[ts_key] = idx
    collector.timestamps.append(ts_key)
    
    # Store time series value (with fallback to zeros)
    if ts_value is not None:
        collector.timeseries.append(np.asarray(ts_value).flatten())
    else:
        n_features = collector.shapes.n_features or 1
        collector.timeseries.append(np.zeros(n_features, dtype=np.float32))
    
    # Store embedding based on num_news_items setting
    # N=1: store as (D,) - just the text embedding
    # N=2: store as (2, D) - text embedding + downtime indicator
    if embedding is not None:
        emb_arr = np.asarray(embedding)
        
        if collector.num_news_items == 1:
            # N=1: Store only the text embedding, flattened to (D,)
            # CRITICAL: Use .copy() to break references to parent arrays (prevents memory leaks)
            # Array slicing creates views that keep references to the original large arrays
            if emb_arr.ndim == 1:
                emb_to_store = emb_arr.copy()  # Copy to break reference
            elif emb_arr.ndim == 2:
                emb_to_store = emb_arr[0].copy()  # Copy slice to break reference to parent
            else:
                emb_to_store = emb_arr.flatten()  # flatten() already creates a copy
            collector.embeddings.append(emb_to_store)
            
            # Update embed_dim if not yet set
            if collector.shapes.embed_dim is None:
                collector.shapes.embed_dim = len(emb_to_store)
        else:
            # N=2: Store full embedding + downtime indicator as (2, D)
            # If embedding is 1D, we need to handle this edge case
            # If 2D with shape (N, D), store the full array
            if emb_arr.ndim == 1:
                # Edge case: 1D embedding provided but N=2 requested
                # Create (2, D) with zeros for downtime
                embed_dim = len(emb_arr)
                emb_to_store = np.zeros((2, embed_dim), dtype=np.float32)
                emb_to_store[0] = emb_arr.copy()  # Copy to break reference
            elif emb_arr.ndim == 2 and emb_arr.shape[0] >= 2:
                # Normal case: (N, D) array, take first 2 items
                # CRITICAL: Use .copy() to break references to parent arrays
                emb_to_store = emb_arr[:2].copy()  # (2, D) - copy to break reference
            elif emb_arr.ndim == 2 and emb_arr.shape[0] == 1:
                # Edge case: Only 1 item, pad with zeros for downtime
                embed_dim = emb_arr.shape[1]
                emb_to_store = np.zeros((2, embed_dim), dtype=np.float32)
                emb_to_store[0] = emb_arr[0]
            else:
                # Fallback: flatten and reshape
                emb_flat = emb_arr.flatten()
                embed_dim = len(emb_flat)
                emb_to_store = np.zeros((2, embed_dim), dtype=np.float32)
                emb_to_store[0] = emb_flat
            
            collector.embeddings.append(emb_to_store)
            
            # Update embed_dim if not yet set (based on actual embedding dim, not total)
            if collector.shapes.embed_dim is None:
                collector.shapes.embed_dim = emb_to_store.shape[1]
    else:
        # Fallback to zeros with appropriate shape
        embed_dim = collector.shapes.embed_dim or 768
        if collector.num_news_items == 1:
            collector.embeddings.append(np.zeros(embed_dim, dtype=np.float32))
        else:
            collector.embeddings.append(np.zeros((2, embed_dim), dtype=np.float32))
    
    # Store hetero time features (with fallback to zeros)
    if hetero_time_feat is not None:
        collector.hetero_time.append(np.asarray(hetero_time_feat).flatten())
    else:
        n_htf = collector.shapes.n_hetero_time_features or 1
        collector.hetero_time.append(np.zeros(n_htf, dtype=np.float32))


def _register_entity_data(
    collector: SharedTableCollector,
    entity_id: str,
    hetero_general: Optional[np.ndarray],
    hetero_channel: Optional[np.ndarray]
) -> None:
    """
    Register entity-level static data if entity not already seen.
    
    WHY ENTITY DEDUPLICATION:
    - hetero_general and hetero_channel are IDENTICAL for all samples from same entity
    - Storing per-sample wastes (n_samples / n_entities) times the space
    - Store once per entity, reference by entity_idx
    
    Args:
        collector: SharedTableCollector to update
        entity_id: Unique entity identifier
        hetero_general: Entity's general embedding
        hetero_channel: Entity's channel embedding
    """
    if entity_id in collector.entity_to_idx:
        return
    
    # Assign next available index
    idx = len(collector.entity_general)
    collector.entity_to_idx[entity_id] = idx
    
    # Store entity data (may be None)
    if hetero_general is not None:
        arr = np.asarray(hetero_general)
        collector.entity_general.append(arr)
        # Update embed_dim if not yet known
        if collector.shapes.embed_dim is None:
            collector.shapes.embed_dim = arr.shape[-1]
    else:
        collector.entity_general.append(None)
    
    collector.entity_channel.append(
        np.asarray(hetero_channel) if hetero_channel is not None else None
    )


def _process_sample_for_collection(
    collector: SharedTableCollector,
    sample: tuple
) -> None:
    """
    Process a single sample, registering all unique timestamps and entity data.
    
    SAMPLE STRUCTURE (Universal_Dataset output):
    - sample[3]: x_time - input timestamps (input_len,)
    - sample[4]: y_time - output timestamps (output_len,)
    - sample[1]: seq_x - input time series (input_len, n_features)
    - sample[2]: seq_y - output time series (output_len, n_features)
    - sample[5]: hetero_x - input embeddings (input_len, num_items, embed_dim) where num_items=2
    - sample[6]: hetero_y - output embeddings (output_len, num_items, embed_dim) where num_items=2
    - sample[7]: hetero_x_time - input hetero time features
    - sample[8]: hetero_y_time - output hetero time features
    
    EMBEDDING HANDLING:
    - The original dataloader produces embeddings with shape (L, N, D) where N=2:
      - [:, 0, :] = actual text embedding
      - [:, 1, :] = downtime indicator (or zeros if no downtime)
    - We pass the FULL embedding (including downtime) to _register_timestamp_data
    - _register_timestamp_data decides what to store based on collector.num_news_items
    
    Args:
        collector: SharedTableCollector to update
        sample: 13-element tuple from dataset.__getitem__
    """
    # Update shapes if not yet inferred
    if collector.shapes.n_features is None:
        sample_shapes = infer_shapes_from_sample(sample)
        collector.shapes.n_features = sample_shapes.n_features
        collector.shapes.embed_dim = sample_shapes.embed_dim or collector.shapes.embed_dim
        collector.shapes.n_hetero_time_features = sample_shapes.n_hetero_time_features
    
    # Process input window timestamps
    x_time = _safe_array(sample[SAMPLE_IDX_X_TIME])
    if x_time is not None:
        x_time_flat = x_time.flatten()
        seq_x = _safe_array(sample[SAMPLE_IDX_SEQ_X])
        hetero_x = _safe_array(sample[SAMPLE_IDX_HETERO_X])
        hetero_x_time = _safe_array(sample[SAMPLE_IDX_HETERO_X_TIME])
        
        for i, ts in enumerate(x_time_flat):
            # Extract value at position i (handling different array shapes)
            ts_val = seq_x[i] if seq_x is not None and i < len(seq_x) else None
            
            # Extract per-timestep embedding (pass full embedding including downtime)
            # The _register_timestamp_data function will handle N=1 vs N=2 based on
            # collector.num_news_items setting determined by downtime detection
            if hetero_x is not None:
                if hetero_x.ndim >= 2 and i < hetero_x.shape[0]:
                    # Per-timestep embedding, may be (num_items, embed_dim) or (embed_dim,)
                    # CRITICAL: Always copy to break reference to parent hetero_x array
                    # Array slicing creates views that keep references to the original large arrays
                    emb = hetero_x[i].copy()
                elif hetero_x.ndim == 1:
                    # Static embedding (same for all timesteps) - copy to break reference
                    emb = hetero_x.copy()
                else:
                    emb = None
            else:
                emb = None
            
            htf = hetero_x_time[i] if hetero_x_time is not None and hetero_x_time.ndim > 1 and i < len(hetero_x_time) else None
            
            _register_timestamp_data(collector, ts, ts_val, emb, htf)
    
    # Process output window timestamps (same pattern as input)
    y_time = _safe_array(sample[SAMPLE_IDX_Y_TIME])
    if y_time is not None:
        y_time_flat = y_time.flatten()
        seq_y = _safe_array(sample[SAMPLE_IDX_SEQ_Y])
        hetero_y = _safe_array(sample[SAMPLE_IDX_HETERO_Y])
        hetero_y_time = _safe_array(sample[SAMPLE_IDX_HETERO_Y_TIME])
        
        for i, ts in enumerate(y_time_flat):
            ts_val = seq_y[i] if seq_y is not None and i < len(seq_y) else None
            
            # Extract per-timestep embedding (pass full embedding including downtime)
            # The _register_timestamp_data function will handle N=1 vs N=2 based on
            # collector.num_news_items setting determined by downtime detection
            # CRITICAL: Create copies to break references to parent hetero_y array
            # Array slicing creates views that keep references to the original large arrays
            if hetero_y is not None:
                if hetero_y.ndim >= 2 and i < hetero_y.shape[0]:
                    # Per-timestep embedding, may be (num_items, embed_dim) or (embed_dim,)
                    emb = hetero_y[i].copy()
                elif hetero_y.ndim == 1:
                    # Static embedding (same for all timesteps) - copy to break reference
                    emb = hetero_y.copy()
                else:
                    emb = None
            else:
                emb = None
            
            htf = hetero_y_time[i] if hetero_y_time is not None and hetero_y_time.ndim > 1 and i < len(hetero_y_time) else None
            
            _register_timestamp_data(collector, ts, ts_val, emb, htf)


def _finalize_shared_tables(
    collector: SharedTableCollector
) -> Tuple[Dict[str, np.ndarray], dict]:
    """
    Convert collected lists to numpy arrays and build index mappings.
    
    WHY SEPARATE FINALIZATION:
    - Lists are efficient for building (O(1) append)
    - Arrays are efficient for storage and access
    - Convert once after all data collected
    
    MEMORY OPTIMIZATION:
    - Clear each collector list immediately after np.stack() conversion
    - This prevents double memory usage (list + array simultaneously)
    - gc.collect() at end to force memory reclamation
    
    Args:
        collector: Completed SharedTableCollector
        
    Returns:
        Tuple of (shared_tables dict, index_mappings dict)
    """
    shared_tables = {}
    
    _debug_memory("_finalize_shared_tables START")
    
    # Convert timestamp data lists to arrays
    # CRITICAL: Clear each list immediately after conversion to prevent memory doubling
    if collector.timestamps:
        shared_tables['timestamps'] = np.array(collector.timestamps, dtype=np.int64)
        collector.timestamps.clear()
        _debug_memory("After timestamps conversion")
    
    if collector.timeseries:
        shared_tables['timeseries'] = np.stack(collector.timeseries).astype(np.float32)
        collector.timeseries.clear()
        _debug_memory("After timeseries conversion")
    
    if collector.embeddings:
        shared_tables['embeddings'] = np.stack(collector.embeddings).astype(np.float32)
        collector.embeddings.clear()
        _debug_memory("After embeddings conversion")
    
    if collector.hetero_time:
        shared_tables['hetero_time'] = np.stack(collector.hetero_time).astype(np.float32)
        collector.hetero_time.clear()
        _debug_memory("After hetero_time conversion")
    
    # Convert entity data lists to arrays (filter out None values)
    valid_generals = [g for g in collector.entity_general if g is not None]
    if valid_generals:
        shared_tables['entity_general'] = np.stack(valid_generals).astype(np.float32)
    collector.entity_general.clear()
    
    valid_channels = [c for c in collector.entity_channel if c is not None]
    if valid_channels:
        shared_tables['entity_channel'] = np.stack(valid_channels).astype(np.float32)
    collector.entity_channel.clear()
    
    # Build index mappings (JSON-serializable)
    index_mappings = {
        'timestamp_to_idx': {str(k): v for k, v in collector.timestamp_to_idx.items()},
        'entity_to_idx': collector.entity_to_idx
    }
    
    # Force garbage collection to reclaim cleared list memory
    gc.collect()
    _debug_memory("_finalize_shared_tables END (after gc.collect)", force_gc=False)
    
    return shared_tables, index_mappings


# =============================================================================
# SECTION 6: INDEX ARRAY WRITING
# =============================================================================
# These functions write per-sample index arrays with checkpointing support.
# The core operation: convert timestamps to indices into shared tables.
# =============================================================================

# Array specifications for V2 indexed format
INDEXED_ARRAY_SPECS = {
    'sample_ids': {'dtype': 'U64'},        # Sample identifiers
    'entity_indices': {'dtype': 'int16'},  # Index into entity tables
    'x_indices': {'dtype': 'int32'},       # Indices into shared tables for input
    'y_indices': {'dtype': 'int32'},       # Indices into shared tables for output
    'x_time_features': {'dtype': 'float32'},  # Per-sample time features (not deduplicated)
    'y_time_features': {'dtype': 'float32'},  # Per-sample time features (not deduplicated)
}

# Shared table specifications
SHARED_TABLE_SPECS = {
    'timeseries': {'dtype': 'float32'},      # Time series values
    'timestamps': {'dtype': 'int64'},        # Timestamp keys
    'embeddings': {'dtype': 'float32'},      # Text embeddings
    'hetero_time': {'dtype': 'float32'},     # Hetero time features
    'entity_general': {'dtype': 'float32'},  # Entity general embeddings
    'entity_channel': {'dtype': 'float32'},  # Entity channel embeddings
}

# Legacy V1 format specifications (for backward compatibility)
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


def _create_index_arrays(
    split_dir: Path,
    shapes: Dict[str, tuple],
    resume_from_existing: bool
) -> Dict[str, np.memmap]:
    """
    Create memory-mapped arrays for index storage.
    
    WHY MEMORY-MAPPED:
    - Arrays can be larger than RAM
    - Data persists across crashes (with flush)
    - Efficient for sequential writes
    
    Args:
        split_dir: Directory for this split's arrays
        shapes: Dict of {array_name: shape_tuple}
        resume_from_existing: If True, open existing arrays in r+ mode
        
    Returns:
        Dict of {array_name: np.memmap}
    """
    arrays = {}
    
    for name, shape in shapes.items():
        dtype = INDEXED_ARRAY_SPECS.get(name, {}).get('dtype', 'float32')
        filepath = split_dir / f"{name}.npy"
        
        if filepath.exists() and resume_from_existing:
            # Open existing for resumption (read-write mode)
            arrays[name] = np.lib.format.open_memmap(
                str(filepath), dtype=dtype, mode='r+', shape=shape
            )
        else:
            # Create new (write mode)
            arrays[name] = np.lib.format.open_memmap(
                str(filepath), dtype=dtype, mode='w+', shape=shape
            )
    
    return arrays


def _write_sample_indices(
    arrays: Dict[str, np.memmap],
    write_idx: int,
    sample: tuple,
    entity_idx: int,
    timestamp_to_idx: Dict[int, int]
) -> None:
    """
    Write index data for a single sample to memory-mapped arrays.
    
    WHAT GETS WRITTEN:
    - sample_ids: The sample identifier string
    - entity_indices: Index into entity tables (same for all samples from entity)
    - x_indices: Convert input timestamps → indices into shared tables
    - y_indices: Convert output timestamps → indices into shared tables
    - x_time_features: Per-sample time features (NOT deduplicated - different per sample)
    - y_time_features: Per-sample time features (NOT deduplicated)
    
    WHY TIME FEATURES NOT DEDUPLICATED:
    - Time features encode position-in-sample information
    - Same timestamp may have different features depending on its position
    - Cost is low (~0.2 GB) compared to embeddings (~2 GB deduplicated)
    
    Args:
        arrays: Memory-mapped arrays to write to
        write_idx: Position in arrays to write
        sample: 13-element tuple from dataset
        entity_idx: Index of this sample's entity in entity tables
        timestamp_to_idx: Mapping from timestamp to shared table index
    """
    # Sample ID
    arrays['sample_ids'][write_idx] = str(sample[SAMPLE_IDX_SAMPLE_ID])
    
    # Entity index
    arrays['entity_indices'][write_idx] = entity_idx
    
    # Convert input timestamps to indices
    x_time = np.asarray(sample[SAMPLE_IDX_X_TIME]).flatten()
    x_indices = np.array(
        [timestamp_to_idx.get(int(ts), 0) for ts in x_time],
        dtype=np.int32
    )
    arrays['x_indices'][write_idx] = x_indices
    
    # Convert output timestamps to indices
    y_time = np.asarray(sample[SAMPLE_IDX_Y_TIME]).flatten()
    y_indices = np.array(
        [timestamp_to_idx.get(int(ts), 0) for ts in y_time],
        dtype=np.int32
    )
    arrays['y_indices'][write_idx] = y_indices
    
    # Per-sample time features (written directly, not indexed)
    x_tf = sample[SAMPLE_IDX_X_TIME_FEATURES]
    if 'x_time_features' in arrays and x_tf is not None:
        arrays['x_time_features'][write_idx] = np.asarray(x_tf)
    
    y_tf = sample[SAMPLE_IDX_Y_TIME_FEATURES]
    if 'y_time_features' in arrays and y_tf is not None:
        arrays['y_time_features'][write_idx] = np.asarray(y_tf)


def _flush_arrays(arrays: Dict[str, np.memmap]) -> None:
    """
    Flush all memory-mapped arrays to disk.
    
    WHY FLUSH:
    - Memory-mapped writes may be buffered
    - Flush ensures data is on disk (crash-safe checkpoint)
    - Call after completing each entity
    
    Args:
        arrays: Dict of memory-mapped arrays
    """
    for arr in arrays.values():
        if hasattr(arr, 'flush'):
            arr.flush()


def _save_checkpoint(
    progress_file: Path,
    config_hash: str,
    completed_entities: set,
    current_idx: int
) -> None:
    """
    Save checkpoint state after completing an entity.
    
    CHECKPOINT CONTENTS:
    - config_hash: For validation on resume (don't mix different configs)
    - completed_entities: Set of entity IDs fully processed
    - current_write_position: Where to continue writing
    - status: 'in_progress' or 'complete'
    
    Args:
        progress_file: Path to progress.json
        config_hash: Hash of current config
        completed_entities: Set of completed entity IDs
        current_idx: Current write position in arrays
    """
    _save_progress_atomic(progress_file, {
        'config_hash': config_hash,
        'completed_entities': list(completed_entities),
        'current_write_position': current_idx,
        'last_updated': datetime.now().isoformat(),
        'status': 'in_progress'
    })


# =============================================================================
# SECTION 7: CACHE GENERATOR
# =============================================================================
# Orchestrates the full cache generation pipeline with progress display.
# =============================================================================

class TensorCacheGenerator:
    """
    Generates pre-computed tensor cache for fast training data loading.
    
    GENERATION PIPELINE:
    1. Build shared tables (first pass - collect unique data)
    2. Save shared tables to shared/ directory
    3. Generate per-split index arrays (with checkpointing)
    4. Save metadata
    
    FEATURES:
    - INDEX-BASED DEDUPLICATION: ~100x disk space reduction
    - ENTITY-LEVEL CHECKPOINTING: Resume interrupted generation
    - NESTED PROGRESS BARS: Better visibility into generation progress
    """

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
            config: Cache config dict from build_cache_config()
            chunk_size: Number of samples to process per chunk (memory management)
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
        Generate cache for specified data splits.
        
        Args:
            flags: List of splits to generate (default: ['train', 'val', 'test'])
            
        Returns:
            Path to cache directory
        """
        if flags is None:
            flags = ['train', 'val', 'test']

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        shared_dir = self.cache_dir / 'shared'
        shared_dir.mkdir(exist_ok=True)

        # Step 1: Build shared tables
        _debug_memory("generate: START")
        logger.info("Building shared tables (collecting unique timestamps)...")
        shared_tables, index_mappings = self._build_shared_tables(flags)
        _debug_memory("generate: after _build_shared_tables")
        
        # Step 2: Save shared tables
        shared_shapes = self._save_shared_tables(shared_dir, shared_tables)
        self._save_index_mappings(shared_dir, index_mappings)
        
        # MEMORY OPTIMIZATION: Release shared_tables after saving to disk
        # They're now on disk and only needed for shape info (already extracted)
        del shared_tables
        gc.collect()
        _debug_memory("generate: after saving shared tables (gc.collect)", force_gc=False)

        # Step 3: Generate per-split index arrays
        shapes, entity_info = self._generate_all_splits(flags, index_mappings)
        _debug_memory("generate: after _generate_all_splits")

        # Step 4: Save metadata
        self._save_metadata(shapes, shared_shapes, entity_info)

        logger.info(f"Cache generated at: {self.cache_dir}")
        return self.cache_dir

    def _build_shared_tables(
        self,
        flags: List[str]
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        """
        Build shared tables by collecting unique data across all splits.
        
        ALGORITHM:
        1. Detect if training data has downtime (determines num_news_items)
        2. Create SharedTableCollector with appropriate num_news_items setting
        3. Iterate through all splits and entities
        4. For each sample, register unique timestamps and entity data
        5. Convert collected lists to numpy arrays
        
        DOWNTIME DETECTION:
        - If training data has non-zero downtime indicators: num_news_items=2
          (stores embedding + downtime indicator)
        - If training data has NO downtime: num_news_items=1
          (stores only embedding, 50% memory savings)
        - Rationale: If model doesn't see downtime during training, it won't
          learn to use it anyway, so storing it wastes memory
        
        Returns:
            Tuple of (shared_tables dict, index_mappings dict)
        """
        # Step 1: Detect downtime in training data to determine num_news_items
        _debug_memory("_build_shared_tables: START")
        has_downtime = _detect_downtime_in_training(self.data_provider)
        # Release train datasets used by downtime detection
        gc.collect()
        _debug_memory("_build_shared_tables: after downtime detection (gc.collect)", force_gc=False)
        
        num_news_items = 2 if has_downtime else 1
        
        # Store for later use in metadata
        self._num_news_items = num_news_items
        
        # Log the decision with clear explanation
        if has_downtime:
            logger.info(
                "[ info ] Downtime detected in training data - storing num_news_items=2 "
                "(embedding + downtime indicator). This matches the original dataloader behavior."
            )
            print(
                "[ info ] Downtime detected in training data - storing embeddings with "
                "downtime indicators (N=2). The model will learn to use downtime information."
            )
        else:
            logger.info(
                "[ info ] No downtime in training data - storing num_news_items=1 "
                "(embedding only). This saves 50% embedding memory."
            )
            print(
                "[ info ] No downtime in training data - storing embeddings without "
                "downtime indicators (N=1). Even if val/test have downtime, the model "
                "won't have learned to use it, so omitting saves memory."
            )
        
        collector = SharedTableCollector()
        collector.num_news_items = num_news_items
        
        # Count total samples for progress tracking
        # MEMORY OPTIMIZATION: Release datasets after counting to avoid holding
        # multiple dataset instances in memory simultaneously
        _debug_memory("_build_shared_tables: before counting")
        total_samples = 0
        for flag in flags:
            datasets = self.data_provider.get_datasets(flag)
            if datasets:
                total_samples += sum(len(ds) for ds in datasets.values())
            # Release this split's datasets before loading next
            del datasets
        gc.collect()
        _debug_memory("_build_shared_tables: after counting (gc.collect)", force_gc=False)
        
        # Process all splits to collect unique data
        if RICH_AVAILABLE and total_samples > 0:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                disable=not RICH_AVAILABLE
            ) as progress:
                task = progress.add_task(
                    "Processing samples",
                    total=total_samples
                )
                
                for flag in flags:
                    _debug_memory(f"_build_shared_tables: before loading {flag}")
                    datasets = self.data_provider.get_datasets(flag)
                    if not datasets:
                        continue
                    
                    _debug_memory(f"_build_shared_tables: after loading {flag}")
                    
                    for entity_id, dataset in datasets.items():
                        if len(dataset) == 0:
                            continue
                        
                        # Register entity (extracts static embeddings from first sample)
                        first_sample = dataset[0]
                        _register_entity_data(
                            collector,
                            entity_id,
                            first_sample[SAMPLE_IDX_HETERO_GENERAL],
                            first_sample[SAMPLE_IDX_HETERO_CHANNEL]
                        )
                        
                        # Process all samples to register unique timestamps
                        # MEMORY OPTIMIZATION: Periodic GC during large loops to prevent unbounded growth
                        samples_per_gc = 10000  # Run GC every 10K samples
                        for sample_idx in range(len(dataset)):
                            sample = dataset[sample_idx]
                            _process_sample_for_collection(collector, sample)
                            progress.update(task, advance=1)
                            
                            # Periodic memory management during processing
                            if (sample_idx + 1) % samples_per_gc == 0:
                                # Force garbage collection periodically to release sample references
                                gc.collect()
                                _debug_memory(
                                    f"_build_shared_tables: processing {entity_id}, "
                                    f"sample {sample_idx+1}/{len(dataset)}",
                                    force_gc=False,
                                    collector=collector
                                )
                                
                                # Always warn if memory usage is high (even without debug mode)
                                process = psutil.Process(os.getpid())
                                rss_gb = process.memory_info().rss / (1024 ** 3)
                                if rss_gb > 50:  # Warn if using more than 50GB
                                    logger.warning(
                                        f"High memory usage detected: {rss_gb:.1f}GB RSS. "
                                        f"Collector has {len(collector.timestamps)} unique timestamps. "
                                        f"Enable TENSOR_CACHE_DEBUG_MEMORY=1 for detailed memory tracking."
                                    )
                    
                    # CRITICAL: Release this split's datasets before loading next split
                    # This prevents multiple dataset instances from accumulating in memory
                    del datasets
                    gc.collect()
                    _debug_memory(f"_build_shared_tables: after processing {flag} (gc.collect)", force_gc=False)
        else:
            # Fallback without progress bar
            for flag in flags:
                _debug_memory(f"_build_shared_tables: before loading {flag}")
                datasets = self.data_provider.get_datasets(flag)
                if not datasets:
                    continue
                
                _debug_memory(f"_build_shared_tables: after loading {flag}")
                
                for entity_id, dataset in datasets.items():
                    if len(dataset) == 0:
                        continue
                    
                    # Register entity (extracts static embeddings from first sample)
                    first_sample = dataset[0]
                    _register_entity_data(
                        collector,
                        entity_id,
                        first_sample[SAMPLE_IDX_HETERO_GENERAL],
                        first_sample[SAMPLE_IDX_HETERO_CHANNEL]
                    )
                    
                    # Process all samples to register unique timestamps
                    # MEMORY OPTIMIZATION: Periodic GC during large loops to prevent unbounded growth
                    samples_per_gc = 10000  # Run GC every 10K samples
                    for sample_idx in range(len(dataset)):
                        sample = dataset[sample_idx]
                        _process_sample_for_collection(collector, sample)
                        
                        # Periodic memory management during processing
                        if (sample_idx + 1) % samples_per_gc == 0:
                            # Force garbage collection periodically to release sample references
                            gc.collect()
                            _debug_memory(
                                f"_build_shared_tables: processing {entity_id}, "
                                f"sample {sample_idx+1}/{len(dataset)}",
                                force_gc=False,
                                collector=collector
                            )
                            
                            # Always warn if memory usage is high (even without debug mode)
                            process = psutil.Process(os.getpid())
                            rss_gb = process.memory_info().rss / (1024 ** 3)
                            if rss_gb > 50:  # Warn if using more than 50GB
                                logger.warning(
                                    f"High memory usage detected: {rss_gb:.1f}GB RSS. "
                                    f"Collector has {len(collector.timestamps)} unique timestamps. "
                                    f"Enable TENSOR_CACHE_DEBUG_MEMORY=1 for detailed memory tracking."
                                )
                
                # CRITICAL: Release this split's datasets before loading next split
                # This prevents multiple dataset instances from accumulating in memory
                del datasets
                gc.collect()
                _debug_memory(f"_build_shared_tables: after processing {flag} (gc.collect)", force_gc=False)
        
        # Capture counts before finalization (which clears the lists)
        n_unique_timestamps = len(collector.timestamps)
        n_entities = len(collector.entity_to_idx)
        
        # Finalize: convert lists to arrays
        logger.info("Converting collected data to arrays...")
        _debug_memory("_build_shared_tables: before finalization")
        shared_tables, index_mappings = _finalize_shared_tables(collector)
        _debug_memory("_build_shared_tables: after finalization")
        
        logger.info(
            f"Built shared tables: {n_unique_timestamps} unique timestamps, "
            f"{n_entities} entities"
        )
        
        return shared_tables, index_mappings

    def _save_shared_tables(
        self,
        shared_dir: Path,
        shared_tables: Dict[str, np.ndarray]
    ) -> Dict[str, List[int]]:
        """
        Save shared tables to disk and return shapes.
        
        Args:
            shared_dir: Directory for shared tables
            shared_tables: Dict of arrays to save
            
        Returns:
            Dict of {table_name: shape_list}
        """
        shared_shapes = {}
        
        for name, data in shared_tables.items():
            if data is not None and len(data) > 0:
                filepath = shared_dir / f"{name}.npy"
                np.save(filepath, data)
                shared_shapes[name] = list(data.shape)
                logger.info(f"Saved shared table {name}: {data.shape}")
        
        return shared_shapes

    def _save_index_mappings(self, shared_dir: Path, index_mappings: dict) -> None:
        """Save index mappings to JSON file."""
        with open(shared_dir / 'index_mappings.json', 'w') as f:
            json.dump(index_mappings, f, indent=2)

    def _generate_all_splits(
        self,
        flags: List[str],
        index_mappings: dict
    ) -> Tuple[dict, dict]:
        """
        Generate index arrays for all requested splits.
        
        MEMORY OPTIMIZATION:
        - Each split is processed sequentially
        - Memory is released between splits via gc.collect() in _generate_split
        - This prevents accumulation of multiple dataset instances
        
        Args:
            flags: List of splits to generate
            index_mappings: Timestamp and entity mappings from shared table building
            
        Returns:
            Tuple of (shapes dict, entity_info dict)
        """
        shapes = {}
        entity_info = {'entity_ids': [], 'samples_per_entity': {}}
        
        _debug_memory("_generate_all_splits: START")
        
        for flag in flags:
            logger.info(f"Generating index arrays for {flag} split...")
            split_shapes, split_entity_info = self._generate_split(flag, index_mappings)
            shapes[flag] = split_shapes
            
            # Merge entity info
            for eid in split_entity_info.get('entity_ids', []):
                if eid not in entity_info['entity_ids']:
                    entity_info['entity_ids'].append(eid)
            entity_info['samples_per_entity'].update(
                split_entity_info.get('samples_per_entity', {})
            )
            
            # gc.collect() is called at the end of _generate_split
            # No need to call again here
        
        _debug_memory("_generate_all_splits: END")
        return shapes, entity_info

    def _generate_split(
        self,
        flag: str,
        index_mappings: dict
    ) -> Tuple[dict, dict]:
        """
        Generate index arrays for a single split with checkpointing.
        
        CHECKPOINTING FLOW:
        1. Check for existing progress file (may be resuming)
        2. Create or open arrays based on resume state
        3. Process entities, skipping completed ones
        4. Checkpoint after each entity (flush + save progress)
        5. Delete progress file on successful completion
        
        Args:
            flag: Split name ('train', 'val', 'test')
            index_mappings: Timestamp and entity index mappings
            
        Returns:
            Tuple of (array_shapes dict, entity_info dict)
        """
        split_dir = self.cache_dir / flag
        split_dir.mkdir(exist_ok=True)
        progress_file = split_dir / 'progress.json'
        
        _debug_memory(f"_generate_split({flag}): START")
        
        # Get datasets
        datasets = self.data_provider.get_datasets(flag)
        if not datasets:
            logger.warning(f"No datasets found for {flag} split")
            return {}, {}
        
        _debug_memory(f"_generate_split({flag}): after loading datasets")
        
        # Calculate totals and get shapes from first sample
        total_samples = sum(len(ds) for ds in datasets.values())
        logger.info(f"Total samples for {flag}: {total_samples:,}")
        
        if total_samples == 0:
            return {}, {}
        
        # Infer shapes from first sample
        first_sample = next(iter(datasets.values()))[0]
        shapes = infer_shapes_from_sample(first_sample)
        
        # Check for resumption
        progress = _load_progress(progress_file)
        completed_entities = set()
        start_idx = 0
        
        if progress and progress.get('config_hash') == self.config_hash:
            completed_entities = set(progress.get('completed_entities', []))
            start_idx = progress.get('current_write_position', 0)
            if completed_entities:
                logger.info(
                    f"Resuming from checkpoint: {len(completed_entities)}/{len(datasets)} "
                    f"entities done, position {start_idx}"
                )
        
        # Determine array shapes
        array_shapes = self._compute_array_shapes(total_samples, shapes)
        
        # Create or open arrays
        arrays = _create_index_arrays(split_dir, array_shapes, bool(completed_entities))
        
        # Convert mappings for efficient lookup
        timestamp_to_idx = {int(k): v for k, v in index_mappings['timestamp_to_idx'].items()}
        entity_to_idx = index_mappings['entity_to_idx']
        
        # Process datasets
        entity_info = {'entity_ids': [], 'samples_per_entity': {}}
        
        if self.console is not None and RICH_AVAILABLE and self.verbose:
            self._process_with_rich_progress(
                flag, datasets, arrays, entity_info, total_samples,
                timestamp_to_idx, entity_to_idx, completed_entities,
                start_idx, progress_file
            )
        else:
            self._process_with_tqdm(
                flag, datasets, arrays, entity_info,
                timestamp_to_idx, entity_to_idx, completed_entities,
                start_idx, progress_file
            )
        
        # Remove progress file on success
        if progress_file.exists():
            progress_file.unlink()
        
        # Capture return values before cleanup
        result_shapes = {name: list(arr.shape) for name, arr in arrays.items()}
        
        # MEMORY OPTIMIZATION: Release datasets and arrays after processing
        # Arrays are memory-mapped so this just releases the Python wrappers
        del datasets
        del arrays
        gc.collect()
        _debug_memory(f"_generate_split({flag}): END (gc.collect)", force_gc=False)
        
        return result_shapes, entity_info

    def _compute_array_shapes(
        self,
        total_samples: int,
        shapes: InferredShapes
    ) -> Dict[str, tuple]:
        """
        Compute array shapes for index arrays.
        
        Args:
            total_samples: Total number of samples in split
            shapes: Inferred shapes from sample
            
        Returns:
            Dict of {array_name: shape_tuple}
        """
        array_shapes = {
            'sample_ids': (total_samples,),
            'entity_indices': (total_samples,),
            'x_indices': (total_samples, shapes.input_len),
            'y_indices': (total_samples, shapes.output_len),
        }
        
        if shapes.n_x_time_features and shapes.n_x_time_features > 0:
            array_shapes['x_time_features'] = (
                total_samples, shapes.input_len, shapes.n_x_time_features
            )
        
        if shapes.n_y_time_features and shapes.n_y_time_features > 0:
            array_shapes['y_time_features'] = (
                total_samples, shapes.output_len, shapes.n_y_time_features
            )
        
        return array_shapes

    def _process_with_rich_progress(
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
    ) -> None:
        """
        Process datasets with Rich progress bars and checkpointing.
        
        PROGRESS DISPLAY:
        - Level 1: Entity progress (e.g., 45/87 entities)
        - Level 2: Current entity samples (e.g., 15000/25000)
        - Level 3: Total samples across all entities
        
        This provides much better visibility than a single bar,
        especially for large entities where processing appears "stuck".
        """
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
            
            # Create progress tasks
            entity_task = progress.add_task(f"[cyan]Entities ({flag})", total=len(datasets))
            samples_task = progress.add_task("[dim]  └─ waiting...[/dim]", total=100)
            total_task = progress.add_task("[green]Total samples", total=total_samples)
            
            # Update for already-completed work
            progress.update(total_task, completed=start_idx)
            
            # Process each entity
            for entity_id, dataset in datasets.items():
                current_idx = self._process_entity(
                    entity_id, dataset, arrays, entity_info,
                    timestamp_to_idx, entity_to_idx, completed_entities,
                    current_idx, progress_file,
                    progress, entity_task, samples_task, total_task
                )
            
            progress.update(samples_task, description="[dim]  └─ complete[/dim]", visible=False)

    def _process_with_tqdm(
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
    ) -> None:
        """
        Process datasets with tqdm progress bar (fallback when Rich unavailable).
        """
        current_idx = start_idx
        
        for entity_id, dataset in tqdm(
            datasets.items(),
            desc=f"Entities ({flag})",
            disable=not self.verbose
        ):
            current_idx = self._process_entity(
                entity_id, dataset, arrays, entity_info,
                timestamp_to_idx, entity_to_idx, completed_entities,
                current_idx, progress_file,
                progress=None, entity_task=None, samples_task=None, total_task=None
            )

    def _process_entity(
        self,
        entity_id: str,
        dataset: Any,
        arrays: Dict[str, np.memmap],
        entity_info: dict,
        timestamp_to_idx: dict,
        entity_to_idx: dict,
        completed_entities: set,
        current_idx: int,
        progress_file: Path,
        progress: Optional[Any] = None,
        entity_task: Optional[int] = None,
        samples_task: Optional[int] = None,
        total_task: Optional[int] = None
    ) -> int:
        """
        Process a single entity's samples with checkpointing.
        
        ENTITY PROCESSING FLOW:
        1. Skip if already completed (resumption)
        2. Update progress display for this entity
        3. Process samples in chunks (memory efficiency)
        4. Checkpoint after entity completes (flush + save progress)
        
        Args:
            entity_id: Entity identifier
            dataset: Dataset for this entity
            arrays: Memory-mapped arrays to write to
            entity_info: Dict to update with entity metadata
            timestamp_to_idx: Timestamp to index mapping
            entity_to_idx: Entity to index mapping
            completed_entities: Set of already-completed entities (modified in place)
            current_idx: Current write position
            progress_file: Path to progress.json
            progress: Optional Rich Progress instance
            entity_task, samples_task, total_task: Optional Rich task IDs
            
        Returns:
            Updated write position after processing entity
        """
        entity_samples = len(dataset)
        
        # Skip completed entities
        if entity_id in completed_entities:
            if progress is not None:
                progress.update(entity_task, advance=1)
            return current_idx
        
        # Skip empty entities
        if entity_samples == 0:
            if progress is not None:
                progress.update(entity_task, advance=1)
            return current_idx
        
        # Track entity info
        entity_info['entity_ids'].append(entity_id)
        entity_info['samples_per_entity'][entity_id] = entity_samples
        
        # Update progress display
        if progress is not None:
            progress.update(
                samples_task,
                description=f"[yellow]  └─ {entity_id}[/yellow]",
                completed=0,
                total=entity_samples
            )
        
        # Get entity index
        entity_idx = entity_to_idx.get(entity_id, 0)
        
        # Process samples in chunks
        for chunk_start in range(0, entity_samples, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, entity_samples)
            chunk_size = chunk_end - chunk_start
            
            # Write each sample in chunk
            for i in range(chunk_start, chunk_end):
                sample = dataset[i]
                write_idx = current_idx + i
                _write_sample_indices(
                    arrays, write_idx, sample, entity_idx, timestamp_to_idx
                )
            
            # Update progress
            if progress is not None:
                progress.update(samples_task, advance=chunk_size)
                progress.update(total_task, advance=chunk_size)
        
        # Checkpoint after entity completes
        current_idx += entity_samples
        _flush_arrays(arrays)
        
        completed_entities.add(entity_id)
        _save_checkpoint(progress_file, self.config_hash, completed_entities, current_idx)
        
        if progress is not None:
            progress.update(entity_task, advance=1)
        
        return current_idx

    def _save_metadata(
        self,
        shapes: dict,
        shared_shapes: dict,
        entity_info: dict
    ) -> None:
        """
        Save cache metadata to metadata.json.
        
        Includes num_news_items which determines embedding shape:
        - N=1: embeddings stored as (D,) per timestamp
        - N=2: embeddings stored as (2, D) per timestamp (embedding + downtime)
        """
        # Get num_news_items from the build process (set in _build_shared_tables)
        num_news_items = getattr(self, '_num_news_items', 1)
        
        metadata = TensorCacheMetadata(
            config_hash=self.config_hash,
            data_config=self.config,
            shapes=shapes,
            dtypes={name: spec['dtype'] for name, spec in INDEXED_ARRAY_SPECS.items()},
            cache_format=TensorCacheMetadata.FORMAT_INDEXED,
            shared_shapes=shared_shapes,
            entity_info=entity_info,
            num_news_items=num_news_items
        )
        metadata.save(self.cache_dir / 'metadata.json')


# =============================================================================
# SECTION 8: CACHE DATASET
# =============================================================================
# Ultra-fast dataset that loads from pre-computed tensor cache.
# Supports both V1 (direct) and V2 (indexed) formats.
# =============================================================================

class TensorCacheDataset(Dataset):
    """
    Ultra-fast dataset that loads from pre-computed tensor cache.
    
    V2 INDEXED FORMAT:
    - Shared tables loaded to RAM at init (~2-4 GB after deduplication)
    - __getitem__ performs index lookups into shared tables
    - Lookup time: ~2-5ms per batch (negligible vs GPU time)
    
    V1 DIRECT FORMAT (legacy):
    - Arrays memory-mapped directly
    - __getitem__ is direct array indexing
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
            preload_to_ram: If True, load all data to RAM (V1 format only)
        """
        # BEGIN DEBUG
        _debug_memory(f"TensorCacheDataset.__init__ START ({flag})")
        # END DEBUG
        
        self.cache_dir = Path(cache_dir)
        self.split_dir = self.cache_dir / flag
        self.flag = flag
        self.preload_to_ram = preload_to_ram

        # Load metadata and detect format
        self.metadata = TensorCacheMetadata.load(self.cache_dir / 'metadata.json')
        
        # Initialize based on format
        if self.metadata.is_indexed:
            self._init_indexed_format()
        else:
            self._init_legacy_format()
        
        # BEGIN DEBUG
        _debug_memory(f"TensorCacheDataset.__init__ END ({flag})")
        # END DEBUG
        
        logger.info(
            f"TensorCacheDataset initialized: {flag}, "
            f"{self.n_samples:,} samples, format={self.metadata.cache_format}, "
            f"hetero_stride={self.hetero_stride}"
        )

    def _init_indexed_format(self) -> None:
        """Initialize for V2 indexed format."""
        # BEGIN DEBUG
        _debug_memory(f"_init_indexed_format START ({self.flag})")
        # END DEBUG
        
        # Load shared tables into RAM (small after deduplication)
        shared_dir = self.cache_dir / 'shared'
        self.shared = {}
        
        for name in SHARED_TABLE_SPECS.keys():
            filepath = shared_dir / f"{name}.npy"
            if filepath.exists():
                # BEGIN DEBUG
                file_size_mb = filepath.stat().st_size / (1024 ** 2)
                print(f"[DEBUG] Loading shared/{name}.npy (file size: {file_size_mb:.1f}MB)")
                # END DEBUG
                # Load to RAM for fast lookup
                self.shared[name] = np.load(filepath, mmap_mode=None)
                # BEGIN DEBUG
                arr = self.shared[name]
                arr_size_mb = arr.nbytes / (1024 ** 2)
                print(f"[DEBUG]   -> shape={arr.shape}, dtype={arr.dtype}, memory={arr_size_mb:.1f}MB")
                _debug_memory(f"  After loading {name}")
                # END DEBUG
        
        # BEGIN DEBUG
        _debug_memory(f"After loading all shared tables ({self.flag})")
        # END DEBUG
        
        # Load per-split index arrays (memory-mapped for efficiency)
        self.arrays = {}
        for name in INDEXED_ARRAY_SPECS.keys():
            filepath = self.split_dir / f"{name}.npy"
            if filepath.exists():
                self.arrays[name] = np.load(filepath, mmap_mode='r', allow_pickle=True)
                # BEGIN DEBUG
                arr = self.arrays[name]
                print(f"[DEBUG] Loaded {self.flag}/{name}.npy: shape={arr.shape}, dtype={arr.dtype}")
                # END DEBUG
        
        # BEGIN DEBUG
        _debug_memory(f"After loading per-split arrays ({self.flag})")
        # END DEBUG
        
        # Determine sample count
        if 'x_indices' in self.arrays:
            self.n_samples = self.arrays['x_indices'].shape[0]
        elif 'sample_ids' in self.arrays:
            self.n_samples = self.arrays['sample_ids'].shape[0]
        else:
            self.n_samples = 0
        
        # Extract hetero_stride from data_config for embedding striding
        # This matches the stride applied by the normal dataloader (data_loader.py)
        # which reduces embedding timesteps from input_len to ceil(input_len/stride)
        self.hetero_stride = self.metadata.data_config.get('hetero_stride', 1)

    def _init_legacy_format(self) -> None:
        """Initialize for V1 direct format (backward compatibility)."""
        self.shared = None
        self.arrays = {}
        
        for name in LEGACY_ARRAY_SPECS.keys():
            filepath = self.split_dir / f"{name}.npy"
            if filepath.exists():
                mmap_mode = None if self.preload_to_ram else 'r'
                self.arrays[name] = np.load(filepath, mmap_mode=mmap_mode, allow_pickle=True)
        
        first_array = next(iter(self.arrays.values()))
        self.n_samples = first_array.shape[0]
        
        # Legacy format already stores pre-strided data, but set stride for consistency
        self.hetero_stride = self.metadata.data_config.get('hetero_stride', 1)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> tuple:
        """
        Get sample by index.
        
        Dispatches to format-specific implementation.
        """
        # BEGIN DEBUG
        global _DEBUG_GETITEM_COUNT
        if _DEBUG_GETITEM_COUNT < _DEBUG_GETITEM_LIMIT:
            _debug_memory(f"__getitem__({index}) START")
        # END DEBUG
        
        if self.metadata.is_indexed:
            result = self._getitem_indexed(index)
        else:
            result = self._getitem_legacy(index)
        
        # BEGIN DEBUG
        if _DEBUG_GETITEM_COUNT < _DEBUG_GETITEM_LIMIT:
            _debug_memory(f"__getitem__({index}) END")
            _DEBUG_GETITEM_COUNT += 1
        # END DEBUG
        
        return result

    def _getitem_indexed(self, index: int) -> tuple:
        """
        Get sample using V2 indexed format.
        
        LOOKUP PROCESS:
        1. Get x_indices and y_indices for this sample
        2. Use indices to look up data from shared tables
        3. Apply hetero_stride to embedding indices (matches normal dataloader behavior)
        4. Handle num_news_items dimension in embeddings (N=1 or N=2)
        5. Return reconstructed sample tuple
        
        EMBEDDING SHAPE HANDLING:
        The embeddings are stored differently based on metadata.num_news_items:
        
        - N=1 (no downtime in training): stored as (L, D), returned as (L, 1, D)
          We add the N dimension with expand_dims
          
        - N=2 (downtime in training): stored as (L, 2, D), returned as (L, 2, D)
          Already has the N dimension, no expansion needed
        
        The original dataloader produces (L, 2, D) where:
        - [:, 0, :] = text embedding
        - [:, 1, :] = downtime indicator (non-zero if sensor was down)
        
        The text_encoder does cross-attention where:
        - Query: channel_description [B, L, C, D] - C channels
        - Key/Value: news_emb [B, L, N, D] - N news items per timestep
        
        Each channel attends to all N news items. With N=2, the model can learn
        to use downtime information. With N=1, we save 50% memory and the model
        only sees text embeddings (appropriate when no downtime in training).
        
        See docs/tensor_cache_embedding_shapes.md for detailed explanation.
        
        HETERO STRIDE:
        - The normal dataloader (data_loader.py) applies hetero_stride to reduce
          embedding timesteps: x_hetero = self.full_hetero[s_begin:s_end:hetero_stride]
        - FiLMGenerator expects this strided length, not full input_len
        - We replicate this behavior by striding the embedding indices
        
        This is the key to deduplication efficiency:
        - Indices are small (int32)
        - Shared tables loaded once at init
        - Numpy advanced indexing is highly optimized
        """
        # Get indices for this sample
        x_idx = self.arrays['x_indices'][index]        # (input_len,)
        y_idx = self.arrays['y_indices'][index]        # (output_len,)
        entity_idx = self.arrays['entity_indices'][index]  # scalar
        
        # Look up time series from shared table
        seq_x = self.shared['timeseries'][x_idx] if 'timeseries' in self.shared else None
        seq_y = self.shared['timeseries'][y_idx] if 'timeseries' in self.shared else None
        
        # Look up timestamps from shared table
        x_time = self.shared['timestamps'][x_idx] if 'timestamps' in self.shared else None
        y_time = self.shared['timestamps'][y_idx] if 'timestamps' in self.shared else None
        
        # Apply hetero_stride to embedding indices
        # This matches the striding done in data_loader.py:
        #   x_hetero = self.full_hetero[s_begin:s_end:self.hetero_stride]
        # Models like FiLMGenerator expect strided embeddings, not full resolution
        x_hetero_idx = x_idx[::self.hetero_stride]
        y_hetero_idx = y_idx[::self.hetero_stride]
        
        # Look up embeddings from shared table using STRIDED indices
        # Shape depends on num_news_items in metadata:
        # - N=1: stored as (L, D), needs expansion to (L, 1, D)
        # - N=2: stored as (L, 2, D), already has N dimension
        hetero_x = self.shared['embeddings'][x_hetero_idx] if 'embeddings' in self.shared else None
        hetero_y = self.shared['embeddings'][y_hetero_idx] if 'embeddings' in self.shared else None
        
        # Get num_news_items from metadata (default to 1 for old caches)
        num_news_items = getattr(self.metadata, 'num_news_items', 1)
        
        # Handle embedding shape based on num_news_items
        if hetero_x is not None:
            # BEGIN DEBUG
            if _DEBUG_GETITEM_COUNT < _DEBUG_GETITEM_LIMIT:
                print(f"[DEBUG]   hetero_x before expand: shape={hetero_x.shape}")
            # END DEBUG
            
            if num_news_items == 1:
                # N=1: stored as (L, D) -> expand to (L, 1, D)
                hetero_x = np.expand_dims(hetero_x, axis=1)
            # else: N=2, already stored as (L, 2, D), no expansion needed
            
            # BEGIN DEBUG
            if _DEBUG_GETITEM_COUNT < _DEBUG_GETITEM_LIMIT:
                hetero_x_mb = hetero_x.nbytes / (1024 ** 2)
                print(f"[DEBUG]   hetero_x after expand: shape={hetero_x.shape}, size={hetero_x_mb:.2f}MB")
            # END DEBUG
        
        if hetero_y is not None:
            if num_news_items == 1:
                # N=1: stored as (L, D) -> expand to (L, 1, D)
                hetero_y = np.expand_dims(hetero_y, axis=1)
            # else: N=2, already stored as (L, 2, D), no expansion needed
            
            # BEGIN DEBUG
            if _DEBUG_GETITEM_COUNT < _DEBUG_GETITEM_LIMIT:
                hetero_y_mb = hetero_y.nbytes / (1024 ** 2)
                print(f"[DEBUG]   hetero_y after expand: shape={hetero_y.shape}, size={hetero_y_mb:.2f}MB")
            # END DEBUG
        
        # Look up hetero time features from shared table (also strided)
        hetero_x_time = self.shared['hetero_time'][x_hetero_idx] if 'hetero_time' in self.shared else None
        hetero_y_time = self.shared['hetero_time'][y_hetero_idx] if 'hetero_time' in self.shared else None
        
        # Look up entity-level static embeddings
        hetero_general = self.shared['entity_general'][entity_idx] if 'entity_general' in self.shared else None
        hetero_channel = self.shared['entity_channel'][entity_idx] if 'entity_channel' in self.shared else None
        
        # Get per-sample data (not deduplicated)
        sample_id = self.arrays['sample_ids'][index] if 'sample_ids' in self.arrays else ''
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
        def get(name: str, default=None):
            if name in self.arrays:
                return self.arrays[name][index]
            return default if default is not None else np.zeros((1,), dtype=np.float32)
        
        return (
            get('sample_ids', default=''),
            get('seq_x'),
            get('seq_y'),
            get('x_time'),
            get('y_time'),
            get('hetero_x'),
            get('hetero_y'),
            get('hetero_x_time'),
            get('hetero_y_time'),
            get('hetero_general'),
            get('hetero_channel'),
            get('x_time_features'),
            get('y_time_features'),
        )

    def get_scaler_params(self) -> Optional[dict]:
        """Get scaler parameters for inverse transform."""
        return self.metadata.scaler_params


# =============================================================================
# SECTION 9: COLLATE FUNCTION FOR NONE HANDLING
# =============================================================================
# Custom collate function that handles None values in batch elements.
# PyTorch's default_collate cannot handle None - it throws TypeError.
# This is needed because tensor cache may not have all optional fields.
# =============================================================================

_DEBUG_COLLATE_COUNT = 0
_DEBUG_COLLATE_LIMIT = 3  # Only print first N collate calls

def tensor_cache_collate_fn(batch: List[tuple]) -> tuple:
    """
    Custom collate function that handles None values in batch elements.
    
    WHY THIS EXISTS:
    - TensorCacheDataset._getitem_indexed() returns None for missing arrays
    - PyTorch's default_collate throws TypeError on None values
    - Some fields (x_time_features, y_time_features) are optional
    - Models should handle None gracefully for optional fields
    
    BEHAVIOR:
    - For non-None values: stack into batched tensor (like default_collate)
    - For None values: return None (entire batch element is None)
    - For string values: return as list (sample_ids)
    
    Args:
        batch: List of sample tuples from TensorCacheDataset
        
    Returns:
        Tuple of batched tensors/arrays, with None preserved for missing fields
    """
    # BEGIN DEBUG
    global _DEBUG_COLLATE_COUNT
    debug_this_call = _DEBUG_COLLATE_COUNT < _DEBUG_COLLATE_LIMIT
    if debug_this_call:
        _debug_memory(f"tensor_cache_collate_fn START (batch_size={len(batch)})")
    # END DEBUG
    
    if not batch:
        return tuple()
    
    # Get number of elements in each sample tuple
    n_elements = len(batch[0])
    
    # Collate each element position across all samples
    collated = []
    for elem_idx in range(n_elements):
        # Extract this element from all samples
        elements = [sample[elem_idx] for sample in batch]
        
        # Check if ALL elements are None
        all_none = all(elem is None for elem in elements)
        if all_none:
            collated.append(None)
            continue
        
        # Check if ANY element is None (mixed None and non-None)
        any_none = any(elem is None for elem in elements)
        if any_none:
            # For mixed case, we could either skip Nones or fail
            # For robustness, filter out Nones and log warning
            # But this changes batch size - safer to replace with zeros
            # For now, return None for entire field if any is None
            # (This shouldn't happen in practice - all samples should be consistent)
            logger.warning(
                f"Mixed None/non-None values at element {elem_idx} in batch. "
                f"Returning None for entire field."
            )
            collated.append(None)
            continue
        
        # Check element type of first non-None element
        first_elem = elements[0]
        
        # Handle string/sample_id (element 0)
        if isinstance(first_elem, (str, np.str_)):
            # Return as list of strings (can't stack strings as tensor)
            collated.append([str(e) for e in elements])
            continue
        
        # Handle numpy arrays - convert to tensor and stack
        if isinstance(first_elem, np.ndarray):
            try:
                # Stack numpy arrays and convert to tensor
                stacked = np.stack(elements)
                collated.append(torch.from_numpy(stacked))
            except Exception as e:
                logger.warning(f"Failed to stack element {elem_idx}: {e}")
                collated.append(None)
            continue
        
        # Handle torch tensors - stack directly
        if isinstance(first_elem, torch.Tensor):
            try:
                collated.append(torch.stack(elements))
            except Exception as e:
                logger.warning(f"Failed to stack tensor element {elem_idx}: {e}")
                collated.append(None)
            continue
        
        # Handle scalars (int, float)
        if isinstance(first_elem, (int, float, np.integer, np.floating)):
            try:
                collated.append(torch.tensor(elements))
            except Exception as e:
                logger.warning(f"Failed to convert scalars at element {elem_idx}: {e}")
                collated.append(None)
            continue
        
        # Unknown type - log warning and return as list
        logger.warning(
            f"Unknown element type at index {elem_idx}: {type(first_elem)}. "
            f"Returning as list."
        )
        collated.append(elements)
    
    # BEGIN DEBUG
    if debug_this_call:
        _debug_memory("tensor_cache_collate_fn END")
        _DEBUG_COLLATE_COUNT += 1
    # END DEBUG
    
    return tuple(collated)


# =============================================================================
# SECTION 10: VALIDATION AND UTILITIES
# =============================================================================
# Functions for validating cache integrity and creating dataloaders.
# =============================================================================

def validate_cache(cache_dir: Union[str, Path], config: dict) -> Tuple[bool, str]:
    """
    Validate that a cache exists and matches the given config.
    
    VALIDATION CHECKS:
    1. Cache directory exists
    2. metadata.json exists and is valid
    3. No incomplete progress files (generation was interrupted)
    4. Config hash matches (cache is not stale)
    5. Required files exist for detected format
    
    Args:
        cache_dir: Path to cache directory
        config: Current config to validate against
        
    Returns:
        Tuple of (is_valid, message)
    """
    cache_dir = Path(cache_dir)

    # Check directory exists
    if not cache_dir.exists():
        return False, f"Cache directory does not exist: {cache_dir}"

    # Check metadata exists
    metadata_path = cache_dir / 'metadata.json'
    if not metadata_path.exists():
        return False, f"Metadata file not found: {metadata_path}"

    # Load metadata
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

    # Check format-specific requirements
    if metadata.is_indexed:
        shared_dir = cache_dir / 'shared'
        if not shared_dir.exists():
            return False, "Missing shared/ directory for indexed format"
    else:
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
    preload_to_ram: bool = False,
    collate_fn: Optional[callable] = None
) -> torch.utils.data.DataLoader:
    """
    Create an optimized DataLoader from tensor cache.
    
    OPTIMIZATION SETTINGS:
    - pin_memory=True: Faster CPU→GPU transfer
    - persistent_workers=True: Avoid worker restart overhead
    - prefetch_factor: Pipeline batches while GPU works
    - drop_last=True for train: Avoid variable batch sizes
    - collate_fn: Uses tensor_cache_collate_fn by default to handle None values
    
    Args:
        cache_dir: Path to tensor cache directory
        flag: Data split ('train', 'val', 'test')
        batch_size: Batch size for DataLoader
        num_workers: Number of data loading workers
        prefetch_factor: Batches to prefetch per worker
        shuffle: Whether to shuffle (default: True for train)
        preload_to_ram: Whether to preload data to RAM
        collate_fn: Custom collate function (default: tensor_cache_collate_fn)
        
    Returns:
        Configured DataLoader
    """
    if shuffle is None:
        shuffle = (flag == 'train')
    
    # Use custom collate function that handles None values
    # This is required because TensorCacheDataset may return None for optional fields
    if collate_fn is None:
        collate_fn = tensor_cache_collate_fn

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
        drop_last=(flag == 'train'),
        collate_fn=collate_fn
    )
