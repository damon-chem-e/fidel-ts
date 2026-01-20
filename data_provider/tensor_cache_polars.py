"""
Polars-based tensor cache generation for optimized performance.

This module provides polars-accelerated versions of tensor cache generation
operations. The key optimizations are:

1. VECTORIZED DEDUPLICATION: Uses polars .unique().sort().with_row_index()
   instead of Python dict-based O(n) insertion

2. VECTORIZED INDEX LOOKUP: Uses polars join instead of dict.get() in list
   comprehensions for O(1) vectorized lookup vs O(n) Python loops

3. BATCH PROCESSING: Processes timestamps in batches using polars DataFrames
   instead of one-at-a-time Python operations

Architecture:
- collect_all_timestamps(): Extract timestamps from all samples
- build_timestamp_index(): Deduplicate and assign indices with polars
- lookup_indices_polars(): Convert timestamps to indices via polars join
- PolarsSharedTableBuilder: Main class for polars-based shared table building

Usage:
    # In TensorCacheGenerator, set use_polars=True (default)
    generator = TensorCacheGenerator(data_provider, cache_dir, config, use_polars=True)
    generator.generate()

See docs/planning/tensor_cache_polars_optimization.md for design details.
"""

import gc
import logging
import numpy as np
import polars as pl
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

from .tensor_cache import (
    SAMPLE_IDX_SAMPLE_ID,
    SAMPLE_IDX_SEQ_X,
    SAMPLE_IDX_SEQ_Y,
    SAMPLE_IDX_X_TIME,
    SAMPLE_IDX_Y_TIME,
    SAMPLE_IDX_HETERO_X,
    SAMPLE_IDX_HETERO_Y,
    SAMPLE_IDX_HETERO_X_TIME,
    SAMPLE_IDX_HETERO_Y_TIME,
    SAMPLE_IDX_HETERO_GENERAL,
    SAMPLE_IDX_HETERO_CHANNEL,
    SAMPLE_IDX_X_TIME_FEATURES,
    SAMPLE_IDX_Y_TIME_FEATURES,
    InferredShapes,
    _safe_array,
    _detect_downtime_in_training,
    infer_shapes_from_sample,
    _is_llm_embedding,
    KNOWN_LLM_EMBED_DIMS,
)
from .dataset_direct_access import DirectAccessMixin

if TYPE_CHECKING:
    from data_provider.data_factory import Data_Provider

logger = logging.getLogger(__name__)


# =============================================================================
# TIMESTAMP COLLECTION
# =============================================================================

def extract_timestamps_from_sample(sample: tuple) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract x_time and y_time arrays from a sample tuple.

    Args:
        sample: 13-element tuple from dataset.__getitem__

    Returns:
        Tuple of (x_timestamps, y_timestamps) as int64 arrays
    """
    x_time = _safe_array(sample[SAMPLE_IDX_X_TIME])
    y_time = _safe_array(sample[SAMPLE_IDX_Y_TIME])

    x_ts = x_time.flatten().astype(np.int64) if x_time is not None else np.array([], dtype=np.int64)
    y_ts = y_time.flatten().astype(np.int64) if y_time is not None else np.array([], dtype=np.int64)

    return x_ts, y_ts


def collect_all_timestamps_from_datasets(
    data_provider: 'Data_Provider',
    flags: List[str]
) -> pl.DataFrame:
    """
    Collect all timestamps from all samples across all splits.

    This function iterates through samples (required by dataset API) but
    collects timestamps in batches for polars processing.

    Args:
        data_provider: Data provider with get_datasets() method
        flags: List of splits to process

    Returns:
        pl.DataFrame with single column 'ts' containing all timestamps
    """
    all_timestamps: List[int] = []

    for flag in flags:
        datasets = data_provider.get_datasets(flag)
        if not datasets:
            continue

        for entity_id, dataset in datasets.items():
            for sample_idx in range(len(dataset)):
                sample = dataset[sample_idx]
                x_ts, y_ts = extract_timestamps_from_sample(sample)
                all_timestamps.extend(x_ts.tolist())
                all_timestamps.extend(y_ts.tolist())

        del datasets
        gc.collect()

    return pl.DataFrame({'ts': all_timestamps})


# =============================================================================
# TIMESTAMP DEDUPLICATION & INDEX ASSIGNMENT
# =============================================================================

def build_timestamp_index(timestamps_df: pl.DataFrame) -> pl.DataFrame:
    """
    Build unique timestamp index using polars vectorized operations.

    This replaces the Python dict-based deduplication with:
    1. polars .unique() - hash-based parallel deduplication
    2. polars .sort() - efficient sorted order
    3. polars .with_row_index() - O(n) index assignment

    Args:
        timestamps_df: DataFrame with column 'ts' containing all timestamps

    Returns:
        DataFrame with columns ['ts', 'ts_idx'] sorted by timestamp
    """
    return (
        timestamps_df
        .unique()
        .sort('ts')
        .with_row_index('ts_idx')
        .cast({'ts_idx': pl.Int32})
    )


def build_timestamp_lookup_dict(index_df: pl.DataFrame) -> Dict[int, int]:
    """
    Convert polars index DataFrame to Python dict for compatibility.

    This is used for the shared table building phase where we need
    O(1) lookup during sample iteration.

    Args:
        index_df: DataFrame with ['ts', 'ts_idx'] columns

    Returns:
        Dict mapping timestamp -> index
    """
    return dict(zip(
        index_df['ts'].to_list(),
        index_df['ts_idx'].to_list()
    ))


# =============================================================================
# VECTORIZED INDEX LOOKUP
# =============================================================================

def lookup_indices_polars(
    timestamps: np.ndarray,
    index_df: pl.DataFrame
) -> np.ndarray:
    """
    Convert timestamps to indices using polars join (vectorized).

    This replaces the Python list comprehension:
        [timestamp_to_idx.get(int(ts), 0) for ts in timestamps]

    With a single polars hash join operation.

    Args:
        timestamps: Array of timestamps to look up
        index_df: DataFrame with ['ts', 'ts_idx'] columns

    Returns:
        np.ndarray of indices (int32), with 0 for missing timestamps
    """
    query_df = pl.DataFrame({'ts': timestamps.astype(np.int64)})

    result = (
        query_df
        .join(index_df.select(['ts', 'ts_idx']), on='ts', how='left')
        .with_columns(pl.col('ts_idx').fill_null(0))
    )

    return result['ts_idx'].to_numpy().astype(np.int32)


def lookup_indices_batch(
    sample_timestamps: List[np.ndarray],
    index_df: pl.DataFrame
) -> List[np.ndarray]:
    """
    Batch lookup indices for multiple samples.

    More efficient than calling lookup_indices_polars for each sample
    when processing many samples.

    Args:
        sample_timestamps: List of timestamp arrays
        index_df: DataFrame with ['ts', 'ts_idx'] columns

    Returns:
        List of index arrays (same order as input)
    """
    if not sample_timestamps:
        return []

    # Build combined query with sample markers
    all_ts = []
    lengths = []
    for ts_arr in sample_timestamps:
        flat = ts_arr.astype(np.int64)
        all_ts.extend(flat.tolist())
        lengths.append(len(flat))

    # Single vectorized lookup
    query_df = pl.DataFrame({'ts': all_ts})
    result = (
        query_df
        .join(index_df.select(['ts', 'ts_idx']), on='ts', how='left')
        .with_columns(pl.col('ts_idx').fill_null(0))
    )

    indices = result['ts_idx'].to_numpy().astype(np.int32)

    # Split back into per-sample arrays
    result_list = []
    offset = 0
    for length in lengths:
        result_list.append(indices[offset:offset + length])
        offset += length

    return result_list


# =============================================================================
# DATA COLLECTION WITH EARLY DEDUPLICATION
# =============================================================================

@dataclass
class PolarsCollectorState:
    """
    State container for polars-based data collection.

    Uses a set for O(1) membership checking during collection,
    then converts to polars for final processing.

    LLM EMBEDDING SUPPORT:
    - For LLM embeddings (shape: embed_dim, n_channels), embeddings are stored
      per-sample in llm_embeddings instead of per-timestamp in embeddings.
    - has_llm_embeddings is set when LLM embeddings are detected.
    - llm_embed_dim and llm_n_channels track the LLM embedding shape.
    """
    seen_timestamps: set = field(default_factory=set)
    timestamps: List[int] = field(default_factory=list)
    timeseries: List[np.ndarray] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)
    hetero_time: List[np.ndarray] = field(default_factory=list)

    entity_to_idx: Dict[str, int] = field(default_factory=dict)
    entity_general: List[Optional[np.ndarray]] = field(default_factory=list)
    entity_channel: List[Optional[np.ndarray]] = field(default_factory=list)

    shapes: InferredShapes = field(default_factory=InferredShapes)
    num_news_items: int = 1

    # LLM embedding fields
    llm_embeddings: List[np.ndarray] = field(default_factory=list)
    has_llm_embeddings: bool = False
    llm_embed_dim: Optional[int] = None
    llm_n_channels: Optional[int] = None


def register_timestamp_data_polars(
    state: PolarsCollectorState,
    timestamp: int,
    ts_value: Optional[np.ndarray],
    embedding: Optional[np.ndarray],
    hetero_time_feat: Optional[np.ndarray]
) -> None:
    """
    Register data for a timestamp if not already seen (polars version).

    Uses a set for O(1) membership check instead of dict.

    Args:
        state: PolarsCollectorState to update
        timestamp: Unix timestamp (int64)
        ts_value: Time series value at this timestamp
        embedding: Text embedding at this timestamp
        hetero_time_feat: Hetero time features at this timestamp
    """
    ts_key = int(timestamp)

    if ts_key in state.seen_timestamps:
        return

    state.seen_timestamps.add(ts_key)
    state.timestamps.append(ts_key)

    # Store time series value
    if ts_value is not None:
        state.timeseries.append(np.asarray(ts_value).flatten().copy())
    else:
        n_features = state.shapes.n_features or 1
        state.timeseries.append(np.zeros(n_features, dtype=np.float32))

    # Store embedding based on num_news_items
    _store_embedding(state, embedding)

    # Store hetero time features
    if hetero_time_feat is not None:
        state.hetero_time.append(np.asarray(hetero_time_feat).flatten().copy())
    else:
        n_htf = state.shapes.n_hetero_time_features or 1
        state.hetero_time.append(np.zeros(n_htf, dtype=np.float32))


def _store_embedding(state: PolarsCollectorState, embedding: Optional[np.ndarray]) -> None:
    """Store embedding with proper shape handling based on num_news_items."""
    if embedding is None:
        embed_dim = state.shapes.embed_dim or 768
        if state.num_news_items == 1:
            state.embeddings.append(np.zeros(embed_dim, dtype=np.float32))
        else:
            state.embeddings.append(np.zeros((2, embed_dim), dtype=np.float32))
        return

    emb_arr = np.asarray(embedding)

    if state.num_news_items == 1:
        # N=1: Store only the text embedding, flattened to (D,)
        if emb_arr.ndim == 1:
            emb_to_store = emb_arr.copy()
        elif emb_arr.ndim == 2:
            emb_to_store = emb_arr[0].copy()
        else:
            emb_to_store = emb_arr.flatten()
        state.embeddings.append(emb_to_store)

        if state.shapes.embed_dim is None:
            state.shapes.embed_dim = len(emb_to_store)
    else:
        # N=2: Store embedding + downtime indicator as (2, D)
        if emb_arr.ndim == 1:
            embed_dim = len(emb_arr)
            emb_to_store = np.zeros((2, embed_dim), dtype=np.float32)
            emb_to_store[0] = emb_arr.copy()
        elif emb_arr.ndim == 2 and emb_arr.shape[0] >= 2:
            emb_to_store = emb_arr[:2].copy()
        elif emb_arr.ndim == 2 and emb_arr.shape[0] == 1:
            embed_dim = emb_arr.shape[1]
            emb_to_store = np.zeros((2, embed_dim), dtype=np.float32)
            emb_to_store[0] = emb_arr[0]
        else:
            emb_flat = emb_arr.flatten()
            embed_dim = len(emb_flat)
            emb_to_store = np.zeros((2, embed_dim), dtype=np.float32)
            emb_to_store[0] = emb_flat

        state.embeddings.append(emb_to_store)

        if state.shapes.embed_dim is None:
            state.shapes.embed_dim = emb_to_store.shape[1]


def register_entity_data_polars(
    state: PolarsCollectorState,
    entity_id: str,
    hetero_general: Optional[np.ndarray],
    hetero_channel: Optional[np.ndarray]
) -> None:
    """Register entity-level static data if entity not already seen."""
    if entity_id in state.entity_to_idx:
        return

    idx = len(state.entity_general)
    state.entity_to_idx[entity_id] = idx

    if hetero_general is not None:
        arr = np.asarray(hetero_general).copy()
        state.entity_general.append(arr)
        if state.shapes.embed_dim is None:
            state.shapes.embed_dim = arr.shape[-1]
    else:
        state.entity_general.append(None)

    state.entity_channel.append(
        np.asarray(hetero_channel).copy() if hetero_channel is not None else None
    )


def process_sample_for_collection_polars(
    state: PolarsCollectorState,
    sample: tuple,
    skip_embeddings: bool = False
) -> None:
    """
    Process a single sample, registering all unique timestamps and data.

    Polars version uses set-based deduplication for O(1) membership check.

    Args:
        state: PolarsCollectorState to update
        sample: 13-element tuple from dataset.__getitem__
        skip_embeddings: If True, skip collecting per-timestamp embeddings
            (used for LLM mode where embeddings are collected per-sample)
    """
    # Update shapes if not yet inferred
    if state.shapes.n_features is None:
        sample_shapes = infer_shapes_from_sample(sample)
        state.shapes.n_features = sample_shapes.n_features
        state.shapes.embed_dim = sample_shapes.embed_dim or state.shapes.embed_dim
        state.shapes.n_hetero_time_features = sample_shapes.n_hetero_time_features

    # Process input window
    x_time = _safe_array(sample[SAMPLE_IDX_X_TIME])
    if x_time is not None:
        _process_window(state, sample, x_time.flatten(), is_input=True, skip_embeddings=skip_embeddings)

    # Process output window
    y_time = _safe_array(sample[SAMPLE_IDX_Y_TIME])
    if y_time is not None:
        _process_window(state, sample, y_time.flatten(), is_input=False, skip_embeddings=skip_embeddings)


def _process_window(
    state: PolarsCollectorState,
    sample: tuple,
    time_array: np.ndarray,
    is_input: bool,
    skip_embeddings: bool = False
) -> None:
    """
    Process input or output window timestamps.

    Args:
        state: PolarsCollectorState to update
        sample: Sample tuple
        time_array: Array of timestamps
        is_input: True for input window, False for output window
        skip_embeddings: If True, skip embedding extraction (for LLM mode)
    """
    if is_input:
        seq = _safe_array(sample[SAMPLE_IDX_SEQ_X])
        hetero = _safe_array(sample[SAMPLE_IDX_HETERO_X]) if not skip_embeddings else None
        htf = _safe_array(sample[SAMPLE_IDX_HETERO_X_TIME])
    else:
        seq = _safe_array(sample[SAMPLE_IDX_SEQ_Y])
        hetero = _safe_array(sample[SAMPLE_IDX_HETERO_Y]) if not skip_embeddings else None
        htf = _safe_array(sample[SAMPLE_IDX_HETERO_Y_TIME])

    for i, ts in enumerate(time_array):
        ts_val = seq[i] if seq is not None and i < len(seq) else None

        if hetero is not None:
            if hetero.ndim >= 2 and i < hetero.shape[0]:
                emb = hetero[i].copy()
            elif hetero.ndim == 1:
                emb = hetero.copy()
            else:
                emb = None
        else:
            emb = None

        htf_val = htf[i] if htf is not None and htf.ndim > 1 and i < len(htf) else None

        register_timestamp_data_polars(state, ts, ts_val, emb, htf_val)


# =============================================================================
# FINALIZATION
# =============================================================================

def finalize_shared_tables_polars(
    state: PolarsCollectorState
) -> Tuple[Dict[str, np.ndarray], dict]:
    """
    Convert collected lists to numpy arrays and build index mappings.

    Uses polars to build the timestamp_to_idx mapping efficiently.

    LLM EMBEDDING SUPPORT:
    - For LLM mode, returns llm_embeddings instead of embeddings
    - LLM embeddings are stored per-sample (not reordered by timestamp)

    Args:
        state: Completed PolarsCollectorState

    Returns:
        Tuple of (shared_tables dict, index_mappings dict)
    """
    shared_tables = {}

    # Convert timestamp data to arrays
    if state.timestamps:
        # Use polars to sort and assign indices
        ts_df = pl.DataFrame({'ts': state.timestamps})
        index_df = build_timestamp_index(ts_df)

        # Create ordered arrays aligned with sorted timestamps
        sorted_ts = index_df['ts'].to_numpy()

        # Build lookup from original order to sorted order
        original_to_sorted = {ts: i for i, ts in enumerate(sorted_ts)}

        # Reorder data arrays to match sorted timestamp order
        n_unique = len(state.timestamps)

        shared_tables['timestamps'] = sorted_ts.astype(np.int64)

        if state.timeseries:
            ts_arr = np.stack(state.timeseries)
            reordered_ts = np.zeros_like(ts_arr)
            for orig_idx, ts in enumerate(state.timestamps):
                sorted_idx = original_to_sorted[ts]
                reordered_ts[sorted_idx] = ts_arr[orig_idx]
            shared_tables['timeseries'] = reordered_ts.astype(np.float32)
            state.timeseries.clear()

        # For LLM mode, skip per-timestamp embeddings (they're collected per-sample)
        if not state.has_llm_embeddings and state.embeddings:
            emb_arr = np.stack(state.embeddings)
            reordered_emb = np.zeros_like(emb_arr)
            for orig_idx, ts in enumerate(state.timestamps):
                sorted_idx = original_to_sorted[ts]
                reordered_emb[sorted_idx] = emb_arr[orig_idx]
            shared_tables['embeddings'] = reordered_emb.astype(np.float32)
            state.embeddings.clear()

        if state.hetero_time:
            htf_arr = np.stack(state.hetero_time)
            reordered_htf = np.zeros_like(htf_arr)
            for orig_idx, ts in enumerate(state.timestamps):
                sorted_idx = original_to_sorted[ts]
                reordered_htf[sorted_idx] = htf_arr[orig_idx]
            shared_tables['hetero_time'] = reordered_htf.astype(np.float32)
            state.hetero_time.clear()

        state.timestamps.clear()
        state.seen_timestamps.clear()

    # LLM embeddings: stored per-sample, not per-timestamp
    if state.has_llm_embeddings and state.llm_embeddings:
        shared_tables['llm_embeddings'] = np.stack(state.llm_embeddings).astype(np.float32)
        state.llm_embeddings.clear()

    # Entity data
    valid_generals = [g for g in state.entity_general if g is not None]
    if valid_generals:
        shared_tables['entity_general'] = np.stack(valid_generals).astype(np.float32)
    state.entity_general.clear()

    valid_channels = [c for c in state.entity_channel if c is not None]
    if valid_channels:
        shared_tables['entity_channel'] = np.stack(valid_channels).astype(np.float32)
    state.entity_channel.clear()

    # Build index mappings (using sorted order)
    if 'timestamps' in shared_tables:
        timestamp_to_idx = {
            str(int(ts)): idx
            for idx, ts in enumerate(shared_tables['timestamps'])
        }
    else:
        timestamp_to_idx = {}

    index_mappings = {
        'timestamp_to_idx': timestamp_to_idx,
        'entity_to_idx': state.entity_to_idx
    }

    gc.collect()
    return shared_tables, index_mappings


# =============================================================================
# MAIN BUILDER CLASS
# =============================================================================

class PolarsSharedTableBuilder:
    """
    Builds shared tables using polars for optimized deduplication and indexing.

    This class is a drop-in replacement for the dict-based SharedTableCollector
    approach, providing identical output with better performance.

    Usage:
        builder = PolarsSharedTableBuilder(data_provider, num_news_items=1)
        shared_tables, index_mappings = builder.build(flags=['train', 'val', 'test'])
    """

    def __init__(
        self,
        data_provider: 'Data_Provider',
        num_news_items: int = 1,
        verbose: bool = True
    ):
        """
        Initialize the builder.

        Args:
            data_provider: Data provider with get_datasets() method
            num_news_items: 1 for embedding only, 2 for embedding + downtime
            verbose: Whether to log progress
        """
        self.data_provider = data_provider
        self.num_news_items = num_news_items
        self.verbose = verbose

        # LLM embedding state (set during build)
        self.has_llm_embeddings = False
        self.llm_embed_dim: Optional[int] = None
        self.llm_n_channels: Optional[int] = None

    def build(self, flags: List[str]) -> Tuple[Dict[str, np.ndarray], dict]:
        """
        Build shared tables for the specified splits.

        Automatically detects if datasets support direct array access and uses
        the optimized path (20x faster) when available. Falls back to per-sample
        iteration if direct access is not supported.

        Args:
            flags: List of splits to process

        Returns:
            Tuple of (shared_tables dict, index_mappings dict)
        """
        total_samples = self._count_samples(flags)

        if self.verbose:
            logger.info(f"Building shared tables: {total_samples:,} samples")

        # Check if all datasets support direct access
        if self._all_support_direct_access(flags):
            if self.verbose:
                logger.info("Using direct array access (20x faster)")
            return self._build_direct(flags)
        else:
            if self.verbose:
                logger.info("Using per-sample iteration (direct access not available)")
            return self._build_iterative(flags)

    def _all_support_direct_access(self, flags: List[str]) -> bool:
        """
        Check if all datasets across all splits support direct array access.

        Direct access requires:
        1. Dataset has DirectAccessMixin
        2. Dataset has supports_direct_access() returning True
        3. Dataset has preload_hetero=True (for embedding access)

        Returns:
            True if all datasets support direct access, False otherwise
        """
        for flag in flags:
            datasets = self.data_provider.get_datasets(flag)
            if not datasets:
                continue

            for entity_id, dataset in datasets.items():
                # Check for mixin
                if not isinstance(dataset, DirectAccessMixin):
                    if self.verbose:
                        logger.debug(f"Dataset {entity_id} does not have DirectAccessMixin")
                    del datasets
                    return False

                # Check supports_direct_access
                if not dataset.supports_direct_access():
                    if self.verbose:
                        logger.debug(f"Dataset {entity_id} does not support direct access")
                    del datasets
                    return False

                # Check preload_hetero (required for embedding access)
                if not getattr(dataset, 'preload_hetero', False):
                    if self.verbose:
                        logger.debug(f"Dataset {entity_id} does not have preload_hetero=True")
                    del datasets
                    return False

            del datasets

        return True

    def _build_direct(self, flags: List[str]) -> Tuple[Dict[str, np.ndarray], dict]:
        """
        Build shared tables using direct array access (20x faster).

        This method bypasses __getitem__ entirely, accessing the underlying
        numpy arrays directly for vectorized operations.

        LLM EMBEDDING SUPPORT:
        - For LLM embeddings (shape: embed_dim, n_channels), embeddings are stored
          per-sample instead of per-timestamp.
        - LLM embeddings are detected by shape heuristics using _is_llm_embedding().
        - Returns llm_embeddings in shared_tables instead of embeddings.

        Args:
            flags: List of splits to process

        Returns:
            Tuple of (shared_tables dict, index_mappings dict)
        """
        import time
        start_time = time.perf_counter()

        # Collect all unique timestamps across all datasets
        all_timestamps_set = set()
        entity_data = {}  # entity_id -> list of (raw_arrays, flag) tuples

        for flag in flags:
            datasets = self.data_provider.get_datasets(flag)
            if not datasets:
                continue

            for entity_id, dataset in datasets.items():
                if len(dataset) == 0:
                    continue

                # Get raw arrays
                raw = dataset.get_raw_arrays()

                # Get unique timestamps this dataset accesses
                unique_ts = dataset.get_all_unique_timestamps()
                all_timestamps_set.update(unique_ts.tolist())

                # Store raw arrays from ALL splits for this entity (with flag for LLM tracking)
                if entity_id not in entity_data:
                    entity_data[entity_id] = []
                entity_data[entity_id].append((raw, flag))

            del datasets

        if self.verbose:
            logger.info(f"Collected {len(all_timestamps_set):,} unique timestamps "
                       f"from {len(entity_data)} entities")

        # Sort timestamps for consistent indexing
        sorted_timestamps = np.array(sorted(all_timestamps_set), dtype=np.int64)
        n_unique = len(sorted_timestamps)

        # Build timestamp -> index mapping
        timestamp_to_idx = {int(ts): idx for idx, ts in enumerate(sorted_timestamps)}

        # Infer shapes from first entity
        first_entity_id = next(iter(entity_data))
        first_raw = entity_data[first_entity_id][0][0]  # First raw from list

        n_features = first_raw.data.shape[1] if first_raw.data.ndim > 1 else 1

        # Detect if this is an LLM embedding config
        # For LLM embeddings: shape is (embed_dim, n_channels) e.g., (768, 1)
        # For news embeddings: shape is (N, num_items, D) or (N, D) where N = timestamps
        is_llm_mode = False
        llm_embed_dim = None
        llm_n_channels = None

        if first_raw.embeddings is not None:
            # Use _is_llm_embedding with seq_len as input_len reference
            is_llm_mode = _is_llm_embedding(first_raw.embeddings, first_raw.seq_len)

            if is_llm_mode:
                # LLM embedding: shape is (embed_dim, n_channels)
                llm_embed_dim = first_raw.embeddings.shape[0]
                llm_n_channels = first_raw.embeddings.shape[1] if first_raw.embeddings.ndim > 1 else 1
                self.has_llm_embeddings = True
                self.llm_embed_dim = llm_embed_dim
                self.llm_n_channels = llm_n_channels

                if self.verbose:
                    logger.info(f"LLM embedding mode detected: shape ({llm_embed_dim}, {llm_n_channels})")

        # Determine embedding dimension for news embeddings
        embed_dim = 768  # Default
        if not is_llm_mode and first_raw.embeddings is not None:
            if first_raw.embeddings.ndim == 3:
                embed_dim = first_raw.embeddings.shape[2]
            elif first_raw.embeddings.ndim == 2:
                embed_dim = first_raw.embeddings.shape[1]
            else:
                embed_dim = first_raw.embeddings.shape[0] if first_raw.embeddings.ndim > 0 else 768
        elif first_raw.hetero_general is not None:
            embed_dim = first_raw.hetero_general.shape[-1]

        # Determine hetero time features dimension
        # NOTE: first_raw.hetero_time will be None because DirectAccessMixin.get_raw_arrays()
        # deliberately does NOT copy Universal_Dataset.hetero_time (which contains matched
        # timestamps as List[str], not time features). RawDataArrays.hetero_time expects
        # numeric time features (np.ndarray), which Universal_Dataset doesn't compute.
        # See RawDataArrays docstring for full explanation of this naming collision.
        n_htf = first_raw.hetero_time.shape[1] if first_raw.hetero_time is not None else 0

        # Pre-allocate shared table arrays
        timeseries_array = np.zeros((n_unique, n_features), dtype=np.float32)

        # For news embeddings, create per-timestamp array
        # For LLM embeddings, we'll collect per-sample later
        embeddings_array = None
        if not is_llm_mode:
            if self.num_news_items == 1:
                embeddings_array = np.zeros((n_unique, embed_dim), dtype=np.float32)
            else:
                embeddings_array = np.zeros((n_unique, 2, embed_dim), dtype=np.float32)

        if n_htf > 0:
            hetero_time_array = np.zeros((n_unique, n_htf), dtype=np.float32)
        else:
            hetero_time_array = None

        # Fill arrays from entity data
        seen_timestamps = set()

        # For LLM mode, collect embeddings per-sample
        llm_embeddings_list = [] if is_llm_mode else None

        for entity_id, raw_flag_list in entity_data.items():
            for raw, flag in raw_flag_list:  # Iterate over ALL splits for this entity
                # Handle 1D data case
                data = raw.data if raw.data.ndim > 1 else raw.data.reshape(-1, 1)

                # For LLM mode, collect per-sample embeddings
                if is_llm_mode and raw.embeddings is not None:
                    # LLM embedding is the same for all samples in this entity/split
                    # We need to add one copy per sample
                    for _ in range(raw.n_samples):
                        llm_embeddings_list.append(raw.embeddings.copy())

                for local_idx in range(len(raw.timestamps)):
                    ts_int = int(raw.timestamps[local_idx])

                    if ts_int in seen_timestamps:
                        continue  # Already have data for this timestamp

                    if ts_int not in timestamp_to_idx:
                        continue  # Timestamp not in unique set (shouldn't happen)

                    unique_idx = timestamp_to_idx[ts_int]
                    seen_timestamps.add(ts_int)

                    # Extract timeseries data
                    timeseries_array[unique_idx] = data[local_idx]

                    # Extract embedding data (news embeddings only)
                    if not is_llm_mode and raw.embeddings is not None and local_idx < len(raw.embeddings):
                        emb = raw.embeddings[local_idx]
                        if self.num_news_items == 1:
                            if emb.ndim == 0:
                                embeddings_array[unique_idx] = emb.flatten()
                            elif emb.ndim == 1:
                                embeddings_array[unique_idx] = emb
                            else:
                                embeddings_array[unique_idx] = emb[0]
                        else:
                            if emb.ndim == 0:
                                embeddings_array[unique_idx, 0] = emb.flatten()
                            elif emb.ndim == 1:
                                embeddings_array[unique_idx, 0] = emb
                            else:
                                embeddings_array[unique_idx] = emb[:2]

                    # Extract hetero time features
                    if raw.hetero_time is not None and hetero_time_array is not None:
                        if local_idx < len(raw.hetero_time):
                            hetero_time_array[unique_idx] = raw.hetero_time[local_idx]

        # Validation: Check for data integrity across splits
        n_filled = np.sum(np.any(timeseries_array != 0, axis=1))
        fill_ratio = n_filled / n_unique if n_unique > 0 else 0

        if fill_ratio < 0.9:
            raise ValueError(
                f"Shared table fill ratio is {fill_ratio:.1%} ({n_filled}/{n_unique}). "
                f"This indicates a bug in data collection across splits. "
                f"Expected >= 90% of timestamps to have non-zero data."
            )
        elif fill_ratio < 0.99:
            logger.warning(
                f"Shared table fill ratio is {fill_ratio:.1%} ({n_filled}/{n_unique}). "
                f"Some timestamps may have missing data."
            )

        if self.verbose:
            logger.info(f"Data integrity: {n_filled:,}/{n_unique:,} timestamps filled ({fill_ratio:.1%})")

        # Build shared tables dict
        shared_tables = {
            'timestamps': sorted_timestamps,
            'timeseries': timeseries_array,
        }

        # Add embeddings based on mode
        if is_llm_mode:
            # Stack LLM embeddings: shape (n_total_samples, embed_dim, n_channels)
            if llm_embeddings_list:
                shared_tables['llm_embeddings'] = np.stack(llm_embeddings_list).astype(np.float32)
                if self.verbose:
                    logger.info(f"LLM embeddings: {len(llm_embeddings_list):,} samples, "
                               f"shape {shared_tables['llm_embeddings'].shape}")
        else:
            shared_tables['embeddings'] = embeddings_array

        if hetero_time_array is not None:
            shared_tables['hetero_time'] = hetero_time_array

        # Entity-level data
        entity_general_list = []
        entity_channel_list = []
        entity_to_idx = {}

        for idx, (entity_id, raw_flag_list) in enumerate(entity_data.items()):
            entity_to_idx[entity_id] = idx
            # Use first raw for static entity data (same across splits)
            first_raw = raw_flag_list[0][0]
            if first_raw.hetero_general is not None:
                entity_general_list.append(first_raw.hetero_general)
            if first_raw.hetero_channel is not None:
                entity_channel_list.append(first_raw.hetero_channel)

        if entity_general_list:
            shared_tables['entity_general'] = np.stack(entity_general_list).astype(np.float32)
        if entity_channel_list:
            shared_tables['entity_channel'] = np.stack(entity_channel_list).astype(np.float32)

        # Build index mappings
        index_mappings = {
            'timestamp_to_idx': {str(k): v for k, v in timestamp_to_idx.items()},
            'entity_to_idx': entity_to_idx
        }

        elapsed = time.perf_counter() - start_time

        if self.verbose:
            logger.info(f"Built shared tables (direct): {n_unique:,} unique timestamps, "
                       f"{len(entity_to_idx)} entities in {elapsed:.2f}s")
            self._validate_splits(flags, timestamp_to_idx, timeseries_array)

        # Clean up
        del entity_data
        gc.collect()

        return shared_tables, index_mappings

    def _validate_splits(
        self,
        flags: List[str],
        timestamp_to_idx: Dict[int, int],
        timeseries_array: np.ndarray
    ) -> None:
        """
        Validate that each split has non-zero data in the shared tables.

        Samples timestamps from the first entity in each split and verifies
        they have non-zero values in the timeseries array.

        Args:
            flags: List of split names to validate
            timestamp_to_idx: Mapping from timestamp to array index
            timeseries_array: The populated timeseries shared table

        Raises:
            ValueError: If any split has > 10% zero-filled timestamps
        """
        for flag in flags:
            datasets = self.data_provider.get_datasets(flag)
            if not datasets:
                continue

            sample_count = 0
            zero_count = 0
            for entity_id, dataset in list(datasets.items())[:1]:  # Check first entity
                if len(dataset) == 0:
                    continue
                raw = dataset.get_raw_arrays()
                sample_ts = raw.timestamps[:min(100, len(raw.timestamps))]
                for ts in sample_ts:
                    ts_int = int(ts)
                    if ts_int in timestamp_to_idx:
                        idx = timestamp_to_idx[ts_int]
                        if np.all(timeseries_array[idx] == 0):
                            zero_count += 1
                        sample_count += 1

            del datasets

            if sample_count > 0:
                zero_ratio = zero_count / sample_count
                if zero_ratio > 0.1:
                    raise ValueError(
                        f"Split '{flag}' has {zero_ratio:.1%} zero-filled timestamps. "
                        f"Val/test data was not properly collected."
                    )
                logger.info(f"Split '{flag}' validated: {zero_ratio:.1%} zero timestamps")

    def _build_iterative(self, flags: List[str]) -> Tuple[Dict[str, np.ndarray], dict]:
        """
        Build shared tables using per-sample iteration (fallback method).

        This is the original implementation that iterates through __getitem__
        for each sample. Used when direct access is not available.

        LLM EMBEDDING SUPPORT:
        - Detects LLM embeddings from first sample's hetero_x
        - Collects LLM embeddings per-sample in state.llm_embeddings
        - Returns llm_embeddings in shared_tables instead of embeddings

        Args:
            flags: List of splits to process

        Returns:
            Tuple of (shared_tables dict, index_mappings dict)
        """
        state = PolarsCollectorState()
        state.num_news_items = self.num_news_items

        # Detect LLM mode from first sample
        self._detect_llm_mode(state, flags)

        self._collect_data(state, flags)

        if self.verbose:
            logger.info(f"Collected {len(state.timestamps):,} unique timestamps")
            if state.has_llm_embeddings:
                logger.info(f"LLM embeddings: {len(state.llm_embeddings):,} samples")

        shared_tables, index_mappings = finalize_shared_tables_polars(state)

        # Transfer LLM state to builder for later use
        if state.has_llm_embeddings:
            self.has_llm_embeddings = True
            self.llm_embed_dim = state.llm_embed_dim
            self.llm_n_channels = state.llm_n_channels

        return shared_tables, index_mappings

    def _detect_llm_mode(self, state: PolarsCollectorState, flags: List[str]) -> None:
        """
        Detect if this is an LLM embedding configuration.

        Examines the first sample's hetero_x to determine if LLM embeddings
        are being used (shape: embed_dim, n_channels).

        Args:
            state: PolarsCollectorState to update with LLM mode info
            flags: List of splits to check
        """
        for flag in flags:
            datasets = self.data_provider.get_datasets(flag)
            if not datasets:
                continue

            for entity_id, dataset in datasets.items():
                if len(dataset) == 0:
                    continue

                # Get first sample to check embedding shape
                first_sample = dataset[0]
                hetero_x = _safe_array(first_sample[SAMPLE_IDX_HETERO_X])

                if hetero_x is not None:
                    # Get input_len from x_time
                    x_time = _safe_array(first_sample[SAMPLE_IDX_X_TIME])
                    input_len = len(x_time) if x_time is not None else 96

                    if _is_llm_embedding(hetero_x, input_len):
                        state.has_llm_embeddings = True
                        state.llm_embed_dim = hetero_x.shape[0]
                        state.llm_n_channels = hetero_x.shape[1] if hetero_x.ndim > 1 else 1

                        if self.verbose:
                            logger.info(f"LLM embedding mode detected: shape "
                                       f"({state.llm_embed_dim}, {state.llm_n_channels})")

                del datasets
                return  # Only need to check one sample

            del datasets
            return

    def _count_samples(self, flags: List[str]) -> int:
        """Count total samples across all splits."""
        total = 0
        for flag in flags:
            datasets = self.data_provider.get_datasets(flag)
            if datasets:
                total += sum(len(ds) for ds in datasets.values())
            del datasets
        gc.collect()
        return total

    def _collect_data(self, state: PolarsCollectorState, flags: List[str]) -> None:
        """
        Collect unique data from all samples (iterative method).

        For LLM embeddings, collects per-sample embeddings instead of
        per-timestamp embeddings.
        """
        for flag in flags:
            datasets = self.data_provider.get_datasets(flag)
            if not datasets:
                continue

            for entity_id, dataset in datasets.items():
                if len(dataset) == 0:
                    continue

                # Register entity from first sample
                first_sample = dataset[0]
                register_entity_data_polars(
                    state,
                    entity_id,
                    first_sample[SAMPLE_IDX_HETERO_GENERAL],
                    first_sample[SAMPLE_IDX_HETERO_CHANNEL]
                )

                # Process all samples
                for sample_idx in range(len(dataset)):
                    sample = dataset[sample_idx]

                    # For LLM mode, collect embedding per-sample
                    if state.has_llm_embeddings:
                        hetero_x = _safe_array(sample[SAMPLE_IDX_HETERO_X])
                        if hetero_x is not None:
                            state.llm_embeddings.append(hetero_x.copy())

                    # Process timestamp data (timeseries, news embeddings if not LLM mode)
                    process_sample_for_collection_polars(state, sample, skip_embeddings=state.has_llm_embeddings)

            del datasets
            gc.collect()


# =============================================================================
# INDEX ARRAY GENERATION
# =============================================================================

def write_sample_indices_polars(
    arrays: Dict[str, np.ndarray],
    write_idx: int,
    sample: tuple,
    entity_idx: int,
    index_df: pl.DataFrame
) -> None:
    """
    Write index data for a single sample using polars lookup.

    This is the polars equivalent of _write_sample_indices, using
    polars join instead of dict.get() for index lookup.

    Args:
        arrays: Memory-mapped arrays to write to
        write_idx: Position in arrays to write
        sample: 13-element tuple from dataset
        entity_idx: Index of this sample's entity in entity tables
        index_df: DataFrame with ['ts', 'ts_idx'] columns for lookup
    """
    arrays['sample_ids'][write_idx] = str(sample[SAMPLE_IDX_SAMPLE_ID])
    arrays['entity_indices'][write_idx] = entity_idx

    # Convert timestamps to indices using polars
    x_time = np.asarray(sample[SAMPLE_IDX_X_TIME]).flatten()
    arrays['x_indices'][write_idx] = lookup_indices_polars(x_time, index_df)

    y_time = np.asarray(sample[SAMPLE_IDX_Y_TIME]).flatten()
    arrays['y_indices'][write_idx] = lookup_indices_polars(y_time, index_df)

    # Time features (direct copy)
    x_tf = sample[SAMPLE_IDX_X_TIME_FEATURES]
    if 'x_time_features' in arrays and x_tf is not None:
        arrays['x_time_features'][write_idx] = np.asarray(x_tf)

    y_tf = sample[SAMPLE_IDX_Y_TIME_FEATURES]
    if 'y_time_features' in arrays and y_tf is not None:
        arrays['y_time_features'][write_idx] = np.asarray(y_tf)


def write_sample_indices_batch_polars(
    arrays: Dict[str, np.ndarray],
    start_idx: int,
    samples: List[tuple],
    entity_idx: int,
    index_df: pl.DataFrame
) -> None:
    """
    Write index data for a batch of samples using batched polars lookup.

    More efficient than calling write_sample_indices_polars for each sample.

    Args:
        arrays: Memory-mapped arrays to write to
        start_idx: Starting position in arrays
        samples: List of 13-element tuples from dataset
        entity_idx: Index of entity in entity tables
        index_df: DataFrame with ['ts', 'ts_idx'] columns for lookup
    """
    if not samples:
        return

    # Collect all timestamps for batch lookup
    x_timestamps = [np.asarray(s[SAMPLE_IDX_X_TIME]).flatten() for s in samples]
    y_timestamps = [np.asarray(s[SAMPLE_IDX_Y_TIME]).flatten() for s in samples]

    # Batch lookup
    x_indices_list = lookup_indices_batch(x_timestamps, index_df)
    y_indices_list = lookup_indices_batch(y_timestamps, index_df)

    # Write to arrays
    for i, sample in enumerate(samples):
        write_idx = start_idx + i

        arrays['sample_ids'][write_idx] = str(sample[SAMPLE_IDX_SAMPLE_ID])
        arrays['entity_indices'][write_idx] = entity_idx
        arrays['x_indices'][write_idx] = x_indices_list[i]
        arrays['y_indices'][write_idx] = y_indices_list[i]

        x_tf = sample[SAMPLE_IDX_X_TIME_FEATURES]
        if 'x_time_features' in arrays and x_tf is not None:
            arrays['x_time_features'][write_idx] = np.asarray(x_tf)

        y_tf = sample[SAMPLE_IDX_Y_TIME_FEATURES]
        if 'y_time_features' in arrays and y_tf is not None:
            arrays['y_time_features'][write_idx] = np.asarray(y_tf)
