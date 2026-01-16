# Tensor Cache Optimization v2: Direct Array Access

**Date**: 2026-01-16
**Status**: Planning
**Priority**: HIGH (directly addresses time constraint)
**Target**: 10-30x speedup for tensor cache generation

---

## Executive Summary

### The Problem

Tensor cache generation is slow because we access data **sample-by-sample through Python**:

```python
# Current: 300k function calls, 300k tuple creations
for i in range(300000):
    sample = dataset[i]  # Slow!
```

### The Solution

Access underlying arrays **directly**, bypassing the dataset API:

```python
# Proposed: Direct vectorized access
timestamps = dataset.get_raw_timestamps()  # Single array access
data = dataset.get_raw_data()              # Single array access
# Then vectorized operations on entire arrays
```

### Expected Results

| Samples | Current | After Optimization | Speedup |
|---------|---------|-------------------|---------|
| 100k | 30s | 3-5s | 6-10x |
| 300k | 90s | 6-10s | 9-15x |
| 726k | ~220s | 15-25s | 9-15x |

With parallel processing added:

| Samples | Direct Access | + Parallel (4 cores) | Total Speedup |
|---------|--------------|---------------------|---------------|
| 300k | 6-10s | 2-3s | **30-45x** |
| 726k | 15-25s | 4-7s | **30-55x** |

---

## Table of Contents

1. [Current Dataset Structure](#1-current-dataset-structure)
2. [Proposed Interface Changes](#2-proposed-interface-changes)
3. [Implementation Plan](#3-implementation-plan)
4. [Tensor Cache Generator Changes](#4-tensor-cache-generator-changes)
5. [Parallel Processing Integration](#5-parallel-processing-integration)
6. [Testing Strategy](#6-testing-strategy)
7. [Migration Guide](#7-migration-guide)
8. [Timeline](#8-timeline)

---

## 1. Current Dataset Structure

### Universal_Dataset Internal Arrays

Based on code analysis, `Universal_Dataset` holds these arrays internally:

| Array | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `self.data` | `(N, n_features)` | float32 | Normalized time series data |
| `self.timestamp` | `(N,)` | int64 | Timestamps (YYYYMMDDHHMMSS format) |
| `self.scaler` | sklearn object | - | For denormalization |

When `preload_hetero=True`:

| Array | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `self.full_hetero` | `(N, num_news_items, embed_dim)` | float32 | Full embeddings |
| `self.hetero_time` | `(N, n_htf)` | float32 | Hetero time features |
| `self.hetero_general` | `(embed_dim,)` | float32 | Entity-level embedding |
| `self.hetero_channel` | `(embed_dim,)` | float32 | Channel embedding |

### Configuration Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `self.seq_len` | int | Input sequence length |
| `self.pred_len` | int | Output/prediction length |
| `self.stride` | int | Stride between samples |
| `self.hetero_stride` | int | Stride for hetero sampling |
| `self.entity_id` | str | Entity identifier |

### Current `__getitem__` (The Bottleneck)

```python
def __getitem__(self, index):
    # Window computation (repeated 300k times)
    s_begin = index
    s_end = s_begin + self.seq_len
    r_begin = s_end
    r_end = r_begin + self.pred_len

    # Array slicing (repeated 300k times)
    seq_x = self.data[s_begin:s_end]
    seq_y = self.data[r_begin:r_end]
    x_time = self.timestamp[s_begin:s_end]
    y_time = self.timestamp[r_begin:r_end]

    # Hetero data access (repeated 300k times)
    if self.preload_hetero:
        x_hetero = self.full_hetero[s_begin:s_end:self.hetero_stride]
        # ...

    # Tuple creation (repeated 300k times)
    return (sample_id, seq_x, seq_y, x_time, y_time, ...)
```

**Problem**: Each operation is trivial, but doing it 300k times = 90 seconds.

---

## 2. Proposed Interface Changes

### New Mixin: `DirectAccessMixin`

Add a mixin class that provides direct array access without breaking existing functionality:

```python
# data_provider/dataset_direct_access.py

"""
Mixin for direct array access in datasets.

This mixin provides methods for accessing underlying arrays directly,
enabling vectorized operations without per-sample iteration overhead.

Usage:
    class Universal_Dataset(DirectAccessMixin, Dataset):
        ...

    # Then in tensor cache generator:
    if dataset.supports_direct_access():
        raw = dataset.get_raw_arrays()
        # Vectorized operations on raw arrays
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class RawDataArrays:
    """Container for raw dataset arrays."""

    # Core time series data
    data: np.ndarray                    # (N, n_features), float32
    timestamps: np.ndarray              # (N,), int64

    # Window configuration
    seq_len: int
    pred_len: int
    stride: int

    # Entity info
    entity_id: str
    n_samples: int                      # Number of valid samples

    # Hetero data (optional)
    embeddings: Optional[np.ndarray] = None      # (N, num_news_items, D) or (N, D)
    hetero_time: Optional[np.ndarray] = None     # (N, n_htf)
    hetero_general: Optional[np.ndarray] = None  # (D,)
    hetero_channel: Optional[np.ndarray] = None  # (D,)
    hetero_stride: int = 1

    # Time features (optional)
    time_features: Optional[np.ndarray] = None   # (N, n_tf)

    def get_window_indices(self, sample_idx: int) -> tuple:
        """Get window start/end indices for a sample."""
        x_start = sample_idx * self.stride
        x_end = x_start + self.seq_len
        y_start = x_end
        y_end = y_start + self.pred_len
        return x_start, x_end, y_start, y_end

    def get_all_window_indices(self) -> Dict[str, np.ndarray]:
        """
        Get window indices for ALL samples at once (vectorized).

        Returns:
            Dict with keys 'x_start', 'x_end', 'y_start', 'y_end',
            each containing array of shape (n_samples,)
        """
        sample_indices = np.arange(self.n_samples)
        x_starts = sample_indices * self.stride
        x_ends = x_starts + self.seq_len
        y_starts = x_ends
        y_ends = y_starts + self.pred_len

        return {
            'x_start': x_starts,
            'x_end': x_ends,
            'y_start': y_starts,
            'y_end': y_ends,
        }


class DirectAccessMixin:
    """
    Mixin that provides direct access to underlying dataset arrays.

    This is a NON-BREAKING addition to existing dataset classes.
    All existing functionality continues to work unchanged.
    """

    def supports_direct_access(self) -> bool:
        """
        Check if this dataset supports direct array access.

        Returns True if the required arrays are available.
        """
        return (
            hasattr(self, 'data') and
            hasattr(self, 'timestamp') and
            hasattr(self, 'seq_len') and
            hasattr(self, 'pred_len')
        )

    def get_raw_arrays(self) -> RawDataArrays:
        """
        Get direct access to underlying arrays.

        Returns:
            RawDataArrays dataclass containing all raw arrays
            and configuration needed for vectorized operations.

        Raises:
            RuntimeError: If direct access is not supported
        """
        if not self.supports_direct_access():
            raise RuntimeError(
                f"Dataset {type(self).__name__} does not support direct access. "
                f"Missing required attributes."
            )

        # Calculate number of valid samples
        total_len = len(self.data)
        stride = getattr(self, 'stride', 1)
        n_samples = (total_len - self.seq_len - self.pred_len) // stride + 1

        # Build RawDataArrays
        raw = RawDataArrays(
            data=self.data,
            timestamps=self.timestamp,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            stride=stride,
            entity_id=getattr(self, 'entity_id', 'unknown'),
            n_samples=n_samples,
        )

        # Add hetero data if available
        if hasattr(self, 'full_hetero') and self.full_hetero is not None:
            raw.embeddings = self.full_hetero
            raw.hetero_stride = getattr(self, 'hetero_stride', 1)

        if hasattr(self, 'hetero_time') and self.hetero_time is not None:
            raw.hetero_time = self.hetero_time

        if hasattr(self, 'hetero_general'):
            raw.hetero_general = self.hetero_general

        if hasattr(self, 'hetero_channel'):
            raw.hetero_channel = self.hetero_channel

        # Add time features if available
        if hasattr(self, 'time_features') and self.time_features is not None:
            raw.time_features = self.time_features

        return raw

    def get_raw_timestamps_for_samples(
        self,
        sample_indices: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get timestamps for multiple samples at once (vectorized).

        Args:
            sample_indices: Array of sample indices. If None, returns for all samples.

        Returns:
            Dict with:
                'x_time': (n_samples, seq_len) array of input timestamps
                'y_time': (n_samples, pred_len) array of output timestamps
        """
        raw = self.get_raw_arrays()

        if sample_indices is None:
            sample_indices = np.arange(raw.n_samples)

        n_samples = len(sample_indices)
        stride = raw.stride

        # Pre-allocate output arrays
        x_times = np.empty((n_samples, raw.seq_len), dtype=np.int64)
        y_times = np.empty((n_samples, raw.pred_len), dtype=np.int64)

        # Vectorized index computation
        x_starts = sample_indices * stride

        # Fill arrays (still a loop, but much tighter than __getitem__)
        for i, x_start in enumerate(x_starts):
            x_end = x_start + raw.seq_len
            y_start = x_end
            y_end = y_start + raw.pred_len

            x_times[i] = raw.timestamps[x_start:x_end]
            y_times[i] = raw.timestamps[y_start:y_end]

        return {
            'x_time': x_times,
            'y_time': y_times,
        }

    def get_raw_data_for_samples(
        self,
        sample_indices: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get all data for multiple samples at once (vectorized).

        Args:
            sample_indices: Array of sample indices. If None, returns for all samples.

        Returns:
            Dict with all sample data arrays
        """
        raw = self.get_raw_arrays()

        if sample_indices is None:
            sample_indices = np.arange(raw.n_samples)

        n_samples = len(sample_indices)
        stride = raw.stride
        n_features = raw.data.shape[1]

        # Pre-allocate output arrays
        result = {
            'seq_x': np.empty((n_samples, raw.seq_len, n_features), dtype=np.float32),
            'seq_y': np.empty((n_samples, raw.pred_len, n_features), dtype=np.float32),
            'x_time': np.empty((n_samples, raw.seq_len), dtype=np.int64),
            'y_time': np.empty((n_samples, raw.pred_len), dtype=np.int64),
        }

        # Add hetero arrays if available
        if raw.embeddings is not None:
            embed_shape = raw.embeddings.shape[1:]  # (num_news_items, D) or (D,)
            hetero_len_x = (raw.seq_len + raw.hetero_stride - 1) // raw.hetero_stride
            hetero_len_y = (raw.pred_len + raw.hetero_stride - 1) // raw.hetero_stride

            result['hetero_x'] = np.empty(
                (n_samples, hetero_len_x) + embed_shape, dtype=np.float32
            )
            result['hetero_y'] = np.empty(
                (n_samples, hetero_len_y) + embed_shape, dtype=np.float32
            )

        # Vectorized index computation
        x_starts = sample_indices * stride

        # Fill arrays
        for i, x_start in enumerate(x_starts):
            x_end = x_start + raw.seq_len
            y_start = x_end
            y_end = y_start + raw.pred_len

            result['seq_x'][i] = raw.data[x_start:x_end]
            result['seq_y'][i] = raw.data[y_start:y_end]
            result['x_time'][i] = raw.timestamps[x_start:x_end]
            result['y_time'][i] = raw.timestamps[y_start:y_end]

            if raw.embeddings is not None:
                result['hetero_x'][i] = raw.embeddings[x_start:x_end:raw.hetero_stride]
                result['hetero_y'][i] = raw.embeddings[y_start:y_end:raw.hetero_stride]

        # Add entity-level data (same for all samples)
        if raw.hetero_general is not None:
            result['hetero_general'] = raw.hetero_general
        if raw.hetero_channel is not None:
            result['hetero_channel'] = raw.hetero_channel

        return result
```

### Integrating with Existing Datasets

**Non-breaking change**: Add mixin to existing classes:

```python
# data_provider/data_loader.py

from data_provider.dataset_direct_access import DirectAccessMixin

class Universal_Dataset(DirectAccessMixin, Dataset):
    """
    Universal dataset for time series forecasting.

    Now supports direct array access via DirectAccessMixin.
    All existing functionality unchanged.
    """
    # ... existing code unchanged ...
```

```python
# data_provider/time_mmd_dataset.py

from data_provider.dataset_direct_access import DirectAccessMixin

class TimeMMD_Dataset(DirectAccessMixin, Dataset):
    """
    TimeMMD dataset for text-augmented time series.

    Now supports direct array access via DirectAccessMixin.
    """
    # ... existing code unchanged ...
```

### Ensuring Hetero Data is Preloaded

For direct access to work with embeddings, we need `preload_hetero=True`:

```python
# In TensorCacheGenerator, ensure preload_hetero=True when getting datasets
datasets = data_provider.get_datasets(flag, preload_hetero=True)
```

Or add a method to force preloading:

```python
class DirectAccessMixin:
    # ... existing methods ...

    def ensure_hetero_preloaded(self):
        """
        Ensure hetero data is fully loaded into memory.

        Call this before using get_raw_arrays() if hetero data
        might be lazy-loaded.
        """
        if hasattr(self, 'hetero_data_getter') and self.hetero_data_getter is not None:
            if not hasattr(self, 'full_hetero') or self.full_hetero is None:
                # Force load all hetero data
                self._preload_all_hetero()

    def _preload_all_hetero(self):
        """Preload all hetero data into memory."""
        if not hasattr(self, 'timestamp') or self.hetero_data_getter is None:
            return

        # Get all unique timestamps
        all_timestamps = self.timestamp

        # Load embeddings for all timestamps
        self.full_hetero = self.hetero_data_getter(all_timestamps)

        # Load other hetero components if available
        if hasattr(self.hetero_data_getter, 'get_general'):
            self.hetero_general = self.hetero_data_getter.get_general()
        if hasattr(self.hetero_data_getter, 'get_channel'):
            self.hetero_channel = self.hetero_data_getter.get_channel()
```

---

## 3. Implementation Plan

### Phase 1: Add DirectAccessMixin (2-3 days)

**Files to create:**
- `data_provider/dataset_direct_access.py` - New file with mixin

**Files to modify:**
- `data_provider/data_loader.py` - Add mixin to `Universal_Dataset`
- `data_provider/time_mmd_dataset.py` - Add mixin to `TimeMMD_Dataset`

**Tests to add:**
- `tests/test_dataset_direct_access.py`

### Phase 2: Update Tensor Cache Generator (3-4 days)

**New method in `TensorCacheGenerator`:**

```python
def _build_shared_tables_direct(
    self,
    flags: List[str]
) -> Tuple[Dict[str, np.ndarray], dict]:
    """
    Build shared tables using direct array access.

    This bypasses the dataset.__getitem__ API entirely,
    accessing underlying arrays directly for vectorized operations.

    Expected speedup: 5-10x over per-sample iteration.
    """
    from data_provider.tensor_cache_polars import (
        PolarsCollectorState,
        build_timestamp_index,
        finalize_shared_tables_polars,
    )

    # Detect downtime (still use per-sample for this small check)
    has_downtime = _detect_downtime_in_training(self.data_provider)
    num_news_items = 2 if has_downtime else 1

    # Collect all unique timestamps across all datasets (vectorized)
    all_timestamps = []
    all_data_map = {}  # ts -> (entity_id, local_idx)

    for flag in flags:
        datasets = self.data_provider.get_datasets(flag)

        for entity_id, dataset in datasets.items():
            # Check if dataset supports direct access
            if not hasattr(dataset, 'supports_direct_access') or \
               not dataset.supports_direct_access():
                # Fall back to per-sample iteration for this dataset
                logger.warning(f"Dataset {entity_id} does not support direct access")
                continue

            # Ensure hetero data is preloaded
            if hasattr(dataset, 'ensure_hetero_preloaded'):
                dataset.ensure_hetero_preloaded()

            # Get raw arrays
            raw = dataset.get_raw_arrays()

            # Get ALL timestamps for this entity (vectorized!)
            ts_data = dataset.get_raw_timestamps_for_samples()
            x_times = ts_data['x_time'].flatten()  # (n_samples * seq_len,)
            y_times = ts_data['y_time'].flatten()  # (n_samples * pred_len,)

            entity_timestamps = np.concatenate([x_times, y_times])
            all_timestamps.append(entity_timestamps)

            # Store reference for data extraction later
            all_data_map[entity_id] = raw

    # Combine and deduplicate timestamps (polars-optimized)
    import polars as pl

    combined_timestamps = np.concatenate(all_timestamps)
    ts_df = pl.DataFrame({'ts': combined_timestamps})
    index_df = build_timestamp_index(ts_df)

    unique_timestamps = index_df['ts'].to_numpy()
    n_unique = len(unique_timestamps)

    if self.verbose:
        logger.info(f"Collected {len(combined_timestamps):,} timestamp references")
        logger.info(f"Deduplicated to {n_unique:,} unique timestamps")
        logger.info(f"Deduplication ratio: {len(combined_timestamps)/n_unique:.1f}:1")

    # Build shared tables by extracting data for unique timestamps
    shared_tables = self._extract_data_for_unique_timestamps(
        unique_timestamps, all_data_map, num_news_items
    )

    # Build index mappings
    timestamp_to_idx = {str(int(ts)): idx for idx, ts in enumerate(unique_timestamps)}
    entity_to_idx = {eid: idx for idx, eid in enumerate(all_data_map.keys())}

    index_mappings = {
        'timestamp_to_idx': timestamp_to_idx,
        'entity_to_idx': entity_to_idx,
    }

    return shared_tables, index_mappings


def _extract_data_for_unique_timestamps(
    self,
    unique_timestamps: np.ndarray,
    all_data_map: Dict[str, RawDataArrays],
    num_news_items: int
) -> Dict[str, np.ndarray]:
    """
    Extract data values for each unique timestamp.

    This finds the first occurrence of each timestamp and extracts
    the corresponding data values.
    """
    n_unique = len(unique_timestamps)

    # Infer shapes from first dataset
    first_raw = next(iter(all_data_map.values()))
    n_features = first_raw.data.shape[1]

    if first_raw.embeddings is not None:
        if first_raw.embeddings.ndim == 3:
            embed_dim = first_raw.embeddings.shape[2]
        else:
            embed_dim = first_raw.embeddings.shape[1]
    else:
        embed_dim = 768  # Default

    n_htf = first_raw.hetero_time.shape[1] if first_raw.hetero_time is not None else 0

    # Pre-allocate arrays
    timestamps_array = unique_timestamps.astype(np.int64)
    timeseries_array = np.zeros((n_unique, n_features), dtype=np.float32)

    if num_news_items == 1:
        embeddings_array = np.zeros((n_unique, embed_dim), dtype=np.float32)
    else:
        embeddings_array = np.zeros((n_unique, 2, embed_dim), dtype=np.float32)

    if n_htf > 0:
        hetero_time_array = np.zeros((n_unique, n_htf), dtype=np.float32)
    else:
        hetero_time_array = None

    # Build reverse lookup: timestamp -> index in unique array
    ts_to_unique_idx = {ts: idx for idx, ts in enumerate(unique_timestamps)}

    # For each entity, find timestamps that match and extract data
    seen_timestamps = set()

    for entity_id, raw in all_data_map.items():
        # Get all timestamps in this entity's data
        entity_timestamps = raw.timestamps

        for local_idx, ts in enumerate(entity_timestamps):
            ts_int = int(ts)

            if ts_int in seen_timestamps:
                continue  # Already have data for this timestamp

            if ts_int not in ts_to_unique_idx:
                continue  # Timestamp not in unique set (shouldn't happen)

            unique_idx = ts_to_unique_idx[ts_int]
            seen_timestamps.add(ts_int)

            # Extract data at this timestamp
            timeseries_array[unique_idx] = raw.data[local_idx]

            if raw.embeddings is not None:
                emb = raw.embeddings[local_idx]
                if num_news_items == 1:
                    if emb.ndim == 1:
                        embeddings_array[unique_idx] = emb
                    else:
                        embeddings_array[unique_idx] = emb[0]
                else:
                    if emb.ndim == 1:
                        embeddings_array[unique_idx, 0] = emb
                    else:
                        embeddings_array[unique_idx] = emb[:2]

            if raw.hetero_time is not None and hetero_time_array is not None:
                hetero_time_array[unique_idx] = raw.hetero_time[local_idx]

    # Build shared tables dict
    shared_tables = {
        'timestamps': timestamps_array,
        'timeseries': timeseries_array,
        'embeddings': embeddings_array,
    }

    if hetero_time_array is not None:
        shared_tables['hetero_time'] = hetero_time_array

    # Entity data
    entity_general_list = []
    entity_channel_list = []

    for entity_id, raw in all_data_map.items():
        if raw.hetero_general is not None:
            entity_general_list.append(raw.hetero_general)
        if raw.hetero_channel is not None:
            entity_channel_list.append(raw.hetero_channel)

    if entity_general_list:
        shared_tables['entity_general'] = np.stack(entity_general_list).astype(np.float32)
    if entity_channel_list:
        shared_tables['entity_channel'] = np.stack(entity_channel_list).astype(np.float32)

    return shared_tables
```

### Phase 3: Add Parallel Processing (3-4 days)

Combine direct array access with parallel entity processing:

```python
def _build_shared_tables_parallel_direct(
    self,
    flags: List[str],
    n_workers: int = None
) -> Tuple[Dict[str, np.ndarray], dict]:
    """
    Build shared tables using parallel processing + direct array access.

    Each worker:
    1. Gets assigned entities
    2. Uses direct array access to extract timestamps and data
    3. Returns partial results

    Master process:
    1. Merges results from all workers
    2. Performs global deduplication
    3. Builds final shared tables

    Expected speedup: 15-30x over sequential per-sample iteration.
    """
    import multiprocessing as mp
    from functools import partial

    if n_workers is None:
        n_workers = mp.cpu_count()

    # Collect all (entity_id, flag) tasks
    tasks = []
    for flag in flags:
        datasets = self.data_provider.get_datasets(flag)
        for entity_id in datasets.keys():
            tasks.append((entity_id, flag))

    if len(tasks) < n_workers:
        # Not enough entities for parallelization
        return self._build_shared_tables_direct(flags)

    # Detect downtime (quick, single-threaded)
    has_downtime = _detect_downtime_in_training(self.data_provider)
    num_news_items = 2 if has_downtime else 1

    # Process entities in parallel
    with mp.Pool(processes=n_workers) as pool:
        worker_func = partial(
            _process_entity_direct_worker,
            data_provider_config=self._serialize_config(),
            num_news_items=num_news_items
        )

        results = pool.map(worker_func, tasks)

    # Merge results from all workers
    return self._merge_parallel_results(results, num_news_items)


def _process_entity_direct_worker(
    task: Tuple[str, str],
    data_provider_config: Dict,
    num_news_items: int
) -> Dict:
    """
    Worker function for parallel entity processing.

    Uses direct array access for maximum speed.
    """
    entity_id, flag = task

    # Reconstruct data provider in worker
    data_provider = _reconstruct_data_provider(data_provider_config)
    datasets = data_provider.get_datasets(flag)

    if entity_id not in datasets:
        return {'entity_id': entity_id, 'timestamps': [], 'data': {}}

    dataset = datasets[entity_id]

    # Use direct access
    if hasattr(dataset, 'ensure_hetero_preloaded'):
        dataset.ensure_hetero_preloaded()

    raw = dataset.get_raw_arrays()

    # Extract all timestamps
    ts_data = dataset.get_raw_timestamps_for_samples()
    all_timestamps = np.concatenate([
        ts_data['x_time'].flatten(),
        ts_data['y_time'].flatten()
    ])

    # Get unique timestamps for this entity
    unique_ts = np.unique(all_timestamps)

    # Extract data for each unique timestamp
    ts_to_local_idx = {ts: np.where(raw.timestamps == ts)[0][0]
                       for ts in unique_ts if ts in raw.timestamps}

    data_dict = {}
    for ts, local_idx in ts_to_local_idx.items():
        data_dict[int(ts)] = {
            'timeseries': raw.data[local_idx],
            'embedding': raw.embeddings[local_idx] if raw.embeddings is not None else None,
            'hetero_time': raw.hetero_time[local_idx] if raw.hetero_time is not None else None,
        }

    return {
        'entity_id': entity_id,
        'timestamps': unique_ts.tolist(),
        'data': data_dict,
        'hetero_general': raw.hetero_general,
        'hetero_channel': raw.hetero_channel,
    }
```

### Phase 4: Integration & Testing (2-3 days)

1. Add feature flags for new implementations
2. Verify correctness against existing implementation
3. Benchmark performance
4. Update documentation

---

## 4. Tensor Cache Generator Changes

### Updated `_build_shared_tables` Method

```python
class TensorCacheGenerator:
    def __init__(
        self,
        data_provider,
        cache_dir,
        config,
        chunk_size: int = 10000,
        verbose: bool = True,
        console = None,
        use_polars: bool = True,
        use_direct_access: bool = True,    # NEW
        use_parallel: bool = True,          # NEW
        n_workers: int = None               # NEW
    ):
        """
        Initialize tensor cache generator.

        Args:
            use_direct_access: Use direct array access (5-10x faster)
            use_parallel: Use parallel entity processing (2-4x faster)
            n_workers: Number of worker processes (default: CPU count)
        """
        self.use_direct_access = use_direct_access
        self.use_parallel = use_parallel
        self.n_workers = n_workers
        # ... rest of init ...

    def _build_shared_tables(self, flags: List[str]):
        """
        Build shared tables with automatic optimization selection.

        Priority:
        1. Parallel + direct access (15-30x faster)
        2. Direct access only (5-10x faster)
        3. Polars-optimized per-sample (1.1x faster)
        4. Standard per-sample (baseline)
        """
        # Check if direct access is available
        can_use_direct = self._check_direct_access_support(flags)

        if can_use_direct and self.use_direct_access:
            if self.use_parallel and self._should_use_parallel(flags):
                logger.info("Using parallel + direct access (fastest)")
                return self._build_shared_tables_parallel_direct(flags)
            else:
                logger.info("Using direct array access")
                return self._build_shared_tables_direct(flags)

        elif self.use_polars:
            logger.info("Using polars-optimized per-sample iteration")
            return self._build_shared_tables_polars(flags)

        else:
            logger.info("Using standard per-sample iteration")
            return self._build_shared_tables_standard(flags)

    def _check_direct_access_support(self, flags: List[str]) -> bool:
        """Check if all datasets support direct array access."""
        for flag in flags:
            datasets = self.data_provider.get_datasets(flag)
            for entity_id, dataset in datasets.items():
                if not hasattr(dataset, 'supports_direct_access'):
                    return False
                if not dataset.supports_direct_access():
                    return False
        return True
```

### CLI Updates

```python
# cli/tensor_cache.py

def generate(
    suite_path: Path,
    splits: str = "train,val,test",
    force: bool = False,
    dry_run: bool = False,
    cpu_only: bool = False,
    use_polars: bool = True,
    use_direct_access: bool = True,  # NEW
    use_parallel: bool = True,        # NEW
    n_workers: int = None             # NEW
):
    """
    Generate tensor cache for all experiments in a suite.

    Performance options:
      --use-direct-access: Use direct array access (default: enabled)
      --use-parallel: Use parallel entity processing (default: enabled)
      --n-workers: Number of parallel workers (default: CPU count)

    Expected speedups:
      - Default (parallel + direct): 15-30x
      - Direct only: 5-10x
      - Polars only: 1.1x
    """
```

---

## 5. Parallel Processing Integration

### Worker Design

```
┌─────────────────────────────────────────────────────────────────┐
│                      Master Process                              │
├─────────────────────────────────────────────────────────────────┤
│  1. Enumerate entities from all splits                          │
│  2. Distribute entities to workers (round-robin)                │
│  3. Wait for worker results                                     │
│  4. Merge results: global deduplication, build shared tables    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ┌─────────┐        ┌─────────┐        ┌─────────┐
   │ Worker 1│        │ Worker 2│        │ Worker N│
   ├─────────┤        ├─────────┤        ├─────────┤
   │Entity A │        │Entity B │        │Entity C │
   │Entity D │        │Entity E │        │Entity F │
   │         │        │         │        │         │
   │ Direct  │        │ Direct  │        │ Direct  │
   │ Access  │        │ Access  │        │ Access  │
   └────┬────┘        └────┬────┘        └────┬────┘
        │                  │                  │
        └──────────────────┴──────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │ Merge Results  │
                  │ - Deduplicate  │
                  │ - Build arrays │
                  └────────────────┘
```

### Memory Considerations

Each worker loads its assigned entities into memory. With N workers and E entities:
- Each worker loads ~E/N entities
- Total memory: Same as sequential (each timestamp loaded once)
- Peak memory: Slightly higher during merge phase

### Scalability

| Workers | Entities | Expected Speedup | Notes |
|---------|----------|------------------|-------|
| 1 | Any | 1x (no parallelism) | Fall back to direct access only |
| 4 | 4+ | 3-3.5x | Good for typical datasets |
| 8 | 8+ | 5-7x | Diminishing returns start |
| 16 | 16+ | 8-12x | Communication overhead increases |

---

## 6. Testing Strategy

### Unit Tests

```python
# tests/test_dataset_direct_access.py

class TestDirectAccessMixin:
    """Tests for DirectAccessMixin functionality."""

    def test_supports_direct_access(self, mock_dataset):
        """Test direct access detection."""
        assert mock_dataset.supports_direct_access()

    def test_get_raw_arrays(self, mock_dataset):
        """Test raw array extraction."""
        raw = mock_dataset.get_raw_arrays()

        assert raw.data is not None
        assert raw.timestamps is not None
        assert raw.n_samples == len(mock_dataset)
        assert raw.seq_len == mock_dataset.seq_len
        assert raw.pred_len == mock_dataset.pred_len

    def test_get_raw_timestamps_matches_getitem(self, mock_dataset):
        """Verify direct timestamps match __getitem__ timestamps."""
        ts_data = mock_dataset.get_raw_timestamps_for_samples()

        # Compare with __getitem__ for first 10 samples
        for i in range(10):
            sample = mock_dataset[i]
            x_time_getitem = sample[3]
            y_time_getitem = sample[4]

            np.testing.assert_array_equal(ts_data['x_time'][i], x_time_getitem)
            np.testing.assert_array_equal(ts_data['y_time'][i], y_time_getitem)

    def test_get_raw_data_matches_getitem(self, mock_dataset):
        """Verify direct data matches __getitem__ data."""
        data = mock_dataset.get_raw_data_for_samples(np.arange(10))

        for i in range(10):
            sample = mock_dataset[i]

            np.testing.assert_array_almost_equal(data['seq_x'][i], sample[1])
            np.testing.assert_array_almost_equal(data['seq_y'][i], sample[2])


class TestTensorCacheDirectAccess:
    """Tests for tensor cache generation with direct access."""

    def test_direct_access_matches_standard(self, mock_data_provider, tmp_path):
        """Verify direct access produces identical cache to standard."""
        # Generate with standard method
        gen_standard = TensorCacheGenerator(
            mock_data_provider, tmp_path / 'standard', {},
            use_direct_access=False, use_polars=False
        )
        tables_standard, mappings_standard = gen_standard._build_shared_tables(['train'])

        # Generate with direct access
        gen_direct = TensorCacheGenerator(
            mock_data_provider, tmp_path / 'direct', {},
            use_direct_access=True
        )
        tables_direct, mappings_direct = gen_direct._build_shared_tables(['train'])

        # Compare results
        for key in tables_standard:
            np.testing.assert_array_equal(
                tables_standard[key],
                tables_direct[key],
                err_msg=f"Mismatch in {key}"
            )

        assert mappings_standard['timestamp_to_idx'] == mappings_direct['timestamp_to_idx']
```

### Integration Tests

```python
class TestEndToEndDirectAccess:
    """End-to-end tests with real datasets."""

    @pytest.mark.slow
    def test_real_dataset_direct_access(self, traffic_dataset):
        """Test direct access on real time_mmd_traffic dataset."""
        if not traffic_dataset.supports_direct_access():
            pytest.skip("Dataset does not support direct access")

        raw = traffic_dataset.get_raw_arrays()

        # Verify arrays are valid
        assert raw.data.shape[0] > 0
        assert raw.timestamps.shape[0] == raw.data.shape[0]
        assert raw.n_samples > 0
```

### Performance Benchmarks

```python
class TestPerformanceBenchmarks:
    """Performance benchmarks for optimization comparison."""

    @pytest.mark.slow
    def test_speedup_direct_vs_standard(self, large_mock_dataset):
        """Benchmark direct access vs standard iteration."""
        import time

        # Standard: per-sample iteration
        start = time.perf_counter()
        for i in range(len(large_mock_dataset)):
            sample = large_mock_dataset[i]
        standard_time = time.perf_counter() - start

        # Direct: batch extraction
        start = time.perf_counter()
        data = large_mock_dataset.get_raw_data_for_samples()
        direct_time = time.perf_counter() - start

        speedup = standard_time / direct_time
        print(f"\nDirect access speedup: {speedup:.1f}x")
        print(f"  Standard: {standard_time:.2f}s")
        print(f"  Direct: {direct_time:.2f}s")

        assert speedup >= 3.0, f"Expected at least 3x speedup, got {speedup:.1f}x"
```

---

## 7. Migration Guide

### For Dataset Maintainers

**Step 1**: Add mixin to your dataset class (non-breaking):

```python
from data_provider.dataset_direct_access import DirectAccessMixin

class MyDataset(DirectAccessMixin, Dataset):
    # Existing code unchanged
    pass
```

**Step 2**: Ensure required attributes exist:

```python
class MyDataset(DirectAccessMixin, Dataset):
    def __init__(self, ...):
        # Required attributes for direct access:
        self.data = ...        # numpy array (N, n_features)
        self.timestamp = ...   # numpy array (N,)
        self.seq_len = ...     # int
        self.pred_len = ...    # int

        # Optional but recommended:
        self.stride = ...      # int (default: 1)
        self.entity_id = ...   # str
        self.full_hetero = ... # numpy array (if embeddings)
```

**Step 3**: Test direct access works:

```python
dataset = MyDataset(...)
assert dataset.supports_direct_access()

raw = dataset.get_raw_arrays()
print(f"Data shape: {raw.data.shape}")
print(f"Timestamps shape: {raw.timestamps.shape}")
print(f"N samples: {raw.n_samples}")
```

### For Tensor Cache Users

No changes required! The generator automatically detects and uses direct access when available.

To force a specific method:

```bash
# Force direct access (fastest if supported)
python -m cli.tensor_cache generate <suite> --use-direct-access --use-parallel

# Disable direct access (fall back to polars)
python -m cli.tensor_cache generate <suite> --no-direct-access

# Disable parallelism
python -m cli.tensor_cache generate <suite> --no-parallel
```

---

## 8. Timeline

### Week 1: Foundation (5 days)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Create `DirectAccessMixin` | `dataset_direct_access.py` |
| 2 | Add mixin to `Universal_Dataset` | Modified `data_loader.py` |
| 3 | Add mixin to `TimeMMD_Dataset` | Modified `time_mmd_dataset.py` |
| 4 | Write unit tests | `test_dataset_direct_access.py` |
| 5 | Verify with real datasets | Test report |

### Week 2: Tensor Cache Integration (5 days)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Implement `_build_shared_tables_direct()` | Modified `tensor_cache.py` |
| 3 | Add feature flags and auto-detection | CLI integration |
| 4 | Write integration tests | Test suite |
| 5 | Benchmark and tune | Performance report |

### Week 3: Parallel Processing (5 days)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Implement parallel worker functions | `tensor_cache_parallel.py` |
| 3 | Implement result merging | Complete parallel implementation |
| 4 | Test on multi-core machines | Scaling test report |
| 5 | Documentation | Updated docs |

### Week 4: Polish & Release (3-5 days)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Edge case handling | Robust error handling |
| 2 | Final benchmarking | Comprehensive benchmark report |
| 3 | Documentation update | User guide |
| 4-5 | Code review & merge | PR merged |

---

## 9. Expected Results

### Performance Projections

| Dataset | Current | Direct Access | + Parallel | Total Speedup |
|---------|---------|---------------|------------|---------------|
| 100k samples | 30s | 4-6s | 1.5-2s | **15-20x** |
| 300k samples | 90s | 9-15s | 3-5s | **18-30x** |
| 726k samples | 220s | 22-35s | 6-10s | **22-37x** |

### Memory Impact

| Method | Peak Memory | Notes |
|--------|-------------|-------|
| Standard | ~2 GB | Per-sample overhead |
| Direct Access | ~1.8 GB | More efficient allocation |
| Parallel (4 workers) | ~2.2 GB | Slightly higher during merge |

### Code Complexity

| Component | Lines of Code | Complexity |
|-----------|---------------|------------|
| DirectAccessMixin | ~200 | Low |
| Direct access generator | ~300 | Medium |
| Parallel processing | ~250 | Medium |
| Tests | ~400 | Low |
| **Total new code** | **~1,150** | - |

---

## 10. Summary

### Key Changes

1. **Non-breaking dataset changes**: Add `DirectAccessMixin` to expose underlying arrays
2. **New generator methods**: Direct array access and parallel processing
3. **Auto-detection**: Generator automatically uses fastest available method
4. **Backward compatible**: All existing code continues to work

### Expected Outcome

**300k samples generation time:**
- Before: 90 seconds
- After: 3-5 seconds
- **Speedup: 18-30x**

This directly addresses your time constraint by making tensor cache generation **dramatically faster** without requiring architectural changes or breaking existing functionality.

### Next Steps

1. **Approve plan**
2. **Begin Week 1**: Create `DirectAccessMixin`
3. **Iterate**: Benchmark after each phase to validate projections
4. **Deploy**: Enable by default once validated

Would you like me to start implementing this plan?
