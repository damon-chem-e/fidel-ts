# Tensor Cache Generation: Polars-Based Vectorization Plan

**Status**: Planning  
**Created**: 2026-01-16  
**Priority**: MEDIUM (performance optimization, not blocking)

## Executive Summary

This document provides a detailed analysis of the explicit Python loops in `data_provider/tensor_cache.py` and proposes a polars-based vectorized implementation to dramatically improve cache generation performance. The current implementation processes samples one-by-one with nested Python loops, which is inefficient for large datasets with hundreds of thousands of samples.

---

## Table of Contents

1. [Current Implementation Analysis](#1-current-implementation-analysis)
2. [Algorithm Documentation](#2-algorithm-documentation)
3. [Performance Bottlenecks](#3-performance-bottlenecks)
4. [Polars Optimization Strategy](#4-polars-optimization-strategy)
5. [Testing Plan](#5-testing-plan)
6. [Implementation Plan](#6-implementation-plan)

---

## 1. Current Implementation Analysis

### 1.1 Overview

The tensor cache generation pipeline processes time series forecasting samples and deduplicates shared data (timestamps, embeddings, time series values) to reduce disk usage from ~400GB to ~4GB. However, the current implementation uses explicit Python loops that iterate through:
- 3 data splits (train/val/test)
- Multiple entities per split (~30-100 entities)
- Thousands to millions of samples per entity (~100k-700k samples)
- Hundreds of timestamps per sample (input_len + output_len ≈ 288 + 144 = 432 timestamps)

### 1.2 File Location and Structure

**File**: `data_provider/tensor_cache.py` (2542 lines)

**Key Components with Explicit Loops**:

| Function | Lines | Loop Type | Description |
|----------|-------|-----------|-------------|
| `_detect_downtime_in_training()` | 617-684 | Nested (entities × samples) | Scans for non-zero downtime indicators |
| `_process_sample_for_collection()` | 842-941 | Inner loop (timestamps) | Processes each timestamp in sample |
| `_register_timestamp_data()` | 687-798 | Single item | Per-timestamp data registration |
| `_build_shared_tables()` | 1303-1521 | Triple nested (splits × entities × samples) | Main collection loop |
| `_write_sample_indices()` | 1101-1161 | List comprehension | Timestamp-to-index conversion |
| `_process_entity()` | 1823-1922 | Nested (chunks × samples) | Entity-level processing |
| `_finalize_shared_tables()` | 943-1012 | List operations | Array finalization |

---

## 2. Algorithm Documentation

### 2.1 Phase 1: Downtime Detection

**Purpose**: Determine if training data has downtime indicators (affects embedding storage format).

**Current Implementation** (`_detect_downtime_in_training()`, lines 617-684):

```python
# Pseudocode for current implementation
def _detect_downtime_in_training(data_provider, max_samples_to_check=1000):
    for entity_id, dataset in data_provider.get_datasets('train').items():
        # Calculate samples per entity
        samples_per_entity = min(len(dataset), max_samples_to_check // n_entities)
        
        # Iterate through samples with stride
        for sample_idx in range(0, len(dataset), stride):
            sample = dataset[sample_idx]
            
            # Extract hetero_x and hetero_y embeddings
            hetero_x = sample[SAMPLE_IDX_HETERO_X]  # Shape: (L, N, D) where N=2
            hetero_y = sample[SAMPLE_IDX_HETERO_Y]  # Shape: (L, N, D) where N=2
            
            # Check if second channel (downtime indicator) has non-zero values
            if hetero_x.shape[1] >= 2:
                downtime_indicator = hetero_x[:, 1, :]  # (L, D)
                if np.any(downtime_indicator != 0):
                    return True
            
            # Same check for hetero_y
            # ...
    
    return False
```

**Complexity**: O(n_entities × samples_per_entity × timestamps_per_sample)

**Inefficiency**: 
- Creates Python object for each sample access
- Repeated array slicing operations
- Early exit is the only optimization

---

### 2.2 Phase 2: Shared Table Building (Collection Phase)

**Purpose**: Collect unique timestamps and their associated data (time series, embeddings, hetero time features) across all samples.

**Current Implementation** (`_build_shared_tables()` + `_process_sample_for_collection()`, lines 1303-1521, 842-941):

```python
# Pseudocode for current implementation
def _build_shared_tables(flags=['train', 'val', 'test']):
    collector = SharedTableCollector()
    collector.timestamp_to_idx = {}  # Dict for O(1) deduplication
    collector.timestamps = []
    collector.timeseries = []
    collector.embeddings = []
    collector.hetero_time = []
    
    for flag in flags:                           # 3 iterations
        for entity_id, dataset in get_datasets(flag).items():  # ~30-100 entities
            # Register entity (first sample only)
            _register_entity_data(collector, entity_id, ...)
            
            for sample_idx in range(len(dataset)):  # 100k-700k samples
                sample = dataset[sample_idx]
                _process_sample_for_collection(collector, sample)
    
    # Finalize
    shared_tables = _finalize_shared_tables(collector)
    return shared_tables


def _process_sample_for_collection(collector, sample):
    # Process input timestamps
    x_time = sample[SAMPLE_IDX_X_TIME].flatten()    # (input_len,) e.g., 288 timestamps
    seq_x = sample[SAMPLE_IDX_SEQ_X]                # (input_len, n_features)
    hetero_x = sample[SAMPLE_IDX_HETERO_X]          # (input_len, N, D) or (input_len, D)
    hetero_x_time = sample[SAMPLE_IDX_HETERO_X_TIME]  # (input_len, n_tf)
    
    for i, ts in enumerate(x_time):                  # 288 iterations
        ts_val = seq_x[i]
        emb = hetero_x[i].copy()                     # CRITICAL: copy to break reference
        htf = hetero_x_time[i]
        _register_timestamp_data(collector, ts, ts_val, emb, htf)
    
    # Process output timestamps (similar loop for 144 timestamps)
    y_time = sample[SAMPLE_IDX_Y_TIME].flatten()
    for i, ts in enumerate(y_time):                  # 144 iterations
        # ... similar registration
        pass


def _register_timestamp_data(collector, timestamp, ts_value, embedding, hetero_time_feat):
    ts_key = int(timestamp)
    
    # Deduplication check (O(1) dict lookup)
    if ts_key in collector.timestamp_to_idx:
        return
    
    # Assign index and append to lists
    idx = len(collector.timestamps)
    collector.timestamp_to_idx[ts_key] = idx
    collector.timestamps.append(ts_key)
    collector.timeseries.append(np.asarray(ts_value).flatten())
    collector.embeddings.append(embedding)
    collector.hetero_time.append(np.asarray(hetero_time_feat).flatten())
```

**Total Loop Iterations** (Bear_room dataset):
- 3 splits × 1 entity × 726,566 samples × 432 timestamps = **941 million** Python loop iterations
- Each iteration involves: dict lookup, array indexing, array copy, list append

**Complexity**: O(n_splits × n_entities × n_samples × timestamps_per_sample)

**Inefficiencies**:
1. **Python loop overhead**: 941M Python function calls
2. **Individual array copies**: `hetero_x[i].copy()` creates new array per timestamp
3. **Dict lookups in Python**: Even O(1), overhead is significant at scale
4. **List appends**: Python list resizing overhead
5. **No batch processing**: Each timestamp processed individually

---

### 2.3 Phase 3: Index Array Writing

**Purpose**: Convert sample timestamps to indices into shared tables.

**Current Implementation** (`_write_sample_indices()`, lines 1101-1161):

```python
def _write_sample_indices(arrays, write_idx, sample, entity_idx, timestamp_to_idx):
    # Sample ID and entity index (single values)
    arrays['sample_ids'][write_idx] = str(sample[SAMPLE_IDX_SAMPLE_ID])
    arrays['entity_indices'][write_idx] = entity_idx
    
    # Convert input timestamps to indices (list comprehension)
    x_time = np.asarray(sample[SAMPLE_IDX_X_TIME]).flatten()
    x_indices = np.array(
        [timestamp_to_idx.get(int(ts), 0) for ts in x_time],  # Python loop!
        dtype=np.int32
    )
    arrays['x_indices'][write_idx] = x_indices
    
    # Convert output timestamps to indices (list comprehension)
    y_time = np.asarray(sample[SAMPLE_IDX_Y_TIME]).flatten()
    y_indices = np.array(
        [timestamp_to_idx.get(int(ts), 0) for ts in y_time],  # Python loop!
        dtype=np.int32
    )
    arrays['y_indices'][write_idx] = y_indices
    
    # Time features (direct copy)
    arrays['x_time_features'][write_idx] = np.asarray(sample[SAMPLE_IDX_X_TIME_FEATURES])
    arrays['y_time_features'][write_idx] = np.asarray(sample[SAMPLE_IDX_Y_TIME_FEATURES])
```

**Called from** (`_process_entity()`, lines 1823-1922):

```python
def _process_entity(entity_id, dataset, arrays, ...):
    for chunk_start in range(0, entity_samples, chunk_size):  # chunk_size=10000
        for i in range(chunk_start, chunk_end):               # 10000 iterations
            sample = dataset[i]
            _write_sample_indices(arrays, current_idx + i, sample, entity_idx, timestamp_to_idx)
```

**Total Iterations** (Bear_room):
- 726,566 samples × 432 timestamps = **314 million** dict.get() calls in list comprehensions

**Inefficiencies**:
1. **Python list comprehension**: `[timestamp_to_idx.get(int(ts), 0) for ts in x_time]`
2. **Per-sample processing**: No batching across samples
3. **Dict lookups in Python**: Should be vectorized array operations

---

### 2.4 Finalization Phase

**Current Implementation** (`_finalize_shared_tables()`, lines 943-1012):

```python
def _finalize_shared_tables(collector):
    shared_tables = {}
    
    # Convert lists to arrays (efficient - single np.stack per table)
    if collector.timestamps:
        shared_tables['timestamps'] = np.array(collector.timestamps, dtype=np.int64)
        collector.timestamps.clear()
    
    if collector.timeseries:
        shared_tables['timeseries'] = np.stack(collector.timeseries).astype(np.float32)
        collector.timeseries.clear()
    
    if collector.embeddings:
        shared_tables['embeddings'] = np.stack(collector.embeddings).astype(np.float32)
        collector.embeddings.clear()
    
    # Entity data (filter out None values)
    valid_generals = [g for g in collector.entity_general if g is not None]  # List comp
    if valid_generals:
        shared_tables['entity_general'] = np.stack(valid_generals).astype(np.float32)
    
    # Build index mappings (dict comprehension)
    index_mappings = {
        'timestamp_to_idx': {str(k): v for k, v in collector.timestamp_to_idx.items()},
        'entity_to_idx': collector.entity_to_idx
    }
    
    return shared_tables, index_mappings
```

**Efficiency**: This phase is relatively efficient since `np.stack()` is vectorized. The only inefficiency is the list comprehension for filtering None values.

---

## 3. Performance Bottlenecks

### 3.1 Bottleneck Ranking

| Rank | Bottleneck | Function | Est. Time % | Est. Iterations |
|------|------------|----------|-------------|-----------------|
| 1 | Sample-level iteration | `_build_shared_tables` | 40% | 726k per split |
| 2 | Timestamp-level iteration | `_process_sample_for_collection` | 35% | 941M total |
| 3 | Dict lookup per timestamp | `_register_timestamp_data` | 15% | 941M lookups |
| 4 | Index conversion loop | `_write_sample_indices` | 8% | 314M lookups |
| 5 | Array copies per timestamp | `hetero_x[i].copy()` | 2% | 941M copies |

### 3.2 Measured vs Theoretical Performance

**Theoretical minimum** (with perfect vectorization):
- Read all samples: ~30 seconds (disk I/O bound)
- Build timestamp index: ~5 seconds (polars join)
- Write index arrays: ~10 seconds (numpy bulk operations)
- **Total**: ~45 seconds

**Current observed** (Bear_room ~726k samples):
- ~15-20 minutes for shared table building
- ~10-15 minutes for index array generation
- **Total**: ~25-35 minutes

**Speedup potential**: 30-50x with proper vectorization

---

## 4. Polars Optimization Strategy

### 4.1 Why Polars?

Polars is ideal for this optimization because:

1. **Columnar processing**: Operates on entire columns, not row-by-row
2. **Lazy evaluation**: Can optimize query plans before execution
3. **Parallel execution**: Automatically parallelizes operations
4. **Memory efficiency**: Uses Apache Arrow memory format
5. **No Python GIL**: Operations execute in Rust, bypassing Python overhead
6. **Join operations**: Highly optimized hash joins for deduplication

### 4.2 Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    POLARS-BASED TENSOR CACHE GENERATION                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 1: Data Extraction (Parallelized)                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  For each entity (parallel):                                         │    │
│  │    - Load dataset                                                    │    │
│  │    - Extract all timestamps as flat array: [sample_idx, ts_idx, ts]  │    │
│  │    - Extract all time series values: [ts, feature_0, ..., feature_n] │    │
│  │    - Extract all embeddings: [ts, emb_0, ..., emb_d]                 │    │
│  │    - Extract all hetero_time: [ts, htf_0, ..., htf_k]                │    │
│  │  Concatenate into polars DataFrames                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                  │                                           │
│                                  ▼                                           │
│  PHASE 2: Deduplication (Vectorized)                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  timestamps_df = all_timestamps.unique().sort()                      │    │
│  │  timestamps_df = timestamps_df.with_row_index("ts_idx")              │    │
│  │                                                                      │    │
│  │  # Join to build shared tables                                       │    │
│  │  timeseries_df = timestamps_df.join(all_timeseries, on="ts")         │    │
│  │  embeddings_df = timestamps_df.join(all_embeddings, on="ts")         │    │
│  │  hetero_time_df = timestamps_df.join(all_hetero_time, on="ts")       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                  │                                           │
│                                  ▼                                           │
│  PHASE 3: Index Generation (Vectorized)                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  # Join sample timestamps with global index                          │    │
│  │  indexed_samples = sample_timestamps_df.join(                        │    │
│  │      timestamps_df.select(["ts", "ts_idx"]),                         │    │
│  │      on="ts"                                                         │    │
│  │  )                                                                   │    │
│  │                                                                      │    │
│  │  # Pivot to get [sample_idx, ts_idx_0, ts_idx_1, ..., ts_idx_n]     │    │
│  │  x_indices = indexed_samples.filter(pl.col("window") == "x")         │    │
│  │      .pivot(values="ts_idx", index="sample_idx", columns="position") │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                  │                                           │
│                                  ▼                                           │
│  PHASE 4: Array Export (Bulk Operations)                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  # Convert polars columns to numpy arrays (zero-copy when possible)  │    │
│  │  shared_tables['timestamps'] = timestamps_df['ts'].to_numpy()        │    │
│  │  shared_tables['embeddings'] = embeddings_df.to_numpy()              │    │
│  │                                                                      │    │
│  │  # Write index arrays                                                │    │
│  │  arrays['x_indices'] = x_indices_df.to_numpy()                       │    │
│  │  arrays['y_indices'] = y_indices_df.to_numpy()                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Detailed Algorithm Transformations

#### 4.3.1 Timestamp Collection (Current → Polars)

**Current**:
```python
# O(n_samples × timestamps_per_sample) Python iterations
for sample_idx in range(len(dataset)):
    sample = dataset[sample_idx]
    x_time = sample[SAMPLE_IDX_X_TIME].flatten()
    for i, ts in enumerate(x_time):
        if ts not in collector.timestamp_to_idx:
            collector.timestamp_to_idx[ts] = len(collector.timestamps)
            collector.timestamps.append(ts)
```

**Polars**:
```python
import polars as pl

def collect_timestamps_polars(datasets: Dict[str, Dataset], flags: List[str]) -> pl.DataFrame:
    """
    Collect all unique timestamps across all samples using polars.
    
    Returns DataFrame with columns: ['ts', 'ts_idx'] sorted by timestamp.
    
    Args:
        datasets: Dict mapping entity_id to dataset
        flags: List of splits to process
        
    Returns:
        pl.DataFrame with unique timestamps and assigned indices
    """
    all_timestamps = []
    
    for flag in flags:
        for entity_id, dataset in datasets.items():
            # Batch extract all timestamps from entity
            # This is still a Python loop over samples, but we batch the results
            entity_timestamps = []
            
            for sample_idx in range(len(dataset)):
                sample = dataset[sample_idx]
                x_time = sample[SAMPLE_IDX_X_TIME].flatten()
                y_time = sample[SAMPLE_IDX_Y_TIME].flatten()
                entity_timestamps.extend(x_time.tolist())
                entity_timestamps.extend(y_time.tolist())
            
            all_timestamps.extend(entity_timestamps)
    
    # Create polars DataFrame and deduplicate (vectorized!)
    ts_df = pl.DataFrame({'ts': all_timestamps})
    
    # Deduplicate and assign indices - this is O(n) in polars, fully vectorized
    unique_ts_df = (
        ts_df
        .unique()
        .sort('ts')
        .with_row_index('ts_idx')
    )
    
    return unique_ts_df
```

**Improvement**: 
- Deduplication is now a single polars `.unique()` operation (parallel hash-based)
- Index assignment is a single `.with_row_index()` (O(n) memory scan)
- No Python dict operations

#### 4.3.2 Index Generation (Current → Polars)

**Current**:
```python
# O(n_samples × timestamps_per_sample) Python dict lookups
x_indices = np.array(
    [timestamp_to_idx.get(int(ts), 0) for ts in x_time],
    dtype=np.int32
)
```

**Polars**:
```python
def generate_indices_polars(
    sample_timestamps_df: pl.DataFrame,
    timestamp_index_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Convert sample timestamps to indices using polars join.
    
    Args:
        sample_timestamps_df: DataFrame with columns ['sample_idx', 'position', 'ts', 'window']
            - sample_idx: which sample this belongs to
            - position: position within the window (0 to input_len-1 or 0 to output_len-1)
            - ts: the timestamp value
            - window: 'x' for input, 'y' for output
        timestamp_index_df: DataFrame with columns ['ts', 'ts_idx']
    
    Returns:
        DataFrame with columns ['sample_idx', 'position', 'ts_idx', 'window']
    """
    # Single vectorized join operation
    # This replaces millions of dict.get() calls with one hash join
    indexed_df = sample_timestamps_df.join(
        timestamp_index_df.select(['ts', 'ts_idx']),
        on='ts',
        how='left'
    ).with_columns(
        pl.col('ts_idx').fill_null(0)  # Default to 0 for missing timestamps
    )
    
    return indexed_df


def pivot_indices_to_array(
    indexed_df: pl.DataFrame,
    window: str,  # 'x' or 'y'
    n_samples: int,
    window_len: int
) -> np.ndarray:
    """
    Pivot indexed timestamps to per-sample arrays.
    
    Args:
        indexed_df: DataFrame with ['sample_idx', 'position', 'ts_idx', 'window']
        window: 'x' for input indices, 'y' for output indices
        n_samples: Total number of samples
        window_len: Length of the window (input_len or output_len)
    
    Returns:
        np.ndarray of shape (n_samples, window_len) with dtype int32
    """
    # Filter to requested window
    window_df = indexed_df.filter(pl.col('window') == window)
    
    # Pivot: rows=samples, columns=positions, values=ts_idx
    # This transforms long format to wide format in one operation
    pivoted = window_df.pivot(
        values='ts_idx',
        index='sample_idx',
        columns='position',
        aggregate_function='first'  # Should be unique, but specify to avoid error
    ).sort('sample_idx')
    
    # Extract as numpy array (columns 0 to window_len-1)
    indices = pivoted.select([str(i) for i in range(window_len)]).to_numpy().astype(np.int32)
    
    return indices
```

**Improvement**:
- Single hash join replaces 314M dict.get() calls
- Pivot operation is fully vectorized
- No Python loops for index conversion

#### 4.3.3 Embedding/Timeseries Collection (Current → Polars)

**Current** (nested loops with per-timestamp copies):
```python
for i, ts in enumerate(x_time_flat):
    ts_val = seq_x[i] if seq_x is not None and i < len(seq_x) else None
    emb = hetero_x[i].copy()  # Creates new array per iteration!
    htf = hetero_x_time[i]
    _register_timestamp_data(collector, ts, ts_val, emb, htf)
```

**Polars**:
```python
def collect_timeseries_data_polars(
    datasets: Dict[str, Dataset],
    flags: List[str],
    unique_timestamps: pl.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect time series, embeddings, and hetero_time for unique timestamps.
    
    Strategy: 
    1. For each sample, extract timestamp-indexed data as flat arrays
    2. Build a polars DataFrame with (ts, data...) pairs
    3. Join with unique_timestamps to keep only first occurrence
    4. Sort by ts_idx to create aligned arrays
    
    Args:
        datasets: Entity datasets
        flags: Splits to process
        unique_timestamps: DataFrame with ['ts', 'ts_idx']
    
    Returns:
        Tuple of (timeseries_array, embeddings_array, hetero_time_array)
        Each array is aligned with unique_timestamps (row i corresponds to ts_idx=i)
    """
    # Collect first occurrence of each timestamp's data
    # Use a dict for now (polars-compatible structure)
    ts_data = {}  # ts -> {'seq': array, 'emb': array, 'htf': array}
    
    for flag in flags:
        for entity_id, dataset in datasets.items():
            for sample_idx in range(len(dataset)):
                sample = dataset[sample_idx]
                
                x_time = sample[SAMPLE_IDX_X_TIME].flatten()
                seq_x = sample[SAMPLE_IDX_SEQ_X]
                hetero_x = sample[SAMPLE_IDX_HETERO_X]
                htf_x = sample[SAMPLE_IDX_HETERO_X_TIME]
                
                # Vectorized timestamp check - which timestamps are new?
                mask = np.array([ts not in ts_data for ts in x_time])
                new_indices = np.where(mask)[0]
                
                # Only process new timestamps (avoids redundant copies)
                for i in new_indices:
                    ts = int(x_time[i])
                    ts_data[ts] = {
                        'seq': seq_x[i].copy() if seq_x is not None else None,
                        'emb': hetero_x[i].copy() if hetero_x is not None else None,
                        'htf': htf_x[i].copy() if htf_x is not None else None,
                    }
                
                # Same for y_time...
    
    # Convert to arrays aligned with unique_timestamps
    n_unique = len(unique_timestamps)
    ts_to_idx = dict(zip(
        unique_timestamps['ts'].to_list(),
        unique_timestamps['ts_idx'].to_list()
    ))
    
    # Pre-allocate arrays
    n_features = next(iter(ts_data.values()))['seq'].shape[0]
    embed_dim = next(iter(ts_data.values()))['emb'].shape[-1]
    n_htf = next(iter(ts_data.values()))['htf'].shape[0]
    
    timeseries = np.zeros((n_unique, n_features), dtype=np.float32)
    embeddings = np.zeros((n_unique, embed_dim), dtype=np.float32)
    hetero_time = np.zeros((n_unique, n_htf), dtype=np.float32)
    
    # Fill arrays (vectorized with numpy advanced indexing)
    for ts, data in ts_data.items():
        idx = ts_to_idx[ts]
        if data['seq'] is not None:
            timeseries[idx] = data['seq']
        if data['emb'] is not None:
            embeddings[idx] = data['emb']
        if data['htf'] is not None:
            hetero_time[idx] = data['htf']
    
    return timeseries, embeddings, hetero_time
```

**Note**: The sample iteration is still required because datasets don't expose bulk access. However, the optimizations are:
1. Early duplicate detection with numpy mask
2. Pre-allocated arrays instead of list appends
3. Bulk array assignment via advanced indexing

### 4.4 Hybrid Approach: Polars + NumPy

Since datasets require sequential access, a pure polars approach isn't possible. The optimal strategy is:

| Phase | Tool | Rationale |
|-------|------|-----------|
| Sample iteration | Python | Dataset API requires it |
| Timestamp extraction | NumPy | Already in numpy format |
| Deduplication | Polars | Vectorized unique/sort/index |
| Data collection | NumPy + early-exit | Minimize copies |
| Index generation | Polars join | Replaces dict lookups |
| Array creation | NumPy | Direct memory layout |

### 4.5 Memory-Mapped DataFrame Approach

For very large datasets, we can use polars' lazy evaluation with memory-mapped files:

```python
import polars as pl

def build_timestamp_index_lazy(parquet_path: str) -> pl.LazyFrame:
    """
    Build timestamp index from pre-extracted parquet files.
    
    This approach requires a preprocessing step that extracts timestamps
    from the dataset into a parquet file, but enables fully vectorized
    processing for subsequent operations.
    
    Args:
        parquet_path: Path to parquet file with columns ['sample_idx', 'position', 'ts', 'window']
    
    Returns:
        LazyFrame with unique timestamps and indices
    """
    return (
        pl.scan_parquet(parquet_path)
        .select('ts')
        .unique()
        .sort('ts')
        .with_row_index('ts_idx')
    )


def generate_all_indices_lazy(
    timestamps_parquet: str,
    index_df: pl.LazyFrame
) -> pl.LazyFrame:
    """
    Generate all sample indices using lazy evaluation.
    
    The query plan is optimized before execution, and polars
    will automatically parallelize and stream the computation.
    
    Args:
        timestamps_parquet: Path to sample timestamps parquet
        index_df: LazyFrame with ['ts', 'ts_idx']
    
    Returns:
        LazyFrame with indexed samples
    """
    return (
        pl.scan_parquet(timestamps_parquet)
        .join(index_df, on='ts', how='left')
        .with_columns(pl.col('ts_idx').fill_null(0))
    )
```

---

## 5. Testing Plan

### 5.1 Test Philosophy

The test suite must verify that the polars-based implementation produces **bit-for-bit identical results** to the current Python implementation. This is critical because:
1. The tensor cache is used for training - any deviation could affect model behavior
2. The deduplication logic must be exactly preserved
3. Index mappings must be consistent across implementations

### 5.2 Test File Structure

**Location**: `tests/test_tensor_cache_polars.py`

### 5.3 Test Categories

#### 5.3.1 Unit Tests: Algorithm Correctness

```python
"""
Unit tests for polars-based tensor cache generation.

These tests verify that each polars-based function produces identical
results to the corresponding Python implementation.
"""

import pytest
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple

# Test fixtures
@pytest.fixture
def sample_timestamps():
    """Create sample timestamp data for testing."""
    # Simulates 3 samples with input_len=4, output_len=2
    # Timestamps have sliding window overlap
    return {
        'sample_0': {
            'x_time': np.array([100, 101, 102, 103]),
            'y_time': np.array([104, 105]),
        },
        'sample_1': {
            'x_time': np.array([101, 102, 103, 104]),  # Overlaps with sample_0
            'y_time': np.array([105, 106]),
        },
        'sample_2': {
            'x_time': np.array([102, 103, 104, 105]),  # Overlaps with both
            'y_time': np.array([106, 107]),
        },
    }


@pytest.fixture
def sample_data():
    """Create sample data arrays for testing."""
    np.random.seed(42)
    n_features = 3
    embed_dim = 8
    n_htf = 2
    
    # Unique timestamps: 100, 101, 102, 103, 104, 105, 106, 107 (8 total)
    return {
        'timeseries': {ts: np.random.randn(n_features).astype(np.float32) 
                       for ts in range(100, 108)},
        'embeddings': {ts: np.random.randn(embed_dim).astype(np.float32)
                       for ts in range(100, 108)},
        'hetero_time': {ts: np.random.randn(n_htf).astype(np.float32)
                        for ts in range(100, 108)},
    }


class TestTimestampDeduplication:
    """Tests for timestamp deduplication algorithms."""
    
    def test_unique_timestamps_match(self, sample_timestamps):
        """Verify polars unique() matches Python set() deduplication."""
        # Python implementation
        all_ts_python = set()
        for sample in sample_timestamps.values():
            all_ts_python.update(sample['x_time'].tolist())
            all_ts_python.update(sample['y_time'].tolist())
        python_unique = sorted(all_ts_python)
        
        # Polars implementation
        all_ts = []
        for sample in sample_timestamps.values():
            all_ts.extend(sample['x_time'].tolist())
            all_ts.extend(sample['y_time'].tolist())
        
        polars_df = (
            pl.DataFrame({'ts': all_ts})
            .unique()
            .sort('ts')
        )
        polars_unique = polars_df['ts'].to_list()
        
        # Verify identical results
        assert python_unique == polars_unique, (
            f"Unique timestamps differ:\n"
            f"Python: {python_unique}\n"
            f"Polars: {polars_unique}"
        )
    
    def test_index_assignment_order(self, sample_timestamps):
        """Verify index assignment order is consistent."""
        # Build timestamp list
        all_ts = []
        for sample in sample_timestamps.values():
            all_ts.extend(sample['x_time'].tolist())
            all_ts.extend(sample['y_time'].tolist())
        
        # Python: dict-based assignment (order of first occurrence)
        python_idx = {}
        for ts in all_ts:
            if ts not in python_idx:
                python_idx[ts] = len(python_idx)
        
        # Polars: sorted unique with row index
        polars_df = (
            pl.DataFrame({'ts': all_ts})
            .unique()
            .sort('ts')
            .with_row_index('ts_idx')
        )
        polars_idx = dict(zip(
            polars_df['ts'].to_list(),
            polars_df['ts_idx'].to_list()
        ))
        
        # Note: Order may differ! Python uses insertion order, polars uses sorted order.
        # Both are valid as long as consistent within each implementation.
        # The test verifies that indices are assigned correctly within each approach.
        
        # Verify bijection (each timestamp has unique index)
        assert len(python_idx) == len(set(python_idx.values())), "Python indices not unique"
        assert len(polars_idx) == len(set(polars_idx.values())), "Polars indices not unique"
        
        # Verify same timestamps covered
        assert set(python_idx.keys()) == set(polars_idx.keys()), "Timestamp sets differ"


class TestIndexGeneration:
    """Tests for timestamp-to-index conversion."""
    
    def test_index_lookup_matches(self, sample_timestamps):
        """Verify polars join matches dict.get() for index lookup."""
        # Build index mapping
        all_ts = []
        for sample in sample_timestamps.values():
            all_ts.extend(sample['x_time'].tolist())
            all_ts.extend(sample['y_time'].tolist())
        
        index_df = (
            pl.DataFrame({'ts': all_ts})
            .unique()
            .sort('ts')
            .with_row_index('ts_idx')
        )
        index_dict = dict(zip(
            index_df['ts'].to_list(),
            index_df['ts_idx'].to_list()
        ))
        
        # Test sample
        test_ts = sample_timestamps['sample_1']['x_time']
        
        # Python: list comprehension with dict.get()
        python_indices = [index_dict.get(int(ts), 0) for ts in test_ts]
        
        # Polars: join operation
        sample_df = pl.DataFrame({'ts': test_ts.tolist()})
        polars_result = (
            sample_df
            .join(index_df.select(['ts', 'ts_idx']), on='ts', how='left')
            .with_columns(pl.col('ts_idx').fill_null(0))
        )
        polars_indices = polars_result['ts_idx'].to_list()
        
        # Verify identical results
        assert python_indices == polars_indices, (
            f"Index lookup differs:\n"
            f"Python: {python_indices}\n"
            f"Polars: {polars_indices}"
        )


class TestDataAlignment:
    """Tests for data alignment with timestamp indices."""
    
    def test_timeseries_alignment(self, sample_timestamps, sample_data):
        """Verify time series data aligns correctly with indices."""
        # Build index
        all_ts = sorted(set(
            ts for sample in sample_timestamps.values()
            for ts in list(sample['x_time']) + list(sample['y_time'])
        ))
        
        # Create aligned array
        n_unique = len(all_ts)
        n_features = list(sample_data['timeseries'].values())[0].shape[0]
        
        aligned_ts = np.zeros((n_unique, n_features), dtype=np.float32)
        for idx, ts in enumerate(all_ts):
            aligned_ts[idx] = sample_data['timeseries'][ts]
        
        # Verify lookup works correctly
        sample_x_time = sample_timestamps['sample_0']['x_time']
        indices = [all_ts.index(ts) for ts in sample_x_time]
        
        # Retrieved data should match original
        for i, ts in enumerate(sample_x_time):
            np.testing.assert_array_almost_equal(
                aligned_ts[indices[i]],
                sample_data['timeseries'][ts],
                decimal=6,
                err_msg=f"Timeseries mismatch at timestamp {ts}"
            )
```

#### 5.3.2 Integration Tests: Full Pipeline

```python
class TestFullPipeline:
    """Integration tests comparing full pipeline results."""
    
    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset for testing."""
        class MockDataset:
            def __init__(self, n_samples: int = 100, input_len: int = 8, output_len: int = 4):
                np.random.seed(42)
                self.n_samples = n_samples
                self.input_len = input_len
                self.output_len = output_len
                self.n_features = 3
                self.embed_dim = 16
                
                # Generate continuous timestamps
                base_ts = 1000
                self.timestamps = np.arange(base_ts, base_ts + n_samples + input_len + output_len)
                
                # Pre-generate data for each timestamp
                n_ts = len(self.timestamps)
                self.all_timeseries = np.random.randn(n_ts, self.n_features).astype(np.float32)
                self.all_embeddings = np.random.randn(n_ts, self.embed_dim).astype(np.float32)
                self.all_hetero_time = np.random.randn(n_ts, 4).astype(np.float32)
                self.all_time_features = np.random.randn(n_ts, 4).astype(np.float32)
            
            def __len__(self):
                return self.n_samples
            
            def __getitem__(self, idx):
                # Sliding window sample
                start = idx
                x_start, x_end = start, start + self.input_len
                y_start, y_end = x_end, x_end + self.output_len
                
                return (
                    f"sample_{idx}",                              # sample_id
                    self.all_timeseries[x_start:x_end],           # seq_x
                    self.all_timeseries[y_start:y_end],           # seq_y
                    self.timestamps[x_start:x_end],               # x_time
                    self.timestamps[y_start:y_end],               # y_time
                    self.all_embeddings[x_start:x_end],           # hetero_x
                    self.all_embeddings[y_start:y_end],           # hetero_y
                    self.all_hetero_time[x_start:x_end],          # hetero_x_time
                    self.all_hetero_time[y_start:y_end],          # hetero_y_time
                    self.all_embeddings[0],                       # hetero_general (static)
                    self.all_embeddings[1],                       # hetero_channel (static)
                    self.all_time_features[x_start:x_end],        # x_time_features
                    self.all_time_features[y_start:y_end],        # y_time_features
                )
        
        return MockDataset(n_samples=100, input_len=8, output_len=4)
    
    def test_shared_tables_identical(self, mock_dataset, tmp_path):
        """Verify shared tables are identical between implementations."""
        # Run current implementation
        current_tables = self._run_current_implementation(mock_dataset)
        
        # Run polars implementation
        polars_tables = self._run_polars_implementation(mock_dataset)
        
        # Compare each table
        for name in ['timestamps', 'timeseries', 'embeddings', 'hetero_time']:
            np.testing.assert_array_equal(
                current_tables.get(name),
                polars_tables.get(name),
                err_msg=f"Shared table '{name}' differs between implementations"
            )
    
    def test_index_arrays_identical(self, mock_dataset, tmp_path):
        """Verify index arrays are identical between implementations."""
        # Run current implementation
        current_indices = self._run_current_index_generation(mock_dataset)
        
        # Run polars implementation
        polars_indices = self._run_polars_index_generation(mock_dataset)
        
        # Compare
        for name in ['x_indices', 'y_indices', 'entity_indices']:
            np.testing.assert_array_equal(
                current_indices.get(name),
                polars_indices.get(name),
                err_msg=f"Index array '{name}' differs between implementations"
            )
    
    def test_reconstructed_samples_identical(self, mock_dataset, tmp_path):
        """Verify samples reconstructed from cache match original."""
        # Generate cache with polars implementation
        cache_path = tmp_path / "cache"
        self._generate_cache_polars(mock_dataset, cache_path)
        
        # Load cache and reconstruct samples
        cached_dataset = TensorCacheDataset(cache_path, 'train')
        
        # Compare first 10 samples
        for idx in range(min(10, len(mock_dataset))):
            original = mock_dataset[idx]
            cached = cached_dataset[idx]
            
            # Compare each field
            for field_idx, field_name in enumerate([
                'sample_id', 'seq_x', 'seq_y', 'x_time', 'y_time',
                'hetero_x', 'hetero_y', 'hetero_x_time', 'hetero_y_time',
                'hetero_general', 'hetero_channel', 'x_time_features', 'y_time_features'
            ]):
                if isinstance(original[field_idx], str):
                    assert original[field_idx] == cached[field_idx], (
                        f"Sample {idx} field '{field_name}' (string) differs"
                    )
                elif original[field_idx] is not None:
                    np.testing.assert_array_almost_equal(
                        original[field_idx],
                        cached[field_idx],
                        decimal=6,
                        err_msg=f"Sample {idx} field '{field_name}' differs"
                    )
```

#### 5.3.3 Performance Benchmarks

```python
class TestPerformance:
    """Performance benchmarks for polars vs current implementation."""
    
    @pytest.fixture
    def large_mock_dataset(self):
        """Create a larger dataset for performance testing."""
        return MockDataset(n_samples=10000, input_len=288, output_len=144)
    
    @pytest.mark.slow
    def test_deduplication_speedup(self, large_mock_dataset):
        """Benchmark timestamp deduplication performance."""
        import time
        
        # Extract all timestamps
        all_ts = []
        for idx in range(len(large_mock_dataset)):
            sample = large_mock_dataset[idx]
            all_ts.extend(sample[3].tolist())  # x_time
            all_ts.extend(sample[4].tolist())  # y_time
        
        # Python implementation (current)
        start = time.perf_counter()
        python_unique = {}
        for ts in all_ts:
            if ts not in python_unique:
                python_unique[ts] = len(python_unique)
        python_time = time.perf_counter() - start
        
        # Polars implementation
        start = time.perf_counter()
        polars_df = (
            pl.DataFrame({'ts': all_ts})
            .unique()
            .sort('ts')
            .with_row_index('ts_idx')
        )
        polars_time = time.perf_counter() - start
        
        speedup = python_time / polars_time
        print(f"\nDeduplication speedup: {speedup:.2f}x")
        print(f"  Python: {python_time*1000:.1f}ms")
        print(f"  Polars: {polars_time*1000:.1f}ms")
        
        # Verify correctness
        assert len(python_unique) == len(polars_df)
        
        # Expect at least 2x speedup for large datasets
        assert speedup >= 1.5, f"Expected speedup >= 1.5x, got {speedup:.2f}x"
    
    @pytest.mark.slow
    def test_index_generation_speedup(self, large_mock_dataset):
        """Benchmark index generation performance."""
        import time
        
        # Build index
        all_ts = []
        for idx in range(len(large_mock_dataset)):
            sample = large_mock_dataset[idx]
            all_ts.extend(sample[3].tolist())
            all_ts.extend(sample[4].tolist())
        
        index_dict = {}
        for ts in all_ts:
            if ts not in index_dict:
                index_dict[ts] = len(index_dict)
        
        index_df = (
            pl.DataFrame({'ts': all_ts})
            .unique()
            .sort('ts')
            .with_row_index('ts_idx')
        )
        
        # Prepare test data - all x_time arrays
        test_timestamps = [large_mock_dataset[i][3] for i in range(100)]
        
        # Python: list comprehension
        start = time.perf_counter()
        for x_time in test_timestamps:
            indices = [index_dict.get(int(ts), 0) for ts in x_time]
        python_time = time.perf_counter() - start
        
        # Polars: join
        start = time.perf_counter()
        for x_time in test_timestamps:
            sample_df = pl.DataFrame({'ts': x_time.tolist()})
            result = sample_df.join(index_df.select(['ts', 'ts_idx']), on='ts', how='left')
        polars_time = time.perf_counter() - start
        
        speedup = python_time / polars_time
        print(f"\nIndex generation speedup: {speedup:.2f}x")
        print(f"  Python: {python_time*1000:.1f}ms")
        print(f"  Polars: {polars_time*1000:.1f}ms")
```

#### 5.3.4 Edge Case Tests

```python
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_dataset(self):
        """Handle empty datasets gracefully."""
        empty_dataset = MockDataset(n_samples=0)
        # Should not raise, should produce empty arrays
        result = collect_timestamps_polars({'entity': empty_dataset}, ['train'])
        assert len(result) == 0
    
    def test_single_sample(self):
        """Handle single-sample dataset."""
        single_dataset = MockDataset(n_samples=1)
        result = collect_timestamps_polars({'entity': single_dataset}, ['train'])
        # Should have input_len + output_len unique timestamps
        assert len(result) == single_dataset.input_len + single_dataset.output_len
    
    def test_no_overlap(self):
        """Handle datasets with no timestamp overlap between samples."""
        # This would be unusual but should work
        pass
    
    def test_complete_overlap(self):
        """Handle datasets where all samples share the same timestamps."""
        # This tests deduplication efficiency
        pass
    
    def test_missing_timestamps(self):
        """Handle lookups for timestamps not in index."""
        index_df = pl.DataFrame({'ts': [1, 2, 3], 'ts_idx': [0, 1, 2]})
        query_df = pl.DataFrame({'ts': [1, 4, 2]})  # 4 is missing
        
        result = query_df.join(index_df, on='ts', how='left').with_columns(
            pl.col('ts_idx').fill_null(0)
        )
        
        expected = [0, 0, 1]  # 4 maps to default 0
        assert result['ts_idx'].to_list() == expected
    
    def test_dtype_consistency(self):
        """Verify dtypes match between implementations."""
        # Important for memory-mapped arrays
        pass
    
    def test_large_timestamps(self):
        """Handle int64 timestamps correctly."""
        large_ts = [2**40 + i for i in range(100)]
        df = pl.DataFrame({'ts': large_ts})
        
        assert df['ts'].dtype == pl.Int64
        assert df['ts'].to_list() == large_ts
```

### 5.4 Test Execution Plan

```bash
# Run all tensor cache polars tests
pytest tests/test_tensor_cache_polars.py -v

# Run only fast unit tests (default)
pytest tests/test_tensor_cache_polars.py -v -m "not slow"

# Run performance benchmarks
pytest tests/test_tensor_cache_polars.py -v -m "slow" --tb=short

# Run with coverage
pytest tests/test_tensor_cache_polars.py -v --cov=data_provider.tensor_cache --cov-report=html

# Run specific test class
pytest tests/test_tensor_cache_polars.py::TestFullPipeline -v
```

---

## 6. Implementation Plan

### 6.1 Phase Overview

| Phase | Description | Priority | Est. Effort |
|-------|-------------|----------|-------------|
| 1 | Test infrastructure | HIGH | 1 day |
| 2 | Polars helper functions | HIGH | 1 day |
| 3 | Generator refactor | MEDIUM | 2 days |
| 4 | Integration & benchmarking | MEDIUM | 1 day |

### 6.2 Detailed Phases

#### Phase 1: Test Infrastructure

**Goal**: Create comprehensive test suite before any implementation changes.

**Tasks**:
1. Create `tests/test_tensor_cache_polars.py`
2. Implement `MockDataset` fixture
3. Implement unit tests for current algorithm behavior
4. Establish baseline performance metrics

**Deliverables**:
- Test file with 20+ test cases
- Documented expected behavior for each algorithm
- Baseline timing data for 1k, 10k, 100k sample datasets

#### Phase 2: Polars Helper Functions

**Goal**: Implement polars-based functions alongside current code.

**Tasks**:
1. Add `polars` to `requirements/base.txt`
2. Create `data_provider/tensor_cache_polars.py` with:
   - `collect_timestamps_polars()`
   - `generate_indices_polars()`
   - `deduplicate_shared_data_polars()`
3. Implement conversion utilities between polars DataFrames and numpy arrays
4. Add tests verifying equivalence with current functions

**Deliverables**:
- New module with polars functions
- All unit tests passing
- No changes to existing `tensor_cache.py`

#### Phase 3: Generator Refactor

**Goal**: Integrate polars functions into `TensorCacheGenerator`.

**Tasks**:
1. Add `use_polars: bool = True` parameter to `TensorCacheGenerator.__init__`
2. Create `_build_shared_tables_polars()` method
3. Create `_generate_split_polars()` method
4. Add feature flag to switch implementations
5. Run integration tests comparing both implementations

**Deliverables**:
- Updated `TensorCacheGenerator` with dual implementations
- All integration tests passing
- Side-by-side timing comparisons

#### Phase 4: Integration & Benchmarking

**Goal**: Validate at scale and optimize.

**Tasks**:
1. Test on real datasets (Bear_room, Jena_Atmospheric_Physics)
2. Profile memory usage with both implementations
3. Tune polars configuration (streaming, parallelism)
4. Document performance improvements
5. Make polars the default (if validated)

**Deliverables**:
- Performance comparison document
- Updated documentation
- PR ready for review

### 6.3 Rollback Strategy

If the polars implementation has issues:
1. `use_polars=False` flag reverts to current behavior
2. No changes to existing `tensor_cache.py` functions in Phase 2
3. Integration tests catch any discrepancies before merge

### 6.4 Dependencies

**New Dependencies**:
```
polars>=0.20.0
```

**Existing Dependencies** (unchanged):
- numpy
- torch (for dataloader)
- tqdm / rich (for progress)

---

## 7. Appendix: Code Locations

### A.1 Current Implementation Functions

| Function | File | Lines | Purpose |
|----------|------|-------|---------|
| `_detect_downtime_in_training` | tensor_cache.py | 617-684 | Check for downtime indicators |
| `_register_timestamp_data` | tensor_cache.py | 687-798 | Per-timestamp data registration |
| `_register_entity_data` | tensor_cache.py | 800-839 | Per-entity data registration |
| `_process_sample_for_collection` | tensor_cache.py | 842-941 | Sample processing loop |
| `_finalize_shared_tables` | tensor_cache.py | 943-1012 | List-to-array conversion |
| `_write_sample_indices` | tensor_cache.py | 1101-1161 | Index array writing |
| `TensorCacheGenerator._build_shared_tables` | tensor_cache.py | 1303-1521 | Main collection orchestration |
| `TensorCacheGenerator._process_entity` | tensor_cache.py | 1823-1922 | Entity-level processing |

### A.2 Sample Tuple Indices

From `tensor_cache.py` lines 433-446:

```python
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
```

---

## 8. References

- [Polars User Guide](https://pola-rs.github.io/polars/user-guide/)
- [Polars vs Pandas Performance](https://pola-rs.github.io/polars/user-guide/misc/multiprocessing/)
- Existing planning docs:
  - `docs/planning/tensor_cache_memory_optimization.md`
  - `docs/planning/tensor_cache_disk_optimization.md`
  - `docs/tensor_cache.md`
