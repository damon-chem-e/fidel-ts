# Tensor Cache Disk Space Optimization

**Status**: Planning → Implementation Required  
**Created**: 2026-01-14  
**Priority**: HIGH (blocking production use)

## Problem Statement

Tensor cache generation produces catastrophically large files due to data duplication
inherent in sliding window sampling. A single dataset (Bear_room train split) produced
**408 GB** of cache data, exceeding a 1 TB disk quota.

### Observed Data

```
Bear_room tensor cache breakdown (train split only):
├── hetero_x.npy          267 GB  (65% of total)
├── hetero_y.npy          134 GB  (33% of total)
├── hetero_general.npy    1.4 GB
├── hetero_channel.npy    1.4 GB
├── seq_x.npy             1.6 GB
├── seq_y.npy             818 MB
├── x_time.npy            1.1 GB
├── y_time.npy            545 MB
├── hetero_x_time.npy     178 MB
├── hetero_y_time.npy     90 MB
└── sample_ids.npy        122 MB
                          --------
                          ~408 GB TOTAL (train only!)

Source data for comparison:
├── 80 parquet files      ~50 MB total
└── embedding cache       ~1.6 GB
```

**Conclusion**: Text embeddings (hetero_x + hetero_y) account for **98%** of cache size.

---

## Root Cause Analysis

### 1. Sliding Window Creates Massive Redundancy

For sliding window sampling with stride=1:
- Sample i: uses timesteps `[t, t+input_len)`
- Sample i+1: uses timesteps `[t+1, t+input_len+1)`
- **Overlap: (input_len-1)/input_len = 287/288 = 99.65%**

For 726,566 samples with input_len=288:
```
Naive storage:   726,566 × 288 × 768 × 4 bytes = ~634 GB
Unique timesteps: ~726,566 unique timestamps
Deduplicated:    726,566 × 768 × 4 bytes = ~2.2 GB

Redundancy factor: 288x (one per window position)
```

### 2. Static Embeddings Stored Per-Sample

`hetero_general` and `hetero_channel` are entity-level constants, but currently stored
for every sample:
```
Current:  726,566 samples × 768 floats × 4 bytes = 2.2 GB (per field)
Optimal:  29 entities × 768 floats × 4 bytes = 89 KB (per field)

Redundancy factor: ~25,000x
```

### 3. No Compression

Arrays stored as raw float32 with no compression. While this enables memory-mapping,
text embeddings have significant compressibility (similar values, patterns).

---

## Quantified Impact

| Data Type | Current Size | Optimal Size | Redundancy Factor |
|-----------|--------------|--------------|-------------------|
| hetero_x (input embeddings) | 267 GB | ~1.5 GB | ~180x |
| hetero_y (output embeddings) | 134 GB | ~0.8 GB | ~170x |
| hetero_general | 1.4 GB | ~90 KB | ~16,000x |
| hetero_channel | 1.4 GB | ~90 KB | ~16,000x |
| seq_x, seq_y | 2.4 GB | ~20 MB | ~120x |
| x_time, y_time | 1.6 GB | ~15 MB | ~110x |
| hetero_x_time, hetero_y_time | 268 MB | ~3 MB | ~90x |
| **TOTAL** | **408 GB** | **~3-4 GB** | **~100-130x** |

**Note**: Time series (seq_x, seq_y) also have sliding window redundancy! Adjacent
samples share 287/288 timesteps for input and 143/144 for output.

---

## Proposed Solutions

### Solution A: Index-Based Embedding Lookup (Recommended)

Instead of storing embeddings directly, store indices into a shared embedding table.

#### Current Schema (Naive)
```
hetero_x:  shape=(N_samples, input_len, embed_dim)  # 267 GB
           Each sample stores full embedding vectors
```

#### Proposed Schema (Deduplicated)
```
embeddings:        shape=(N_unique_timestamps, embed_dim)  # ~2 GB
                   Shared table of all unique embeddings

x_embed_indices:   shape=(N_samples, input_len)  # ~0.8 GB
                   dtype=int32, indices into embeddings table

y_embed_indices:   shape=(N_samples, output_len)  # ~0.4 GB
                   dtype=int32, indices into embeddings table
```

#### Training-Time Lookup
```python
# In __getitem__ or collate_fn:
def get_hetero_x(self, sample_idx):
    indices = self.x_embed_indices[sample_idx]  # (input_len,) int32
    return self.embeddings[indices]  # (input_len, embed_dim) float32
```

**Overhead**: One extra indexing operation per batch. With numpy advanced indexing
or torch.index_select, this is ~1-2ms per batch - negligible vs data loading.

#### Space Savings
```
Before: 267 GB + 134 GB = 401 GB
After:  2 GB (embeddings) + 1.2 GB (indices) = 3.2 GB
Savings: 398 GB (99.2% reduction)
```

---

### Solution B: Timestamp-Based Lazy Lookup

Don't cache embeddings at all - look them up from the embedding cache at training time.

#### Schema
```
# Only store timestamp indices
x_timestamps:  shape=(N_samples, input_len)  dtype=int64
y_timestamps:  shape=(N_samples, output_len) dtype=int64

# Load embedding dict once at dataset init
self.embeddings = load_embedding_cache(entity_id)  # {timestamp_str: embedding}
```

#### Training-Time Lookup
```python
def get_hetero_x(self, sample_idx):
    timestamps = self.x_timestamps[sample_idx]
    return np.stack([self.embeddings[str(ts)] for ts in timestamps])
```

**Pros**: Zero embedding storage in tensor cache
**Cons**: Dict lookups per sample (slower than array indexing)

---

### Solution C: Entity-Level Static Embedding Table

For `hetero_general` and `hetero_channel`, store once per entity, not per sample.

#### Current Schema
```
hetero_general:  shape=(N_samples, embed_dim)  # 1.4 GB
hetero_channel:  shape=(N_samples, embed_dim)  # 1.4 GB
```

#### Proposed Schema
```
entity_ids:             shape=(N_samples,) dtype=int16        # ~1.5 MB
entity_general:         shape=(N_entities, embed_dim)         # ~90 KB
entity_channel:         shape=(N_entities, embed_dim)         # ~90 KB
entity_id_to_idx:       {entity_id: idx}                      # metadata
```

#### Training-Time Lookup
```python
def get_hetero_general(self, sample_idx):
    entity_idx = self.entity_ids[sample_idx]
    return self.entity_general[entity_idx]
```

**Savings**: 2.8 GB → ~2 MB (99.9% reduction)

---

### Solution D: Index-Based Time Series Lookup

The same sliding window redundancy applies to time series data (seq_x, seq_y).

#### Current Schema
```
seq_x:  shape=(N_samples, input_len, n_features)   # 1.6 GB
seq_y:  shape=(N_samples, output_len, n_features)  # 818 MB
```

#### Proposed Schema
```
timeseries:       shape=(N_unique_timestamps, n_features)  # ~10 MB
x_ts_indices:     shape=(N_samples, input_len) int32       # ~0.8 GB
y_ts_indices:     shape=(N_samples, output_len) int32      # ~0.4 GB
```

#### Why This Matters
For Bear_room with ~726k samples and ~726k unique timestamps:
- Current: 726,566 × 288 × 1 feature × 4 bytes = 836 MB for seq_x
- Deduplicated: 726,566 × 1 × 4 bytes (data) + indices = ~13 MB

For datasets with more features or future growth, this becomes more significant.
The same infrastructure supports both embeddings and time series.

---

### Solution E: Compression (Supplementary)

Use compression for arrays that are accessed sequentially (not random access).

| Compression Method | Ratio | Random Access | Notes |
|--------------------|-------|---------------|-------|
| gzip (np.savez_compressed) | 2-5x | No | Good for archival |
| LZ4 | 1.5-2x | Partial | Fast decompression |
| Blosc | 2-4x | Partial | Best for numeric |
| HDF5 chunked | 2-4x | Yes | Supports mmap-like |

**Recommendation**: Use for rarely-accessed arrays (val/test splits), but not for
training arrays where throughput matters.

---

## Recommended Implementation Plan

### Phase 1: Index-Based Embedding Lookup (HIGH PRIORITY)

**Goal**: Reduce hetero_x/hetero_y from 401 GB to ~3 GB

1. **Modify cache generation**:
   - Build timestamp→index mapping during first pass
   - Store embedding table once (deduplicated)
   - Store index arrays instead of embedding arrays

2. **Modify cache loading**:
   - Load embedding table into memory (2 GB, fits easily)
   - Implement index-based lookup in `__getitem__`

3. **Backward compatibility**:
   - Detect cache version from metadata
   - Support both old (direct) and new (indexed) formats

### Phase 2: Entity-Level Static Embeddings (MEDIUM PRIORITY)

**Goal**: Reduce hetero_general/hetero_channel from 2.8 GB to ~2 MB

1. Store entity-level arrays separately
2. Store entity_id per sample (int16)
3. Lookup at training time

### Phase 3: Time Array Deduplication (LOW PRIORITY)

**Goal**: Reduce time arrays using same index approach

Similar pattern - store unique timestamps once, reference by index.

---

## Implementation Details

### Modified ARRAY_SPECS

```python
ARRAY_SPECS_V2 = {
    # Per-sample metadata only
    'sample_ids': {'dtype': 'U64'},           # (N_samples,)
    'entity_indices': {'dtype': 'int16'},     # (N_samples,) - index into entity tables
    
    # Index arrays - reference into shared tables
    'x_indices': {'dtype': 'int32'},          # (N_samples, input_len) - into timeseries/embeddings
    'y_indices': {'dtype': 'int32'},          # (N_samples, output_len) - into timeseries/embeddings
    
    # Time features - still per-sample (small, different structure)
    'x_time_features': {'dtype': 'float32'},  # (N_samples, input_len, n_time_features)
    'y_time_features': {'dtype': 'float32'},  # (N_samples, output_len, n_time_features)
}

# Shared tables (stored once in shared/ directory, not per-sample)
SHARED_TABLES = {
    # Time series data - one row per unique timestamp
    'timeseries': {'dtype': 'float32'},       # (N_unique_timestamps, n_features)
    'timestamps': {'dtype': 'int64'},         # (N_unique_timestamps,) - actual timestamp values
    
    # Embeddings - one row per unique timestamp (may be subset of timeseries)
    'embeddings': {'dtype': 'float32'},       # (N_unique_timestamps, embed_dim)
    'hetero_time_features': {'dtype': 'float32'},  # (N_unique_timestamps, n_hetero_time_features)
    
    # Entity-level static data - one row per entity
    'entity_general': {'dtype': 'float32'},   # (N_entities, embed_dim)
    'entity_channel': {'dtype': 'float32'},   # (N_entities, embed_dim)
}
```

**Key insight**: `x_indices` and `y_indices` serve as universal indices into ALL
shared tables. The same index retrieves the timeseries value, embedding, and
timestamp for a given position. This unifies the lookup mechanism.

### Modified Cache Structure

```
tensor_cache/{hash}/
├── metadata.json              # Version, config, shapes, index mappings
├── shared/                    # Shared across ALL splits (deduplicated)
│   ├── timeseries.npy         # (N_unique_ts, n_features) - raw time series
│   ├── timestamps.npy         # (N_unique_ts,) int64 - timestamp values
│   ├── embeddings.npy         # (N_unique_ts, embed_dim) - text embeddings
│   ├── hetero_time.npy        # (N_unique_ts, n_time_features) - hetero time features
│   ├── entity_general.npy     # (N_entities, embed_dim) - static general embeddings
│   ├── entity_channel.npy     # (N_entities, embed_dim) - static channel embeddings
│   └── index_mappings.json    # {timestamp_str: idx}, {entity_id: idx}
├── train/
│   ├── sample_ids.npy         # (N,) str - sample identifiers
│   ├── entity_indices.npy     # (N,) int16 - index into entity tables
│   ├── x_indices.npy          # (N, input_len) int32 - indices into shared tables
│   ├── y_indices.npy          # (N, output_len) int32 - indices into shared tables
│   ├── x_time_features.npy    # (N, input_len, n_tf) - per-sample time features
│   └── y_time_features.npy    # (N, output_len, n_tf) - per-sample time features
├── val/
│   └── ... (same structure as train)
└── test/
    └── ... (same structure as train)
```

**Size comparison** (Bear_room train):
```
BEFORE (direct storage):     AFTER (indexed):
├── hetero_x.npy    267 GB   ├── shared/
├── hetero_y.npy    134 GB   │   ├── embeddings.npy    ~2 GB
├── seq_x.npy       1.6 GB   │   ├── timeseries.npy    ~3 MB
├── seq_y.npy       818 MB   │   └── ...               ~5 MB
├── x_time.npy      1.1 GB   ├── train/
├── y_time.npy      545 MB   │   ├── x_indices.npy     ~0.8 GB
└── ...             ~2 GB    │   ├── y_indices.npy     ~0.4 GB
                             │   └── ...               ~0.2 GB
─────────────────────────    ─────────────────────────────────
TOTAL:              ~408 GB  TOTAL:                    ~3.5 GB
```

### Modified TensorCacheDataset.__getitem__

```python
def __getitem__(self, index: int) -> tuple:
    # Get universal indices for this sample
    x_idx = self.arrays['x_indices'][index]      # (input_len,) int32
    y_idx = self.arrays['y_indices'][index]      # (output_len,) int32
    entity_idx = self.arrays['entity_indices'][index]  # scalar int16
    
    # Use indices to look up ALL data from shared tables
    # Time series
    seq_x = self.shared['timeseries'][x_idx]     # (input_len, n_features)
    seq_y = self.shared['timeseries'][y_idx]     # (output_len, n_features)
    
    # Timestamps
    x_time = self.shared['timestamps'][x_idx]    # (input_len,)
    y_time = self.shared['timestamps'][y_idx]    # (output_len,)
    
    # Text embeddings
    hetero_x = self.shared['embeddings'][x_idx]  # (input_len, embed_dim)
    hetero_y = self.shared['embeddings'][y_idx]  # (output_len, embed_dim)
    
    # Hetero time features
    hetero_x_time = self.shared['hetero_time'][x_idx]  # (input_len, n_hetero_tf)
    hetero_y_time = self.shared['hetero_time'][y_idx]  # (output_len, n_hetero_tf)
    
    # Entity-level static embeddings (same for all samples from this entity)
    hetero_general = self.shared['entity_general'][entity_idx]  # (embed_dim,)
    hetero_channel = self.shared['entity_channel'][entity_idx]  # (embed_dim,)
    
    # Per-sample arrays (not deduplicated - different structure)
    sample_id = self.arrays['sample_ids'][index]
    x_time_features = self.arrays['x_time_features'][index]
    y_time_features = self.arrays['y_time_features'][index]
    
    return (sample_id, seq_x, seq_y, x_time, y_time,
            hetero_x, hetero_y, hetero_x_time, hetero_y_time,
            hetero_general, hetero_channel, x_time_features, y_time_features)
```

**Performance note**: The index lookups (`shared['timeseries'][x_idx]`) use numpy
advanced indexing which is highly optimized. For a batch of 768 samples, the total
lookup overhead is ~2-5ms - negligible compared to GPU training time.

---

## Performance Considerations

### Training Throughput Impact

| Operation | Time per Batch (est.) | Notes |
|-----------|----------------------|-------|
| Array indexing (current) | ~0.5 ms | Direct mmap read |
| Embedding lookup (new) | ~1-2 ms | np.take or torch.index_select |
| Entity lookup (new) | ~0.1 ms | Single index per sample |

**Expected impact**: <5% throughput reduction, well worth 100x disk savings.

### Memory Usage

```
Current (load embedding arrays):
  hetero_x mmap: 267 GB virtual, pages loaded on access
  Peak RSS during batch: ~1-2 GB

Proposed (load shared embedding table):
  embeddings array: ~2 GB loaded fully into RAM
  Index arrays: mmap as before
  Peak RSS: ~3-4 GB (acceptable on GPU nodes)
```

---

## Migration Path

### Version Detection

```python
# In metadata.json
{
    "version": "2.0.0",  # Bump for new format
    "cache_format": "indexed_embeddings",  # vs "direct_embeddings"
    ...
}
```

### Backward Compatibility

```python
class TensorCacheDataset:
    def __init__(self, cache_dir, ...):
        self.metadata = load_metadata(cache_dir)
        
        if self.metadata.get('cache_format') == 'indexed_embeddings':
            self._init_indexed_format()
        else:
            self._init_legacy_format()
```

---

## Expected Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Bear_room train cache | 408 GB | ~3.5 GB | **~115x smaller** |
| Full Bear_room cache (train+val+test) | ~600 GB | ~5 GB | **~120x smaller** |
| NYC_traffic_speed (estimated) | ~850 GB | ~8 GB | **~105x smaller** |
| Cache generation time | Similar | Slightly faster | Better (less I/O) |
| Training throughput | Baseline | -2-5% | Acceptable |

**Why training throughput impact is minimal**:
- Shared tables loaded into RAM once at dataset init (~2-3 GB)
- Index lookups use numpy advanced indexing (C-optimized)
- Lookup time ~2-5ms per batch vs ~100-500ms model forward pass
- Reduced disk I/O may actually improve throughput on some systems

---

## Files to Modify

1. **`data_provider/tensor_cache.py`**:
   - `TensorCacheGenerator`: Build embedding index, store indices
   - `TensorCacheDataset`: Load shared tables, implement lookup
   - `ARRAY_SPECS`: Update for new format

2. **`cli/tensor_cache.py`**:
   - Add `--format` option for legacy vs optimized
   - Update info command to show format

3. **New file: `data_provider/embedding_index.py`** (optional):
   - Centralized embedding index building
   - Timestamp → index mapping utilities

---

## Testing Plan

1. **Correctness**: Compare `__getitem__` output between old and new formats
2. **Size verification**: Confirm ~80x reduction on Bear_room
3. **Throughput benchmark**: Verify <5% slowdown
4. **Memory profiling**: Confirm embedding table fits in RAM
5. **Edge cases**: Empty embeddings, missing timestamps

---

## Timeline Estimate

- Phase 1 (Index-based embeddings): 4-6 hours
- Phase 2 (Entity-level statics): 2-3 hours
- Testing and validation: 2-3 hours
- **Total**: ~1 day of implementation

---

## Summary

The current tensor cache implementation has a fundamental design flaw: it stores
embedding vectors redundantly for every overlapping window position. For typical
configurations (input_len=288, stride=1), this creates **~288x data duplication**.

The solution is straightforward: store embeddings once in a shared table, and store
indices per sample. This preserves all the amortization benefits (no computation at
training time) while reducing disk usage from ~400 GB to ~5 GB per dataset.

The implementation is backward compatible and has negligible performance impact.
