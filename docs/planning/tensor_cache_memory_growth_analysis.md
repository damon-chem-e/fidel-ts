# Memory Growth Analysis During Shared Table Building

## Overview

During the `_build_shared_tables()` processing loop, memory grows in several areas. This document breaks down what accumulates and why.

## Memory Consumers

### 1. Collector Lists (Growing Unbounded)

**Location**: `SharedTableCollector` dataclass

As we process samples and discover unique timestamps, arrays are appended to these lists:

```python
timestamps: List[int]           # ~8 bytes per timestamp
timeseries: List[np.ndarray]    # One array per unique timestamp
embeddings: List[np.ndarray]    # One array per unique timestamp  
hetero_time: List[np.ndarray]   # One array per unique timestamp
```

**For Jena (~367K unique timestamps):**

| List | Items | Size per Item | Total Data | Python Overhead | Total |
|------|-------|---------------|------------|-----------------|-------|
| `timestamps` | 367K | 8 bytes | ~3 MB | ~20 MB (list pointers) | **~23 MB** |
| `timeseries` | 367K | 17 features × 4 bytes = 68 bytes | ~25 MB | ~50 MB (array objects + pointers) | **~75 MB** |
| `embeddings` | 367K | 768 dim × 4 bytes = 3,072 bytes | ~1.1 GB | ~50 MB (array objects + pointers) | **~1.15 GB** |
| `hetero_time` | 367K | ~4 features × 4 bytes = 16 bytes | ~6 MB | ~50 MB | **~56 MB** |
| **TOTAL** | | | | | **~1.3 GB** |

**Key Issue**: These lists grow continuously as unique timestamps are discovered. By 90% completion, most unique timestamps have been seen, so this should stabilize.

**Python Array Overhead Breakdown:**
- Each numpy array object: ~96-128 bytes (PyObject header + metadata)
- List entry pointer: 8 bytes
- Memory allocator overhead: ~8-16 bytes
- Total overhead per array: ~110-150 bytes
- For 367K arrays: ~40-55 MB overhead total

### 2. Sample Tuple References (Temporary, But Accumulating)

**Location**: `for sample_idx in range(len(dataset)): sample = dataset[sample_idx]`

Each `dataset[sample_idx]` call returns a 13-element tuple containing numpy arrays:

```python
sample = (
    sample_id,        # str
    seq_x,           # (360, 17) float32 array - ~24 KB
    seq_y,           # (168, 17) float32 array - ~11 KB
    x_time,          # (360,) int64 array - ~3 KB
    y_time,          # (168,) int64 array - ~1.4 KB
    x_hetero,        # (360, 768) float32 array OR view - ~1.1 MB
    y_hetero,        # (168, 768) float32 array OR view - ~516 KB
    hetero_x_time,   # (360, 4) float32 array - ~6 KB
    hetero_y_time,   # (168, 4) float32 array - ~3 KB
    hetero_general,  # (768,) float32 array - ~3 KB
    hetero_channel,  # (768,) float32 array - ~3 KB
    x_time_features, # Variable
    y_time_features  # Variable
)
```

**Memory per sample tuple:**
- **If arrays are views** (references to preloaded data): ~100-200 bytes overhead
- **If arrays are copies**: ~1.7 MB per sample

**For Jena with 36,763 training samples:**
- If views: ~7 MB (just tuple overhead)
- If copies: **~62 GB** ❌ (clearly not happening, or we'd OOM immediately)

**Actual Behavior**: 
- If `preload_hetero=True`: Arrays are likely views into `self.full_hetero`, so minimal overhead
- However, array slicing operations in processing (e.g., `hetero_x[i]`) may create views that keep references to the original large arrays
- Python's garbage collector may not immediately free the sample tuple if there are lingering references

### 3. Array Copies During Processing

**Location**: `_register_timestamp_data()` lines 727, 748, 768

Several operations create **copies** of data:

```python
# Line 727: .flatten() ALWAYS creates a copy
collector.timeseries.append(np.asarray(ts_value).flatten())

# Line 745: Array slicing may create a view, but could become a copy
emb_to_store = emb_arr[0]  # View of (D,) array

# Line 768: .flatten() creates a copy
collector.hetero_time.append(np.asarray(hetero_time_feat).flatten())
```

**Impact**:
- `.flatten()` creates a new contiguous array (copy)
- For each unique timestamp, we're creating 2-3 new arrays via `.flatten()`
- These copies stay in memory until finalization

### 4. Dictionary Growth (timestamp_to_idx)

**Location**: `collector.timestamp_to_idx: Dict[int, int]`

**Memory**: 
- ~367K entries
- Python dict overhead: ~24 bytes per entry (hash, key, value, pointers)
- Total: ~9 MB

This is small compared to collector lists, but still contributes.

### 5. Dataset Instance Memory (If preload_hetero=True)

**Location**: `datasets = self.data_provider.get_datasets(flag)`

If `preload_hetero=True`, each `Universal_Dataset` instance holds:

```python
self.full_hetero  # (367K, 768, 2) float32 = ~2.3 GB per dataset
```

**However**: With our sequential processing fix, only ONE dataset instance should exist at a time (per split).

**If memory isn't being released**:
- Multiple dataset instances could accumulate if `del datasets; gc.collect()` isn't working
- Python's memory allocator keeps freed memory in pools (doesn't return to OS immediately)
- The dataset's internal arrays might have lingering references

### 6. Intermediate Arrays in Processing Loop

**Location**: Inside `_process_sample_for_collection()`

Temporary arrays created during processing:

```python
x_time_flat = x_time.flatten()  # Copy of x_time
y_time_flat = y_time.flatten()  # Copy of y_time
emb = hetero_x[i]               # View or copy of embedding slice
```

These should be garbage collected quickly, but if they're views into large arrays, they might prevent garbage collection of the parent arrays.

## Memory Growth Timeline

### Phase 1: Counting Samples (Low Memory)
- Loads each split's datasets briefly
- Just counts samples, doesn't process
- **Expected memory**: < 1 GB

### Phase 2: Processing Train Split (~36K samples)
- Collector grows as unique timestamps are discovered
- Most unique timestamps are found in train data
- **Memory at end**: ~1.5 GB (collector) + 2-3 GB (train dataset if preload_hetero)

### Phase 3: Processing Val Split (~9K samples)
- Fewer new unique timestamps (most overlap with train)
- Collector continues growing slowly
- **Memory at end**: ~1.6 GB (collector) + 2-3 GB (val dataset)

### Phase 4: Processing Test Split (~9K samples)
- Fewest new unique timestamps
- Collector nearly complete
- **Memory at end**: ~1.7 GB (collector) + 2-3 GB (test dataset)

### Phase 5: Finalization
- `np.stack()` creates new arrays (doubles memory temporarily)
- Collector lists cleared after each conversion
- **Peak memory**: ~3.4 GB (collector arrays + stacked arrays)
- **After cleanup**: ~1.7 GB (just stacked arrays)

## Why Memory Might Still Grow Unbounded

### Hypothesis 1: Python Memory Pools
Python's memory allocator (pymalloc) keeps freed memory in pools and doesn't immediately return it to the OS. Even after `gc.collect()`, RSS might not decrease.

**Check**: Monitor `VMS` (virtual memory) vs `RSS` (resident set size). If VMS stays constant but RSS grows, it's Python's allocator.

### Hypothesis 2: Lingering References
- Array views might keep references to parent arrays
- Sample tuples might not be garbage collected if references persist
- Collector arrays might reference data from samples

**Check**: Look for `.flatten()`, array slicing, or views that might create reference chains.

### Hypothesis 3: Dataset Preloading
If `preload_hetero=True` and datasets aren't being released, multiple 2-3 GB instances accumulate.

**Check**: The periodic memory warnings should catch this (warns at 50GB RSS).

### Hypothesis 4: Collector Growth Exceeds Estimate
If there are MORE unique timestamps than expected (e.g., 500K instead of 367K), collector memory would be proportionally larger.

**Check**: The debug output will show collector size at each GC checkpoint.

## Recommendations

1. **Monitor collector size during processing** (done via debug prints)
2. **Force explicit array copying** instead of views if views are causing reference issues
3. **Consider chunked processing** for very large datasets (process N samples, checkpoint collector, continue)
4. **Disable preload_hetero during cache generation** (saves 2-3 GB per dataset)
5. **Use memory profiling tools** like `memory_profiler` to pinpoint exact growth locations

## Memory Calculation Summary

For Jena with 367K unique timestamps:

| Component | Estimated Memory |
|-----------|------------------|
| Collector lists (data) | ~1.1 GB |
| Collector lists (overhead) | ~0.2 GB |
| timestamp_to_idx dict | ~0.01 GB |
| One dataset (if preload_hetero) | ~2.3 GB |
| Sample tuples (temporary, minimal) | ~0.01 GB |
| **Total (ideal)** | **~3.6 GB** |
| **With Python allocator overhead** | **~5-7 GB** |

If memory is > 50 GB, something else is accumulating (likely multiple dataset instances or memory leak).
