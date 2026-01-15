# Tensor Cache Generation: Memory and Performance Optimization Plan

## Overview

This document identifies major inefficiencies in the tensor cache generation process that cause OOM errors on large datasets (e.g., Jena_Atmospheric_Physics killed at 90% on 64GB RAM).

**Root Causes Identified:**
1. **Multiple dataset loading AND retention** - Datasets loaded 3-4 times per split and kept in memory simultaneously
2. **preload_hetero memory impact** - Each dataset with preloaded hetero holds ~2-3GB
3. **Unnecessary memory growth** - Collector lists and finalization doubling
4. **No memory estimation** - Processing starts without knowing if it will fit in memory

---

## Issue 1: Multiple Dataset Loading AND Retention (PRIMARY CAUSE)

### Problem Summary

The tensor cache generation process loads train/val/test datasets multiple times AND keeps multiple instances in memory simultaneously:
- **Train**: Loaded 4 times (downtime detection + counting + processing + index generation)
- **Val/Test**: Loaded 3 times each (counting + processing + index generation)

Each `get_datasets(flag)` call creates NEW `Universal_Dataset` instances. Previous instances remain in memory until garbage collected.

### Evidence from OOM Failure

```
Loading train datasets ━━━ 100% • 1/1  [appears 4 times]
Loading val datasets   ━━━ 100% • 1/1  [appears 2 times, one taking 17s]  
Loading test datasets  ━━━ 100% • 1/1  [appears 2 times]
⠼ Processing samples ━━━━━━━━━━━━━━━━━━━━━ 89211/99110 • 0:10:51
Killed
```

With 8 dataset loads and potential retention, memory accumulates rapidly.

### Memory Impact

For Jena with `preload_hetero=True`:
- Each dataset holds `full_hetero`: ~367K timestamps × 768 dim × 4 bytes × 2 = **~2.3GB per entity**
- Multiple dataset instances staying in memory = **20-40GB consumed by datasets alone**

### Current Flow (PROBLEMATIC)

```
generate()
  └─> _build_shared_tables()
      ├─> _detect_downtime_in_training()  [Load train #1, not released]
      ├─> get_datasets(flag) for counting  [Load all splits #2, not released]
      └─> get_datasets(flag) for processing [Load all splits #3, not released]
  └─> _generate_all_splits()
      └─> _generate_split(flag) for each
          └─> get_datasets(flag)            [Load each split #4, not released]
```

### Solution: Sequential Processing with Explicit Memory Release

**IMPORTANT**: Do NOT cache all datasets upfront (this makes OOM worse by loading all splits simultaneously).

Instead, process phases sequentially with explicit memory release:

```python
import gc

def _build_shared_tables(self, flags: List[str]):
    # Detect downtime (train only)
    has_downtime = _detect_downtime_in_training(self.data_provider)
    gc.collect()  # Release train datasets from detection
    
    collector = SharedTableCollector()
    collector.num_news_items = 2 if has_downtime else 1
    
    # Count total samples (quick scan, release after)
    total_samples = 0
    for flag in flags:
        datasets = self.data_provider.get_datasets(flag)
        if datasets:
            total_samples += sum(len(ds) for ds in datasets.values())
        del datasets  # Explicit release
    gc.collect()
    
    # Process each split sequentially
    for flag in flags:
        datasets = self.data_provider.get_datasets(flag)
        if not datasets:
            continue
        
        # Process all samples in this split
        for entity_id, dataset in datasets.items():
            for sample_idx in range(len(dataset)):
                sample = dataset[sample_idx]
                _process_sample_for_collection(collector, sample)
        
        # CRITICAL: Release this split's datasets before loading next
        del datasets
        gc.collect()
    
    # Finalize
    shared_tables, index_mappings = _finalize_shared_tables(collector)
    
    # Clear collector lists to free memory before returning
    collector.timeseries.clear()
    collector.embeddings.clear()
    collector.hetero_time.clear()
    collector.timestamps.clear()
    gc.collect()
    
    return shared_tables, index_mappings
```

---

## Issue 2: preload_hetero Memory Impact

### Problem Summary

When `Universal_Dataset` is created with `preload_hetero=True`, it calls `__preload_hetero__()` which loads the ENTIRE heterogeneous embedding array into `self.full_hetero`.

For Jena:
- ~367K timestamps × 768 dim × 4 bytes × 2 (N) = **~2.3GB per dataset instance**

### Why This Matters for Cache Generation

- Cache generation accesses each sample **exactly once** (sequential scan)
- `preload_hetero` is designed for **training** where samples are accessed multiple times
- Sequential access gets no benefit from preloading

### Solution

Force `preload_hetero=False` during cache generation by modifying how `TensorCacheGenerator` creates its data provider or by adding a parameter.

**Option A**: Modify data provider initialization in TensorCacheGenerator
**Option B**: Add `low_memory_mode` parameter that forces sequential hetero loading

---

## Issue 3: Unnecessary Memory Growth During Processing

### 3.1 Double Memory Usage During Finalization

**Location**: `_finalize_shared_tables()` lines 935-941

**Problem**: `np.stack()` creates new arrays while original lists still exist:

```python
if collector.timeseries:
    shared_tables['timeseries'] = np.stack(collector.timeseries).astype(np.float32)
    # ^ Creates NEW array, but collector.timeseries list still exists
```

**Impact**: Memory doubles temporarily during finalization.

**Solution**: Clear collector lists immediately after each conversion:

```python
if collector.timeseries:
    shared_tables['timeseries'] = np.stack(collector.timeseries).astype(np.float32)
    collector.timeseries.clear()  # Free list memory immediately
    
if collector.embeddings:
    shared_tables['embeddings'] = np.stack(collector.embeddings).astype(np.float32)
    collector.embeddings.clear()
    
# ... etc for all lists
```

### 3.2 Python List Overhead

**Problem**: Each collector list entry is a separate numpy array with ~100-150 bytes overhead.

For 367K unique timestamps:
- 367K × 140 bytes overhead = **~50MB** just in object overhead

**Future optimization**: Use pre-allocated growable arrays instead of lists of arrays.

### 3.3 Array Copies vs Views

**Location**: `_register_timestamp_data()` lines 702, 768

```python
collector.timeseries.append(np.asarray(ts_value).flatten())  # Creates copy
```

**Note**: May be necessary for data integrity, but worth investigating.

---

## Issue 4: No Memory Estimation

### Problem

Processing starts without knowing if it will fit in memory. OOM happens at 90% completion, wasting time.

### Solution: Pre-flight Memory Check

```python
def _estimate_memory_requirements(self, flags: List[str]) -> dict:
    """
    Estimate memory requirements before starting generation.
    
    Performs a quick scan of timestamps to count unique values,
    then estimates memory based on data shapes.
    """
    import psutil
    
    # Quick scan to count unique timestamps (don't process samples)
    unique_timestamps = set()
    
    for flag in flags:
        datasets = self.data_provider.get_datasets(flag)
        if not datasets:
            continue
        
        # Sample a subset for efficiency
        for entity_id, dataset in datasets.items():
            # Check every Nth sample for large datasets
            step = max(1, len(dataset) // 1000)
            for i in range(0, len(dataset), step):
                sample = dataset[i]
                x_time = sample[3]  # SAMPLE_IDX_X_TIME
                y_time = sample[4]  # SAMPLE_IDX_Y_TIME
                if x_time is not None:
                    unique_timestamps.update(x_time.flatten().tolist())
                if y_time is not None:
                    unique_timestamps.update(y_time.flatten().tolist())
        
        del datasets
        gc.collect()
    
    # Extrapolate to full dataset
    n_unique_estimate = len(unique_timestamps) * (1000 / max(1, len(dataset) // 1000))
    n_unique_estimate = int(n_unique_estimate * 1.2)  # 20% safety margin
    
    # Estimate memory per unique timestamp
    # Get shapes from config or first sample
    n_features = 17  # Jena default, should be inferred
    embed_dim = 768  # Default, should be inferred
    num_news_items = 1  # Default
    n_hetero_time = 4  # Default
    
    per_ts_bytes = (
        8 +                                      # timestamp int64
        n_features * 4 +                         # timeseries float32
        embed_dim * 4 * num_news_items +         # embeddings
        n_hetero_time * 4 +                      # hetero_time
        150                                      # Python object overhead
    )
    
    collector_gb = (n_unique_estimate * per_ts_bytes) / (1024**3)
    
    # Get available memory
    available_gb = psutil.virtual_memory().available / (1024**3)
    
    return {
        'estimated_unique_timestamps': n_unique_estimate,
        'estimated_collector_memory_gb': collector_gb,
        'available_memory_gb': available_gb,
        'will_likely_fit': collector_gb < available_gb * 0.7
    }
```

---

## Memory Tracing Strategy

To diagnose memory growth issues, debug prints should be added at key points.

### Debug Function

```python
import psutil
import gc

def _debug_memory(label: str, force_gc: bool = False) -> None:
    """Print memory usage for debugging. Enable via TENSOR_CACHE_DEBUG_MEMORY=1."""
    import os
    if os.environ.get('TENSOR_CACHE_DEBUG_MEMORY', '0') != '1':
        return
    
    if force_gc:
        gc.collect()
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_gb = mem_info.rss / (1024 ** 3)
    
    print(f"[DEBUG MEM] {label}: RSS={rss_gb:.2f}GB")
```

### Key Debug Points

1. `generate()` start - Baseline
2. After `_detect_downtime_in_training()` - Train dataset impact
3. Before/after each `get_datasets()` call - Track loading
4. After `del datasets; gc.collect()` - Verify release
5. During sample processing (every 10K samples) - Track collector growth
6. Before/after `_finalize_shared_tables()` - Peak memory
7. After clearing collector lists - Verify cleanup

---

## Implementation Priority

### Critical (Fix OOM on 64GB)

| Priority | Issue | Solution | Impact |
|----------|-------|----------|--------|
| **1** | Multiple dataset retention | Sequential processing with `del datasets; gc.collect()` | **High** - Primary OOM cause |
| **2** | Finalization doubling | Clear collector lists after each `np.stack()` | **Medium** - Reduces peak |
| **3** | No memory release | Add `gc.collect()` between all phases | **Medium** - Forces cleanup |

### High Priority (Robustness)

| Priority | Issue | Solution | Impact |
|----------|-------|----------|--------|
| **4** | preload_hetero | Force False during cache generation | **High** - 2-3GB per entity |
| **5** | No memory estimation | Add pre-flight check with warning | **Medium** - Fail fast |
| **6** | No debug visibility | Add conditional memory logging | **Low** - Debugging aid |

### Medium Priority (Scalability)

| Priority | Issue | Solution | Impact |
|----------|-------|----------|--------|
| **7** | Python list overhead | Pre-allocated growable arrays | **Low** - ~50MB savings |
| **8** | Very large datasets | Chunked/streaming processing | **High** - Enables huge datasets |
| **9** | Partial progress lost | Checkpoint within `_build_shared_tables` | **Medium** - Resume support |

---

## Testing Strategy

After implementing fixes:

1. **Verify correctness** - Run on small dataset, compare output
2. **Memory monitoring** - Run with `TENSOR_CACHE_DEBUG_MEMORY=1`
3. **Test on Jena** - Should complete on 64GB RAM
4. **Measure improvement**:
   - Peak memory usage (before/after)
   - Total generation time
   - Number of `gc.collect()` calls

### Test Commands

```bash
# Enable memory debugging
export TENSOR_CACHE_DEBUG_MEMORY=1

# Test on small dataset first
python -m cli.tensor_cache generate configs/experiment_suites/lynx_film_raw/bear_room.yaml

# Then test on Jena
python -m cli.tensor_cache generate configs/experiment_suites/lynx_film_raw/jena_atmospheric.yaml
```

---

## Memory Budget for Jena (Expected After Fix)

| Component | Before Fix | After Fix |
|-----------|------------|-----------|
| Collector (~367K unique timestamps) | ~1.2 GB | ~1.2 GB |
| Datasets in memory simultaneously | ~20-40 GB | **~2-3 GB** (one split at a time) |
| Peak during finalization | ~2.4 GB | **~1.2 GB** (clear lists) |
| **Total Peak** | **~25-45 GB** | **~5-7 GB** |

---

## Notes

- The existing `_debug_memory()` function (lines 125-131) can be reused
- Debug prints should be conditional via `TENSOR_CACHE_DEBUG_MEMORY` env var
- Consider adding `--low-memory` CLI flag for explicit memory-saving mode
- Python's garbage collector may not immediately release memory; explicit `gc.collect()` helps
