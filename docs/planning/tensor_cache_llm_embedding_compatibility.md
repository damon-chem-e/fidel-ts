# Tensor Cache and LLM Embedding Compatibility Analysis

**Date**: January 2025
**Status**: Analysis Complete, Implementation Deferred
**Affected Models**: TimeCMA, MMTSFLib, and any model using `llm_embedding` config section

---

## Executive Summary

The current tensor cache implementation **does not support LLM embeddings** used by models like TimeCMA and MMTSFLib. This is due to a **shape semantics mismatch** between how tensor cache processes embeddings (expecting per-timestamp data) and how LLM embeddings are structured (per-sample data with embedding dimensions first).

**Current workaround**: Run TimeCMA/MMTSFLib without tensor cache (`use_tensor_cache: false`).

**Future option**: Implement tensor cache v2 with explicit LLM embedding support for significant training speedup.

---

## Table of Contents

1. [Current Tensor Cache Architecture](#1-current-tensor-cache-architecture)
2. [What Tensor Cache Currently Supports](#2-what-tensor-cache-currently-supports)
3. [LLM Embedding Architecture](#3-llm-embedding-architecture)
4. [The Incompatibility: Shape Semantics Mismatch](#4-the-incompatibility-shape-semantics-mismatch)
5. [Why This Causes Failure](#5-why-this-causes-failure)
6. [Cache Hash Considerations](#6-cache-hash-considerations)
7. [Current Workaround](#7-current-workaround)
8. [Future: Tensor Cache v2 for LLM Embeddings](#8-future-tensor-cache-v2-for-llm-embeddings)

---

## 1. Current Tensor Cache Architecture

### Overview

The tensor cache system pre-computes all dataloader operations and stores results as memory-mapped numpy arrays, providing 100-1000x faster data loading during training.

### Key Files

| File | Purpose |
|------|---------|
| `cli/tensor_cache.py` | CLI commands for generate/validate/info |
| `data_provider/tensor_cache.py` | Core cache generation and loading logic |
| `utils/experiment_config_builder.py` | Cache hash computation via `build_cache_config()` |

### Generation Pipeline

```
Phase 1: Collection
    Data_Provider.get_datasets()
    → Universal_Dataset.__getitem__()
    → _process_sample_for_collection()
    → SharedTableCollector (deduplicated storage)

Phase 2: Save Shared Tables
    → shared/timeseries.npy
    → shared/timestamps.npy
    → shared/embeddings.npy
    → shared/entity_general.npy
    → shared/entity_channel.npy

Phase 3: Generate Per-Split Indices
    → train/x_indices.npy, y_indices.npy, entity_indices.npy
    → val/...
    → test/...

Phase 4: Save Metadata
    → metadata.json
```

### Cache Directory Structure

```
data/{dataset}/tensor_cache/{hash}/
├── shared/
│   ├── timeseries.npy       # (N_unique_timestamps, n_features)
│   ├── timestamps.npy       # (N_unique_timestamps,)
│   ├── embeddings.npy       # (N_unique_timestamps, [2,] embed_dim)
│   ├── entity_general.npy   # (N_entities, embed_dim)
│   └── entity_channel.npy   # (N_entities, embed_dim)
├── train/
│   ├── x_indices.npy        # (N_samples, input_len) - indices into shared tables
│   ├── y_indices.npy        # (N_samples, output_len)
│   └── entity_indices.npy   # (N_samples,)
├── val/
│   └── ...
├── test/
│   └── ...
└── metadata.json
```

---

## 2. What Tensor Cache Currently Supports

### 2.1 Dynamic Per-Timestamp Embeddings (hetero_x, hetero_y)

**Used by**: TGTSF models (Lynx, lynx_film_raw), Time-MMD models

**Shape**: `(input_len, num_items, embed_dim)` e.g., `(96, 2, 768)`

**Semantics**:
- First dimension = timestamps
- Each timestamp has its own embedding (news article, weather forecast at that time)
- The embedding at timestamp T is **identical** for all samples that include T

**How tensor cache handles this**:

```python
# From data_provider/tensor_cache.py lines 933-956
for i, ts in enumerate(x_time_flat):  # Iterate over timestamps
    if hetero_x.ndim >= 2 and i < hetero_x.shape[0]:
        emb = hetero_x[i].copy()  # Extract embedding at timestamp i
    _register_timestamp_data(collector, ts, ts_val, emb, htf)
```

**Deduplication**: By timestamp - if timestamp T appears in 1000 samples, its embedding is stored once.

**Result**: Works correctly. Each sample gets its unique window of timestamp-indexed embeddings.

### 2.2 Static Entity Data (hetero_general, hetero_channel)

**Used by**: All TGTSF models with entity-level descriptions

**Shape**: `(embed_dim,)` e.g., `(384,)`

**Semantics**:
- Per-entity static data (channel descriptions, general entity information)
- **Identical** for all samples from the same entity

**How tensor cache handles this**:

```python
# From data_provider/tensor_cache.py lines 1463-1470
first_sample = dataset[0]
_register_entity_data(
    collector,
    entity_id,
    first_sample[SAMPLE_IDX_HETERO_GENERAL],
    first_sample[SAMPLE_IDX_HETERO_CHANNEL]
)
```

**Deduplication**: By entity - each entity's static data stored once.

**Result**: Works correctly.

### 2.3 Time Series Data (seq_x, seq_y)

**Shape**: `(input_len, n_features)` and `(output_len, n_features)`

**Handling**: Stored per-timestamp in shared tables, indexed per-sample.

**Result**: Works correctly.

---

## 3. LLM Embedding Architecture

### Overview

LLM embeddings (used by TimeCMA, MMTSFLib) are fundamentally different from news/weather embeddings.

### How LLM Embeddings Are Generated

1. **Input**: Time series window `(input_len, n_features)`
2. **Process**:
   - Convert time series to text prompt (e.g., "Given the following time series values: [1.2, 3.4, ...]")
   - Run through LLM (GPT-2, Qwen2.5-0.5B-Instruct, etc.)
   - Extract hidden state (last token or pooled)
3. **Output**: Single embedding vector for the entire sample

### LLM Embedding Shape

**Shape**: `(embed_dim, n_channels)` e.g., `(768, 1)` for GPT-2

| Model | embed_dim |
|-------|-----------|
| GPT-2 | 768 |
| GPT-2 Medium | 1024 |
| Qwen2.5-0.5B-Instruct | 896 |

### Where LLM Embeddings Go

From `data_provider/data_loader.py` lines 390-394:

```python
if self.llm_embedding_provider is not None:
    # LLM embeddings mode: x_hetero comes from precomputed LLM cache
    # Per-sample embeddings go to x_hetero (maps to historical_events in model forward)
    x_hetero = self.llm_embedding_provider[index]
```

**Key point**: LLM embeddings are placed in `x_hetero` (same field as news embeddings), but with completely different shape semantics.

### LLM Embedding Uniqueness

- Each sample has a **unique** LLM embedding based on its specific time series values
- Sample 0 (timestamps 0-95) has different embedding than Sample 1 (timestamps 1-96)
- No meaningful deduplication possible (unlike timestamp-based news embeddings)

---

## 4. The Incompatibility: Shape Semantics Mismatch

### The Core Problem

| Data Type | Shape | First Dimension | Deduplication |
|-----------|-------|-----------------|---------------|
| News/weather embeddings | `(input_len, num_items, embed_dim)` | Timestamps | By timestamp |
| LLM embeddings | `(embed_dim, n_channels)` | Embedding features | None possible |

Tensor cache assumes `hetero_x` has **timestamps as the first dimension**. LLM embeddings have **embedding features as the first dimension**.

### Visual Comparison

**News embeddings** (what tensor cache expects):
```
hetero_x shape: (96, 2, 768)
         dim 0: timestamps (96 timesteps)
         dim 1: items (news + downtime indicator)
         dim 2: embedding dimension
```

**LLM embeddings** (what TimeCMA/MMTSFLib produce):
```
hetero_x shape: (768, 1)
         dim 0: embedding dimension (NOT timestamps!)
         dim 1: channels
```

---

## 5. Why This Causes Failure

### What Happens During Cache Generation

When tensor cache processes an LLM embedding:

```python
# hetero_x has shape (768, 1) for LLM embedding
# x_time_flat has shape (96,) for input_len=96

for i, ts in enumerate(x_time_flat):  # i goes 0 to 95
    if hetero_x.ndim >= 2 and i < hetero_x.shape[0]:  # True: ndim=2, i < 768
        emb = hetero_x[i].copy()  # Extracts ROW i of (768, 1) matrix
```

For timestamp 0: `emb = hetero_x[0]` → shape `(1,)` (the first element of embedding!)
For timestamp 1: `emb = hetero_x[1]` → shape `(1,)` (the second element of embedding!)
...
For timestamp 95: `emb = hetero_x[95]` → shape `(1,)` (the 96th element of embedding!)

### Result: Garbled Data

Instead of storing one `(768, 1)` embedding per sample, tensor cache stores:
- 96 "embeddings" of shape `(1,)`
- Each containing a single scalar from the original embedding
- Completely destroys the semantic content

### Failure Mode During Training

When loading from corrupted cache:
1. Model expects `hetero_x` with shape `(input_len, ?, embed_dim)` or `(embed_dim, n_channels)`
2. Gets reconstructed data with wrong shapes/values
3. Dimension mismatch errors or silent incorrect training

---

## 6. Cache Hash Considerations

### Current Cache Hash Parameters

From `utils/experiment_config_builder.py` lines 382-394:

```python
return {
    'input_len': args.input_len,
    'output_len': args.output_len,
    'scale': args.scale,
    'truncate_train_for_purge': args.truncate_train_for_purge,
    'downsample': args.downsample,
    'data_name': args.data,
    'hetero_stride': hetero_stride,
    'hetero_type': hetero_type,
    'timemmd_text_output': timemmd_text_output,
    'missing_value_strategy': ...,
    'split_info': ...,
}
```

### Missing from Hash: LLM Embedding Config

The `llm_embedding` configuration is **NOT included** in the cache hash:
- `model_name` (gpt2 vs Qwen2.5-0.5B-Instruct)
- `prompt_template` (timecma_v1 vs mmtsflib_v1)
- `extraction_mode` (last_token vs pooled)
- `d_llm` (embedding dimension)

### Risk if LLM Embeddings Were Cached

Without including LLM config in hash:
- Experiment A (GPT-2, d_llm=768) generates cache
- Experiment B (Qwen2.5, d_llm=896) incorrectly reuses same cache
- Dimension mismatch → training failure

**Any future implementation MUST include LLM embedding config in cache hash.**

---

## 7. Current Workaround

### Disable Tensor Cache for LLM-Based Models

In experiment config:

```yaml
training:
  use_tensor_cache: false  # Required for TimeCMA, MMTSFLib
```

### Performance Impact

Without tensor cache:
- `__getitem__` time: ~18.6ms per sample (includes LLM embedding lookup from HDF5)
- With tensor cache (non-LLM models): ~0.02ms per sample

**Expected slowdown**: ~1000x slower data loading, but:
- GPU compute typically dominates training time
- LLM embedding lookup from HDF5 is already optimized (memory-mapped)
- May be acceptable for smaller datasets

### Recommendation

1. Start training without tensor cache
2. Monitor GPU utilization and data loading bottleneck
3. If data loading becomes bottleneck (GPU util < 80%), consider implementing tensor cache v2

---

## 8. Future: Tensor Cache v2 for LLM Embeddings

If training speed becomes unacceptable, here's a plan for tensor cache v2 with LLM embedding support.

### Design Goals

1. Store LLM embeddings as per-sample data (not per-timestamp)
2. Include LLM config in cache hash for proper invalidation
3. Maintain backward compatibility with existing non-LLM caches
4. Achieve similar speedup to current tensor cache (~1000x)

### Proposed Architecture

#### New Cache Structure

```
data/{dataset}/tensor_cache/{hash}/
├── shared/
│   ├── timeseries.npy
│   ├── timestamps.npy
│   ├── embeddings.npy           # News/weather (unchanged)
│   ├── entity_general.npy
│   ├── entity_channel.npy
│   └── llm_embeddings.npy       # NEW: (N_samples, embed_dim, n_channels)
├── train/
│   ├── x_indices.npy
│   ├── y_indices.npy
│   ├── entity_indices.npy
│   └── llm_indices.npy          # NEW: (N_samples,) → index into llm_embeddings
├── val/
│   └── ...
├── test/
│   └── ...
└── metadata.json
```

#### Detection Logic

```python
def _is_llm_embedding(hetero_x: np.ndarray, input_len: int) -> bool:
    """
    Detect if hetero_x is an LLM embedding based on shape.

    News embeddings: (input_len, num_items, embed_dim) - first dim matches input_len
    LLM embeddings: (embed_dim, n_channels) - first dim is large (768+), doesn't match input_len
    """
    if hetero_x is None or hetero_x.ndim != 2:
        return False

    # LLM embeddings have embed_dim as first dimension (768, 896, 1024, etc.)
    # News embeddings have input_len as first dimension (typically 24-720)
    first_dim = hetero_x.shape[0]

    # Heuristic: LLM embed_dim is typically 768+ and != input_len
    return first_dim >= 512 and first_dim != input_len
```

#### Modified Sample Processing

```python
def _process_sample_for_collection(collector, sample, input_len):
    hetero_x = _safe_array(sample[SAMPLE_IDX_HETERO_X])

    if _is_llm_embedding(hetero_x, input_len):
        # LLM embedding path: store as per-sample data
        _register_llm_embedding(collector, hetero_x)
    else:
        # News/weather path: existing per-timestamp logic
        for i, ts in enumerate(x_time_flat):
            emb = hetero_x[i].copy() if hetero_x is not None else None
            _register_timestamp_data(collector, ts, ts_val, emb, htf)
```

#### LLM Embedding Registration

```python
def _register_llm_embedding(collector, llm_embedding: np.ndarray) -> int:
    """
    Register per-sample LLM embedding.

    Unlike timestamp deduplication, LLM embeddings are unique per sample,
    so we store them sequentially without deduplication.
    """
    idx = len(collector.llm_embeddings)
    collector.llm_embeddings.append(llm_embedding.copy())
    return idx
```

#### Updated Cache Hash

```python
def build_cache_config(args: dotdict) -> dict:
    # ... existing parameters ...

    # NEW: Include LLM embedding config if present
    llm_config_for_hash = None
    if hasattr(args, 'llm_embedding') and args.llm_embedding:
        llm_config_for_hash = {
            'model_name': args.llm_embedding.get('model_name'),
            'prompt_template': args.llm_embedding.get('prompt_template'),
            'extraction_mode': args.llm_embedding.get('extraction_mode', 'last_token'),
        }

    return {
        # ... existing keys ...
        'llm_embedding': llm_config_for_hash,  # NEW
    }
```

#### Loading LLM Embeddings

```python
class TensorCacheDataset(Dataset):
    def __init__(self, cache_dir, split, ...):
        # ... existing loading ...

        # NEW: Load LLM embeddings if present
        llm_path = shared_dir / "llm_embeddings.npy"
        if llm_path.exists():
            self.llm_embeddings = np.load(llm_path, mmap_mode='r')
            self.llm_indices = np.load(split_dir / "llm_indices.npy", mmap_mode='r')
        else:
            self.llm_embeddings = None
            self.llm_indices = None

    def __getitem__(self, index):
        # ... existing logic ...

        # NEW: Get LLM embedding if available
        if self.llm_embeddings is not None:
            llm_idx = int(self.llm_indices[index])
            x_hetero = self.llm_embeddings[llm_idx]
        else:
            # Existing news/weather reconstruction
            x_hetero = self._reconstruct_hetero_x(index)

        return sample_tuple
```

### Implementation Effort Estimate

| Component | Lines of Code | Complexity |
|-----------|---------------|------------|
| Detection logic | ~20 | Low |
| LLM embedding collection | ~50 | Medium |
| LLM embedding saving | ~30 | Low |
| Index generation | ~40 | Medium |
| Cache loading | ~40 | Low |
| Hash update | ~20 | Low |
| Validation | ~30 | Low |
| Testing | ~100 | Medium |
| **Total** | **~330** | **Medium** |

### Expected Performance Improvement

| Metric | Without Cache | With Cache v2 |
|--------|---------------|---------------|
| `__getitem__` time | ~18.6ms | ~0.05ms |
| Speedup | 1x | ~370x |
| Memory overhead | None | ~N_samples × embed_dim × 4 bytes |

For a dataset with 100K samples and GPT-2 (768D):
- LLM embeddings storage: 100K × 768 × 4 bytes = ~307 MB
- Acceptable for most systems

### Backward Compatibility

- Existing caches (without `llm_embeddings.npy`) continue to work unchanged
- Non-LLM models are unaffected
- Detection is automatic based on shape heuristics

---

## Appendix: Key Code Locations

### Tensor Cache Core

| Location | Purpose |
|----------|---------|
| `data_provider/tensor_cache.py:212-226` | `CACHE_RELEVANT_KEYS` |
| `data_provider/tensor_cache.py:228-254` | `compute_cache_hash()` |
| `data_provider/tensor_cache.py:848-887` | `_register_entity_data()` |
| `data_provider/tensor_cache.py:890-989` | `_process_sample_for_collection()` |
| `data_provider/tensor_cache.py:1264-1500` | `TensorCacheGenerator._build_shared_tables()` |
| `data_provider/tensor_cache.py:2215-2405` | `TensorCacheDataset` |

### LLM Embedding System

| Location | Purpose |
|----------|---------|
| `data_provider/data_factory.py:163-224` | `_get_llm_embedding_provider()` |
| `data_provider/data_loader.py:390-394` | LLM embedding injection into `x_hetero` |
| `embedder/llm_embedding_provider.py:58-157` | `LLMEmbeddingProvider.from_experiment_config()` |
| `embedder/llm_embedder.py` | LLM embedding generation |
| `embedder/llm_cache.py` | LLM embedding HDF5 caching |

### Cache Hash Computation

| Location | Purpose |
|----------|---------|
| `utils/experiment_config_builder.py:349-394` | `build_cache_config()` |

---

## Revision History

| Date | Change |
|------|--------|
| 2025-01-19 | Initial analysis documenting incompatibility and future plans |
