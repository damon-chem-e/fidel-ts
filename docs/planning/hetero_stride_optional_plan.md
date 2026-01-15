# Hetero Stride: Current State and Optional Striding Plan

## Overview

This document describes the current implementation of `hetero_stride` (text embedding striding), where it is applied across the codebase, and a plan to make striding optional via experiment config.

## What is Hetero Stride?

`hetero_stride` reduces the number of text embedding timesteps to match the patch stride used by time series models. For example:
- `input_len = 24` timesteps
- `stride = 3` (from model config)
- `hetero_stride = 3` when `hetero_align_stride = True`
- Result: `ceil(24/3) = 8` text embedding timesteps instead of 24

**Purpose:** Originally designed to reduce RAM usage and align text embeddings with patch boundaries in TGTSF-style models.

---

## Current Implementation Status

### 1. Dataloader (`data_provider/data_loader.py`)

**STATUS: IMPLEMENTED**

```python
# Initialization (line 128)
self.hetero_stride = hetero_stride  # From model_config.stride if hetero_align_stride else 1

# __getitem__ (lines 371-393)
# Preloaded mode:
x_hetero = self.full_hetero[s_begin:s_end:self.hetero_stride]
y_hetero = self.full_hetero[r_begin:r_end:self.hetero_stride]

# On-demand mode:
x_hetero = self.hetero_data_getter(x_time[::self.hetero_stride])
```

**Striding Enabled For:**
- All datasets (Time-MMD, TTC, Fidel-TS)
- When model config has `hetero_align_stride: True`

---

### 2. Data Factory (`data_provider/data_factory.py`)

**STATUS: IMPLEMENTED (passes stride to dataloader)**

```python
# Line 764
hetero_stride=self.args.model_config.stride if self.args.model_config.hetero_align_stride else 1
```

---

### 3. Tensor Cache (`data_provider/tensor_cache.py`)

**STATUS: RECENTLY FIXED (applies stride on READ)**

```python
# _init_indexed_format (line 1561)
self.hetero_stride = self.metadata.data_config.get('hetero_stride', 1)

# _getitem_indexed (lines 1645-1646)
x_hetero_idx = x_idx[::self.hetero_stride]
y_hetero_idx = y_idx[::self.hetero_stride]
```

**Note:** Tensor cache stores FULL resolution embeddings and applies stride on read.

---

### 4. Embedders

#### Main Embedder (`embedder/embedder.py`, `embedder/fidel_ts_embedder.py`)

**STATUS: NO STRIDING (generates full resolution)**

The main embedder used for Time-MMD, TTC, and Fidel-TS datasets has **no striding logic whatsoever**. It generates embeddings for every timestamp at full resolution.

- `embedder.py` - Base embedding functionality
- `fidel_ts_embedder.py` - Fidel-TS specific embedder
- `cache_manager.py` - Embedding cache management
- `llm_embedding_provider.py` - Per-sample embedding provider

None of these files reference `stride` or `hetero_stride`.

**Result:** Embeddings are always generated at full resolution, which is correct.

#### LLM Embedder (`embedder/llm_embedder.py`)

**STATUS: HARDCODED TO stride=1 (only used for TimeCMA/MMTSFlib)**

```python
# Lines 1091-1092, 1311-1312, 1401-1402
'model_config': dotdict({
    'stride': 1,
    'hetero_align_stride': False,
    ...
})
```

**Note:** This embedder is only used for TimeCMA and MMTSFlib models, NOT for lynx/TGTSF models.

---

### 5. Time-MMD Dataset (`data_provider/time_mmd_dataset.py`)

**STATUS: INHERITED FROM Universal_Dataset**

```python
# Line 560: Accepts hetero_stride parameter
def __init__(self, ..., hetero_stride=1, ...):
    ...
    super().__init__(..., hetero_stride=hetero_stride, ...)
```

---

### 6. Models Using Stride for FiLM

#### `layers/lynx_film_layers.py` (iTransformerFilm)

**STATUS: CALCULATES text_seq_len FROM STRIDE**

```python
# Lines 77-95
hetero_align_stride = getattr(configs, 'hetero_align_stride', True)
hetero_stride = stride if hetero_align_stride else 1

if timestamp_semantics == 't_about':
    self.text_seq_len = int(np.ceil(self.pred_len / hetero_stride))
else:  # t_known
    self.text_seq_len = int(np.ceil(self.seq_len / hetero_stride))
```

**Critical:** FiLMGenerator has FIXED input dimension = `text_seq_len * text_dim`. Mismatch causes runtime error.

#### `layers/FiLM_layers.py` (FiLMGenerator)

```python
# Line 60-61
input_dim = seq_len * text_dim  # FIXED at init time
self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), ...)
```

#### `models/lynx_film_enhanced.py`

Similar pattern to `lynx_film_layers.py`.

#### `layers/enhanced_film_layers.py`

Uses `text_seq_len` parameter passed from model.

---

### 7. Model Configs with `hetero_align_stride`

| Model Config | stride | hetero_align_stride | Notes |
|-------------|--------|---------------------|-------|
| lynx.yaml | 3 | True | Uses hetero striding |
| lynx_film.yaml | 3 | True | Uses hetero striding |
| lynx_film_raw.yaml | 3 | True | Uses hetero striding |
| lynx_film_enhanced.yaml | 3 | True | Uses hetero striding |
| TGTSF.yaml | 3 | True | Uses hetero striding |
| TGTSF-Bear.yaml | 6 | True | Uses hetero striding |
| TGTSF-CAISO.yaml | 3 | True | Uses hetero striding |

### 8. Model Configs with `stride` but NO `hetero_align_stride`

These models use `stride` for time series patching only (NOT for text striding):

| Model Config | stride | Notes |
|-------------|--------|-------|
| PatchTST.yaml | 8 | Patching only |
| GPT4TS.yaml | 8 | Patching only |
| GPT4MTS.yaml | 3 | Patching only |
| LeRet.yaml | 8 | Patching only |
| MMTSFlib.yaml | 8 | Patching only |
| TimeLLM.yaml | 8 | Patching only |
| ZhangHanBest.yaml | 8 | Patching only |

---

## Current Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EMBEDDING GENERATION                              │
│  embedder/embedder.py, fidel_ts_embedder.py                                 │
│  - NO STRIDING - generates full resolution embeddings                       │
│  - Stores in embedding cache (.pkl files)                                   │
│  - Used for: Time-MMD, TTC, Fidel-TS datasets                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TENSOR CACHE GENERATION                           │
│  data_provider/tensor_cache.py (TensorCacheGenerator)                       │
│  - Reads from dataloader (which MAY apply stride)                           │
│  - Stores hetero_stride in metadata                                         │
│  - Stores embeddings at whatever resolution dataloader provides             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING                                       │
├─────────────────────────────┬───────────────────────────────────────────────┤
│   NORMAL DATALOADER         │   TENSOR CACHE DATALOADER                     │
│   data_loader.py            │   tensor_cache.py (TensorCacheDataset)        │
│   - Applies hetero_stride   │   - Applies hetero_stride on READ             │
│     at __getitem__ time     │     from metadata.data_config                 │
└─────────────────────────────┴───────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               MODEL                                         │
│  - Calculates text_seq_len based on hetero_stride                           │
│  - FiLMGenerator expects text_seq_len timesteps                             │
│  - Dimension mismatch = RuntimeError                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Problems with Current Implementation

1. **No way to disable striding** - `hetero_align_stride` is in model config, not experiment config
2. **Inconsistent application** - Some models use stride for patching only, others for text
3. **Tensor cache complexity** - Cache must store stride in metadata, apply on read
4. **Cache invalidation** - Changing stride requires cache regeneration
5. **No full-resolution option** - Cannot easily test models with full text resolution

---

## Plan: Make Striding Optional

### Goal

Allow experiments to specify whether to use strided or full-resolution text embeddings via experiment config, independent of model's patching stride.

### Proposed Config Structure

```yaml
# Experiment config
training:
  # ... other training params ...
  
  # NEW: Control text embedding resolution
  # Options: "aligned" (use model stride), "full" (no striding), or explicit integer
  text_embedding_stride: "full"  # or "aligned" or 3
```

### Implementation Plan

#### Phase 1: Add Config Parameter

**Files to modify:**
- `utils/experiment_config_builder.py` - Extract new param
- `cli/config/models.py` - Add to training config model

**New parameter:** `text_embedding_stride`
- `"aligned"` (default): Use `model_config.stride` if `hetero_align_stride=True`, else 1
- `"full"`: Always use stride=1 (full resolution)
- `<integer>`: Use explicit stride value

---

#### Phase 2: Update Data Factory

**File:** `data_provider/data_factory.py`

```python
# Current (line 764):
hetero_stride = self.args.model_config.stride if self.args.model_config.hetero_align_stride else 1

# Proposed:
def _compute_hetero_stride(self):
    """Compute effective hetero_stride from config."""
    text_stride_config = getattr(self.args, 'text_embedding_stride', 'aligned')
    
    if text_stride_config == 'full':
        return 1
    elif text_stride_config == 'aligned':
        if getattr(self.args.model_config, 'hetero_align_stride', False):
            return self.args.model_config.stride
        return 1
    elif isinstance(text_stride_config, int):
        return text_stride_config
    else:
        raise ValueError(f"Invalid text_embedding_stride: {text_stride_config}")
```

---

#### Phase 3: Update Tensor Cache

**File:** `data_provider/tensor_cache.py`

1. **Cache generation** - Store effective `hetero_stride` in metadata (already done)
2. **Cache reading** - Apply stride from metadata (recently fixed)
3. **Cache hash** - Include `text_embedding_stride` in hash

**File:** `utils/experiment_config_builder.py`

```python
# Update build_cache_config() to use resolved stride
def build_cache_config(args: dotdict) -> dict:
    # Compute effective stride
    text_stride_config = getattr(args, 'text_embedding_stride', 'aligned')
    if text_stride_config == 'full':
        hetero_stride = 1
    elif text_stride_config == 'aligned':
        hetero_stride = args.model_config.get('stride', 1) if args.model_config.get('hetero_align_stride') else 1
    else:
        hetero_stride = int(text_stride_config)
    
    return {
        ...,
        'hetero_stride': hetero_stride,
        ...
    }
```

---

#### Phase 4: Update Models

**Files to modify:**
- `layers/lynx_film_layers.py`
- `layers/enhanced_film_layers.py`
- `models/lynx_film_enhanced.py`

**Change:** Read `text_seq_len` from config instead of calculating from stride.

```python
# Current:
hetero_stride = stride if hetero_align_stride else 1
self.text_seq_len = int(np.ceil(self.seq_len / hetero_stride))

# Proposed:
# Option A: Pass text_seq_len directly in config
self.text_seq_len = configs.text_seq_len  # Computed by data_factory

# Option B: Compute from resolved hetero_stride
self.text_seq_len = int(np.ceil(self.seq_len / configs.hetero_stride))
```

**Preferred: Option A** - Data factory computes `text_seq_len` and passes it to model config.

---

#### Phase 5: Validation

1. **Test with tensor cache** - Verify stride applied correctly on read
2. **Test without tensor cache** - Verify stride applied in dataloader
3. **Test all datasets** - Time-MMD, TTC, Fidel-TS
4. **Test all affected models** - lynx, lynx_film, lynx_film_raw, lynx_film_enhanced, TGTSF

---

### Migration Path

1. **Default to current behavior** - `text_embedding_stride: "aligned"` as default
2. **Add deprecation warnings** - Warn if using old config structure
3. **Document breaking changes** - Note that cache regeneration may be required

---

## Summary Table: Component Status

| Component | Current Status | Needs Update | Priority |
|-----------|---------------|--------------|----------|
| data_loader.py | Striding applied | Add config param | P1 |
| data_factory.py | Computes stride | Use new config | P1 |
| tensor_cache.py | Stride on read (fixed) | Include in hash | P1 |
| experiment_config_builder.py | Has hetero_stride | Compute from new param | P1 |
| lynx_film_layers.py | Calculates text_seq_len | Use config value | P2 |
| enhanced_film_layers.py | Calculates text_seq_len | Use config value | P2 |
| models/lynx_film_enhanced.py | Calculates text_seq_len | Use config value | P2 |
| embedder/embedder.py | No striding (full res) | No change needed | - |
| embedder/fidel_ts_embedder.py | No striding (full res) | No change needed | - |
| embedder/llm_embedder.py | Hardcoded stride=1 (TimeCMA only) | No change needed | - |
| time_mmd_dataset.py | Inherits stride | No change needed | - |

---

## Testing Matrix

| Dataset | Model | text_embedding_stride | Expected text_seq_len |
|---------|-------|----------------------|----------------------|
| Time-MMD Traffic | lynx_film_raw | "aligned" (stride=3) | ceil(24/3) = 8 |
| Time-MMD Traffic | lynx_film_raw | "full" | 24 |
| TTC Medical | lynx_film_raw | "aligned" (stride=3) | ceil(24/3) = 8 |
| TTC Medical | lynx_film_raw | "full" | 24 |
| Bear Room | TGTSF | "aligned" (stride=6) | ceil(288/6) = 48 |
| Bear Room | TGTSF | "full" | 288 |
| California ISO | lynx | "aligned" (stride=3) | ceil(168/3) = 56 |
| California ISO | lynx | "full" | 168 |

---

## Full Resolution vs Strided: Trade-off Analysis

### What Information is Lost with Striding?

Striding **discards text embeddings** at non-stride timesteps. This is a form of temporal downsampling that assumes text information is smooth/redundant across adjacent timesteps.

#### Example 1: Hourly Energy Data with stride=3

```
Input window: 24 hours (t0 to t23)
Stride: 3

FULL RESOLUTION (24 embeddings):
  t0:  "Morning demand begins, solar output low"
  t1:  "Peak breakfast cooking load"
  t2:  "Industrial facilities coming online"
  t3:  "Solar generation ramping up"
  t4:  "Air conditioning loads increasing"
  t5:  "UNEXPECTED: Grid frequency event at 11am"  ← CRITICAL EVENT
  t6:  "Recovery from frequency event"
  ...
  t23: "Night demand trough"

STRIDED (8 embeddings, stride=3):
  t0:  "Morning demand begins, solar output low"     ✓ kept
  t3:  "Solar generation ramping up"                 ✓ kept
  t6:  "Recovery from frequency event"               ✓ kept
  t9:  ...                                           ✓ kept
  
  LOST: t1, t2, t4, t5, t7, t8, t10, t11, ...
  
  ⚠️ The critical grid frequency event at t5 is COMPLETELY LOST!
```

#### Example 2: Traffic Data with stride=3

```
Input window: 24 hours (hourly)
Stride: 3

Timeline with events:
  t0:  "Normal traffic flow"
  t1:  "Minor accident on highway" ← LOST
  t2:  "Accident cleared" ← LOST  
  t3:  "Traffic returning to normal"
  t4:  "School dismissal begins" ← LOST
  t5:  "Heavy school zone traffic" ← LOST
  t6:  "Evening commute starts"
  ...

With stride=3, the model sees: t0, t3, t6, t9, ...
It MISSES the accident (t1-t2) and school traffic (t4-t5) entirely!
```

#### Example 3: When Striding is Safe

Striding works well when text information is:
- **Temporally smooth**: Same general context across hours (e.g., "sunny day" persists)
- **Redundant**: Adjacent timestamps have similar text
- **Low event density**: Important events are sparse and align with stride

```
Safe scenario (weather forecasts):
  t0:  "Clear skies, high of 72°F"
  t1:  "Clear skies, high of 72°F"     ← Same as t0
  t2:  "Clear skies, high of 72°F"     ← Same as t0
  t3:  "Clear skies, high of 72°F"     ← Stride captures this
  ...
  
Here, losing t1, t2 doesn't lose information because they're identical to t0.
```

### Resource Impact Analysis

#### FiLMGenerator Parameters

```
FiLMGenerator architecture:
  Input:  text_seq_len × text_dim → flattened
  Layer1: Linear(input_dim, hidden_dim) + GELU
  Layer2: Linear(hidden_dim, output_dim × 4)  # gamma1, beta1, gamma2, beta2

Example with input_len=24, text_dim=256:

STRIDED (stride=3):
  text_seq_len = ceil(24/3) = 8
  input_dim = 8 × 256 = 2,048
  Layer1: 2,048 → 256 = 524,288 params + 256 bias = 524,544 params
  Layer2: 256 → 1,024 = 262,144 params + 1,024 bias = 263,168 params
  Total per FiLMGenerator: ~788K params

FULL RESOLUTION:
  text_seq_len = 24
  input_dim = 24 × 256 = 6,144
  Layer1: 6,144 → 256 = 1,572,864 params + 256 bias = 1,573,120 params
  Layer2: 256 → 1,024 = 262,144 params + 1,024 bias = 263,168 params
  Total per FiLMGenerator: ~1.84M params

With e_layers=3 (3 FiLMGenerators):
  STRIDED: 3 × 788K = 2.36M params
  FULL:    3 × 1.84M = 5.52M params
  
Increase: +3.16M params (+134%)
```

#### VRAM Impact

```
Per-batch memory for text embeddings:
  Shape: [batch_size, text_seq_len, num_items, embed_dim]
  
Example: batch_size=128, num_items=1, embed_dim=768

STRIDED (text_seq_len=8):
  128 × 8 × 1 × 768 × 4 bytes = 3.15 MB

FULL (text_seq_len=24):
  128 × 24 × 1 × 768 × 4 bytes = 9.44 MB
  
Increase: +6.29 MB per batch (+200%)

FiLMGenerator activations (approximate):
  STRIDED: ~5 MB per batch
  FULL:    ~15 MB per batch
  
Total VRAM increase: ~15-20 MB per batch
This is NEGLIGIBLE on modern GPUs (40-80GB)
```

#### Training Time Impact

```
With Tensor Cache:
  - Embeddings are PRE-COMPUTED and stored
  - No embedding generation during training
  - Only difference: FiLM forward pass computation

FiLM computation increase:
  - Input Linear: 3× more multiply-adds (2048 → 6144 input)
  - Hidden/Output Linear: SAME (256 → 1024)
  
Estimated overhead per forward pass: <1ms on GPU
As percentage of total forward pass: <5%

With deduplicated tensor cache:
  - Embedding lookup is O(1) index operation
  - No additional disk I/O for full resolution
  - Memory-mapped numpy arrays make access fast
```

### Recommendation: Use Full Resolution for lynx_film_raw

Given the analysis above, **full resolution is recommended** for `configs/experiment_suites/lynx_film_raw/` because:

1. **Information preservation**: No risk of losing important events
2. **Minimal VRAM impact**: ~15-20 MB extra per batch (negligible on 40GB+ GPUs)
3. **Minimal training time impact**: <5% overhead from FiLM computation
4. **Tensor cache efficiency**: Deduplicated storage means full resolution doesn't multiply disk usage
5. **Research validity**: Full information allows fair comparison of model architectures

#### When to Use Striding

Striding may still be useful for:
- **VRAM-constrained environments**: <16GB GPU memory
- **Very long input windows**: input_len > 288 where text_seq_len becomes large
- **Matching original TGTSF paper**: For reproduction studies
- **Known-redundant text**: When text is genuinely uniform across timesteps

### Quick Config Change for lynx_film_raw

To use full resolution in experiment configs, override `hetero_align_stride`:

```yaml
# In configs/experiment_suites/lynx_film_raw/*.yaml
model_config_overrides:
  hetero_align_stride: false  # Disables striding, uses full resolution
```

Or with the proposed `text_embedding_stride` parameter (after implementation):

```yaml
training:
  text_embedding_stride: "full"  # Explicit full resolution
```

---

## Open Questions

1. **Should full-resolution be the default?** 
   - **Recommendation: YES for lynx_film and lynx_film_raw**
   - Pro: More information, no risk of missing alignment
   - Con: Higher memory usage, may not match original TGTSF design
   - Con is negligible with modern GPUs and tensor cache

2. **Should embedder generate strided embeddings?**
   - Current: Main embedder (`embedder.py`, `fidel_ts_embedder.py`) has NO striding
   - This is CORRECT - striding should happen at load time, not generation
   - Generating at full resolution allows flexibility to choose stride at training time

3. **Cache regeneration requirement**
   - Changing `text_embedding_stride` requires tensor cache regeneration
   - Embedding cache does NOT need regeneration (already full resolution)
   - Should we support multiple stride values per tensor cache? (Complex, probably not worth it)

4. **Ablation study opportunity**
   - Compare full resolution vs strided on same dataset/model
   - Hypothesis: Full resolution will improve performance on event-dense data
   - May have minimal impact on smooth/redundant text data
