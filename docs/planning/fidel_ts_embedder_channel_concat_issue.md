# Fidel-TS Embedder Channel Concatenation Issue

## Executive Summary

**Issue**: The `fidel_ts_embedder.py` embedding generation code concatenates per-variable channel descriptions into a single embedding for multi-variable datasets, causing runtime failures in TGTSF and suboptimal performance in LYNX/FILM models.

**Impact**:
- **TGTSF**: Runtime failure (`RuntimeError: shape '[256, 119, 256]' is invalid for input of size 163774464`)
- **LYNX/FILM**: Silent degradation (single description broadcast to all variables)
- **Affected Datasets**: Jena_Atmospheric_Physics (21 vars), Bear_room (16 vars), California_ISO (8 vars), and any dataset with nested `channel_info` structure

**Root Cause**: Line 796-803 in `embedder/fidel_ts_embedder.py` flattens nested channel descriptions instead of creating per-variable embeddings.

**Status**:
- ✅ Root cause identified
- ✅ Workaround designed (broadcasting with warning)
- ⏳ Proper fix designed (per-variable embedding generation)
- ⏳ Implementation pending

---

## Table of Contents

1. [Problem Description](#problem-description)
2. [Root Cause Analysis](#root-cause-analysis)
3. [Why FILM Works But TGTSF Fails](#why-film-works-but-tgtsf-fails)
4. [Affected Datasets](#affected-datasets)
5. [Solution 1: Immediate Workaround (Broadcasting)](#solution-1-immediate-workaround-broadcasting)
6. [Solution 2: Proper Fix (Per-Variable Embeddings)](#solution-2-proper-fix-per-variable-embeddings)
7. [Implementation Plan](#implementation-plan)
8. [Testing Strategy](#testing-strategy)

---

## Problem Description

### Symptom

When training TGTSF on Jena_Atmospheric_Physics dataset:

```
RuntimeError: shape '[256, 119, 256]' is invalid for input of size 163774464

from user code:
   File "models/TGTSF.py", line 400, in forward
    x, mix_weights = self.mixer(t, x)
  File "layers/TGTSF_torch.py", line 98, in forward
    temp_emb=temp_emb.reshape(B*C, L_temp, D_temp)
```

### Expected vs Actual Behavior

**Expected**:
- Jena has 21 time series variables (temperature, pressure, humidity, etc.)
- Each variable should have its own channel description embedding
- `channel_description` tensor shape: `[batch_size, nvars, embed_dim]` = `[256, 21, 768]`

**Actual**:
- All 21 variable descriptions concatenated into single string
- Single embedding generated for the concatenated string
- `channel_description` tensor shape: `[batch_size, 1, embed_dim]` = `[256, 1, 768]`

**Consequence**: TGTSF mixer expects C=21 but receives C=1, causing reshape failure.

---

## Root Cause Analysis

### Data Flow Architecture

```
static_info.json
    └─> channel_info: {entity_id: {var1: desc1, var2: desc2, ...}}
             ↓
    fidel_ts_embedder.py::_compute_static_embeddings()
             ↓ [BUG HERE: concatenates all variable descriptions]
    static_embeddings.pkl
             ↓ channel_info: {entity_id: single_concatenated_embedding}
    data_loader.py::init_hetero_data()
             ↓ [normalizes to (1, embed_dim)]
    hetero_data_getter(timestamps)
             ↓ returns channel_info: (1, 768)
    Universal_Dataset::__getitem__()
             ↓
    DataLoader collate
             ↓ stacks batch: (B, 1, 768)
    Model forward()
             ↓ expects: (B, nvars, 768)
    FAILURE: C=1 instead of C=nvars
```

### Bug Location

**File**: `embedder/fidel_ts_embedder.py`
**Lines**: 796-803
**Function**: `_compute_static_embeddings()`

```python
# Collect channel_info texts
if 'channel_info' in static_text:
    channel_info_text = static_text['channel_info']
    if isinstance(channel_info_text, dict):
        for channel_id, channel_text in channel_info_text.items():
            # Handle nested dict structure (e.g., Bear_room has {measurement_type: description})
            if isinstance(channel_text, dict):
                # ❌ BUG: Concatenates all measurement descriptions into a single string
                flattened_text = ' '.join(str(v) for v in channel_text.values() if v)
                if flattened_text.strip():
                    static_texts_list.append(flattened_text)  # Single embedding!
                    key = f'channel_info_{channel_id}'
                    text_keys_list.append(key)
                    keys_to_type[key] = 'channel_info'
```

**What Should Happen**: Create **separate embedding for each variable** in the nested dict.

**What Actually Happens**: Concatenates all variable descriptions → creates **one embedding per entity**.

### Example: Jena_Atmospheric_Physics

**Static Info Structure**:
```json
{
  "channel_info": {
    "weather_large": {
      "p (mbar)": "Atmospheric pressure measured in millibars...",
      "T (degC)": "Temperature at the point of observation...",
      "Tpot (K)": "Potential temperature, given in Kelvin...",
      ...  // 21 variables total
    }
  }
}
```

**Current Behavior**:
- Concatenates all 21 descriptions: `"Atmospheric pressure... Temperature... Potential temperature..."`
- Generates 1 embedding: `(768,)` → normalized to `(1, 768)`
- Result: `channel_info["weather_large"]` = single 768-dim array

**Expected Behavior**:
- Generate 21 separate embeddings, one per variable
- Result: `channel_info["weather_large"]` = dict of 21 embeddings OR array of shape `(21, 768)`

---

## Why FILM Works But TGTSF Fails

This is the KEY insight explaining why LYNX/FILM models succeed on multi-variable datasets while TGTSF crashes.

### Architecture Comparison

#### TGTSF Architecture

```
Input:
  - x (time series): [B, seq_len, nvars]
  - channel_description: [B, nvars, embed_dim]  ← REQUIRES per-variable!
  - text_input (news): [B, L, num_news, embed_dim]

Processing:
  1. TS_encoder(x) → [B, patch_num, nvars, d_model]
  2. text_encoder(text_input, channel_description) → [B, L, nvars, d_model]
     └─> Repeats channel_description along L dimension: [B, 1, nvars, D] → [B, L, nvars, D]
  3. mixer(text_emb, ts_emb) → [B, patch_num, nvars, d_model]
     ├─> Extracts C from text_emb: B, L_text, C, D_text = text_emb.shape
     ├─> Permutes both: text_emb → [B, C, L, D], temp_emb → [B, C, N, D]
     └─> Reshapes both: text_emb.reshape(B*C, L, D), temp_emb.reshape(B*C, N, D)
         ⚠️ REQUIRES: C from text matches C from time series!
```

**CRITICAL LINE** (`layers/TGTSF_torch.py:86-98`):
```python
def forward(self, text_emb, temp_emb):
    # text_emb:  [b, l, c, d]
    # temp_emb:  [b, n, c, d]

    B, L_text, C, D_text = text_emb.shape  # ← C extracted from text!
    _, L_temp, _, D_temp = temp_emb.shape

    text_emb = text_emb.permute(0, 2, 1, 3)  # [b, c, l, d]
    temp_emb = temp_emb.permute(0, 2, 1, 3)  # [b, c_ts, n, d]  ← c_ts = actual nvars

    text_emb = text_emb.reshape(B*C, L_text, D_text)
    temp_emb = temp_emb.reshape(B*C, L_temp, D_temp)  # ← Uses C from text!
    # ❌ FAILS if C (from text) ≠ c_ts (from time series)
```

**For Jena with bug**:
- text_emb: `[256, L, 1, 256]` → C=1
- temp_emb: `[256, 119, 21, 256]` → c_ts=21
- Line 98: `temp_emb.reshape(256*1, 119, 256)` tries to reshape `[256, 21, 119, 256]` to `[256, 119, 256]`
- **Size mismatch**: 163,774,464 elements → 7,798,784 elements ❌

#### LYNX/FILM Architecture

```
Input:
  - x (time series): [B, seq_len, nvars]
  - channel_description: [B, 1, embed_dim]  ← Works with single description!
  - text_input (news): [B, L, num_news, embed_dim]

Processing:
  1. unimodal_wrapper.predict(x) → [B, pred_len, nvars] (frozen baseline)
  2. text_encoder(text_input, channel_description) → [B, L, 1, text_dim]
     └─> Unsqueeze & repeat: [B, 1, D] → [B, 1, 1, D] → [B, L, 1, D]
  3. Permute: text_emb → [B, 1, L, D]
  4. iTransformerFilm(x_norm, text_emb) → [B, pred_len, nvars]
     └─> FiLM modulation: broadcasts single text_emb across all nvars
  5. Final: residual + baseline

⚠️ NO MIXER! No dimension matching required!
```

**Key Difference**:
```python
# FILM (models/lynx_film.py:190-195)
if len(channel_description.shape) == 3:  # [B, 1, D]
    channel_description = channel_description.unsqueeze(1)  # [B, 1, 1, D]
description = channel_description.repeat(1, text_input.shape[1], 1, 1)  # [B, L, 1, D]
text_emb = self.text_encoder(text_input, description)  # [B, L, 1, D]

# text_emb has C=1, but this is FINE because:
# - iTransformerFilm broadcasts FiLM params across all variables
# - No reshape operation that requires C to match nvars
```

### Why FILM "Works" (But Suboptimally)

✅ **No runtime error**: FiLM broadcasting allows C=1 to work with any nvars
⚠️ **Semantic loss**: All variables get same channel description (concatenated mess)
⚠️ **Suboptimal performance**: Model can't distinguish between variables semantically

**Example**: For Jena, FILM sees:
- Single description: "Atmospheric pressure... Temperature... Humidity..." (all concatenated)
- Broadcasts this to all 21 variables
- Cannot learn that "pressure" description applies specifically to pressure measurements

**Proper behavior** would give each variable its own description:
- Variable 0 (pressure): "Atmospheric pressure measured in millibars..."
- Variable 1 (temperature): "Temperature at the point of observation..."
- Variable 2 (humidity): "Relative humidity, expressed as a percentage..."

---

## Affected Datasets

### Datasets with Nested channel_info (Multi-Variable)

These datasets have `channel_info[entity_id] = {var1: desc1, var2: desc2, ...}` structure:

1. **Jena_Atmospheric_Physics**
   - Entities: 1 (`weather_large`)
   - Variables per entity: 21 (pressure, temperature, humidity, etc.)
   - Status: ❌ TGTSF fails, ⚠️ FILM suboptimal

2. **Bear_room**
   - Entities: Multiple rooms
   - Variables per entity: 16 measurements (temperature, CO2, humidity, etc.)
   - Status: ❌ TGTSF fails, ⚠️ FILM suboptimal

3. **California_ISO (CAISO)**
   - Entities: Multiple zones
   - Variables per entity: 8 (solar, wind, load, etc.)
   - Status: ❌ TGTSF fails, ⚠️ FILM suboptimal

### Datasets with Flat channel_info (Single Variable)

These datasets have `channel_info[entity_id] = "description"` structure:

1. **Germany_Renewable_Power_Grid**
   - Entities: 8 (solar_50Hertz, wind_Amprion, etc.)
   - Variables per entity: 1
   - Status: ✅ Works correctly (no concatenation issue)

2. **NYC_traffic_speed**
   - Entities: Multiple road segments
   - Variables per entity: 1 (speed)
   - Status: ✅ Works correctly

**Why single-variable datasets work**: No nested dict → no concatenation → correct embedding generated.

---

## Solution 1: Immediate Workaround (Broadcasting)

### Objective

Fix TGTSF runtime errors WITHOUT regenerating embeddings by automatically broadcasting single channel description to all variables.

### Implementation

#### Location 1: TGTSF Model (`models/TGTSF.py`)

**Insert after line 381** (after `_project_text_embeddings`):

```python
# Project text embeddings if input dimension differs from operational dimension
text_input, channel_description = self._project_text_embeddings(text_input, channel_description)

# ============================================================================
# WORKAROUND: Handle channel dimension mismatch (concatenated descriptions)
# ============================================================================
# channel_description should have shape [B, C, D] where C = number of variables
# Due to embedding generation bug, it may arrive as [B, 1, D] (concatenated)
# We broadcast the single description to all variables to prevent reshape errors
C_time_series = x.shape[2]  # Number of variables in time series

if channel_description.ndim == 3:
    # Expected: [B, C, D] where C = nvars
    C_desc = channel_description.shape[1]
    if C_desc == 1 and C_time_series > 1:
        # Mismatch detected: single description for multi-variable dataset
        print(f'[ WARNING ] TGTSF: channel_description has C={C_desc} but time series has '
              f'C={C_time_series} variables. Broadcasting single channel description to all '
              f'variables. This is a WORKAROUND due to embedding generation concatenating '
              f'per-variable descriptions. For optimal performance with per-variable semantic '
              f'descriptions, regenerate embeddings with the fixed fidel_ts_embedder.')
        channel_description = channel_description.repeat(1, C_time_series, 1)  # [B, 1, D] → [B, C, D]
elif channel_description.ndim == 4:
    # Expected: [B, 1, C, D] from some preprocessing
    C_desc = channel_description.shape[2]
    if C_desc == 1 and C_time_series > 1:
        print(f'[ WARNING ] TGTSF: channel_description has C={C_desc} but time series has '
              f'C={C_time_series} variables. Broadcasting single channel description.')
        channel_description = channel_description.repeat(1, 1, C_time_series, 1)  # [B, 1, 1, D] → [B, 1, C, D]

# Continue with existing code
channel_description = channel_description.unsqueeze(1) # [bs, 1, nvars, text_dim]
description = channel_description.repeat(1, text_input.shape[1], 1, 1) # [bs, l, nvars, text_dim]
```

#### Location 2: LYNX/FILM Models

**Purpose**: Add warning to inform users they're using concatenated descriptions.

**File**: `models/lynx_film.py` (and `lynx_film_raw.py`, `lynx_film_enhanced.py`)
**Insert after line 184** (after `_project_text_embeddings`):

```python
# Step 3: Project text embeddings if input dimension differs from operational dimension
text_input, channel_description = self._project_text_embeddings(text_input, channel_description)

# ============================================================================
# PERFORMANCE WARNING: Check for concatenated channel descriptions
# ============================================================================
# LYNX/FILM models can operate with C=1 (single description broadcast to all variables)
# due to FiLM's inherent broadcasting mechanism. However, this is suboptimal when
# per-variable descriptions are available but were concatenated during embedding generation.
C_time_series = x.shape[2]  # Number of variables in time series

if len(channel_description.shape) == 3:  # [B, C, D]
    C_desc = channel_description.shape[1]
    if C_desc == 1 and C_time_series > 1:
        print(f'[ INFO ] LYNX/FILM: Using single channel description for {C_time_series} variables. '
              f'This works but is suboptimal if per-variable descriptions exist. '
              f'Consider regenerating embeddings with fixed fidel_ts_embedder for improved '
              f'semantic alignment between text and time series variables.')
        # Note: No broadcasting needed for FILM - it naturally handles C=1

# Step 4: Get Text Embeddings
# ... existing code continues
```

### Trade-offs

#### Pros
✅ Immediate fix - no re-embedding required
✅ Backward compatible with existing tensor caches
✅ Works with current deployment
✅ No GPU needed (pure model-level fix)

#### Cons
❌ All variables receive same (concatenated) description
❌ Loss of semantic information (can't distinguish pressure from temperature)
❌ Suboptimal model performance
❌ Warning spam in logs for affected datasets

### Expected Behavior After Fix

**TGTSF on Jena**:
```
[ WARNING ] TGTSF: channel_description has C=1 but time series has C=21 variables.
Broadcasting single channel description to all variables. This is a WORKAROUND...

Training starts successfully ✅
```

**LYNX/FILM on Jena**:
```
[ INFO ] LYNX/FILM: Using single channel description for 21 variables. This works
but is suboptimal...

Training starts successfully ✅ (was already working, now with informative message)
```

---

## Solution 2: Proper Fix (Per-Variable Embeddings)

### Objective

Generate separate embeddings for each variable in nested `channel_info` structures, enabling proper semantic alignment between text descriptions and time series variables.

### Implementation

#### Phase 1: Embedding Generation Fix

**File**: `embedder/fidel_ts_embedder.py`
**Function**: `_compute_static_embeddings()`
**Lines**: 788-836

```python
# Collect channel_info texts
# Handles both flat (channel_id -> string) and nested (channel_id -> {measurement_type: string}) structures
if 'channel_info' in static_text:
    channel_info_text = static_text['channel_info']
    if isinstance(channel_info_text, dict):
        for channel_id, channel_text in channel_info_text.items():
            # ====================================================================
            # FIXED: Create separate embedding for each variable
            # ====================================================================
            if isinstance(channel_text, dict):
                # Nested structure: channel_id -> {var_name: description}
                # Generate per-variable embeddings and store as nested dict
                for var_name, var_description in channel_text.items():
                    if isinstance(var_description, str) and var_description.strip():
                        static_texts_list.append(var_description)
                        key = f'channel_info_{channel_id}_{var_name}'
                        text_keys_list.append(key)
                        keys_to_type[key] = 'channel_info_nested'
            elif isinstance(channel_text, str) and channel_text.strip():
                # Flat structure: channel_id -> description (Germany, NYC)
                # Keep existing behavior for single-variable datasets
                static_texts_list.append(channel_text)
                key = f'channel_info_{channel_id}'
                text_keys_list.append(key)
                keys_to_type[key] = 'channel_info'
        else:
            print(f'[ warning ] channel_info is not a dictionary, skipping')

# ... embed all texts in batch ...

# Map embeddings back to the correct structure
static_embeddings = {}

for i, key in enumerate(text_keys_list):
    emb = embeddings_array[i]
    text_type = keys_to_type[key]

    if text_type == 'general_info':
        static_embeddings['general_info'] = emb
    elif text_type == 'downtime_prompt':
        static_embeddings['downtime_prompt'] = emb
    elif text_type == 'channel_info':
        # Flat structure (single variable per entity)
        channel_id = key.replace('channel_info_', '')
        if 'channel_info' not in static_embeddings:
            static_embeddings['channel_info'] = {}
        static_embeddings['channel_info'][channel_id] = emb
    elif text_type == 'channel_info_nested':
        # ====================================================================
        # FIXED: Nested structure (multiple variables per entity)
        # ====================================================================
        # Parse key: 'channel_info_{entity_id}_{var_name}'
        parts = key.replace('channel_info_', '').split('_', 1)
        if len(parts) == 2:
            entity_id, var_name = parts
        else:
            # Fallback for complex variable names
            entity_id = parts[0]
            var_name = '_'.join(parts[1:]) if len(parts) > 1 else parts[0]

        if 'channel_info' not in static_embeddings:
            static_embeddings['channel_info'] = {}
        if entity_id not in static_embeddings['channel_info']:
            # Initialize as dict to hold per-variable embeddings
            static_embeddings['channel_info'][entity_id] = {}

        static_embeddings['channel_info'][entity_id][var_name] = emb

print(f'[ info ] Computed static embeddings with aggregation method: {self.aggregation_method}')
return static_embeddings
```

**Result**:
```python
# Old (buggy) format:
static_embeddings = {
    'channel_info': {
        'weather_large': np.array([...])  # Single (768,) array
    }
}

# New (fixed) format:
static_embeddings = {
    'channel_info': {
        'weather_large': {
            'p (mbar)': np.array([...]),      # (768,)
            'T (degC)': np.array([...]),      # (768,)
            'Tpot (K)': np.array([...]),      # (768,)
            ...  # 21 embeddings total
        }
    }
}
```

#### Phase 2: Data Loader Fix

**File**: `data_provider/data_loader.py`
**Function**: `init_hetero_data()`
**Line**: 797

```python
def init_hetero_data(self, id):
    # ... downtime handling code ...

    general_info = self.static_data['general_info']
    channel_info = self.static_data['channel_info'][id]

    # ============================================================================
    # FIXED: Handle nested channel_info (per-variable embeddings)
    # ============================================================================
    if isinstance(channel_info, dict):
        # Nested structure: per-variable embeddings
        # Stack into (nvars, embed_dim) array
        # CRITICAL: Variable order must match parquet file column order!

        # Get variable names from time series data to ensure correct ordering
        # Note: This assumes the Universal_Dataset has access to variable names
        # For now, we sort alphabetically (consistent with parquet column order)
        var_names = sorted(channel_info.keys())

        # Stack embeddings in sorted order
        channel_info = np.stack([channel_info[var_name] for var_name in var_names])
        # Result shape: (nvars, embed_dim)

    elif isinstance(channel_info, np.ndarray):
        # Flat structure: single embedding (already correct for single-variable datasets)
        # Normalize shape to (1, embedding_dim)
        channel_info = self._normalize_embedding_shape(channel_info)
    else:
        raise TypeError(
            f"Unexpected channel_info type: {type(channel_info)}. "
            f"Expected dict (nested) or np.ndarray (flat)."
        )

    # Note: channel_info now has shape (nvars, embed_dim) for nested structure
    # or (1, embed_dim) for flat structure

    downtime_prompt = self.static_data['downtime_prompt']
    # ... rest of function ...
```

**Critical Consideration**: Variable ordering must match time series column order in parquet files!

#### Phase 3: Tensor Cache Compatibility

**Impact**: Tensor cache automatically picks up new format through the data pipeline:

```
embedder (fixed)
  → generates per-variable embeddings
    → data_loader (fixed) stacks into (nvars, embed_dim)
      → Universal_Dataset passes to collate
        → DataLoader batches into (B, nvars, embed_dim)
          → Tensor cache stores in shared/entity_channel array
            → TensorCacheDataset loads correctly
```

**Action Required**:
1. Regenerate embeddings for affected datasets (Jena, Bear_room, CAISO)
2. Regenerate tensor cache for affected datasets
3. Verify cache hash changes (different embedding shapes → different hash)

#### Phase 4: Variable Name Ordering Issue

**Problem**: Variable order in embeddings must match time series column order.

**Solution Options**:

**Option A: Store Variable Metadata in Embeddings** (Recommended)
```python
# In fidel_ts_embedder.py, save metadata alongside embeddings
metadata = {
    'variable_order': {
        'weather_large': ['p (mbar)', 'T (degC)', 'Tpot (K)', ...]
    }
}
# Save in cache: metadata.json alongside embeddings
```

**Option B: Infer from Parquet File**
```python
# In data_loader.py, read parquet column names at init
parquet_file = self.root_path / self.formatter.format(i=id)
df = pd.read_parquet(parquet_file, nrows=0)  # Read just schema
var_names = df.columns.tolist()
```

**Recommendation**: Option A for robustness and performance (no file I/O at runtime).

### Trade-offs

#### Pros
✅ Proper semantic alignment (each variable gets its own description)
✅ Optimal model performance
✅ Clean architecture (no workarounds)
✅ Enables variable-specific text-TS relationships

#### Cons
❌ Requires re-embedding (GPU needed)
❌ Requires regenerating tensor cache
❌ More complex data structure
❌ Variable ordering dependency (fragile if not handled correctly)
❌ Breaks backward compatibility with existing embeddings

### Migration Strategy

1. **Implement Phase 1-2** (embedding generation + data loader)
2. **Test on small dataset** (verify shapes, ordering)
3. **Regenerate embeddings** for one affected dataset (e.g., Jena)
4. **Regenerate tensor cache** for that dataset
5. **Run training comparison**:
   - Old (concatenated): With workaround broadcasting
   - New (per-variable): With proper embeddings
6. **Measure performance improvement**
7. **Roll out to all affected datasets** if successful

---

## Implementation Plan

### Stage 1: Immediate Workaround (Week 1)

**Goal**: Unblock TGTSF training on multi-variable datasets.

**Tasks**:
1. ✅ Investigate root cause (COMPLETED)
2. ✅ Document issue comprehensively (COMPLETED - this document)
3. [ ] Implement broadcasting fix in `models/TGTSF.py`
4. [ ] Add informational warnings to `models/lynx_film*.py`
5. [ ] Test on Jena_Atmospheric_Physics:
   - Verify TGTSF trains without crashes
   - Verify warning message appears
   - Monitor training loss curves
6. [ ] Test on Bear_room and California_ISO
7. [ ] Update model documentation with workaround notes

**Deliverables**:
- Working TGTSF on all datasets (with suboptimal embeddings)
- Clear warnings guiding users toward proper fix
- Documentation of limitations

### Stage 2: Proper Fix Implementation (Week 2-3)

**Goal**: Enable per-variable embedding generation.

**Tasks**:
1. [ ] Implement embedding generation fix (Phase 1)
2. [ ] Implement data loader fix (Phase 2)
3. [ ] Add variable ordering metadata to embedding cache
4. [ ] Write unit tests:
   - Test nested channel_info embedding
   - Test variable ordering consistency
   - Test backward compatibility with flat structure
5. [ ] Update documentation

**Deliverables**:
- Fixed `fidel_ts_embedder.py`
- Fixed `data_loader.py`
- Test suite
- Updated documentation

### Stage 3: Re-Embedding & Validation (Week 4)

**Goal**: Regenerate embeddings for affected datasets and measure improvement.

**Tasks**:
1. [ ] Regenerate embeddings for Jena_Atmospheric_Physics
   - Run on GPU node
   - Verify per-variable embeddings created
   - Verify metadata saved correctly
2. [ ] Regenerate tensor cache for Jena
3. [ ] Run training experiments:
   - **Baseline**: TGTSF with old concatenated embeddings + broadcasting
   - **Proposed**: TGTSF with new per-variable embeddings
   - **Comparison**: LYNX/FILM with old vs new
4. [ ] Measure performance improvement:
   - MSE/MAE on test set
   - Attention weight analysis (do variables attend to correct descriptions?)
   - Ablation: per-variable vs concatenated
5. [ ] Document results
6. [ ] Repeat for Bear_room and California_ISO if successful

**Deliverables**:
- Re-embedded datasets with per-variable descriptions
- Performance comparison report
- Decision on rollout strategy

### Stage 4: Rollout & Cleanup (Week 5)

**Goal**: Apply proper fix to all datasets and deprecate workaround.

**Tasks**:
1. [ ] Re-embed all affected datasets
2. [ ] Regenerate all tensor caches
3. [ ] Remove/update workaround warnings:
   - Keep broadcasting code for backward compatibility
   - Change warning to deprecation notice
4. [ ] Update training scripts and documentation
5. [ ] Archive old embedding files
6. [ ] Communicate changes to team

**Deliverables**:
- All datasets using per-variable embeddings
- Deprecated workaround (kept for compatibility)
- Updated documentation
- Migration guide for other users

---

## Testing Strategy

### Unit Tests

```python
# tests/test_fidel_ts_embedder_channel_fix.py

def test_nested_channel_info_embedding():
    """Test that nested channel_info generates per-variable embeddings."""
    static_text = {
        'channel_info': {
            'entity1': {
                'var1': 'description 1',
                'var2': 'description 2',
                'var3': 'description 3'
            }
        }
    }
    embedder = FidelTSEmbedder(...)
    static_embeddings = embedder._compute_static_embeddings_from_text(static_text)

    # Verify nested structure
    assert 'channel_info' in static_embeddings
    assert 'entity1' in static_embeddings['channel_info']
    assert isinstance(static_embeddings['channel_info']['entity1'], dict)

    # Verify per-variable embeddings
    assert 'var1' in static_embeddings['channel_info']['entity1']
    assert 'var2' in static_embeddings['channel_info']['entity1']
    assert 'var3' in static_embeddings['channel_info']['entity1']

    # Verify shapes
    for var_emb in static_embeddings['channel_info']['entity1'].values():
        assert var_emb.shape == (768,)  # CLS/average aggregation

def test_flat_channel_info_backward_compatibility():
    """Test that flat channel_info still works (Germany, NYC datasets)."""
    static_text = {
        'channel_info': {
            'entity1': 'single description'
        }
    }
    embedder = FidelTSEmbedder(...)
    static_embeddings = embedder._compute_static_embeddings_from_text(static_text)

    # Verify flat structure
    assert 'channel_info' in static_embeddings
    assert 'entity1' in static_embeddings['channel_info']
    assert isinstance(static_embeddings['channel_info']['entity1'], np.ndarray)
    assert static_embeddings['channel_info']['entity1'].shape == (768,)

def test_data_loader_stacking():
    """Test that data_loader correctly stacks per-variable embeddings."""
    static_data = {
        'channel_info': {
            'entity1': {
                'var1': np.random.rand(768),
                'var2': np.random.rand(768),
                'var3': np.random.rand(768)
            }
        }
    }
    loader = Heterogeneous_Dataset(...)
    loader.static_data = static_data

    hetero_getter = loader.init_hetero_data('entity1')
    # ... call hetero_getter and verify shape ...
```

### Integration Tests

```python
# tests/test_tgtsf_channel_broadcasting.py

def test_tgtsf_with_concatenated_channel_desc():
    """Test TGTSF with broadcasting workaround (C=1 → C=nvars)."""
    # Setup: Load Jena with old concatenated embeddings
    config = load_config('configs/experiments/jena_tgtsf.yaml')
    config.data.use_old_embeddings = True  # Use concatenated format

    model = Model(config.model)
    data_provider = Data_Provider(config.data)
    train_loader, val_loader, test_loader = data_provider.get_data(...)

    # Verify shapes from dataloader
    for batch in train_loader:
        _, x, y, _, _, x_hetero, y_hetero, _, _, hetero_general, hetero_channel, _, _ = batch
        assert hetero_channel.shape == (config.training.batch_size, 1, 768)  # C=1
        break

    # Test forward pass (should broadcast automatically)
    output = model(x=x, news=y_hetero, channel_description=hetero_channel)
    assert output.shape == (config.training.batch_size, config.training.output_len, 21)

def test_tgtsf_with_proper_channel_desc():
    """Test TGTSF with proper per-variable embeddings (C=nvars)."""
    # Setup: Load Jena with new per-variable embeddings
    config = load_config('configs/experiments/jena_tgtsf.yaml')
    config.data.use_old_embeddings = False  # Use per-variable format

    model = Model(config.model)
    data_provider = Data_Provider(config.data)
    train_loader, val_loader, test_loader = data_provider.get_data(...)

    # Verify shapes from dataloader
    for batch in train_loader:
        _, x, y, _, _, x_hetero, y_hetero, _, _, hetero_general, hetero_channel, _, _ = batch
        assert hetero_channel.shape == (config.training.batch_size, 21, 768)  # C=21!
        break

    # Test forward pass (no broadcasting needed)
    output = model(x=x, news=y_hetero, channel_description=hetero_channel)
    assert output.shape == (config.training.batch_size, config.training.output_len, 21)
```

### Performance Comparison Tests

```python
# tests/benchmarks/test_tgtsf_embedding_comparison.py

def test_training_loss_comparison():
    """Compare training dynamics with concatenated vs per-variable embeddings."""
    # Run training for 10 epochs with both configurations
    results_old = train_tgtsf(use_old_embeddings=True, epochs=10)
    results_new = train_tgtsf(use_old_embeddings=False, epochs=10)

    # Log results for manual inspection
    print("Old (concatenated) final loss:", results_old['val_loss'][-1])
    print("New (per-variable) final loss:", results_new['val_loss'][-1])

    # We expect new to be better or equal
    # Note: This is a weak assertion - main value is in logged metrics
    assert results_new['val_loss'][-1] <= results_old['val_loss'][-1] * 1.1
```

---

## Appendix: Technical Details

### Embedding Shape Evolution

```
Static Info JSON
    channel_info["weather_large"]: {
        "p (mbar)": "...",
        "T (degC)": "...",
        ...  # 21 variables
    }
         ↓
OLD BEHAVIOR (BUG):
    Concatenate all → single string
         ↓
    Embed single string
         ↓
    channel_info["weather_large"]: (768,)
         ↓
    Normalize: (1, 768)
         ↓
    Collate batch: (B, 1, 768)
         ↓
    TGTSF receives: [B, 1, 768] where C=1 ❌
         ↓
    CRASH in mixer reshape

NEW BEHAVIOR (FIXED):
    Embed each variable separately
         ↓
    channel_info["weather_large"]: {
        "p (mbar)": (768,),
        "T (degC)": (768,),
        ...  # 21 embeddings
    }
         ↓
    Stack in data_loader: (21, 768)
         ↓
    Collate batch: (B, 21, 768)
         ↓
    TGTSF receives: [B, 21, 768] where C=21 ✅
         ↓
    SUCCESS - mixer reshape works correctly
```

### Memory Impact

**Old Format**:
- Per entity: 1 embedding × 768 dims × 4 bytes = 3 KB
- For 8 entities (Germany): 24 KB

**New Format**:
- Per entity: 21 embeddings × 768 dims × 4 bytes = 63 KB (Jena)
- Per entity: 16 embeddings × 768 dims × 4 bytes = 48 KB (Bear_room)

**Impact**:
- Negligible for embedding storage (~50 KB increase per entity)
- Tensor cache size unchanged (already stores per-sample data)
- Runtime memory: Minimal (channel descriptions are small)

### Backward Compatibility

**Breaking Changes**:
- Embedding file format changes (nested dict structure)
- Cache hash changes (different embedding shapes)

**Compatibility Strategy**:
- Keep `use_old_embeddings` flag for transition period
- Broadcasting workaround allows old embeddings to work with new model code
- Clear migration path: regenerate embeddings → regenerate cache → retrain

---

## References

- **Original Issue**: TGTSF training failure on Jena_Atmospheric_Physics
- **Root Cause**: `embedder/fidel_ts_embedder.py:796-803`
- **Related Models**: TGTSF, LYNX/FILM (all variants)
- **Related Code**:
  - `layers/TGTSF_torch.py`: mixer reshape logic
  - `data_provider/data_loader.py`: channel_info normalization
  - `models/TGTSF.py`: model forward pass
  - `models/lynx_film.py`: FILM text encoding

---

**Document Version**: 1.0
**Date**: 2026-01-20
**Author**: Claude Code (claude-sonnet-4-5)
**Status**: Ready for Implementation
