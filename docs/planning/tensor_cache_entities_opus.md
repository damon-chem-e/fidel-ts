# Comprehensive Bug Trace: Entity Mixing in Multi-Entity Datasets

## Executive Summary

The tensor cache and new text embedding cache both use **timestamp-only keys**, which causes **entity mixing** in multi-entity datasets like TTC Medical where different patients have different time series values and text at the same timestamps.

---

## Part 1: TTC Medical Dataset Structure

From `context/data_local/ttc/medical/id_info.json`:

```json
{
  "10576": { "description": "Time-MMD dataset entry: patient_10576.csv" },
  "10910": { "description": "Time-MMD dataset entry: patient_10910.csv" },
  ...
  // 72 total patients
}
```

**Key insight**: Each patient is a **separate entity** with:
- **Separate CSV file**: `patient_10576.csv`, `patient_10910.csv`, etc.
- **Unique time series values**: Each patient has different vital signs at any given time
- **Unique text notes**: Each patient has different clinical notes at any given time
- **Overlapping timestamps**: Medical data often uses absolute timestamps like `20200115120000`

This contrasts with TTC Climate:
```json
{ "all": { "description": "Time-MMD dataset" } }
```
Single-entity: all data comes from one file with one time series.

---

## Part 2: Correct Data Flow (Non-Tensor-Cache Path)

### Step 2.1: Dataset Creation in `data_factory.py`

```python
def get_datasets(self, flag):
    for i in self.id_list:  # e.g., ["10576", "10910", ...]
        if self._is_time_mmd_dataset():
            dataset = self._create_time_mmd_dataset(i, flag, llm_embedding_provider)
        datasets[i] = dataset  # Each entity gets its own dataset
```

Each entity (patient) gets its own `TimeMMD_Dataset` instance.

### Step 2.2: TimeMMD_Dataset Initialization

```python
# data_provider/time_mmd_dataset.py lines 557-636
def __init__(self, root_path, flag='train', data_path='ETTh1.csv', ...):
    # Store entity_id passed from data_factory
    # data_path is something like "patient_10576.csv" (per-entity file)
    
    # Call parent Universal_Dataset.__init__ which calls __read_data__
    super().__init__(root_path=root_path, data_path=data_path, ...)
    
    # Setup text getter after data is loaded
    self._setup_text_getter()
```

### Step 2.3: Loading Time Series Data (Correct - Per-Entity)

```python
# data_provider/time_mmd_dataset.py lines 736-906
def __read_data__(self):
    # Load THIS entity's CSV file
    df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
    # e.g., ./data/ttc/medical/patient_10576.csv
    
    # Extract text data indexed by timestamp
    text_series = df_raw[[self.timestamp_col, self._text_column_name]].copy()
    text_series[self.timestamp_col] = text_series[self.timestamp_col].dt.strftime('%Y%m%d%H%M%S').astype(np.int64)
    self._text_data = text_series.set_index(self.timestamp_col)[self._text_column_name]
    # self._text_data: {20200115120000: "Patient notes...", ...}
    
    # Time series data is stored in self.data, self.timestamp
```

**This is correct**: Each entity loads its own CSV file with its own time series values.

### Step 2.4: Text Embedding Creation

```python
# data_provider/time_mmd_dataset.py lines 691-734
def _setup_text_getter(self):
    text_getter = TimeMMD_HeteroGetter(
        text_data=self._text_data,          # THIS entity's text data
        timestamps=self.timestamp,           # THIS entity's timestamps
        root_path=self.root_path,
        data_path=self.data_path,           # "patient_10576.csv"
        ...
    )
    self.hetero_data_getter = text_getter
```

### Step 2.5: TimeMMD_HeteroGetter._load_or_create_embeddings() - **PROBLEM STARTS HERE**

```python
# data_provider/time_mmd_dataset.py lines 203-293
def _load_or_create_embeddings(self):
    # Convert text_data to dict format
    text_dict = {str(ts): text for ts, text in self.text_data.items()}
    # KEY FORMAT: Just timestamp strings like "20200115120000"
    # NO entity_id prefix!
    
    if self.use_old_pkl:
        pkl_path = self._get_embedding_path()  # data/ttc/medical/patient_10576.pkl
        # OLD PATH: Per-entity file - CORRECT!
        self.embeddings = joblib.load(pkl_path)
        return
    
    # NEW CACHE SYSTEM - PROBLEM!
    self._init_embedder()
    
    # embed_text_dict uses cache_manager with cache_path based on:
    # - dataset_root = self.root_path (./data/ttc/medical)
    # - dataset_path = subdirectory from data_path
    # NOT including entity_id!
    embeddings_dict = self.embedder.embed_text_dict(text_dict, format_for_time_mmd=True)
```

### Step 2.6: EmbeddingCacheManager Path Construction

```python
# embedder/cache_manager.py lines 25-48
def __init__(self, dataset_root: str, dataset_path: str = ''):
    self.dataset_root = Path(dataset_root)   # ./data/ttc/medical
    self.dataset_path = Path(dataset_path)   # Empty or subdirectory
    self.cache_base = self.dataset_root / self.dataset_path
    # Result: ./data/ttc/medical/

def get_cache_dir(self, metadata: EmbeddingMetadata) -> Path:
    hash_id = metadata.compute_hash()
    return self.cache_base / f"embeddings_{hash_id}"
    # Result: ./data/ttc/medical/embeddings_a1b2c3d4...
```

**BUG**: The cache directory is **shared across all patients** because entity_id is not in the path.

### Step 2.7: Embedding Keys Are Timestamps Only

```python
# embedder/embedder.py lines 510-556
def embed_text_dict(self, text_dict: Dict[str, str], ...):
    keys = list(text_dict.keys())  # ["20200115120000", "20200115130000", ...]
    texts = [text_dict[key] for key in keys]
    
    embeddings_array = self.embed_texts(texts, text_keys=keys, ...)
    # Cache lookup uses these timestamp-only keys!
```

**BUG**: Cache keys are timestamps only. If Patient A's timestamp `20200115120000` is embedded first, Patient B's request for the same timestamp will return Patient A's embedding!

---

## Part 3: Tensor Cache Entity Mixing (More Severe)

### Step 3.1: Tensor Cache Generation

```python
# data_provider/tensor_cache.py lines 735-764
def _register_timestamp_data(
    collector: SharedTableCollector,
    timestamp: int,
    ts_value: Optional[np.ndarray],
    embedding: Optional[np.ndarray],
    ...
) -> None:
    ts_key = int(timestamp)
    
    # CRITICAL BUG: Only checks timestamp, not (entity_id, timestamp)
    if ts_key in collector.timestamp_to_idx:
        return  # Skip - already have data for this timestamp
    
    # Store data for this timestamp (from whichever entity came first)
    idx = len(collector.timestamps)
    collector.timestamp_to_idx[ts_key] = idx
    collector.timestamps.append(ts_key)
    collector.timeseries.append(np.asarray(ts_value).flatten())  # Entity A's values!
    collector.embeddings.append(emb_to_store)  # Entity A's embeddings!
```

### Step 3.2: Sample Processing Flow

```python
# data_provider/tensor_cache.py lines 890-989
def _process_sample_for_collection(collector: SharedTableCollector, sample: tuple):
    x_time = _safe_array(sample[SAMPLE_IDX_X_TIME])
    if x_time is not None:
        x_time_flat = x_time.flatten()
        seq_x = _safe_array(sample[SAMPLE_IDX_SEQ_X])
        hetero_x = _safe_array(sample[SAMPLE_IDX_HETERO_X])
        
        for i, ts in enumerate(x_time_flat):
            ts_val = seq_x[i] if seq_x is not None else None
            emb = hetero_x[i].copy() if hetero_x is not None else None
            
            # THIS CALL DEDUPLICATES BY TIMESTAMP ONLY!
            _register_timestamp_data(collector, ts, ts_val, emb, ...)
```

### Step 3.3: What Happens at __getitem__ Time

```python
# data_provider/tensor_cache.py lines 2246-2378
def _getitem_indexed(self, index: int) -> tuple:
    # Get indices for this sample
    x_idx = self.arrays['x_indices'][index]    # [234, 235, 236, ...]
    
    # Look up time series from SHARED table
    seq_x = self.shared['timeseries'][x_idx]   # WRONG ENTITY'S DATA!
    
    # Look up embeddings from SHARED table
    hetero_x = self.shared['embeddings'][x_hetero_idx]  # WRONG ENTITY'S EMBEDDING!
```

**THE BUG**: Sample from Patient B uses indices into shared tables that were populated by Patient A (whoever processed first with that timestamp).

---

## Part 4: What Each Text Type Represents

### 4.1: For TTC Medical (Time-MMD Format)

| Component | Description | Source | Per-Entity? |
|-----------|-------------|--------|-------------|
| **Variables (channels)** | Vital signs: heart rate, blood pressure, etc. | CSV columns | ✅ Yes - each patient has unique values |
| **channel_info** | Static description of variables | `channel_info` config param | ❌ No - shared description like "Patient vital signs" |
| **general_info** | Static dataset description | `general_info` config param | ❌ No - shared description |
| **text (news/dynamic)** | Clinical notes at each timestamp | `text` column in CSV | ✅ Yes - each patient has unique notes |

### 4.2: For Fidel-TS Datasets (e.g., Bear_room)

| Component | Description | Source | Per-Entity? |
|-----------|-------------|--------|-------------|
| **Variables (channels)** | Sensor readings | CSV file | Depends on `hetero_type` |
| **channel_info** | Description per channel | `static_info.json[channel_info][entity_id]` | ✅ Yes |
| **general_info** | Dataset description | `static_info.json[general_info]` | ❌ No |
| **news (dynamic)** | Timestamped events/forecasts | JSON files keyed by timestamp | Depends on `hetero_type` |
| **downtime_prompt** | Sensor outage indicator | `static_info.json[downtime_prompt]` | ❌ No |

---

## Part 5: Correct Embedding Flow (What Should Happen)

### 5.1: TimeMMD_HeteroGetter.__call__() - At Training Time

```python
# data_provider/time_mmd_dataset.py lines 459-517
def __call__(self, timestamps):
    # Match timestamps to text data (backward matching)
    matched_times, matched_texts = self._match_timestamps(timestamps)
    
    if self.output_format == 'embedding':
        # Load embeddings if not already loaded
        if self.embeddings is None:
            self._load_or_create_embeddings()
        
        # Fetch embeddings for matched timestamps
        embedding_list, missing_count = self._fetch_and_validate_embeddings(matched_times)
        
        # Stack embeddings: (num_timesteps, 1, bert_dim)
        output_dynamic = np.stack(embedding_list, axis=0)
    
    # Prepare channel and general info embeddings
    general_info_emb, channel_info_emb = self._prepare_channel_and_general_info()
    
    return matched_times, general_info_emb, channel_info_emb, output_dynamic
```

### 5.2: Universal_Dataset.__getitem__() Returns

```python
# data_provider/data_loader.py lines 328-440
def __getitem__(self, index):
    # Time series slicing
    seq_x = self.data[s_begin:s_end]  # THIS entity's time series
    seq_y = self.data[r_begin:r_end]
    
    if self.preload_hetero:
        # Use preloaded hetero data (from THIS entity's hetero_data_getter)
        hetero_general = self.hetero_general
        hetero_channel = self.hetero_channel
        x_hetero = self.full_hetero[s_begin:s_end:self.hetero_stride]
        y_hetero = self.full_hetero[r_begin:r_end:self.hetero_stride]
    
    return sample_id, seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, ...
```

### 5.3: Model Forward Pass (lynx_film_raw example)

```python
# models/lynx_film_raw.py lines 303-378
def forward(self, x, news=None, channel_description=None, historical_events=None, **kwargs):
    # Select text input based on timestamp_semantics
    if self.timestamp_semantics == 't_known':
        # Time-MMD/TTC: Use historical_events (x_hetero) - avoids lookahead
        text_input = historical_events  # [B, seq_len, num_items, text_dim]
    
    # Project text embeddings if needed (768 -> text_dim)
    text_input, channel_description = self._project_text_embeddings(text_input, channel_description)
    
    # Expand channel_description to match text time dimension
    description = channel_description.unsqueeze(1).repeat(1, text_input.shape[1], 1, 1)
    
    # Text encoder combines news and channel description
    text_emb = self.text_encoder(text_input, description)  # [B, L, C, text_dim]
    
    # FiLM modulation of time series prediction
    text_emb = text_emb.permute(0, 2, 1, 3).contiguous()  # [B, C, L, D]
    pred_norm = self.model(x_norm, text_emb)
```

---

## Part 6: The Entity Mixing Bug in Detail

### Scenario: TTC Medical with 72 Patients

1. **Patient 10576** has timestamp `20200115120000`:
   - Time series: `[heart_rate=72, bp=120, ...]`
   - Text: `"Patient stable, no acute distress"`

2. **Patient 10910** has same timestamp `20200115120000`:
   - Time series: `[heart_rate=95, bp=180, ...]`  (different patient!)
   - Text: `"Patient in acute respiratory distress"`  (different text!)

### With Tensor Cache (Bug Active):

```
Tensor Cache Generation:
1. Process Patient 10576 first
2. Register timestamp 20200115120000:
   - timeseries[idx] = [72, 120, ...]  (Patient 10576's data)
   - embeddings[idx] = embed("Patient stable...")  (Patient 10576's text)
   - timestamp_to_idx[20200115120000] = idx

3. Process Patient 10910
4. Try to register timestamp 20200115120000:
   - Check: 20200115120000 in timestamp_to_idx? YES
   - SKIP! (Don't store Patient 10910's data)

At Training Time:
- Sample from Patient 10910 at timestamp 20200115120000
- x_indices points to idx in shared table
- Gets Patient 10576's time series values!
- Gets Patient 10576's text embedding!

RESULT: Model learns wrong patient-text associations!
```

### With Old .pkl Format (Correct):

```
1. Load Patient 10576 dataset
   - Load ./data/ttc/medical/patient_10576.pkl
   - Contains only Patient 10576's embeddings

2. Load Patient 10910 dataset  
   - Load ./data/ttc/medical/patient_10910.pkl
   - Contains only Patient 10910's embeddings

3. Each dataset has its own embedding dict
   - No collision possible!
```

---

## Part 7: Summary of Affected Components

| Component | Bug? | Description |
|-----------|------|-------------|
| **Old .pkl embedding cache** | ✅ Safe | Per-entity file: `{base_name}.pkl` |
| **New embedding cache** | ⚠️ BUG | Shared directory, timestamp-only keys |
| **Tensor cache (shared tables)** | 🔴 BUG | Global timestamp dedup across all entities |
| **Non-cache dataloader path** | ✅ Safe | Each entity loads its own CSV |

---

## Part 8: Impact Analysis

### Single-Entity Datasets (Not Affected):
- TTC Climate (`id: "all"`)
- Time-MMD subdatasets with single entity
- Most Fidel-TS datasets with `hetero_type: "all_for_one"`

### Multi-Entity Datasets (AFFECTED):
- **TTC Medical** (72 patients)
- **NYC Traffic Speed** (if multiple sensors with overlapping timestamps)
- Any dataset with `id: "all"` that expands to multiple entity files

### Why Your Jan12 Run Was Okay:
Likely used `use_old_pkl: true` or didn't use tensor cache, so embeddings were loaded per-entity.

### Why Current Runs Are Worse:
Using new embedding cache and/or tensor cache with `use_tensor_cache: true`, causing entity mixing.

---

## Part 9: Recommended Fixes

### Immediate Workaround:
```yaml
# In sweep/experiment config:
data_config:
  use_old_pkl: true  # Force per-entity .pkl files

training:
  use_tensor_cache: false  # Disable tensor cache for multi-entity
```

### Code Fix Option 1: Prefix Keys with Entity ID
```python
# In time_mmd_dataset.py _load_or_create_embeddings():
text_dict = {f"{self.entity_id}|{str(ts)}": text for ts, text in self.text_data.items()}

# In tensor_cache.py _register_timestamp_data():
ts_key = (entity_id, int(timestamp))  # Tuple key
```

### Code Fix Option 2: Per-Entity Cache Directories
```python
# In cache_manager.py:
self.cache_base = self.dataset_root / self.dataset_path / f"entity_{entity_id}"
```

### Code Fix Option 3: Per-Entity Shared Tables (Tensor Cache)
```python
# Store separate shared tables per entity, or compound keys in mappings
collector.timestamp_to_idx: Dict[Tuple[str, int], int]  # (entity_id, timestamp) -> idx
```

---

## Part 10: Detailed Code Locations

### Text Embedding Cache Bug:
- `data_provider/time_mmd_dataset.py` lines 203-293: `_load_or_create_embeddings()`
- `embedder/cache_manager.py` lines 25-48: `EmbeddingCacheManager.__init__()`
- `embedder/embedder.py` lines 510-556: `embed_text_dict()`

### Tensor Cache Bug:
- `data_provider/tensor_cache.py` lines 735-846: `_register_timestamp_data()`
- `data_provider/tensor_cache.py` lines 890-989: `_process_sample_for_collection()`
- `data_provider/tensor_cache.py` lines 2246-2378: `_getitem_indexed()`

### Correct Per-Entity Loading:
- `data_provider/time_mmd_dataset.py` lines 159-176: `_get_embedding_path()` (old .pkl path)
- `data_provider/data_factory.py` lines 757-870: `_create_time_mmd_dataset()`
- `data_provider/data_factory.py` lines 980-1113: `get_datasets()`

---

## Part 11: Rigorous Fix Plan

### 11.1 Design Principles

1. **Entity Isolation**: Each entity's data must be stored and retrieved independently
2. **Backward Compatibility**: Single-entity datasets must continue to work without changes
3. **Shape Preservation**: Output tensor shapes must remain identical for all models
4. **Deduplication Benefits**: Within an entity, timestamp deduplication should still work
5. **Minimal Invasiveness**: Changes should be localized to cache key/path logic

### 11.2 Two-Part Fix Strategy

The fix requires changes to two independent systems:

| System | Problem | Solution |
|--------|---------|----------|
| **Text Embedding Cache** | Shared cache directory, timestamp-only keys | Entity-prefixed keys in embedding dict |
| **Tensor Cache** | Global `timestamp_to_idx` across all entities | Compound keys: `(entity_id, timestamp)` |

---

### 11.3 Fix Part A: Text Embedding Cache (TimeMMD_HeteroGetter)

#### 11.3.1 Current Buggy Flow

```
TimeMMD_Dataset.__init__(entity_id="10576", data_path="patient_10576.csv")
  └── _setup_text_getter()
        └── TimeMMD_HeteroGetter(text_data={20200115120000: "Patient stable..."})
              └── _load_or_create_embeddings()
                    └── text_dict = {str(ts): text for ts, text in self.text_data.items()}
                          # Keys: ["20200115120000", ...]  ← NO ENTITY PREFIX!
                    └── embedder.embed_text_dict(text_dict)
                          # Cache: data/ttc/medical/embeddings_HASH/embeddings.pkl
                          # Dict keys inside: {"20200115120000": embedding, ...}
```

#### 11.3.2 Fixed Flow

```
TimeMMD_Dataset.__init__(entity_id="10576", data_path="patient_10576.csv")
  └── _setup_text_getter()
        └── TimeMMD_HeteroGetter(text_data={...}, entity_id="10576")  ← PASS ENTITY_ID
              └── _load_or_create_embeddings()
                    └── text_dict = {f"{self.entity_id}|{str(ts)}": text ...}
                          # Keys: ["10576|20200115120000", ...]  ← ENTITY PREFIX!
                    └── embedder.embed_text_dict(text_dict)
                          # Cache: data/ttc/medical/embeddings_HASH/embeddings.pkl
                          # Dict keys inside: {"10576|20200115120000": embedding, ...}
              
              └── _fetch_and_validate_embeddings(matched_times)
                    └── for ts in matched_times:
                          key = f"{self.entity_id}|{ts}"  ← LOOKUP WITH PREFIX
                          emb = self.embeddings[key]
```

#### 11.3.3 Code Changes Required

**File: `data_provider/time_mmd_dataset.py`**

```python
# Change 1: Add entity_id to TimeMMD_HeteroGetter.__init__
class TimeMMD_HeteroGetter:
    def __init__(self, text_data, timestamps, general_info='', channel_info='', 
                 output_format='json', embed_model_name='bert-base-uncased', 
                 embed_dim=768, force_reembed=False, hf_cache_dir='./HF_cache/',
                 root_path=None, data_path=None, device='cpu', num_channels=None, 
                 channel_names=None, aggregation_method='cls', use_old_pkl=False,
                 entity_id=None):  # ← ADD THIS PARAMETER
        # ... existing init ...
        self.entity_id = entity_id  # ← STORE IT

# Change 2: Update _load_or_create_embeddings to use entity-prefixed keys
def _load_or_create_embeddings(self):
    # Create entity-prefixed keys for multi-entity safety
    if self.entity_id is not None:
        text_dict = {f"{self.entity_id}|{str(ts)}": text 
                     for ts, text in self.text_data.items()}
    else:
        # Backward compatibility: no entity_id means single-entity dataset
        text_dict = {str(ts): text for ts, text in self.text_data.items()}
    
    # ... rest of method unchanged ...

# Change 3: Update _fetch_and_validate_embeddings to use entity-prefixed keys
def _fetch_and_validate_embeddings(self, matched_times: List[str]) -> Tuple[...]:
    embedding_list = []
    missing_count = 0
    
    for ts in matched_times:
        # Build key with entity prefix if available
        if self.entity_id is not None:
            key = f"{self.entity_id}|{ts}"
        else:
            key = ts
        
        if key in self.embeddings:
            emb = self.embeddings[key]
            emb = self._normalize_embedding_shape(emb)
        else:
            missing_count += 1
            emb = self._create_zero_embedding()
        
        embedding_list.append(emb)
    
    return embedding_list, missing_count

# Change 4: Pass entity_id when creating TimeMMD_HeteroGetter in _setup_text_getter
def _setup_text_getter(self):
    # ... existing code ...
    
    # Extract entity_id from data_path (e.g., "patient_10576.csv" -> "10576")
    # Or use self.entity_id if passed to TimeMMD_Dataset
    entity_id = getattr(self, 'entity_id', None)
    if entity_id is None:
        # Try to extract from data_path
        import re
        match = re.search(r'patient_(\d+)\.csv', self.data_path)
        if match:
            entity_id = match.group(1)
    
    text_getter = TimeMMD_HeteroGetter(
        text_data=self._text_data,
        timestamps=self.timestamp,
        # ... other params ...
        entity_id=entity_id,  # ← PASS ENTITY_ID
    )
```

#### 11.3.4 Verification: Output Shapes Unchanged

| Output | Before Fix | After Fix | Model Expectation |
|--------|------------|-----------|-------------------|
| `output_dynamic` | `(L, 1, 768)` | `(L, 1, 768)` | ✅ Same |
| `general_info_emb` | `(1, 768)` | `(1, 768)` | ✅ Same |
| `channel_info_emb` | `(C, 768)` | `(C, 768)` | ✅ Same |

The entity prefix is **only in the cache keys**, not in the returned data structure.

---

### 11.4 Fix Part B: Tensor Cache (SharedTableCollector)

#### 11.4.1 Current Buggy Flow

```
TensorCacheGenerator.generate()
  └── _build_shared_tables()
        └── SharedTableCollector()
              timestamp_to_idx: Dict[int, int] = {}  ← GLOBAL, NO ENTITY
        
        └── For each entity, for each sample:
              _process_sample_for_collection(collector, sample)
                └── for ts in x_time:
                      _register_timestamp_data(collector, ts, ts_val, emb, ...)
                        └── ts_key = int(timestamp)  ← NO ENTITY!
                        └── if ts_key in collector.timestamp_to_idx:
                              return  ← SKIP! Entity B's data lost!
```

#### 11.4.2 Fixed Flow

```
TensorCacheGenerator.generate()
  └── _build_shared_tables()
        └── SharedTableCollector()
              timestamp_to_idx: Dict[Tuple[str, int], int] = {}  ← COMPOUND KEY
        
        └── For each entity, for each sample:
              current_entity_id = extract_entity_id(sample)
              _process_sample_for_collection(collector, sample, entity_id)
                └── for ts in x_time:
                      _register_timestamp_data(collector, ts, ts_val, emb, entity_id)
                        └── ts_key = (entity_id, int(timestamp))  ← COMPOUND KEY
                        └── if ts_key in collector.timestamp_to_idx:
                              return  ← Only skip if SAME entity + timestamp
```

#### 11.4.3 Code Changes Required

**File: `data_provider/tensor_cache.py`**

```python
# Change 1: Update SharedTableCollector to use compound keys
@dataclass
class SharedTableCollector:
    # Change from Dict[int, int] to Dict[Tuple[str, int], int]
    timestamp_to_idx: Dict[Tuple[str, int], int] = field(default_factory=dict)
    
    # ... rest unchanged ...

# Change 2: Update _register_timestamp_data to accept and use entity_id
def _register_timestamp_data(
    collector: SharedTableCollector,
    timestamp: int,
    ts_value: Optional[np.ndarray],
    embedding: Optional[np.ndarray],
    hetero_time_feat: Optional[np.ndarray],
    entity_id: str = ""  # ← ADD THIS PARAMETER
) -> None:
    """
    Register data for a (entity_id, timestamp) pair if not already seen.
    
    Uses compound key (entity_id, timestamp) to ensure entity isolation.
    Same timestamp from different entities will be stored separately.
    """
    # FIXED: Use compound key (entity_id, timestamp)
    ts_key = (entity_id, int(timestamp))
    
    # Skip if already registered for THIS entity at THIS timestamp
    if ts_key in collector.timestamp_to_idx:
        return
    
    # Assign next available index
    idx = len(collector.timestamps)
    collector.timestamp_to_idx[ts_key] = idx
    
    # Store the (flattened) timestamp for reconstruction
    # Note: We store the raw timestamp, not the compound key
    collector.timestamps.append(int(timestamp))
    
    # ... rest of storage logic unchanged ...

# Change 3: Update _process_sample_for_collection to extract and pass entity_id
def _process_sample_for_collection(
    collector: SharedTableCollector,
    sample: tuple,
    entity_id: str = ""  # ← ADD THIS PARAMETER
) -> None:
    """
    Process a single sample, registering all unique (entity, timestamp) pairs.
    """
    # ... existing shape inference ...
    
    # Process input window timestamps
    x_time = _safe_array(sample[SAMPLE_IDX_X_TIME])
    if x_time is not None:
        x_time_flat = x_time.flatten()
        seq_x = _safe_array(sample[SAMPLE_IDX_SEQ_X])
        hetero_x = _safe_array(sample[SAMPLE_IDX_HETERO_X])
        hetero_x_time = _safe_array(sample[SAMPLE_IDX_HETERO_X_TIME])
        
        for i, ts in enumerate(x_time_flat):
            ts_val = seq_x[i] if seq_x is not None else None
            emb = hetero_x[i].copy() if hetero_x is not None and ... else None
            htf = hetero_x_time[i] if hetero_x_time is not None and ... else None
            
            # FIXED: Pass entity_id to _register_timestamp_data
            _register_timestamp_data(collector, ts, ts_val, emb, htf, entity_id)
    
    # Process output window timestamps (same pattern)
    y_time = _safe_array(sample[SAMPLE_IDX_Y_TIME])
    if y_time is not None:
        # ... similar changes ...
        _register_timestamp_data(collector, ts, ts_val, emb, htf, entity_id)

# Change 4: Update callers to pass entity_id through the chain
# In TensorCacheGenerator._build_shared_tables():
for flag in flags:
    datasets = self.data_provider.get_datasets(flag)
    for entity_id, dataset in datasets.items():
        # ... entity registration ...
        for sample_idx in range(len(dataset)):
            sample = dataset[sample_idx]
            # FIXED: Pass entity_id
            _process_sample_for_collection(collector, sample, entity_id=str(entity_id))

# Change 5: Update _finalize_shared_tables to handle compound keys
def _finalize_shared_tables(collector: SharedTableCollector) -> Tuple[Dict[str, np.ndarray], dict]:
    # ... existing array conversion ...
    
    # Build index mappings (JSON-serializable)
    # Convert compound keys to string format: "entity_id|timestamp"
    index_mappings = {
        'timestamp_to_idx': {
            f"{entity_id}|{ts}": v 
            for (entity_id, ts), v in collector.timestamp_to_idx.items()
        },
        'entity_to_idx': collector.entity_to_idx
    }
    
    return shared_tables, index_mappings

# Change 6: Update _write_sample_indices to use compound key lookup
def _write_sample_indices(
    arrays: Dict[str, np.memmap],
    write_idx: int,
    sample: tuple,
    entity_idx: int,
    timestamp_to_idx: Dict[str, int],  # Now string keys: "entity_id|timestamp"
    entity_id: str  # ← ADD THIS PARAMETER
) -> None:
    # ... existing code ...
    
    # Convert input timestamps to indices using compound key
    x_time = np.asarray(sample[SAMPLE_IDX_X_TIME]).flatten()
    x_indices = np.array(
        [timestamp_to_idx.get(f"{entity_id}|{int(ts)}", 0) for ts in x_time],
        dtype=np.int32
    )
    arrays['x_indices'][write_idx] = x_indices
    
    # Same for y_indices
    y_time = np.asarray(sample[SAMPLE_IDX_Y_TIME]).flatten()
    y_indices = np.array(
        [timestamp_to_idx.get(f"{entity_id}|{int(ts)}", 0) for ts in y_time],
        dtype=np.int32
    )
    arrays['y_indices'][write_idx] = y_indices
```

#### 11.4.4 Verification: Output Shapes Unchanged

The fix changes **how indices are computed and stored**, but the final output shapes in `__getitem__` remain identical:

| Output | Before Fix | After Fix | Model Expectation |
|--------|------------|-----------|-------------------|
| `seq_x` | `(input_len, n_features)` | `(input_len, n_features)` | ✅ Same |
| `seq_y` | `(output_len, n_features)` | `(output_len, n_features)` | ✅ Same |
| `hetero_x` | `(L, N, embed_dim)` | `(L, N, embed_dim)` | ✅ Same |
| `hetero_y` | `(L, N, embed_dim)` | `(L, N, embed_dim)` | ✅ Same |
| `hetero_general` | `(embed_dim,)` | `(embed_dim,)` | ✅ Same |
| `hetero_channel` | `(embed_dim,)` | `(embed_dim,)` | ✅ Same |

**Why shapes don't change:**
1. Indices now correctly point to THIS entity's data in shared tables
2. The lookup `self.shared['timeseries'][x_idx]` returns correct values
3. The embedding lookup `self.shared['embeddings'][x_hetero_idx]` returns correct embeddings

---

### 11.5 Robustness Verification

#### 11.5.1 Single-Entity Dataset (Backward Compatibility)

For TTC Climate with `id: "all"`:

```
Before Fix:
  timestamp_to_idx[20200115120000] = 0
  
After Fix:
  timestamp_to_idx[("all", 20200115120000)] = 0
  # Or: timestamp_to_idx["all|20200115120000"] = 0

Result: Identical behavior, just different key format
```

#### 11.5.2 Multi-Entity Dataset (Fix Verification)

For TTC Medical with 72 patients:

```
Before Fix (BUG):
  timestamp_to_idx[20200115120000] = 0  # Patient 10576's data
  # Patient 10910's request for same timestamp → gets Patient 10576's data!

After Fix (CORRECT):
  timestamp_to_idx[("10576", 20200115120000)] = 0   # Patient 10576's data
  timestamp_to_idx[("10910", 20200115120000)] = 47  # Patient 10910's data
  # Each patient's data stored separately!
```

#### 11.5.3 Memory Impact

The fix increases shared table size proportionally to entity count:

| Metric | Single Entity | 72 Entities (TTC Medical) |
|--------|---------------|---------------------------|
| Unique timestamps (before) | ~10,000 | ~10,000 (BUG: collapsed!) |
| Unique (entity, timestamp) (after) | ~10,000 | ~720,000 (correct) |
| Memory increase | 0% | ~72x for timestamp-keyed data |

**However**, this is the **correct** memory usage. The "savings" before were due to data loss!

For embeddings (the largest component):
- Before: ~10K × 768 × 4 bytes = 30 MB (BUT WRONG DATA!)
- After: ~720K × 768 × 4 bytes = 2.2 GB (CORRECT DATA!)

This is still ~100x smaller than V1 format which stored per-sample:
- V1: ~5M samples × 768 × 4 bytes = 15 GB

---

### 11.6 Model-Specific Verification

#### 11.6.1 TGTSF Model

```python
# models/TGTSF.py forward()
def forward(self, x, news=None, channel_description=None, historical_events=None, **kwargs):
    # For t_known (Time-MMD/TTC): uses historical_events (x_hetero)
    text_input = historical_events  # Shape: [B, seq_len, num_items, text_dim]
    
    # Expected shapes from dataloader:
    # - historical_events: [B, L, N, D] where N=1 or 2, D=768
    # - channel_description: [B, C, D] where C=num_channels, D=768
```

**Fix verification**: The compound key ensures `historical_events` contains THIS patient's text embeddings, not another patient's.

#### 11.6.2 lynx_film_raw Model

```python
# models/lynx_film_raw.py forward()
def forward(self, x, news=None, channel_description=None, historical_events=None, **kwargs):
    if self.timestamp_semantics == 't_known':
        text_input = historical_events  # [B, seq_len, num_items, text_dim]
    
    # Project if needed: 768 -> text_dim
    text_input, channel_description = self._project_text_embeddings(text_input, channel_description)
    
    # Text encoder: [B, L, N, D] + [B, L, C, D] -> [B, L, C, D]
    text_emb = self.text_encoder(text_input, description)
```

**Fix verification**: Shape `[B, L, N, D]` is unchanged. Only the VALUES are now correct (THIS entity's embeddings).

#### 11.6.3 lynx_film and lynx_film_enhanced

Same pattern as lynx_film_raw - shapes unchanged, values now correct.

---

### 11.7 Implementation Order

1. **Phase 1: Text Embedding Cache Fix** (Lower risk)
   - Modify `TimeMMD_HeteroGetter` to use entity-prefixed keys
   - Test with `use_tensor_cache: false`
   - Verify TTC Medical performance improves

2. **Phase 2: Tensor Cache Fix** (Higher complexity)
   - Update `SharedTableCollector` to use compound keys
   - Update all functions in the chain to pass `entity_id`
   - Regenerate tensor caches for multi-entity datasets
   - Verify shapes match expectations

3. **Phase 3: Integration Testing**
   - Run full training on TTC Medical with both fixes
   - Compare metrics to Jan12 baseline (which used old .pkl)
   - Verify single-entity datasets still work (TTC Climate)

---

### 11.8 Test Cases

```python
def test_multi_entity_embedding_isolation():
    """Verify entity A's embedding is not returned for entity B's timestamp."""
    # Setup: Two entities with same timestamp, different text
    entity_a = TimeMMD_Dataset(data_path="patient_10576.csv", entity_id="10576")
    entity_b = TimeMMD_Dataset(data_path="patient_10910.csv", entity_id="10910")
    
    # Both have timestamp 20200115120000
    emb_a = entity_a.hetero_data_getter([20200115120000])[3]  # output_dynamic
    emb_b = entity_b.hetero_data_getter([20200115120000])[3]
    
    # Embeddings must be different (different clinical notes!)
    assert not np.allclose(emb_a, emb_b), "Entity embeddings should differ!"

def test_tensor_cache_entity_isolation():
    """Verify tensor cache stores each entity's data separately."""
    # Generate tensor cache for TTC Medical
    cache = TensorCacheDataset(cache_dir, flag='train')
    
    # Find two samples from different entities at same timestamp
    sample_a = cache[find_sample_idx(entity="10576", timestamp=20200115120000)]
    sample_b = cache[find_sample_idx(entity="10910", timestamp=20200115120000)]
    
    # Time series values must differ (different patients!)
    seq_x_a, seq_x_b = sample_a[1], sample_b[1]
    assert not np.allclose(seq_x_a, seq_x_b), "Time series should differ!"
    
    # Embeddings must differ (different clinical notes!)
    hetero_x_a, hetero_x_b = sample_a[5], sample_b[5]
    assert not np.allclose(hetero_x_a, hetero_x_b), "Embeddings should differ!"

def test_output_shapes_unchanged():
    """Verify model-expected shapes are preserved after fix."""
    cache = TensorCacheDataset(cache_dir, flag='train')
    sample = cache[0]
    
    seq_x = sample[1]       # (input_len, n_features)
    hetero_x = sample[5]    # (L, N, embed_dim)
    hetero_channel = sample[10]  # (embed_dim,) or (n_channels, embed_dim)
    
    assert seq_x.shape == (input_len, n_features)
    assert hetero_x.shape[2] == 768  # embed_dim
    assert len(hetero_channel.shape) in [1, 2]  # 1D or 2D
```

---

### 11.9 Rollback Plan

If issues arise:

1. **Immediate**: Set `use_tensor_cache: false` and `use_old_pkl: true` in configs
2. **Cache invalidation**: Delete `embeddings_*` directories and `tensor_cache/` directories
3. **Code rollback**: Revert changes to `time_mmd_dataset.py` and `tensor_cache.py`

The compound key approach is designed to be **additive** - it doesn't break existing caches for single-entity datasets, they just won't benefit from the fix.
