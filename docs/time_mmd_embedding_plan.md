## Time-MMD On-the-Fly Text Embedding Plan

This document outlines a plan to add **optional on-the-fly text embedding** for Time-MMD datasets, integrated into `TimeMMD_HeteroGetter`, with:
- Predictable `.pkl` storage under `data/time_mmd/{subdataset}/`
- Default use of pre-computed embeddings when available
- Automatic embedding + save if missing
- Option to force re-embedding
- Output formats compatible with **TGTSF** and **LYNX** (lynx_film, lynx_tgtsf)
- Configurable choice between **raw text** (for ChatTime) and **embeddings** (for TGTSF/LYNX)

---

### 1. Goals

1. **Add embedding mode** to `TimeMMD_HeteroGetter`:
   - Convert raw text (from CSV) to embeddings using a HuggingFace model (e.g., BERT, MiniLM).
   - Save embeddings to disk in a predictable `.pkl` file.
   - Reuse embeddings on subsequent runs.
2. **Match existing embedding format** used by `Heterogeneous_Dataset` when `output_format='embedding'`:
   - Ensure compatibility with models expecting `news` embeddings (`TGTSF`, `lynx_film`, `lynx`).
3. **Config-driven behavior**:
   - `output_format: text` → raw text JSON/dict (for ChatTime, etc.).
   - `output_format: embedding` → pre-computed or on-the-fly embeddings.
   - `force_reembed: true/false` to control re-embedding.

---

### 2. File Layout and Storage Convention

#### 2.1. Embedding File Path

For each Time-MMD subdataset:

```text
data/time_mmd/{subdataset}/{filename}.csv     # Original CSV
data/time_mmd/{subdataset}/{filename}.pkl     # Embeddings (to be created)
```

Examples:
- `data/time_mmd/Traffic/US_VMT_Month.csv` → `data/time_mmd/Traffic/US_VMT_Month.pkl`
- `data/time_mmd/Economy/US_TradeBalance_Month.csv` → `.../US_TradeBalance_Month.pkl`

#### 2.2. Embedding File Format

To match `Heterogeneous_Dataset.load_embedding()` conventions, use a dict-of-arrays format:

```python
{
    "YYYYMMDDHHMMSS_0": np.ndarray(shape=(n, d), dtype=np.float32),
    "YYYYMMDDHHMMSS_1": np.ndarray(shape=(n, d), dtype=np.float32),
    ...
}
```

- **Key**: unique timestamp identifier (string) — can be the raw int timestamp or timestamp with index suffix.
- **Value**: `np.ndarray` of shape `(news_num, text_dim)`:
  - `news_num`: number of text snippets per time step (Time-MMD: typically 1).
  - `text_dim`: embedding dimension (e.g., 768).

This aligns with `Heterogeneous_Dataset.get_hetero_data()` when `output_format='embedding'`, which expects per-time embeddings retrievable by timestamp.

---

### 3. Configuration Additions

#### 3.1. Data Config (Time-MMD)

Extend `data_configs/time_mmd/{subdataset}/config.yaml` with:

```yaml
# Existing
dataset_type: time_mmd
text_column: auto
use_closedllm: false
text_len: 4

# New
timemmd_text_output: text        # 'text' | 'embedding'
timemmd_embed_model: bert-base-uncased  # HF model name/path (default: bert-base-uncased)
timemmd_embed_dim: 768           # Embedding dimension (default: 768 for BERT)
timemmd_force_reembed: false     # If true, ignore existing .pkl and recompute
hf_cache_dir: ./HF_cache/  # Local cache for HF models (default: ./HF_cache/)
```

**Interpretation:**
- `timemmd_text_output: text` → `TimeMMD_HeteroGetter` returns JSON strings (current default).
- `timemmd_text_output: embedding` → `TimeMMD_HeteroGetter` returns embeddings (using `.pkl`).
- `timemmd_embed_model`: HF model for encoding text (`AutoTokenizer` + `AutoModel`). **Defaults to `bert-base-uncased`** (same as used elsewhere in fidel-ts).
- `hf_cache_dir`: Local directory for caching HF models. **Defaults to `./HF_cache/`**. Models are cached here after first download to avoid re-downloading. This key is **data-agnostic** and can be shared across datasets.
- `timemmd_force_reembed: true`:
  - Always recompute embeddings and overwrite `.pkl`.

#### 3.2. Model Config (Optional)

Optionally, add model-level hints (for auto-switching defaults):

```yaml
# In model_configs/general/TGTSF.yaml
expects_text_embeddings: true

# In model_configs/LLM/ChatTime.yaml
expects_raw_text: true
```

These are *hints* only; final behavior is determined by `timemmd_text_output`.

---

### 4. TimeMMD_HeteroGetter Changes (Plan Only)

#### 4.1. Initialization

Add new init parameters (wired from `TimeMMD_Dataset`):

```python
class TimeMMD_HeteroGetter:
    def __init__(
        self,
        text_data,
        timestamps,
        general_info='',
        channel_info='',
        output_format='json',           # 'json' | 'dict' | 'csv' | 'embedding'
        embed_model_name='bert-base-uncased',  # Default: bert-base-uncased (same as fidel-ts)
        embed_dim=768,                  # Default: 768 for BERT
        force_reembed=False,
        hf_cache_dir='./HF_cache/',     # Local cache for HF models (data-agnostic)
        root_path=None,                 # dataset root_path
        data_path=None,                 # filename.csv
        device='cpu'                    # Device for embedding model
    ):
        ...
```

**Responsibilities:**
- Decide whether to operate in **text mode** or **embedding mode** based on `output_format`.
- In embedding mode:
  - Compute `.pkl` file path from `root_path` and `data_path`.
  - Load existing embeddings if available and `force_reembed=False`.
  - Otherwise embed on-the-fly and save to `.pkl`.

#### 4.2. Embedding Loader/Creator

Add internal helper methods:

```python
def _get_embedding_path(self):
    # root_path: ./data/time_mmd/Traffic
    # data_path: US_VMT_Month.csv
    base, _ = os.path.splitext(self.data_path)
    return os.path.join(self.root_path, f"{base}.pkl")

def _load_or_create_embeddings(self):
    pkl_path = self._get_embedding_path()
    if os.path.exists(pkl_path) and not self.force_reembed:
        # Load precomputed embeddings
        self.embeddings = joblib.load(pkl_path)   # or np.load/pickle
    else:
        # Compute embeddings on-the-fly
        self.embeddings = self._embed_text_corpus()
        joblib.dump(self.embeddings, pkl_path)
```

`_embed_text_corpus()`:
- Iterates over `self.text_data` (Series indexed by timestamp).
- Uses `AutoTokenizer` + `AutoModel` to get embeddings (CLS or pooled output).
- **Model loading strategy**:
  - First checks `hf_cache_dir` for local model files.
  - If not found locally, downloads from HF and caches to `hf_cache_dir`.
  - Subsequent runs reuse the cached model (no re-download).
- Stores in the dict-of-arrays format described above.

**Model Caching Details:**
- Uses HuggingFace's `cache_dir` parameter in `from_pretrained()`.
- Models are stored in `{hf_cache_dir}/{model_name}/` (HF's standard structure).
- Only downloads once per model; subsequent loads use cached version.

#### 4.3. __call__() Logic

Current logic (simplified):

```python
def __call__(self, timestamps):
    matched_times, matched_texts = self._match_timestamps(timestamps)
    if self.output_format == 'json':
        ...
    elif self.output_format == 'embedding':
        # currently: zeros -> change to fetch from self.embeddings
```

Planned change for `output_format == 'embedding'`:

```python
if self.output_format == 'embedding':
    # Ensure embeddings are loaded/created
    if not hasattr(self, 'embeddings'):
        self._load_or_create_embeddings()

    embedding_list = []
    for ts in matched_times:
        # ts is string 'YYYYMMDDHHMMSS' -> use as key or map to nearest
        if ts in self.embeddings:
            emb = self.embeddings[ts]              # shape: (news_num, embed_dim)
        else:
            emb = np.zeros((1, self.embed_dim), dtype=np.float32)
        embedding_list.append(emb)

    # Convert to np.array: (len(matched_times), news_num, embed_dim)
    output_dynamic = np.stack(embedding_list, axis=0)
    return matched_times, self.general_info, self.channel_info, output_dynamic
```

This matches the spirit of `Heterogeneous_Dataset.get_hetero_data()` when `output_format='embedding'`:
- Downstream, `Universal_Dataset.__getitem__()` will:
  - For `x_hetero`: receive array of shape `(seq_len, news_num, embed_dim)`.
  - Dataloader will collate to shape `[B, l, n, d]` for `news`.

---

### 5. Compatibility with TGTSF and LYNX

#### 5.1. TGTSF

**Expected `news` shape** (line 189 in `models/TGTSF.py`):

```python
def forward(self, x, news, channel_description, **kwargs):
    # x: [B, input_len, C]
    # news: [B, l, news_num, text_dim]
```

Our planned `output_dynamic`:
- Per sample: `(seq_len, news_num, embed_dim)`
- Per batch: Dataloader collates to `[B, seq_len, news_num, embed_dim]`

This matches TGTSF's expectation (with `l = seq_len`).

#### 5.2. LYNX / LYNX_FiLM

**Expected `news` shape**:
- Documented as: `news: [B, l, news_num, text_dim]`
- Same as TGTSF

Therefore, the same embedding format works for both TGTSF and LYNX-style models.

---

### 6. Text vs Embedding Mode Selection

#### 6.1. Data Config Driven

**Goal**: Let the data config decide how `TimeMMD_HeteroGetter` behaves:

```yaml
timemmd_text_output: text      # for ChatTime, DLinear (ignores text)
# or
timemmd_text_output: embedding # for TGTSF, LYNX
```

Mapping inside `TimeMMD_Dataset.__init__()`:

```python
mode = self.args.data_config.get('timemmd_text_output', 'text')
if mode == 'text':
    output_format = 'json'
elif mode == 'embedding':
    output_format = 'embedding'
else:
    raise ValueError(...)
```

Additional options from config:

```yaml
timemmd_embed_model: bert-base-uncased  # Default: bert-base-uncased
timemmd_embed_dim: 768                  # Default: 768 for BERT
timemmd_force_reembed: false
hf_cache_dir: ./HF_cache/       # Default: ./HF_cache/ (data-agnostic)
```

Pass these into `TimeMMD_HeteroGetter` constructor.

#### 6.2. Model-Specific Suggestions

We can provide recommended settings in docs:

- **DLinear / TSF models**:
  - `timemmd_text_output: text` or omit (DLinear ignores text anyway)
- **ChatTime**:
  - `timemmd_text_output: text` (LLM consumes raw text)
- **TGTSF / LYNX**:
  - `timemmd_text_output: embedding`
  - `timemmd_embed_model: bert-base-uncased` (default, same as used elsewhere in fidel-ts)
  - `timemmd_hf_cache_dir: ./HF_cache/` (default, avoids re-downloading models)

---

### 7. Force Re-Embedding Behavior

Use `timemmd_force_reembed` to control recomputation:

```python
def _load_or_create_embeddings(self):
    pkl_path = self._get_embedding_path()
    if os.path.exists(pkl_path) and not self.force_reembed:
        self.embeddings = joblib.load(pkl_path)
    else:
        self.embeddings = self._embed_text_corpus()
        joblib.dump(self.embeddings, pkl_path)
```

**Use cases:**
- Change embedding model → set `timemmd_force_reembed: true`
- Text data changes → set `timemmd_force_reembed: true`

---

### 8. Testing Plan

1. **Unit tests** for `TimeMMD_HeteroGetter`:
   - Given a small `text_data` Series, check:
     - `.pkl` creation
     - Reload behavior
     - Shape and dtype of embeddings
2. **Integration tests** with TGTSF:
   - Config: `timemmd_text_output: embedding`
   - Ensure:
     - No shape mismatch errors
     - `news` argument in TGTSF forward has expected shape `[B, l, n, d]`
3. **Integration tests** with ChatTime:
   - Config: `timemmd_text_output: text`
   - Ensure:
     - Text strings are passed through unchanged
     - ChatTime builds prompts from raw text
4. **Performance tests**:
   - Measure embedding time for on-the-fly embedding
   - Confirm `.pkl` reuse significantly reduces subsequent startup time

---

### 9. Model Caching Strategy

#### 9.1. Local Cache Directory

- **Default location**: `./HF_cache/`
- **Purpose**: Store downloaded HuggingFace models locally to avoid re-downloading
- **Configurable**: Via `hf_cache_dir` in config (data-agnostic; shared across datasets)

#### 9.2. Model Loading Logic

```python
def _load_embedding_model(self):
    """Load tokenizer and model, using local cache if available."""
    import os
    from transformers import AutoTokenizer, AutoModel
    
    # Ensure cache directory exists
    os.makedirs(self.hf_cache_dir, exist_ok=True)
    
    # Use cache_dir parameter to store models locally
    self.tokenizer = AutoTokenizer.from_pretrained(
        self.embed_model_name,
        cache_dir=self.hf_cache_dir
    )
    self.model = AutoModel.from_pretrained(
        self.embed_model_name,
        cache_dir=self.hf_cache_dir
    ).to(self.device)
    
    # First call downloads and caches; subsequent calls use cache
```

**Behavior:**
- First run: Downloads model from HF → saves to `./HF_cache/{model_name}/`
- Subsequent runs: Loads from `./HF_cache/{model_name}/` (no download)
- If cache directory is changed: Downloads again to new location

#### 9.3. Default Model

- **Default**: `bert-base-uncased` (same as used in `Heterogeneous_Dataset` with `postemb`)
- **Embedding dimension**: 768 (standard BERT dimension)
- **Rationale**: Consistency with existing fidel-ts codebase

### 10. Summary

This plan:
- Adds an **optional on-the-fly embedding path** to `TimeMMD_HeteroGetter`
- Uses a **predictable `.pkl` location**: `data/time_mmd/{subdataset}/{filename}.pkl`
- Ensures embeddings match the **expected format** for TGTSF and LYNX:
  - Per-sample: `(seq_len, news_num, embed_dim)`
  - Per-batch: `[B, l, n, d]` where `l = seq_len`
- Provides a **configurable switch** between raw text (`text`) and embeddings (`embedding`)
- Supports **force re-embedding** for experimentation and model upgrades
- **Caches HF models locally** in `./HF_cache/` (configurable) to avoid re-downloading
- **Defaults to BERT** (`bert-base-uncased`) for consistency with fidel-ts


