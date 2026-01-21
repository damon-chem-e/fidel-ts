# Embedding Version Tracking Implementation

**Date**: 2026-01-21
**Purpose**: Document how embedding version information is tracked and can be accessed for experiment metadata

## Summary

Implemented `embedding_version` field to distinguish between:
- **Version 1.0**: Old .pkl files with concatenated channel descriptions (buggy)
- **Version 2.0**: Fixed per-variable embeddings with parquet column order alignment

## Implementation

### 1. Embedding Metadata (embedder/metadata.py)

Added `embedding_version` parameter to `EmbeddingMetadata`:

```python
def __init__(self,
             ...
             embedding_version: str = '2.0',  # Defaults to new fixed version
             ...):
```

**Included in cache hash**: Version changes invalidate old caches automatically.

### 2. Embedding Loader (embedder/fidel_ts_embedder.py)

- `_create_metadata_without_model()`: Sets `embedding_version='2.0'` for new embeddings
- `_load_old_embeddings()`: Sets `embedding_version='1.0'` for old .pkl files
- `_load_from_cache()`: Loads version from cache `metadata.json`
- `get_embedding_metadata_dict()`: Exposes metadata as dict for experiment tracking

### 3. Data Loader (data_provider/data_loader.py:673)

Stores embedding metadata in `Heterogeneous_Dataset`:

```python
self.embedding_metadata = loader.get_embedding_metadata_dict()
```

## How to Access Embedding Version in Experiments

### Option 1: Access from Dataset (Currently Implemented)

The embedding metadata is stored in the `Heterogeneous_Dataset` instance:

```python
# In experiment code, after creating dataset:
dataset = data_factory.get_dataset(...)
if hasattr(dataset, 'embedding_metadata') and dataset.embedding_metadata:
    embedding_version = dataset.embedding_metadata['embedding_version']
    print(f"Using embeddings version: {embedding_version}")
```

### Option 2: Add to Experiment Metadata (Recommended)

To automatically save embedding version to experiment metadata, modify `exp/manager.py`:

1. Pass dataset to ExperimentManager during init
2. In `_capture_metadata()`, add:

```python
# Capture embedding metadata if available
if hasattr(self, 'train_dataset'):
    dataset = self.train_dataset
    if hasattr(dataset, 'embedding_metadata') and dataset.embedding_metadata:
        metadata["embedding_info"] = dataset.embedding_metadata
```

This would save embedding metadata to `{experiment_dir}/metadata.json`.

### Option 3: Check Cache Directory

Embedding version is always saved in the cache `metadata.json`:

```bash
# Find cache directory
ls data/Jena_Atmospheric_Physics/weather/embeddings_cache/

# Check metadata
cat data/Jena_Atmospheric_Physics/weather/embeddings_cache/embeddings_{hash}/metadata.json
```

Look for `"embedding_version": "2.0"` or `"1.0"`.

## Verification

### New Embeddings (v2.0)
- Generated with current code
- Cache directory: `embeddings_{new_hash}/`
- `metadata.json` contains: `"embedding_version": "2.0"`

### Old Embeddings (v1.0)
- Loaded from old `.pkl` files with `use_old_embeddings=True`
- Metadata created in-memory, marked as version `1.0`
- Cache directory: `embeddings_{old_hash}/` (if exists)

### Different Hash Values

Because `embedding_version` is included in the hash computation, v1.0 and v2.0 use **different cache directories**:

- Old buggy embeddings: `embeddings_abc123.../` (version 1.0)
- New fixed embeddings: `embeddings_def456.../` (version 2.0)

This ensures no accidental mixing of old and new embeddings.

## ✅ Automatic Experiment Metadata Integration (IMPLEMENTED)

**Status**: COMPLETE - Embedding version is now automatically saved to experiment metadata.json

### Implementation Details

1. **ExperimentManager** (`exp/manager.py:1130-1172`):
   - Added `capture_embedding_metadata(dataset)` method
   - Extracts `embedding_metadata` from dataset and saves to `metadata.json`
   - Logs embedding version and model info

2. **Data_Provider** (`data_provider/data_factory.py:51,897-900`):
   - Accepts `exp_manager` parameter
   - Calls `exp_manager.capture_embedding_metadata()` after creating train_dataset

3. **Exp_Basic** (`exp/exp_basic.py:62-67`):
   - Passes `exp_manager` to Data_Provider during initialization

4. **Lightning Support** (`data_provider/lightning_data_module.py:24,68-72`, `exp/exp_lightning.py:339`):
   - TimeSeriesDataModule accepts and stores `exp_manager`
   - Passes it to Data_Provider during setup
   - train_lightning_model passes exp_manager to TimeSeriesDataModule

### Result

Every experiment's `{experiment_dir}/metadata.json` now contains:
```json
{
  "experiment_id": "...",
  "experiment_name": "...",
  "embedding_info": {
    "model_name": "bert-base-uncased",
    "aggregation_method": "cls",
    "embedding_version": "2.0",
    "embedding_dim": 768,
    "max_length": 512,
    "created_at": "2026-01-21T..."
  },
  ...
}
```

### Backwards Compatibility

- **New experiments**: Have `"embedding_info": {...}` with version 2.0 or 1.0
- **Old experiments** (pre-version-tracking): Have `"embedding_info": null`
- **No embeddings used**: Have `"embedding_info": null`

This allows you to identify which embedding version was used by checking the metadata.json:
- `null` = pre-version-tracking (buggy embeddings)
- `"embedding_version": "1.0"` = old .pkl files (buggy)
- `"embedding_version": "2.0"` = new fixed embeddings

## Quick Check Command

```bash
# Check all embedding caches and their versions
find data -name "metadata.json" -path "*/embeddings_*/metadata.json" -exec sh -c 'echo "File: $1"; jq -r ".embedding_version" "$1"' _ {} \;
```

## Version History

- **1.0**: Original implementation with concatenated channel descriptions (buggy)
  - Nested channel_info: All variable descriptions concatenated into single string
  - Single embedding broadcast across all variables
  - TGTSF models crash with shape mismatch on multi-variable datasets

- **2.0**: Fixed per-variable embeddings (2026-01-21)
  - Nested channel_info: Separate embedding for each variable
  - Embeddings stacked in parquet column order
  - TGTSF models work correctly on all datasets
  - Includes Stages 1, 2, and 2.5 fixes
