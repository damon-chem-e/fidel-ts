# Time-MMD Embedding Implementation Summary

## Overview

On-the-fly text embedding functionality has been implemented for Time-MMD datasets, allowing models like TGTSF and LYNX to use pre-computed embeddings instead of raw text strings.

## Implementation Details

### 1. TimeMMD_HeteroGetter Enhancements

**Location**: `data_provider/time_mmd_dataset.py`

**New Features**:
- **Embedding generation**: Converts text to embeddings using HuggingFace models (default: BERT)
- **Caching**: Saves embeddings to `.pkl` files for reuse
- **Model caching**: Caches HF models locally in `./HF_cache/` to avoid re-downloading
- **Lazy loading**: Only loads embedding model when `output_format='embedding'`

**New Parameters**:
- `embed_model_name`: HF model name (default: `'bert-base-uncased'`)
- `embed_dim`: Embedding dimension (default: `768`)
- `force_reembed`: Force recomputation (default: `False`)
- `hf_cache_dir`: Local cache directory (default: `'./HF_cache/'`)
- `root_path`: Dataset root path (for embedding file location)
- `data_path`: Data file path (for embedding file location)
- `device`: Device for embedding model (default: `'cpu'`)

**Key Methods**:
- `_load_embedding_model()`: Loads tokenizer and model with local caching
- `_embed_text_corpus()`: Embeds all text data in batches
- `_load_or_create_embeddings()`: Loads from `.pkl` or creates on-the-fly
- `_get_embedding_path()`: Computes `.pkl` file path

### 2. TimeMMD_Dataset Updates

**Location**: `data_provider/time_mmd_dataset.py`

**Changes**:
- Added embedding parameters to `__init__()`
- Passes embedding config to `TimeMMD_HeteroGetter`

### 3. Data Factory Updates

**Location**: `data_provider/data_factory.py`

**Changes**:
- Reads `timemmd_text_output` from data config
- Maps `timemmd_text_output` to `output_format`:
  - `'text'` → `output_format='json'` (for ChatTime)
  - `'embedding'` → `output_format='embedding'` (for TGTSF/LYNX)
- Passes embedding parameters to `TimeMMD_Dataset`

### 4. Configuration

**Data Config Example** (`data_configs/time_mmd/{subdataset}/config.yaml`):

```yaml
# Text embedding configuration
timemmd_text_output: embedding  # 'text' | 'embedding'
timemmd_embed_model: bert-base-uncased  # HF model name
timemmd_embed_dim: 768  # Embedding dimension
timemmd_force_reembed: false  # Force recomputation
hf_cache_dir: ./HF_cache/  # Local cache directory (data-agnostic)
```

## Usage

### For ChatTime (Raw Text)

```yaml
timemmd_text_output: text
```

### For TGTSF/LYNX (Embeddings)

```yaml
timemmd_text_output: embedding
timemmd_embed_model: bert-base-uncased
timemmd_embed_dim: 768
```

## File Structure

```
data/time_mmd/{subdataset}/
  ├── {filename}.csv      # Original CSV with text
  └── {filename}.pkl      # Cached embeddings (auto-generated)

HF_cache/                 # Local HF model cache
  └── bert-base-uncased/  # Cached BERT model
```

## Embedding Format

**Storage** (`.pkl` file):
```python
{
    "YYYYMMDDHHMMSS": np.ndarray(shape=(1, embed_dim), dtype=np.float32),
    ...
}
```

**Output Shape**:
- Per sample: `(seq_len, 1, embed_dim)` where `seq_len` is the sequence length
- Per batch: `[B, seq_len, 1, embed_dim]` after DataLoader collation
- Matches TGTSF expectation: `news: [B, l, news_num, text_dim]` where `news_num=1`

## Model Caching

- **First run**: Downloads model from HuggingFace → saves to `./HF_cache/{model_name}/`
- **Subsequent runs**: Loads from `./HF_cache/{model_name}/` (no download)
- **Cache location**: Configurable via `hf_cache_dir` (shared across datasets)

## Performance Notes

- **First embedding**: May take several minutes depending on dataset size
- **Subsequent runs**: Fast loading from `.pkl` file
- **Model loading**: First time downloads model (~400MB for BERT), cached thereafter

## Testing

To test embedding functionality:

1. **Set config**:
   ```yaml
   timemmd_text_output: embedding
   ```

2. **Run training**:
   ```bash
   python -m cli.train pytorch configs/experiments/time_mmd_test_tgtsf.yaml
   ```

3. **Check output**:
   - Embeddings will be computed on first run
   - `.pkl` file will be created in `data/time_mmd/{subdataset}/`
   - Subsequent runs will load from `.pkl`

## Troubleshooting

### Force Re-embedding

If you change the embedding model or want to recompute:

```yaml
timemmd_force_reembed: true
```

### Custom Cache Directory

```yaml
hf_cache_dir: /path/to/custom/cache/
```

### Different Embedding Model

```yaml
timemmd_embed_model: sentence-transformers/all-MiniLM-L6-v2
timemmd_embed_dim: 384  # Update dimension to match model
```

## Compatibility

- ✅ **TGTSF**: Uses embeddings `[B, l, n, d]`
- ✅ **LYNX**: Uses embeddings `[B, l, n, d]`
- ✅ **ChatTime**: Uses raw text (set `timemmd_text_output: text`)
- ✅ **DLinear**: Ignores text (works with either mode)

