# LLM Embedding Integration Plan

## Overview

**Goal**: Connect precomputed LLM embeddings (from `cli.inference generate-suite`) to the training pipeline so models like TimeCMA can access them as `hetero_channel`.

**Current Gap**: Embeddings are cached at `data/{dataset_path}/llm_embeddings/llm_{hash}/` but the data loader doesn't know how to find or load them.

**Scope**: Support LLM embeddings for all dataset types:
- Time-MMD datasets (`time_mmd_*`)
- TTC datasets (`ttc_*`)
- Fidel-TS datasets (`fidel_*`)

---

## Architecture Analysis

### Current Data Flow

```
Training Request
    ↓
Data_Provider.__init__()
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Time-MMD/TTC:                  │ Fidel-TS:                      │
│   TimeMMD_Dataset              │   Universal_Dataset            │
│   └─ TimeMMD_HeteroGetter      │   └─ Heterogeneous_Dataset     │
│      (BERT text embeddings)    │      (Fidel-TS embeddings)     │
└─────────────────────────────────────────────────────────────────┘
    ↓
__getitem__(index) → returns hetero_channel 
    (currently: text embeddings, Fidel-TS embeddings, or None)
```

### Required Data Flow with LLM Embeddings

```
Training Request (with llm_embedding config)
    ↓
Data_Provider.__init__()
    ↓
Detect llm_embedding config → Load LLMEmbeddingProvider
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ All Dataset Types with LLM Embeddings:                         │
│   Dataset (TimeMMD_Dataset or Universal_Dataset)               │
│   └─ LLMEmbeddingProvider (precomputed GPT-2/LLM embeddings)   │
└─────────────────────────────────────────────────────────────────┘
    ↓
__getitem__(index) → LLMEmbeddingProvider[index] 
    → returns LLM embeddings as hetero_channel
```

---

## LLM Embedding Cache Structure

```
data/
├── time_mmd/
│   ├── Traffic/
│   │   ├── US_VMT_Month.csv
│   │   └── llm_embeddings/
│   │       └── llm_{hash}/
│   │           ├── metadata.json
│   │           ├── train/embeddings.h5  # [N, embed_dim, C]
│   │           ├── val/embeddings.h5
│   │           └── test/embeddings.h5
│   └── Climate/
│       └── llm_embeddings/...
├── ttc/
│   └── {domain}/
│       └── llm_embeddings/...
└── {fidel_dataset}/
    └── llm_embeddings/...
```

**Embedding Shape**: `[N, embed_dim, C]`
- `N` = number of samples in split
- `embed_dim` = LLM hidden dimension (768 for GPT-2)
- `C` = number of channels

**Hash Computation** (must match `cli.inference generate`):
- `model_name`, `prompt_template`, `prompt_config`
- `input_len`, `output_len` from training config

---

## Implementation Phases

### Phase 1: Create LLM Embedding Provider

**File**: `embedder/llm_embedding_provider.py` (NEW)

```python
"""
LLM Embedding Provider for Training

Loads precomputed LLM embeddings from cache and provides them
to the data loader during training.

Supports all dataset types: Time-MMD, TTC, and Fidel-TS.
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Any, Optional, List

from embedder.llm_cache import LLMEmbeddingCache, LLMEmbeddingMetadata


class LLMEmbeddingProvider:
    """
    Provides precomputed LLM embeddings for training.
    
    Loads embeddings from the LLM cache and provides them by sample index.
    Implements the interface expected by the data loader for hetero_channel.
    
    Usage:
        provider = LLMEmbeddingProvider.from_experiment_config(config, split='train')
        embedding = provider[sample_index]  # Returns [embed_dim, num_channels]
    
    Supports:
        - Time-MMD datasets (time_mmd_*)
        - TTC datasets (ttc_*)
        - Fidel-TS datasets (fidel_*)
    """
    
    def __init__(self, embeddings: np.ndarray, metadata: LLMEmbeddingMetadata):
        """
        Initialize provider with preloaded embeddings.
        
        Args:
            embeddings: Preloaded embeddings array [N, embed_dim, C]
            metadata: Cache metadata for validation
        """
        self.embeddings = embeddings  # [N, embed_dim, C]
        self.metadata = metadata
        self.num_samples = embeddings.shape[0]
        self.embed_dim = embeddings.shape[1]
        self.num_channels = embeddings.shape[2]
    
    @classmethod
    def from_experiment_config(
        cls, 
        experiment_config: Dict[str, Any], 
        split: str,
        validate: bool = True
    ) -> 'LLMEmbeddingProvider':
        """
        Create provider from experiment configuration.
        
        This method mirrors the hash computation from cli.inference generate
        to find the correct cache directory.
        
        Args:
            experiment_config: Experiment configuration dict containing:
                - data.name: Dataset name (e.g., 'time_mmd_traffic')
                - training.input_len, training.output_len
                - llm_embedding: LLM embedding configuration
            split: Data split ('train', 'val', 'test')
            validate: Whether to validate embeddings after loading
        
        Returns:
            LLMEmbeddingProvider instance
        
        Raises:
            FileNotFoundError: If embeddings not found in cache
            ValueError: If embeddings don't match expected configuration
        """
        from embedder.llm_embedder import LLMEmbedder
        
        # Extract configuration
        dataset_name = experiment_config.get('data', {}).get('name')
        if not dataset_name:
            raise ValueError("Experiment config missing data.name")
        
        llm_config = experiment_config.get('llm_embedding', {})
        if not llm_config:
            raise ValueError("Experiment config missing llm_embedding section")
        
        training = experiment_config.get('training', {})
        input_len = training.get('input_len', 96)
        output_len = training.get('output_len', 96)
        
        # Create embedder to resolve paths and build metadata
        # (Uses same logic as cli.inference generate)
        embedder = LLMEmbedder(
            model_name=llm_config.get('model_name', 'gpt2'),
            cache_dir=llm_config.get('cache_dir', './LLM_cache/'),
            data_root=experiment_config.get('base_data_path', './data/'),
            prompt_template=llm_config.get('prompt_template', 'timecma_v1'),
            prompt_config=llm_config.get('prompt_config', {
                'value_format': 'integer', 
                'include_timestamps': True
            }),
            input_len=input_len,
            output_len=output_len,
            scale=training.get('scale', True),
            data_config_path=experiment_config.get('data', {}).get('config_path'),
        )
        
        # Resolve data directory (handles time_mmd_, ttc_, fidel_ prefixes)
        data_dir = embedder._get_data_directory(dataset_name)
        
        # Build metadata for cache lookup
        metadata = embedder._build_metadata(dataset_name)
        
        # Create cache and load embeddings
        cache = LLMEmbeddingCache(str(data_dir), dataset_name)
        
        if not cache.cache_exists(metadata, split):
            cache_path = cache.get_cache_dir(metadata)
            raise FileNotFoundError(
                f"LLM embeddings not found for '{dataset_name}/{split}'.\n"
                f"Expected cache at: {cache_path}\n"
                f"Hash: {metadata.compute_hash()}\n\n"
                f"Generate embeddings with:\n"
                f"  python -m cli.inference generate-suite <suite_config.yaml>\n"
                f"Or:\n"
                f"  python -m cli.inference generate <experiment_config.yaml>"
            )
        
        # Load embeddings
        embeddings = cache.load_embeddings(metadata, split)
        
        provider = cls(embeddings, metadata)
        
        if validate:
            provider._validate_config(experiment_config)
        
        return provider
    
    def _validate_config(self, experiment_config: Dict[str, Any]):
        """Validate loaded embeddings match experiment configuration."""
        # Validate embedding dimension matches model config
        model_overrides = experiment_config.get('model_config_overrides', {})
        expected_d_llm = model_overrides.get('d_llm', 768)
        
        if self.embed_dim != expected_d_llm:
            raise ValueError(
                f"LLM embedding dimension mismatch: "
                f"cached embeddings have dim={self.embed_dim}, "
                f"but model expects d_llm={expected_d_llm}. "
                f"Regenerate embeddings with matching LLM model."
            )
    
    def validate_sample_count(self, dataset_length: int):
        """
        Validate embedding count matches dataset length.
        
        Call this after dataset is created to ensure alignment.
        """
        if self.num_samples != dataset_length:
            raise ValueError(
                f"LLM embedding count mismatch: "
                f"embeddings={self.num_samples}, dataset={dataset_length}. "
                f"This can happen if:\n"
                f"  1. Dataset was modified after embedding generation\n"
                f"  2. input_len/output_len changed\n"
                f"  3. Split ratios changed\n"
                f"Regenerate embeddings with 'cli.inference generate-suite'."
            )
    
    def __getitem__(self, index: int) -> np.ndarray:
        """
        Return embedding for sample index.
        
        Args:
            index: Sample index (0 to N-1)
        
        Returns:
            Embedding array with shape [embed_dim, num_channels]
        """
        if index < 0 or index >= self.num_samples:
            raise IndexError(
                f"Sample index {index} out of range [0, {self.num_samples})"
            )
        return self.embeddings[index]
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples
    
    def get_for_batch(self, indices: List[int]) -> np.ndarray:
        """
        Return embeddings for batch of indices.
        
        Args:
            indices: List of sample indices
        
        Returns:
            Embeddings array with shape [B, embed_dim, C]
        """
        return self.embeddings[indices]
    
    @property
    def shape(self) -> tuple:
        """Return embedding array shape: (N, embed_dim, C)."""
        return self.embeddings.shape
    
    def __repr__(self) -> str:
        return (
            f"LLMEmbeddingProvider("
            f"samples={self.num_samples}, "
            f"embed_dim={self.embed_dim}, "
            f"channels={self.num_channels}, "
            f"model='{self.metadata.model_name}')"
        )
```

---

### Phase 2: Modify Data Factory

**File**: `data_provider/data_factory.py`

**Changes Required**:

#### 2.1 Add LLM embedding detection in `__init__()`

```python
def __init__(self, args, flag):
    ...
    # Detect LLM embedding configuration
    # This is set when experiment config has llm_embedding section
    self.llm_embedding_config = getattr(args, 'llm_embedding', None)
    self.llm_embedding_provider = None
    
    if self.llm_embedding_config:
        self._setup_llm_embedding_provider(flag)
    ...
```

#### 2.2 Add provider setup method

```python
def _setup_llm_embedding_provider(self, flag):
    """
    Load precomputed LLM embeddings for this split.
    
    Supports all dataset types: Time-MMD, TTC, and Fidel-TS.
    
    Args:
        flag: Dataset split ('train', 'val', 'test')
    
    Raises:
        FileNotFoundError: If embeddings not found (user should run generate-suite)
    """
    from embedder.llm_embedding_provider import LLMEmbeddingProvider
    
    # Build experiment config dict from args
    experiment_config = {
        'data': {
            'name': self._get_dataset_name(),
            'config_path': getattr(self.dataset_config, 'config_path', None),
        },
        'training': {
            'input_len': self.args.input_len,
            'output_len': self.args.output_len,
            'scale': getattr(self.args, 'scale', True),
        },
        'llm_embedding': self.llm_embedding_config,
        'base_data_path': getattr(self.args, 'base_data_path', './data/'),
        'model_config_overrides': getattr(self.args, 'model_config_overrides', {}),
    }
    
    try:
        self.llm_embedding_provider = LLMEmbeddingProvider.from_experiment_config(
            experiment_config, 
            split=flag
        )
        print(f"[ LLM Embeddings ] Loaded {len(self.llm_embedding_provider)} embeddings "
              f"for {flag} (shape: {self.llm_embedding_provider.shape})")
    except FileNotFoundError as e:
        # Re-raise with helpful message
        raise

def _get_dataset_name(self) -> str:
    """
    Get dataset name from configuration.
    
    Handles different config structures for Time-MMD, TTC, and Fidel-TS.
    """
    # Try different locations where dataset name might be stored
    if hasattr(self.dataset_config, 'name'):
        return self.dataset_config.name
    if hasattr(self.args, 'data_name'):
        return self.args.data_name
    if hasattr(self.args.data_config, 'name'):
        return self.args.data_config.name
    raise ValueError("Cannot determine dataset name from configuration")
```

#### 2.3 Pass provider to dataset creation

For Time-MMD/TTC datasets:
```python
def _create_time_mmd_dataset(self, i, flag):
    ...
    return TimeMMD_Dataset(
        ...
        llm_embedding_provider=self.llm_embedding_provider,  # NEW
        ...
    )
```

For Fidel-TS datasets (in `get_datasets()`):
```python
def get_datasets(self, flag):
    ...
    for i in data_id_list:
        if is_time_mmd:
            dataset = self._create_time_mmd_dataset(i, flag)
        else:
            dataset = Universal_Dataset(
                ...
                llm_embedding_provider=self.llm_embedding_provider,  # NEW
                ...
            )
    ...
```

---

### Phase 3: Modify Dataset Classes

#### 3.1 Modify Universal_Dataset (base class)

**File**: `data_provider/data_loader.py`

Add support for LLM embedding provider:

```python
class Universal_Dataset(Dataset):
    def __init__(self, ..., llm_embedding_provider=None, ...):
        ...
        self.llm_embedding_provider = llm_embedding_provider
        self.use_llm_embeddings = llm_embedding_provider is not None
        ...
    
    def __getitem__(self, index):
        ...
        # Get base hetero_channel from normal getter
        hetero_channel = np.zeros((1), dtype=np.float32)
        
        # Override with LLM embeddings if provider is set
        if self.use_llm_embeddings:
            hetero_channel = self.llm_embedding_provider[index]
        elif self.preload_hetero:
            hetero_channel = self.hetero_channel
        elif 'x_hetero' in self.custom_input or 'y_hetero' in self.custom_input:
            # Existing hetero getter logic
            ...
        
        return (sample_id, seq_x, seq_y, x_time, y_time, 
                x_hetero, y_hetero, hetero_x_time, hetero_y_time, 
                hetero_general, hetero_channel, 
                x_time_features, y_time_features)
```

#### 3.2 Modify TimeMMD_Dataset

**File**: `data_provider/time_mmd_dataset.py`

```python
class TimeMMD_Dataset(Universal_Dataset):
    def __init__(self, ..., llm_embedding_provider=None, ...):
        ...
        # Pass to parent
        super().__init__(..., llm_embedding_provider=llm_embedding_provider, ...)
        ...
    
    def _setup_hetero_getter(self):
        """Setup heterogeneous data getter based on configuration."""
        if self.use_llm_embeddings:
            # LLM embeddings are handled by parent class __getitem__
            # Skip text getter setup
            print("[ TimeMMD ] Using LLM embeddings (skipping text getter)")
            return
        else:
            # Normal text embeddings mode
            self._setup_text_getter()
```

---

### Phase 4: Propagate Configuration

#### 4.1 Suite Executor

**File**: `runs/suite_executor.py`

Ensure `llm_embedding` config is included when building experiment args:

```python
def _execute_experiment(self, exp_config):
    ...
    final_config = merge_configs(template, overrides)
    
    # Ensure llm_embedding is passed through
    # (Should already work if merge_configs handles it correctly)
    ...
```

#### 4.2 PyTorch Runner

**File**: `runs/pytorch.py`

Ensure `llm_embedding` is set on args:

```python
def run(experiment_config, ...):
    ...
    # Convert experiment config to args
    args = build_args_from_config(experiment_config)
    
    # Ensure llm_embedding is on args
    if 'llm_embedding' in experiment_config:
        args.llm_embedding = experiment_config['llm_embedding']
    ...
```

#### 4.3 Experiment Config Models

**File**: `cli/config/models.py`

Already has `llm_embedding: Optional[LLMEmbeddingConfig]` - verify it's propagated:

```python
class ExperimentConfig(BaseModel):
    ...
    llm_embedding: Optional[LLMEmbeddingConfig] = Field(
        default=None, 
        description="LLM embedding configuration (for TimeCMA-style models)"
    )
```

---

### Phase 5: Validation and Error Handling

#### 5.1 Sample Count Validation

In `Data_Provider.get_datasets()` after creating datasets:

```python
def get_datasets(self, flag):
    ...
    # After all datasets created, validate LLM embedding count
    if self.llm_embedding_provider:
        total_samples = sum(len(d) for d in datasets.values())
        self.llm_embedding_provider.validate_sample_count(total_samples)
    ...
```

#### 5.2 Dimension Validation

In model initialization (e.g., `models/TimeCMA.py`):

```python
def __init__(self, configs):
    ...
    self.d_llm = getattr(configs, 'd_llm', 768)
    
    # Validate at forward() time if embeddings provided
    ...
```

#### 5.3 Helpful Error Messages

Already included in `LLMEmbeddingProvider.from_experiment_config()`.

---

## Dataset Type Support Matrix

| Dataset Type | Prefix | Data Directory Resolution | Supported |
|--------------|--------|---------------------------|-----------|
| Time-MMD | `time_mmd_*` | `data/time_mmd/{Domain}/` | ✓ |
| TTC | `ttc_*` | `data/ttc/{domain}/` | ✓ |
| Fidel-TS | `fidel_*` | `data/{Dataset}/` | ✓ |

All dataset types use the same `LLMEmbeddingProvider` - the resolution logic is in `LLMEmbedder._get_data_directory()` which already handles all three prefixes.

---

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `embedder/llm_embedding_provider.py` | **CREATE** | New class to load and provide LLM embeddings |
| `data_provider/data_factory.py` | **MODIFY** | Detect `llm_embedding` config, create provider |
| `data_provider/data_loader.py` | **MODIFY** | Add `llm_embedding_provider` to `Universal_Dataset` |
| `data_provider/time_mmd_dataset.py` | **MODIFY** | Accept provider, skip text getter when using LLM |
| `runs/pytorch.py` | **MODIFY** | Ensure `llm_embedding` config passed to args |
| `runs/suite_executor.py` | **VERIFY** | Confirm config merging works correctly |

---

## Testing Plan

### Unit Tests

1. `LLMEmbeddingProvider` tests:
   - `from_experiment_config()` correctly resolves cache for all dataset types
   - Hash computation matches between generate and load
   - Indexing returns correct shapes
   - Validation catches mismatched dimensions

2. `Data_Provider` tests:
   - Detects `llm_embedding` config
   - Creates provider correctly
   - Passes provider to datasets

### Integration Tests

1. End-to-end for each dataset type:
   ```bash
   # Time-MMD
   python -m cli.inference generate configs/experiments/timecma_time_mmd.yaml
   python -m cli.suite run configs/experiment_suites/timecma_test.yaml
   
   # TTC (when configs exist)
   python -m cli.inference generate configs/experiments/timecma_ttc.yaml
   
   # Fidel-TS (when configs exist)
   python -m cli.inference generate configs/experiments/timecma_fidel.yaml
   ```

2. Error handling:
   - Missing embeddings → clear error message
   - Dimension mismatch → validation error
   - Sample count mismatch → validation error

### Manual Verification Checklist

- [ ] `generate-suite` creates embeddings in correct location
- [ ] `verify-suite` finds and validates embeddings
- [ ] `suite run` loads embeddings without error
- [ ] Training completes successfully with embeddings
- [ ] `hetero_channel` tensor has correct shape in model forward

---

## Estimated Effort

| Phase | Effort | Priority |
|-------|--------|----------|
| Phase 1: LLMEmbeddingProvider | 2-3 hours | High |
| Phase 2: Data Factory changes | 1-2 hours | High |
| Phase 3: Dataset class changes | 2-3 hours | High |
| Phase 4: Config propagation | 30 min | Medium |
| Phase 5: Validation | 1 hour | Medium |
| Testing | 2-3 hours | High |

**Total**: ~10-14 hours

---

## Future Enhancements

1. **Lazy Loading**: Load embeddings on-demand instead of all at once (for very large datasets)
2. **Memory Mapping**: Use `h5py` memory mapping for huge embedding files
3. **Multi-GPU**: Ensure embeddings are correctly distributed across GPUs
4. **Caching**: Cache provider instances to avoid reloading between epochs
5. **Mixed Mode**: Support both LLM embeddings AND text embeddings simultaneously

