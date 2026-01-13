# Tensor Cache Test Plan

This document outlines the testing strategy for the tensor cache implementation.

## Overview

The tensor cache pre-computes all CPU-intensive dataloader operations and stores results as memory-mapped numpy arrays. Testing should verify:
1. Cache generation correctness
2. Cache validation
3. Training integration
4. Performance improvements

## Cache Storage Design

Tensor caches are stored alongside dataset data using a hash-based directory structure:

```
data/fidel-ts/germany_renewable/
├── time_series/              # Original dataset files
├── embeddings/               # Pre-computed embeddings
└── tensor_cache/
    └── a1b2c3d4e5f67890/    # Config hash directory
        ├── metadata.json     # Cache metadata
        ├── train/
        │   ├── seq_x.npy
        │   ├── seq_y.npy
        │   └── ...
        ├── val/
        └── test/
```

**Config parameters that affect the hash:**
- `input_len`, `output_len`
- `scale` (normalization)
- `truncate_train_for_purge`
- `downsample`
- `data_name`
- `hetero_stride`
- `hetero_type`
- `missing_value_strategy`
- `split_info`

If any of these change, a new cache directory is created with a different hash.

## Test Phases

### Phase 1: Unit Tests (CLI Commands)

#### 1.1 Multi-Experiment Cache Generation
```bash
# Generate caches for ALL experiments in a suite
python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/suite.yaml
```

Expected:
- Analyzes all experiments and groups by config hash
- Shows deduplication (experiments with same hash share cache)
- Generates unique caches only
- Shows final summary with counts

#### 1.2 Filtered Generation
```bash
# Generate only for experiments matching pattern
python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/suite.yaml --filter germany
```

Expected:
- Only processes experiments with "germany" in name
- Shows filter applied in output

#### 1.3 Dry Run
```bash
# Preview what would be generated without generating
python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/suite.yaml --dry-run
```

Expected:
- Shows analysis of experiments
- Shows unique caches that would be generated
- Does NOT create any files

#### 1.4 Cache Reuse (Skip Regeneration)
```bash
# Run same command again - should skip valid caches
python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/suite.yaml
```

Expected:
- Message: "Cache already valid, skipping" for each existing valid cache
- Only generates missing/invalid caches

#### 1.5 Multi-Experiment Validation
```bash
# Validate caches for all experiments
python -m cli.tensor_cache validate configs/experiment_suites/lynx_film/suite.yaml
```

Expected:
- Shows validation status for each unique cache
- Lists experiments covered by each cache
- Summary shows valid/invalid counts

#### 1.6 Filtered Validation
```bash
# Validate only for experiments matching pattern
python -m cli.tensor_cache validate configs/experiment_suites/lynx_film/suite.yaml --filter renewable
```

Expected:
- Only validates caches for matching experiments

#### 1.7 Cache Info
```bash
# View cache information (use actual cache path from generate output)
python -m cli.tensor_cache info data/fidel-ts/germany_renewable/tensor_cache/a1b2c3d4e5f67890/
```

Expected:
- Shows metadata (version, config hash, creation date)
- Shows shapes and sizes for each split
- Shows entity information

#### 1.8 Cache Benchmark
```bash
# Benchmark cache loading speed
python -m cli.tensor_cache benchmark data/fidel-ts/germany_renewable/tensor_cache/a1b2c3d4e5f67890/
```

Expected:
- Shows throughput comparison (samples/sec)
- Should show 10-100x improvement over baseline

### Phase 2: Integration Tests

#### 2.1 Training with Tensor Cache (Auto-Discovery)

Enable tensor cache in experiment config - no need to specify directory:

```yaml
# configs/test/tensor_cache_test.yaml
model:
  name: DLinear
  config_path: configs/models/DLinear.yaml

data:
  name: germany_renewable
  config_path: configs/data/fidel-ts/germany_renewable.yaml

training:
  epochs: 1
  batch_size: 32
  input_len: 96
  output_len: 96
  use_tensor_cache: true
  # tensor_cache_dir is auto-resolved to: data/fidel-ts/germany_renewable/tensor_cache/<hash>/
```

Workflow:
```bash
# 1. First generate the cache
python -m cli.tensor_cache generate configs/test/tensor_cache_test.yaml

# 2. Run training - auto-discovers cache via hash
python -m runs.pytorch --config configs/test/tensor_cache_test.yaml
```

Expected:
- Training starts successfully
- Log shows "Using tensor cache from: data/fidel-ts/germany_renewable/tensor_cache/<hash>/"
- Training completes without errors

#### 2.2 Cache Miss Handling

Test with missing cache:
```bash
# Try to train with use_tensor_cache: true but no cache generated
python -m runs.pytorch --config configs/test/tensor_cache_test.yaml
```

Expected:
- Clear error message: "Tensor cache requested but invalid: Cache directory does not exist"
- Shows expected path and suggests running cache generation command

### Phase 3: Correctness Validation

#### 3.1 Output Comparison

Compare outputs between cached and non-cached loading:

```python
# scripts/test_tensor_cache_correctness.py
import torch
from data_provider.data_factory import Data_Provider
from data_provider.tensor_cache import TensorCacheDataset

# Load same sample from both sources
# Compare tensor values with torch.allclose()
```

Run comparison:
```bash
python scripts/test_tensor_cache_correctness.py --config <config> --cache-dir <cache>
```

Expected:
- All tensor values match within floating point tolerance
- Sample IDs match exactly

### Phase 4: Performance Validation

#### 4.1 Throughput Measurement

Measure actual training throughput improvement:

```bash
# Without cache (baseline)
python -m runs.pytorch --config configs/test/tensor_cache_baseline.yaml 2>&1 | tee baseline.log

# With cache
python -m runs.pytorch --config configs/test/tensor_cache_enabled.yaml 2>&1 | tee cached.log

# Compare training times
```

Expected:
- Significant reduction in per-epoch time
- GPU utilization should increase (less waiting on data)

#### 4.2 Memory Usage

Monitor memory during cache usage:

```bash
# Watch memory while training with large cache
watch -n 1 free -h

# Or use nvidia-smi for GPU memory
watch -n 1 nvidia-smi
```

Expected:
- Memory-mapped arrays should NOT load entire cache to RAM
- RAM usage should be reasonable (not explode with large datasets)

## Test Checklist

### Generation Tests
- [ ] Cache generates without errors for Fidel-TS dataset
- [ ] Cache generates without errors for Time-MMD dataset
- [ ] Metadata file contains correct information
- [ ] All expected array files are created

### Validation Tests
- [ ] Valid cache passes validation
- [ ] Cache with different config hash fails validation
- [ ] Missing metadata file reports clear error
- [ ] Missing array files report clear error

### Integration Tests
- [ ] Data_Provider uses cache when enabled
- [ ] Training completes successfully with cache
- [ ] Clear error when cache missing/invalid
- [ ] return_type='set' raises appropriate error

### Correctness Tests
- [ ] Cached seq_x matches original seq_x
- [ ] Cached seq_y matches original seq_y
- [ ] Cached hetero_x matches original (if applicable)
- [ ] Sample ordering is consistent

### Performance Tests
- [ ] Benchmark shows significant speedup
- [ ] GPU utilization improves during training
- [ ] Memory usage is reasonable

## Quick Validation Commands

After implementation, run these commands to verify basic functionality:

```bash
# 1. Preview what caches would be generated (dry run)
python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/suite.yaml --dry-run

# 2. Generate caches for all experiments in suite
python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/suite.yaml

# 3. Generate only for filtered experiments
python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/suite.yaml --filter germany

# 4. Validate all caches
python -m cli.tensor_cache validate configs/experiment_suites/lynx_film/suite.yaml

# 5. View cache info (use path from generate output)
python -m cli.tensor_cache info data/fidel-ts/germany_renewable/tensor_cache/<hash>/

# 6. Run benchmark
python -m cli.tensor_cache benchmark data/fidel-ts/germany_renewable/tensor_cache/<hash>/

# 7. Run training with cache
# (Set use_tensor_cache: true in config, no need to specify directory)
```

## Known Limitations

1. **return_type='set' not supported**: Tensor cache only returns DataLoaders, not raw datasets
2. **Scaler parameters**: Currently not stored in cache (inverse_transform may need original dataset)
3. **Dynamic data augmentation**: Cache captures single snapshot, no runtime augmentation

## Cleanup

Tensor caches are stored alongside dataset data, so cleanup is straightforward:

```bash
# Remove all tensor caches for a dataset
rm -rf data/fidel-ts/germany_renewable/tensor_cache/

# Or remove a specific cache by hash
rm -rf data/fidel-ts/germany_renewable/tensor_cache/a1b2c3d4e5f67890/
```

Old caches with different hashes can accumulate over time if configs change frequently.
Consider periodic cleanup of unused cache directories.
