# Tensor Cache: Amortized Data Loading for Fast Training

## Overview

The tensor cache is a performance optimization that pre-computes all CPU-intensive dataloader operations and stores the results as memory-mapped numpy arrays. This enables **100-1000x faster data loading** during training by eliminating per-sample computation overhead.

## Problem Statement

### The CPU Bottleneck

Training on GPU clusters with limited vCPUs (e.g., RTX 5000 Ada with 6 vCPU) often results in:
- **Low CPU utilization**: ~20% despite being the bottleneck
- **Sparse GPU utilization**: GPU starved waiting for data
- **Long training times**: >1 day for experiments that should complete in hours

### Root Cause: Per-Sample Computation

Profiling revealed that each sample retrieval (`__getitem__`) takes **~18.6 ms** due to:

| Operation | Time (μs) | % of Total |
|-----------|-----------|------------|
| `hetero_downtime_check` | 3,171 | 34.2% |
| `hetero_embedding_lookup` | 2,989 | 32.3% |
| `hetero_time_matching` | 1,490 | 16.1% |
| `ts_array_slicing` | 534 | 5.8% |
| `normalization` | 312 | 3.4% |
| Other operations | 756 | 8.2% |

The primary bottleneck (`hetero_downtime_check`) is a non-vectorized Python loop that checks timestamps against downtime intervals - this cannot be easily parallelized within the DataLoader.

### The Amortization Solution

Instead of computing these operations for every sample during every epoch, we:
1. **Pre-compute once**: Run all operations in a CPU-only job
2. **Store as arrays**: Save results as memory-mapped numpy arrays
3. **Load instantly**: Training reads pre-computed arrays with O(1) lookups

This is the "second amortization" - similar to how embeddings are pre-computed, but for the entire dataloader pipeline.

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CACHE GENERATION (CPU Job)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Data_Provider│───▶│ Universal_   │───▶│ TensorCache  │       │
│  │              │    │ Dataset      │    │ Generator    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                 │                │
│                                                 ▼                │
│                           ┌─────────────────────────────────────┐│
│                           │  Memory-Mapped Arrays (.npy)        ││
│                           │  • seq_x, seq_y                     ││
│                           │  • hetero_x, hetero_y               ││
│                           │  • x_time, y_time                   ││
│                           │  • metadata.json                    ││
│                           └─────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING (GPU Job)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ TensorCache  │───▶│  DataLoader  │───▶│    Model     │       │
│  │ Dataset      │    │  (fast!)     │    │   Training   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                                                        │
│         ▼                                                        │
│  __getitem__(idx):                                               │
│    return arrays[idx]  # O(1) - just array indexing!            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Storage Structure

Tensor caches are stored alongside dataset data using a **hash-based directory structure**:

```
data/fidel-ts/germany_renewable/
├── time_series/                    # Original dataset files
├── embeddings/                     # Pre-computed text embeddings
└── tensor_cache/
    └── a1b2c3d4e5f67890/          # 16-character config hash
        ├── metadata.json           # Cache metadata & validation info
        ├── train/
        │   ├── sample_ids.npy      # Sample identifiers
        │   ├── seq_x.npy           # Input sequences
        │   ├── seq_y.npy           # Target sequences
        │   ├── x_time.npy          # Input timestamps
        │   ├── y_time.npy          # Target timestamps
        │   ├── hetero_x.npy        # Heterogeneous input features
        │   ├── hetero_y.npy        # Heterogeneous target features
        │   └── ...
        ├── val/
        │   └── ...
        └── test/
            └── ...
```

### Config Hash

The 16-character hash is computed from configuration parameters that affect cache validity:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `input_len` | Input sequence length | 336 |
| `output_len` | Prediction horizon | 168 |
| `scale` | Whether normalization is applied | true |
| `truncate_train_for_purge` | Lookahead bias prevention | false |
| `downsample` | Downsampling factor | null |
| `data_name` | Dataset identifier | "germany_renewable" |
| `hetero_stride` | Heterogeneous data stride | 8 |
| `hetero_type` | Type of heterogeneous data | "embedding" |
| `missing_value_strategy` | How missing values are handled | "none" |
| `split_info` | Train/val/test split configuration | "2020-01-01,2021-01-01" |

**If any of these parameters change, a new cache directory is created with a different hash.** This ensures:
- Old caches are never incorrectly reused
- Multiple experiment configurations can coexist
- Cache validity is automatically verified

## Usage

### Step 1: Generate the Cache

Use the CLI to pre-generate the tensor cache:

```bash
# Generate cache for an experiment suite
python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/germany_renewable.yaml
```

Output:
```
Generating tensor cache for: germany_renewable_lynx_film
Model: LynxFilm
Data: germany_renewable
Input len: 336, Output len: 168
Config hash: a1b2c3d4e5f67890
Cache directory: data/fidel-ts/germany_renewable/tensor_cache/a1b2c3d4e5f67890

Initializing Data_Provider...
Loading train datasets ━━━━━━━━━━━━━━━━━━━━ 100% • 50/50
Generating cache for splits: ['train', 'val', 'test']

Generating tensor cache...
Entities (train): 100%|██████████| 50/50
Entities (val):   100%|██████████| 50/50
Entities (test):  100%|██████████| 50/50

Cache generated successfully at: data/fidel-ts/germany_renewable/tensor_cache/a1b2c3d4e5f67890

┏━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Split ┃ Samples  ┃ seq_x shape        ┃
┡━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ train │ 125,000  │ [125000, 336, 4]   │
│ val   │ 18,000   │ [18000, 336, 4]    │
│ test  │ 36,000   │ [36000, 336, 4]    │
└───────┴──────────┴────────────────────┘
```

### Step 2: Enable in Experiment Config

Add `use_tensor_cache: true` to your training config:

```yaml
# configs/experiment_suites/lynx_film/germany_renewable.yaml
model:
  name: LynxFilm
  config_path: configs/models/lynx_film.yaml

data:
  name: germany_renewable
  config_path: configs/data/fidel-ts/germany_renewable.yaml

training:
  epochs: 30
  batch_size: 768
  input_len: 336
  output_len: 168
  use_tensor_cache: true    # Enable tensor cache
  # tensor_cache_dir: ...   # Optional: auto-resolved from config hash
```

### Step 3: Run Training

Training automatically discovers and uses the cache:

```bash
python -m runs.pytorch --config configs/experiment_suites/lynx_film/germany_renewable.yaml
```

Output:
```
[ info ] Using tensor cache from: data/fidel-ts/germany_renewable/tensor_cache/a1b2c3d4e5f67890
>>>>>>>start training : germany_renewable_lynx_film_20240115_143022>>>>>>>>>>>>>>>>>>>>>>>>>>
```

## CLI Commands

### `generate` - Create Tensor Cache

Processes **all experiments** in a suite, with automatic deduplication by config hash.

```bash
python -m cli.tensor_cache generate <suite_config.yaml> [OPTIONS]

Options:
  --output-dir, -o    Override output directory (default: auto-generated)
  --filter, -f        Filter experiments by name pattern (case-insensitive)
  --chunk-size        Samples per processing chunk (default: 10000)
  --splits            Comma-separated splits to generate (default: train,val,test)
  --force             Overwrite existing cache
  --dry-run           Show what would be generated without generating
```

**Examples:**
```bash
# Generate caches for ALL experiments in suite
python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/suite.yaml

# Generate only for experiments matching "germany"
python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/suite.yaml --filter germany

# Preview what would be generated
python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/suite.yaml --dry-run
```

**Deduplication**: Experiments with identical config parameters share the same cache:
```
Analyzing experiments...
  ● germany_dlinear: hash=a1b2c3d4 (new)
  ○ germany_lynx: hash=a1b2c3d4 (same as germany_dlinear)
  ● canada_dlinear: hash=f9e8d7c6 (new)

Unique caches to generate: 2
```

### `validate` - Check Cache Validity

Validates caches for **all experiments** in a suite.

```bash
python -m cli.tensor_cache validate <suite_config.yaml> [OPTIONS]

Options:
  --cache-dir, -c     Override cache directory (default: auto-resolved)
  --filter, -f        Filter experiments by name pattern (case-insensitive)
```

**Examples:**
```bash
# Validate caches for all experiments
python -m cli.tensor_cache validate configs/experiment_suites/lynx_film/suite.yaml

# Validate only for experiments matching pattern
python -m cli.tensor_cache validate configs/experiment_suites/lynx_film/suite.yaml --filter renewable
```

### `info` - Display Cache Information

```bash
python -m cli.tensor_cache info <cache_dir>
```

Output:
```
Tensor Cache Info
Directory: data/fidel-ts/germany_renewable/tensor_cache/a1b2c3d4e5f67890
Version: 1.0.0
Created: 2024-01-15T14:30:22.123456
Config hash: a1b2c3d4e5f67890

Data Configuration
  input_len: 336
  output_len: 168
  scale: True
  data_name: germany_renewable

┏━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Split ┃ Array       ┃ Shape              ┃
┡━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ train │ seq_x       │ [125000, 336, 4]   │
│ train │ seq_y       │ [125000, 168, 4]   │
│ train │ hetero_x    │ [125000, 42, 768]  │
...

Disk Usage
  train: 2.34 GB
  val: 0.34 GB
  test: 0.67 GB
  Total: 3.35 GB
```

### `benchmark` - Measure Loading Speed

```bash
python -m cli.tensor_cache benchmark <cache_dir> [OPTIONS]

Options:
  --num-batches, -n   Number of batches to benchmark (default: 100)
  --batch-size, -b    Batch size (default: 768)
  --num-workers, -w   Number of dataloader workers (default: 4)
  --preload           Preload data to RAM
```

## Configuration Options

### TrainingConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_tensor_cache` | bool | false | Enable tensor cache for data loading |
| `tensor_cache_dir` | string | null | Override cache directory (auto-resolved if null) |

### Existing Optimization Fields

These fields are independent of tensor cache and affect standard data loading:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_workers` | int | 0 | Number of DataLoader worker processes |
| `prefetch_factor` | int | 2 | Batches to prefetch per worker |

## Workflow for SLURM Clusters

### Recommended Setup

1. **Cache Generation Job** (CPU-only, high memory)
   ```bash
   #!/bin/bash
   #SBATCH --job-name=tensor_cache
   #SBATCH --cpus-per-task=16
   #SBATCH --mem=64G
   #SBATCH --time=4:00:00
   #SBATCH --partition=cpu

   python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/germany_renewable.yaml
   ```

2. **Training Job** (GPU, low CPU)
   ```bash
   #!/bin/bash
   #SBATCH --job-name=train_lynx
   #SBATCH --gres=gpu:1
   #SBATCH --cpus-per-task=6
   #SBATCH --mem=32G
   #SBATCH --time=8:00:00
   #SBATCH --partition=gpu

   python -m runs.pytorch --config configs/experiment_suites/lynx_film/germany_renewable.yaml
   ```

### Cache Sharing

Since caches are stored by config hash, multiple jobs with the same configuration automatically share the cache:

```
Job A: input_len=336, output_len=168 → hash: a1b2c3d4e5f67890
Job B: input_len=336, output_len=168 → hash: a1b2c3d4e5f67890 (same, reuses cache)
Job C: input_len=720, output_len=168 → hash: f9e8d7c6b5a43210 (different, new cache)
```

## Troubleshooting

### "Tensor cache requested but invalid"

```
FileNotFoundError: Tensor cache requested but invalid: Cache directory does not exist
Cache directory: data/fidel-ts/germany_renewable/tensor_cache/a1b2c3d4e5f67890

Generate the cache with:
  python -m cli.tensor_cache generate <config.yaml>

Or disable tensor cache by setting:
  training.use_tensor_cache: false
```

**Solution**: Generate the cache before running training.

### "Config hash mismatch"

```
Cache is invalid: Config hash mismatch (cache may be stale).
Expected: a1b2c3d4e5f67890, Got: f9e8d7c6b5a43210
```

**Cause**: The experiment configuration changed since the cache was generated.

**Solution**: Regenerate the cache with `--force`:
```bash
python -m cli.tensor_cache generate config.yaml --force
```

### "return_type='set' is not supported"

```
ValueError: return_type='set' is not supported with tensor cache.
Use 'loader' instead, or disable tensor cache.
```

**Cause**: Some code is requesting raw Dataset objects, which tensor cache doesn't support.

**Solution**: Use `return_type='loader'` or disable tensor cache for that experiment.

### Large Disk Usage

Tensor caches can be large (several GB per dataset). To clean up:

```bash
# Remove all caches for a dataset
rm -rf data/fidel-ts/germany_renewable/tensor_cache/

# Remove specific cache
rm -rf data/fidel-ts/germany_renewable/tensor_cache/a1b2c3d4e5f67890/
```

## Performance Expectations

### Before Tensor Cache

| Metric | Value |
|--------|-------|
| Per-sample `__getitem__` time | ~18.6 ms |
| Effective throughput | ~110 samples/sec |
| GPU utilization | 20-40% (data-starved) |
| Training time (30 epochs) | >24 hours |

### After Tensor Cache

| Metric | Value |
|--------|-------|
| Per-sample `__getitem__` time | ~0.01 ms |
| Effective throughput | 10,000-50,000 samples/sec |
| GPU utilization | 80-95% |
| Training time (30 epochs) | 2-4 hours |

**Speedup: 100-1000x for data loading, 6-12x for total training time**

## Limitations

1. **No raw Dataset access**: Tensor cache only returns DataLoaders, not Dataset objects
2. **Static snapshot**: Cache captures data at generation time; no runtime augmentation
3. **Disk space**: Large datasets require significant storage
4. **Regeneration required**: Any config change affecting the hash requires regeneration

## Implementation Details

### Key Files

| File | Purpose |
|------|---------|
| `data_provider/tensor_cache.py` | Core implementation (Generator, Dataset, validation) |
| `cli/tensor_cache.py` | CLI commands (generate, validate, info, benchmark) |
| `data_provider/data_factory.py` | Integration with Data_Provider |
| `cli/config/models.py` | Config fields (use_tensor_cache, tensor_cache_dir) |
| `runs/pytorch.py` | Training loop integration |

### Memory-Mapped Arrays

The cache uses numpy memory-mapped arrays (`np.memmap`) which:
- Load data on-demand from disk
- Share memory across worker processes
- Avoid loading entire dataset into RAM
- Provide O(1) random access

## References

- [Profiling Results](../context/performance_optimization/profiling_germany_renewable.json)
- [Training Optimization Plan](../context/performance_optimization/training_optimization_plan.md)
- [Test Plan](../context/performance_optimization/tensor_cache_test_plan.md)
