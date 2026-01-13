# Training Pipeline Optimization Plan

## Executive Summary

**Problem**: Training on RTX 5000 Ada with 6 vCPU (AMD EPYC 7272) shows:
- CPU utilization: ~20% constant
- GPU utilization: Sparse, infrequent spurts
- Training time: >1 day for a single experiment

**Root Cause Hypothesis**: The dataloader is CPU-bound, unable to feed the GPU fast enough. The GPU starves waiting for data.

**Goal**: Achieve 10-100x speedup by amortizing CPU-bound dataloader operations into a preprocessing step, creating GPU-ready tensors that can be loaded with minimal CPU overhead.

**Note**: DataLoader configuration options (`num_workers` and `prefetch_factor`) already exist in the training config (`cli/config/models.py:65-66`), allowing adjustment to available compute resources.

---

## Part 1: Bottleneck Identification Plan

### 1.1 Profiling Strategy

#### A. PyTorch DataLoader Profiling

```python
# Add to exp/exp_universal.py or create new profiling script
import torch.profiler as profiler
import time

def profile_dataloader(data_loader, num_batches=100):
    """Profile time spent in data loading vs GPU operations."""

    data_times = []
    gpu_times = []

    for i, batch in enumerate(data_loader):
        if i >= num_batches:
            break

        # Time GPU transfer
        start_gpu = time.perf_counter()
        batch_gpu = [t.cuda() if isinstance(t, torch.Tensor) else t for t in batch]
        torch.cuda.synchronize()
        gpu_times.append(time.perf_counter() - start_gpu)

    # Data loading time is implicitly the time between loop iterations
    # Use iter() to measure explicitly

    iterator = iter(data_loader)
    for i in range(num_batches):
        start_data = time.perf_counter()
        batch = next(iterator)
        data_times.append(time.perf_counter() - start_data)

    print(f"Avg data loading time: {np.mean(data_times)*1000:.2f} ms")
    print(f"Avg GPU transfer time: {np.mean(gpu_times)*1000:.2f} ms")
    print(f"Data loading accounts for: {np.sum(data_times)/(np.sum(data_times)+np.sum(gpu_times))*100:.1f}%")
```

#### B. Per-Operation Profiling in `__getitem__()`

Add timing instrumentation to `data_provider/data_loader.py`:

```python
# In Universal_Dataset.__getitem__() and Heterogeneous_Dataset.__getitem__()
import time

# Class-level accumulators
_profile_data = {
    'slice_time': [],      # Time series slicing
    'timestamp_time': [],  # Timestamp extraction
    'hetero_match_time': [],  # Temporal matching
    'hetero_lookup_time': [], # Embedding lookup
    'hetero_concat_time': [], # Concatenation
    'time_features_time': [], # Time feature generation
    'total_time': []
}

def __getitem__(self, index):
    total_start = time.perf_counter()

    # Slicing
    start = time.perf_counter()
    s_begin, s_end = index, index + self.seq_len
    r_begin, r_end = s_end, s_end + self.pred_len
    seq_x = self.data[s_begin:s_end]
    seq_y = self.data[r_begin:r_end]
    self._profile_data['slice_time'].append(time.perf_counter() - start)

    # ... continue for each operation
```

#### C. CPU Profiler (cProfile/py-spy)

```bash
# Profile the entire training run
py-spy record -o profile.svg --pid <PID>

# Or use cProfile
python -m cProfile -o profile.stats -m cli.suite run configs/experiment_suites/lynx_film/canada_photovoltaics.yaml
```

#### D. NVIDIA Nsight Systems

```bash
# Profile GPU utilization and data transfer patterns
nsys profile -o training_profile python -m cli.suite run configs/experiment_suites/lynx_film/canada_photovoltaics.yaml
```

### 1.2 Key Metrics to Capture

| Metric | Tool | Expected Bottleneck |
|--------|------|---------------------|
| DataLoader worker utilization | Custom timing | Low if num_workers insufficient |
| GPU idle time between batches | nsys | High if CPU-bound |
| Per-operation time in `__getitem__` | Custom timing | Hetero matching/lookup |
| Memory copy time (CPU→GPU) | torch.profiler | Should be minimal with pin_memory |
| Worker process overhead | py-spy | Fork/IPC overhead |

### 1.3 Profiling Commands (IMPLEMENTED)

The profiling infrastructure has been implemented. Use these commands to identify bottlenecks:

```bash
# Basic profiling - profiles __getitem__ and DataLoader throughput
python scripts/profile_dataloader.py configs/experiment_suites/lynx_film/canada_photovoltaics.yaml

# Profile with more samples for accurate statistics
python scripts/profile_dataloader.py configs/experiment_suites/lynx_film/canada_photovoltaics.yaml \
    --num-samples 10000 \
    --num-batches 200

# Save results to JSON for analysis
python scripts/profile_dataloader.py configs/experiment_suites/lynx_film/canada_photovoltaics.yaml \
    --output results/profiling/canada_photovoltaics_profile.json

# Profile worker scaling (tests num_workers=0,1,2,4,8)
python scripts/profile_dataloader.py configs/experiment_suites/lynx_film/canada_photovoltaics.yaml \
    --worker-scaling \
    --max-workers 6

# Skip __getitem__ profiling, only measure DataLoader throughput
python scripts/profile_dataloader.py configs/experiment_suites/lynx_film/canada_photovoltaics.yaml \
    --skip-getitem

# Profile specific experiment from suite
python scripts/profile_dataloader.py configs/experiment_suites/lynx_film/canada_photovoltaics.yaml \
    --experiment "lynx_film_canada_photovoltaics"
```

**Profiled Operations:**

The profiler measures the following operations in `__getitem__`:
- `ts_array_slicing` - Time series array slicing (seq_x, seq_y, timestamps)
- `sample_id_generation` - Generating deterministic sample IDs
- `hetero_data_getter_x` - Fetching heterogeneous data for input sequence
- `hetero_data_getter_y` - Fetching heterogeneous data for output sequence
- `time_features_generation` - Generating time features (if enabled)
- `llm_embedding_lookup` - LLM embedding lookup (if using TimeCMA-style models)
- `preloaded_hetero_slicing` - Slicing preloaded hetero data (if preload_hetero=True)

Within `get_hetero_data`:
- `hetero_time_matching` - Temporal matching (pd.searchsorted)
- `hetero_downtime_check` - Checking downtime ranges
- `hetero_embedding_lookup` - Dictionary lookups + shape normalization
- `hetero_array_construction` - np.array and np.concatenate operations

**Expected Output:**

```
================================================================================
DATALOADER PROFILING SUMMARY
================================================================================
Total samples processed: 5,000
Total elapsed time: 12.34 seconds
Avg throughput: 405.2 samples/sec

Operation                           Count       Total (ms)     %   Mean (μs)     P95 (μs)
-----------------------------------------------------------------------------------------------
hetero_time_matching                 5000          4523.1  45.2%       904.6      1234.5
hetero_embedding_lookup              5000          2845.2  28.4%       569.0       789.2
hetero_array_construction            5000          1523.4  15.2%       304.7       423.1
...
-----------------------------------------------------------------------------------------------
TOTAL                                            10012.3

Per-sample overhead: 2002.5 μs

TOP BOTTLENECKS (by total time):
  1. hetero_time_matching: 4523.1ms (45.2%) - 904.6μs/call
  2. hetero_embedding_lookup: 2845.2ms (28.4%) - 569.0μs/call
  ...
================================================================================
```

---

## Part 2: Dataloader Operations Mapping

### 2.1 Operation Categories

#### A. One-Time Operations (Per Dataset Init)

| Operation | Location | Est. Time | Notes |
|-----------|----------|-----------|-------|
| CSV/Parquet loading | `data_loader.py:179-186` | 0.5-5s per file | I/O bound, uses data_buffer cache |
| Timestamp conversion | `data_loader.py:189-198` | 50-200ms | Pandas datetime → int64 |
| Missing value handling | `data_loader.py:200-208` | 100-500ms | Creates indicator columns |
| Data splitting | `data_helper.py:83-185` | 10-50ms | Array slicing |
| StandardScaler fit | `data_loader.py:268-272` | 50-100ms | Computes mean/std |
| StandardScaler transform | `data_loader.py:269` | 20-100ms | Applies normalization |
| Embedding loading | `data_loader.py:569-628` | 1-10s | Large pickle/cache files |
| Embedding index building | `data_loader.py:631-660` | 100-500ms | Creates timestamp→embedding map |

**Total one-time cost**: 2-20 seconds per entity (acceptable)

#### B. Per-Sample Operations (Critical Path - `__getitem__()`)

| Operation | Location | Est. Time per Sample | Samples/Epoch | Total per Epoch |
|-----------|----------|---------------------|---------------|-----------------|
| Array slicing | `data_loader.py:295-310` | 1-5 μs | 100k-1M | 0.1-5s |
| Timestamp extraction | `data_loader.py:312-318` | 2-10 μs | 100k-1M | 0.2-10s |
| **Temporal matching** | `data_loader.py:773-799` | **50-200 μs** | 100k-1M | **5-200s** |
| **Embedding lookup** | `data_loader.py:812-839` | **10-50 μs** | 100k-1M | **1-50s** |
| **Shape normalization** | `data_loader.py:841-870` | **5-20 μs** | 100k-1M | **0.5-20s** |
| **Downtime checking** | `data_loader.py:801-810` | **20-100 μs** | 100k-1M | **2-100s** |
| Concatenation | `data_loader.py:908-924` | 5-20 μs | 100k-1M | 0.5-20s |
| Time feature gen | `data_loader.py:388-394` | 10-50 μs | 100k-1M | 1-50s |

**Total per-sample overhead**: 100-450 μs → **10-450 seconds per epoch**

#### C. Per-Batch Operations (Collation)

| Operation | Location | Est. Time per Batch | Batches/Epoch | Total per Epoch |
|-----------|----------|---------------------|---------------|-----------------|
| default_collate | PyTorch | 0.5-2 ms | 130-1300 | 0.065-2.6s |
| Tensor stacking | PyTorch | 0.2-1 ms | 130-1300 | 0.026-1.3s |
| Pin memory copy | PyTorch | 0.1-0.5 ms | 130-1300 | 0.013-0.65s |

**Total collation overhead**: ~1-5 seconds per epoch (acceptable)

### 2.2 Critical Bottlenecks Identified

```
BOTTLENECK RANK:
┌────────────────────────────────────────────────────────────────┐
│ 1. Temporal Matching (pd.searchsorted per sample)              │
│    - Called in hetero_data_getter() for EVERY sample           │
│    - 50-200 μs × 100k-1M samples = 5-200 seconds/epoch         │
├────────────────────────────────────────────────────────────────┤
│ 2. Embedding Lookup (dictionary access + numpy operations)     │
│    - Dict lookup + shape validation per sample                 │
│    - 10-50 μs × 100k-1M samples = 1-50 seconds/epoch           │
├────────────────────────────────────────────────────────────────┤
│ 3. Downtime Checking (boolean array operations)                │
│    - Range comparison per timestamp per sample                 │
│    - 20-100 μs × 100k-1M samples = 2-100 seconds/epoch         │
├────────────────────────────────────────────────────────────────┤
│ 4. Shape Normalization (numpy reshape/concatenate)             │
│    - Memory allocation per sample                              │
│    - 5-20 μs × 100k-1M samples = 0.5-20 seconds/epoch          │
└────────────────────────────────────────────────────────────────┘
```

### 2.3 DataLoader Configuration Analysis

Current config (`data_factory.py:843-895`):
```python
DataLoader(
    dataset,
    batch_size=768,       # Large batch - good
    shuffle=True,         # Required for training
    num_workers=4,        # May be insufficient
    pin_memory=True,      # Good for GPU transfer
    persistent_workers=True,  # Good - avoids fork overhead
    prefetch_factor=2     # Prefetches 2×num_workers batches
)
```

**Issues**:
1. `num_workers=4` with 6 vCPU leaves headroom but workers are blocked on CPU operations
2. Prefetch can't help if each `__getitem__` is CPU-bound
3. `shuffle=True` prevents sequential memory access patterns

---

## Part 3: Non-Dataloader CPU-Bound Operations

### 3.1 Model Forward Pass

| Operation | Location | Est. Time | Frequency | Notes |
|-----------|----------|-----------|-----------|-------|
| Tensor device transfer | `exp_universal.py:160-175` | 0.1-1 ms/batch | Every batch | Already optimized with pin_memory |
| Time mark generation | `exp_universal.py:180-190` | 0.5-2 ms/batch | Every batch | Only for specific models |
| Loss computation | `exp_universal.py:200-210` | 0.01-0.1 ms/batch | Every batch | Minimal |

### 3.2 Training Loop Overhead

| Operation | Location | Est. Time | Frequency | Notes |
|-----------|----------|-----------|-----------|-------|
| Gradient accumulation | `exp_universal.py:240-260` | N/A | If enabled | Currently not a bottleneck |
| Optimizer step | PyTorch | 0.5-5 ms/batch | Every batch | GPU-bound, fine |
| Learning rate scheduling | `exp_universal.py:280` | <0.01 ms | Every batch | Negligible |
| Logging/metrics | `exp_universal.py:290-310` | 0.1-1 ms/batch | Every batch | Minimal |
| Checkpoint saving | `exp_universal.py:320-350` | 100-500 ms | Every N epochs | Infrequent |

### 3.3 Validation/Testing

| Operation | Location | Est. Time | Frequency | Notes |
|-----------|----------|-----------|-----------|-------|
| Inverse scaling | `exp_universal.py:400-420` | 5-50 ms/batch | Val/Test only | Not training bottleneck |
| Metric computation | `exp_universal.py:430-450` | 10-100 ms/epoch | Val/Test only | Not training bottleneck |

### 3.4 Memory Management

| Operation | Est. Time | Notes |
|-----------|-----------|-------|
| Python GC | Variable | Can cause jitter if large objects created per sample |
| Numpy array allocation | ~1-10 μs/array | Significant when multiplied by samples |
| Dict operations | ~0.1-1 μs/lookup | Fast but adds up |

---

## Part 4: Second Amortization Step - Detailed Plan

### 4.1 Concept Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        CURRENT PIPELINE (CPU-BOUND)                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Raw CSV  ──► Data Loading ──► Per-Sample Processing ──► GPU            │
│      │              │                    │                 │             │
│      │              │         ┌──────────┴──────────┐      │             │
│      │              │         │ Temporal Matching   │      │             │
│    [Disk]        [CPU]       │ Embedding Lookup    │   [Fast]            │
│                              │ Shape Normalization │                     │
│                              │ Downtime Checking   │                     │
│                              └─────────────────────┘                     │
│                                   BOTTLENECK!                            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                    PROPOSED PIPELINE (GPU-OPTIMIZED)                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   STEP 1: EMBEDDING GENERATION (Already Amortized)                       │
│   ─────────────────────────────────────────────────                      │
│   Raw Text ──► Text Embeddings ──► Cache (.pkl/.npy)                     │
│       │              │                   │                               │
│    [Disk]         [CPU/GPU]           [Disk]                             │
│                                                                          │
│   STEP 2: SAMPLE PREPROCESSING (NEW - This Plan)                         │
│   ────────────────────────────────────────────────                       │
│   Raw CSV + Embeddings ──► Pre-built Samples ──► TensorCache             │
│            │                      │                   │                  │
│         [CPU]              [CPU - Slurm Job]       [Disk]                │
│                                                                          │
│   All temporal matching, embedding lookup, normalization                 │
│   done ONCE in preprocessing                                             │
│                                                                          │
│   STEP 3: TRAINING (Trivial Data Loading)                                │
│   ───────────────────────────────────────                                │
│   TensorCache ──► Memory-Mapped Arrays ──► GPU                           │
│        │                  │                 │                            │
│     [Disk]         [Minimal CPU]        [Fast!]                          │
│                                                                          │
│   __getitem__: Just array indexing + memcpy                              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.2 TensorCache Format Design

#### A. Directory Structure

```
experiment_dir/
├── checkpoints/
├── logs/
└── tensor_cache/
    ├── metadata.json           # Cache configuration & validation
    ├── train/
    │   ├── seq_x.npy          # Memory-mapped: (N_train, seq_len, n_features)
    │   ├── seq_y.npy          # Memory-mapped: (N_train, pred_len, n_features)
    │   ├── hetero_x.npy       # Memory-mapped: (N_train, hetero_seq_len, embed_dim)
    │   ├── hetero_y.npy       # Memory-mapped: (N_train, hetero_pred_len, embed_dim)
    │   ├── hetero_general.npy # Memory-mapped: (N_train, 1, embed_dim)
    │   ├── hetero_channel.npy # Memory-mapped: (N_train, n_channels, embed_dim)
    │   ├── x_time.npy         # Memory-mapped: (N_train, seq_len)
    │   ├── y_time.npy         # Memory-mapped: (N_train, pred_len)
    │   ├── sample_ids.npy     # Memory-mapped: (N_train,)
    │   └── entity_indices.npy # Memory-mapped: (N_train,) - for entity-level metrics
    ├── val/
    │   └── ... (same structure)
    └── test/
        └── ... (same structure)
```

#### B. Metadata Schema

```json
{
    "version": "1.0.0",
    "created_at": "2024-01-15T10:30:00Z",
    "config_hash": "sha256:abc123...",
    "data_config": {
        "dataset_name": "Canada_photovoltaics_plants",
        "input_len": 360,
        "output_len": 168,
        "stride": 1,
        "hetero_stride": 12,
        "hetero_type": "all_for_one",
        "normalize": true,
        "truncate_train_for_purge": true
    },
    "shapes": {
        "train": {
            "seq_x": [523456, 360, 12],
            "seq_y": [523456, 168, 12],
            "hetero_x": [523456, 30, 768],
            "hetero_y": [523456, 14, 768]
        },
        "val": { ... },
        "test": { ... }
    },
    "dtypes": {
        "seq_x": "float32",
        "seq_y": "float32",
        "hetero_x": "float32",
        "hetero_y": "float32",
        "x_time": "int64",
        "y_time": "int64"
    },
    "scaler_params": {
        "mean": [...],
        "std": [...]
    },
    "entity_info": {
        "entity_ids": ["plant_001", "plant_002", ...],
        "samples_per_entity": {"plant_001": 12345, ...}
    }
}
```

### 4.3 Cache Generation Pipeline

#### A. New CLI Command

```bash
# Generate tensor cache for an experiment
python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/canada_photovoltaics.yaml

# Options:
# --output-dir: Override cache location
# --num-workers: Parallel processing workers (for cache generation, not dataloader)
# --chunk-size: Samples per chunk (memory management)
# --force: Overwrite existing cache
```

**Note**: The `--num-workers` option here controls parallel processing during cache generation (multiprocessing for sample preprocessing), which is distinct from the DataLoader `num_workers` config that controls parallel data loading during training.

#### B. Cache Generator Module

Create `data_provider/tensor_cache.py` and `cli/tensor_cache.py`:

```python
"""
Tensor Cache Generator for Amortized Data Loading

This module pre-computes all CPU-intensive dataloader operations:
- Temporal slicing
- Embedding temporal matching
- Shape normalization
- Time feature generation

The output is memory-mapped numpy arrays that can be loaded with
near-zero CPU overhead during training.
"""

import numpy as np
import json
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import multiprocessing as mp

class TensorCacheGenerator:
    """Generates pre-computed tensor cache for fast training data loading."""

    def __init__(
        self,
        data_provider: 'Data_Provider',
        cache_dir: Path,
        config_hash: str,
        num_workers: int = 4,
        chunk_size: int = 10000
    ):
        self.data_provider = data_provider
        self.cache_dir = Path(cache_dir)
        self.config_hash = config_hash
        self.num_workers = num_workers
        self.chunk_size = chunk_size

    def generate(self, flags: list = ['train', 'val', 'test']):
        """Generate cache for specified data splits."""

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        metadata = self._init_metadata()

        for flag in flags:
            print(f"Generating cache for {flag} split...")
            self._generate_split(flag, metadata)

        # Save metadata
        with open(self.cache_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Cache generated at: {self.cache_dir}")

    def _generate_split(self, flag: str, metadata: dict):
        """Generate cache for a single split."""

        split_dir = self.cache_dir / flag
        split_dir.mkdir(exist_ok=True)

        # Get datasets (without DataLoader overhead)
        datasets = self.data_provider.get_datasets(flag)

        # Calculate total samples
        total_samples = sum(len(ds) for ds in datasets.values())

        # Pre-allocate memory-mapped arrays
        sample_shape = self._get_sample_shape(datasets)
        arrays = self._create_mmap_arrays(split_dir, total_samples, sample_shape)

        # Process in chunks with progress bar
        current_idx = 0
        for entity_id, dataset in tqdm(datasets.items(), desc=f"Entities ({flag})"):
            entity_samples = len(dataset)

            # Process entity in chunks
            for chunk_start in range(0, entity_samples, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, entity_samples)
                chunk_indices = range(chunk_start, chunk_end)

                # Parallel sample processing
                chunk_data = self._process_chunk(dataset, chunk_indices)

                # Write to memory-mapped arrays
                write_start = current_idx + chunk_start
                write_end = current_idx + chunk_end
                self._write_chunk(arrays, chunk_data, write_start, write_end)

            current_idx += entity_samples

        # Flush to disk
        for arr in arrays.values():
            arr.flush()

        metadata['shapes'][flag] = {k: list(v.shape) for k, v in arrays.items()}

    def _process_chunk(self, dataset, indices) -> Dict[str, np.ndarray]:
        """Process a chunk of samples, optionally in parallel."""

        if self.num_workers > 1:
            with mp.Pool(self.num_workers) as pool:
                samples = pool.map(dataset.__getitem__, indices)
        else:
            samples = [dataset[i] for i in indices]

        # Collate into arrays
        return self._collate_samples(samples)

    def _collate_samples(self, samples) -> Dict[str, np.ndarray]:
        """Collate list of samples into numpy arrays."""

        # Unpack sample tuples
        # (sample_id, seq_x, seq_y, x_time, y_time, x_hetero, y_hetero,
        #  hetero_x_time, hetero_y_time, hetero_general, hetero_channel,
        #  x_time_features, y_time_features)

        return {
            'sample_ids': np.array([s[0] for s in samples]),
            'seq_x': np.stack([s[1] for s in samples]),
            'seq_y': np.stack([s[2] for s in samples]),
            'x_time': np.stack([s[3] for s in samples]),
            'y_time': np.stack([s[4] for s in samples]),
            'hetero_x': np.stack([s[5] for s in samples]) if samples[0][5] is not None else None,
            'hetero_y': np.stack([s[6] for s in samples]) if samples[0][6] is not None else None,
            'hetero_general': np.stack([s[9] for s in samples]) if samples[0][9] is not None else None,
            'hetero_channel': np.stack([s[10] for s in samples]) if samples[0][10] is not None else None,
            # Time features optional
        }

    def _create_mmap_arrays(
        self,
        split_dir: Path,
        n_samples: int,
        shapes: dict
    ) -> Dict[str, np.memmap]:
        """Create memory-mapped arrays for cache storage."""

        arrays = {}
        for name, (sample_shape, dtype) in shapes.items():
            full_shape = (n_samples,) + tuple(sample_shape)
            filepath = split_dir / f"{name}.npy"
            arrays[name] = np.memmap(
                filepath,
                dtype=dtype,
                mode='w+',
                shape=full_shape
            )
        return arrays


class TensorCacheDataset(torch.utils.data.Dataset):
    """
    Ultra-fast dataset that loads from pre-computed tensor cache.

    __getitem__ is just array indexing - no computation.
    """

    def __init__(self, cache_dir: Path, flag: str = 'train'):
        self.cache_dir = Path(cache_dir) / flag
        self.metadata = self._load_metadata()

        # Memory-map all arrays (lazy loading)
        self.arrays = self._load_mmap_arrays()
        self.n_samples = self.arrays['seq_x'].shape[0]

    def _load_mmap_arrays(self) -> Dict[str, np.memmap]:
        """Load memory-mapped arrays."""

        arrays = {}
        for name in ['seq_x', 'seq_y', 'x_time', 'y_time',
                     'hetero_x', 'hetero_y', 'hetero_general',
                     'hetero_channel', 'sample_ids']:
            filepath = self.cache_dir / f"{name}.npy"
            if filepath.exists():
                arrays[name] = np.memmap(filepath, mode='r')
        return arrays

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        """
        Ultra-fast sample retrieval - just array indexing.

        Time complexity: O(1) with memory-mapped I/O
        No computation, no dict lookups, no temporal matching.
        """

        return (
            self.arrays['sample_ids'][index],
            self.arrays['seq_x'][index],
            self.arrays['seq_y'][index],
            self.arrays['x_time'][index],
            self.arrays['y_time'][index],
            self.arrays.get('hetero_x', [None])[index] if 'hetero_x' in self.arrays else None,
            self.arrays.get('hetero_y', [None])[index] if 'hetero_y' in self.arrays else None,
            None,  # hetero_x_time (not needed)
            None,  # hetero_y_time (not needed)
            self.arrays.get('hetero_general', [None])[index] if 'hetero_general' in self.arrays else None,
            self.arrays.get('hetero_channel', [None])[index] if 'hetero_channel' in self.arrays else None,
            None,  # x_time_features (pre-computed if needed)
            None,  # y_time_features
        )
```

### 4.4 Integration with Training Pipeline

#### A. Config Flag

Add to `cli/config/models.py` TrainingConfig class:
```python
# In TrainingConfig (around line 99)
use_tensor_cache: bool = Field(
    default=False,
    description="Use pre-generated tensor cache for fast data loading"
)
tensor_cache_dir: Optional[str] = Field(
    default=None,
    description="Path to tensor cache directory (auto-generated in experiment_dir if null)"
)
```

Then in experiment config:
```yaml
training:
  use_tensor_cache: true
  tensor_cache_dir: null  # Auto-generated if null
  num_workers: 8          # Increase for fast cache (default: 0)
  prefetch_factor: 8      # Increase for fast cache (default: 2)
  # If cache doesn't exist, falls back to regular dataloader
```

#### B. Data_Provider Modification

```python
# In data_factory.py

class Data_Provider:
    def __init__(self, ...):
        # ... existing init ...

        # Check for tensor cache
        self.tensor_cache_dir = self._resolve_cache_dir()
        self.use_tensor_cache = (
            self.args.use_tensor_cache and
            self._validate_cache()
        )

    def _resolve_cache_dir(self) -> Optional[Path]:
        """Resolve tensor cache directory."""

        if self.args.tensor_cache_dir:
            return Path(self.args.tensor_cache_dir)

        # Default: experiment_dir/tensor_cache
        if hasattr(self.args, 'experiment_dir'):
            return Path(self.args.experiment_dir) / 'tensor_cache'

        return None

    def _validate_cache(self) -> bool:
        """Check if cache exists and matches current config."""

        if not self.tensor_cache_dir or not self.tensor_cache_dir.exists():
            return False

        metadata_path = self.tensor_cache_dir / 'metadata.json'
        if not metadata_path.exists():
            return False

        with open(metadata_path) as f:
            metadata = json.load(f)

        # Validate config hash
        current_hash = self._compute_config_hash()
        if metadata.get('config_hash') != current_hash:
            print(f"Warning: Cache config mismatch. Regenerate with cli.cache")
            return False

        return True

    def get_dataloader(self, flag: str, ...):
        """Get dataloader, using tensor cache if available."""

        if self.use_tensor_cache:
            dataset = TensorCacheDataset(self.tensor_cache_dir, flag)
            return DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=(flag == 'train'),
                num_workers=self.args.num_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4  # Can increase with fast __getitem__
            )
        else:
            # Existing dataloader logic
            return self._get_regular_dataloader(flag, ...)
```

### 4.5 Slurm Job for Cache Generation

Create `scripts/slurm/generate_tensor_cache.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=tensor_cache
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/cache_%j.out
#SBATCH --error=logs/cache_%j.err

# Load environment
source ~/.bashrc
conda activate fidel-ts

# Generate cache
CONFIG_FILE=${1:-"configs/experiment_suites/lynx_film/canada_photovoltaics.yaml"}
OUTPUT_DIR=${2:-""}

# Note: --num-workers here is for cache generation parallelism (multiprocessing)
# This is separate from the DataLoader num_workers in the training config
python -m cli.tensor_cache generate "$CONFIG_FILE" \
    --num-workers 32 \
    --chunk-size 50000 \
    ${OUTPUT_DIR:+--output-dir "$OUTPUT_DIR"}

echo "Cache generation complete!"
```

**Usage**:
```bash
# Submit cache generation job
sbatch scripts/slurm/generate_tensor_cache.sh configs/experiment_suites/lynx_film/canada_photovoltaics.yaml

# Then run training with the cache
python -m cli.suite run configs/experiment_suites/lynx_film/canada_photovoltaics.yaml
```

### 4.6 Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE OPTIMIZED PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: EMBEDDING GENERATION (Existing - Run Once per Dataset)             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Raw Text Data                                                             │
│        │                                                                    │
│        ▼                                                                    │
│   ┌──────────────────┐                                                      │
│   │ Text Embedder    │  ← Can run on GPU or CPU                             │
│   │ (BERT/GPT-2)     │                                                      │
│   └────────┬─────────┘                                                      │
│            │                                                                │
│            ▼                                                                │
│   ┌──────────────────┐                                                      │
│   │ Embedding Cache  │  → embeddings/{dataset}_{model}_{hash}.pkl           │
│   │ (Per-Dataset)    │                                                      │
│   └──────────────────┘                                                      │
│                                                                             │
│   Time: 10-60 minutes per dataset (one-time cost)                           │
│   Storage: 100MB - 2GB per dataset                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: TENSOR CACHE GENERATION (NEW - Run Once per Experiment Config)     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│   │ Raw CSV/     │    │ Embedding    │    │ Experiment   │                  │
│   │ Parquet Data │    │ Cache        │    │ Config       │                  │
│   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                  │
│          │                   │                   │                          │
│          └───────────────────┼───────────────────┘                          │
│                              │                                              │
│                              ▼                                              │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │              TensorCacheGenerator (CPU-Only Slurm Job)            │     │
│   ├───────────────────────────────────────────────────────────────────┤     │
│   │                                                                   │     │
│   │   For each entity_id:                                             │     │
│   │     1. Load CSV/Parquet data                                      │     │
│   │     2. Apply normalization (fit scaler on train)                  │     │
│   │     3. Split into train/val/test                                  │     │
│   │                                                                   │     │
│   │   For each sample index i:                                        │     │
│   │     4. Slice: seq_x[i:i+seq_len], seq_y[i+seq_len:...]            │     │
│   │     5. Extract timestamps: x_time, y_time                         │     │
│   │     6. Temporal match: find nearest embeddings                    │     │
│   │     7. Load embeddings from cache                                 │     │
│   │     8. Normalize shapes: (1, embed_dim)                           │     │
│   │     9. Check downtime: add indicators                             │     │
│   │    10. Generate time features (if needed)                         │     │
│   │                                                                   │     │
│   │   Write to memory-mapped numpy arrays                             │     │
│   │                                                                   │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │                    Tensor Cache Directory                          │     │
│   ├───────────────────────────────────────────────────────────────────┤     │
│   │   experiment_dir/tensor_cache/                                    │     │
│   │   ├── metadata.json      (config hash, shapes, dtypes)            │     │
│   │   ├── train/                                                      │     │
│   │   │   ├── seq_x.npy      (N_train, seq_len, n_features)           │     │
│   │   │   ├── seq_y.npy      (N_train, pred_len, n_features)          │     │
│   │   │   ├── hetero_x.npy   (N_train, hetero_len, embed_dim)         │     │
│   │   │   └── ...                                                     │     │
│   │   ├── val/                                                        │     │
│   │   └── test/                                                       │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│   Time: 5-30 minutes (Slurm job, 32 CPU cores)                              │
│   Storage: 5-50 GB per experiment config                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: TRAINING (GPU Job - Optimized Data Loading)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │                    TensorCacheDataset                              │     │
│   ├───────────────────────────────────────────────────────────────────┤     │
│   │                                                                   │     │
│   │   __getitem__(index):                                             │     │
│   │       return (                                                    │     │
│   │           self.seq_x[index],      # Direct array access           │     │
│   │           self.seq_y[index],      # O(1) operation                │     │
│   │           self.hetero_x[index],   # No computation                │     │
│   │           ...                                                     │     │
│   │       )                                                           │     │
│   │                                                                   │     │
│   │   Time per sample: <1 μs (vs 100-450 μs before)                   │     │
│   │   Speedup: 100-450x per sample                                    │     │
│   │                                                                   │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │                      PyTorch DataLoader                            │     │
│   ├───────────────────────────────────────────────────────────────────┤     │
│   │                                                                   │     │
│   │   - num_workers: 4-8 (now effective, not CPU-bound)               │     │
│   │   - prefetch_factor: 4-8 (workers can prefetch many batches)      │     │
│   │   - pin_memory: True (fast CPU→GPU transfer)                      │     │
│   │   - persistent_workers: True (no fork overhead)                   │     │
│   │                                                                   │     │
│   │   Batch prep time: <10ms (vs 50-200ms before)                     │     │
│   │                                                                   │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │                         GPU Training                               │     │
│   ├───────────────────────────────────────────────────────────────────┤     │
│   │                                                                   │     │
│   │   GPU utilization: 80-95% (vs sparse spurts before)               │     │
│   │   Training time: Hours (vs >1 day before)                         │     │
│   │                                                                   │     │
│   │   The GPU is now continuously fed with data!                      │     │
│   │                                                                   │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.7 Modern Best Practices for GPU Feeding

#### A. Memory-Mapped Files vs. Pre-loaded Tensors

| Approach | Pros | Cons | When to Use |
|----------|------|------|-------------|
| **Memory-Mapped (np.memmap)** | Low RAM usage, OS manages caching | Slightly slower first access | Large datasets (>RAM) |
| **Pre-loaded to RAM** | Fastest access | High RAM usage | Small-medium datasets |
| **Pre-loaded to GPU** | Zero transfer time | Limited by VRAM | Very small datasets |

**Recommendation**: Use memory-mapped for flexibility, with option to pre-load to RAM if dataset fits.

#### B. DataLoader Configuration for Fast Cache

```python
# Optimized DataLoader for TensorCacheDataset
DataLoader(
    tensor_cache_dataset,
    batch_size=768,           # Match original
    shuffle=True,             # Required for training
    num_workers=4,            # Now effective!
    pin_memory=True,          # Fast GPU transfer
    persistent_workers=True,  # No fork overhead
    prefetch_factor=8,        # Increase - workers are fast now
    drop_last=True            # Consistent batch sizes
)
```

**Existing Configuration Support**: The codebase already supports tuning these parameters via the experiment config:

```yaml
# configs/experiment_suites/lynx_film/canada_photovoltaics.yaml
training:
  num_workers: 8            # Adjust to available vCPUs (default: 0)
  prefetch_factor: 8        # Increase with fast cache (default: 2)
  batch_size: 768
  # ... other settings
```

These are defined in `cli/config/models.py:65-66` and automatically passed to the DataLoader via `data_factory.py:876,879,889,892`. With the tensor cache making `__getitem__` trivial, increasing both parameters will significantly improve GPU feeding rate.

#### C. NVIDIA DALI Alternative (Advanced)

For maximum performance, consider NVIDIA DALI:
```python
from nvidia.dali.plugin.pytorch import DALIGenericIterator

# DALI can load directly from cache files with GPU decoding
# Eliminates Python GIL bottleneck entirely
```

#### D. PyTorch 2.0+ Optimizations

```python
# Enable with TensorCacheDataset
with torch.amp.autocast('cuda'):  # Mixed precision
    output = model(batch)

# torch.compile for model
model = torch.compile(model, mode='reduce-overhead')

# Tensor cores utilization
torch.set_float32_matmul_precision('medium')  # TF32 on Ampere+
```

### 4.8 Cache Invalidation Strategy

```python
def compute_config_hash(config: dict) -> str:
    """
    Compute hash of config parameters that affect cache validity.

    Cache must be regenerated if ANY of these change:
    - input_len / output_len
    - hetero_stride
    - normalize settings
    - split settings
    - truncate_train_for_purge
    - Entity filtering settings
    - Missing value handling strategy
    """

    relevant_keys = [
        'input_len', 'output_len', 'hetero_stride',
        'normalize', 'split', 'truncate_train_for_purge',
        'entity_filter', 'missing_value_strategy',
        'downsample', 'hetero_type'
    ]

    relevant_config = {k: config.get(k) for k in relevant_keys}
    config_str = json.dumps(relevant_config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]
```

### 4.9 Disk Space Estimation

For Canada Photovoltaics with:
- ~500k training samples
- seq_len=360, pred_len=168
- n_features=12
- embed_dim=768

```
Per-sample storage:
- seq_x: 360 × 12 × 4 bytes = 17.3 KB
- seq_y: 168 × 12 × 4 bytes = 8.1 KB
- hetero_x: 30 × 768 × 4 bytes = 92.2 KB
- hetero_y: 14 × 768 × 4 bytes = 43.0 KB
- timestamps + metadata: ~1 KB
Total per sample: ~162 KB

For 500k train + 100k val + 100k test = 700k samples:
Total storage: 700k × 162 KB ≈ 113 GB

Trade-off: 113 GB disk → 10-100x training speedup
```

### 4.10 Implementation Priority

```
PHASE 1 (High Impact, Low Effort) - IMPLEMENTED:
├── [x] Add profiling instrumentation to __getitem__ (data_provider/data_loader.py)
├── [x] Create profiling module (data_provider/profiling.py)
├── [x] Create profiling CLI script (scripts/profile_dataloader.py)
└── [ ] Run profiling and document actual bottleneck percentages

PHASE 2 (Core Implementation):
├── [ ] Implement TensorCacheGenerator class (data_provider/tensor_cache.py)
├── [ ] Implement TensorCacheDataset class (data_provider/tensor_cache.py)
├── [ ] Add cli.tensor_cache command module (cli/tensor_cache.py)
├── [ ] Add use_tensor_cache config flag to TrainingConfig
└── [ ] Document num_workers/prefetch_factor tuning guidelines for tensor cache

PHASE 3 (Integration & Testing):
├── [ ] Modify Data_Provider for cache support
├── [ ] Add cache validation logic
├── [ ] Create Slurm job template (scripts/slurm/generate_tensor_cache.sh)
├── [ ] Add tuning guidelines for num_workers/prefetch_factor with tensor cache
└── [ ] Benchmark end-to-end speedup

PHASE 4 (Optimization):
├── [ ] Profile memory-mapped vs pre-loaded performance
├── [ ] Tune DataLoader prefetch settings
├── [ ] Consider NVIDIA DALI for further optimization
└── [ ] Add cache compression option (for storage-constrained systems)
```

---

## Part 5: Expected Performance Improvements

### 5.1 Before vs. After Comparison

| Metric | Before (Current) | After (With Cache) | Improvement |
|--------|------------------|-------------------|-------------|
| `__getitem__` time | 100-450 μs | <1 μs | 100-450x |
| Batch prep time | 50-200 ms | <10 ms | 5-20x |
| GPU utilization | 5-20% | 80-95% | 4-20x |
| Epoch time | 30-60 min | 3-10 min | 3-10x |
| **Total training time** | **>24 hours** | **2-6 hours** | **4-12x** |

### 5.2 Resource Trade-offs

| Resource | Before | After | Notes |
|----------|--------|-------|-------|
| Disk usage | Low | +50-200 GB | Acceptable trade-off |
| RAM usage | Moderate | Lower (mmap) | OS manages page cache |
| CPU during training | Bottleneck | Minimal | Workers feed GPU easily |
| GPU utilization | Poor | Excellent | Continuous feeding |
| Preprocessing time | 0 | 5-30 min | One-time Slurm job |

### 5.3 Break-Even Analysis

Cache generation cost: 30 minutes (Slurm CPU job)
Per-epoch savings: 25 minutes (conservative estimate)

**Break-even: ~1.2 epochs**

For 50-epoch training: 50 × 25 min = 1250 minutes = 20.8 hours saved

---

## Appendix A: Alternative Approaches Considered

### A.1 Increase num_workers

**Approach**: Use more DataLoader workers (e.g., 16-32)
**Why not sufficient**: Workers are CPU-bound on the same operations. More workers = more CPU contention, not more throughput. May help marginally but doesn't solve the fundamental problem.

### A.2 GPU-Accelerated Data Loading

**Approach**: Move temporal matching to GPU
**Why not chosen**:
- Complex to implement
- GPU memory limited
- Dictionary lookups don't parallelize well on GPU
- Cache approach is simpler and more effective

### A.3 Just-In-Time Compilation (Numba)

**Approach**: JIT compile `__getitem__` operations
**Why not sufficient**:
- Dictionary operations not JIT-friendly
- Numpy/Pandas operations have overhead
- Doesn't solve the fundamental per-sample work problem

### A.4 LMDB/HDF5 Storage

**Approach**: Use LMDB or HDF5 instead of numpy memmap
**Trade-offs**:
- More complex implementation
- Similar performance for sequential access
- Consider if need random access optimization

---

## Appendix B: DataLoader Tuning Guidelines

### B.1 Tuning num_workers and prefetch_factor

With tensor cache making `__getitem__` nearly instant (<1 μs), DataLoader configuration becomes critical:

#### Without Tensor Cache (Current - CPU-bound __getitem__)
```yaml
training:
  num_workers: 0-4         # More workers don't help - still CPU-bound
  prefetch_factor: 2       # Limited benefit
```

#### With Tensor Cache (Optimized - Fast __getitem__)
```yaml
training:
  num_workers: 4-16        # Scale with available vCPUs
  prefetch_factor: 4-16    # Increase to keep workers busy
  batch_size: 768          # May increase further with better feeding
```

#### Tuning Strategy

1. **Start Conservative**: `num_workers=4, prefetch_factor=4`
2. **Monitor GPU Utilization**: Use `nvidia-smi dmon -s u`
3. **Increase if GPU Underutilized**: Increment both by 2-4
4. **Watch for Diminishing Returns**: Beyond ~16 workers, benefit plateaus
5. **Memory Considerations**: Each worker holds `prefetch_factor × batch_size` in memory

#### Example Configurations by Hardware

| Hardware | vCPUs | Recommended num_workers | Recommended prefetch_factor |
|----------|-------|-------------------------|----------------------------|
| RTX 5000 Ada + 6 vCPU | 6 | 4-6 | 4-8 |
| A100 + 16 vCPU | 16 | 8-12 | 8-16 |
| H100 + 32 vCPU | 32 | 12-16 | 8-16 |

**Key Insight**: With tensor cache, you're no longer CPU-bound on computation, but you may become I/O bound on memory-mapped file access. More workers help parallelize disk I/O.

---

## Appendix C: Profiling Code Snippets

### C.1 Quick DataLoader Benchmark

```python
# scripts/benchmark_dataloader.py
import time
import torch
from data_provider.data_factory import Data_Provider

def benchmark_dataloader(data_provider, num_batches=100):
    loader = data_provider.get_train(return_type='loader')

    times = []
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        start = time.perf_counter()
        # Simulate GPU transfer
        _ = [t.cuda() if isinstance(t, torch.Tensor) else t for t in batch]
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    print(f"Avg batch time: {sum(times)/len(times)*1000:.2f} ms")
    print(f"Throughput: {num_batches/sum(times):.1f} batches/sec")
```

### C.2 Per-Sample Operation Timing

```python
# Add to data_loader.py for detailed profiling
import functools
import time

def profile_operation(name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            if not hasattr(wrapper, '_times'):
                wrapper._times = []
            wrapper._times.append(elapsed)
            return result
        return wrapper
    return decorator
```
