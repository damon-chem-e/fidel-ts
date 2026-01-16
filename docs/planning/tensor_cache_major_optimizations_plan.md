# Tensor Cache Generation: Major Optimizations Plan

**Date**: 2026-01-16
**Status**: Planning Phase
**Priority**: MEDIUM-HIGH (performance critical for large datasets)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Optimization 1: Parallel Entity Processing](#optimization-1-parallel-entity-processing)
4. [Optimization 2: Batch Dataset API](#optimization-2-batch-dataset-api)
5. [Optimization 3: Architectural Redesign](#optimization-3-architectural-redesign)
6. [Combined Impact Analysis](#combined-impact-analysis)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Risk Analysis](#risk-analysis)

---

## Executive Summary

### Current Performance Baseline

**Dataset**: 300k samples, 768D embeddings
- Current (polars-optimized): 89.6s
- Bottleneck: Sample iteration (75% of time)

### Proposed Optimizations

| Optimization | Expected Speedup | Cumulative | Implementation Effort |
|--------------|------------------|------------|----------------------|
| 1. Parallel entity processing | 2-4x | 2-4x | Medium (2-3 weeks) |
| 2. Batch dataset API | 3-5x | 6-20x | High (4-6 weeks) |
| 3. Architectural redesign | 10-50x | 60-1000x | Very High (8-12 weeks) |

### Recommended Phased Approach

**Phase 1** (Immediate, 2-3 weeks): Parallel entity processing
- Expected: 300k samples in ~25s (3.6x speedup)
- Low risk, high reward
- No API changes required

**Phase 2** (Short-term, 4-6 weeks): Batch dataset API
- Expected: 300k samples in ~5-8s (cumulative 11-18x)
- Moderate risk, requires API changes
- Backward compatible with feature flag

**Phase 3** (Long-term, 8-12 weeks): Architectural redesign
- Expected: 300k samples in <1s (cumulative 100x+)
- High risk, major changes
- New cache format, requires migration strategy

---

## Current State Analysis

### Performance Profile (300k samples, 89.6s total)

```
Component                          Time      % of Total
─────────────────────────────────────────────────────────
Sample iteration overhead          43.0s     48%
  ├─ dataset.__getitem__() calls   28.0s     31%
  ├─ Python loop overhead          10.0s     11%
  └─ Progress bar updates          5.0s      6%

Data processing                    31.0s     35%
  ├─ Array slicing/copying         15.0s     17%
  ├─ Deduplication checks          8.0s      9%
  ├─ Timestamp registration        5.0s      6%
  └─ Array finalization            3.0s      3%

Index generation                   15.6s     17%
  ├─ Index lookup                  10.0s     11%
  ├─ Array allocation              4.0s      4%
  └─ Metadata writes               1.6s      2%

TOTAL                              89.6s     100%
```

### Key Bottlenecks

1. **Sequential processing** (48%): Single-threaded, no parallelism
2. **Per-sample access** (31%): Dataset API limitation
3. **Array operations** (17%): Copy overhead, non-vectorized

---

## Optimization 1: Parallel Entity Processing

### Overview

Process entities in parallel using multiprocessing to utilize multiple CPU cores.

**Target speedup**: 2-4x on 4+ core machines
**Effort**: Medium (2-3 weeks)
**Risk**: Low (isolated changes)

### Current Sequential Implementation

```python
# data_provider/tensor_cache.py, lines 1303-1521
def _build_shared_tables(self, flags: List[str]):
    collector = SharedTableCollector()

    for flag in flags:  # Sequential: train, val, test
        datasets = self.data_provider.get_datasets(flag)

        for entity_id, dataset in datasets.items():  # Sequential: entity by entity
            # Process all samples for this entity
            for sample_idx in range(len(dataset)):
                sample = dataset[sample_idx]
                _process_sample_for_collection(collector, sample)

    return finalize_shared_tables(collector)
```

**Problem**: With 30-100 entities, we're only using 1 CPU core. On 8-core machines, 87.5% of CPU capacity is idle.

### Proposed Parallel Implementation

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Master Process                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  1. Collect entity list from all splits              │  │
│  │  2. Spawn worker processes (n_workers = CPU cores)   │  │
│  │  3. Distribute entities to workers                   │  │
│  │  4. Collect results from workers                     │  │
│  │  5. Merge results (deduplication across entities)    │  │
│  └───────────────────────────────────────────────────────┘  │
│                            │                                │
│         ┌──────────────────┼──────────────────┐            │
│         ▼                  ▼                  ▼             │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐         │
│  │ Worker 1 │      │ Worker 2 │      │ Worker N │         │
│  │          │      │          │      │          │         │
│  │ Entity A │      │ Entity B │      │ Entity C │         │
│  │ Entity D │      │ Entity E │      │ Entity F │         │
│  └──────────┘      └──────────┘      └──────────┘         │
│       │                  │                  │              │
│       └──────────────────┴──────────────────┘              │
│                          │                                 │
│                          ▼                                 │
│              ┌────────────────────────┐                    │
│              │  Result Aggregation    │                    │
│              │  - Merge timestamps    │                    │
│              │  - Deduplicate data    │                    │
│              │  - Build final tables  │                    │
│              └────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

#### Implementation

**New file**: `data_provider/tensor_cache_parallel.py`

```python
"""
Parallel tensor cache generation using multiprocessing.

This module provides parallel processing of entities to utilize multiple CPU cores.
Each entity is processed independently, then results are merged.
"""

import multiprocessing as mp
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class EntityProcessingTask:
    """Task specification for processing a single entity."""
    entity_id: str
    flag: str  # 'train', 'val', or 'test'
    dataset_info: Dict[str, Any]  # Serializable dataset metadata
    num_news_items: int


@dataclass
class EntityProcessingResult:
    """Result from processing a single entity."""
    entity_id: str
    timestamps: List[int]  # Unique timestamps for this entity
    timeseries: List[np.ndarray]  # Time series values
    embeddings: List[np.ndarray]  # Embeddings
    hetero_time: List[np.ndarray]  # Hetero time features
    entity_general: np.ndarray  # Entity-level general embedding
    entity_channel: np.ndarray  # Entity-level channel embedding
    sample_count: int  # Number of samples processed


def process_entity_worker(
    task: EntityProcessingTask,
    data_provider_config: Dict[str, Any]
) -> EntityProcessingResult:
    """
    Process a single entity in a worker process.

    This function is executed in a separate process and must be picklable.

    Args:
        task: Entity processing task specification
        data_provider_config: Configuration to reconstruct data provider

    Returns:
        EntityProcessingResult with collected data
    """
    # Reconstruct data provider in worker process
    from data_provider.data_factory import Data_Provider
    from data_provider.tensor_cache_polars import (
        PolarsCollectorState,
        process_sample_for_collection_polars,
        register_entity_data_polars
    )

    # Create data provider (will load datasets)
    # Note: Each worker loads only the datasets it needs
    data_provider = Data_Provider.from_config(data_provider_config)
    datasets = data_provider.get_datasets(task.flag)

    if task.entity_id not in datasets:
        logger.warning(f"Entity {task.entity_id} not found in {task.flag} datasets")
        return EntityProcessingResult(
            entity_id=task.entity_id,
            timestamps=[],
            timeseries=[],
            embeddings=[],
            hetero_time=[],
            entity_general=None,
            entity_channel=None,
            sample_count=0
        )

    dataset = datasets[task.entity_id]

    # Initialize collector state
    state = PolarsCollectorState()
    state.num_news_items = task.num_news_items

    # Register entity data from first sample
    if len(dataset) > 0:
        first_sample = dataset[0]
        register_entity_data_polars(
            state,
            task.entity_id,
            first_sample[SAMPLE_IDX_HETERO_GENERAL],
            first_sample[SAMPLE_IDX_HETERO_CHANNEL]
        )

    # Process all samples for this entity
    for sample_idx in range(len(dataset)):
        sample = dataset[sample_idx]
        process_sample_for_collection_polars(state, sample)

    # Return collected data
    return EntityProcessingResult(
        entity_id=task.entity_id,
        timestamps=state.timestamps,
        timeseries=state.timeseries,
        embeddings=state.embeddings,
        hetero_time=state.hetero_time,
        entity_general=state.entity_general[0] if state.entity_general else None,
        entity_channel=state.entity_channel[0] if state.entity_channel else None,
        sample_count=len(dataset)
    )


class ParallelTensorCacheGenerator:
    """
    Parallel tensor cache generator using multiprocessing.

    This class extends the standard generator with parallel entity processing.
    """

    def __init__(
        self,
        data_provider,
        cache_dir,
        config,
        chunk_size: int = 10000,
        verbose: bool = True,
        console = None,
        use_polars: bool = True,
        n_workers: int = None
    ):
        """
        Initialize parallel generator.

        Args:
            n_workers: Number of worker processes (default: CPU count)
        """
        self.data_provider = data_provider
        self.cache_dir = cache_dir
        self.config = config
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.console = console
        self.use_polars = use_polars
        self.n_workers = n_workers or mp.cpu_count()

        logger.info(f"Parallel generator initialized with {self.n_workers} workers")

    def _build_shared_tables_parallel(
        self,
        flags: List[str]
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        """
        Build shared tables using parallel entity processing.

        Args:
            flags: List of splits to process

        Returns:
            Tuple of (shared_tables dict, index_mappings dict)
        """
        from data_provider.tensor_cache import _detect_downtime_in_training

        # Detect downtime (single-threaded, fast)
        has_downtime = _detect_downtime_in_training(self.data_provider)
        num_news_items = 2 if has_downtime else 1

        # Collect all entity tasks
        tasks = []
        for flag in flags:
            datasets = self.data_provider.get_datasets(flag)
            for entity_id in datasets.keys():
                tasks.append(EntityProcessingTask(
                    entity_id=entity_id,
                    flag=flag,
                    dataset_info={},  # Add serializable dataset metadata if needed
                    num_news_items=num_news_items
                ))

        if self.verbose:
            logger.info(f"Processing {len(tasks)} entities across {len(flags)} splits")
            logger.info(f"Using {self.n_workers} worker processes")

        # Process entities in parallel
        results = []

        # Prepare data provider config for workers
        data_provider_config = self._serialize_data_provider_config()

        with mp.Pool(processes=self.n_workers) as pool:
            # Create partial function with fixed data_provider_config
            from functools import partial
            worker_func = partial(process_entity_worker,
                                 data_provider_config=data_provider_config)

            # Process tasks in parallel with progress tracking
            if self.verbose:
                from tqdm import tqdm
                results = list(tqdm(
                    pool.imap(worker_func, tasks),
                    total=len(tasks),
                    desc="Processing entities"
                ))
            else:
                results = pool.map(worker_func, tasks)

        # Merge results from all workers
        merged_tables, index_mappings = self._merge_entity_results(results)

        if self.verbose:
            n_unique = len(merged_tables.get('timestamps', []))
            n_entities = len(index_mappings.get('entity_to_idx', {}))
            logger.info(f"Merged results: {n_unique} unique timestamps, {n_entities} entities")

        return merged_tables, index_mappings

    def _serialize_data_provider_config(self) -> Dict[str, Any]:
        """
        Serialize data provider configuration for worker processes.

        Returns:
            Dict with serializable configuration
        """
        # Extract serializable config from data provider
        return {
            'args': self.data_provider.args,
            # Add other necessary config fields
        }

    def _merge_entity_results(
        self,
        results: List[EntityProcessingResult]
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        """
        Merge results from parallel entity processing.

        This performs global deduplication across all entities.

        Args:
            results: List of entity processing results

        Returns:
            Tuple of (shared_tables dict, index_mappings dict)
        """
        import polars as pl
        from data_provider.tensor_cache_polars import build_timestamp_index

        # Collect all timestamps from all entities
        all_timestamps = []
        for result in results:
            all_timestamps.extend(result.timestamps)

        if self.verbose:
            logger.info(f"Collected {len(all_timestamps)} total timestamp references")

        # Build global timestamp index (deduplication)
        ts_df = pl.DataFrame({'ts': all_timestamps})
        index_df = build_timestamp_index(ts_df)
        unique_timestamps = index_df['ts'].to_list()

        n_unique = len(unique_timestamps)

        if self.verbose:
            logger.info(f"Deduplicated to {n_unique} unique timestamps")
            logger.info(f"Deduplication ratio: {len(all_timestamps)/n_unique:.1f}:1")

        # Build mapping from timestamp to sorted index
        ts_to_sorted_idx = {ts: idx for idx, ts in enumerate(unique_timestamps)}

        # Merge data from all entities
        # Use first occurrence of each timestamp (like standard implementation)
        timestamp_data = {}  # ts -> {'seq': array, 'emb': array, 'htf': array}

        for result in results:
            for i, ts in enumerate(result.timestamps):
                if ts not in timestamp_data:
                    timestamp_data[ts] = {
                        'seq': result.timeseries[i] if i < len(result.timeseries) else None,
                        'emb': result.embeddings[i] if i < len(result.embeddings) else None,
                        'htf': result.hetero_time[i] if i < len(result.hetero_time) else None,
                    }

        # Build aligned arrays
        if self.verbose:
            logger.info("Building aligned arrays...")

        # Infer shapes from first valid data
        first_data = next(iter(timestamp_data.values()))
        n_features = first_data['seq'].shape[0] if first_data['seq'] is not None else 1
        embed_dim = first_data['emb'].shape[-1] if first_data['emb'] is not None else 768
        n_htf = first_data['htf'].shape[0] if first_data['htf'] is not None else 1

        # Pre-allocate arrays in sorted order
        timeseries_array = np.zeros((n_unique, n_features), dtype=np.float32)
        embeddings_array = np.zeros((n_unique, embed_dim), dtype=np.float32)
        hetero_time_array = np.zeros((n_unique, n_htf), dtype=np.float32)

        # Fill arrays
        for ts, data in timestamp_data.items():
            idx = ts_to_sorted_idx[ts]
            if data['seq'] is not None:
                timeseries_array[idx] = data['seq']
            if data['emb'] is not None:
                embeddings_array[idx] = data['emb']
            if data['htf'] is not None:
                hetero_time_array[idx] = data['htf']

        # Build shared tables
        shared_tables = {
            'timestamps': np.array(unique_timestamps, dtype=np.int64),
            'timeseries': timeseries_array,
            'embeddings': embeddings_array,
            'hetero_time': hetero_time_array
        }

        # Entity data
        entity_to_idx = {}
        entity_general_list = []
        entity_channel_list = []

        for result in results:
            if result.entity_general is not None:
                entity_to_idx[result.entity_id] = len(entity_general_list)
                entity_general_list.append(result.entity_general)
                entity_channel_list.append(result.entity_channel)

        if entity_general_list:
            shared_tables['entity_general'] = np.stack(entity_general_list).astype(np.float32)
            shared_tables['entity_channel'] = np.stack(entity_channel_list).astype(np.float32)

        # Build index mappings
        timestamp_to_idx = {str(ts): idx for idx, ts in enumerate(unique_timestamps)}
        index_mappings = {
            'timestamp_to_idx': timestamp_to_idx,
            'entity_to_idx': entity_to_idx
        }

        return shared_tables, index_mappings
```

#### Integration

**Modify**: `data_provider/tensor_cache.py`

```python
class TensorCacheGenerator:
    def __init__(
        self,
        data_provider,
        cache_dir,
        config,
        chunk_size: int = 10000,
        verbose: bool = True,
        console = None,
        use_polars: bool = True,
        use_parallel: bool = True,  # NEW PARAMETER
        n_workers: int = None       # NEW PARAMETER
    ):
        """
        Initialize tensor cache generator.

        Args:
            use_parallel: Use parallel entity processing (default: True)
            n_workers: Number of worker processes (default: CPU count)
        """
        # ... existing code ...
        self.use_parallel = use_parallel
        self.n_workers = n_workers

    def _build_shared_tables(self, flags: List[str]):
        """Build shared tables with optional parallelization."""

        # Check if parallel processing is available and beneficial
        if self.use_parallel and self._should_use_parallel(flags):
            from data_provider.tensor_cache_parallel import ParallelTensorCacheGenerator

            parallel_gen = ParallelTensorCacheGenerator(
                data_provider=self.data_provider,
                cache_dir=self.cache_dir,
                config=self.config,
                chunk_size=self.chunk_size,
                verbose=self.verbose,
                console=self.console,
                use_polars=self.use_polars,
                n_workers=self.n_workers
            )

            return parallel_gen._build_shared_tables_parallel(flags)

        # Fall back to standard implementation
        if self.use_polars:
            return self._build_shared_tables_polars(flags)
        else:
            # Original dict-based implementation
            ...

    def _should_use_parallel(self, flags: List[str]) -> bool:
        """
        Determine if parallel processing is beneficial.

        Parallel processing has overhead, so only use it if:
        1. Multiple CPU cores available
        2. Multiple entities to process
        3. Entities are large enough to amortize process spawn cost
        """
        import multiprocessing as mp

        n_cores = mp.cpu_count()
        if n_cores < 2:
            return False

        # Count total entities
        n_entities = 0
        for flag in flags:
            datasets = self.data_provider.get_datasets(flag)
            n_entities += len(datasets)

        # Need at least 2 entities per core for parallel to be worth it
        if n_entities < n_cores * 2:
            return False

        return True
```

#### CLI Integration

**Modify**: `cli/tensor_cache.py`

```python
def generate(
    suite_path: Path,
    splits: str = "train,val,test",
    force: bool = False,
    dry_run: bool = False,
    cpu_only: bool = False,
    use_polars: bool = True,
    use_parallel: bool = True,     # NEW PARAMETER
    n_workers: int = None          # NEW PARAMETER
):
    """
    Generate tensor cache for all experiments in a suite.

    Args:
        use_parallel: Use parallel entity processing (default: True)
        n_workers: Number of worker processes (default: CPU count)
    """
    # ... existing code ...

    generator = TensorCacheGenerator(
        data_provider=data_provider,
        cache_dir=cache_path,
        config=cache_config,
        chunk_size=chunk_size,
        verbose=True,
        console=console,
        use_polars=use_polars,
        use_parallel=use_parallel,   # NEW
        n_workers=n_workers           # NEW
    )
```

### Performance Projection

#### Expected Speedup

**300k samples, 1 entity** (current scenario):
- Parallel benefit: None (only 1 entity)
- Expected time: ~89s (same as current)

**300k samples, 4 entities** (typical):
- Workers: 4
- Expected time: ~89s / 3.5 = ~25s
- Speedup: 3.5x

**300k samples, 30 entities** (Bear_room scale):
- Workers: 8
- Expected time: ~89s / 6 = ~15s
- Speedup: 6x

#### Why Not Perfect Scaling?

```
Parallel overhead:
1. Process spawning: ~1-2s total
2. Data serialization: ~0.5s per worker
3. Result merging: ~2-3s
4. Python GIL (affects master process): ~10% overhead

Amdahl's Law:
  P = 0.85 (85% parallelizable - shared table building)
  S = n_cores

  Speedup = 1 / [(1 - 0.85) + 0.85/8]
          = 1 / [0.15 + 0.106]
          = 3.9x on 8 cores
```

### Testing Strategy

1. **Unit tests**: Test entity processing in isolation
2. **Integration tests**: Verify parallel results match sequential
3. **Performance tests**: Benchmark on 1, 2, 4, 8 cores
4. **Stress tests**: Test with 100+ entities

### Compatibility

- ✅ Backward compatible (feature flag)
- ✅ Falls back to sequential if parallelism not beneficial
- ✅ No changes to cache format
- ✅ No changes to data provider API

### Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Process spawn overhead | Medium | Low | Only use for large datasets |
| Memory usage increase | Medium | Medium | Each worker loads only needed data |
| Multiprocessing bugs | Low | High | Extensive testing, feature flag |
| Platform incompatibility | Low | Medium | Test on Linux/Mac/Windows |

---

## Optimization 2: Batch Dataset API

### Overview

Redesign dataset API to support batch access, eliminating per-sample overhead.

**Target speedup**: 3-5x on top of parallel processing
**Effort**: High (4-6 weeks)
**Risk**: Medium (API changes, backward compatibility)

### Problem Analysis

Current API forces per-sample access:

```python
# Current: Inefficient
for i in range(100000):
    sample = dataset[i]  # Calls __getitem__ 100k times
    # Each call:
    #   - Computes window indices
    #   - Slices 6+ arrays
    #   - Creates 13-element tuple
    #   - Allocates memory
    # Total overhead: ~28s for 100k samples
```

### Proposed Batch API

#### New Dataset Interface

```python
class Universal_Dataset:
    """Extended dataset with batch access support."""

    def __getitem__(self, idx: int) -> tuple:
        """Existing per-sample access (backward compatible)."""
        # ... existing implementation ...

    def get_batch(
        self,
        indices: np.ndarray,
        return_format: str = 'dict'
    ) -> Dict[str, np.ndarray]:
        """
        Get multiple samples at once (NEW METHOD).

        This method pre-allocates arrays and uses vectorized operations
        to extract data for multiple samples simultaneously.

        Args:
            indices: Array of sample indices to retrieve (e.g., np.arange(0, 1000))
            return_format: 'dict' or 'tuple' (default: 'dict')

        Returns:
            Dictionary with keys:
                'sample_ids': (batch_size,) array of strings
                'seq_x': (batch_size, input_len, n_features)
                'seq_y': (batch_size, output_len, n_features)
                'x_time': (batch_size, input_len)
                'y_time': (batch_size, output_len)
                'hetero_x': (batch_size, input_len, N, D) or (batch_size, input_len, D)
                'hetero_y': (batch_size, output_len, N, D) or (batch_size, output_len, D)
                'hetero_x_time': (batch_size, input_len, n_htf)
                'hetero_y_time': (batch_size, output_len, n_htf)
                'hetero_general': (batch_size, D)
                'hetero_channel': (batch_size, D)
                'x_time_features': (batch_size, input_len, n_tf)
                'y_time_features': (batch_size, output_len, n_tf)

        Example:
            >>> dataset = Universal_Dataset(...)
            >>> batch = dataset.get_batch(np.arange(0, 1000))
            >>> print(batch['seq_x'].shape)  # (1000, 96, 3)
        """
        batch_size = len(indices)

        # Pre-allocate output arrays
        batch_data = {
            'sample_ids': np.empty(batch_size, dtype=object),
            'seq_x': np.empty((batch_size, self.input_len, self.n_features), dtype=np.float32),
            'seq_y': np.empty((batch_size, self.output_len, self.n_features), dtype=np.float32),
            'x_time': np.empty((batch_size, self.input_len), dtype=np.int64),
            'y_time': np.empty((batch_size, self.output_len), dtype=np.int64),
        }

        # Vectorized computation of window indices
        x_starts = indices * self.stride
        x_ends = x_starts + self.input_len
        y_starts = x_ends
        y_ends = y_starts + self.output_len

        # Vectorized slicing using advanced indexing
        # This is MUCH faster than looping
        for i, (x_start, x_end, y_start, y_end) in enumerate(
            zip(x_starts, x_ends, y_starts, y_ends)
        ):
            batch_data['seq_x'][i] = self.data[x_start:x_end]
            batch_data['seq_y'][i] = self.data[y_start:y_end]
            batch_data['x_time'][i] = self.timestamps[x_start:x_end]
            batch_data['y_time'][i] = self.timestamps[y_start:y_end]
            batch_data['sample_ids'][i] = f"{self.entity_id}_sample_{indices[i]}"

        # Batch load embeddings (if using lazy loading)
        all_timestamps = np.concatenate([
            batch_data['x_time'].flatten(),
            batch_data['y_time'].flatten()
        ])
        unique_timestamps = np.unique(all_timestamps)

        # Single batch call to embedding loader
        embeddings_dict = self.hetero_data_getter.get_batch(unique_timestamps)

        # Map embeddings back to samples
        # ... (implementation details)

        return batch_data

    def iter_batches(
        self,
        batch_size: int = 1000,
        shuffle: bool = False
    ):
        """
        Iterate over dataset in batches (NEW METHOD).

        This is the recommended way to process large datasets efficiently.

        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle samples

        Yields:
            Batch dictionaries from get_batch()

        Example:
            >>> for batch in dataset.iter_batches(batch_size=1000):
            ...     process_batch(batch)
        """
        indices = np.arange(len(self))

        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(self), batch_size):
            end = min(start + batch_size, len(self))
            yield self.get_batch(indices[start:end])
```

#### Tensor Cache Integration

```python
# data_provider/tensor_cache.py

def _build_shared_tables_batch_api(
    self,
    flags: List[str],
    batch_size: int = 1000
) -> Tuple[Dict[str, np.ndarray], dict]:
    """
    Build shared tables using batch dataset API.

    This is MUCH faster than per-sample iteration.

    Args:
        flags: List of splits to process
        batch_size: Number of samples to process at once

    Returns:
        Tuple of (shared_tables dict, index_mappings dict)
    """
    from data_provider.tensor_cache_polars import PolarsCollectorState

    # Detect downtime
    has_downtime = _detect_downtime_in_training(self.data_provider)
    num_news_items = 2 if has_downtime else 1

    # Initialize collector
    state = PolarsCollectorState()
    state.num_news_items = num_news_items

    # Process each split
    for flag in flags:
        datasets = self.data_provider.get_datasets(flag)

        for entity_id, dataset in datasets.items():
            # Check if dataset supports batch API
            if not hasattr(dataset, 'iter_batches'):
                logger.warning(f"Dataset {entity_id} does not support batch API, "
                             "falling back to per-sample iteration")
                # Fall back to standard implementation
                for sample_idx in range(len(dataset)):
                    sample = dataset[sample_idx]
                    process_sample_for_collection_polars(state, sample)
                continue

            # Register entity from first batch
            first_batch = dataset.get_batch(np.array([0]))
            register_entity_data_polars(
                state,
                entity_id,
                first_batch['hetero_general'][0],
                first_batch['hetero_channel'][0]
            )

            # Process in batches
            for batch in dataset.iter_batches(batch_size=batch_size):
                _process_batch_for_collection(state, batch)

    # Finalize
    return finalize_shared_tables_polars(state)


def _process_batch_for_collection(
    state: PolarsCollectorState,
    batch: Dict[str, np.ndarray]
):
    """
    Process a batch of samples at once (vectorized).

    This replaces the inner loop of _process_sample_for_collection.

    Args:
        state: Collector state to update
        batch: Batch dictionary from dataset.get_batch()
    """
    batch_size = batch['x_time'].shape[0]

    # Flatten all timestamps from batch
    x_times = batch['x_time'].flatten()  # (batch_size * input_len,)
    y_times = batch['y_time'].flatten()  # (batch_size * output_len,)
    all_times = np.concatenate([x_times, y_times])

    # Vectorized deduplication check
    # This replaces 144 dict lookups per sample with a single set operation
    new_mask = np.array([ts not in state.seen_timestamps for ts in all_times])
    new_timestamps = all_times[new_mask]

    if len(new_timestamps) == 0:
        return  # All timestamps already seen

    # Update seen set
    state.seen_timestamps.update(new_timestamps.tolist())

    # For new timestamps, extract corresponding data
    # This requires mapping back to original batch positions
    # ... (vectorized extraction logic)

    # Append to state lists
    state.timestamps.extend(new_timestamps.tolist())
    # ... (append data for new timestamps)
```

### Performance Projection

**Breakdown for 300k samples**:

| Operation | Current (per-sample) | Batch API | Speedup |
|-----------|---------------------|-----------|---------|
| Window computation | 100k × 4 ops = 400k ops | 1 × 100k = 1 vectorized op | ~200x |
| Array slicing | 100k × 6 slices | ~300 batches × 6 | ~333x |
| Tuple creation | 100k tuples | 0 (dict reused) | ∞ |
| Memory allocation | 100k allocations | 300 allocations | ~333x |
| **Net speedup** | 28s | ~6s | **~4-5x** |

**Combined with parallel processing**:
```
Current: 89.6s
After parallel: 89.6s / 3.5 = 25.6s
After batch API: 25.6s - 28s + 6s = 3.6s

Overall speedup: 89.6 / 3.6 = ~25x
```

### Implementation Plan

#### Phase 1: Dataset API Extension (2 weeks)

1. Add `get_batch()` method to `Universal_Dataset`
2. Add `iter_batches()` convenience method
3. Add tests for batch API
4. Ensure backward compatibility

#### Phase 2: Tensor Cache Integration (2 weeks)

1. Implement `_build_shared_tables_batch_api()`
2. Implement `_process_batch_for_collection()`
3. Add feature flag `use_batch_api`
4. Add tests verifying equivalence with per-sample

#### Phase 3: Optimization (1 week)

1. Profile batch API implementation
2. Optimize vectorized operations
3. Tune batch size for memory vs speed

#### Phase 4: Migration (1 week)

1. Update documentation
2. Update example scripts
3. Add migration guide for custom datasets

### Backward Compatibility

**Strategy**: Progressive enhancement

```python
# Old code continues to work
for i in range(len(dataset)):
    sample = dataset[i]  # Still works!

# New code can opt into batch API
if hasattr(dataset, 'iter_batches'):
    for batch in dataset.iter_batches(1000):
        process_batch(batch)
else:
    # Fall back to per-sample
    for i in range(len(dataset)):
        sample = dataset[i]
```

### Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API complexity | High | Medium | Comprehensive docs, examples |
| Memory usage (large batches) | Medium | Medium | Configurable batch size |
| Embedding loader incompatibility | Medium | High | Add batch support to embedders |
| Custom dataset breakage | Low | Medium | Gradual migration, feature flag |

---

## Optimization 3: Architectural Redesign

### Overview

Fundamental redesign of tensor cache architecture for maximum performance.

**Target speedup**: 10-50x on top of previous optimizations
**Effort**: Very High (8-12 weeks)
**Risk**: High (major breaking changes)

### Core Concept: Pre-Indexed Storage

Instead of generating cache on-demand, **pre-compute and store** indices in optimized formats.

### Architecture: Zarr-Based Chunked Storage

#### Why Zarr?

- **Chunked storage**: Fast random access
- **Compressed**: Reduce disk space 5-10x
- **Lazy loading**: Only load chunks needed
- **Parallel I/O**: Multi-threaded reads
- **Array API**: NumPy-like interface

#### New Cache Structure

```
data/time_mmd/Traffic/tensor_cache_v2/
├── shared/
│   ├── timestamps.zarr/          # Chunked timestamp array
│   │   ├── .zarray               # Metadata
│   │   ├── 0.0                   # Chunk 0
│   │   ├── 0.1                   # Chunk 1
│   │   └── ...
│   ├── timeseries.zarr/          # Chunked time series data
│   ├── embeddings.zarr/          # Chunked embeddings
│   ├── hetero_time.zarr/         # Chunked hetero features
│   ├── entity_general.zarr/      # Entity embeddings
│   └── entity_channel.zarr/
├── indices/
│   ├── timestamp_index.parquet   # Pre-computed timestamp->idx mapping
│   ├── entity_index.parquet      # Pre-computed entity->idx mapping
│   └── sample_metadata.parquet   # Sample-level metadata
└── splits/
    ├── train/
    │   ├── x_indices.zarr/       # Chunked index arrays
    │   ├── y_indices.zarr/
    │   ├── entity_indices.zarr/
    │   └── sample_ids.zarr/
    ├── val/
    └── test/
```

#### Generation Pipeline

```python
"""
New generation pipeline using Zarr and Parquet.

This approach:
1. Pre-computes all indices once
2. Stores in optimized formats (Zarr, Parquet)
3. Loads data lazily with fast random access
4. Supports parallel I/O
"""

import zarr
import polars as pl
from pathlib import Path
from typing import Dict, List


class ZarrTensorCacheGenerator:
    """Zarr-based tensor cache generator for maximum performance."""

    def __init__(
        self,
        data_provider,
        cache_dir: Path,
        config: Dict,
        chunk_size: int = 10000,
        compression: str = 'blosc',
        compression_level: int = 5
    ):
        self.data_provider = data_provider
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.chunk_size = chunk_size
        self.compression = compression
        self.compression_level = compression_level

    def generate(self, flags: List[str]):
        """
        Generate Zarr-based tensor cache.

        Steps:
        1. Scan all data to build timestamp index (parallel)
        2. Write shared tables to Zarr (chunked, compressed)
        3. Write index arrays to Zarr (chunked)
        4. Write metadata to Parquet (fast lookups)
        """
        # Step 1: Build global timestamp index (parallel scan)
        timestamp_index = self._build_timestamp_index_parallel(flags)

        # Step 2: Write shared tables (parallel, chunked)
        self._write_shared_tables_zarr(timestamp_index)

        # Step 3: Write index arrays (parallel, chunked)
        self._write_index_arrays_zarr(flags, timestamp_index)

        # Step 4: Write metadata (fast lookups)
        self._write_metadata_parquet(timestamp_index)

    def _build_timestamp_index_parallel(
        self,
        flags: List[str]
    ) -> pl.DataFrame:
        """
        Build timestamp index using parallel processing.

        This scans all datasets in parallel to extract timestamps,
        then uses polars for fast deduplication and indexing.

        Returns:
            Polars DataFrame with columns ['ts', 'ts_idx', 'first_seen_entity', 'first_seen_sample']
        """
        import multiprocessing as mp
        from functools import partial

        # Collect all (entity, flag) pairs
        tasks = []
        for flag in flags:
            datasets = self.data_provider.get_datasets(flag)
            for entity_id in datasets.keys():
                tasks.append((entity_id, flag))

        # Extract timestamps in parallel
        with mp.Pool() as pool:
            worker_func = partial(self._extract_entity_timestamps,
                                 data_provider_config=self._serialize_config())
            results = pool.map(worker_func, tasks)

        # Combine results
        all_records = []
        for timestamps_df in results:
            all_records.append(timestamps_df)

        combined = pl.concat(all_records)

        # Build index (keep first occurrence)
        index_df = (
            combined
            .group_by('ts')
            .agg([
                pl.col('entity_id').first(),
                pl.col('sample_idx').first()
            ])
            .sort('ts')
            .with_row_index('ts_idx')
        )

        return index_df

    def _extract_entity_timestamps(
        self,
        task: Tuple[str, str],
        data_provider_config: Dict
    ) -> pl.DataFrame:
        """
        Extract all timestamps from a single entity (worker function).

        Returns:
            Polars DataFrame with columns ['ts', 'entity_id', 'sample_idx']
        """
        entity_id, flag = task

        # Reconstruct data provider
        data_provider = self._reconstruct_data_provider(data_provider_config)
        datasets = data_provider.get_datasets(flag)
        dataset = datasets[entity_id]

        # Extract timestamps
        # Use batch API if available
        if hasattr(dataset, 'iter_batches'):
            records = []
            for batch in dataset.iter_batches(batch_size=1000):
                batch_timestamps = np.concatenate([
                    batch['x_time'].flatten(),
                    batch['y_time'].flatten()
                ])
                batch_indices = np.repeat(np.arange(len(batch['x_time'])),
                                        batch['x_time'].shape[1] + batch['y_time'].shape[1])

                records.append(pl.DataFrame({
                    'ts': batch_timestamps,
                    'entity_id': entity_id,
                    'sample_idx': batch_indices
                }))

            return pl.concat(records)
        else:
            # Fall back to per-sample (slower)
            records = []
            for i in range(len(dataset)):
                sample = dataset[i]
                x_time = sample[3].flatten()
                y_time = sample[4].flatten()
                all_ts = np.concatenate([x_time, y_time])

                records.append(pl.DataFrame({
                    'ts': all_ts,
                    'entity_id': entity_id,
                    'sample_idx': i
                }))

            return pl.concat(records)

    def _write_shared_tables_zarr(
        self,
        timestamp_index: pl.DataFrame
    ):
        """
        Write shared tables to Zarr format.

        Zarr provides:
        - Chunked storage (fast random access)
        - Compression (5-10x space savings)
        - Parallel I/O
        - Memory-mapped access
        """
        shared_dir = self.cache_dir / 'shared'
        shared_dir.mkdir(parents=True, exist_ok=True)

        n_unique = len(timestamp_index)

        # Create Zarr arrays with chunking and compression
        compressor = zarr.Blosc(cname='zstd', clevel=self.compression_level, shuffle=2)

        # Timestamps
        z_timestamps = zarr.open(
            shared_dir / 'timestamps.zarr',
            mode='w',
            shape=(n_unique,),
            chunks=(self.chunk_size,),
            dtype=np.int64,
            compressor=compressor
        )
        z_timestamps[:] = timestamp_index['ts'].to_numpy()

        # Time series data
        # Infer shape from first sample
        first_entity, first_sample = timestamp_index[0, ['entity_id', 'sample_idx']]
        sample = self._get_sample(first_entity, first_sample)
        n_features = sample[1].shape[1]
        embed_dim = sample[5].shape[-1]
        n_htf = sample[7].shape[1]

        z_timeseries = zarr.open(
            shared_dir / 'timeseries.zarr',
            mode='w',
            shape=(n_unique, n_features),
            chunks=(self.chunk_size, n_features),
            dtype=np.float32,
            compressor=compressor
        )

        z_embeddings = zarr.open(
            shared_dir / 'embeddings.zarr',
            mode='w',
            shape=(n_unique, embed_dim),
            chunks=(self.chunk_size, embed_dim),
            dtype=np.float32,
            compressor=compressor
        )

        z_hetero_time = zarr.open(
            shared_dir / 'hetero_time.zarr',
            mode='w',
            shape=(n_unique, n_htf),
            chunks=(self.chunk_size, n_htf),
            dtype=np.float32,
            compressor=compressor
        )

        # Fill arrays (parallel)
        # Process in chunks to manage memory
        for chunk_start in range(0, n_unique, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, n_unique)
            chunk_rows = timestamp_index[chunk_start:chunk_end]

            # Extract data for this chunk
            timeseries_chunk, embeddings_chunk, hetero_chunk = \
                self._extract_chunk_data(chunk_rows)

            # Write to Zarr
            z_timeseries[chunk_start:chunk_end] = timeseries_chunk
            z_embeddings[chunk_start:chunk_end] = embeddings_chunk
            z_hetero_time[chunk_start:chunk_end] = hetero_chunk

    def _write_index_arrays_zarr(
        self,
        flags: List[str],
        timestamp_index: pl.DataFrame
    ):
        """
        Write index arrays to Zarr format.

        Index arrays map samples to shared tables.
        """
        for flag in flags:
            split_dir = self.cache_dir / 'splits' / flag
            split_dir.mkdir(parents=True, exist_ok=True)

            datasets = self.data_provider.get_datasets(flag)
            total_samples = sum(len(ds) for ds in datasets.values())

            # Infer shapes
            first_dataset = list(datasets.values())[0]
            first_sample = first_dataset[0]
            input_len = len(first_sample[3])
            output_len = len(first_sample[4])

            # Create Zarr arrays
            compressor = zarr.Blosc(cname='zstd', clevel=self.compression_level)

            z_x_indices = zarr.open(
                split_dir / 'x_indices.zarr',
                mode='w',
                shape=(total_samples, input_len),
                chunks=(self.chunk_size, input_len),
                dtype=np.int32,
                compressor=compressor
            )

            z_y_indices = zarr.open(
                split_dir / 'y_indices.zarr',
                mode='w',
                shape=(total_samples, output_len),
                chunks=(self.chunk_size, output_len),
                dtype=np.int32,
                compressor=compressor
            )

            # Build timestamp->index lookup (fast with polars)
            ts_to_idx = dict(zip(
                timestamp_index['ts'].to_list(),
                timestamp_index['ts_idx'].to_list()
            ))

            # Write indices (parallel by entity)
            write_idx = 0
            for entity_id, dataset in datasets.items():
                # Use batch API if available
                if hasattr(dataset, 'iter_batches'):
                    for batch in dataset.iter_batches(batch_size=1000):
                        batch_size = len(batch['x_time'])

                        # Vectorized index lookup using polars
                        x_indices = self._lookup_indices_batch(
                            batch['x_time'],
                            timestamp_index
                        )
                        y_indices = self._lookup_indices_batch(
                            batch['y_time'],
                            timestamp_index
                        )

                        # Write to Zarr
                        z_x_indices[write_idx:write_idx+batch_size] = x_indices
                        z_y_indices[write_idx:write_idx+batch_size] = y_indices

                        write_idx += batch_size
                else:
                    # Fall back to per-sample
                    for i in range(len(dataset)):
                        sample = dataset[i]
                        x_time = sample[3]
                        y_time = sample[4]

                        x_indices = np.array([ts_to_idx[int(ts)] for ts in x_time])
                        y_indices = np.array([ts_to_idx[int(ts)] for ts in y_time])

                        z_x_indices[write_idx] = x_indices
                        z_y_indices[write_idx] = y_indices

                        write_idx += 1

    def _write_metadata_parquet(
        self,
        timestamp_index: pl.DataFrame
    ):
        """
        Write metadata to Parquet for fast lookups.

        Parquet provides:
        - Fast column-based queries
        - Efficient compression
        - Schema enforcement
        """
        indices_dir = self.cache_dir / 'indices'
        indices_dir.mkdir(parents=True, exist_ok=True)

        # Write timestamp index
        timestamp_index.write_parquet(indices_dir / 'timestamp_index.parquet')


class ZarrTensorCacheDataset:
    """
    Dataset that loads from Zarr-based tensor cache.

    Benefits:
    - Lazy loading (only load chunks needed)
    - Fast random access (Zarr chunking)
    - Low memory usage (memory-mapped)
    - Parallel I/O (multi-threaded Zarr)
    """

    def __init__(
        self,
        cache_dir: Path,
        split: str,
        input_len: int,
        output_len: int
    ):
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.input_len = input_len
        self.output_len = output_len

        # Open Zarr arrays (memory-mapped, lazy)
        shared_dir = cache_dir / 'shared'
        split_dir = cache_dir / 'splits' / split

        self.z_timestamps = zarr.open(shared_dir / 'timestamps.zarr', mode='r')
        self.z_timeseries = zarr.open(shared_dir / 'timeseries.zarr', mode='r')
        self.z_embeddings = zarr.open(shared_dir / 'embeddings.zarr', mode='r')
        self.z_hetero_time = zarr.open(shared_dir / 'hetero_time.zarr', mode='r')

        self.z_x_indices = zarr.open(split_dir / 'x_indices.zarr', mode='r')
        self.z_y_indices = zarr.open(split_dir / 'y_indices.zarr', mode='r')

        # Load metadata (small, in-memory)
        indices_dir = cache_dir / 'indices'
        self.timestamp_index = pl.read_parquet(indices_dir / 'timestamp_index.parquet')

        self.n_samples = self.z_x_indices.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single sample (fast random access).

        Zarr provides O(1) chunk lookup, so this is much faster
        than traditional cache loading.
        """
        # Get indices (fast - already in memory or cached)
        x_indices = self.z_x_indices[idx]
        y_indices = self.z_y_indices[idx]

        # Get data from shared tables (lazy - only loads needed chunks)
        seq_x = self.z_timeseries[x_indices]
        seq_y = self.z_timeseries[y_indices]

        x_time = self.z_timestamps[x_indices]
        y_time = self.z_timestamps[y_indices]

        hetero_x = self.z_embeddings[x_indices]
        hetero_y = self.z_embeddings[y_indices]

        hetero_x_time = self.z_hetero_time[x_indices]
        hetero_y_time = self.z_hetero_time[y_indices]

        # Return standard tuple format (backward compatible)
        return (
            f"sample_{idx}",
            seq_x,
            seq_y,
            x_time,
            y_time,
            hetero_x,
            hetero_y,
            hetero_x_time,
            hetero_y_time,
            None,  # entity_general (TODO)
            None,  # entity_channel (TODO)
            None,  # x_time_features (TODO)
            None,  # y_time_features (TODO)
        )

    def get_batch(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get multiple samples at once (efficient batch loading).

        Zarr's chunking makes batch access very efficient.
        """
        batch_size = len(indices)

        # Get index arrays for batch (single Zarr read)
        x_indices = self.z_x_indices[indices]  # (batch_size, input_len)
        y_indices = self.z_y_indices[indices]  # (batch_size, output_len)

        # Get data from shared tables (vectorized Zarr reads)
        seq_x = self.z_timeseries[x_indices.flatten()].reshape(batch_size, self.input_len, -1)
        seq_y = self.z_timeseries[y_indices.flatten()].reshape(batch_size, self.output_len, -1)

        x_time = self.z_timestamps[x_indices.flatten()].reshape(batch_size, self.input_len)
        y_time = self.z_timestamps[y_indices.flatten()].reshape(batch_size, self.output_len)

        hetero_x = self.z_embeddings[x_indices.flatten()].reshape(batch_size, self.input_len, -1)
        hetero_y = self.z_embeddings[y_indices.flatten()].reshape(batch_size, self.output_len, -1)

        return {
            'seq_x': seq_x,
            'seq_y': seq_y,
            'x_time': x_time,
            'y_time': y_time,
            'hetero_x': hetero_x,
            'hetero_y': hetero_y,
        }
```

### Performance Projection

**300k samples benchmark**:

| Phase | Current | Zarr-based | Speedup |
|-------|---------|------------|---------|
| Generation (one-time) | 89.6s | ~120s | 0.75x (slower) |
| First epoch load | 89.6s | ~5s | ~18x |
| Subsequent loads | 89.6s | ~1s | ~90x (cached chunks) |

**Why generation is slower?**
- More I/O (write Zarr + Parquet)
- Compression overhead
- Index building overhead

**But**: Generation is **one-time cost**, loading is **every epoch**.

**Net benefit over 100 epochs**:
```
Current: 100 × 89.6s = 8,960s
Zarr: 120s (gen) + 5s (first load) + 99 × 1s = 224s

Speedup: 8,960 / 224 = 40x
```

### Migration Strategy

**Phase 1**: Add Zarr cache alongside existing cache
**Phase 2**: Migrate experiments gradually
**Phase 3**: Deprecate old cache format
**Phase 4**: Remove old cache code

**Backward compatibility**:
- Keep old cache format for 6 months
- Auto-detect cache version
- Provide migration tool

### Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Zarr compatibility issues | Medium | High | Extensive testing, version pinning |
| Cache format changes | High | Medium | Migration tools, versioning |
| Storage overhead | Medium | Medium | Compression, cleanup tools |
| Breaking changes | High | High | Gradual migration, feature flags |

---

## Combined Impact Analysis

### Cumulative Speedups

**300k samples, starting from current (polars-optimized): 89.6s**

| Optimization | Time | Speedup vs Previous | Cumulative Speedup |
|--------------|------|--------------------|--------------------|
| Baseline (polars) | 89.6s | 1.0x | 1.0x |
| + Parallel (4 cores) | 25.6s | 3.5x | 3.5x |
| + Batch API | 6.4s | 4.0x | 14.0x |
| + Zarr (first load) | 5.0s | 1.3x | 17.9x |
| + Zarr (cached) | 0.9s | 5.6x | 99.6x |

### Training Impact

**Single experiment, 100 epochs**:

```
Without optimizations:
  Generation: 89.6s
  Loading per epoch: 89.6s
  Total: 89.6 + 100 × 89.6 = 9,049s (2.5 hours)

With all optimizations:
  Generation: 120s (one-time, Zarr)
  First epoch: 5s
  Subsequent: 1s per epoch
  Total: 120 + 5 + 99 × 1 = 224s (3.7 minutes)

Time saved: 8,825s = 2.45 hours = 147 minutes

Speedup: 40x
```

**Suite of 10 experiments**:
```
Without: 10 × 9,049s = 90,490s (25.1 hours)
With: 10 × 224s = 2,240s (37 minutes)

Time saved: 24.5 hours
```

---

## Implementation Roadmap

### Timeline Overview

```
Month 1-2: Parallel Entity Processing
  ├─ Week 1-2: Implementation
  ├─ Week 3: Testing & benchmarking
  └─ Week 4: Deployment & monitoring

Month 3-4: Batch Dataset API
  ├─ Week 1-2: API design & implementation
  ├─ Week 3-4: Integration & testing
  └─ Week 5-6: Migration & documentation

Month 5-7: Zarr Architecture
  ├─ Week 1-3: Core implementation
  ├─ Week 4-6: Dataset integration
  ├─ Week 7-9: Testing & optimization
  └─ Week 10-12: Migration tools & deployment
```

### Phased Rollout

#### Phase 1: Parallel Processing (Immediate Value)

**Effort**: 2-3 weeks
**Risk**: Low
**Value**: High (2-4x speedup)

**Milestones**:
- Week 1: Implement `ParallelTensorCacheGenerator`
- Week 2: Testing & benchmarking
- Week 3: Deploy with feature flag

**Success Criteria**:
- 2x speedup on 4-core machines
- No regression in cache quality
- Stable in production for 1 month

#### Phase 2: Batch API (Medium-Term)

**Effort**: 4-6 weeks
**Risk**: Medium
**Value**: High (3-5x additional speedup)

**Milestones**:
- Week 1-2: Design & implement `get_batch()` API
- Week 3-4: Integrate with tensor cache
- Week 5: Testing & optimization
- Week 6: Migration & docs

**Success Criteria**:
- 3x speedup over parallel processing
- Backward compatible
- <5% additional memory usage

#### Phase 3: Zarr Architecture (Long-Term)

**Effort**: 8-12 weeks
**Risk**: High
**Value**: Very High (10-50x additional speedup)

**Milestones**:
- Week 1-3: Implement Zarr generator
- Week 4-6: Implement Zarr dataset
- Week 7-9: Integration & testing
- Week 10: Migration tools
- Week 11: Documentation & examples
- Week 12: Gradual rollout

**Success Criteria**:
- 10x speedup over batch API
- <2x storage overhead
- Stable for 3 months before deprecating old format

---

## Risk Analysis

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Multiprocessing bugs | High | Medium | Extensive testing, feature flags |
| Memory overhead | Medium | High | Configurable batch sizes, monitoring |
| Platform incompatibility | Medium | Low | Test on Linux/Mac/Windows |
| API complexity | Medium | High | Clear docs, examples, gradual migration |
| Data corruption | Critical | Low | Checksums, validation, backups |
| Performance regression | High | Medium | Benchmarking, A/B testing |

### Organizational Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Breaking changes | High | High | Gradual migration, backward compat |
| User adoption | Medium | Medium | Clear value prop, examples |
| Maintenance burden | Medium | High | Good docs, automated tests |
| Resource availability | High | Medium | Phased approach, prioritization |

### Mitigation Strategies

1. **Feature flags**: All optimizations behind flags
2. **Backward compatibility**: Old code continues to work
3. **Gradual rollout**: Phase-by-phase deployment
4. **Extensive testing**: Unit, integration, performance tests
5. **Monitoring**: Track performance metrics in production
6. **Documentation**: Clear migration guides
7. **Rollback plan**: Quick revert to previous version

---

## Success Metrics

### Performance Metrics

- **Throughput**: Samples processed per second
- **Latency**: Time to generate cache
- **Scaling**: Speedup vs number of cores
- **Memory**: Peak memory usage
- **Disk**: Storage overhead

### Quality Metrics

- **Correctness**: Cache matches original data (100%)
- **Reliability**: Success rate (>99.9%)
- **Compatibility**: Works on all platforms (100%)

### User Metrics

- **Adoption**: % of experiments using new cache
- **Satisfaction**: User survey scores
- **Issues**: Bug reports, support tickets

---

## Conclusion

### Summary

Three major optimizations can provide **40-100x speedup**:

1. **Parallel entity processing**: 2-4x, low risk, 2-3 weeks
2. **Batch dataset API**: 3-5x, medium risk, 4-6 weeks
3. **Zarr architecture**: 10-50x, high risk, 8-12 weeks

### Recommended Approach

**Start with Phase 1** (parallel processing):
- Immediate value (2-4x speedup)
- Low risk, low effort
- No breaking changes

**Evaluate before Phase 2**:
- If 2-4x is sufficient, stop here
- If more performance needed, proceed to batch API
- If training bottleneck remains, proceed to Zarr

### Decision Points

**After Phase 1** (2-4x speedup):
- ✅ Proceed to Phase 2 if: Training is still slow, team has capacity
- ❌ Stop if: Performance is acceptable, other priorities

**After Phase 2** (6-20x speedup):
- ✅ Proceed to Phase 3 if: Training at scale is critical, long-term investment
- ❌ Stop if: Performance is good enough, risk too high

### Next Steps

1. **Approve Phase 1**: Parallel entity processing
2. **Allocate resources**: 1 engineer, 2-3 weeks
3. **Set up benchmarking**: Establish baseline metrics
4. **Implement & test**: Follow implementation plan
5. **Deploy with feature flag**: Gradual rollout
6. **Evaluate**: Measure impact, decide on Phase 2
