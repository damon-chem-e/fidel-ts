#!/usr/bin/env python
"""
Benchmark script for DirectAccessMixin vs __getitem__ iteration.

This script compares the performance of:
1. Per-sample __getitem__ iteration (current approach)
2. Direct array access (new optimized approach)

Run with:
    source .venv/bin/activate
    python scripts/benchmark_direct_access.py
"""

import gc
import time
import logging
import numpy as np
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_provider.dataset_direct_access import (
    DirectAccessMixin,
    build_shared_tables_direct,
)
from data_provider.tensor_cache_polars import (
    PolarsCollectorState,
    process_sample_for_collection_polars,
    register_entity_data_polars,
    finalize_shared_tables_polars,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkDataset(DirectAccessMixin):
    """
    Mock dataset for benchmarking with realistic dimensions.

    Uses:
    - 5 time series features
    - 768-dim embeddings (BERT-like)
    - 4 hetero time features
    - seq_len=288, pred_len=144 (realistic for hourly data with daily patterns)
    """

    def __init__(self, n_timestamps: int, entity_id: str = "test_entity"):
        """
        Initialize benchmark dataset.

        Args:
            n_timestamps: Total number of timestamps (determines n_samples)
            entity_id: Entity identifier
        """
        self.data = np.random.randn(n_timestamps, 5).astype(np.float32)
        self.timestamp = np.arange(
            20200101000000,  # Start timestamp
            20200101000000 + n_timestamps,
            dtype=np.int64
        )
        self.seq_len = 288  # 288 hours = 12 days at hourly
        self.pred_len = 144  # 144 hours = 6 days
        self.stride = 1
        self.entity_id = entity_id

        # Hetero data
        self.full_hetero = np.random.randn(n_timestamps, 768).astype(np.float32)
        self.hetero_time = np.random.randn(n_timestamps, 4).astype(np.float32)
        self.hetero_general = np.random.randn(768).astype(np.float32)
        self.hetero_channel = np.random.randn(768).astype(np.float32)
        self.hetero_stride = 1

        # Preload flag (for tensor cache)
        self.preload_hetero = True

    def __len__(self):
        length = len(self.data) - self.seq_len - self.pred_len + 1
        return max(0, length)

    def __getitem__(self, idx):
        """Standard __getitem__ as in Universal_Dataset."""
        x_start = idx
        x_end = x_start + self.seq_len
        y_start = x_end
        y_end = y_start + self.pred_len

        return (
            f"{self.entity_id}|{self.timestamp[x_start]}|{idx}",  # sample_id
            self.data[x_start:x_end],                             # seq_x
            self.data[y_start:y_end],                             # seq_y
            self.timestamp[x_start:x_end],                        # x_time
            self.timestamp[y_start:y_end],                        # y_time
            self.full_hetero[x_start:x_end],                      # hetero_x
            self.full_hetero[y_start:y_end],                      # hetero_y
            self.hetero_time[x_start:x_end],                      # hetero_x_time
            self.hetero_time[y_start:y_end],                      # hetero_y_time
            self.hetero_general.copy(),                           # hetero_general
            self.hetero_channel.copy(),                           # hetero_channel
            np.array([]).astype(np.float32),                      # x_time_features
            np.array([]).astype(np.float32),                      # y_time_features
        )


def estimate_memory_gb(n_timestamps: int) -> float:
    """Estimate memory needed for dataset in GB."""
    data_bytes = n_timestamps * 5 * 4  # data: (N, 5) float32
    ts_bytes = n_timestamps * 8  # timestamps: (N,) int64
    emb_bytes = n_timestamps * 768 * 4  # embeddings: (N, 768) float32
    htf_bytes = n_timestamps * 4 * 4  # hetero_time: (N, 4) float32

    total_bytes = data_bytes + ts_bytes + emb_bytes + htf_bytes
    return total_bytes / (1024 ** 3)


def benchmark_getitem_iteration(dataset, num_samples: int = None) -> Tuple[float, dict]:
    """
    Benchmark per-sample __getitem__ iteration.

    Returns:
        Tuple of (time_seconds, results_dict)
    """
    if num_samples is None:
        num_samples = len(dataset)

    gc.collect()

    start = time.perf_counter()

    # Collect timestamps (simulating shared table building)
    seen_ts = set()
    timestamps = []
    timeseries = []
    embeddings = []

    for i in range(num_samples):
        sample = dataset[i]
        x_time = sample[3].flatten()
        y_time = sample[4].flatten()
        seq_x = sample[1]
        seq_y = sample[2]
        hetero_x = sample[5]
        hetero_y = sample[6]

        for j, ts in enumerate(x_time):
            ts_int = int(ts)
            if ts_int not in seen_ts:
                seen_ts.add(ts_int)
                timestamps.append(ts_int)
                timeseries.append(seq_x[j].copy())
                embeddings.append(hetero_x[j].copy())

        for j, ts in enumerate(y_time):
            ts_int = int(ts)
            if ts_int not in seen_ts:
                seen_ts.add(ts_int)
                timestamps.append(ts_int)
                timeseries.append(seq_y[j].copy())
                embeddings.append(hetero_y[j].copy())

    elapsed = time.perf_counter() - start

    return elapsed, {
        'n_unique_ts': len(timestamps),
        'n_samples': num_samples,
        'method': '__getitem__ iteration'
    }


def benchmark_polars_collection(dataset, num_samples: int = None) -> Tuple[float, dict]:
    """
    Benchmark polars-based collection (current implementation).

    Returns:
        Tuple of (time_seconds, results_dict)
    """
    if num_samples is None:
        num_samples = len(dataset)

    gc.collect()

    start = time.perf_counter()

    state = PolarsCollectorState()
    state.num_news_items = 1

    # Register entity
    first_sample = dataset[0]
    register_entity_data_polars(state, dataset.entity_id, first_sample[9], first_sample[10])

    # Process samples
    for i in range(num_samples):
        sample = dataset[i]
        process_sample_for_collection_polars(state, sample)

    # Finalize
    shared_tables, index_mappings = finalize_shared_tables_polars(state)

    elapsed = time.perf_counter() - start

    return elapsed, {
        'n_unique_ts': len(shared_tables['timestamps']),
        'n_samples': num_samples,
        'method': 'polars collection'
    }


def benchmark_direct_access(datasets: Dict[str, BenchmarkDataset]) -> Tuple[float, dict]:
    """
    Benchmark direct array access (new optimized approach).

    Returns:
        Tuple of (time_seconds, results_dict)
    """
    gc.collect()

    start = time.perf_counter()

    shared_tables, index_mappings = build_shared_tables_direct(
        datasets,
        num_news_items=1,
        verbose=False
    )

    elapsed = time.perf_counter() - start

    total_samples = sum(len(ds) for ds in datasets.values())

    return elapsed, {
        'n_unique_ts': len(shared_tables['timestamps']),
        'n_samples': total_samples,
        'method': 'direct array access'
    }


def run_benchmark(n_timestamps: int, log_file: Path) -> dict:
    """
    Run full benchmark for a given dataset size.

    Args:
        n_timestamps: Number of timestamps in dataset
        log_file: Path to log file for results

    Returns:
        Dict with benchmark results
    """
    mem_estimate = estimate_memory_gb(n_timestamps)

    logger.info(f"=" * 60)
    logger.info(f"Benchmarking with {n_timestamps:,} timestamps")
    logger.info(f"Estimated memory: {mem_estimate:.2f} GB")

    with open(log_file, 'a') as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"n_timestamps: {n_timestamps:,}\n")
        f.write(f"Estimated memory: {mem_estimate:.2f} GB\n")

    # Create dataset
    logger.info("Creating dataset...")
    dataset = BenchmarkDataset(n_timestamps)
    n_samples = len(dataset)
    logger.info(f"Dataset has {n_samples:,} samples")

    with open(log_file, 'a') as f:
        f.write(f"n_samples: {n_samples:,}\n\n")

    results = {}

    # Benchmark 1: __getitem__ iteration
    logger.info("Benchmarking __getitem__ iteration...")
    try:
        time_getitem, info_getitem = benchmark_getitem_iteration(dataset)
        results['getitem'] = {
            'time': time_getitem,
            'samples_per_sec': n_samples / time_getitem,
            **info_getitem
        }
        logger.info(f"  Time: {time_getitem:.2f}s ({n_samples / time_getitem:.0f} samples/sec)")

        with open(log_file, 'a') as f:
            f.write(f"__getitem__ iteration:\n")
            f.write(f"  Time: {time_getitem:.2f}s\n")
            f.write(f"  Samples/sec: {n_samples / time_getitem:.0f}\n")
            f.write(f"  Unique timestamps: {info_getitem['n_unique_ts']:,}\n\n")
    except Exception as e:
        logger.error(f"  Failed: {e}")
        results['getitem'] = {'error': str(e)}

    gc.collect()

    # Benchmark 2: Polars collection (current implementation)
    logger.info("Benchmarking polars collection...")
    try:
        time_polars, info_polars = benchmark_polars_collection(dataset)
        results['polars'] = {
            'time': time_polars,
            'samples_per_sec': n_samples / time_polars,
            **info_polars
        }
        logger.info(f"  Time: {time_polars:.2f}s ({n_samples / time_polars:.0f} samples/sec)")

        with open(log_file, 'a') as f:
            f.write(f"Polars collection:\n")
            f.write(f"  Time: {time_polars:.2f}s\n")
            f.write(f"  Samples/sec: {n_samples / time_polars:.0f}\n")
            f.write(f"  Unique timestamps: {info_polars['n_unique_ts']:,}\n\n")
    except Exception as e:
        logger.error(f"  Failed: {e}")
        results['polars'] = {'error': str(e)}

    gc.collect()

    # Benchmark 3: Direct access (new approach)
    logger.info("Benchmarking direct array access...")
    try:
        datasets_dict = {'entity_1': dataset}
        time_direct, info_direct = benchmark_direct_access(datasets_dict)
        results['direct'] = {
            'time': time_direct,
            'samples_per_sec': n_samples / time_direct,
            **info_direct
        }
        logger.info(f"  Time: {time_direct:.2f}s ({n_samples / time_direct:.0f} samples/sec)")

        with open(log_file, 'a') as f:
            f.write(f"Direct array access:\n")
            f.write(f"  Time: {time_direct:.2f}s\n")
            f.write(f"  Samples/sec: {n_samples / time_direct:.0f}\n")
            f.write(f"  Unique timestamps: {info_direct['n_unique_ts']:,}\n\n")
    except Exception as e:
        logger.error(f"  Failed: {e}")
        results['direct'] = {'error': str(e)}

    # Calculate speedups
    if 'time' in results.get('getitem', {}) and 'time' in results.get('direct', {}):
        speedup_vs_getitem = results['getitem']['time'] / results['direct']['time']
        results['speedup_vs_getitem'] = speedup_vs_getitem
        logger.info(f"Speedup (direct vs __getitem__): {speedup_vs_getitem:.1f}x")

        with open(log_file, 'a') as f:
            f.write(f"SPEEDUP (direct vs __getitem__): {speedup_vs_getitem:.1f}x\n")

    if 'time' in results.get('polars', {}) and 'time' in results.get('direct', {}):
        speedup_vs_polars = results['polars']['time'] / results['direct']['time']
        results['speedup_vs_polars'] = speedup_vs_polars
        logger.info(f"Speedup (direct vs polars): {speedup_vs_polars:.1f}x")

        with open(log_file, 'a') as f:
            f.write(f"SPEEDUP (direct vs polars): {speedup_vs_polars:.1f}x\n")

    with open(log_file, 'a') as f:
        f.write(f"{'=' * 60}\n")

    # Clean up
    del dataset
    gc.collect()

    return results


def main():
    """Run benchmarks at various scales."""
    log_file = Path("logs/benchmark_direct_access.log")
    log_file.parent.mkdir(exist_ok=True)

    with open(log_file, 'w') as f:
        f.write(f"DirectAccessMixin Benchmark Results\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")
        f.write(f"{'=' * 60}\n")

    logger.info("Starting DirectAccessMixin benchmarks")
    logger.info(f"Results will be logged to: {log_file}")

    # Test sizes: 100k, 300k, 500k samples
    # seq_len=288, pred_len=144, so n_timestamps = n_samples + 288 + 144 - 1
    sample_targets = [10_000, 50_000, 100_000, 300_000, 500_000]

    all_results = {}

    for target_samples in sample_targets:
        # Calculate required timestamps
        n_timestamps = target_samples + 288 + 144 - 1
        mem_gb = estimate_memory_gb(n_timestamps)

        # Skip if would use too much memory (16GB safety limit)
        if mem_gb > 14:
            logger.warning(f"Skipping {target_samples:,} samples - would use {mem_gb:.1f}GB")
            continue

        try:
            results = run_benchmark(n_timestamps, log_file)
            all_results[target_samples] = results
        except Exception as e:
            logger.error(f"Benchmark failed for {target_samples:,} samples: {e}")
            all_results[target_samples] = {'error': str(e)}

        gc.collect()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    with open(log_file, 'a') as f:
        f.write(f"\n\n{'=' * 60}\n")
        f.write(f"SUMMARY\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"\n{'Samples':>15} | {'__getitem__':>12} | {'Direct':>12} | {'Speedup':>10}\n")
        f.write(f"{'-' * 55}\n")

    print(f"\n{'Samples':>15} | {'__getitem__':>12} | {'Direct':>12} | {'Speedup':>10}")
    print(f"{'-' * 55}")

    for target_samples, results in all_results.items():
        if 'error' in results:
            print(f"{target_samples:>15,} | {'ERROR':>12} | {'ERROR':>12} | {'N/A':>10}")
            continue

        getitem_time = results.get('getitem', {}).get('time', float('inf'))
        direct_time = results.get('direct', {}).get('time', float('inf'))
        speedup = results.get('speedup_vs_getitem', 0)

        print(f"{target_samples:>15,} | {getitem_time:>11.2f}s | {direct_time:>11.2f}s | {speedup:>9.1f}x")

        with open(log_file, 'a') as f:
            f.write(f"{target_samples:>15,} | {getitem_time:>11.2f}s | {direct_time:>11.2f}s | {speedup:>9.1f}x\n")

    with open(log_file, 'a') as f:
        f.write(f"\nCompleted: {datetime.now().isoformat()}\n")

    logger.info(f"\nResults saved to: {log_file}")


if __name__ == '__main__':
    main()
