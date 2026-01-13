"""
Profiling utilities for dataloader performance analysis.

This module provides instrumentation for identifying CPU-bound bottlenecks
in the data loading pipeline, particularly in __getitem__ operations.

Usage:
    from data_provider.profiling import DataloaderProfiler, profile_operation

    # Enable profiling globally
    DataloaderProfiler.enable()

    # Run your training/data loading code...

    # Get results
    DataloaderProfiler.print_summary()
    DataloaderProfiler.save_results("profile_results.json")
"""

import time
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
from contextlib import contextmanager
import functools
import numpy as np


class DataloaderProfiler:
    """
    Global profiler for dataloader operations.

    Thread-safe accumulator for timing data across multiple workers.
    """

    _enabled: bool = False
    _lock = threading.Lock()
    _timings: Dict[str, List[float]] = defaultdict(list)
    _call_counts: Dict[str, int] = defaultdict(int)
    _start_time: Optional[float] = None
    _total_samples: int = 0

    @classmethod
    def enable(cls):
        """Enable profiling globally."""
        with cls._lock:
            cls._enabled = True
            cls._start_time = time.perf_counter()
            cls._timings.clear()
            cls._call_counts.clear()
            cls._total_samples = 0

    @classmethod
    def disable(cls):
        """Disable profiling globally."""
        with cls._lock:
            cls._enabled = False

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if profiling is enabled."""
        return cls._enabled

    @classmethod
    def record(cls, operation: str, duration: float):
        """Record a timing measurement (thread-safe)."""
        if not cls._enabled:
            return
        with cls._lock:
            cls._timings[operation].append(duration)
            cls._call_counts[operation] += 1

    @classmethod
    def record_sample(cls):
        """Record that a sample was processed."""
        if not cls._enabled:
            return
        with cls._lock:
            cls._total_samples += 1

    @classmethod
    def get_stats(cls) -> Dict[str, Dict[str, float]]:
        """Get statistics for all recorded operations."""
        with cls._lock:
            stats = {}
            for op, times in cls._timings.items():
                if times:
                    times_arr = np.array(times)
                    stats[op] = {
                        'count': len(times),
                        'total_ms': np.sum(times_arr) * 1000,
                        'mean_us': np.mean(times_arr) * 1e6,
                        'std_us': np.std(times_arr) * 1e6,
                        'min_us': np.min(times_arr) * 1e6,
                        'max_us': np.max(times_arr) * 1e6,
                        'median_us': np.median(times_arr) * 1e6,
                        'p95_us': np.percentile(times_arr, 95) * 1e6,
                        'p99_us': np.percentile(times_arr, 99) * 1e6,
                    }
            return stats

    @classmethod
    def print_summary(cls):
        """Print a formatted summary of profiling results."""
        stats = cls.get_stats()

        if not stats:
            print("No profiling data collected. Did you enable profiling?")
            return

        elapsed = time.perf_counter() - cls._start_time if cls._start_time else 0

        print("\n" + "=" * 80)
        print("DATALOADER PROFILING SUMMARY")
        print("=" * 80)
        print(f"Total samples processed: {cls._total_samples:,}")
        print(f"Total elapsed time: {elapsed:.2f} seconds")
        if cls._total_samples > 0:
            print(f"Avg throughput: {cls._total_samples / elapsed:.1f} samples/sec")
        print()

        # Sort by total time (descending)
        sorted_ops = sorted(stats.items(), key=lambda x: x[1]['total_ms'], reverse=True)

        # Calculate total overhead
        total_overhead_ms = sum(s['total_ms'] for _, s in sorted_ops)

        print(f"{'Operation':<35} {'Count':>10} {'Total (ms)':>12} {'%':>6} {'Mean (μs)':>12} {'P95 (μs)':>12}")
        print("-" * 95)

        for op, s in sorted_ops:
            pct = (s['total_ms'] / total_overhead_ms * 100) if total_overhead_ms > 0 else 0
            print(f"{op:<35} {s['count']:>10,} {s['total_ms']:>12.1f} {pct:>5.1f}% {s['mean_us']:>12.1f} {s['p95_us']:>12.1f}")

        print("-" * 95)
        print(f"{'TOTAL':<35} {'':<10} {total_overhead_ms:>12.1f}")
        print()

        # Per-sample breakdown
        if cls._total_samples > 0:
            print(f"Per-sample overhead: {total_overhead_ms / cls._total_samples * 1000:.1f} μs")
            print()

            # Top bottlenecks
            print("TOP BOTTLENECKS (by total time):")
            for i, (op, s) in enumerate(sorted_ops[:5], 1):
                pct = (s['total_ms'] / total_overhead_ms * 100) if total_overhead_ms > 0 else 0
                print(f"  {i}. {op}: {s['total_ms']:.1f}ms ({pct:.1f}%) - {s['mean_us']:.1f}μs/call")

        print("=" * 80)

    @classmethod
    def save_results(cls, filepath: str):
        """Save profiling results to JSON file."""
        stats = cls.get_stats()
        elapsed = time.perf_counter() - cls._start_time if cls._start_time else 0

        results = {
            'total_samples': cls._total_samples,
            'elapsed_seconds': elapsed,
            'throughput_samples_per_sec': cls._total_samples / elapsed if elapsed > 0 else 0,
            'operations': stats
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Profiling results saved to: {filepath}")

    @classmethod
    def reset(cls):
        """Reset all profiling data."""
        with cls._lock:
            cls._timings.clear()
            cls._call_counts.clear()
            cls._total_samples = 0
            cls._start_time = None


@contextmanager
def timed_operation(name: str):
    """
    Context manager for timing an operation.

    Usage:
        with timed_operation("my_operation"):
            # code to time
    """
    if not DataloaderProfiler.is_enabled():
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        DataloaderProfiler.record(name, duration)


def profile_operation(name: str):
    """
    Decorator for profiling a function.

    Usage:
        @profile_operation("my_function")
        def my_function():
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not DataloaderProfiler.is_enabled():
                return func(*args, **kwargs)

            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            DataloaderProfiler.record(name, duration)
            return result
        return wrapper
    return decorator


class BatchProfiler:
    """
    Profiler for batch-level timing (DataLoader iteration).

    Usage:
        profiler = BatchProfiler()
        for batch in dataloader:
            profiler.record_batch()
            # process batch
        profiler.print_summary()
    """

    def __init__(self):
        self.batch_times: List[float] = []
        self.gpu_transfer_times: List[float] = []
        self.start_time: Optional[float] = None
        self.last_batch_end: Optional[float] = None

    def start(self):
        """Start the profiler."""
        self.start_time = time.perf_counter()
        self.last_batch_end = self.start_time

    def record_batch_start(self):
        """Record the start of batch fetching (after previous batch processing)."""
        if self.last_batch_end is None:
            self.start()

    def record_batch_ready(self):
        """Record when a batch is ready from the DataLoader."""
        now = time.perf_counter()
        if self.last_batch_end is not None:
            self.batch_times.append(now - self.last_batch_end)

    def record_gpu_transfer(self, duration: float):
        """Record GPU transfer time."""
        self.gpu_transfer_times.append(duration)

    def record_batch_end(self):
        """Record the end of batch processing."""
        self.last_batch_end = time.perf_counter()

    def print_summary(self):
        """Print batch timing summary."""
        if not self.batch_times:
            print("No batch timing data collected.")
            return

        batch_arr = np.array(self.batch_times) * 1000  # Convert to ms

        print("\n" + "=" * 60)
        print("BATCH TIMING SUMMARY")
        print("=" * 60)
        print(f"Total batches: {len(self.batch_times)}")
        print(f"Batch fetch time (ms):")
        print(f"  Mean:   {np.mean(batch_arr):.2f}")
        print(f"  Std:    {np.std(batch_arr):.2f}")
        print(f"  Min:    {np.min(batch_arr):.2f}")
        print(f"  Max:    {np.max(batch_arr):.2f}")
        print(f"  P50:    {np.percentile(batch_arr, 50):.2f}")
        print(f"  P95:    {np.percentile(batch_arr, 95):.2f}")
        print(f"  P99:    {np.percentile(batch_arr, 99):.2f}")

        if self.gpu_transfer_times:
            gpu_arr = np.array(self.gpu_transfer_times) * 1000
            print(f"\nGPU transfer time (ms):")
            print(f"  Mean:   {np.mean(gpu_arr):.2f}")
            print(f"  Max:    {np.max(gpu_arr):.2f}")

        total_batch_time = np.sum(batch_arr)
        total_gpu_time = np.sum(self.gpu_transfer_times) * 1000 if self.gpu_transfer_times else 0

        print(f"\nTime breakdown:")
        print(f"  Data loading: {total_batch_time:.1f}ms ({total_batch_time/(total_batch_time+total_gpu_time)*100:.1f}%)")
        print(f"  GPU transfer: {total_gpu_time:.1f}ms ({total_gpu_time/(total_batch_time+total_gpu_time)*100:.1f}%)")
        print("=" * 60)
