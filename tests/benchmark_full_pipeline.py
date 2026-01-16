"""
Full pipeline benchmark for tensor cache generation: polars vs current implementation.

This script compares the performance of the polars-optimized implementation
against the standard dict-based implementation on a large mock dataset.

Usage:
    python tests/benchmark_full_pipeline.py
"""

import sys
import time
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_provider.tensor_cache import TensorCacheGenerator
from tests.test_tensor_cache_polars import MockDataset


class MockDataProvider:
    """Mock data provider for benchmarking."""

    def __init__(self, datasets_dict: Dict[str, Any]):
        self.datasets = datasets_dict

    def get_datasets(self, flag: str):
        """Return datasets for the given flag."""
        return self.datasets.get(flag, {})


def benchmark_tensor_cache_generation(
    n_samples: int = 100000,
    input_len: int = 96,
    output_len: int = 48,
    embed_dim: int = 768,
    use_polars: bool = True
):
    """
    Benchmark full tensor cache generation pipeline.

    Args:
        n_samples: Number of samples in dataset
        input_len: Input window length
        output_len: Output window length
        embed_dim: Embedding dimension
        use_polars: Whether to use polars optimization

    Returns:
        Dict with timing breakdown
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking Tensor Cache Generation")
    print(f"  Implementation: {'polars-optimized' if use_polars else 'standard (dict-based)'}")
    print(f"  Samples: {n_samples:,}")
    print(f"  Input length: {input_len}")
    print(f"  Output length: {output_len}")
    print(f"  Embedding dim: {embed_dim}")
    print(f"{'='*80}\n")

    # Create mock dataset
    print("Creating mock dataset...")
    dataset = MockDataset(
        n_samples=n_samples,
        input_len=input_len,
        output_len=output_len,
        embed_dim=embed_dim,
        n_features=3,
        n_hetero_time_features=4,
        n_time_features=4,
        seed=42
    )
    print(f"  Dataset created: {len(dataset):,} samples")
    print(f"  Expected unique timestamps: ~{n_samples + input_len + output_len:,}")
    print(f"  Total timestamp references: {n_samples * (input_len + output_len):,}")

    # Create mock data provider
    data_provider = MockDataProvider({
        'train': {'entity_0': dataset},
        'val': {},
        'test': {}
    })

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"

        # Mock config
        config = {
            'dataset': 'mock',
            'input_len': input_len,
            'output_len': output_len,
            'features': 'M',
            'target': 'OT',
            'scale': True
        }

        # Create generator
        generator = TensorCacheGenerator(
            data_provider=data_provider,
            cache_dir=cache_dir,
            config=config,
            chunk_size=10000,
            verbose=True,
            use_polars=use_polars
        )

        # Benchmark: Shared table building
        print("\nPhase 1: Building shared tables (collecting unique data)...")
        start_time = time.perf_counter()

        shared_tables, index_mappings = generator._build_shared_tables(['train'])

        shared_tables_time = time.perf_counter() - start_time
        print(f"  ✓ Shared tables built in {shared_tables_time:.2f}s")
        print(f"    - Unique timestamps: {len(shared_tables.get('timestamps', [])):,}")
        print(f"    - Timeseries shape: {shared_tables.get('timeseries', np.array([])).shape}")
        print(f"    - Embeddings shape: {shared_tables.get('embeddings', np.array([])).shape}")

        # Calculate deduplication ratio
        n_unique = len(shared_tables.get('timestamps', []))
        n_total_refs = n_samples * (input_len + output_len)
        dedup_ratio = n_total_refs / n_unique if n_unique > 0 else 0
        print(f"    - Deduplication ratio: {dedup_ratio:.1f}:1")

        # Benchmark: Index array generation (simulate)
        print("\nPhase 2: Generating index arrays...")
        start_time = time.perf_counter()

        # Simulate index generation for all samples
        timestamp_to_idx = index_mappings['timestamp_to_idx']
        entity_to_idx = index_mappings['entity_to_idx']

        # Count index conversions
        n_conversions = 0
        for i in range(min(1000, n_samples)):  # Sample 1000 to estimate
            sample = dataset[i]
            x_time = sample[3].flatten()
            y_time = sample[4].flatten()

            # Simulate index lookup (this is what happens in _write_sample_indices)
            if use_polars:
                # With polars: vectorized join (simulated here with dict for simplicity)
                x_indices = np.array([timestamp_to_idx.get(str(int(ts)), 0) for ts in x_time])
                y_indices = np.array([timestamp_to_idx.get(str(int(ts)), 0) for ts in y_time])
            else:
                # Standard: list comprehension
                x_indices = np.array([timestamp_to_idx.get(str(int(ts)), 0) for ts in x_time])
                y_indices = np.array([timestamp_to_idx.get(str(int(ts)), 0) for ts in y_time])

            n_conversions += len(x_time) + len(y_time)

        index_gen_time = time.perf_counter() - start_time

        # Extrapolate to full dataset
        estimated_total_time = index_gen_time * (n_samples / 1000)
        print(f"  ✓ Index generation (1000 samples) in {index_gen_time:.2f}s")
        print(f"    - Estimated full dataset: {estimated_total_time:.2f}s")
        print(f"    - Conversions performed: {n_conversions:,}")

        # Total time
        total_time = shared_tables_time + estimated_total_time

        print(f"\n{'='*80}")
        print(f"TOTAL TIME: {total_time:.2f}s")
        print(f"  - Shared table building: {shared_tables_time:.2f}s ({shared_tables_time/total_time*100:.1f}%)")
        print(f"  - Index generation (est): {estimated_total_time:.2f}s ({estimated_total_time/total_time*100:.1f}%)")
        print(f"{'='*80}\n")

        return {
            'implementation': 'polars' if use_polars else 'standard',
            'n_samples': n_samples,
            'n_unique_timestamps': n_unique,
            'deduplication_ratio': dedup_ratio,
            'shared_tables_time': shared_tables_time,
            'index_generation_time': estimated_total_time,
            'total_time': total_time
        }


def main():
    """Run benchmarks and compare implementations."""

    # Configuration
    n_samples = 100000
    input_len = 96
    output_len = 48
    embed_dim = 768

    print("\n" + "="*80)
    print("TENSOR CACHE GENERATION BENCHMARK")
    print("="*80)

    # Run standard implementation
    print("\n\n>>> RUNNING STANDARD IMPLEMENTATION (dict-based)")
    standard_results = benchmark_tensor_cache_generation(
        n_samples=n_samples,
        input_len=input_len,
        output_len=output_len,
        embed_dim=embed_dim,
        use_polars=False
    )

    # Run polars implementation
    print("\n\n>>> RUNNING POLARS IMPLEMENTATION")
    polars_results = benchmark_tensor_cache_generation(
        n_samples=n_samples,
        input_len=input_len,
        output_len=output_len,
        embed_dim=embed_dim,
        use_polars=True
    )

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    print(f"\nDataset:")
    print(f"  Samples: {n_samples:,}")
    print(f"  Unique timestamps: {standard_results['n_unique_timestamps']:,}")
    print(f"  Deduplication ratio: {standard_results['deduplication_ratio']:.1f}:1")

    print(f"\nShared Table Building:")
    print(f"  Standard: {standard_results['shared_tables_time']:.2f}s")
    print(f"  Polars:   {polars_results['shared_tables_time']:.2f}s")
    speedup = standard_results['shared_tables_time'] / polars_results['shared_tables_time']
    print(f"  Speedup:  {speedup:.2f}x {'✓' if speedup > 1 else '✗'}")

    print(f"\nIndex Generation (estimated):")
    print(f"  Standard: {standard_results['index_generation_time']:.2f}s")
    print(f"  Polars:   {polars_results['index_generation_time']:.2f}s")
    speedup = standard_results['index_generation_time'] / polars_results['index_generation_time']
    print(f"  Speedup:  {speedup:.2f}x {'✓' if speedup > 1 else '✗'}")

    print(f"\nTotal Time:")
    print(f"  Standard: {standard_results['total_time']:.2f}s")
    print(f"  Polars:   {polars_results['total_time']:.2f}s")
    overall_speedup = standard_results['total_time'] / polars_results['total_time']
    print(f"  Speedup:  {overall_speedup:.2f}x {'✓' if overall_speedup > 1 else '✗'}")

    if overall_speedup > 1:
        time_saved = standard_results['total_time'] - polars_results['total_time']
        print(f"  Time saved: {time_saved:.2f}s ({time_saved/standard_results['total_time']*100:.1f}%)")

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80 + "\n")

    # Write results to file
    results_file = Path(__file__).parent.parent / "docs/planning/benchmark_results_100k.txt"
    with open(results_file, 'w') as f:
        f.write("Tensor Cache Generation Benchmark Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: {n_samples:,} samples, {input_len} input + {output_len} output, {embed_dim}D embeddings\n")
        f.write(f"Unique timestamps: {standard_results['n_unique_timestamps']:,}\n")
        f.write(f"Deduplication ratio: {standard_results['deduplication_ratio']:.1f}:1\n\n")
        f.write("Timing Results:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Phase':<30} {'Standard':<15} {'Polars':<15} {'Speedup':<10}\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Shared table building':<30} {standard_results['shared_tables_time']:>10.2f}s    {polars_results['shared_tables_time']:>10.2f}s    {standard_results['shared_tables_time'] / polars_results['shared_tables_time']:>6.2f}x\n")
        f.write(f"{'Index generation (est)':<30} {standard_results['index_generation_time']:>10.2f}s    {polars_results['index_generation_time']:>10.2f}s    {standard_results['index_generation_time'] / polars_results['index_generation_time']:>6.2f}x\n")
        f.write(f"{'TOTAL':<30} {standard_results['total_time']:>10.2f}s    {polars_results['total_time']:>10.2f}s    {overall_speedup:>6.2f}x\n")
        f.write("-"*80 + "\n")

    print(f"Results written to: {results_file}")


if __name__ == '__main__':
    main()
