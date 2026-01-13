#!/usr/bin/env python
"""
Dataloader Profiling Script

This script profiles the dataloader to identify CPU-bound bottlenecks.
It measures per-operation timing in __getitem__ and batch-level timing.

Usage:
    # Profile a suite config (uses first experiment)
    python scripts/profile_dataloader.py configs/experiment_suites/lynx_film/canada_photovoltaics.yaml

    # Profile with options
    python scripts/profile_dataloader.py configs/experiment_suites/lynx_film/canada_photovoltaics.yaml \
        --num-samples 5000 \
        --num-batches 100 \
        --output profile_results.json

    # Profile specific experiment from suite
    python scripts/profile_dataloader.py configs/experiment_suites/lynx_film/canada_photovoltaics.yaml \
        --experiment "lynx_film_canada_photovoltaics"
"""

import sys
import os
import time
import argparse
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
import numpy as np
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

from data_provider.profiling import DataloaderProfiler, BatchProfiler
from utils.tools import dotdict
from utils.config_utils import merge_configs


console = Console()


def load_suite_config(suite_path: str) -> dict:
    """Load a suite configuration file."""
    with open(suite_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_template(template_path: str) -> dict:
    """Load a template configuration file."""
    with open(template_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def build_args_from_config(config: dict) -> dotdict:
    """
    Build an args object from experiment config.

    This replicates the essential parts of config_to_args from runs/pytorch.py
    """
    args = dotdict()

    # Model config
    args.model = config.get('model', {}).get('name', 'unknown')
    args.model_config = config.get('model', {}).get('config_path', '')

    # Data config - load the actual data config file
    data_config_path = config.get('data', {}).get('config_path', '')
    if data_config_path and Path(data_config_path).exists():
        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)
        args.data_config = dotdict(data_config)
    else:
        args.data_config = dotdict({})

    args.data = config.get('data', {}).get('name', 'unknown')

    # Training config
    training = config.get('training', {})
    args.scale = training.get('scale', True)
    args.disable_buffer = training.get('disable_buffer', False)
    args.preload_hetero = training.get('preload_hetero', False)
    args.prefetch_factor = training.get('prefetch_factor', 2)
    args.noise = training.get('noise', 0.0)
    args.downsample = training.get('downsample', None)
    args.num_workers = training.get('num_workers', 0)
    args.batch_size = training.get('batch_size', 32)
    args.truncate_train_for_purge = training.get('truncate_train_for_purge', False)

    # Task config
    args.ahead = training.get('ahead', None)
    args.output_len = training.get('output_len', 96)
    args.input_len = training.get('input_len', 336)

    # GPU config
    device_config = config.get('device', {})
    args.use_gpu = device_config.get('use_gpu', torch.cuda.is_available())
    args.gpu = device_config.get('gpu', 0)

    # Load model config if available
    if args.model_config and Path(args.model_config).exists():
        with open(args.model_config, 'r') as f:
            model_config = yaml.safe_load(f)
        args.model_config = dotdict(model_config)

    return args


def get_experiment_config(suite_config: dict, experiment_name: str = None) -> dict:
    """
    Get a single experiment config from a suite.

    Args:
        suite_config: Full suite configuration
        experiment_name: Optional name of experiment to use. If None, uses first.

    Returns:
        Merged experiment config
    """
    suite_info = suite_config.get('suite', {})
    experiments = suite_info.get('experiments', [])

    if not experiments:
        raise ValueError("No experiments found in suite config")

    # Find the experiment
    exp_config = None
    if experiment_name:
        for exp in experiments:
            if exp.get('name') == experiment_name:
                exp_config = exp
                break
        if exp_config is None:
            available = [e.get('name') for e in experiments]
            raise ValueError(f"Experiment '{experiment_name}' not found. Available: {available}")
    else:
        exp_config = experiments[0]

    # Load template and merge with overrides
    template_path = exp_config.get('template', '')
    if not template_path:
        raise ValueError(f"No template specified for experiment '{exp_config.get('name')}'")

    template = load_template(template_path)
    overrides = exp_config.get('overrides', {})

    return merge_configs(template, overrides)


def profile_getitem_operations(data_provider, num_samples: int = 1000) -> dict:
    """
    Profile individual __getitem__ operations.

    Returns detailed timing breakdown per operation.
    """
    console.print("\n[bold blue]Profiling __getitem__ operations...[/bold blue]")

    # Enable profiling
    DataloaderProfiler.enable()

    # Get training dataset
    datasets = data_provider.get_datasets('train')

    if not datasets:
        console.print("[red]No datasets found[/red]")
        return {}

    # Use first dataset for profiling
    dataset_id = list(datasets.keys())[0]
    dataset = datasets[dataset_id]

    console.print(f"Dataset: {dataset_id}, Length: {len(dataset)}")

    # Sample indices
    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    # Profile __getitem__ calls
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Profiling samples", total=num_samples)

        for idx in indices:
            _ = dataset[idx]
            DataloaderProfiler.record_sample()
            progress.advance(task)

    # Get and print results
    DataloaderProfiler.print_summary()

    stats = DataloaderProfiler.get_stats()
    DataloaderProfiler.disable()

    return stats


def profile_dataloader_throughput(data_provider, num_batches: int = 100) -> dict:
    """
    Profile DataLoader throughput and batch timing.
    """
    console.print("\n[bold blue]Profiling DataLoader throughput...[/bold blue]")

    batch_profiler = BatchProfiler()
    batch_profiler.start()

    # Get training dataloader
    train_loader = data_provider.get_train(return_type='loader')

    console.print(f"Batch size: {data_provider.batch_size}")
    console.print(f"Num workers: {data_provider.args.num_workers}")
    console.print(f"Prefetch factor: {data_provider.args.prefetch_factor}")

    batch_times = []
    gpu_transfer_times = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Profiling batches", total=num_batches)

        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break

            # Record batch fetch time
            batch_profiler.record_batch_ready()

            # Time GPU transfer if available
            if torch.cuda.is_available():
                start_gpu = time.perf_counter()
                # Move tensors to GPU
                for item in batch:
                    if isinstance(item, torch.Tensor):
                        item.to(device, non_blocking=True)
                torch.cuda.synchronize()
                gpu_time = time.perf_counter() - start_gpu
                batch_profiler.record_gpu_transfer(gpu_time)

            batch_profiler.record_batch_end()
            progress.advance(task)

    batch_profiler.print_summary()

    return {
        'batch_times_ms': [t * 1000 for t in batch_profiler.batch_times],
        'gpu_transfer_times_ms': [t * 1000 for t in batch_profiler.gpu_transfer_times]
    }


def profile_worker_scaling(data_provider, max_workers: int = 8, num_batches: int = 50) -> dict:
    """
    Profile throughput with different num_workers settings.
    """
    console.print("\n[bold blue]Profiling worker scaling...[/bold blue]")

    results = {}
    original_workers = data_provider.args.num_workers

    for num_workers in [0, 1, 2, 4, min(8, max_workers)]:
        if num_workers > max_workers:
            continue

        console.print(f"\n[yellow]Testing num_workers={num_workers}[/yellow]")
        data_provider.args.num_workers = num_workers

        # Re-create dataloader with new settings
        train_loader = data_provider.get_train(return_type='loader')

        times = []
        start = time.perf_counter()

        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break
            times.append(time.perf_counter() - start)
            start = time.perf_counter()

        if times:
            avg_time = np.mean(times[1:]) * 1000 if len(times) > 1 else times[0] * 1000  # Skip first batch
            throughput = num_batches / sum(times)
            results[num_workers] = {
                'avg_batch_time_ms': avg_time,
                'throughput_batches_per_sec': throughput
            }
            console.print(f"  Avg batch time: {avg_time:.2f}ms, Throughput: {throughput:.1f} batches/sec")

    # Restore original
    data_provider.args.num_workers = original_workers

    return results


def main():
    parser = argparse.ArgumentParser(description='Profile dataloader performance')
    parser.add_argument('config_path', type=str, help='Path to suite config YAML')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Name of specific experiment to profile (default: first)')
    parser.add_argument('--num-samples', type=int, default=5000,
                        help='Number of samples to profile for __getitem__ (default: 5000)')
    parser.add_argument('--num-batches', type=int, default=100,
                        help='Number of batches to profile for throughput (default: 100)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--skip-getitem', action='store_true',
                        help='Skip per-sample __getitem__ profiling')
    parser.add_argument('--skip-throughput', action='store_true',
                        help='Skip DataLoader throughput profiling')
    parser.add_argument('--worker-scaling', action='store_true',
                        help='Profile with different num_workers settings')
    parser.add_argument('--max-workers', type=int, default=8,
                        help='Maximum workers to test in scaling (default: 8)')

    args = parser.parse_args()

    console.print("[bold green]Dataloader Profiling Script[/bold green]")
    console.print(f"Config: {args.config_path}")

    # Load suite config
    suite_config = load_suite_config(args.config_path)

    # Get experiment config
    exp_config = get_experiment_config(suite_config, args.experiment)
    exp_name = exp_config.get('experiment', {}).get('name', 'unknown')
    console.print(f"Experiment: {exp_name}")

    # Build args
    exp_args = build_args_from_config(exp_config)
    console.print(f"Model: {exp_args.model}")
    console.print(f"Data: {exp_args.data}")
    console.print(f"Input len: {exp_args.input_len}, Output len: {exp_args.output_len}")
    console.print(f"Batch size: {exp_args.batch_size}")

    # Create Data_Provider
    from data_provider.data_factory import Data_Provider

    console.print("\n[yellow]Initializing Data_Provider...[/yellow]")
    data_provider = Data_Provider(exp_args, buffer=not exp_args.disable_buffer, console=console)

    results = {
        'config': {
            'experiment': exp_name,
            'model': exp_args.model,
            'data': exp_args.data,
            'input_len': exp_args.input_len,
            'output_len': exp_args.output_len,
            'batch_size': exp_args.batch_size,
            'num_workers': exp_args.num_workers,
            'prefetch_factor': exp_args.prefetch_factor
        }
    }

    # Profile __getitem__ operations
    if not args.skip_getitem:
        getitem_stats = profile_getitem_operations(data_provider, args.num_samples)
        results['getitem_operations'] = getitem_stats

    # Profile DataLoader throughput
    if not args.skip_throughput:
        throughput_stats = profile_dataloader_throughput(data_provider, args.num_batches)
        results['dataloader_throughput'] = throughput_stats

    # Profile worker scaling
    if args.worker_scaling:
        scaling_stats = profile_worker_scaling(data_provider, args.max_workers, args.num_batches)
        results['worker_scaling'] = scaling_stats

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        console.print(f"\n[green]Results saved to: {output_path}[/green]")

    console.print("\n[bold green]Profiling complete![/bold green]")

    return results


if __name__ == '__main__':
    main()
