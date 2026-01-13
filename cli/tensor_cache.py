"""
CLI module for tensor cache generation and management.

This module provides commands for generating and validating tensor caches
that enable ultra-fast data loading during training.

Usage:
    # Generate cache for an experiment suite
    python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/canada_photovoltaics.yaml

    # Validate existing cache
    python -m cli.tensor_cache validate configs/experiment_suites/lynx_film/canada_photovoltaics.yaml

    # Show cache info
    python -m cli.tensor_cache info ./output/tensor_cache/

See context/performance_optimization/training_optimization_plan.md for details.
"""

import typer
import yaml
import logging
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from utils.tools import dotdict
from utils.config_utils import merge_configs

app = typer.Typer(help="Generate and manage tensor caches for fast data loading")
console = Console()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_suite_config(suite_path: str) -> dict:
    """Load a suite configuration file."""
    with open(suite_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_template(template_path: str) -> dict:
    """Load a template configuration file."""
    with open(template_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def get_experiment_config(suite_config: dict, experiment_name: str = None) -> dict:
    """Get a single experiment config from a suite."""
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


def build_args_from_config(config: dict) -> dotdict:
    """Build an args object from experiment config."""
    import torch

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


def build_cache_config(args: dotdict) -> dict:
    """
    Build config dict for tensor cache hash computation.

    These parameters determine cache uniqueness - if any change, the cache must be regenerated.

    Args:
        args: Argument object from build_args_from_config

    Returns:
        Dict of parameters that affect cache validity
    """
    # Get hetero_stride from model config if available
    hetero_stride = 1
    if hasattr(args, 'model_config') and isinstance(args.model_config, dict):
        hetero_stride = args.model_config.get('stride', 1)

    # Get hetero_type from data config if available
    hetero_type = None
    if hasattr(args, 'data_config') and args.data_config.get('hetero_info'):
        hetero_type = args.data_config.hetero_info.get('hetero_type')

    return {
        'input_len': args.input_len,
        'output_len': args.output_len,
        'scale': args.scale,
        'truncate_train_for_purge': args.truncate_train_for_purge,
        'downsample': args.downsample,
        'data_name': args.data,
        'hetero_stride': hetero_stride,
        'hetero_type': hetero_type,
        'missing_value_strategy': args.data_config.get('missing_value_strategy', 'none') if args.data_config else 'none',
        'split_info': str(args.data_config.get('split_info', '')) if args.data_config else '',
    }


def resolve_cache_dir(args: dotdict, explicit_dir: Optional[str] = None) -> Path:
    """
    Resolve tensor cache directory path.

    If explicit_dir is provided, use it. Otherwise, auto-generate based on
    dataset root_path and config hash.

    The auto-generated path follows the pattern:
        {dataset_root_path}/../tensor_cache/{config_hash}/

    This co-locates caches with dataset data and enables automatic reuse
    when the same config hash is encountered again.

    Args:
        args: Argument object from build_args_from_config
        explicit_dir: Optional explicit directory override

    Returns:
        Path to tensor cache directory
    """
    import os
    from data_provider.tensor_cache import compute_config_hash

    if explicit_dir:
        return Path(explicit_dir)

    # Build config and compute hash
    config = build_cache_config(args)
    config_hash = compute_config_hash(config)

    # Get dataset root path (e.g., data/fidel-ts/germany_renewable/time_series/)
    # Cache goes in parent: data/fidel-ts/germany_renewable/tensor_cache/<hash>/
    root_path = args.data_config.get('root_path', './data')
    dataset_dir = os.path.dirname(root_path.rstrip('/\\'))

    return Path(dataset_dir) / 'tensor_cache' / config_hash


@app.command()
def generate(
    config_path: str = typer.Argument(..., help="Path to suite config YAML"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Override output directory (default: auto-generated in dataset dir)"),
    experiment: Optional[str] = typer.Option(None, "--experiment", "-e", help="Specific experiment name"),
    chunk_size: int = typer.Option(10000, "--chunk-size", help="Samples per processing chunk"),
    splits: str = typer.Option("train,val,test", "--splits", help="Comma-separated splits to generate"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing cache")
):
    """
    Generate tensor cache for an experiment suite.

    This pre-computes all CPU-intensive dataloader operations:
    - Temporal matching
    - Embedding lookups
    - Array construction

    The resulting cache enables 100-1000x faster data loading during training.

    By default, the cache is stored alongside the dataset data:
        data/<dataset>/tensor_cache/<config_hash>/

    This enables automatic cache reuse when the same config is used again.

    Example:
        python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/canada_photovoltaics.yaml
    """
    config_path = Path(config_path)

    if not config_path.exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        raise typer.Exit(code=1)

    try:
        # Load suite config
        suite_config = load_suite_config(str(config_path))

        # Get experiment config
        exp_config = get_experiment_config(suite_config, experiment)
        exp_name = exp_config.get('experiment', {}).get('name', 'unknown')
        console.print(f"[green]Generating tensor cache for: {exp_name}[/green]")

        # Build args
        exp_args = build_args_from_config(exp_config)
        console.print(f"Model: {exp_args.model}")
        console.print(f"Data: {exp_args.data}")
        console.print(f"Input len: {exp_args.input_len}, Output len: {exp_args.output_len}")

        # Determine output directory using hash-based path resolution
        cache_dir = resolve_cache_dir(exp_args, output_dir)
        cache_config = build_cache_config(exp_args)

        from data_provider.tensor_cache import compute_config_hash
        config_hash = compute_config_hash(cache_config)

        console.print(f"Config hash: {config_hash}")
        console.print(f"Cache directory: {cache_dir}")

        # Check for existing cache
        if cache_dir.exists():
            if force:
                console.print(f"[yellow]Overwriting existing cache (--force specified)[/yellow]")
            else:
                # Check if cache is valid - if so, skip regeneration
                from data_provider.tensor_cache import validate_cache
                is_valid, message = validate_cache(cache_dir, cache_config)
                if is_valid:
                    console.print(f"[green]Cache already exists and is valid. Skipping regeneration.[/green]")
                    console.print(f"[green]Use --force to regenerate anyway.[/green]")
                    raise typer.Exit(code=0)
                else:
                    console.print(f"[yellow]Existing cache is invalid: {message}[/yellow]")
                    console.print(f"[yellow]Regenerating...[/yellow]")

        # Create Data_Provider
        from data_provider.data_factory import Data_Provider
        from data_provider.tensor_cache import TensorCacheGenerator

        console.print("\n[yellow]Initializing Data_Provider...[/yellow]")
        data_provider = Data_Provider(exp_args, buffer=not exp_args.disable_buffer, console=console)

        # Create generator
        generator = TensorCacheGenerator(
            data_provider=data_provider,
            cache_dir=cache_dir,
            config=cache_config,
            chunk_size=chunk_size,
            verbose=True
        )

        # Parse splits
        split_list = [s.strip() for s in splits.split(',')]
        console.print(f"Generating cache for splits: {split_list}")

        # Generate cache
        console.print("\n[yellow]Generating tensor cache...[/yellow]")
        cache_path = generator.generate(flags=split_list)

        console.print(f"\n[green]Cache generated successfully at: {cache_path}[/green]")

        # Show summary
        from data_provider.tensor_cache import TensorCacheMetadata
        metadata = TensorCacheMetadata.load(cache_path / 'metadata.json')

        table = Table(title="Cache Summary")
        table.add_column("Split", style="cyan")
        table.add_column("Samples", style="green")
        table.add_column("seq_x shape", style="yellow")

        for split, shapes in metadata.shapes.items():
            if 'seq_x' in shapes:
                samples = shapes['seq_x'][0]
                shape_str = str(shapes['seq_x'])
                table.add_row(split, f"{samples:,}", shape_str)

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error generating cache: {str(e)}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def validate(
    config_path: str = typer.Argument(..., help="Path to suite config YAML"),
    cache_dir: Optional[str] = typer.Option(None, "--cache-dir", "-c", help="Override cache directory (default: auto-resolved)"),
    experiment: Optional[str] = typer.Option(None, "--experiment", "-e", help="Specific experiment name")
):
    """
    Validate that a tensor cache matches the experiment config.

    Checks:
    - Cache directory exists
    - Metadata is valid
    - Config hash matches (cache is not stale)

    By default, looks for cache at:
        data/<dataset>/tensor_cache/<config_hash>/

    Example:
        python -m cli.tensor_cache validate configs/experiment_suites/lynx_film/canada_photovoltaics.yaml
    """
    config_path = Path(config_path)

    if not config_path.exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        raise typer.Exit(code=1)

    try:
        # Load suite config
        suite_config = load_suite_config(str(config_path))
        exp_config = get_experiment_config(suite_config, experiment)
        exp_args = build_args_from_config(exp_config)

        # Determine cache directory using hash-based path resolution
        cache_path = resolve_cache_dir(exp_args, cache_dir)
        cache_config = build_cache_config(exp_args)

        from data_provider.tensor_cache import compute_config_hash
        config_hash = compute_config_hash(cache_config)

        console.print(f"Config hash: {config_hash}")
        console.print(f"Validating cache at: {cache_path}")

        # Validate
        from data_provider.tensor_cache import validate_cache

        is_valid, message = validate_cache(cache_path, cache_config)

        if is_valid:
            console.print(f"[green]Cache is valid: {message}[/green]")
        else:
            console.print(f"[red]Cache is invalid: {message}[/red]")
            raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[red]Error validating cache: {str(e)}[/red]")
        raise typer.Exit(code=1)


@app.command()
def info(
    cache_dir: str = typer.Argument(..., help="Path to tensor cache directory")
):
    """
    Show information about a tensor cache.

    Displays:
    - Cache metadata
    - Array shapes and sizes
    - Entity information

    Example:
        python -m cli.tensor_cache info ./tensor_cache/Germany_Renewable_Power_Grid_360_168/
    """
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        console.print(f"[red]Error: Cache directory not found: {cache_path}[/red]")
        raise typer.Exit(code=1)

    try:
        from data_provider.tensor_cache import TensorCacheMetadata

        metadata = TensorCacheMetadata.load(cache_path / 'metadata.json')

        # Basic info
        console.print(f"\n[bold cyan]Tensor Cache Info[/bold cyan]")
        console.print(f"Directory: {cache_path}")
        console.print(f"Version: {metadata.version}")
        console.print(f"Created: {metadata.created_at}")
        console.print(f"Config hash: {metadata.config_hash}")

        # Data config
        console.print(f"\n[bold yellow]Data Configuration[/bold yellow]")
        for key, value in metadata.data_config.items():
            console.print(f"  {key}: {value}")

        # Shapes table
        table = Table(title="Array Shapes")
        table.add_column("Split", style="cyan")
        table.add_column("Array", style="green")
        table.add_column("Shape", style="yellow")

        for split, shapes in metadata.shapes.items():
            for arr_name, shape in shapes.items():
                table.add_row(split, arr_name, str(shape))

        console.print(table)

        # Entity info
        if metadata.entity_info:
            console.print(f"\n[bold magenta]Entity Information[/bold magenta]")
            entity_ids = metadata.entity_info.get('entity_ids', [])
            console.print(f"Total entities: {len(entity_ids)}")
            if len(entity_ids) <= 10:
                for eid in entity_ids:
                    samples = metadata.entity_info.get('samples_per_entity', {}).get(eid, 0)
                    console.print(f"  {eid}: {samples:,} samples")

        # Disk usage
        console.print(f"\n[bold green]Disk Usage[/bold green]")
        total_size = 0
        for split_dir in cache_path.iterdir():
            if split_dir.is_dir():
                split_size = sum(f.stat().st_size for f in split_dir.glob('*.npy'))
                total_size += split_size
                console.print(f"  {split_dir.name}: {split_size / 1e9:.2f} GB")

        console.print(f"  [bold]Total: {total_size / 1e9:.2f} GB[/bold]")

    except Exception as e:
        console.print(f"[red]Error reading cache info: {str(e)}[/red]")
        raise typer.Exit(code=1)


@app.command()
def benchmark(
    cache_dir: str = typer.Argument(..., help="Path to tensor cache directory"),
    num_batches: int = typer.Option(100, "--num-batches", "-n", help="Number of batches to benchmark"),
    batch_size: int = typer.Option(768, "--batch-size", "-b", help="Batch size"),
    num_workers: int = typer.Option(4, "--num-workers", "-w", help="Number of dataloader workers"),
    preload: bool = typer.Option(False, "--preload", help="Preload data to RAM")
):
    """
    Benchmark tensor cache loading speed.

    Measures:
    - Batch loading time
    - Throughput (samples/sec)
    - GPU transfer time

    Example:
        python -m cli.tensor_cache benchmark ./tensor_cache/Germany_Renewable_Power_Grid_360_168/
    """
    import time
    import torch
    import numpy as np

    cache_path = Path(cache_dir)

    if not cache_path.exists():
        console.print(f"[red]Error: Cache directory not found: {cache_path}[/red]")
        raise typer.Exit(code=1)

    try:
        from data_provider.tensor_cache import get_tensor_cache_dataloader

        console.print(f"[green]Benchmarking tensor cache: {cache_path}[/green]")
        console.print(f"Batch size: {batch_size}, Workers: {num_workers}, Preload: {preload}")

        # Create dataloader
        loader = get_tensor_cache_dataloader(
            cache_dir=cache_path,
            flag='train',
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=4,
            preload_to_ram=preload
        )

        console.print(f"Dataset size: {len(loader.dataset):,} samples")
        console.print(f"Batches available: {len(loader):,}")

        # Warmup
        console.print("\n[yellow]Warming up...[/yellow]")
        warmup_iter = iter(loader)
        for _ in range(min(5, len(loader))):
            _ = next(warmup_iter)

        # Benchmark
        console.print(f"[yellow]Benchmarking {num_batches} batches...[/yellow]")
        batch_times = []
        gpu_times = []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        start_total = time.perf_counter()
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break

            batch_start = time.perf_counter()

            # GPU transfer
            if torch.cuda.is_available():
                gpu_start = time.perf_counter()
                for item in batch:
                    if isinstance(item, torch.Tensor):
                        item.to(device, non_blocking=True)
                torch.cuda.synchronize()
                gpu_times.append(time.perf_counter() - gpu_start)

            batch_times.append(time.perf_counter() - batch_start)

        total_time = time.perf_counter() - start_total
        total_samples = num_batches * batch_size

        # Results
        batch_arr = np.array(batch_times) * 1000  # ms

        console.print(f"\n[bold cyan]Benchmark Results[/bold cyan]")
        console.print(f"Total time: {total_time:.2f} seconds")
        console.print(f"Total samples: {total_samples:,}")
        console.print(f"Throughput: {total_samples / total_time:.0f} samples/sec")
        console.print(f"\nBatch time (ms):")
        console.print(f"  Mean: {np.mean(batch_arr):.2f}")
        console.print(f"  Std: {np.std(batch_arr):.2f}")
        console.print(f"  Min: {np.min(batch_arr):.2f}")
        console.print(f"  Max: {np.max(batch_arr):.2f}")
        console.print(f"  P50: {np.percentile(batch_arr, 50):.2f}")
        console.print(f"  P95: {np.percentile(batch_arr, 95):.2f}")

        if gpu_times:
            gpu_arr = np.array(gpu_times) * 1000
            console.print(f"\nGPU transfer time (ms):")
            console.print(f"  Mean: {np.mean(gpu_arr):.2f}")

        # Comparison
        console.print(f"\n[bold green]Comparison with baseline[/bold green]")
        console.print(f"Baseline (profiled): ~7000 ms/batch, ~110 samples/sec")
        console.print(f"Tensor cache: {np.mean(batch_arr):.0f} ms/batch, {total_samples / total_time:.0f} samples/sec")
        speedup = 7000 / np.mean(batch_arr) if np.mean(batch_arr) > 0 else float('inf')
        console.print(f"[bold]Speedup: {speedup:.0f}x[/bold]")

    except Exception as e:
        console.print(f"[red]Error benchmarking: {str(e)}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
