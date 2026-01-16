"""
CLI module for tensor cache generation and management.

================================================================================
OVERVIEW
================================================================================

This module provides commands for generating and validating tensor caches
that enable ultra-fast data loading during training (100-1000x speedup).

================================================================================
GPU VS CPU SEPARATION
================================================================================

There are TWO distinct phases in the data pipeline, with different GPU requirements:

PHASE 1: EMBEDDING GENERATION (GPU REQUIRED)
--------------------------------------------
  - Runs BERT/transformer models on text data
  - Produces embedding vectors (768-dim for BERT-base)
  - MUST run on GPU (CPU is 10-100x slower, impractical)
  - Output: Embedding cache files (.pkl)
  - Location: data/{dataset}/embeddings_{hash}/

PHASE 2: TENSOR CACHE GENERATION (CPU OK)
-----------------------------------------
  - Reads pre-computed embeddings from disk
  - Performs temporal matching, indexing, array construction
  - Pure numpy/CPU operations
  - CAN run on CPU-only nodes IF embeddings are pre-computed
  - Output: Tensor cache files (.npy)
  - Location: data/{dataset}/tensor_cache/{hash}/

================================================================================
--cpu-only FLAG SEMANTICS
================================================================================

The --cpu-only flag enables tensor cache generation on CPU-only nodes.

IMPORTANT: This flag does NOT enable CPU embedding computation!
           Embedding computation ALWAYS requires GPU.

What --cpu-only does:
  1. Sets device='cpu' for Data_Provider
  2. Validates that embeddings are ALREADY cached
  3. If embeddings are NOT cached → FAIL with helpful error
  4. If embeddings ARE cached → Proceed with tensor cache generation

Typical workflow:
  1. GPU node:  Generate embeddings (automatic, part of data loading)
  2. CPU node:  python -m cli.tensor_cache generate suite.yaml --cpu-only
                (uses pre-computed embeddings, generates tensor cache)

================================================================================
USAGE EXAMPLES
================================================================================

# Generate cache (auto-detect GPU/CPU, compute embeddings if needed)
python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/suite.yaml

# Generate cache on CPU-only node (REQUIRES pre-computed embeddings)
python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/suite.yaml --cpu-only

# Validate existing cache
python -m cli.tensor_cache validate configs/experiment_suites/lynx_film/suite.yaml

# Show cache info
python -m cli.tensor_cache info ./output/tensor_cache/

See context/performance_optimization/training_optimization_plan.md for details.
"""

import typer
import yaml
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from rich.console import Console
from rich.table import Table

from utils.tools import dotdict
from utils.config_utils import merge_configs
from utils.experiment_config_builder import (
    build_experiment_args,
    build_cache_config as build_cache_config_centralized,
)

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


def get_all_experiments(
    suite_config: dict,
    filter_pattern: Optional[str] = None
) -> List[Tuple[str, dict]]:
    """
    Get all experiments from a suite, optionally filtered by name pattern.

    Args:
        suite_config: Loaded suite configuration dict
        filter_pattern: Optional case-insensitive substring filter

    Returns:
        List of (experiment_name, merged_config) tuples
    """
    suite_info = suite_config.get('suite', {})
    experiments = suite_info.get('experiments', [])

    if not experiments:
        raise ValueError("No experiments found in suite config")

    # Filter by enabled flag first
    experiments = [exp for exp in experiments if exp.get('enabled', True)]

    # Apply name filter if provided
    if filter_pattern:
        experiments = [
            exp for exp in experiments
            if filter_pattern.lower() in exp.get('name', '').lower()
        ]

    if not experiments:
        raise ValueError(f"No experiments match filter '{filter_pattern}'")

    # Load and merge configs for each experiment
    result = []
    for exp in experiments:
        exp_name = exp.get('name', 'unknown')
        template_path = exp.get('template', '')

        if not template_path:
            logger.warning(f"No template specified for experiment '{exp_name}', skipping")
            continue

        try:
            template = load_template(template_path)
            overrides = exp.get('overrides', {})
            merged_config = merge_configs(template, overrides)
            result.append((exp_name, merged_config))
        except Exception as e:
            logger.warning(f"Error loading config for '{exp_name}': {e}, skipping")
            continue

    return result


def build_args_from_config(config: dict) -> dotdict:
    """
    Build an args object from experiment config.
    
    Uses centralized config builder to ensure data_config overrides are applied.
    This fixes the bug where experiment-level overrides (e.g., timemmd_text_output: embedding)
    were not being merged into the base data config.
    
    Args:
        config: Merged experiment config (template + overrides)
    
    Returns:
        Complete args dotdict ready for Data_Provider
    """
    return build_experiment_args(config, include_gpu=True)


def build_cache_config(args: dotdict) -> dict:
    """
    Build config dict for tensor cache hash computation.

    Uses centralized config builder to ensure cache hash includes all relevant parameters.
    These parameters determine cache uniqueness - if any change, the cache must be regenerated.

    Args:
        args: Argument object from build_args_from_config

    Returns:
        Dict of parameters that affect cache validity
    """
    return build_cache_config_centralized(args)


def _check_embeddings_cached(args: dotdict) -> Tuple[bool, str]:
    """
    Check if embeddings are cached for the given experiment configuration.
    
    This function performs a LIGHTWEIGHT check that does NOT load any models.
    It's designed to be called on CPU-only nodes to validate that embeddings
    exist before attempting tensor cache generation.
    
    How it works:
    -------------
    1. For Fidel-TS datasets (with hetero_info):
       - Creates a FidelTSEmbeddingLoader with the same config
       - Calls has_cached_embeddings()
    
    2. For Time-MMD datasets (with timemmd_text_output: embedding):
       - Uses TextEmbedder's cache manager directly
       - Checks for matching embeddings_{hash} directory
    
    3. For datasets without embeddings:
       - Returns True (no embeddings needed)
    
    Why this is important:
    ----------------------
    The --cpu-only flag promises that no GPU will be needed. But if embeddings
    aren't cached, the Data_Provider will try to compute them, which requires
    loading BERT on GPU. By validating upfront, we can fail fast with a helpful
    error instead of failing deep in the embedding loading code.
    
    Args:
        args: Experiment arguments (dotdict) containing data_config
    
    Returns:
        Tuple of (is_cached, message)
        - is_cached: True if embeddings are cached and ready to load, False otherwise
        - message: Descriptive message about cache status
    """
    from embedder.fidel_ts_embedder import FidelTSEmbeddingLoader
    from embedder.cache_manager import EmbeddingCacheManager
    from embedder.metadata import EmbeddingMetadata
    from embedder.embedder import TextEmbedder
    from pathlib import Path
    
    # Get hetero_info from args (embedding configuration for Fidel-TS datasets)
    hetero_info = args.data_config.get('hetero_info', {})
    
    # Check if this is a Time-MMD dataset with embedding output format
    timemmd_text_output = args.data_config.get('timemmd_text_output', 'text')
    is_time_mmd_embedding = (timemmd_text_output == 'embedding')
    
    # =========================================================================
    # CASE 1: Time-MMD dataset with embedding output
    # =========================================================================
    if is_time_mmd_embedding:
        root_path = args.data_config.get('root_path', './data')
        embed_model_name = args.data_config.get('timemmd_embed_model', 'bert-base-uncased')
        aggregation_method = args.data_config.get('aggregation_method', 'cls')
        max_length = 512  # Default max_length
        
        # Get embedding dimension from lookup table (no model loading)
        embedding_dim = TextEmbedder._get_embedding_dim_without_model(embed_model_name)
        if embedding_dim is None:
            return False, f"Unknown model '{embed_model_name}' - cannot check cache without loading model"
        
        # Create metadata to compute expected hash
        metadata = EmbeddingMetadata(
            tokenizer_name=embed_model_name,
            model_name=embed_model_name,
            aggregation_method=aggregation_method,
            embedding_dim=embedding_dim,
            max_length=max_length
        )
        expected_hash = metadata.compute_hash()[:16]
        
        # Check if cache directory exists
        cache_base = Path(root_path)
        if not cache_base.exists():
            return False, f"Cache base directory does not exist: {cache_base}"
        
        # Look for embeddings_{hash} directories
        expected_dir = cache_base / f"embeddings_{expected_hash}"
        if expected_dir.exists():
            # Verify metadata file exists
            metadata_path = expected_dir / 'metadata.json'
            embeddings_path = expected_dir / 'embeddings.pkl'
            
            if metadata_path.exists() and embeddings_path.exists():
                return True, f"Found Time-MMD embeddings at {expected_dir}"
            else:
                return False, f"Cache directory exists but incomplete: {expected_dir}"
        
        # Cache not found - list what IS available for debugging
        available_caches = [d.name for d in cache_base.iterdir() if d.is_dir() and d.name.startswith('embeddings_')]
        if available_caches:
            return False, (
                f"Time-MMD embedding cache not found. "
                f"Looking for: embeddings_{expected_hash} in {cache_base}\n"
                f"Available caches: {available_caches}\n"
                f"Config: model={embed_model_name}, aggregation={aggregation_method}, dim={embedding_dim}"
            )
        else:
            return False, f"No embedding caches found in {cache_base}. Expected: embeddings_{expected_hash}"
    
    # =========================================================================
    # CASE 2: Fidel-TS dataset with hetero_info
    # =========================================================================
    if hetero_info:
        dataset_name = args.data
        root_path = args.data_config.get('root_path', './data')
        embed_model_name = hetero_info.get('embed_model_name', 'bert-base-uncased')
        aggregation_method = hetero_info.get('aggregation_method', 'cls')
        use_old_embeddings = hetero_info.get('use_old_embeddings', False)
        
        try:
            loader = FidelTSEmbeddingLoader(
                dataset_name=dataset_name,
                hetero_info=hetero_info,
                base_data_path=root_path,
                embed_model_name=embed_model_name,
                aggregation_method=aggregation_method,
                device='cpu',
                hf_cache_dir=args.get('hf_cache_dir', './HF_cache/'),
                use_old_embeddings=use_old_embeddings
            )
            
            if loader.has_cached_embeddings():
                return True, "Found Fidel-TS embeddings in cache"
            else:
                return False, "Fidel-TS embeddings not found in cache"
        
        except Exception as e:
            return False, f"Error checking Fidel-TS embedding cache: {e}"
    
    # =========================================================================
    # CASE 3: No embeddings needed
    # =========================================================================
    return True, "No embeddings needed for this dataset configuration"


def resolve_cache_dir(args: dotdict, explicit_dir: Optional[str] = None) -> Path:
    """
    Resolve tensor cache directory path.

    If explicit_dir is provided, use it. Otherwise, auto-generate based on
    dataset root_path and config hash.

    The auto-generated path follows the pattern:
        {dataset_root_path}/tensor_cache/{config_hash}/

    This co-locates caches with dataset data and enables automatic reuse
    when the same config hash is encountered again.

    For time_mmd datasets with root_path='data/time_mmd/Traffic':
        -> Cache: data/time_mmd/Traffic/tensor_cache/<hash>/

    For other datasets with root_path='data/fidel-ts/germany_renewable/time_series':
        -> Cache: data/fidel-ts/germany_renewable/time_series/tensor_cache/<hash>/

    Args:
        args: Argument object from build_args_from_config
        explicit_dir: Optional explicit directory override

    Returns:
        Path to tensor cache directory
    """
    from data_provider.tensor_cache import compute_config_hash

    if explicit_dir:
        return Path(explicit_dir)

    # Build config and compute hash
    config = build_cache_config(args)
    config_hash = compute_config_hash(config)

    # Get dataset root path - cache goes directly under this directory
    root_path = args.data_config.get('root_path', './data')
    dataset_dir = root_path.rstrip('/\\')

    return Path(dataset_dir) / 'tensor_cache' / config_hash


@app.command()
def generate(
    config_path: str = typer.Argument(..., help="Path to suite config YAML"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Override output directory (default: auto-generated in dataset dir)"),
    filter_experiments: Optional[str] = typer.Option(None, "--filter", "-f", help="Filter experiments by name pattern (case-insensitive)"),
    chunk_size: int = typer.Option(10000, "--chunk-size", help="Samples per processing chunk"),
    splits: str = typer.Option("train,val,test", "--splits", help="Comma-separated splits to generate"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing cache"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be generated without generating"),
    cpu_only: bool = typer.Option(False, "--cpu-only", help="CPU-only mode: requires pre-computed embeddings"),
    use_polars: bool = typer.Option(True, "--use-polars/--no-polars", help="Use polars-optimized implementation (default: enabled)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose output (hetero data loading messages)")
):
    """
    Generate tensor cache for all experiments in a suite.

    This pre-computes all CPU-intensive dataloader operations:
    - Temporal matching
    - Embedding lookups  
    - Array construction

    The resulting cache enables 100-1000x faster data loading during training.

    By default, the cache is stored alongside the dataset data:
        data/<dataset>/tensor_cache/<config_hash>/

    Experiments with the same config hash share the same cache (deduplication).

    GPU vs CPU:
    -----------
    By default, this command will use GPU for embedding computation if needed.
    Use --cpu-only to run on CPU-only nodes, but ONLY if embeddings are already
    cached. The --cpu-only flag does NOT enable CPU embedding computation.

    Examples:
        # Generate for all experiments in suite (uses GPU for embeddings if needed)
        python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/suite.yaml

        # Generate only for experiments matching pattern
        python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/suite.yaml --filter germany
        
        # CPU-only mode (REQUIRES pre-computed embeddings!)
        python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/suite.yaml --cpu-only
    """
    from data_provider.tensor_cache import compute_config_hash, validate_cache

    config_path = Path(config_path)

    if not config_path.exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        raise typer.Exit(code=1)

    try:
        # Load suite config
        suite_config = load_suite_config(str(config_path))
        suite_info = suite_config.get('suite', {})
        suite_name = suite_info.get('name', 'unknown')

        # Get all experiments (filtered if requested)
        experiments = get_all_experiments(suite_config, filter_experiments)

        console.print(f"\n[bold cyan]Tensor Cache Generation for Suite: {suite_name}[/bold cyan]")
        console.print(f"  Total experiments: [green]{len(experiments)}[/green]")
        if filter_experiments:
            console.print(f"  Filter: [yellow]'{filter_experiments}'[/yellow]")
        if cpu_only:
            console.print(f"  Device: [yellow]CPU-only mode (GPU disabled)[/yellow]")
        console.print(f"  Implementation: [green]{'polars-optimized' if use_polars else 'standard'}[/green]")
        console.print()

        # Analyze experiments and group by config hash (deduplication)
        # Key: config_hash -> Value: (cache_dir, cache_config, exp_args, [experiment_names])
        cache_groups: Dict[str, Tuple[Path, dict, dotdict, List[str]]] = {}
        skipped_experiments = []

        console.print("[bold]Analyzing experiments...[/bold]")
        for exp_name, exp_config in experiments:
            try:
                exp_args = build_args_from_config(exp_config)
                cache_config = build_cache_config(exp_args)
                config_hash = compute_config_hash(cache_config)
                cache_dir = resolve_cache_dir(exp_args, output_dir)

                if config_hash in cache_groups:
                    # Same hash - add to existing group
                    cache_groups[config_hash][3].append(exp_name)
                    console.print(f"  [dim]○[/dim] {exp_name}: hash={config_hash[:8]} (same as {cache_groups[config_hash][3][0]})")
                else:
                    # New hash
                    cache_groups[config_hash] = (cache_dir, cache_config, exp_args, [exp_name])
                    console.print(f"  [green]●[/green] {exp_name}: hash={config_hash[:8]} (new)")

            except Exception as e:
                console.print(f"  [yellow]⚠[/yellow] {exp_name}: Error analyzing: {e}")
                skipped_experiments.append(exp_name)
                continue

        console.print()
        console.print(f"[bold]Unique caches to generate: {len(cache_groups)}[/bold]")

        if dry_run:
            console.print(f"\n[yellow]Dry run - no caches will be generated[/yellow]")
            for config_hash, (cache_dir, cache_config, _, exp_names) in cache_groups.items():
                console.print(f"  {config_hash[:8]}: {cache_dir}")
                console.print(f"    Experiments: {', '.join(exp_names)}")
            raise typer.Exit(code=0)

        # Generate caches
        from data_provider.data_factory import Data_Provider
        from data_provider.tensor_cache import TensorCacheGenerator, TensorCacheMetadata

        split_list = [s.strip() for s in splits.split(',')]
        generated_count = 0
        skipped_valid_count = 0

        for i, (config_hash, (cache_dir, cache_config, exp_args, exp_names)) in enumerate(cache_groups.items(), 1):
            console.print(f"\n[bold cyan]Cache {i}/{len(cache_groups)}: {config_hash[:8]}[/bold cyan]")
            console.print(f"  Path: {cache_dir}")
            console.print(f"  Experiments: {', '.join(exp_names)}")
            console.print(f"  Data: {exp_args.data}, Input: {exp_args.input_len}, Output: {exp_args.output_len}")

            # Check for existing valid cache
            if cache_dir.exists() and not force:
                is_valid, message = validate_cache(cache_dir, cache_config)
                if is_valid:
                    console.print(f"  [green]✓ Cache already valid, skipping[/green]")
                    skipped_valid_count += 1
                    continue
                else:
                    console.print(f"  [yellow]Cache invalid: {message}, regenerating...[/yellow]")

            # Generate cache
            try:
                # =================================================================
                # CPU-ONLY MODE VALIDATION
                # =================================================================
                # When --cpu-only is set, we MUST verify embeddings are cached.
                # This is because:
                #   1. Embedding computation requires GPU (BERT is GPU-intensive)
                #   2. We explicitly do NOT support CPU fallback for embeddings
                #   3. Better to fail fast with clear error than fail deep in code
                #
                # The validation uses has_cached_embeddings() which does NOT load
                # the embedding model - it only checks if cache directories exist.
                # =================================================================
                
                if cpu_only:
                    console.print("  [yellow]CPU-only mode: Validating embedding cache...[/yellow]")
                    
                    # Check if embeddings are cached (without loading model)
                    embeddings_cached, cache_message = _check_embeddings_cached(exp_args)
                    
                    if not embeddings_cached:
                        console.print("  [red]✗ Embeddings NOT cached![/red]")
                        console.print(f"  [red]  {cache_message}[/red]")
                        console.print("  [red]  CPU-only mode requires pre-computed embeddings.[/red]")
                        console.print("  [yellow]  Solutions:[/yellow]")
                        console.print("  [yellow]    1. Run without --cpu-only on a GPU node first[/yellow]")
                        console.print("  [yellow]    2. Generate embeddings separately on GPU[/yellow]")
                        console.print("  [yellow]    3. Check embedding cache path and config hash match[/yellow]")
                        continue  # Skip this cache group
                    
                    console.print(f"  [green]✓ {cache_message}[/green]")
                    
                    # Set device to CPU
                    original_device = exp_args.device
                    exp_args.device = 'cpu'
                    if original_device != 'cpu':
                        console.print(f"  [dim]Device: {original_device} -> cpu[/dim]")
                
                # Force preload_hetero=True for direct array access optimization
                # This enables 20x faster shared table building by bypassing __getitem__
                # Safe for tensor cache generation (one-time batch operation)
                original_preload = getattr(exp_args, 'preload_hetero', False)
                exp_args.preload_hetero = True
                if not original_preload:
                    console.print("  [dim]preload_hetero: False -> True (for direct access optimization)[/dim]")

                # Set verbose flag for hetero data preloading messages
                exp_args.verbose_hetero_preload = verbose

                console.print("  [yellow]Initializing Data_Provider...[/yellow]")
                data_provider = Data_Provider(exp_args, buffer=not exp_args.disable_buffer, console=console)

                generator = TensorCacheGenerator(
                    data_provider=data_provider,
                    cache_dir=cache_dir,
                    config=cache_config,
                    chunk_size=chunk_size,
                    verbose=True,
                    console=console,  # Enable Rich nested progress bars
                    use_polars=use_polars
                )

                console.print(f"  [yellow]Generating for splits: {split_list}[/yellow]")
                cache_path = generator.generate(flags=split_list)

                # Show summary for this cache
                metadata = TensorCacheMetadata.load(cache_path / 'metadata.json')
                # V2 indexed format uses x_indices, V1 direct format uses seq_x
                # Try x_indices first (V2), then seq_x (V1), then sample_ids (fallback)
                total_samples = sum(
                    shapes.get('x_indices', shapes.get('seq_x', shapes.get('sample_ids', [0])))[0]
                    for shapes in metadata.shapes.values()
                )
                console.print(f"  [green]✓ Generated: {total_samples:,} total samples[/green]")
                generated_count += 1

            except Exception as e:
                console.print(f"  [red]✗ Error: {e}[/red]")
                import traceback
                traceback.print_exc()
                continue

        # Final summary
        console.print(f"\n[bold green]═══ Generation Complete ═══[/bold green]")
        console.print(f"  Caches generated: {generated_count}")
        console.print(f"  Caches skipped (already valid): {skipped_valid_count}")
        console.print(f"  Experiments covered: {sum(len(g[3]) for g in cache_groups.values())}")
        if skipped_experiments:
            console.print(f"  Experiments skipped (errors): {len(skipped_experiments)}")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error generating cache: {str(e)}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def validate(
    config_path: str = typer.Argument(..., help="Path to suite config YAML"),
    cache_dir: Optional[str] = typer.Option(None, "--cache-dir", "-c", help="Override cache directory (default: auto-resolved)"),
    filter_experiments: Optional[str] = typer.Option(None, "--filter", "-f", help="Filter experiments by name pattern (case-insensitive)")
):
    """
    Validate tensor caches for all experiments in a suite.

    Checks for each unique cache:
    - Cache directory exists
    - Metadata is valid
    - Config hash matches (cache is not stale)

    By default, looks for cache at:
        data/<dataset>/tensor_cache/<config_hash>/

    Examples:
        # Validate caches for all experiments
        python -m cli.tensor_cache validate configs/experiment_suites/lynx_film/suite.yaml

        # Validate only for experiments matching pattern
        python -m cli.tensor_cache validate configs/experiment_suites/lynx_film/suite.yaml --filter germany
    """
    from data_provider.tensor_cache import compute_config_hash, validate_cache as validate_cache_fn

    config_path = Path(config_path)

    if not config_path.exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        raise typer.Exit(code=1)

    try:
        # Load suite config
        suite_config = load_suite_config(str(config_path))
        suite_info = suite_config.get('suite', {})
        suite_name = suite_info.get('name', 'unknown')

        # Get all experiments (filtered if requested)
        experiments = get_all_experiments(suite_config, filter_experiments)

        console.print(f"\n[bold cyan]Tensor Cache Validation for Suite: {suite_name}[/bold cyan]")
        console.print(f"  Total experiments: [green]{len(experiments)}[/green]")
        if filter_experiments:
            console.print(f"  Filter: [yellow]'{filter_experiments}'[/yellow]")
        console.print()

        # Group by config hash (same as generate)
        cache_groups: Dict[str, Tuple[Path, dict, List[str]]] = {}

        for exp_name, exp_config in experiments:
            try:
                exp_args = build_args_from_config(exp_config)
                cache_config = build_cache_config(exp_args)
                config_hash = compute_config_hash(cache_config)
                cache_path = resolve_cache_dir(exp_args, cache_dir)

                if config_hash not in cache_groups:
                    cache_groups[config_hash] = (cache_path, cache_config, [exp_name])
                else:
                    cache_groups[config_hash][2].append(exp_name)
            except Exception as e:
                console.print(f"  [yellow]⚠[/yellow] {exp_name}: Error analyzing: {e}")
                continue

        # Validate each unique cache
        valid_count = 0
        invalid_count = 0

        console.print(f"[bold]Validating {len(cache_groups)} unique caches...[/bold]\n")

        for config_hash, (cache_path, cache_config, exp_names) in cache_groups.items():
            is_valid, message = validate_cache_fn(cache_path, cache_config)

            if is_valid:
                console.print(f"  [green]✓[/green] {config_hash[:8]}: Valid")
                console.print(f"      Path: {cache_path}")
                console.print(f"      Experiments: {', '.join(exp_names)}")
                valid_count += 1
            else:
                console.print(f"  [red]✗[/red] {config_hash[:8]}: {message}")
                console.print(f"      Path: {cache_path}")
                console.print(f"      Experiments: {', '.join(exp_names)}")
                invalid_count += 1

        # Summary
        console.print(f"\n[bold]═══ Validation Summary ═══[/bold]")
        console.print(f"  Valid caches: [green]{valid_count}[/green]")
        console.print(f"  Invalid/missing caches: [red]{invalid_count}[/red]")

        if invalid_count > 0:
            console.print(f"\n[yellow]Run 'python -m cli.tensor_cache generate {config_path}' to create missing caches[/yellow]")
            raise typer.Exit(code=1)

    except typer.Exit:
        raise
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
