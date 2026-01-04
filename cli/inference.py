"""
CLI module for LLM inference and embedding precomputation.

Generates LLM embeddings for use with TimeCMA and similar models.
All embeddings must be precomputed before training - no on-the-fly inference supported.

EXPERIMENT-DRIVEN EMBEDDING GENERATION:
    Embeddings are generated using the experiment configuration file, which ensures
    consistency between training and embedding generation. The experiment config
    specifies input_len, output_len, and the llm_embedding settings.
    
CURRENT SUPPORT:
    Time Series → Text → LLM Embeddings (via TSPromptBuilder)
    - Converts time series values to natural language prompts
    - Uses templates like 'timecma_v1' from the llm_embedding config section
    - Extracts last-token hidden states from LLM
    
FUTURE SUPPORT (not yet implemented):
    Raw Text → LLM Embeddings
    - Direct embedding of text data (news articles, reports, etc.)
    - Will use LLMEmbedder.embed_texts() method
    - Useful for multimodal forecasting with external text sources
    
The current embedding process (time series):
    1. Load experiment config to get input_len, output_len, and llm_embedding settings
    2. Load time series data from experiment's dataset
    3. Convert each (sample, channel) to a text prompt via TSPromptBuilder
       (controlled by prompt_template in llm_embedding section)
    4. Run LLM to extract last-token hidden states
    5. Cache embeddings for use during training

This CLI provides commands for:
- generate: Generate LLM embeddings using experiment config
- generate-suite: Generate LLM embeddings for all experiments in a suite
- verify: Verify that embeddings exist and are valid for an experiment
- verify-suite: Verify embeddings for all experiments in a suite
- list-cached: List cached embeddings for an experiment config
- estimate-memory: Estimate GPU memory requirements for a model
- list-models: List supported LLM models with specifications
- gpu-info: Display current GPU memory information

Examples:
    # Generate embeddings for an experiment config
    python -m cli.inference generate configs/experiments/timecma_test.yaml
    
    # Generate embeddings for all experiments in a suite
    python -m cli.inference generate-suite configs/experiment_suites/timecma_test.yaml
    
    # Force regeneration of test split only
    python -m cli.inference generate configs/experiments/timecma_test.yaml --splits test --force
    
    # Verify embeddings exist for an experiment
    python -m cli.inference verify configs/experiments/timecma_test.yaml
    
    # Verify embeddings for all experiments in a suite
    python -m cli.inference verify-suite configs/experiment_suites/timecma_test.yaml
    
    # List cached embeddings for an experiment
    python -m cli.inference list-cached configs/experiments/timecma_test.yaml
    
    # Check memory requirements
    python -m cli.inference estimate-memory Qwen/Qwen2.5-72B-Instruct --quantization 4bit
"""

import typer
import torch
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
# Rich imports are used within LLMEmbedder, not directly in CLI


app = typer.Typer(help="LLM inference and embedding precomputation")
console = Console()


@app.command()
def generate(
    experiment_config: str = typer.Argument(..., help="Experiment configuration file (e.g., configs/experiments/timecma_test.yaml)"),
    splits: List[str] = typer.Option(
        ["train", "val", "test"],
        "--splits", "-s",
        help="Data splits to process"
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device", "-d",
        help="Override device from experiment config"
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size", "-b",
        help="Override batch size from config"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Force recompute existing embeddings"
    ),
    memory_efficient: bool = typer.Option(
        False,
        "--memory-efficient", "-m",
        help="Use memory-efficient chunked processing (for large datasets)"
    ),
    chunk_size: int = typer.Option(
        1000,
        "--chunk-size",
        help="Samples per chunk in memory-efficient mode"
    ),
    gpu_monitor: bool = typer.Option(
        True,
        "--gpu-monitor/--no-gpu-monitor",
        help="Enable GPU utilization monitoring in memory-efficient mode"
    ),
):
    """
    Generate LLM embeddings for an experiment (time series → prompts → embeddings).
    
    Uses the experiment configuration file to ensure consistency between
    training and embedding generation. The experiment config must have an
    'llm_embedding' section that specifies the LLM model and settings.
    
    The input_len and output_len from the experiment's training section are
    used to generate embeddings with matching sequence lengths.
    
    This command generates embeddings by:
    1. Loading time series data from the experiment's dataset
    2. Converting each (sample, channel) to a text prompt via TSPromptBuilder
       (template controlled by prompt_template in llm_embedding section)
    3. Running the LLM to extract last-token embeddings
    4. Caching results for use during training
    
    Examples:
        # Generate embeddings for a TimeCMA experiment
        python -m cli.inference generate configs/experiments/timecma_test.yaml
        
        # Generate only test split
        python -m cli.inference generate configs/experiments/timecma_test.yaml --splits test
        
        # Force regenerate all splits
        python -m cli.inference generate configs/experiments/timecma_test.yaml --force
        
        # Use specific GPU
        python -m cli.inference generate configs/experiments/timecma_test.yaml --device cuda:1
    """
    import yaml
    from embedder.llm_embedder import LLMEmbedder
    
    # Experiment config is REQUIRED
    config_path = Path(experiment_config)
    if not config_path.exists():
        console.print(f"[red]Error: Experiment config file not found: {config_path}[/red]")
        raise typer.Exit(code=1)
    
    # Load experiment config to extract dataset name
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check for llm_embedding section
    if 'llm_embedding' not in config:
        console.print(f"[red]Error: Experiment config does not have 'llm_embedding' section[/red]")
        console.print("[dim]Add llm_embedding configuration to generate LLM embeddings:[/dim]")
        console.print("""
[yellow]llm_embedding:
  model_name: "gpt2"           # or "Qwen/Qwen2.5-7B-Instruct"
  batch_size: 64
  cache_dir: "./LLM_cache/"
  prompt_template: "timecma_v1"[/yellow]
""")
        raise typer.Exit(code=1)
    
    # Extract dataset name from config
    dataset = config.get('data', {}).get('name')
    if not dataset:
        console.print("[red]Error: Experiment config missing data.name[/red]")
        raise typer.Exit(code=1)
    
    # Load embedder from experiment config with console for progress bars
    try:
        embedder = LLMEmbedder.from_experiment_config(
            experiment_config_path=str(config_path),
            device=device,
            console=console,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)
    
    # Batch size priority: CLI > config > default
    effective_batch_size = batch_size if batch_size is not None else embedder.default_batch_size
    
    # Determine memory-efficient mode: CLI flag > config > default (False)
    llm_config = config.get('llm_embedding', {})
    use_memory_efficient = memory_efficient  # CLI flag takes priority
    if not memory_efficient:
        # Check config if CLI flag not set
        use_memory_efficient = llm_config.get('memory_efficient', False)
    
    # Determine chunk size: CLI (if not default) > config > default
    effective_chunk_size = chunk_size
    if chunk_size == 1000:  # Default value, check config
        effective_chunk_size = llm_config.get('chunk_size', 1000)
    
    console.print("\n[bold cyan]LLM Embedding Generation (Experiment-Driven)[/bold cyan]")
    console.print(f"  Experiment: [green]{config_path.name}[/green]")
    console.print(f"  Dataset: [green]{dataset}[/green]")
    console.print(f"  Model: [green]{embedder.model_name}[/green]")
    console.print(f"  Quantization: [green]{embedder.quantization or 'None (fp16)'}[/green]")
    console.print(f"  Device: [green]{embedder.device}[/green]")
    console.print(f"  Input Length: [green]{embedder.input_len}[/green]")
    console.print(f"  Output Length: [green]{embedder.output_len}[/green]")
    console.print(f"  Splits: [green]{', '.join(splits)}[/green]")
    if use_memory_efficient:
        console.print(f"  Mode: [yellow]Memory-Efficient (chunk_size={effective_chunk_size})[/yellow]")
    console.print()
    
    succeeded = []
    failed = []
    
    for split in splits:
        console.print(f"[bold]━━━ {split} ━━━[/bold]")
        
        try:
            if use_memory_efficient:
                # Use memory-efficient chunked processing
                embedder.generate_ts_embeddings_chunked(
                    dataset=dataset,
                    split=split,
                    batch_size=effective_batch_size,
                    chunk_size=effective_chunk_size,
                    force=force,
                    enable_gpu_monitor=gpu_monitor,
                )
            else:
                # Use standard processing
                embedder.generate_ts_embeddings(
                    dataset=dataset,
                    split=split,
                    batch_size=effective_batch_size,
                    force=force,
                )
            
            console.print(f"  [green]✓[/green] {split} complete")
            succeeded.append(split)
            
        except Exception as e:
            console.print(f"  [red]✗[/red] {split} failed: {str(e)}")
            console.print_exception(show_locals=False)
            if not force:
                console.print("    [dim]Use --force to regenerate[/dim]")
            failed.append(split)
        
        finally:
            # Explicit memory cleanup between splits to prevent accumulation
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        console.print()  # Blank line between splits
    
    # Print appropriate summary based on results
    if failed and not succeeded:
        console.print(f"[red]✗ All splits failed for {dataset}[/red]")
        raise typer.Exit(code=1)
    elif failed:
        console.print(f"[yellow]⚠ Partial success for {dataset}: {len(succeeded)}/{len(splits)} splits completed[/yellow]")
        console.print(f"  [green]Succeeded:[/green] {', '.join(succeeded)}")
        console.print(f"  [red]Failed:[/red] {', '.join(failed)}")
    else:
        console.print(f"[green]✓ Embeddings generated for {dataset} ({len(succeeded)} splits)[/green]")


@app.command()
def verify(
    experiment_config: str = typer.Argument(..., help="Experiment configuration file"),
    device: Optional[str] = typer.Option(
        None,
        "--device", "-d",
        help="Override device from experiment config"
    ),
):
    """
    Verify that embeddings exist and are valid for an experiment.
    
    Checks that:
    - Cache directory exists
    - Metadata file is valid
    - Embeddings exist for all splits
    - H5 files are readable
    - Embeddings match experiment's input_len/output_len
    
    Examples:
        python -m cli.inference verify configs/experiments/timecma_test.yaml
    """
    import yaml
    from embedder.llm_embedder import LLMEmbedder
    
    # Experiment config is REQUIRED
    config_path = Path(experiment_config)
    if not config_path.exists():
        console.print(f"[red]Error: Experiment config file not found: {config_path}[/red]")
        raise typer.Exit(code=1)
    
    # Load experiment config to extract dataset name
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check for llm_embedding section
    if 'llm_embedding' not in config:
        console.print(f"[yellow]Note: Experiment config does not have 'llm_embedding' section[/yellow]")
        console.print("[dim]This experiment does not use LLM embeddings.[/dim]")
        raise typer.Exit(code=0)
    
    # Extract dataset name from config
    dataset = config.get('data', {}).get('name')
    if not dataset:
        console.print("[red]Error: Experiment config missing data.name[/red]")
        raise typer.Exit(code=1)
    
    try:
        embedder = LLMEmbedder.from_experiment_config(
            experiment_config_path=str(config_path),
            device=device,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)
    
    status = embedder.verify_cache(dataset)
    
    console.print(f"\n[bold cyan]Embedding Cache Verification[/bold cyan]")
    console.print(f"  Experiment: [green]{config_path.name}[/green]")
    console.print(f"  Dataset: [green]{dataset}[/green]")
    console.print(f"  Model: [green]{embedder.model_name}[/green]")
    console.print(f"  Input Length: [green]{embedder.input_len}[/green]")
    console.print(f"  Output Length: [green]{embedder.output_len}[/green]")
    console.print(f"  Cache: [dim]{status.get('cache_dir', 'N/A')}[/dim]")
    console.print()
    
    if status['valid']:
        console.print("[green]✓ All embeddings verified and complete[/green]")
        for split, valid in status.get('splits', {}).items():
            console.print(f"  [green]✓[/green] {split}: complete")
    else:
        # Check if any splits are resumable
        resumable = status.get('resumable', {})
        if resumable:
            console.print("[yellow]⚡ Partial embeddings found (resumable)[/yellow]")
            for split, samples in resumable.items():
                console.print(f"  [yellow]⚡[/yellow] {split}: {samples:,} samples (can resume)")
            for split, valid in status.get('splits', {}).items():
                if valid:
                    console.print(f"  [green]✓[/green] {split}: complete")
                elif split not in resumable:
                    console.print(f"  [red]✗[/red] {split}: missing")
        else:
            console.print("[red]✗ Verification failed[/red]")
            for issue in status.get('issues', []):
                console.print(f"  [yellow]• {issue}[/yellow]")
        
        console.print(f"\n[dim]Run 'python -m cli.inference generate {experiment_config}' to fix/resume[/dim]")
        raise typer.Exit(code=1)


@app.command("verify-suite")
def verify_suite(
    suite_config_path: str = typer.Argument(..., help="Path to experiment suite configuration file"),
    filter_experiments: Optional[str] = typer.Option(
        None,
        "--filter",
        help="Filter experiments by name pattern (case-insensitive)"
    ),
):
    """
    Verify that embeddings exist for all experiments in a suite.
    
    Iterates over enabled experiments in the suite, checks which ones
    require LLM embeddings, and verifies the cache status for each.
    
    Examples:
        # Verify all experiments in a suite
        python -m cli.inference verify-suite configs/experiment_suites/timecma_test.yaml
        
        # Verify only specific experiments
        python -m cli.inference verify-suite configs/experiment_suites/timecma_test.yaml --filter traffic
    """
    from runs.suite_executor import load_suite_config, load_template
    from utils.config_utils import merge_configs
    from embedder.llm_embedder import LLMEmbedder
    
    # Suite config is REQUIRED
    config_path = Path(suite_config_path)
    if not config_path.exists():
        console.print(f"[red]Error: Suite config file not found: {config_path}[/red]")
        raise typer.Exit(code=1)
    
    # Load suite config
    try:
        suite_config = load_suite_config(str(suite_config_path))
    except Exception as e:
        console.print(f"[red]Error loading suite config: {e}[/red]")
        raise typer.Exit(code=1)
    
    suite_info = suite_config.get('suite', {})
    suite_name = suite_info.get('name', 'unknown')
    
    # Get enabled experiments
    experiments = [
        exp for exp in suite_info.get('experiments', [])
        if exp.get('enabled', True)
    ]
    
    # Filter experiments if requested
    if filter_experiments:
        experiments = [
            exp for exp in experiments
            if filter_experiments.lower() in exp.get('name', '').lower()
        ]
        if not experiments:
            console.print(f"[yellow]No experiments match filter '{filter_experiments}'[/yellow]")
            raise typer.Exit(code=0)
    
    console.print(f"\n[bold cyan]LLM Embedding Verification for Suite: {suite_name}[/bold cyan]")
    console.print(f"  Experiments: [green]{len(experiments)}[/green]\n")
    
    # Collect unique embedding configurations (same dedup as generate-suite)
    embedding_configs = {}
    skipped_experiments = []
    
    for exp in experiments:
        exp_name = exp.get('name', 'unknown')
        template_path = exp.get('template')
        overrides = exp.get('overrides', {})
        
        if not template_path:
            skipped_experiments.append(exp_name)
            continue
        
        # Load template and merge with overrides
        try:
            template = load_template(template_path)
            final_config = merge_configs(template, overrides)
        except Exception as e:
            console.print(f"  [yellow]⚠[/yellow] {exp_name}: Error loading config: {e}")
            skipped_experiments.append(exp_name)
            continue
        
        # Check for llm_embedding section
        llm_embedding = final_config.get('llm_embedding')
        if not llm_embedding:
            console.print(f"  [dim]○[/dim] {exp_name}: No llm_embedding section")
            skipped_experiments.append(exp_name)
            continue
        
        # Extract key parameters
        dataset = final_config.get('data', {}).get('name')
        if not dataset:
            skipped_experiments.append(exp_name)
            continue
        
        training = final_config.get('training', {})
        input_len = training.get('input_len', 96)
        output_len = training.get('output_len', 96)
        model_name = llm_embedding.get('model_name', 'gpt2')
        
        # Create deduplication key
        config_key = (dataset, input_len, output_len, model_name)
        
        if config_key not in embedding_configs:
            embedding_configs[config_key] = (exp_name, final_config)
    
    if not embedding_configs:
        console.print(f"[yellow]No experiments require LLM embeddings[/yellow]")
        raise typer.Exit(code=0)
    
    console.print(f"[bold]Verifying {len(embedding_configs)} unique configurations...[/bold]\n")
    
    all_valid = True
    verified = []
    failed = []
    
    for config_key, (exp_name, final_config) in embedding_configs.items():
        dataset, input_len, output_len, model_name = config_key
        
        console.print(f"[bold cyan]━━━ {dataset} (in={input_len}, out={output_len}, model={model_name}) ━━━[/bold cyan]")
        
        # Create embedder to get proper paths
        llm_config = final_config.get('llm_embedding', {})
        training = final_config.get('training', {})
        
        embedder = LLMEmbedder(
            model_name=llm_config.get('model_name', 'gpt2'),
            cache_dir=llm_config.get('cache_dir', './LLM_cache/'),
            data_root=final_config.get('base_data_path', './data/'),
            prompt_template=llm_config.get('prompt_template', 'timecma_v1'),
            prompt_config=llm_config.get('prompt_config', {'value_format': 'integer', 'include_timestamps': True}),
            input_len=input_len,
            output_len=output_len,
            scale=training.get('scale', True),
            data_config_path=final_config.get('data', {}).get('config_path'),
        )
        
        # Verify cache
        try:
            status = embedder.verify_cache(dataset)
            
            if status['valid']:
                console.print(f"  [green]✓[/green] All splits verified and complete")
                for split in status.get('splits', {}).keys():
                    console.print(f"    [green]✓[/green] {split}")
                verified.append(dataset)
            else:
                # Check for resumable splits
                resumable = status.get('resumable', {})
                if resumable:
                    console.print(f"  [yellow]⚡[/yellow] Partial (resumable)")
                    for split, samples in resumable.items():
                        console.print(f"    [yellow]⚡[/yellow] {split}: {samples:,} samples")
                    for split, valid in status.get('splits', {}).items():
                        if valid:
                            console.print(f"    [green]✓[/green] {split}: complete")
                else:
                    console.print(f"  [red]✗[/red] Verification failed")
                    for issue in status.get('issues', []):
                        console.print(f"    [yellow]• {issue}[/yellow]")
                failed.append(dataset)
                all_valid = False
        except Exception as e:
            console.print(f"  [red]✗[/red] Error: {e}")
            failed.append(dataset)
            all_valid = False
        
        console.print()
    
    # Summary
    if all_valid:
        console.print(f"[green]✓ All embeddings verified ({len(verified)} configurations)[/green]")
    else:
        console.print(f"[red]✗ Verification failed[/red]")
        console.print(f"  [green]Verified:[/green] {len(verified)}")
        console.print(f"  [red]Failed:[/red] {len(failed)}")
        console.print(f"\n[dim]Run 'python -m cli.inference generate-suite {suite_config_path}' to fix[/dim]")
        raise typer.Exit(code=1)


@app.command("generate-suite")
def generate_suite(
    suite_config_path: str = typer.Argument(..., help="Path to experiment suite configuration file"),
    splits: List[str] = typer.Option(
        ["train", "val", "test"],
        "--splits", "-s",
        help="Data splits to process"
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device", "-d",
        help="Override device from experiment config"
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size", "-b",
        help="Override batch size from config"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Force recompute existing embeddings"
    ),
    filter_experiments: Optional[str] = typer.Option(
        None,
        "--filter",
        help="Filter experiments by name pattern (case-insensitive)"
    ),
    memory_efficient: bool = typer.Option(
        False,
        "--memory-efficient", "-m",
        help="Use memory-efficient chunked processing (for large datasets)"
    ),
    chunk_size: int = typer.Option(
        1000,
        "--chunk-size",
        help="Samples per chunk in memory-efficient mode"
    ),
    gpu_monitor: bool = typer.Option(
        True,
        "--gpu-monitor/--no-gpu-monitor",
        help="Enable GPU utilization monitoring in memory-efficient mode"
    ),
):
    """
    Generate LLM embeddings for all experiments in a suite.
    
    Iterates over enabled experiments in the suite, finds those with
    llm_embedding configurations, and generates embeddings for each
    unique (dataset, input_len, output_len, model) combination.
    
    Duplicate embedding requests (same dataset/lengths/model) are
    automatically deduplicated to avoid redundant computation.
    
    Examples:
        # Generate embeddings for all experiments in a suite
        python -m cli.inference generate-suite configs/experiment_suites/timecma_test.yaml
        
        # Generate only for specific experiments
        python -m cli.inference generate-suite configs/experiment_suites/timecma_test.yaml --filter traffic
        
        # Force regenerate all
        python -m cli.inference generate-suite configs/experiment_suites/timecma_test.yaml --force
    """
    from runs.suite_executor import load_suite_config, load_template
    from utils.config_utils import merge_configs
    from embedder.llm_embedder import LLMEmbedder
    
    # Suite config is REQUIRED
    config_path = Path(suite_config_path)
    if not config_path.exists():
        console.print(f"[red]Error: Suite config file not found: {config_path}[/red]")
        raise typer.Exit(code=1)
    
    # Load suite config
    try:
        suite_config = load_suite_config(str(suite_config_path))
    except Exception as e:
        console.print(f"[red]Error loading suite config: {e}[/red]")
        raise typer.Exit(code=1)
    
    suite_info = suite_config.get('suite', {})
    suite_name = suite_info.get('name', 'unknown')
    
    # Get enabled experiments
    experiments = [
        exp for exp in suite_info.get('experiments', [])
        if exp.get('enabled', True)
    ]
    
    # Filter experiments if requested
    if filter_experiments:
        experiments = [
            exp for exp in experiments
            if filter_experiments.lower() in exp.get('name', '').lower()
        ]
        if not experiments:
            console.print(f"[yellow]No experiments match filter '{filter_experiments}'[/yellow]")
            raise typer.Exit(code=0)
    
    console.print(f"\n[bold cyan]LLM Embedding Generation for Suite: {suite_name}[/bold cyan]")
    console.print(f"  Experiments: [green]{len(experiments)}[/green]")
    if memory_efficient:
        console.print(f"  Mode: [yellow]Memory-Efficient forced via CLI (chunk_size={chunk_size})[/yellow]")
    else:
        console.print(f"  Mode: [dim]Per-experiment config (use --memory-efficient to override all)[/dim]")
    console.print()
    
    # Collect unique embedding configurations to avoid duplicates
    # Key: (dataset, input_len, output_len, model_name)
    # Value: (experiment_name, final_config)
    embedding_configs = {}
    skipped_experiments = []
    
    for exp in experiments:
        exp_name = exp.get('name', 'unknown')
        template_path = exp.get('template')
        overrides = exp.get('overrides', {})
        
        if not template_path:
            console.print(f"  [yellow]⚠[/yellow] {exp_name}: No template specified, skipping")
            skipped_experiments.append(exp_name)
            continue
        
        # Load template and merge with overrides
        try:
            template = load_template(template_path)
            final_config = merge_configs(template, overrides)
        except Exception as e:
            console.print(f"  [yellow]⚠[/yellow] {exp_name}: Error loading config: {e}")
            skipped_experiments.append(exp_name)
            continue
        
        # Check for llm_embedding section
        llm_embedding = final_config.get('llm_embedding')
        if not llm_embedding:
            console.print(f"  [dim]○[/dim] {exp_name}: No llm_embedding section, skipping")
            skipped_experiments.append(exp_name)
            continue
        
        # Extract key parameters
        dataset = final_config.get('data', {}).get('name')
        if not dataset:
            console.print(f"  [yellow]⚠[/yellow] {exp_name}: No data.name, skipping")
            skipped_experiments.append(exp_name)
            continue
        
        training = final_config.get('training', {})
        input_len = training.get('input_len', 96)
        output_len = training.get('output_len', 96)
        model_name = llm_embedding.get('model_name', 'gpt2')
        
        # Create deduplication key
        config_key = (dataset, input_len, output_len, model_name)
        
        if config_key in embedding_configs:
            # Already have this configuration
            existing_exp = embedding_configs[config_key][0]
            console.print(f"  [dim]○[/dim] {exp_name}: Same as {existing_exp}, will reuse")
        else:
            embedding_configs[config_key] = (exp_name, final_config)
            console.print(f"  [green]●[/green] {exp_name}: {dataset} (in={input_len}, out={output_len}, model={model_name})")
    
    if not embedding_configs:
        console.print(f"\n[yellow]No experiments require LLM embedding generation[/yellow]")
        raise typer.Exit(code=0)
    
    console.print(f"\n[bold]Generating embeddings for {len(embedding_configs)} unique configurations...[/bold]\n")
    
    succeeded = []
    failed = []
    
    for config_key, (exp_name, final_config) in embedding_configs.items():
        dataset, input_len, output_len, model_name = config_key
        
        console.print(f"[bold cyan]━━━ {dataset} (in={input_len}, out={output_len}) ━━━[/bold cyan]")
        
        # Create embedder from the merged config
        llm_config = final_config.get('llm_embedding', {})
        training = final_config.get('training', {})
        
        # Determine device
        effective_device = device
        if effective_device is None:
            device_config = final_config.get('device', {})
            gpu = device_config.get('gpu', 0)
            use_gpu = device_config.get('use_gpu', True)
            effective_device = f'cuda:{gpu}' if use_gpu else 'cpu'
        
        # Create embedder manually with merged config values and console for progress bars
        embedder = LLMEmbedder(
            model_name=llm_config.get('model_name', 'gpt2'),
            device=effective_device,
            cache_dir=llm_config.get('cache_dir', './LLM_cache/'),
            data_root=final_config.get('base_data_path', './data/'),
            quantization=llm_config.get('quantization'),
            extraction_mode=llm_config.get('extraction_mode', 'last_token'),
            prompt_template=llm_config.get('prompt_template', 'timecma_v1'),
            prompt_config=llm_config.get('prompt_config', {'value_format': 'integer', 'include_timestamps': True}),
            max_length=llm_config.get('max_length', 512),
            input_len=input_len,
            output_len=output_len,
            scale=training.get('scale', True),
            data_config_path=final_config.get('data', {}).get('config_path'),
            console=console,
        )
        embedder.default_batch_size = llm_config.get('batch_size', 64)
        
        # Override batch size if provided via CLI
        effective_batch_size = batch_size if batch_size is not None else embedder.default_batch_size
        
        # Determine memory-efficient mode for this experiment:
        # CLI flag > experiment config > default (False)
        use_memory_efficient = memory_efficient  # CLI flag takes priority
        if not memory_efficient:
            # Check experiment config
            use_memory_efficient = llm_config.get('memory_efficient', False)
        
        # Determine chunk size: CLI (if not default) > experiment config > default
        effective_chunk_size = chunk_size
        if chunk_size == 1000:  # Default value, check config
            effective_chunk_size = llm_config.get('chunk_size', 1000)
        
        # Show mode for this dataset if memory-efficient
        if use_memory_efficient:
            console.print(f"  [dim]Mode: Memory-Efficient (chunk_size={effective_chunk_size})[/dim]")
        
        for split in splits:
            console.print(f"  [bold]{split}:[/bold]")
            try:
                if use_memory_efficient:
                    # Use memory-efficient chunked processing
                    embedder.generate_ts_embeddings_chunked(
                        dataset=dataset,
                        split=split,
                        batch_size=effective_batch_size,
                        chunk_size=effective_chunk_size,
                        force=force,
                        enable_gpu_monitor=gpu_monitor,
                    )
                else:
                    # Use standard processing
                    embedder.generate_ts_embeddings(
                        dataset=dataset,
                        split=split,
                        batch_size=effective_batch_size,
                        force=force,
                    )
                console.print(f"    [green]✓[/green] {split} complete")
                succeeded.append(f"{dataset}/{split}")
            except Exception as e:
                console.print(f"    [red]✗[/red] {split}: {str(e)}")
                console.print_exception(show_locals=False)
                failed.append(f"{dataset}/{split}")
            finally:
                # Explicit memory cleanup between splits to prevent accumulation
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        console.print()  # Blank line between datasets
    
    # Print summary
    console.print()
    if failed and not succeeded:
        console.print(f"[red]✗ All embedding generation failed[/red]")
        raise typer.Exit(code=1)
    elif failed:
        console.print(f"[yellow]⚠ Partial success: {len(succeeded)} succeeded, {len(failed)} failed[/yellow]")
    else:
        console.print(f"[green]✓ All embeddings generated successfully ({len(succeeded)} total)[/green]")


@app.command("estimate-memory")
def estimate_memory(
    model: str = typer.Argument(..., help="Model name (e.g., Qwen/Qwen2.5-72B-Instruct)"),
    quantization: Optional[str] = typer.Option(
        None,
        "--quantization", "-q",
        help="Quantization: '4bit', '8bit', or None for fp16"
    ),
):
    """
    Estimate GPU memory requirements for a model.
    
    Provides memory estimates and recommended hardware for running
    a given model with optional quantization.
    
    Examples:
        python -m cli.inference estimate-memory gpt2
        python -m cli.inference estimate-memory Qwen/Qwen2.5-72B-Instruct --quantization 4bit
    """
    from embedder.llm_utils import estimate_model_memory
    
    result = estimate_model_memory(model, quantization)
    
    console.print(f"\n[bold cyan]Memory Estimate: {model}[/bold cyan]")
    console.print(f"  Quantization: [green]{quantization or 'None (fp16)'}[/green]")
    console.print()
    console.print(f"  Parameters: [yellow]{result['params_billions']:.1f}B[/yellow]")
    console.print(f"  Embedding Dim: [yellow]{result['embed_dim']}[/yellow]")
    console.print(f"  Estimated VRAM: [yellow]{result['vram_gb']:.1f} GB[/yellow]")
    console.print(f"  Recommended GPU: [green]{result['recommended_gpu']}[/green]")


@app.command("list-models")
def list_models():
    """
    List supported LLM models with their specifications.
    
    Shows all pre-configured models with parameter counts,
    embedding dimensions, and memory requirements.
    """
    from embedder.llm_registry import LLMRegistry
    
    models = LLMRegistry.list_supported_models()
    
    table = Table(title="Supported LLM Models")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Parameters", style="green", justify="right")
    table.add_column("Embed Dim", style="yellow", justify="right")
    table.add_column("FP16 VRAM", style="magenta", justify="right")
    table.add_column("4-bit VRAM", style="blue", justify="right")
    
    for model_info in models:
        table.add_row(
            model_info['name'],
            model_info['params'],
            str(model_info['embed_dim']),
            f"{model_info['min_vram_fp16']:.0f} GB",
            f"{model_info['min_vram_4bit']:.0f} GB",
        )
    
    console.print(table)
    
    console.print("\n[dim]Use 'python -m cli.inference estimate-memory <model>' for detailed estimates[/dim]")


@app.command("list-cached")
def list_cached(
    experiment_config: str = typer.Argument(..., help="Experiment configuration file"),
):
    """
    List cached embeddings for an experiment.
    
    Shows all cached embedding configurations with their
    models, templates, and creation times.
    
    Examples:
        python -m cli.inference list-cached configs/experiments/timecma_test.yaml
    """
    import yaml
    from embedder.llm_cache import LLMEmbeddingCache
    from embedder.llm_embedder import LLMEmbedder
    
    # Experiment config is REQUIRED
    config_path = Path(experiment_config)
    if not config_path.exists():
        console.print(f"[red]Error: Experiment config file not found: {config_path}[/red]")
        raise typer.Exit(code=1)
    
    # Load experiment config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check for llm_embedding section
    if 'llm_embedding' not in config:
        console.print("[yellow]Note: Experiment config does not have 'llm_embedding' section[/yellow]")
        console.print("[dim]This experiment does not use LLM embeddings.[/dim]")
        raise typer.Exit(code=0)
    
    # Extract dataset name from config
    dataset = config.get('data', {}).get('name')
    if not dataset:
        console.print("[red]Error: Experiment config missing data.name[/red]")
        raise typer.Exit(code=1)
    
    # Create embedder to resolve path
    embedder = LLMEmbedder()
    
    try:
        data_dir = embedder._get_data_directory(dataset)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)
    
    cache = LLMEmbeddingCache(str(data_dir), dataset)
    configs = cache.list_cached_configs()
    
    if not configs:
        console.print(f"[yellow]No cached embeddings found for {dataset}[/yellow]")
        console.print(f"[dim]Run 'python -m cli.inference generate {experiment_config}' to create[/dim]")
        return
    
    console.print(f"\n[bold cyan]Cached Embeddings[/bold cyan]")
    console.print(f"  Experiment: [green]{config_path.name}[/green]")
    console.print(f"  Dataset: [green]{dataset}[/green]")
    console.print(f"  Cache Dir: [dim]{data_dir}/llm_embeddings/[/dim]\n")
    
    table = Table()
    table.add_column("Hash", style="dim", no_wrap=True)
    table.add_column("Model", style="cyan")
    table.add_column("Template", style="green")
    table.add_column("Input/Output", style="yellow")
    table.add_column("Created", style="dim")
    
    for cached_config in configs:
        # Try to get input/output len from metadata if available
        input_len = cached_config.get('input_len', '?')
        output_len = cached_config.get('output_len', '?')
        
        table.add_row(
            cached_config['hash'][:8] + "...",
            cached_config['model_name'],
            cached_config['prompt_template'],
            f"{input_len}/{output_len}",
            cached_config['created_at'][:19],  # Trim to datetime
        )
    
    console.print(table)


@app.command("gpu-info")
def gpu_info():
    """
    Display current GPU memory information.
    
    Shows total, used, and available VRAM for planning
    which models can be loaded.
    """
    from embedder.llm_utils import get_gpu_memory_info
    import torch
    
    if not torch.cuda.is_available():
        console.print("[red]No CUDA GPUs available[/red]")
        raise typer.Exit(code=1)
    
    num_gpus = torch.cuda.device_count()
    
    console.print("\n[bold cyan]GPU Memory Information[/bold cyan]")
    console.print(f"  Available GPUs: [green]{num_gpus}[/green]\n")
    
    table = Table()
    table.add_column("Device", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Total", style="yellow", justify="right")
    table.add_column("Used", style="magenta", justify="right")
    table.add_column("Free", style="blue", justify="right")
    
    for i in range(num_gpus):
        device = f"cuda:{i}"
        info = get_gpu_memory_info(device)
        name = torch.cuda.get_device_name(i)
        
        table.add_row(
            device,
            name[:30],  # Truncate long names
            f"{info['total_gb']:.1f} GB",
            f"{info['used_gb']:.1f} GB",
            f"{info['free_gb']:.1f} GB",
        )
    
    console.print(table)


if __name__ == "__main__":
    app()

