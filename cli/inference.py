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
- verify: Verify that embeddings exist and are valid for an experiment
- estimate-memory: Estimate GPU memory requirements for a model
- list-models: List supported LLM models with specifications
- list-cached: List cached embeddings for a dataset
- list-datasets: List available datasets for embedding generation
- gpu-info: Display current GPU memory information

Examples:
    # List available datasets
    python -m cli.inference list-datasets
    
    # Generate embeddings for an experiment
    python -m cli.inference generate configs/experiments/timecma_test.yaml
    
    # Force regeneration of test split only
    python -m cli.inference generate configs/experiments/timecma_test.yaml --splits test --force
    
    # Verify embeddings exist for an experiment
    python -m cli.inference verify configs/experiments/timecma_test.yaml
    
    # Check memory requirements
    python -m cli.inference estimate-memory Qwen/Qwen2.5-72B-Instruct --quantization 4bit
"""

import typer
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn


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
    
    # Load embedder from experiment config
    try:
        embedder = LLMEmbedder.from_experiment_config(
            experiment_config_path=str(config_path),
            device=device,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)
    
    # Batch size priority: CLI > config > default
    effective_batch_size = batch_size if batch_size is not None else embedder.default_batch_size
    
    console.print("\n[bold cyan]LLM Embedding Generation (Experiment-Driven)[/bold cyan]")
    console.print(f"  Experiment: [green]{config_path.name}[/green]")
    console.print(f"  Dataset: [green]{dataset}[/green]")
    console.print(f"  Model: [green]{embedder.model_name}[/green]")
    console.print(f"  Quantization: [green]{embedder.quantization or 'None (fp16)'}[/green]")
    console.print(f"  Device: [green]{embedder.device}[/green]")
    console.print(f"  Input Length: [green]{embedder.input_len}[/green]")
    console.print(f"  Output Length: [green]{embedder.output_len}[/green]")
    console.print(f"  Splits: [green]{', '.join(splits)}[/green]")
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        overall_task = progress.add_task(
            f"[cyan]Processing {dataset}...",
            total=len(splits)
        )
        
        succeeded = []
        failed = []
        
        for split in splits:
            progress.update(overall_task, description=f"[cyan]Processing {split} split...")
            
            try:
                # Create a sub-progress for batches
                def progress_callback(current, total):
                    # Update would be too frequent, skip for now
                    pass
                
                embedder.generate_ts_embeddings(
                    dataset=dataset,
                    split=split,
                    batch_size=effective_batch_size,
                    force=force,
                    progress_callback=progress_callback,
                )
                
                console.print(f"  [green]✓[/green] {split} complete")
                succeeded.append(split)
                
            except Exception as e:
                console.print(f"  [red]✗[/red] {split} failed: {str(e)}")
                console.print_exception(show_locals=False)
                if not force:
                    console.print("    [dim]Use --force to regenerate[/dim]")
                failed.append(split)
            
            progress.advance(overall_task)
    
    # Print appropriate summary based on results
    if failed and not succeeded:
        console.print(f"\n[red]✗ All splits failed for {dataset}[/red]")
        raise typer.Exit(code=1)
    elif failed:
        console.print(f"\n[yellow]⚠ Partial success for {dataset}: {len(succeeded)}/{len(splits)} splits completed[/yellow]")
        console.print(f"  [green]Succeeded:[/green] {', '.join(succeeded)}")
        console.print(f"  [red]Failed:[/red] {', '.join(failed)}")
    else:
        console.print(f"\n[green]✓ Embeddings generated for {dataset} ({len(succeeded)} splits)[/green]")


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
        console.print("[green]✓ All embeddings verified[/green]")
        for split, valid in status.get('splits', {}).items():
            console.print(f"  [green]✓[/green] {split}")
    else:
        console.print("[red]✗ Verification failed[/red]")
        for issue in status.get('issues', []):
            console.print(f"  [yellow]• {issue}[/yellow]")
        
        console.print(f"\n[dim]Run 'python -m cli.inference generate {experiment_config}' to fix[/dim]")
        raise typer.Exit(code=1)


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
    dataset: str = typer.Argument(..., help="Dataset name"),
    data_root: str = typer.Option(
        "./data/",
        "--data-root",
        help="Root directory for datasets"
    ),
):
    """
    List cached embeddings for a dataset.
    
    Shows all cached embedding configurations with their
    models, templates, and creation times.
    """
    from embedder.llm_cache import LLMEmbeddingCache
    
    cache = LLMEmbeddingCache(data_root, dataset)
    configs = cache.list_cached_configs()
    
    if not configs:
        console.print(f"[yellow]No cached embeddings found for {dataset}[/yellow]")
        console.print(f"[dim]Run 'python -m cli.inference generate {dataset}' to create[/dim]")
        return
    
    table = Table(title=f"Cached Embeddings: {dataset}")
    table.add_column("Hash", style="dim", no_wrap=True)
    table.add_column("Model", style="cyan")
    table.add_column("Template", style="green")
    table.add_column("Created", style="yellow")
    
    for config in configs:
        table.add_row(
            config['hash'][:8] + "...",
            config['model_name'],
            config['prompt_template'],
            config['created_at'][:19],  # Trim to datetime
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


@app.command("list-datasets")
def list_datasets():
    """
    List available datasets for LLM embedding generation.
    
    Shows all datasets that can be used with the 'generate' command,
    organized by type (Time-MMD, TTC, Fidel-TS).
    
    Dataset naming conventions:
        - Time-MMD: time_mmd_<domain>     (e.g., time_mmd_traffic)
        - TTC:      ttc_<domain>          (e.g., ttc_climate)
        - Fidel-TS: fidel_<dataset>       (e.g., fidel_ETT)
        - Fidel-TS: fidel_<dataset>:<cfg> (e.g., fidel_ETT:fullETT_M)
    
    Examples:
        python -m cli.inference list-datasets
    """
    from pathlib import Path
    
    console.print("\n[bold cyan]Available Datasets for LLM Embedding Generation[/bold cyan]\n")
    
    # ==========================================================================
    # Time-MMD Datasets
    # ==========================================================================
    time_mmd_path = Path("data_configs/time_mmd")
    
    if time_mmd_path.exists():
        console.print("[bold magenta]1. Time-MMD Datasets[/bold magenta]")
        console.print("[dim]Usage: python -m cli.inference generate time_mmd_<domain> <config>[/dim]\n")
        
        table = Table()
        table.add_column("Dataset Name", style="cyan")
        table.add_column("Domain", style="green")
        table.add_column("Config Path", style="dim")
        
        domains = sorted([d.name for d in time_mmd_path.iterdir() if d.is_dir() and not d.name.startswith('.')])
        
        for domain in domains:
            config_file = time_mmd_path / domain / "config.yaml"
            if config_file.exists():
                dataset_name = f"time_mmd_{domain.lower()}"
                table.add_row(
                    dataset_name,
                    domain,
                    str(config_file),
                )
        
        console.print(table)
        console.print()
    else:
        console.print("[yellow]Time-MMD datasets not found at data_configs/time_mmd/[/yellow]\n")
    
    # ==========================================================================
    # TTC Datasets
    # ==========================================================================
    ttc_path = Path("data_configs/ttc")
    
    if ttc_path.exists():
        console.print("[bold magenta]2. TTC Datasets (Time-Text Corpus)[/bold magenta]")
        console.print("[dim]Usage: python -m cli.inference generate ttc_<domain> <config>[/dim]\n")
        
        table = Table()
        table.add_column("Dataset Name", style="cyan")
        table.add_column("Domain", style="green")
        table.add_column("Config Path", style="dim")
        
        domains = sorted([d.name for d in ttc_path.iterdir() if d.is_dir() and not d.name.startswith('.')])
        
        for domain in domains:
            config_file = ttc_path / domain / "config.yaml"
            if config_file.exists():
                dataset_name = f"ttc_{domain.lower()}"
                table.add_row(
                    dataset_name,
                    domain,
                    str(config_file),
                )
        
        console.print(table)
        console.print()
    else:
        console.print("[yellow]TTC datasets not found at data_configs/ttc/[/yellow]\n")
    
    # ==========================================================================
    # Fidel-TS Datasets
    # ==========================================================================
    console.print("[bold magenta]3. Fidel-TS Datasets[/bold magenta]")
    console.print("[dim]Usage: python -m cli.inference generate fidel_<dataset> <config>[/dim]")
    console.print("[dim]       python -m cli.inference generate fidel_<dataset>:<config_name> <config>[/dim]\n")
    
    # Default config mappings for Fidel-TS datasets
    fidel_datasets = {
        'Bear_room': ('fullBear', 'Bear room temperature & weather'),
        'California_ISO': ('fullCAISO', 'California energy grid data'),
        'Canada_photovoltaics_plants': ('fullCPP', 'Canadian solar power plants'),
        'electricity': ('fullelectricity', 'Electricity consumption'),
        'ETT': ('fullETT_H', 'Electricity Transformer Temperature'),
        'Germany_Renewable_Power_Grid': ('fullGRPG', 'German renewable energy grid'),
        'Jena_Atmospheric_Physics': ('fullJAP', 'Jena weather station data'),
        'NYC_traffic_speed': ('fullNYCTS', 'NYC traffic speed data'),
        'traffic': ('fulltraffic', 'Road traffic data'),
        'weather': ('weather', 'Weather forecasting data'),
    }
    
    table = Table()
    table.add_column("Dataset Name", style="cyan")
    table.add_column("Default Config", style="green")
    table.add_column("Description", style="dim")
    table.add_column("Other Configs", style="yellow")
    
    data_configs_path = Path("data_configs")
    
    for dataset_name, (default_config, description) in sorted(fidel_datasets.items()):
        dataset_dir = data_configs_path / dataset_name
        if dataset_dir.exists():
            # Get all yaml configs in this directory
            all_configs = sorted([f.stem for f in dataset_dir.glob('*.yaml')])
            other_configs = [c for c in all_configs if c != default_config]
            other_configs_str = ", ".join(other_configs[:3])  # Show first 3
            if len(other_configs) > 3:
                other_configs_str += f" (+{len(other_configs) - 3} more)"
            
            table.add_row(
                f"fidel_{dataset_name}",
                default_config,
                description,
                other_configs_str if other_configs else "-",
            )
    
    console.print(table)
    console.print()
    
    console.print("[dim]To use a non-default config: fidel_<dataset>:<config_name>[/dim]")
    console.print("[dim]Example: fidel_ETT:fullETT_M uses fullETT_M.yaml instead of fullETT_H.yaml[/dim]\n")
    
    # ==========================================================================
    # Example Commands (Experiment-Driven)
    # ==========================================================================
    console.print("[bold]Example Commands (Experiment-Driven)[/bold]\n")
    
    console.print("  [dim]# Generate embeddings for an experiment config[/dim]")
    console.print("  python -m cli.inference generate configs/experiments/timecma_test.yaml\n")
    
    console.print("  [dim]# Generate only test split[/dim]")
    console.print("  python -m cli.inference generate configs/experiments/timecma_test.yaml --splits test\n")
    
    console.print("  [dim]# Force regenerate all splits[/dim]")
    console.print("  python -m cli.inference generate configs/experiments/timecma_test.yaml --force\n")
    
    console.print("  [dim]# Verify embeddings exist for an experiment[/dim]")
    console.print("  python -m cli.inference verify configs/experiments/timecma_test.yaml\n")
    
    # ==========================================================================
    # LLM Embedding Configuration Note
    # ==========================================================================
    console.print("[bold]LLM Embedding Configuration[/bold]\n")
    console.print("[dim]LLM embeddings are configured in your experiment config file.[/dim]")
    console.print("[dim]Add an 'llm_embedding' section to your experiment config:[/dim]\n")
    console.print("""[yellow]llm_embedding:
  model_name: "gpt2"           # HuggingFace model name
  batch_size: 64               # Batch size for LLM inference
  cache_dir: "./LLM_cache/"    # Where to cache model weights
  prompt_template: "timecma_v1" # Prompt format[/yellow]
""")


if __name__ == "__main__":
    app()

