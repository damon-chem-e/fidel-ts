"""
CLI module for LLM inference and embedding precomputation.

Generates LLM embeddings from prompts for use with TimeCMA and similar models.
All embeddings must be precomputed before training - no on-the-fly inference supported.

This CLI provides commands for:
- generate: Generate LLM embeddings for a dataset
- verify: Verify that embeddings exist and are valid
- estimate-memory: Estimate GPU memory requirements for a model
- list-models: List supported LLM models with specifications
- list-cached: List cached embeddings for a dataset

Examples:
    # Generate embeddings with default config
    python -m cli.inference generate ETTh1
    
    # Use Qwen 70B with 4-bit quantization
    python -m cli.inference generate ETTh1 --model Qwen/Qwen2.5-72B-Instruct --quantization 4bit
    
    # Force regeneration of test split only
    python -m cli.inference generate ETTh1 --splits test --force
    
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
    dataset: str = typer.Argument(..., help="Dataset name (e.g., ETTh1, ETTm1)"),
    config: str = typer.Argument(..., help="LLM embedding configuration file (required)"),
    splits: List[str] = typer.Option(
        ["train", "val", "test"],
        "--splits", "-s",
        help="Data splits to process"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Override LLM model from config (e.g., 'gpt2', 'Qwen/Qwen2.5-72B-Instruct')"
    ),
    device: str = typer.Option(
        "cuda:0",
        "--device", "-d",
        help="Device for inference"
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
    quantization: Optional[str] = typer.Option(
        None,
        "--quantization", "-q",
        help="Override quantization from config: '4bit', '8bit', or None for fp16"
    ),
    data_root: Optional[str] = typer.Option(
        None,
        "--data-root",
        help="Override data_root from config"
    ),
):
    """
    Generate LLM embeddings for a dataset (time series → prompts → embeddings).
    
    This command generates embeddings by:
    1. Loading time series data from the dataset
    2. Converting each sample to a text prompt via TSPromptBuilder
    3. Running the LLM to extract last-token embeddings
    4. Caching results for use during training
    
    Examples:
        # Generate with default config (Qwen 72B on H200)
        python -m cli.inference generate ETTh1 model_configs/llm_embedding/default.yaml
        
        # Use smaller model config
        python -m cli.inference generate ETTh1 model_configs/llm_embedding/qwen_7b.yaml
        
        # Override model from config
        python -m cli.inference generate ETTh1 config.yaml --model gpt2
        
        # Regenerate only test split
        python -m cli.inference generate ETTh1 config.yaml --splits test --force
    """
    from embedder.llm_embedder import LLMEmbedder
    
    # Config is REQUIRED - no silent fallback to defaults
    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        console.print("[dim]Config is required. Create one in model_configs/llm_embedding/[/dim]")
        raise typer.Exit(code=1)
    
    # Load embedder from config
    embedder = LLMEmbedder.from_config(
        config_path=str(config_path),
        model_override=model,
        device=device,
        quantization=quantization,
    )
    
    # Override data_root if provided via CLI (CLI takes precedence over config)
    if data_root is not None:
        embedder.data_root = data_root
    
    # Batch size priority: CLI > config > default (16)
    if batch_size is not None:
        effective_batch_size = batch_size
    elif hasattr(embedder, 'default_batch_size'):
        effective_batch_size = embedder.default_batch_size
    else:
        effective_batch_size = 16
    
    console.print(f"\n[bold cyan]LLM Embedding Generation[/bold cyan]")
    console.print(f"  Dataset: [green]{dataset}[/green]")
    console.print(f"  Model: [green]{embedder.model_name}[/green]")
    console.print(f"  Quantization: [green]{embedder.quantization or 'None (fp16)'}[/green]")
    console.print(f"  Device: [green]{embedder.device}[/green]")
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
                
            except Exception as e:
                console.print(f"  [red]✗[/red] {split} failed: {str(e)}")
                if not force:
                    console.print(f"    [dim]Use --force to regenerate[/dim]")
            
            progress.advance(overall_task)
    
    console.print(f"\n[green]✓ Embeddings generated for {dataset}[/green]")


@app.command()
def verify(
    dataset: str = typer.Argument(..., help="Dataset name"),
    config: str = typer.Argument(..., help="LLM embedding configuration file (required)"),
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Override model from config"
    ),
    data_root: Optional[str] = typer.Option(
        None,
        "--data-root",
        help="Override data_root from config"
    ),
):
    """
    Verify that embeddings exist and are valid for a dataset.
    
    Checks that:
    - Cache directory exists
    - Metadata file is valid
    - Embeddings exist for all splits
    - H5 files are readable
    """
    from embedder.llm_embedder import LLMEmbedder
    
    # Config is REQUIRED
    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        raise typer.Exit(code=1)
    
    embedder = LLMEmbedder.from_config(str(config_path), model_override=model)
    
    if data_root is not None:
        embedder.data_root = data_root
    
    status = embedder.verify_cache(dataset)
    
    console.print(f"\n[bold cyan]Embedding Cache Verification: {dataset}[/bold cyan]")
    console.print(f"  Model: [green]{embedder.model_name}[/green]")
    console.print(f"  Cache: [dim]{status.get('cache_dir', 'N/A')}[/dim]")
    console.print()
    
    if status['valid']:
        console.print(f"[green]✓ All embeddings verified[/green]")
        for split, valid in status.get('splits', {}).items():
            console.print(f"  [green]✓[/green] {split}")
    else:
        console.print(f"[red]✗ Verification failed[/red]")
        for issue in status.get('issues', []):
            console.print(f"  [yellow]• {issue}[/yellow]")
        
        console.print(f"\n[dim]Run 'python -m cli.inference generate {dataset}' to fix[/dim]")
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
    
    console.print(f"\n[bold cyan]GPU Memory Information[/bold cyan]")
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

