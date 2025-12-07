"""
Evaluation command-line interface for Fidel-TS.

This module provides CLI commands for evaluating trained models:
- Per-sample metrics evaluation (computes per-sample metrics on trained checkpoints)
"""

import typer
from pathlib import Path
from typing import Optional
from cli.config.loader import load_config
from cli.config.models import ExperimentConfig
from cli.utils import handle_error, console, RICH_AVAILABLE
from utils.tools import dotdict

app = typer.Typer(
    name="evaluate",
    help="Evaluate trained time series forecasting models",
    add_completion=False
)


def _find_experiment_directory(experiment_id: str, output_dir: str = "./output") -> Path:
    """
    Find experiment directory from experiment ID.
    
    Supports both standalone experiments and suite experiments:
    - Standalone: experiment_id
    - Suite: suite_name/experiment_id
    
    Args:
        experiment_id: Experiment ID (may include suite name as prefix)
        output_dir: Base output directory
        
    Returns:
        Path to experiment directory
        
    Raises:
        FileNotFoundError: If experiment directory not found
    """
    output_path = Path(output_dir)
    
    # Check if experiment_id contains suite name (has a slash)
    if '/' in experiment_id:
        # Suite experiment: suite_name/experiment_id
        suite_name, exp_id = experiment_id.split('/', 1)
        exp_dir = output_path / suite_name / exp_id
    else:
        # Standalone experiment
        exp_dir = output_path / experiment_id
    
    if not exp_dir.exists():
        # Try to find by partial match
        if '/' in experiment_id:
            suite_name, exp_id = experiment_id.split('/', 1)
            suite_dir = output_path / suite_name
            if suite_dir.exists():
                # Search for matching experiment ID
                matching_dirs = [d for d in suite_dir.iterdir() if d.is_dir() and exp_id in d.name]
                if matching_dirs:
                    exp_dir = matching_dirs[0]
        else:
            # Search in output directory
            matching_dirs = [d for d in output_path.iterdir() if d.is_dir() and experiment_id in d.name]
            if matching_dirs:
                exp_dir = matching_dirs[0]
    
    if not exp_dir.exists():
        raise FileNotFoundError(
            f"Experiment directory not found: {exp_dir}\n"
            f"  Searched in: {output_path}\n"
            f"  Experiment ID: {experiment_id}"
        )
    
    return exp_dir


def _load_experiment_config(experiment_dir: Path) -> dict:
    """
    Load experiment configuration from experiment directory.
    
    Tries multiple locations:
    1. configs/experiment_config.yaml (new format)
    2. args.json (legacy format)
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If no config file found
    """
    # Try new format first
    config_path = experiment_dir / "configs" / "experiment_config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    # Try legacy format
    args_path = experiment_dir / "args.json"
    if args_path.exists():
        import json
        with open(args_path, 'r') as f:
            return json.load(f)
    
    raise FileNotFoundError(
        f"No configuration file found in experiment directory: {experiment_dir}\n"
        f"  Expected: configs/experiment_config.yaml or args.json"
    )


def _create_evaluation_config(experiment_dir: Path, experiment_config: dict, 
                               splits: Optional[list] = None) -> dotdict:
    """
    Create evaluation configuration from experiment directory and config.
    
    Args:
        experiment_dir: Path to experiment directory
        experiment_config: Loaded experiment configuration
        splits: Optional list of splits to process (default: ['train', 'val', 'test'])
        
    Returns:
        dotdict: Evaluation configuration compatible with evaluate_per_sample
    """
    # Extract model and data info from config
    model_name = experiment_config.get('model', {}).get('name', 'unknown')
    data_name = experiment_config.get('data', {}).get('name', 'unknown')
    
    # Extract training config for input/output lengths
    training_config = experiment_config.get('training', {})
    input_len = training_config.get('input_len', 360)
    output_len = training_config.get('output_len', 24)
    
    # Determine task type (default to TSF)
    task = experiment_config.get('experiment', {}).get('type', 'TSF')
    if task not in ['TSF', 'TGTSF', 'MTSF']:
        # Infer from model name or config
        if 'tgtsf' in model_name.lower() or 'lynx' in model_name.lower():
            task = 'TGTSF'
        elif 'mtsf' in model_name.lower():
            task = 'MTSF'
        else:
            task = 'TSF'
    
    # Create evaluation config
    eval_config = dotdict({
        'model': model_name,
        'data': data_name,
        'version': 'latest',  # Not used when checkpoint path is direct
        'checkpoint_base': str(experiment_dir),  # Use experiment dir as checkpoint base
        'input_len': input_len,
        'output_len': output_len,
        'batch_size': training_config.get('batch_size', 128),
        'task': task,
        'device': '0',  # Default GPU 0
        'splits': splits or ['train', 'val', 'test'],
        'filtered_samples': None,
        'channel_wise': False
    })
    
    return eval_config


@app.command()
def per_sample(
    experiment_id: str = typer.Argument(..., help="Experiment ID (or suite_name/experiment_id for suite experiments)"),
    output_dir: str = typer.Option("./output", "--output-dir", "-o", help="Base output directory containing experiments"),
    splits: Optional[str] = typer.Option(None, "--splits", "-s", help="Comma-separated list of splits to process (default: train,val,test)"),
    device: str = typer.Option("0", "--device", "-d", help="GPU device ID (or 'cpu')"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate experiment directory without running evaluation")
):
    """
    Compute per-sample metrics for a trained model checkpoint.
    
    This command loads a trained model from an experiment directory and computes
    per-sample metrics on train/val/test splits. Metrics are saved to parquet files
    in the experiment's metrics directory.
    
    Examples:
        # Evaluate standalone experiment
        python -m cli.evaluate per_sample 20240101-abc123def456
        
        # Evaluate suite experiment
        python -m cli.evaluate per_sample linear_models/20240101-abc123def456
        
        # Evaluate only test split
        python -m cli.evaluate per_sample 20240101-abc123def456 --splits test
        
        # Use specific output directory
        python -m cli.evaluate per_sample 20240101-abc123def456 --output-dir ./experiments
    """
    try:
        # Find experiment directory
        experiment_dir = _find_experiment_directory(experiment_id, output_dir)
        
        if RICH_AVAILABLE:
            console.print(f"[green]Found experiment directory:[/green] {experiment_dir}")
        else:
            typer.echo(f"Found experiment directory: {experiment_dir}")
        
        # Load experiment configuration
        experiment_config = _load_experiment_config(experiment_dir)
        
        # Parse splits
        split_list = None
        if splits:
            split_list = [s.strip() for s in splits.split(',')]
        
        # Create evaluation configuration
        eval_config = _create_evaluation_config(experiment_dir, experiment_config, split_list)
        eval_config.device = device
        
        if dry_run:
            if RICH_AVAILABLE:
                console.print(f"[green]✓[/green] Experiment validated: {experiment_id}")
                console.print(f"  Model: {eval_config.model}")
                console.print(f"  Data: {eval_config.data}")
                console.print(f"  Task: {eval_config.task}")
                console.print(f"  Input/Output length: {eval_config.input_len}/{eval_config.output_len}")
                console.print(f"  Splits: {eval_config.splits}")
                console.print(f"  Checkpoint directory: {experiment_dir / 'checkpoints'}")
            else:
                typer.echo(f"✓ Experiment validated: {experiment_id}")
                typer.echo(f"  Model: {eval_config.model}")
                typer.echo(f"  Data: {eval_config.data}")
                typer.echo(f"  Task: {eval_config.task}")
                typer.echo(f"  Splits: {eval_config.splits}")
            return
        
        # Import here to avoid circular imports
        from evaluation.per_sample import evaluate_per_sample
        
        # Create config structure expected by evaluate_per_sample
        config = dotdict({
            'evaluation': eval_config,
            'data_config': None  # Will use checkpoint's data config
        })
        
        if RICH_AVAILABLE:
            console.print(f"[green]Starting per-sample evaluation for:[/green] {experiment_id}")
        else:
            typer.echo(f"Starting per-sample evaluation for: {experiment_id}")
        
        # Run evaluation with direct checkpoint path
        results = evaluate_per_sample(config, checkpoint_path=str(experiment_dir))
        
        if RICH_AVAILABLE:
            console.print(f"[green]✓[/green] Per-sample evaluation completed!")
            console.print(f"  Results saved to: {experiment_dir / 'metrics' / 'per_sample'}")
        else:
            typer.echo(f"✓ Per-sample evaluation completed!")
            typer.echo(f"  Results saved to: {experiment_dir / 'metrics' / 'per_sample'}")
        
    except FileNotFoundError as e:
        if RICH_AVAILABLE:
            console.print(f"[bold red]Error:[/bold red] {e}")
        else:
            typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        handle_error(e, "Error during per-sample evaluation")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

