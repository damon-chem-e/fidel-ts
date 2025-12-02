"""
Training command-line interface for Fidel-TS.

This module provides CLI commands for training models using different frameworks:
- PyTorch (standard torch training)
- PyTorch Lightning (lightning training with multi-GPU support)
- LLM (Large Language Model based forecasting)
- FM (Foundation Models forecasting)
"""

import typer
from pathlib import Path
from cli.config.loader import load_config_with_nested
from cli.utils import handle_error, console, RICH_AVAILABLE

app = typer.Typer(
    name="train",
    help="Train time series forecasting models",
    add_completion=False
)


@app.command()
def pytorch(
    config_path: str = typer.Argument(..., help="Path to experiment configuration file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without running training")
):
    """
    Train using PyTorch pipeline.
    
    This command runs training using the standard PyTorch training loop.
    Good for development and debugging.
    
    Example:
        python -m cli.train pytorch configs/experiments/dlinear_solar.yaml
    """
    try:
        config_hierarchy = load_config_with_nested(config_path)
        config = config_hierarchy['primary']
        
        if dry_run:
            typer.echo(f"✓ Config validated: {config_path}")
            typer.echo(f"  Model: {config.model.name}")
            typer.echo(f"  Data: {config.data.name}")
            return
        
        # Import here to avoid circular imports
        from runs.pytorch import run
        
        typer.echo(f"Starting PyTorch training with config: {config_path}")
        run(config)
        
    except FileNotFoundError as e:
        if RICH_AVAILABLE:
            console.print(f"[bold red]Error:[/bold red] Config file not found: {e}")
        else:
            typer.echo(f"Error: Config file not found: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        handle_error(e, "Error during training")
        raise typer.Exit(code=1)


@app.command()
def lightning(
    config_path: str = typer.Argument(..., help="Path to experiment configuration file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without running training")
):
    """
    Train using PyTorch Lightning pipeline.
    
    This command runs training using PyTorch Lightning for better
    multi-GPU support, experiment tracking, and structured training.
    
    Example:
        python -m cli.train lightning configs/experiments/dlinear_solar.yaml
    """
    try:
        config_hierarchy = load_config_with_nested(config_path)
        config = config_hierarchy['primary']
        
        if dry_run:
            typer.echo(f"✓ Config validated: {config_path}")
            typer.echo(f"  Model: {config.model.name}")
            typer.echo(f"  Data: {config.data.name}")
            return
        
        # Import here to avoid circular imports
        from runs.lightning import run
        
        typer.echo(f"Starting Lightning training with config: {config_path}")
        run(config)
        
    except FileNotFoundError as e:
        if RICH_AVAILABLE:
            console.print(f"[bold red]Error:[/bold red] Config file not found: {e}")
        else:
            typer.echo(f"Error: Config file not found: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        handle_error(e, "Error during training")
        raise typer.Exit(code=1)


@app.command()
def llm(
    config_path: str = typer.Argument(..., help="Path to experiment configuration file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without running")
):
    """
    Run LLM-based time series forecasting experiments.
    
    This command runs Large Language Model based forecasting experiments.
    Typically used for inference/testing rather than training.
    
    Example:
        python -m cli.train llm configs/experiments/llm_solar.yaml
    """
    try:
        config_hierarchy = load_config_with_nested(config_path)
        config = config_hierarchy['primary']
        
        if dry_run:
            typer.echo(f"✓ Config validated: {config_path}")
            typer.echo(f"  Model: {config.model.name}")
            typer.echo(f"  Data: {config.data.name}")
            return
        
        # Import here to avoid circular imports
        from runs.llm import run
        
        typer.echo(f"Starting LLM experiment with config: {config_path}")
        run(config)
        
    except FileNotFoundError as e:
        if RICH_AVAILABLE:
            console.print(f"[bold red]Error:[/bold red] Config file not found: {e}")
        else:
            typer.echo(f"Error: Config file not found: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        handle_error(e, "Error during LLM experiment")
        raise typer.Exit(code=1)


@app.command()
def fm(
    config_path: str = typer.Argument(..., help="Path to experiment configuration file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without running")
):
    """
    Test Foundation Models (FM) for time series forecasting.
    
    This command runs Foundation Model testing experiments.
    Foundation models are pre-trained models that can be tested directly.
    
    Example:
        python -m cli.train fm configs/experiments/fm_solar.yaml
    """
    try:
        config_hierarchy = load_config_with_nested(config_path)
        config = config_hierarchy['primary']
        
        if dry_run:
            typer.echo(f"✓ Config validated: {config_path}")
            typer.echo(f"  Model: {config.model.name}")
            typer.echo(f"  Data: {config.data.name}")
            return
        
        # Import here to avoid circular imports
        from runs.fm import run
        
        typer.echo(f"Starting Foundation Model testing with config: {config_path}")
        run(config)
        
    except FileNotFoundError as e:
        if RICH_AVAILABLE:
            console.print(f"[bold red]Error:[/bold red] Config file not found: {e}")
        else:
            typer.echo(f"Error: Config file not found: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        handle_error(e, "Error during FM testing")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

