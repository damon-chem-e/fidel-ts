"""
Testing/Evaluation command-line interface for Fidel-TS.

This module provides CLI commands for evaluating trained models:
- Standard evaluation (for PyTorch-trained models)
- Lightning evaluation (for Lightning-trained models)
- LLM evaluation (for LLM-based forecasting results)
"""

import typer
from cli.config.loader import load_config_with_nested

app = typer.Typer(
    name="test",
    help="Evaluate trained time series forecasting models",
    add_completion=False
)


@app.command()
def standard(
    config_path: str = typer.Argument(..., help="Path to evaluation configuration file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without running evaluation")
):
    """
    Evaluate standard PyTorch-trained models.
    
    This command evaluates models trained with the standard PyTorch pipeline.
    Supports channel-wise evaluation and filtered sample testing.
    
    Example:
        python -m cli.test standard configs/experiments/dlinear_solar.yaml
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
        from evaluation.standard import evaluate
        
        typer.echo(f"Starting standard evaluation with config: {config_path}")
        evaluate(config)
        
    except FileNotFoundError as e:
        typer.echo(f"Error: Config file not found: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error during evaluation: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def lightning(
    config_path: str = typer.Argument(..., help="Path to evaluation configuration file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without running evaluation")
):
    """
    Evaluate PyTorch Lightning-trained models.
    
    This command evaluates models trained with PyTorch Lightning.
    Handles Lightning checkpoint format and state dict loading.
    
    Example:
        python -m cli.test lightning configs/experiments/dlinear_solar.yaml
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
        from evaluation.lightning import evaluate
        
        typer.echo(f"Starting Lightning evaluation with config: {config_path}")
        evaluate(config)
        
    except FileNotFoundError as e:
        typer.echo(f"Error: Config file not found: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error during evaluation: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def llm(
    config_path: str = typer.Argument(..., help="Path to evaluation configuration file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without running evaluation")
):
    """
    Evaluate LLM-based forecasting results.
    
    This command evaluates predictions from LLM-based forecasting experiments.
    Reads JSON prediction files and calculates standardized metrics.
    
    Example:
        python -m cli.test llm configs/experiments/llm_solar.yaml
    """
    try:
        config_hierarchy = load_config_with_nested(config_path)
        config = config_hierarchy['primary']
        
        if dry_run:
            typer.echo(f"✓ Config validated: {config_path}")
            typer.echo(f"  Checkpoint: {config.evaluation.ckpt_id}")
            return
        
        # Import here to avoid circular imports
        from evaluation.llm import evaluate
        
        typer.echo(f"Starting LLM evaluation with config: {config_path}")
        evaluate(config)
        
    except FileNotFoundError as e:
        typer.echo(f"Error: Config file not found: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error during evaluation: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

