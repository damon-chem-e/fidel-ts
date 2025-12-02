"""
Visualization command-line interface for Fidel-TS.

This module provides CLI commands for visualizing model predictions:
- TSF/TGTSF visualization (standard time series forecasting)
- LLM visualization (LLM-based forecasting results)
- Lightning visualization (Lightning-trained model predictions)
"""

import typer
from cli.config.loader import load_config_with_nested

app = typer.Typer(
    name="visualize",
    help="Visualize time series forecasting model predictions",
    add_completion=False
)


@app.command()
def tsf(
    config_path: str = typer.Argument(..., help="Path to visualization configuration file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without generating visualization")
):
    """
    Visualize TSF/TGTSF model predictions.
    
    This command creates prediction plots comparing input history,
    ground truth, and model predictions for standard time series models.
    
    Example:
        python -m cli.visualize tsf configs/experiments/dlinear_solar.yaml
    """
    try:
        config_hierarchy = load_config_with_nested(config_path)
        config = config_hierarchy['primary']
        
        # Load plotting subconfig if available
        plotting_config = config_hierarchy['nested'].get('plotting', None)
        
        if dry_run:
            typer.echo(f"✓ Config validated: {config_path}")
            typer.echo(f"  Model: {config.model.name}")
            typer.echo(f"  Checkpoint: {config.visualization.ckpt_id}")
            if plotting_config:
                typer.echo(f"  Plotting config: loaded")
            return
        
        # Import here to avoid circular imports
        from visualization.tsf import visualize
        
        typer.echo(f"Generating TSF visualization with config: {config_path}")
        visualize(config, plotting_config)
        
    except FileNotFoundError as e:
        typer.echo(f"Error: Config file not found: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error during visualization: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def llm(
    config_path: str = typer.Argument(..., help="Path to visualization configuration file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without generating visualization")
):
    """
    Visualize LLM-based forecasting results.
    
    This command visualizes predictions from LLM experiments by reading
    JSON result files generated during LLM inference.
    
    Example:
        python -m cli.visualize llm configs/experiments/llm_solar.yaml
    """
    try:
        config_hierarchy = load_config_with_nested(config_path)
        config = config_hierarchy['primary']
        
        # Load plotting subconfig if available
        plotting_config = config_hierarchy['nested'].get('plotting', None)
        
        if dry_run:
            typer.echo(f"✓ Config validated: {config_path}")
            typer.echo(f"  Checkpoint: {config.visualization.ckpt_id}")
            if plotting_config:
                typer.echo(f"  Plotting config: loaded")
            return
        
        # Import here to avoid circular imports
        from visualization.llm import visualize
        
        typer.echo(f"Generating LLM visualization with config: {config_path}")
        visualize(config, plotting_config)
        
    except FileNotFoundError as e:
        typer.echo(f"Error: Config file not found: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error during visualization: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def lightning(
    config_path: str = typer.Argument(..., help="Path to visualization configuration file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without generating visualization")
):
    """
    Visualize PyTorch Lightning-trained model predictions.
    
    This command visualizes predictions from models trained with Lightning.
    Supports flexible checkpoint selection (best/last).
    
    Example:
        python -m cli.visualize lightning configs/experiments/dlinear_solar.yaml
    """
    try:
        config_hierarchy = load_config_with_nested(config_path)
        config = config_hierarchy['primary']
        
        # Load plotting subconfig if available
        plotting_config = config_hierarchy['nested'].get('plotting', None)
        
        if dry_run:
            typer.echo(f"✓ Config validated: {config_path}")
            typer.echo(f"  Model: {config.model.name}")
            typer.echo(f"  Checkpoint: {config.visualization.ckpt_id}")
            if plotting_config:
                typer.echo(f"  Plotting config: loaded")
            return
        
        # Import here to avoid circular imports
        from visualization.lightning import visualize
        
        typer.echo(f"Generating Lightning visualization with config: {config_path}")
        visualize(config, plotting_config)
        
    except FileNotFoundError as e:
        typer.echo(f"Error: Config file not found: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error during visualization: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

