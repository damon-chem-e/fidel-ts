"""
Filtering command-line interface for Fidel-TS.

This module provides CLI commands for filtering reasoning samples:
- Reasoning sample filtering (compare TST vs TGTSF performance)
- CSV-based filtering (use pre-computed loss dataframes)
"""

import typer
from cli.config.loader import load_config_with_nested

app = typer.Typer(
    name="filter",
    help="Filter reasoning samples for LLM training",
    add_completion=False
)


@app.command()
def reasoning(
    config_path: str = typer.Argument(..., help="Path to filtering configuration file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without running filtering")
):
    """
    Filter reasoning samples by comparing TST vs TGTSF performance.
    
    This command identifies samples where TGTSF significantly outperforms
    or underperforms compared to TST, generating filtered sample indexes
    for use in LLM reasoning training.
    
    Example:
        python -m cli.filter reasoning configs/experiments/filter_reasoning.yaml
    """
    try:
        config_hierarchy = load_config_with_nested(config_path)
        config = config_hierarchy['primary']
        
        if dry_run:
            typer.echo(f"✓ Config validated: {config_path}")
            typer.echo(f"  Data: {config.data.name}")
            typer.echo(f"  Sampling rate: {config.filtering.sampling_rate}")
            return
        
        # Import here to avoid circular imports
        from filtering.reasoning import filter_samples
        
        typer.echo(f"Starting reasoning sample filtering with config: {config_path}")
        filter_samples(config)
        
    except FileNotFoundError as e:
        typer.echo(f"Error: Config file not found: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error during filtering: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def from_csv(
    config_path: str = typer.Argument(..., help="Path to filtering configuration file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without running filtering")
):
    """
    Filter reasoning samples from pre-computed loss CSV files.
    
    This command filters samples using pre-computed loss dataframes stored
    as CSV files, avoiding the need to run model inference again.
    More efficient for re-filtering with different sampling rates.
    
    Example:
        python -m cli.filter from_csv configs/experiments/filter_reasoning.yaml
    """
    try:
        config_hierarchy = load_config_with_nested(config_path)
        config = config_hierarchy['primary']
        
        if dry_run:
            typer.echo(f"✓ Config validated: {config_path}")
            typer.echo(f"  Data: {config.data.name}")
            typer.echo(f"  Sampling rate: {config.filtering.sampling_rate}")
            typer.echo(f"  Using pre-computed CSV files")
            return
        
        # Import here to avoid circular imports
        from filtering.reasoning import filter_from_csv
        
        typer.echo(f"Starting CSV-based filtering with config: {config_path}")
        filter_from_csv(config)
        
    except FileNotFoundError as e:
        typer.echo(f"Error: Config file not found: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error during filtering: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

