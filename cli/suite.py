"""
CLI module for experiment suite management.

This module provides commands for running, listing, and validating experiment suites.
"""

import typer
import builtins
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from runs.suite_executor import load_suite_config, execute_suite, SuiteExecutor

app = typer.Typer(help="Manage and execute experiment suites")
console = Console()


@app.command()
def run(
    suite_config_path: str = typer.Argument(..., help="Path to experiment suite configuration file"),
    filter_experiments: Optional[str] = typer.Option(None, "--filter", "-f", help="Filter experiments by name pattern"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without running experiments"),
    init_only: bool = typer.Option(False, "--init-only", help="Initialize experiment structures without running them")
):
    """
    Run an experiment suite from a suite config file.
    
    Example:
        python -m cli.suite run configs/experiment_suites/linear_models.yaml
    """
    suite_config_path = Path(suite_config_path)
    
    if not suite_config_path.exists():
        console.print(f"[red]Error: Suite config file not found: {suite_config_path}[/red]")
        raise typer.Exit(code=1)
    
    try:
        suite_config = load_suite_config(str(suite_config_path))
        suite_info = suite_config.get('suite', {})
        suite_name = suite_info.get('name', 'unknown')
        
        # Filter experiments if requested
        if filter_experiments:
            original_experiments = suite_info.get('experiments', [])
            filtered_experiments = [
                exp for exp in original_experiments
                if filter_experiments.lower() in exp.get('name', '').lower()
            ]
            suite_info['experiments'] = filtered_experiments
            console.print(f"[yellow]Filtered to {len(filtered_experiments)} experiments matching '{filter_experiments}'[/yellow]")
        
        if dry_run:
            console.print(f"[green]Dry run: Validated suite '{suite_name}' with {len(suite_info.get('experiments', []))} experiments[/green]")
            return
        
        if init_only:
            console.print(f"[green]Initializing suite structure (no execution): {suite_name}[/green]")
        else:
            console.print(f"[green]Running suite: {suite_name}[/green]")
            
        executor = SuiteExecutor(suite_config, init_only=init_only)
        executor.execute()
        
        if init_only:
            console.print(f"[green]Suite initialization completed[/green]")
            console.print(f"Use the following resume_suite_id for your job: [bold cyan]{executor.suite_name}[/bold cyan]")
        else:
            console.print(f"[green]Suite execution completed[/green]")
        
    except Exception as e:
        console.print(f"[red]Error executing suite: {str(e)}[/red]")
        raise typer.Exit(code=1)


@app.command()
def list(
    suites_dir: str = typer.Option("configs/experiment_suites", "--dir", "-d", help="Directory containing suite configs")
):
    """
    List all available experiment suites.
    
    Example:
        python -m cli.suite list
    """
    # Ensure suites_dir is a string or Path-like object
    # Convert to string if it's already a Path object, or handle other path-like objects
    if isinstance(suites_dir, Path):
        suites_path = suites_dir
    elif isinstance(suites_dir, str):
        suites_path = Path(suites_dir)
    else:
        console.print(f"[red]Error: Invalid suites_dir type: {type(suites_dir).__name__}. Expected a string path or Path object.[/red]")
        raise typer.Exit(code=1)
    
    if not suites_path.exists():
        console.print(f"[red]Error: Suites directory not found: {suites_path}[/red]")
        raise typer.Exit(code=1)
    
    # Find all YAML files in suites directory
    # Use builtins.list() to avoid shadowing the function name 'list'
    suite_files = builtins.list(suites_path.glob("*.yaml")) + builtins.list(suites_path.glob("*.yml"))
    
    if not suite_files:
        console.print(f"[yellow]No suite configs found in {suites_path}[/yellow]")
        return
    
    # Create table
    table = Table(title="Available Experiment Suites")
    table.add_column("Suite Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Experiments", justify="right", style="green")
    table.add_column("File", style="dim")
    
    for suite_file in sorted(suite_files):
        try:
            suite_config = load_suite_config(str(suite_file))
            suite_info = suite_config.get('suite', {})
            suite_name = suite_info.get('name', suite_file.stem)
            description = suite_info.get('description', 'No description')
            num_experiments = len(suite_info.get('experiments', []))
            
            table.add_row(
                suite_name,
                description[:60] + "..." if len(description) > 60 else description,
                str(num_experiments),
                suite_file.name
            )
        except Exception as e:
            table.add_row(
                suite_file.stem,
                f"[red]Error loading: {str(e)}[/red]",
                "-",
                suite_file.name
            )
    
    console.print(table)


@app.command()
def validate(
    suite_config_path: str = typer.Argument(..., help="Path to experiment suite configuration file")
):
    """
    Validate a suite config file.
    
    Example:
        python -m cli.suite validate configs/experiment_suites/linear_models.yaml
    """
    suite_config_path = Path(suite_config_path)
    
    if not suite_config_path.exists():
        console.print(f"[red]Error: Suite config file not found: {suite_config_path}[/red]")
        raise typer.Exit(code=1)
    
    try:
        suite_config = load_suite_config(str(suite_config_path))
        suite_info = suite_config.get('suite', {})
        
        # Validate structure
        errors = []
        warnings = []
        
        # Check required fields
        if 'name' not in suite_info:
            errors.append("Missing required field: suite.name")
        if 'experiments' not in suite_info:
            errors.append("Missing required field: suite.experiments")
        
        # Validate experiments
        experiments = suite_info.get('experiments', [])
        if not isinstance(experiments, list):
            errors.append("suite.experiments must be a list")
        else:
            for i, exp in enumerate(experiments):
                if not isinstance(exp, dict):
                    errors.append(f"Experiment {i} must be a dictionary")
                    continue
                
                if 'name' not in exp:
                    errors.append(f"Experiment {i} missing required field: name")
                if 'template' not in exp:
                    errors.append(f"Experiment {i} missing required field: template")
                elif not Path(exp['template']).exists():
                    warnings.append(f"Experiment {i} template file not found: {exp['template']}")
        
        # Print results
        if errors:
            console.print("[red]Validation Errors:[/red]")
            for error in errors:
                console.print(f"  [red]✗[/red] {error}")
        
        if warnings:
            console.print("[yellow]Validation Warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  [yellow]⚠[/yellow] {warning}")
        
        if not errors and not warnings:
            console.print(f"[green]✓ Suite config is valid[/green]")
            console.print(f"  Suite: {suite_info.get('name', 'unknown')}")
            console.print(f"  Experiments: {len(experiments)}")
            console.print(f"  Enabled: {sum(1 for exp in experiments if exp.get('enabled', True))}")
            return
        
        if errors:
            raise typer.Exit(code=1)
        
    except Exception as e:
        console.print(f"[red]Error validating suite config: {str(e)}[/red]")
        raise typer.Exit(code=1)


@app.command()
def info(
    suite_config_path: str = typer.Argument(..., help="Path to experiment suite configuration file")
):
    """
    Show detailed information about a suite.
    
    Example:
        python -m cli.suite info configs/experiment_suites/linear_models.yaml
    """
    suite_config_path = Path(suite_config_path)
    
    if not suite_config_path.exists():
        console.print(f"[red]Error: Suite config file not found: {suite_config_path}[/red]")
        raise typer.Exit(code=1)
    
    try:
        suite_config = load_suite_config(str(suite_config_path))
        suite_info = suite_config.get('suite', {})
        
        console.print(f"\n[bold cyan]Suite Information[/bold cyan]")
        console.print(f"  Name: {suite_info.get('name', 'unknown')}")
        console.print(f"  Description: {suite_info.get('description', 'No description')}")
        
        execution = suite_info.get('execution', {})
        console.print(f"\n[bold cyan]Execution Settings[/bold cyan]")
        console.print(f"  Parallel: {execution.get('parallel', False)}")
        console.print(f"  Continue on error: {execution.get('continue_on_error', True)}")
        console.print(f"  Log directory: {execution.get('log_dir', './logs/suites')}")
        
        experiments = suite_info.get('experiments', [])
        console.print(f"\n[bold cyan]Experiments ({len(experiments)})[/bold cyan]")
        
        for i, exp in enumerate(experiments, 1):
            enabled = exp.get('enabled', True)
            status = "[green]✓[/green]" if enabled else "[red]✗[/red]"
            console.print(f"  {status} {i}. {exp.get('name', 'unknown')}")
            if exp.get('description'):
                console.print(f"     {exp.get('description')}")
        
    except Exception as e:
        console.print(f"[red]Error loading suite info: {str(e)}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

