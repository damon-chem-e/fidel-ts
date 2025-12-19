#!/usr/bin/env python3
"""
W&B Cleanup Utility

A Typer CLI tool for cleaning up Weights & Biases runs.
Provides safe deletion operations with dry-run mode by default.

Usage:
    # List all runs in a project
    python scripts/wandb_cleanup.py list-runs -e <entity> -p <project>
    
    # Delete all runs except specific IDs (dry run by default)
    python scripts/wandb_cleanup.py keep-ids -e <entity> -p <project> -k <id1> -k <id2>
    
    # Delete runs older than 30 days
    python scripts/wandb_cleanup.py by-age -e <entity> -p <project> -d 30 --execute
    
    # Delete runs with specific tags
    python scripts/wandb_cleanup.py by-tags -e <entity> -p <project> -t <tag1> -t <tag2> --execute
    
    # Keep runs with specific tags (delete all others)
    python scripts/wandb_cleanup.py by-tags -e <entity> -p <project> -t <tag1> --keep-tagged --execute
"""

import typer
from typing import List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json

try:
    import wandb
    from wandb.apis.public import Api
except ImportError:
    print("Error: wandb is not installed. Please install it with: pip install wandb")
    raise

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Initialize Typer app and Rich console
app = typer.Typer(
    name="wandb-cleanup",
    help="Clean up Weights & Biases runs with various filtering options",
    add_completion=False,
)
console = Console()


def get_api() -> Api:
    """
    Initialize and return W&B API client.
    
    Returns:
        Api: Initialized W&B API client
        
    Raises:
        typer.Exit: If API initialization fails
    """
    try:
        api = wandb.Api()
        return api
    except Exception as e:
        console.print(f"[red]Error initializing W&B API: {e}[/red]")
        console.print("[yellow]Make sure you're logged in with: wandb login[/yellow]")
        raise typer.Exit(1)


def get_runs(api: Api, entity: str, project: str) -> List:
    """
    Fetch all runs from a W&B project.
    
    Args:
        api: W&B API client
        entity: W&B entity (username or team name)
        project: W&B project name
        
    Returns:
        List of run objects
        
    Raises:
        typer.Exit: If fetching runs fails
    """
    project_path = f"{entity}/{project}"
    try:
        console.print(f"[cyan]Fetching runs from: {project_path}[/cyan]")
        runs = list(api.runs(project_path))
        return runs
    except Exception as e:
        console.print(f"[red]Error fetching runs: {e}[/red]")
        console.print(f"[yellow]Check that entity '{entity}' and project '{project}' exist and you have access[/yellow]")
        raise typer.Exit(1)


def filter_runs_by_ids(runs: List, keep_ids: List[str]) -> tuple[List, List]:
    """
    Filter runs into those to keep and those to delete based on ID list.
    
    Args:
        runs: List of all runs
        keep_ids: List of run IDs to keep
        
    Returns:
        Tuple of (runs_to_keep, runs_to_delete)
    """
    keep_ids_set = set(keep_ids)
    runs_to_keep = [r for r in runs if r.id in keep_ids_set]
    runs_to_delete = [r for r in runs if r.id not in keep_ids_set]
    return runs_to_keep, runs_to_delete


def filter_runs_by_age(runs: List, days: int) -> tuple[List, List]:
    """
    Filter runs by age - delete runs older than specified days.
    
    Args:
        runs: List of all runs
        days: Number of days - runs older than this will be deleted
        
    Returns:
        Tuple of (runs_to_keep, runs_to_delete)
    """
    cutoff_date = datetime.now() - timedelta(days=days)
    runs_to_keep = []
    runs_to_delete = []
    
    for run in runs:
        # Parse run creation time
        run_date = datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
        if run_date > cutoff_date:
            runs_to_keep.append(run)
        else:
            runs_to_delete.append(run)
    
    return runs_to_keep, runs_to_delete


def filter_runs_by_tags(runs: List, tags: List[str], keep_tagged: bool = False) -> tuple[List, List]:
    """
    Filter runs by tags - delete runs with/without specified tags.
    
    Args:
        runs: List of all runs
        tags: List of tags to filter by
        keep_tagged: If True, keep runs with tags; if False, delete runs with tags
        
    Returns:
        Tuple of (runs_to_keep, runs_to_delete)
    """
    runs_to_keep = []
    runs_to_delete = []
    tags_set = set(tags)
    
    for run in runs:
        run_tags = set(run.tags) if run.tags else set()
        has_tag = bool(tags_set & run_tags)
        
        if (keep_tagged and has_tag) or (not keep_tagged and not has_tag):
            runs_to_keep.append(run)
        else:
            runs_to_delete.append(run)
    
    return runs_to_keep, runs_to_delete


def display_runs_table(runs: List, title: str):
    """
    Display runs in a formatted Rich table.
    
    Args:
        runs: List of runs to display
        title: Title for the table
    """
    if not runs:
        console.print(f"[dim]No runs to display for: {title}[/dim]")
        return
    
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("State", style="yellow")
    table.add_column("Created", style="blue")
    table.add_column("Tags", style="dim")
    
    for run in runs[:50]:  # Limit to 50 for display
        created = datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
        tags_str = ", ".join(run.tags[:3]) if run.tags else "None"
        if run.tags and len(run.tags) > 3:
            tags_str += "..."
        
        table.add_row(
            run.id,
            run.name[:50] if run.name else "N/A",
            run.state,
            created.strftime("%Y-%m-%d %H:%M"),
            tags_str
        )
    
    if len(runs) > 50:
        table.add_row("...", f"... and {len(runs) - 50} more runs", "", "", "")
    
    console.print(table)


def delete_runs(runs: List, dry_run: bool = True) -> int:
    """
    Delete a list of runs with progress tracking.
    
    Args:
        runs: List of runs to delete
        dry_run: If True, don't actually delete (just simulate)
        
    Returns:
        Number of successfully deleted runs
    """
    if not runs:
        return 0
    
    deleted_count = 0
    failed_count = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(
            f"{'[DRY RUN] ' if dry_run else ''}Deleting runs...",
            total=len(runs)
        )
        
        for run in runs:
            if dry_run:
                deleted_count += 1
                progress.update(task, advance=1, description=f"[DRY RUN] Would delete: {run.id}")
            else:
                try:
                    run.delete()
                    deleted_count += 1
                    progress.update(task, advance=1, description=f"Deleted: {run.id}")
                except Exception as e:
                    failed_count += 1
                    console.print(f"[red]Failed to delete {run.id}: {e}[/red]")
                    progress.update(task, advance=1)
    
    return deleted_count


@app.command()
def keep_ids(
    entity: str = typer.Option(..., "--entity", "-e", help="W&B entity (username or team)"),
    project: str = typer.Option(..., "--project", "-p", help="W&B project name"),
    keep: List[str] = typer.Option(..., "--keep", "-k", help="Run IDs to keep (can specify multiple)"),
    dry_run: bool = typer.Option(True, "--execute/--dry-run", help="Actually delete (default: dry run)"),
    show_runs: bool = typer.Option(True, "--show-runs/--no-show-runs", help="Display runs table"),
    ids_file: Optional[Path] = typer.Option(None, "--ids-file", "-f", help="JSON file with list of IDs to keep"),
):
    """
    Delete all runs EXCEPT the specified run IDs.
    
    This is useful when you want to keep only specific runs and delete everything else.
    """
    # Load IDs from file if provided
    if ids_file:
        if not ids_file.exists():
            console.print(f"[red]Error: File not found: {ids_file}[/red]")
            raise typer.Exit(1)
        try:
            with open(ids_file, 'r') as f:
                file_ids = json.load(f)
                if isinstance(file_ids, list):
                    keep.extend(file_ids)
                else:
                    console.print("[red]Error: JSON file must contain a list of IDs[/red]")
                    raise typer.Exit(1)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing JSON file: {e}[/red]")
            raise typer.Exit(1)
    
    if not keep:
        console.print("[red]Error: No run IDs specified to keep[/red]")
        raise typer.Exit(1)
    
    # Initialize API and fetch runs
    api = get_api()
    all_runs = get_runs(api, entity, project)
    
    # Filter runs
    runs_to_keep, runs_to_delete = filter_runs_by_ids(all_runs, keep)
    
    # Display summary
    console.print(Panel.fit(
        f"[bold]Summary[/bold]\n"
        f"Total runs: {len(all_runs)}\n"
        f"Runs to keep: [green]{len(runs_to_keep)}[/green]\n"
        f"Runs to delete: [red]{len(runs_to_delete)}[/red]",
        title="Run Filtering Results"
    ))
    
    # Display runs if requested
    if show_runs:
        if runs_to_delete:
            display_runs_table(runs_to_delete, "Runs to be Deleted")
        if runs_to_keep:
            display_runs_table(runs_to_keep, "Runs to be Kept")
    
    # Confirm deletion
    if runs_to_delete:
        if not dry_run:
            if not typer.confirm(f"\n[red]Are you sure you want to delete {len(runs_to_delete)} runs?[/red]"):
                console.print("[yellow]Deletion cancelled[/yellow]")
                raise typer.Exit(0)
        
        deleted = delete_runs(runs_to_delete, dry_run=dry_run)
        
        if dry_run:
            console.print(f"\n[green]Dry run complete. {deleted} runs would be deleted.[/green]")
            console.print("[yellow]Run with --execute to actually delete[/yellow]")
        else:
            console.print(f"\n[green]Successfully deleted {deleted} runs[/green]")
    else:
        console.print("[green]No runs to delete[/green]")


@app.command()
def by_age(
    entity: str = typer.Option(..., "--entity", "-e", help="W&B entity (username or team)"),
    project: str = typer.Option(..., "--project", "-p", help="W&B project name"),
    days: int = typer.Option(..., "--days", "-d", help="Delete runs older than this many days"),
    dry_run: bool = typer.Option(True, "--execute/--dry-run", help="Actually delete (default: dry run)"),
    show_runs: bool = typer.Option(True, "--show-runs/--no-show-runs", help="Display runs table"),
):
    """
    Delete runs older than a specified number of days.
    
    Useful for cleaning up old experiments automatically.
    """
    if days < 1:
        console.print("[red]Error: Days must be >= 1[/red]")
        raise typer.Exit(1)
    
    # Initialize API and fetch runs
    api = get_api()
    all_runs = get_runs(api, entity, project)
    
    # Filter runs by age
    runs_to_keep, runs_to_delete = filter_runs_by_age(all_runs, days)
    
    # Display summary
    cutoff_date = datetime.now() - timedelta(days=days)
    console.print(Panel.fit(
        f"[bold]Summary[/bold]\n"
        f"Cutoff date: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Total runs: {len(all_runs)}\n"
        f"Runs to keep: [green]{len(runs_to_keep)}[/green]\n"
        f"Runs to delete: [red]{len(runs_to_delete)}[/red]",
        title="Age-based Filtering Results"
    ))
    
    # Display runs if requested
    if show_runs and runs_to_delete:
        display_runs_table(runs_to_delete, f"Runs Older Than {days} Days")
    
    # Confirm deletion
    if runs_to_delete:
        if not dry_run:
            if not typer.confirm(f"\n[red]Are you sure you want to delete {len(runs_to_delete)} runs?[/red]"):
                console.print("[yellow]Deletion cancelled[/yellow]")
                raise typer.Exit(0)
        
        deleted = delete_runs(runs_to_delete, dry_run=dry_run)
        
        if dry_run:
            console.print(f"\n[green]Dry run complete. {deleted} runs would be deleted.[/green]")
            console.print("[yellow]Run with --execute to actually delete[/yellow]")
        else:
            console.print(f"\n[green]Successfully deleted {deleted} runs[/green]")
    else:
        console.print("[green]No runs to delete[/green]")


@app.command()
def by_tags(
    entity: str = typer.Option(..., "--entity", "-e", help="W&B entity (username or team)"),
    project: str = typer.Option(..., "--project", "-p", help="W&B project name"),
    tags: List[str] = typer.Option(..., "--tag", "-t", help="Tags to filter by (can specify multiple)"),
    keep_tagged: bool = typer.Option(False, "--keep-tagged/--delete-tagged", help="Keep runs with tags (default: delete tagged runs)"),
    dry_run: bool = typer.Option(True, "--execute/--dry-run", help="Actually delete (default: dry run)"),
    show_runs: bool = typer.Option(True, "--show-runs/--no-show-runs", help="Display runs table"),
):
    """
    Delete runs based on tags.
    
    By default, deletes runs that HAVE the specified tags.
    Use --keep-tagged to delete runs that DON'T have the tags.
    """
    if not tags:
        console.print("[red]Error: At least one tag must be specified[/red]")
        raise typer.Exit(1)
    
    # Initialize API and fetch runs
    api = get_api()
    all_runs = get_runs(api, entity, project)
    
    # Filter runs by tags
    runs_to_keep, runs_to_delete = filter_runs_by_tags(all_runs, tags, keep_tagged)
    
    # Display summary
    action = "keep" if keep_tagged else "delete"
    console.print(Panel.fit(
        f"[bold]Summary[/bold]\n"
        f"Tags: {', '.join(tags)}\n"
        f"Action: {action} runs with these tags\n"
        f"Total runs: {len(all_runs)}\n"
        f"Runs to keep: [green]{len(runs_to_keep)}[/green]\n"
        f"Runs to delete: [red]{len(runs_to_delete)}[/red]",
        title="Tag-based Filtering Results"
    ))
    
    # Display runs if requested
    if show_runs and runs_to_delete:
        display_runs_table(runs_to_delete, f"Runs to be Deleted (tags: {', '.join(tags)})")
    
    # Confirm deletion
    if runs_to_delete:
        if not dry_run:
            if not typer.confirm(f"\n[red]Are you sure you want to delete {len(runs_to_delete)} runs?[/red]"):
                console.print("[yellow]Deletion cancelled[/yellow]")
                raise typer.Exit(0)
        
        deleted = delete_runs(runs_to_delete, dry_run=dry_run)
        
        if dry_run:
            console.print(f"\n[green]Dry run complete. {deleted} runs would be deleted.[/green]")
            console.print("[yellow]Run with --execute to actually delete[/yellow]")
        else:
            console.print(f"\n[green]Successfully deleted {deleted} runs[/green]")
    else:
        console.print("[green]No runs to delete[/green]")


@app.command()
def list_runs(
    entity: str = typer.Option(..., "--entity", "-e", help="W&B entity (username or team)"),
    project: str = typer.Option(..., "--project", "-p", help="W&B project name"),
    limit: int = typer.Option(50, "--limit", "-l", help="Maximum number of runs to display"),
):
    """
    List all runs in a project with their details.
    
    Useful for exploring what runs exist before deciding what to delete.
    """
    # Initialize API and fetch runs
    api = get_api()
    all_runs = get_runs(api, entity, project)
    
    console.print(f"\n[bold cyan]Found {len(all_runs)} runs in {entity}/{project}[/bold cyan]\n")
    
    # Display runs
    display_runs_table(all_runs[:limit], f"Runs (showing {min(limit, len(all_runs))} of {len(all_runs)})")
    
    if len(all_runs) > limit:
        console.print(f"\n[yellow]Showing first {limit} runs. Use --limit to see more[/yellow]")


if __name__ == "__main__":
    app()
