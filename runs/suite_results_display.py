"""
Display suite results using rich tables.

This module provides functionality to render experiment results in
rich-formatted tables with color-coded status indicators.
"""

from pathlib import Path
from typing import List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from runs.suite_results_collector import ExperimentResult


# Status abbreviation mapping
STATUS_ABBREV = {
    'completed': 'COMP',
    'failed': 'FAIL',
    'skipped': 'SKIP',
    'not_run': 'NORUN',
    'partial': 'PART'
}

STATUS_LEGEND = {
    'COMP': 'Completed (all epochs or early stopping)',
    'FAIL': 'Failed (error during execution)',
    'SKIP': 'Skipped (already complete)',
    'NORUN': 'Not run (suite stopped before this experiment)',
    'PART': 'Partial (some epochs completed)'
}

# Completion reason abbreviation mapping
REASON_ABBREV = {
    'all_epochs': 'ALL',
    'early_stopping': 'EARLY',
    'timeout': 'TIME',
    'error': 'ERR',
    None: 'UNK'
}

REASON_LEGEND = {
    'ALL': 'All epochs completed',
    'EARLY': 'Early stopping triggered',
    'TIME': 'Timeout (SLURM or manual)',
    'ERR': 'Error during training',
    'UNK': 'Unknown (reason not recorded)'
}


class SuiteResultsDisplay:
    """Display suite results as rich-formatted tables."""

    def __init__(self, console: Console = None):
        """
        Initialize display manager.

        Args:
            console: Rich Console instance (creates new one if not provided)
        """
        self.console = console or Console()

    def display_all_tables(
        self,
        results: List[ExperimentResult],
        suite_name: str,
        suite_id: str
    ):
        """
        Display both metrics and status tables.

        Args:
            results: List of ExperimentResult objects
            suite_name: Base suite name (without timestamp)
            suite_id: Full suite ID (with timestamp)
        """
        self.console.print("\n")
        self.console.print(Panel.fit(
            f"[bold cyan]Suite Results Summary[/bold cyan]\n"
            f"Suite: [yellow]{suite_name}[/yellow]",
            border_style="cyan"
        ))
        self.console.print("\n")

        # Display metrics table
        self._display_metrics_table(results)
        self.console.print(f"\n[dim]Suite ID: {suite_id}[/dim]\n")

        # Display status table
        self._display_status_table(results)

        # Display legends
        self._display_legends()

    def _display_metrics_table(self, results: List[ExperimentResult]):
        """Display table with train/val/test MSE metrics."""
        table = Table(
            title="Experiment Metrics (Normalized MSE)",
            show_header=True,
            header_style="bold magenta",
            title_style="bold white"
        )

        table.add_column("Experiment Name", style="cyan", no_wrap=False)
        table.add_column("Experiment ID", style="dim", no_wrap=True)
        table.add_column("Train MSE", justify="right", style="green")
        table.add_column("Val MSE", justify="right", style="yellow")
        table.add_column("Test MSE", justify="right", style="blue")

        for result in results:
            # Only include experiments that have at least some metrics
            # (skip experiments that were never run)
            if result.status == "not_run":
                continue

            # Format experiment ID (show last 12 chars for readability)
            exp_id_short = result.experiment_id[-12:] if len(result.experiment_id) > 12 else result.experiment_id

            # Format metrics (N/A if not available)
            train_mse_str = f"{result.train_mse:.7f}" if result.train_mse is not None else "N/A"
            val_mse_str = f"{result.val_mse:.7f}" if result.val_mse is not None else "N/A"
            test_mse_str = f"{result.test_mse:.7f}" if result.test_mse is not None else "N/A"

            table.add_row(
                result.name,
                exp_id_short,
                train_mse_str,
                val_mse_str,
                test_mse_str
            )

        self.console.print(table)

    def _display_status_table(self, results: List[ExperimentResult]):
        """Display table with experiment status and completion info."""
        table = Table(
            title="Experiment Status",
            show_header=True,
            header_style="bold magenta",
            title_style="bold white"
        )

        table.add_column("Experiment Name", style="cyan", no_wrap=False)
        table.add_column("Exp ID", style="dim", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Epochs", justify="center", style="yellow")
        table.add_column("Reason", justify="center", style="blue")

        for result in results:
            # Format experiment ID (last 8 chars)
            exp_id_short = result.experiment_id[-8:] if len(result.experiment_id) > 8 else result.experiment_id

            # Get status abbreviation
            status_abbrev = STATUS_ABBREV.get(result.status, result.status.upper()[:5])

            # Color status based on completion
            if result.status == 'completed':
                status_str = f"[green]{status_abbrev}[/green]"
            elif result.status == 'failed':
                status_str = f"[red]{status_abbrev}[/red]"
            elif result.status == 'partial':
                status_str = f"[yellow]{status_abbrev}[/yellow]"
            else:
                status_str = f"[dim]{status_abbrev}[/dim]"

            # Format epochs
            epochs_str = f"{result.epochs_completed}/{result.total_epochs}"
            if result.total_epochs == 0:
                epochs_str = "N/A"

            # Get reason abbreviation
            reason_abbrev = REASON_ABBREV.get(result.completion_reason, 'UNK')

            table.add_row(
                result.name,
                exp_id_short,
                status_str,
                epochs_str,
                reason_abbrev
            )

        self.console.print(table)

    def _display_legends(self):
        """Display legends for abbreviations used in tables."""
        self.console.print("\n[bold]Legends:[/bold]")

        # Status legend
        self.console.print("  [underline]Status:[/underline]")
        for abbrev, desc in STATUS_LEGEND.items():
            self.console.print(f"    • {abbrev} = {desc}")

        # Reason legend
        self.console.print("\n  [underline]Completion Reason:[/underline]")
        for abbrev, desc in REASON_LEGEND.items():
            self.console.print(f"    • {abbrev} = {desc}")

        self.console.print("\n")
