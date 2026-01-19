"""Progress bar utilities for optional progress display."""

from typing import Optional, Any


class ProgressWrapper:
    """
    Wrapper that provides progress bar when console is available,
    or no-op operations when console is None.

    This allows single-loop code structure without duplication
    for if/else console availability logic.

    Supports custom fields for inline metrics display.
    """

    def __init__(self, console: Optional[Any], show_metrics: bool = False):
        """
        Initialize progress wrapper.

        Args:
            console: Rich console instance, or None for no-op mode
            show_metrics: If True, include columns for inline metrics display
        """
        self.console = console
        self.progress = None
        self.show_metrics = show_metrics

    def __enter__(self):
        if self.console is not None:
            from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

            if self.show_metrics:
                # Progress bar with inline metrics
                progress_columns = [
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TextColumn("•"),
                    TextColumn("{task.fields[metrics]}"),  # Dynamic metrics field
                    TimeElapsedColumn(),
                ]
            else:
                # Basic progress bar (original)
                progress_columns = [
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                ]

            self.progress = Progress(*progress_columns, console=self.console)
            self.progress.__enter__()
        return self

    def __exit__(self, *args):
        if self.progress is not None:
            self.progress.__exit__(*args)

    def add_task(self, description: str, total: int, **kwargs) -> Optional[Any]:
        """
        Add a task to progress bar (no-op if console is None).

        Args:
            description: Task description
            total: Total number of items
            **kwargs: Custom fields for inline metrics (e.g., metrics="MSE: 0.123")
        """
        if self.progress is not None:
            # Set default metrics field if show_metrics is True
            if self.show_metrics and 'metrics' not in kwargs:
                kwargs['metrics'] = ""
            return self.progress.add_task(description, total=total, **kwargs)
        return None

    def update(self, task_id: Optional[Any], advance: int = 1, **kwargs) -> None:
        """
        Update task progress (no-op if console is None or task_id is None).

        Args:
            task_id: Task ID from add_task()
            advance: Number of items to advance
            **kwargs: Custom fields to update (e.g., metrics="MSE: 0.123")
        """
        if self.progress is not None and task_id is not None:
            self.progress.update(task_id, advance=advance, **kwargs)

    def remove_task(self, task_id: Optional[Any]) -> None:
        """Remove a task from progress bar (no-op if console is None or task_id is None)."""
        if self.progress is not None and task_id is not None:
            self.progress.remove_task(task_id)
