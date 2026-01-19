"""Progress bar utilities for optional progress display."""

from typing import Optional, Any


class ProgressWrapper:
    """
    Wrapper that provides progress bar when console is available,
    or no-op operations when console is None.

    This allows single-loop code structure without duplication
    for if/else console availability logic.
    """

    def __init__(self, console: Optional[Any]):
        """
        Initialize progress wrapper.

        Args:
            console: Rich console instance, or None for no-op mode
        """
        self.console = console
        self.progress = None

    def __enter__(self):
        if self.console is not None:
            from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
            self.progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console
            )
            self.progress.__enter__()
        return self

    def __exit__(self, *args):
        if self.progress is not None:
            self.progress.__exit__(*args)

    def add_task(self, description: str, total: int) -> Optional[Any]:
        """Add a task to progress bar (no-op if console is None)."""
        if self.progress is not None:
            return self.progress.add_task(description, total=total)
        return None

    def update(self, task_id: Optional[Any], advance: int = 1) -> None:
        """Update task progress (no-op if console is None or task_id is None)."""
        if self.progress is not None and task_id is not None:
            self.progress.update(task_id, advance=advance)

    def remove_task(self, task_id: Optional[Any]) -> None:
        """Remove a task from progress bar (no-op if console is None or task_id is None)."""
        if self.progress is not None and task_id is not None:
            self.progress.remove_task(task_id)
