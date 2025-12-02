"""
Shared utilities for CLI modules.
"""

import sys
import traceback

try:
    from rich.console import Console
    from rich.traceback import Traceback
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


def handle_error(error: Exception, context: str = "Error", use_rich: bool = True):
    """
    Handle and display errors with Rich formatting if available.
    
    Args:
        error: The exception that was raised
        context: Context string describing where the error occurred
        use_rich: Whether to use Rich formatting (if available)
    """
    if RICH_AVAILABLE and use_rich:
        console.print(Panel(
            Traceback.from_exception(type(error), error, error.__traceback__),
            title=f"[bold red]{context}[/bold red]",
            border_style="red"
        ))
    else:
        print(f"{context}: {error}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

