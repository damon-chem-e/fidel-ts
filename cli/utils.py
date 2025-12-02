"""
Shared utilities for CLI modules.
"""

import sys
import traceback
from typing import Any

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


def safe_float(value: Any, default: float) -> float:
    """
    Safely convert a value to float, handling string representations.
    
    This function ensures that YAML values (which may be loaded as strings)
    are properly converted to float types. Handles scientific notation strings
    like "5e-4" that may come from YAML parsing.
    
    Args:
        value: Value to convert (can be int, float, or string)
        default: Default value to return if conversion fails
    
    Returns:
        float: The converted float value or default if conversion fails
    
    Example:
        >>> safe_float("5e-4", 0.001)
        0.0005
        >>> safe_float(5e-4, 0.001)
        0.0005
        >>> safe_float(None, 0.001)
        0.001
    """
    if value is None:
        return default
    
    # If already a float or int, return as float
    if isinstance(value, (int, float)):
        return float(value)
    
    # If string, try to convert
    if isinstance(value, str):
        try:
            # Handle scientific notation strings
            return float(value)
        except (ValueError, TypeError):
            return default
    
    return default


def safe_int(value: Any, default: int) -> int:
    """
    Safely convert a value to int, handling string representations.
    
    This function ensures that YAML values (which may be loaded as strings)
    are properly converted to int types.
    
    Args:
        value: Value to convert (can be int, float, or string)
        default: Default value to return if conversion fails
    
    Returns:
        int: The converted int value or default if conversion fails
    
    Example:
        >>> safe_int("96", 32)
        96
        >>> safe_int(96, 32)
        96
        >>> safe_int(96.0, 32)
        96
        >>> safe_int(None, 32)
        32
    """
    if value is None:
        return default
    
    # If already an int, return as is
    if isinstance(value, int):
        return value
    
    # If float, convert to int
    if isinstance(value, float):
        return int(value)
    
    # If string, try to convert
    if isinstance(value, str):
        try:
            # Convert float string to int if needed
            return int(float(value))
        except (ValueError, TypeError):
            return default
    
    return default


def safe_bool(value: Any, default: bool) -> bool:
    """
    Safely convert a value to bool, handling various representations.
    
    This function ensures that YAML boolean values (which may be loaded as strings)
    are properly converted to bool types.
    
    Args:
        value: Value to convert (can be bool, str, int, etc.)
        default: Default value to return if conversion fails
    
    Returns:
        bool: The converted bool value or default if conversion fails
    
    Example:
        >>> safe_bool("true", False)
        True
        >>> safe_bool(1, False)
        True
        >>> safe_bool("false", True)
        False
        >>> safe_bool(None, True)
        True
    """
    if value is None:
        return default
    
    # If already a bool, return as is
    if isinstance(value, bool):
        return value
    
    # If int, convert (0 = False, non-zero = True)
    if isinstance(value, int):
        return bool(value)
    
    # If string, check for common boolean representations
    if isinstance(value, str):
        lower_value = value.lower().strip()
        if lower_value in ('true', 'yes', '1', 'on'):
            return True
        elif lower_value in ('false', 'no', '0', 'off'):
            return False
    
    return default

