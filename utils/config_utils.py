"""
Configuration utility functions.

This module provides utilities for merging and manipulating configuration dictionaries.
"""

from typing import Dict, Any


def merge_configs(template: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge template configuration with overrides using deep merge.
    
    This function performs a deep merge, where overrides take precedence
    over template values. Nested dictionaries are merged recursively.
    
    Args:
        template: Base template configuration
        overrides: Configuration overrides
    
    Returns:
        Merged configuration dictionary
    
    Example:
        >>> template = {'a': 1, 'b': {'c': 2, 'd': 3}}
        >>> overrides = {'b': {'c': 4}}
        >>> merge_configs(template, overrides)
        {'a': 1, 'b': {'c': 4, 'd': 3}}
    """
    result = template.copy()
    
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Override with new value
            result[key] = value
    
    return result

