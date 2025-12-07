"""
Utility functions for data path replacement in configurations.

This module provides functions to replace relative data paths with absolute paths
in data configuration dictionaries.
"""

from typing import Dict, Any


def replace_data_paths(data_config: Dict[str, Any], base_data_path: str) -> Dict[str, Any]:
    """
    Recursively replace './data' with base_data_path in all path fields of data config.
    
    This function traverses the data config dictionary and replaces any string value
    that starts with './data' with the provided base_data_path, preserving the rest
    of the path structure.
    
    Args:
        data_config: Data configuration dictionary (may be nested)
        base_data_path: Base path to replace './data' with (e.g., '/nfs/.../data')
    
    Returns:
        Modified data configuration dictionary with paths replaced
    
    Example:
        >>> config = {'root_path': './data/dataset', 'hetero_info': {'root_path': './data/hetero'}}
        >>> result = replace_data_paths(config, '/nfs/data')
        >>> result['root_path']
        '/nfs/data/dataset'
        >>> result['hetero_info']['root_path']
        '/nfs/data/hetero'
    """
    if not base_data_path:
        return data_config
    
    # Normalize base_data_path to remove trailing slash
    base_data_path = base_data_path.rstrip('/')
    
    result = {}
    for key, value in data_config.items():
        if isinstance(value, str):
            # Replace './data' at the start of the string
            if value.startswith('./data'):
                # Replace './data' with base_data_path, preserving the rest
                result[key] = value.replace('./data', base_data_path, 1)
            else:
                result[key] = value
        elif isinstance(value, dict):
            # Recursively process nested dictionaries
            result[key] = replace_data_paths(value, base_data_path)
        elif isinstance(value, list):
            # Process lists (may contain strings or dicts)
            result[key] = [
                replace_data_paths(item, base_data_path) if isinstance(item, dict)
                else item.replace('./data', base_data_path, 1) if isinstance(item, str) and item.startswith('./data')
                else item
                for item in value
            ]
        else:
            result[key] = value
    
    return result

