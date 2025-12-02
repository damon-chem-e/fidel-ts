"""
Configuration loader with support for nested subconfigs.

This module provides functionality to load primary configuration files
that may reference nested subconfigs (e.g., plotting, evaluation configs).
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from utils.tools import dotdict


def _recursive_dotdict(obj):
    """
    Recursively convert nested dictionaries to dotdict objects.
    
    Args:
        obj: Dictionary or other object to convert
    
    Returns:
        dotdict or original object if not a dict
    """
    if isinstance(obj, dict):
        result = dotdict()
        for key, value in obj.items():
            result[key] = _recursive_dotdict(value)
        return result
    elif isinstance(obj, list):
        return [_recursive_dotdict(item) for item in obj]
    else:
        return obj


def resolve_config_path(config_path: str, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve a config path to an absolute path.
    
    Args:
        config_path: Relative or absolute path to config file
        base_dir: Base directory for resolving relative paths (default: current working directory)
    
    Returns:
        Absolute Path object to the config file
    """
    if base_dir is None:
        base_dir = Path.cwd()
    
    config_path_obj = Path(config_path)
    if config_path_obj.is_absolute():
        return config_path_obj
    else:
        return (base_dir / config_path_obj).resolve()


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Dictionary containing config contents
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        return {}
    
    return config


def load_config(config_path: str, base_dir: Optional[Path] = None) -> dotdict:
    """
    Load a primary configuration file and convert to dotdict.
    
    This function loads a single config file without resolving nested subconfigs.
    For nested config support, use load_config_with_nested().
    
    Args:
        config_path: Path to primary config file
        base_dir: Base directory for resolving relative paths
    
    Returns:
        dotdict object containing config contents
    
    Example:
        >>> config = load_config("configs/experiments/dlinear_solar.yaml")
        >>> print(config.model.name)
        'DLinear'
    """
    resolved_path = resolve_config_path(config_path, base_dir)
    config = load_yaml_config(resolved_path)
    return _recursive_dotdict(config)


def load_config_with_nested(config_path: str, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load a primary configuration file and all referenced nested subconfigs.
    
    This function:
    1. Loads the primary config file
    2. Identifies references to subconfigs (keys with string values that end in .yaml/.yml)
    3. Loads each subconfig recursively
    4. Returns a hierarchy with primary config and nested configs
    
    Args:
        config_path: Path to primary config file
        base_dir: Base directory for resolving relative paths
    
    Returns:
        Dictionary with structure:
        {
            'primary': dotdict(...),  # Primary config as dotdict
            'nested': {
                'plotting': dotdict(...),  # If plotting: "configs/plotting/default.yaml" exists
                'evaluation': dotdict(...),  # If evaluation: "configs/evaluation/default.yaml" exists
                ...
            },
            'config_paths': {
                'primary': Path(...),
                'plotting': Path(...),
                ...
            }
        }
    
    Example:
        >>> result = load_config_with_nested("configs/experiments/dlinear_solar.yaml")
        >>> primary_config = result['primary']
        >>> plotting_config = result['nested'].get('plotting')
    """
    if base_dir is None:
        base_dir = Path.cwd()
    
    # Load primary config
    primary_path = resolve_config_path(config_path, base_dir)
    primary_config = load_yaml_config(primary_path)
    
    # Track all config paths
    config_paths = {'primary': primary_path}
    nested_configs = {}
    
    # Find and load nested configs
    # Look for keys that have string values ending in .yaml or .yml
    def find_subconfigs(config_dict: Dict[str, Any], parent_path: Path) -> None:
        """Recursively find and load subconfig references."""
        for key, value in config_dict.items():
            if isinstance(value, str) and (value.endswith('.yaml') or value.endswith('.yml')):
                # This looks like a subconfig reference
                try:
                    subconfig_path = resolve_config_path(value, parent_path.parent)
                    if subconfig_path.exists():
                        subconfig = load_yaml_config(subconfig_path)
                        nested_configs[key] = _recursive_dotdict(subconfig)
                        config_paths[key] = subconfig_path
                        # Recursively check for nested configs within this subconfig
                        find_subconfigs(subconfig, subconfig_path)
                except (FileNotFoundError, yaml.YAMLError) as e:
                    # If subconfig doesn't exist or fails to load, skip it
                    # This allows optional subconfigs
                    pass
            elif isinstance(value, dict):
                # Recursively check nested dictionaries
                find_subconfigs(value, parent_path)
    
    find_subconfigs(primary_config, primary_path)
    
    return {
        'primary': _recursive_dotdict(primary_config),
        'nested': {k: v for k, v in nested_configs.items()},
        'config_paths': config_paths
    }


def get_config_hierarchy(config_path: str) -> Dict[str, Any]:
    """
    Get complete configuration hierarchy for tracking and documentation.
    
    This is a convenience function that returns a structured representation
    of all configs for use in experiment tracking.
    
    Args:
        config_path: Path to primary config file
    
    Returns:
        Dictionary with complete config hierarchy
    """
    result = load_config_with_nested(config_path)
    
    return {
        'primary_config': result['primary'],
        'nested_configs': result['nested'],
        'config_paths': result['config_paths'],
        'all_configs': {
            'primary': result['primary'],
            **result['nested']
        }
    }

