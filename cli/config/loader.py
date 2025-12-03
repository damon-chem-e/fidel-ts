"""
Configuration loader with support for nested subconfigs using Pydantic.

This module provides functionality to load primary configuration files
that may reference nested subconfigs (e.g., plotting, evaluation configs).
All configs are loaded as Pydantic models for type safety and validation.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union

from cli.config.models import ExperimentConfig


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


def load_config(config_path: str, base_dir: Optional[Path] = None) -> ExperimentConfig:
    """
    Load a primary configuration file as Pydantic model.
    
    This function loads a single config file and validates it using Pydantic.
    For nested config support with full hierarchy, use load_config_with_nested().
    
    Args:
        config_path: Path to primary config file
        base_dir: Base directory for resolving relative paths
    
    Returns:
        ExperimentConfig instance
    
    Example:
        >>> config = load_config("configs/experiments/dlinear_solar.yaml")
        >>> print(config.model.name)
        'DLinear'
    """
    resolved_path = resolve_config_path(config_path, base_dir)
    return ExperimentConfig.from_yaml(resolved_path)


def load_config_with_nested(config_path: str, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load a primary configuration file and all referenced nested subconfigs.
    
    This function:
    1. Loads the primary config file as Pydantic model
    2. Identifies references to subconfigs (keys with string values that end in .yaml/.yml)
    3. Loads each subconfig as raw dictionaries (for nested configs like plotting, evaluation)
    4. Returns a hierarchy with primary config (Pydantic) and nested configs (dicts)
    
    Args:
        config_path: Path to primary config file
        base_dir: Base directory for resolving relative paths
    
    Returns:
        Dictionary with structure:
        {
            'primary': ExperimentConfig(...),  # Primary config as Pydantic model
            'nested': {
                'plotting': {...},  # If plotting: "configs/plotting/default.yaml" exists
                'evaluation': {...},  # If evaluation: "configs/evaluation/default.yaml" exists
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
        >>> primary_config = result['primary']  # ExperimentConfig instance
        >>> plotting_config = result['nested'].get('plotting')  # dict
    """
    if base_dir is None:
        base_dir = Path.cwd()
    
    # Load primary config as Pydantic model
    primary_path = resolve_config_path(config_path, base_dir)
    primary_config = ExperimentConfig.from_yaml(primary_path)
    
    # Track all config paths
    config_paths = {'primary': primary_path}
    nested_configs = {}
    
    # Find and load nested configs (plotting, evaluation, etc.)
    # These are stored as raw dicts since they're not part of the main ExperimentConfig schema
    # Note: model.config_path and data.config_path are NOT nested configs, they're just file paths
    def find_subconfigs(config_dict: Dict[str, Any], parent_path: Path) -> None:
        """Recursively find and load subconfig references."""
        for key, value in config_dict.items():
            if isinstance(value, str) and (value.endswith('.yaml') or value.endswith('.yml')):
                # This looks like a subconfig reference
                # Only treat top-level keys 'plotting' and 'evaluation' as nested configs
                # model.config_path and data.config_path are just file paths, not nested configs
                if key in ['plotting', 'evaluation']:
                    try:
                        subconfig_path = resolve_config_path(value, parent_path.parent)
                        if subconfig_path.exists():
                            subconfig = load_yaml_config(subconfig_path)
                            nested_configs[key] = subconfig
                            config_paths[key] = subconfig_path
                            # Recursively check for nested configs within this subconfig
                            find_subconfigs(subconfig, subconfig_path)
                    except (FileNotFoundError, yaml.YAMLError):
                        # If subconfig doesn't exist or fails to load, skip it
                        # This allows optional subconfigs
                        pass
            elif isinstance(value, dict):
                # Recursively check nested dictionaries
                find_subconfigs(value, parent_path)
    
    # Extract dict from Pydantic model to search for nested configs
    primary_dict = primary_config.model_dump()
    find_subconfigs(primary_dict, primary_path)
    
    return {
        'primary': primary_config,
        'nested': nested_configs,
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
        Dictionary with complete config hierarchy:
        {
            'primary_config': ExperimentConfig(...),
            'nested_configs': {...},
            'config_paths': {...},
            'all_configs': {
                'primary': ExperimentConfig(...),
                **nested_configs
            }
        }
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
