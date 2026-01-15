"""
Centralized experiment configuration builder.

This module provides a single source of truth for building experiment args
from suite/experiment configs, ensuring consistent override application across:
- Tensor cache generation
- Profiling tools
- (Future) Training, evaluation

The key function is `build_experiment_args()` which:
1. Loads base data config from YAML file
2. Applies data_config overrides from experiment config (deep merge)
3. Loads base model config from YAML file  
4. Applies model_config_overrides from experiment config (deep merge)
5. Computes effective hetero_stride from text_embedding_stride config
6. Builds complete args dotdict

This ensures tensor cache generation uses the SAME effective config as training.

Usage:
    from utils.experiment_config_builder import build_experiment_args
    
    # experiment_config is the merged config (template + overrides)
    args = build_experiment_args(experiment_config)
    data_provider = Data_Provider(args)
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

from utils.tools import dotdict
from utils.config_utils import merge_configs
from utils.data_path_utils import replace_data_paths

logger = logging.getLogger(__name__)


def compute_effective_hetero_stride(
    text_embedding_stride: Optional[Any],
    model_config: Optional[dotdict]
) -> int:
    """
    Compute the effective hetero_stride from text_embedding_stride config.
    
    This centralizes the stride computation logic used by data loading,
    tensor cache, and models.
    
    Args:
        text_embedding_stride: Config value - can be:
            - None or 'aligned': Use model_config.stride if hetero_align_stride=True, else 1
            - 'full': Always use 1 (full resolution)
            - int: Use explicit value
        model_config: Model configuration dotdict (may contain stride, hetero_align_stride)
        
    Returns:
        Effective hetero_stride (1 = full resolution, >1 = strided)
        
    Examples:
        # Full resolution
        compute_effective_hetero_stride('full', model_config) -> 1
        
        # Aligned with model (stride=3, hetero_align_stride=True)
        compute_effective_hetero_stride('aligned', model_config) -> 3
        
        # Aligned but hetero_align_stride=False
        compute_effective_hetero_stride('aligned', model_config) -> 1
        
        # Explicit stride
        compute_effective_hetero_stride(6, model_config) -> 6
    """
    # Handle 'full' - always use stride=1
    if text_embedding_stride == 'full':
        return 1
    
    # Handle explicit integer
    if isinstance(text_embedding_stride, int):
        if text_embedding_stride < 1:
            raise ValueError(f"text_embedding_stride must be >= 1, got {text_embedding_stride}")
        return text_embedding_stride
    
    # Handle None or 'aligned' - use model config's stride if hetero_align_stride=True
    if text_embedding_stride is None or text_embedding_stride == 'aligned':
        if model_config is None:
            return 1
        
        # Get hetero_align_stride (default True for backward compatibility)
        hetero_align_stride = model_config.get('hetero_align_stride', True) if hasattr(model_config, 'get') else True
        
        if hetero_align_stride:
            # Use model's stride
            stride = model_config.get('stride', 1) if hasattr(model_config, 'get') else 1
            return stride if stride else 1
        else:
            return 1
    
    # Invalid value
    raise ValueError(
        f"Invalid text_embedding_stride: {text_embedding_stride!r}. "
        f"Expected 'full', 'aligned', None, or positive integer."
    )


def load_and_merge_data_config(
    data_config_path: str,
    data_config_overrides: Optional[Dict[str, Any]] = None,
    base_data_path: Optional[str] = None
) -> dotdict:
    """
    Load base data config and apply overrides using deep merge.
    
    This ensures experiment-level overrides (e.g., timemmd_text_output: embedding)
    are properly applied on top of the base data config file.
    
    Args:
        data_config_path: Path to base data config YAML file
        data_config_overrides: Optional overrides to merge (from experiment config)
        base_data_path: Optional base path for data directory substitution
    
    Returns:
        Merged data config as dotdict
        
    Example:
        Base config (data_configs/time_mmd/Traffic/config.yaml):
            timemmd_text_output: text
            root_path: ./data/time_mmd/Traffic
            
        Overrides (from experiment suite):
            timemmd_text_output: embedding
            
        Result:
            timemmd_text_output: embedding  (override wins)
            root_path: ./data/time_mmd/Traffic  (from base)
    """
    # Load base config from file
    if data_config_path and Path(data_config_path).exists():
        with open(data_config_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f) or {}
    else:
        data_config = {}
        if data_config_path:
            logger.warning(f"Data config file not found: {data_config_path}")
    
    # Apply overrides using deep merge
    # This is the critical step that was missing in tensor_cache.py!
    if data_config_overrides and isinstance(data_config_overrides, dict):
        data_config = merge_configs(data_config, data_config_overrides)
        logger.debug(f"Applied data_config overrides: {list(data_config_overrides.keys())}")
    
    # Apply base_data_path substitution if specified (uses same utility as training code)
    # This recursively replaces './data' in all path fields including nested structures
    if base_data_path:
        data_config = replace_data_paths(data_config, base_data_path)
        logger.debug(f"Applied base_data_path substitution: {base_data_path}")
    
    return dotdict(data_config)


def load_and_merge_model_config(
    model_config_path: str,
    model_config_overrides: Optional[Dict[str, Any]] = None
) -> dotdict:
    """
    Load base model config and apply overrides using deep merge.
    
    Args:
        model_config_path: Path to base model config YAML file
        model_config_overrides: Optional overrides to merge (from experiment config)
    
    Returns:
        Merged model config as dotdict
    """
    # Load base config from file
    if model_config_path and Path(model_config_path).exists():
        with open(model_config_path, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f) or {}
    else:
        model_config = {}
        if model_config_path:
            logger.warning(f"Model config file not found: {model_config_path}")
    
    # Apply overrides using deep merge
    if model_config_overrides and isinstance(model_config_overrides, dict):
        model_config = merge_configs(model_config, model_config_overrides)
        logger.debug(f"Applied model_config overrides: {list(model_config_overrides.keys())}")
    
    return dotdict(model_config)


def build_experiment_args(
    experiment_config: Dict[str, Any],
    include_gpu: bool = True
) -> dotdict:
    """
    Build complete experiment args from merged experiment config.
    
    This is the SINGLE SOURCE OF TRUTH for building args from experiment configs.
    Tools like tensor_cache and profile_dataloader should use this function.
    
    IMPORTANT: Training code (runs/pytorch.py, etc.) has its own config_to_args()
    function that is intentionally NOT changed. This function replicates the
    same logic to ensure tensor cache uses identical config to training.
    
    Args:
        experiment_config: Merged experiment config (template + overrides from suite)
        include_gpu: Whether to include GPU-related args (default: True)
    
    Returns:
        Complete args dotdict ready for Data_Provider
        
    Config structure expected:
        model:
            name: "lynx_film_raw"
            config_path: "model_configs/general/lynx_film_raw.yaml"
        model_config_overrides:  # Optional overrides for model config
            d_model: 1024
        data:
            name: "time_mmd_traffic"
            config_path: "data_configs/time_mmd/Traffic/config.yaml"
        data_config:  # Optional overrides for data config
            timemmd_text_output: embedding
        training:
            input_len: 24
            output_len: 6
            ...
    """
    args = dotdict()
    
    # === Model config (with overrides) ===
    model_section = experiment_config.get('model', {})
    model_config_path = model_section.get('config_path', '')
    model_config_overrides = experiment_config.get('model_config_overrides', {})
    
    args.model = model_section.get('name', 'unknown')
    args.model_config = load_and_merge_model_config(model_config_path, model_config_overrides)
    
    # === Data config (with overrides) - THE CRITICAL FIX ===
    data_section = experiment_config.get('data', {})
    data_config_path = data_section.get('config_path', '')
    data_config_overrides = experiment_config.get('data_config', {})
    base_data_path = experiment_config.get('base_data_path')
    
    args.data = data_section.get('name', 'unknown')
    args.data_config = load_and_merge_data_config(
        data_config_path,
        data_config_overrides,
        base_data_path
    )
    
    # Store config_path and name in data_config for downstream use
    args.data_config.config_path = data_config_path
    args.data_config.name = data_section.get('name', 'unknown')
    
    # === Training config ===
    training = experiment_config.get('training', {})
    args.input_len = training.get('input_len', 336)
    args.output_len = training.get('output_len', 96)
    args.scale = training.get('scale', True)
    args.disable_buffer = training.get('disable_buffer', False)
    args.preload_hetero = training.get('preload_hetero', False)
    args.prefetch_factor = training.get('prefetch_factor', 2)
    args.noise = training.get('noise', 0.0)
    args.downsample = training.get('downsample', None)
    args.num_workers = training.get('num_workers', 0)
    args.batch_size = training.get('batch_size', 32)
    args.truncate_train_for_purge = training.get('truncate_train_for_purge', False)
    args.ahead = training.get('ahead', None)
    
    # === Text embedding stride (controls hetero data resolution) ===
    # Extract from training config
    args.text_embedding_stride = training.get('text_embedding_stride', None)
    
    # Compute effective hetero_stride from text_embedding_stride
    # This replaces the scattered computation logic in data_factory and models
    args.hetero_stride = compute_effective_hetero_stride(
        args.text_embedding_stride,
        args.model_config
    )
    
    # Inject resolved hetero_stride into both model_config and data_config
    # - model_config: so models can access it during initialization
    # - data_config: so tensor_cache can read it from metadata.data_config
    # This ensures all components use the same stride value
    if args.model_config is not None:
        args.model_config['hetero_stride'] = args.hetero_stride
    if args.data_config is not None:
        args.data_config['hetero_stride'] = args.hetero_stride
    
    # === GPU config (optional) ===
    if include_gpu:
        import torch
        device_config = experiment_config.get('device', {})
        args.use_gpu = device_config.get('use_gpu', torch.cuda.is_available())
        args.gpu = device_config.get('gpu', 0)
    
    return args


def build_cache_config(args: dotdict) -> dict:
    """
    Build config dict for tensor cache hash computation.
    
    These parameters determine cache uniqueness - if any change, cache must be regenerated.
    This should match the parameters that affect data loading behavior.
    
    Args:
        args: Argument object from build_experiment_args
    
    Returns:
        Dict of parameters that affect cache validity
    """
    # Use pre-computed hetero_stride if available, otherwise compute it
    # This ensures consistency with build_experiment_args
    if hasattr(args, 'hetero_stride'):
        hetero_stride = args.hetero_stride
    else:
        # Fallback for args not built by build_experiment_args
        text_embedding_stride = getattr(args, 'text_embedding_stride', None)
        model_config = getattr(args, 'model_config', None)
        hetero_stride = compute_effective_hetero_stride(text_embedding_stride, model_config)

    # Get hetero_type from data config if available
    hetero_type = None
    if hasattr(args, 'data_config') and args.data_config.get('hetero_info'):
        hetero_type = args.data_config.hetero_info.get('hetero_type')

    # Get timemmd_text_output - critical for cache validity!
    timemmd_text_output = None
    if hasattr(args, 'data_config'):
        timemmd_text_output = args.data_config.get('timemmd_text_output')

    return {
        'input_len': args.input_len,
        'output_len': args.output_len,
        'scale': args.scale,
        'truncate_train_for_purge': args.truncate_train_for_purge,
        'downsample': args.downsample,
        'data_name': args.data,
        'hetero_stride': hetero_stride,
        'hetero_type': hetero_type,
        'timemmd_text_output': timemmd_text_output,  # Include in hash!
        'missing_value_strategy': args.data_config.get('missing_value_strategy', 'none') if args.data_config else 'none',
        'split_info': str(args.data_config.get('split_info', '')) if args.data_config else '',
    }
