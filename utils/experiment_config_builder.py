"""
Centralized experiment configuration builder.

This module provides a single source of truth for building experiment args
from suite/experiment configs, ensuring consistent override application across:
- Training (runs/pytorch.py)
- Tensor cache generation (cli/tensor_cache.py)
- Profiling tools

The key function is `build_experiment_args()` which:
1. Loads base data config from YAML file
2. Applies data_config overrides from experiment config (deep merge)
3. Loads base model config from YAML file  
4. Applies model_config_overrides from experiment config (deep merge)
5. Computes effective hetero_stride from text_embedding_stride config
6. Builds complete args dotdict

This ensures tensor cache generation uses the SAME effective config as training,
preventing hash mismatches between cache generation and training runtime.

Usage:
    from utils.experiment_config_builder import build_experiment_args
    
    # experiment_config is the merged config (template + overrides)
    args = build_experiment_args(experiment_config)
    data_provider = Data_Provider(args)
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

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
    include_gpu: bool = True,
    include_training: bool = True
) -> dotdict:
    """
    Build complete experiment args from merged experiment config.
    
    This is the SINGLE SOURCE OF TRUTH for building args from experiment configs.
    All tools (tensor_cache, profile_dataloader, training) should use this function
    to ensure consistent config handling, especially for hetero_stride computation.
    
    Args:
        experiment_config: Merged experiment config (template + overrides from suite)
        include_gpu: Whether to include GPU-related args (default: True)
        include_training: Whether to include full training args like epochs, lr (default: True)
    
    Returns:
        Complete args dotdict ready for Data_Provider and training
        
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
            epochs: 50
            learning_rate: 0.001
            ...
    """
    args = dotdict()
    
    # === Model config (with overrides) ===
    model_section = experiment_config.get('model', {})
    model_config_path = model_section.get('config_path', '')
    model_config_overrides = experiment_config.get('model_config_overrides', {})
    
    args.model = model_section.get('name', 'unknown')
    args.model_config = load_and_merge_model_config(model_config_path, model_config_overrides)
    
    # Store model_config_overrides for downstream use (e.g., LLM embedding validation)
    args.model_config_overrides = model_config_overrides or {}
    
    # === Data config (with overrides) - THE CRITICAL FIX ===
    data_section = experiment_config.get('data', {})
    data_config_path = data_section.get('config_path', '')
    data_config_overrides = experiment_config.get('data_config', {})
    base_data_path = experiment_config.get('base_data_path')
    
    args.data = data_section.get('name', 'unknown')
    args.data_name = args.data  # Alias for backward compatibility
    args.data_config = load_and_merge_data_config(
        data_config_path,
        data_config_overrides,
        base_data_path
    )
    
    # Store config_path and name in data_config for downstream use
    args.data_config.config_path = data_config_path
    args.data_config.name = data_section.get('name', 'unknown')
    
    # Store base_data_path for LLM embedding provider
    args.base_data_path = base_data_path or './data/'
    
    # === Training config (core parameters) ===
    # NOTE: Use `or` pattern for fields that may be None in Pydantic model_dump()
    # This ensures we get the fallback when the value is None, not just missing
    training = experiment_config.get('training', {})
    
    # input_len/output_len can be None in Pydantic - use 1000 fallback to match old behavior
    args.input_len = training.get('input_len') or 1000
    args.output_len = training.get('output_len') or 1000
    
    # Boolean and numeric fields with proper defaults
    args.scale = training.get('scale', True)
    args.disable_buffer = training.get('disable_buffer', False)
    args.preload_hetero = training.get('preload_hetero', False)
    args.prefetch_factor = training.get('prefetch_factor', 2)
    args.noise = training.get('noise', 0.0)
    args.downsample = training.get('downsample', None)
    args.num_workers = training.get('num_workers', 0)
    args.batch_size = training.get('batch_size', 96)  # Pydantic default is 96
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
    
    # === Tensor cache settings ===
    args.use_tensor_cache = training.get('use_tensor_cache', False)
    args.tensor_cache_dir = training.get('tensor_cache_dir', None)
    
    # === PyTorch compile settings ===
    args.torch_compile = training.get('torch_compile', False)
    args.compile_mode = training.get('compile_mode', 'reduce-overhead')
    
    # === Full training parameters (epochs, learning rate, etc.) ===
    # NOTE: Defaults match Pydantic TrainingConfig defaults for consistency
    if include_training:
        args.train_epochs = training.get('epochs', 20)  # Pydantic default is 20
        args.patience = training.get('patience', 3)
        args.learning_rate = training.get('learning_rate', 5e-4)  # Pydantic default is 5e-4
        # Differential learning rate for projection layers (e.g., ZhangHanBest residual_proj)
        # If None, uses the same learning_rate for all parameters (backward compatible)
        args.projector_learning_rate = training.get('projector_learning_rate', None)
        args.loss = training.get('loss', 'mse')
        args.lradj = training.get('lradj', 'type3')  # Pydantic default is 'type3'
        args.track_per_sample = training.get('track_per_sample', False)
        args.evaluate_test_during_training = training.get('evaluate_test_during_training', False)
    
    # === Extract model architecture parameters from model config ===
    # (Used by model-specific trainers for loss computation)
    args.patch_len = args.model_config.get('patch_len', 16)
    args.stride = args.model_config.get('stride', 8)
    
    # === GPU config (optional) ===
    if include_gpu:
        import torch
        device_config = experiment_config.get('device', {})
        args.use_gpu = device_config.get('use_gpu', torch.cuda.is_available())
        args.gpu = device_config.get('gpu', 0)
        args.use_multi_gpu = device_config.get('use_multi_gpu', False)
        args.devices = device_config.get('devices', '0')
    
    # === LLM embedding config ===
    llm_embedding = experiment_config.get('llm_embedding')
    args.llm_embedding = llm_embedding if llm_embedding else None
    
    # === Environment variables ===
    args.hf_mirror = experiment_config.get('hf_mirror', False)
    args.hf_offline = experiment_config.get('hf_offline', False)
    
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

    # Get embedding configuration - critical for cache validity!
    # If embeddings change (different version, model, or aggregation), cache must be regenerated
    embedding_version = None
    embedding_model = None
    embedding_aggregation = None
    if hasattr(args, 'data_config') and args.data_config.get('hetero_info'):
        hetero_info = args.data_config.hetero_info

        # Check if using old embeddings (legacy .pkl files)
        use_old_embeddings = hetero_info.get('use_old_embeddings', False) if isinstance(hetero_info, dict) else getattr(hetero_info, 'use_old_embeddings', False)

        if use_old_embeddings:
            # Using old .pkl files - mark as version 1.0 (buggy concatenated embeddings)
            embedding_version = '1.0'
            # Old embeddings don't have standardized model/aggregation tracking
            embedding_model = None
            embedding_aggregation = None
        else:
            # Using new embedding system - embedding_config MUST exist
            # Handle both dict and dotdict for embedding_config
            emb_cfg = hetero_info.get('embedding_config') if isinstance(hetero_info, dict) else getattr(hetero_info, 'embedding_config', None)

            if emb_cfg:
                # Handle both dict and dotdict for embedding config values
                embedding_model = emb_cfg.get('model_name', 'bert-base-uncased') if isinstance(emb_cfg, dict) else getattr(emb_cfg, 'model_name', 'bert-base-uncased')
                embedding_aggregation = emb_cfg.get('aggregation_method', 'cls') if isinstance(emb_cfg, dict) else getattr(emb_cfg, 'aggregation_method', 'cls')
                # Version defaults to '2.0' for new embeddings (will be set explicitly by embedder)
                # This ensures tensor caches built with old embeddings are invalidated
                embedding_version = '2.0'  # Current embedding implementation version
            else:
                # ERROR: hetero_info exists but no embedding_config and not using old embeddings
                # This is a configuration error - fail loudly
                raise ValueError(
                    "hetero_info is present but missing 'embedding_config'. "
                    "This indicates a configuration error. "
                    "Either provide 'embedding_config' with 'model_name' and 'aggregation_method', "
                    "or set 'use_old_embeddings: true' to use legacy .pkl files. "
                    "See data_configs/ for examples."
                )

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
        'embedding_version': embedding_version,  # Invalidate cache when embedding version changes
        'embedding_model': embedding_model,  # Invalidate cache when embedding model changes
        'embedding_aggregation': embedding_aggregation,  # Invalidate cache when aggregation changes
    }
