"""
PyTorch training execution module.

This module provides the execution logic for standard PyTorch-based training,
migrated from run.py. Uses Pydantic configs and ExperimentManager.
"""

import os
import torch
import random
import numpy as np
from typing import Optional, Dict, Any
from utils.tools import dotdict
from utils.task import ahead_task_parser
from utils.gpu_monitor import gpu_monitoring_context
from exp.exp_universal import Experiment
from cli.config.models import ExperimentConfig
from exp.manager import ExperimentManager
from utils.experiment_config_builder import build_experiment_args


def pydantic_config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Convert Pydantic ExperimentConfig to dict format for build_experiment_args.
    
    This bridges the gap between Pydantic config models and the centralized
    config builder that expects dict input.
    
    Args:
        config: Pydantic ExperimentConfig instance
    
    Returns:
        Dict in the format expected by build_experiment_args
    """
    # Start with base model dump
    config_dict = config.model_dump() if hasattr(config, 'model_dump') else dict(config)
    
    # Ensure training section has all fields from Pydantic model
    # (Pydantic config uses 'epochs' but build_experiment_args expects it in training section)
    if 'training' not in config_dict:
        config_dict['training'] = {}
    
    # The Pydantic model has training as a nested object, ensure it's a dict
    training = config_dict.get('training', {})
    if hasattr(training, 'model_dump'):
        config_dict['training'] = training.model_dump()
    elif not isinstance(training, dict):
        config_dict['training'] = dict(training)
    
    # Same for model section
    model = config_dict.get('model', {})
    if hasattr(model, 'model_dump'):
        config_dict['model'] = model.model_dump()
    elif not isinstance(model, dict):
        config_dict['model'] = dict(model)
    
    # Same for data section
    data = config_dict.get('data', {})
    if hasattr(data, 'model_dump'):
        config_dict['data'] = data.model_dump()
    elif not isinstance(data, dict):
        config_dict['data'] = dict(data)
    
    # Same for device section
    device = config_dict.get('device', {})
    if hasattr(device, 'model_dump'):
        config_dict['device'] = device.model_dump()
    elif not isinstance(device, dict):
        config_dict['device'] = dict(device)
    
    # Handle llm_embedding (can be Pydantic model or None)
    llm_embedding = config_dict.get('llm_embedding')
    if llm_embedding is not None and hasattr(llm_embedding, 'model_dump'):
        config_dict['llm_embedding'] = llm_embedding.model_dump()
    
    return config_dict


def config_to_args(config: ExperimentConfig, exp_manager: ExperimentManager):
    """
    Convert ExperimentConfig to argparse-like args object.
    
    Uses the centralized build_experiment_args to ensure consistent config handling
    across all tools (CLI, training, tensor cache). This fixes the hetero_stride
    hash mismatch between tensor cache generation and training.
    
    Args:
        config: ExperimentConfig instance containing experiment configuration
        exp_manager: ExperimentManager instance for experiment tracking
    
    Returns:
        dotdict object compatible with Experiment class
    """
    # Convert Pydantic config to dict format
    config_dict = pydantic_config_to_dict(config)
    
    # Use centralized builder for consistent config handling
    # This ensures hetero_stride is computed the same way as CLI tools
    args = build_experiment_args(config_dict, include_gpu=True, include_training=True)
    
    # === Runtime-specific overrides ===
    # These are set by ExperimentManager and aren't part of the config file
    args.checkpoints = str(exp_manager.get_checkpoint_dir())
    
    # === Apply model-specific training configs ===
    # (e.g., LeRet two-stage training)
    from cli.config.model_training import apply_model_configs_to_args
    apply_model_configs_to_args(args, config.training, config.model.name)
    
    # === Handle ahead task parsing ===
    # Override input_len/output_len if ahead task is specified
    if args.ahead is not None:
        assert args.ahead in ['day', 'week', 'month'], 'ahead task not supported, or add your own parser'
        try:
            args.output_len, args.input_len = ahead_task_parser(args.ahead, args.data_config.sampling_rate)
        except:
            raise ValueError('sampling rate not found in data config, fall back to default, input output length')
    
    # === GPU availability check ===
    # Override use_gpu based on actual CUDA availability
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    # === Configure device IDs for multi-GPU training ===
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    return args


def run(config: ExperimentConfig, suite_name: Optional[str] = None, suite_info: Optional[Dict[str, Any]] = None, output_dir: Optional[str] = None, init_only: bool = False, return_ids: bool = False, sweep: bool = False):
    """
    Run PyTorch training experiment.
    
    This function executes a complete PyTorch training pipeline based on
    the provided Pydantic configuration, with full experiment tracking.
    
    Args:
        config: ExperimentConfig instance containing experiment configuration
        suite_name: Optional suite name if experiment is part of a suite
        suite_info: Optional suite information dictionary
        output_dir: Optional base output directory (overrides config setting)
        init_only: If True, only initialize experiment structure without running training
        return_ids: If True, return experiment_id and suite_name as dict (useful for wandb sweeps)
        sweep: If True, running in wandb sweep context (pass to ExperimentManager)
    
    Returns:
        If return_ids=True, returns dict with 'experiment_id' and 'suite_name' keys.
        Otherwise returns None.
    
    Example:
        >>> from cli.config.loader import load_config
        >>> config = load_config("configs/experiments/dlinear_solar.yaml")
        >>> run(config)
    """
    # Detect SLURM job ID from environment if available
    slurm_job_id = os.environ.get('SLURM_JOB_ID', config.job_id)
    slurm_job_name = os.environ.get('SLURM_JOB_NAME', config.job_name)
    
    # Initialize experiment manager
    base_output_dir = output_dir or config.training.experiment_output or "./output"
    exp_manager = ExperimentManager(
        config=config,
        output_dir=base_output_dir,
        experiment_name=config.experiment_name,
        job_id=slurm_job_id,
        job_name=slurm_job_name,
        suite_name=suite_name,
        suite_info=suite_info,
        init_only=init_only,
        sweep=sweep
    )
    
    if init_only:
        if return_ids:
            return {
                'experiment_id': exp_manager.get_experiment_id(),
                'suite_name': exp_manager.suite_name
            }
        return

    # Set environment variables for HuggingFace
    if config.hf_mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    if config.hf_offline:
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
    
    # Convert config to args format
    args = config_to_args(config, exp_manager)
    
    # Use experiment ID instead of setting string
    experiment_id = exp_manager.get_experiment_id()
    
    # Set seeds for reproducibility (from config)
    fix_seed = config.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Run experiment with GPU monitoring context
    with gpu_monitoring_context(args, exp_manager, log_interval_s=30.0):
        # Check for model-specific training handler
        from exp.model_specific import has_custom_trainer, get_model_trainer
        
        if has_custom_trainer(args.model, framework="pytorch"):
            # Use model-specific trainer (e.g., LeRet two-stage training)
            trainer_fn = get_model_trainer(args.model, framework="pytorch")
            print(f'>>>>>>>start training : {experiment_id}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            trainer_fn(args, exp_manager=exp_manager)
        else:
            # Standard PyTorch training via Experiment class
            exp = Experiment(args, exp_manager=exp_manager)
            print(f'>>>>>>>start training : {experiment_id}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train()
    
    # Return IDs if requested (for wandb sweep integration)
    if return_ids:
        return {
            'experiment_id': exp_manager.get_experiment_id(),
            'suite_name': exp_manager.suite_name
        }

