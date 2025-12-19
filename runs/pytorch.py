"""
PyTorch training execution module.

This module provides the execution logic for standard PyTorch-based training,
migrated from run.py. Uses Pydantic configs and ExperimentManager.
"""

import os
import torch
import random
import numpy as np
import yaml
from typing import Optional, Dict, Any
from utils.tools import dotdict
from utils.task import ahead_task_parser
from utils.gpu_monitor import gpu_monitoring_context
from exp.exp_universal import Experiment
from cli.config.models import ExperimentConfig
from exp.manager import ExperimentManager
from utils.data_path_utils import replace_data_paths


def config_to_args(config: ExperimentConfig, exp_manager: ExperimentManager):
    """
    Convert ExperimentConfig to argparse-like args object.
    
    This function bridges the gap between the new Pydantic config system
    and the existing Experiment class that expects argparse args.
    
    Args:
        config: ExperimentConfig instance containing experiment configuration
        exp_manager: ExperimentManager instance for experiment tracking
    
    Returns:
        dotdict object compatible with Experiment class
    """
    args = dotdict()
    
    # Model config
    args.model = config.model.name
    args.model_config = config.model.config_path
    
    # Data config
    args.data = config.data.name
    args.data_config = config.data.config_path
    args.checkpoints = str(exp_manager.get_checkpoint_dir())
    args.scale = config.training.scale
    args.disable_buffer = config.training.disable_buffer
    args.preload_hetero = config.training.preload_hetero
    args.prefetch_factor = config.training.prefetch_factor
    args.noise = config.training.noise
    args.downsample = config.training.downsample
    
    # Forecasting task
    args.ahead = config.training.ahead
    args.output_len = config.training.output_len or 1000
    args.input_len = config.training.input_len or 1000
    
    # Optimization
    args.num_workers = config.training.num_workers
    args.train_epochs = config.training.epochs
    args.batch_size = config.training.batch_size
    args.patience = config.training.patience
    args.learning_rate = config.training.learning_rate
    args.loss = config.training.loss
    args.lradj = config.training.lradj
    args.track_per_sample = config.training.track_per_sample
    args.evaluate_test_during_training = config.training.evaluate_test_during_training
    
    # GPU
    args.use_gpu = config.device.use_gpu
    args.gpu = config.device.gpu
    args.use_multi_gpu = config.device.use_multi_gpu
    args.devices = config.device.devices
    
    # Environment variables
    args.hf_mirror = config.hf_mirror
    args.hf_offline = config.hf_offline
    
    # Load model and data configs as dotdict (for backward compatibility)
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Merge model_config overrides if present (from experiment suite)
    # Supports both explicit model_config_overrides field and legacy model_config extra field
    # Explicit field takes precedence for clarity and type safety
    # Note: model_config_overrides is at top level of config (flattened from overrides by merge_configs)
    if hasattr(config, 'model_config_overrides') and config.model_config_overrides is not None:
        model_config.update(config.model_config_overrides)
    # Fallback: check legacy 'model_config' extra field (for backward compatibility)
    elif hasattr(config, 'model_config') and isinstance(getattr(config, 'model_config', None), dict):
        model_config.update(getattr(config, 'model_config'))
    # Final fallback: check model_dump() which should include extra fields
    elif hasattr(config, 'model_dump'):
        config_dict = config.model_dump(mode='python')
        if 'model_config' in config_dict and isinstance(config_dict['model_config'], dict):
            model_config.update(config_dict['model_config'])
    
    args.model_config = dotdict(model_config)
    
    with open(args.data_config, 'r') as f:
        data_configs = yaml.safe_load(f)
    
    # Merge data_config overrides if present (from experiment suite)
    # Pydantic models with extra="allow" store extra fields in model_extra or model_dump()
    if hasattr(config, 'model_dump'):
        config_dict = config.model_dump()
        if 'data_config' in config_dict and isinstance(config_dict['data_config'], dict):
            data_configs.update(config_dict['data_config'])
    # Also check if data_config is directly accessible (for backwards compatibility)
    elif hasattr(config, 'data_config') and isinstance(config.data_config, dict):
        data_configs.update(config.data_config)
    
    # Replace './data' with base_data_path if specified
    if config.base_data_path:
        data_configs = replace_data_paths(data_configs, config.base_data_path)
    
    args.data_config = dotdict(data_configs)
    
    # Handle ahead task
    if args.ahead is not None:
        assert args.ahead in ['day', 'week', 'month'], 'ahead task not supported, or add your own parser'
        try:
            args.output_len, args.input_len = ahead_task_parser(args.ahead, args.data_config.sampling_rate)
        except:
            raise ValueError('sampling rate not found in data config, fall back to default, input output length')
    
    # Set GPU availability
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    # Configure device IDs for multi-GPU training
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    return args


def run(config: ExperimentConfig, suite_name: Optional[str] = None, suite_info: Optional[Dict[str, Any]] = None, output_dir: Optional[str] = None, init_only: bool = False):
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
        init_only=init_only
    )
    
    if init_only:
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
        # Initialize and run experiment (pass exp_manager for tracking)
        exp = Experiment(args, exp_manager=exp_manager)
        print(f'>>>>>>>start training : {experiment_id}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train()

