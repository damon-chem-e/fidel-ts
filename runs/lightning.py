"""
PyTorch Lightning training execution module.

This module provides the execution logic for PyTorch Lightning-based training,
migrated from run_lightning.py. Now uses Pydantic configs and ExperimentManager.
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
from exp.exp_lightning import train_lightning_model
from cli.config.models import ExperimentConfig
from exp.manager import ExperimentManager
from utils.data_path_utils import replace_data_paths
from utils.config_utils import merge_configs


def config_to_args(config: ExperimentConfig, exp_manager: ExperimentManager):
    """
    Convert ExperimentConfig to argparse-like args object for Lightning training.
    
    Args:
        config: ExperimentConfig instance containing experiment configuration
        exp_manager: ExperimentManager instance for experiment tracking
    
    Returns:
        dotdict object compatible with train_lightning_model function
    """
    args = dotdict()
    
    # Model config
    args.model = config.model.name
    args.model_config = config.model.config_path
    args.last_ckpt = config.training.last_ckpt
    
    # Data config
    args.data = config.data.name
    args.data_config = config.data.config_path
    args.checkpoints = str(exp_manager.get_checkpoint_dir())
    args.scale = config.training.scale
    args.disable_buffer = config.training.disable_buffer
    args.preload_hetero = config.training.preload_hetero
    args.prefetch_factor = config.training.prefetch_factor
    args.noise = config.training.noise
    
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
    
    # PyTorch Lightning specific
    args.precision = config.training.precision or '32'
    args.gradient_clip_val = config.training.gradient_clip_val or 0.0
    args.test_after_epoch = config.training.test_after_epoch
    args.test = False  # Default, can be overridden
    
    # Apply model-specific training configs (e.g., LeRet two-stage training)
    # This abstracts away model-specific config handling
    from cli.config.model_training import apply_model_configs_to_args
    apply_model_configs_to_args(args, config.training, config.model.name)
    
    # Environment variables
    args.hf_mirror = config.hf_mirror
    
    # Load model and data configs
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Merge model_config overrides if present.
    # NOTE: Legacy 'model_config' overrides were removed because 'model_config' is reserved in Pydantic v2.
    if config.model_config_overrides is not None:
        model_config = merge_configs(model_config, config.model_config_overrides)
    
    args.model_config = dotdict(model_config)
    
    # Extract model architecture parameters from model config
    # (These are used by model-specific trainers for loss computation)
    args.patch_len = model_config.get('patch_len', 16)
    args.stride = model_config.get('stride', 8)
    
    with open(args.data_config, 'r') as f:
        data_configs = yaml.safe_load(f)
    
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


def run(config: ExperimentConfig, suite_name: Optional[str] = None, suite_info: Optional[Dict[str, Any]] = None, output_dir: Optional[str] = None, init_only: bool = False, return_ids: bool = False):
    """
    Run PyTorch Lightning training experiment.
    
    This function executes a complete Lightning training pipeline based on
    the provided Pydantic configuration, with full experiment tracking.
    
    Args:
        config: ExperimentConfig instance containing experiment configuration
        suite_name: Optional suite name if experiment is part of a suite
        suite_info: Optional suite information dictionary
        output_dir: Optional base output directory (overrides config setting)
        init_only: If True, only initialize experiment structure without running training
        return_ids: If True, return experiment_id and suite_name as dict (useful for wandb sweeps)
    
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
        init_only=init_only
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
    
    # Make training faster
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')
    
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
    
    # Run training with GPU monitoring context
    with gpu_monitoring_context(args, exp_manager, log_interval_s=30.0):
        # Check for model-specific training handler
        from exp.model_specific import has_custom_trainer, get_model_trainer
        
        if has_custom_trainer(args.model, framework="lightning"):
            # Use model-specific trainer (e.g., LeRet two-stage training)
            trainer_fn = get_model_trainer(args.model, framework="lightning")
            model = trainer_fn(args, exp_manager=exp_manager)
        else:
            # Standard Lightning training for other models
            model = train_lightning_model(args, exp_manager=exp_manager)
        
        print(f'>>>>>>>training completed : {experiment_id}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    
    # Return IDs if requested (for wandb sweep integration)
    if return_ids:
        return {
            'experiment_id': exp_manager.get_experiment_id(),
            'suite_name': exp_manager.suite_name
        }

