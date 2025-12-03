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
from utils.tools import dotdict
from utils.task import ahead_task_parser
from exp.exp_lightning import train_lightning_model
from cli.config.models import ExperimentConfig
from exp.manager import ExperimentManager


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
    
    # Environment variables
    args.hf_mirror = config.hf_mirror
    
    # Load model and data configs
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    args.model_config = dotdict(model_config)
    
    with open(args.data_config, 'r') as f:
        data_configs = yaml.safe_load(f)
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


def run(config: ExperimentConfig):
    """
    Run PyTorch Lightning training experiment.
    
    This function executes a complete Lightning training pipeline based on
    the provided Pydantic configuration, with full experiment tracking.
    
    Args:
        config: ExperimentConfig instance containing experiment configuration
    
    Example:
        >>> from cli.config.loader import load_config
        >>> config = load_config("configs/experiments/dlinear_solar.yaml")
        >>> run(config)
    """
    # Initialize experiment manager
    output_dir = config.training.checkpoints or "./outputs"
    exp_manager = ExperimentManager(
        config=config,
        output_dir=output_dir,
        experiment_name=config.experiment_name,
        job_id=config.job_id,
        job_name=config.job_name
    )
    
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
    
    # Train model
    model = train_lightning_model(args, experiment_id)
    print(f'>>>>>>>training completed : {experiment_id}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    
    # Final cleanup
    torch.cuda.empty_cache()

