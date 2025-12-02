"""
PyTorch Lightning training execution module.

This module provides the execution logic for PyTorch Lightning-based training,
migrated from run_lightning.py.
"""

import os
import torch
import random
import numpy as np
import time
import yaml
from utils.tools import dotdict
from utils.task import ahead_task_parser
from exp.exp_lightning import train_lightning_model


def config_to_args(config):
    """
    Convert config dotdict to argparse-like args object for Lightning training.
    
    Args:
        config: dotdict containing experiment configuration
    
    Returns:
        dotdict object compatible with train_lightning_model function
    """
    args = dotdict()
    
    # Model config
    args.model = config.model.name
    args.model_config = config.model.config_path
    args.last_ckpt = config.training.get('last_ckpt', None)
    
    # Data config
    args.data = config.data.name
    args.data_config = config.data.config_path
    args.checkpoints = config.training.get('checkpoints', './checkpoints/')
    args.scale = config.training.get('scale', True)
    args.disable_buffer = config.training.get('disable_buffer', False)
    args.preload_hetero = config.training.get('preload_hetero', False)
    args.prefetch_factor = config.training.get('prefetch_factor', 2)
    args.noise = config.training.get('noise', 0.0)
    
    # Forecasting task
    args.ahead = config.training.get('ahead', None)
    args.output_len = config.training.get('output_len', 1000)
    args.input_len = config.training.get('input_len', 1000)
    
    # Optimization
    args.num_workers = config.training.get('num_workers', 4)
    args.train_epochs = config.training.get('epochs', 30)
    args.batch_size = config.training.get('batch_size', 96)
    args.patience = config.training.get('patience', 3)
    args.learning_rate = config.training.get('learning_rate', 5e-4)
    args.loss = config.training.get('loss', 'mse')
    args.lradj = config.training.get('lradj', 'type3')
    
    # GPU
    args.use_gpu = config.device.get('use_gpu', True)
    args.gpu = config.device.get('gpu', 0)
    args.use_multi_gpu = config.device.get('use_multi_gpu', True)
    args.devices = config.device.get('devices', '3')
    
    # PyTorch Lightning specific
    args.precision = config.training.get('precision', '32')
    args.gradient_clip_val = config.training.get('gradient_clip_val', 0.0)
    args.test_after_epoch = config.training.get('test_after_epoch', False)
    args.test = config.training.get('test', False)
    
    # Environment variables
    args.hf_mirror = config.get('hf_mirror', False)
    
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


def run(config):
    """
    Run PyTorch Lightning training experiment.
    
    This function executes a complete Lightning training pipeline based on
    the provided configuration.
    
    Args:
        config: dotdict containing experiment configuration with structure:
            - model: {name, config_path}
            - data: {name, config_path}
            - training: {epochs, batch_size, learning_rate, precision, ...}
            - device: {use_gpu, gpu, use_multi_gpu, devices}
    
    Example:
        >>> from cli.config.loader import load_config
        >>> config = load_config("configs/experiments/dlinear_solar.yaml")
        >>> run(config)
    """
    # Set environment variables for HuggingFace
    if config.get('hf_mirror', False):
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # Make training faster
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')
    
    # Convert config to args format
    args = config_to_args(config)
    
    # Generate experiment setting name
    current_time = 'lightning_' + time.strftime("%m-%d-%H%M", time.localtime())
    
    if args.ahead is not None:
        setting = f'{current_time}_{args.model}_{args.data}_{args.ahead}_ahead_pl'
    else:
        setting = f'{current_time}_{args.model}_{args.data}_{args.output_len}_{args.input_len}_pl'
    
    # Set seeds for reproducibility
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    print('Args in experiment:')
    print(args)
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Train model
    model = train_lightning_model(args, setting)
    print(f'>>>>>>>training completed : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    
    # Final cleanup
    torch.cuda.empty_cache()

