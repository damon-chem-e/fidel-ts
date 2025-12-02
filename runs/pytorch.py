"""
PyTorch training execution module.

This module provides the execution logic for standard PyTorch-based training,
migrated from run.py.
"""

import os
import torch
import random
import numpy as np
import time
import yaml
from pathlib import Path
from utils.tools import dotdict
from utils.task import ahead_task_parser
from exp.exp_universal import Experiment


def config_to_args(config):
    """
    Convert config dotdict to argparse-like args object.
    
    This function bridges the gap between the new config-based system
    and the existing Experiment class that expects argparse args.
    
    Args:
        config: dotdict containing experiment configuration
    
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
    args.checkpoints = config.training.get('checkpoints', './checkpoints/')
    args.scale = config.training.get('scale', True)
    args.disable_buffer = config.training.get('disable_buffer', False)
    args.preload_hetero = config.training.get('preload_hetero', False)
    args.prefetch_factor = config.training.get('prefetch_factor', 2)
    args.noise = config.training.get('noise', 0.0)
    args.downsample = config.training.get('downsample', None)
    
    # Forecasting task
    args.ahead = config.training.get('ahead', None)
    args.output_len = config.training.get('output_len', 1000)
    args.input_len = config.training.get('input_len', 1000)
    
    # Optimization
    args.num_workers = config.training.get('num_workers', 0)
    args.train_epochs = config.training.get('epochs', 20)
    args.batch_size = config.training.get('batch_size', 96)
    args.patience = config.training.get('patience', 3)
    args.learning_rate = config.training.get('learning_rate', 5e-4)
    args.loss = config.training.get('loss', 'mse')
    args.lradj = config.training.get('lradj', 'type3')
    
    # GPU
    args.use_gpu = config.device.get('use_gpu', True)
    args.gpu = config.device.get('gpu', 0)
    args.use_multi_gpu = config.device.get('use_multi_gpu', False)
    args.devices = config.device.get('devices', '0,1,2,3')
    
    # Environment variables
    args.hf_mirror = config.get('hf_mirror', False)
    args.hf_offline = config.get('hf_offline', False)
    
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
    Run PyTorch training experiment.
    
    This function executes a complete PyTorch training pipeline based on
    the provided configuration.
    
    Args:
        config: dotdict containing experiment configuration with structure:
            - model: {name, config_path}
            - data: {name, config_path}
            - training: {epochs, batch_size, learning_rate, ...}
            - device: {use_gpu, gpu, use_multi_gpu, devices}
    
    Example:
        >>> from cli.config.loader import load_config
        >>> config = load_config("configs/experiments/dlinear_solar.yaml")
        >>> run(config)
    """
    # Set environment variables for HuggingFace
    if config.get('hf_mirror', False):
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    if config.get('hf_offline', False):
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
    
    # Convert config to args format
    args = config_to_args(config)
    
    # Generate experiment setting name
    current_time = time.strftime('%m-%d-%H%M', time.localtime(time.time()))
    
    if args.ahead is not None:
        setting = f'{current_time}_{args.model}_{args.data}_{args.ahead}_ahead'
    else:
        setting = f'{current_time}_{args.model}_{args.data}_{args.output_len}_{args.input_len}'
    
    # Set seeds for reproducibility
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    print('Args in experiment:')
    print(args)
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Initialize and run experiment
    exp = Experiment(args)
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    # Final cleanup
    torch.cuda.empty_cache()

