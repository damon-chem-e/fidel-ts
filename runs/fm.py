"""
Foundation Model testing execution module.

This module provides the execution logic for Foundation Model testing,
migrated from run_fm.py.
"""

import os
import torch
import random
import numpy as np
import time
import yaml
from utils.tools import dotdict
from utils.task import ahead_task_parser
from exp.exp_fm import Experiment
from cli.utils import safe_float, safe_int, safe_bool


def config_to_args(config):
    """
    Convert config dotdict to argparse-like args object for FM testing.
    
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
    args.scale = safe_bool(config.training.get('scale', True), True)
    args.disable_buffer = safe_bool(config.training.get('disable_buffer', False), False)
    args.preload_hetero = safe_bool(config.training.get('preload_hetero', False), False)
    args.prefetch_factor = safe_int(config.training.get('prefetch_factor', 2), 2)
    args.noise = safe_float(config.training.get('noise', 0.0), 0.0)
    args.downsample = config.training.get('downsample', None)
    if args.downsample is not None:
        args.downsample = safe_int(args.downsample, None)
    
    # Forecasting task
    args.task = config.training.get('task', 'TSF')
    args.ahead = config.training.get('ahead', None)
    args.output_len = safe_int(config.training.get('output_len', 1000), 1000)
    args.input_len = safe_int(config.training.get('input_len', 1000), 1000)
    args.filtered_samples = config.training.get('filtered_samples', None)
    args.individual = safe_bool(config.training.get('individual', True), True)
    
    # Optimization
    args.num_workers = safe_int(config.training.get('num_workers', 0), 0)
    args.batch_size = safe_int(config.training.get('batch_size', 96), 96)
    args.loss = config.training.get('loss', 'mse')
    
    # GPU
    args.use_gpu = safe_bool(config.device.get('use_gpu', True), True)
    args.gpu = safe_int(config.device.get('gpu', 0), 0)
    args.use_multi_gpu = safe_bool(config.device.get('use_multi_gpu', False), False)
    args.devices = config.device.get('devices', '0,1,2,3')
    
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
    
    # Use batch size = 1 for filtered samples
    if args.filtered_samples is not None:
        args.batch_size = 1
    
    # Configure device IDs for multi-GPU training
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    return args


def run(config):
    """
    Run Foundation Model testing experiment.
    
    This function executes Foundation Model testing based on the provided configuration.
    Foundation models are pre-trained models that can be tested directly.
    
    Args:
        config: dotdict containing experiment configuration with structure:
            - model: {name, config_path}
            - data: {name, config_path}
            - training: {task, filtered_samples, individual, ...}
            - device: {use_gpu, gpu, use_multi_gpu, devices}
    
    Example:
        >>> from cli.config.loader import load_config
        >>> config = load_config("configs/experiments/fm_solar.yaml")
        >>> run(config)
    """
    # Set environment variables for HuggingFace
    if config.get('hf_mirror', False):
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # Convert config to args format
    args = config_to_args(config)
    
    # Generate experiment setting name
    current_time = time.strftime('%m-%d-%H%M', time.localtime(time.time()))
    
    if args.ahead is not None:
        setting = f'{current_time}_{args.model}_{args.data}_{args.ahead}_ahead'
    else:
        if args.filtered_samples is not None:
            setting = f'filtered_{current_time}_{args.model}_{args.data}_{args.output_len}_{args.input_len}'
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
    print('>>>>>>>start testing: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.test(setting)
    
    # Final cleanup
    torch.cuda.empty_cache()

