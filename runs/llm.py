"""
LLM experiment execution module.

This module provides the execution logic for LLM-based time series forecasting,
migrated from run_llm.py.
"""

import os
import time
import yaml
from utils.tools import dotdict
from utils.task import ahead_task_parser
from exp.exp_llm import Experiment

# SSL certificate setup for OpenAI/API calls
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()


def config_to_args(config):
    """
    Convert config dotdict to argparse-like args object for LLM experiments.
    
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
    args.scale = config.training.get('scale', False)
    args.disable_buffer = config.training.get('disable_buffer', False)
    args.filtered_samples = config.training.get('filtered_samples', None)
    args.preload_hetero = config.training.get('preload_hetero', False)
    args.noise = config.training.get('noise', 0.0)
    
    # Forecasting task
    args.ahead = config.training.get('ahead', None)
    args.output_len = config.training.get('output_len', 1000)
    args.input_len = config.training.get('input_len', 1000)
    args.sample_step = config.training.get('sample_step', 24)
    args.no_parallel = config.training.get('no_parallel', False)
    args.valisets = config.training.get('valisets', 'full')
    
    # Optimization (mostly unused for LLM inference)
    args.num_workers = config.training.get('num_workers', 0)
    args.train_epochs = config.training.get('epochs', 50)
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
    
    # LLM-specific
    args.amlt = config.get('amlt', False)
    
    # Load model and data configs
    with open(args.model_config, 'r', encoding='utf-8') as f:
        model_configs = yaml.safe_load(f)
    args.model_config = dotdict(model_configs)
    
    with open(args.data_config, 'r', encoding='utf-8') as f:
        data_configs = yaml.safe_load(f)
    args.data_config = dotdict(data_configs)
    
    # Handle ahead task
    if args.ahead is not None:
        assert args.ahead in ['day', 'week', 'month'], 'ahead task not supported, or add your own parser'
        try:
            args.output_len, args.input_len = ahead_task_parser(args.ahead, args.data_config.sampling_rate)
        except:
            raise ValueError('sampling rate not found in data config, fall back to default, input output length')
    
    return args


def run(config):
    """
    Run LLM-based time series forecasting experiment.
    
    This function executes LLM inference/testing based on the provided configuration.
    LLM experiments typically focus on inference rather than training.
    
    Args:
        config: dotdict containing experiment configuration with structure:
            - model: {name, config_path}
            - data: {name, config_path}
            - training: {filtered_samples, sample_step, valisets, ...}
            - device: {use_gpu, gpu, use_multi_gpu, devices}
    
    Example:
        >>> from cli.config.loader import load_config
        >>> config = load_config("configs/experiments/llm_solar.yaml")
        >>> run(config)
    """
    import torch
    print(torch.cuda.device_count())
    
    # Convert config to args format
    args = config_to_args(config)
    
    # Generate experiment setting name
    if args.amlt:
        current_time = 'amlt'
    else:
        current_time = time.strftime('%m-%d-%H%M', time.localtime(time.time()))
    
    # Remove the "/" "\" in model name
    _model = args.model.replace('/', '-').replace('\\', '-')
    
    if args.ahead is not None:
        setting = f'{current_time}_{_model}_{args.data}_{args.ahead}_ahead'
    else:
        setting = f'{current_time}_{_model}_{args.data}_{args.output_len}_{args.input_len}'
    
    # Check if the checkpoint path exists
    if not os.path.exists(os.path.join(args.checkpoints, setting)):
        os.makedirs(os.path.join(args.checkpoints, setting))
    
    # Initialize and run experiment
    exp = Experiment(args)
    exp.test(savepath=os.path.join(args.checkpoints, setting), valiset=args.valisets)

