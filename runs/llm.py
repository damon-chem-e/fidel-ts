"""
LLM experiment execution module.

This module provides the execution logic for LLM-based time series forecasting,
migrated from run_llm.py. Uses Pydantic configs and ExperimentManager.
"""

import os
import yaml
import torch
from utils.tools import dotdict
from utils.task import ahead_task_parser
from utils.gpu_monitor import gpu_monitoring_context
from exp.exp_llm import Experiment
from cli.config.models import ExperimentConfig
from exp.manager import ExperimentManager

# SSL certificate setup for OpenAI/API calls
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()


def config_to_args(config: ExperimentConfig, exp_manager: ExperimentManager):
    """
    Convert ExperimentConfig to argparse-like args object for LLM experiments.
    
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
    args.filtered_samples = config.training.filtered_samples
    args.preload_hetero = config.training.preload_hetero
    args.noise = config.training.noise
    
    # Forecasting task
    args.ahead = config.training.ahead
    args.output_len = config.training.output_len or 1000
    args.input_len = config.training.input_len or 1000
    if args.input_len != 'ntp':
        args.input_len = int(args.input_len) if isinstance(args.input_len, (int, str)) and str(args.input_len).isdigit() else args.input_len
    args.sample_step = config.training.sample_step
    args.no_parallel = config.training.no_parallel
    args.valisets = config.training.valisets
    
    # Optimization (mostly unused for LLM inference)
    args.num_workers = config.training.num_workers
    args.train_epochs = config.training.epochs
    args.batch_size = config.training.batch_size
    args.patience = config.training.patience
    args.learning_rate = config.training.learning_rate
    args.loss = config.training.loss
    args.lradj = config.training.lradj
    args.track_per_sample = config.training.track_per_sample
    
    # GPU
    args.use_gpu = config.device.use_gpu
    args.gpu = config.device.gpu
    args.use_multi_gpu = config.device.use_multi_gpu
    args.devices = config.device.devices
    
    # LLM-specific (from extra fields)
    args.amlt = config.model_dump().get('amlt', False)
    
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


def run(config: ExperimentConfig):
    """
    Run LLM-based time series forecasting experiment.
    
    This function executes LLM inference/testing based on the provided Pydantic
    configuration, with full experiment tracking.
    
    Args:
        config: ExperimentConfig instance containing experiment configuration
    
    Example:
        >>> from cli.config.loader import load_config
        >>> config = load_config("configs/experiments/llm_solar.yaml")
        >>> run(config)
    """
    import torch
    print(torch.cuda.device_count())
    
    # Initialize experiment manager
    output_dir = config.training.experiment_output or "./output"
    exp_manager = ExperimentManager(
        config=config,
        output_dir=output_dir,
        experiment_name=config.experiment_name,
        job_id=config.job_id,
        job_name=config.job_name
    )
    
    # Convert config to args format
    args = config_to_args(config, exp_manager)
    
    # Use experiment ID instead of setting string
    experiment_id = exp_manager.get_experiment_id()
    
    # Remove the "/" "\" in model name for path safety
    _model = args.model.replace('/', '-').replace('\\', '-')
    
    # Run experiment with GPU monitoring context
    with gpu_monitoring_context(args, exp_manager, log_interval_s=30.0):
        # Initialize and run experiment
        exp = Experiment(args)
        exp.test(savepath=str(exp_manager.get_checkpoint_dir()), valiset=args.valisets)

