"""
Configuration builder for evaluation.

This module provides functions to build evaluation configurations from experiment configs,
eliminating redundant configuration by inferring all parameters from training configs.
"""

import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from utils.tools import dotdict
from cli.config.models import ExperimentConfig


def load_checkpoint_config(experiment_dir: Path) -> Dict[str, Any]:
    """
    Load checkpoint configuration from experiment directory.
    
    Only supports the new format: configs/experiment_config.yaml
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        dict: Checkpoint configuration dictionary
        
    Raises:
        FileNotFoundError: If no checkpoint config found
    """
    config_path = experiment_dir / "configs" / "experiment_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No checkpoint config found in {experiment_dir}\n"
            f"  Expected: configs/experiment_config.yaml"
        )
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def infer_task_type(config: ExperimentConfig) -> str:
    """
    Infer task type from experiment config.
    
    Priority order:
    1. Model config YAML task field
    2. Experiment type (if exists as extra field)
    3. Model name inference
    
    Args:
        config: ExperimentConfig instance
        
    Returns:
        str: Task type (TSF, TGTSF, MTSF, etc.)
        
    Raises:
        ValueError: If task type cannot be determined
    """
    # 1. Try model config YAML
    try:
        model_config_path = Path(config.model.config_path)
        if not model_config_path.is_absolute():
            model_config_path = Path.cwd() / model_config_path
        if model_config_path.exists():
            with open(model_config_path, 'r', encoding='utf-8') as f:
                model_config = yaml.safe_load(f)
                if isinstance(model_config, dict) and 'task' in model_config:
                    task = model_config['task']
                    if task in ['TSF', 'TGTSF', 'MTSF', 'Reasoning']:
                        return task
    except Exception as e:
        pass
    
    # 2. Try experiment type (if exists as extra field)
    config_dict = config.model_dump()
    if 'experiment' in config_dict and isinstance(config_dict['experiment'], dict):
        exp_type = config_dict['experiment'].get('type')
        if exp_type in ['TSF', 'TGTSF', 'MTSF']:
            return exp_type
    
    # 3. Infer from model name
    model_name_lower = config.model.name.lower()
    if 'tgtsf' in model_name_lower or 'lynx' in model_name_lower:
        return 'TGTSF'
    elif 'mtsf' in model_name_lower:
        return 'MTSF'
    
    # If we can't determine task type, raise an error
    raise ValueError(
        f"Could not determine task type from config.\n"
        f"  Model: {config.model.name}\n"
        f"  Model config path: {config.model.config_path}\n"
        f"  Please specify 'task' in model config YAML or 'experiment.type' in experiment config."
    )


def build_evaluation_config_from_experiment_config(
    config: ExperimentConfig,
    resume_experiment_id: str,
    resume_suite_id: Optional[str] = None,
    output_dir: str = "./output",
    version: str = "best",
    device_override: Optional[str] = None,
    batch_size_override: Optional[int] = None,
    evaluation_overrides: Optional[Dict[str, Any]] = None
) -> dotdict:
    """
    Build evaluation configuration from experiment config.
    
    This function infers all evaluation parameters from the experiment config
    and checkpoint directory, eliminating redundant configuration.
    
    Args:
        config: ExperimentConfig from training config file
        resume_experiment_id: Experiment ID to evaluate (required)
        resume_suite_id: Suite ID if experiment is part of a suite (optional)
        output_dir: Base output directory (default: "./output")
        version: Checkpoint version - "best", "latest", or specific pattern (default: "best")
        device_override: Optional device override (overrides config.device.gpu)
        batch_size_override: Optional batch size override (overrides config.training.batch_size)
        evaluation_overrides: Optional dict of evaluation options to override (e.g., from suite config)
        
    Returns:
        dotdict: Evaluation configuration compatible with evaluation functions
        
    Raises:
        FileNotFoundError: If experiment directory or checkpoint config not found
        ValueError: If required parameters are missing
    """
    output_path = Path(output_dir).resolve()
    
    # 1. Find experiment directory
    if resume_suite_id:
        experiment_dir = output_path / resume_suite_id / resume_experiment_id
    else:
        experiment_dir = output_path / resume_experiment_id
    
    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"Experiment directory not found: {experiment_dir}\n"
            f"  Searched in: {output_path}\n"
            f"  Experiment ID: {resume_experiment_id}"
            + (f"\n  Suite ID: {resume_suite_id}" if resume_suite_id else "")
        )
    
    # 2. Load checkpoint config (contains actual training parameters used)
    checkpoint_config = load_checkpoint_config(experiment_dir)
    
    # 3. Extract input_len and output_len from checkpoint config
    # New format: experiment_config.yaml with nested training config
    training_config = checkpoint_config.get('training', {})
    input_len = training_config.get('input_len')
    output_len = training_config.get('output_len')
    
    # Validate that input_len/output_len are present
    if input_len is None or output_len is None:
        # Fallback to training config (from the experiment config file passed to CLI)
        if config.training.input_len is not None and config.training.output_len is not None:
            input_len = config.training.input_len
            output_len = config.training.output_len
        else:
            raise ValueError(
                f"Could not determine input_len/output_len from checkpoint config or training config.\n"
                f"  Experiment directory: {experiment_dir}\n"
                f"  Checked: experiment_config.yaml, training config\n"
                f"  Please ensure training completed successfully with input_len/output_len specified."
            )
    
    # 4. Extract other parameters
    model_name = config.model.name
    data_name = config.data.name
    batch_size = batch_size_override if batch_size_override is not None else config.training.batch_size
    
    # Device handling
    if device_override is not None:
        device = device_override
    elif config.device.use_gpu:
        device = str(config.device.gpu)
    else:
        device = "0"  # Default GPU 0
    
    # 5. Infer task type
    task = infer_task_type(config)
    
    # 6. Handle filtered_samples
    filtered_samples = config.training.filtered_samples
    
    # 7. Read evaluation-specific options from config if present
    # These can be specified in the experiment config under 'evaluation' key
    config_dict = config.model_dump()
    evaluation_options = config_dict.get('evaluation', {})
    # Handle case where evaluation key exists but is None
    if evaluation_options is None:
        evaluation_options = {}
    
    # Merge evaluation overrides from suite config (takes precedence)
    if evaluation_overrides is not None:
        if isinstance(evaluation_options, dict):
            evaluation_options = {**evaluation_options, **evaluation_overrides}
        else:
            evaluation_options = evaluation_overrides
    
    # 8. Build evaluation config
    eval_config = dotdict({
        'model': model_name,
        'data': data_name,
        'version': version,
        'checkpoint_base': str(experiment_dir),  # Use experiment dir as checkpoint base
        'input_len': input_len,
        'output_len': output_len,
        'batch_size': batch_size,
        'task': task,
        'device': device,
        'filtered_samples': filtered_samples,
        'channel_wise': False,
        'experiment_dir': str(experiment_dir),  # Store for convenience
    })
    
    # 9. Merge evaluation options (e.g., nan_aware_aggregation, channel_wise, etc.)
    if isinstance(evaluation_options, dict):
        for key, value in evaluation_options.items():
            if value is not None:  # Only override if value is explicitly set
                eval_config[key] = value
    
    return eval_config
