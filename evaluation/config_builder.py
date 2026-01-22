"""
Configuration builder for evaluation.

This module provides functions to build evaluation configurations from experiment configs,
eliminating redundant configuration by inferring all parameters from training configs.
"""

import copy
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from utils.tools import dotdict
from cli.config.models import ExperimentConfig
from utils.experiment_config_builder import build_experiment_args


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


def _load_saved_experiment_config(experiment_dir: Path) -> Dict[str, Any]:
    """
    Load experiment_config.yaml and point model/data config paths to saved copies.

    This ensures evaluation uses the exact config files saved during training,
    even if the original config paths have changed.
    """
    experiment_config = load_checkpoint_config(experiment_dir)

    # Use saved model/data config paths when available
    resolved_config = copy.deepcopy(experiment_config)
    model_section = resolved_config.get('model') or {}
    data_section = resolved_config.get('data') or {}

    model_config_path = experiment_dir / "configs" / "model_config.yaml"
    data_config_path = experiment_dir / "configs" / "data_config.yaml"

    if model_config_path.exists():
        model_section['config_path'] = str(model_config_path)
    if data_config_path.exists():
        data_section['config_path'] = str(data_config_path)

    resolved_config['model'] = model_section
    resolved_config['data'] = data_section

    return resolved_config


def build_eval_args_from_experiment_dir(
    experiment_dir: Path,
    data_config_override: Optional[str] = None
) -> dotdict:
    """
    Build training-equivalent args from a saved experiment directory.

    This uses the saved experiment_config.yaml plus saved model/data configs
    to recreate the training runtime configuration (including tensor cache
    settings, hetero_stride, base_data_path, and llm_embedding).
    """
    experiment_config = _load_saved_experiment_config(experiment_dir)
    args = build_experiment_args(experiment_config, include_gpu=True, include_training=True)

    # Allow evaluation-time data config override (CLI), if provided
    if data_config_override:
        with open(data_config_override, 'r', encoding='utf-8') as f:
            args.data_config = dotdict(yaml.safe_load(f) or {})
        args.data_config.config_path = data_config_override
        if not args.data_config.get('name'):
            args.data_config.name = args.data

    return args


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
    
    # 2. Load saved experiment config (contains actual training parameters used)
    checkpoint_config = _load_saved_experiment_config(experiment_dir)

    # Build training-equivalent args (for consistent defaults and derived fields)
    training_args = build_experiment_args(checkpoint_config, include_gpu=True, include_training=True)
    
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
    model_name = checkpoint_config.get('model', {}).get('name', training_args.model)
    data_name = checkpoint_config.get('data', {}).get('name', training_args.data)
    batch_size = batch_size_override if batch_size_override is not None else training_args.batch_size
    
    # Device handling
    if device_override is not None:
        device = device_override
    elif config.device.use_gpu:
        device = str(config.device.gpu)
    else:
        device = "0"  # Default GPU 0
    
    # 5. Infer task type
    try:
        saved_config = ExperimentConfig.from_yaml(experiment_dir / "configs" / "experiment_config.yaml")
        task = infer_task_type(saved_config)
    except Exception:
        task = infer_task_type(config)
    
    # 6. Handle filtered_samples
    filtered_samples = checkpoint_config.get('training', {}).get('filtered_samples')
    
    # 7. Read evaluation-specific options from config if present
    # These can be specified in the experiment config under 'evaluation' key
    config_dict = checkpoint_config or {}
    evaluation_options = config_dict.get('evaluation', {})
    # Handle case where evaluation key exists but is None
    if evaluation_options is None:
        evaluation_options = {}

    # Fallback to passed config if saved config has no evaluation options
    if not evaluation_options:
        fallback_dict = config.model_dump()
        evaluation_options = fallback_dict.get('evaluation', {}) or {}
    
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
