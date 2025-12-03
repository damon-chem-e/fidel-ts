"""
Suite execution engine for running experiment suites defined in YAML configs.

This module provides functionality to execute collections of related experiments
defined in suite configuration files, replacing the need for bash scripts.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from cli.config.loader import load_config, resolve_config_path
from cli.config.models import ExperimentConfig
from runs.pytorch import run as run_pytorch
from runs.lightning import run as run_lightning
from runs.llm import run as run_llm
from runs.fm import run as run_fm
from evaluation.standard import evaluate as evaluate_standard
from evaluation.lightning import evaluate as evaluate_lightning


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_template(template_path: str, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load a template configuration file.
    
    Args:
        template_path: Path to template YAML file
        base_dir: Base directory for resolving relative paths
    
    Returns:
        Dictionary containing template configuration
    """
    resolved_path = resolve_config_path(template_path, base_dir)
    with open(resolved_path, 'r', encoding='utf-8') as f:
        template = yaml.safe_load(f)
    
    if template is None:
        return {}
    
    return template


def merge_configs(template: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge template configuration with overrides.
    
    This function performs a deep merge, where overrides take precedence
    over template values. Nested dictionaries are merged recursively.
    
    Args:
        template: Base template configuration
        overrides: Configuration overrides
    
    Returns:
        Merged configuration dictionary
    """
    result = template.copy()
    
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Override with new value
            result[key] = value
    
    return result


def substitute_placeholders(config: Dict[str, Any], experiment_name: str, 
                            model_name: Optional[str] = None,
                            model_config_path: Optional[str] = None,
                            data_name: Optional[str] = None,
                            data_config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Substitute placeholder values in configuration.
    
    Replaces placeholders like "${experiment_name}" with actual values.
    
    Args:
        config: Configuration dictionary (may contain placeholders)
        experiment_name: Experiment name to substitute
        model_name: Model name to substitute (optional)
        model_config_path: Model config path to substitute (optional)
        data_name: Data name to substitute (optional)
        data_config_path: Data config path to substitute (optional)
    
    Returns:
        Configuration dictionary with placeholders replaced
    """
    def substitute_value(value: Any) -> Any:
        """Recursively substitute placeholders in a value."""
        if isinstance(value, str):
            # Replace placeholders
            value = value.replace("${experiment_name}", experiment_name)
            if model_name:
                value = value.replace("${model_name}", model_name)
            if model_config_path:
                value = value.replace("${model_config_path}", model_config_path)
            if data_name:
                value = value.replace("${data_name}", data_name)
            if data_config_path:
                value = value.replace("${data_config_path}", data_config_path)
            return value
        elif isinstance(value, dict):
            return {k: substitute_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [substitute_value(item) for item in value]
        else:
            return value
    
    return substitute_value(config)


class SuiteExecutor:
    """Execute experiment suites defined in YAML configs."""
    
    def __init__(self, suite_config: Dict[str, Any], log_dir: Optional[str] = None):
        """
        Initialize suite executor.
        
        Args:
            suite_config: Suite configuration dictionary
            log_dir: Directory for suite execution logs (optional)
        """
        self.suite_config = suite_config
        self.suite_info = suite_config.get('suite', {})
        self.execution_config = self.suite_info.get('execution', {})
        self.log_dir = log_dir or self.execution_config.get('log_dir', './logs/suites')
        
        # Create log directory
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Set up suite-specific logging
        suite_name = self.suite_info.get('name', 'unknown')
        log_file = Path(self.log_dir) / f"{suite_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info(f"Initialized suite executor for: {suite_name}")
    
    def execute(self) -> None:
        """Execute all experiments in the suite."""
        experiments = [
            exp for exp in self.suite_info.get('experiments', [])
            if exp.get('enabled', True)
        ]
        
        logger.info(f"Starting suite execution: {len(experiments)} experiments")
        
        success_count = 0
        error_count = 0
        
        for exp_config in experiments:
            exp_name = exp_config.get('name', 'unknown')
            try:
                logger.info(f"Executing experiment: {exp_name}")
                self._execute_experiment(exp_config)
                success_count += 1
                logger.info(f"Successfully completed experiment: {exp_name}")
            except Exception as e:
                error_count += 1
                error_msg = f"Error executing experiment '{exp_name}': {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                if not self.execution_config.get('continue_on_error', True):
                    logger.error("Stopping suite execution due to error")
                    raise
                else:
                    logger.warning(f"Continuing suite execution despite error in '{exp_name}'")
        
        logger.info(f"Suite execution completed: {success_count} successful, {error_count} errors")
    
    def _execute_experiment(self, exp_config: Dict[str, Any]) -> None:
        """
        Execute a single experiment from the suite.
        
        Args:
            exp_config: Experiment configuration dictionary
        """
        exp_name = exp_config.get('name', 'unknown')
        template_path = exp_config.get('template')
        overrides = exp_config.get('overrides', {})
        
        if not template_path:
            raise ValueError(f"Template path not specified for experiment '{exp_name}'")
        
        # Load template
        template = load_template(template_path)
        
        # Merge template with overrides
        final_config = merge_configs(template, overrides)
        
        # Extract values for placeholder substitution from merged config
        # This ensures we get the actual values after merging
        model_name = final_config.get('model', {}).get('name', '')
        model_config_path = final_config.get('model', {}).get('config_path', '')
        data_name = final_config.get('data', {}).get('name', '')
        data_config_path = final_config.get('data', {}).get('config_path', '')
        
        # Substitute placeholders (handles nested structures recursively)
        final_config = substitute_placeholders(
            final_config,
            experiment_name=exp_name,
            model_name=model_name,
            model_config_path=model_config_path,
            data_name=data_name,
            data_config_path=data_config_path
        )
        
        # Handle multiple output_lens if specified
        training_overrides = overrides.get('training', {})
        if 'output_lens' in training_overrides:
            output_lens = training_overrides['output_lens']
            for output_len in output_lens:
                # Create a copy of config with this output_len
                config_copy = final_config.copy()
                config_copy['training'] = final_config['training'].copy()
                config_copy['training']['output_len'] = output_len
                
                # Update experiment name to include output_len
                exp_name_with_len = f"{exp_name}_output{output_len}"
                config_copy['experiment']['name'] = exp_name_with_len
                
                self._run_single_experiment(config_copy, exp_name_with_len)
        else:
            self._run_single_experiment(final_config, exp_name)
    
    def _run_single_experiment(self, config: Dict[str, Any], experiment_name: str) -> None:
        """
        Run a single experiment configuration.
        
        Args:
            config: Complete experiment configuration dictionary
            experiment_name: Name of the experiment
        """
        exp_type = config.get('experiment', {}).get('type', 'pytorch')
        
        logger.info(f"Running experiment '{experiment_name}' with type '{exp_type}'")
        
        # Execute based on experiment type
        if exp_type == 'evaluation':
            # Evaluation uses a different config structure (dotdict with evaluation section)
            from utils.tools import dotdict
            # Convert config dict to dotdict format expected by evaluate()
            eval_config = dotdict(config)
            # Ensure evaluation section exists
            if 'evaluation' not in eval_config:
                raise ValueError("Evaluation config must have 'evaluation' section")
            # Add data_config override if specified in evaluation section
            if 'data_config' in eval_config.evaluation and eval_config.evaluation.data_config:
                eval_config.data_config = eval_config.evaluation.data_config
            # Determine which evaluation function to use based on model type
            # Lightning models use evaluate_lightning, others use evaluate_standard
            eval_model = eval_config.evaluation.get('model', '').lower()
            if eval_model in ['patchtst', 'itransformer', 'tgtsf']:
                evaluate_lightning(eval_config)
            else:
                evaluate_standard(eval_config)
        else:
            # Convert dict to ExperimentConfig for training experiments
            try:
                experiment_config = ExperimentConfig(**config)
            except Exception as e:
                raise ValueError(f"Invalid experiment configuration: {str(e)}")
            
            # Set experiment name
            experiment_config.experiment_name = experiment_name
            
            # Execute based on experiment type
            if exp_type == 'pytorch':
                run_pytorch(experiment_config)
            elif exp_type == 'lightning':
                run_lightning(experiment_config)
            elif exp_type == 'llm':
                run_llm(experiment_config)
            elif exp_type == 'fm':
                # FM experiments may have task specified at experiment level
                if 'task' in config.get('experiment', {}):
                    # Add task to config dict before creating ExperimentConfig
                    config['task'] = config['experiment']['task']
                    # Recreate ExperimentConfig with task field
                    experiment_config = ExperimentConfig(**config)
                    experiment_config.experiment_name = experiment_name
                run_fm(experiment_config)
            else:
                raise ValueError(f"Unknown experiment type: {exp_type}")


def load_suite_config(suite_config_path: str) -> Dict[str, Any]:
    """
    Load a suite configuration file.
    
    Args:
        suite_config_path: Path to suite YAML config file
    
    Returns:
        Suite configuration dictionary
    """
    resolved_path = resolve_config_path(suite_config_path)
    with open(resolved_path, 'r', encoding='utf-8') as f:
        suite_config = yaml.safe_load(f)
    
    if suite_config is None:
        raise ValueError(f"Suite config file is empty: {suite_config_path}")
    
    return suite_config


def execute_suite(suite_config_path: str) -> None:
    """
    Execute an experiment suite from a config file.
    
    Args:
        suite_config_path: Path to suite YAML config file
    """
    suite_config = load_suite_config(suite_config_path)
    executor = SuiteExecutor(suite_config)
    executor.execute()

