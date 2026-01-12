"""
Suite execution engine for running experiment suites defined in YAML configs.

This module provides functionality to execute collections of related experiments
defined in suite configuration files, replacing the need for bash scripts.
"""

import yaml
import json
import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from cli.config.loader import resolve_config_path
from cli.config.models import ExperimentConfig
from utils.gpu_monitor import GpuMonitor
from utils.config_utils import merge_configs
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


def _check_experiment_complete(output_dir: Path, suite_name: str, resume_experiment_id: str, total_epochs: int) -> bool:
    """
    Check if an experiment is already complete by reading job_history.json.
    
    Args:
        output_dir: Base output directory
        suite_name: Suite directory name (with timestamp if resuming)
        resume_experiment_id: Experiment ID to check
        total_epochs: Total epochs for the experiment
    
    Returns:
        True if experiment is complete, False otherwise
    """
    experiment_dir = output_dir / suite_name / resume_experiment_id
    job_history_path = experiment_dir / "job_history.json"
    
    if not job_history_path.exists():
        return False
    
    try:
        with open(job_history_path, 'r', encoding='utf-8') as f:
            job_history = json.load(f)
        
        # Check if training is complete
        # Training is complete if last_epoch >= total_epochs
        completed_jobs = [job for job in job_history.get("jobs", []) if job.get("status") in ["completed", "timeout"]]
        if completed_jobs:
            last_job = completed_jobs[-1]
            last_epoch = last_job.get("end_epoch", job_history.get("current_epoch", 0))
            total_epochs_from_history = job_history.get("total_epochs", total_epochs)
            if last_epoch >= total_epochs_from_history:
                return True
        
        return False
    except (json.JSONDecodeError, IOError, KeyError):
        # If we can't read/parse the file, assume not complete
        return False


class SuiteExecutor:
    """Execute experiment suites defined in YAML configs."""
    
    def __init__(self, 
                 suite_config: Dict[str, Any], 
                 log_dir: Optional[str] = None, 
                 output_dir: Optional[str] = None, 
                 init_only: bool = False, 
                 return_ids: bool = False, 
                 sweep: bool = False,
                 force_rerun: bool = False):
        """
        Initialize suite executor.
        
        Args:
            suite_config: Suite configuration dictionary
            log_dir: Directory for suite execution logs (optional)
            output_dir: Base directory for experiment outputs (optional, defaults to ./output)
            init_only: If True, only initialize experiment structures without running them
            return_ids: If True, track and return experiment IDs (useful for wandb sweeps)
            sweep: If True, running in wandb sweep context (pass to ExperimentManager)
            force_rerun: If True, re-run experiments even if they are already complete
        """
        self.suite_config = suite_config
        self.init_only = init_only
        self.return_ids = return_ids
        self.sweep = sweep
        self.force_rerun = force_rerun
        # Track experiment IDs when return_ids is enabled
        self.experiment_ids: Dict[str, str] = {}
        self.suite_info = suite_config.get('suite', {})
        self.execution_config = self.suite_info.get('execution', {})
        self.log_dir = log_dir or self.execution_config.get('log_dir', './logs/suites')
        self.output_dir = Path(output_dir or self.execution_config.get('output_dir', './output'))
        
        # Create log directory
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Set up suite-specific logging
        suite_name = self.suite_info.get('name', 'unknown')
        
        # Check for suite-level resume configuration
        # If resuming, all experiments in the suite must be resuming from the same suite
        resume_suite_id = self.suite_info.get('resume_suite_id')
        
        if resume_suite_id:
            # Resuming - use existing suite directory
            self.suite_name_base = suite_name
            # Extract timestamp from resume_suite_id (format: suite_name_YYYYMMDD_HHMMSS)
            # The resume_suite_id should be the full suite directory name
            self.suite_name = resume_suite_id
            # Try to extract timestamp if possible, otherwise use current time for logging
            if '_' in resume_suite_id:
                parts = resume_suite_id.rsplit('_', 1)
                if len(parts) == 2 and len(parts[1]) == 15:  # YYYYMMDD_HHMMSS format
                    self.suite_timestamp = parts[1]
                else:
                    self.suite_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            else:
                self.suite_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.suite_dir = self.output_dir / self.suite_name
            # Don't create directory if it doesn't exist - it should already exist for resume
            if not self.suite_dir.exists():
                raise ValueError(
                    f"Cannot resume suite: suite directory '{self.suite_dir}' does not exist. "
                    f"Ensure resume_suite_id '{resume_suite_id}' is correct."
                )
            logger.info(f"Resuming suite: {self.suite_name}")
        else:
            # New suite - create new directory with timestamp
            suite_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.suite_name_base = suite_name
            self.suite_timestamp = suite_timestamp
            self.suite_name = f"{suite_name}_{suite_timestamp}"
            self.suite_dir = self.output_dir / self.suite_name
            self.suite_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized new suite: {self.suite_name}")
        
        # Set up logging (use current timestamp for log file even when resuming)
        log_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = Path(self.log_dir) / f"{suite_name}_{log_timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Save suite-level metadata (only for new suites, or update existing for resume)
        if not resume_suite_id:
            self._save_suite_metadata()
        else:
            # When resuming, don't overwrite existing metadata, but log that we're resuming
            logger.info(f"Resuming existing suite (metadata preserved): {self.suite_name}")
    
    def _save_suite_metadata(self) -> None:
        """Save suite-level metadata to suite directory."""
        suite_metadata = {
            "suite_name": self.suite_name_base,
            "suite_name_with_timestamp": self.suite_name,
            "timestamp": datetime.now().isoformat(),
            "suite_timestamp": self.suite_timestamp,
            "description": self.suite_info.get('description', ''),
            "tags": self.suite_info.get('tags', []),
            "execution": self.execution_config,
            "experiments": [
                {
                    "name": exp.get('name', 'unknown'),
                    "description": exp.get('description', ''),
                    "enabled": exp.get('enabled', True),
                    "template": exp.get('template', '')
                }
                for exp in self.suite_info.get('experiments', [])
            ]
        }
        
        # Save suite metadata JSON
        metadata_path = self.suite_dir / "suite_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(suite_metadata, f, indent=2, default=str)
        
        # Save suite config file
        config_path = self.suite_dir / "suite_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.suite_config, f, default_flow_style=False, sort_keys=False)
    
    def execute(self) -> Optional[Dict[str, Any]]:
        """
        Execute all experiments in the suite.
        
        Returns:
            If return_ids=True, returns dict with:
                - 'suite_id': Suite name with timestamp
                - 'experiment_ids': Dictionary mapping experiment names to experiment IDs
            Otherwise returns None.
        """
        experiments = [
            exp for exp in self.suite_info.get('experiments', [])
            if exp.get('enabled', True)
        ]
        
        # Validate resume configuration: if suite is resuming, all experiments must have resume_experiment_id
        if self.suite_info.get('resume_suite_id'):
            missing_resume_ids = []
            for exp in experiments:
                overrides = exp.get('overrides', {})
                if 'resume_experiment_id' not in overrides:
                    missing_resume_ids.append(exp.get('name', 'unknown'))
            
            if missing_resume_ids:
                raise ValueError(
                    f"Suite is resuming (resume_suite_id: {self.suite_info.get('resume_suite_id')}), "
                    f"but the following experiments are missing 'resume_experiment_id' in their overrides: "
                    f"{', '.join(missing_resume_ids)}. "
                    f"When resuming a suite, all enabled experiments must specify their resume_experiment_id."
                )
        
        logger.info(f"Starting suite execution: {len(experiments)} experiments")
        
        # Initialize suite-level GPU monitor if GPU is available
        suite_gpu_monitor = None
        if torch.cuda.is_available():
            suite_csv_path = str(Path(self.log_dir) / "suite_gpu_telemetry.csv")
            suite_gpu_monitor = GpuMonitor(device_index=0, out_csv=suite_csv_path)
            suite_gpu_monitor.start()
            logger.info("Started suite-level GPU monitoring")
        
        success_count = 0
        error_count = 0
        
        try:
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
        finally:
            # Stop suite-level GPU monitoring
            if suite_gpu_monitor:
                suite_gpu_monitor.stop()
                summary = suite_gpu_monitor.summary()
                if summary:
                    logger.info(f"Suite GPU monitoring summary: {summary}")
        
        logger.info(f"Suite execution completed: {success_count} successful, {error_count} errors")
        
        # Return suite and experiment IDs if requested
        if self.return_ids:
            return {
                'suite_id': self.suite_name,
                'experiment_ids': self.experiment_ids.copy()
            }
        return None
    
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
        
        # If suite is resuming, inject resume_suite_id into experiment overrides
        # and validate that resume_experiment_id is set for this experiment
        resume_experiment_id = None
        if self.suite_info.get('resume_suite_id'):
            if 'resume_experiment_id' not in overrides:
                raise ValueError(
                    f"Suite is resuming (resume_suite_id: {self.suite_info.get('resume_suite_id')}), "
                    f"but experiment '{exp_name}' does not have 'resume_experiment_id' set in its overrides. "
                    f"When resuming a suite, all experiments must specify their resume_experiment_id."
                )
            resume_experiment_id = overrides.get('resume_experiment_id')
            # Inject resume_suite_id from suite level into experiment overrides
            overrides['resume_suite_id'] = self.suite_info.get('resume_suite_id')
        
        # Load template (needed for both completion check and execution)
        template = load_template(template_path)
        
        # Merge template with overrides to get final config
        merged_config = merge_configs(template, overrides)
        
        # Check if experiment is already complete (unless force_rerun is True)
        if not self.force_rerun and resume_experiment_id:
            total_epochs = merged_config.get('training', {}).get('epochs', 3)
            is_complete = _check_experiment_complete(
                self.output_dir,
                self.suite_name,
                resume_experiment_id,
                total_epochs
            )
            if is_complete:
                logger.info(f"Skipping experiment '{exp_name}': training already completed (epoch {total_epochs}/{total_epochs})")
                return
        
        # Use merged_config as final_config (already merged)
        final_config = merged_config
        
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
                
                experiment_id = self._run_single_experiment(config_copy, exp_name_with_len)
                if experiment_id:
                    self.experiment_ids[exp_name_with_len] = experiment_id
        else:
            experiment_id = self._run_single_experiment(final_config, exp_name)
            if experiment_id:
                self.experiment_ids[exp_name] = experiment_id
    
    def _run_single_experiment(self, config: Dict[str, Any], experiment_name: str) -> Optional[str]:
        """
        Run a single experiment configuration.
        
        Args:
            config: Complete experiment configuration dictionary
            experiment_name: Name of the experiment
        
        Returns:
            experiment_id if return_ids is enabled, None otherwise
        """
        exp_type = config.get('experiment', {}).get('type', 'pytorch')
        
        logger.info(f"Running experiment '{experiment_name}' with type '{exp_type}'")
        
        # Prepare suite information to pass to experiments
        # Use the timestamped suite name so experiments are saved in the correct directory
        suite_info = {
            "name": self.suite_name_base,
            "name_with_timestamp": self.suite_name,
            "timestamp": self.suite_timestamp,
            "description": self.suite_info.get('description', ''),
            "tags": self.suite_info.get('tags', [])
        }
        
        # Execute based on experiment type
        if exp_type == 'evaluation':
            if self.init_only:
                logger.info(f"Skipping evaluation experiment '{experiment_name}' during initialization.")
                return

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
            # Pass the timestamped suite name so experiments are saved in the correct directory
            result = None
            if exp_type == 'pytorch':
                result = run_pytorch(experiment_config, suite_name=self.suite_name, suite_info=suite_info, output_dir=str(self.output_dir), init_only=self.init_only, return_ids=self.return_ids, sweep=self.sweep)
            elif exp_type == 'lightning':
                result = run_lightning(experiment_config, suite_name=self.suite_name, suite_info=suite_info, output_dir=str(self.output_dir), init_only=self.init_only, return_ids=self.return_ids, sweep=self.sweep)
            elif exp_type == 'llm':
                result = run_llm(experiment_config, suite_name=self.suite_name, suite_info=suite_info, output_dir=str(self.output_dir), init_only=self.init_only, return_ids=self.return_ids, sweep=self.sweep)
            elif exp_type == 'fm':
                # FM experiments may have task specified at experiment level
                if 'task' in config.get('experiment', {}):
                    # Add task to config dict before creating ExperimentConfig
                    config['task'] = config['experiment']['task']
                    # Recreate ExperimentConfig with task field
                    experiment_config = ExperimentConfig(**config)
                    experiment_config.experiment_name = experiment_name
                result = run_fm(experiment_config, suite_name=self.suite_name, suite_info=suite_info, output_dir=str(self.output_dir), init_only=self.init_only, return_ids=self.return_ids, sweep=self.sweep)
            else:
                raise ValueError(f"Unknown experiment type: {exp_type}")
            
            # Extract experiment_id from result if available
            if self.return_ids and result:
                return result.get('experiment_id')
            return None


def load_suite_config(suite_config_path: str) -> Dict[str, Any]:
    """
    Load a suite configuration file.
    
    Args:
        suite_config_path: Path to suite YAML config file
    
    Returns:
        Suite configuration dictionary
    
    Raises:
        ValueError: If config file is empty or invalid
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    try:
        resolved_path = resolve_config_path(suite_config_path)
        with open(resolved_path, 'r', encoding='utf-8') as f:
            suite_config = yaml.safe_load(f)
        
        if suite_config is None:
            raise ValueError(f"Suite config file is empty: {suite_config_path}")
        
        if not isinstance(suite_config, dict):
            raise ValueError(f"Suite config must be a dictionary, got {type(suite_config).__name__}")
        
        return suite_config
    except FileNotFoundError:
        raise
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML in suite config: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading suite config: {str(e)}")


def execute_suite(suite_config_path: str, init_only: bool = False) -> None:
    """
    Execute an experiment suite from a config file.
    
    Args:
        suite_config_path: Path to suite YAML config file
        init_only: If True, only initialize experiment structures without running them
    """
    suite_config = load_suite_config(suite_config_path)
    executor = SuiteExecutor(suite_config, init_only=init_only)
    executor.execute()

