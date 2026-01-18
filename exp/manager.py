"""
Experiment Manager for comprehensive experiment tracking.

This module provides the ExperimentManager class that handles:
- Experiment directory creation and management
- Metadata capture (git hash, timestamp, configs)
- Local metrics logging
- WandB integration for cloud-based experiment tracking
- Complete experiment reproducibility
- Sweep integration for hyperparameter optimization (optional)

Sweep Integration
-----------------
The ExperimentManager provides completion signals to the sweep system:

1. Training code can call set_completion_reason() when training ends
   for reasons other than completing all epochs (e.g., early stopping)

2. ExperimentManager uses _detect_training_completed() to determine if
   training finished successfully (all epochs, early stopping, etc.)

3. On end_experiment(), the manager updates wandb.summary with completion
   status (for visibility, but NOT authoritative for resumption)

Note: The SweepManager (see exp/sweep_manager.py) owns the sweep registry
and reads completion status from job_history.json. ExperimentManager does
not directly update the registry - this ensures single ownership of state.

Completion Detection
--------------------
The ExperimentManager detects training completion by checking:
1. current_epoch >= total_epochs in job_history
2. Any explicitly set completion reason (early stopping, etc.)

Training code should call set_completion_reason() when training ends
for reasons other than completing all epochs (e.g., early stopping).

Usage:
    # Basic experiment (no sweep)
    manager = ExperimentManager(config, output_dir="./output")
    # ... training ...
    manager.end_experiment(final_metrics)

    # With sweep integration (SweepManager reads job_history.json)
    manager = ExperimentManager(config, output_dir="./output")
    # ... training ...
    manager.set_completion_reason("early_stopping")  # if applicable
    manager.end_experiment(final_metrics)
"""

import os
import json
import hashlib
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import subprocess
import yaml

from rich.console import Console
from rich.logging import RichHandler

from cli.config.models import ExperimentConfig

# Import sweep registry for completion tracking (optional - may not be in sweep context)
try:
    from exp.sweep_registry import SweepRegistry, get_location, CompletionReason
    SWEEP_REGISTRY_AVAILABLE = True
except ImportError:
    SWEEP_REGISTRY_AVAILABLE = False
    SweepRegistry = None
    get_location = None
    CompletionReason = None


class ExperimentManager:
    """
    Manages experiment tracking, metadata capture, and output organization.
    
    This class provides a centralized way to:
    - Create and manage experiment directories
    - Capture complete experiment metadata
    - Log metrics and checkpoints (local and wandb)
    - Ensure reproducibility
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str = "./output",
        experiment_name: Optional[str] = None,
        job_id: Optional[str] = None,
        job_name: Optional[str] = None,
        suite_name: Optional[str] = None,
        suite_info: Optional[Dict[str, Any]] = None,
        init_only: bool = False,
        sweep: bool = False
    ):
        """
        Initialize ExperimentManager.
        
        Args:
            config: ExperimentConfig instance containing complete experiment configuration
            output_dir: Base directory for experiment outputs
            experiment_name: Optional experiment name (overrides config)
            job_id: Optional job ID from cluster/scheduler
            job_name: Optional job name from cluster/scheduler
            suite_name: Optional suite name if experiment is part of a suite
            suite_info: Optional suite information dictionary (name, description, tags, etc.)
            init_only: If True, only initialize experiment structure without starting job
            sweep: If True, running in wandb sweep context (use existing wandb.run instead of creating new)
        """
        self.config = config
        self.init_only = init_only
        self.sweep = sweep
        # Ensure output_dir is absolute for reliable path operations
        self.output_dir = Path(output_dir).resolve()
        self.experiment_name = experiment_name or config.experiment_name
        self.job_id = job_id or config.job_id
        self.job_name = job_name or config.job_name
        self.suite_name = suite_name
        self.suite_info = suite_info or {}
        
        # Determine experiment ID: use explicit resume_experiment_id if provided, otherwise generate new
        if config.resume_experiment_id:
            # Resuming existing experiment - use provided ID
            self.experiment_id = config.resume_experiment_id
        else:
            # New experiment - generate ID from config hash
            config_hash = self._generate_config_hash()
            self.experiment_id = self._generate_experiment_id(config_hash)
        
        # Create experiment directory structure
        # If part of a suite, create directory under suite folder
        if self.suite_name:
            # If resuming, use explicit resume_suite_id if provided
            if config.resume_experiment_id:
                if config.resume_suite_id:
                    # Use the explicitly provided suite directory
                    suite_dir = self.output_dir / config.resume_suite_id
                    self.experiment_dir = (suite_dir / self.experiment_id).resolve()
                    # Update suite_name to match the resume suite
                    self.suite_name = config.resume_suite_id
                else:
                    # Resuming but no resume_suite_id provided - this is an error for suite experiments
                    raise ValueError(
                        f"Cannot resume suite experiment {self.experiment_id} without resume_suite_id. "
                        f"When resuming a suite experiment, both resume_experiment_id and resume_suite_id must be provided."
                    )
            else:
                # New experiment - create new suite directory if it doesn't exist
                suite_dir = self.output_dir / self.suite_name
                suite_dir.mkdir(parents=True, exist_ok=True)
                self.experiment_dir = (suite_dir / self.experiment_id).resolve()
        else:
            self.experiment_dir = (self.output_dir / self.experiment_id).resolve()
        
        self._create_experiment_structure()
        
        # Set up proper logging (doesn't interfere with tqdm/rich progress bars)
        self._setup_logging()
        
        # Log resume detection after logger is set up
        if config.resume_experiment_id:
            if self.suite_name:
                self.logger.info(f"Resuming existing experiment: {self.experiment_id} in suite: {self.suite_name}")
            else:
                self.logger.info(f"Resuming existing experiment: {self.experiment_id}")
        else:
            self.logger.info(f"Starting new experiment: {self.experiment_id}")
        
        # Capture metadata
        self.metadata = self._capture_metadata()
        self._save_metadata()
        
        # Metrics storage
        self.metrics: Dict[str, Any] = {}
        
        # GPU monitor reference (set by gpu_monitoring_context)
        self.gpu_monitor = None
        
        # Completion reason tracking (can be set by training code)
        # This is used by the sweep system to detect completion (early stopping, etc.)
        self._completion_reason: Optional[str] = None
        
        # Load job history if experiment already exists (for resume)
        # This must happen before wandb init so we can resume the same wandb run
        self.job_history = self._load_job_history()
        
        if self.init_only:
            # Pre-generate or use existing WandB run ID if enabled
            if self.config.wandb.enabled and not self.job_history.get("wandb_run_id"):
                try:
                    import wandb
                    if self.sweep and wandb.run is not None and hasattr(wandb.run, 'id'):
                        # In sweep context: use existing wandb.run.id from wandb.agent()
                        self.job_history["wandb_run_id"] = wandb.run.id
                        self.logger.info(f"Using existing WandB sweep run ID: {self.job_history['wandb_run_id']}")
                    else:
                        # Not in sweep context: generate a unique run ID offline
                        self.job_history["wandb_run_id"] = wandb.util.generate_id()
                        self.logger.info(f"Pre-generated WandB run ID: {self.job_history['wandb_run_id']}")
                except ImportError:
                    self.logger.warning("WandB not installed. Skipping run ID generation.")
            
            # Save the initialized (potentially empty) job history
            self._save_job_history()
            self.logger.info(f"Initialized experiment structure for: {self.experiment_id}")
            return

        # Detect if we're resuming
        self.resume_info = self._detect_resume()
        
        # Register job start for ALL jobs (new and resume)
        self._register_job_start()
        
        # Initialize wandb if enabled
        # If resuming, use the stored wandb_run_id to resume the same run
        self.wandb_run = None
        if config.wandb.enabled:
            # Check if we have a stored wandb run_id from previous jobs
            wandb_run_id = self.job_history.get("wandb_run_id")
            if wandb_run_id and not config.wandb.run_id:
                # Resume existing wandb run - temporarily set run_id in config
                original_run_id = config.wandb.run_id
                config.wandb.run_id = wandb_run_id
                self._init_wandb()
                # Restore original run_id (don't modify user's config permanently)
                config.wandb.run_id = original_run_id
            else:
                # New experiment or user explicitly provided run_id
                self._init_wandb()
    
    
    def _generate_config_hash(self) -> str:
        """
        Generate config hash for experiment matching.
        
        Returns:
            12-character hex hash of config
        """
        # Serialize config to dict (excluding metadata fields that don't affect reproducibility)
        config_dict = self.config.model_dump(exclude={
            'experiment_name', 'job_id', 'job_name', 'random_seed'
        })
        
        # Convert to JSON with sorted keys for deterministic hashing
        config_json = json.dumps(config_dict, sort_keys=True, default=str)
        
        # Generate hash
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:12]
        
        return config_hash
    
    def _generate_experiment_id(self, config_hash: Optional[str] = None) -> str:
        """
        Generate unique experiment ID from config hash.
        
        Args:
            config_hash: Optional pre-computed config hash
        
        Returns:
            Experiment ID in format: {timestamp}_{short_hash}
        """
        if config_hash is None:
            config_hash = self._generate_config_hash()
        
        # Combine with timestamp for human readability
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        
        return f"{timestamp}_{config_hash}"
    
    
    def _create_experiment_structure(self) -> None:
        """Create experiment directory structure."""
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.experiment_dir / "checkpoints").mkdir(exist_ok=True)
        (self.experiment_dir / "visualizations").mkdir(exist_ok=True)
        (self.experiment_dir / "logs").mkdir(exist_ok=True)
        (self.experiment_dir / "configs").mkdir(exist_ok=True)
        (self.experiment_dir / "metrics").mkdir(exist_ok=True)
        (self.experiment_dir / "metadata").mkdir(exist_ok=True)
        
        # Create wandb directory if enabled
        if self.config.wandb.enabled:
            (self.experiment_dir / "wandb").mkdir(exist_ok=True)
    
    def _setup_logging(self) -> None:
        """
        Set up Rich-based logging system.
        
        Creates a Rich Console for terminal output and progress bars.
        All logging goes through the logger, giving us full control without
        needing to redirect stdout/stderr.
        """
        log_dir = self.experiment_dir / "logs"
        
        # Create Rich Console for beautiful terminal output
        self.console = Console(width=None, force_terminal=True)
        
        # Set up Python logging with Rich handler
        self.logger = logging.getLogger(f"experiment.{self.experiment_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()
        
        # File handler for structured logs
        log_file = log_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler: use RichHandler for beautiful output
        console_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True
        )
        
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        # Store handlers for cleanup
        self._log_handlers = [file_handler, console_handler]
        
        # Store file handler reference for file-only logging
        self._file_handler = file_handler
    
    def log_file_only(self, message: str, level: int = logging.INFO) -> None:
        """
        Log a message only to the file handler, bypassing console output.
        
        This is useful for logging messages that should be recorded but not displayed
        on the console, such as during progress bar operations where console output
        would interfere with the display.
        
        Args:
            message (str): The message to log
            level (int): Logging level (default: logging.INFO)
        
        Example:
            ```python
            exp_manager.log_file_only("Test loss for dataset_123: 0.456")
            ```
        """
        record = logging.LogRecord(
            name=self.logger.name,
            level=level,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        record.created = datetime.now().timestamp()
        self._file_handler.emit(record)
    
    def _capture_metadata(self) -> Dict[str, Any]:
        """
        Capture complete experiment metadata.
        
        Returns:
            Dictionary containing all metadata
        """
        metadata = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "job_id": self.job_id,
            "job_name": self.job_name,
            "random_seed": self.config.random_seed,
        }
        
        # Suite information (if experiment is part of a suite)
        if self.suite_name:
            metadata["suite"] = {
                "suite_name": self.suite_name,
                **self.suite_info
            }
        
        # Git information
        git_info = self._get_git_info()
        metadata.update(git_info)
        
        # Environment information
        metadata["environment"] = {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "cwd": str(Path.cwd()),
        }
        
        return metadata
    
    def _get_git_info(self) -> Dict[str, Any]:
        """
        Get Git repository information.
        
        Returns:
            Dictionary with git commit hash, branch, and status
        """
        git_info = {
            "git_commit_hash": None,
            "git_branch": None,
            "git_is_dirty": None,
        }
        
        try:
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                timeout=5
            )
            if result.returncode == 0:
                git_info["git_commit_hash"] = result.stdout.strip()
            
            # Get branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                timeout=5
            )
            if result.returncode == 0:
                git_info["git_branch"] = result.stdout.strip()
            
            # Check if working directory is dirty
            result = subprocess.run(
                ["git", "diff", "--quiet"],
                cwd=Path.cwd(),
                timeout=5
            )
            git_info["git_is_dirty"] = result.returncode != 0
            
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            # Git not available or not a git repo
            pass
        
        return git_info
    
    def _save_metadata(self) -> None:
        """Save metadata to experiment directory."""
        # Save metadata JSON
        metadata_path = self.experiment_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        # Save complete config hierarchy
        self._save_config_hierarchy()
    
    def _save_config_hierarchy(self) -> None:
        """Save complete config hierarchy to experiment directory with recursive copying."""
        # Save primary config
        primary_config_path = self.experiment_dir / "configs" / "experiment_config.yaml"
        self.config.to_yaml(primary_config_path)
        
        # Track copied configs to avoid duplicates
        self._copied_configs = set()
        
        # Load and save nested configs (model, data, etc.) with recursive copying
        self._copy_config_file_recursive(self.config.model.config_path, "model_config.yaml")
        self._copy_config_file_recursive(self.config.data.config_path, "data_config.yaml")
        
        # Save nested configs if they exist
        if self.config.plotting:
            self._copy_config_file_recursive(self.config.plotting, "plotting_config.yaml")
        if self.config.evaluation:
            self._copy_config_file_recursive(self.config.evaluation, "evaluation_config.yaml")
    
    def _copy_config_file_recursive(self, source_path: str, dest_name: str, base_dir: Optional[Path] = None) -> None:
        """
        Copy a config file to experiment directory and recursively copy any nested configs it references.
        
        This method:
        1. Copies the source config file
        2. Parses the YAML to find any references to other config files
        3. Recursively copies those referenced configs
        
        Args:
            source_path: Path to source config file
            dest_name: Destination filename in experiment configs directory
            base_dir: Base directory for resolving relative paths in nested configs
        """
        source = Path(source_path)
        if not source.is_absolute():
            source = Path.cwd() / source
        
        if not source.exists():
            return
        
        # Avoid copying the same file twice
        if str(source.resolve()) in self._copied_configs:
            return
        
        # Copy the main config file
        dest = self.experiment_dir / "configs" / dest_name
        shutil.copy2(source, dest)
        self._copied_configs.add(str(source.resolve()))
        
        # Parse the config file to find nested config references
        try:
            with open(source, 'r', encoding='utf-8') as f:
                config_content = yaml.safe_load(f)
            
            if config_content is None:
                return
            
            # Use source file's directory as base for resolving relative paths
            if base_dir is None:
                base_dir = source.parent
            
            # Recursively search for config file references
            self._find_and_copy_nested_configs(config_content, base_dir, dest_name)
            
        except (yaml.YAMLError, Exception):
            # If we can't parse the config, just copy it and continue
            # This handles non-YAML files or corrupted files gracefully
            pass
    
    def _find_and_copy_nested_configs(self, config_dict: Any, base_dir: Path, parent_name: str) -> None:
        """
        Recursively find and copy nested config file references in a config dictionary.
        
        Args:
            config_dict: Dictionary or value to search for config references
            base_dir: Base directory for resolving relative paths
            parent_name: Name of parent config file (for naming nested configs)
        """
        if isinstance(config_dict, dict):
            for key, value in config_dict.items():
                if isinstance(value, str) and (value.endswith('.yaml') or value.endswith('.yml')):
                    # This looks like a config file reference
                    # Check if it's a valid file path
                    nested_config_path = Path(value)
                    if not nested_config_path.is_absolute():
                        nested_config_path = base_dir / nested_config_path
                    
                    if nested_config_path.exists() and nested_config_path.is_file():
                        # Generate a unique name for the nested config
                        # Use parent name and key to create a descriptive name
                        nested_dest_name = f"{parent_name.replace('.yaml', '').replace('.yml', '')}_{key}.yaml"
                        
                        # Recursively copy this nested config
                        self._copy_config_file_recursive(
                            str(nested_config_path), 
                            nested_dest_name, 
                            base_dir=nested_config_path.parent
                        )
                elif isinstance(value, (dict, list)):
                    # Recursively search nested structures
                    self._find_and_copy_nested_configs(value, base_dir, parent_name)
        elif isinstance(config_dict, list):
            for item in config_dict:
                if isinstance(item, (dict, list)):
                    self._find_and_copy_nested_configs(item, base_dir, parent_name)
    
    def _copy_config_file(self, source_path: str, dest_name: str) -> None:
        """
        Copy a config file to experiment directory (legacy method for backward compatibility).
        
        Args:
            source_path: Path to source config file
            dest_name: Destination filename in experiment configs directory
        """
        self._copy_config_file_recursive(source_path, dest_name)
    
    def _init_wandb(self) -> None:
        """
        Initialize Weights & Biases run for experiment tracking.
        
        Creates a wandb run with experiment metadata, config, and links to job info.
        Handles offline mode gracefully if wandb is not available.
        """
        try:
            import wandb
            
            # Flatten config for wandb (wandb prefers flat configs)
            wandb_config = self._flatten_config(self.config.model_dump())
            
            # Add metadata to config
            wandb_config.update({
                'git_commit': self.metadata.get('git_commit_hash'),
                'git_branch': self.metadata.get('git_branch'),
                'git_is_dirty': self.metadata.get('git_is_dirty'),
                'job_id': self.job_id,
                'job_name': self.job_name,
                'random_seed': self.config.random_seed,
                'experiment_id': self.experiment_id,
            })
            
            # Determine run name (use fixed name if provided, otherwise use experiment_id)
            run_name = self.config.wandb.run_name if self.config.wandb.run_name else self.experiment_id
            
            # Ensure wandb directory path is absolute and exists
            # This is critical when base_data_path is set and we're not in the project root
            wandb_dir = (self.experiment_dir / "wandb").resolve()
            wandb_dir.mkdir(parents=True, exist_ok=True)
            
            # Build wandb.init() arguments
            init_kwargs = {
                'project': self.config.wandb.project,
                'entity': self.config.wandb.entity,
                'name': run_name,
                'tags': self.config.wandb.tags,
                'notes': self.config.wandb.notes,
                'config': wandb_config,
                'mode': self.config.wandb.mode,
                'dir': str(wandb_dir),  # Use absolute path to avoid working directory issues
                'reinit': False
            }
            
            # Add run_id if specified (for resuming/overwriting runs)
            if self.config.wandb.run_id:
                init_kwargs['id'] = self.config.wandb.run_id
                # Use 'allow' to support both:
                # 1. First run with pre-generated ID (creates new run on server)
                # 2. Resuming existing run (resumes run on server)
                init_kwargs['resume'] = 'allow'
            
            # Check if we're in a sweep context (wandb.run already exists from wandb.agent)
            if self.sweep and wandb.run is not None:
                # Use existing wandb run from sweep context
                self.wandb_run = wandb.run
                self.logger.info(f"Using existing WandB sweep run: {self.wandb_run.id}")
                
                # Update config with metadata only (exclude sweep-controlled parameters)
                # Sweep parameters like training.batch_size, training.learning_rate are locked by wandb
                # Only update metadata fields that we explicitly add, avoiding sweep-controlled params
                metadata_only = {
                    'git_commit': self.metadata.get('git_commit_hash'),
                    'git_branch': self.metadata.get('git_branch'),
                    'git_is_dirty': self.metadata.get('git_is_dirty'),
                    'job_id': self.job_id,
                    'job_name': self.job_name,
                    'random_seed': self.config.random_seed,
                    'experiment_id': self.experiment_id,
                }
                # Update only metadata fields (these are not sweep-controlled, so no warnings)
                self.wandb_run.config.update(metadata_only, allow_val_change=True)
            else:
                # Not in sweep context - initialize new wandb run
                self.wandb_run = wandb.init(**init_kwargs)

            # Define custom step metrics for batch vs epoch logging
            # This prevents step counter conflicts between batch and epoch metrics
            if self.wandb_run is not None:
                wandb.define_metric("batch_step")
                wandb.define_metric("epoch")

                # Batch-level metrics use batch_step as x-axis
                wandb.define_metric("batch_loss", step_metric="batch_step")
                wandb.define_metric("batch_grad_norm", step_metric="batch_step")

                # Epoch-level metrics use epoch as x-axis
                wandb.define_metric("train_loss", step_metric="epoch")
                wandb.define_metric("val_loss", step_metric="epoch")
                wandb.define_metric("test_loss", step_metric="epoch")
                wandb.define_metric("learning_rate", step_metric="epoch")

            # Store wandb run_id in job_history for resume capability
            if self.wandb_run and hasattr(self.wandb_run, 'id'):
                wandb_run_id = self.wandb_run.id
                if not self.job_history.get("wandb_run_id"):
                    # First job - store the run_id
                    self.job_history["wandb_run_id"] = wandb_run_id
                    self._save_job_history()
                elif self.job_history.get("wandb_run_id") != wandb_run_id:
                    # Run ID mismatch - update job_history with correct ID (sweep context)
                    # This can happen if init_only pre-generated a different ID before sweep run was created
                    old_id = self.job_history.get("wandb_run_id")
                    self.job_history["wandb_run_id"] = wandb_run_id
                    self._save_job_history()
                    if self.sweep:
                        self.logger.warning(
                            f"Updated WandB run_id in job_history: {old_id} -> {wandb_run_id} "
                            f"(using sweep run ID). This should not happen."
                        )
                    else:
                        self.logger.warning(
                            f"Wandb run_id mismatch: job_history has {old_id}, "
                            f"but resumed with {wandb_run_id}. This may indicate a configuration issue."
                        )
            
            # Log config files as artifacts
            self._log_config_artifacts()
            
        except ImportError:
            # Wandb not installed, continue without it
            print("Warning: wandb not installed. Continuing without wandb logging.")
            self.wandb_run = None
        except Exception as e:
            # Other errors (e.g., authentication, network)
            print(f"Warning: Failed to initialize wandb: {e}. Continuing without wandb logging.")
            self.wandb_run = None
    
    def _flatten_config(self, config_dict: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        Flatten nested dictionary for wandb config.
        
        Note: This method currently only handles single-level nesting (e.g., 
        {'model': {'name': 'DLinear'}} becomes {'model.name': 'DLinear'}). 
        If the config structure includes deeper nesting (e.g., 
        {'model': {'config': {'param': 'value'}}}), this method will need 
        to be updated to handle multi-level nesting recursively.
        
        Args:
            config_dict: Nested dictionary to flatten
            parent_key: Parent key prefix for nested items
            sep: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in config_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_config(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _log_config_artifacts(self) -> None:
        """Log configuration files as wandb artifacts."""
        if self.wandb_run is None:
            return
        
        try:
            import wandb
            
            # Create artifact for configs
            config_artifact = wandb.Artifact(
                name=f"configs_{self.experiment_id}",
                type="config",
                description="Experiment configuration files"
            )
            
            # Add config files to artifact
            configs_dir = self.experiment_dir / "configs"
            for config_file in configs_dir.glob("*.yaml"):
                config_artifact.add_file(str(config_file), name=config_file.name)
            
            # Log artifact
            self.wandb_run.log_artifact(config_artifact)
            
        except Exception as e:
            print(f"Warning: Failed to log config artifacts to wandb: {e}")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric value to both local storage and wandb.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step/epoch number
        """
        if step is not None:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append({"step": step, "value": value})
        else:
            self.metrics[name] = value
        
        # Log to wandb if enabled and run is active
        if self.wandb_run is not None:
            try:
                # Check if wandb run is still active
                if hasattr(self.wandb_run, '_wandb') and self.wandb_run._wandb.run is None:
                    # Run is finished, skip logging
                    return
                
                if step is not None:
                    self.wandb_run.log({name: value}, step=step)
                else:
                    # Log without step - use commit=False to avoid step conflicts
                    # GPU metrics and other non-step metrics should not interfere with training step tracking
                    self.wandb_run.log({name: value}, commit=False)
            except Exception as e:
                # Suppress warnings about step ordering and finished runs
                error_msg = str(e).lower()
                if ("step" not in error_msg and 
                    "finished" not in error_msg and 
                    "is finished" not in error_msg):
                    print(f"Warning: Failed to log metric to wandb: {e}")
        
        # Save metrics to file
        self._save_metrics()
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log multiple metrics at once to both local storage and wandb.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step/epoch number
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def log_batch_metrics(self, metrics: Dict[str, Any], batch_step: int) -> None:
        """
        Log batch-level metrics with batch_step as the step counter.

        Uses wandb.define_metric() to ensure batch metrics are plotted
        against batch_step, not the global step counter.

        Args:
            metrics: Dictionary of metric names to values (e.g., batch_loss, batch_grad_norm)
            batch_step: Global batch step number (epoch * total_batches + batch_idx)
        """
        # Store locally
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append({"step": batch_step, "value": value})

        # Log to wandb with batch_step as the step metric
        if self.wandb_run is not None:
            try:
                if hasattr(self.wandb_run, '_wandb') and self.wandb_run._wandb.run is None:
                    return  # Run is finished
                log_dict = {"batch_step": batch_step, **metrics}
                self.wandb_run.log(log_dict)
            except Exception as e:
                error_msg = str(e).lower()
                if "step" not in error_msg and "finished" not in error_msg:
                    print(f"Warning: Failed to log batch metrics to wandb: {e}")

        self._save_metrics()

    def log_epoch_metrics(self, metrics: Dict[str, Any], epoch: int) -> None:
        """
        Log epoch-level metrics with epoch as the step counter.

        Uses wandb.define_metric() to ensure epoch metrics are plotted
        against epoch number, not the global step counter.

        Args:
            metrics: Dictionary of metric names to values (e.g., train_loss, val_loss)
            epoch: Epoch number (1-indexed)
        """
        # Store locally
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append({"step": epoch, "value": value})

        # Log to wandb with epoch as the step metric
        if self.wandb_run is not None:
            try:
                if hasattr(self.wandb_run, '_wandb') and self.wandb_run._wandb.run is None:
                    return  # Run is finished
                log_dict = {"epoch": epoch, **metrics}
                self.wandb_run.log(log_dict)
            except Exception as e:
                error_msg = str(e).lower()
                if "step" not in error_msg and "finished" not in error_msg:
                    print(f"Warning: Failed to log epoch metrics to wandb: {e}")

        self._save_metrics()

    def _save_metrics(self) -> None:
        """Save metrics to JSON file."""
        metrics_path = self.experiment_dir / "metrics" / "metrics.json"
        # Ensure directory exists before writing
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, default=str)
    
    def get_checkpoint_dir(self) -> Path:
        """Get path to checkpoint directory."""
        return self.experiment_dir / "checkpoints"
    
    def get_log_dir(self) -> Path:
        """Get path to log directory."""
        return self.experiment_dir / "logs"
    
    def get_visualization_dir(self) -> Path:
        """Get path to visualization directory."""
        return self.experiment_dir / "visualizations"
    
    def save_checkpoint(
        self, 
        checkpoint: Dict[str, Any], 
        filename: str = "checkpoint.pth",
        is_best: bool = False
    ) -> Path:
        """
        Save model checkpoint to local storage and wandb artifact.
        
        Args:
            checkpoint: Dictionary containing checkpoint data
            filename: Checkpoint filename
            is_best: Whether this is the best checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        import torch
        checkpoint_path = self.get_checkpoint_dir() / filename
        torch.save(checkpoint, checkpoint_path)
        
        # Log checkpoint as wandb artifact if enabled
        if self.wandb_run is not None and is_best:
            try:
                import wandb
                
                checkpoint_artifact = wandb.Artifact(
                    name=f"checkpoint_{self.experiment_id}",
                    type="model",
                    description=f"Model checkpoint: {filename}"
                )
                checkpoint_artifact.add_file(str(checkpoint_path), name=filename)
                self.wandb_run.log_artifact(checkpoint_artifact)
                
            except Exception as e:
                print(f"Warning: Failed to log checkpoint artifact to wandb: {e}")
        
        return checkpoint_path
    
    def save_visualization(self, fig, filename: str, step: Optional[int] = None) -> Path:
        """
        Save visualization figure to local storage and wandb.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            step: Optional step/epoch number for wandb logging
            
        Returns:
            Path to saved visualization
        """
        vis_path = self.get_visualization_dir() / filename
        fig.savefig(vis_path, dpi=300, bbox_inches='tight')
        
        # Log to wandb if enabled
        if self.wandb_run is not None:
            try:
                import wandb
                self.wandb_run.log({filename: wandb.Image(str(vis_path))}, step=step)
            except Exception as e:
                print(f"Warning: Failed to log visualization to wandb: {e}")
        
        return vis_path
    
    def end_experiment(self, final_metrics: Optional[Dict[str, Any]] = None, sweep_completed: Optional[bool] = None) -> None:
        """
        Finalize experiment, log final metrics, and close wandb run.
        
        Args:
            final_metrics: Optional dictionary of final metrics to log
            sweep_completed: Whether to mark the sweep run as completed in wandb.summary.
                           - True: Explicitly mark as completed (all epochs finished)
                           - False: Explicitly mark as incomplete (interrupted, should resume)
                           - None (default): Auto-detect based on job_history epochs
                           
                           Auto-detection checks if current_epoch >= total_epochs in job_history.
                           This prevents false positives where a partial run is marked complete.
        """
        # Auto-detect sweep_completed if not explicitly provided
        if sweep_completed is None:
            sweep_completed = self._detect_training_completed()
            if sweep_completed:
                self.logger.info("Auto-detected training as COMPLETED (all epochs finished)")
            else:
                self.logger.info("Auto-detected training as INCOMPLETE (not all epochs finished)")
        
        # Get GPU monitor summary before closing wandb (if monitor is active)
        gpu_metrics = None
        if self.gpu_monitor is not None:
            try:
                # Get summary without stopping the monitor (it will be stopped in context cleanup)
                summary = self.gpu_monitor.summary()
                if summary:
                    gpu_metrics = {
                        'gpu_avg_util_pct': summary.get('avg_util_gpu_pct'),
                        'gpu_p95_util_pct': summary.get('p95_util_gpu_pct'),
                        'gpu_avg_mem_used_mib': summary.get('avg_mem_used_mib'),
                        'gpu_max_mem_used_mib': summary.get('max_mem_used_mib'),
                        'gpu_mem_total_mib': summary.get('mem_total_mib'),
                        'gpu_avg_power_w': summary.get('avg_power_w'),
                        'gpu_max_temp_c': summary.get('max_temp_c'),
                        'gpu_num_samples': summary.get('num_samples'),
                    }
                    # Filter out None values
                    gpu_metrics = {k: v for k, v in gpu_metrics.items() if v is not None}
            except Exception as e:
                print(f"Warning: Failed to get GPU monitor summary: {e}")
        
        # Stop GPU monitor Rich Live display before wandb finishes
        # This ensures wandb's colored output displays properly
        try:
            from utils.gpu_monitor import _stop_gpu_monitor_display
            _stop_gpu_monitor_display()
        except Exception:
            # If GPU monitor is not available, continue silently
            pass
        
        # Merge final metrics with GPU metrics
        all_final_metrics = final_metrics.copy() if final_metrics else {}
        if gpu_metrics:
            all_final_metrics.update(gpu_metrics)
        
        # Log final metrics if provided
        if all_final_metrics:
            self.log_metrics(all_final_metrics)
            
            # Update wandb summary with final metrics (including GPU metrics)
            if self.wandb_run is not None:
                try:
                    self.wandb_run.summary.update(all_final_metrics)
                except Exception as e:
                    print(f"Warning: Failed to update wandb summary: {e}")
        
        # Get epoch info for completion tracking
        current_epoch = self.job_history.get("current_epoch", 0) if hasattr(self, 'job_history') else 0
        total_epochs = self.job_history.get("total_epochs", self.config.training.epochs) if hasattr(self, 'job_history') else self.config.training.epochs
        
        # Determine completion reason
        completion_reason = self._completion_reason
        if completion_reason is None and sweep_completed:
            # Default to "all_epochs" if completed and no specific reason set
            completion_reason = "all_epochs"
        
        # Save completion_reason to job_history.json for sweep resumption tracking
        # This ensures the sweep system can detect early stopping, hyperband pruning, etc.
        if hasattr(self, 'job_history') and completion_reason:
            self.job_history["completion_reason"] = completion_reason
            self._save_job_history()
            self.logger.debug(f"Saved completion_reason to job_history: {completion_reason}")
        
        # Note: Sweep registry updates are handled by SweepManager._finalize_run()
        # The ExperimentManager no longer directly updates the registry.
        # This ensures single ownership of registry state.
        
        # Mark sweep completion in wandb.summary BEFORE calling finish()
        # This is for central visibility but NOT authoritative for resumption
        if self.wandb_run is not None:
            try:
                completion_status = {
                    '_sweep_completed': sweep_completed,
                    '_sweep_completion_time': datetime.now().isoformat(),
                    '_experiment_id': self.experiment_id,
                    '_final_epoch': current_epoch,
                    '_total_epochs': total_epochs,
                    '_completion_reason': completion_reason,
                    '_location': get_location() if SWEEP_REGISTRY_AVAILABLE and get_location else None,
                }
                if self.suite_name:
                    completion_status['_suite_name'] = self.suite_name
                
                # Filter out None values
                completion_status = {k: v for k, v in completion_status.items() if v is not None}
                
                self.wandb_run.summary.update(completion_status)
                
                # Also try to update config (may fail if sweep wrapper already called finish)
                try:
                    self.wandb_run.config.update({
                        '_job_status': 'completed' if sweep_completed else 'interrupted'
                    }, allow_val_change=True)
                except Exception:
                    # Config update failed - this is expected if run is finishing
                    pass
                    
            except Exception as e:
                print(f"Warning: Failed to update wandb completion status: {e}")
        
        # Close wandb run
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception as e:
                print(f"Warning: Failed to finish wandb run: {e}")
        
        # Close logging handlers
        for handler in getattr(self, '_log_handlers', []):
            handler.close()
            if handler in self.logger.handlers:
                self.logger.removeHandler(handler)
        
        # Save final metadata
        self.metadata["end_timestamp"] = datetime.now().isoformat()
        
        # In sweep mode, ensure wandb config (including sweep hyperparameters) is saved to metadata
        # This ensures consistency between local metadata and wandb config for resumption
        if self.sweep and self.wandb_run is not None:
            try:
                # Save wandb run config to metadata (includes sweep hyperparameters)
                wandb_config = dict(self.wandb_run.config)
                self.metadata["wandb_config"] = wandb_config
                # Also save config_path for easy access during resumption
                if "_config_path" in wandb_config:
                    self.metadata["config_path"] = wandb_config["_config_path"]
                self.logger.debug("Saved wandb config to metadata for sweep consistency")
            except Exception as e:
                self.logger.warning(f"Could not save wandb config to metadata: {e}")
        
        metadata_path = self.experiment_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    
    def get_experiment_id(self) -> str:
        """Get experiment ID."""
        return self.experiment_id
    
    def get_experiment_dir(self) -> Path:
        """Get experiment directory path."""
        return self.experiment_dir
    
    def get_console(self) -> Console:
        """Get Rich Console instance for terminal output."""
        return self.console
    
    def _load_job_history(self) -> Dict[str, Any]:
        """
        Load job history from experiment directory if it exists.
        
        Returns:
            Dictionary containing job history, or empty dict if no history exists
        """
        job_history_path = self.experiment_dir / "job_history.json"
        
        # If resuming, validate that experiment exists
        if self.config.resume_experiment_id:
            if not job_history_path.exists():
                raise ValueError(
                    f"Cannot resume experiment {self.experiment_id}: job_history.json not found. "
                    f"Ensure the experiment_id is correct and the experiment directory exists."
                )
            if not self.experiment_dir.exists():
                raise ValueError(
                    f"Cannot resume experiment {self.experiment_id}: experiment directory not found. "
                    f"Ensure the experiment_id is correct."
                )
        
        if not job_history_path.exists():
            # First job - initialize empty history
            return {
                "experiment_id": self.experiment_id,
                "total_epochs": self.config.training.epochs,
                "jobs": [],
                "current_epoch": 0,
                "last_checkpoint": None,
                "best_checkpoint": None,
                "wandb_run_id": None  # Will be set when wandb is initialized
            }
        
        try:
            with open(job_history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            # Validate history structure and experiment_id match
            if "experiment_id" not in history or history["experiment_id"] != self.experiment_id:
                if self.config.resume_experiment_id:
                    raise ValueError(
                        f"Experiment ID mismatch: expected {self.experiment_id}, "
                        f"but job_history.json has {history.get('experiment_id')}. "
                        f"Ensure resume_experiment_id matches exactly."
                    )
                else:
                    self.logger.warning(f"Job history experiment_id mismatch. Expected {self.experiment_id}, got {history.get('experiment_id')}. Creating new history.")
                    return {
                        "experiment_id": self.experiment_id,
                        "total_epochs": self.config.training.epochs,
                        "jobs": [],
                        "current_epoch": 0,
                        "last_checkpoint": None,
                        "best_checkpoint": None,
                        "wandb_run_id": None
                    }
            
            return history
        except (json.JSONDecodeError, IOError) as e:
            if self.config.resume_experiment_id:
                raise ValueError(
                    f"Failed to load job history for resume: {e}. "
                    f"Ensure the experiment exists and job_history.json is valid."
                )
            self.logger.warning(f"Failed to load job history: {e}. Creating new history.")
            return {
                "experiment_id": self.experiment_id,
                "total_epochs": self.config.training.epochs,
                "jobs": [],
                "current_epoch": 0,
                "last_checkpoint": None,
                "best_checkpoint": None,
                "wandb_run_id": None
            }
    
    def _save_job_history(self) -> None:
        """Save job history to experiment directory."""
        job_history_path = self.experiment_dir / "job_history.json"
        with open(job_history_path, 'w', encoding='utf-8') as f:
            json.dump(self.job_history, f, indent=2, default=str)
    
    def _detect_resume(self) -> Optional[Dict[str, Any]]:
        """
        Detect if we should resume from a previous checkpoint.
        
        Returns:
            Dictionary with resume information (start_epoch, checkpoint_path) or None if starting fresh
        """
        # Check if we have any completed jobs
        if not self.job_history.get("jobs"):
            # No previous jobs - start fresh
            return None
        
        # Check for running jobs first
        running_jobs = [job for job in self.job_history["jobs"] if job.get("status") == "running"]
        if running_jobs:
            # There's a running job - require explicit confirmation that it's complete
            if not self.config.mark_last_job_complete:
                last_job = running_jobs[-1]
                raise ValueError(
                    f"Another job is currently running on this experiment. "
                    f"Job {last_job.get('slurm_job_id', last_job.get('job_id', 'unknown'))} "
                    f"started at {last_job.get('start_time')} and is marked as 'running'. "
                    f"If that job has completed or timed out, set 'mark_last_job_complete: true' "
                    f"in your config to mark it as complete and resume."
                )
            
            # Explicitly mark the last running job as complete/timeout (mark_last_job_complete: true)
            last_job = running_jobs[-1]
            last_epoch = self.job_history.get("current_epoch", 0)
            
            if last_epoch == 0:
                raise ValueError(
                    "Cannot resume: last running job has current_epoch=0 in job_history. "
                    "This indicates the job history was not properly updated. "
                    "Please check the job_history.json file and ensure current_epoch reflects the actual last completed epoch."
                )
            
            # Mark the job as timeout (since it didn't complete normally)
            last_job["status"] = "timeout"
            last_job["end_time"] = datetime.now().isoformat()
            last_job["end_epoch"] = last_epoch
            self._save_job_history()
            self.logger.info(f"Marked last running job as timeout, ended at epoch {last_epoch}")
        
        # Get last completed job (now includes the job we just marked as timeout)
        completed_jobs = [job for job in self.job_history["jobs"] if job.get("status") in ["completed", "timeout"]]
        if not completed_jobs:
            # No completed jobs - start fresh
            return None
        
        # Get the most recent completed job
        last_job = completed_jobs[-1]
        last_epoch = last_job.get("end_epoch", self.job_history.get("current_epoch", 0))
        
        # Check if training is complete
        if last_epoch >= self.job_history.get("total_epochs", self.config.training.epochs):
            self.logger.info(f"Training already completed (epoch {last_epoch}/{self.job_history.get('total_epochs', self.config.training.epochs)})")
            return None
        
        # Validate that we have a valid epoch
        if last_epoch == 0:
            raise ValueError(
                "Cannot resume: last completed job has end_epoch=0. "
                "This indicates the job history was not properly updated. "
                "Please check the job_history.json file and ensure end_epoch reflects the actual last completed epoch."
            )
        
        # Find the latest checkpoint
        checkpoint_path = self._find_latest_checkpoint()
        if not checkpoint_path:
            self.logger.warning("Resume detected but no checkpoint found. Starting from last completed epoch.")
            return {
                "start_epoch": last_epoch + 1,
                "checkpoint_path": None,
                "last_epoch": last_epoch
            }
        
        self.logger.info(f"Resuming training from epoch {last_epoch + 1} (last completed: {last_epoch})")
        return {
            "start_epoch": last_epoch + 1,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "last_epoch": last_epoch
        }
    
    def _find_latest_checkpoint(self) -> Optional[Path]:
        """
        Find the latest checkpoint in the checkpoint directory.
        
        Returns:
            Path to latest checkpoint, or None if no checkpoint found
        """
        checkpoint_dir = self.get_checkpoint_dir()
        
        if not checkpoint_dir.exists():
            return None
        
        # For PyTorch: look for checkpoint.pth or epoch-based checkpoints
        # For Lightning: look for last.ckpt or epoch-based checkpoints
        
        # Try Lightning last checkpoint first
        last_ckpt = checkpoint_dir / "last.ckpt"
        if last_ckpt.exists():
            return last_ckpt
        
        # Try PyTorch checkpoint.pth
        pytorch_ckpt = checkpoint_dir / "checkpoint.pth"
        if pytorch_ckpt.exists():
            return pytorch_ckpt
        
        # Look for epoch-based checkpoints (Lightning format: checkpoint-{epoch:02d}-{val_loss:.6f}.ckpt)
        epoch_checkpoints = list(checkpoint_dir.glob("checkpoint-*.ckpt"))
        if epoch_checkpoints:
            # Sort by modification time, return most recent
            epoch_checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return epoch_checkpoints[0]
        
        # Look for any .pth files
        pth_checkpoints = list(checkpoint_dir.glob("*.pth"))
        if pth_checkpoints:
            pth_checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return pth_checkpoints[0]
        
        return None
    
    def _register_job_start(self) -> None:
        """Register the start of a new job in job history."""
        # Determine start epoch: use resume_info if resuming, otherwise use current_epoch from history
        if self.resume_info:
            start_epoch = self.resume_info["start_epoch"]
        else:
            # New job - start from current_epoch (which should be 0 for new experiments)
            start_epoch = self.job_history.get("current_epoch", 0)
        
        job_entry = {
            "job_id": self.job_id,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", self.job_id),
            "job_name": self.job_name or os.environ.get("SLURM_JOB_NAME", "training_job"),
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "start_epoch": start_epoch,
            "end_epoch": None,
            "epochs_completed": [],
            "status": "running",
            "checkpoint_path": None,
            "final_train_loss": None,
            "final_val_loss": None
        }
        
        self.job_history["jobs"].append(job_entry)
        self._save_job_history()
        
        if self.resume_info:
            self.logger.info(f"Registered job start: SLURM_JOB_ID={job_entry['slurm_job_id']}, resuming from epoch {start_epoch}")
        else:
            self.logger.info(f"Registered job start: SLURM_JOB_ID={job_entry['slurm_job_id']}, starting from epoch {start_epoch}")
    
    def register_job_end(
        self,
        end_epoch: int,
        status: str = "completed",
        checkpoint_path: Optional[str] = None,
        final_train_loss: Optional[float] = None,
        final_val_loss: Optional[float] = None
    ) -> None:
        """
        Register the end of the current job in job history.
        
        Args:
            end_epoch: Last epoch completed in this job
            status: Job status ("completed", "timeout", "failed")
            checkpoint_path: Path to final checkpoint for this job
            final_train_loss: Final training loss
            final_val_loss: Final validation loss
        """
        if not self.job_history.get("jobs"):
            self.logger.warning("No job history to update. Job may not have been registered.")
            return
        
        # Update the last (current) job entry
        current_job = self.job_history["jobs"][-1]
        current_job["end_time"] = datetime.now().isoformat()
        current_job["end_epoch"] = end_epoch
        current_job["status"] = status
        current_job["checkpoint_path"] = checkpoint_path
        current_job["final_train_loss"] = final_train_loss
        current_job["final_val_loss"] = final_val_loss
        
        # Update global tracking
        self.job_history["current_epoch"] = end_epoch
        if checkpoint_path:
            self.job_history["last_checkpoint"] = checkpoint_path
        
        self._save_job_history()
        self.logger.info(f"Registered job end: epoch {end_epoch}, status={status}")
    
    def check_should_stop(self) -> tuple[bool, Optional[str]]:
        """
        Check if training should stop early due to external signals.
        
        This method checks for stop signals that should be checked after each epoch:
        1. wandb.run.should_stop() - Set by Hyperband/ASHA scheduler when run is pruned
        2. wandb.run.stopped - Alternative check for Hyperband pruning
        
        This is particularly important for sweep runs where:
        - Hyperband may prune runs early based on performance
        - Shutdown signals may be received (though these are typically handled
          by the training loop checking shutdown_check callback)
        
        Returns:
            Tuple of (should_stop: bool, reason: Optional[str])
            - should_stop: True if training should exit gracefully
            - reason: Completion reason if stopping ("hyperband", etc.), None otherwise
        
        Example:
            # In training loop after each epoch:
            should_stop, reason = exp_manager.check_should_stop()
            if should_stop:
                if reason:
                    exp_manager.set_completion_reason(reason)
                break  # Exit training loop gracefully
        """
        # Check wandb for Hyperband/ASHA pruning
        try:
            import wandb
            if wandb.run is not None:
                # Check should_stop() method (preferred)
                if hasattr(wandb.run, 'should_stop') and callable(wandb.run.should_stop):
                    if wandb.run.should_stop():
                        self.logger.info("Training stopped: wandb.run.should_stop() returned True (Hyperband pruning)")
                        return True, "hyperband"
                
                # Check stopped attribute (fallback)
                if getattr(wandb.run, 'stopped', False):
                    self.logger.info("Training stopped: wandb.run.stopped is True (Hyperband pruning)")
                    return True, "hyperband"
        except Exception:
            pass  # wandb not available or error
        
        return False, None
    
    def update_current_epoch(self, epoch: int, checkpoint_path: Optional[str] = None) -> bool:
        """
        Update current epoch in job history (called after each epoch).
        
        This method:
        1. Updates job_history.json with the current epoch progress
        2. Automatically checks for early termination signals (Hyperband, shutdown, etc.)
        3. Returns True if training should stop, False otherwise
        
        For sweep runs, this automatically detects Hyperband pruning via
        wandb.run.should_stop() and sets the completion reason appropriately.
        
        Args:
            epoch: Current epoch number (1-indexed)
            checkpoint_path: Optional path to checkpoint saved at this epoch
            
        Returns:
            True if training should stop (e.g., Hyperband pruning), False otherwise
            
        Example:
            # In training loop after each epoch:
            should_stop = exp_manager.update_current_epoch(epoch, checkpoint_path)
            if should_stop:
                break  # Exit training loop gracefully
        """
        if not self.job_history.get("jobs"):
            return False
        
        # Update current job's epochs_completed list
        current_job = self.job_history["jobs"][-1]
        if epoch not in current_job["epochs_completed"]:
            current_job["epochs_completed"].append(epoch)
        
        # Update global tracking
        self.job_history["current_epoch"] = epoch
        if checkpoint_path:
            self.job_history["last_checkpoint"] = checkpoint_path
        
        # Save periodically (every epoch)
        # This ensures job_history.json is always up-to-date, even if SIGTERM
        # kills the process before end_experiment() can run. The registry
        # reconciliation in SweepManager will read this to determine final state.
        self._save_job_history()
        
        # Check for early termination signals (Hyperband pruning, etc.)
        # This is called automatically after each epoch to detect external stop signals
        should_stop, reason = self.check_should_stop()
        if should_stop:
            # Set completion reason if provided (e.g., "hyperband")
            if reason:
                self.set_completion_reason(reason)
            return True
        
        return False
    
    def _detect_training_completed(self) -> bool:
        """
        Auto-detect whether training has completed successfully.
        
        This method determines if training is COMPLETE (should NOT be resumed).
        It checks multiple signals in priority order:
        
        1. Explicit completion reason (set via set_completion_reason())
           - early_stopping, hyperband, converged, manual_stop
        
        2. wandb.run.stopped (set by Hyperband/ASHA scheduler)
           - Indicates sweep controller pruned the run
        
        3. Epoch comparison (current_epoch >= total_epochs)
           - All planned epochs have been completed
        
        If none of these are true, the run is considered INCOMPLETE and
        should be resumed later.
        
        Returns:
            True if training completed successfully (should NOT be resumed)
            False if training is incomplete (should be resumed)
            
        Note:
            This method is used by end_experiment() to auto-detect completion
            status when sweep_completed is not explicitly provided.
        """
        # If init_only mode, training hasn't started yet
        if self.init_only:
            return False
        
        # Priority 1: Check for explicit completion reason
        # This handles early stopping, hyperband, convergence, etc.
        if self._completion_reason is not None:
            self.logger.debug(f"Training completed: explicit reason set ({self._completion_reason})")
            return True
        
        # Priority 2: Check wandb.run.stopped (Hyperband/ASHA pruning)
        # The sweep controller sets this flag when pruning a run
        try:
            import wandb
            if wandb.run is not None and getattr(wandb.run, 'stopped', False):
                self.logger.debug("Training completed: wandb.run.stopped is True (Hyperband pruning)")
                self._completion_reason = "hyperband"  # Record the reason
                return True
        except Exception:
            pass  # wandb not available or error
        
        # Priority 3: Check job_history for epoch completion
        if not hasattr(self, 'job_history') or not self.job_history:
            # No job history - can't determine, assume incomplete for safety
            self.logger.warning("No job_history available for completion detection, assuming incomplete")
            return False
        
        current_epoch = self.job_history.get("current_epoch", 0)
        total_epochs = self.job_history.get("total_epochs", 0)
        
        # If total_epochs is 0 or not set, try to get from config
        if total_epochs == 0:
            total_epochs = self.config.training.epochs
        
        # Check if current epoch meets or exceeds total epochs
        if current_epoch >= total_epochs and total_epochs > 0:
            self.logger.debug(f"Training completed: epoch {current_epoch} >= total {total_epochs}")
            return True
        else:
            self.logger.debug(f"Training incomplete: epoch {current_epoch} < total {total_epochs}")
            return False
    
    def get_resume_info(self) -> Optional[Dict[str, Any]]:
        """
        Get resume information for training.
        
        Returns:
            Dictionary with resume info (start_epoch, checkpoint_path) or None
        """
        return self.resume_info
    
    def set_completion_reason(self, reason: str) -> None:
        """
        Set the reason for training completion.
        
        This method should be called by training code when training ends for a
        specific reason OTHER than completing all epochs. This ensures the sweep
        system correctly identifies the run as COMPLETE (not needing resumption).
        
        Why This Matters
        ----------------
        Without setting a completion reason, the sweep system would see that
        current_epoch < total_epochs and incorrectly mark the run for resumption.
        By setting an explicit reason, we signal that the run is intentionally
        complete even though it didn't finish all epochs.
        
        Valid reasons (see exp.sweep_executor.CompletionReason):
        - "all_epochs": Completed all planned epochs (usually auto-detected)
        - "early_stopping": Patience-based early stopping triggered
        - "hyperband": Sweep controller (Hyperband/ASHA) pruned the run
        - "converged": Reached convergence threshold
        - "manual_stop": Intentionally stopped (not the same as interrupted)
        
        When to Call
        ------------
        - Early stopping callback triggers → set_completion_reason("early_stopping")
        - Convergence threshold reached → set_completion_reason("converged")
        - Manual stop requested by user → set_completion_reason("manual_stop")
        - Hyperband: Usually detected automatically via wandb.run.stopped
        
        Args:
            reason: Completion reason string (see valid reasons above)
            
        Example:
            # In your training loop's early stopping callback:
            if early_stopping.should_stop:
                exp_manager.set_completion_reason("early_stopping")
                break
        """
        self._completion_reason = reason
        self.logger.info(f"Training completion reason set: {reason}")
    

