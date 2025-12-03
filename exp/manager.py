"""
Experiment Manager for comprehensive experiment tracking.

This module provides the ExperimentManager class that handles:
- Experiment directory creation and management
- Metadata capture (git hash, timestamp, configs)
- Local metrics logging
- WandB integration for cloud-based experiment tracking
- Complete experiment reproducibility
"""

import os
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
import subprocess
import yaml

from cli.config.models import ExperimentConfig


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
        output_dir: str = "./outputs",
        experiment_name: Optional[str] = None,
        job_id: Optional[str] = None,
        job_name: Optional[str] = None
    ):
        """
        Initialize ExperimentManager.
        
        Args:
            config: ExperimentConfig instance containing complete experiment configuration
            output_dir: Base directory for experiment outputs
            experiment_name: Optional experiment name (overrides config)
            job_id: Optional job ID from cluster/scheduler
            job_name: Optional job name from cluster/scheduler
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name or config.experiment_name
        self.job_id = job_id or config.job_id
        self.job_name = job_name or config.job_name
        
        # Generate experiment ID from config hash
        self.experiment_id = self._generate_experiment_id()
        
        # Create experiment directory structure
        self.experiment_dir = self.output_dir / self.experiment_id
        self._create_experiment_structure()
        
        # Capture metadata
        self.metadata = self._capture_metadata()
        self._save_metadata()
        
        # Metrics storage
        self.metrics: Dict[str, Any] = {}
        
        # Initialize wandb if enabled
        self.wandb_run = None
        if config.wandb.enabled:
            self._init_wandb()
    
    def _generate_experiment_id(self) -> str:
        """
        Generate unique experiment ID from config hash.
        
        Returns:
            Experiment ID in format: {timestamp}_{short_hash}
        """
        # Serialize config to dict (excluding metadata fields that don't affect reproducibility)
        config_dict = self.config.model_dump(exclude={
            'experiment_name', 'job_id', 'job_name', 'random_seed'
        })
        
        # Convert to JSON with sorted keys for deterministic hashing
        config_json = json.dumps(config_dict, sort_keys=True, default=str)
        
        # Generate hash
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:12]
        
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
        metadata_path = self.experiment_dir / "metadata" / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        # Save complete config hierarchy
        self._save_config_hierarchy()
    
    def _save_config_hierarchy(self) -> None:
        """Save complete config hierarchy to experiment directory."""
        # Save primary config
        primary_config_path = self.experiment_dir / "configs" / "experiment_config.yaml"
        self.config.to_yaml(primary_config_path)
        
        # Load and save nested configs (model, data, etc.)
        self._copy_config_file(self.config.model.config_path, "model_config.yaml")
        self._copy_config_file(self.config.data.config_path, "data_config.yaml")
        
        # Save nested configs if they exist
        if self.config.plotting:
            self._copy_config_file(self.config.plotting, "plotting_config.yaml")
        if self.config.evaluation:
            self._copy_config_file(self.config.evaluation, "evaluation_config.yaml")
    
    def _copy_config_file(self, source_path: str, dest_name: str) -> None:
        """
        Copy a config file to experiment directory.
        
        Args:
            source_path: Path to source config file
            dest_name: Destination filename in experiment configs directory
        """
        source = Path(source_path)
        if not source.is_absolute():
            source = Path.cwd() / source
        
        if source.exists():
            dest = self.experiment_dir / "configs" / dest_name
            shutil.copy2(source, dest)
    
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
            
            # Initialize wandb run
            self.wandb_run = wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                name=self.experiment_id,
                tags=self.config.wandb.tags,
                notes=self.config.wandb.notes,
                config=wandb_config,
                mode=self.config.wandb.mode,
                dir=str(self.experiment_dir / "wandb"),
                reinit=False
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
        
        # Log to wandb if enabled
        if self.wandb_run is not None:
            try:
                if step is not None:
                    self.wandb_run.log({name: value}, step=step)
                else:
                    self.wandb_run.log({name: value})
            except Exception as e:
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
    
    def _save_metrics(self) -> None:
        """Save metrics to JSON file."""
        metrics_path = self.experiment_dir / "metrics" / "metrics.json"
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
    
    def end_experiment(self, final_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Finalize experiment, log final metrics, and close wandb run.
        
        Args:
            final_metrics: Optional dictionary of final metrics to log
        """
        # Log final metrics if provided
        if final_metrics:
            self.log_metrics(final_metrics)
            
            # Update wandb summary with final metrics
            if self.wandb_run is not None:
                try:
                    self.wandb_run.summary.update(final_metrics)
                except Exception as e:
                    print(f"Warning: Failed to update wandb summary: {e}")
        
        # Close wandb run
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception as e:
                print(f"Warning: Failed to finish wandb run: {e}")
        
        # Save final metadata
        self.metadata["end_timestamp"] = datetime.now().isoformat()
        metadata_path = self.experiment_dir / "metadata" / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def get_experiment_id(self) -> str:
        """Get experiment ID."""
        return self.experiment_id
    
    def get_experiment_dir(self) -> Path:
        """Get experiment directory path."""
        return self.experiment_dir

