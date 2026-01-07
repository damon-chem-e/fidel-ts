"""
Sweep Executor for running individual sweep trials.

This module provides the SweepExecutor class which handles the execution of
a single hyperparameter sweep trial. It is a pure executor - it runs training
and reports results back to the SweepManager.

Key Design Principles:
    - Pure execution: Does NOT manage state or registry
    - Single responsibility: Execute trial, detect completion, return result
    - No signals: Signal handling is managed by SweepManager
    - Clean interface: Takes parameters, returns SweepTrialResult

The SweepExecutor is designed to be called by SweepManager, which owns
all state and makes decisions about how to record results.

Usage:
    executor = SweepExecutor()
    result = executor.execute_trial(
        config_path="configs/experiment_suites/my_suite.yaml",
        sweep_params={"training.learning_rate": 0.001},
        shutdown_check=lambda: manager.shutdown_requested
    )
    
    if result.is_complete:
        print(f"Trial completed: {result.completion_reason}")
    elif result.should_resume:
        print(f"Trial interrupted at epoch {result.final_epoch}")
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from copy import deepcopy
from enum import Enum

# Add project root to path if needed
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class CompletionReason(str, Enum):
    """
    Reason why a sweep trial completed successfully.
    
    These indicate the trial is DONE and should NOT be resumed.
    """
    ALL_EPOCHS = "all_epochs"           # Finished all planned epochs
    EARLY_STOPPING = "early_stopping"   # Patience-based early stopping triggered
    HYPERBAND = "hyperband"             # Sweep controller pruned (Hyperband/ASHA)
    CONVERGED = "converged"             # Reached convergence threshold
    MANUAL_STOP = "manual_stop"         # Intentionally stopped (not interrupted)


class InterruptReason(str, Enum):
    """
    Reason why a sweep trial was interrupted.
    
    These indicate the trial should be RESUMED.
    """
    SLURM_TIMEOUT = "slurm_timeout"     # SLURM job time limit reached
    SIGTERM = "sigterm"                 # Generic SIGTERM signal
    SIGINT = "sigint"                   # Ctrl+C / SIGINT signal
    PREEMPTION = "preemption"           # Cloud/cluster preemption
    OOM = "oom"                         # Out of memory (if recoverable)
    UNKNOWN = "unknown"                 # Unknown interruption


@dataclass
class SweepTrialResult:
    """
    Result from executing a single sweep trial.
    
    This is the communication contract between SweepExecutor and SweepManager.
    The executor populates this dataclass after training completes (or fails),
    and the manager uses it to update the registry and wandb.
    
    Attributes:
        success: Whether training ran without exceptions
        interrupted: Whether training was interrupted by a signal
        final_epoch: Last completed epoch (0 if none completed)
        total_epochs: Total epochs that were planned
        completion_reason: Why training completed (None if not complete)
        interrupt_reason: Why training was interrupted (None if not interrupted)
        error: Error message if training failed
        experiment_id: Local experiment ID for this trial
        suite_id: Suite ID if part of a suite
        wandb_run_id: W&B run ID for this trial
        hyperparams: Hyperparameters used for this trial
        metrics: Final metrics from training
    
    Example:
        # Successful completion
        result = SweepTrialResult(
            success=True,
            final_epoch=100,
            total_epochs=100,
            completion_reason=CompletionReason.ALL_EPOCHS
        )
        
        # Interrupted by timeout
        result = SweepTrialResult(
            success=True,  # No exception
            interrupted=True,
            final_epoch=47,
            total_epochs=100,
            interrupt_reason=InterruptReason.SLURM_TIMEOUT
        )
    """
    success: bool = False
    interrupted: bool = False
    final_epoch: int = 0
    total_epochs: int = 0
    completion_reason: Optional[str] = None
    interrupt_reason: Optional[str] = None
    error: Optional[str] = None
    experiment_id: Optional[str] = None
    suite_id: Optional[str] = None
    wandb_run_id: Optional[str] = None
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_complete(self) -> bool:
        """
        Is this run complete (should NOT be resumed)?
        
        A run is complete if it has a valid completion reason. This includes:
        - Finished all epochs
        - Early stopping triggered
        - Hyperband/ASHA pruned
        - Convergence reached
        - Manual (intentional) stop
        
        Returns:
            True if the run is complete and should not be resumed
        """
        if self.completion_reason in [
            CompletionReason.ALL_EPOCHS.value,
            CompletionReason.EARLY_STOPPING.value,
            CompletionReason.HYPERBAND.value,
            CompletionReason.CONVERGED.value,
            CompletionReason.MANUAL_STOP.value,
            # Also accept string versions
            'all_epochs', 'early_stopping', 'hyperband', 'converged', 'manual_stop'
        ]:
            return True
        
        # Check if finished all epochs even without explicit reason
        if self.final_epoch >= self.total_epochs > 0:
            return True
        
        return False
    
    @property
    def should_resume(self) -> bool:
        """
        Should this run be resumed later?
        
        A run should be resumed if:
        1. It was interrupted (not a crash/failure)
        2. It's not already complete
        
        Note: We allow resuming even if final_epoch == 0 (e.g., timed out
        during first epoch). The ExperimentManager handles missing checkpoints
        gracefully by starting fresh.
        
        Returns:
            True if the run should be resumed
        """
        # Already complete? Don't resume
        if self.is_complete:
            return False
        
        # Any interruption is resumable (even if no progress yet)
        if self.interrupted:
            return True
        
        # Not successful but no error? Might be recoverable
        if not self.success and not self.error:
            return True
        
        return False
    
    @property
    def is_failed(self) -> bool:
        """
        Did this run fail (should NOT be resumed)?
        
        A run failed if:
        1. It's not complete
        2. It's not resumable (no progress or has error)
        
        Returns:
            True if the run failed and should not be resumed
        """
        return not self.is_complete and not self.should_resume


class SweepExecutor:
    """
    Executes a single sweep trial.
    
    This is a pure executor - it runs training and reports results.
    It does NOT:
        - Manage the sweep registry (that's SweepManager's job)
        - Handle signals directly (SweepManager sets shutdown flag)
        - Make decisions about resumption (returns result for manager to decide)
    
    The executor receives a shutdown_check callback from the manager which
    it can use to detect when training should stop gracefully.
    
    Attributes:
        None (stateless executor)
    
    Example:
        executor = SweepExecutor()
        
        # Execute a new trial
        result = executor.execute_trial(
            config_path="configs/my_suite.yaml",
            sweep_params={"training.learning_rate": 0.001},
            shutdown_check=lambda: manager.shutdown_requested
        )
        
        # Resume an incomplete trial
        result = executor.execute_trial(
            config_path="configs/my_suite.yaml",
            sweep_params=stored_params,
            resume_ids={"suite_id": "...", "experiment_id": "..."},
            shutdown_check=lambda: manager.shutdown_requested
        )
    """
    
    def __init__(self):
        """Initialize the executor (stateless)."""
        pass
    
    def execute_trial(
        self,
        config_path: str,
        sweep_params: Dict[str, Any],
        experiment_type: str = "pytorch",
        resume_ids: Optional[Dict[str, str]] = None,
        shutdown_check: Optional[Callable[[], bool]] = None,
        wandb_run_id: Optional[str] = None
    ) -> SweepTrialResult:
        """
        Execute a single sweep trial.
        
        This method runs training with the given hyperparameters and returns
        a SweepTrialResult indicating how training ended.
        
        Args:
            config_path: Path to config file (suite or single experiment)
            sweep_params: Hyperparameters from sweep controller
            experiment_type: Type of experiment (pytorch, lightning, llm, fm)
            resume_ids: Optional dict with suite_id and experiment_id for resuming
            shutdown_check: Callback that returns True if shutdown was requested.
                           Training should check this periodically and exit gracefully.
            wandb_run_id: Optional W&B run ID (for resuming same wandb run)
        
        Returns:
            SweepTrialResult with completion status and metadata
        
        Raises:
            Exception: Only if config loading fails. Training exceptions are
                      caught and returned in the result.
        """
        # Initialize result
        result = SweepTrialResult(hyperparams=sweep_params)
        
        try:
            # Determine if this is a suite or single experiment
            is_suite = self._is_suite_config(config_path)
            
            if is_suite:
                result = self._execute_suite_trial(
                    config_path=config_path,
                    sweep_params=sweep_params,
                    resume_ids=resume_ids,
                    shutdown_check=shutdown_check,
                    wandb_run_id=wandb_run_id
                )
            else:
                result = self._execute_single_trial(
                    config_path=config_path,
                    sweep_params=sweep_params,
                    experiment_type=experiment_type,
                    resume_ids=resume_ids,
                    shutdown_check=shutdown_check,
                    wandb_run_id=wandb_run_id
                )
            
            # Detect completion reason if not set
            if result.success and not result.completion_reason and not result.interrupted:
                result.completion_reason = self._detect_completion_reason(result)
            
            return result
            
        except Exception as e:
            # Catch any exception during execution
            result.success = False
            result.error = str(e)
            return result
    
    def _is_suite_config(self, config_path: str) -> bool:
        """
        Check if config path points to a suite config.
        
        Args:
            config_path: Path to config file
            
        Returns:
            True if suite config, False if single experiment config
        """
        from runs.suite_executor import load_suite_config
        from cli.config.loader import load_config
        
        try:
            config = load_suite_config(config_path)
            return 'suite' in config
        except Exception:
            try:
                load_config(config_path)
                return False
            except Exception:
                return 'suite' in str(config_path).lower()
    
    def _execute_suite_trial(
        self,
        config_path: str,
        sweep_params: Dict[str, Any],
        resume_ids: Optional[Dict[str, str]] = None,
        shutdown_check: Optional[Callable[[], bool]] = None,
        wandb_run_id: Optional[str] = None
    ) -> SweepTrialResult:
        """
        Execute a suite sweep trial.
        
        Args:
            config_path: Path to suite config
            sweep_params: Hyperparameters from sweep controller
            resume_ids: Optional dict with suite_id and experiment_id for resuming
            shutdown_check: Callback for graceful shutdown
            wandb_run_id: Optional W&B run ID for resuming
            
        Returns:
            SweepTrialResult with execution results
        """
        from runs.suite_executor import SuiteExecutor, load_suite_config
        
        result = SweepTrialResult(hyperparams=sweep_params)
        
        # Load base suite config
        suite_config = load_suite_config(config_path)
        
        # Apply sweep parameters
        suite_config = self._apply_sweep_params_to_suite(suite_config, sweep_params)
        
        # Get output directory and total epochs for tracking
        output_dir = Path(suite_config.get('suite', {}).get('output_dir', './output')).resolve()
        experiments = suite_config.get('suite', {}).get('experiments', [])
        first_exp = next((e for e in experiments if e.get('enabled', True)), experiments[0] if experiments else {})
        
        # Get total epochs - must be set in config
        epochs = first_exp.get('overrides', {}).get('training', {}).get('epochs')
        if epochs is None:
            result.error = "Configuration error: training.epochs must be set in experiment overrides"
            return result
        result.total_epochs = epochs
        
        # Set resume IDs if provided
        if resume_ids:
            suite_config['suite']['resume_suite_id'] = resume_ids.get('suite_id')
            for exp in suite_config['suite'].get('experiments', []):
                if exp.get('enabled', True):
                    exp_name = exp.get('name', '')
                    exp_ids = resume_ids.get('experiment_ids', {})
                    for name, exp_id in exp_ids.items():
                        if exp_name in name or name in exp_name:
                            exp.setdefault('overrides', {})['resume_experiment_id'] = exp_id
                            break
        
        try:
            # Check for shutdown before starting
            if shutdown_check and shutdown_check():
                result.interrupted = True
                result.interrupt_reason = InterruptReason.SIGTERM.value
                return result
            
            # First: Initialize to get IDs (if not resuming)
            if not resume_ids:
                executor = SuiteExecutor(suite_config, init_only=True, return_ids=True)
                init_result = executor.execute()
                
                if not init_result:
                    result.error = "Failed to initialize experiment"
                    return result
                
                result.suite_id = init_result['suite_id']
                result.experiment_id = list(init_result['experiment_ids'].values())[0] if init_result['experiment_ids'] else None
                
                # Set resume IDs for execution
                suite_config['suite']['resume_suite_id'] = result.suite_id
                for exp_name, exp_id in init_result['experiment_ids'].items():
                    for exp in suite_config['suite']['experiments']:
                        if exp.get('name') == exp_name or f"{exp.get('name')}_output" in exp_name:
                            exp.setdefault('overrides', {})['resume_experiment_id'] = exp_id
                            break
            else:
                result.suite_id = resume_ids.get('suite_id')
                exp_ids = resume_ids.get('experiment_ids', {})
                result.experiment_id = list(exp_ids.values())[0] if exp_ids else resume_ids.get('experiment_id')
            
            # Check for shutdown again
            if shutdown_check and shutdown_check():
                result.interrupted = True
                result.interrupt_reason = InterruptReason.SIGTERM.value
                return result
            
            # Execute training
            executor = SuiteExecutor(suite_config, init_only=False)
            executor.execute()
            
            # If we get here, training completed without exception
            result.success = True
            
            # Note: We don't set final_epoch here because we don't know if training
            # completed all epochs or stopped early (early stopping, hyperband, etc.).
            # The final_epoch will be read from job_history.json by SweepManager
            # after training completes. This ensures we get the actual last completed
            # epoch as tracked by ExperimentManager.
            
            # Check if shutdown was requested during training
            if shutdown_check and shutdown_check():
                result.interrupted = True
                result.interrupt_reason = InterruptReason.SIGTERM.value
            
        except Exception as e:
            result.error = str(e)
            # Check if this was due to shutdown
            if shutdown_check and shutdown_check():
                result.interrupted = True
                result.interrupt_reason = InterruptReason.SIGTERM.value
                result.success = True  # Clean exit, not a failure
                result.error = None
        
        return result
    
    def _execute_single_trial(
        self,
        config_path: str,
        sweep_params: Dict[str, Any],
        experiment_type: str = "pytorch",
        resume_ids: Optional[Dict[str, str]] = None,
        shutdown_check: Optional[Callable[[], bool]] = None,
        wandb_run_id: Optional[str] = None
    ) -> SweepTrialResult:
        """
        Execute a single experiment sweep trial.
        
        Args:
            config_path: Path to experiment config
            sweep_params: Hyperparameters from sweep controller
            experiment_type: Type of experiment
            resume_ids: Optional dict for resuming
            shutdown_check: Callback for graceful shutdown
            wandb_run_id: Optional W&B run ID
            
        Returns:
            SweepTrialResult with execution results
        """
        from cli.config.loader import load_config
        from cli.config.models import ExperimentConfig
        
        result = SweepTrialResult(hyperparams=sweep_params)
        
        # Load base experiment config
        experiment_config = load_config(config_path)
        
        # Apply sweep parameters
        experiment_config = self._apply_sweep_params_to_experiment(experiment_config, sweep_params)
        
        # Get total epochs - must be set in config
        if not hasattr(experiment_config, 'training') or not hasattr(experiment_config.training, 'epochs'):
            result.error = "Configuration error: experiment_config.training.epochs must be set"
            return result
        result.total_epochs = experiment_config.training.epochs
        
        # Set resume IDs if provided
        if resume_ids:
            experiment_config.resume_experiment_id = resume_ids.get('experiment_id')
            if resume_ids.get('suite_id'):
                experiment_config.resume_suite_id = resume_ids.get('suite_id')
        
        # Import appropriate run function
        if experiment_type == "pytorch":
            from runs.pytorch import run
        elif experiment_type == "lightning":
            from runs.lightning import run
        elif experiment_type == "llm":
            from runs.llm import run
        elif experiment_type == "fm":
            from runs.fm import run
        else:
            result.error = f"Unknown experiment type: {experiment_type}"
            return result
        
        try:
            # Check for shutdown before starting
            if shutdown_check and shutdown_check():
                result.interrupted = True
                result.interrupt_reason = InterruptReason.SIGTERM.value
                return result
            
            # Initialize to get IDs if not resuming
            if not resume_ids:
                init_result = run(experiment_config, init_only=True, return_ids=True)
                
                if not init_result:
                    result.error = "Failed to initialize experiment"
                    return result
                
                result.experiment_id = init_result['experiment_id']
                result.suite_id = init_result.get('suite_name')
                
                # Set resume ID for execution
                experiment_config.resume_experiment_id = result.experiment_id
                if result.suite_id:
                    experiment_config.resume_suite_id = result.suite_id
            else:
                result.experiment_id = resume_ids.get('experiment_id')
                result.suite_id = resume_ids.get('suite_id')
            
            # Check for shutdown again
            if shutdown_check and shutdown_check():
                result.interrupted = True
                result.interrupt_reason = InterruptReason.SIGTERM.value
                return result
            
            # Execute training
            run(experiment_config, init_only=False)
            
            # If we get here, training completed without exception
            result.success = True
            
            # Note: We don't set final_epoch here because we don't know if training
            # completed all epochs or stopped early (early stopping, hyperband, etc.).
            # The final_epoch will be read from job_history.json by SweepManager
            # after training completes. This ensures we get the actual last completed
            # epoch as tracked by ExperimentManager.
            
            # Check if shutdown was requested during training
            if shutdown_check and shutdown_check():
                result.interrupted = True
                result.interrupt_reason = InterruptReason.SIGTERM.value
                
        except Exception as e:
            result.error = str(e)
            if shutdown_check and shutdown_check():
                result.interrupted = True
                result.interrupt_reason = InterruptReason.SIGTERM.value
                result.success = True
                result.error = None
        
        return result
    
    def _apply_sweep_params_to_suite(
        self,
        suite_config: Dict[str, Any],
        sweep_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply wandb sweep hyperparameters to suite config.
        
        Args:
            suite_config: Suite configuration dictionary
            sweep_params: Hyperparameters from wandb.config
            
        Returns:
            Modified suite config with sweep parameters applied
        """
        suite_config = deepcopy(suite_config)
        
        experiments = suite_config.get('suite', {}).get('experiments', [])
        if not experiments:
            return suite_config
        
        # Find first enabled experiment
        exp = next((e for e in experiments if e.get('enabled', True)), None)
        if not exp:
            return suite_config
        
        overrides = exp.setdefault('overrides', {})
        
        # Apply sweep parameters using dot notation
        for key, value in sweep_params.items():
            if key.startswith('_'):
                continue
            
            keys = key.split('.')
            if len(keys) == 1:
                overrides[key] = value
            else:
                current = overrides
                for k in keys[:-1]:
                    current = current.setdefault(k, {})
                current[keys[-1]] = value
        
        return suite_config
    
    def _apply_sweep_params_to_experiment(
        self,
        experiment_config,
        sweep_params: Dict[str, Any]
    ):
        """
        Apply wandb sweep hyperparameters to single experiment config.
        
        Args:
            experiment_config: ExperimentConfig instance
            sweep_params: Hyperparameters from wandb.config
            
        Returns:
            Modified ExperimentConfig
        """
        from cli.config.models import ExperimentConfig
        
        config_dict = experiment_config.model_dump()
        
        for key, value in sweep_params.items():
            if key.startswith('_'):
                continue
            
            keys = key.split('.')
            if len(keys) == 1:
                config_dict[key] = value
            else:
                current = config_dict
                for k in keys[:-1]:
                    current = current.setdefault(k, {})
                current[keys[-1]] = value
        
        return ExperimentConfig(**config_dict)
    
    def _read_final_epoch_from_job_history(
        self,
        experiment_id: str,
        suite_id: Optional[str],
        output_dir: Path
    ) -> Optional[int]:
        """
        Read the final completed epoch from job_history.json.
        
        The ExperimentManager updates job_history.json after each epoch,
        so this gives us the authoritative final epoch even if training
        was interrupted mid-epoch.
        
        Args:
            experiment_id: Experiment ID
            suite_id: Optional suite ID (if part of suite)
            output_dir: Base output directory
            
        Returns:
            Final epoch number, or None if job_history not found
        """
        import json
        
        # Determine experiment directory path
        if suite_id:
            exp_dir = output_dir / suite_id / experiment_id
        else:
            exp_dir = output_dir / experiment_id
        
        job_history_path = exp_dir / "job_history.json"
        
        if not job_history_path.exists():
            return None
        
        try:
            with open(job_history_path, 'r', encoding='utf-8') as f:
                job_history = json.load(f)
            return job_history.get('current_epoch', 0)
        except Exception:
            return None
    
    def _detect_completion_reason(self, result: SweepTrialResult) -> Optional[str]:
        """
        Detect why training completed if no explicit reason was set.
        
        This checks in priority order:
        1. Check if wandb.run.stopped (Hyperband pruning)
        2. Check if all epochs completed
        
        Args:
            result: Partially filled SweepTrialResult
            
        Returns:
            Completion reason string or None
        """
        # Check Hyperband pruning
        try:
            import wandb
            if wandb.run and getattr(wandb.run, 'stopped', False):
                return CompletionReason.HYPERBAND.value
        except Exception:
            pass
        
        # Check if all epochs completed (only if final_epoch is set)
        if result.final_epoch > 0 and result.final_epoch >= result.total_epochs:
            return CompletionReason.ALL_EPOCHS.value
        
        return None
