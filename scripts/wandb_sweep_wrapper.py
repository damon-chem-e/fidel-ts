#!/usr/bin/env python3
"""
WandB Sweep Wrapper for Experiment Suite System

This script integrates wandb sweeps with the experiment suite system, supporting:
- Hyperparameter sweeps on suite configs or single experiment configs
- Automatic initialization and resume ID management
- Job status tracking (not_started, initialized, running, completed)

Usage:
    # Initialize sweep first
    wandb sweep sweep_config.yaml --project fidel-ts
    
    # Run agent (this script is called by wandb agent)
    wandb agent SWEEP_ID --project fidel-ts
"""

import os
import sys
import signal
from pathlib import Path
from typing import Dict, Any
from copy import deepcopy
from datetime import datetime

import wandb


# Global flag for graceful shutdown on SLURM timeout
_shutdown_requested = False


def setup_signal_handlers():
    """
    Set up signal handlers for graceful shutdown on SLURM timeout.
    
    SLURM sends SIGTERM before killing jobs. We catch this to mark runs
    for resumption in both wandb and the local registry.
    """
    def signal_handler(signum, frame):
        global _shutdown_requested, _sweep_registry
        _shutdown_requested = True
        signal_name = signal.Signals(signum).name
        print(f"\n[Sweep] Received {signal_name} signal. Requesting graceful shutdown...")
        
        # Determine interrupt reason from signal
        if signum == signal.SIGTERM:
            interrupt_reason = InterruptReason.SLURM_TIMEOUT.value
        elif signum == signal.SIGINT:
            interrupt_reason = InterruptReason.SIGINT.value
        else:
            interrupt_reason = InterruptReason.SIGTERM.value
        
        # Mark current wandb run for resumption
        if wandb.run is not None:
            try:
                wandb.run.config.update({
                    '_needs_resume': True,
                    '_interrupted_at': datetime.now().isoformat(),
                    '_interrupted_by': signal_name
                }, allow_val_change=True)
                print(f"[Sweep] Marked run {wandb.run.id} for resumption in wandb")
            except Exception as e:
                print(f"[Sweep] Warning: Could not mark run for resumption in wandb: {e}")
            
            # Mark in local registry (this is the authoritative source)
            if _sweep_registry is not None:
                try:
                    run_info = _sweep_registry.get_run(wandb.run.id)
                    if run_info:
                        _sweep_registry.mark_needs_resume(
                            wandb_run_id=wandb.run.id,
                            interrupt_reason=interrupt_reason,
                            final_epoch=run_info.get('final_epoch', 0) or 0,
                            total_epochs=run_info.get('total_epochs', 0) or 0
                        )
                        print(f"[Sweep] Marked run {wandb.run.id} for resumption in local registry")
                except Exception as e:
                    print(f"[Sweep] Warning: Could not mark run in local registry: {e}")
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    try:
        signal.signal(signal.SIGUSR1, signal_handler)
    except (AttributeError, ValueError):
        pass  # Not available on Windows


def is_shutdown_requested() -> bool:
    """Check if graceful shutdown has been requested."""
    return _shutdown_requested

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cli.config.loader import load_config
from runs.suite_executor import SuiteExecutor, load_suite_config
from cli.config.models import ExperimentConfig
from exp.sweep_registry import SweepRegistry, get_location, CompletionReason, InterruptReason


# Global registry instance (set when sweep starts)
_sweep_registry: SweepRegistry = None


# Job status constants
STATUS_NOT_STARTED = "not_started"
STATUS_INITIALIZED = "initialized"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"  # For SLURM timeout / signal-based interruption


def is_suite_config(config_path: str) -> bool:
    """
    Check if config path points to a suite config.
    
    Args:
        config_path: Path to config file
        
    Returns:
        True if suite config, False if single experiment config
    """
    try:
        config = load_suite_config(config_path)
        return 'suite' in config
    except Exception:
        # If loading as suite fails, try as single experiment
        try:
            load_config(config_path)
            return False
        except Exception:
            # If both fail, assume suite based on path
            return 'suite' in str(config_path).lower()


def apply_sweep_parameters_to_suite(suite_config: Dict[str, Any], sweep_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply wandb sweep hyperparameters to suite config.
    
    Args:
        suite_config: Suite configuration dictionary
        sweep_params: Hyperparameters from wandb.config (e.g., {'training.learning_rate': 0.001})
        
    Returns:
        Modified suite config with sweep parameters applied
    """
    suite_config = deepcopy(suite_config)
    
    # Get first experiment (for single-experiment suites, or use first enabled experiment)
    experiments = suite_config.get('suite', {}).get('experiments', [])
    if not experiments:
        raise ValueError("Suite config has no experiments")
    
    # Find first enabled experiment
    exp = None
    for e in experiments:
        if e.get('enabled', True):
            exp = e
            break
    
    if not exp:
        raise ValueError("No enabled experiments found in suite")
    
    overrides = exp.setdefault('overrides', {})
    
    # Apply sweep parameters (handle nested keys like 'training.learning_rate' or 'model_config_overrides.e_layers')
    for key, value in sweep_params.items():
        if key.startswith('_'):  # Skip internal wandb metadata
            continue
            
        keys = key.split('.')
        if len(keys) == 1:
            # Top-level parameter
            overrides[key] = value
        elif len(keys) == 2:
            # Nested parameter (e.g., 'training.learning_rate' or 'model_config_overrides.e_layers')
            section = overrides.setdefault(keys[0], {})
            section[keys[1]] = value
        else:
            # Deeper nesting - handle recursively (e.g., 'model_config_overrides.some.nested.param')
            current = overrides
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = value
    
    return suite_config


def apply_sweep_parameters_to_experiment(experiment_config: ExperimentConfig, sweep_params: Dict[str, Any]) -> ExperimentConfig:
    """
    Apply wandb sweep hyperparameters to single experiment config.
    
    Args:
        experiment_config: ExperimentConfig instance
        sweep_params: Hyperparameters from wandb.config
        
    Returns:
        Modified ExperimentConfig with sweep parameters applied
    """
    # Convert to dict, modify, then recreate
    config_dict = experiment_config.model_dump()
    
    # Apply sweep parameters
    for key, value in sweep_params.items():
        if key.startswith('_'):  # Skip internal wandb metadata
            continue
            
        keys = key.split('.')
        if len(keys) == 1:
            config_dict[key] = value
        elif len(keys) == 2:
            section = config_dict.setdefault(keys[0], {})
            section[keys[1]] = value
        else:
            # Deeper nesting
            current = config_dict
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = value
    
    # Recreate ExperimentConfig
    return ExperimentConfig(**config_dict)


def get_job_status(wandb_run) -> str:
    """
    Get job status from wandb run config.
    
    Args:
        wandb_run: wandb run object
        
    Returns:
        Job status string
    """
    return wandb_run.config.get('_job_status', STATUS_NOT_STARTED)


def set_job_status(wandb_run, status: str):
    """
    Set job status in wandb run config and summary.
    
    Updates both config and summary to handle the race condition where
    wandb.finish() may be called before we can update config. Summary
    updates are more reliable as they persist even after the run ends.
    
    Args:
        wandb_run: wandb run object
        status: Job status string
    """
    config_updated = False
    summary_updated = False
    
    # Try to update config (primary method, but may fail if run is finished)
    try:
        wandb_run.config.update({'_job_status': status}, allow_val_change=True)
        config_updated = True
    except Exception as e:
        # Run may already be finished - this is expected in many cases
        pass
    
    # Also update summary (backup method, more reliable for completion detection)
    # The smart_sweep_agent checks both config._job_status AND summary._sweep_completed
    try:
        summary_update = {'_job_status_summary': status}
        if status == STATUS_COMPLETED:
            summary_update['_sweep_completed'] = True
            summary_update['_sweep_completion_time'] = datetime.now().isoformat()
        elif status == STATUS_FAILED:
            summary_update['_sweep_completed'] = False
            summary_update['_sweep_failed'] = True
        elif status == STATUS_INTERRUPTED:
            summary_update['_sweep_completed'] = False
            summary_update['_needs_resume'] = True
        
        wandb_run.summary.update(summary_update)
        summary_updated = True
    except Exception as e:
        pass
    
    # Log status of update attempts
    if not config_updated and not summary_updated:
        print(f"[Sweep] Warning: Could not update job status to '{status}' (run may be finished)")
    elif not config_updated:
        print(f"[Sweep] Note: Updated job status to '{status}' in summary only (config update failed)")


def run_suite_sweep(suite_config_path: str, sweep_params: Dict[str, Any]) -> None:
    """
    Run a wandb sweep on a suite config.
    
    Args:
        suite_config_path: Path to suite config file
        sweep_params: Hyperparameters from wandb.config
    """
    global _sweep_registry
    
    wandb_run = wandb.run
    location = get_location()
    
    # Store location in wandb config (for central tracking)
    try:
        wandb_run.config.update({'_location': location}, allow_val_change=True)
    except Exception:
        pass
    
    # Load base suite config
    suite_config = load_suite_config(suite_config_path)
    
    # Apply sweep parameters
    suite_config = apply_sweep_parameters_to_suite(suite_config, sweep_params)
    
    # Get output directory for registry
    output_dir = Path(suite_config.get('suite', {}).get('output_dir', './output')).resolve()
    
    # Check job status
    job_status = get_job_status(wandb_run)
    suite_id = None
    experiment_ids = {}
    
    if job_status == STATUS_NOT_STARTED:
        # Phase 1: Initialize to get IDs
        print("[Sweep] Initializing experiment structure...")
        executor = SuiteExecutor(suite_config, init_only=True, return_ids=True)
        result = executor.execute()
        
        if not result:
            raise RuntimeError("Failed to get experiment IDs from initialization")
        
        suite_id = result['suite_id']
        experiment_ids = result['experiment_ids']
        
        # Store IDs and status in wandb (includes location for central tracking)
        wandb_run.config.update({
            '_suite_id': suite_id,
            '_experiment_ids': experiment_ids,
            '_job_status': STATUS_INITIALIZED,
            '_location': location,
        }, allow_val_change=True)
        
        print(f"[Sweep] Initialized suite: {suite_id}")
        print(f"[Sweep] Experiment IDs: {experiment_ids}")
        print(f"[Sweep] Location: {location}")
        
        # Initialize local registry for this sweep
        suite_dir = output_dir / suite_id
        _sweep_registry = SweepRegistry(
            suite_dir=str(suite_dir),
            sweep_id=wandb_run.sweep_id,
            location=location
        )
        
        # Register this run in the local registry
        _sweep_registry.register_run(
            wandb_run_id=wandb_run.id,
            experiment_id=list(experiment_ids.values())[0] if experiment_ids else suite_id,
            hyperparams=sweep_params,
            suite_id=suite_id
        )
        print(f"[Sweep] Registered run in local registry")
        
        # Update suite config with resume IDs
        suite_config['suite']['resume_suite_id'] = suite_id
        for exp_name, exp_id in experiment_ids.items():
            for exp in suite_config['suite']['experiments']:
                if exp.get('name') == exp_name or f"{exp.get('name')}_output" in exp_name:
                    exp.setdefault('overrides', {})['resume_experiment_id'] = exp_id
                    break
        
        set_job_status(wandb_run, STATUS_RUNNING)
    
    elif job_status in [STATUS_INITIALIZED, STATUS_RUNNING]:
        # Resume from initialization or running - get IDs from wandb config
        suite_id = wandb_run.config.get('_suite_id')
        experiment_ids = wandb_run.config.get('_experiment_ids', {})
        
        if not suite_id or not experiment_ids:
            raise RuntimeError("Missing suite_id or experiment_ids in wandb config. Cannot resume.")
        
        # Initialize registry for this sweep (may already exist)
        suite_dir = output_dir / suite_id
        _sweep_registry = SweepRegistry(
            suite_dir=str(suite_dir),
            sweep_id=wandb_run.sweep_id,
            location=location
        )
        
        # Update registry if run exists, otherwise register it
        if not _sweep_registry.run_exists(wandb_run.id):
            _sweep_registry.register_run(
                wandb_run_id=wandb_run.id,
                experiment_id=list(experiment_ids.values())[0] if experiment_ids else suite_id,
                hyperparams=sweep_params,
                suite_id=suite_id
            )
        
        # Set resume IDs in config
        suite_config['suite']['resume_suite_id'] = suite_id
        for exp_name, exp_id in experiment_ids.items():
            for exp in suite_config['suite']['experiments']:
                if exp.get('name') == exp_name or f"{exp.get('name')}_output" in exp_name:
                    exp.setdefault('overrides', {})['resume_experiment_id'] = exp_id
                    break
        
        if job_status == STATUS_INITIALIZED:
            set_job_status(wandb_run, STATUS_RUNNING)
        else:
            print("[Sweep] Job already marked as running, continuing...")
    
    # Get total epochs for registry tracking
    total_epochs = suite_config.get('suite', {}).get('experiments', [{}])[0].get('overrides', {}).get('training', {}).get('epochs', 100)
    
    # Mark as running in registry
    if _sweep_registry is not None:
        _sweep_registry.mark_running(wandb_run.id, total_epochs)
    
    # Phase 2: Run actual training
    print("[Sweep] Starting training execution...")
    try:
        executor = SuiteExecutor(suite_config, init_only=False)
        executor.execute()
        
        # Mark complete in local registry (authoritative)
        if _sweep_registry is not None:
            _sweep_registry.mark_complete(
                wandb_run_id=wandb_run.id,
                completion_reason=CompletionReason.ALL_EPOCHS.value,  # Will be overridden by ExperimentManager if different
                final_epoch=total_epochs,
                total_epochs=total_epochs
            )
            print("[Sweep] Marked run as complete in local registry")
        
        set_job_status(wandb_run, STATUS_COMPLETED)
        print("[Sweep] Training completed successfully")
    except Exception as e:
        # Mark failed in local registry
        if _sweep_registry is not None:
            _sweep_registry.mark_failed(wandb_run.id, error_message=str(e))
        
        set_job_status(wandb_run, STATUS_FAILED)
        print(f"[Sweep] Training failed: {e}")
        raise


def run_single_experiment_sweep(experiment_config_path: str, sweep_params: Dict[str, Any], experiment_type: str = "pytorch") -> None:
    """
    Run a wandb sweep on a single experiment config.
    
    Args:
        experiment_config_path: Path to single experiment config file
        sweep_params: Hyperparameters from wandb.config
        experiment_type: Experiment type (pytorch, lightning, llm, fm)
    """
    global _sweep_registry
    
    wandb_run = wandb.run
    location = get_location()
    
    # Store location in wandb config (for central tracking)
    try:
        wandb_run.config.update({'_location': location}, allow_val_change=True)
    except Exception:
        pass
    
    # Load base experiment config
    experiment_config = load_config(experiment_config_path)
    
    # Apply sweep parameters
    experiment_config = apply_sweep_parameters_to_experiment(experiment_config, sweep_params)
    
    # Get output directory for registry
    output_dir = Path(experiment_config.output_dir if hasattr(experiment_config, 'output_dir') else './output').resolve()
    
    # Check job status
    job_status = get_job_status(wandb_run)
    experiment_id = None
    suite_name = None
    
    if job_status == STATUS_NOT_STARTED:
        # Phase 1: Initialize to get IDs
        print("[Sweep] Initializing experiment structure...")
        
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
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        
        result = run(experiment_config, init_only=True, return_ids=True)
        
        if not result:
            raise RuntimeError("Failed to get experiment ID from initialization")
        
        experiment_id = result['experiment_id']
        suite_name = result.get('suite_name')
        
        # Store IDs and status in wandb (includes location for central tracking)
        wandb_run.config.update({
            '_experiment_id': experiment_id,
            '_suite_name': suite_name,
            '_job_status': STATUS_INITIALIZED,
            '_location': location,
        }, allow_val_change=True)
        
        print(f"[Sweep] Initialized experiment: {experiment_id}")
        print(f"[Sweep] Location: {location}")
        
        # Initialize local registry for this sweep
        # For single experiments, registry goes in the suite dir or output dir
        if suite_name:
            registry_dir = output_dir / suite_name
        else:
            registry_dir = output_dir / f"sweep_{wandb_run.sweep_id}"
        
        _sweep_registry = SweepRegistry(
            suite_dir=str(registry_dir),
            sweep_id=wandb_run.sweep_id,
            location=location
        )
        
        # Register this run in the local registry
        _sweep_registry.register_run(
            wandb_run_id=wandb_run.id,
            experiment_id=experiment_id,
            hyperparams=sweep_params,
            suite_id=suite_name
        )
        print("[Sweep] Registered run in local registry")
        
        # Set resume ID
        experiment_config.resume_experiment_id = experiment_id
        if suite_name:
            experiment_config.resume_suite_id = suite_name
        
        set_job_status(wandb_run, STATUS_RUNNING)
    
    elif job_status in [STATUS_INITIALIZED, STATUS_RUNNING]:
        # Resume from initialization or running
        experiment_id = wandb_run.config.get('_experiment_id')
        suite_name = wandb_run.config.get('_suite_name')
        
        if not experiment_id:
            raise RuntimeError("Missing experiment_id in wandb config. Cannot resume.")
        
        # Initialize registry for this sweep (may already exist)
        if suite_name:
            registry_dir = output_dir / suite_name
        else:
            registry_dir = output_dir / f"sweep_{wandb_run.sweep_id}"
        
        _sweep_registry = SweepRegistry(
            suite_dir=str(registry_dir),
            sweep_id=wandb_run.sweep_id,
            location=location
        )
        
        # Update registry if run exists, otherwise register it
        if not _sweep_registry.run_exists(wandb_run.id):
            _sweep_registry.register_run(
                wandb_run_id=wandb_run.id,
                experiment_id=experiment_id,
                hyperparams=sweep_params,
                suite_id=suite_name
            )
        
        experiment_config.resume_experiment_id = experiment_id
        if suite_name:
            experiment_config.resume_suite_id = suite_name
        
        if job_status == STATUS_INITIALIZED:
            set_job_status(wandb_run, STATUS_RUNNING)
        else:
            print("[Sweep] Job already marked as running, continuing...")
    
    # Get total epochs for registry tracking
    total_epochs = experiment_config.training.epochs if hasattr(experiment_config, 'training') else 100
    
    # Mark as running in registry
    if _sweep_registry is not None:
        _sweep_registry.mark_running(wandb_run.id, total_epochs)
    
    # Phase 2: Run actual training
    print("[Sweep] Starting training execution...")
    try:
        if experiment_type == "pytorch":
            from runs.pytorch import run
        elif experiment_type == "lightning":
            from runs.lightning import run
        elif experiment_type == "llm":
            from runs.llm import run
        elif experiment_type == "fm":
            from runs.fm import run
        
        run(experiment_config, init_only=False)
        
        # Mark complete in local registry (authoritative)
        if _sweep_registry is not None:
            _sweep_registry.mark_complete(
                wandb_run_id=wandb_run.id,
                completion_reason=CompletionReason.ALL_EPOCHS.value,
                final_epoch=total_epochs,
                total_epochs=total_epochs
            )
            print("[Sweep] Marked run as complete in local registry")
        
        set_job_status(wandb_run, STATUS_COMPLETED)
        print("[Sweep] Training completed successfully")
    except Exception as e:
        # Mark failed in local registry
        if _sweep_registry is not None:
            _sweep_registry.mark_failed(wandb_run.id, error_message=str(e))
        
        set_job_status(wandb_run, STATUS_FAILED)
        print(f"[Sweep] Training failed: {e}")
        raise


def main():
    """
    Main entry point for wandb sweep wrapper.
    
    Expects environment variables or wandb config:
    - WANDB_SWEEP_CONFIG_PATH: Path to base config (suite or experiment)
    - WANDB_SWEEP_EXPERIMENT_TYPE: Experiment type if using single experiment (pytorch, lightning, llm, fm)
    """
    # Set up signal handlers for graceful shutdown on SLURM timeout
    setup_signal_handlers()
    
    # Initialize wandb run (wandb agent handles this, but we need to access it)
    wandb.init()
    wandb_run = wandb.run
    
    # Get config path from multiple sources (in order of preference):
    # 1. From wandb run config (if passed)
    # 2. From sweep config (accessed via API)
    # 3. From environment variable
    config_path = wandb_run.config.get('_config_path')
    
    if not config_path:
        # Try to get from sweep config via API
        try:
            api = wandb.Api()
            sweep = api.sweep(f"{wandb_run.entity}/{wandb_run.project}/{wandb_run.sweep_id}")
            config_path = sweep.config.get('_config_path')
        except Exception as e:
            print(f"[Sweep] Could not access sweep config via API: {e}")
    
    if not config_path:
        # Fall back to environment variable
        config_path = os.environ.get('WANDB_SWEEP_CONFIG_PATH')
    
    if not config_path:
        raise ValueError(
            "Config path not specified. Set WANDB_SWEEP_CONFIG_PATH environment variable "
            "or '_config_path' in wandb sweep config."
        )
    
    # Get experiment type if using single experiment config
    experiment_type = wandb_run.config.get('_experiment_type') or os.environ.get('WANDB_SWEEP_EXPERIMENT_TYPE', 'pytorch')
    
    # Get sweep parameters (exclude internal metadata)
    sweep_params = {
        k: v for k, v in wandb_run.config.items()
        if not k.startswith('_')
    }
    
    print(f"[Sweep] Running sweep with config: {config_path}")
    print(f"[Sweep] Sweep parameters: {list(sweep_params.keys())}")
    print(f"[Sweep] WandB run ID: {wandb_run.id}")
    print(f"[Sweep] WandB sweep ID: {wandb_run.sweep_id}")
    
    # Determine if suite or single experiment
    if is_suite_config(config_path):
        print("[Sweep] Detected suite config")
        run_suite_sweep(config_path, sweep_params)
    else:
        print(f"[Sweep] Detected single experiment config (type: {experiment_type})")
        run_single_experiment_sweep(config_path, sweep_params, experiment_type)
    
    print("[Sweep] Sweep run completed")


if __name__ == "__main__":
    main()
