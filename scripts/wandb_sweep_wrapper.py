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
from pathlib import Path
from typing import Dict, Any
from copy import deepcopy

import wandb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cli.config.loader import load_config
from runs.suite_executor import SuiteExecutor, load_suite_config
from cli.config.models import ExperimentConfig


# Job status constants
STATUS_NOT_STARTED = "not_started"
STATUS_INITIALIZED = "initialized"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


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
    Set job status in wandb run config.
    
    Args:
        wandb_run: wandb run object
        status: Job status string
    """
    wandb_run.config['_job_status'] = status
    wandb.run.config.update({'_job_status': status})


def run_suite_sweep(suite_config_path: str, sweep_params: Dict[str, Any]) -> None:
    """
    Run a wandb sweep on a suite config.
    
    Args:
        suite_config_path: Path to suite config file
        sweep_params: Hyperparameters from wandb.config
    """
    wandb_run = wandb.run
    
    # Load base suite config
    suite_config = load_suite_config(suite_config_path)
    
    # Apply sweep parameters
    suite_config = apply_sweep_parameters_to_suite(suite_config, sweep_params)
    
    # Check job status
    job_status = get_job_status(wandb_run)
    
    if job_status == STATUS_NOT_STARTED:
        # Phase 1: Initialize to get IDs
        print("[Sweep] Initializing experiment structure...")
        executor = SuiteExecutor(suite_config, init_only=True, return_ids=True)
        result = executor.execute()
        
        if not result:
            raise RuntimeError("Failed to get experiment IDs from initialization")
        
        suite_id = result['suite_id']
        experiment_ids = result['experiment_ids']
        
        # Store IDs and status in wandb
        wandb_run.config.update({
            '_suite_id': suite_id,
            '_experiment_ids': experiment_ids,
            '_job_status': STATUS_INITIALIZED
        })
        
        print(f"[Sweep] Initialized suite: {suite_id}")
        print(f"[Sweep] Experiment IDs: {experiment_ids}")
        
        # Update suite config with resume IDs
        suite_config['suite']['resume_suite_id'] = suite_id
        for exp_name, exp_id in experiment_ids.items():
            # Find experiment in suite and set resume_experiment_id
            for exp in suite_config['suite']['experiments']:
                if exp.get('name') == exp_name or f"{exp.get('name')}_output" in exp_name:
                    exp.setdefault('overrides', {})['resume_experiment_id'] = exp_id
                    break
        
        # Update status
        set_job_status(wandb_run, STATUS_RUNNING)
    
    elif job_status == STATUS_INITIALIZED:
        # Resume from initialization - get IDs from wandb config
        suite_id = wandb_run.config.get('_suite_id')
        experiment_ids = wandb_run.config.get('_experiment_ids', {})
        
        if not suite_id or not experiment_ids:
            raise RuntimeError("Missing suite_id or experiment_ids in wandb config. Cannot resume.")
        
        # Set resume IDs in config
        suite_config['suite']['resume_suite_id'] = suite_id
        for exp_name, exp_id in experiment_ids.items():
            for exp in suite_config['suite']['experiments']:
                if exp.get('name') == exp_name or f"{exp.get('name')}_output" in exp_name:
                    exp.setdefault('overrides', {})['resume_experiment_id'] = exp_id
                    break
        
        set_job_status(wandb_run, STATUS_RUNNING)
    
    elif job_status == STATUS_RUNNING:
        # Already running - this shouldn't happen in normal flow, but handle gracefully
        print(f"[Sweep] Job already marked as running, continuing...")
    
    # Phase 2: Run actual training
        print("[Sweep] Starting training execution...")
    try:
        executor = SuiteExecutor(suite_config, init_only=False)
        executor.execute()
        set_job_status(wandb_run, STATUS_COMPLETED)
        print("[Sweep] Training completed successfully")
    except Exception as e:
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
    wandb_run = wandb.run
    
    # Load base experiment config
    experiment_config = load_config(experiment_config_path)
    
    # Apply sweep parameters
    experiment_config = apply_sweep_parameters_to_experiment(experiment_config, sweep_params)
    
    # Check job status
    job_status = get_job_status(wandb_run)
    
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
        
        # Store IDs and status in wandb
        wandb_run.config.update({
            '_experiment_id': experiment_id,
            '_suite_name': suite_name,
            '_job_status': STATUS_INITIALIZED
        })
        
        print(f"[Sweep] Initialized experiment: {experiment_id}")
        
        # Set resume ID
        experiment_config.resume_experiment_id = experiment_id
        if suite_name:
            experiment_config.resume_suite_id = suite_name
        
        set_job_status(wandb_run, STATUS_RUNNING)
    
    elif job_status == STATUS_INITIALIZED:
        # Resume from initialization
        experiment_id = wandb_run.config.get('_experiment_id')
        suite_name = wandb_run.config.get('_suite_name')
        
        if not experiment_id:
            raise RuntimeError("Missing experiment_id in wandb config. Cannot resume.")
        
        experiment_config.resume_experiment_id = experiment_id
        if suite_name:
            experiment_config.resume_suite_id = suite_name
        
        set_job_status(wandb_run, STATUS_RUNNING)
    
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
        set_job_status(wandb_run, STATUS_COMPLETED)
        print("[Sweep] Training completed successfully")
    except Exception as e:
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
