#!/usr/bin/env python3
"""
Smart WandB Sweep Agent with Local Registry-Based Resumption.

This script provides a robust solution for hyperparameter sweeps on SLURM clusters
with strict time constraints. It handles:
- LOCAL registry-based detection of incomplete runs (no race conditions)
- Location-aware resumption (runs resume where their checkpoints exist)
- SLURM signal handling for graceful timeout
- Checkpoint-based resumption across job boundaries
- Run prioritization (incomplete runs first, then new combinations)

Key Design:
- Uses a LOCAL file-based registry (per-sweep) as the authoritative source for
  which runs need resumption. This eliminates the race condition with wandb server.
- Only resumes runs that were started on this machine/location (checkpoints are local).
- Wandb is updated for central visibility but is NOT the source of truth for resumption.

Usage:
    # Initialize sweep first (standard wandb sweep command)
    wandb sweep sweep_config.yaml --project fidel-ts
    
    # Run smart agent (replaces wandb agent)
    python scripts/smart_sweep_agent.py SWEEP_ID --project fidel-ts --count 5
    
    # Specify location and output directory
    SWEEP_AGENT_LOCATION=slurm-mit python scripts/smart_sweep_agent.py SWEEP_ID \\
        --project fidel-ts --output-dir /scratch/output
    
    # Or run in SLURM job with automatic signal handling
    sbatch --wrap="python scripts/smart_sweep_agent.py SWEEP_ID --project fidel-ts"
"""

import os
import sys
import signal
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from exp.sweep_registry import SweepRegistry, get_location


# Global flag for graceful shutdown
_shutdown_requested = False
_current_run = None


def setup_signal_handlers():
    """
    Set up signal handlers for graceful shutdown on SLURM timeout.
    
    SLURM sends SIGTERM before killing jobs (configurable via --signal).
    We catch this to save checkpoints and mark runs appropriately.
    """
    def signal_handler(signum, frame):
        global _shutdown_requested
        _shutdown_requested = True
        signal_name = signal.Signals(signum).name
        print(f"\n[SmartAgent] Received {signal_name} signal. Requesting graceful shutdown...")
        
        # If we have an active run, mark it for resumption
        if _current_run is not None:
            try:
                # Update run config to indicate it needs resumption
                _current_run.config.update({
                    '_needs_resume': True,
                    '_interrupted_at': datetime.now().isoformat(),
                    '_interrupted_by': signal_name
                }, allow_val_change=True)
                print(f"[SmartAgent] Marked run {_current_run.id} for resumption")
            except Exception as e:
                print(f"[SmartAgent] Warning: Could not mark run for resumption: {e}")
    
    # Register handlers for common termination signals
    signal.signal(signal.SIGTERM, signal_handler)  # SLURM timeout
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    
    # USR1 is commonly used for preemption warnings
    try:
        signal.signal(signal.SIGUSR1, signal_handler)
    except (AttributeError, ValueError):
        # SIGUSR1 may not be available on Windows
        pass


def is_run_completed(run) -> bool:
    """
    Check if a W&B run is truly completed using explicit completion markers.
    
    This handles the race condition where wandb.finish() is called before
    _job_status can be updated in config. We check explicit markers only:
    
    1. summary._sweep_completed == True (most reliable - set just before finish())
    2. config._job_status == 'completed' (may fail to update)
    
    IMPORTANT: We do NOT use heuristics like "has final metrics" because:
    - A run that completed 50/100 epochs might have metrics but isn't done
    - Only explicit completion markers should be trusted
    
    Args:
        run: wandb.Run object from API
        
    Returns:
        True if the run has an explicit completion marker, False otherwise
    """
    # Check 1: summary._sweep_completed == True (most reliable)
    # This is set by ExperimentManager.end_experiment(sweep_completed=True)
    # just before wandb.finish() is called
    try:
        # Must be explicitly True, not just truthy
        if run.summary.get('_sweep_completed') is True:
            return True
    except Exception:
        pass
    
    # Check 2: config._job_status == 'completed'
    # This may fail to update if run finishes before sweep wrapper can set it
    try:
        if run.config.get('_job_status', '') == 'completed':
            return True
    except Exception:
        pass
    
    # No explicit completion marker found
    return False


def get_incomplete_runs(api, entity: str, project: str, sweep_id: str) -> List[Dict[str, Any]]:
    """
    Query W&B API for incomplete runs in the sweep that need resumption.
    
    An incomplete run is one that:
    - Has NOT been marked complete via summary._sweep_completed
    - Has status "running", "crashed", or "failed" (but may be resumable)
    - Has _job_status != "completed" in config (fallback check)
    - Has _needs_resume flag set
    - Started training but didn't finish all epochs
    
    Uses multiple completion indicators to avoid false positives from the
    race condition where wandb.finish() is called before _job_status is set.
    
    Args:
        api: wandb.Api instance
        entity: W&B entity/username
        project: W&B project name
        sweep_id: W&B sweep ID
        
    Returns:
        List of incomplete run info dictionaries, sorted by priority
    """
    incomplete_runs = []
    
    try:
        # Get all runs in the sweep
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        runs = sweep.runs
        
        for run in runs:
            # Skip runs that are confirmed completed (using robust multi-check)
            if is_run_completed(run):
                continue
            
            # Skip runs that finished successfully with no partial progress
            # (these are runs that failed immediately before any training)
            if run.state == "finished":
                # Only consider for resume if it has experiment IDs (partial progress)
                has_progress = (
                    run.config.get('_experiment_id') or 
                    run.config.get('_experiment_ids') or
                    run.config.get('_job_status') in ['initialized', 'running']
                )
                if not has_progress:
                    continue
            
            # Check if run needs resumption
            needs_resume = False
            priority = 0  # Higher = more urgent
            
            # Check explicit needs_resume flag (set by signal handler)
            if run.config.get('_needs_resume', False):
                needs_resume = True
                priority = 100  # Highest priority
            
            # Check if run was running but crashed/stopped
            elif run.state in ['running', 'crashed', 'failed']:
                job_status = run.config.get('_job_status', 'not_started')
                
                # If initialized or running, it needs resume
                if job_status in ['initialized', 'running']:
                    needs_resume = True
                    priority = 80 if run.state == 'running' else 60
                
                # Check if it has resume IDs stored (indicates partial progress)
                elif run.config.get('_experiment_id') or run.config.get('_experiment_ids'):
                    needs_resume = True
                    priority = 50
            
            # Check for finished runs that didn't complete (race condition case)
            elif run.state == 'finished':
                job_status = run.config.get('_job_status', '')
                # If finished but not marked completed, likely a race condition
                if job_status in ['initialized', 'running']:
                    needs_resume = True
                    priority = 70  # High priority - likely just missed completion
                elif run.config.get('_experiment_id') or run.config.get('_experiment_ids'):
                    # Has experiment IDs but no completion marker
                    needs_resume = True
                    priority = 45
            
            # Check for stale "running" state (stuck runs)
            if run.state == 'running' and not needs_resume:
                # If heartbeat is old (> 10 min), consider it stuck
                try:
                    heartbeat = run.heartbeat_at
                    if heartbeat:
                        age = datetime.now() - datetime.fromisoformat(heartbeat.replace('Z', '+00:00').replace('+00:00', ''))
                        if age > timedelta(minutes=10):
                            needs_resume = True
                            priority = 40
                except Exception:
                    pass
            
            if needs_resume:
                # Get summary info for better diagnostics
                try:
                    summary_dict = dict(run.summary) if run.summary else {}
                except Exception:
                    summary_dict = {}
                
                incomplete_runs.append({
                    'run_id': run.id,
                    'run_name': run.name,
                    'state': run.state,
                    'job_status': run.config.get('_job_status', 'unknown'),
                    'sweep_completed': summary_dict.get('_sweep_completed', False),
                    'config': dict(run.config),
                    'summary': summary_dict,
                    'priority': priority,
                    'created_at': run.created_at,
                    'experiment_id': run.config.get('_experiment_id'),
                    'experiment_ids': run.config.get('_experiment_ids'),
                    'suite_id': run.config.get('_suite_id'),
                })
        
        # Sort by priority (highest first), then by creation time (oldest first)
        incomplete_runs.sort(key=lambda x: (-x['priority'], x['created_at']))
        
    except Exception as e:
        print(f"[SmartAgent] Warning: Could not query incomplete runs: {e}")
    
    return incomplete_runs


def resume_incomplete_run(run_info: Dict[str, Any], entity: str, project: str) -> bool:
    """
    Resume an incomplete W&B run.
    
    Args:
        run_info: Dictionary with incomplete run information
        entity: W&B entity
        project: W&B project
        
    Returns:
        True if run completed successfully, False otherwise
    """
    global _current_run, _shutdown_requested
    
    import wandb
    from scripts.wandb_sweep_wrapper import (
        run_suite_sweep, 
        run_single_experiment_sweep,
        is_suite_config,
        STATUS_RUNNING
    )
    
    run_id = run_info['run_id']
    config = run_info['config']
    
    print(f"[SmartAgent] Resuming incomplete run: {run_id} (state: {run_info['state']}, job_status: {run_info['job_status']})")
    
    try:
        # Resume the wandb run
        _current_run = wandb.init(
            entity=entity,
            project=project,
            id=run_id,
            resume='must',  # Must resume existing run
            reinit=True
        )
        
        # Clear the needs_resume flag
        _current_run.config.update({
            '_needs_resume': False,
            '_resumed_at': datetime.now().isoformat(),
            '_job_status': STATUS_RUNNING
        }, allow_val_change=True)
        
        # Get config path
        config_path = config.get('_config_path')
        if not config_path:
            print(f"[SmartAgent] Error: No _config_path in run config. Cannot resume.")
            return False
        
        # Get sweep parameters (exclude internal metadata)
        sweep_params = {k: v for k, v in config.items() if not k.startswith('_')}
        
        # Check for shutdown before starting
        if _shutdown_requested:
            print(f"[SmartAgent] Shutdown requested before run started. Aborting.")
            return False
        
        # Run the appropriate sweep function
        experiment_type = config.get('_experiment_type', 'pytorch')
        
        if is_suite_config(config_path):
            run_suite_sweep(config_path, sweep_params)
        else:
            run_single_experiment_sweep(config_path, sweep_params, experiment_type)
        
        print(f"[SmartAgent] Successfully completed resumed run: {run_id}")
        return True
        
    except Exception as e:
        print(f"[SmartAgent] Error resuming run {run_id}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        _current_run = None
        if wandb.run is not None:
            try:
                wandb.finish()
            except Exception:
                pass


def run_new_sweep_trial(entity: str, project: str, sweep_id: str) -> bool:
    """
    Run a new sweep trial (standard wandb agent behavior for one run).
    
    Args:
        entity: W&B entity
        project: W&B project
        sweep_id: W&B sweep ID
        
    Returns:
        True if run completed successfully, False otherwise
    """
    global _current_run, _shutdown_requested
    
    import wandb
    from scripts.wandb_sweep_wrapper import main as sweep_wrapper_main
    
    print(f"[SmartAgent] Starting new sweep trial...")
    
    try:
        # Use wandb.agent with count=1 to run exactly one trial
        # We wrap this to set up our global run reference
        
        def run_one_trial():
            global _current_run
            
            # Initialize is handled by wandb.agent internally
            wandb.init()
            _current_run = wandb.run
            
            try:
                # Call the sweep wrapper main function (minus the wandb.init)
                sweep_wrapper_main()
            finally:
                _current_run = None
        
        # Run one trial
        wandb.agent(
            sweep_id,
            function=run_one_trial,
            entity=entity,
            project=project,
            count=1
        )
        
        print(f"[SmartAgent] Completed new sweep trial")
        return True
        
    except Exception as e:
        print(f"[SmartAgent] Error in new sweep trial: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        _current_run = None


def find_local_registries(output_dir: Path, sweep_id: str) -> List[SweepRegistry]:
    """
    Find all local sweep registries for a given sweep ID.
    
    Searches the output directory for sweep registry files that match
    the given sweep ID.
    
    Args:
        output_dir: Base output directory to search
        sweep_id: W&B sweep ID
        
    Returns:
        List of SweepRegistry instances found
    """
    registries = []
    location = get_location()
    
    if not output_dir.exists():
        return registries
    
    # Search for registry files in subdirectories
    for registry_file in output_dir.rglob(SweepRegistry.REGISTRY_FILENAME):
        try:
            registry = SweepRegistry(
                suite_dir=str(registry_file.parent),
                sweep_id=sweep_id,
                location=location,
                create_if_missing=False
            )
            # Check if this registry is for our sweep
            data = registry._load()
            if data.get('sweep_id') == sweep_id:
                registries.append(registry)
        except Exception:
            continue
    
    return registries


def get_local_incomplete_runs(output_dir: Path, sweep_id: str) -> List[Dict[str, Any]]:
    """
    Get all incomplete runs from local registries for this location.
    
    This is the authoritative source for resumption - no wandb API calls,
    no race conditions. Only returns runs that need resumption AND were
    started on this machine (so checkpoints exist locally).
    
    Args:
        output_dir: Base output directory
        sweep_id: W&B sweep ID
        
    Returns:
        List of incomplete run info dictionaries, sorted by priority
    """
    all_incomplete = []
    
    # Find all registries for this sweep
    registries = find_local_registries(output_dir, sweep_id)
    
    for registry in registries:
        incomplete = registry.get_runs_needing_resume()
        for run_info in incomplete:
            run_info['registry'] = registry  # Keep reference to registry
            run_info['suite_dir'] = str(registry.suite_dir)
        all_incomplete.extend(incomplete)
    
    # Sort by priority (highest first)
    all_incomplete.sort(key=lambda x: -x.get('priority', 0))
    
    return all_incomplete


def resume_local_run(run_info: Dict[str, Any], entity: str, project: str) -> bool:
    """
    Resume an incomplete run using local registry info.
    
    Args:
        run_info: Run info from local registry
        entity: W&B entity
        project: W&B project
        
    Returns:
        True if run completed successfully, False otherwise
    """
    global _current_run, _shutdown_requested
    
    import wandb
    from scripts.wandb_sweep_wrapper import (
        run_suite_sweep, 
        run_single_experiment_sweep,
        is_suite_config,
        STATUS_RUNNING
    )
    
    wandb_run_id = run_info['wandb_run_id']
    experiment_id = run_info.get('experiment_id')
    registry = run_info.get('registry')
    
    print(f"[SmartAgent] Resuming local incomplete run: {wandb_run_id}")
    print(f"[SmartAgent]   Experiment ID: {experiment_id}")
    print(f"[SmartAgent]   Progress: {run_info.get('final_epoch', 0)}/{run_info.get('total_epochs', '?')} epochs")
    
    try:
        # Resume the wandb run
        _current_run = wandb.init(
            entity=entity,
            project=project,
            id=wandb_run_id,
            resume='must',  # Must resume existing run
            reinit=True
        )
        
        # Get config from wandb run
        config = dict(_current_run.config)
        
        # Clear the needs_resume flag in wandb
        _current_run.config.update({
            '_needs_resume': False,
            '_resumed_at': datetime.now().isoformat(),
            '_job_status': STATUS_RUNNING
        }, allow_val_change=True)
        
        # Mark as running in local registry
        if registry:
            registry.mark_running(
                wandb_run_id,
                run_info.get('total_epochs', 100)
            )
        
        # Get config path
        config_path = config.get('_config_path')
        if not config_path:
            print("[SmartAgent] Error: No _config_path in run config. Cannot resume.")
            return False
        
        # Get sweep parameters (exclude internal metadata)
        sweep_params = {k: v for k, v in config.items() if not k.startswith('_')}
        
        # Check for shutdown before starting
        if _shutdown_requested:
            print("[SmartAgent] Shutdown requested before run started. Aborting.")
            return False
        
        # Run the appropriate sweep function
        experiment_type = config.get('_experiment_type', 'pytorch')
        
        if is_suite_config(config_path):
            run_suite_sweep(config_path, sweep_params)
        else:
            run_single_experiment_sweep(config_path, sweep_params, experiment_type)
        
        print(f"[SmartAgent] Successfully completed resumed run: {wandb_run_id}")
        return True
        
    except Exception as e:
        print(f"[SmartAgent] Error resuming run {wandb_run_id}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        _current_run = None
        if wandb.run is not None:
            try:
                wandb.finish()
            except Exception:
                pass


def smart_agent_loop(
    entity: str,
    project: str,
    sweep_id: str,
    output_dir: str = "./output",
    count: Optional[int] = None,
    resume_only: bool = False,
    new_only: bool = False
):
    """
    Main smart agent loop that prioritizes incomplete runs from LOCAL registry.
    
    This replaces the standard `wandb agent` command with intelligent
    incomplete run detection and resumption based on LOCAL file registry.
    
    Key design:
    - Uses LOCAL registry as authoritative source (no race conditions)
    - Only resumes runs started on this machine (checkpoints are local)
    - Wandb API is only used for new trials and sweep state checking
    
    Args:
        entity: W&B entity/username
        project: W&B project name
        sweep_id: W&B sweep ID
        output_dir: Base output directory (where sweep registries are stored)
        count: Maximum number of runs to execute (None = unlimited)
        resume_only: If True, only resume incomplete runs (don't start new)
        new_only: If True, skip incomplete runs (behave like standard agent)
    """
    global _shutdown_requested
    
    import wandb
    
    # Set up signal handlers for graceful shutdown
    setup_signal_handlers()
    
    location = get_location()
    output_path = Path(output_dir).resolve()
    api = wandb.Api()
    runs_completed = 0
    
    print(f"[SmartAgent] Starting smart sweep agent")
    print(f"[SmartAgent] Sweep: {entity}/{project}/{sweep_id}")
    print(f"[SmartAgent] Location: {location}")
    print(f"[SmartAgent] Output dir: {output_path}")
    print(f"[SmartAgent] Max runs: {count if count else 'unlimited'}")
    print(f"[SmartAgent] Mode: {'resume_only' if resume_only else ('new_only' if new_only else 'smart (local resume first)')}")
    
    while True:
        # Check if we've hit our run limit
        if count is not None and runs_completed >= count:
            print(f"[SmartAgent] Completed {runs_completed} runs. Exiting.")
            break
        
        # Check for shutdown signal
        if _shutdown_requested:
            print(f"[SmartAgent] Shutdown requested. Exiting after {runs_completed} runs.")
            break
        
        # Step 1: Check LOCAL registry for incomplete runs (unless new_only mode)
        if not new_only:
            print(f"[SmartAgent] Checking local registry for incomplete runs...")
            incomplete_runs = get_local_incomplete_runs(output_path, sweep_id)
            
            if incomplete_runs:
                print(f"[SmartAgent] Found {len(incomplete_runs)} local incomplete run(s)")
                for i, run in enumerate(incomplete_runs[:3]):  # Show top 3
                    print(f"[SmartAgent]   {i+1}. {run['wandb_run_id']} - {run.get('final_epoch', 0)}/{run.get('total_epochs', '?')} epochs")
                
                # Resume the highest priority incomplete run
                run_info = incomplete_runs[0]
                success = resume_local_run(run_info, entity, project)
                
                if success:
                    runs_completed += 1
                else:
                    # Failed to resume - brief pause before continuing
                    print(f"[SmartAgent] Failed to resume run. Trying next...")
                    time.sleep(5)
                
                continue  # Loop back to check for more incomplete runs
        
        # Step 2: Run new sweep trial (unless resume_only mode)
        if resume_only:
            print("[SmartAgent] No local incomplete runs and resume_only=True. Exiting.")
            break
        
        print("[SmartAgent] No local incomplete runs. Starting new sweep trial...")
        
        # Check sweep state before starting new run
        try:
            sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
            if sweep.state == 'FINISHED':
                print("[SmartAgent] Sweep is finished. Exiting.")
                break
            elif sweep.state == 'PAUSED':
                print("[SmartAgent] Sweep is paused. Waiting...")
                time.sleep(60)
                continue
        except Exception as e:
            print(f"[SmartAgent] Warning: Could not check sweep state: {e}")
        
        # Run new trial
        success = run_new_sweep_trial(entity, project, sweep_id)
        
        if success:
            runs_completed += 1
        else:
            # Check if sweep might be exhausted
            time.sleep(10)
    
    print(f"[SmartAgent] Agent finished. Total runs completed: {runs_completed}")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Smart WandB Sweep Agent with local registry-based resumption',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run smart agent on a sweep (resumes local incomplete runs first, then new)
    python smart_sweep_agent.py SWEEP_ID --project fidel-ts
    
    # Specify output directory (where sweep registries are stored)
    python smart_sweep_agent.py SWEEP_ID --project fidel-ts --output-dir /scratch/output
    
    # Run with entity specified
    python smart_sweep_agent.py SWEEP_ID --entity my-team --project fidel-ts
    
    # Limit to 5 runs (good for SLURM jobs with time limits)
    python smart_sweep_agent.py SWEEP_ID --project fidel-ts --count 5
    
    # Only resume incomplete runs (don't start new combinations)
    python smart_sweep_agent.py SWEEP_ID --project fidel-ts --resume-only
    
    # Only run new trials (ignore incomplete runs, like standard agent)
    python smart_sweep_agent.py SWEEP_ID --project fidel-ts --new-only
    
    # Specify location explicitly (default: from SWEEP_AGENT_LOCATION env or hostname)
    SWEEP_AGENT_LOCATION=slurm-mit python smart_sweep_agent.py SWEEP_ID --project fidel-ts

Environment Variables:
    SWEEP_AGENT_LOCATION  - Identifies where this agent is running (e.g., 'slurm-mit', 'runpod-a100')
    WANDB_ENTITY          - Default W&B entity if not specified via --entity
        """
    )
    
    parser.add_argument('sweep_id', type=str, help='WandB sweep ID')
    parser.add_argument('--entity', '-e', type=str, default=None,
                        help='WandB entity/username (default: from wandb config)')
    parser.add_argument('--project', '-p', type=str, required=True,
                        help='WandB project name')
    parser.add_argument('--output-dir', '-o', type=str, default='./output',
                        help='Output directory where sweep registries are stored (default: ./output)')
    parser.add_argument('--count', '-c', type=int, default=None,
                        help='Maximum number of runs to execute')
    parser.add_argument('--resume-only', action='store_true',
                        help='Only resume incomplete runs, do not start new combinations')
    parser.add_argument('--new-only', action='store_true',
                        help='Skip incomplete runs (behave like standard wandb agent)')
    
    args = parser.parse_args()
    
    # Validate mutually exclusive options
    if args.resume_only and args.new_only:
        parser.error("--resume-only and --new-only are mutually exclusive")
    
    # Get entity from wandb if not specified
    entity = args.entity
    if entity is None:
        try:
            import wandb
            entity = wandb.Api().default_entity
            if entity is None:
                # Try to get from environment
                entity = os.environ.get('WANDB_ENTITY')
        except Exception:
            pass
    
    if entity is None:
        parser.error("Could not determine WandB entity. Specify with --entity or WANDB_ENTITY env var")
    
    # Run the smart agent loop
    smart_agent_loop(
        entity=entity,
        project=args.project,
        sweep_id=args.sweep_id,
        output_dir=args.output_dir,
        count=args.count,
        resume_only=args.resume_only,
        new_only=args.new_only
    )


if __name__ == "__main__":
    main()
