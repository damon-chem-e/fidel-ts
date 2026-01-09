#!/usr/bin/env python3
"""
Local Sweep Agent - CLI entry point for sweep execution with local-first resumption.

This script provides a command-line interface for running W&B sweeps with
intelligent resumption handling. It uses a local registry to track which
runs need resumption, eliminating race conditions with the wandb server.

Key Features:
    - Local-first resumption: Uses file-based registry (no race conditions)
    - Location-aware: Only resumes runs where checkpoints exist
    - Signal handling: Graceful shutdown on SIGTERM/SIGINT
    - Flexible modes: Resume-only, new-only, or smart (resume first)

Architecture:
    This script is a thin CLI wrapper around SweepManager, which handles:
        - Signal registration and shutdown coordination
        - Registry management (single owner)
        - Trial execution via SweepExecutor
        - Status finalization

Usage:
    # Run sweep agent (resumes incomplete runs first, then gets new hparams)
    python scripts/local_sweep_agent.py SWEEP_ID --project fidel-ts
    
    # Specify output directory
    python scripts/local_sweep_agent.py SWEEP_ID --project fidel-ts -o /scratch/output
    
    # Limit number of runs (for SLURM time limits)
    python scripts/local_sweep_agent.py SWEEP_ID --project fidel-ts --count 3
    
    # Only resume incomplete runs (don't start new)
    python scripts/local_sweep_agent.py SWEEP_ID --project fidel-ts --resume-only
    
    # Only run new trials (skip incomplete check)
    python scripts/local_sweep_agent.py SWEEP_ID --project fidel-ts --new-only

Environment Variables:
    SWEEP_AGENT_LOCATION  - Identifies this machine (e.g., 'slurm-mit', 'runpod-a100')
                           Default: hostname
    WANDB_ENTITY          - Default W&B entity if not specified via --entity

Example SLURM Job:
    #!/bin/bash
    #SBATCH --job-name=sweep-agent
    #SBATCH --time=04:00:00
    #SBATCH --signal=B:TERM@120
    
    export SWEEP_AGENT_LOCATION="slurm-mit"
    python scripts/local_sweep_agent.py $SWEEP_ID --project fidel-ts --count 3
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """
    Main entry point - parse arguments and run sweep manager.
    
    This function:
        1. Parses command-line arguments
        2. Resolves entity from args, env, or wandb config
        3. Creates SweepManager instance
        4. Runs the main loop
    """
    parser = argparse.ArgumentParser(
        description='Local Sweep Agent - Run W&B sweeps with local-first resumption',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run sweep (resumes incomplete first, then new)
    python local_sweep_agent.py SWEEP_ID --project fidel-ts
    
    # Specify output directory
    python local_sweep_agent.py SWEEP_ID --project fidel-ts -o /scratch/output
    
    # Limit runs (for SLURM time limits)
    python local_sweep_agent.py SWEEP_ID --project fidel-ts --count 3
    
    # Only resume incomplete runs
    python local_sweep_agent.py SWEEP_ID --project fidel-ts --resume-only
    
    # Only run new trials
    python local_sweep_agent.py SWEEP_ID --project fidel-ts --new-only

Environment Variables:
    SWEEP_AGENT_LOCATION  - Location identifier (default: hostname)
    WANDB_ENTITY          - Default W&B entity
        """
    )
    
    # Required arguments
    parser.add_argument(
        'sweep_id',
        type=str,
        help='W&B sweep ID'
    )
    
    parser.add_argument(
        '--project', '-p',
        type=str,
        required=True,
        help='W&B project name'
    )
    
    # Optional arguments
    parser.add_argument(
        '--entity', '-e',
        type=str,
        default=None,
        help='W&B entity/username (default: from wandb config or WANDB_ENTITY)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./output',
        help='Output directory for experiments (default: ./output)'
    )
    
    parser.add_argument(
        '--count', '-c',
        type=int,
        default=None,
        help='Maximum number of runs to execute (default: unlimited)'
    )
    
    parser.add_argument(
        '--resume-only',
        action='store_true',
        help='Only resume incomplete runs, do not start new'
    )
    
    parser.add_argument(
        '--new-only',
        action='store_true',
        help='Skip incomplete run check, only run new trials'
    )
    
    parser.add_argument(
        '--experiment-type',
        type=str,
        default='pytorch',
        choices=['pytorch', 'lightning', 'llm', 'fm'],
        help='Experiment type (default: pytorch)'
    )
    
    parser.add_argument(
        '--no-reconcile-registry',
        action='store_true',
        help='Skip registry reconciliation on startup. Use when running concurrent agents to avoid conflicts.'
    )
    
    parser.add_argument(
        '--lock-staleness-hours',
        type=float,
        default=48.0,
        help='Hours threshold for detecting stale locks (default: 48.0). Locks older than this are considered stale and released.'
    )
    
    args = parser.parse_args()
    
    # Validate mutually exclusive options
    if args.resume_only and args.new_only:
        parser.error("--resume-only and --new-only are mutually exclusive")
    
    # Resolve entity
    entity = args.entity
    if entity is None:
        # Try wandb config
        try:
            import wandb
            entity = wandb.Api().default_entity
        except Exception:
            pass
    
    if entity is None:
        # Try environment variable
        entity = os.environ.get('WANDB_ENTITY')
    
    if entity is None:
        parser.error(
            "Could not determine W&B entity. "
            "Specify with --entity or set WANDB_ENTITY environment variable."
        )
    
    # Import and create manager
    from exp.sweep_manager import SweepManager
    
    manager = SweepManager(
        entity=entity,
        project=args.project,
        sweep_id=args.sweep_id,
        output_dir=args.output_dir,
        experiment_type=args.experiment_type
    )
    
    # Run the main loop
    runs_completed = manager.run_loop(
        count=args.count,
        resume_only=args.resume_only,
        new_only=args.new_only,
        reconcile_registry=not args.no_reconcile_registry,
        lock_staleness_hours=args.lock_staleness_hours
    )
    
    print(f"\n[LocalSweepAgent] Finished. Runs completed: {runs_completed}")
    
    # Exit with appropriate code
    sys.exit(0 if runs_completed > 0 or args.resume_only else 1)


if __name__ == "__main__":
    main()
