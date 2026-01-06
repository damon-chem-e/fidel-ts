#!/usr/bin/env python3
"""
Sweep Job Submission Helper.

This script helps submit multiple SLURM jobs for a W&B sweep, with intelligent
handling of incomplete runs and time constraints.

Usage:
    # Submit 10 jobs with 3 runs each
    python scripts/submit_sweep_jobs.py SWEEP_ID --project fidel-ts --jobs 10
    
    # Submit jobs with custom time limit
    python scripts/submit_sweep_jobs.py SWEEP_ID --project fidel-ts --jobs 5 --time 08:00:00
    
    # Check status of incomplete runs without submitting
    python scripts/submit_sweep_jobs.py SWEEP_ID --project fidel-ts --status-only
"""

import argparse
import subprocess
import os
import sys
from pathlib import Path
from typing import Optional


def check_sweep_status(entity: str, project: str, sweep_id: str) -> dict:
    """
    Check the status of a W&B sweep.
    
    Returns:
        Dictionary with sweep status information
    """
    try:
        import wandb
        api = wandb.Api()
        
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        runs = sweep.runs
        
        status = {
            'sweep_state': sweep.state,
            'total_runs': len(runs),
            'completed': 0,
            'running': 0,
            'failed': 0,
            'crashed': 0,
            'incomplete_resumable': 0,
        }
        
        for run in runs:
            if run.state == 'finished':
                job_status = run.config.get('_job_status', '')
                if job_status == 'completed':
                    status['completed'] += 1
                else:
                    status['incomplete_resumable'] += 1
            elif run.state == 'running':
                status['running'] += 1
                # Check if it's actually stuck
                if run.config.get('_needs_resume', False):
                    status['incomplete_resumable'] += 1
            elif run.state == 'failed':
                status['failed'] += 1
                # Check if resumable
                if run.config.get('_experiment_id') or run.config.get('_experiment_ids'):
                    status['incomplete_resumable'] += 1
            elif run.state == 'crashed':
                status['crashed'] += 1
                status['incomplete_resumable'] += 1
        
        return status
        
    except Exception as e:
        print(f"Error checking sweep status: {e}")
        return {}


def submit_slurm_job(
    sweep_id: str,
    project: str,
    entity: str,
    time_limit: str = "04:00:00",
    runs_per_job: int = 3,
    partition: str = "gpu",
    gpus: int = 1,
    memory: str = "32G",
    resume_only: bool = False,
    dry_run: bool = False
) -> Optional[str]:
    """
    Submit a single SLURM job for the sweep.
    
    Returns:
        SLURM job ID if successful, None otherwise
    """
    # Build sbatch command
    script_path = Path(__file__).parent / "slurm_templates" / "smart_sweep_job.sh"
    
    # If template doesn't exist, use inline sbatch
    if not script_path.exists():
        # Build inline sbatch command
        cmd = [
            "sbatch",
            f"--job-name=sweep-{sweep_id[:8]}",
            f"--partition={partition}",
            f"--gpus-per-node={gpus}",
            f"--mem={memory}",
            f"--time={time_limit}",
            "--signal=B:TERM@120",
            "--output=logs/sweep_%j.out",
            "--error=logs/sweep_%j.err",
            "--wrap",
        ]
        
        # Build the wrap command
        agent_cmd = f"python scripts/smart_sweep_agent.py {sweep_id} --project {project} --entity {entity} --count {runs_per_job}"
        if resume_only:
            agent_cmd += " --resume-only"
        
        cmd.append(agent_cmd)
    else:
        # Use template script with environment variables
        cmd = [
            "sbatch",
            f"--export=SWEEP_ID={sweep_id},PROJECT={project},ENTITY={entity},RUNS_PER_JOB={runs_per_job}",
            f"--time={time_limit}",
            f"--partition={partition}",
            f"--gpus-per-node={gpus}",
            f"--mem={memory}",
            str(script_path)
        ]
    
    if dry_run:
        print(f"[DRY RUN] Would submit: {' '.join(cmd)}")
        return "dry-run-job-id"
    
    try:
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Parse job ID from output
            output = result.stdout.strip()
            job_id = output.split()[-1] if output else None
            return job_id
        else:
            print(f"sbatch error: {result.stderr}")
            return None
            
    except FileNotFoundError:
        print("Error: sbatch command not found. Are you on a SLURM cluster?")
        return None
    except Exception as e:
        print(f"Error submitting job: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Submit multiple SLURM jobs for a W&B sweep',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('sweep_id', type=str, help='WandB sweep ID')
    parser.add_argument('--project', '-p', type=str, required=True,
                        help='WandB project name')
    parser.add_argument('--entity', '-e', type=str, default=None,
                        help='WandB entity (default: from wandb config)')
    parser.add_argument('--jobs', '-j', type=int, default=1,
                        help='Number of SLURM jobs to submit (default: 1)')
    parser.add_argument('--runs-per-job', '-r', type=int, default=3,
                        help='Number of runs per job (default: 3)')
    parser.add_argument('--time', '-t', type=str, default='04:00:00',
                        help='Time limit per job (default: 04:00:00)')
    parser.add_argument('--partition', type=str, default='gpu',
                        help='SLURM partition (default: gpu)')
    parser.add_argument('--gpus', type=int, default=1,
                        help='GPUs per job (default: 1)')
    parser.add_argument('--memory', type=str, default='32G',
                        help='Memory per job (default: 32G)')
    parser.add_argument('--resume-only', action='store_true',
                        help='Only resume incomplete runs, do not start new')
    parser.add_argument('--status-only', action='store_true',
                        help='Only show sweep status, do not submit jobs')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be submitted without actually submitting')
    
    args = parser.parse_args()
    
    # Get entity
    entity = args.entity
    if entity is None:
        try:
            import wandb
            entity = wandb.Api().default_entity
        except Exception:
            pass
    
    if entity is None:
        entity = os.environ.get('WANDB_ENTITY')
    
    if entity is None:
        print("Error: Could not determine WandB entity. Use --entity or set WANDB_ENTITY")
        sys.exit(1)
    
    # Check sweep status
    print(f"\n=== Sweep Status: {args.sweep_id} ===")
    status = check_sweep_status(entity, args.project, args.sweep_id)
    
    if status:
        print(f"State: {status.get('sweep_state', 'unknown')}")
        print(f"Total runs: {status.get('total_runs', 0)}")
        print(f"  Completed: {status.get('completed', 0)}")
        print(f"  Running: {status.get('running', 0)}")
        print(f"  Failed: {status.get('failed', 0)}")
        print(f"  Crashed: {status.get('crashed', 0)}")
        print(f"  Incomplete (resumable): {status.get('incomplete_resumable', 0)}")
    else:
        print("Could not fetch sweep status")
    
    print("=" * 40)
    
    if args.status_only:
        return
    
    # Submit jobs
    print(f"\nSubmitting {args.jobs} job(s)...")
    print(f"  Runs per job: {args.runs_per_job}")
    print(f"  Time limit: {args.time}")
    print(f"  Resume only: {args.resume_only}")
    print()
    
    submitted_jobs = []
    for i in range(args.jobs):
        job_id = submit_slurm_job(
            sweep_id=args.sweep_id,
            project=args.project,
            entity=entity,
            time_limit=args.time,
            runs_per_job=args.runs_per_job,
            partition=args.partition,
            gpus=args.gpus,
            memory=args.memory,
            resume_only=args.resume_only,
            dry_run=args.dry_run
        )
        
        if job_id:
            submitted_jobs.append(job_id)
            print(f"  Submitted job {i+1}/{args.jobs}: {job_id}")
        else:
            print(f"  Failed to submit job {i+1}/{args.jobs}")
    
    print(f"\nSubmitted {len(submitted_jobs)}/{args.jobs} jobs")
    
    if submitted_jobs and not args.dry_run:
        print(f"\nMonitor with: squeue -u $USER")
        print(f"Cancel all: scancel {' '.join(submitted_jobs)}")


if __name__ == "__main__":
    main()
