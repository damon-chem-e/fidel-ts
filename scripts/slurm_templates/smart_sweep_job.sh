#!/bin/bash
#SBATCH --job-name=sweep-agent
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err

# ============================================================================
# Smart WandB Sweep Agent SLURM Job Template
# ============================================================================
#
# This template demonstrates how to run the smart sweep agent on SLURM.
# The smart agent will:
#   1. Check for incomplete runs from previous jobs that timed out
#   2. Resume incomplete runs before starting new hyperparameter combinations
#   3. Handle SIGTERM gracefully to mark runs for future resumption
#
# Key Features:
#   - Signal handling: SLURM sends SIGTERM 120s before job ends (configurable)
#   - Run prioritization: Incomplete runs are resumed before new ones
#   - Checkpoint-based resumption: Training continues from last epoch
#
# Usage:
#   1. Create sweep:     wandb sweep configs/sweep_configs/your_sweep.yaml
#   2. Submit job:       SWEEP_ID=abc123 sbatch scripts/slurm_templates/smart_sweep_job.sh
#   3. Submit more:      SWEEP_ID=abc123 sbatch scripts/slurm_templates/smart_sweep_job.sh
#
# The magic: Multiple jobs can be submitted. Each will resume incomplete runs
# before starting new combinations, ensuring no training progress is lost.
# ============================================================================

# === Configuration (modify these) ===
PROJECT="fidel-ts"
ENTITY="${WANDB_ENTITY:-your-wandb-username}"  # Set via env var or change default

# Sweep ID must be passed as environment variable
if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: SWEEP_ID environment variable not set"
    echo "Usage: SWEEP_ID=abc123 sbatch $0"
    exit 1
fi

# Number of runs per job - adjust based on your time limit and expected run duration
# Rule of thumb: (time_limit_hours * 60) / (expected_run_minutes) - 1
# E.g., 4 hours with ~45 min runs: (4*60)/45 - 1 ≈ 4 runs
RUNS_PER_JOB=3

# === Signal Configuration ===
# Tell SLURM to send SIGTERM 120 seconds before timeout
# This gives our signal handler time to save checkpoints
#SBATCH --signal=B:TERM@120

# === Environment Setup ===
echo "=== Smart Sweep Agent Starting ==="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Time Limit: $SLURM_TIMELIMIT"
echo "Sweep ID: $SWEEP_ID"
echo "Project: $PROJECT"
echo "Entity: $ENTITY"
echo "Runs per job: $RUNS_PER_JOB"
echo "=================================="

# Activate your conda environment (modify path as needed)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate fidel-ts

# Or use module system
# module load cuda/12.1
# module load python/3.10

# Change to project directory
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Create logs directory if it doesn't exist
mkdir -p logs

# === Run Smart Agent ===
python scripts/smart_sweep_agent.py "$SWEEP_ID" \
    --project "$PROJECT" \
    --entity "$ENTITY" \
    --count "$RUNS_PER_JOB"

exit_code=$?

echo "=== Smart Sweep Agent Finished ==="
echo "Exit code: $exit_code"
echo "=================================="

exit $exit_code
