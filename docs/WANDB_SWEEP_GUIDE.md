# WandB Sweep Integration Guide

This guide explains how to use WandB sweeps with the experiment suite system for hyperparameter optimization.

## Overview

The sweep integration supports:
- **Suite sweeps**: Optimize hyperparameters across experiments in a suite
- **Single experiment sweeps**: Optimize hyperparameters for a single experiment
- **Automatic initialization**: Experiments are initialized first to get IDs, then run with resume IDs
- **Job status tracking**: Tracks job status (not_started, initialized, running, completed, failed)

## Quick Start

### 1. Create Sweep Configuration

Create a YAML file defining your sweep:

```yaml
# sweep_config.yaml
program: scripts/wandb_sweep_wrapper.py
method: bayes
metric:
  name: val_loss
  goal: minimize

# Specify your base config (suite or single experiment)
_config_path: "configs/experiment_suites/lynx_film_raw_test.yaml"

# Hyperparameters to sweep
parameters:
  training.learning_rate:
    distribution: log_uniform
    min: 1e-5
    max: 1e-2
  training.batch_size:
    values: [256, 512, 768, 1024]
```

### 2. Initialize Sweep

```bash
wandb sweep sweep_config.yaml --project fidel-ts
# Returns: entity/project/sweep_id
```

### 3. Run Agents

**Option A: Single SLURM Job with Multiple Agents**

```bash
#!/bin/bash
#SBATCH --job-name=wandb_sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/sweep_%j.out

module load python/3.9
source venv/bin/activate

SWEEP_ID="entity/project/sweep_id"

# Run 4 agents in parallel
for i in {0..3}; do
    CUDA_VISIBLE_DEVICES=$i wandb agent $SWEEP_ID --project fidel-ts &
done

wait
```

**Option B: Multiple SLURM Jobs (Recommended for Large Sweeps)**

```bash
#!/bin/bash
#SBATCH --job-name=wandb_agent
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sweep_agent_%j.out
#SBATCH --array=0-19  # Run 20 agents

module load python/3.9
source venv/bin/activate

# Set GPU from SLURM allocation
if [ -n "$SLURM_JOB_GPUS" ]; then
    export CUDA_VISIBLE_DEVICES=$(echo $SLURM_JOB_GPUS | sed 's/GPU-//g')
fi

SWEEP_ID="entity/project/sweep_id"
wandb agent $SWEEP_ID --project fidel-ts --count 1
```

## Sweep Configuration Options

### Suite Sweep

For sweeping over suite configs:

```yaml
program: scripts/wandb_sweep_wrapper.py
method: bayes
metric:
  name: val_loss
  goal: minimize

_config_path: "configs/experiment_suites/my_suite.yaml"

parameters:
  training.learning_rate:
    distribution: log_uniform
    min: 1e-5
    max: 1e-2
  training.batch_size:
    values: [256, 512, 768]
```

### Single Experiment Sweep

For sweeping over a single experiment:

```yaml
program: scripts/wandb_sweep_wrapper.py
method: random
metric:
  name: val_loss
  goal: minimize

_config_path: "configs/experiments/my_experiment.yaml"
_experiment_type: "pytorch"  # pytorch, lightning, llm, or fm

parameters:
  training.learning_rate:
    distribution: log_uniform
    min: 1e-5
    max: 1e-2
```

## How It Works

### Two-Phase Execution

1. **Initialization Phase** (first run):
   - Runs with `init_only=True` and `return_ids=True`
   - Gets experiment IDs and suite ID
   - Stores IDs in wandb config as `_experiment_id`, `_suite_id`, etc.
   - Sets job status to `initialized`

2. **Execution Phase**:
   - Loads stored IDs from wandb config
   - Sets `resume_experiment_id` and `resume_suite_id` in config
   - Runs actual training
   - Sets job status to `running`, then `completed` or `failed`

### Job Status Tracking

The wrapper tracks job status in wandb config:
- `not_started`: Job hasn't been initialized yet
- `initialized`: Experiment structure created, IDs stored
- `running`: Training in progress
- `completed`: Training finished successfully
- `failed`: Training failed

### Resume Capability

If a SLURM job times out:
- WandB agent can resume the same run
- Wrapper detects `initialized` status
- Loads stored IDs from wandb config
- Continues with resume IDs set
- Your existing resume system handles checkpoint loading

## Parameter Naming

Sweep parameters use dot notation to access nested config:
- `training.learning_rate` → `config.training.learning_rate`
- `training.batch_size` → `config.training.batch_size`
- `model.dropout` → `config.model.dropout` (if in model config)

For suite configs, parameters are applied to the first enabled experiment's overrides.

## Examples

### Example 1: Learning Rate and Batch Size Sweep

```yaml
program: scripts/wandb_sweep_wrapper.py
method: bayes
metric:
  name: val_loss
  goal: minimize

_config_path: "configs/experiment_suites/lynx_film_raw_hparam_sweep.yaml"

parameters:
  training.learning_rate:
    distribution: log_uniform
    min: 1e-5
    max: 1e-2
  training.batch_size:
    values: [256, 512, 768, 1024]
```

### Example 2: Grid Search

```yaml
program: scripts/wandb_sweep_wrapper.py
method: grid
metric:
  name: val_loss
  goal: minimize

_config_path: "configs/experiment_suites/my_suite.yaml"

parameters:
  training.learning_rate:
    values: [1e-4, 5e-4, 1e-3]
  training.patience:
    values: [3, 5, 10]
```

### Example 3: Bayesian Optimization

```yaml
program: scripts/wandb_sweep_wrapper.py
method: bayes
metric:
  name: val_loss
  goal: minimize

_config_path: "configs/experiment_suites/my_suite.yaml"

parameters:
  training.learning_rate:
    distribution: log_uniform
    min: 1e-5
    max: 1e-2
  training.batch_size:
    distribution: q_uniform
    min: 64
    max: 1024
    q: 64
```

## Monitoring

- **WandB Dashboard**: View all runs at `https://wandb.ai/entity/project/sweeps/SWEEP_ID`
- **Job Status**: Check `_job_status` in wandb run config
- **SLURM Logs**: Check `logs/sweep_agent_*.out` for agent logs
- **Experiment Logs**: Check experiment directories in `output/`

## Troubleshooting

### Job Stuck in "initialized" Status

If a job is stuck, it may have failed during initialization. Check:
- WandB run logs
- SLURM job logs
- Experiment directory for errors

You can manually set status in wandb or restart the agent.

### Missing Experiment IDs

If `_experiment_id` or `_suite_id` are missing:
- Check that initialization completed successfully
- Verify wandb config was updated
- Re-run initialization if needed

### Parameter Not Applied

If a sweep parameter isn't being applied:
- Check parameter name matches config structure (use dot notation)
- Verify parameter is in the correct section (training, model, etc.)
- Check that parameter exists in the base config

## Best Practices

1. **Start Small**: Test with a few runs first
2. **Use Bayesian**: For continuous parameters, use `bayes` method
3. **Grid for Discrete**: For categorical parameters, use `grid` or `random`
4. **Monitor Resources**: Watch GPU/memory usage across agents
5. **Set Time Limits**: Use SLURM time limits to prevent runaway jobs
6. **Checkpoint Early**: Your resume system handles timeouts automatically

## Integration with Existing Resume System

The sweep wrapper integrates seamlessly with your existing resume capability:
- Each sweep run is a separate experiment with its own ID
- Resume IDs are set automatically from initialization
- If a SLURM job times out, the next agent run continues from checkpoints
- Job history is maintained per experiment as usual
