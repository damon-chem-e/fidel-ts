# Job Resumption for Long-Running Experiments

This document describes how to resume training experiments across multiple SLURM jobs when individual jobs have time limits (e.g., 6 hours) but training requires longer (e.g., 15-30 hours).

## Overview

The experiment manager supports automatic resumption of training across multiple SLURM jobs. When a job times out, the next job automatically detects the existing experiment, loads the last checkpoint, and continues training from the next epoch.

## Key Concepts

### Experiment ID

Each experiment has a unique **experiment ID** in the format: `{timestamp}_{config_hash}`

- **timestamp**: When the experiment was first created (format: `YYYYMMDD-HHMMSS`)
- **config_hash**: 12-character hash of the experiment configuration

The experiment ID uniquely identifies an experiment and remains constant across all jobs that resume it.

### Job History

The experiment manager maintains a `job_history.json` file in the experiment directory that tracks:
- All SLURM jobs that have run for this experiment
- Start/end times for each job
- Epochs completed by each job
- SLURM job IDs
- Checkpoint paths
- Final metrics for each job

### Checkpoint Management

- Checkpoints are saved after each epoch
- The latest checkpoint is automatically detected and loaded when resuming
- Both PyTorch and Lightning training paths are supported

## Workflow

### 1. Initial Job (First Run)

1. Create your experiment configuration file
2. Submit the first SLURM job
3. The experiment manager:
   - Generates a new experiment ID
   - Creates experiment directory structure
   - Starts training from epoch 0
   - Saves checkpoints after each epoch
   - Registers job start in `job_history.json`

4. If the job times out:
   - The last checkpoint is saved
   - Job end is registered in `job_history.json` with status "timeout"

### 2. Resumption Job (Subsequent Runs)

1. **Important**: Use the same experiment configuration file, but specify the `resume_experiment_id` field:

```yaml
# Your existing config...
model:
  name: "DLinear"
  config_path: "model_configs/dlinear.yaml"
data:
  name: "ETTh1"
  config_path: "data_configs/ETTh1.yaml"
training:
  epochs: 100
  batch_size: 32
  # ... other training config

# Add this field for resumption:
resume_experiment_id: "20240101-120000_abc123def456"
```

2. Submit the next SLURM job with the same config (including `resume_experiment_id`)
3. The experiment manager:
   - Detects the `resume_experiment_id` in config
   - Loads the existing experiment directory
   - Finds the latest checkpoint
   - Determines the next epoch to start from
   - Registers the new job start
   - Resumes training from the checkpoint

4. Repeat steps 2-3 until all epochs are complete

## Git Worktree Best Practice

**Critical**: For reproducibility, use a git worktree locked to a specific commit for all jobs in an experiment.

### Why Use a Worktree?

The experiment ID is based on the configuration, not the code. If the main workspace changes commits between jobs:
- The experiment will still match (same config hash)
- But the code may have changed, leading to inconsistent results
- This makes debugging difficult and breaks reproducibility

### How to Use a Worktree

1. **Create a frozen worktree** at the commit you want to use:

```bash
# Get the commit hash you want to use
git log --oneline -1

# Create a worktree at that commit
git worktree add /path/to/frozen-worktree <commit-hash>

# Example:
git worktree add ~/experiments/exp_20240101 abc123def456
```

2. **Run all SLURM jobs from the frozen worktree**:

```bash
cd ~/experiments/exp_20240101
python -m cli.train lightning configs/experiments/my_experiment.yaml
```

3. **Benefits**:
   - Code stays locked to the original commit
   - Main workspace can change without affecting the experiment
   - Clear reproducibility - you know exactly which code version was used
   - Easy to identify which commit was used for each experiment

### Worktree Management

```bash
# List all worktrees
git worktree list

# Remove a worktree when experiment is complete
git worktree remove /path/to/frozen-worktree

# Or if the worktree directory was deleted manually
git worktree prune
```

## Configuration

### New Experiment

```yaml
model:
  name: "DLinear"
  config_path: "model_configs/dlinear.yaml"
data:
  name: "ETTh1"
  config_path: "data_configs/ETTh1.yaml"
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  # ... other config
# No resume_experiment_id - creates new experiment
```

### Resuming Experiment

```yaml
model:
  name: "DLinear"
  config_path: "model_configs/dlinear.yaml"
data:
  name: "ETTh1"
  config_path: "data_configs/ETTh1.yaml"
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  # ... other config (must match original)
# Specify the experiment ID to resume
resume_experiment_id: "20240101-120000_abc123def456"
```

**Important**: The configuration (except `resume_experiment_id`) must match the original experiment exactly, or the experiment ID hash will not match.

## SLURM Integration

The system automatically detects SLURM environment variables:

- `SLURM_JOB_ID`: Automatically captured and stored in job history
- `SLURM_JOB_NAME`: Automatically captured if available

These are logged in `job_history.json` for tracking which SLURM job ran which epochs.

## Job History Structure

The `job_history.json` file contains:

```json
{
  "experiment_id": "20240101-120000_abc123def456",
  "total_epochs": 100,
  "jobs": [
    {
      "job_id": "12345",
      "slurm_job_id": "12345",
      "job_name": "training_job_1",
      "start_time": "2024-01-01T12:00:00",
      "end_time": "2024-01-01T18:00:00",
      "start_epoch": 0,
      "end_epoch": 15,
      "epochs_completed": [1, 2, 3, ..., 15],
      "status": "timeout",
      "checkpoint_path": "checkpoints/checkpoint-15-0.123456.pth",
      "final_train_loss": 0.123,
      "final_val_loss": 0.456
    },
    {
      "job_id": "12346",
      "slurm_job_id": "12346",
      "start_time": "2024-01-01T18:05:00",
      "end_time": null,
      "start_epoch": 16,
      "end_epoch": null,
      "status": "running"
    }
  ],
  "current_epoch": 16,
  "last_checkpoint": "checkpoints/checkpoint-15-0.123456.pth",
  "best_checkpoint": "checkpoints/checkpoint-12-0.111111.pth"
}
```

## Example Workflow

### Step 1: Initial Job

```bash
# In your frozen worktree
cd ~/experiments/exp_20240101

# Submit first job
sbatch scripts/train.sh configs/experiments/my_experiment.yaml
```

The job runs for 6 hours, completes epochs 0-15, then times out.

### Step 2: Check Progress

```bash
# Check job history
cat output/my_experiment/20240101-120000_abc123def456/job_history.json

# Check latest checkpoint
ls -lh output/my_experiment/20240101-120000_abc123def456/checkpoints/
```

### Step 3: Resume Job

Edit your config to add `resume_experiment_id`:

```yaml
# configs/experiments/my_experiment.yaml
resume_experiment_id: "20240101-120000_abc123def456"
# ... rest of config unchanged
```

Submit the next job:

```bash
# Still in the same frozen worktree
sbatch scripts/train.sh configs/experiments/my_experiment.yaml
```

The job automatically:
- Detects the existing experiment
- Loads checkpoint from epoch 15
- Resumes from epoch 16
- Continues until timeout or completion

### Step 4: Repeat Until Complete

Continue submitting jobs with the same config (including `resume_experiment_id`) until all epochs are complete.

## Troubleshooting

### Experiment Not Found

**Error**: "Experiment ID not found"

**Solution**: 
- Verify the `resume_experiment_id` matches exactly (including timestamp)
- Check that the experiment directory exists in the output path
- Ensure you're using the same output directory as the original job

### Checkpoint Not Found

**Error**: "Resume detected but no checkpoint found"

**Solution**:
- Check that checkpoints were saved in the first job
- Verify checkpoint directory exists: `{experiment_dir}/checkpoints/`
- For Lightning: look for `last.ckpt` or `checkpoint-*.ckpt`
- For PyTorch: look for `checkpoint.pth`

### Config Mismatch

**Error**: Experiment resumes but results are inconsistent

**Solution**:
- Ensure all config fields (except `resume_experiment_id`) match the original exactly
- Verify you're using the same git worktree/commit
- Check that model config and data config paths are identical

## Best Practices

1. **Always use a git worktree** for multi-job experiments
2. **Document the commit hash** used for each experiment
3. **Keep config files version-controlled** so you can verify they match
4. **Check job_history.json** after each job to verify progress
5. **Use descriptive job names** in SLURM to track which job is which
6. **Save experiment IDs** in a log file or notes for easy reference

## Implementation Details

### Experiment ID Generation

The experiment ID is generated from:
- Configuration hash (model, data, training parameters)
- Timestamp of first job

The config hash excludes:
- `experiment_name`
- `job_id`
- `job_name`
- `random_seed` (though this should ideally be the same)

### Resume Detection

When `resume_experiment_id` is provided:
1. System looks for experiment directory with that exact ID
2. Loads `job_history.json`
3. Determines last completed epoch
4. Finds latest checkpoint
5. Resumes from next epoch

If `resume_experiment_id` is not provided:
- New experiment is created
- New experiment ID is generated
- Training starts from epoch 0

### Checkpoint Loading

- **PyTorch**: Loads model state dict and optimizer state from checkpoint
- **Lightning**: Uses `trainer.fit(ckpt_path=...)` for automatic resume

Both paths maintain epoch continuity across jobs.
