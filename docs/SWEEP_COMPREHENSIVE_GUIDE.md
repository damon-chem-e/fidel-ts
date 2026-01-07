# W&B Sweep System - Comprehensive Guide

This guide provides complete documentation for the W&B hyperparameter sweep system, including architecture, configuration, execution, and best practices.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Hyperparameter Strategy](#hyperparameter-strategy)
6. [Run Lifecycle](#run-lifecycle)
7. [Local Registry System](#local-registry-system)
8. [SLURM Integration](#slurm-integration)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Overview

The sweep system enables hyperparameter optimization across distributed compute environments (SLURM clusters, RunPod, etc.) with intelligent local-first resumption. Key features:

- **Local Registry**: File-based tracking eliminates race conditions with W&B server
- **Location-Aware Resumption**: Runs resume only where checkpoints exist
- **Graceful Shutdown**: Signal handling for clean SLURM timeout handling
- **Multi-Platform**: Works across SLURM, RunPod, and local machines

### What It Supports

| Feature | Description |
|---------|-------------|
| Suite sweeps | Optimize hyperparameters across experiments in a suite |
| Single experiment sweeps | Optimize hyperparameters for a single experiment |
| Bayesian optimization | Efficient search using Gaussian processes |
| Grid/Random search | Traditional search methods |
| Hyperband early termination | Stop poor runs early |
| SLURM time limits | Graceful handling of job timeouts |
| Multi-location | Run agents on SLURM + RunPod simultaneously |

---

## Architecture

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LOCAL SWEEP AGENT (CLI)                              │
│                     scripts/local_sweep_agent.py                            │
│                                                                             │
│  - CLI argument parsing                                                     │
│  - Entry point for users                                                    │
│  - Creates SweepManager and calls run_loop()                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SWEEP MANAGER                                      │
│                        exp/sweep_manager.py                                  │
│                                                                             │
│  OWNS:                                                                      │
│  - SweepRegistry instance                                                   │
│  - Shutdown state (no globals)                                              │
│  - Signal handlers (SIGTERM, SIGINT)                                        │
│  - All status updates (single point of truth)                               │
│                                                                             │
│  RESPONSIBILITIES:                                                          │
│  - Check local registry for runs needing resumption                        │
│  - Request new hyperparameters from wandb controller                       │
│  - Delegate execution to SweepExecutor                                     │
│  - Receive completion signals and update registry                          │
│  - Coordinate with wandb for visibility updates                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SWEEP EXECUTOR                                      │
│                       exp/sweep_executor.py                                  │
│                                                                             │
│  OWNS:                                                                      │
│  - Nothing stateful (pure executor)                                         │
│                                                                             │
│  RESPONSIBILITIES:                                                          │
│  - Execute a single sweep trial (suite or single experiment)               │
│  - Apply hyperparameters to config                                         │
│  - Run training via SuiteExecutor or run functions                         │
│  - Detect completion signals (early stopping, hyperband, epochs)           │
│  - Return SweepTrialResult to SweepManager                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       EXPERIMENT MANAGER                                     │
│                         exp/manager.py                                       │
│                                                                             │
│  OWNS:                                                                      │
│  - Experiment directory and metadata                                        │
│  - Job history and checkpoint tracking                                      │
│  - Wandb run (if enabled)                                                   │
│                                                                             │
│  RESPONSIBILITIES:                                                          │
│  - Track epochs and checkpoints                                             │
│  - Provide completion reason signals via set_completion_reason()           │
│  - Manage experiment lifecycle                                              │
│                                                                             │
│  NOTE: Not every experiment is part of a sweep. Sweep-related              │
│  functionality is optional and only activated when in sweep context.       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Single Ownership**: Only SweepManager updates the registry
2. **No Globals**: All state is instance attributes
3. **Clear Signal Flow**: Signals → SweepManager → Registry
4. **Local-First**: Registry is authoritative for resumption, wandb is for visibility

---

## Quick Start

### 1. Create Sweep Configuration

Create a YAML file defining your sweep:

```yaml
# configs/sweep_configs/my_sweep.yaml
program: scripts/wandb_sweep_wrapper.py  # Still needed for wandb sweep create
method: bayes
metric:
  name: val_loss
  goal: minimize

# Path to your base config (suite or single experiment)
_config_path: "configs/experiment_suites/my_suite.yaml"

# Hyperparameters to sweep
parameters:
  training.learning_rate:
    distribution: log_uniform
    min: 1e-5
    max: 1e-2
  training.batch_size:
    values: [256, 512, 768, 1024]

# Optional: Early termination
early_terminate:
  type: hyperband
  min_iter: 3
  max_iter: 7
  s: 2
```

### 2. Initialize Sweep in W&B

```bash
wandb sweep configs/sweep_configs/my_sweep.yaml --project my-project
# Returns: entity/project/sweep_id
```

### 3. Run Local Sweep Agent

```bash
# Run sweep agent (resumes incomplete runs first, then gets new hparams)
python scripts/local_sweep_agent.py SWEEP_ID --project my-project

# Specify output directory
python scripts/local_sweep_agent.py SWEEP_ID --project my-project -o /scratch/output

# Limit number of runs (for SLURM time limits)
python scripts/local_sweep_agent.py SWEEP_ID --project my-project --count 3

# Only resume incomplete runs (don't start new)
python scripts/local_sweep_agent.py SWEEP_ID --project my-project --resume-only

# Only run new trials (skip incomplete check)
python scripts/local_sweep_agent.py SWEEP_ID --project my-project --new-only
```

---

## Configuration

### Suite Sweep Configuration

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
  model_config_overrides.e_layers:
    values: [2, 3, 4]
  model_config_overrides.d_model:
    values: [128, 256, 512]
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

### Parameter Naming Convention

Sweep parameters use dot notation to access nested config:

| Parameter | Maps To |
|-----------|---------|
| `training.learning_rate` | `config.training.learning_rate` |
| `training.batch_size` | `config.training.batch_size` |
| `model_config_overrides.e_layers` | `config.model_config_overrides.e_layers` |
| `model.dropout` | `config.model.dropout` |

### Distribution Types

```yaml
# Continuous distributions
parameter_name:
  distribution: uniform
  min: 0.0
  max: 1.0

parameter_name:
  distribution: log_uniform
  min: 1e-5
  max: 1e-2

parameter_name:
  distribution: normal
  mu: 0.5
  sigma: 0.1

# Discrete distributions
parameter_name:
  values: [1, 2, 4, 8]

parameter_name:
  distribution: q_uniform
  min: 64
  max: 1024
  q: 64  # Quantization step
```

---

## Hyperparameter Strategy

### Recommended Approach: Combined Sweep

For most use cases (model architecture + training hyperparameters), use a **combined sweep** with Bayesian optimization:

**Advantages:**
- Efficiency: Learns from all parameter combinations simultaneously
- Interactions: Captures interactions between architecture and training parameters
- State-of-the-art: Modern HPO approach
- Time savings: Single sweep vs. multiple sequential sweeps

### Recommended Parameter Ranges

**Architecture Parameters:**
```yaml
model_config_overrides.e_layers:
  values: [2, 3, 4, 5]        # Number of transformer layers
model_config_overrides.d_model:
  values: [128, 256, 512]      # Model dimension
model_config_overrides.n_heads:
  values: [4, 8]               # Attention heads
```

**Training Parameters:**
```yaml
training.learning_rate:
  distribution: log_uniform
  min: 1e-5
  max: 1e-2
training.batch_size:
  values: [256, 512, 768, 1024]
```

### Search Methods

| Method | Best For | Description |
|--------|----------|-------------|
| `bayes` | Continuous parameters | Uses Gaussian Process to model performance surface |
| `grid` | Small discrete spaces | Exhaustive search of all combinations |
| `random` | Large spaces | Random sampling, good baseline |

### Early Termination (Hyperband)

Stop poor runs early to save compute:

```yaml
early_terminate:
  type: hyperband
  min_iter: 3      # Minimum epochs before stopping
  max_iter: 7      # Maximum epochs for evaluation
  s: 2             # Successive halving factor (keep top 50%)
```

**How Hyperband Works with Asynchronous Jobs:**

1. **Bracket Formation**: Runs are grouped into brackets based on start time
2. **Asynchronous Evaluation**: When a run completes a checkpoint, W&B compares to others
3. **Flexible Pruning**: Significantly worse runs can be stopped early

**Detection in Code:**
```python
# The sweep system detects Hyperband pruning via:
if wandb.run and getattr(wandb.run, 'stopped', False):
    # Run was pruned by Hyperband
    completion_reason = "hyperband"
```

### Sequential/Tiered Sweeps (Alternative)

When compute is very limited:

**Stage 1: Architecture Search**
- Fix training params (use defaults)
- Sweep: e_layers, d_model, n_heads
- Find best architecture

**Stage 2: Training Optimization**
- Fix architecture (from Stage 1)
- Sweep: learning_rate, batch_size
- Fine-tune training

---

## Run Lifecycle

### Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SWEEP INITIALIZATION                                  │
│  wandb sweep configs/sweep_configs/my_sweep.yaml                        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    SWEEP CREATED IN W&B                                  │
│  • Sweep ID: entity/project/sweep_abc123                                │
│  • Parameters and method stored in W&B cloud                            │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LOCAL SWEEP AGENT STARTS                             │
│  python scripts/local_sweep_agent.py SWEEP_ID                          │
│                                                                          │
│  1. Creates SweepManager                                                 │
│  2. Registers signal handlers                                            │
│  3. Calls run_loop()                                                     │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    CHECK FOR INCOMPLETE RUNS                            │
│  SweepManager queries local registry for runs needing resume           │
│                                                                          │
│  If incomplete runs exist:                                               │
│    → Resume highest priority run (most progress)                         │
│                                                                          │
│  If no incomplete runs:                                                  │
│    → Request new hyperparameters from W&B                               │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    EXECUTE TRIAL                                         │
│  SweepExecutor.execute_trial()                                          │
│                                                                          │
│  1. Apply sweep parameters to config                                     │
│  2. Initialize experiment (get IDs)                                      │
│  3. Run training via SuiteExecutor or run function                      │
│  4. Detect completion reason                                             │
│  5. Return SweepTrialResult                                             │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    FINALIZE RUN                                          │
│  SweepManager._finalize_run(result)                                     │
│                                                                          │
│  Based on SweepTrialResult:                                             │
│  • COMPLETED → registry.mark_complete()                                 │
│  • INTERRUPTED → registry.mark_needs_resume()                           │
│  • FAILED → registry.mark_failed()                                      │
│                                                                          │
│  Also updates wandb.summary (for visibility)                            │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LOOP CONTINUES                                        │
│  • Check run limit                                                       │
│  • Check shutdown flag                                                   │
│  • Repeat from "Check for incomplete runs"                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### Completion Detection

The system detects completion in priority order:

| Priority | Signal | Source | Detection Method |
|----------|--------|--------|------------------|
| 1 | Early stopping | Training callback | `exp_manager.set_completion_reason("early_stopping")` |
| 2 | Hyperband pruning | Sweep controller | `wandb.run.stopped == True` |
| 3 | Convergence | Training code | `exp_manager.set_completion_reason("converged")` |
| 4 | All epochs | Training loop | `current_epoch >= total_epochs` |

**Setting Completion Reason in Training Code:**

```python
# In your training loop's early stopping callback:
if early_stopping.should_stop:
    exp_manager.set_completion_reason("early_stopping")
    break

# For convergence detection:
if loss < convergence_threshold:
    exp_manager.set_completion_reason("converged")
    break
```

### Interruption Handling

When a signal is received (SIGTERM, SIGINT):

1. **SweepManager's signal handler** sets `shutdown_requested = True`
2. **Training detects shutdown** via callback and exits gracefully
3. **SweepExecutor returns** `SweepTrialResult` with `interrupted=True`
4. **SweepManager finalizes** and marks run as `NEEDS_RESUME`

---

## Local Registry System

### Why Local Registry?

The local registry solves several problems:

1. **No Race Conditions**: Local file I/O is atomic with locking
2. **Location-Aware**: Runs only resume where checkpoints exist
3. **Fast**: No API calls needed to check resumption status
4. **Robust**: Works even if W&B is temporarily unavailable

### Registry Location

Each sweep has its own registry file:

```
output/
└── suite_name_timestamp/
    ├── .sweep_registry.json      # Registry for this sweep
    ├── .sweep_registry.json.lock # Lock file
    └── experiment_id/
        ├── checkpoints/
        ├── job_history.json
        └── ...
```

### Registry Status Values

| Status | Meaning |
|--------|---------|
| `registered` | Run started but not yet training |
| `running` | Currently training |
| `needs_resume` | Interrupted, should be resumed |
| `completed` | Successfully finished |
| `failed` | Failed, should not be resumed |

### Completion Reasons

| Reason | Description |
|--------|-------------|
| `all_epochs` | Finished all planned epochs |
| `early_stopping` | Patience-based early stopping triggered |
| `hyperband` | Sweep controller (Hyperband) pruned |
| `converged` | Reached convergence threshold |
| `manual_stop` | Intentionally stopped |

### Interrupt Reasons

| Reason | Description |
|--------|-------------|
| `slurm_timeout` | SLURM job time limit reached |
| `sigterm` | Generic SIGTERM signal |
| `sigint` | Ctrl+C / SIGINT signal |
| `preemption` | Cloud/cluster preemption |
| `unknown` | Unknown interruption |

---

## SLURM Integration

### Recommended SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=sweep-agent
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/sweep_agent_%j.out
#SBATCH --signal=B:TERM@120   # Send SIGTERM 120s before time limit

# Set location identifier for this cluster
export SWEEP_AGENT_LOCATION="slurm-cluster-name"

# Activate environment
source venv/bin/activate

# Run sweep agent
# --count 3 limits runs to stay within time limit
python scripts/local_sweep_agent.py $SWEEP_ID \
    --project my-project \
    --output-dir /scratch/$USER/output \
    --count 3
```

### SLURM Array Jobs

For running multiple agents in parallel:

```bash
#!/bin/bash
#SBATCH --job-name=sweep-array
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-9  # Run 10 agents
#SBATCH --output=logs/sweep_agent_%A_%a.out
#SBATCH --signal=B:TERM@120

export SWEEP_AGENT_LOCATION="slurm-cluster-name"
source venv/bin/activate

python scripts/local_sweep_agent.py $SWEEP_ID \
    --project my-project \
    --count 3
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SWEEP_AGENT_LOCATION` | Location identifier for this machine | hostname |
| `SWEEP_AGENT_MACHINE_ID` | Unique machine ID | hostname |
| `WANDB_ENTITY` | Default W&B entity | None |

---

## Troubleshooting

### Run Marked as Incomplete When Complete

**Symptom:** Run finished early stopping but is being resumed.

**Cause:** `set_completion_reason()` was not called.

**Fix:** In your training code, call:
```python
exp_manager.set_completion_reason("early_stopping")
```

### Runs Not Resuming

**Symptom:** Interrupted runs are not being picked up for resumption.

**Causes and Fixes:**

1. **Wrong location**: Runs only resume on the machine where checkpoints exist.
   - Check `SWEEP_AGENT_LOCATION` matches the original run.

2. **Registry not found**: Agent can't find the registry.
   - Ensure `--output-dir` points to the same location.

3. **Run marked as failed**: Run had an error and can't be resumed.
   - Check registry status: `cat output/suite_dir/.sweep_registry.json`

### Sweep Shows as Finished But Runs Incomplete

**Symptom:** W&B sweep dashboard shows "FINISHED" but local runs need resumption.

**Explanation:** W&B sweep state is based on agent activity, not run completion.

**Fix:** Use local registry as source of truth. Run agent with `--resume-only`:
```bash
python scripts/local_sweep_agent.py SWEEP_ID --project my-project --resume-only
```

### Signal Handler Conflicts

**Symptom:** Training doesn't exit gracefully on SIGTERM.

**Cause:** Multiple signal handlers registered.

**Fix:** The new architecture ensures SweepManager is the single owner of signal handling. Ensure you're using `local_sweep_agent.py`, not the old `smart_sweep_agent.py`.

### Registry Lock Timeout

**Symptom:** Agent hangs when accessing registry.

**Cause:** Another process holds the lock file.

**Fix:** 
1. Check for zombie processes: `ps aux | grep sweep`
2. Remove lock file if needed: `rm output/suite_dir/.sweep_registry.json.lock`

---

## Best Practices

### General Recommendations

1. **Start Small**: Test with a few runs before launching full sweep
2. **Use Bayesian**: For continuous parameters, `bayes` method is most efficient
3. **Set Time Limits**: Use SLURM `--signal=B:TERM@120` for graceful shutdown
4. **Limit Runs**: Use `--count` to stay within SLURM time limits
5. **Monitor Progress**: Check both W&B dashboard and local registry

### For SLURM Users

1. **Pre-warm Workers**: First job is slower due to loading. Run warmup jobs if needed.
2. **Choose Run Count Wisely**: Estimate runs per time slot, add buffer.
3. **Use Shared Storage**: Output directory must be accessible for resumption.
4. **Array Jobs**: Use SLURM arrays for parallel agents instead of multiple submissions.

### For Multi-Location Sweeps

When running on SLURM + RunPod simultaneously:

1. **Set Location**: Use `SWEEP_AGENT_LOCATION` to identify each platform.
2. **Separate Storage**: Each location needs its own registry (automatic).
3. **Same Output Path**: Keep output directory structure consistent.
4. **Resume on Same Location**: Checkpoints stay where they were created.

### Metrics to Track

| Metric | Purpose |
|--------|---------|
| `val_loss` | Primary optimization target |
| `train_loss` | Detect overfitting |
| `val_mae`, `val_rmse` | Additional evaluation |
| Training time per epoch | Resource planning |

### Common Pitfalls to Avoid

1. **Don't forget `set_completion_reason()`** when training ends early
2. **Don't use multiple agents on same output directory** without location IDs
3. **Don't ignore the 120s buffer** for SLURM signal handling
4. **Don't rely on W&B state** for resumption decisions (use local registry)

---

## File Reference

### New Architecture Files

| File | Purpose |
|------|---------|
| `exp/sweep_manager.py` | Central orchestrator, owns registry and signals |
| `exp/sweep_executor.py` | Pure executor, runs single trials |
| `exp/sweep_registry.py` | Local file-based registry |
| `scripts/local_sweep_agent.py` | CLI entry point |

### Configuration Files

| File | Purpose |
|------|---------|
| `configs/sweep_configs/*.yaml` | Sweep configuration files |
| `configs/experiment_suites/*.yaml` | Suite configurations |
| `configs/experiments/*.yaml` | Single experiment configurations |

### Output Files

| File | Purpose |
|------|---------|
| `output/suite_dir/.sweep_registry.json` | Local run status registry |
| `output/suite_dir/exp_id/job_history.json` | Experiment job tracking |
| `output/suite_dir/exp_id/checkpoints/` | Model checkpoints |

---

## API Reference

### SweepManager

```python
from exp.sweep_manager import SweepManager

manager = SweepManager(
    entity="my-team",           # W&B entity
    project="my-project",       # W&B project
    sweep_id="abc123",          # W&B sweep ID
    output_dir="./output",      # Output directory
    location=None,              # Auto-detect from env
    config_path=None,           # Base config (can be in sweep config)
    experiment_type="pytorch"   # Default experiment type
)

# Run the main loop
runs_completed = manager.run_loop(
    count=5,            # Max runs (None = unlimited)
    resume_only=False,  # Only resume incomplete runs
    new_only=False      # Skip incomplete check
)
```

### SweepTrialResult

```python
from exp.sweep_executor import SweepTrialResult, CompletionReason

result = SweepTrialResult(
    success=True,
    interrupted=False,
    final_epoch=100,
    total_epochs=100,
    completion_reason=CompletionReason.ALL_EPOCHS.value,
    interrupt_reason=None,
    error=None,
    experiment_id="20240106_abc123",
    suite_id="my_suite_20240106",
    wandb_run_id="xyz789",
    hyperparams={"lr": 0.001},
    metrics={"val_loss": 0.15}
)

# Properties
result.is_complete    # True if should NOT be resumed
result.should_resume  # True if should be resumed
result.is_failed      # True if failed permanently
```

### ExperimentManager Integration

```python
from exp.manager import ExperimentManager

# In training loop, when early stopping triggers:
if early_stopping.should_stop:
    exp_manager.set_completion_reason("early_stopping")
    break

# When convergence is reached:
if loss < threshold:
    exp_manager.set_completion_reason("converged")
    break
```

---

## Summary

The sweep system provides robust hyperparameter optimization with:

- **Local-first design**: Registry eliminates race conditions
- **Location awareness**: Runs resume where checkpoints exist
- **Clean architecture**: Single owner of state and signals
- **Flexible execution**: Works on SLURM, RunPod, and local machines

For questions or issues, check the troubleshooting section or examine the local registry (`cat output/suite_dir/.sweep_registry.json`) for debugging information.
