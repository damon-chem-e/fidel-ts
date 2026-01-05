# WandB Sweep Flow: Complete Pathway from Start to Completion

This document explains the complete flow of a WandB hyperparameter sweep, from initialization through execution to completion, showing where all information is stored.

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    WANDB SWEEP INITIALIZATION                            │
│  wandb sweep configs/sweep_configs/lynx_film_raw_architecture_sweep.yaml│
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    WANDB SWEEP CREATED                                   │
│  • Sweep ID: entity/project/sweep_abc123                                │
│  • Sweep config stored in WandB cloud                                    │
│  • Parameters defined: e_layers, d_model, n_heads, lr, batch_size       │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    AGENT STARTS (wandb agent SWEEP_ID)                  │
│  • Agent polls WandB for new hyperparameter combinations                │
│  • Gets suggested params: {e_layers: 3, d_model: 256, lr: 0.001, ...} │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    WANDB RUN CREATED                                     │
│  • Run ID: run_xyz789                                                   │
│  • Status: not_started                                                  │
│  • Config: {hyperparameters + _config_path, _job_status: "not_started"} │
│  • Stored: WandB cloud (wandb.ai)                                       │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    SWEEP WRAPPER: PHASE 1 - INITIALIZATION               │
│  scripts/wandb_sweep_wrapper.py                                        │
│                                                                          │
│  1. Load base suite config                                               │
│     configs/experiment_suites/lynx_film_raw_hparam_sweep.yaml          │
│                                                                          │
│  2. Apply sweep hyperparameters                                         │
│     • training.learning_rate → suite.experiments[0].overrides.training   │
│     • model_config_overrides.e_layers → suite.experiments[0].overrides  │
│                                                                          │
│  3. Check job status: "not_started"                                     │
│                                                                          │
│  4. Run SuiteExecutor(init_only=True, return_ids=True)                  │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT INITIALIZATION                            │
│  SuiteExecutor.execute() with init_only=True                            │
│                                                                          │
│  1. Create suite directory:                                              │
│     outputs/suites/lynx_film_raw_hparam_sweep_20250109_120000/          │
│                                                                          │
│  2. For each experiment:                                                 │
│     • Create ExperimentManager                                           │
│     • Generate experiment_id: "20250109-120000_abc123def456"            │
│     • Create experiment directory structure                              │
│     • Save configs, metadata, job_history.json                          │
│                                                                          │
│  3. Return IDs:                                                          │
│     {                                                                     │
│       'suite_id': 'lynx_film_raw_hparam_sweep_20250109_120000',         │
│       'experiment_ids': {                                                │
│         'lynx_film_raw_nyc_traffic_speed_24_sweep':                     │
│           '20250109-120000_abc123def456'                                │
│       }                                                                  │
│     }                                                                    │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STORE IDs IN WANDB                                    │
│  wandb_run.config.update({                                              │
│    '_suite_id': 'lynx_film_raw_hparam_sweep_20250109_120000',          │
│    '_experiment_ids': {...},                                             │
│    '_job_status': 'initialized'                                          │
│  })                                                                      │
│                                                                          │
│  Stored: WandB cloud (wandb.ai) - accessible across all agent runs     │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    SWEEP WRAPPER: PHASE 2 - EXECUTION                    │
│                                                                          │
│  1. Set resume IDs in suite config:                                      │
│     • suite.resume_suite_id = stored suite_id                           │
│     • experiment.overrides.resume_experiment_id = stored exp_id         │
│                                                                          │
│  2. Update job status: "running"                                         │
│                                                                          │
│  3. Run SuiteExecutor(init_only=False)                                  │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT EXECUTION                                  │
│  SuiteExecutor.execute() with init_only=False                           │
│                                                                          │
│  1. Load existing experiment (via resume_experiment_id)                  │
│     • ExperimentManager detects resume                                  │
│     • Loads job_history.json                                            │
│     • Gets stored wandb_run_id from job_history                         │
│                                                                          │
│  2. Resume WandB run:                                                    │
│     • Uses stored wandb_run_id                                          │
│     • Continues logging to same WandB run                                │
│                                                                          │
│  3. Training loop:                                                       │
│     • Load data, initialize model with hyperparameters                   │
│     • Train for 7 epochs                                                 │
│     • Log metrics to WandB (val_loss, train_loss, etc.)                  │
│     • Save checkpoints locally                                           │
│                                                                          │
│  4. Final metrics:                                                       │
│     • Logged to WandB run                                                │
│     • Saved in experiment directory                                     │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMPLETION                                            │
│                                                                          │
│  1. Update job status: "completed"                                       │
│  2. Final metrics logged to WandB                                        │
│  3. Agent reports completion to WandB                                    │
│  4. Agent requests next hyperparameter combination                       │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    REPEAT FOR NEXT HYPERPARAMETER COMBINATION            │
│  (Each combination is a separate WandB run with its own experiment)     │
└─────────────────────────────────────────────────────────────────────────┘
```

## Information Storage Locations

### 1. WandB Cloud (wandb.ai)

**Sweep Level:**
- **Location**: `https://wandb.ai/entity/project/sweeps/SWEEP_ID`
- **Contains**:
  - Sweep configuration (parameters, method, metric)
  - All run IDs in the sweep
  - Best run identification
  - Sweep progress and statistics

**Run Level (per hyperparameter combination):**
- **Location**: `https://wandb.ai/entity/project/runs/RUN_ID`
- **Contains**:
  - Hyperparameters for this run
  - All logged metrics (val_loss, train_loss per epoch)
  - System metrics (GPU usage, memory)
  - Config metadata:
    - `_config_path`: Base config file
    - `_suite_id`: Suite directory name
    - `_experiment_ids`: Mapping of experiment names to IDs
    - `_job_status`: Current job status
  - Artifacts (if logged)
  - Run logs

### 2. Local Filesystem

**Suite Directory:**
```
outputs/suites/lynx_film_raw_hparam_sweep_20250109_120000/
├── suite_metadata.json          # Suite-level metadata
├── suite_config.yaml            # Complete suite config
└── [experiment directories]
```

**Experiment Directory (per hyperparameter combination):**
```
outputs/suites/lynx_film_raw_hparam_sweep_20250109_120000/
└── 20250109-120000_abc123def456/  # Unique experiment ID
    ├── configs/
    │   ├── experiment_config.yaml  # Complete experiment config
    │   ├── model_config.yaml       # Model config with overrides
    │   └── data_config.yaml        # Data config
    ├── metadata.json               # Experiment metadata
    ├── job_history.json            # Job tracking (SLURM jobs, epochs)
    │                               # Contains: wandb_run_id
    ├── checkpoints/                # Model checkpoints
    │   ├── checkpoint-0-*.pth
    │   ├── checkpoint-1-*.pth
    │   └── ...
    ├── metrics/                    # Local metrics storage
    │   └── metrics.json
    ├── logs/                       # Training logs
    └── wandb/                      # Local WandB cache
        └── [wandb internal files]
```

**Key Files:**

1. **`job_history.json`**:
   ```json
   {
     "experiment_id": "20250109-120000_abc123def456",
     "wandb_run_id": "run_xyz789",
     "jobs": [
       {
         "job_id": "12345",
         "slurm_job_id": "12345",
         "start_epoch": 0,
         "end_epoch": 7,
         "status": "completed"
       }
     ],
     "current_epoch": 7
   }
   ```

2. **`experiment_config.yaml`**:
   - Complete experiment configuration
   - Includes all hyperparameters (from sweep)
   - Includes resume IDs

3. **`metadata.json`**:
   - Git commit hash
   - Timestamp
   - System info
   - Job info

## Data Flow for Each Hyperparameter Combination

### Step-by-Step for One Run

1. **Agent gets hyperparameters**:
   - WandB suggests: `{e_layers: 3, d_model: 256, lr: 0.001, batch_size: 512}`
   - Creates new WandB run with these params

2. **Initialization**:
   - Wrapper applies params to suite config
   - Creates experiment structure
   - Generates unique experiment_id (based on config hash)
   - Stores IDs in WandB run config

3. **Execution**:
   - Loads experiment via resume_experiment_id
   - Resumes WandB run (same run_id)
   - Trains with hyperparameters
   - Logs metrics to WandB

4. **Completion**:
   - Final metrics logged
   - Status updated to "completed"
   - Agent requests next combination

### Multiple Runs in Parallel

When running multiple agents:
- Each agent gets different hyperparameter combinations
- Each creates its own WandB run
- Each creates its own experiment directory
- All runs visible in WandB sweep dashboard

## Resume Scenario

If a SLURM job times out:

1. **Agent resumes**:
   - WandB agent continues same run
   - Wrapper detects `_job_status: "initialized"` or `"running"`

2. **Load stored IDs**:
   - Gets `_suite_id` and `_experiment_ids` from WandB config
   - Sets resume IDs in config

3. **Resume training**:
   - ExperimentManager loads checkpoint
   - Continues from last epoch
   - Logs to same WandB run

## Key Points

1. **Each hyperparameter combination = separate experiment**:
   - Unique experiment_id (from config hash)
   - Unique WandB run
   - Separate experiment directory

2. **IDs stored in WandB for persistence**:
   - Survives agent restarts
   - Accessible across SLURM jobs
   - Links WandB run to local experiment

3. **Two-phase execution**:
   - Phase 1: Initialize → get IDs → store in WandB
   - Phase 2: Load IDs → set resume → execute

4. **Results aggregation**:
   - WandB sweep dashboard shows all runs
   - Compare hyperparameter combinations
   - Identify best configuration
   - Local directories contain full experiment data

## Example: Tracking a Specific Run

To find information for a specific hyperparameter combination:

1. **In WandB**:
   - Go to sweep dashboard
   - Find run with desired hyperparameters
   - View metrics, config, logs

2. **Locally**:
   - Get `_experiment_id` from WandB run config
   - Navigate to: `outputs/suites/{suite_id}/{experiment_id}/`
   - View checkpoints, configs, logs

3. **Resume if needed**:
   - Use `_suite_id` and `_experiment_id` from WandB
   - Set in suite config as resume IDs
   - Continue training
