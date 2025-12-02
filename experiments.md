# Experiments Documentation and Tracking Plan

This document describes the experiment classes in the `exp/` directory and provides a comprehensive plan for implementing detailed experiment tracking with Weights & Biases (wandb) integration.

## Current Experiment Classes

### Base Class: `Exp_Basic`

**Location**: `exp/exp_basic.py`

**Purpose**: Abstract base class providing core functionality for all time series forecasting experiments.

**Key Features**:
- Device management (CPU/GPU, multi-GPU support)
- Model initialization framework
- Data provider setup
- Optimizer and criterion selection
- Model profiling utilities (FLOPs, parameter count)
- Abstract methods for training, validation, and testing

**Methods**:
- `_acquire_device()`: Configures computing device (CPU or CUDA)
- `_select_optimizer()`: Creates optimizer (default: Adam)
- `_select_criterion()`: Creates loss function (L1 or MSE)
- `_get_profile()`: Analyzes model computational complexity
- `_build_model()`: Abstract method for model construction
- `_get_data()`: Abstract method for data retrieval
- `train()`, `vali()`, `test()`: Abstract methods for experiment phases

**Usage**: Not used directly; serves as base class for all experiment implementations.

---

### `Experiment` (Universal PyTorch)

**Location**: `exp/exp_universal.py`

**Purpose**: Main experiment orchestrator for universal time series forecasting models using standard PyTorch.

**Key Features**:
- Multi-GPU training support via DataParallel
- Early stopping with configurable patience
- Learning rate scheduling (multiple strategies)
- Cross-modal data handling (text, events, heterogeneous data)
- Comprehensive checkpointing
- Progress tracking with detailed metrics

**Training Workflow**:
1. Loads train/val/test data loaders
2. Initializes optimizer and loss function
3. Training loop with epoch-wise validation
4. Early stopping based on validation loss
5. Saves best model checkpoint
6. Logs training/validation/test metrics per epoch

**Checkpoint Structure**:
- `checkpoints/{setting}/args.json`: Full experiment configuration
- `checkpoints/{setting}/checkpoint.pth`: Best model weights

**Metrics Tracked**:
- Training loss (per epoch)
- Validation loss (per epoch)
- Test loss (per epoch)
- Training speed (iterations/second)
- Estimated time remaining

**Dependencies**: `exp.exp_basic`, `models`, `utils.tools`, `data_provider.data_factory`

---

### `TimeSeriesLightningModel` and `train_lightning_model`

**Location**: `exp/exp_lightning.py`

**Purpose**: PyTorch Lightning-based experiment framework for structured training and better multi-GPU support.

**Key Features**:
- PyTorch Lightning module wrapper
- Automatic mixed precision training (32/16/bf16)
- Gradient clipping support
- TensorBoard logging integration
- DDP (Distributed Data Parallel) for multi-GPU
- Test after epoch option
- Automatic checkpoint management

**Training Workflow**:
1. Initializes Lightning data module
2. Configures Lightning Trainer with callbacks
3. Early stopping and model checkpointing callbacks
4. TensorBoard logger for experiment tracking
5. Training with automatic validation
6. Optional test after each epoch
7. Final test on best model

**Checkpoint Structure**:
- `checkpoints/{setting}/args.json`: Experiment configuration
- `checkpoints/{setting}/checkpoint-{epoch}-{val_loss}.ckpt`: Versioned checkpoints
- `checkpoints/{setting}/last.ckpt`: Last checkpoint
- `checkpoints/tb_logs/{setting}/`: TensorBoard logs
- `checkpoints/{setting}/test_results.json`: Per-subset test results
- `checkpoints/{setting}/test_results_average.json`: Average test loss

**Metrics Tracked**:
- Training loss (per step and epoch)
- Validation loss (per epoch)
- Test loss (per subset and overall)
- Learning rate (per epoch)

**Dependencies**: `pytorch_lightning`, `exp.exp_basic`, `models`, `data_provider.lightning_data_module`

---

### `Experiment` (LLM)

**Location**: `exp/exp_llm.py`

**Purpose**: Specialized experiment orchestrator for Large Language Model (LLM) based time series forecasting.

**Key Features**:
- LLM-specific model initialization
- Memory-efficient inference
- Robust error handling and logging
- Batch processing optimization
- Parallel processing support (ThreadPoolExecutor)
- Filtered sample support
- Error logging with detailed stack traces

**Testing Workflow**:
1. Loads test datasets (supports filtered samples)
2. Processes samples in parallel or sequentially
3. Handles errors gracefully with detailed logging
4. Saves per-sample results as JSON files
5. Aggregates results per dataset subset
6. Generates comprehensive error reports

**Output Structure**:
- `checkpoints/{setting}/{subset_id}/{date}_result.json`: Per-sample prediction results
- `checkpoints/{setting}/{subset_id}/{subset_id}_result.json`: Aggregated results per subset
- `checkpoints/{setting}/error_log.txt`: Detailed error log with timestamps

**Metrics Tracked**:
- Successfully processed samples
- Skipped samples (already exist)
- Prediction shape mismatch errors
- General errors
- Per-subset and overall statistics

**Dependencies**: `exp.exp_basic`, `models`, `data_provider.data_factory`, `utils.tools`

---

### `Experiment` (Foundation Models)

**Location**: `exp/exp_fm.py`

**Purpose**: Experiment orchestrator for Foundation Model (FM) based time series forecasting.

**Key Features**:
- Foundation model initialization with pre-trained weights
- Individual channel training support
- Memory-optimized device management
- Cross-modal data support (TSF and TGTSF tasks)
- Filtered sample testing
- Comprehensive error handling

**Testing Workflow**:
1. Loads test data loaders (supports filtered samples)
2. Processes samples with individual or joint channel handling
3. Handles model failures gracefully
4. Saves metrics per subset and overall
5. Tracks errors separately

**Output Structure**:
- `checkpoints/{setting}/all_test_metrics.json`: Per-subset test metrics
- `checkpoints/{setting}/final_test_result.json`: Overall test result
- `checkpoints/{setting}/overall_error.json`: Error statistics

**Metrics Tracked**:
- Test loss per dataset subset
- Overall test loss
- Error count per subset
- Overall error count

**Dependencies**: `exp.exp_basic`, `models`, `data_provider.data_factory`, `utils.tools`

---

## Experiment Tracking Implementation Plan

### Overview

The goal is to implement comprehensive experiment tracking that:
- Links to Weights & Biases (wandb) for cloud-based experiment management
- Maintains local experiment folders with complete reproducibility information
- Tracks all metrics both locally and in wandb
- Links nested configuration structures
- Captures environment and execution metadata

### Target Directory Structure

```
outputs/
├── {experiment_id}/                    # Unique experiment folder
│   ├── configs/                        # Complete configuration hierarchy
│   │   ├── primary_config.yaml         # Main experiment config
│   │   ├── model_config.yaml           # Model-specific config
│   │   ├── data_config.yaml            # Data-specific config
│   │   ├── plotting_config.yaml        # Visualization config (if used)
│   │   └── evaluation_config.yaml      # Evaluation config (if used)
│   ├── checkpoints/                    # Model checkpoints
│   │   ├── best.pth                    # Best model checkpoint
│   │   ├── last.pth                    # Last checkpoint
│   │   └── epoch_*.pth                 # Epoch-specific checkpoints (optional)
│   ├── visualizations/                 # Generated plots and figures
│   │   ├── training_curves.png
│   │   ├── predictions_*.png
│   │   └── ...
│   ├── logs/                           # Log files
│   │   ├── experiment.log              # Main experiment log
│   │   ├── training.log                # Training-specific log
│   │   ├── errors.log                  # Error log (if any)
│   │   └── wandb_sync.log              # Wandb sync log
│   ├── metrics/                        # Local metrics storage
│   │   ├── training_metrics.json       # Training metrics per epoch
│   │   ├── validation_metrics.json     # Validation metrics per epoch
│   │   ├── test_metrics.json           # Test metrics
│   │   └── summary.json                # Experiment summary
│   ├── metadata/                       # Experiment metadata
│   │   ├── git_info.json               # Git commit hash, branch, diff
│   │   ├── environment.json            # Python version, package versions
│   │   ├── system_info.json            # System information
│   │   ├── job_info.json               # Job ID, job name, scheduler info
│   │   └── timestamp.json              # Start/end timestamps
│   ├── wandb/                          # Local wandb files (if offline)
│   │   └── ...
│   └── README.md                       # Human-readable experiment summary
```

### Implementation Components

#### 1. Experiment Manager (`exp/experiment_manager.py`)

**Purpose**: Centralized experiment tracking and management.

**Key Responsibilities**:
- Create and manage experiment directories
- Initialize wandb runs
- Track configuration files (primary + nested)
- Capture git information
- Log system and environment information
- Manage job IDs and names
- Coordinate metric logging (local + wandb)
- Handle experiment lifecycle (start, checkpoint, end)

**Key Methods**:
```python
class ExperimentManager:
    def __init__(self, config_path, output_base_dir="outputs"):
        """Initialize experiment manager with config and output directory."""
        
    def start_experiment(self):
        """Create experiment directory, initialize wandb, capture metadata."""
        
    def log_configs(self, primary_config, nested_configs):
        """Save all configuration files to experiment folder."""
        
    def capture_git_info(self):
        """Capture git commit hash, branch, and diff."""
        
    def capture_environment(self):
        """Capture Python version, package versions, system info."""
        
    def set_job_info(self, job_id, job_name):
        """Set job ID and name (from scheduler or manual)."""
        
    def log_metrics(self, metrics_dict, step=None, commit=True):
        """Log metrics to both local JSON and wandb."""
        
    def save_checkpoint(self, model, optimizer, epoch, metrics):
        """Save model checkpoint and link to wandb artifact."""
        
    def save_visualization(self, fig, name):
        """Save visualization and log to wandb."""
        
    def end_experiment(self, final_metrics):
        """Finalize experiment, close wandb run, generate summary."""
```

#### 2. Configuration Loader Enhancement (`cli/config/loader.py`)

**Purpose**: Enhanced config loading with nested config support and tracking.

**Enhancements**:
- Load primary config with nested config references
- Resolve all nested config paths
- Validate config structure
- Return config hierarchy for tracking
- Support config inheritance and overrides

**Key Methods**:
```python
def load_config_with_nested(config_path):
    """Load primary config and all nested configs, return hierarchy."""
    
def resolve_config_paths(config, base_dir):
    """Resolve relative paths in config to absolute paths."""
    
def validate_config_structure(config):
    """Validate config structure and required fields."""
    
def get_config_hierarchy(config):
    """Return complete config hierarchy for tracking."""
```

#### 3. Wandb Integration (`exp/wandb_integration.py`)

**Purpose**: Wandb-specific integration utilities.

**Key Features**:
- Initialize wandb run with proper configuration
- Link job ID and job name to wandb run
- Log nested configs as wandb config
- Create wandb artifacts for checkpoints
- Log visualizations as wandb images
- Sync metrics in real-time
- Handle offline mode gracefully

**Key Methods**:
```python
def init_wandb_run(experiment_id, config, job_id=None, job_name=None):
    """Initialize wandb run with experiment metadata."""
    
def log_config_to_wandb(config_hierarchy):
    """Log complete config hierarchy to wandb."""
    
def log_checkpoint_artifact(checkpoint_path, experiment_id):
    """Create wandb artifact for model checkpoint."""
    
def log_visualization_to_wandb(fig, name, step=None):
    """Log visualization to wandb as image."""
    
def log_metrics_to_wandb(metrics, step=None):
    """Log metrics dictionary to wandb."""
```

#### 4. Metadata Capture (`exp/metadata_capture.py`)

**Purpose**: Capture and store experiment metadata.

**Key Features**:
- Git information capture (commit hash, branch, diff)
- Environment capture (Python version, package versions)
- System information (OS, CPU, GPU, memory)
- Job information (ID, name, scheduler type)
- Timestamp tracking (start, checkpoints, end)

**Key Methods**:
```python
def capture_git_info(repo_path="."):
    """Capture git commit hash, branch, and uncommitted diff."""
    
def capture_environment():
    """Capture Python version and installed package versions."""
    
def capture_system_info():
    """Capture system information (OS, CPU, GPU, memory)."""
    
def capture_job_info():
    """Capture job ID and name from environment variables or scheduler."""
    
def get_timestamps():
    """Get current timestamp for experiment tracking."""
```

#### 5. Metrics Logger (`exp/metrics_logger.py`)

**Purpose**: Unified metrics logging to local files and wandb.

**Key Features**:
- Log metrics to local JSON files (per epoch, per subset)
- Log metrics to wandb in real-time
- Support for nested metrics (per-subset, per-channel)
- Automatic metric aggregation
- Metric history tracking

**Key Methods**:
```python
class MetricsLogger:
    def log_training_metrics(self, epoch, metrics):
        """Log training metrics for an epoch."""
        
    def log_validation_metrics(self, epoch, metrics):
        """Log validation metrics for an epoch."""
        
    def log_test_metrics(self, metrics, subset=None):
        """Log test metrics (overall or per-subset)."""
        
    def get_metric_history(self, metric_name):
        """Retrieve metric history for analysis."""
```

### Integration Points

#### Modified Experiment Classes

Each experiment class will be modified to:

1. **Initialize ExperimentManager**:
   ```python
   def __init__(self, args):
       super().__init__(args)
       self.exp_manager = ExperimentManager(
           config_path=args.config_path,
           output_base_dir=args.output_dir or "outputs"
       )
       self.exp_manager.start_experiment()
   ```

2. **Log Configurations**:
   ```python
   # At experiment start
   config_hierarchy = load_config_with_nested(args.config_path)
   self.exp_manager.log_configs(
       primary_config=config_hierarchy['primary'],
       nested_configs=config_hierarchy['nested']
   )
   ```

3. **Log Metrics During Training**:
   ```python
   # In training loop
   metrics = {
       'train_loss': train_loss,
       'val_loss': val_loss,
       'test_loss': test_loss,
       'learning_rate': current_lr
   }
   self.exp_manager.log_metrics(metrics, step=epoch)
   ```

4. **Save Checkpoints**:
   ```python
   # When saving checkpoint
   self.exp_manager.save_checkpoint(
       model=self.model,
       optimizer=optimizer,
       epoch=epoch,
       metrics=metrics
   )
   ```

5. **Save Visualizations**:
   ```python
   # When creating plots
   fig = create_prediction_plot(...)
   self.exp_manager.save_visualization(fig, name=f"prediction_epoch_{epoch}")
   ```

6. **Finalize Experiment**:
   ```python
   # At experiment end
   final_metrics = {
       'final_train_loss': ...,
       'final_val_loss': ...,
       'final_test_loss': ...,
       'best_epoch': ...
   }
   self.exp_manager.end_experiment(final_metrics)
   ```

### Configuration Structure

#### Primary Config Example

```yaml
# configs/experiments/dlinear_solar.yaml
experiment:
  name: "dlinear_solar_day_ahead"
  type: "pytorch"  # pytorch, lightning, llm, fm
  output_dir: "outputs"
  
model:
  name: "DLinear"
  config_path: "model_configs/general/DLinear.yaml"
  
data:
  name: "solar"
  config_path: "data_configs/fullsolar.yaml"
  
training:
  epochs: 20
  batch_size: 96
  learning_rate: 5e-4
  patience: 3
  loss: "mse"
  
wandb:
  project: "fidel-ts"
  entity: "your-entity"  # Optional
  tags: ["dlinear", "solar", "day-ahead"]
  notes: "Baseline DLinear model on solar dataset"
  
plotting: "configs/plotting/default.yaml"  # Optional nested config
evaluation: "configs/evaluation/default.yaml"  # Optional nested config
```

#### Metadata Auto-Capture

The system will automatically capture:
- **Git Info**: Commit hash, branch name, uncommitted diff (if any)
- **Environment**: Python version, all package versions from `requirements.txt`
- **System Info**: OS, CPU info, GPU info (if available), memory
- **Job Info**: From environment variables (`SLURM_JOB_ID`, `SLURM_JOB_NAME`, etc.) or manual setting
- **Timestamps**: Experiment start time, checkpoint times, end time

### Wandb Integration Details

#### Run Initialization

```python
wandb.init(
    project=config['wandb']['project'],
    entity=config['wandb'].get('entity'),
    name=experiment_id,
    tags=config['wandb'].get('tags', []),
    notes=config['wandb'].get('notes', ''),
    config={
        **flatten_config_hierarchy(config_hierarchy),
        'git_commit': git_info['commit_hash'],
        'git_branch': git_info['branch'],
        'job_id': job_info['job_id'],
        'job_name': job_info['job_name'],
    },
    dir=experiment_dir / "wandb"
)
```

#### Artifact Management

- **Model Checkpoints**: Saved as wandb artifacts with versioning
- **Configurations**: Saved as wandb artifacts for reproducibility
- **Visualizations**: Logged as wandb images with step tracking

#### Metric Logging

- **Real-time Logging**: Metrics logged during training/validation
- **Summary Metrics**: Final metrics logged at experiment end
- **Custom Metrics**: Support for per-subset, per-channel metrics

### Implementation Phases

#### Phase 1: Core Infrastructure (Week 1)
1. Create `ExperimentManager` class with basic functionality
2. Implement metadata capture utilities
3. Create experiment directory structure
4. Implement local metrics logging

#### Phase 2: Configuration System (Week 1-2)
1. Enhance config loader with nested config support
2. Implement config hierarchy tracking
3. Add config validation
4. Test config loading with various structures

#### Phase 3: Wandb Integration (Week 2)
1. Implement wandb initialization
2. Add wandb metric logging
3. Implement checkpoint artifact creation
4. Add visualization logging to wandb
5. Test wandb integration (online and offline modes)

#### Phase 4: Experiment Class Integration (Week 2-3)
1. Integrate `ExperimentManager` into `exp_universal.py`
2. Integrate into `exp_lightning.py`
3. Integrate into `exp_llm.py`
4. Integrate into `exp_fm.py`
5. Test each experiment type

#### Phase 5: Enhanced Features (Week 3)
1. Add visualization saving and tracking
2. Implement experiment summary generation
3. Add experiment comparison utilities
4. Create experiment search/filter utilities
5. Add experiment resume functionality

#### Phase 6: Documentation and Testing (Week 3-4)
1. Write comprehensive documentation
2. Create example configurations
3. Write unit tests for all components
4. Test end-to-end workflows
5. Create migration guide for existing experiments

### Benefits

1. **Reproducibility**: Complete config hierarchy, git info, and environment captured
2. **Traceability**: Every experiment linked to code version and configuration
3. **Organization**: Structured output directories with clear hierarchy
4. **Collaboration**: Wandb integration enables team-wide experiment sharing
5. **Analysis**: Easy comparison of experiments with consistent structure
6. **Debugging**: Comprehensive logs and error tracking
7. **Scalability**: Job ID/name linking supports cluster/scheduler integration

### Migration Strategy

For existing experiments:
1. Create wrapper that captures metadata from existing checkpoint directories
2. Generate experiment folders with captured information
3. Optionally sync to wandb retroactively
4. Maintain backward compatibility with existing checkpoint structure

### Future Enhancements

1. **Experiment Comparison Dashboard**: Compare multiple experiments side-by-side
2. **Automatic Hyperparameter Search**: Integration with wandb sweep
3. **Model Registry**: Track model versions and performance
4. **Experiment Templates**: Pre-configured experiment templates
5. **Automated Reporting**: Generate experiment reports automatically
6. **Integration with MLflow**: Optional MLflow backend in addition to wandb

---

## Configuration System Modernization Plan

This section outlines a comprehensive plan to modernize the configuration system, eliminate fragile string-based experiment identifiers, improve type safety, and ensure complete experiment reproducibility.

### 1. Eliminating 'Settings' Strings

#### Current Problem

Currently, experiments use "settings" strings that are constructed by concatenating config values with underscores:
- Example: `f'{current_time}_{args.model}_{args.data}_{args.output_len}_{args.input_len}'`
- Example: `f'{current_time}_{args.model}_{args.data}_{args.ahead}_ahead'`
- Example: `f'filtered_{current_time}_{args.model}_{args.data}_{args.output_len}_{args.input_len}'`

**Issues with this approach:**
1. **Fragility**: String concatenation is error-prone and breaks easily when config values change
2. **Incompleteness**: Only a subset of config values are included (model, data, output_len, input_len)
3. **Ambiguity**: Different experiments with different configs can produce identical setting strings
4. **Maintenance burden**: Adding new config fields requires updating setting string generation logic
5. **No validation**: No way to ensure setting strings are unique or meaningful
6. **Hard to search**: Difficult to find experiments based on specific config combinations

#### Proposed Solution

**Replace settings with experiment IDs derived from complete config hash:**

1. **Experiment ID Generation**:
   - Generate a deterministic hash (e.g., SHA256) from the complete nested config structure
   - Include all config values (primary + nested configs) in the hash
   - Use first 8-12 characters of hash as experiment ID
   - Format: `{timestamp}_{short_hash}` for human readability and uniqueness

2. **Complete Config as Experiment Identifier**:
   - The entire nested config structure IS the experiment identifier
   - No separate "setting" string needed
   - Experiment folder name: `outputs/{experiment_id}/` where `experiment_id` is derived from config hash
   - Config hash ensures uniqueness and reproducibility

3. **Benefits**:
   - **Robust**: Changes to any config value automatically produce different experiment ID
   - **Complete**: All config values are considered, not just a subset
   - **Deterministic**: Same config always produces same experiment ID
   - **Searchable**: Can search experiments by config values (not string matching)
   - **No maintenance**: No need to update string generation when adding config fields

4. **Implementation**:
   ```python
   def generate_experiment_id(config_hierarchy, timestamp=None):
       """Generate experiment ID from complete config hierarchy."""
       import hashlib
       import json
       
       # Serialize complete config (primary + nested) to JSON
       config_json = json.dumps(
           {
               'primary': config_hierarchy['primary'],
               'nested': config_hierarchy['nested']
           },
           sort_keys=True  # Ensure deterministic ordering
       )
       
       # Generate hash
       config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:12]
       
       # Combine with timestamp for human readability
       if timestamp is None:
           timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
       
       return f"{timestamp}_{config_hash}"
   ```

5. **Migration Strategy**:
   - Phase 1: Generate experiment IDs from config hash, but still support legacy setting strings for backward compatibility
   - Phase 2: Update all experiment classes to use experiment ID instead of setting
   - Phase 3: Remove setting string generation entirely
   - Phase 4: Update checkpoint paths and references to use experiment ID

### 2. Configuration Structure: dotdict vs Dataclasses

#### Current State: dotdict

**dotdict** is a custom dictionary subclass that allows attribute-style access:
```python
class dotdict(dict):
    def __getattr__(self, name):
        return self[name] if name in self.keys() else None
    def __setattr__(self, name, value):
        self[name] = value
```

**Pros:**
- Simple and lightweight
- Easy to convert from YAML (just wrap dict)
- Flexible: can add arbitrary keys dynamically
- Familiar dict-like interface
- Works well with nested structures (with recursive conversion)

**Cons:**
- No type checking or validation
- No IDE autocomplete support
- No default values (returns None for missing keys)
- No schema validation
- Easy to introduce typos (e.g., `config.learing_rate` instead of `config.learning_rate`)
- No documentation of expected fields
- Runtime errors instead of static analysis

#### Alternative: Dataclasses

**Dataclasses** provide structured, typed configuration:

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    name: str
    config_path: str

@dataclass
class DataConfig:
    name: str
    config_path: str

@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 96
    learning_rate: float = 5e-4
    patience: int = 3
    loss: str = "mse"
    # ... other fields with defaults

@dataclass
class ExperimentConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    device: DeviceConfig
    # ... nested configs
```

**Pros:**
- **Type safety**: Static type checking with mypy/pyright
- **IDE support**: Autocomplete, type hints, refactoring
- **Validation**: Can add `__post_init__` for validation
- **Documentation**: Type hints serve as documentation
- **Default values**: Clear defaults for optional fields
- **Catch errors early**: Typos caught at import/validation time
- **Standard library**: Well-supported, familiar to Python developers

**Cons:**
- More verbose: need to define classes for each config section
- Less flexible: can't add arbitrary fields easily
- YAML loading: Need custom loader to convert YAML → dataclass
- Migration effort: Requires refactoring existing code

#### Hybrid Approach: Pydantic Models

**Pydantic** combines benefits of both:

```python
from pydantic import BaseModel, Field
from typing import Optional

class ModelConfig(BaseModel):
    name: str
    config_path: str

class TrainingConfig(BaseModel):
    epochs: int = Field(default=20, ge=1)
    batch_size: int = Field(default=96, ge=1)
    learning_rate: float = Field(default=5e-4, gt=0)
    patience: int = Field(default=3, ge=1)
    loss: str = Field(default="mse", pattern="^(mse|l1)$")

class ExperimentConfig(BaseModel):
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    # ... nested configs
    
    class Config:
        extra = "forbid"  # Reject unknown fields
```

**Pros:**
- **Type safety**: Like dataclasses
- **Validation**: Automatic validation with helpful error messages
- **YAML support**: Built-in YAML loading via `pydantic.yaml`
- **Flexible**: Can allow extra fields or forbid them
- **Rich error messages**: Clear validation errors
- **JSON schema**: Can generate JSON schemas for documentation
- **Default values**: Clear defaults with validation

**Cons:**
- External dependency: Requires `pydantic` package
- Slightly more complex: More features than needed for simple cases

#### Recommendation

**Migrate to Pydantic models** for the following reasons:

1. **Best of both worlds**: Type safety + validation + YAML support
2. **Validation**: Catches config errors early with clear messages
3. **Documentation**: Type hints + JSON schema generation
4. **Future-proof**: Well-maintained, widely used in ML ecosystem
5. **Migration path**: Can gradually migrate, starting with new code

**Migration Strategy:**
1. Phase 1: Define Pydantic models for all config structures
2. Phase 2: Migrate config loader to use Pydantic (completely replace dotdict, no backward compatibility)
3. Phase 3: Update experiment classes to accept Pydantic models directly
4. Phase 4: Remove all dotdict usage from codebase entirely
5. Phase 5: Update all config references to use Pydantic models

**Note**: The experiment class updates to accept configs directly will mesh nicely with the Pydantic migration - both changes can be done together, as experiment classes will naturally work with Pydantic's type-safe config objects. We will completely phase out dotdict usage everywhere with no backward compatibility layer.

### 3. Eliminating config_to_args Methods

#### Current Problem

The `config_to_args()` methods in `runs/*.py` convert config objects to argparse-like objects:
- `runs/pytorch.py`: `config_to_args(config)`
- `runs/lightning.py`: `config_to_args(config)`
- `runs/llm.py`: `config_to_args(config)`
- `runs/fm.py`: `config_to_args(config)`

**Issues:**
1. **Unnecessary conversion**: Config → Args → Config (redundant)
2. **Maintenance burden**: Must update conversion logic when config structure changes
3. **Error-prone**: Easy to miss fields or convert incorrectly
4. **Tight coupling**: Experiment classes depend on argparse structure
5. **Loss of type information**: Args are untyped, lose validation

#### Proposed Solution

**Make experiment classes natively accept config structure:**

1. **Update Experiment Classes**:
   - Modify `Exp_Basic.__init__()` to accept config object directly
   - Remove dependency on argparse-like structure
   - Access config values directly: `config.training.epochs` instead of `args.train_epochs`

2. **Unified Config Structure**:
   - All experiment classes use the same config structure
   - Config includes: model, data, training, device, wandb, job info, etc.
   - No conversion needed: config → experiment class directly

3. **Benefits**:
   - **Simpler**: No intermediate conversion step
   - **Type-safe**: With Pydantic, get validation and type checking
   - **Maintainable**: One config structure, not two (config + args)
   - **Consistent**: Same config format across all experiment types

4. **Implementation Example**:
   ```python
   # Before (current)
   def run(config):
       args = config_to_args(config)  # Convert config → args
       exp = Experiment(args)  # Experiment expects args
   
   # After (proposed)
   def run(config: ExperimentConfig):
       exp = Experiment(config)  # Experiment accepts config directly
   ```

5. **Config Structure Includes Everything**:
   ```python
   class ExperimentConfig(BaseModel):
       # Core experiment info
       experiment_id: Optional[str] = None  # Auto-generated if not provided
       experiment_name: Optional[str] = None
       
       # Model and data
       model: ModelConfig
       data: DataConfig
       
       # Training parameters
       training: TrainingConfig
       
       # Device configuration
       device: DeviceConfig
       
       # WandB configuration
       wandb: WandBConfig
       
       # Job information (from scheduler or manual)
       job: JobConfig  # job_id, job_name
       
       # Reproducibility
       random_seed: int = Field(default=2021, description="Random seed for reproducibility")
       
       # Embedding metadata (see section 7)
       embeddings: Optional[EmbeddingConfig] = None
   ```

### 4. Random Seed in Config

#### Current Problem

Random seed is hardcoded in multiple places:
- `runs/pytorch.py`: `fix_seed = 2021`
- `runs/lightning.py`: `fix_seed = 2021`
- `runs/fm.py`: `fix_seed = 2021`
- `run.py`: `fix_seed = 2021`

**Issues:**
1. **Not configurable**: Can't vary random seed without code changes
2. **Inconsistent**: Different values in different files (potential)
3. **Not tracked**: Seed value not saved in experiment metadata
4. **Hard to reproduce**: If seed changes, can't reproduce exact experiment

#### Proposed Solution

1. **Add to Config**:
   ```yaml
   # configs/experiments/example.yaml
   training:
     random_seed: 2021  # Default, but configurable
     epochs: 20
     # ...
   ```

2. **Default Value**: Default to 2021 for backward compatibility, but allow override

3. **Track in Metadata**: Save actual seed value used (even if default) in experiment metadata

4. **Set All Random Seeds**:
   ```python
   def set_random_seed(seed: int):
       """Set random seed for all random number generators."""
       random.seed(seed)
       torch.manual_seed(seed)
       np.random.seed(seed)
       if torch.cuda.is_available():
           torch.cuda.manual_seed_all(seed)
       # Set Python hash seed for deterministic dict ordering
       os.environ['PYTHONHASHSEED'] = str(seed)
   ```

5. **Metadata Tracking**: Include in experiment metadata:
   ```json
   {
     "random_seed": 2021,
     "random_seed_source": "config.training.random_seed (default)"
   }
   ```

### 5. Complete Config File Copying

#### Current Problem

Experiments reference config files but don't copy them:
- `model_config_path: "model_configs/general/TGTSF.yaml"`
- If `TGTSF.yaml` is modified after experiment, can't reproduce

#### Proposed Solution

**Copy all referenced config files to experiment folder:**

1. **Config Copying Strategy**:
   - Copy primary config file to `outputs/{experiment_id}/configs/primary_config.yaml`
   - Copy all nested configs to `outputs/{experiment_id}/configs/nested/`
   - Copy model config: `outputs/{experiment_id}/configs/model_config.yaml`
   - Copy data config: `outputs/{experiment_id}/configs/data_config.yaml`
   - Copy any other referenced configs

2. **Implementation**:
   ```python
   def copy_config_files(config_hierarchy, experiment_dir):
       """Copy all config files to experiment directory."""
       configs_dir = experiment_dir / "configs"
       configs_dir.mkdir(parents=True, exist_ok=True)
       
       # Copy primary config
       shutil.copy(
           config_hierarchy['config_paths']['primary'],
           configs_dir / "primary_config.yaml"
       )
       
       # Copy nested configs
       for key, path in config_hierarchy['config_paths'].items():
           if key != 'primary':
               shutil.copy(path, configs_dir / f"{key}_config.yaml")
   ```

3. **Benefits**:
   - **Reproducibility**: Know exactly what configs were used
   - **Immutable**: Config files in experiment folder never change
   - **Self-contained**: Experiment folder has everything needed to reproduce

4. **Metadata Tracking**:
   - Save config file paths and hashes in metadata
   - Verify config files haven't changed if re-running

### 6. Removing Args Printing

#### Current Problem

Experiment args are printed at the beginning of training:
```python
print('Args in experiment:')
print(args)
```

**Issues:**
1. **Not pretty-printed**: Hard to read, especially with nested structures
2. **Redundant**: All this info is saved in experiment folder anyway
3. **Clutters output**: Makes logs harder to read
4. **Inconsistent**: Different formats in different experiment types

#### Proposed Solution

1. **Remove Print Statements**: Remove all `print(args)` statements

2. **Log to File Instead**: 
   - Save config to `outputs/{experiment_id}/configs/resolved_config.yaml` (with all defaults filled in)
   - Save pretty-printed config to `outputs/{experiment_id}/metadata/config_summary.txt`

3. **Optional Verbose Mode**: 
   - Add `--verbose` flag to CLI to optionally print config summary
   - Use Rich for pretty printing if verbose mode enabled

4. **Config Summary File**:
   ```python
   def save_config_summary(config, experiment_dir):
       """Save human-readable config summary."""
       summary_path = experiment_dir / "metadata" / "config_summary.txt"
       with open(summary_path, 'w') as f:
           f.write("Experiment Configuration Summary\n")
           f.write("=" * 50 + "\n\n")
           f.write(f"Model: {config.model.name}\n")
           f.write(f"Data: {config.data.name}\n")
           f.write(f"Epochs: {config.training.epochs}\n")
           # ... formatted summary
   ```

### 7. Embedding Metadata Tracking

#### Current Problem

Embedding information is not properly tracked:
- Pre-computed embeddings: No metadata about which model was used, when generated, from what text
- On-the-fly embeddings: Model is specified in config but not tracked in experiment metadata
- Raw text: Not documented that raw text was used (vs embeddings)

#### Proposed Solution

**Add comprehensive embedding metadata to config and experiment tracking:**

1. **Embedding Config Structure**:
   ```yaml
   # In data config or experiment config
   embeddings:
     # Option 1: Pre-computed embeddings
     source: "precomputed"  # precomputed, on_the_fly, raw_text
     embedding_model: "bert-base-uncased"  # Model used to generate
     embedding_dim: 768
     generation_date: "2024-01-15"  # When embeddings were generated
     source_text_files: 
       - "data/Canada_photovoltaics_plants/weather/calgary/report_embedding/formal_report/fast_general_formal_embeddings_2021.pkl"
     embedding_metadata:
       token_extraction: "CLS"  # CLS token, last token, mean pooling, etc.
       max_length: 512
       batch_size: 200
     notes: "Embeddings generated using BERT-base-uncased, extracted from HuggingFace cache"
     
     # Option 2: On-the-fly embeddings
     source: "on_the_fly"
     embedding_model: "bert-base-uncased"
     embedding_dim: 768
     token_extraction: "CLS"
     max_length: 512
     batch_size: 200
     
     # Option 3: Raw text (no embeddings)
     source: "raw_text"
     notes: "Text passed directly to LLM model without embedding"
   ```

2. **Auto-Detection from Config**:
   ```python
   def detect_embedding_config(data_config):
       """Detect embedding configuration from data config."""
       embedding_config = {}
       
       if data_config.hetero_info.input_format == 'embedding':
           # Pre-computed embeddings
           embedding_config['source'] = 'precomputed'
           # Try to infer model from file names or metadata
           # If no metadata, note that it's from HuggingFace
           embedding_config['notes'] = "Pre-computed embeddings, source model unknown (likely BERT-base-uncased from HuggingFace)"
       elif hasattr(data_config, 'postemb') and data_config.postemb:
           # On-the-fly embeddings
           embedding_config['source'] = 'on_the_fly'
           embedding_config['embedding_model'] = data_config.postemb_model
           embedding_config['embedding_dim'] = data_config.postemb_d
           embedding_config['token_extraction'] = 'CLS'  # Default, could be configurable
       else:
           # Raw text
           embedding_config['source'] = 'raw_text'
       
       return embedding_config
   ```

3. **Track in Experiment Metadata**:
   - Save embedding config to `outputs/{experiment_id}/metadata/embeddings.json`
   - Include in WandB config
   - Document in experiment README

4. **Data File Tracking**:
   - Track which data files were used (already partially done in data config)
   - Include file paths and checksums in metadata
   - Document text source files for embeddings

5. **Implementation**:
   ```python
   def capture_embedding_metadata(data_config, experiment_dir):
       """Capture and save embedding metadata."""
       embedding_config = detect_embedding_config(data_config)
       
       # Save to metadata
       metadata_path = experiment_dir / "metadata" / "embeddings.json"
       with open(metadata_path, 'w') as f:
           json.dump(embedding_config, f, indent=2)
       
       # Also include in main experiment metadata
       return embedding_config
   ```

### Implementation Phases

#### Phase 1: Foundation (Week 1)
1. Add random seed to config with default value
2. Implement config file copying to experiment folder
3. Remove args printing, replace with file logging
4. Add embedding metadata detection and tracking

#### Phase 2: Config Structure (Week 2)
1. Define Pydantic models for config structure
2. Create config loader that returns Pydantic models
3. Implement experiment ID generation from config hash
4. Update one experiment class (e.g., `exp_universal.py`) to accept config directly

#### Phase 3: Migration (Week 3-4)
1. Update all experiment classes to accept config directly
2. Remove `config_to_args()` methods
3. Replace setting strings with experiment IDs
4. Update checkpoint paths to use experiment IDs

#### Phase 4: Validation and Testing (Week 4-5)
1. Add config validation with Pydantic
2. Test experiment ID generation and uniqueness
3. Verify config copying works correctly
4. Test embedding metadata tracking
5. Update documentation

### Benefits Summary

1. **Robustness**: No fragile string concatenation for experiment IDs
2. **Type Safety**: Pydantic models provide validation and type checking
3. **Simplicity**: No config_to_args conversion needed
4. **Reproducibility**: Complete config files copied, all defaults tracked
5. **Traceability**: Embedding metadata fully documented
6. **Maintainability**: Single config structure, no duplication
7. **Developer Experience**: IDE autocomplete, type hints, validation errors

---

## Bash Scripts to Config Migration Plan

### Current State Analysis

The `scripts/` directory contains numerous bash scripts that orchestrate experiments:

**Script Categories**:
1. **Individual Experiment Scripts**: Run single model/dataset combinations (e.g., `scripts/NYC_traffic_speed/dlinear.sh`)
2. **Batch Execution Scripts**: Run multiple experiments in sequence (e.g., `run_all_linear.sh`, `run_all_trans.sh`)
3. **Test Scripts**: Evaluate models on test sets (e.g., `test_all_trans_on_samples.sh`)
4. **Variation Scripts**: Specialized versions:
   - `*_on_samples.sh`: Use filtered samples for testing
   - `*_zero_shot.sh`: Zero-shot evaluation
   - `*_m.sh`: Multi-GPU training
   - `*_test.sh`: Testing only

**Current Issues**:
- Decentralized and ad-hoc organization
- Hard to maintain and modify
- Difficult to track which experiments were run
- No clear documentation of experiment parameters
- Repetitive code with slight variations
- No easy way to reproduce exact experiment configurations

### Target Architecture

Migrate all bash scripts to a well-organized config-based system with:

1. **Experiment Suite Configs**: Define collections of related experiments
2. **Template Configs**: Reusable base configurations
3. **CLI-Based Execution**: Use the new CLI system to run experiments
4. **Documentation**: Clear documentation of all experiment configurations

### Proposed Directory Structure

```
configs/
├── experiment_suites/          # Collections of related experiments
│   ├── linear_models.yaml      # All linear model experiments
│   ├── transformer_models.yaml # All transformer model experiments
│   ├── foundation_models.yaml   # All foundation model experiments
│   ├── llm_models.yaml         # All LLM model experiments
│   ├── zero_shot.yaml          # All zero-shot experiments
│   └── filtered_samples.yaml  # All filtered sample experiments
├── templates/                  # Reusable experiment templates
│   ├── pytorch_training.yaml   # Base PyTorch training template
│   ├── lightning_training.yaml # Base Lightning training template
│   ├── llm_testing.yaml        # Base LLM testing template
│   ├── fm_testing.yaml         # Base FM testing template
│   └── evaluation.yaml         # Base evaluation template
├── datasets/                   # Dataset-specific configs
│   ├── NYC_traffic_speed/
│   │   ├── base.yaml           # Base dataset config
│   │   ├── day_ahead.yaml      # Day-ahead forecasting config
│   │   ├── week_ahead.yaml     # Week-ahead forecasting config
│   │   └── filtered_samples.yaml # Filtered samples config
│   ├── Bear_room/
│   │   └── ...
│   └── ...
└── models/                     # Model-specific configs (existing)
    ├── general/
    └── ...
```

### Experiment Suite Config Format

**Example: `configs/experiment_suites/linear_models.yaml`**

```yaml
# Experiment Suite: Linear Models
# Description: Comprehensive evaluation of linear models (DLinear, FITS) across all datasets
# Usage: python -m cli.train suite configs/experiment_suites/linear_models.yaml

suite:
  name: "linear_models_comprehensive"
  description: "Evaluate DLinear and FITS models across all datasets with multiple forecast horizons"
  tags: ["linear", "baseline", "comprehensive"]
  
  # Execution settings
  execution:
    parallel: false  # Run experiments sequentially
    max_parallel: 1  # If parallel=true, max concurrent experiments
    continue_on_error: true  # Continue if one experiment fails
    log_dir: "./logs/suites/linear_models"
    
  # Experiments in this suite
  experiments:
    - name: "dlinear_nyc_traffic_speed"
      description: "DLinear on NYC Traffic Speed dataset"
      enabled: true
      template: "templates/pytorch_training.yaml"
      overrides:
        model:
          name: "DLinear"
          config_path: "model_configs/general/DLinear.yaml"
        data:
          name: "NYC_traffic_speed"
          config_path: "data_configs/NYC_traffic_speed/fullNYCTS_H.yaml"
        training:
          input_len: 360
          output_lens: [24, 168, 336, 720]  # Multiple forecast horizons
          batch_size: 512
        experiment:
          type: "pytorch"
          output_dir: "outputs/suites/linear_models"
          
    - name: "fits_nyc_traffic_speed"
      description: "FITS on NYC Traffic Speed dataset"
      enabled: true
      template: "templates/pytorch_training.yaml"
      overrides:
        model:
          name: "FITS"
          config_path: "model_configs/general/FITS.yaml"
        data:
          name: "NYC_traffic_speed"
          config_path: "data_configs/NYC_traffic_speed/fullNYCTS_H.yaml"
        training:
          input_len: 360
          output_lens: [24, 168, 336, 720]
          batch_size: 512
        experiment:
          type: "pytorch"
          output_dir: "outputs/suites/linear_models"
          
    # ... more experiments for other datasets
    - name: "dlinear_california_iso"
      description: "DLinear on California ISO dataset"
      enabled: true
      template: "templates/pytorch_training.yaml"
      overrides:
        model:
          name: "DLinear"
          config_path: "model_configs/general/DLinear.yaml"
        data:
          name: "California_ISO"
          config_path: "data_configs/California_ISO/fullCAISO_H.yaml"
        training:
          input_len: 360
          output_lens: [24, 168, 336, 720]
          batch_size: 512
        experiment:
          type: "pytorch"
          output_dir: "outputs/suites/linear_models"
```

**Example: `configs/experiment_suites/filtered_samples.yaml`**

```yaml
# Experiment Suite: Filtered Sample Testing
# Description: Test models on filtered sample sets for comparative analysis
# Usage: python -m cli.test suite configs/experiment_suites/filtered_samples.yaml

suite:
  name: "filtered_samples_testing"
  description: "Evaluate models on filtered sample sets across datasets"
  tags: ["testing", "filtered_samples", "comparison"]
  
  execution:
    parallel: false
    continue_on_error: true
    log_dir: "./logs/suites/filtered_samples"
    
  experiments:
    - name: "dlinear_nyc_traffic_speed_day_samples"
      description: "DLinear on NYC Traffic Speed day-ahead filtered samples"
      enabled: true
      template: "templates/evaluation.yaml"
      overrides:
        model:
          name: "DLinear"
        data:
          name: "NYC_traffic_speed"
        evaluation:
          checkpoint_version: "latest"
          input_len: 360
          output_len: 24
          filtered_samples: "sample_indexes/NYC_traffic_speed_sample_day.json"
          batch_size: 1
          device: "2"
        experiment:
          type: "pytorch"
          output_dir: "outputs/suites/filtered_samples"
          
    - name: "dlinear_nyc_traffic_speed_week_samples"
      description: "DLinear on NYC Traffic Speed week-ahead filtered samples"
      enabled: true
      template: "templates/evaluation.yaml"
      overrides:
        model:
          name: "DLinear"
        data:
          name: "NYC_traffic_speed"
        evaluation:
          checkpoint_version: "latest"
          input_len: 360
          output_len: 168
          filtered_samples: "sample_indexes/NYC_traffic_speed_sample_week.json"
          batch_size: 1
          device: "2"
        experiment:
          type: "pytorch"
          output_dir: "outputs/suites/filtered_samples"
```

### Template Config Format

**Example: `configs/templates/pytorch_training.yaml`**

```yaml
# Template: PyTorch Training
# Description: Base template for PyTorch-based model training
# This template can be extended by experiment suite configs

experiment:
  name: "${experiment_name}"  # Placeholder filled by suite
  type: "pytorch"
  output_dir: "outputs"
  
model:
  name: "${model_name}"  # Placeholder
  config_path: "${model_config_path}"  # Placeholder
  
data:
  name: "${data_name}"  # Placeholder
  config_path: "${data_config_path}"  # Placeholder
  
training:
  epochs: 20
  batch_size: 96
  learning_rate: 5e-4
  patience: 3
  loss: "mse"
  lradj: "type3"
  input_len: 360
  output_len: 24  # Can be overridden with output_lens list
  scale: true
  disable_buffer: false
  num_workers: 0
  
device:
  use_gpu: true
  gpu: 0
  use_multi_gpu: false
  devices: "0,1,2,3"
  
wandb:
  project: "fidel-ts"
  tags: []
  notes: ""
```

**Example: `configs/templates/evaluation.yaml`**

```yaml
# Template: Model Evaluation
# Description: Base template for model evaluation/testing
# This template is used for testing trained models

experiment:
  name: "${experiment_name}"
  type: "evaluation"
  output_dir: "outputs"
  
model:
  name: "${model_name}"
  
data:
  name: "${data_name}"
  
evaluation:
  checkpoint_version: "latest"  # latest, oldest, or specific date
  checkpoint_base: "./checkpoints"
  input_len: 360
  output_len: 24
  batch_size: 128
  device: "0"
  task: "TSF"  # TSF, TGTSF, MTSF
  filtered_samples: null  # Path to filtered samples JSON (optional)
  channel_wise: false
```

### CLI Suite Execution

**New CLI Command: `cli/suite.py`**

```python
import typer
from cli.config.loader import load_suite_config
from runs import execute_experiment_suite

app = typer.Typer()

@app.command()
def run(suite_config_path: str):
    """Run an experiment suite from a suite config file."""
    suite_config = load_suite_config(suite_config_path)
    execute_experiment_suite(suite_config)

@app.command()
def list():
    """List all available experiment suites."""
    # List all suite configs in configs/experiment_suites/
    pass

@app.command()
def validate(suite_config_path: str):
    """Validate a suite config file."""
    # Validate suite config structure
    pass

if __name__ == "__main__":
    app()
```

**Usage**:
```bash
# Run an experiment suite
python -m cli.suite run configs/experiment_suites/linear_models.yaml

# List available suites
python -m cli.suite list

# Validate a suite config
python -m cli.suite validate configs/experiment_suites/linear_models.yaml
```

### Suite Execution Engine

**New Module: `runs/suite_executor.py`**

```python
class SuiteExecutor:
    """Execute experiment suites defined in YAML configs."""
    
    def __init__(self, suite_config):
        self.suite_config = suite_config
        self.log_dir = suite_config['suite']['execution']['log_dir']
        
    def execute(self):
        """Execute all experiments in the suite."""
        experiments = [
            exp for exp in self.suite_config['suite']['experiments']
            if exp.get('enabled', True)
        ]
        
        for exp_config in experiments:
            try:
                self._execute_experiment(exp_config)
            except Exception as e:
                if not self.suite_config['suite']['execution']['continue_on_error']:
                    raise
                log_error(exp_config['name'], e)
    
    def _execute_experiment(self, exp_config):
        """Execute a single experiment from the suite."""
        # Load template
        template = load_template(exp_config['template'])
        
        # Merge template with overrides
        final_config = merge_configs(template, exp_config['overrides'])
        
        # Handle multiple output_lens if specified
        if 'output_lens' in exp_config['overrides']['training']:
            for output_len in exp_config['overrides']['training']['output_lens']:
                final_config['training']['output_len'] = output_len
                self._run_single_experiment(final_config, exp_config['name'])
        else:
            self._run_single_experiment(final_config, exp_config['name'])
    
    def _run_single_experiment(self, config, experiment_name):
        """Run a single experiment configuration."""
        # Determine experiment type and call appropriate runner
        exp_type = config['experiment']['type']
        
        if exp_type == 'pytorch':
            from runs.pytorch import run
            run(config)
        elif exp_type == 'lightning':
            from runs.lightning import run
            run(config)
        elif exp_type == 'evaluation':
            from evaluation.standard import evaluate
            evaluate(config)
        # ... other types
```

### Migration Strategy

#### Phase 1: Create Config Structure (Week 1)
1. Create `configs/experiment_suites/` directory
2. Create `configs/templates/` directory
3. Document config format and structure
4. Create example suite and template configs
5. Move `scripts/download_datasets.py` to `utils/download_datasets.py` (or appropriate utility folder) as part of the bash script migration

#### Phase 2: Implement Suite Execution (Week 1-2)
1. Create `cli/suite.py` with suite commands
2. Implement `runs/suite_executor.py`
3. Implement config merging logic
4. Add suite validation
5. Test with simple suite

#### Phase 3: Migrate Linear Model Scripts (Week 2)
1. Convert `run_all_linear.sh` to `linear_models.yaml`
2. Convert individual `dlinear.sh` and `FITS.sh` scripts
3. Test migrated experiments
4. Document migration process

#### Phase 4: Migrate Transformer Scripts (Week 2-3)
1. Convert `run_all_trans.sh` to `transformer_models.yaml`
2. Convert individual transformer scripts
3. Handle Lightning-specific configs
4. Test migrated experiments

#### Phase 5: Migrate Foundation Model Scripts (Week 3)
1. Convert `run_all_FM.sh` to `foundation_models.yaml`
2. Convert individual FM scripts
3. Handle FM-specific testing configs
4. Test migrated experiments

#### Phase 6: Migrate Test Scripts (Week 3-4)
1. Convert `test_all_*.sh` scripts to evaluation suites
2. Convert `*_on_samples.sh` scripts
3. Convert `*_zero_shot.sh` scripts
4. Test all evaluation workflows

#### Phase 7: Migrate LLM Scripts (Week 4)
1. Convert LLM-related scripts
2. Handle LLM-specific configs
3. Test LLM workflows

#### Phase 8: Documentation and Cleanup (Week 4-5)
1. Document all experiment suites
2. Create migration guide
3. Deprecate old bash scripts (keep for reference)
4. Update README with new workflow
5. Create suite execution examples

### Benefits of Config-Based Approach

1. **Centralized Management**: All experiment configurations in one place
2. **Reproducibility**: Exact configurations saved and version-controlled
3. **Documentation**: Configs serve as documentation
4. **Flexibility**: Easy to enable/disable experiments, modify parameters
5. **Scalability**: Easy to add new experiments or datasets
6. **Integration**: Works seamlessly with experiment tracking system
7. **Validation**: Config validation before execution
8. **Template Reuse**: Common patterns captured in templates
9. **Batch Execution**: Run multiple related experiments easily
10. **Maintainability**: Changes in one place affect all related experiments

### Example Migration

**Before (bash script)**:
```bash
# scripts/NYC_traffic_speed/dlinear.sh
for output_len in 24 168 336 720
do
python -u run.py \
    --model 'DLinear' \
    --model_config 'model_configs/general/DLinear.yaml' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 512 | tee -a ./logs/Linear.log
done
```

**After (config-based)**:
```yaml
# configs/experiment_suites/linear_models.yaml
experiments:
  - name: "dlinear_nyc_traffic_speed"
    template: "templates/pytorch_training.yaml"
    overrides:
      model:
        name: "DLinear"
        config_path: "model_configs/general/DLinear.yaml"
      data:
        name: "NYC_traffic_speed"
        config_path: "data_configs/NYC_traffic_speed/fullNYCTS_H.yaml"
      training:
        input_len: 360
        output_lens: [24, 168, 336, 720]
        batch_size: 512
```

**Execution**:
```bash
# Old way
bash scripts/NYC_traffic_speed/dlinear.sh

# New way
python -m cli.suite run configs/experiment_suites/linear_models.yaml --filter dlinear_nyc_traffic_speed
```

### Documentation Structure

Each experiment suite should include:
- **Purpose**: What the suite tests/evaluates
- **Experiments**: List of all experiments in the suite
- **Parameters**: Key parameters varied in the suite
- **Expected Outputs**: What results to expect
- **Dependencies**: Required data, models, or previous experiments
- **Usage Examples**: How to run the suite or individual experiments

### Future Enhancements

1. **Suite Comparison**: Compare results across different suites
2. **Parameter Sweeps**: Automatic hyperparameter search within suites
3. **Conditional Execution**: Run experiments based on previous results
4. **Suite Templates**: Templates for common suite patterns
5. **Visualization Suites**: Automated visualization generation for suites
6. **Report Generation**: Automatic report generation for completed suites

