# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Fidel-TS is a universal cross-modal time series forecasting framework supporting:
- Multiple backends: PyTorch, PyTorch Lightning, LLM-based reasoning
- Multimodal data: Time series + text embeddings (weather forecasts, news, etc.)
- Flexible configuration: YAML-based experiment suites, templates, and single runs
- Performance optimization: Tensor caching for 100-1000x faster data loading

## Remote Development Environment

**IMPORTANT**: The local Windows environment lacks required dependencies. Use the remote RunPod instance for all Python commands and testing.

### Instance Connection
```bash
# SSH connection
ssh 708s8s9tby38jc-64411b16@ssh.runpod.io -i ~/.ssh/id_ed25519

# One-liner: Connect and attach to tmux session
ssh 708s8s9tby38jc-64411b16@ssh.runpod.io -i ~/.ssh/id_ed25519 -t "cd /workspace/fidel-ts && tmux attach -t claude-main || tmux new -s claude-main"
```

### Workspace Details
- **Workspace path**: `/workspace/fidel-ts`
- **Dataset**: canada photovoltaics data
- **Python env**: `.venv` (activate with `source .venv/bin/activate`)

### Tmux Session Management

Always use tmux for persistent sessions on RunPod. Prefix Claude sessions with `claude-` to differentiate from user sessions.

```bash
# Create/attach sessions
tmux new-session -s claude-main        # Primary testing
tmux new-session -s claude-build       # Build/compile
tmux new-session -s claude-test        # Running tests
tmux attach-session -t claude-main     # Attach to existing
tmux list-sessions                     # List all sessions

# Save session logs (run inside tmux: Ctrl+B, then :, then enter command)
capture-pane -pS -32768 > /workspace/claude-sessions/claude-main.log

# Create log directory if needed
mkdir -p /workspace/claude-sessions
```

### Testing Workflow

**Always sync code before testing on RunPod:**

1. **Local (Windows)**: Commit and push
   ```bash
   git add -A && git commit -m "description" && git push
   ```

2. **RunPod**: Pull changes and test
   ```bash
   cd /workspace/fidel-ts
   git fetch && git pull
   source .venv/bin/activate
   python -m cli.train pytorch configs/experiments/example.yaml
   ```

## CLI Commands

The framework uses Typer-based CLI with subcommands. All experiment configurations use YAML files.

### Training

```bash
# PyTorch training (good for debugging)
python -m cli.train pytorch configs/experiments/dlinear_solar.yaml

# PyTorch Lightning (better for multi-GPU, experiment tracking)
python -m cli.train lightning configs/experiments/dlinear_solar.yaml

# LLM-based forecasting
python -m cli.train llm configs/experiments/llm_solar.yaml

# Foundation model testing
python -m cli.train fm configs/experiments/fm_solar.yaml

# Initialize experiment structure without running (useful for inspection)
python -m cli.train pytorch configs/experiments/example.yaml --init-only

# Validate config without training
python -m cli.train pytorch configs/experiments/example.yaml --dry-run
```

### Testing/Evaluation

```bash
# Evaluate standard PyTorch models
python -m cli.test standard configs/experiments/dlinear_solar.yaml

# Evaluate Lightning models
python -m cli.test lightning configs/experiments/dlinear_solar.yaml

# Evaluate LLM predictions
python -m cli.test llm configs/experiments/llm_solar.yaml
```

### Visualization

```bash
# Visualize TSF/TGTSF model predictions
python -m cli.visualize tsf configs/experiments/dlinear_solar.yaml

# Visualize Lightning model predictions
python -m cli.visualize lightning configs/experiments/dlinear_solar.yaml

# Visualize LLM results
python -m cli.visualize llm configs/experiments/llm_solar.yaml
```

### Experiment Suites

Suites run multiple experiments with shared templates and overrides.

```bash
# Run full suite
python -m cli.suite run configs/experiment_suites/linear_models.yaml

# Filter experiments by name
python -m cli.suite run configs/experiment_suites/linear_models.yaml --filter "dlinear"

# Initialize suite structure without running
python -m cli.suite run configs/experiment_suites/linear_models.yaml --init-only

# Force re-run completed experiments
python -m cli.suite run configs/experiment_suites/linear_models.yaml --force-rerun

# Dry run (validate only)
python -m cli.suite run configs/experiment_suites/linear_models.yaml --dry-run
```

### Tensor Cache Generation

Tensor caching pre-computes dataloader operations for 100-1000x faster training.

```bash
# Generate cache (auto-detect GPU/CPU)
python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/suite.yaml

# Generate on CPU-only node (requires pre-computed embeddings)
python -m cli.tensor_cache generate configs/experiment_suites/lynx_film/suite.yaml --cpu-only

# Validate existing cache
python -m cli.tensor_cache validate configs/experiment_suites/lynx_film/suite.yaml

# Show cache info
python -m cli.tensor_cache info ./output/tensor_cache/
```

**Note**: The `--cpu-only` flag is for tensor cache generation only. Embedding computation ALWAYS requires GPU and must be done first.

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_config_loading.py

# Run with verbose output
pytest tests/ -v
```

## Configuration System

The framework uses a **three-level configuration hierarchy**:

1. **Experiment configs** (`configs/experiments/*.yaml`): Single experiment definitions
2. **Suite configs** (`configs/experiment_suites/*.yaml`): Multiple experiments with shared settings
3. **Templates** (`configs/templates/*.yaml`): Base configurations for suites

### Referenced Configs

Experiments reference model and data configs:

- **Model configs** (`model_configs/`): Model-specific hyperparameters
- **Data configs** (`data_configs/`): Dataset paths, splits, heterogeneous data settings

### Config Merging

Suite experiments use **deep recursive merging**:
- `dict + dict`: Merged recursively (right-hand side wins)
- `list + list`: Override list fully replaces template list (no item-wise merge)
- `scalar`: Overridden

Placeholders like `${experiment_name}` are substituted recursively.

### Example Experiment Config

```yaml
model:
  name: DLinear
  config_path: model_configs/general/DLinear.yaml

data:
  name: solar
  config_path: data_configs/fullsolar.yaml

training:
  epochs: 20
  batch_size: 96
  learning_rate: 5e-4
  ahead: day  # Auto-sets input_len and output_len based on sampling_rate

device:
  use_gpu: true
  gpu: 0
  use_multi_gpu: false
  devices: "0,1,2,3"

# Optional: Nested subconfigs
plotting: "configs/plotting/default.yaml"
evaluation: "configs/evaluation/default.yaml"
```

## Architecture Overview

### Directory Structure

```
cli/                   # Typer-based CLI (train, test, visualize, suite, tensor_cache)
├── config/            # Config loading and Pydantic models
├── train.py           # Training commands
├── test.py            # Testing/evaluation commands
├── visualize.py       # Visualization commands
├── suite.py           # Suite execution
└── tensor_cache.py    # Tensor cache generation

data_provider/         # Data loading and preparation
├── data_factory.py    # Dataset factory
├── data_loader.py     # Universal_Dataset class
├── data_helper.py     # Splitting, filtering helpers
├── tensor_cache.py    # Standard tensor cache (pandas-based)
└── tensor_cache_polars.py  # Optimized Polars-based cache

models/                # Model definitions (DLinear, TGTSF, PatchTST, etc.)
├── DLinear.py         # Simple linear baseline
├── TGTSF.py           # Text-guided time series forecasting
├── TimeCMA.py         # Cross-modal attention model
└── ...                # Many other model architectures

exp/                   # Experiment management
├── exp_universal.py   # PyTorch experiment class
├── exp_lightning.py   # Lightning experiment class
├── exp_llm.py         # LLM experiment class
├── manager.py         # Experiment directory/metadata management
└── sweep_manager.py   # W&B hyperparameter sweep management

runs/                  # Training execution modules
├── pytorch.py         # PyTorch training loop
├── lightning.py       # Lightning training loop
├── suite_executor.py  # Suite execution logic
└── ...

embedder/              # Text and LLM embedding infrastructure
├── llm_embedding_provider.py  # LLM embeddings (GPT-2, Qwen)
├── llm_cache.py       # LLM embedding caching
├── metadata.py        # Text embedder for BERT/RoBERTa
└── llm_registry.py    # LLM model registry

layers/                # Model building blocks
utils/                 # Utility functions
configs/               # Experiment configurations
├── experiments/       # Single experiment configs
├── experiment_suites/ # Suite configs
└── templates/         # Template configs for suites

model_configs/         # Model-specific YAML configs
data_configs/          # Dataset-specific YAML configs
```

### Key Concepts

#### Task Types

Models specify task type in their config (`model_configs/`):
- **TSF**: Standard time series forecasting (time series input only)
- **TGTSF**: Text-guided time series forecasting (time series + text embeddings)
- **Reasoning**: LLM-based reasoning task
- **Custom**: Comma-separated list of data components (e.g., `'seq_x,seq_y,hetero_x,hetero_y'`)

Task type determines what data is loaded and passed to the model.

#### Heterogeneous Data

Fidel-TS supports multimodal data via `hetero_info` in data configs:

```yaml
hetero_info:
  sampling_rate: 1day           # Sampling rate of heterogeneous data
  root_path: /path/to/hetero    # Path to heterogeneous data
  formatter: weather_????.json   # Filename pattern
  matching: single              # Time alignment (nearest/forward/backward/single)
  input_format: json            # Format (json/dict/csv/embedding)
  static_path: static_info.json # Static information file
```

**Data flow**:
1. Heterogeneous data loaded by `Heterogeneous_Dataset` or `TimeMMD_HeteroGetter`
2. Time alignment performed based on `matching` strategy
3. Passed to `Universal_Dataset` via callable functions
4. Available in model forward pass as `news`, `historical_events`, `channel_description`

#### Tensor Cache

**Problem**: DataLoader `__getitem__` takes ~18.6ms due to per-sample operations (downtime checks, embedding lookups, time matching).

**Solution**: Pre-compute all operations once and store as memory-mapped numpy arrays.

**Architecture**:
- **Generation phase** (CPU job): `Data_Provider` → `Universal_Dataset` → `TensorCacheGenerator` → `.npy` files
- **Training phase** (GPU job): `TensorCacheDataset` → `__getitem__` returns `arrays[idx]` (O(1) lookup)

**Cache location**: `data/{dataset}/tensor_cache/{hash}/` where hash is computed from:
- `input_len`, `output_len`, `scale`, `data_name`, `hetero_stride`, `split_info`, etc.
- Different configs get different cache directories automatically

**Important**: Embedding computation requires GPU. The `--cpu-only` flag is for tensor cache generation AFTER embeddings are pre-computed.

#### Timestamp Semantics

The framework distinguishes between two timestamp types:
- **t_about**: When event is *about* (e.g., weather forecast for next 24 hours)
- **t_known**: When information became *known* (avoids lookahead bias)

TGTSF models automatically select text source based on `timestamp_semantics` config parameter.

#### Ahead Task Definition

Shorthand for prediction horizons that auto-align with sampling rate:

```yaml
training:
  ahead: day  # For hourly data: input_len=168 (7 days), output_len=24 (1 day)
```

Predefined tasks in `utils/task.py`:
- `day`: 1 day ahead, 7 day lookback
- `week`: 7 day ahead, 30 day lookback
- `month`: 30 day ahead, 60 day lookback

## Key Files

### Configuration and Execution
- `cli/config/loader.py`: Config loading with validation
- `cli/config/models.py`: Pydantic models for config schema
- `runs/suite_executor.py`: Suite execution logic with deep merge

### Data Pipeline
- `data_provider/data_factory.py`: Creates datasets and dataloaders
- `data_provider/data_loader.py`: `Universal_Dataset` class (main dataset class)
- `data_provider/data_helper.py`: Splitting logic, filtering, time alignment
- `data_provider/tensor_cache_polars.py`: Optimized Polars-based tensor cache

### Experiment Management
- `exp/manager.py`: Experiment directory structure, metadata, checkpointing
- `exp/exp_universal.py`: PyTorch training loop
- `exp/exp_lightning.py`: Lightning training loop
- `exp/sweep_manager.py`: W&B hyperparameter sweep coordination

### Models
- `models/DLinear.py`: Simple linear baseline
- `models/TGTSF.py`: Text-guided forecasting with cross-attention
- `models/TimeCMA.py`: Cross-modal attention with LLM embeddings

## Important Notes

### PyTorch vs Lightning

The framework supports both:
- **PyTorch** (`exp/exp_universal.py`): Manual training loop, granular control, good for debugging
- **Lightning** (`exp/exp_lightning.py`): Structured approach, better multi-GPU support, experiment tracking

Use PyTorch for development/debugging, Lightning for production training.

**IMPORTANT**: Do NOT run FITS model with PyTorch Lightning (loss explodes due to complex number computation issues).

### Config State and Naming

- The repository uses **YAML + Pydantic + custom deep-merge** (not Hydra/OmegaConf)
- `model_config` terminology can conflict with Pydantic v2's protected namespace
- Suite templates use `extra="allow"` in ExperimentConfig to support non-modeled sections like `experiment:`

### W&B Integration

The framework has comprehensive W&B integration:
- Experiment tracking and logging
- Hyperparameter sweeps with local registry (avoids race conditions)
- Location-aware resumption (runs resume only where checkpoints exist)
- Graceful shutdown handling for SLURM timeouts

Configure in experiment YAML:
```yaml
wandb:
  project: my_project
  entity: my_team
  enabled: true
  mode: online  # or offline
```

### Common Pitfalls

1. **Tensor cache requires pre-computed embeddings on CPU-only nodes**: Run embedding generation on GPU first
2. **Config hash changes invalidate cache**: Ensure config is stable before generating large caches
3. **Suite deep merge replaces lists entirely**: Override full lists, not individual items
4. **Ahead task requires sampling_rate in data config**: Define `sampling_rate` and `base_T`
5. **FITS model + Lightning = loss explosion**: Use PyTorch for FITS

## Documentation References

For deeper dives, see:
- `docs/tensor_cache.md`: Detailed tensor cache architecture and usage
- `docs/text_information_flow.md`: How text data flows through the system
- `docs/embedder.md`: Text vs LLM embeddings, input sources
- `docs/config_state.md`: Complete config system mapping
- `docs/SWEEP_COMPREHENSIVE_GUIDE.md`: W&B hyperparameter sweeps
- `docs/timestamp_semantics.md`: Understanding t_about vs t_known

## Development Workflow

1. **Local (Windows)**: Edit code, commit changes
2. **Remote (RunPod)**: Pull changes, test with real data
3. **Iterate**: Fix issues locally, push, test remotely
4. **Use tmux**: Keep persistent sessions, save logs regularly
5. **Prefix sessions**: Use `claude-*` naming for Claude sessions
