# Evaluation Configuration Simplification Plan

## Problem Statement

Currently, evaluation requires redundant configuration that duplicates information already present in training configs:
- `checkpoint_base` should be inferable from `resume_experiment_id` + `resume_suite_id`
- `model`, `data` are already in the config
- `input_len`, `output_len`, `batch_size` are in `training` config
- `task` can be inferred from model config or experiment type
- `version` should default to "best" (instead of "latest")

## Goals

1. **Single config for training and evaluation**: Use the same experiment config file for both
2. **Required `resume_experiment_id`**: This is the primary way to identify which experiment to evaluate
3. **Auto-inference of all evaluation parameters** from training config and experiment directory structure
4. **Default to best checkpoint**: Use `version: "best"` as default instead of requiring explicit specification

## Implementation Plan

### Phase 1: Update Evaluation Functions to Accept Experiment Config

#### 1.1 Modify `cli/test.py` to accept experiment config directly

**Current behavior:**
- Loads config with `load_config_with_nested`
- Expects `config.evaluation` dictionary
- Passes primary config to evaluation functions

**New behavior:**
- Load experiment config (same as training)
- Require `resume_experiment_id` in config (or as CLI argument)
- Infer evaluation parameters from training config
- Build evaluation config internally

**Changes needed:**
- Update `standard()`, `lightning()`, `llm()` commands to:
  1. Accept config path (same as training)
  2. Require `resume_experiment_id` (or get from CLI arg if not in config)
  3. Call new helper function to build evaluation config from experiment config

#### 1.2 Create helper function to build evaluation config from experiment config

**New function: `evaluation/config_builder.py`** (or add to existing evaluation module)

```python
def build_evaluation_config_from_experiment_config(
    config: ExperimentConfig,
    resume_experiment_id: str,
    resume_suite_id: Optional[str] = None,
    output_dir: str = "./output",
    version: str = "best",  # Default to "best"
    device_override: Optional[str] = None,
    batch_size_override: Optional[int] = None
) -> dotdict:
    """
    Build evaluation configuration from experiment config.
    
    Args:
        config: ExperimentConfig from training config file
        resume_experiment_id: Experiment ID to evaluate (required)
        resume_suite_id: Suite ID if experiment is part of a suite (optional)
        output_dir: Base output directory (default: "./output")
        version: Checkpoint version - "best", "latest", or specific pattern (default: "best")
        device_override: Optional device override (overrides config.device.gpu)
        batch_size_override: Optional batch size override (overrides config.training.batch_size)
        
    Returns:
        dotdict: Evaluation configuration compatible with evaluation functions
    """
```

**This function should:**
1. **Find experiment directory:**
   - If `resume_suite_id`: `output_dir / resume_suite_id / resume_experiment_id`
   - Else: `output_dir / resume_experiment_id`
   - Validate directory exists

2. **Load checkpoint config:**
   - Load from `experiment_dir / "configs" / "experiment_config.yaml"` (preferred, new format)
   - Fallback to `experiment_dir / "args.json"` (legacy format)
   - The checkpoint config contains the actual `input_len` and `output_len` values used during training
   - These are already resolved (if training used `ahead`, it was resolved in `config_to_args()` during training)

3. **Extract from checkpoint/training config:**
   - `model`: `config.model.name`
   - `data`: `config.data.name`
   - `input_len`: From checkpoint config `args.input_len` (or legacy `args.json` format) - these are the actual values used during training
   - `output_len`: From checkpoint config `args.output_len` (or legacy `args.json` format) - these are the actual values used during training
   - `batch_size`: `config.training.batch_size` (unless overridden)
   - `device`: `config.device.gpu` or `device_override` or "0"
   - **Important:** Use checkpoint config values for `input_len`/`output_len` since those match exactly what was used during training

4. **Infer task type:**
   - Load model config YAML from `config.model.config_path`
   - Read `task` field from model config (TSF, TGTSF, MTSF, etc.)
   - Fallback: infer from model name (e.g., "TGTSF" or "lynx" → TGTSF)

5. **Build checkpoint path:**
   - `checkpoint_base`: experiment directory path (not base output dir)
   - For version="best": prefer `best_checkpoint.*`, fallback to `checkpoint.*`
   - For version="latest": use latest checkpoint by modification time
   - For specific version: use pattern matching (backward compatibility)

6. **Handle filtered_samples:**
   - If `config.training.filtered_samples` exists, use it
   - Otherwise, `None`

**Important Note:** We do NOT need to handle `ahead` resolution in evaluation. The checkpoint config already contains the resolved `input_len` and `output_len` values that were actually used during training. Evaluation should simply use those values directly to match training exactly.

### Phase 2: Update Checkpoint Finding Logic

#### 2.1 Modify `evaluation/standard.py` and `evaluation/lightning.py`

**Current `find_checkpoint()` issues:**
- Uses pattern matching: `_{model}_{data}_{output_len}_{input_len}`
- Requires `checkpoint_base` to be parent directory
- Assumes experiments are named with this pattern

**New approach:**
- If `checkpoint_base` points directly to experiment directory (contains `checkpoints/` subdirectory):
  - Use `checkpoint_base / "checkpoints"` directly
  - No pattern matching needed
- If `checkpoint_base` points to parent directory (for backward compatibility):
  - Fall back to pattern matching
  - Emit deprecation warning

**New function:**
```python
def find_checkpoint_from_experiment_dir(
    experiment_dir: Path,
    version: str = "best"
) -> Path:
    """
    Find checkpoint in experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory (e.g., output/experiment_id)
        version: "best", "latest", or specific pattern
        
    Returns:
        Path to checkpoint file (not directory)
    """
    checkpoints_dir = experiment_dir / "checkpoints"
    
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    if version == "best":
        # Prefer best_checkpoint.*
        best_ckpt = list(checkpoints_dir.glob("best_checkpoint*"))
        if best_ckpt:
            return best_ckpt[0]
        # Fallback to checkpoint.pth or last.ckpt
        if (checkpoints_dir / "checkpoint.pth").exists():
            return checkpoints_dir / "checkpoint.pth"
        if (checkpoints_dir / "last.ckpt").exists():
            return checkpoints_dir / "last.ckpt"
        # Find latest by modification time
        all_ckpts = list(checkpoints_dir.glob("checkpoint*")) + list(checkpoints_dir.glob("*.ckpt"))
        if all_ckpts:
            return max(all_ckpts, key=lambda p: p.stat().st_mtime)
        raise FileNotFoundError(f"No checkpoint found in {checkpoints_dir}")
    
    elif version == "latest":
        # Find by modification time
        all_ckpts = list(checkpoints_dir.glob("checkpoint*")) + list(checkpoints_dir.glob("*.ckpt"))
        if not all_ckpts:
            raise FileNotFoundError(f"No checkpoint found in {checkpoints_dir}")
        return max(all_ckpts, key=lambda p: p.stat().st_mtime)
    
    else:
        # Specific version pattern (backward compatibility)
        # ... existing pattern matching logic
```

#### 2.2 Update evaluation functions to use new checkpoint finding

**In `evaluation/standard.py`, `evaluation/lightning.py`:**
- Accept experiment directory path directly (not just checkpoint_base)
- Call `find_checkpoint_from_experiment_dir()` instead of `find_checkpoint()`
- Load config from `experiment_dir / "configs" / "experiment_config.yaml"` (preferred)
- Fallback to `experiment_dir / "args.json"` for backward compatibility

### Phase 3: Update CLI Interface

#### 3.1 Modify `cli/test.py` commands

**New signature:**
```python
@app.command()
def standard(
    config_path: str = typer.Argument(..., help="Path to experiment configuration file (same as training)"),
    resume_experiment_id: Optional[str] = typer.Option(None, "--resume-id", "-r", help="Experiment ID to evaluate (overrides config.resume_experiment_id)"),
    resume_suite_id: Optional[str] = typer.Option(None, "--resume-suite-id", help="Suite ID if experiment is part of a suite (overrides config.resume_suite_id)"),
    output_dir: str = typer.Option("./output", "--output-dir", "-o", help="Base output directory"),
    version: str = typer.Option("best", "--version", "-v", help="Checkpoint version: 'best', 'latest', or specific pattern"),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="GPU device ID (overrides config.device.gpu)"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size (overrides config.training.batch_size)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without running evaluation")
):
```

**Validation:**
- If `resume_experiment_id` not provided in config or CLI, raise error
- If `resume_suite_id` provided (either in config or CLI), validate suite directory exists
- Validate experiment directory exists
- Validate checkpoint directory exists

**Example usage:**
```bash
# Using config with resume_experiment_id
python -m cli.test standard configs/experiments/my_experiment.yaml

# Override resume ID via CLI
python -m cli.test standard configs/experiments/my_experiment.yaml --resume-id 20240101-abc123

# Suite experiment
python -m cli.test standard configs/experiments/my_experiment.yaml --resume-id 20240101-abc123 --resume-suite-id my_suite_20240101_120000

# Override version
python -m cli.test standard configs/experiments/my_experiment.yaml --version latest

# Override device and batch size
python -m cli.test standard configs/experiments/my_experiment.yaml --device 1 --batch-size 64
```

### Phase 4: Handle Task Type Inference

#### 4.1 Task type resolution logic

**Priority order:**
1. **Model config YAML** (`config.model.config_path`): Read `task` field
2. **Experiment type** (`config.experiment.type` if exists): TSF, TGTSF, MTSF
3. **Model name inference**: 
   - Contains "tgtsf" or "lynx" → TGTSF
   - Contains "mtsf" → MTSF
   - Default → TSF

**Implementation:**
```python
def infer_task_type(config: ExperimentConfig) -> str:
    """Infer task type from config."""
    # 1. Try model config
    try:
        model_config_path = Path(config.model.config_path)
        if not model_config_path.is_absolute():
            model_config_path = Path.cwd() / model_config_path
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
            if 'task' in model_config:
                task = model_config['task']
                if task in ['TSF', 'TGTSF', 'MTSF', 'Reasoning']:
                    return task
    except Exception:
        pass
    
    # 2. Try experiment type (if exists as extra field)
    if hasattr(config, 'experiment') and hasattr(config.experiment, 'type'):
        task = config.experiment.type
        if task in ['TSF', 'TGTSF', 'MTSF']:
            return task
    
    # 3. Infer from model name
    model_name_lower = config.model.name.lower()
    if 'tgtsf' in model_name_lower or 'lynx' in model_name_lower:
        return 'TGTSF'
    elif 'mtsf' in model_name_lower:
        return 'MTSF'
    
    # 4. Default
    return 'TSF'
```

### Phase 5: Load Checkpoint Config for Training Parameters

#### 5.1 Use checkpoint config for input_len/output_len

**Important:** Evaluation should use the exact `input_len` and `output_len` values that were used during training, which are stored in the checkpoint config.

**Approach:**
1. **Load checkpoint config** from `experiment_dir / "configs" / "experiment_config.yaml"` (new format) or `experiment_dir / "args.json"` (legacy)
2. **Extract resolved values**: The checkpoint config contains the actual `input_len` and `output_len` used during training
   - If training used `ahead`, these will already be resolved (training resolved them in `config_to_args()`)
   - If training used explicit `input_len`/`output_len`, these will be in the checkpoint config
3. **No `ahead` resolution needed**: Evaluation doesn't need to resolve `ahead` - it just uses the values from the checkpoint

**Implementation:**
```python
def load_checkpoint_config(experiment_dir: Path) -> dict:
    """
    Load checkpoint configuration (preferred: experiment_config.yaml, fallback: args.json).
    
    The checkpoint config contains the actual training parameters used,
    including resolved input_len/output_len if ahead was used during training.
    """
    # Try new format first
    config_path = experiment_dir / "configs" / "experiment_config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    # Fallback to legacy args.json
    args_path = experiment_dir / "args.json"
    if args_path.exists():
        with open(args_path, 'r') as f:
            return json.load(f)
    
    raise FileNotFoundError(f"No checkpoint config found in {experiment_dir}")
```

**Note:** If the checkpoint config doesn't have `input_len`/`output_len` (shouldn't happen if training completed), we can fall back to the training config's values. But ideally, we always use the checkpoint config values since those are what was actually used during training.

### Phase 6: Backward Compatibility

#### 6.1 Maintain backward compatibility for old evaluation configs

**Detection:**
- If config has `evaluation` section as dictionary (not path), use old behavior
- If config has `evaluation` as path string, load nested config
- Emit deprecation warning suggesting migration to `resume_experiment_id`

**Migration path:**
- Old evaluation configs can still work
- But recommend adding `resume_experiment_id` instead

### Phase 7: Testing

#### 7.1 Test cases needed

1. **Standalone experiment evaluation:**
   - Config with `resume_experiment_id`
   - CLI override of `resume_experiment_id`
   - Validation that experiment directory exists

2. **Suite experiment evaluation:**
   - Config with both `resume_experiment_id` and `resume_suite_id`
   - CLI override of suite ID
   - Validation that suite directory exists

3. **Checkpoint version handling:**
   - "best" version (prefer `best_checkpoint.pth`)
   - "latest" version (by modification time)
   - Specific pattern (backward compatibility)

4. **Parameter inference:**
   - `input_len`/`output_len` from checkpoint config (already resolved from training)
   - `batch_size` from `training` config
   - `task` from model config
   - `device` from device config

5. **Backward compatibility:**
   - Old evaluation config format still works
   - Pattern-based checkpoint finding still works
   - Deprecation warnings emitted

## File Changes Summary

### New Files
- `evaluation/config_builder.py` - Build evaluation config from experiment config
- `evaluation/checkpoint_finder.py` - Improved checkpoint finding logic
- `docs/evaluation_config_simplification_plan.md` - This plan

### Modified Files
- `cli/test.py` - Update commands to use new config builder
- `evaluation/standard.py` - Use new checkpoint finder, accept experiment dir
- `evaluation/lightning.py` - Use new checkpoint finder, accept experiment dir
- `evaluation/per_sample.py` - Update to use new config builder (if needed)

### Configuration Changes
- No changes to `ExperimentConfig` model needed (already has `resume_experiment_id`, `resume_suite_id`)
- Evaluation config template becomes optional/deprecated

## Benefits

1. **Single source of truth**: Training config is the only config needed
2. **Less redundancy**: No duplicate model/data/lengths information
3. **Safer defaults**: Defaults to best checkpoint (not latest)
4. **Clearer intent**: `resume_experiment_id` makes it obvious which experiment to evaluate
5. **Better validation**: Can validate experiment directory exists before starting
6. **Easier maintenance**: Changes to training config automatically apply to evaluation

## Migration Guide

**Old way:**
```yaml
# evaluation_config.yaml
evaluation:
  model: TGTSF
  data: Canada_photovoltaics_plants
  version: "latest"
  checkpoint_base: "./output"
  input_len: 360
  output_len: 24
  batch_size: 32
  device: "0"
  task: "TGTSF"
```

**New way:**
```yaml
# training_config.yaml (same file used for training)
model:
  name: TGTSF
  config_path: model_configs/general/TGTSF.yaml
data:
  name: Canada_photovoltaics_plants
  config_path: data_configs/Canada_photovoltaics_plants/fullCPP_hetero_TGTSF.yaml
training:
  input_len: 360
  output_len: 24
  batch_size: 32
device:
  gpu: 0

# Add this for evaluation:
resume_experiment_id: "20240101-abc123def456"  # Required for evaluation
resume_suite_id: "my_suite_20240101_120000"    # Optional, only if part of suite
```

**CLI usage:**
```bash
# Old way (still works, but deprecated)
python -m cli.test standard configs/evaluation_config.yaml

# New way (recommended)
python -m cli.test standard configs/experiments/my_experiment.yaml

# Or override resume ID via CLI
python -m cli.test standard configs/experiments/my_experiment.yaml --resume-id 20240101-abc123
```
