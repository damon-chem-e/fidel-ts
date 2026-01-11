# Implementation Plan: Training Data Purge Truncation

## Goal

Add a configuration option `truncate_train_for_purge` that removes the last `pred_len` samples from training data, eliminating lookahead bias and making validation loss reliable for hyperparameter tuning and early stopping.

## Problem Recap

Currently, the last training sample predicts `[train_split, train_split + pred_len)`, which overlaps with validation data. By truncating training data by `pred_len`, the last training sample predicts `[train_split - pred_len, train_split)`, which ends before validation targets begin.

## Implementation Steps

### 1. Add Configuration Option

**File:** `cli/config/models.py`

**Location:** In `TrainingConfig` class, after `evaluate_test_during_training` (around line 85)

```python
# Purge period truncation to remove lookahead bias
truncate_train_for_purge: bool = Field(
    default=False,
    description=(
        "If True, truncate training data by pred_len to remove lookahead bias. "
        "This ensures training predictions don't overlap with validation data, "
        "making validation loss reliable for hyperparameter tuning. "
        "See docs/train_val_test_purge_period_issue.md for details."
    )
)
```

### 2. Pass Option Through Config Pipeline

**Files:** `runs/pytorch.py`, `runs/lightning.py`, `runs/llm.py`, `runs/fm.py`

**Location:** In `config_to_args()` function, add:

```python
args.truncate_train_for_purge = config.training.truncate_train_for_purge
```

### 3. Add Parameter to Dataset Constructors

**File:** `data_provider/data_loader.py`

**Location:** `Universal_Dataset.__init__()` signature (around line 72-79)

```python
def __init__(self, root_path, flag='train', data_path='ETTh1.csv',
             seq_len=24, pred_len=24, spliter=ratio_spliter, timestamp_col='date',
             target='OT', scale=True, data_buffer=None, hetero_data_getter=None, 
             preload_hetero=False, hetero_stride=1, task=None, custom_input=None, 
             timezone=None, downsample=None, entity_id=None, 
             missing_value_strategy='none', required_indicators=None,
             generate_time_features=False, time_feature_freq='h',
             llm_embedding_provider=None,
             truncate_train_for_purge=False):  # NEW PARAMETER
    # ...
    self.truncate_train_for_purge = truncate_train_for_purge
```

### 4. Implement Truncation Logic

**File:** `data_provider/data_loader.py`

**Location:** In `Universal_Dataset.__read_data__()`, after timestamp extraction (after line 223)

```python
self.timestamp = self.data[self.timestamp_col].values.copy()

# Truncate training data by pred_len to remove lookahead bias (if enabled)
# This ensures training predictions don't overlap with validation data.
# NOTE: train_data (used for scaler fitting) remains full - only self.data is truncated.
if self.set_type == 'train' and self.truncate_train_for_purge:
    if len(self.data) > self.pred_len:
        original_len = len(self.data)
        self.data = self.data.iloc[:-self.pred_len]  # DataFrame truncation
        self.timestamp = self.timestamp[:-self.pred_len]
        logger.info(f"Truncated training data by {self.pred_len} points to remove lookahead bias. "
                   f"Original: {original_len}, New: {len(self.data)}")
    else:
        logger.warning(f"Cannot truncate training data: length ({len(self.data)}) <= pred_len ({self.pred_len})")
```

**Key insight:** `train_data` (used for scaler fitting at line 242) remains full because it's a separate reference after the `.drop()` operation at line 231/233. This means:
- Scaler fits on full training distribution (correct for normalization)
- Training uses truncated data (no lookahead bias)

### 5. Update TimeMMD_Dataset

**File:** `data_provider/time_mmd_dataset.py`

**Location:** `TimeMMD_Dataset.__init__()` and `__read_data__()`

Same changes as `Universal_Dataset`:
1. Add `truncate_train_for_purge=False` parameter to constructor
2. Add truncation logic after timestamp extraction (around line 829)

### 6. Pass Parameter in Data Factory

**File:** `data_provider/data_factory.py`

**Location:** Where datasets are created (lines ~560, ~734, ~773)

```python
dataset = Universal_Dataset(
    ...,
    truncate_train_for_purge=getattr(self.args, 'truncate_train_for_purge', False),
    ...
)
```

### 7. Update Documentation

**File:** `docs/train_val_test_purge_period_issue.md`

Add section explaining this option as the proper fix (vs. optimizing on test loss as a workaround).

## Usage Example

```yaml
# configs/experiment_suites/my_sweep.yaml
training:
  truncate_train_for_purge: true  # Enable purge truncation
  evaluate_test_during_training: false  # No longer needed - val_loss is now reliable!
```

```yaml
# configs/sweep_configs/my_sweep.yaml
metric:
  name: val_loss  # Can now safely optimize on val_loss
  goal: minimize
```

## Behavior Summary

| Setting | Training Data | Scaler Fitting | Val Loss | Use Case |
|---------|--------------|----------------|----------|----------|
| `false` (default) | Full `[0, train_split)` | Full | Biased (lookahead) | Backward compatibility |
| `true` | Truncated `[0, train_split - pred_len)` | Full | Reliable | HPO, early stopping |

## Testing Checklist

- [ ] Dataset length is reduced by `pred_len` when enabled
- [ ] Scaler statistics match full training data
- [ ] Validation loss is comparable to test loss (no artificial improvement)
- [ ] Works with `Universal_Dataset` and `TimeMMD_Dataset`
- [ ] Warning logged when dataset too small to truncate
- [ ] Backward compatible (default `false` preserves existing behavior)

## Files to Modify

1. `cli/config/models.py` - Add config field
2. `runs/pytorch.py` - Pass to args
3. `runs/lightning.py` - Pass to args  
4. `runs/llm.py` - Pass to args
5. `runs/fm.py` - Pass to args
6. `data_provider/data_loader.py` - Constructor + truncation logic
7. `data_provider/time_mmd_dataset.py` - Constructor + truncation logic
8. `data_provider/data_factory.py` - Pass parameter to datasets
9. `docs/train_val_test_purge_period_issue.md` - Document the option
