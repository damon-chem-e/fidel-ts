# Implementation Plan: Automatic Test Evaluation at End of Training

## Overview

Add automatic comprehensive test evaluation at the end of training for both PyTorch and Lightning backends. Results will be saved to `metrics/test_results.json` and logged to WandB summary.

**Key Requirements:**
- Implement for BOTH PyTorch and Lightning
- Keep existing per-epoch test options independent
- Save to `metrics/test_results.json` AND log to WandB
- Always run at end of training (not optional initially)
- Compute comprehensive metrics: MSE/MAE (normalized + denormalized)

## Current State

### PyTorch Backend (`exp/exp_universal.py`)
- **No final test evaluation currently exists**
- Training flow (lines 889-956):
  1. `_setup_training()` - Initialize components
  2. Epoch loop (927-948) - Train and evaluate per epoch
  3. `_finalize_training()` (line 951) - Load best checkpoint
  4. `_register_job_end()` (line 954) - Register completion
  5. Return model (line 956)
- Has `test()` method (lines 1056-1134) that runs test evaluation during epochs if `evaluate_test_during_training=True`

### Lightning Backend (`exp/exp_lightning.py`)
- **Has basic final test evaluation** (lines 459-482)
- **Problem:** Only computes and saves loss values, not comprehensive metrics
- Saves to `checkpoints/test_results.json` instead of `metrics/test_results.json`
- Doesn't log comprehensive metrics to WandB summary

### Evaluation Infrastructure
- `evaluation/standard.py`: `evaluate_full_dataset()` returns `(mse_norm, mae_norm, mse_denorm, mae_denorm, num_samples)`
- `evaluation/lightning.py`: `run_test()` returns only normalized MSE/MAE
- Both support per-entity evaluation via dict of loaders

## Implementation Steps

### Step 1: Add Result Formatting Helper Function

**File:** Create or add to `utils/tools.py`

**Function:** `format_test_results(per_entity_metrics, best_epoch, checkpoint_path)`

**Purpose:** Standardize result formatting across PyTorch and Lightning backends

**Returns:**
```json
{
  "overall": {
    "mse_normalized": 0.0234,
    "mae_normalized": 0.1234,
    "mse_denormalized": 45.67,
    "mae_denormalized": 5.43,
    "num_samples": 12000
  },
  "per_entity": {
    "entity_1": {
      "mse_normalized": 0.0234,
      "mae_normalized": 0.1234,
      "mse_denormalized": 45.67,
      "mae_denormalized": 5.43,
      "num_samples": 4000
    }
  },
  "metadata": {
    "timestamp": "2026-01-17T10:30:00",
    "best_epoch": 15,
    "best_checkpoint": "checkpoint.pth"
  }
}
```

**Implementation:**
- Compute weighted average across entities (weighted by num_samples)
- Handle case where denormalized metrics are missing (no scaler)
- Include metadata for traceability

---

### Step 2: PyTorch - Add `_run_final_test_evaluation()` Method

**File:** `exp/exp_universal.py`

**Location:** Add around line 950 (before `_finalize_training`)

**Method Signature:**
```python
def _run_final_test_evaluation(self, test_loader, criterion):
    """
    Run comprehensive test evaluation at end of training.

    Computes MSE/MAE (normalized and denormalized) for all test subsets.
    Saves results to metrics/test_results.json and logs to WandB.

    Args:
        test_loader: Test data loader (dict of loaders by entity)
        criterion: Loss function

    Returns:
        dict: Test metrics with overall, per_entity, and metadata sections
    """
```

**Implementation Logic:**
1. Set model to eval mode
2. Get best epoch from early_stopping or current_epoch
3. Loop through test_loader (dict of entity_id -> loader):
   - Get dataset from loader for scaler access
   - Call `evaluation.standard.evaluate_full_dataset(loader, model, config, device, indexes=None, channel_wise=False, dataset=dataset, filter_nan_samples=False)`
   - Extract: `(mse_norm, mae_norm, mse_denorm, mae_denorm, num_samples)`
   - Build per_entity_metrics dict
4. Call `format_test_results()` helper to create standardized output
5. Save to `self.exp_manager.experiment_dir / "metrics" / "test_results.json"`
6. Log to WandB summary via `self.exp_manager` if enabled
7. Return metrics dict

**Error Handling:**
- Handle None test_loader gracefully (return None)
- Wrap evaluation in try-except, log errors but don't crash
- Handle missing scaler (denormalized metrics will be absent)
- Handle file write failures separately from computation failures

**Console Output:**
- Log "Running final test evaluation..."
- Show progress bar for entity evaluation
- Log overall MSE/MAE results to console

---

### Step 3: PyTorch - Integrate into `train()` Method

**File:** `exp/exp_universal.py`

**Location:** Lines 951-956 (after `_finalize_training`, before `_register_job_end`)

**Current Code:**
```python
# Line 951
self._finalize_training(path, model_optim, train_loss, vali_loss, test_loss)

# Line 954
self._register_job_end(path, train_loss, vali_loss)

# Line 956
return self.model
```

**New Code:**
```python
# Line 951
self._finalize_training(path, model_optim, train_loss, vali_loss, test_loss)

# NEW: Run final comprehensive test evaluation
final_test_metrics = None
if test_loader is not None:
    try:
        self.exp_manager.logger.info("Running final test evaluation on best checkpoint...")
        final_test_metrics = self._run_final_test_evaluation(test_loader, criterion)

        if final_test_metrics:
            self.exp_manager.logger.info(f"Final test MSE (normalized): {final_test_metrics['overall']['mse_normalized']:.7f}")
            self.exp_manager.logger.info(f"Final test MAE (normalized): {final_test_metrics['overall']['mae_normalized']:.7f}")

            if 'mse_denormalized' in final_test_metrics['overall']:
                self.exp_manager.logger.info(f"Final test MSE (denormalized): {final_test_metrics['overall']['mse_denormalized']:.4f}")
                self.exp_manager.logger.info(f"Final test MAE (denormalized): {final_test_metrics['overall']['mae_denormalized']:.4f}")
    except Exception as e:
        self.exp_manager.logger.error(f"Failed to run final test evaluation: {e}")
        import traceback
        traceback.print_exc()

# Line 954
self._register_job_end(path, train_loss, vali_loss)

# Line 956
return self.model
```

---

### Step 4: Lightning - Enhance Existing Test Evaluation

**File:** `exp/exp_lightning.py`

**Location:** Replace lines 459-482

**Current Code Issues:**
- Only computes loss via `trainer.test()` and `trainer.callback_metrics['test_loss']`
- Saves to `checkpoint_path/test_results.json` (should be `metrics/`)
- Doesn't compute comprehensive metrics (MSE/MAE normalized/denormalized)
- Doesn't log to WandB summary properly

**New Implementation:**

```python
# Line 459: Keep logging message
exp_manager.logger.info(f'>>>>>>>final testing on best model: {best_model_path}>>>>>>>>>>>>>>>>>>>>>>>>>>>')

data_module.setup(stage='test')
test_loaders = data_module.test_dataloader()

# Import evaluation function
from evaluation.lightning import run_test

# Collect per-entity metrics
per_entity_metrics = {}

for subset_id, loader in test_loaders.items():
    exp_manager.logger.info(f"Testing {subset_id}...")

    # Get dataset from loader for scaler access
    dataset = loader.dataset if hasattr(loader, 'dataset') else None

    # Call run_test to get MSE/MAE (normalized only from this function)
    mse_norm, mae_norm, num_samples = run_test(
        loader=loader,
        model=model,
        config=config,
        device=device,
        indexes=None,
        channel_wise=False
    )

    # Build entity metrics dict
    entity_metrics = {
        'mse_normalized': mse_norm,
        'mae_normalized': mae_norm,
        'num_samples': num_samples
    }

    # Compute denormalized metrics if scaler available
    if dataset is not None and hasattr(dataset, 'scaler') and dataset.scaler is not None:
        # Need to run inference again to get predictions for denormalization
        # This is a limitation - Lightning's run_test doesn't return predictions
        # We'll need to either enhance run_test or compute inline here

        # OPTION: Compute denormalized metrics inline
        model.eval()
        mse_denorm, mae_denorm = 0.0, 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                # Move batch to device
                batch_x = batch['x'].to(device)
                batch_y = batch['y'].to(device)

                # Get predictions (handle task type)
                if config.model.task == 'TSF':
                    pred = model(x=batch_x)
                elif config.model.task == 'TGTSF':
                    pred = model(
                        x=batch_x,
                        news=batch.get('y_hetero'),
                        channel_description=batch.get('hetero_channel'),
                        historical_events=batch.get('x_hetero')
                    )
                else:
                    pred = model(x=batch_x)

                pred = pred[:, -config.training.output_len:, :]
                true = batch_y[:, -config.training.output_len:, :]

                # Denormalize
                pred_denorm = dataset.scaler.inverse_transform(pred.cpu().numpy().reshape(-1, pred.shape[-1]))
                true_denorm = dataset.scaler.inverse_transform(true.cpu().numpy().reshape(-1, true.shape[-1]))

                pred_denorm = torch.from_numpy(pred_denorm.reshape(pred.shape))
                true_denorm = torch.from_numpy(true_denorm.reshape(true.shape))

                # Compute denormalized losses
                mse_denorm += F.mse_loss(pred_denorm, true_denorm, reduction='sum').item()
                mae_denorm += F.l1_loss(pred_denorm, true_denorm, reduction='sum').item()
                total_samples += pred.numel()

        entity_metrics['mse_denormalized'] = mse_denorm / total_samples
        entity_metrics['mae_denormalized'] = mae_denorm / total_samples

    per_entity_metrics[subset_id] = entity_metrics

    exp_manager.logger.info(f"Test MSE (norm) for {subset_id}: {mse_norm:.7f}")
    exp_manager.logger.info(f"Test MAE (norm) for {subset_id}: {mae_norm:.7f}")

# Format results using helper function
from utils.tools import format_test_results
final_test_metrics = format_test_results(
    per_entity_metrics=per_entity_metrics,
    best_epoch=trainer.current_epoch if hasattr(trainer, 'current_epoch') else args.train_epochs,
    checkpoint_path=best_model_path
)

# Save to metrics/test_results.json (not checkpoints/)
if trainer.is_global_zero:
    import json
    from pathlib import Path

    metrics_dir = Path(exp_manager.experiment_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    results_path = metrics_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(final_test_metrics, f, indent=2)

    exp_manager.logger.info(f"Test results saved to {results_path}")

    # Log to WandB summary
    if exp_manager.wandb_run is not None:
        wandb_metrics = {
            'test/mse_normalized': final_test_metrics['overall']['mse_normalized'],
            'test/mae_normalized': final_test_metrics['overall']['mae_normalized'],
            'test/num_samples': final_test_metrics['overall']['num_samples'],
            'test/num_entities': len(per_entity_metrics)
        }

        if 'mse_denormalized' in final_test_metrics['overall']:
            wandb_metrics['test/mse_denormalized'] = final_test_metrics['overall']['mse_denormalized']
            wandb_metrics['test/mae_denormalized'] = final_test_metrics['overall']['mae_denormalized']

        exp_manager.wandb_run.summary.update(wandb_metrics)
        exp_manager.logger.info("Test metrics logged to WandB summary")

# Keep backward compatibility: also save legacy format to checkpoints/
if trainer.is_global_zero:
    legacy_results = {k: v['mse_normalized'] for k, v in per_entity_metrics.items()}
    with open(os.path.join(checkpoint_path, 'test_results.json'), 'w') as f:
        json.dump(legacy_results, f)
    with open(os.path.join(checkpoint_path, 'test_results_average.json'), 'w') as f:
        json.dump({'average loss of all subsets': final_test_metrics['overall']['mse_normalized']}, f)
```

**Note:** The denormalization loop is verbose but necessary since `run_test()` doesn't return predictions. Alternative is to enhance `evaluation/lightning.py` to support denormalization.

---

### Step 5: WandB Logging Strategy

**For both backends:**

**WandB Summary Fields:**
- `test/mse_normalized`: Overall normalized MSE
- `test/mae_normalized`: Overall normalized MAE
- `test/mse_denormalized`: Overall denormalized MSE (if scaler available)
- `test/mae_denormalized`: Overall denormalized MAE (if scaler available)
- `test/num_samples`: Total number of test samples
- `test/num_entities`: Number of test entities/subsets

**Optional (if few entities):**
- `test/{entity_id}/mse_normalized`: Per-entity MSE
- `test/{entity_id}/mae_normalized`: Per-entity MAE

**Implementation:**
```python
if self.exp_manager.wandb_run is not None:
    wandb_metrics = {
        'test/mse_normalized': final_test_metrics['overall']['mse_normalized'],
        'test/mae_normalized': final_test_metrics['overall']['mae_normalized'],
        'test/num_samples': final_test_metrics['overall']['num_samples'],
        'test/num_entities': len(final_test_metrics['per_entity'])
    }

    if 'mse_denormalized' in final_test_metrics['overall']:
        wandb_metrics['test/mse_denormalized'] = final_test_metrics['overall']['mse_denormalized']
        wandb_metrics['test/mae_denormalized'] = final_test_metrics['overall']['mae_denormalized']

    # Update summary (not run.log, since this is end-of-training)
    self.exp_manager.wandb_run.summary.update(wandb_metrics)
```

---

## Edge Cases and Error Handling

### 1. No Test Data
- Check `if test_loader is None` at start of evaluation
- Log info message and skip evaluation
- Don't save file or log to WandB

### 2. Scaler Not Available
- Check `if dataset.scaler is None` or `not hasattr(dataset, 'scaler')`
- Only compute normalized metrics
- Don't include denormalized keys in output
- Log info message (not error)

### 3. Evaluation Fails
- Wrap entire evaluation in try-except
- Log error with traceback
- Continue with training finalization (don't crash entire run)
- Don't save partial results that might be misleading

### 4. File Write Fails
- Wrap JSON save in separate try-except
- Log error but continue (WandB might still work)
- Error shouldn't prevent training completion

### 5. WandB Not Initialized
- Check `if self.exp_manager.wandb_run is not None` before logging
- Skip WandB logging gracefully
- Local save should still work

### 6. Multiple Entities with Mixed Scalers
- Compute denormalized metrics per-entity where scaler exists
- Overall average only includes entities with denormalized metrics
- Warn if some entities missing denormalization

---

## Testing and Verification

### Manual Testing Checklist

**PyTorch Backend:**
1. ✓ Single entity dataset with scaler → verify full metrics
2. ✓ Multiple entity dataset with scaler → verify per-entity + overall
3. ✓ Dataset without scaler → verify only normalized metrics
4. ✓ No test data (test_loader=None) → verify graceful skip
5. ✓ With `evaluate_test_during_training=True` → verify both per-epoch and final
6. ✓ With `evaluate_test_during_training=False` → verify only final

**Lightning Backend:**
1. ✓ Single entity dataset with scaler → verify full metrics
2. ✓ Multiple entity dataset with scaler → verify per-entity + overall
3. ✓ Dataset without scaler → verify only normalized metrics
4. ✓ No test data → verify graceful skip
5. ✓ With `test_after_epoch=True` → verify both per-epoch and final
6. ✓ With `test_after_epoch=False` → verify only final

**File Verification:**
- ✓ `metrics/test_results.json` exists and has correct structure
- ✓ JSON is valid and properly formatted
- ✓ Metadata includes timestamp, best_epoch, checkpoint_path

**WandB Verification:**
- ✓ WandB summary contains `test/mse_normalized`, `test/mae_normalized`
- ✓ Denormalized metrics present when scaler available
- ✓ Metrics visible in WandB web UI under "Summary" tab

**Console Output Verification:**
- ✓ "Running final test evaluation..." logged
- ✓ Per-entity results logged
- ✓ Overall metrics logged with proper formatting

**Consistency Check:**
- ✓ Compare with standalone `python -m cli.test` command output
- ✓ Metrics should match (allowing for floating point precision)

---

## Critical Files Summary

### Files to Modify:
1. **`utils/tools.py`** (or create new file)
   - Add `format_test_results()` helper function

2. **`exp/exp_universal.py`**
   - Add `_run_final_test_evaluation()` method (~line 950)
   - Modify `train()` method to call it (~line 951-954)

3. **`exp/exp_lightning.py`**
   - Replace test evaluation code (lines 459-482)
   - Add denormalization logic
   - Change save location to `metrics/`
   - Add WandB summary logging

### Files Referenced (no changes):
- `evaluation/standard.py` - Use `evaluate_full_dataset()`
- `evaluation/lightning.py` - Use `run_test()`
- `exp/manager.py` - Use `wandb_run.summary.update()`

---

## Implementation Order

### Phase 1: Helper Function
1. Add `format_test_results()` to `utils/tools.py`
2. Test helper function in isolation

### Phase 2: PyTorch Backend
1. Add `_run_final_test_evaluation()` method
2. Integrate into `train()` method
3. Test with sample experiments
4. Verify file save and WandB logging

### Phase 3: Lightning Backend
1. Enhance existing test evaluation
2. Add denormalization logic
3. Update save location and WandB logging
4. Test with sample experiments

### Phase 4: Integration Testing
1. Run full experiments with both backends
2. Verify metrics match standalone test command
3. Check WandB summary in web UI
4. Verify per-epoch test still works independently

---

## Future Enhancements (Optional)

1. **Config Option:** Add `training.run_final_test_evaluation: bool = True` to allow disabling
2. **Per-Channel Metrics:** Support channel-wise metrics in final test
3. **Enhance `evaluation/lightning.py`:** Add denormalization support to avoid duplication
4. **Per-Sample Test Metrics:** Support `track_per_sample=True` for final test
5. **Visualization:** Auto-generate test prediction plots at end of training
