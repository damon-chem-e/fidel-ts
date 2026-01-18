# Plan: NaN Detection and Early Stopping

## Problem Statement

Currently, when training encounters NaN losses, the training loop continues running through all epochs without triggering early stopping. This wastes computational resources and makes debugging difficult.

## Objectives

1. Detect NaN losses immediately after they occur (both train and validation)
2. Stop training when NaN is detected instead of continuing to max epochs
3. Mark the experiment as failed with appropriate error reporting
4. Preserve debugging information for later analysis

## Proposed Implementation

### 1. Loss Validation in Training Loop

**Location**: `exp/exp_universal.py` (or `exp/exp_lightning.py` for Lightning)

**Implementation Points**:

#### A. After Training Step Loss Computation
```python
# In _train_step method, after loss computation (line ~441)
loss_value = loss.item()

# Check for NaN/Inf in training loss
if math.isnan(loss_value) or math.isinf(loss_value):
    logger.error(f"[Epoch {epoch}] NaN/Inf detected in training loss")
    # Raise exception to stop training
    raise ValueError(f"Training loss became NaN/Inf at epoch {epoch}, batch {batch_idx}")
```

#### B. After Validation Epoch
```python
# In vali method, after computing average validation loss (line ~XXX)
vali_loss = np.average(total_loss)

if math.isnan(vali_loss) or math.isinf(vali_loss):
    logger.error(f"[Epoch {epoch}] NaN/Inf detected in validation loss: {vali_loss}")
    # Raise exception to stop training
    raise ValueError(f"Validation loss became NaN/Inf at epoch {epoch}")
```

### 2. Exception Handling in Main Training Loop

**Location**: `exp/exp_universal.py` - `train()` method

**Implementation**:

```python
def train(self):
    """Main training loop with NaN detection."""
    try:
        # Existing training loop
        for epoch in range(self.args.train_epochs):
            # ... training code ...

            # After validation
            vali_loss = self.vali(...)

            # Early stopping check (existing)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    except ValueError as e:
        # Catch NaN-related errors
        if "NaN" in str(e) or "Inf" in str(e):
            logger.error(f"Training stopped due to NaN/Inf: {e}")
            # Mark experiment as failed
            if self.exp_manager:
                self.exp_manager.mark_failed(reason=str(e))
            # Re-raise to propagate to suite executor
            raise
        else:
            # Other ValueError, re-raise
            raise

    finally:
        # Cleanup code (if needed)
        pass
```

### 3. Experiment Manager Integration

**Location**: `exp/manager.py` - `ExperimentManager` class

**New Method**:

```python
def mark_failed(self, reason: str = "Unknown error"):
    """
    Mark experiment as failed with reason.

    Args:
        reason: Description of why experiment failed
    """
    # Update experiment state
    self.state.status = "failed"
    self.state.error_message = reason
    self.state.failed_at = datetime.now().isoformat()

    # Save state
    self.save_state()

    # Log to experiment directory
    failure_log = self.experiment_dir / "FAILED.txt"
    with open(failure_log, 'w') as f:
        f.write(f"Experiment failed at {self.state.failed_at}\n")
        f.write(f"Reason: {reason}\n")

    logger.error(f"Experiment {self.experiment_id} marked as failed: {reason}")
```

### 4. Suite Executor Handling

**Location**: `runs/suite_executor.py`

**Enhancement**:

```python
def run_experiment(self, experiment_config):
    """Run single experiment with failure handling."""
    try:
        # Existing experiment execution
        result = self._execute_experiment(experiment_config)
        return result

    except ValueError as e:
        if "NaN" in str(e) or "Inf" in str(e):
            # NaN-related failure
            logger.error(f"Experiment {experiment_config['name']} failed due to NaN: {e}")

            # Check continue_on_error setting
            if self.suite_config.get('execution', {}).get('continue_on_error', False):
                logger.warning(f"Continuing to next experiment despite NaN failure")
                return {"status": "failed", "reason": str(e)}
            else:
                # Stop entire suite
                logger.error("Stopping suite execution due to NaN failure")
                raise
        else:
            # Other errors
            raise
```

### 5. Additional Validation Points

#### Option A: Per-Batch NaN Detection (Aggressive)
- Check loss after every batch
- Pros: Catches NaN immediately, saves computation
- Cons: Slight performance overhead from frequent checks

#### Option B: Per-Epoch NaN Detection (Conservative)
- Check only after validation at epoch end
- Pros: Minimal overhead
- Cons: Wastes computation on failed batches within epoch

**Recommendation**: Start with per-epoch (Option B), add per-batch if needed.

### 6. Debugging Information Preservation

When NaN is detected, save diagnostic information:

```python
def save_nan_diagnostics(self, epoch, batch_idx, model_state):
    """Save debugging information when NaN is detected."""
    diag_dir = self.experiment_dir / "nan_diagnostics"
    diag_dir.mkdir(exist_ok=True)

    # Save model state at failure
    torch.save(model_state, diag_dir / f"model_state_nan_epoch{epoch}_batch{batch_idx}.pt")

    # Save optimizer state
    torch.save(self.optimizer.state_dict(), diag_dir / "optimizer_state_nan.pt")

    # Save training metrics history
    with open(diag_dir / "metrics_at_failure.json", 'w') as f:
        json.dump({
            "epoch": epoch,
            "batch": batch_idx,
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history,
        }, f, indent=2)

    logger.info(f"NaN diagnostics saved to {diag_dir}")
```

## Implementation Priority

1. **High Priority** (Implement First):
   - Validation loss NaN detection (after each validation epoch)
   - Exception handling in main training loop
   - Experiment failure marking

2. **Medium Priority** (Implement Second):
   - Training loss NaN detection (per-batch or per-epoch)
   - Suite executor NaN handling
   - Basic diagnostics saving

3. **Low Priority** (Optional Enhancements):
   - Advanced diagnostics (gradient norms, activation statistics)
   - Automatic recovery attempts (reduce LR, reset optimizer)
   - Email/Slack notifications for NaN failures

## Testing Strategy

1. **Unit Tests**:
   - Test NaN detection in loss computation
   - Test exception handling and propagation
   - Test experiment failure marking

2. **Integration Tests**:
   - Inject NaN into mock model output
   - Verify training stops immediately
   - Verify experiment marked as failed
   - Verify suite continues (if continue_on_error=true)

3. **Manual Testing**:
   - Use TimeLLM on problematic datasets
   - Verify NaN detected and training stopped
   - Verify diagnostic information saved

## Backward Compatibility

- All changes are additive (no breaking changes)
- Existing experiments without NaN will behave identically
- NaN detection is always active (no opt-out needed)

## Performance Impact

- Minimal: `math.isnan()` and `math.isinf()` are O(1) operations
- Per-batch checking adds ~0.1ms overhead per batch (negligible)
- Diagnostic saving only occurs on failure (no normal-case impact)

## Configuration Options (Optional)

Add to experiment config for advanced control:

```yaml
training:
  nan_detection:
    enabled: true  # Enable/disable NaN detection
    check_frequency: "per_epoch"  # "per_batch" or "per_epoch"
    save_diagnostics: true  # Save debug info on NaN
    stop_on_nan: true  # Stop training vs. continue with warning
```

## Related Files to Modify

1. `exp/exp_universal.py` - Main training loop
2. `exp/exp_lightning.py` - Lightning training loop
3. `exp/manager.py` - Experiment state management
4. `runs/suite_executor.py` - Suite-level failure handling
5. `utils/early_stopping.py` (optional) - Integrate NaN detection into early stopping

## Success Criteria

✅ Training stops immediately when NaN loss detected
✅ Experiment marked as failed with clear error message
✅ Diagnostic information saved for debugging
✅ Suite execution continues if continue_on_error=true
✅ No performance degradation for normal training
✅ Works for both PyTorch and Lightning experiments

## Future Enhancements

1. **Gradient Anomaly Detection**: Detect exploding/vanishing gradients before NaN
2. **Automatic Recovery**: Reduce LR and restore from last checkpoint
3. **Trend Analysis**: Predict potential NaN from loss trends
4. **Model-Specific Checks**: Add custom NaN checks for specific model architectures
