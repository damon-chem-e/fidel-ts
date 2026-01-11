# Plan: Robust Normalized and Denormalized Evaluation Metrics

**Status**: Planning  
**Created**: 2026-01-11  
**Related Issue**: Loss discrepancy between training and evaluation due to missing `scale` parameter

## Executive Summary

This document outlines a plan to make `cli.test` evaluation metrics robust for both normalized and denormalized MSE/MAE, regardless of whether data comes from a ConcatDataset (train/val splits) or per-entity dict of loaders (test split).

## Current State Analysis

### Problem Summary

| Scenario | Normalized Metrics | Denormalized Metrics |
|----------|-------------------|---------------------|
| Single entity | ✅ Correct | ✅ Correct (scaler available) |
| Dict of loaders (test) | ✅ Correct | ⚠️ Per-entity scalers exist but not aggregated properly |
| ConcatDataset (train/val) | ⚠️ Consistent with training but semantically mixed | ❌ Uses wrong scaler for non-first entities |

### Root Causes

1. **ConcatDataset breaks scaler association**: When datasets are concatenated, we lose track of which sample came from which dataset/scaler. The current code extracts the first underlying dataset's scaler, which is incorrect for samples from other entities.

2. **No per-sample scaler tracking**: Batches don't carry scaler information, so there's no way to know which scaler to apply for denormalization.

3. **Inconsistent evaluation paths**: Train/val use ConcatDataset (for efficient batching during training), while test uses dict of loaders (for per-entity evaluation).

### Why This Matters

Each entity in a multi-entity dataset has its own `StandardScaler` fitted on its training data:

```
Entity A (high-traffic city): scaler.mean_ = 10000, scaler.scale_ = 2000
Entity B (low-traffic suburb): scaler.mean_ = 500,   scaler.scale_ = 100
```

A normalized prediction error of 0.1 means:
- **Entity A**: 0.1 × 2000 = **200 actual units** of error
- **Entity B**: 0.1 × 100 = **10 actual units** of error

Using Entity A's scaler to denormalize Entity B's predictions gives completely wrong results.

### Impact on Normalized Metrics

Even normalized metrics have semantic issues when aggregating across entities:
- All samples are transformed to mean=0, std=1 using their respective scalers
- Averaging normalized MSE across entities treats all errors equally
- But a normalized error of 0.1 represents vastly different real-world magnitudes depending on the entity's original scale
- This is **consistent with training** (model optimized this metric) but not directly interpretable in real-world units

---

## Proposed Architecture

### Option A: Per-Entity Evaluation for All Splits (Recommended)

**Core Idea**: Never use ConcatDataset for evaluation. Always evaluate per-entity, then aggregate.

#### Changes Required

**1. Modify `_run_evaluation_loop()` in `standard.py`**

```
Current flow:
  train/val: ConcatDataset loader → evaluate_full_dataset() → aggregate
  test: dict of loaders → per-entity evaluate_full_dataset() → aggregate

Proposed flow:
  ALL splits: dict of loaders → per-entity evaluate_full_dataset() → aggregate
```

**2. Add new methods to `Data_Provider`**

```python
def get_train_eval(self, return_type='loader'):
    """Get training data as dict of loaders (for evaluation, not training)."""
    self.train_dataset = self.get_datasets('train')
    if return_type == 'set':
        return self.train_dataset
    elif return_type == 'loader':
        return self.get_dataloader(self.train_dataset, shuffle=False, drop_last=False, concat=False)
    # ...

def get_val_eval(self, return_type='loader'):
    """Get validation data as dict of loaders (for evaluation)."""
    # Similar implementation
```

**3. Create `MetricsAggregator` helper class**

```python
class MetricsAggregator:
    """
    Aggregates per-entity metrics into summary statistics.
    
    Supports multiple aggregation strategies:
    - sample_weighted: Weight by number of samples (default)
    - entity_weighted: Equal weight per entity
    """
    
    def __init__(self):
        self.entity_results = {}
    
    def add_entity_result(self, entity_id: str, metrics: dict):
        """
        Add results for a single entity.
        
        Args:
            entity_id: Unique entity identifier
            metrics: Dict with mse_norm, mae_norm, mse_denorm, mae_denorm, n_samples
        """
        self.entity_results[entity_id] = metrics
    
    def get_aggregate(self, method: str = 'sample_weighted') -> dict:
        """
        Compute aggregated metrics across all entities.
        
        Args:
            method: 'sample_weighted' or 'entity_weighted'
        
        Returns:
            Dict with aggregated metrics
        """
        if method == 'sample_weighted':
            return self._sample_weighted_aggregate()
        elif method == 'entity_weighted':
            return self._entity_weighted_aggregate()
    
    def get_per_entity_report(self) -> dict:
        """Return detailed per-entity breakdown."""
        return self.entity_results
```

**4. Update `evaluate()` function**

```python
def evaluate(config):
    # ...
    
    # For evaluation, always use per-entity loaders (not ConcatDataset)
    train_loaders = data_provider.get_train_eval("loader")  # Returns dict
    val_loaders = data_provider.get_val_eval("loader")      # Returns dict
    test_loaders = data_provider.get_test("loader")         # Already returns dict
    
    # Get datasets for scaler access
    train_datasets = data_provider.get_train_eval("set")
    val_datasets = data_provider.get_val_eval("set")
    test_datasets = data_provider.get_test("set")
    
    # Evaluate each split with proper per-entity handling
    # ...
```

#### Aggregation Strategy

```
Per-entity evaluation:
  Entity A: 100 samples, MSE_norm=0.02, MSE_denorm=50000
  Entity B: 200 samples, MSE_norm=0.01, MSE_denorm=100

Sample-weighted aggregate:
  MSE_norm = (100*0.02 + 200*0.01) / 300 = 0.0133
  MSE_denorm = (100*50000 + 200*100) / 300 = 16733.33

Entity-weighted aggregate (optional):
  MSE_norm = (0.02 + 0.01) / 2 = 0.015
  MSE_denorm = (50000 + 100) / 2 = 25050
```

---

### Option B: Per-Sample Scaler Tracking (Alternative)

**Core Idea**: Include scaler parameters in each sample's batch data.

#### Changes Required

**1. Modify dataset `__getitem__()` methods**

```python
def __getitem__(self, index):
    # ... existing code ...
    
    # Add scaler parameters to return tuple
    scaler_mean = self.scaler.mean_ if self.scale else None
    scaler_scale = self.scaler.scale_ if self.scale else None
    
    return (sample_id, seq_x, seq_y, ..., scaler_mean, scaler_scale)
```

**2. Update evaluation loop**

```python
for batch in loader:
    # Unpack including scaler params
    *standard_batch, scaler_means, scaler_scales = batch
    
    # Denormalize using per-sample scalers
    pred_denorm = prediction * scaler_scales + scaler_means
    target_denorm = batch_y * scaler_scales + scaler_means
```

#### Pros/Cons Comparison

| Aspect | Option A (Per-Entity) | Option B (Per-Sample Scaler) |
|--------|----------------------|------------------------------|
| Accuracy | ✅ Perfect | ✅ Perfect |
| Complexity | Medium (new loader method) | High (dataset interface change) |
| Memory overhead | Same as current | Higher (scaler params in every sample) |
| Breaking changes | Low (evaluation only) | Medium (dataset interface) |
| Training impact | None | None |
| Batch efficiency | Slightly lower (per-entity iteration) | Same as current |

---

## Recommended Implementation Plan

### Phase 1: Data Provider Enhancement
**File**: `data_provider/data_factory.py`

1. Add `get_train_eval()` method
   - Returns dict of loaders (not concatenated)
   - Used only for evaluation, not training
   
2. Add `get_val_eval()` method
   - Same pattern as above

3. Keep existing `get_train()` and `get_val()` unchanged
   - Training continues to use ConcatDataset for efficiency

### Phase 2: Metrics Aggregation
**File**: `evaluation/standard.py` (new class)

1. Create `MetricsAggregator` class
   - Accumulate per-entity metrics
   - Support sample-weighted and entity-weighted aggregation
   - Track per-entity results for detailed reporting

### Phase 3: Evaluation Loop Refactor
**File**: `evaluation/standard.py`

1. Refactor `_run_evaluation_loop()`
   - Always iterate over entities (never ConcatDataset)
   - Use per-entity scaler for denormalization
   - Accumulate via MetricsAggregator

2. Update `evaluate()` function
   - Request per-entity loaders for all splits
   - Pass entity-specific datasets to `evaluate_full_dataset()`

### Phase 4: Output Enhancement
**File**: `evaluation/standard.py`

1. Enhance `_print_summary()`
   - Show aggregate metrics (current behavior)
   - Add optional per-entity breakdown
   - Remove warning about ConcatDataset denormalization

2. Add metrics to saved results
   - Per-entity results JSON
   - Aggregate results with method noted

### Phase 5: CLI Integration
**File**: `cli/test.py`

1. Add `--per-entity-report` flag
   - Shows detailed per-entity breakdown

2. Add `--aggregation-method` flag
   - Options: `sample_weighted` (default), `entity_weighted`

---

## Expected Output After Implementation

```
==================================================
               Evaluation Summary
==================================================

TRAIN Set (3 entities, 336 samples):
  Normalized   - MSE: 0.0142, MAE: 0.0891
  Denormalized - MSE: 12453.21, MAE: 89.32

  Per-entity breakdown:
    entity_A (100 samples): MSE_norm=0.018, MSE_denorm=45000.00
    entity_B (150 samples): MSE_norm=0.012, MSE_denorm=200.50
    entity_C (86 samples):  MSE_norm=0.014, MSE_denorm=1500.00

VAL Set (3 entities, 48 samples):
  Normalized   - MSE: 0.0189, MAE: 0.1023
  Denormalized - MSE: 15234.56, MAE: 102.11

TEST Set (3 entities, 102 samples):
  Normalized   - MSE: 0.0234, MAE: 0.1156
  Denormalized - MSE: 18234.12, MAE: 115.23
==================================================
```

---

## Files to Modify

| File | Changes | Priority |
|------|---------|----------|
| `data_provider/data_factory.py` | Add `get_train_eval()`, `get_val_eval()` methods | High |
| `evaluation/standard.py` | Add `MetricsAggregator`, refactor `_run_evaluation_loop()` | High |
| `cli/test.py` | Add new CLI flags | Medium |
| `evaluation/config_builder.py` | Add aggregation config support | Low |

---

## Migration Path

1. **Backward compatible**: Existing behavior preserved by default
2. **Opt-in detailed metrics**: Use flags for per-entity reporting
3. **No training changes**: Only evaluation path affected
4. **Gradual rollout**: Can implement phases independently

---

## Current Workaround

Until this plan is implemented, users should be aware:

1. **Denormalized metrics for train/val (ConcatDataset) are NOT valid** for multi-entity datasets
2. **Normalized metrics are consistent with training** but represent different real-world magnitudes across entities
3. **Test split metrics are more reliable** since evaluation is already per-entity

A warning is displayed in evaluation output when denormalized metrics are shown for concatenated datasets.

---

## References

- Related fix: `scale` parameter loading in `_load_checkpoint_config()` (2026-01-11)
- ConcatDataset scaler extraction: `evaluate_full_dataset()` lines 59-63
