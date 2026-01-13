# Evaluation NaN Handling Options

This document describes the available evaluation configuration options for handling NaN (Not a Number) values that may occur during model evaluation.

## Overview

NaN values can appear in evaluation metrics for various reasons:
- Missing or corrupted data in the dataset
- Numerical instability in model predictions
- Unhandled edge cases in data preprocessing
- Issues with specific entities or samples in multi-entity datasets

The evaluation system provides configurable options to handle these NaN values at different levels of granularity.

## Configuration

Evaluation options are specified in the experiment suite configuration file under the `evaluation` key within an experiment's `overrides` section:

```yaml
experiments:
  - name: my_experiment
    overrides:
      evaluation:
        nan_aware_aggregation: true
        # Future option:
        # filter_nan_samples: true
```

## Available Options

### 1. `nan_aware_aggregation` (Implemented)

**Level:** Entity-level aggregation

**Description:** When enabled, this option excludes entire entities that produce NaN metrics from the final aggregated results. This is useful for multi-entity datasets where some entities may have problematic data that causes NaN predictions.

**How it works:**
1. During evaluation, metrics are computed for each entity separately
2. After all entities are evaluated, the aggregation step checks each entity's metrics (MSE and MAE, both normalized and denormalized)
3. Entities with any NaN values in their metrics are:
   - Excluded from the final aggregation
   - Logged with a warning message indicating which entity was excluded
4. Final metrics are computed only from entities with valid (non-NaN) metrics
5. If all entities produce NaN, an error is reported and no results are returned

**Use cases:**
- Multi-entity datasets where some entities have missing or corrupted data
- Cases where specific entities cause numerical instability
- When you want to get aggregate metrics despite some entities failing

**Example output:**
```
[Warning] Entity 'entity_5' produced NaN metrics - excluded from aggregation
[Warning] Entity 'entity_7' produced NaN metrics - excluded from aggregation

[Warning] 2 entity/entities with NaN metrics (excluded): entity_5, entity_7

-> Results for 'test' (normalized): MSE = 0.1234567, MAE = 0.2345678
```

**Configuration:**
```yaml
evaluation:
  nan_aware_aggregation: true
```

**Implementation location:**
- `evaluation/standard.py`: `_aggregate_entity_metrics()` function (lines 253-321)
- Called from `_run_evaluation_loop()` when processing multi-entity datasets

---

### 2. `filter_nan_samples` (Planned - Not Yet Implemented)

**Level:** Sample-level filtering

**Description:** When enabled, this option filters out individual samples that produce NaN losses during evaluation, before they contribute to entity-level metrics. **Important:** This filters at the sample level within batches, preserving valid samples even when other samples in the same batch have NaN. This provides finer-grained control than entity-level filtering.

**How it will work:**
1. During the evaluation loop in `evaluate_full_dataset()`, after getting model predictions for each batch
2. Compute per-sample losses (using `reduction='none'` in loss functions) instead of batch-averaged losses
3. Check each sample's losses for NaN values
4. Create a mask identifying valid samples (samples without NaN in their losses)
5. Filter predictions and targets to include only valid samples
6. Compute metrics (MSE/MAE) only on the valid samples
7. Accumulate metrics with the correct sample count (only valid samples)
8. Entity-level metrics are computed only from valid (non-NaN) samples
9. This prevents NaN values from propagating to entity-level metrics while preserving valid samples

**Use cases:**
- Datasets with occasional corrupted or problematic samples
- When you want to evaluate as many samples as possible, excluding only the problematic ones
- Fine-grained control over which samples contribute to metrics
- Cases where NaN occurs at the sample level rather than affecting entire entities
- When batches contain a mix of valid and invalid samples - this preserves the valid ones

**Expected behavior:**
- More samples may be evaluated compared to entity-level filtering
- Entity metrics will be computed from a subset of samples (excluding NaN-producing ones)
- Aggregate metrics will reflect only valid samples

**Configuration (planned):**
```yaml
evaluation:
  filter_nan_samples: true
```

**Implementation location (planned):**
- `evaluation/standard.py`: `evaluate_full_dataset()` function (lines 20-199)
- Will require checking for NaN after loss computation and before adding to running totals

---

## Comparison

| Feature | `nan_aware_aggregation` | `filter_nan_samples` (planned) |
|---------|-------------------------|--------------------------------|
| **Granularity** | Entity-level | Sample-level |
| **When applied** | After entity evaluation, during aggregation | During sample evaluation loop |
| **What gets filtered** | Entire entities with NaN metrics | Individual samples with NaN losses (within batches) |
| **Use case** | Entities with systematic issues | Occasional problematic samples |
| **Impact** | Excludes all samples from problematic entities | Excludes only problematic samples |
| **Status** | ✅ Implemented | 📋 Planned |

## Combining Options

When both options are implemented, they can potentially be used together:
1. `filter_nan_samples` would filter NaN samples during evaluation
2. `nan_aware_aggregation` would provide a safety net for any entities that still produce NaN after sample filtering

However, in most cases, using `filter_nan_samples` alone should be sufficient, as it prevents NaN from reaching entity-level metrics in the first place.

## Related Code

- **Entity evaluation:** `evaluation/standard.py::_evaluate_single_entity()` (lines 205-250)
- **Entity aggregation:** `evaluation/standard.py::_aggregate_entity_metrics()` (lines 253-321)
- **Sample evaluation:** `evaluation/standard.py::evaluate_full_dataset()` (lines 20-199)
- **Config building:** `evaluation/config_builder.py::build_evaluation_config_from_experiment_config()`
- **Suite evaluation:** `cli/test.py` (suite evaluation loop)
