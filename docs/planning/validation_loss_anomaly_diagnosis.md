# Validation Loss Anomaly Diagnosis

## Problem Statement

When running `python -m cli.suite run configs/experiment_suites/lynx_film_raw/nyc_traffic_speed.yaml` with tensor cache generated using the new polars + direct access optimizations:

- **Train loss**: ~0.2 (reasonable)
- **Validation loss**: ~0.000005 (unreasonable - orders of magnitude lower)

Previous runs without full resolution text embeddings showed val_loss ~0.6 and train_loss ~0.3-0.5.

---

## Root Cause: Data Not Populated for Validation/Test Splits in Direct Access Mode

**Location**: `data_provider/tensor_cache_polars.py`, `_build_direct()` method, lines 686-706

### The Bug

In the `_build_direct` method, when collecting raw data from all splits, only the **first occurrence** of each entity_id is stored:

```python
for flag in flags:  # flags = ['train', 'val', 'test']
    datasets = self.data_provider.get_datasets(flag)
    ...
    for entity_id, dataset in datasets.items():
        ...
        raw = dataset.get_raw_arrays()
        unique_ts = dataset.get_all_unique_timestamps()
        all_timestamps_set.update(unique_ts.tolist())

        # BUG: Only stores first occurrence per entity
        if entity_id not in entity_data:
            entity_data[entity_id] = (raw, dataset)
```

Since `flags = ['train', 'val', 'test']` and train is processed first, **only train data is stored** for each entity.

### Consequence

Later, when populating the shared tables (lines 762-802):

```python
for entity_id, (raw, dataset) in entity_data.items():
    # This only has train data, NOT val/test data!

    for local_idx in range(len(raw.timestamps)):
        ts_int = int(raw.timestamps[local_idx])

        if ts_int in seen_timestamps:
            continue

        unique_idx = timestamp_to_idx[ts_int]
        seen_timestamps.add(ts_int)

        # Only train timestamps get filled
        timeseries_array[unique_idx] = data[local_idx]
        ...
        embeddings_array[unique_idx] = emb
```

**Result**:
- Shared tables are sized correctly for ALL timestamps (train + val + test)
- But only TRAIN timestamps are populated with actual data
- Val/test timestamps remain as **zeros** (from `np.zeros` initialization)

---

## Why Validation Loss is ~0

1. **During training**: The model learns from train data (non-zero values)
2. **During validation**:
   - Val samples have `x_indices` and `y_indices` pointing to shared table positions
   - For val timestamps, `shared['timeseries'][idx]` returns **zeros**
   - `seq_x = zeros`, `seq_y = zeros`
3. **Forward pass**:
   - Model receives input `seq_x = zeros`
   - Model outputs something close to zero (learned from normalized data with zero mean)
   - Ground truth `seq_y = zeros`
   - `Loss = MSE(~0, 0) ≈ 0.000005`

---

## Supporting Evidence

### 1. Timestamp-based Splitting

The `nyc_traffic_speed.yaml` config uses timestamp splitting:
```yaml
# Split points: {Jan. 1, 2021, Jan. 1, 2022}
```

This means train, val, and test have **completely disjoint timestamps** via `timestamp_spliter`:
- Train: timestamps < Jan 1, 2021
- Val: Jan 1, 2021 <= timestamps < Jan 1, 2022
- Test: timestamps >= Jan 1, 2022

### 2. Entity Continuity

The NYC traffic speed dataset has the same sensors (entities) across all time periods. So:
- `entity_data['sensor_1']` is set from train
- When val is processed, `entity_id='sensor_1'` already exists, so val's `raw` data is discarded
- Same for test

### 3. Direct Access Mode Trigger

The bug only manifests when:
1. `preload_hetero=True` is forced (line 553 in `cli/tensor_cache.py`)
2. `_all_support_direct_access()` returns `True` (line 614-617)
3. `_build_direct()` is called instead of `_build_iterative()` (line 617)

---

## Verification Steps

To confirm this diagnosis:

1. **Check shared table population**:
   ```python
   # After tensor cache generation
   timeseries = np.load('tensor_cache/xxx/shared/timeseries.npy')
   timestamps = np.load('tensor_cache/xxx/shared/timestamps.npy')

   # Find val timestamps (2021 data)
   val_mask = (timestamps >= 20210101000000) & (timestamps < 20220101000000)
   val_timeseries = timeseries[val_mask]

   # Check if mostly zeros
   print(f"Val data zeros: {np.sum(val_timeseries == 0)} / {val_timeseries.size}")
   print(f"Val data non-zeros: {np.sum(val_timeseries != 0)}")
   ```

2. **Check index array data retrieval**:
   ```python
   # Load val indices
   y_indices = np.load('tensor_cache/xxx/val/y_indices.npy')

   # Sample some indices
   sample_indices = y_indices[0]  # First val sample's output indices
   sample_data = timeseries[sample_indices]
   print(f"Sample seq_y values: {sample_data}")
   # Expected: all or mostly zeros if bug confirmed
   ```

---

## Remediation Plan

### Fix 1: Store Data Per Split in `_build_direct`

The core fix is to collect data from ALL splits, not just the first occurrence:

```python
# Current (buggy):
if entity_id not in entity_data:
    entity_data[entity_id] = (raw, dataset)

# Fixed approach - collect ALL data across splits:
for flag in flags:
    datasets = self.data_provider.get_datasets(flag)
    for entity_id, dataset in datasets.items():
        raw = dataset.get_raw_arrays()

        # Store timestamps and data for this split
        for local_idx in range(len(raw.timestamps)):
            ts_int = int(raw.timestamps[local_idx])
            if ts_int not in timestamp_data:
                timestamp_data[ts_int] = {
                    'timeseries': raw.data[local_idx],
                    'embedding': raw.embeddings[local_idx] if raw.embeddings is not None else None
                }
```

### Fix 2: Restructure Shared Table Building

Alternative approach - separate entity-level data from timestamp-level data:

1. **First pass**: Collect ALL unique timestamps and their data across ALL splits
2. **Second pass**: Extract entity-level static data (hetero_general, hetero_channel)

### Fix 3: Add Validation Assertions

Add runtime checks in tensor cache generation:

```python
# After building shared tables
n_filled = np.sum(np.any(timeseries_array != 0, axis=1))
n_total = len(sorted_timestamps)
fill_ratio = n_filled / n_total

if fill_ratio < 0.9:
    raise ValueError(
        f"Shared table fill ratio is {fill_ratio:.2%} ({n_filled}/{n_total}). "
        f"This indicates a bug in data collection across splits."
    )
```

---

## Immediate Workaround

Until the bug is fixed, disable direct access optimization:

**Option A**: Regenerate tensor cache without direct access

This would require modifying `_all_support_direct_access()` to return `False`, forcing the iterative fallback path which doesn't have this bug.

**Option B**: Use iterative mode explicitly

Set `preload_hetero=False` before tensor cache generation (but this may break other things).

---

## Files Affected

| File | Line(s) | Issue |
|------|---------|-------|
| `data_provider/tensor_cache_polars.py` | 686-706 | Only first entity occurrence stored |
| `data_provider/tensor_cache_polars.py` | 762-802 | Only train data populates shared tables |
| `cli/tensor_cache.py` | 553 | Forces `preload_hetero=True` |

---

## Related Code Analysis

### Why Iterative Mode Works

The iterative fallback (`_build_iterative`) processes samples via `__getitem__`, which correctly retrieves data for each split:

```python
for flag in flags:
    datasets = self.data_provider.get_datasets(flag)
    for entity_id, dataset in datasets.items():
        for sample_idx in range(len(dataset)):
            sample = dataset[sample_idx]  # Gets correct split data
            _process_sample_for_collection(collector, sample)
```

### Previous Behavior

Without the direct access optimization (before the polars + direct access changes), the iterative path was always used, which correctly populated data for all splits.

---

## Timeline

- **Commit `59a3556`**: Introduced direct access integration
- **Commit `be49a5e`**: Attempted bug fix for hetero time (suggests awareness of issues)
- **Commit `b26ae9e`**: Made verbose hetero preload false (hides potential warnings)

---

## Summary

The validation loss anomaly is caused by a bug in the `_build_direct` method where only TRAIN data is used to populate the shared tables, leaving VALIDATION and TEST timestamps with zero values. This causes validation loss to be essentially zero since MSE(predictions, 0) ≈ 0 when the model outputs small normalized values.

**Priority**: CRITICAL - This bug causes completely invalid validation metrics, making model selection and early stopping unreliable.
