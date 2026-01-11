# Train/Val/Test Split Purge Period Issue: Lookahead Bias in Validation

## Executive Summary

Both **fidel-ts** and **MM-TSFlib** repositories exhibit a critical issue in their train/validation/test split logic: **the absence of a purge period between splits equal to at least the prediction length (`pred_len`)**. This causes **lookahead bias** in validation loss metrics, making them appear artificially good, especially when the validation period is small relative to the prediction length (common in Time-MMD small datasets).

**Key Finding**: Validation loss should **NOT** be trusted as a reliable metric. Instead, **test loss** should be the primary metric for model evaluation, as it is not affected by this lookahead bias.

## The Problem: Why a Purge Period is Required

In time series forecasting, when training on data `[1, ..., T]` with prediction length `p`:

1. **Training data** uses inputs up to time `t` (index `train_split`)
2. **Model predictions** from the last training samples extend to time `t + p` (i.e., `train_split + pred_len`)
3. **Validation data** should start **no earlier than** `t + p + 1` to avoid temporal leakage

If validation starts earlier (e.g., at `train_split - seq_len` or even at `train_split`), then:
- Training samples with indices near the end of training predict values that overlap with the validation period
- The model has effectively "seen" validation data during training (through the prediction targets)
- Validation loss is artificially low because the model is being evaluated on data it was indirectly trained on

### Mathematical Formulation

Given:
- Training data: `[0, train_split)` (indices 0 to `train_split - 1`)
- Prediction length: `p` (`pred_len`)
- Sequence length: `s` (`seq_len`)

The last training sample uses:
- Input: indices `[train_split - s, train_split)`
- Target: indices `[train_split, train_split + p)`

**Validation must start at index `train_split + p` or later** to avoid overlap with training predictions.

## Issue in fidel-ts (fidel-ts-worktree-lynx)

### Current Implementation

The `ratio_spliter` function in `data_provider/data_helper.py` implements splits as follows:

```python
def ratio_spliter(split=(7,1,2), seq_len=0, df=None):
    # ... ratio calculations ...
    train_split = int(raw_data_len * train_split)  # e.g., 0.7 * total
    val_split = int(raw_data_len * val_split)      # e.g., 0.8 * total
    
    train_data = df[0:train_split]                    # [0, train_split)
    val_data = df[train_split-seq_len:val_split]     # [train_split-seq_len, val_split)
    test_data = df[val_split-seq_len:]                # [val_split-seq_len, end)
```

**Problem**: Validation starts at `train_split - seq_len`, which is **before** the end of training data at `train_split - 1`. The overlap of `seq_len` is intentional for input sequence construction, but **there is no purge period for predictions**.

In `Universal_Dataset.__getitem__` (lines 290-293 of `data_provider/data_loader.py`):

```python
s_begin = index
s_end = s_begin + self.seq_len
r_begin = s_end                                    # Prediction starts immediately after input
r_end = r_begin + self.pred_len
```

The last training sample (with `index = train_split - seq_len - 1`, such that `s_end ≈ train_split`) predicts:
- Target indices: `[train_split, train_split + pred_len)`

But validation data starts at `train_split - seq_len`, meaning indices `[train_split, train_split + pred_len)` are **within the validation period**, creating lookahead bias.

### Example

For a dataset with:
- `total_length = 1000`
- `split = (7, 1, 2)` → 70% train, 10% val, 20% test
- `seq_len = 96`
- `pred_len = 24`

Current split boundaries:
- Training: `[0, 700)` (indices 0-699)
- Validation: `[700 - 96, 800)` = `[604, 800)` (indices 604-799)
- Test: `[800 - 96, 1000)` = `[704, 1000)` (indices 704-999)

The last training sample with input `[604, 700)` predicts `[700, 724)`, which **overlaps with validation data** (validation includes indices 700-799). This creates lookahead bias.

**Required purge**: Validation should start at `700 + 24 = 724` (or later), not at 604.

## Issue in MM-TSFlib

### Current Implementation

The `Dataset_Custom` class in `data_provider/data_loader.py` implements splits as follows:

```python
num_train = int(len(df_raw) * 0.7)
num_test = int(len(df_raw) * 0.2)
num_vali = len(df_raw) - num_train - num_test

border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
border2s = [num_train, num_train + num_vali, len(df_raw)]
```

Where:
- Training: `[border1s[0], border2s[0])` = `[0, num_train)`
- Validation: `[border1s[1], border2s[1])` = `[num_train - seq_len, num_train + num_vali)`
- Test: `[border1s[2], border2s[2])` = `[len(df_raw) - num_test - seq_len, len(df_raw))`

**Problem**: Similar to fidel-ts, validation starts at `num_train - seq_len`, with no purge period for predictions.

In `Dataset_Custom.__getitem__` (lines 151-157):

```python
s_begin = index % self.tot_len
s_end = s_begin + self.seq_len
r_begin = s_end - self.label_len
r_end = r_begin + self.label_len + self.pred_len
```

The last training sample (with `s_end = num_train`) predicts:
- Target indices: `[num_train - label_len, num_train + pred_len)`

But validation data starts at `num_train - seq_len`, and includes indices `[num_train, num_train + pred_len)`, creating lookahead bias.

### Example

For the same dataset configuration:
- Training: `[0, 700)`
- Validation: `[700 - 96, 800)` = `[604, 800)`
- Test: `[1000 - 200 - 96, 1000)` = `[704, 1000)`

The last training sample predicts `[num_train, num_train + pred_len)` = `[700, 724)`, which overlaps with validation data, creating lookahead bias.

## Impact on Validation Loss: Artificially Good Metrics

### Why Validation Loss Appears Too Good

The lookahead bias causes validation loss to be **artificially low** because:

1. **Temporal Leakage**: Training samples near the end of the training period predict values that fall within the validation period
2. **Indirect Training on Validation Data**: The model has effectively been trained to predict validation data (through the overlap in prediction targets)
3. **Optimistic Evaluation**: Validation loss reflects the model's performance on data it was indirectly trained on, not truly unseen future data

### Critical Impact on Small Datasets (Time-MMD)

This issue is **especially problematic** for small datasets like those in Time-MMD, where:

- **Validation periods are short** (often only 10% of data, or ~100-200 timesteps)
- **Prediction lengths are significant** relative to validation size (e.g., `pred_len = 24` or `48` for validation periods of ~100 timesteps)
- **Overlap proportion is large**: When `pred_len` is 20-50% of the validation period size, a large fraction of validation data overlaps with training predictions
- **Validation loss becomes meaningless**: The metric is so polluted by lookahead bias that it cannot be trusted

**Example**: For a Time-MMD dataset with:
- Total length: 500 timesteps
- Validation: 50 timesteps (10%)
- `pred_len = 24`

If validation starts at `train_split` (350) instead of `train_split + pred_len` (374), then **48% of the validation period** (indices 350-373) overlaps with training predictions, making validation loss unreliable.

## Test Loss: The Reliable Metric

### Why Test Loss is Not Affected

Test loss is **NOT polluted** by this lookahead bias because:

1. **Distance from Training**: Test data starts well after training data ends
2. **No Direct Overlap**: Training predictions (extending to `train_split + pred_len`) do not reach into the test period
3. **Only Validation is Polluted**: The lookahead bias affects validation only; it does not stretch all the way to the test period

### Validation of Test Loss Reliability

Even though validation data overlaps with training predictions:
- Training predictions extend to: `train_split + pred_len`
- Validation ends at: `val_split`
- Test starts at: `val_split - seq_len` (or `val_split` in some configurations)

The gap between `train_split + pred_len` and the test period is typically **much larger than `pred_len`**, ensuring no overlap with training predictions.

**Example** (fidel-ts with `split = (7,1,2)`, `total = 1000`, `pred_len = 24`):
- Training predictions extend to: `700 + 24 = 724`
- Validation ends at: `800`
- Test starts at: `800 - 96 = 704` (with overlap) or could be `800` (without overlap)

Even with the overlap in test start, training predictions (up to index 724) do not reach the test period (starting at 704, but actual test evaluation uses indices well beyond 800). The **effective test period** (excluding the initial overlap) starts after index 724, ensuring no contamination.

## Our Approach: Following Original Paper Conventions

### Decision: Maintain Current Split Logic

We will **NOT modify the split logic** in either repository to maintain compatibility with:

1. **Original paper conventions**: Both repositories follow established conventions in the time series forecasting literature
2. **Reproducibility**: Changing split logic would make results incomparable with prior work
3. **Backward compatibility**: Existing experiments and configurations rely on current split behavior

### Solution: Focus on Test Loss, Not Validation Loss

Since we cannot change the split logic, we must **change our evaluation strategy**:

1. **Primary Metric: Test Loss**
   - Use test loss as the primary metric for model evaluation
   - Test loss is reliable and not affected by lookahead bias
   - Report test loss in all experiments and comparisons

2. **Secondary Metric: Validation Loss (with caveats)**
   - Acknowledge that validation loss is polluted by lookahead bias
   - Do not use validation loss for model selection or hyperparameter tuning in critical decisions
   - Only use validation loss for early stopping or rough guidance, with full awareness of its limitations
   - When validation loss appears "too good to be true," trust test loss instead

3. **Documentation and Reporting**
   - Always report both validation and test loss
   - Clearly state that validation loss may be artificially low due to lookahead bias
   - Emphasize test loss as the reliable metric
   - In publications, prioritize test loss in results tables and analysis

### When Validation Loss Might Be Acceptable

Validation loss may still be useful for:
- **Early stopping**: Even with bias, it can indicate overfitting relative to training loss
- **Relative comparisons**: Comparing validation loss across different models may still be informative (though test loss is preferred)
- **Rough guidance**: Quick checks during development, with final evaluation on test set

However, **never rely solely on validation loss** for final model selection or performance claims.

## Implementation Notes

### Current Code Behavior

Both repositories' split functions intentionally create overlaps for sequence construction:
- `seq_len` overlap: Allows input sequences to use historical context across split boundaries
- **Missing**: `pred_len` purge period to prevent prediction overlap

The overlap is **necessary for input sequences** but creates **undesirable overlap in prediction targets**.

### Why Changing Split Logic is Difficult

Changing the split logic would require:
1. Modifying `ratio_spliter` in fidel-ts
2. Modifying `Dataset_Custom.__read_data__` in MM-TSFlib
3. Updating all existing experiments and configurations
4. Breaking reproducibility with published results
5. Potentially reducing usable data (purge period reduces effective dataset size)

These costs outweigh the benefits, especially since **test loss remains reliable**.

## Recommendations

### For Researchers Using These Repositories

1. **Always report test loss** as the primary performance metric
2. **Acknowledge validation loss limitations** in papers and documentation
3. **Be skeptical** of validation loss that seems "too good," especially on small datasets
4. **Compare test losses** across models, not validation losses
5. **Use validation loss only for early stopping**, with final evaluation on test set

### For Future Work

1. **Document this issue** in repository README files
2. **Add warnings** in evaluation code about validation loss reliability
3. **Consider adding optional purge period** in future versions (with clear documentation)
4. **Report both metrics** but emphasize test loss in all results

## Conclusion

Both fidel-ts and MM-TSFlib exhibit a purge period issue that causes lookahead bias in validation loss. This is especially problematic for small datasets like Time-MMD, where validation periods are short relative to prediction lengths. 

**Key Takeaway**: Validation loss is unreliable due to lookahead bias. **Test loss is the trusted metric** and should be used as the primary evaluation criterion. We will maintain current split logic to preserve compatibility with original papers, but will always prioritize test loss in evaluations and reporting.
