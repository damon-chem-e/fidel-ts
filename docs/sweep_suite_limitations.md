# Sweep Suite Limitations: Multiple Experiments

## Overview

When running a wandb sweep on an experiment suite that contains **more than one experiment**, there is an important limitation regarding how the sweep objective metric is determined.

## The Issue

### Current Behavior

When a suite contains multiple experiments, all experiments run sequentially within the **same wandb run**. Each experiment logs its own `val_loss` (or other metrics) at different steps/epochs. 

**Wandb uses the LAST logged value of the objective metric** (e.g., `val_loss`) to determine the sweep objective for that trial. This means:

- Only the **last experiment's validation loss** affects the sweep optimization
- Earlier experiments' validation losses are effectively ignored for sweep purposes
- The sweep will optimize hyperparameters based solely on the performance of the final experiment in the suite

### Example

Consider a suite with 3 experiments:

```
Experiment 1: logs val_loss = 0.5 at step 10
Experiment 2: logs val_loss = 0.3 at step 20  
Experiment 3: logs val_loss = 0.4 at step 30
```

Wandb will use `val_loss = 0.4` (from Experiment 3) as the objective value for this sweep trial. The losses from Experiments 1 and 2 are logged but do not influence the sweep optimization.

### Why This Happens

1. All experiments in a suite share the same wandb run (created by `wandb.agent()`)
2. Each experiment logs metrics independently using `wandb.run.log()`
3. Wandb sweep controller reads the **final value** of the specified metric from the run summary
4. There is no built-in aggregation of metrics across multiple experiments in a suite

## Current Recommendation

**For sweeps, use suites with only ONE experiment.**

If you need to sweep hyperparameters:
- Create a suite with a single experiment
- Or run separate sweeps for each experiment you want to optimize

This ensures the sweep objective accurately reflects the performance you want to optimize.

## Future Plans

We plan to implement a feature that allows you to specify an **aggregation method** for validation loss (or other metrics) across experiments in a suite. This would enable options such as:

- **Sum**: `suite_val_loss = sum(val_loss_1, val_loss_2, val_loss_3)`
- **Mean**: `suite_val_loss = mean(val_loss_1, val_loss_2, val_loss_3)`
- **Weighted average**: `suite_val_loss = weighted_mean(val_loss_1, val_loss_2, val_loss_3, weights=[w1, w2, w3])`

The aggregated metric would be logged as a suite-level metric (e.g., `suite_val_loss`) that can be used as the sweep objective, allowing proper optimization across all experiments in the suite.

## Related Documentation

- [SWEEP_COMPREHENSIVE_GUIDE.md](./SWEEP_COMPREHENSIVE_GUIDE.md) - General sweep usage guide
- [job_resumption.md](./job_resumption.md) - Information about resuming sweep runs
