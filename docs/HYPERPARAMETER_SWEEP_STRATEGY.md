# Hyperparameter Sweep Strategy Recommendations

## Recommended Approach: Combined Sweep

For your use case (model architecture + training hyperparameters), I recommend a **combined sweep** using Bayesian optimization. Here's why:

### Advantages of Combined Sweep

1. **Efficiency**: Bayesian optimization learns from all parameter combinations simultaneously, finding better configurations faster
2. **Interactions**: Captures interactions between architecture and training parameters (e.g., larger models may need different learning rates)
3. **State-of-the-art**: Modern HPO (e.g., Optuna, WandB sweeps) uses combined optimization
4. **Time savings**: Single sweep vs. multiple sequential sweeps

### Recommended Parameters

Based on your model (`lynx_film_raw`) and the sweep config I created:

**Architecture Parameters:**
- `e_layers` (depth): [2, 3, 4, 5] - Number of transformer layers
- `d_model` (width): [128, 256, 512] - Model dimension
- `n_heads`: [4, 8] - Attention heads

**Training Parameters:**
- `learning_rate`: log_uniform(1e-5, 1e-2) - Learning rate
- `batch_size`: [256, 512, 768, 1024] - Batch size

**Why these ranges:**
- `e_layers`: 2-5 covers shallow to moderately deep
- `d_model`: 128-512 covers small to medium models
- `n_heads`: 4-8 are common values (must divide d_model)
- `learning_rate`: Wide log range to find optimal LR
- `batch_size`: Practical range for GPU memory

### Additional Parameters to Consider

1. **`d_ff` (feed-forward dimension)**:
   - Typically 4×d_model, but can be swept
   - Range: [512, 1024, 2048, 4096]
   - **Recommendation**: Keep at 4×d_model (derived) or add as separate parameter

2. **`dropout`**:
   - Range: [0.0, 0.1, 0.2, 0.3, 0.5]
   - **Recommendation**: Include if overfitting is a concern

3. **`text_dim`** (for FiLM):
   - Range: [128, 256, 512]
   - **Recommendation**: Include if text embeddings are used

4. **`patch_len` and `stride`**:
   - Current: patch_len=6, stride=3
   - **Recommendation**: Keep fixed for initial sweep, add later if needed

### Sequential/Tiered Sweeps (Alternative)

If compute is very limited, consider a two-stage approach:

**Stage 1: Architecture Search**
- Fix training params (use defaults)
- Sweep: e_layers, d_model, n_heads
- Find best architecture

**Stage 2: Training Optimization**
- Fix architecture (from Stage 1)
- Sweep: learning_rate, batch_size
- Fine-tune training

**When to use:**
- Very limited compute budget
- Architecture has much larger impact than training params
- Need to understand architecture effects separately

**Disadvantages:**
- Misses architecture-training interactions
- Takes longer (two sweeps)
- May not find global optimum

## State-of-the-Art Practices

### 1. Bayesian Optimization (Recommended)

**Method**: `bayes` in WandB sweeps
- Uses Gaussian Process to model performance surface
- Actively learns which regions to explore
- Efficient for continuous parameters

**Best for**: Your use case (mixed continuous/discrete)

### 2. Early Stopping

**Hyperband** (included in your sweep config):
- Stops poor runs early
- Saves compute for promising configurations
- `min_iter: 3` means runs must complete 3 epochs before stopping

#### How Hyperband Works with Asynchronous Jobs

**Traditional Hyperband (Synchronous)**:
- Assumes all runs start and complete checkpoints at the same time
- At each checkpoint (e.g., after 3 epochs), compares all runs and prunes worst ones
- Requires synchronized evaluation

**WandB Sweeps (Asynchronous)**:
- Multiple agents run simultaneously, starting at different times
- Runs complete epochs at different rates and finish at different times
- No global synchronization

**How WandB Adapts Hyperband for Asynchrony**:

1. **Bracket Formation**:
   - Runs are grouped into brackets based on when they start
   - Each bracket has a target size (e.g., 8 runs)

2. **Asynchronous Evaluation**:
   - When a run completes a checkpoint (e.g., epoch 3), WandB checks:
     - How many runs in this bracket have completed this checkpoint?
     - What are their metrics?
   - If enough runs have completed and some are clearly worse, WandB can:
     - Stop the worst runs that haven't reached the next checkpoint yet
     - Or mark them for early termination

3. **Flexible Pruning**:
   - Unlike synchronous Hyperband, WandB doesn't wait for all runs to reach a checkpoint
   - Uses statistical comparisons: if a run is significantly worse than others in its bracket, it can be stopped
   - Works even if runs finish at different times

**Does Asynchrony Matter?**

Yes, but WandB's implementation handles it:

**Advantages**:
- ✅ Still provides early stopping benefits
- ✅ Adapts to asynchronous execution
- ✅ Works with multiple agents running simultaneously
- ✅ Better than no early stopping

**Limitations**:
- ⚠️ Less strict than synchronous Hyperband
- ⚠️ May not prune as aggressively as the ideal synchronous case
- ⚠️ Efficiency gains are still present but may be slightly reduced

**Practical Impact**:
- Multiple agents can run simultaneously
- Each run trains independently
- WandB tracks metrics asynchronously
- Hyperband still prunes poor runs, just not as strictly as synchronous execution
- You still save compute by stopping bad configurations early

**Your Configuration**:
```yaml
early_terminate:
  type: hyperband
  min_iter: 3      # Runs must complete at least 3 epochs
  max_iter: 7      # Maximum epochs
  s: 2             # Successive halving factor (keep top 50%)
```

**How this works with async agents**:
1. Each agent gets a hyperparameter combination and starts training
2. After 3 epochs, WandB evaluates the run's performance
3. WandB compares it to other runs in the same bracket
4. If it's in the bottom 50% (s=2 means keep top 50%), WandB signals early termination
5. The wrapper can check for early termination signals and stop training
6. The agent then requests the next combination

**Bottom Line**: Hyperband works with asynchronous jobs in WandB sweeps. The implementation is adapted for this scenario, and you'll still see efficiency gains from early stopping, even if not as strict as theoretical synchronous Hyperband.

### 3. Parameter Importance

After sweep completes:
- Use WandB's parameter importance plots
- Identify which parameters matter most
- Focus future sweeps on important parameters

### 4. Pruning Strategy

**Progressive pruning**:
1. Start with wide ranges
2. Analyze results
3. Narrow ranges around best values
4. Run focused sweep

## Recommended Sweep Configuration

I've created `configs/sweep_configs/lynx_film_raw_architecture_sweep.yaml` with:

```yaml
method: bayes
parameters:
  model_config_overrides.e_layers: [2, 3, 4, 5]
  model_config_overrides.d_model: [128, 256, 512]
  model_config_overrides.n_heads: [4, 8]
  training.learning_rate: log_uniform(1e-5, 1e-2)
  training.batch_size: [256, 512, 768, 1024]
```

**Total combinations**: 4 × 3 × 2 × (continuous) × 4 = Many, but Bayesian will sample efficiently

**Expected runs**: 20-50 for good coverage (Bayesian optimization)

## Execution Strategy

### Phase 1: Initial Sweep (Current Config)
- 7 epochs (fast iteration)
- 20-30 runs
- Identify promising regions

### Phase 2: Focused Sweep (Optional)
- Narrow parameter ranges around best values
- Increase epochs to 20-30
- 10-15 runs for fine-tuning

### Phase 3: Final Validation
- Best configuration from sweep
- Full training (50 epochs)
- Final evaluation

## Monitoring Recommendations

1. **Track these metrics**:
   - `val_loss` (primary - used for optimization)
   - `train_loss` (check for overfitting)
   - `val_mae`, `val_rmse` (additional metrics)
   - Training time per epoch

2. **Watch for**:
   - Overfitting (train_loss << val_loss)
   - Underfitting (both high)
   - Training instability (loss spikes)

3. **Parameter interactions to observe**:
   - Larger models (higher d_model, e_layers) may need lower LR
   - Larger batch_size may need higher LR
   - More heads may need more regularization

## Summary

**Recommendation**: Use the combined sweep I created with Bayesian optimization. It's:
- ✅ State-of-the-art approach
- ✅ Efficient (learns from all parameters)
- ✅ Captures interactions
- ✅ Single sweep (faster)

**Additional parameters to consider later**:
- `dropout` (if overfitting)
- `d_ff` (if you want to decouple from d_model)
- `text_dim` (if using text embeddings)

**Start with**: The sweep config I created, run 20-30 runs, analyze results, then decide on next steps.
