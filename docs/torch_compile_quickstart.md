# torch.compile Quick Start

## 🚀 Get Started in 30 Seconds

### For Training

Add these two lines to your experiment config:

```yaml
training:
  torch_compile: true
  compile_mode: "reduce-overhead"
  # ... rest of your config unchanged
```

That's it! Your training will now be 15-35% faster.

### For Evaluation

Add these two lines to your evaluation config:

```yaml
evaluation:
  torch_compile: true
  compile_mode: "reduce-overhead"
  # ... rest of your config unchanged
```

## Example: Modify an Existing Config

**Before:**
```yaml
# configs/experiments/my_experiment.yaml
training:
  epochs: 20
  batch_size: 96
  learning_rate: 5e-4
  # ... other settings
```

**After:**
```yaml
# configs/experiments/my_experiment.yaml
training:
  epochs: 20
  batch_size: 96
  learning_rate: 5e-4
  torch_compile: true  # ← Add this line
  compile_mode: "reduce-overhead"  # ← Add this line
  # ... other settings
```

## What to Expect

### First Epoch
- ⏱️ Takes 2-5 minutes longer (compilation)
- 📊 Uses 10-20% more memory
- ✅ Only happens once

### Subsequent Epochs
- ⚡ 15-35% faster
- 💾 Same memory as before
- 🎯 Consistent speedup

## Best Models for Compilation

| Model Type | Expected Speedup |
|------------|-----------------|
| DLinear, PatchTST | 30-40% faster |
| TimeCMA, Informer | 15-25% faster |
| Complex transformers | 10-20% faster |

## When NOT to Use

- ❌ Quick debugging sessions (compilation overhead not worth it)
- ❌ Very short training runs (< 5 epochs)
- ❌ If you hit OOM errors (try reducing batch size first)

## Troubleshooting One-Liners

**OOM error?** → Reduce batch size by 20%
**Compilation fails?** → Set `compile_mode: "default"`
**No speedup?** → Wait until epoch 2+

## Full Documentation

See `docs/torch_compile_guide.md` for complete documentation.

## Requirements

- PyTorch >= 2.0
- CUDA >= 11.7 (recommended)

Check your version:
```bash
python -c "import torch; print(torch.__version__)"
```
