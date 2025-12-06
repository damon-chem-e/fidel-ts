# Patch Reconstruction Optimization Tests

## Purpose

This test suite validates a critical optimization to the TGTSF model's patch reconstruction step. The original implementation uses a sequential Python loop that becomes a bottleneck with larger batch sizes. The vectorized version uses `torch.nn.functional.fold` which is fully parallelized.

## Background

The patch reconstruction step in TGTSF (models/TGTSF.py:104-116) reconstructs overlapping patch predictions into a continuous sequence by averaging overlapping regions. This is necessary because:

1. TGTSF processes time series in overlapping patches
2. Each patch produces predictions for its time steps
3. Multiple patches can predict the same time step (overlap)
4. The final prediction averages all overlapping patch predictions

## Test Structure

- **Original Implementation**: Exact copy of the loop-based code from TGTSF
- **Vectorized Implementation**: Uses `torch.nn.functional.fold` for parallel processing
- **Test Fixtures**: Multiple configurations from small to xlarge scale
- **Validation**: Exact numerical equivalence + performance benchmarks

## Running Tests

```bash
# Run all tests
pytest tests/test_patch_reconstruction.py -v

# Run only correctness tests (no performance)
pytest tests/test_patch_reconstruction.py -v -k "not performance"

# Run only performance tests
pytest tests/test_patch_reconstruction.py -v -k "performance"

# Run with CUDA (if available)
CUDA_VISIBLE_DEVICES=0 pytest tests/test_patch_reconstruction.py -v
```

## Test Coverage

### Correctness Tests
- ✅ Small, medium, large, xlarge configurations
- ✅ TGTSF-like configuration (matching actual usage)
- ✅ Various stride/patch_len combinations
- ✅ Edge cases: no overlap, high overlap
- ✅ Gradient flow verification

### Performance Tests
- ✅ Medium config performance
- ✅ Large config performance  
- ✅ XLarge config performance
- ✅ Batch size scaling

## Expected Results

- **Correctness**: Outputs should match within `rtol=1e-6, atol=1e-6`
- **Performance**: Vectorized version should be 1.5-3x faster for large configs
- **Gradients**: Should flow correctly through vectorized implementation

## Implementation Notes

The vectorized implementation uses `torch.nn.functional.fold`, which:
- Is the inverse operation of `unfold`
- Automatically handles overlapping patches with averaging
- Is fully differentiable
- Is deterministic on CUDA
- Uses optimized CUDA kernels

## References

- Original implementation: `models/TGTSF.py:104-116`
- Optimization analysis: `docs/TGTSF_PERFORMANCE_OPTIMIZATION.md`
- Patch reconstruction explanation: `docs/PATCH_RECONSTRUCTION_EXPLAINED.md`

