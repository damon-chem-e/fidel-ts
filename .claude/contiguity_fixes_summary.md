# Tensor Contiguity Fixes - Implementation Summary

## Status: ✓ COMPLETE

All three models (`lynx_film_raw`, `lynx_film`, and `TGTSF`) are now protected against tensor contiguity errors when using `torch.compile`.

## Error Fixed
```
RuntimeError: (*bias): last dimension must be contiguous
```

This error occurred during training with `torch.compile(mode='reduce-overhead')` due to non-contiguous tensor memory layouts in attention operations.

## Changes Implemented

### 1. lynx_film_raw.py (Line 303) ✓
**Status**: FIXED and TESTED - Training works successfully

**Change**:
```python
# Before:
text_emb = text_emb.permute(0, 2, 1, 3) # [B, C, L, D]

# After:
# Ensure contiguity after permute for torch.compile compatibility
# The compiled attention kernels require contiguous tensors
text_emb = text_emb.permute(0, 2, 1, 3).contiguous()  # [B, C, L, D]
```

**Impact**: Fixed the immediate compilation error. Model trains successfully with torch.compile.

### 2. lynx_film.py (Line 203) ✓
**Status**: FIXED (preventive)

**Change**:
```python
# Before:
text_emb = text_emb.permute(0, 2, 1, 3) # [B, C, L, D]

# After:
# Ensure contiguity after permute for torch.compile compatibility
# The compiled attention kernels require contiguous tensors
text_emb = text_emb.permute(0, 2, 1, 3).contiguous()  # [B, C, L, D]
```

**Impact**: Same pattern as lynx_film_raw. Prevents the same error from occurring when lynx_film is used with torch.compile.

### 3. TGTSF_torch.py - text_encoder.forward() ✓

#### Fix 3a: Lines 167 & 172
**Status**: FIXED (preventive)

**Change**:
```python
# Before:
news_emb = news_emb.view(B*L, news_emb.shape[2], D)  # [b*l, n, d]
description_emb = description_emb.view(B*L, description_emb.shape[2], D)  # [b*l, c, d]

# After:
# Ensure contiguity before view operations for torch.compile
news_emb = news_emb.contiguous().view(B*L, news_emb.shape[2], D)  # [b*l, n, d]
description_emb = description_emb.contiguous().view(B*L, description_emb.shape[2], D)  # [b*l, c, d]
```

**Impact**: Ensures tensors are contiguous before `.view()` operations, which is required for torch.compile's optimized kernels.

#### Fix 3b: Line 211
**Status**: FIXED (preventive)

**Change**:
```python
# Before:
x = rearrange(text_emb, 'b l c d -> (b c) l d', b=B, c=C)

# After:
# Ensure contiguity after rearrange for attention operations
x = rearrange(text_emb, 'b l c d -> (b c) l d', b=B, c=C).contiguous()
```

**Impact**: Ensures tensor is contiguous after `rearrange()` before being passed to attention operations.

### 4. TGTSF.py ✓
**Status**: NO CHANGES NEEDED

The TGTSF model already had `.contiguous()` at line 283:
```python
x_reshaped = x.reshape(B * C, N, L).permute(0, 2, 1).contiguous()  # Already correct!
```

**Impact**: No changes needed. This model was already protected.

## Models Protected

### ✓ lynx_film_raw
- **Status**: TESTED and working with torch.compile
- **Files modified**: `models/lynx_film_raw.py`
- **Fixes applied**: Phase 1 + Phase 2 (shared layers)

### ✓ lynx_film
- **Status**: PROTECTED (same pattern as lynx_film_raw)
- **Files modified**: `models/lynx_film.py`
- **Fixes applied**: Phase 1 + Phase 2 (shared layers)

### ✓ TGTSF
- **Status**: PROTECTED (uses shared text_encoder with fixes)
- **Files modified**: None (already correct) + Phase 2 in shared layers
- **Fixes applied**: Phase 2 (shared layers)

## Shared Components

All three models use the same `text_encoder` from `layers/TGTSF_torch.py`, which has been fixed. This means:
- Any future models using `text_encoder` are automatically protected
- The fixes are centralized in the shared layer code
- No need to add `.contiguous()` in each model individually (except for model-specific permutes)

## Testing Status

### lynx_film_raw
✓ **CONFIRMED WORKING** - User reported "that worked" after Phase 1 implementation

### lynx_film
⚠️ **NOT YET TESTED** - Should work based on identical pattern to lynx_film_raw

**Recommended test**:
```bash
python -m runs.suite_executor \
    --suite configs/experiment_suites/lynx_film/[appropriate_config].yaml \
    --max_epochs 1 \
    --max_experiments 1
```

### TGTSF
⚠️ **NOT YET TESTED** - Protected by shared layer fixes

**Recommended test**:
```bash
python -m runs.suite_executor \
    --suite configs/experiment_suites/tgtsf_test.yaml \
    --max_epochs 1 \
    --max_experiments 1
```

## Performance Impact

**Expected**: NEGLIGIBLE
- `.contiguous()` is a no-op when tensor is already contiguous (most cases)
- Only creates a copy when necessary (one-time cost per batch)
- Memory copy cost << attention computation cost
- torch.compile speedup far exceeds any copy overhead

**Measured**: Not yet benchmarked
- Recommend comparing training speed before/after fixes
- Expect 1.2-2x overall speedup from torch.compile (even with .contiguous() calls)

## Why These Fixes Work

1. **torch.compile is strict**: Compiled attention kernels require contiguous memory layouts for efficiency
2. **permute() creates non-contiguous tensors**: Changes stride pattern without copying data
3. **.contiguous() forces contiguity**: Creates a copy with sequential memory layout when needed
4. **Minimal overhead**: Only copies when necessary; no-op otherwise

## Future Considerations

### When Adding New Models
If you create new models that:
1. Use `.permute()` on tensors
2. Pass those tensors to attention operations
3. Use `torch.compile`

**Always add `.contiguous()` after permute operations**:
```python
tensor = tensor.permute(...).contiguous()  # ✓ Safe for torch.compile
```

### When Modifying Existing Code
Be careful with these operations:
- `.permute()` - Always add `.contiguous()` if tensor goes to attention
- `.transpose()` - Same as permute
- `.view()` - Requires input to be contiguous (use `.contiguous()` before)
- `.reshape()` - More forgiving but may create copies; prefer `.view()` + `.contiguous()`

## Files Modified

1. `models/lynx_film_raw.py` - Line 303
2. `models/lynx_film.py` - Line 203
3. `layers/TGTSF_torch.py` - Lines 167, 172, 211

## Commit Recommendation

**Commit Message**:
```
Fix tensor contiguity for torch.compile compatibility

Add .contiguous() calls after permute/reshape operations in lynx_film_raw,
lynx_film, and shared TGTSF_torch text_encoder to fix RuntimeError when
using torch.compile. Compiled attention kernels require contiguous tensors.

Fixes:
- models/lynx_film_raw.py: Add .contiguous() after permute (line 303)
- models/lynx_film.py: Add .contiguous() after permute (line 203)
- layers/TGTSF_torch.py: Add .contiguous() in text_encoder (lines 167, 172, 211)

Impact: Minimal performance overhead, enables torch.compile speedups
Tested: lynx_film_raw trains successfully with torch.compile
```

## References

- **Diagnosis Document**: `.claude/contiguity_error_diagnosis.md`
- **Fix Plan**: `.claude/contiguity_error_fix_plan.md`
- **PyTorch Docs**: https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html
- **torch.compile Tutorial**: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

---

**Date**: 2026-01-15  
**Status**: Complete  
**Tested**: lynx_film_raw ✓  
**Risk Level**: LOW  
**Performance Impact**: NEGLIGIBLE
