# Tensor Cache Fixes

## Summary

Fixed three critical issues with tensor cache generation and validation for time_mmd/ttc datasets:

1. **Cache directory path** - Fixed to save caches in the correct location within dataset directories
2. **Config hash mismatch** - Fixed inconsistent hash computation between CLI and generator
3. **Memory-mapped file format** - Fixed to use proper .npy format for efficient memory-mapped I/O

## Issue 1: Cache Directory Path

### Problem
For time_mmd and ttc datasets, caches were being saved to:
```
data/time_mmd/tensor_cache/<hash>/          # WRONG - too high in hierarchy
```

Instead of the correct location:
```
data/time_mmd/Traffic/tensor_cache/<hash>/   # CORRECT - within dataset dir
```

### Root Cause
In `cli/tensor_cache.py`, the `resolve_cache_dir()` function was using:
```python
dataset_dir = os.path.dirname(root_path.rstrip('/\\'))
```

This went **one directory level too high** by taking the parent of `root_path`.

### Fix
Changed to use the `root_path` directly:
```python
dataset_dir = root_path.rstrip('/\\')
return Path(dataset_dir) / 'tensor_cache' / config_hash
```

### Result
Now caches are correctly saved as:
- time_mmd datasets: `data/time_mmd/Traffic/tensor_cache/<hash>/`
- Other datasets: `data/fidel-ts/germany_renewable/time_series/tensor_cache/<hash>/`

This matches the structure used for embeddings and other dataset artifacts.

## Issue 2: Config Hash Mismatch

### Problem
After generating a cache, validation immediately reported it as invalid:
```
Expected: a2f9f7f43f9a548f
Got:      446b4deae4deba88
```

This caused the cache to regenerate every time, even though it was just created.

### Root Cause
The CLI and the `TensorCacheGenerator` were computing hashes from **different sets of parameters**:

**CLI** (via `utils.experiment_config_builder.build_cache_config()`):
- Includes: `input_len`, `output_len`, `scale`, `truncate_train_for_purge`, `downsample`, `data_name`
- **PLUS**: `hetero_stride`, `hetero_type`, `timemmd_text_output`, `missing_value_strategy`, `split_info`

**Generator** (via `TensorCacheGenerator._extract_cache_config()`):
- Only includes: `input_len`, `output_len`, `scale`, `truncate_train_for_purge`, `downsample`, `data_name`

The missing parameters (especially `timemmd_text_output`) caused different hashes.

### Fix
Modified `TensorCacheGenerator` to use the config passed in from the centralized builder instead of extracting its own:

**Before:**
```python
self.config_hash = compute_config_hash(self._extract_cache_config())
```

**After:**
```python
self.config_hash = compute_config_hash(config)  # Use config from centralized builder
```

The `_extract_cache_config()` method is now deprecated and marked as such.

### Result
Both CLI and Generator now compute identical hashes from the same parameter set, ensuring:
- Validation succeeds after generation
- No unnecessary regeneration
- Cache reuse works correctly across experiments with identical configs

## Files Modified

1. **cli/tensor_cache.py**
   - `resolve_cache_dir()`: Fixed to use root_path directly (not parent)
   - Updated docstring to clarify cache location behavior

2. **data_provider/tensor_cache.py**
   - Module docstring: Updated usage example to show centralized config builder
   - `TensorCacheGenerator.__init__()`: Now uses config passed in directly
   - `TensorCacheGenerator._create_mmap_arrays()`: Changed from `np.memmap` to `np.lib.format.open_memmap`
   - `_extract_cache_config()`: Deprecated with warning
   - `generate()`: Uses centralized config for metadata

3. **data_provider/data_factory.py** (Additional fixes for runtime)
   - `_resolve_tensor_cache_dir()`: Fixed to use root_path directly (not parent)
   - `_build_tensor_cache_config()`: Added missing `timemmd_text_output` parameter
   - Updated docstrings to clarify cache location and config consistency

## Testing

**IMPORTANT:** Old caches created with `np.memmap` are incompatible and must be regenerated.

To verify the fixes work:

```bash
# Clean up old incompatible caches
rm -rf data/time_mmd/*/tensor_cache/

# Generate cache with proper format
python -m cli.tensor_cache generate configs/experiment_suites/lynx_film_raw/time_mmd_ttc.yaml --filter "traffic"

# Validate (should succeed)
python -m cli.tensor_cache validate configs/experiment_suites/lynx_film_raw/time_mmd_ttc.yaml --filter "traffic"

# Test training with tensor cache
python -m cli.suite run configs/experiment_suites/tensor_cache_quick_test.yaml

# Verify cache location
ls data/time_mmd/Traffic/tensor_cache/
```

Expected results:
- Cache generated at `data/time_mmd/Traffic/tensor_cache/<hash>/`
- Validation reports cache as valid
- Training loads from cache without errors
- Running generate again skips (cache already valid)

## Additional Testing

Created a quick test suite to verify tensor cache works during training:
- **File:** `configs/experiment_suites/tensor_cache_quick_test.yaml`
- **Usage:** `python -m cli.suite run configs/experiment_suites/tensor_cache_quick_test.yaml`
- **Features:**
  - Tests lynx_film_raw training with tensor cache
  - Auto-resolves cache directory (no hardcoded paths)
  - Quick 2-epoch test for fast verification
  - Uses time_mmd Traffic dataset

## Design Notes

### Why Use Centralized Config Builder?

The centralized `build_cache_config()` in `utils/experiment_config_builder.py` is the single source of truth for:
- Which parameters affect cache validity
- How configs are merged (template + overrides)
- Consistent behavior across all tools (CLI, profiling, training)

This prevents subtle bugs where different parts of the codebase have different ideas about what makes a cache unique.

**Critical:** The `_build_tensor_cache_config()` method in `Data_Provider` must include the same parameters as the centralized builder, especially `timemmd_text_output` for time_mmd datasets!

## Issue 3: Memory-Mapped File Format

### Problem
After fixing the first two issues, loading cached tensors failed with:
```
_pickle.UnpicklingError: Failed to interpret file PosixPath('data/time_mmd/Traffic/tensor_cache/.../sample_ids.npy') as a pickle
```

### Root Cause
The code was mixing two different approaches to memory-mapped files:

**Saving** (in `_create_mmap_arrays()`):
```python
arrays[name] = np.memmap(filepath, dtype=dtype, mode='w+', shape=shape)
```
This creates a **raw binary file** with `.npy` extension but no proper .npy format headers.

**Loading** (in `_load_arrays()`):
```python
arrays[name] = np.load(filepath, mmap_mode='r', allow_pickle=True)
```
This expects a **proper .npy format file** with headers and metadata.

When `np.load` tries to read a raw binary file created by `np.memmap`, it fails to parse the header and tries to interpret it as a pickle file, causing the error.

### Fix
Use `np.lib.format.open_memmap()` instead of `np.memmap()` for creating arrays:

**Before:**
```python
arrays[name] = np.memmap(filepath, dtype=dtype, mode='w+', shape=shape)
```

**After:**
```python
arrays[name] = np.lib.format.open_memmap(
    str(filepath), dtype=dtype, mode='w+', shape=shape
)
```

### Why This is Better
`np.lib.format.open_memmap()`:
- Creates files in proper .npy format with headers
- Supports memory-mapped I/O (efficient, no full load into RAM)
- Compatible with `np.load(..., mmap_mode='r')` for loading
- No conversion step needed - arrays are written in the correct format
- Maintains full efficiency of memory-mapped operations

This is the **canonical way** to create memory-mapped .npy files in NumPy.

### Cache Location Strategy

Caches are stored directly under the dataset's `root_path`:
```
{root_path}/tensor_cache/{config_hash}/
```

This ensures:
- Caches are co-located with their data
- Easy to find and clean up per-dataset
- No confusion about which cache belongs to which dataset
- Consistent with other dataset artifacts (embeddings, id_info, etc.)
