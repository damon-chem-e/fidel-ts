# Time-MMD Implementation - Fixes Summary

## All Issues Addressed

### ✅ 1. Embeddings and Output Format

**Issue**: Time-MMD returns zero embeddings when `output_format == 'embedding'`

**Clarification**:
- `output_format == 'embedding'` expects **pre-computed embeddings from files** (`.pkl`)
- Time-MMD has text in CSV columns, not pre-computed embeddings
- **Current behavior is correct**: Returns zeros for 'embedding' format
- **For embeddings**: Use `output_format='json'` or `'dict'` (models handle text)
- **Future**: Support `postemb` to create embeddings during initialization

**When embeddings are created on-the-fly**:
- When `postemb` is set in `hetero_info`, embeddings are created **during initialization** (not during `__call__`)
- `output_format` is still `'json'` or `'dict'`, but actual data is embeddings (torch tensors)
- Time-MMD doesn't support `postemb` yet (future enhancement)

### ✅ 2. Timestamp Matching Modularity

**Fixed**: Extracted `_match_timestamps()` method in `TimeMMD_HeteroGetter`
- Clean separation of concerns
- Easier to test and maintain

### ✅ 3. Output Format Determination

**Answer**: 
- `output_format` comes from `args.data_config.hetero_info.input_format`
- Default: `'json'` if `hetero_info` not specified
- When calling from `cli.train`: Uses `hetero_info.input_format` from model config

**Usage**:
- `json`: Default, used by most models (ChatTime, etc.)
- `dict`: Alternative text format  
- `csv`: CSV string format
- `embedding`: Pre-computed embeddings (Time-MMD returns zeros - correct behavior)

### ✅ 4. Text Column Detection - No Fallbacks

**Fixed**: Removed all fallbacks in `_detect_text_column()`
- If `text_column != 'auto'` and not found → raises `ValueError` (strict)
- If `text_column == 'auto'` and `use_closedllm=True` → only looks for `Final_Output` (no fallback)
- If `text_column == 'auto'` and `use_closedllm=False` → only looks for `Final_Search_{text_len}` (no fallback to other text_len)

### ✅ 5. Text Columns Excluded from Time Series Data

**Fixed**: ALL text columns and metadata columns are now excluded:
- ALL `Final_Search_*` columns (regardless of which one we're using)
- `Final_Output` column (if present)
- `start_date`, `end_date`
- These are dropped before extracting time series data

### ✅ 6. Downsampling - No Lookahead Bias

**Fixed**: Uses `reindex()` with forward fill (ffill) method
- **Method**: `self._text_data.reindex(self.timestamp, method='ffill')`
- **Why ffill**: Forward fill uses the last valid observation (previous value in time)
- **No lookahead bias**: Only uses past/present text values, never future values
- **Handles missing timestamps**: If a downsampled timestamp doesn't exist in text_data, uses the most recent previous text value
- Matches `Universal_Dataset` behavior for time series: `self.data[::self.downsample]`

### ✅ 7. Modular Time-MMD Check

**Fixed**: Extracted to two methods in `data_factory.py`:
- `_is_time_mmd_dataset()`: Checks if dataset is Time-MMD
- `_create_time_mmd_dataset(i, flag)`: Creates Time-MMD dataset instance
- Both code paths (with/without progress bar) use these methods
- Single source of truth for Time-MMD instantiation

### ✅ 8. Path Rename: time-mmd → time_mmd

**Fixed**: 
- Renamed directory: `data_configs/time-mmd/` → `data_configs/time_mmd/`
- Updated all config files: `./data/time-mmd/` → `./data/time_mmd/`
- Updated documentation

### ✅ 9. base_data_path Support

**Verified**: `base_data_path` works correctly
- `replace_data_paths()` in `utils/data_path_utils.py` recursively replaces `'./data'` with `base_data_path`
- Time-MMD configs use `root_path: ./data/time_mmd/...`
- Automatically handled by run scripts before data loading
- **Example**: `./data/time_mmd/Traffic` → `/nfs/data/time_mmd/Traffic` (if `base_data_path='/nfs/data'`)

## Files Modified

1. **`data_provider/time_mmd_dataset.py`**
   - Extracted `_match_timestamps()` method
   - Removed fallbacks in `_detect_text_column()`
   - Excluded text/metadata columns from time series data
   - Fixed downsampling (no lookahead bias)
   - Improved embedding format documentation

2. **`data_provider/data_factory.py`**
   - Added `_is_time_mmd_dataset()` method
   - Added `_create_time_mmd_dataset(i, flag)` method
   - Both code paths use these methods
   - Fixed import formatting

3. **Config Files**
   - Updated paths: `time-mmd` → `time_mmd`
   - All configs in `data_configs/time_mmd/`

4. **Documentation**
   - Updated README with new paths
   - Created `TIME_MMD_QUESTIONS_ANSWERS.md`
   - Created `TIME_MMD_FIXES_SUMMARY.md`

## Testing Recommendations

1. **Test with different output formats**:
   - `json`: Should work (default)
   - `dict`: Should work
   - `csv`: Should work
   - `embedding`: Returns zeros (correct for pre-computed embeddings)

2. **Test text column detection**:
   - `text_column='auto'` with `Final_Search_4` present
   - `text_column='auto'` with `Final_Output` present
   - `text_column='Final_Search_2'` (should fail if not present, no fallback)

3. **Test downsampling**:
   - Verify no lookahead bias
   - Text data should align with downsampled timestamps

4. **Test base_data_path**:
   - Set `base_data_path` in config
   - Verify paths are correctly replaced

5. **Test without text column**:
   - Dataset should work as time-series-only
   - No errors, just info messages

## Known Limitations

1. **`postemb` not supported**: Embeddings created during initialization (future enhancement)
2. **Pre-computed embeddings**: `output_format='embedding'` returns zeros (correct, but not useful for Time-MMD)

## Future Enhancements

1. Support `postemb` to create embeddings during initialization from text
2. Support pre-computed embeddings from files (if needed)
3. Support `prior_history_avg` as additional feature

