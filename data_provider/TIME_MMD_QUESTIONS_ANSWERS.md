# Time-MMD Implementation - Questions and Answers

## 1. Embeddings and Output Format

### Question: What determines output_format? When embeddings are required, will it provide correct embeddings?

**Answer:**

The `output_format` is determined by `hetero_info.input_format` in the model/data config. There are two scenarios:

#### Scenario A: `output_format == 'embedding'` (Pre-computed embeddings)
- Used when embeddings are pre-computed and stored in files (`.pkl` files)
- Time-MMD currently returns zero arrays when `output_format == 'embedding'`
- This is **correct behavior** because:
  - Pre-computed embeddings should come from files, not from CSV text columns
  - If you have pre-computed embeddings, you should use `Heterogeneous_Dataset` with `output_format='embedding'` instead
- **Current limitation**: Time-MMD doesn't support pre-computed embeddings from files

#### Scenario B: `postemb` is set (On-the-fly embeddings)
- When `postemb` is set in `hetero_info`, embeddings are created **during initialization** (not on-the-fly during `__call__`)
- The `output_format` is still `'json'` or `'dict'`, but the actual data returned is embeddings (torch tensors)
- This happens in `Heterogeneous_Dataset.load_data()` → `convert_df_text_to_embeddings()`
- **Current limitation**: Time-MMD doesn't support `postemb` yet (embeddings created during initialization)

**When different output formats are called:**
- `json`: Default, used by most models (ChatTime, etc.)
- `dict`: Alternative text format
- `csv`: CSV string format
- `embedding`: Pre-computed embeddings from files

**For Time-MMD with embeddings:**
- Currently, use `output_format='json'` or `'dict'` and let models handle text
- Future enhancement: Support `postemb` to create embeddings during initialization

## 2. Timestamp Matching Modularity

**Fixed**: Extracted `_match_timestamps()` method in `TimeMMD_HeteroGetter` for modularity and cleanliness.

## 3. Output Format Usage

**Answer:**
- `output_format` comes from `args.data_config.hetero_info.input_format` in the config
- When calling from `cli.train` with a Time-MMD dataset:
  - If `hetero_info` is specified in model config → uses `hetero_info.input_format` (default: `'json'`)
  - If `hetero_info` is not specified → defaults to `'json'`
- Most models use `'json'` format for text data

## 4. Text Column Detection Fallbacks

**Fixed**: Removed all fallbacks in `_detect_text_column()`:
- If `text_column != 'auto'` and not found → raises `ValueError` (no fallback)
- If `text_column == 'auto'` and `use_closedllm=True` → only looks for `Final_Output` (no fallback)
- If `text_column == 'auto'` and `use_closedllm=False` → only looks for `Final_Search_{text_len}` (no fallback to other text_len values)

## 5. Text Columns in Time Series Data

**Fixed**: Text columns and metadata columns (`start_date`, `end_date`, `prior_history_avg`, `prior_history_std`) are now excluded from time series data extraction.

## 6. Downsampling and Lookahead Bias

**Fixed**: 
- Changed from `reindex(..., method='nearest')` (which can cause lookahead bias)
- Now uses simple indexing: `self._text_data.iloc[downsampled_indices]`
- This matches `Universal_Dataset` behavior: `self.data[::self.downsample]`
- No lookahead bias - only uses past/present data

## 7. Modular Time-MMD Check

**Fixed**: Extracted to two methods in `data_factory.py`:
- `_is_time_mmd_dataset()`: Checks if dataset is Time-MMD
- `_create_time_mmd_dataset(i, flag)`: Creates Time-MMD dataset instance
- Both instances (with/without progress bar) use these methods

## 8. Path Rename: time-mmd → time_mmd

**Fixed**: 
- Renamed directory: `data_configs/time-mmd/` → `data_configs/time_mmd/`
- Updated all config files: `./data/time-mmd/` → `./data/time_mmd/`
- Updated documentation

## 9. base_data_path Support

**Answer:**

Yes, `base_data_path` works with Time-MMD datasets. The `replace_data_paths()` function in `utils/data_path_utils.py` recursively replaces `'./data'` with `base_data_path` in all string values.

Since Time-MMD configs use `root_path: ./data/time_mmd/...`, the `base_data_path` will correctly replace `./data` with the specified base path.

**Example:**
```yaml
# Config
root_path: ./data/time_mmd/Traffic

# With base_data_path: /nfs/data
# Becomes: /nfs/data/time_mmd/Traffic
```

This is handled automatically by the run scripts (`runs/pytorch.py`, `runs/lightning.py`, etc.) before data loading.

## Summary of Changes

✅ **Fixed Issues:**
1. Extracted timestamp matching to `_match_timestamps()` method
2. Removed all fallbacks in `_detect_text_column()`
3. Excluded text and metadata columns from time series data
4. Fixed downsampling to avoid lookahead bias (simple indexing)
5. Extracted Time-MMD check to modular methods
6. Renamed all paths from `time-mmd` to `time_mmd`
7. Verified `base_data_path` support

⚠️ **Current Limitations:**
1. `output_format='embedding'` returns zeros (correct for pre-computed embeddings, but Time-MMD doesn't support file-based embeddings)
2. `postemb` not yet supported (embeddings created during initialization)

🔮 **Future Enhancements:**
1. Support `postemb` to create embeddings during initialization from text
2. Support pre-computed embeddings from files (if needed)

