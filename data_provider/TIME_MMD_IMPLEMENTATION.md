# Time-MMD Dataset Integration - Implementation Summary

## Overview

Successfully implemented integration of MM-TSFlib (Time-MMD) datasets into fidel-ts. The implementation allows loading Time-MMD CSV files (with embedded text columns) and using them with existing fidel-ts models without any code changes to models, training, or evaluation.

## Files Created/Modified

### New Files

1. **`data_provider/time_mmd_dataset.py`**
   - `TimeMMD_Dataset`: Dataset class that inherits from `Universal_Dataset`
   - `TimeMMD_HeteroGetter`: Text data getter that adapts MM-TSFlib format to fidel-ts interface
   - Handles CSV loading, text extraction, and format conversion

2. **`data_configs/time-mmd/README.md`**
   - Documentation for Time-MMD dataset configuration

3. **Example Config Files**:
   - `data_configs/time-mmd/Traffic/config.yaml`
   - `data_configs/time-mmd/Public_Health/config.yaml`
   - `data_configs/time-mmd/Energy/config.yaml`

### Modified Files

1. **`data_provider/data_factory.py`**
   - Added detection for `dataset_type: time_mmd`
   - Instantiates `TimeMMD_Dataset` when detected
   - Falls back to `Universal_Dataset` for other dataset types

## Key Features

### 1. Automatic Text Column Detection
- Auto-detects `Final_Search_{text_len}` or `Final_Output` columns
- Supports manual specification via `text_column` config
- Gracefully handles missing text columns (works as time-series-only)

### 2. Text Format Conversion
- Converts MM-TSFlib text format (single text at sequence end) to fidel-ts format
- Supports multiple output formats: `json`, `dict`, `csv`, `embedding`
- Uses backward matching to align text with timestamps

### 3. Data Compatibility
- Returns data in fidel-ts expected format
- Compatible with all existing models (TSF, TGTSF, MTSF, Reasoning)
- Supports all fidel-ts features (normalization, downsampling, timezone handling)

### 4. Empty Value Handling
- Returns empty strings for `hetero_general` and `hetero_channel` when not provided
- Compatible with string-concatenating models (ChatTime)
- Compatible with models that convert to float (with proper string checking)

## Usage

### 1. Prepare Data
Place Time-MMD CSV files in:
```
data/time-mmd/{subdataset}/{filename}.csv
```

### 2. Create Config File
Create `data_configs/time-mmd/{subdataset}/config.yaml` with:
```yaml
root_path: ./data/time-mmd/{subdataset}
data_path: {filename}.csv
dataset_type: time_mmd  # Required flag
timestamp_col: date
target: OT
spliter: ratio
split_info: [7, 1, 2]
text_column: auto
use_closedllm: false
text_len: 4
general_info: "Dataset description"
channel_info: ""
```

### 3. Use in Experiments
Reference the config in your experiment YAML:
```yaml
data_config: data_configs/time-mmd/Traffic/config.yaml
```

## Implementation Details

### Text Alignment Strategy
- MM-TSFlib stores text at sequence end point (`s_end`)
- fidel-ts expects text aligned to individual timestamps
- Solution: Use backward matching (text at or before timestamp)
- This ensures text describes context up to that point

### Data Flow
1. Load CSV file
2. Extract time series columns (normalize if requested)
3. Extract text column (if present)
4. Create `TimeMMD_HeteroGetter` with text data
5. Set as `hetero_data_getter` for `Universal_Dataset`
6. Return data in fidel-ts format via parent class

### Backward Compatibility
- No changes to existing datasets
- `dataset_type` is optional (defaults to standard behavior)
- Existing code continues to work unchanged

## Testing Checklist

- [ ] Load dataset with text column
- [ ] Load dataset without text column (time-series-only)
- [ ] Test with different text output formats (json, dict, csv)
- [ ] Test with different tasks (TSF, TGTSF, MTSF)
- [ ] Test with downsampling
- [ ] Test with normalization
- [ ] Verify text alignment correctness
- [ ] Test with models that use text (ChatTime, lynx, etc.)

## Future Enhancements

1. Support for `prior_history_avg` column as additional feature
2. Support for multiple text columns
3. Automatic config generation from CSV headers
4. Text embedding pre-computation option
5. Support for embedding format with proper dimension handling

## Notes

- Text data is stored in memory (aligned with timestamps)
- Empty strings are returned for static info when not provided
- The implementation maintains modularity and avoids code duplication
- All fidel-ts features work seamlessly with Time-MMD datasets

