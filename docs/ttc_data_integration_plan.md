# TTC Data Integration Plan

## Overview

This document outlines the plan to integrate TTC (Time-to-Come) datasets into fidel-ts. These datasets include:
- **Climate data**: Single CSV file with weather time series and text descriptions
- **Medical data**: Multiple CSV files (one per patient) with health metrics and text notes

Both datasets follow a similar structure to Time-MMD datasets (CSV with embedded text column), but use a simpler column name: `text` instead of `Final_Search_*` or `Final_Output`.

## Data Structure

### Climate Dataset
- **Location**: `fidel-ts/data/ttc/climate/climate_2014_2023_final.csv`
- **Structure**: Single CSV file
- **Columns**:
  - `date`: Timestamp (YYYY-MM-DD format)
  - `temp`: Temperature (f64)
  - `precip`: Precipitation (f64)
  - `humidity`: Humidity (f64)
  - `windspeed`: Wind speed (f64)
  - `text`: Text descriptions (string)

### Medical Dataset
- **Location**: `fidel-ts/data/ttc/medical/patient_{number}.csv`
- **Structure**: Multiple CSV files, one per patient
- **Columns**:
  - `date`: Timestamp (YYYY-MM-DD format, may have future dates like 2157-12-13)
  - `Respiratory_Rate`: Respiratory rate (f64)
  - `Heart_Rate`: Heart rate (f64)
  - `SaO2`: Oxygen saturation (f64)
  - `FiO2`: Fraction of inspired oxygen (f64)
  - `text`: Medical notes (string)

## Key Differences from Time-MMD

1. **Text column name**: Uses `text` instead of `Final_Search_*` or `Final_Output`
2. **Column naming**: Different column names (not `OT` for target)
3. **Multi-entity support**: Medical dataset has multiple files (similar to standard fidel-ts multi-entity setup)
4. **No metadata columns**: No `start_date`, `end_date`, `prior_history_avg` columns

## Implementation Strategy

### Phase 1: Extend TimeMMD_Dataset Support

The existing `TimeMMD_Dataset` class can be reused with minimal modifications:

1. **Text column detection**: Extend `_detect_text_column()` to handle `text` column
   - Current logic: Detects `Final_Search_{text_len}` or `Final_Output`
   - New logic: Also detect `text` column when specified or when `text_column='text'`
   - Priority: If `text_column='text'` is specified, use that. Otherwise, fall back to existing Time-MMD detection logic.

2. **Target column flexibility**: 
   - Current: Assumes target column named `OT` or `all` for all columns
   - TTC: Climate uses all columns except `date` and `text`, Medical uses all columns except `date` and `text`
   - Solution: Use `target: 'all'` in config and ensure text/metadata columns are excluded

3. **No changes needed to TimeMMD_HeteroGetter**: It already handles any text column name correctly since it works with the extracted text series.

### Phase 2: Create Data Configurations

#### Climate Dataset Config
**File**: `data_configs/ttc/climate/config.yaml`

```yaml
root_path: ./data/ttc/climate
data_path: climate_2014_2023_final.csv
spliter: ratio
split_info: [7, 1, 2]  # 70% train, 10% val, 20% test
timestamp_col: date
target: all  # Use all columns (temp, precip, humidity, windspeed)
id: all
formatter: climate_2014_2023_final.csv
id_info: id_info.json  # Will be auto-created if not present
dataset_type: time_mmd  # Use TimeMMD_Dataset
text_column: text  # Explicitly specify text column
use_closedllm: false  # Not using Final_Output
text_len: 4  # Not relevant for 'text' column, but required for compatibility
general_info: "Climate data with weather forecasts and text descriptions"
channel_info: ""  # Single entity dataset
time_zone: null
downsample: null

# Text embedding configuration
timemmd_text_output: text  # 'text' | 'embedding'
timemmd_embed_model: bert-base-uncased
timemmd_embed_dim: 768
timemmd_force_reembed: false
hf_cache_dir: ./HF_cache/
```

#### Medical Dataset Config
**File**: `data_configs/ttc/medical/config.yaml`

```yaml
root_path: ./data/ttc/medical
data_path: null  # Not used, will use formatter with {i} placeholder
spliter: ratio
split_info: [7, 1, 2]  # 70% train, 10% val, 20% test
timestamp_col: date
target: all  # Use all columns (Respiratory_Rate, Heart_Rate, SaO2, FiO2)
id: all  # Process all patient files
formatter: patient_{i}.csv  # Pattern for patient files
id_info: id_info.json  # Required: lists all patient IDs
dataset_type: time_mmd  # Use TimeMMD_Dataset
text_column: text  # Explicitly specify text column
use_closedllm: false  # Not using Final_Output
text_len: 4  # Not relevant for 'text' column
general_info: "Medical patient data with health metrics and text notes"
channel_info: ""  # Will be set per patient if needed
time_zone: null
downsample: null

# Text embedding configuration
timemmd_text_output: text  # 'text' | 'embedding'
timemmd_embed_model: bert-base-uncased
timemmd_embed_dim: 768
timemmd_force_reembed: false
hf_cache_dir: ./HF_cache/
```

### Phase 3: Create id_info.json Files

#### Climate Dataset id_info.json
**File**: `data/ttc/climate/id_info.json`

```json
{
  "climate": {
    "description": "Climate dataset with weather time series and text descriptions"
  }
}
```

Or use auto-creation (Time-MMD already supports this).

#### Medical Dataset id_info.json
**File**: `data/ttc/medical/id_info.json`

This file must list all patient IDs. It can be auto-generated by scanning the directory for `patient_*.csv` files.

**Script to generate**: `scripts/generate_medical_id_info.py` (or manual creation)

Example structure:
```json
{
  "517": {
    "description": "Patient 517 medical data"
  },
  "518": {
    "description": "Patient 518 medical data"
  },
  ...
}
```

### Phase 4: Code Modifications

#### Modification 1: Extend `_detect_text_column()` in `TimeMMD_Dataset`

**File**: `data_provider/time_mmd_dataset.py`

**Change**: Update `_detect_text_column()` method to handle explicit `text` column specification:

```python
def _detect_text_column(self, df_raw):
    """
    Detect text column name from CSV.
    
    Supports:
    - Explicit specification: text_column='text' or any column name
    - Time-MMD auto-detection: Final_Search_{text_len} or Final_Output
    - Explicit 'text' column: Checks for 'text' column when text_column='auto' (fallback)
    
    Args:
        df_raw: Raw DataFrame loaded from CSV
        
    Returns:
        str or None: Column name if found, None otherwise
    """
    # If explicitly specified, use that column (or raise error if not found)
    if self.text_column != 'auto':
        if self.text_column in df_raw.columns:
            return self.text_column
        else:
            raise ValueError(
                f'Specified text column "{self.text_column}" not found in CSV columns: '
                f'{list(df_raw.columns)}'
            )
    
    # Auto-detect logic: Try Time-MMD patterns first, then 'text' column
    if self.use_closedllm:
        if 'Final_Output' in df_raw.columns:
            return 'Final_Output'
    
    # Look for Final_Search_{text_len} pattern
    pattern = f'Final_Search_{self.text_len}'
    if pattern in df_raw.columns:
        return pattern
    
    # Fallback: Check for 'text' column (for TTC and similar datasets)
    if 'text' in df_raw.columns:
        return 'text'
    
    # No text column found
    return None
```

**Rationale**: This maintains backward compatibility with Time-MMD datasets while adding support for the `text` column used in TTC datasets.

#### Modification 2: Verify Text Column Exclusion

**File**: `data_provider/time_mmd_dataset.py`

**Change**: Ensure `text` column is properly excluded from time series data in `__read_data__()`.

Current code already handles this correctly by excluding any column matching patterns or explicitly specified text columns. Just need to ensure `text` is in the exclude list when found.

**Verification**: The existing code at lines 623-627 already handles this:
```python
# Exclude ALL Final_Search_* and Final_Output columns
for col in df_raw.columns:
    if re.match(r'Final_Search_\d+', col) or col == 'Final_Output':
        if col not in exclude_cols:
            exclude_cols.append(col)
```

We should add `text` to this exclusion logic:
```python
# Exclude text columns (Time-MMD and TTC formats)
for col in df_raw.columns:
    if (re.match(r'Final_Search_\d+', col) or 
        col == 'Final_Output' or 
        col == 'text'):
        if col not in exclude_cols:
            exclude_cols.append(col)
```

#### Modification 3: Create Helper Script for Medical id_info.json

**File**: `scripts/generate_ttc_medical_id_info.py`

```python
"""
Generate id_info.json for TTC medical dataset by scanning for patient_*.csv files.
"""
import os
import json
import glob
import re
from pathlib import Path

def generate_medical_id_info(root_path='./data/ttc/medical', output_file='id_info.json'):
    """
    Generate id_info.json by scanning for patient_*.csv files.
    
    Args:
        root_path: Root directory containing patient CSV files
        output_file: Output filename (relative to root_path)
    """
    root_path = Path(root_path)
    id_info = {}
    
    # Find all patient_*.csv files
    pattern = str(root_path / 'patient_*.csv')
    patient_files = glob.glob(pattern)
    
    # Extract patient IDs from filenames
    for file_path in sorted(patient_files):
        filename = os.path.basename(file_path)
        match = re.match(r'patient_(\d+)\.csv', filename)
        if match:
            patient_id = match.group(1)
            id_info[patient_id] = {
                'description': f'Patient {patient_id} medical data'
            }
    
    # Write id_info.json
    output_path = root_path / output_file
    with open(output_path, 'w') as f:
        json.dump(id_info, f, indent=2)
    
    print(f'[ info ] Generated id_info.json with {len(id_info)} patients')
    print(f'[ info ] Output: {output_path}')
    
    return id_info

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate id_info.json for TTC medical dataset')
    parser.add_argument('--root_path', type=str, default='./data/ttc/medical',
                       help='Root directory containing patient CSV files')
    parser.add_argument('--output', type=str, default='id_info.json',
                       help='Output filename (relative to root_path)')
    args = parser.parse_args()
    
    generate_medical_id_info(args.root_path, args.output)
```

## Testing Plan

### Test 1: Climate Dataset (Single Entity)

1. **Setup**:
   - Place `climate_2014_2023_final.csv` in `data/ttc/climate/`
   - Create `data_configs/ttc/climate/config.yaml` with config above
   - Create minimal `id_info.json` or rely on auto-creation

2. **Test**:
   ```python
   # Quick test script
   from data_provider.data_factory import Data_Provider
   from utils.tools import dotdict
   import yaml
   
   # Load config
   with open('data_configs/ttc/climate/config.yaml') as f:
       config = yaml.safe_load(f)
   
   args = dotdict({
       'data_config': dotdict(config),
       'batch_size': 32,
       'input_len': 96,
       'output_len': 24,
       'scale': True,
       'preload_hetero': False,
       'num_workers': 0,
       'prefetch_factor': 2,
       'model_config': dotdict({
           'task': 'MTSF',
           'stride': 1,
           'hetero_align_stride': False
       })
   })
   
   # Create data provider
   data_provider = Data_Provider(args, buffer=False)
   
   # Test loading
   train_loader = data_provider.get_train(return_type='loader')
   val_loader = data_provider.get_val(return_type='loader')
   test_loader = data_provider.get_test(return_type='loader')
   
   # Get a sample
   batch = next(iter(train_loader))
   print(f"Batch keys: {batch.keys() if isinstance(batch, dict) else 'Not a dict'}")
   print(f"Sample shape: {batch[0].shape if isinstance(batch, tuple) else 'Check structure'}")
   ```

### Test 2: Medical Dataset (Multi-Entity)

1. **Setup**:
   - Place patient CSV files in `data/ttc/medical/`
   - Generate `id_info.json` using helper script
   - Create `data_configs/ttc/medical/config.yaml`

2. **Test**: Similar to Test 1, but verify multiple entities are loaded correctly

### Test 3: Verify Text Column Handling

1. **Test with text column**:
   - Ensure text is extracted correctly
   - Ensure text is excluded from time series data
   - Verify text is accessible via hetero_data_getter

2. **Test with missing text column**:
   - Dataset should work as time-series-only (no errors)

## Implementation Checklist

- [ ] **Phase 1: Code Modifications**
  - [ ] Update `_detect_text_column()` to handle `text` column
  - [ ] Update column exclusion logic to include `text` column
  - [ ] Test backward compatibility with Time-MMD datasets

- [ ] **Phase 2: Configuration Files**
  - [ ] Create `data_configs/ttc/climate/config.yaml`
  - [ ] Create `data_configs/ttc/medical/config.yaml`
  - [ ] Create `data/ttc/climate/id_info.json` (or verify auto-creation)
  - [ ] Create helper script `scripts/generate_ttc_medical_id_info.py`
  - [ ] Generate `data/ttc/medical/id_info.json` using helper script

- [ ] **Phase 3: Testing**
  - [ ] Test climate dataset loading (single entity)
  - [ ] Test medical dataset loading (multi-entity)
  - [ ] Test text column extraction and exclusion
  - [ ] Test with different task types (TSF, TGTSF, MTSF)
  - [ ] Test with both text and embedding output formats
  - [ ] Verify backward compatibility with existing Time-MMD datasets

- [ ] **Phase 4: Documentation**
  - [ ] Update `docs/time_mmd.md` to mention TTC datasets
  - [ ] Add TTC-specific notes to data config README
  - [ ] Document the `text` column format option

## Alternative Approaches Considered

### Alternative 1: Create Separate TTC_Dataset Class
**Pros**: Clear separation, no risk of breaking Time-MMD
**Cons**: Code duplication, maintenance overhead
**Decision**: Not chosen - TTC format is very similar to Time-MMD, just simpler

### Alternative 2: Use Standard Universal_Dataset + Heterogeneous_Dataset
**Pros**: Uses existing infrastructure
**Cons**: Requires splitting CSV files into separate time series and text files, more complex setup
**Decision**: Not chosen - Time-MMD approach is cleaner for embedded text columns

## Notes

1. **Target Column**: Both datasets should use `target: all` to include all numerical columns as features/targets.

2. **Text Column Name**: The explicit `text_column: text` specification ensures we use the correct column. The auto-detection fallback to `text` provides convenience.

3. **Multi-Entity Medical Dataset**: The formatter `patient_{i}.csv` pattern allows Data_Provider to load multiple files, similar to standard fidel-ts multi-entity datasets.

4. **Backward Compatibility**: All changes maintain backward compatibility with existing Time-MMD datasets that use `Final_Search_*` or `Final_Output` columns.
