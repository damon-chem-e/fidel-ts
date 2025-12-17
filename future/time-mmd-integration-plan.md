# Time-MMD Dataset Integration Plan

## Overview

This document outlines the plan to integrate MM-TSFlib datasets (Time-MMD) into fidel-ts, enabling training and evaluation on these multimodal time series datasets without modifying the rest of the fidel-ts codebase.

## Key Differences Between MM-TSFlib and fidel-ts Data Providers

### MM-TSFlib (`Dataset_Custom`)
- **Data Format**: Single CSV file containing:
  - Time series columns (target + features)
  - Text column: `Final_Search_{text_len}` or `Final_Output`
  - Metadata columns: `date`, `start_date`, `end_date`, `prior_history_avg`
- **Text Handling**: 
  - Text stored as raw strings in CSV column
  - Retrieved via `get_text(index)` method at sequence end point (`s_end`)
  - One text string per sample
- **Return Format**: `(seq_x, seq_y, seq_x_mark, seq_y_mark, index)`
- **Text Processing**: Done externally (Doc2Vec, LLM embeddings) before model forward pass

### fidel-ts (`Universal_Dataset`)
- **Data Format**: 
  - Time series: Separate CSV/parquet files
  - Text: Separate JSON/CSV files (via `Heterogeneous_Dataset`)
- **Text Handling**:
  - Text stored in separate files with timestamps
  - Retrieved via `hetero_data_getter(timestamps)` function
  - Multiple texts per sequence (aligned to timestamps)
- **Return Format**: `(sample_id, seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel)`
- **Text Processing**: Can be done via `Heterogeneous_Dataset` with post-embedding support

## Integration Strategy

### Approach: Adapter Pattern with Minimal Code Duplication

Create a new dataset class `TimeMMD_Dataset` that:
1. Inherits from or wraps `Universal_Dataset` to maintain compatibility
2. Handles MM-TSFlib CSV format internally
3. Converts MM-TSFlib data format to fidel-ts expected format
4. Provides text data via `hetero_data_getter` interface

## Implementation Plan

### Phase 1: Create Time-MMD Dataset Adapter

#### 1.1 Create `TimeMMD_Dataset` Class
**Location**: `data_provider/time_mmd_dataset.py`

**Responsibilities**:
- Load MM-TSFlib CSV format (single file with all columns)
- Extract time series data (normalized)
- Extract text data from CSV column
- Provide text via `hetero_data_getter` compatible interface
- Return data in fidel-ts format

**Key Design Decisions**:
- Inherit from `Universal_Dataset` where possible, or compose it
- Store text data in memory (aligned with timestamps)
- Create a `hetero_data_getter` function that returns text in fidel-ts format
- Handle text column name detection (`Final_Search_*` vs `Final_Output`)

#### 1.2 Text Data Format Conversion

**MM-TSFlib Format**:
- Text at sequence end point (`s_end`)
- Single text string per sample
- Raw string format

**fidel-ts Format**:
- Text aligned to timestamps
- Returns tuple: `(matched_times, general_info, channel_info, output_dynamic)`
- `output_dynamic` can be: list of strings (json format), embeddings, or dicts

**Conversion Strategy**:
- Create a `TimeMMD_HeteroGetter` class that:
  - Stores text data indexed by timestamp
  - Implements `get_hetero_data(timestamps)` method
  - Returns text aligned to requested timestamps
  - Uses 'nearest' or 'backward' matching (since text is at sequence end)
  - Returns format compatible with fidel-ts expectations

#### 1.3 Data Splitting

**MM-TSFlib**: Uses fixed ratio split (70% train, 10% val, 20% test)
**fidel-ts**: Supports both ratio and timestamp-based splitting

**Solution**: Use fidel-ts `ratio_spliter` with (7, 1, 2) ratios, matching MM-TSFlib behavior.

### Phase 2: Create Data Configuration Support

#### 2.1 Data Config Structure
**Location**: `data_configs/time-mmd/{subdataset}/config.yaml`

**Required Fields**:
```yaml
root_path: ./data/time-mmd/{subdataset}
data_path: {subdataset_file}.csv  # e.g., US_VMT_Month.csv
spliter: ratio
split_info: [7, 1, 2]  # Match MM-TSFlib default
timestamp_col: date
target: OT  # or appropriate target column name
id: all  # or specific IDs if multi-entity
formatter: '{data_path}'  # Single file format
dataset_type: time_mmd  # Flag to use TimeMMD_Dataset
text_column: auto  # Auto-detect Final_Search_* or Final_Output
use_closedllm: false  # Whether to use Final_Output column
text_len: 4  # For Final_Search_{text_len} detection
```

#### 2.2 Dataset Type Detection
**Location**: `data_provider/data_factory.py`

**Modification**: Add logic to detect `dataset_type: time_mmd` and instantiate `TimeMMD_Dataset` instead of `Universal_Dataset`.

### Phase 3: Text Format Handling

#### 3.1 Text Output Format Options

Support multiple text output formats to match fidel-ts expectations:

1. **JSON Format** (default for compatibility):
   - Convert text string to JSON: `{"text": "<text_content>"}`
   - Return as list of JSON strings

2. **Dict Format**:
   - Return as list of dictionaries: `[{"text": "<text_content>"}]`

3. **Embedding Format** (if post-embedding enabled):
   - Use `Heterogeneous_Dataset` post-embedding capabilities
   - Pre-compute embeddings during initialization

#### 3.2 Static Information

**MM-TSFlib**: No explicit static info
**fidel-ts**: Expects `hetero_general` and `hetero_channel`

**Solution**: 
- `hetero_general`: Dataset description (e.g., "US Vehicle Miles Traveled monthly data")
- `hetero_channel`: Empty string or dataset-specific description
- Can be configured in data config or auto-generated from dataset name

**Empty Value Handling**:

**Current fidel-ts Behavior**:
- When no hetero data is requested (not in `custom_input`): `hetero_general` and `hetero_channel` default to `np.zeros((1), dtype=np.float32)` in `Universal_Dataset.__getitem__()`
- When hetero data IS requested via `hetero_data_getter`: They come from `static_data` as **strings** (regardless of `output_format`)
- The `output_format` parameter only affects `x_hetero`/`y_hetero`, not `hetero_general`/`hetero_channel`

**For Time-MMD Integration**:
- **Default approach**: Return empty strings `""` when no static info is available
  - Compatible with string-concatenating models (e.g., `ChatTime`): `"" + "" + batch_y_hetero` = `batch_y_hetero`
  - Compatible with `output_format == 'embedding'`: `hetero_general`/`hetero_channel` remain strings even when `x_hetero`/`y_hetero` are embeddings
  - Matches behavior when `static_path` is None in `Heterogeneous_Dataset` (provides default string values)

**Optional: Empty Embeddings Support**:
- If models require embeddings for `hetero_general`/`hetero_channel` (not currently the case in fidel-ts), we can:
  1. Detect when `output_format == 'embedding'` AND static info is empty
  2. Return zero tensors: `np.zeros((embedding_dim,), dtype=np.float32)` or `torch.zeros(embedding_dim)`
  3. Shape should match expected embedding dimension (typically from model config or `postemb_d`)
- **Note**: This is not currently needed since fidel-ts models treat `hetero_general`/`hetero_channel` as strings/metadata, not embeddings

### Phase 4: Integration Points

#### 4.1 Data Factory Integration
**File**: `data_provider/data_factory.py`

**Changes**:
```python
def get_datasets(self, flag):
    # ... existing code ...
    if self.dataset_config.get('dataset_type') == 'time_mmd':
        from data_provider.time_mmd_dataset import TimeMMD_Dataset
        dataset = TimeMMD_Dataset(...)
    else:
        dataset = Universal_Dataset(...)
```

#### 4.2 Heterogeneous Data Getter
**File**: `data_provider/time_mmd_dataset.py`

**Implementation**:
- Create `TimeMMD_HeteroGetter` class
- Implements `get_hetero_data(timestamps)` method
- Returns tuple: `(matched_times, general_info, channel_info, output_dynamic)`
- Handles timestamp matching (nearest/backward to sequence end)

### Phase 5: File Structure

```
fidel-ts/
├── data/
│   └── time-mmd/
│       ├── Agriculture/
│       │   └── US_RetailBroilerComposite_Month.csv
│       ├── Climate/
│       │   └── US_precipitation_month.csv
│       ├── Economy/
│       │   └── US_TradeBalance_Month.csv
│       ├── Energy/
│       │   └── US_GasolinePrice_Week.csv
│       ├── Public_Health/
│       │   └── US_FLURATIO_Week.csv
│       ├── Security/
│       │   └── US_FEMAGrant_Month.csv
│       ├── SocialGood/
│       │   └── Unadj_UnemploymentRate_ALL_processed.csv
│       └── Traffic/
│           └── US_VMT_Month.csv
├── data_configs/
│   └── time-mmd/
│       ├── Agriculture/
│       │   └── config.yaml
│       ├── Climate/
│       │   └── config.yaml
│       ├── Economy/
│       │   └── config.yaml
│       ├── Energy/
│       │   └── config.yaml
│       ├── Public_Health/
│       │   └── config.yaml
│       ├── Security/
│       │   └── config.yaml
│       ├── SocialGood/
│       │   └── config.yaml
│       └── Traffic/
│           └── config.yaml
└── data_provider/
    ├── time_mmd_dataset.py  # NEW
    ├── time_mmd_hetero_getter.py  # NEW (optional, can be in same file)
    ├── data_loader.py  # Existing
    └── data_factory.py  # Modified
```

## Detailed Implementation Steps

### Step 1: Create `TimeMMD_Dataset` Class

**File**: `data_provider/time_mmd_dataset.py`

**Key Methods**:
1. `__init__()`: 
   - Load CSV file
   - Detect text column name
   - Extract time series and text data
   - Initialize text getter
   - Call parent `Universal_Dataset.__init__()` with text getter

2. `_detect_text_column()`:
   - Check for `Final_Output` column (if `use_closedllm=True`)
   - Otherwise, check for `Final_Search_{text_len}` pattern
   - Return column name or None

3. `_extract_text_data()`:
   - Extract text column from CSV
   - Align with timestamps
   - Store in format accessible by text getter

**Inheritance Strategy**:
- Option A: Inherit from `Universal_Dataset` and override `__read_data__()` and `__init__()`
- Option B: Compose `Universal_Dataset` and delegate most operations
- **Recommendation**: Option A (inheritance) for cleaner integration

### Step 2: Create `TimeMMD_HeteroGetter` Class

**File**: `data_provider/time_mmd_dataset.py` (same file)

**Key Methods**:
1. `__init__(text_data, timestamps, general_info, channel_info, output_format)`:
   - Store text data indexed by timestamp
   - Store static info
   - Set output format

2. `get_hetero_data(timestamps)`:
   - Match timestamps to text data (nearest/backward matching)
   - Format text according to `output_format`
   - Return tuple: `(matched_times, general_info, channel_info, output_dynamic)`

**Matching Strategy**:
- Since MM-TSFlib text is at sequence end point, use 'backward' matching
- For each requested timestamp, find the text at or before that timestamp
- This ensures text describes context up to that point

### Step 3: Modify Data Factory

**File**: `data_provider/data_factory.py`

**Changes**:
```python
def get_datasets(self, flag):
    datasets = {}
    # ... existing progress bar code ...
    for i in self.id_list:
        # Check if this is a Time-MMD dataset
        if self.dataset_config.get('dataset_type') == 'time_mmd':
            from data_provider.time_mmd_dataset import TimeMMD_Dataset
            dataset = TimeMMD_Dataset(
                root_path=self.dataset_config.root_path,
                data_path=self.formatter.format(i=i) if '{i}' in self.formatter else self.dataset_config.data_path,
                flag=flag,
                seq_len=self.args.input_len,
                pred_len=self.args.output_len,
                spliter=self.spliter,
                timestamp_col=self.dataset_config.timestamp_col,
                target=self.dataset_config.target,
                scale=self.args.scale,
                text_column=self.dataset_config.get('text_column', 'auto'),
                use_closedllm=self.dataset_config.get('use_closedllm', False),
                text_len=self.dataset_config.get('text_len', 4),
                output_format=self.args.data_config.hetero_info.input_format if self.args.data_config.hetero_info else 'json',
                general_info=self.dataset_config.get('general_info', ''),
                channel_info=self.dataset_config.get('channel_info', ''),
                task=self.args.model_config.task,
                custom_input=self.args.model_config.custom_input,
                timezone=self.dataset_config.time_zone,
                downsample=self.dataset_config.downsample,
                entity_id=i
            )
        else:
            # Existing Universal_Dataset code
            # ...
        datasets[i] = dataset
    return datasets
```

### Step 4: Create Data Configurations

**Template**: `data_configs/time-mmd/{subdataset}/config.yaml`

**Example for Traffic dataset**:
```yaml
root_path: ./data/time-mmd/Traffic
data_path: US_VMT_Month.csv
spliter: ratio
split_info: [7, 1, 2]
timestamp_col: date
target: OT
id: all
formatter: US_VMT_Month.csv
dataset_type: time_mmd
text_column: auto
use_closedllm: false
text_len: 4
general_info: "US Vehicle Miles Traveled monthly time series data"
channel_info: ""
time_zone: null
downsample: null
```

### Step 5: Handle Edge Cases

#### 5.1 Missing Text Column
- If text column not found, set `hetero_data_getter=None`
- Dataset works as time-series-only (TSF task)

#### 5.2 Empty Text Values
- Handle NaN/empty strings in text column
- Return empty string or placeholder text

#### 5.3 Timestamp Mismatch
- Ensure text timestamps match time series timestamps
- Use sequence end point for text alignment

#### 5.4 Multiple Features
- Support both 'S' (univariate) and 'M'/'MS' (multivariate) modes
- Extract appropriate columns based on `target` config

## Testing Strategy

### Unit Tests
1. Test `TimeMMD_Dataset` initialization with various CSV formats
2. Test text column detection logic
3. Test text getter matching strategies
4. Test data splitting (ratio-based)

### Integration Tests
1. Test end-to-end data loading with fidel-ts models
2. Test with different tasks (TSF, TGTSF, MTSF)
3. Test with different text output formats (json, dict, embedding)

### Validation Tests
1. Compare loaded data with MM-TSFlib original behavior
2. Verify text alignment correctness
3. Verify normalization matches MM-TSFlib behavior

## Code Reuse and Modularity

### Shared Components
1. **Data Splitting**: Reuse `ratio_spliter` from `data_helper.py`
2. **Normalization**: Reuse `StandardScaler` approach from `Universal_Dataset`
3. **Timestamp Handling**: Reuse timestamp conversion logic
4. **Text Formatting**: Reuse text-to-json/dict conversion patterns from `Heterogeneous_Dataset`

### Avoid Duplication
1. **Don't duplicate**: CSV reading, normalization, splitting logic
2. **Do create**: Text getter adapter, text column detection
3. **Do reuse**: Existing `Universal_Dataset` infrastructure where possible

## Migration Path

### For Existing MM-TSFlib Users
1. Copy CSV files to `fidel-ts/data/time-mmd/{subdataset}/`
2. Create data config YAML file
3. Use fidel-ts training/evaluation scripts with new config

### Backward Compatibility
- No changes to existing fidel-ts datasets
- `dataset_type` field is optional (defaults to standard behavior)
- Existing code continues to work unchanged

## Future Enhancements

1. **Support for Prior History**: MM-TSFlib has `prior_history_avg` column - could be used as additional feature
2. **Support for Multiple Text Columns**: If datasets evolve to have multiple text sources
3. **Automatic Config Generation**: Script to auto-generate configs from CSV headers
4. **Text Embedding Pre-computation**: Option to pre-compute embeddings during dataset initialization

## Summary

This plan provides a modular, non-invasive way to integrate MM-TSFlib datasets into fidel-ts by:
1. Creating a new `TimeMMD_Dataset` class that adapts MM-TSFlib format to fidel-ts format
2. Using adapter pattern for text data (via `hetero_data_getter`)
3. Maintaining compatibility with existing fidel-ts code
4. Minimizing code duplication through inheritance and composition
5. Providing clear configuration interface via YAML files

The implementation maintains separation of concerns and allows gradual migration while preserving all existing functionality.

