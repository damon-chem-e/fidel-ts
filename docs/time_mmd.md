# Time-MMD Dataset Integration

This document explains how Time-MMD datasets from MM-TSFlib are integrated into fidel-ts, including data format, loading mechanism, compatibility layer, and testing procedures.

## Data Format on Disk

Time-MMD datasets are stored as CSV files with the following structure:

### File Location
```
data/time_mmd/{subdataset}/{filename}.csv
```

Example: `data/time_mmd/Traffic/US_VMT_Month.csv`

### CSV Structure

Each CSV file contains:
1. **Timestamp column**: Typically named `date` (configurable via `timestamp_col`)
2. **Time series column(s)**: Numerical time series data, with target column typically named `OT`
3. **Text columns**: One or more text columns containing search results or LLM outputs:
   - `Final_Search_{text_len}`: Search results with specified text length (e.g., `Final_Search_4`)
   - `Final_Output`: LLM-generated text (when using closed-source LLMs)
4. **Metadata columns** (optional): `start_date`, `end_date`, `prior_history_avg`, `prior_history_std`

### Example CSV Structure
```csv
date,OT,Final_Search_4,start_date,end_date
1980-01-01,116462.0,"Available facts are as follows: 1980-01-21: NA...",1980-01-01,1980-12-31
1980-02-01,107338.0,"Available facts are as follows: 1980-02-15: The United...",1980-01-01,1980-12-31
...
```

## How Data is Loaded

### 1. Dataset Detection

The system detects Time-MMD datasets via the `dataset_type: time_mmd` flag in the data config:

```yaml
dataset_type: time_mmd  # Flag to use TimeMMD_Dataset
```

### 2. Data Loading Process

**Step 1: CSV Loading**
- CSV file is loaded using pandas
- Timestamp column is converted to datetime and then to int64 format (YYYYMMDDHHMMSS)

**Step 2: Text Column Detection**
- Auto-detection: Looks for `Final_Search_{text_len}` or `Final_Output` columns
- Manual specification: Can specify exact column name via `text_column` config
- If no text column found: Dataset works as time-series-only (no errors)

**Step 3: Data Extraction**
- Time series data: All columns except timestamp, text columns, and metadata
- Text data: Extracted and indexed by timestamp for fast lookup
- Metadata: Excluded from time series but available for reference

**Step 4: Data Splitting**
- Uses configured splitter (typically `ratio` with `[7, 1, 2]` for 70% train, 10% val, 20% test)
- Splits are applied to both time series and text data

**Step 5: Normalization**
- If `scale: true`, applies StandardScaler to training data
- Handles both multi-column (`target: all`) and single-column (`target: OT`) cases

**Step 6: Downsampling** (if configured)
- Applies downsampling to time series data
- Text data is reindexed using forward fill (no lookahead bias)

### 3. Text Data Getter

The `TimeMMD_HeteroGetter` class provides text data on-demand:

- **Timestamp Matching**: Uses backward matching to find the closest text at or before each timestamp
- **Output Formats**: Supports `json`, `dict`, `csv`, or `embedding` formats
- **Empty Handling**: Returns empty strings/arrays when text is not available

## Porting to fidel-ts Compatibility

### Architecture

Time-MMD datasets are integrated via an adapter pattern:

```
TimeMMD_Dataset (inherits from Universal_Dataset)
    ↓
TimeMMD_HeteroGetter (implements hetero_data_getter interface)
    ↓
fidel-ts models (no changes needed)
```

### Key Components

1. **`TimeMMD_Dataset`** (`data_provider/time_mmd_dataset.py`)
   - Inherits from `Universal_Dataset`
   - Overrides `__read_data__()` to handle CSV format
   - Sets up `TimeMMD_HeteroGetter` as `hetero_data_getter`

2. **`TimeMMD_HeteroGetter`**
   - Implements fidel-ts's `hetero_data_getter` interface
   - Converts MM-TSFlib format (single text at sequence end) to fidel-ts format (text per timestamp)
   - Handles timestamp matching and format conversion

3. **`Data_Provider`** (`data_provider/data_factory.py`)
   - Detects `dataset_type: time_mmd` in config
   - Instantiates `TimeMMD_Dataset` instead of `Universal_Dataset`
   - Creates minimal `id_info.json` if not present

### Data Format Compatibility

Time-MMD datasets return data in the exact same format as standard fidel-ts datasets:

```python
(sample_id, seq_x, seq_y, x_time, y_time, 
 x_hetero, y_hetero, hetero_x_time, hetero_y_time, 
 hetero_general, hetero_channel)
```

Where:
- `seq_x`, `seq_y`: Time series sequences (normalized if `scale: true`)
- `x_time`, `y_time`: Timestamps (int64 YYYYMMDDHHMMSS format)
- `x_hetero`, `y_hetero`: Text data (format depends on `output_format`)
- `hetero_general`, `hetero_channel`: Static info (from config, typically empty strings)

### Model Compatibility

All fidel-ts models work without modification:
- **TSF models**: Use time series only (text is available but ignored)
- **TGTSF models**: Use both time series and text data
- **MTSF models**: Multi-channel support (if multiple time series columns)
- **Reasoning models**: Can use text for reasoning tasks

## Testing

### Quick Test Script

Use the test script to verify dataset loading:

```bash
python tests/test_time_mmd_dataset.py data_configs/time_mmd/Traffic/config.yaml
```

This tests:
- Dataset loading and structure
- Text column detection
- Data shapes and formats
- Text retrieval functionality
- DataLoader creation

### Training Test

Run a single epoch to test end-to-end:

```bash
python -m cli.train pytorch configs/experiments/time_mmd_test.yaml
```

### What to Verify

1. **Dataset Loading**: No errors, correct data shapes
2. **Text Detection**: Text column found (or graceful fallback)
3. **Data Shapes**: 
   - `seq_x`: `(batch_size, seq_len, features)`
   - `seq_y`: `(batch_size, pred_len, features)`
4. **Text Retrieval**: Text data available via `hetero_data_getter`
5. **Training**: Model forward pass succeeds

### Testing Text Embeddings

See the section below on "Testing Text Embeddings" for details on verifying that text is properly embedded vs. returning zero embeddings.

## Configuration

### Data Config Structure

```yaml
root_path: ./data/time_mmd/Traffic
data_path: US_VMT_Month.csv
spliter: ratio
split_info: [7, 1, 2]
timestamp_col: date
target: OT
id: all
formatter: US_VMT_Month.csv
id_info: id_info.json
dataset_type: time_mmd
text_column: auto
use_closedllm: false
text_len: 4
general_info: "Dataset description"
channel_info: ""
time_zone: null
downsample: null
sampling_rate: 1month
base_T: 12
```

### Key Parameters

- `dataset_type: time_mmd`: Required flag to use Time-MMD dataset
- `text_column`: `auto` (detect) or specific column name
- `use_closedllm`: `true` if using `Final_Output`, `false` for `Final_Search_*`
- `text_len`: Text length for `Final_Search_{text_len}` detection
- `general_info`: Static general information (empty string if not provided)
- `channel_info`: Static channel information (empty string for single-entity datasets)

## Testing Text Embeddings

### Understanding Embedding Modes

Time-MMD datasets support two text handling modes:

1. **Raw Text Mode** (default): Text is returned as JSON/dict strings
   - Used by models that handle text directly (ChatTime, TGTSF, etc.)
   - No embeddings needed
   - Output format: `json` or `dict`

2. **Embedding Mode**: Text is converted to embeddings
   - Used when `output_format='embedding'` or `postemb` is enabled
   - Requires embedding model configuration
   - Output format: torch tensors

### Current Limitations

- **`output_format='embedding'`**: Returns zero arrays (expects pre-computed embeddings from files)
- **`postemb`**: Not yet supported (embeddings created during initialization)

### How to Test Text Embeddings

#### Method 1: Inspect Output Format

Check the `output_dynamic` from `hetero_data_getter`:

```python
matched_times, general_info, channel_info, output_dynamic = dataset.hetero_data_getter(timestamps)

# For raw text mode:
if isinstance(output_dynamic, list):
    # Should contain text strings, not zeros
    print(f"Text sample: {output_dynamic[0]}")
    
# For embedding mode:
if isinstance(output_dynamic, torch.Tensor):
    # Check if embeddings are non-zero
    print(f"Embedding shape: {output_dynamic.shape}")
    print(f"Non-zero count: {(output_dynamic != 0).sum()}")
    if (output_dynamic == 0).all():
        print("WARNING: All embeddings are zero!")
```

#### Method 2: Use Test Script

The test script (`tests/test_time_mmd_dataset.py`) shows:
- Text data format
- Sample text content
- Whether text is available

#### Method 3: Model Training

If using a text-guided model (TGTSF, ChatTime):
- Model should use text during forward pass
- Check model logs/attention weights to verify text is being used
- Compare performance with/without text

#### Method 4: Debug in Dataset

Add debug prints in `TimeMMD_HeteroGetter.__call__()`:

```python
# In time_mmd_dataset.py, TimeMMD_HeteroGetter.__call__()
if self.output_format == 'embedding':
    # Check if embeddings are being generated
    print(f"Generating embeddings for {len(matched_texts)} texts")
    embeddings = self._get_embeddings(matched_texts)
    print(f"Embedding stats: min={embeddings.min()}, max={embeddings.max()}, mean={embeddings.mean()}")
```

### Expected Behavior

**Raw Text Mode** (`output_format='json'` or `'dict'`):
- `output_dynamic` is a list of strings (JSON/dict formatted)
- Text content should be visible in the strings
- Empty strings `""` are OK (means no text for that timestamp)

**Embedding Mode** (`output_format='embedding'`):
- Currently returns zero arrays (not yet implemented)
- Future: Should return actual embeddings when `postemb` is supported

### Troubleshooting

**Issue**: All text is empty strings
- **Cause**: CSV file doesn't have text data for those timestamps
- **Solution**: Check CSV file, verify text column exists and has data

**Issue**: Zero embeddings when `output_format='embedding'`
- **Cause**: Expected behavior (not yet implemented)
- **Solution**: Use `output_format='json'` for now, or implement `postemb` support

**Issue**: Text not being used by model
- **Cause**: Model might not be configured to use text, or task is TSF (time-series-only)
- **Solution**: Use TGTSF task or text-guided model, check model config

