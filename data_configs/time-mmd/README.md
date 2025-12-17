# Time-MMD Dataset Configuration

This directory contains configuration files for Time-MMD datasets (from MM-TSFlib).

## Dataset Structure

Time-MMD datasets should be placed in `./data/time-mmd/{subdataset}/` with the CSV file directly in that directory.

Example structure:
```
data/
└── time-mmd/
    ├── Traffic/
    │   └── US_VMT_Month.csv
    ├── Public_Health/
    │   └── US_FLURATIO_Week.csv
    └── Energy/
        └── US_GasolinePrice_Week.csv
```

## Configuration Files

Each subdataset should have a `config.yaml` file in `data_configs/time-mmd/{subdataset}/`.

### Required Fields

- `root_path`: Path to dataset directory (e.g., `./data/time-mmd/Traffic`)
- `data_path`: CSV filename (e.g., `US_VMT_Month.csv`)
- `dataset_type: time_mmd`: **Required** flag to use TimeMMD_Dataset
- `timestamp_col`: Name of timestamp column (usually `date`)
- `target`: Target column name (usually `OT`)
- `spliter`: Data splitting method (`ratio` recommended)
- `split_info`: Split ratios `[7, 1, 2]` for 70% train, 10% val, 20% test

### Text Column Configuration

- `text_column`: Text column name or `'auto'` to auto-detect
- `use_closedllm`: `false` to use `Final_Search_*`, `true` to use `Final_Output`
- `text_len`: Text length for `Final_Search_{text_len}` pattern (default: 4)

### Optional Fields

- `general_info`: Dataset description string (default: empty string)
- `channel_info`: Channel-specific description (default: empty string)
- `time_zone`: Timezone for timestamp conversion (default: `null`)
- `downsample`: Downsampling factor (default: `null`)

## Example Usage

After setting up the data and config files, use the dataset in your experiment config:

```yaml
data_config: data_configs/time-mmd/Traffic/config.yaml
```

The dataset will automatically:
1. Load the CSV file
2. Extract time series data (normalized)
3. Extract text data from CSV columns
4. Provide text via fidel-ts's hetero_data_getter interface
5. Return data in fidel-ts expected format

## Text Format

Text data is provided in the format specified by `hetero_info.input_format` in your model config:
- `json`: List of JSON strings `[{"text": "..."}]`
- `dict`: List of dictionaries `[{"text": "..."}]`
- `csv`: CSV string format
- `embedding`: Zero array (embeddings should be pre-computed)

## Notes

- Text is aligned to sequence end point (s_end) using backward matching
- Empty strings are returned for `hetero_general` and `hetero_channel` if not specified
- The dataset works as time-series-only if no text column is found

