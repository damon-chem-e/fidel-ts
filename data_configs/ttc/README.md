# TTC Dataset Configurations

This directory contains configuration files for TTC (Time Text Corpus) datasets integrated into fidel-ts.

## Datasets

### Climate Dataset
- **Data Location**: `data/ttc/climate/climate_2014_2023_final.csv`
- **Config**: `climate/config.yaml`
- **Description**: Weather time series data with temperature, precipitation, humidity, windspeed, and text descriptions
- **Type**: Single entity dataset

### Medical Dataset
- **Data Location**: `data/ttc/medical/patient_{number}.csv`
- **Config**: `medical/config.yaml`
- **Description**: Medical patient data with health metrics (Respiratory_Rate, Heart_Rate, SaO2, FiO2) and text notes
- **Type**: Multi-entity dataset (one CSV per patient)

## Configuration Notes

### Text Column
Both datasets use a `text` column for embedded text descriptions. This is specified explicitly in the config:
```yaml
text_column: text  # Explicitly specify text column (TTC format)
```

### Target Columns
Both datasets use `target: all` to include all numerical columns:
- **Climate**: temp, precip, humidity, windspeed
- **Medical**: Respiratory_Rate, Heart_Rate, SaO2, FiO2

### Text Embedding
For multimodal models (TGTSF, LYNX), text is converted to embeddings:
```yaml
timemmd_text_output: embedding  # Use embeddings for TGTSF/LYNX
timemmd_embed_model: bert-base-uncased
timemmd_embed_dim: 768
```

### Medical Dataset id_info.json
The medical dataset requires an `id_info.json` file listing all patient IDs. Use the helper script to generate it:
```bash
python scripts/generate_ttc_medical_id_info.py --root_path ./data/ttc/medical
```

Or create it manually following the pattern in `docs/ttc_data_integration_plan.md`.

## Usage

These configs are designed to work with `dataset_type: time_mmd`, which uses the `TimeMMD_Dataset` class that has been extended to support TTC format datasets.

## Example Experiment Suite

See `configs/experiment_suites/ttc_test.yaml` for example experiments using these datasets.
