# Experiment Suites

This directory contains experiment suite configurations that replace the need for bash scripts. Experiment suites allow you to define collections of related experiments in YAML format, making it easy to run multiple experiments with consistent configurations.

## Usage

### Running a Suite

```bash
# Run all experiments in a suite
python -m cli.suite run configs/experiment_suites/linear_models.yaml

# Run with filtering (only experiments matching pattern)
python -m cli.suite run configs/experiment_suites/linear_models.yaml --filter dlinear

# Dry run (validate without executing)
python -m cli.suite run configs/experiment_suites/linear_models.yaml --dry-run
```

### Listing Available Suites

```bash
python -m cli.suite list
```

### Validating a Suite Config

```bash
python -m cli.suite validate configs/experiment_suites/linear_models.yaml
```

### Getting Suite Information

```bash
python -m cli.suite info configs/experiment_suites/linear_models.yaml
```

## Suite Configuration Structure

A suite configuration file has the following structure:

```yaml
suite:
  name: "suite_name"
  description: "Description of what this suite tests"
  tags: ["tag1", "tag2"]
  
  execution:
    parallel: false  # Run experiments sequentially
    continue_on_error: true  # Continue if one experiment fails
    log_dir: "./logs/suites/suite_name"
    
  experiments:
    - name: "experiment_name"
      description: "Description of this experiment"
      enabled: true  # Set to false to skip this experiment
      template: "configs/templates/pytorch_training.yaml"
      overrides:
        model:
          name: "DLinear"
          config_path: "model_configs/general/DLinear.yaml"
        data:
          name: "NYC_traffic_speed"
          config_path: "data_configs/NYC_traffic_speed/fullNYCTS_H.yaml"
        training:
          input_len: 360
          output_lens: [24, 168, 336, 720]  # Multiple output lengths
          batch_size: 512
```

## Available Templates

- `configs/templates/pytorch_training.yaml` - For PyTorch-based training
- `configs/templates/lightning_training.yaml` - For PyTorch Lightning training
- `configs/templates/llm_testing.yaml` - For LLM model testing
- `configs/templates/fm_testing.yaml` - For Foundation Model testing
- `configs/templates/evaluation.yaml` - For model evaluation/testing

## Multiple Output Lengths

To run an experiment with multiple output lengths, use `output_lens` instead of `output_len`:

```yaml
training:
  output_lens: [24, 168, 336, 720]  # Will create separate experiments for each
```

## Example Suites

- `linear_models.yaml` - DLinear and FITS across all datasets
- `transformer_models.yaml` - PatchTST and iTransformer across all datasets
- `foundation_models.yaml` - Sundial, TimeMoE, and Chronos across datasets
- `filtered_samples.yaml` - Testing on filtered sample sets

## Migration from Bash Scripts

The suite system replaces bash scripts like `run_all_linear.sh`. Instead of:

```bash
bash scripts/NYC_traffic_speed/dlinear.sh
bash scripts/California_ISO/dlinear.sh
```

You can now use:

```bash
python -m cli.suite run configs/experiment_suites/linear_models.yaml --filter dlinear
```

## Benefits

1. **Centralized Configuration**: All experiment parameters in one place
2. **Reproducibility**: Exact configurations saved and version-controlled
3. **Flexibility**: Easy to enable/disable experiments, modify parameters
4. **Documentation**: Configs serve as documentation
5. **Validation**: Config validation before execution
6. **Template Reuse**: Common patterns captured in templates

