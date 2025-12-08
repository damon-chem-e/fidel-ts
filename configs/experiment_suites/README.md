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
  
  # Resume configuration (suite-level, optional)
  # When resuming a suite, all experiments in the suite must resume from the same suite directory
  # Each experiment must also specify its own resume_experiment_id in its overrides
  # resume_suite_id: "suite_name_{timestamp}"  # Uncomment and set when resuming
    
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
        # Resume configuration (experiment-level, required when suite is resuming)
        # When suite-level resume_suite_id is set, each experiment must specify its resume_experiment_id
        # resume_experiment_id: "{experiment_id}"  # Uncomment and set when resuming
        mark_last_job_complete: true  # Optional: mark previous running job as complete/timeout
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

## Resuming Suites

When resuming a suite that was interrupted (e.g., due to SLURM job time limits), you must:

1. **Set suite-level resume configuration**: Add `resume_suite_id` at the suite level (the full suite directory name, e.g., `"tgtsf_test_20251207_145025"`)

2. **Set experiment-level resume configuration**: Each experiment in the suite must specify its own `resume_experiment_id` in its `overrides` section

3. **Important constraint**: When a suite is resuming (has `resume_suite_id` set), **all enabled experiments** in that suite must have `resume_experiment_id` set. This ensures consistency - either all experiments resume together, or none do.

Example resume configuration:

```yaml
suite:
  name: "tgtsf_test"
  # ... other suite config ...
  
  # Suite-level resume (all experiments resume from this suite directory)
  resume_suite_id: "suite_name_{timestamp}"
  
  experiments:
    - name: "experiment_1"
      overrides:
        # ... other overrides ...
        # Experiment-level resume (each experiment specifies its own experiment ID)
        resume_experiment_id: "{experiment_id}"
        mark_last_job_complete: true  # Optional: mark previous job as timeout
    
    - name: "experiment_2"
      overrides:
        # ... other overrides ...
        resume_experiment_id: "{experiment_id}"  # Different experiment ID
        mark_last_job_complete: true
```

**Note**: The `resume_suite_id` is the full suite directory name (including timestamp) found in your `output/` directory. Each experiment's `resume_experiment_id` is the experiment ID (including timestamp) found within that suite directory.

## Benefits

1. **Centralized Configuration**: All experiment parameters in one place
2. **Reproducibility**: Exact configurations saved and version-controlled
3. **Flexibility**: Easy to enable/disable experiments, modify parameters
4. **Documentation**: Configs serve as documentation
5. **Validation**: Config validation before execution
6. **Template Reuse**: Common patterns captured in templates
7. **Consistent Resumption**: Suite-level resume ensures all experiments resume from the same suite directory

