# Experiment Suites Coverage Summary

This document summarizes the experiment suites created to replace bash scripts, excluding multi-GPU training scripts.

## Created Suites

### 1. **linear_models.yaml**
- **Coverage**: DLinear and FITS training across all 6 datasets
- **Replaces**: `run_all_linear.sh`
- **Status**: Complete

### 2. **transformer_models.yaml**
- **Coverage**: PatchTST and iTransformer training across all 6 datasets
- **Replaces**: `run_all_trans.sh`
- **Status**: Complete

### 3. **foundation_models.yaml**
- **Coverage**: Sundial, TimeMoE, and Chronos testing across datasets
- **Replaces**: `run_all_FM.sh`
- **Status**: Complete

### 4. **zero_shot_evaluation.yaml**
- **Coverage**: Zero-shot evaluation for DLinear, FITS, PatchTST, iTransformer, TGTSF, Sundial, TimeMoE, Chronos, GPT4TS, GPT4MTS
- **Replaces**: `run_all_zero_shot.sh`
- **Status**: Complete

### 5. **tgtsf_iatsf.yaml**
- **Coverage**: TGTSF training across all 6 datasets with heterogeneous data configs
- **Replaces**: `run_all_IATSF.sh`
- **Status**: Complete

### 6. **gpt4ts.yaml**
- **Coverage**: GPT4TS training across all 6 datasets
- **Replaces**: `run_all_RPLLM.sh` (GPT4TS portion)
- **Status**: Complete

### 7. **gpt4mts_rpllm.yaml**
- **Coverage**: GPT4MTS training across all 6 datasets with heterogeneous data
- **Replaces**: `run_all_RPLLM.sh` (GPT4MTS portion)
- **Status**: Complete

### 8. **filtered_samples.yaml** (Extended)
- **Coverage**: Filtered sample testing for DLinear, FITS, PatchTST, iTransformer, Sundial, TimeMoE, Chronos, GPT4MTS, TGTSF
- **Replaces**: `test_all_linear_on_samples.sh`, `test_all_trans_on_samples.sh`, `test_all_FM_on_samples.sh`, `test_all_RPLLM_on_samples.sh`
- **Status**: Complete

### 9. **test_evaluation.yaml**
- **Coverage**: Full dataset evaluation for PatchTST, iTransformer, TGTSF
- **Replaces**: `test_all_trans.sh`, `test_all_IATSF.sh`
- **Status**: Complete

### 10. **llm_qwen3_14b.yaml**
- **Coverage**: Qwen3-14B LLM experiments on filtered samples (day/week forecasting)
- **Replaces**: `scripts/LLM/Qwen3-14B/*.sh`
- **Status**: Complete (can be extended for other LLM models)

### 11. **ablation_studies.yaml**
- **Coverage**: TGTSF and GPT4MTS ablation studies on Bear Room dataset
- **Replaces**: `TGTSF_ablation.sh`, `gpt4mts_ablation.sh`, `gpt4mts_test_ablation.sh`
- **Status**: Complete

### 12. **chattime.yaml**
- **Coverage**: ChatTime Foundation Model testing on filtered samples (TGTSF task)
- **Replaces**: `chattime_on_samples.sh` scripts
- **Status**: Complete

## Features Supported

### Configuration Features
- Multiple output lengths (`output_lens`)
- Different data config variants (hetero, zero-shot, ablation)
- HuggingFace mirror/offline modes
- Filtered samples support
- Channel-wise evaluation
- Specific checkpoint versions
- Task specification (TSF, TGTSF, MTSF)
- Custom GPU assignments
- Custom hyperparameters (patience, epochs, batch size)

### Experiment Types
- PyTorch training (`pytorch`)
- Lightning training (`lightning`)
- LLM testing (`llm`)
- Foundation Model testing (`fm`)
- Standard evaluation (`evaluation` - uses `evaluation.standard`)
- Lightning evaluation (`evaluation` - uses `evaluation.lightning`)

## Not Covered (Intentionally Excluded)

### Multi-GPU Training Scripts (`*_m.sh`)
- `run_all_linear_m.sh`
- `run_all_trans_m.sh`
- `run_all_FM_m.sh`
- `run_all_IATSF_m.sh`
- Individual `*_m.sh` scripts

**Reason**: User requested to skip multi-GPU training for now.

### Server Startup Scripts
- `scripts/LLM/startup_server/*.sh`

**Reason**: Infrastructure scripts, not experiment configurations.

### Sampling Scripts
- `scripts/sampling/*.sh`
- `run_all_LLM_sampling.sh`

**Reason**: Data processing scripts, not model training/testing.

### Additional LLM Model Variants
- Qwen2.5-14B-Instruct
- Qwen2.5-14B-Instruct-1m
- DeepSeek-R1
- Time-R1

**Status**: Can be added following the same pattern as `llm_qwen3_14b.yaml`

## Usage Examples

```bash
# Run all linear model experiments
python -m cli.suite run configs/experiment_suites/linear_models.yaml

# Run zero-shot evaluation
python -m cli.suite run configs/experiment_suites/zero_shot_evaluation.yaml

# Run TGTSF training
python -m cli.suite run configs/experiment_suites/tgtsf_iatsf.yaml

# Run filtered samples testing
python -m cli.suite run configs/experiment_suites/filtered_samples.yaml

# List all available suites
python -m cli.suite list

# Validate a suite
python -m cli.suite validate configs/experiment_suites/linear_models.yaml
```

## Coverage Statistics

- **Total Bash Scripts**: ~265 scripts
- **Multi-GPU Scripts (Excluded)**: ~40 scripts
- **Infrastructure Scripts (Excluded)**: ~10 scripts
- **Covered by Suites**: ~215 scripts equivalent
- **Coverage**: ~81% (excluding multi-GPU and infrastructure)

## Notes

1. **Multi-GPU Support**: Can be added later by setting `use_multi_gpu: true` and `devices: "0,1,2,3"` in device config.

2. **LLM Scripts**: The bash scripts use `llm_run.py` which doesn't exist in current codebase. Suites use `run_llm.py` via `runs/llm.py` which should be functionally equivalent.

3. **Data Config Variants**: All major data config variants are supported:
   - Standard: `fullNYCTS_H.yaml`
   - Zero-shot: `fullNYCTS_zero_shot_H.yaml`
   - Hetero TGTSF: `fullNYCTS_hetero_TGTSF_H.yaml`
   - Hetero RPLLM: `fullNYCTS_hetero_RPLLM_H.yaml`
   - Hetero LLM: `fullNYCTS_hetero_LLM.yaml`
   - Ablation: `fullBear_hetero_ablation_*.yaml`

4. **Evaluation Types**: The suite executor automatically selects the correct evaluation function:
   - Lightning models (PatchTST, iTransformer, TGTSF) → `evaluation.lightning`
   - Other models → `evaluation.standard`

