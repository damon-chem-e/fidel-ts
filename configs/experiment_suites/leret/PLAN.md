# Plan: LeRet Experiment Suite for Fidel-TS Datasets

## Executive Summary

**Goal**: Create `configs/experiment_suites/leret/` directory with experiment configs for all fidel-ts datasets using LeRet model with text embeddings.

**Key Requirements**:
- Use `y_hetero` (text embeddings aligned to prediction window)
- Use `timestamp_semantics: t_about` (already set in data configs)
- Use original LeRet hyperparameters from `model_configs/general/LeRet.yaml`
- Match input/output lengths from `lynx_film_raw` suite
- Enable text embeddings via `use_language: True`

**Key Decisions Made**:
1. **Pretrain/Finetune Split**: 25 epochs pretrain, 25 epochs finetune (50/50 split)
2. **Batch Size & Learning Rate**: Match `lynx_film_raw` exactly (batch_size=768, lr=8.66e-4)
3. **Enc_in**: Omit from configs - let code infer automatically from `data_config.input_channel`
4. **Task Type**: Set `task: TGTSF` in `model_config_overrides` explicitly

**Critical Issue**: LeRet experiment code currently doesn't pass text embeddings to model. Need to modify `exp/model_specific/leret.py` to extract `batch_y_hetero` and pass as `language_embeddings` parameter.

**Files to Create**: 7 config files (6 individual datasets + 1 combined suite)

## Overview
Create experiment suite configs for LeRet (Language-Enhanced Retention Network) on all fidel-ts datasets, using:
- `y_hetero` (text embeddings aligned to prediction window)
- `timestamp_semantics: t_about` (text describes prediction window)
- Original LeRet hyperparameters from model config
- Input/output lengths matching `lynx_film_raw` suite

## Key Findings from Analysis

### 1. LeRet-Specific Config Options (from `leret_test.yaml`)
```yaml
training:
  leret:
    training_stage: "both"  # or "pretrain" or "finetune"
    pretrain_epochs: 1
    pretrain_loss: "mse"
    finetune_loss: "mse"
    pretrain_learning_rate: 1e-3
```

### 2. Original LeRet Hyperparameters (from `model_configs/general/LeRet.yaml`)
```yaml
d_model: 128
n_heads: 8
e_layers: 3
d_ff: 256
patch_len: 16
stride: 8
padding_patch: end
revin: True
affine: False
subtract_last: False
decomposition: False
kernel_size: 25
individual: False
dropout: 0.05
fc_dropout: 0.05
head_dropout: 0.0
use_language: False  # Need to set to True for text embeddings
language_dim: 768   # Standard embedding dimension
```

### 3. Text Embedding Configuration
- **Data configs** already have `timestamp_semantics: t_about` set
- **y_hetero** contains text embeddings aligned to prediction window
- **LeRet model** accepts `language_embeddings` parameter but needs:
  - `use_language: True` in model config
  - Text embeddings extracted from `y_hetero` and passed to model
  - Format: `[text_num, language_dim]` (needs aggregation from `[B, pred_len, num_items, text_dim]`)

### 4. Fidel-TS Datasets (from `lynx_film_raw` suite)

#### Individual Dataset Configs:
1. **Bear_room** (`bear_room.yaml`)
   - Input: 288 (1 day at 5-min intervals)
   - Output: 144 (12 hours)
   - Data config: `data_configs/Bear_room/fullBear_hetero_TGTSF.yaml`
   - Batch size: 768
   - Learning rate: 8.66e-4

2. **California_ISO** (`california_iso.yaml`)
   - Input: 360 (15 days at hourly intervals)
   - Output: 168 (1 week)
   - Data config: `data_configs/California_ISO/fullCAISO_hetero_TGTSF_H.yaml`
   - Batch size: 768
   - Learning rate: 8.66e-4

3. **Canada_photovoltaics** (`canada_photovoltaics.yaml`)
   - Input: 360 (15 days at hourly intervals)
   - Output: 168 (1 week)
   - Data config: `data_configs/Canada_photovoltaics_plants/fullCPP_hetero_TGTSF.yaml`
   - Batch size: 768
   - Learning rate: 8.66e-4

4. **Germany_Renewable_Power_Grid** (`germany_renewable.yaml`)
   - Input: 360 (15 days at hourly intervals)
   - Output: 168 (1 week)
   - Data config: `data_configs/Germany_Renewable_Power_Grid/fullGRPG_hetero_TGTSF_H.yaml`
   - Batch size: 768
   - Learning rate: 8.66e-4

5. **Jena_Atmospheric_Physics** (`jena_atmospheric.yaml`)
   - Input: 360 (15 days at hourly intervals)
   - Output: 168 (1 week)
   - Data config: `data_configs/Jena_Atmospheric_Physics/fullJAP_hetero_TGTSF_H.yaml`
   - Batch size: 768
   - Learning rate: 8.66e-4

6. **NYC_traffic_speed** (`nyc_traffic_speed.yaml`)
   - Input: 360 (15 days at hourly intervals)
   - Output: 168 (1 week)
   - Data config: `data_configs/NYC_traffic_speed/fullNYCTS_hetero_TGTSF_H.yaml`
   - Batch size: 768
   - Learning rate: 8.66e-4

#### Combined Suite (Time-MMD + TTC):
7. **time_mmd_ttc.yaml** - Contains all Time-MMD and TTC datasets:
   - Time-MMD datasets (all use input=24, output varies):
     - Agriculture: output=6
     - Climate: output=6
     - Economy: output=6
     - Energy: output=12
     - Environment: output=48
     - Health: output=12
     - Socialgood: output=6
     - Traffic: output=6
   - TTC datasets (both use input=7, output=7):
     - Weather: `data_configs/ttc/climate/config.yaml`
     - Medical: `data_configs/ttc/medical/config.yaml`

## Configuration Structure Plan

### Template Structure (per dataset):
```yaml
suite:
  name: "leret_<dataset_name>"
  description: "Train LeRet model on <dataset_name> with text embeddings"
  tags: ["leret", "retention", "pytorch", "fidel-ts", "<dataset_name>", "text-embeddings"]
  
  execution:
    parallel: false
    continue_on_error: true
    log_dir: "./logs/suites/leret/<dataset_name>"
  
  experiments:
    - name: "leret_<dataset_name>"
      description: "LeRet with text embeddings on <dataset_name>"
      enabled: true
      template: "configs/templates/pytorch_training.yaml"
      overrides:
        model:
          name: "LeRet"
          config_path: "model_configs/general/LeRet.yaml"
        data:
          name: "<Dataset_Name>"
          config_path: "<data_config_path>"
        training:
          input_len: <from_lynx_film_raw>
          output_len: <from_lynx_film_raw>
          batch_size: <from_lynx_film_raw>  # May need adjustment for LeRet
          learning_rate: <from_lynx_film_raw>  # May need adjustment
          patience: 5
          epochs: 50  # Total epochs (pretrain + finetune)
          track_per_sample: false
          evaluate_test_during_training: false
          truncate_train_for_purge: true
          torch_compile: true
          compile_mode: "reduce-overhead"
          # LeRet-specific two-stage training
          leret:
            training_stage: "both"
            pretrain_epochs: <TBD>  # Need to determine split
            pretrain_loss: "mse"
            finetune_loss: "mse"
            pretrain_learning_rate: 1e-3
        wandb:
          tags: ["leret", "<dataset_name>"]
        experiment:
          type: "pytorch"
          output_dir: "outputs/suites/leret/<dataset_name>"
        model_config_overrides:
          # Original LeRet hyperparameters
          d_model: 128
          n_heads: 8
          e_layers: 3
          d_ff: 256
          patch_len: 16
          stride: 8
          padding_patch: end
          revin: True
          affine: False
          subtract_last: False
          decomposition: False
          kernel_size: 25
          individual: False
          dropout: 0.05
          fc_dropout: 0.05
          head_dropout: 0.0
          # Text embedding configuration
          use_language: True
          language_dim: 768
          # Dataset-specific
          # enc_in: Omit - let code infer from data_config.input_channel automatically
          seq_len: <input_len>
          pred_len: <output_len>
```

## Key Decisions Needed

### 1. Text Embedding Integration
**Issue**: LeRet model accepts `language_embeddings` but:
- Current experiment code (`exp/model_specific/leret.py`) doesn't extract/pass text embeddings
  - Line 541: `forecast, auto_y = self.model(x=batch_x)` - only passes `x`, not text embeddings
  - `batch_y_hetero` is available in batch but not used
- Model forward doesn't currently use `language_embeddings` parameter (defined but unused in forward logic)
- Need to aggregate `y_hetero` from `[B, pred_len, num_items, text_dim]` to format expected by model
- Other models (lynx_film_raw, TGTSF) receive `news=batch_y_hetero` directly in forward via experiment code

**Options**:
- A) Modify LeRet experiment code (`exp/model_specific/leret.py`) to:
  - Extract `batch_y_hetero` from batch
  - Aggregate from `[B, pred_len, num_items, text_dim]` to `[text_num, language_dim]`
  - Pass as `language_embeddings` parameter to model forward
- B) Modify LeRet model forward to:
  - Accept `news` or `y_hetero` in kwargs
  - Extract and aggregate internally
  - Use in language integrator if `use_language=True`
- C) Follow universal experiment pattern (but LeRet uses model-specific trainer)

**Recommendation**: 
- **Option A** is cleaner - modify experiment code to extract and pass embeddings
- Need to determine aggregation strategy (mean pooling? max pooling? flatten?)
- Check `LanguageIntegrator` expected input format

### 2. Pretrain Epochs Split
**Decision**: Split 50 total epochs evenly: **25 pretrain, 25 finetune**

**Rationale**: User specified 50/50 split for balanced two-stage training.

### 3. Batch Size and Learning Rate
**Decision**: Match `lynx_film_raw` values exactly:
- **Batch size**: 768 (same as lynx_film_raw)
- **Learning rate**: 8.66e-4 (same as lynx_film_raw)

**Rationale**: User specified to match lynx_film_raw for consistency across model comparisons.

### 4. Task Type
**Issue**: LeRet model config has `task: TSF` but we need `TGTSF` for text embeddings.

**Options**:
- A) Override in `model_config_overrides: {task: TGTSF}`
- B) Check if task is read from data config instead
- C) Verify if LeRet experiment code handles this automatically

**Recommendation**: Override to `TGTSF` in model_config_overrides to be explicit.

### 5. Enc_in (Number of Channels)
**Decision**: **Omit `enc_in` from model_config_overrides** - let code infer automatically

**Findings from Code Analysis**:
- `models/__init__.py` (line 53) automatically sets `configs['input_channel'] = data_configs.input_channel` if available
- LeRet model (line 110-117) checks for `enc_in` first, then falls back to `input_channel`
- If neither is found, LeRet raises an error requiring explicit `enc_in`
- User believes fidel-ts datasets can infer this automatically

**Implementation Note**: 
- Do NOT set `enc_in` in `model_config_overrides` initially
- If inference fails at runtime, we can add it explicitly per dataset
- The code path: `model_init()` → tries `data_configs.input_channel` → LeRet checks `enc_in` then `input_channel`

## Implementation Checklist

### Before Implementation:
- [ ] Verify how text embeddings are extracted from `y_hetero` in other models (lynx_film_raw, TGTSF)
- [ ] Check if LeRet model forward needs modification to use `language_embeddings`
- [ ] Determine pretrain/finetune epoch split strategy
- [ ] Verify task type handling (TSF vs TGTSF)
- [ ] Get `enc_in` values for each dataset
- [ ] Test LeRet with text embeddings on one dataset first

### Implementation Steps:
1. [ ] Create `configs/experiment_suites/leret/` directory
2. [ ] Create individual dataset configs (6 files):
   - `bear_room.yaml`
   - `california_iso.yaml`
   - `canada_photovoltaics.yaml`
   - `germany_renewable.yaml`
   - `jena_atmospheric.yaml`
   - `nyc_traffic_speed.yaml`
3. [ ] Create combined suite config:
   - `time_mmd_ttc.yaml` (with all Time-MMD and TTC datasets)
4. [ ] Verify all configs use:
   - `timestamp_semantics: t_about` (in data configs, already set)
   - `y_hetero` for text embeddings
   - Original LeRet hyperparameters
   - Matching input/output lengths from lynx_film_raw
   - Matching batch_size (768) and learning_rate (8.66e-4) from lynx_film_raw
   - `use_language: True` in model_config_overrides
   - `task: TGTSF` in model_config_overrides
   - `pretrain_epochs: 25` in training.leret
   - Do NOT set `enc_in` (let code infer automatically)

## Files to Create

1. `configs/experiment_suites/leret/bear_room.yaml`
2. `configs/experiment_suites/leret/california_iso.yaml`
3. `configs/experiment_suites/leret/canada_photovoltaics.yaml`
4. `configs/experiment_suites/leret/germany_renewable.yaml`
5. `configs/experiment_suites/leret/jena_atmospheric.yaml`
6. `configs/experiment_suites/leret/nyc_traffic_speed.yaml`
7. `configs/experiment_suites/leret/time_mmd_ttc.yaml`

## Notes

- All fidel-ts data configs already have `timestamp_semantics: t_about` set
- Text embeddings are pre-computed and stored in pickle files
- LeRet uses two-stage training (pretrain + finetune) which is handled by experiment code
- Model supports language integration via `LanguageIntegrator` layer when `use_language=True`
- Need to ensure experiment code passes text embeddings to model forward method
