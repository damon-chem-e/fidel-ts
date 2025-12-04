# lynx Model Implementation Plan

## Overview
lynx is a modified version of TGTSF that learns the **residual** to a pretrained frozen unimodal model (default: iTransformer) instead of learning the entire prediction from scratch.

## Architecture
- **Base Model**: TGTSF (Text-Guided Time Series Forecasting)
- **Unimodal Model**: Pretrained frozen model (default: iTransformer)
- **Output**: `final_prediction = unimodal_prediction + residual_prediction`

## Implementation Steps

### 1. Create Model File: `models/lynx.py`

#### 1.1 Model Structure
- Inherit from `nn.Module` (same as TGTSF)
- Include all TGTSF components:
  - `TS_encoder`: Time series encoder
  - `text_encoder`: Text encoder for news and channel descriptions
  - `mixer`: Cross-modal attention mixer
  - `head`: Output projection head
- Add new components:
  - `unimodal_model`: Frozen pretrained unimodal model (default: iTransformer)
  - `pretrained_model_path`: Path to checkpoint
  - `pretrained_model_type`: Type of pretrained model (default: 'iTransformer')

#### 1.2 Initialization (`__init__`)
```python
def __init__(self, configs):
    # Load all TGTSF parameters (same as TGTSF)
    # Load pretrained unimodal model:
    #   - Read configs.pretrained_model_path (required)
    #   - Read configs.pretrained_model_type (default: 'iTransformer')
    #   - Initialize the unimodal model
    #   - Load checkpoint from pretrained_model_path
    #   - Freeze all parameters (requires_grad=False)
    #   - Set to eval mode
```

#### 1.3 Forward Pass
```python
def forward(self, x, news, channel_description, **kwargs):
    # Step 1: Apply normalization matching the pretrained model
    # If pretrained model uses use_norm (iTransformer):
    #   - Apply iTransformer normalization: x_norm = (x - mean) / std
    #   - Store mean and std for denormalization
    #   - Disable RevIN in TGTSF components
    # If pretrained model uses RevIN:
    #   - Use TGTSF's RevIN as normal
    
    x_norm, norm_params = self._apply_normalization(x)
    
    # Step 2: Get unimodal prediction (frozen model, operates on normalized input)
    with torch.no_grad():
        unimodal_pred_norm = self.unimodal_model(x_norm)  # [B, pred_len, nvars]
    
    # Step 3: Get TGTSF residual prediction (operates on normalized input)
    # Note: RevIN is disabled if pretrained model uses use_norm
    residual_pred_norm = self._tgtsf_forward(x_norm, news, channel_description)
    
    # Step 4: Combine predictions in normalized space
    final_pred_norm = unimodal_pred_norm + residual_pred_norm
    
    # Step 5: Denormalize final prediction
    final_pred = self._denormalize(final_pred_norm, norm_params)
    
    return final_pred
```

#### 1.4 Helper Methods
- `_load_pretrained_model()`: Load and freeze pretrained model
  - Detect normalization scheme from pretrained model config
  - Store normalization type for use in forward pass
- `_apply_normalization()`: Apply normalization matching pretrained model
  - If pretrained uses `use_norm`: Apply iTransformer normalization, return params
  - If pretrained uses RevIN: Apply RevIN normalization
- `_denormalize()`: Denormalize predictions using stored normalization parameters
- `_tgtsf_forward()`: Extract TGTSF forward logic (reuse existing code)
  - Modified to accept `disable_revin` flag when pretrained model uses different norm
- `move_to_device()`: Same as TGTSF (for compatibility)

### 2. Model Configuration: `model_configs/general/lynx.yaml`

Create configuration file with:
- All TGTSF parameters (inherit from TGTSF.yaml)
- New parameters:
  - `pretrained_model_path`: Path to pretrained checkpoint (required)
  - `pretrained_model_type`: Type of model (default: 'iTransformer')
  - `pretrained_model_config_path`: Optional path to pretrained model config (if different from default)
- **Normalization**: Match the pretrained model's normalization scheme (see section 6.4)

### 3. Checkpoint Loading Logic

#### 3.1 Loading Pretrained Unimodal Model
- Support loading from:
  - Lightning checkpoints (`.ckpt` files)
  - PyTorch checkpoints (`.pth` files)
  - Directory with checkpoint files
- Handle state_dict key mapping (remove "model." prefix if needed)
- Validate model compatibility (seq_len, pred_len, enc_in)

#### 3.2 Model Compatibility Checks
- Verify `seq_len` matches (or can be adapted)
- Verify `pred_len` matches
- Verify `enc_in` (number of channels) matches
- Handle dimension mismatches gracefully

### 4. Integration Points

#### 4.1 Model Registration
- No changes needed to `models/__init__.py` (uses dynamic import)
- Model will be automatically discoverable as "lynx" (lowercase)

#### 4.2 Training Compatibility
- Compatible with both PyTorch and Lightning training
- Loss computation remains the same (computed externally)
- Model outputs final prediction, loss is `criterion(output, gt)`

#### 4.3 Evaluation Compatibility
- Works with existing evaluation scripts
- No special handling needed

### 5. Configuration Examples

#### 5.1 Basic Configuration
```yaml
model: lynx
pretrained_model_path: "checkpoints/iTransformer_nyc_24_360/checkpoint.pth"
pretrained_model_type: "iTransformer"
# ... all TGTSF parameters ...
```

#### 5.2 With Custom Pretrained Model Config
```yaml
model: lynx
pretrained_model_path: "checkpoints/custom_model/checkpoint.ckpt"
pretrained_model_type: "iTransformer"
pretrained_model_config_path: "model_configs/general/iTransformer.yaml"
# ... all TGTSF parameters ...
```

### 6. Implementation Details

#### 6.1 Freezing Pretrained Model
```python
# After loading checkpoint
for param in self.unimodal_model.parameters():
    param.requires_grad = False
self.unimodal_model.eval()
```

#### 6.2 Handling Different Model Types
- Support iTransformer (default)
- Design extensible for future models (PatchTST, etc.)
- Use factory pattern or conditional loading

#### 6.3 Error Handling
- Check if pretrained_model_path exists
- Validate checkpoint format
- Handle missing keys in state_dict
- Provide clear error messages

#### 6.4 Normalization Scheme Matching
**Critical**: lynx must use the same normalization scheme as the pretrained unimodal model.

- **iTransformer**: Uses `use_norm: True` (Non-stationary Transformer normalization)
  - Normalizes: `x = (x - mean) / std` before forward pass
  - Denormalizes: `output = output * std + mean` after forward pass
- **TGTSF**: Uses `revin: True` (RevIN normalization)
  - Similar normalization but implemented differently

**Solution**: Modify lynx to:
1. Detect the pretrained model's normalization scheme
2. Apply the same normalization to input before both models
3. Ensure both models operate on the same normalized space
4. Apply denormalization only once at the end

**Implementation**:
- If pretrained model uses `use_norm=True` (iTransformer), disable RevIN in TGTSF components
- Apply iTransformer's normalization before both models
- Combine predictions in normalized space
- Denormalize final output using the same scheme

### 7. Testing Considerations

#### 7.1 Unit Tests
- Test model initialization with valid checkpoint
- Test forward pass produces correct shape
- Test that unimodal model is frozen
- Test residual learning (output = base + residual)

#### 7.2 Integration Tests
- Test with Lightning training
- Test with PyTorch training
- Test evaluation pipeline
- Test checkpoint saving/loading

### 8. Documentation

#### 8.1 Code Documentation
- Docstrings for all methods
- Inline comments explaining residual learning
- Configuration parameter descriptions

#### 8.2 Usage Documentation
- Example configuration files
- Training command examples
- Notes on pretraining the unimodal model

## File Structure

```
models/
  ├── lynx.py                    # New model file (lowercase)
  └── ...

model_configs/
  └── general/
      └── lynx.yaml              # New config file (lowercase)

configs/
  ├── experiments/
  │   └── lynx_example.yaml      # Example experiment config
  └── experiment_suites/
      ├── itransformer_pretraining.yaml  # Suite 1: Train iTransformer on all datasets
      └── lynx_training.yaml            # Suite 2: Train lynx using iTransformer checkpoints
```

## Implementation Order

1. **Phase 1: Core Model** (Priority: High)
   - Create `models/lynx.py` with basic structure
   - Implement pretrained model loading (iTransformer only)
   - Implement normalization matching logic
   - Implement forward pass with residual learning
   - Test basic functionality

2. **Phase 2: Configuration** (Priority: High)
   - Create `model_configs/general/lynx.yaml`
   - Add configuration validation
   - Create example experiment config

3. **Phase 3: Experiment Suites** (Priority: High)
   - Create `configs/experiment_suites/itransformer_pretraining.yaml`
     - Train iTransformer on all 6 datasets
     - Use same configs as transformer_models.yaml (input_len: 360, output_lens: [24, 168, 336, 720])
     - Save checkpoints for each dataset/output_len combination
   - Create `configs/experiment_suites/lynx_training.yaml`
     - Train lynx on all 6 datasets
     - Reference iTransformer checkpoints from pretraining suite
     - Use TGTSF data configs (heterogeneous data)
     - Ensure compatibility: same input_len, output_len, normalization

4. **Phase 4: Robustness** (Priority: Medium)
   - Add error handling
   - Support multiple checkpoint formats
   - Add compatibility checks
   - Extend to support other pretrained models

5. **Phase 5: Testing & Documentation** (Priority: Medium)
   - Add unit tests
   - Integration testing
   - Documentation
   - Example usage

## Key Design Decisions

1. **Residual Learning**: The model learns `residual = target - unimodal_prediction`, so the final output is `unimodal_prediction + residual_prediction`.

2. **Frozen Base Model**: The pretrained unimodal model is completely frozen to preserve its learned representations and ensure the residual model focuses on what the base model misses.

3. **Default Model**: iTransformer is chosen as default because:
   - It's a strong unimodal baseline
   - It's already in the codebase
   - It has a simple interface compatible with TGTSF

4. **Checkpoint Path**: Required parameter to ensure explicit model selection and avoid ambiguity.

5. **Compatibility**: The model maintains the same interface as TGTSF (same forward signature) for seamless integration.

## Potential Challenges & Solutions

1. **Challenge**: Dimension mismatches between pretrained model and TGTSF
   - **Solution**: Add validation checks and clear error messages

2. **Challenge**: Different normalization schemes
   - **Solution**: lynx detects and matches the pretrained model's normalization scheme. For iTransformer (use_norm=True), lynx disables RevIN and applies iTransformer's normalization to both models, ensuring they operate in the same normalized space.

3. **Challenge**: Checkpoint format variations
   - **Solution**: Support both Lightning (.ckpt) and PyTorch (.pth) formats with key mapping

4. **Challenge**: Memory usage (two models)
   - **Solution**: Unimodal model is in eval mode and frozen, so gradients aren't computed for it. User confirmed memory is not a concern (only 2% VRAM usage currently).

## Experiment Suites

### Suite 1: iTransformer Pretraining (`itransformer_pretraining.yaml`)

**Purpose**: Train iTransformer models on all datasets and save checkpoints for use as frozen base models in lynx.

**Datasets** (6 total):
- NYC_traffic_speed
- California_ISO
- Canada_photovoltaics_plants
- Germany_Renewable_Power_Grid
- Jena_Atmospheric_Physics
- Bear_room

**Configuration**:
- Model: iTransformer
- Input length: 360 (consistent across all)
- Output lengths: [24, 168, 336, 720] (creates 4 experiments per dataset = 24 total)
- Data configs: Use homogeneous configs (e.g., `fullNYCTS_H.yaml`)
- Training: Lightning training
- Output directory: `outputs/suites/itransformer_pretraining`

**Checkpoint Naming Convention**:
- Checkpoints saved as: `{output_dir}/{experiment_name}/checkpoint.ckpt`
- Experiment names: `itransformer_{dataset}_{output_len}`
- Example: `itransformer_nyc_traffic_speed_24`

**Usage**:
```bash
python -m cli.suite run configs/experiment_suites/itransformer_pretraining.yaml
```

### Suite 2: lynx Training (`lynx_training.yaml`)

**Purpose**: Train lynx models using pretrained iTransformer checkpoints as frozen base models.

**Datasets** (6 total, same as Suite 1):
- NYC_traffic_speed
- California_ISO
- Canada_photovoltaics_plants
- Germany_Renewable_Power_Grid
- Jena_Atmospheric_Physics
- Bear_room

**Configuration**:
- Model: lynx
- Input length: 360 (must match iTransformer pretraining)
- Output lengths: [24, 168, 336, 720] (must match iTransformer pretraining)
- Data configs: Use TGTSF heterogeneous configs (e.g., `fullNYCTS_hetero_TGTSF_H.yaml`)
- Pretrained model path: References checkpoints from Suite 1
- Training: Lightning training
- Output directory: `outputs/suites/lynx_training`

**Checkpoint Path Resolution**:
- Base path: `outputs/suites/itransformer_pretraining`
- Experiment name pattern: `itransformer_{dataset}_{output_len}`
- Checkpoint file: `checkpoint.ckpt` (Lightning format)
- Full path example: `outputs/suites/itransformer_pretraining/itransformer_nyc_traffic_speed_24/checkpoint.ckpt`

**Compatibility Requirements**:
1. **Input/Output Lengths**: Must exactly match between iTransformer pretraining and lynx training
2. **Normalization**: lynx automatically matches iTransformer's normalization scheme
3. **Number of Channels**: Must match (handled by data configs)
4. **Checkpoint Format**: Supports Lightning `.ckpt` format

**Usage**:
```bash
# First run Suite 1 to generate checkpoints
python -m cli.suite run configs/experiment_suites/itransformer_pretraining.yaml

# Then run Suite 2 (after Suite 1 completes)
python -m cli.suite run configs/experiment_suites/lynx_training.yaml
```

**Suite 2 Structure**:
```yaml
suite:
  name: "lynx_training"
  description: "Train lynx models using pretrained iTransformer checkpoints"
  tags: ["lynx", "residual", "lightning", "heterogeneous"]
  
  execution:
    parallel: false
    continue_on_error: true
    log_dir: "./logs/suites/lynx_training"
  
  experiments:
    # For each dataset and output_len combination:
    - name: "lynx_nyc_traffic_speed_24"
      description: "lynx on NYC Traffic Speed with 24-step prediction"
      enabled: true
      template: "configs/templates/lightning_training.yaml"
      overrides:
        model:
          name: "lynx"
          config_path: "model_configs/general/lynx.yaml"
        data:
          name: "NYC_traffic_speed"
          config_path: "data_configs/NYC_traffic_speed/fullNYCTS_hetero_TGTSF_H.yaml"
        training:
          input_len: 360
          output_len: 24  # Must match iTransformer pretraining
          batch_size: 256
          patience: 5
          epochs: 50
        experiment:
          type: "lightning"
          output_dir: "outputs/suites/lynx_training"
```

**Model Config Override** (in experiment suite):
The `lynx.yaml` config will need `pretrained_model_path` set per experiment. This can be done via:
1. Template variables in the suite config
2. Or dynamically resolved based on dataset/output_len in the model initialization

**Recommended Approach**: Use dynamic path resolution in `lynx.py`:
- If `pretrained_model_path` is not provided, construct it from:
  - Base checkpoint directory (configurable via `pretrained_checkpoint_base` in config)
  - Dataset name (from data config or all_args)
  - Output length (from training config or all_args)
  - Model type (iTransformer)
- Path pattern: `{base_dir}/itransformer_{dataset}_{output_len}/checkpoint.ckpt`
- Example: `outputs/suites/itransformer_pretraining/itransformer_nyc_traffic_speed_24/checkpoint.ckpt`

**Alternative**: Set `pretrained_model_path` explicitly in each experiment override:
```yaml
overrides:
  model:
    name: "lynx"
    config_path: "model_configs/general/lynx.yaml"
    # Override pretrained_model_path in model config
    pretrained_model_path: "outputs/suites/itransformer_pretraining/itransformer_nyc_traffic_speed_24/checkpoint.ckpt"
```

**Implementation Note**: The model initialization in `models/__init__.py` passes `all_args` which contains `input_len`, `output_len`, and `data_config`. These can be used to construct checkpoint paths dynamically.

## Future Enhancements

1. Support for other pretrained models (PatchTST, DLinear, etc.)
2. Optional fine-tuning of pretrained model (with flag)
3. Multi-model ensemble (combine multiple pretrained models)
4. Adaptive weighting between base and residual predictions
5. Automatic checkpoint path resolution based on experiment naming conventions

