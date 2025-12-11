# Text Noise Injection Implementation Plan (v3)
## Integrated with Embedding Sequence Dimension Support

## 1. Overview

This plan extends the text noise injection system (see [PLAN.md](./PLAN.md)) to incorporate embedding sequence dimension support (see [embeddings_upgrades.md](./embeddings_upgrades.md)). The integrated system ensures:

1. **Embedding Version Identification**: Dataset IDs include embedding pooling type (`postemb_pooling`)
2. **Config Alignment**: Model configs specify expected embedding type, with validation to ensure data and model configs align
3. **Backward Compatibility**: Support for both 4D (`[B, L, N, text_dim]`) and 5D (`[B, L, N, seq_len, text_dim]`) embeddings
4. **Reproducibility**: Complete tracking of embedding configuration in generated datasets

## 2. Integration Points

### 2.1 Embedding Versioning in Dataset IDs

**Reference:** [PLAN.md Section 3.1](./PLAN.md#31-identification--versioning)

**Enhancement:** Dataset IDs must include embedding pooling type to ensure compatibility.

**Current Dataset ID Format:**
```
{timestamp}_{config_hash}
```

**New Dataset ID Format:**
```
{timestamp}_{config_hash}_{embedding_version}
```

Where `embedding_version` encodes:
- `postemb_pooling` value (`cls`, `mean`, or `none`)
- Model name and version used for embeddings (optional, for reproducibility)
- Example: `20251210_a1b2c3d4_cls` or `20251210_a1b2c3d4_none_bert-base-uncased`

**Implementation:**
- Update `text_noise_injection/utils.py` to include `postemb_pooling` in hash calculation
- Store embedding config in `config.yaml` within generated dataset directory

### 2.2 Enhanced Directory Structure

**Reference:** [PLAN.md Section 3.1](./PLAN.md#31-identification--versioning)

**Updated Structure:**
```
data/NYC_traffic_speed/
└── {INJECTED_DATASET_ID}/
    ├── config.yaml             # Injection config + embedding config
    ├── embedding_config.yaml    # Detailed embedding generation config
    ├── metadata.json            # Per-sample injection logs
    ├── raw_text/                # Structure mirroring original raw data
    │   └── weather/
    │       └── merged_general_report/
    │           └── merged_general_weather_report.json
    └── embeddings/              # Pre-computed embeddings
        └── weather/
            └── merged_report_embedding/
                └── fast_general_formal_embeddings_2017.pkl
```

**New Files:**
- `embedding_config.yaml`: Records the exact embedding configuration used:
  ```yaml
  postemb_pooling: 'none'  # 'cls', 'mean', or 'none'
  model_name: 'bert-base-uncased'
  max_length: 512
  embedding_shape: [seq_len, hidden_dim]  # or [hidden_dim] for 'cls'/'mean'
  ```

## 3. Enhanced Components

### 3.1 Updated `config.py`

**Reference:** [PLAN.md Section 3.2](./PLAN.md#32-components)

**Add to `InjectionConfig`:**
```python
class InjectionConfig:
    # ... existing fields ...
    
    # Embedding configuration
    postemb_pooling: str = 'cls'  # 'cls', 'mean', or 'none'
    embedding_model_name: Optional[str] = None  # e.g., 'bert-base-uncased'
    embedding_max_length: int = 512  # Token sequence max length
```

**Rationale:** The injection config must specify how embeddings will be generated to ensure consistent dataset IDs and compatibility.

### 3.2 Enhanced `injector.py`

**Reference:** [PLAN.md Section 3.2](./PLAN.md#32-components)

**Changes:**
1. Include `postemb_pooling` in dataset ID generation
2. Save embedding config to `embedding_config.yaml`
3. Update `metadata.json` to include embedding version info

**Updated Metadata Schema:**
```json
{
  "dataset_id": "20251210_a1b2c3d4_none",
  "base_dataset": "NYC_traffic_speed",
  "embedding_config": {
    "postemb_pooling": "none",
    "model_name": "bert-base-uncased",
    "max_length": 512
  },
  "injections": {
    "20170101": {
      "brooklyn": {
        "is_injected": false,
        "type": "none"
      },
      "fake_loc_1": {
        "is_injected": true,
        "type": "type1",
        "source_time": "20180512",
        "source_location": "queens"
      }
    }
  }
}
```

### 3.3 Enhanced `embedder.py`

**Reference:** [PLAN.md Section 3.2](./PLAN.md#32-components) and [embeddings_upgrades.md Section 1](./embeddings_upgrades.md#1-data-provider-changes)

**Key Changes:**
1. **Use Shared Embedding Utility**: Extract embedding logic to `utils/text_embedding.py` (as planned)
2. **Respect `postemb_pooling`**: Use the pooling method specified in injection config
3. **Validate Output Shapes**: Ensure generated embeddings match expected shape for pooling type
4. **Save Embedding Config**: Write `embedding_config.yaml` alongside embeddings

**Implementation:**
```python
# In embedder.py
from utils.text_embedding import generate_embeddings

def generate_embeddings_for_dataset(
    noisy_json_path: str,
    output_dir: str,
    postemb_pooling: str = 'cls',
    embedding_model_name: str = 'bert-base-uncased',
    max_length: int = 512
):
    """
    Generate embeddings for noisy dataset with specified pooling.
    
    Args:
        noisy_json_path: Path to noisy JSON data
        output_dir: Directory to save embeddings
        postemb_pooling: 'cls', 'mean', or 'none'
        embedding_model_name: HuggingFace model identifier
        max_length: Maximum token sequence length
    
    Returns:
        Path to generated embedding file
    """
    # Load noisy data
    data = load_json(noisy_json_path)
    
    # Generate embeddings using shared utility
    embeddings = generate_embeddings(
        texts=data,
        pooling=postemb_pooling,
        model_name=embedding_model_name,
        max_length=max_length
    )
    
    # Validate shape
    expected_shape = validate_embedding_shape(embeddings, postemb_pooling)
    
    # Save embeddings
    save_path = save_embeddings(embeddings, output_dir)
    
    # Save embedding config
    embedding_config = {
        'postemb_pooling': postemb_pooling,
        'model_name': embedding_model_name,
        'max_length': max_length,
        'embedding_shape': list(expected_shape)
    }
    save_yaml(embedding_config, os.path.join(output_dir, 'embedding_config.yaml'))
    
    return save_path
```

## 4. Configuration Alignment & Validation

### 4.1 Model Config Specification

**Reference:** [embeddings_upgrades.md Section 3.1](./embeddings_upgrades.md#31-add-to-config-files)

**Model configs must specify expected embedding type:**

**Location:** Model config YAML files (e.g., `model_configs/general/lynx_mmitransformer.yaml`)

```yaml
# Text Embedding Parameters
postemb_pooling: 'none'  # 'cls' (default), 'mean', or 'none' (keep sequence)
postemb_pooling_required: true  # If true, validation will enforce exact match
```

**Rationale:** Models that require sequence dimension (e.g., for text self-attention) must explicitly declare this requirement.

### 4.2 Data Config Specification

**Location:** Data config YAML files

```yaml
# Text Embedding Parameters
postemb_pooling: 'none'  # 'cls' (default), 'mean', or 'none'
# If using pre-computed embeddings, this must match the embedding_config.yaml
```

**For Injected Datasets:**
- Data config should reference the injected dataset directory
- The `embedding_config.yaml` in that directory specifies the actual pooling used
- Data loader should read and validate against this config

### 4.3 Validation Layer

**Location:** `data_provider/data_loader.py` or new `utils/config_validator.py`

**Purpose:** Ensure data config and model config (with suite overrides) align on embedding type.

**Implementation:**
```python
def validate_embedding_compatibility(
    data_config: dict,
    model_config: dict,
    dataset_path: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate that data and model configs are compatible for embeddings.
    
    Args:
        data_config: Data configuration dict
        model_config: Model configuration dict (may include suite overrides)
        dataset_path: Optional path to dataset directory (for injected datasets)
    
    Returns:
        (is_valid, error_message)
    """
    # Get expected pooling from model config
    model_pooling = model_config.get('postemb_pooling', 'cls')
    model_required = model_config.get('postemb_pooling_required', False)
    
    # Get data pooling
    data_pooling = data_config.get('postemb_pooling', 'cls')
    
    # For injected datasets, check embedding_config.yaml
    if dataset_path:
        embedding_config_path = os.path.join(dataset_path, 'embedding_config.yaml')
        if os.path.exists(embedding_config_path):
            with open(embedding_config_path, 'r') as f:
                embedding_config = yaml.safe_load(f)
            data_pooling = embedding_config.get('postemb_pooling', data_pooling)
    
    # Validation logic
    if model_required and model_pooling != data_pooling:
        return False, (
            f"Model requires postemb_pooling='{model_pooling}' "
            f"but data provides '{data_pooling}'. "
            f"Update data config or use compatible dataset."
        )
    
    # Warn if mismatch but not required
    if model_pooling != data_pooling:
        warnings.warn(
            f"Model expects postemb_pooling='{model_pooling}' "
            f"but data provides '{data_pooling}'. "
            f"Model will adapt, but results may differ."
        )
    
    return True, None
```

**Integration Points:**
1. **Data Factory** (`data_provider/data_factory.py`): Call validator before creating dataset
2. **Lightning Module** (`exp/exp_lightning.py`): Validate during setup
3. **Model Initialization** (`models/__init__.py`): Validate before model creation

### 4.4 Suite Override Handling

**Reference:** `runs/suite_executor.py` and `runs/lightning.py`

**Issue:** Suite overrides may change `postemb_pooling` in model config, creating mismatches.

**Solution:**
1. Apply suite overrides to model config (already done in `runs/lightning.py`)
2. Run validation **after** overrides are applied
3. Raise clear error if override creates incompatibility

**Example:**
```python
# In runs/lightning.py, after merging model_config overrides
is_valid, error_msg = validate_embedding_compatibility(
    data_configs,
    args.model_config,
    dataset_path=args.data_config.get('dataset_path')
)
if not is_valid:
    raise ValueError(f"Embedding compatibility error: {error_msg}")
```

## 5. Implementation Steps

### Phase 1: Embedding Infrastructure (Prerequisite)
**Reference:** [embeddings_upgrades.md Section 1](./embeddings_upgrades.md#1-data-provider-changes)

1. **Refactor Embedding Logic:**
   - Create `utils/text_embedding.py`
   - Move `convert_df_text_to_embeddings` logic there
   - Support `postemb_pooling` parameter ('cls', 'mean', 'none')
   - Update `data_loader.py` to use shared utility

2. **Update Data Provider:**
   - Add `postemb_pooling` parameter to `Heterogeneous_Dataset`
   - Modify embedding functions to support all pooling options
   - Update storage/padding logic for both 4D and 5D cases
   - Add shape validation in `load_embedding()`

### Phase 2: Configuration System
**Reference:** [embeddings_upgrades.md Section 2-3](./embeddings_upgrades.md#2-data-factory-changes)

1. **Add to Config Files:**
   - Add `postemb_pooling` to model config YAML files
   - Add `postemb_pooling` to data config YAML files
   - Add `postemb_pooling_required` flag to model configs

2. **Update Data Factory:**
   - Pass `postemb_pooling` from config to `Heterogeneous_Dataset`
   - Read `embedding_config.yaml` for injected datasets

3. **Create Validation Layer:**
   - Implement `validate_embedding_compatibility()`
   - Integrate into data factory and lightning module
   - Add validation after suite overrides

### Phase 3: Noise Injection Module
**Reference:** [PLAN.md Section 4](./PLAN.md#4-implementation-steps)

1. **Update Injection Config:**
   - Add `postemb_pooling` field
   - Add embedding model parameters

2. **Enhance Injector:**
   - Include `postemb_pooling` in dataset ID hash
   - Save `embedding_config.yaml`
   - Update metadata schema

3. **Implement Embedder:**
   - Use shared `utils/text_embedding.py`
   - Respect `postemb_pooling` from injection config
   - Validate output shapes
   - Save embedding config

4. **Update Utilities:**
   - Modify `generate_dataset_id()` to include embedding version
   - Add functions to read/write `embedding_config.yaml`

### Phase 4: Model Updates
**Reference:** [embeddings_upgrades.md Section 4](./embeddings_upgrades.md#4-model-changes)

1. **Update Models:**
   - Add shape detection (4D vs 5D)
   - Handle both cases gracefully
   - Add text self-attention when sequence available
   - Validate input shape matches expected `postemb_pooling`

2. **Model-Specific Changes:**
   - `lynx_mmitransformer`: Primary target for sequence dimension
   - Other models: Update as needed

### Phase 5: Testing & Validation

1. **Unit Tests:**
   - Embedding generation with all pooling types
   - Shape validation
   - Config validation
   - Dataset ID generation with embedding version

2. **Integration Tests:**
   - End-to-end noise injection with different pooling types
   - Config alignment validation
   - Model forward pass with 4D and 5D inputs
   - Suite override compatibility

3. **Backward Compatibility Tests:**
   - Old datasets without `embedding_config.yaml`
   - Old configs without `postemb_pooling` (defaults to 'cls')
   - Models handling both 4D and 5D inputs

## 6. Backward Compatibility

### 6.1 Default Behavior

- **Missing `postemb_pooling` in configs**: Defaults to `'cls'` (current behavior)
- **Missing `embedding_config.yaml`**: Assume `'cls'` for pre-computed embeddings
- **Old dataset IDs**: Continue to work, but new datasets include embedding version

### 6.2 Migration Path

1. **Existing Pre-computed Embeddings:**
   - If shape is `[hidden_dim]`, assume `postemb_pooling='cls'`
   - If shape is `[seq_len, hidden_dim]`, assume `postemb_pooling='none'`
   - Add `embedding_config.yaml` to existing datasets if needed

2. **Config Migration:**
   - Old configs without `postemb_pooling` work with default `'cls'`
   - Models can declare `postemb_pooling_required=true` to enforce explicit config

### 6.3 Breaking Changes

**None** - All changes are backward compatible with proper defaults and graceful degradation.

## 7. Error Messages & Diagnostics

### 7.1 Common Errors

1. **Embedding Shape Mismatch:**
   ```
   Error: Expected embedding shape [hidden_dim] for postemb_pooling='cls',
   but found [seq_len, hidden_dim]. Check embedding_config.yaml or regenerate embeddings.
   ```

2. **Config Mismatch:**
   ```
   Error: Model requires postemb_pooling='none' but data provides 'cls'.
   Update data config or use a dataset with sequence dimension embeddings.
   ```

3. **Missing Embedding Config:**
   ```
   Warning: embedding_config.yaml not found in dataset directory.
   Assuming postemb_pooling='cls' based on embedding shape.
   ```

### 7.2 Diagnostic Tools

Create `text_noise_injection/diagnostics.py`:
```python
def inspect_dataset_embeddings(dataset_path: str) -> dict:
    """Inspect embedding configuration and shapes in a dataset."""
    # Read embedding_config.yaml
    # Load sample embeddings
    # Report shape, pooling type, compatibility
    pass

def validate_dataset_model_compatibility(
    dataset_path: str,
    model_config_path: str
) -> dict:
    """Validate if dataset is compatible with model config."""
    # Load both configs
    # Run validation
    # Return detailed report
    pass
```

## 8. Documentation Updates

### 8.1 Update Existing Docs

- **README**: Document embedding versioning in dataset IDs
- **Config Documentation**: Explain `postemb_pooling` options
- **Model Docs**: Document which models require sequence dimension

### 8.2 New Documentation

1. **Embedding Versioning Guide**: How embedding types affect dataset IDs
2. **Config Alignment Guide**: How to ensure data and model configs match
3. **Migration Guide**: Upgrading existing datasets and configs

## 9. Performance Considerations

**Reference:** [embeddings_upgrades.md Section 7](./embeddings_upgrades.md#7-performance-considerations)

### 9.1 Memory Impact

- **With `postemb_pooling='none'`**: Significant memory increase (see embeddings_upgrades.md)
- **Noise Injection**: Additional memory for storing full sequence embeddings
- **Recommendation**: Use `postemb_pooling='none'` only when text self-attention is needed

### 9.2 Storage Impact

- **Embedding Files**: Larger `.pkl` files for sequence dimension
- **Metadata**: Additional `embedding_config.yaml` files
- **Dataset IDs**: Slightly longer IDs with embedding version

## 10. Future Enhancements

1. **Automatic Embedding Regeneration**: Re-generate embeddings if pooling type changes
2. **Embedding Cache**: Cache embeddings with version tags
3. **Multi-Pooling Support**: Generate multiple pooling types in one pass
4. **Embedding Compression**: Compress sequence dimension embeddings for storage

## 11. References

- [PLAN.md](./PLAN.md): Original text noise injection plan
- [embeddings_upgrades.md](./embeddings_upgrades.md): Embedding sequence dimension upgrade plan
- `data_provider/data_loader.py`: Current embedding implementation
- `models/lynx_mmitransformer.py`: Target model for sequence dimension support

