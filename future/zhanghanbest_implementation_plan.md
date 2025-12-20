# ZhangHanBest Model Implementation Plan

## Overview
ZhangHanBest is a multimodal benchmark model based on Zhang, Han, et al. (2025) from the Amazon team. It combines time series and text representations through a learned fusion mechanism.

## Architecture
- **Time Series Encoder**: Configurable unimodal time series model that outputs representations before the prediction block
- **Text Encoder**: BERT with average pooling over tokens (not CLS token) to create text representations
- **Residual Projection**: Projects text representation to time series representation space
- **Fusion**: Weighted element-wise addition: `w * text_representation + (1-w) * time_series_representation`
- **Prediction Head**: Uses fused representation to predict over `pred_len`

**Trainable Components**: Time series model, residual projection, prediction head (BERT can be frozen or trainable)

## Implementation Steps

### Phase 1: Extend Unimodal Models to Support Representation Extraction

#### 1.1 Design Pattern: `return_representations` Flag

Add a configurable option to unimodal models to return intermediate representations instead of predictions.

**Key Insight**: This avoids code duplication by allowing any unimodal model to be used within ZhangHanBest.

#### 1.2 Models to Extend

Based on the codebase and the Zhang, Han et al. paper, extend the following unimodal models:
- `PatchTST` (required)
- `DLinear` (required)
- `Sundial` (ideally)
- `TimeMoE` (ideally)

**Note**: iTransformer is NOT included as the Zhang, Han paper did not use it.

#### 1.3 Implementation Pattern for Each Model

Each model should support:
- **Config parameter**: `return_representations` (bool, default: False)
- **Representation extraction point**: Before the final projection/head layer
- **Shape consistency**: Representations should be extractable at a consistent interface point

**Example for PatchTST:**
```python
# In models/PatchTST.py

def forward(self, x, return_representations=False, **kwargs):
    """
    Forward pass.
    
    Args:
        x: Input time series [B, seq_len, C]
        return_representations: If True, return representations before prediction head
        **kwargs: Additional arguments
        
    Returns:
        If return_representations=False: predictions [B, pred_len, C]
        If return_representations=True: aggregated representation [B, d_model]
    """
    if self.decomposition:
        res_init, trend_init = self.decomp_module(x)
        res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)
        
        if return_representations:
            # Get representations from both branches
            res_repr = self.model_res.backbone(res_init)  # [B, C, d_model, patch_num]
            trend_repr = self.model_trend.backbone(trend_init)  # [B, C, d_model, patch_num]
            # Aggregate: mean over patches and channels -> [B, d_model]
            combined = (res_repr + trend_repr).mean(dim=(1, 3))  # [B, d_model]
            return combined
        
        res = self.model_res(res_init)
        trend = self.model_trend(trend_init)
        x = res + trend
        x = x.permute(0,2,1)
    else:
        x = x.permute(0,2,1)
        if return_representations:
            # Get representation from backbone before head
            repr = self.model.backbone(x)  # [B, C, d_model, patch_num]
            # Aggregate: mean over patches and channels -> [B, d_model]
            repr = repr.mean(dim=(1, 3))  # [B, d_model]
            return repr
        x = self.model(x)
        x = x.permute(0,2,1)
    return x
```

**Example for DLinear:**
```python
# In models/DLinear.py

def forward(self, x, return_representations=False, **kwargs):
    """
    Forward pass.
    
    Args:
        x: Input time series [B, seq_len, C]
        return_representations: If True, return representations before linear heads
        **kwargs: Additional arguments
        
    Returns:
        If return_representations=False: predictions [B, pred_len, C]
        If return_representations=True: aggregated representation [B, seq_len] -> aggregate to [B, 1] or use mean
    """
    seasonal_init, trend_init = self.decompsition(x)
    
    if return_representations:
        # Use decomposed components as representations
        # Aggregate seasonal and trend, then aggregate over channels and time
        combined = (seasonal_init + trend_init).mean(dim=(1, 2))  # [B] -> need to expand to [B, 1]
        # For consistency with other models, we can use a learnable aggregation or simple mean
        # Since DLinear doesn't have d_model, we'll need to define a representation dimension
        # Option: Use flattened seasonal+trend, then project to d_model if needed
        # For now, return mean over channels and time: [B]
        return combined.unsqueeze(-1)  # [B, 1] - will need dimension matching in fusion
        
    seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
    # ... rest of forward pass ...
```

**Representation Shape Considerations:**
- Different models will have different representation shapes
- PatchTST: Can extract from backbone [B, C, d_model, patch_num] -> aggregate to [B, d_model]
- DLinear: No explicit d_model, need to handle decomposition components
- Sundial/TimeMoE: These are foundation models - need to check their internal structure

**Design Decision**: For ZhangHanBest, we always aggregate representations to a single global representation per sample [B, d_model]:
- Aggregate over any per-variate, per-patch, or per-timestep dimensions
- Final representation is always [B, d_model] before fusion
- This matches the architecture requirement for aggregated representations only

#### 1.4 Testing Representation Extraction

For each modified model:
- Test that `return_representations=False` produces identical output to original
- Test that `return_representations=True` returns correct shape and requires_grad=True
- Ensure backward compatibility (default behavior unchanged)

### Phase 2: Text Input via Pre-computed Embeddings

#### 2.1 Embeddings as Input (Like TGTSF and lynx_film_raw)

**Design Decision**: ZhangHanBest will require embeddings as inputs, just like TGTSF and lynx_film_raw. This means:
- Text embeddings are pre-computed (not computed in-model)
- Embeddings are provided via the data loader (in `hetero_channel`, `hetero_general`, etc.)
- The centralized embedding module (see embeddings centralization plan) handles embedding computation
- Average pooling will be configured in the embedding module, not in the model

#### 2.2 Text Input Format

Based on dataset types:
- **Time-MMD and TTC**: Aggregate text representations only (no per-channel text)
- **Fidel-TS**: Dynamic aggregate text (ignore static channel descriptions, as per paper)

**Input shape**:
- Text embeddings provided as: [B, text_dim] where text_dim = embedding dimension (typically 768 for BERT)
- Aggregation happens during embedding computation (via centralized embedder module)
- Model receives already-aggregated text representation per sample

#### 2.3 Handling Text Embeddings in Forward Pass

The model will simply use the pre-computed embeddings directly:

```python
def forward(self, x, **kwargs):
    # Extract pre-computed text embeddings from kwargs
    # For Time-MMD/TTC: aggregate text is in hetero_general or similar
    # For Fidel-TS: dynamic aggregate text
    text_repr = self._get_text_embeddings(kwargs)  # [B, text_dim]
    
    # Rest of forward pass uses text_repr directly
    # ...
```

No need for in-model BERT encoding - embeddings come pre-computed from the data loader.

### Phase 3: Create Residual Projection Block

#### 3.1 Residual Block Design (Simplified)

According to the paper, residual projection is "more robust as it preserves the original temporal information" and maps text embeddings to the time series feature space. The residual projection should always use a residual connection.

**Architecture** (simplified based on paper description):
```python
class ResidualProjection(nn.Module):
    """
    Residual block to project text representation to time series representation space.
    
    Architecture (per paper):
    - Linear projection: text_dim -> ts_rep_dim
    - Residual connection (always used, with projection if dims differ)
    - Optional: LayerNorm, activation, dropout
    
    Note: Residual connection always used (as per paper recommendation).
    """
    def __init__(self, text_dim, ts_rep_dim, 
                 use_layer_norm=True, activation='gelu', dropout=0.1):
        super().__init__()
        self.text_dim = text_dim
        self.ts_rep_dim = ts_rep_dim
        
        # Main projection
        self.projection = nn.Linear(text_dim, ts_rep_dim)
        
        # Residual projection (if dims differ, need to project residual too)
        if text_dim != ts_rep_dim:
            self.residual_proj = nn.Linear(text_dim, ts_rep_dim)
        else:
            self.residual_proj = None
        
        # Optional components
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(ts_rep_dim)
        else:
            self.layer_norm = None
            
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = None
            
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, text_repr):
        """
        Args:
            text_repr: [B, text_dim] - text representation to project
        
        Returns:
            projected_repr: [B, ts_rep_dim] - projected to TS representation space
        """
        # Project input
        out = self.projection(text_repr)  # [B, ts_rep_dim]
        
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(text_repr)  # [B, ts_rep_dim]
        else:
            residual = text_repr  # [B, text_dim] = [B, ts_rep_dim] (same dim)
        
        out = out + residual  # Residual connection (always used)
        
        # Optional normalization, activation, dropout
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        
        if self.activation is not None:
            out = self.activation(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out
```

#### 3.2 Dimension Handling

- Text representation: [B, text_dim] (aggregated, from embeddings)
- TS representation: [B, d_model] (aggregated, from unimodal model)
- Residual projection: Maps [B, text_dim] -> [B, ts_rep_dim] where ts_rep_dim = d_model
- Both representations are already aggregated, so no broadcasting needed

### Phase 4: Implement Fusion Mechanism

#### 4.1 Weighted Element-wise Addition

Fuse text and time series representations:
```
fused = w * text_repr + (1-w) * time_series_repr
```

Where `w` can be either:
- **Fixed hyperparameter**: Configurable value in [0.0, 1.0]
- **Learned parameter**: Learned during training (with sigmoid to constrain to [0, 1])

#### 4.2 Implementation

```python
def fuse_representations(self, text_repr, ts_repr, fusion_weight):
    """
    Fuse text and time series representations via weighted addition.
    
    Args:
        text_repr: Text representation [B, d_model] (already projected)
        ts_repr: Time series representation [B, d_model] (already aggregated)
        fusion_weight: Weight w (scalar or tensor [1] if learned)
        
    Returns:
        fused_repr: Fused representation [B, d_model]
    """
    # Both inputs are already [B, d_model], so shapes match
    # If fusion_weight is learned, it's a parameter (sigmoid applied in forward if needed)
    fused = fusion_weight * text_repr + (1 - fusion_weight) * ts_repr
    return fused
```

#### 4.3 Fusion Weight Configuration

Support both fixed and learned weights:

```python
class Model(nn.Module):
    def __init__(self, configs):
        # ...
        
        # Fusion weight configuration
        fusion_weight_mode = getattr(configs, 'fusion_weight_mode', 'fixed')  # 'fixed' or 'learned'
        
        if fusion_weight_mode == 'learned':
            # Learned parameter (will be constrained via sigmoid in forward)
            self.fusion_weight_raw = nn.Parameter(torch.tensor(0.5))  # Initial value 0.5
            self.fusion_weight_mode = 'learned'
        else:
            # Fixed hyperparameter
            fusion_weight_value = getattr(configs, 'fusion_weight', 0.5)
            self.register_buffer('fusion_weight_raw', torch.tensor(fusion_weight_value))
            self.fusion_weight_mode = 'fixed'
    
    def forward(self, x, **kwargs):
        # ...
        
        # Get fusion weight
        if self.fusion_weight_mode == 'learned':
            # Apply sigmoid to constrain to [0, 1]
            fusion_weight = torch.sigmoid(self.fusion_weight_raw)
        else:
            fusion_weight = self.fusion_weight_raw
        
        fused_repr = self.fuse_representations(text_repr_proj, ts_repr_agg, fusion_weight)
        # ...
```

### Phase 5: Create Prediction Head

#### 5.1 Prediction Head Design (Aggregated Representations Only)

Map fused representation to prediction space. **Only aggregated representations are supported** (as per architecture requirements).

**Architecture**:
- Input: Fused representation [B, d_model] (aggregated, single representation per sample)
- Output: Predictions [B, pred_len, C] where C = number of channels/variates

Since input is aggregated, we need to map from a single d_model vector to predictions for all channels:
- Option A: Use same prediction for all channels: [B, d_model] -> [B, pred_len] -> broadcast to [B, pred_len, C]
- Option B: Predict per-channel: [B, d_model] -> [B, pred_len * C] -> reshape to [B, pred_len, C]

**Design Decision**: Use Option B (predict per-channel) for flexibility, but can simplify to Option A if needed.

#### 5.2 Implementation

```python
class PredictionHead(nn.Module):
    """
    Prediction head that maps fused representation to predictions.
    
    Input: Aggregated fused representation [B, d_model]
    Output: Predictions [B, pred_len, C]
    """
    def __init__(self, d_model, pred_len, n_variates, 
                 use_mlp=False, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.n_variates = n_variates
        
        output_dim = pred_len * n_variates  # Total output dimensions
        
        if use_mlp and hidden_dim is not None:
            # Multi-layer prediction head
            self.head = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            # Simple linear projection
            self.head = nn.Linear(d_model, output_dim)
    
    def forward(self, fused_repr):
        """
        Args:
            fused_repr: [B, d_model] - aggregated fused representation
            
        Returns:
            predictions: [B, pred_len, C] where C = n_variates
        """
        # [B, d_model] -> [B, pred_len * C]
        pred_flat = self.head(fused_repr)  # [B, pred_len * n_variates]
        
        # Reshape to [B, pred_len, n_variates]
        pred = pred_flat.view(-1, self.pred_len, self.n_variates)
        
        return pred
```

### Phase 6: Assemble ZhangHanBest Model

#### 6.1 Model Structure

```python
class Model(nn.Module):
    """
    ZhangHanBest Model: Multimodal fusion of time series and text.
    
    Architecture:
    1. Time series encoder (unimodal model, returns representations)
    2. Text encoder (BERT with average pooling)
    3. Residual projection (text -> TS representation space)
    4. Fusion (weighted addition)
    5. Prediction head (fused representation -> predictions)
    """
    def __init__(self, configs):
        super().__init__()
        
        # Store config
        self.configs = configs
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        
        # 1. Time series encoder (unimodal model)
        self.unimodal_model_type = getattr(configs, 'unimodal_model_type', 'PatchTST')
        self.ts_encoder = self._create_unimodal_encoder(configs)
        
        # Get representation dimension from TS encoder
        self.ts_rep_dim = getattr(configs, 'd_model', 512)  # Default from config
        
        # 2. Text input dimension (from pre-computed embeddings)
        self.text_dim = getattr(configs, 'input_text_dim', 768)  # Embedding dimension (typically 768 for BERT)
        
        # 3. Residual projection (always uses residual connection)
        self.residual_proj = ResidualProjection(
            text_dim=self.text_dim,
            ts_rep_dim=self.ts_rep_dim,
            use_layer_norm=getattr(configs, 'residual_proj_use_layer_norm', True),
            activation=getattr(configs, 'residual_proj_activation', 'gelu'),
            dropout=getattr(configs, 'residual_proj_dropout', 0.1)
        )
        
        # 4. Fusion weight (fixed or learned)
        fusion_weight_mode = getattr(configs, 'fusion_weight_mode', 'fixed')  # 'fixed' or 'learned'
        if fusion_weight_mode == 'learned':
            # Learned parameter (sigmoid applied in forward to constrain to [0, 1])
            initial_value = getattr(configs, 'fusion_weight_initial', 0.5)
            self.fusion_weight_raw = nn.Parameter(torch.tensor(initial_value))
            self.fusion_weight_mode = 'learned'
        else:
            # Fixed hyperparameter
            fusion_weight_value = getattr(configs, 'fusion_weight', 0.5)
            self.register_buffer('fusion_weight_raw', torch.tensor(fusion_weight_value))
            self.fusion_weight_mode = 'fixed'
        
        # 5. Prediction head
        self.prediction_head = PredictionHead(
            d_model=self.ts_rep_dim,
            pred_len=self.pred_len,
            n_variates=self.enc_in,
            use_mlp=getattr(configs, 'pred_head_use_mlp', False),
            hidden_dim=getattr(configs, 'pred_head_hidden_dim', None),
            dropout=getattr(configs, 'pred_head_dropout', 0.1)
        )
        
    
    def _create_unimodal_encoder(self, configs):
        """Create and configure unimodal time series encoder."""
        # Create unimodal model config (copy from configs, adjust as needed)
        unimodal_configs = self._prepare_unimodal_configs(configs)
        
        # Import and instantiate unimodal model
        if self.unimodal_model_type == 'PatchTST':
            from models.PatchTST import Model as PatchTSTModel
            return PatchTSTModel(unimodal_configs)
        elif self.unimodal_model_type == 'DLinear':
            from models.DLinear import Model as DLinearModel
            return DLinearModel(unimodal_configs)
        elif self.unimodal_model_type == 'Sundial':
            from models.Sundial import Model as SundialModel
            return SundialModel(unimodal_configs)
        elif self.unimodal_model_type == 'TimeMoE':
            from models.TimeMoE import Model as TimeMoEModel
            return TimeMoEModel(unimodal_configs)
        else:
            raise ValueError(f"Unsupported unimodal_model_type: {self.unimodal_model_type}")
    
    
    def forward(self, x, **kwargs):
        """
        Forward pass.
        
        Args:
            x: Time series input [B, seq_len, C]
            **kwargs: May include text inputs:
                - hetero_channel: Channel descriptions
                - hetero_general: General text
                - Or text as pre-computed embeddings
        
        Returns:
            predictions: [B, pred_len, C]
        """
        # 1. Get time series representations (aggregated)
        # Call unimodal model with return_representations=True
        ts_repr = self.ts_encoder(x, return_representations=True)  # [B, d_model] (aggregated)
        
        # 2. Get text representation from pre-computed embeddings
        text_repr = self._get_text_embeddings(kwargs)  # [B, text_dim]
        
        # 3. Project text to TS representation space (residual projection)
        text_repr_proj = self.residual_proj(text_repr)  # [B, d_model]
        
        # 4. Get fusion weight (fixed or learned)
        if self.fusion_weight_mode == 'learned':
            fusion_weight = torch.sigmoid(self.fusion_weight_raw)  # Constrain to [0, 1]
        else:
            fusion_weight = self.fusion_weight_raw  # Fixed value
        
        # 5. Fuse representations (both are [B, d_model])
        fused_repr = self.fuse_representations(text_repr_proj, ts_repr, fusion_weight)  # [B, d_model]
        
        # 6. Prediction (aggregated representation -> predictions for all channels)
        predictions = self.prediction_head(fused_repr)  # [B, pred_len, C]
        
        # 7. Denormalize predictions (if TS model normalized internally)
        predictions = self._denormalize_predictions(predictions, x)
        
        return predictions
    
    def _get_text_embeddings(self, kwargs):
        """
        Extract pre-computed text embeddings from kwargs.
        
        For Time-MMD/TTC: Aggregate text is in hetero_general or similar
        For Fidel-TS: Dynamic aggregate text (ignore static channel descriptions)
        
        Returns:
            text_repr: [B, text_dim] - pre-computed text embeddings
        """
        # Extract from kwargs (format depends on data loader)
        # For Time-MMD: typically in hetero_general or similar field
        # For Fidel-TS: dynamic aggregate text
        # Implementation depends on actual data loader output format
        # Should return [B, text_dim] tensor
        
        # Placeholder - actual implementation depends on data loader format
        # Example:
        # if 'hetero_general' in kwargs:
        #     return kwargs['hetero_general']  # Assuming [B, text_dim]
        # elif 'text_embeddings' in kwargs:
        #     return kwargs['text_embeddings']
        # else:
        #     raise ValueError("Text embeddings not found in kwargs")
        
        raise NotImplementedError("Text embedding extraction - implement based on data loader format")
    
    def _denormalize_predictions(self, predictions, x):
        """
        Denormalize predictions if TS model normalized internally.
        
        Args:
            predictions: [B, pred_len, C] - normalized predictions
            x: [B, seq_len, C] - original input (for normalization params)
        
        Returns:
            predictions: [B, pred_len, C] - denormalized predictions
        """
        # Track if TS model normalized internally
        # Extract normalization parameters from TS model if available
        # Apply inverse normalization
        
        # For now, return as-is (denormalization logic depends on TS model type)
        # This should be implemented based on how each unimodal model handles normalization
        # when return_representations=True
        
        # Note: If TS model returns normalized representations, predictions are normalized
        # Need to denormalize using same scheme as TS model
        
        return predictions  # Placeholder - implement denormalization logic
    
    def fuse_representations(self, text_repr, ts_repr, fusion_weight):
        """Fuse text and time series representations."""
        fused = fusion_weight * text_repr + (1 - fusion_weight) * ts_repr
        return fused
```

#### 6.2 Text Input Format

**Design Decision**: Text inputs are always pre-computed embeddings (like TGTSF and lynx_film_raw):
- Embeddings computed by centralized embedding module (see embeddings centralization plan)
- Average pooling configured in embedding module (not in model)
- Model receives embeddings via data loader in kwargs
- For Time-MMD/TTC: Aggregate text representations only
- For Fidel-TS: Dynamic aggregate text (ignore static channel descriptions, as per paper)

#### 6.3 Normalization Handling

**Design Decision**: Let the time series model handle its own normalization internally:
- Pass raw input `x` to TS encoder
- TS encoder handles normalization (if it does) when extracting representations
- Representations returned are in normalized space
- Prediction head outputs normalized predictions
- **Denormalize predictions after prediction head** using the same normalization scheme as the TS model

**Implementation**:
- Track normalization parameters from TS model (if applicable)
- Store normalization scheme type (e.g., 'use_norm', 'revin', 'none')
- In `_denormalize_predictions()`, apply inverse normalization based on scheme
- Extract normalization parameters from input `x` (mean, std, etc.)
- Apply inverse transformation to predictions

### Phase 7: Model Configuration

#### 7.1 Configuration File: `model_configs/general/ZhangHanBest.yaml`

```yaml
# ZhangHanBest Model Configuration
# Based on Zhang, Han, et al. (2025) - Amazon team

model: ZhangHanBest

# Unimodal time series model configuration
unimodal_model_type: "PatchTST"  # Options: PatchTST, DLinear, Sundial, TimeMoE
# Unimodal model configs will be inherited/merged from corresponding model config
# Or can be specified here to override

# Time series encoder parameters (passed to unimodal model)
seq_len: 96
pred_len: 96
enc_in: 1  # Number of channels/variates
d_model: 512  # Representation dimension (should match unimodal model's d_model)
use_norm: True  # If using iTransformer, this controls normalization
# ... other unimodal model specific params ...

# Text input configuration
input_text_dim: 768  # Dimension of input text embeddings (from centralized embedder)
# Text embeddings are pre-computed by centralized embedding module (like TGTSF/lynx_film_raw)
# Embedding aggregation method (CLS/average/none) configured in embeddings config

# Residual projection configuration
# Note: Residual connection is ALWAYS used (as per paper)
residual_proj_use_layer_norm: True
residual_proj_activation: "gelu"  # Options: gelu, relu, none
residual_proj_dropout: 0.1

# Fusion configuration
fusion_weight_mode: "fixed"  # Options: "fixed" or "learned"
fusion_weight: 0.5  # Weight w in w*text + (1-w)*ts (range: 0.0 to 1.0)
# If mode="learned", fusion_weight is initial value, then learned during training
fusion_weight_initial: 0.5  # Initial value for learned fusion weight (if mode="learned")

# Prediction head configuration
pred_head_use_mlp: False  # Whether to use MLP instead of single linear layer
pred_head_hidden_dim: null  # If use_mlp=True, specify hidden dimension
pred_head_dropout: 0.1

# Task type
task: TGTSF  # Text-guided time series forecasting
individual: False

# Other parameters
dropout: 0.1
```

#### 7.2 Configuration Inheritance

Consider inheriting base configs from unimodal models:
- Load base config (e.g., `iTransformer.yaml`)
- Override/add ZhangHanBest-specific parameters
- Ensure compatibility

### Phase 8: Testing and Validation

#### 8.1 Unit Tests

1. **Text Encoder with Average Pooling**:
   - Test average pooling produces correct shape
   - Test masking of padding tokens
   - Compare with CLS token output (should be different)

2. **Residual Projection**:
   - Test dimension matching
   - Test residual connection (when enabled)
   - Test gradient flow

3. **Fusion Mechanism**:
   - Test weighted addition
   - Test shape broadcasting
   - Test edge cases (w=0, w=1)

4. **Prediction Head**:
   - Test output shape correctness
   - Test with different input representation shapes

5. **Full Model Forward**:
   - Test end-to-end forward pass
   - Test gradient computation
   - Test with different batch sizes

#### 8.2 Integration Tests

1. **Data Loader Integration**:
   - Test with TimeMMD datasets
   - Test with heterogeneous datasets
   - Test text input format handling

2. **Training Integration**:
   - Test training loop compatibility
   - Test checkpoint saving/loading
   - Test with different optimizers

3. **Evaluation Integration**:
   - Test with evaluation scripts
   - Test metric computation

#### 8.3 Backward Compatibility

- Ensure modified unimodal models maintain backward compatibility
- Test existing code using unimodal models still works
- Test that `return_representations=False` (default) produces identical results

### Phase 9: Documentation

#### 9.1 Code Documentation

- Docstrings for all classes and methods
- Inline comments for complex logic
- Type hints where applicable

#### 9.2 Usage Documentation

- README section for ZhangHanBest
- Configuration examples
- Training examples
- How to use different unimodal models

#### 9.3 Architecture Documentation

- Diagram of model architecture
- Explanation of design decisions
- Comparison with baseline models

## Implementation Order

### Phase 1: Foundation (Week 1)
1. Extend PatchTST to support `return_representations` (aggregated to [B, d_model])
2. Extend DLinear to support `return_representations` (aggregated representation)
3. Test PatchTST and DLinear modifications
4. (Optional) Extend Sundial and TimeMoE to support `return_representations`

### Phase 2: Core Components (Week 2)
5. Implement ResidualProjection module
6. Implement PredictionHead module
7. Test individual components

### Phase 3: Model Assembly (Week 3)
8. Create ZhangHanBest model class
9. Implement forward pass logic
10. Handle text input format
11. Handle normalization

### Phase 4: Configuration & Integration (Week 4)
12. Create model configuration file
13. Integrate with data loaders
14. Test with actual datasets

### Phase 5: Testing & Refinement (Week 5)
15. Comprehensive testing
16. Fix bugs and edge cases
17. Optimize performance
18. Documentation

### Phase 6: Extension (Optional, Week 6+)
19. Extend Sundial and TimeMoE if not done in Phase 1
20. Add more fusion mechanisms (concatenation, attention, etc.)
21. Experiment with different aggregation strategies for TS representations

## Key Design Decisions Summary

1. **Representation Extraction**: Add `return_representations` flag to unimodal models (PatchTST, DLinear, Sundial, TimeMoE) - always return aggregated [B, d_model]
2. **Text Input**: Pre-computed embeddings (like TGTSF/lynx_film_raw), configured via centralized embedding module
3. **Text Aggregation**: For Time-MMD/TTC use aggregate text only; for Fidel-TS use dynamic aggregate text (ignore static channel descriptions)
4. **Residual Projection**: Always uses residual connection (simplified design per paper)
5. **Fusion Weight**: Support both fixed hyperparameter and learned parameter (with sigmoid constraint)
6. **Prediction Head**: Only handles aggregated representations [B, d_model] -> [B, pred_len, C]
7. **Normalization**: Let TS model handle normalization internally, denormalize predictions after prediction head

## Potential Challenges and Solutions

### Challenge 1: Representation Shape Inconsistency
**Problem**: Different unimodal models produce different representation shapes
**Solution**: 
- Define aggregation strategy per model type
- Use adapter layers if needed
- Document expected shapes

### Challenge 2: Text Embedding Extraction
**Problem**: Need to extract pre-computed embeddings from data loader kwargs
**Solution**:
- Coordinate with centralized embedding module implementation
- Define standard format for embeddings in kwargs
- Handle Time-MMD vs Fidel-TS format differences

### Challenge 3: Normalization Complexity
**Problem**: Unimodal models have different normalization schemes, need to denormalize after prediction
**Solution**:
- Track normalization scheme type per model
- Extract normalization parameters from input
- Implement denormalization logic per scheme (use_norm, revin, none)
- Test with each supported unimodal model

### Challenge 4: DLinear Representation Dimension
**Problem**: DLinear doesn't have explicit d_model dimension, needs special handling
**Solution**:
- Use flattened decomposition components as representation
- May need projection layer to match d_model dimension
- Or define d_model for DLinear in config

## References

- Zhang, Han, et al. (2025) - Amazon team (paper reference when available)
- BERT: Devlin et al. (2018) "BERT: Pre-training of Deep Bidirectional Transformers"
- iTransformer: Liu et al. (2024) "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
- TGTSF: Framework for text-guided time series forecasting

## Notes

- This implementation plan assumes the codebase structure as of the current state
- Some details may need adjustment based on actual data formats and requirements
- The plan is designed to be modular and extensible
- Consider performance optimizations (e.g., pre-computing text embeddings) for large-scale training

