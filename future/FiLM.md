# lynx-film: FiLM-Modulated Residual Learning Plan

## Overview
`lynx-film` is a new multimodal time-series forecasting model that extends the `lynx` architecture. Like `lynx`, it learns a residual correction to a frozen unimodal baseline (defaulting to `iTransformer`). 

**Key Difference**: Instead of using the Cross-Attention mechanism (TGTSF) for the residual learner, `lynx-film` uses a modified `iTransformer` whose attention blocks are modulated by text representations using **FiLM (Feature-wise Linear Modulation)**.

## Architecture

The model consists of three main components:
1.  **Frozen Unimodal Baseline**: A pre-trained `iTransformer` (frozen).
2.  **FiLM Generator (Controller)**: A shallow encoder that projects text embeddings into modulation parameters ($\gamma, \beta$).
3.  **Residual Learner (`iTransformerFilm`)**: A trainable `iTransformer` variant where encoder layers are modulated by the FiLM parameters.

### Data Flow
1.  **Input**: Time series $X$, Text Embeddings $E_{text}$.
2.  **Unimodal Branch**: $Y_{base} = \text{FrozenModel}(X)$.
3.  **FiLM Generation**:
    -   Text embeddings (aligned with channels/variates) are passed through the **FiLM Generator**.
    -   Output: $\gamma$ (scale) and $\beta$ (shift) parameters for each attention block.
4.  **Residual Branch**:
    -   $X$ is processed by `iTransformerFilm`.
    -   Inside each Transformer block, feature maps are modulated: $F' = \gamma \cdot F + \beta$.
    -   Output: $Y_{residual}$.
5.  **Combination**: $Y_{final} = Y_{base} + Y_{residual}$.

## Detailed Component Design

### 1. FiLM Generator (Shallow Encoder)
The text representation needs to be mapped to the dimensions of the internal feature maps of the `iTransformer` blocks.
-   **Input**: `text_emb` of shape `[Batch, n_vars, text_dim]`.
-   **Architecture**: A simple MLP (Multi-Layer Perceptron).
    -   `Linear(text_dim -> hidden_dim)`
    -   `Activation (ReLU/GELU)`
    -   `Linear(hidden_dim -> d_model * 2)` (for $\gamma$ and $\beta$)
-   **Output**: Modulation parameters of shape `[Batch, n_vars, d_model, 2]`.
-   **Scope**: Can be shared across layers or separate per layer. For `lynx-film`, we will use **layer-specific generators** or a shared generator with layer-specific heads to allow different modulation at different depths.

### 2. `EncoderLayerFilm`
A modified version of the standard Transformer `EncoderLayer`.
-   **Inputs**: Feature map `x`, Attention Mask `attn_mask`, FiLM params `gamma`, `beta`.
-   **Modulation Point**: Applied after the LayerNorm operations, following the standard FiLM approach in residual networks.
    -   Standard Post-Norm Transformer Block:
        ```python
        # Self Attention
        _x = x
        x = self.norm1(x + self.dropout(self.attention(x...)))
        
        # APPLY FiLM HERE
        x = x * gamma1 + beta1
        
        # FFN
        y = x
        x = self.norm2(x + self.dropout(self.ffn(x)))
        
        # APPLY FiLM HERE
        x = x * gamma2 + beta2
        ```
    -   *Note*: Using `gamma + 1` is often more stable for initialization (starts as identity).

### 3. `iTransformerFilm` (Residual Model)
This replaces the TGTSF component in the original `lynx`.
-   Inherits the inverted embedding strategy of `iTransformer` (Time steps -> Embedding dimension, Variates -> Tokens).
-   **Forward Pass**:
    -   Accepts `x` (time series) and `text_emb`.
    -   Generates FiLM params from `text_emb`.
    -   Passes `x` and params through the stack of `EncoderLayerFilm`.
    -   Projects output to prediction length.

### 4. `LynxFilm` Wrapper
-   Maintains the interface of `lynx`.
-   Manages the `unimodal_wrapper` (normalization/denormalization, frozen forward pass).
-   Manages the `iTransformerFilm` (residual forward pass).
-   Combines outputs.

## Implementation Steps

### Step 1: Define FiLM Components
Create a new file `layers/FiLM_layers.py` (or add to `models/lynx_film.py`) containing:
-   `FiLMGenerator`: The MLP module.
-   `EncoderLayerFilm`: The modified encoder layer.
-   `EncoderFilm`: The stack of layers that propagates FiLM params.

```python
class FiLMGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim * 2) # gamma, beta
        )
        
    def forward(self, x):
        # x: [B, N, D]
        params = self.net(x)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        return gamma, beta # [B, N, output_dim]
```

### Step 2: Implement `iTransformerFilm`
Create `models/lynx_film_components.py` (or similar) to house the modified iTransformer.
-   Copy `iTransformer` logic but replace `Encoder` with `EncoderFilm`.
-   Ensure `forward` accepts text embeddings.

### Step 3: Implement `LynxFilm` Model
Create `models/lynx_film.py`.
-   Class `Model(nn.Module)`
-   **Init**:
    -   Load configs.
    -   Initialize `UnimodalModelWrapper` (loads frozen iTransformer).
    -   Initialize `text_encoder` (from TGTSF layers, to get embeddings).
    -   Initialize `residual_model` (`iTransformerFilm`).
-   **Forward**:
    -   `x_norm = normalize(x)`
    -   `pred_base = frozen_model(x_norm)`
    -   `text_emb = text_encoder(news, description)`
    -   `pred_residual = residual_model(x_norm, text_emb)`
    -   `final = denormalize(pred_base + pred_residual)`

### Step 4: Configuration
-   Add `lynx_film` to `model_dict`.
-   Create `model_configs/general/lynx_film.yaml` with relevant hyperparameters (FiLM hidden dim, layers, etc.).

## Advantages
-   **Parameter Efficiency**: The shallow encoder adds minimal parameters compared to full cross-attention.
-   **Feature-wise Control**: Text can specifically upweight/downweight specific variate features or time-step features (depending on embedding dimension) in the residual learner.
-   **Modularity**: Reuses the robust `iTransformer` backbone.

