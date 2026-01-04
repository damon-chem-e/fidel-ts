# MM-TSFlib Chimera Model Implementation Plan for Fidel-TS

This document provides a detailed analysis of the MM-TSFlib repository's multimodal forecasting implementation (ChimeraTransformer) and a comprehensive plan to integrate it into the fidel-ts framework.

---

## Table of Contents

1. [MM-TSFlib Overview](#1-mm-tsflib-overview)
2. [ChimeraTransformer Architecture](#2-chimeratransformer-architecture)
3. [Key Components Analysis](#3-key-components-analysis)
4. [Comparison with Fidel-TS](#4-comparison-with-fidel-ts)
5. [Implementation Plan](#5-implementation-plan)
6. [File Structure](#6-file-structure)
7. [Implementation Details](#7-implementation-details)
8. [Testing Strategy](#8-testing-strategy)
9. [Configuration](#9-configuration)
10. [Timeline and Milestones](#10-timeline-and-milestones)

---

## 1. MM-TSFlib Overview

MM-TSFlib is an open-source library for multimodal time series forecasting based on the [Time-MMD](https://github.com/AdityaLab/Time-MMD/) dataset. The repository extends the Time-Series-Library with multimodal capabilities.

### 1.1 Repository Structure

```
MM-TSFlib/
├── models/
│   ├── ChimeraTransformer.py    # Main multimodal model (Chimera)
│   ├── iTransformer.py          # Base time series model
│   └── ... (20+ other baseline models)
├── layers/
│   ├── CrossAttention.py        # Cross-modal attention mechanism
│   ├── GatingMechanism.py       # Feature gating for fusion
│   ├── Embed.py                 # Time series embedding layers
│   ├── Transformer_EncDec.py    # Standard transformer components
│   └── SelfAttention_Family.py  # Attention layer variants
├── data_provider/
│   ├── data_loader.py           # Dataset classes with text support
│   └── data_factory.py          # Data loading factory
├── exp/
│   └── exp_long_term_forecasting.py  # Training/evaluation with LLM
├── run.py                       # Main entry point
└── utils/
    └── ...                      # Metrics, tools, etc.
```

### 1.2 Design Philosophy

MM-TSFlib implements **two fusion approaches**:

1. **Late Fusion (Default for most models)**: Time series predictions are combined with LLM-derived predictions using a weighted sum with a fixed `prompt_weight` (default: 0.01).

2. **Token-Level Fusion (ChimeraTransformer)**: Deep integration using cross-attention between time series tokens (from iTransformer) and text embeddings, with learned gating mechanisms.

---

## 2. ChimeraTransformer Architecture

The ChimeraTransformer is the most sophisticated multimodal model in MM-TSFlib, implementing token-level fusion.

### 2.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ChimeraTransformer Architecture                       │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐     ┌─────────────────┐
                    │   Text Input    │     │  Time Series    │
                    │   (from LLM)    │     │  Input [B,T,N]  │
                    └────────┬────────┘     └────────┬────────┘
                             │                       │
                             ▼                       ▼
                    ┌─────────────────┐     ┌─────────────────┐
                    │  Text Self-Attn │     │  RevIN Norm     │
                    │  (Optional)     │     │                 │
                    │  [B, L, d_llm]  │     └────────┬────────┘
                    └────────┬────────┘              │
                             │                       ▼
                             │              ┌─────────────────┐
                             │              │  iTransformer   │
                             │              │  Embedding      │
                             │              │  (Inverted)     │
                             │              └────────┬────────┘
                             │                       │
                             │                       ▼
                             │              ┌─────────────────┐
                             │              │  iTransformer   │
                             │              │  Encoder        │
                             │              │  [B, V, d_model]│
                             │              └────────┬────────┘
                             │                       │
                             ▼                       │
                    ┌─────────────────────────────────┐
                    │       Cross-Attention           │
                    │   (MultiheadLatentAttention)    │
                    │   Q: TS features (d_model)      │
                    │   K,V: Text features (d_llm)    │
                    │   Output: [B, V, d_latent]      │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │    Post-Fusion Self-Attention   │
                    │         (Optional)              │
                    │       [B, V, d_latent]          │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │       Feature Gate              │
                    │   Projects d_latent → d_model   │
                    │   Computes gate α ∈ [0,1]       │
                    │   Output = α·TS + (1-α)·Fused   │
                    └────────────────┬────────────────┘
                                     │
               ┌─────────────────────┴───────────────────┐
               │                                         │
               ▼                                         │
    ┌──────────────────┐                                │
    │ Final Self-Attn  │◄───────────────────────────────┘
    │   (Optional)     │    (Skip connection for
    └────────┬─────────┘     raw_skip_dual_gate)
             │
             ▼
    ┌──────────────────┐
    │   Projection     │
    │   [B, pred_len, N]│
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  RevIN Denorm    │
    └────────┬─────────┘
             │
             ▼
        Predictions
```

### 2.2 Key Architectural Choices

1. **iTransformer as Backbone**: Uses inverted attention (variates as tokens, not time steps) for effective multivariate modeling.

2. **Cross-Attention Direction**: Time series features are queries, text features are keys/values. This allows each time series variate to attend to relevant text information.

3. **Latent Space Fusion**: Both modalities are projected to a shared latent space (`d_latent`) before attention, enabling dimension-agnostic fusion.

4. **Learned Gating**: Instead of fixed fusion weights, the model learns to dynamically balance TS vs. fused features based on input content.

5. **Multiple Skip Connection Variants**:
   - `post_attn_skip`: Skip from encoder output (default)
   - `raw_skip`: Skip from embedding output (before encoder)
   - `raw_skip_dual_gate`: Two gating mechanisms

---

## 3. Key Components Analysis

### 3.1 MultiheadLatentAttention (`layers/CrossAttention.py`)

**Purpose**: Performs cross-modal attention in a shared latent space.

```python
class MultiheadLatentAttention(nn.Module):
    """
    Projects Q (from TS) and K,V (from text) to shared latent space.
    
    Input:
        queries: [B, V, d_model]     # Time series features
        keys: [B, L, d_llm]          # Text features
        values: [B, L, d_llm]        # Text features
    
    Output:
        fused: [B, V, d_latent]      # Fused in latent space
    """
    def __init__(self, query_dim, key_dim, latent_dim, num_heads, dropout):
        # Project both modalities to latent_dim
        self.q_proj = nn.Linear(query_dim, latent_dim)
        self.k_proj = nn.Linear(key_dim, latent_dim)
        self.v_proj = nn.Linear(key_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
```

**Key Design**: 
- Separate projections for each modality allow different input dimensions
- Standard scaled dot-product attention after projection
- Output projection maintains latent dimensionality

### 3.2 FeatureGate (`layers/GatingMechanism.py`)

**Purpose**: Learns to dynamically balance time series vs. fused features.

**Gate Types Supported**:

| Gate Type | Formula | Shape of α | Description |
|-----------|---------|------------|-------------|
| `mlp` | σ(MLP([F,T])) | [B,V,d_model] | Full MLP with hidden layer |
| `linear` | σ(W[F,T]+b) | [B,V,d_model] | Single linear layer |
| `linear_norm` | σ(LN(W[F,T]+b)) | [B,V,d_model] | Linear + LayerNorm |
| `per_token_scalar` | σ(W[F,T]+b) | [B,V,1] | One scalar per variate |
| `global_scalar` | σ(W·mean([F,T])+b) | [B,1,1] | One scalar per batch |

**Gating Formula**:
$$
O = \alpha \odot T + (1 - \alpha) \odot F'
$$

Where:
- $F$ = Fused latent features
- $F'$ = Projected fused features (d_latent → d_model)
- $T$ = Original time series features
- $\alpha$ = Learned gate value

### 3.3 Text Embedding Pipeline (`exp/exp_long_term_forecasting.py`)

The experiment class handles LLM loading and text embedding:

```python
# 1. Load LLM (BERT, GPT2, LLaMA, etc.) - FROZEN
self.llm_model = BertModel.from_pretrained('bert-base-uncased')
for param in self.llm_model.parameters():
    param.requires_grad = False

# 2. Create prompt from text
prompt = f"<|start_prompt|>Make predictions based on: {text}<|end_prompt|>"

# 3. Tokenize and embed
tokens = self.tokenizer(prompt, padding=True, truncation=True)
embeddings = self.llm_model.get_input_embeddings()(tokens)

# 4. Optionally pass through full LLM
if use_fullmodel:
    embeddings = self.llm_model(inputs_embeds=embeddings).last_hidden_state

# 5. Pass to ChimeraTransformer
outputs, gate_value, final_gate_value = model(x, x_mark, ..., text_embeddings=embeddings)
```

### 3.4 Data Loading with Text (`data_provider/data_loader.py`)

Text is stored alongside time series in CSV files:

```python
# CSV columns: date, <features>, <target>, prior_history_avg, start_date, end_date, Final_Search_<N>
# Final_Search_<N> contains pre-matched text for each time step

def get_text(self, indices):
    """Retrieve text for given sample indices."""
    s_begins = indices % self.tot_len
    s_ends = s_begins + self.seq_len
    text = np.array([self.text[s_end] for s_end in s_ends])
    return text
```

### 3.5 Training Loop with Gate Regularization

```python
# Forward pass
outputs, gate_value, final_gate_value = model(batch_x, ..., prompt_emb)

# Compute main loss
main_loss = criterion(outputs, batch_y)

# Add gate regularization (optional, pushes gate toward 0.5)
if gate_regularization_lambda > 0:
    reg_loss = lambda * mean(|gate_value - 0.5|)
    total_loss = main_loss + reg_loss
```

---

## 4. Comparison with Fidel-TS

### 4.1 Existing Multimodal Models in Fidel-TS

| Model | Fusion Type | Text Source | Key Difference |
|-------|-------------|-------------|----------------|
| TimeLLM | Reprogramming | Dynamic prompts | TS → LLM space, not cross-attention |
| TimeCMA | Cross-modal | Pre-computed embeddings | Fixed fusion, no gating |
| TGTSF | Text-guided | Retrieved text | Concatenation-based fusion |
| Lynx | FiLM + Cross | Pre-computed embeddings | Different architecture |

### 4.2 Fidel-TS Data Pipeline

Fidel-TS already supports:
- Pre-computed LLM embeddings via `LLMEmbeddingProvider`
- Channel descriptions via `hetero_channel`
- FidelTSEmbeddingLoader for cached embeddings
- Multiple embedding aggregation strategies

### 4.3 Key Differences to Bridge

| Aspect | MM-TSFlib | Fidel-TS | Adaptation Needed |
|--------|-----------|----------|-------------------|
| Text Input | Raw text → tokenized | Pre-computed embeddings | Use existing provider |
| LLM Loading | In experiment class | LLMRegistry/Provider | Reuse existing infra |
| Config System | argparse | YAML configs | Create model config |
| Data Format | CSV with text column | Universal_Dataset | Extend data loader |
| Training | Custom exp class | Lightning module | Adapt to Lightning |

---

## 5. Implementation Plan

### 5.1 Phase 1: Core Model Implementation (Priority: High)

**Goal**: Implement ChimeraTransformer model and layers.

**Tasks**:
1. Create `layers/ChimeraTransformer_layers.py`:
   - `MultiheadLatentAttention` class
   - `FeatureGate` class with all gate types

2. Create `models/ChimeraTransformer.py`:
   - Main model class following fidel-ts patterns
   - Support for pre-computed embeddings (like TimeCMA)
   - Integration with existing `Normalize` layer

### 5.2 Phase 2: Configuration (Priority: High)

**Tasks**:
1. Create `model_configs/general/ChimeraTransformer.yaml`
2. Add to model registry in `models/__init__.py`
3. Define default hyperparameters

### 5.3 Phase 3: Data Integration (Priority: Medium)

**Tasks**:
1. Extend `Universal_Dataset` to support Chimera's text requirements
2. Ensure compatibility with `LLMEmbeddingProvider`
3. Add text data fields to Time-MMD data configs

### 5.4 Phase 4: Training Integration (Priority: Medium)

**Tasks**:
1. Add gate regularization loss option to Lightning training
2. Create experiment-specific training logic if needed
3. Support staged training (TS-only → full model)

### 5.5 Phase 5: Testing & Validation (Priority: Medium)

**Tasks**:
1. Unit tests for new layers
2. Integration tests with sample data
3. Validate against MM-TSFlib results on Time-MMD

---

## 6. File Structure

### 6.1 New Files to Create

```
fidel-ts/
├── layers/
│   └── ChimeraTransformer_layers.py     # NEW: Cross-attention + Gating
│
├── models/
│   └── ChimeraTransformer.py            # NEW: Main model
│
├── model_configs/
│   └── general/
│       └── ChimeraTransformer.yaml      # NEW: Model config
│
├── data_configs/
│   └── time_mmd/
│       └── <existing configs>           # MODIFY: Add text paths
│
└── tests/
    └── test_chimera.py                  # NEW: Unit tests
```

### 6.2 Files to Modify

```
fidel-ts/
├── models/__init__.py                   # Add ChimeraTransformer import
├── data_provider/data_loader.py         # Add text retrieval methods
└── runs/lightning.py                    # Add gate regularization option
```

---

## 7. Implementation Details

### 7.1 `layers/ChimeraTransformer_layers.py`

```python
"""
ChimeraTransformer Layers for Fidel-TS.

Implements the cross-modal attention and gating mechanisms from MM-TSFlib's
ChimeraTransformer for token-level multimodal fusion.

Components:
    - MultiheadLatentAttention: Cross-modal attention in latent space
    - FeatureGate: Learned gating for fusion control

Reference:
    MM-TSFlib: https://github.com/AdityaLab/MM-TSFlib
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiheadLatentAttention(nn.Module):
    """
    Multihead Latent Cross-Attention for token-level multimodal fusion.
    
    Projects query (time series) and key/value (text) features into a shared
    latent space where cross-attention is performed.
    
    Args:
        query_dim: Dimension of query features (d_model from TS encoder)
        key_dim: Dimension of key/value features (d_llm from text encoder)
        latent_dim: Dimension of shared latent space
        num_heads: Number of attention heads
        dropout: Dropout rate for attention weights
    
    Input:
        queries: Time series features [B, V, query_dim]
        keys: Text features [B, L, key_dim]
        values: Text features [B, L, key_dim]
    
    Output:
        Fused features in latent space [B, V, latent_dim]
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        latent_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Validate dimensions
        assert latent_dim % num_heads == 0, \
            f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})"
        
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.head_dim = latent_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projection layers to latent space
        self.q_proj = nn.Linear(query_dim, latent_dim)
        self.k_proj = nn.Linear(key_dim, latent_dim)
        self.v_proj = nn.Linear(key_dim, latent_dim)
        
        # Output projection
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform cross-attention in latent space.
        
        Args:
            queries: TS features [B, V, query_dim]
            keys: Text features [B, L, key_dim]
            values: Text features [B, L, key_dim]
            attention_mask: Optional mask [B, V, L]
        
        Returns:
            Fused features [B, V, latent_dim]
        """
        batch_size = queries.shape[0]
        
        # Project to latent space
        q = self.q_proj(queries)  # [B, V, latent_dim]
        k = self.k_proj(keys)     # [B, L, latent_dim]
        v = self.v_proj(values)   # [B, L, latent_dim]
        
        # Reshape for multihead attention
        # [B, seq, latent] -> [B, heads, seq, head_dim]
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [B, heads, V, head_dim]
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.latent_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output


class FeatureGate(nn.Module):
    """
    Learned gating mechanism for multimodal fusion.
    
    Projects fused features from latent to TS dimension and computes
    gate values to balance fused vs. original time series features.
    
    Gate Types:
        - 'mlp': Two-layer MLP gate (most expressive)
        - 'linear': Single linear layer gate
        - 'linear_norm': Linear + LayerNorm gate
        - 'per_token_scalar': One scalar per variate/token
        - 'global_scalar': One scalar per batch
    
    Args:
        fused_dim: Dimension of fused features (d_latent)
        ts_dim: Dimension of time series features (d_model)
        gate_type: Type of gating mechanism
        hidden_dim: Hidden dimension for MLP gate (optional)
    
    Input:
        fused_features: Cross-attention output [B, V, fused_dim]
        ts_features: Original TS features [B, V, ts_dim]
    
    Output:
        gated_output: Combined features [B, V, ts_dim]
        gate_value: Computed gate for analysis/regularization
    """
    
    GATE_TYPES = {'mlp', 'linear', 'linear_norm', 'per_token_scalar', 'global_scalar'}
    
    def __init__(
        self,
        fused_dim: int,
        ts_dim: int,
        gate_type: str = 'per_token_scalar',
        hidden_dim: Optional[int] = None
    ):
        super().__init__()
        
        # Validate gate type
        if gate_type not in self.GATE_TYPES:
            raise ValueError(
                f"gate_type must be one of {self.GATE_TYPES}, got '{gate_type}'"
            )
        
        self.gate_type = gate_type
        self.fused_dim = fused_dim
        self.ts_dim = ts_dim
        
        # Set hidden dimension
        hidden_dim = hidden_dim if hidden_dim is not None else 2 * ts_dim
        
        # Projection: fused_dim -> ts_dim
        self.fused_projection = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ts_dim)
        )
        
        # Gate network based on type
        concat_dim = fused_dim + ts_dim
        
        if gate_type == 'mlp':
            self.gate_network = nn.Sequential(
                nn.Linear(concat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, ts_dim),
                nn.Sigmoid()
            )
        elif gate_type == 'linear':
            self.gate_network = nn.Sequential(
                nn.Linear(concat_dim, ts_dim),
                nn.Sigmoid()
            )
        elif gate_type == 'linear_norm':
            self.gate_network = nn.Sequential(
                nn.Linear(concat_dim, ts_dim),
                nn.LayerNorm(ts_dim),
                nn.Sigmoid()
            )
        elif gate_type in ('per_token_scalar', 'global_scalar'):
            self.gate_network = nn.Sequential(
                nn.Linear(concat_dim, 1),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        fused_features: torch.Tensor,
        ts_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply gating to control fusion.
        
        Args:
            fused_features: [B, V, fused_dim]
            ts_features: [B, V, ts_dim]
        
        Returns:
            gated_output: [B, V, ts_dim]
            gate_value: Gate tensor for regularization
        """
        # Project fused features to TS dimension
        projected_fused = self.fused_projection(fused_features)  # [B, V, ts_dim]
        
        # Concatenate for gate computation
        concat_features = torch.cat([fused_features, ts_features], dim=-1)
        
        # Compute gate based on type
        if self.gate_type == 'global_scalar':
            # Mean pool then compute scalar
            pooled = concat_features.mean(dim=1)  # [B, fused+ts]
            gate_value = self.gate_network(pooled).unsqueeze(1)  # [B, 1, 1]
        else:
            gate_value = self.gate_network(concat_features)  # [B, V, ts_dim] or [B, V, 1]
        
        # Apply gate: O = α·TS + (1-α)·Fused
        gated_output = gate_value * ts_features + (1 - gate_value) * projected_fused
        
        return gated_output, gate_value
```

### 7.2 `models/ChimeraTransformer.py`

```python
"""
ChimeraTransformer Model for Fidel-TS.

Implements token-level multimodal fusion between time series and text embeddings
using cross-attention in a shared latent space with learned gating.

Architecture:
    1. iTransformer encoder for time series
    2. Optional text self-attention encoder
    3. Cross-modal attention (TS queries, Text keys/values)
    4. Optional post-fusion self-attention
    5. Feature gating for fusion control
    6. Optional final self-attention layers
    7. Linear projection to predictions

Key Features:
    - Uses pre-computed LLM embeddings (like TimeCMA)
    - Multiple gating mechanisms available
    - Multiple skip connection architectures
    - Gate regularization for interpretability

Reference:
    MM-TSFlib ChimeraTransformer: https://github.com/AdityaLab/MM-TSFlib
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from layers.StandardNorm import Normalize
from layers.ChimeraTransformer_layers import MultiheadLatentAttention, FeatureGate
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """
    ChimeraTransformer: Token-level Multimodal Fusion Model.
    
    Fuses time series features with pre-computed text embeddings using
    cross-attention and learned gating mechanisms.
    
    Args:
        configs: Configuration object with:
            Required:
                - seq_len: Input sequence length
                - pred_len: Prediction horizon
                - enc_in: Number of input channels/variates
            
            iTransformer (time series encoder):
                - d_model: Model dimension (default: 512)
                - n_heads: Number of attention heads (default: 8)
                - num_layers: Number of encoder layers (default: 2)
                - d_ff: Feed-forward dimension (default: 2048)
                - dropout: Dropout rate (default: 0.1)
                - activation: Activation function (default: 'gelu')
            
            Text encoder:
                - d_llm: LLM embedding dimension (default: 768)
                - num_layers_llm: Text self-attention layers (default: 0)
            
            Cross-modal fusion:
                - d_latent: Latent space dimension (default: 512)
                - fusion_heads: Cross-attention heads (default: 4)
                - post_fusion_layers: Post-fusion self-attn layers (default: 0)
            
            Gating:
                - gate_type: Gate mechanism type (default: 'per_token_scalar')
                - gate_hidden_dim: MLP gate hidden dim (optional)
                - architecture: Skip connection type (default: 'post_attn_skip')
            
            Final processing:
                - final_layers: Final self-attention layers (default: 0)
    
    Input:
        - x: Time series [B, seq_len, enc_in]
        - channel_description: Pre-computed LLM embeddings [B, d_llm, enc_in] 
          or [B, L, d_llm] depending on provider
        - x_mark_enc: Time features (optional)
    
    Output:
        - predictions: [B, pred_len, enc_in]
        - gate_value: Gate values for analysis (during training)
        - final_gate_value: Second gate values if dual_gate (during training)
    """
    
    ARCHITECTURES = {'post_attn_skip', 'raw_skip', 'raw_skip_dual_gate'}
    
    def __init__(self, configs):
        super().__init__()
        
        # =====================================================================
        # Configuration
        # =====================================================================
        
        # Required parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # iTransformer parameters
        self.d_model = getattr(configs, 'd_model', 512)
        self.n_heads = getattr(configs, 'n_heads', 8)
        self.num_layers = getattr(configs, 'num_layers', 2)
        self.d_ff = getattr(configs, 'd_ff', 2048)
        self.dropout = getattr(configs, 'dropout', 0.1)
        self.activation = getattr(configs, 'activation', 'gelu')
        self.freq = getattr(configs, 'freq', 'h')
        
        # Text encoder parameters
        self.d_llm = getattr(configs, 'd_llm', 768)
        self.num_layers_llm = getattr(configs, 'num_layers_llm', 0)
        
        # Cross-modal fusion parameters
        self.d_latent = getattr(configs, 'd_latent', min(self.d_model, self.d_llm))
        self.fusion_heads = getattr(configs, 'fusion_heads', 4)
        self.post_fusion_layers = getattr(configs, 'post_fusion_layers', 0)
        
        # Gating parameters
        self.gate_type = getattr(configs, 'gate_type', 'per_token_scalar')
        self.gate_hidden_dim = getattr(configs, 'gate_hidden_dim', None)
        self.architecture = getattr(configs, 'architecture', 'post_attn_skip')
        
        # Final processing parameters
        self.final_layers = getattr(configs, 'final_layers', 0)
        
        # Gate regularization flag
        self.gate_regularization = getattr(configs, 'gate_regularization_lambda', 0) > 0
        
        # Validate architecture
        assert self.architecture in self.ARCHITECTURES, \
            f"architecture must be one of {self.ARCHITECTURES}"
        
        if self.architecture == 'raw_skip_dual_gate':
            assert self.final_layers > 0, \
                "raw_skip_dual_gate requires final_layers > 0"
        
        # =====================================================================
        # Build Model Components
        # =====================================================================
        
        # RevIN normalization
        self.normalize_layers = Normalize(self.enc_in, affine=False)
        
        # iTransformer components
        self._build_itransformer()
        
        # Text self-attention (optional)
        self._build_text_encoder()
        
        # Cross-modal fusion
        self._build_cross_attention()
        
        # Post-fusion self-attention (optional)
        self._build_post_fusion()
        
        # Feature gating
        self._build_gating()
        
        # Final self-attention (optional)
        self._build_final_layers()
        
        # Output projection
        self.projection = nn.Linear(self.d_model, self.pred_len, bias=True)
    
    def _build_itransformer(self):
        """Build iTransformer embedding and encoder."""
        # Inverted embedding: treats variates as tokens
        self.enc_embedding = DataEmbedding_inverted(
            c_in=self.seq_len,
            d_model=self.d_model,
            freq=self.freq,
            dropout=self.dropout
        )
        
        # Transformer encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            attention_dropout=self.dropout
                        ),
                        self.d_model,
                        self.n_heads
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                )
                for _ in range(self.num_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model)
        )
    
    def _build_text_encoder(self):
        """Build optional text self-attention encoder."""
        self.text_encoder = None
        if self.num_layers_llm > 0:
            self.text_encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(
                                mask_flag=False,
                                attention_dropout=self.dropout
                            ),
                            self.d_llm,
                            self.n_heads
                        ),
                        self.d_llm,
                        self.d_ff,
                        dropout=self.dropout,
                        activation=self.activation
                    )
                    for _ in range(self.num_layers_llm)
                ],
                norm_layer=nn.LayerNorm(self.d_llm)
            )
    
    def _build_cross_attention(self):
        """Build cross-modal attention layer."""
        self.cross_attention = MultiheadLatentAttention(
            query_dim=self.d_model,
            key_dim=self.d_llm,
            latent_dim=self.d_latent,
            num_heads=self.fusion_heads,
            dropout=self.dropout
        )
    
    def _build_post_fusion(self):
        """Build optional post-fusion self-attention."""
        self.post_fusion_encoder = None
        if self.post_fusion_layers > 0:
            self.post_fusion_encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(
                                mask_flag=False,
                                attention_dropout=self.dropout
                            ),
                            self.d_latent,
                            self.n_heads
                        ),
                        self.d_latent,
                        self.d_ff,
                        dropout=self.dropout,
                        activation=self.activation
                    )
                    for _ in range(self.post_fusion_layers)
                ],
                norm_layer=nn.LayerNorm(self.d_latent)
            )
    
    def _build_gating(self):
        """Build feature gating mechanism."""
        self.feature_gate = FeatureGate(
            fused_dim=self.d_latent,
            ts_dim=self.d_model,
            gate_type=self.gate_type,
            hidden_dim=self.gate_hidden_dim
        )
        
        # Second gate for dual_gate architecture
        self.final_feature_gate = None
        if self.architecture == 'raw_skip_dual_gate':
            self.final_feature_gate = FeatureGate(
                fused_dim=self.d_model,
                ts_dim=self.d_model,
                gate_type=self.gate_type,
                hidden_dim=self.gate_hidden_dim
            )
    
    def _build_final_layers(self):
        """Build optional final self-attention layers."""
        self.final_encoder = None
        if self.final_layers > 0:
            self.final_encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(
                                mask_flag=False,
                                attention_dropout=self.dropout
                            ),
                            self.d_model,
                            self.n_heads
                        ),
                        self.d_model,
                        self.d_ff,
                        dropout=self.dropout,
                        activation=self.activation
                    )
                    for _ in range(self.final_layers)
                ],
                norm_layer=nn.LayerNorm(self.d_model)
            )
    
    def forward(
        self,
        x: torch.Tensor,
        channel_description: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with multimodal fusion.
        
        Args:
            x: Time series input [B, seq_len, enc_in]
            channel_description: Pre-computed LLM embeddings
                Shape: [B, d_llm, enc_in] or [B, L, d_llm]
            x_mark_enc: Time features (optional, not used)
            **kwargs: Additional arguments for compatibility
        
        Returns:
            predictions: [B, pred_len, enc_in]
            gate_value: Gate values (if training with regularization)
            final_gate_value: Second gate values (if dual_gate architecture)
        """
        # Ensure float32
        x = x.float()
        channel_description = channel_description.float()
        
        # Handle embedding shape
        # Expected: [B, L, d_llm] where L is text sequence length
        # If [B, d_llm, N], transpose to [B, N, d_llm]
        if channel_description.dim() == 3:
            if channel_description.shape[1] == self.d_llm:
                # Shape is [B, d_llm, N], transpose
                text_embeddings = channel_description.permute(0, 2, 1)
            else:
                # Shape is already [B, L, d_llm]
                text_embeddings = channel_description
        elif channel_description.dim() == 4:
            # Shape is [B, d_llm, N, 1], squeeze and transpose
            text_embeddings = channel_description.squeeze(-1).permute(0, 2, 1)
        else:
            text_embeddings = channel_description
        
        # =====================================================================
        # RevIN Normalization
        # =====================================================================
        x = self.normalize_layers(x, 'norm')
        
        B, T, N = x.shape
        
        # =====================================================================
        # Time Series Encoding (iTransformer)
        # =====================================================================
        # Embedding: [B, T, N] -> [B, N, d_model] (inverted)
        enc_out = self.enc_embedding(x, x_mark_enc)
        
        # Encoder: [B, N, d_model] -> [B, N, d_model]
        ts_features, _ = self.encoder(enc_out, attn_mask=None)
        
        # =====================================================================
        # Text Encoding (Optional)
        # =====================================================================
        if self.text_encoder is not None:
            text_features, _ = self.text_encoder(text_embeddings, attn_mask=None)
        else:
            text_features = text_embeddings
        
        # =====================================================================
        # Cross-Modal Fusion
        # =====================================================================
        # Q: TS features [B, N, d_model]
        # K,V: Text features [B, L, d_llm]
        # Output: [B, N, d_latent]
        fused_features = self.cross_attention(
            queries=ts_features,
            keys=text_features,
            values=text_features
        )
        
        # Post-fusion self-attention (optional)
        if self.post_fusion_encoder is not None:
            fused_features, _ = self.post_fusion_encoder(fused_features, attn_mask=None)
        
        # =====================================================================
        # Gated Fusion
        # =====================================================================
        # Determine skip connection source
        if self.architecture == 'post_attn_skip':
            skip_features = ts_features
        else:  # raw_skip or raw_skip_dual_gate
            skip_features = enc_out
        
        # Apply feature gate
        final_features, gate_value = self.feature_gate(fused_features, skip_features)
        
        # =====================================================================
        # Final Processing
        # =====================================================================
        final_gate_value = None
        
        if self.final_encoder is not None:
            final_features, _ = self.final_encoder(final_features, attn_mask=None)
            
            # Second gate for dual_gate architecture
            if self.final_feature_gate is not None:
                final_features, final_gate_value = self.final_feature_gate(
                    final_features, enc_out
                )
        
        # =====================================================================
        # Output Projection
        # =====================================================================
        # [B, N, d_model] -> [B, N, pred_len] -> [B, pred_len, N]
        dec_out = self.projection(final_features)
        dec_out = dec_out.permute(0, 2, 1)[:, :, :N]
        
        # =====================================================================
        # RevIN Denormalization
        # =====================================================================
        dec_out = self.normalize_layers(dec_out, 'denorm')
        
        return dec_out, gate_value, final_gate_value
    
    def forward_ts_only(
        self,
        x: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with time series only (no text fusion).
        
        Used for pre-training the TS encoder before multimodal training.
        
        Args:
            x: Time series input [B, seq_len, enc_in]
            x_mark_enc: Time features (optional)
        
        Returns:
            predictions: [B, pred_len, enc_in]
        """
        x = x.float()
        
        # RevIN normalization
        x = self.normalize_layers(x, 'norm')
        B, T, N = x.shape
        
        # iTransformer encoding
        enc_out = self.enc_embedding(x, x_mark_enc)
        ts_features, _ = self.encoder(enc_out, attn_mask=None)
        
        # Direct projection (no fusion)
        dec_out = self.projection(ts_features)
        dec_out = dec_out.permute(0, 2, 1)[:, :, :N]
        
        # RevIN denormalization
        dec_out = self.normalize_layers(dec_out, 'denorm')
        
        return dec_out
    
    def count_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_frozen_params(self) -> int:
        """Count frozen parameters."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
```

### 7.3 `model_configs/general/ChimeraTransformer.yaml`

```yaml
# ChimeraTransformer Configuration
# Token-level multimodal fusion model based on MM-TSFlib

model: ChimeraTransformer

# iTransformer (Time Series Encoder)
d_model: 512
n_heads: 8
num_layers: 2
d_ff: 2048
dropout: 0.1
activation: gelu

# Text Encoder
d_llm: 768  # Match embedding dimension from LLMEmbeddingProvider
num_layers_llm: 0  # Optional self-attention on text embeddings

# Cross-Modal Fusion
d_latent: 512  # Shared latent space dimension
fusion_heads: 4  # Cross-attention heads
post_fusion_layers: 0  # Optional post-fusion self-attention

# Gating Mechanism
gate_type: per_token_scalar  # Options: mlp, linear, linear_norm, per_token_scalar, global_scalar
gate_hidden_dim: null  # Auto: 2 * d_model
architecture: post_attn_skip  # Options: post_attn_skip, raw_skip, raw_skip_dual_gate

# Final Processing
final_layers: 0  # Optional final self-attention

# Training
gate_regularization_lambda: 0.0  # L1 regularization on gate values
ts_only_epochs: 0  # Epochs to train TS encoder only
freeze_ts: false  # Freeze TS encoder after ts_only_epochs

# These are typically set by data config
# seq_len: 96
# pred_len: 96
# enc_in: 7
```

---

## 8. Testing Strategy

### 8.1 Unit Tests (`tests/test_chimera.py`)

```python
"""
Unit tests for ChimeraTransformer components.
"""

import pytest
import torch
from layers.ChimeraTransformer_layers import MultiheadLatentAttention, FeatureGate


class TestMultiheadLatentAttention:
    """Tests for cross-attention layer."""
    
    def test_shapes(self):
        """Test output shapes are correct."""
        batch_size, num_variates, text_len = 4, 7, 32
        query_dim, key_dim, latent_dim = 512, 768, 256
        
        layer = MultiheadLatentAttention(
            query_dim=query_dim,
            key_dim=key_dim,
            latent_dim=latent_dim,
            num_heads=4
        )
        
        queries = torch.randn(batch_size, num_variates, query_dim)
        keys = torch.randn(batch_size, text_len, key_dim)
        values = torch.randn(batch_size, text_len, key_dim)
        
        output = layer(queries, keys, values)
        
        assert output.shape == (batch_size, num_variates, latent_dim)
    
    def test_different_dimensions(self):
        """Test with various dimension combinations."""
        configs = [
            (256, 768, 128, 4),   # Small
            (512, 768, 512, 8),   # Medium
            (1024, 4096, 512, 8), # Large (LLaMA-sized text)
        ]
        
        for query_dim, key_dim, latent_dim, n_heads in configs:
            layer = MultiheadLatentAttention(
                query_dim=query_dim,
                key_dim=key_dim,
                latent_dim=latent_dim,
                num_heads=n_heads
            )
            
            q = torch.randn(2, 7, query_dim)
            k = torch.randn(2, 16, key_dim)
            v = torch.randn(2, 16, key_dim)
            
            output = layer(q, k, v)
            assert output.shape == (2, 7, latent_dim)


class TestFeatureGate:
    """Tests for gating mechanism."""
    
    @pytest.mark.parametrize("gate_type", [
        'mlp', 'linear', 'linear_norm', 'per_token_scalar', 'global_scalar'
    ])
    def test_gate_types(self, gate_type):
        """Test all gate types produce correct shapes."""
        batch_size, num_variates = 4, 7
        fused_dim, ts_dim = 256, 512
        
        gate = FeatureGate(
            fused_dim=fused_dim,
            ts_dim=ts_dim,
            gate_type=gate_type
        )
        
        fused = torch.randn(batch_size, num_variates, fused_dim)
        ts = torch.randn(batch_size, num_variates, ts_dim)
        
        output, gate_value = gate(fused, ts)
        
        assert output.shape == (batch_size, num_variates, ts_dim)
        
        # Check gate value shape
        if gate_type in ('mlp', 'linear', 'linear_norm'):
            assert gate_value.shape == (batch_size, num_variates, ts_dim)
        elif gate_type == 'per_token_scalar':
            assert gate_value.shape == (batch_size, num_variates, 1)
        elif gate_type == 'global_scalar':
            assert gate_value.shape == (batch_size, 1, 1)
    
    def test_gate_value_range(self):
        """Test gate values are in [0, 1]."""
        gate = FeatureGate(fused_dim=256, ts_dim=512, gate_type='linear')
        
        fused = torch.randn(4, 7, 256)
        ts = torch.randn(4, 7, 512)
        
        _, gate_value = gate(fused, ts)
        
        assert (gate_value >= 0).all()
        assert (gate_value <= 1).all()


class TestChimeraTransformer:
    """Integration tests for full model."""
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        from models.ChimeraTransformer import Model
        from utils.tools import dotdict
        
        configs = dotdict({
            'seq_len': 96,
            'pred_len': 24,
            'enc_in': 7,
            'd_model': 128,
            'n_heads': 4,
            'd_llm': 768,
        })
        
        model = Model(configs)
        
        x = torch.randn(4, 96, 7)
        text_emb = torch.randn(4, 768, 7)  # Per-channel embeddings
        
        output, gate, _ = model(x, text_emb)
        
        assert output.shape == (4, 24, 7)
        assert gate is not None
    
    def test_ts_only_forward(self):
        """Test TS-only forward pass."""
        from models.ChimeraTransformer import Model
        from utils.tools import dotdict
        
        configs = dotdict({
            'seq_len': 96,
            'pred_len': 24,
            'enc_in': 7,
        })
        
        model = Model(configs)
        x = torch.randn(4, 96, 7)
        
        output = model.forward_ts_only(x)
        
        assert output.shape == (4, 24, 7)
```

### 8.2 Integration Tests

```python
def test_with_lightning_data():
    """Test with actual data loader."""
    from data_provider.data_factory import get_datasets
    from models.ChimeraTransformer import Model
    
    # Load Time-MMD data with embeddings
    train_dataset = get_datasets(
        data_config='time_mmd/public_health',
        flag='train',
        llm_embedding_provider=provider
    )
    
    # Get sample batch
    batch = train_dataset[0]
    x, y, x_mark, y_mark, ..., channel_desc = batch
    
    # Run model
    model = Model(configs)
    output, gate, _ = model(x.unsqueeze(0), channel_desc.unsqueeze(0))
    
    assert output.shape[1] == configs.pred_len
```

---

## 9. Configuration

### 9.1 Data Configuration for Time-MMD

Update existing Time-MMD configs to include embedding paths:

```yaml
# data_configs/time_mmd/public_health.yaml

# Existing config...
root_path: ./data/time_mmd/Public_Health
data_path: US_FLURATIO_Week.csv

# Add embedding configuration
embedding:
  provider: llm
  model: gpt2  # or bert, llama, etc.
  cache_path: ./embeddings/time_mmd/public_health
  text_column: Final_Search_4  # Column containing text for each sample
```

### 9.2 Experiment Suite Configuration

```yaml
# configs/experiment_suites/chimera_benchmark.yaml

suite_name: chimera_time_mmd_benchmark

model_config: ChimeraTransformer

# Shared settings
shared:
  seq_len: 48
  pred_len: 24
  learning_rate: 0.0001
  batch_size: 32
  train_epochs: 50
  patience: 10
  
  # Chimera-specific
  d_model: 512
  d_llm: 768  # GPT-2 dimension
  d_latent: 512
  gate_type: per_token_scalar
  architecture: post_attn_skip

# Dataset variations
experiments:
  - name: public_health_flu
    data_config: time_mmd/public_health_flu
    
  - name: economy_trade
    data_config: time_mmd/economy_trade
    
  - name: energy_gas
    data_config: time_mmd/energy_gas
```

---

## 10. Timeline and Milestones

### Phase 1: Core Implementation (Week 1-2)
- [ ] Implement `ChimeraTransformer_layers.py`
- [ ] Implement `ChimeraTransformer.py`
- [ ] Add to model registry
- [ ] Create model config YAML

### Phase 2: Integration (Week 2-3)
- [ ] Extend data loader for text retrieval
- [ ] Integrate with LLMEmbeddingProvider
- [ ] Add gate regularization to training loop

### Phase 3: Testing (Week 3)
- [ ] Unit tests for layers
- [ ] Integration tests with Time-MMD data
- [ ] Benchmark against MM-TSFlib results

### Phase 4: Documentation (Week 4)
- [ ] Update README
- [ ] Add example scripts
- [ ] Document configuration options

---

## Appendix A: Key Differences from Original Implementation

| Aspect | MM-TSFlib | Fidel-TS Implementation |
|--------|-----------|------------------------|
| Text Input | Raw text → tokenized at runtime | Pre-computed embeddings |
| LLM Loading | Custom in exp class | Existing LLMEmbeddingProvider |
| Training | Custom experiment class | Lightning module |
| Config | argparse | YAML configs |
| Data | CSV with text column | Universal_Dataset with embedding support |
| Normalization | In-model RevIN | Existing Normalize layer |

## Appendix B: Hyperparameter Recommendations

Based on MM-TSFlib experiments:

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| d_model | 512 | Match iTransformer default |
| d_llm | 768 | GPT-2 base dimension |
| d_latent | 512 | Often matches d_model |
| fusion_heads | 4 | Fewer than n_heads is fine |
| gate_type | per_token_scalar | Good balance of interpretability/performance |
| post_fusion_layers | 0-2 | More layers = more compute |
| final_layers | 0-2 | Depends on architecture |
| gate_regularization_lambda | 0.0-0.001 | Optional, aids interpretability |

## Appendix C: Future Extensions

1. **Bidirectional Cross-Attention**: Allow text to also query time series
2. **Multi-Scale Fusion**: Fuse at multiple encoder depths
3. **Adaptive Embedding Selection**: Learn which text tokens are most relevant
4. **Online Text Processing**: Support real-time text encoding
5. **Gate Visualization**: Tools for analyzing learned gating patterns
