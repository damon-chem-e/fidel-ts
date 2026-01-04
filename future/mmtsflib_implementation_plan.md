# MM-TSFlib Late Fusion Implementation Plan for Fidel-TS

This document provides a detailed plan to implement the MM-TSFlib prediction-level late fusion approach in the fidel-ts framework, following the same patterns used for ZhangHanBest.

---

## Table of Contents

1. [Overview](#1-overview)
2. [MM-TSFlib vs ZhangHanBest Comparison](#2-mm-tsflib-vs-zhanghanbest-comparison)
3. [Implementation Plan](#3-implementation-plan)
4. [File Structure](#4-file-structure)
5. [Detailed Implementation](#5-detailed-implementation)
6. [Configuration](#6-configuration)
7. [Testing Strategy](#7-testing-strategy)
8. [Timeline](#8-timeline)

---

## 1. Overview

### 1.1 What is MM-TSFlib Late Fusion?

MM-TSFlib implements a simple but effective **prediction-level ensemble** approach for multimodal time series forecasting:

1. **Time Series Branch**: Any unimodal model produces predictions
2. **Text Branch**: LLM embeddings → MLP → pooling → normalized text-derived predictions
3. **Ensemble**: Weighted combination of both predictions

```
Text Embeddings ──→ MLP ──→ Pool ──→ Normalize ──→ text_pred
                                                        ↓
                              (1-w)*ts_pred + w*(text_pred + prior)
                                                        ↑
Time Series ──→ Unimodal Model ──→ ts_pred
```

### 1.2 Key Characteristics

| Aspect | MM-TSFlib Approach |
|--------|-------------------|
| Fusion Level | Prediction-level (late fusion) |
| Text Processing | MLP projection: `d_llm → d_llm/8 → pred_len` |
| Pooling Options | avg, max, min, attention |
| Mixing Weight | Fixed hyperparameter (default: 0.01) |
| Prior Integration | Adds historical average to text prediction |
| TS Model | Any unimodal model (returns predictions directly) |

### 1.3 Source Code Location in MM-TSFlib

The implementation lives in `exp/exp_long_term_forecasting.py`:
- **MLP class**: Lines 30-44
- **Initialization**: Lines 71-80 (MLP sizes, prompt_weight)
- **Text embedding**: Lines 575-591
- **Pooling**: Lines 630-653
- **Ensemble**: Lines 654-655

---

## 2. MM-TSFlib vs ZhangHanBest Comparison

Understanding the differences helps inform design decisions:

| Aspect | MM-TSFlib | ZhangHanBest |
|--------|-----------|--------------|
| **Fusion Level** | Prediction-level | Representation-level |
| **What's Fused** | Final predictions | Latent representations |
| **TS Model Output** | Predictions `[B, pred_len, C]` | Representations `[B, d_model]` |
| **Text Projection** | `d_llm → d_llm/8 → pred_len` | `text_dim → d_model` (residual) |
| **Pooling** | avg/max/min/attention | Pre-aggregated |
| **Mixing Default** | 0.01 (99% TS, 1% text) | 0.5 (50/50) |
| **Prediction Head** | Built into TS model | Separate head after fusion |
| **Prior History** | Yes (adds to text pred) | No |

### Key Design Decision

We will adapt MM-TSFlib to work with **pre-computed embeddings** (like ZhangHanBest) rather than runtime LLM tokenization, since fidel-ts already has the `LLMEmbeddingProvider` infrastructure.

---

## 3. Implementation Plan

### 3.1 Phase 1: Core Model Implementation

**Goal**: Create `MMTSFlib` model class that wraps any unimodal model.

**Tasks**:
1. Create `layers/MMTSFlib_layers.py`:
   - `TextToPredsProjection`: MLP for text → prediction space
   - `TextPooling`: Different pooling strategies
   
2. Create `models/MMTSFlib.py`:
   - Wrap any unimodal model (like ZhangHanBest)
   - Implement prediction-level ensemble
   - Support optional prior integration

### 3.2 Phase 2: Configuration

**Tasks**:
1. Create `model_configs/general/MMTSFlib.yaml`
2. Add to model registry in `models/__init__.py`

### 3.3 Phase 3: Testing

**Tasks**:
1. Unit tests for layers
2. Integration tests with different unimodal models
3. Validate against Time-MMD dataset

---

## 4. File Structure

### 4.1 New Files to Create

```
fidel-ts/
├── layers/
│   └── MMTSFlib_layers.py          # NEW: Text projection and pooling
│
├── models/
│   └── MMTSFlib.py                 # NEW: Main model (wraps unimodal)
│
├── model_configs/
│   └── general/
│       └── MMTSFlib.yaml           # NEW: Model config
│
└── tests/
    └── test_mmtsflib.py            # NEW: Unit tests
```

### 4.2 Files to Modify

```
fidel-ts/
└── models/__init__.py              # Add MMTSFlib import
```

---

## 5. Detailed Implementation

### 5.1 `layers/MMTSFlib_layers.py`

```python
"""
MM-TSFlib Layers for Fidel-TS.

Implements the text-to-prediction projection and pooling mechanisms
from MM-TSFlib's late fusion approach.

Components:
    - TextToPredsProjection: MLP that projects text embeddings to prediction space
    - TextPooling: Various pooling strategies for aggregating token embeddings

Reference:
    MM-TSFlib: https://github.com/AdityaLab/MM-TSFlib
    Time-MMD: https://github.com/AdityaLab/Time-MMD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


class TextToPredsProjection(nn.Module):
    """
    MLP that projects text embeddings to prediction space.
    
    Following MM-TSFlib architecture:
        d_llm → d_llm/reduction_factor → pred_len
    
    This maps LLM token embeddings to a dimension matching predictions,
    allowing direct prediction-level ensemble.
    
    Args:
        d_llm: Dimension of input LLM embeddings (e.g., 768 for BERT)
        pred_len: Prediction horizon length (output dimension)
        reduction_factor: Hidden layer size = d_llm / reduction_factor (default: 8)
        dropout: Dropout rate (default: 0.3)
    
    Input:
        text_emb: [B, L, d_llm] - Token-level embeddings
        
    Output:
        text_proj: [B, L, pred_len] - Projected to prediction dimension
    """
    
    def __init__(
        self,
        d_llm: int,
        pred_len: int,
        reduction_factor: int = 8,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.d_llm = d_llm
        self.pred_len = pred_len
        
        # Hidden dimension from reduction
        hidden_dim = max(d_llm // reduction_factor, pred_len)
        
        # Two-layer MLP with ReLU and dropout
        self.layers = nn.Sequential(
            nn.Linear(d_llm, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len)
        )
    
    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Project text embeddings to prediction dimension.
        
        Args:
            text_emb: [B, L, d_llm] or [B, d_llm] (if already pooled)
        
        Returns:
            text_proj: [B, L, pred_len] or [B, pred_len]
        """
        return self.layers(text_emb)


class TextPooling(nn.Module):
    """
    Pooling layer to aggregate token-level embeddings into a single vector.
    
    Supports multiple pooling strategies from MM-TSFlib:
        - 'avg': Global average pooling
        - 'max': Global max pooling
        - 'min': Global min pooling (via negated max)
        - 'attention': Attention-weighted pooling using TS predictions
    
    Args:
        pool_type: Pooling strategy ('avg', 'max', 'min', 'attention')
    
    Input:
        text_emb: [B, L, D] - Token-level embeddings
        ts_preds: [B, pred_len, C] - TS predictions (only for attention pooling)
        
    Output:
        pooled: [B, D, 1] - Pooled embeddings (with trailing dim for broadcast)
    """
    
    POOL_TYPES = {'avg', 'max', 'min', 'attention'}
    
    def __init__(self, pool_type: Literal['avg', 'max', 'min', 'attention'] = 'avg'):
        super().__init__()
        
        if pool_type not in self.POOL_TYPES:
            raise ValueError(f"pool_type must be one of {self.POOL_TYPES}, got '{pool_type}'")
        
        self.pool_type = pool_type
    
    def forward(
        self,
        text_emb: torch.Tensor,
        ts_preds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool token embeddings into single vector.
        
        Args:
            text_emb: [B, L, D] - Token-level embeddings
            ts_preds: [B, pred_len, C] - TS predictions (required for attention pooling)
        
        Returns:
            pooled: [B, D, 1] - Pooled embeddings
        """
        # text_emb shape: [B, L, D]
        
        if self.pool_type == 'avg':
            # Global average pooling: [B, L, D] -> [B, D]
            pooled = F.adaptive_avg_pool1d(
                text_emb.transpose(1, 2),  # [B, D, L]
                1  # Output length
            ).squeeze(2)  # [B, D]
            
        elif self.pool_type == 'max':
            # Global max pooling
            pooled = F.adaptive_max_pool1d(
                text_emb.transpose(1, 2),
                1
            ).squeeze(2)
            
        elif self.pool_type == 'min':
            # Global min pooling (via negated max)
            pooled = -F.adaptive_max_pool1d(
                -text_emb.transpose(1, 2),
                1
            ).squeeze(2)
            
        elif self.pool_type == 'attention':
            # Attention-weighted pooling using TS predictions
            if ts_preds is None:
                raise ValueError("ts_preds required for attention pooling")
            
            # Normalize for attention computation
            # text_emb: [B, L, D], ts_preds: [B, pred_len, C]
            text_norm = F.normalize(text_emb, p=2, dim=2)  # [B, L, D]
            ts_norm = F.normalize(ts_preds, p=2, dim=1)    # [B, pred_len, C]
            
            # Compute attention scores
            # [B, L, D] x [B, D, C] -> [B, L, C] (if D == pred_len)
            # Note: In MM-TSFlib, D == pred_len after MLP projection
            attention_scores = torch.bmm(text_norm, ts_norm)  # [B, L, C]
            attention_weights = F.softmax(attention_scores, dim=1)  # [B, L, C]
            
            # Weighted sum over tokens
            # [B, L, D] * [B, L, C] -> need broadcasting
            # For simplicity, average attention over C dimension
            attention_weights_avg = attention_weights.mean(dim=2, keepdim=True)  # [B, L, 1]
            pooled = (text_emb * attention_weights_avg).sum(dim=1)  # [B, D]
        
        # Add trailing dimension for broadcast compatibility
        return pooled.unsqueeze(-1)  # [B, D, 1]


def normalize_embeddings(emb: torch.Tensor) -> torch.Tensor:
    """
    Instance normalization for embeddings (from MM-TSFlib).
    
    Normalizes each sample independently:
        emb_norm = (emb - mean) / std
    
    Args:
        emb: Input tensor [B, D, 1] or [B, D]
    
    Returns:
        emb_norm: Normalized tensor (same shape)
    """
    # Handle both [B, D, 1] and [B, D] inputs
    squeeze_output = emb.dim() == 2
    if squeeze_output:
        emb = emb.unsqueeze(-1)
    
    # Compute mean and std over D dimension
    mean = emb.mean(dim=1, keepdim=True).detach()
    var = torch.var(emb, dim=1, keepdim=True, unbiased=False) + 1e-5
    std = torch.sqrt(var)
    
    # Normalize
    emb_norm = (emb - mean) / std
    
    if squeeze_output:
        emb_norm = emb_norm.squeeze(-1)
    
    return emb_norm
```

### 5.2 `models/MMTSFlib.py`

```python
"""
MM-TSFlib Model: Prediction-Level Late Fusion for Fidel-TS.

Implements the late fusion approach from MM-TSFlib where time series
predictions are ensembled with text-derived predictions at the output level.

Architecture:
1. Time series encoder (any unimodal model, returns predictions)
2. Text embeddings (pre-computed via LLMEmbeddingProvider)
3. MLP projection (text_dim -> pred_len)
4. Pooling (avg/max/min/attention)
5. Normalization
6. Prediction-level ensemble: (1-w)*ts_pred + w*(text_pred + prior)

Key Features:
- Works with ANY unimodal model (PatchTST, DLinear, iTransformer, etc.)
- Uses pre-computed embeddings (no runtime LLM)
- Supports multiple pooling strategies
- Optional prior history integration
- Configurable mixing weight (prompt_weight)

Reference:
    MM-TSFlib: https://github.com/AdityaLab/MM-TSFlib
    Time-MMD: https://github.com/AdityaLab/Time-MMD
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Literal

from layers.MMTSFlib_layers import (
    TextToPredsProjection,
    TextPooling,
    normalize_embeddings
)


class Model(nn.Module):
    """
    MM-TSFlib: Prediction-Level Late Fusion Model.
    
    Ensembles time series predictions with text-derived predictions using
    a fixed or learned mixing weight.
    
    Args:
        configs: Configuration object with:
            Required:
                - seq_len: Input sequence length
                - pred_len: Prediction horizon
                - enc_in: Number of input channels/variates
            
            Unimodal model:
                - unimodal_model_type: Base model ('PatchTST', 'DLinear', 'iTransformer', etc.)
                - (other model-specific configs passed through)
            
            Text processing:
                - input_text_dim: Dimension of input text embeddings (default: 768)
                - text_mlp_reduction: Hidden dim = d_llm / reduction (default: 8)
                - text_mlp_dropout: Dropout in text MLP (default: 0.3)
                - pool_type: Pooling strategy (default: 'avg')
            
            Fusion:
                - prompt_weight: Mixing weight w (default: 0.01)
                - use_prior: Whether to add prior history (default: True)
    
    Input:
        - x: Time series [B, seq_len, enc_in]
        - **kwargs: Must include text embeddings via 'dataset_description' or 'hetero_general'
    
    Output:
        - predictions: [B, pred_len, enc_in]
    """
    
    # Supported unimodal models
    SUPPORTED_MODELS = {
        'PatchTST', 'DLinear', 'iTransformer', 'FEDformer', 
        'Informer', 'FITS', 'TimeMixer', 'Autoformer'
    }
    
    def __init__(self, configs):
        super().__init__()
        
        # =====================================================================
        # Configuration
        # =====================================================================
        
        # Required parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # Unimodal model configuration
        self.unimodal_model_type = getattr(configs, 'unimodal_model_type', 'iTransformer')
        
        # Text processing configuration
        self.text_dim = getattr(configs, 'input_text_dim', 768)
        self.text_mlp_reduction = getattr(configs, 'text_mlp_reduction', 8)
        self.text_mlp_dropout = getattr(configs, 'text_mlp_dropout', 0.3)
        self.pool_type = getattr(configs, 'pool_type', 'avg')
        
        # Fusion configuration
        self.prompt_weight = getattr(configs, 'prompt_weight', 0.01)
        self.use_prior = getattr(configs, 'use_prior', False)
        
        # Store configs for unimodal model
        self.configs = configs
        
        # =====================================================================
        # Build Model Components
        # =====================================================================
        
        # 1. Unimodal time series model (returns predictions directly)
        self.ts_model = self._create_unimodal_model(configs)
        
        # 2. Text-to-predictions projection MLP
        self.text_projection = TextToPredsProjection(
            d_llm=self.text_dim,
            pred_len=self.pred_len,
            reduction_factor=self.text_mlp_reduction,
            dropout=self.text_mlp_dropout
        )
        
        # 3. Pooling layer
        self.pooling = TextPooling(pool_type=self.pool_type)
    
    def _create_unimodal_model(self, configs):
        """
        Create and configure unimodal time series model.
        
        The model should return predictions [B, pred_len, C] in its forward pass.
        """
        model_type = self.unimodal_model_type
        
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported unimodal_model_type: {model_type}. "
                f"Supported: {self.SUPPORTED_MODELS}"
            )
        
        # Import and instantiate
        if model_type == 'PatchTST':
            from models.PatchTST import Model as PatchTSTModel
            return PatchTSTModel(configs)
        elif model_type == 'DLinear':
            from models.DLinear import Model as DLinearModel
            return DLinearModel(configs)
        elif model_type == 'iTransformer':
            from models.iTransformer import Model as iTransformerModel
            return iTransformerModel(configs)
        elif model_type == 'FEDformer':
            from models.FEDformer import Model as FEDformerModel
            return FEDformerModel(configs)
        elif model_type == 'Informer':
            from models.Informer import Model as InformerModel
            return InformerModel(configs)
        elif model_type == 'FITS':
            from models.FITS import Model as FITSModel
            return FITSModel(configs)
        elif model_type == 'TimeMixer':
            from models.TimeMixer import Model as TimeMixerModel
            return TimeMixerModel(configs)
        elif model_type == 'Autoformer':
            from models.Autoformer import Model as AutoformerModel
            return AutoformerModel(configs)
        else:
            raise ValueError(f"Model {model_type} not implemented")
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass with prediction-level late fusion.
        
        Args:
            x: Time series input [B, seq_len, enc_in]
            **kwargs: Must include text embeddings:
                - 'dataset_description' or 'hetero_general': [B, L, text_dim] or [B, text_dim]
                - 'prior_y' (optional): Historical prior [B, pred_len, 1]
        
        Returns:
            predictions: [B, pred_len, enc_in]
        """
        # =====================================================================
        # 1. Time Series Predictions
        # =====================================================================
        
        # Get predictions from unimodal model
        # Most models accept x directly, some need additional args
        ts_preds = self._forward_ts_model(x, **kwargs)  # [B, pred_len, C]
        
        # =====================================================================
        # 2. Text Processing
        # =====================================================================
        
        # Get text embeddings from kwargs
        text_emb = self._get_text_embeddings(kwargs)  # [B, L, text_dim] or [B, text_dim]
        
        # Ensure 3D for token-level processing
        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(1)  # [B, 1, text_dim]
        
        # Project text to prediction dimension
        text_proj = self.text_projection(text_emb)  # [B, L, pred_len]
        
        # Pool to single vector
        if self.pool_type == 'attention':
            text_pooled = self.pooling(text_proj, ts_preds)  # [B, pred_len, 1]
        else:
            text_pooled = self.pooling(text_proj)  # [B, pred_len, 1]
        
        # Normalize
        text_pred = normalize_embeddings(text_pooled)  # [B, pred_len, 1]
        
        # =====================================================================
        # 3. Prior Integration (Optional)
        # =====================================================================
        
        if self.use_prior and 'prior_y' in kwargs and kwargs['prior_y'] is not None:
            prior_y = kwargs['prior_y']  # [B, pred_len, 1]
            if isinstance(prior_y, np.ndarray):
                prior_y = torch.from_numpy(prior_y).float()
            prior_y = prior_y.to(text_pred.device)
            text_pred = text_pred + prior_y
        
        # =====================================================================
        # 4. Prediction-Level Ensemble
        # =====================================================================
        
        # Broadcast text_pred to match ts_preds channels
        # text_pred: [B, pred_len, 1], ts_preds: [B, pred_len, C]
        
        # Ensemble: (1 - w) * ts_pred + w * text_pred
        predictions = (1 - self.prompt_weight) * ts_preds + self.prompt_weight * text_pred
        
        return predictions
    
    def _forward_ts_model(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through unimodal time series model.
        
        Handles different model interfaces.
        """
        # Most models just need x
        # Some models (like Informer) need decoder input, time marks, etc.
        
        if self.unimodal_model_type in ['Informer', 'FEDformer', 'Autoformer']:
            # Encoder-decoder models need more inputs
            x_mark_enc = kwargs.get('x_mark_enc', None)
            x_mark_dec = kwargs.get('x_mark_dec', None)
            
            # Create decoder input (zeros for prediction part)
            B, T, C = x.shape
            dec_inp = torch.zeros(B, self.pred_len, C).to(x.device)
            
            # Forward with all inputs
            if hasattr(self.ts_model, 'forward'):
                preds = self.ts_model(x, x_mark_enc, dec_inp, x_mark_dec)
            else:
                preds = self.ts_model(x)
        else:
            # Most models just need x
            preds = self.ts_model(x)
        
        # Ensure output shape is [B, pred_len, C]
        if preds.dim() == 2:
            preds = preds.unsqueeze(-1)
        
        # Take last pred_len if needed
        preds = preds[:, -self.pred_len:, :]
        
        return preds
    
    def _get_text_embeddings(self, kwargs) -> torch.Tensor:
        """
        Extract pre-computed text embeddings from kwargs.
        
        Priority: dataset_description > hetero_general
        
        Returns:
            text_emb: [B, L, text_dim] or [B, text_dim]
        """
        text_emb = None
        
        # Try different keys (same as ZhangHanBest)
        if 'dataset_description' in kwargs and kwargs['dataset_description'] is not None:
            text_emb = kwargs['dataset_description']
        elif 'hetero_general' in kwargs and kwargs['hetero_general'] is not None:
            text_emb = kwargs['hetero_general']
        else:
            raise ValueError(
                "Text embeddings not found in kwargs. "
                "Expected 'dataset_description' or 'hetero_general' with pre-computed embeddings."
            )
        
        # Convert numpy to tensor
        if isinstance(text_emb, np.ndarray):
            text_emb = torch.from_numpy(text_emb).float()
        
        # Move to correct device
        device = next(self.parameters()).device
        text_emb = text_emb.to(device)
        
        # Handle various input shapes
        if text_emb.dim() == 4:
            # [B, seq_len, num_items, text_dim] -> [B, text_dim]
            text_emb = text_emb.mean(dim=(1, 2))
        elif text_emb.dim() == 3:
            # [B, L, text_dim] - keep as is for token-level processing
            # OR [B, 1, text_dim] -> squeeze if single token
            if text_emb.shape[1] == 1:
                text_emb = text_emb.squeeze(1)  # [B, text_dim]
        # [B, text_dim] is fine as-is
        
        return text_emb
    
    def count_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

### 5.3 `model_configs/general/MMTSFlib.yaml`

```yaml
# MM-TSFlib Model Configuration
# Prediction-level late fusion based on MM-TSFlib / Time-MMD
#
# Architecture:
#   1. Any unimodal TS model produces predictions
#   2. Text embeddings projected via MLP to prediction dim
#   3. Pooled and normalized
#   4. Ensemble: (1-w)*ts_pred + w*(text_pred + prior)

model: MMTSFlib

# =============================================================================
# Unimodal Time Series Model
# =============================================================================

# Which unimodal model to use as base
# Options: PatchTST, DLinear, iTransformer, FEDformer, Informer, FITS, TimeMixer, Autoformer
unimodal_model_type: "iTransformer"

# Base time series parameters (passed to unimodal model)
seq_len: 96
pred_len: 24
enc_in: 1

# Model-specific parameters (adjust based on unimodal_model_type)
# For iTransformer:
d_model: 512
n_heads: 8
e_layers: 2
d_ff: 2048
dropout: 0.1
activation: gelu
use_norm: True

# For PatchTST (if using):
# d_model: 512
# n_heads: 4
# e_layers: 3
# d_ff: 128
# patch_len: 16
# stride: 8
# revin: True

# For DLinear (if using):
# individual: False

# =============================================================================
# Text Processing Configuration
# =============================================================================

# Dimension of input text embeddings (from LLMEmbeddingProvider)
# Common values: 768 (BERT, GPT-2), 4096 (LLaMA)
input_text_dim: 768

# MLP projection parameters
# Hidden dim = input_text_dim / text_mlp_reduction
text_mlp_reduction: 8  # e.g., 768/8 = 96
text_mlp_dropout: 0.3

# Pooling strategy for aggregating token embeddings
# Options: avg, max, min, attention
pool_type: "avg"

# =============================================================================
# Fusion Configuration
# =============================================================================

# Mixing weight w in: (1-w)*ts_pred + w*text_pred
# MM-TSFlib default is 0.01 (99% TS, 1% text)
# Can tune this for specific datasets
prompt_weight: 0.01

# Whether to add historical prior to text predictions
# Original MM-TSFlib uses prior_history_avg from dataset
use_prior: False

# =============================================================================
# Task Configuration
# =============================================================================

task: TGTSF  # Text-guided time series forecasting

# =============================================================================
# Training Configuration (used by trainer)
# =============================================================================

learning_rate: 0.0001
batch_size: 32
train_epochs: 50
patience: 10
```

---

## 6. Configuration

### 6.1 Example Experiment Suite

```yaml
# configs/experiment_suites/mmtsflib_time_mmd.yaml

suite_name: mmtsflib_time_mmd_benchmark

model_config: MMTSFlib

# Shared settings
shared:
  seq_len: 48
  pred_len: 24
  input_text_dim: 768  # GPT-2 / BERT embeddings
  prompt_weight: 0.01
  pool_type: avg
  
  # iTransformer base model settings
  unimodal_model_type: iTransformer
  d_model: 512
  n_heads: 8
  e_layers: 2
  
  # Training
  learning_rate: 0.0001
  batch_size: 32
  train_epochs: 50

# Dataset variations (Time-MMD)
experiments:
  - name: public_health_flu
    data_config: time_mmd/public_health_flu
    
  - name: economy_trade
    data_config: time_mmd/economy_trade
    
  - name: energy_gas
    data_config: time_mmd/energy_gas

# Ablation: different mixing weights
ablations:
  prompt_weight_sweep:
    base_experiment: public_health_flu
    vary:
      prompt_weight: [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
```

### 6.2 Using Different Unimodal Models

```yaml
# configs/experiment_suites/mmtsflib_model_comparison.yaml

suite_name: mmtsflib_model_comparison

model_config: MMTSFlib

shared:
  seq_len: 96
  pred_len: 24
  input_text_dim: 768
  prompt_weight: 0.01
  data_config: time_mmd/public_health_flu

experiments:
  - name: with_itransformer
    model_config_overrides:
      unimodal_model_type: iTransformer
      d_model: 512
      n_heads: 8
      e_layers: 2
      
  - name: with_patchtst
    model_config_overrides:
      unimodal_model_type: PatchTST
      d_model: 512
      n_heads: 4
      e_layers: 3
      patch_len: 16
      
  - name: with_dlinear
    model_config_overrides:
      unimodal_model_type: DLinear
      individual: False
```

---

## 7. Testing Strategy

### 7.1 Unit Tests (`tests/test_mmtsflib.py`)

```python
"""
Unit tests for MMTSFlib model and layers.
"""

import pytest
import torch
from layers.MMTSFlib_layers import TextToPredsProjection, TextPooling, normalize_embeddings


class TestTextToPredsProjection:
    """Tests for text-to-predictions MLP."""
    
    def test_output_shape(self):
        """Test output dimension matches pred_len."""
        d_llm, pred_len = 768, 24
        proj = TextToPredsProjection(d_llm=d_llm, pred_len=pred_len)
        
        # Token-level input
        x = torch.randn(4, 32, d_llm)  # [B, L, d_llm]
        out = proj(x)
        assert out.shape == (4, 32, pred_len)
        
        # Already pooled input
        x = torch.randn(4, d_llm)  # [B, d_llm]
        out = proj(x)
        assert out.shape == (4, pred_len)
    
    def test_reduction_factor(self):
        """Test different reduction factors."""
        for reduction in [4, 8, 16]:
            proj = TextToPredsProjection(d_llm=768, pred_len=24, reduction_factor=reduction)
            x = torch.randn(2, 768)
            out = proj(x)
            assert out.shape == (2, 24)


class TestTextPooling:
    """Tests for pooling strategies."""
    
    @pytest.mark.parametrize("pool_type", ['avg', 'max', 'min'])
    def test_basic_pooling(self, pool_type):
        """Test avg/max/min pooling."""
        pooling = TextPooling(pool_type=pool_type)
        
        x = torch.randn(4, 32, 24)  # [B, L, D]
        out = pooling(x)
        
        assert out.shape == (4, 24, 1)
    
    def test_attention_pooling(self):
        """Test attention-based pooling."""
        pooling = TextPooling(pool_type='attention')
        
        text_emb = torch.randn(4, 32, 24)  # [B, L, D]
        ts_preds = torch.randn(4, 24, 7)   # [B, pred_len, C]
        
        out = pooling(text_emb, ts_preds)
        assert out.shape == (4, 24, 1)
    
    def test_attention_requires_ts_preds(self):
        """Test attention pooling raises without ts_preds."""
        pooling = TextPooling(pool_type='attention')
        x = torch.randn(4, 32, 24)
        
        with pytest.raises(ValueError):
            pooling(x)  # Missing ts_preds


class TestNormalization:
    """Tests for normalize_embeddings."""
    
    def test_normalization(self):
        """Test instance normalization."""
        x = torch.randn(4, 24, 1)
        out = normalize_embeddings(x)
        
        assert out.shape == x.shape
        # Check normalized (mean ≈ 0, std ≈ 1 per sample)
        assert torch.allclose(out.mean(dim=1), torch.zeros(4, 1), atol=1e-5)


class TestMMTSFlibModel:
    """Integration tests for full model."""
    
    def test_forward_pass(self):
        """Test full forward pass."""
        from models.MMTSFlib import Model
        from utils.tools import dotdict
        
        configs = dotdict({
            'seq_len': 96,
            'pred_len': 24,
            'enc_in': 7,
            'unimodal_model_type': 'DLinear',
            'input_text_dim': 768,
            'prompt_weight': 0.01,
            'pool_type': 'avg',
            'individual': False,
        })
        
        model = Model(configs)
        
        x = torch.randn(4, 96, 7)
        text_emb = torch.randn(4, 768)
        
        out = model(x, dataset_description=text_emb)
        
        assert out.shape == (4, 24, 7)
    
    def test_different_unimodal_models(self):
        """Test with different base models."""
        from models.MMTSFlib import Model
        from utils.tools import dotdict
        
        for model_type in ['DLinear', 'iTransformer']:
            configs = dotdict({
                'seq_len': 96,
                'pred_len': 24,
                'enc_in': 1,
                'unimodal_model_type': model_type,
                'input_text_dim': 768,
                'prompt_weight': 0.01,
                'd_model': 512,
                'n_heads': 8,
                'e_layers': 2,
                'd_ff': 2048,
                'dropout': 0.1,
                'activation': 'gelu',
                'use_norm': True,
                'individual': False,
            })
            
            model = Model(configs)
            x = torch.randn(2, 96, 1)
            text = torch.randn(2, 768)
            
            out = model(x, dataset_description=text)
            assert out.shape == (2, 24, 1), f"Failed for {model_type}"
```

---

## 8. Timeline

### Phase 1: Core Implementation (3-4 days)
- [ ] Implement `layers/MMTSFlib_layers.py`
- [ ] Implement `models/MMTSFlib.py`
- [ ] Add to model registry

### Phase 2: Configuration (1 day)
- [ ] Create `model_configs/general/MMTSFlib.yaml`
- [ ] Create example experiment suite

### Phase 3: Testing (2 days)
- [ ] Unit tests for layers
- [ ] Integration tests with different unimodal models
- [ ] Test with Time-MMD data

### Phase 4: Documentation (1 day)
- [ ] Update docs
- [ ] Add usage examples

---

## Appendix A: Comparison Summary

| | MM-TSFlib | ZhangHanBest |
|--|-----------|--------------|
| Fusion Level | Prediction | Representation |
| Formula | `(1-w)*ts + w*(text+prior)` | `w*text + (1-w)*ts` |
| Default w | 0.01 | 0.5 |
| TS Model Returns | Predictions | Representations |
| Text Projects To | pred_len | d_model |
| Has Prior | Yes | No |
| Has Pooling | Yes (avg/max/min/attn) | No (pre-aggregated) |

## Appendix B: Hyperparameter Recommendations

Based on MM-TSFlib experiments:

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| prompt_weight | 0.01 - 0.1 | Start with 0.01, tune per dataset |
| pool_type | avg | Most stable, attention may help some cases |
| text_mlp_reduction | 8 | Original MM-TSFlib setting |
| unimodal_model_type | iTransformer | Best general performance |
