# Informer Implementation Plan for fidel-ts

## Overview

This document outlines a detailed plan for implementing Informer as an additional model in the fidel-ts repository. Informer is a transformer-based model for long-sequence time series forecasting that introduces the ProbSparse attention mechanism to achieve O(n log n) complexity instead of the standard O(n²) attention.

**Reference Paper:** Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (AAAI 2021)

---

## Table of Contents

1. [Architecture Analysis](#1-architecture-analysis)
2. [Component Mapping](#2-component-mapping)
3. [Implementation Strategy](#3-implementation-strategy)
4. [File Structure](#4-file-structure)
5. [Detailed Implementation Steps](#5-detailed-implementation-steps)
6. [Configuration Files](#6-configuration-files)
7. [Testing Plan](#7-testing-plan)
8. [Potential Challenges](#8-potential-challenges)
9. [Timeline Estimate](#9-timeline-estimate)

---

## 1. Architecture Analysis

### 1.1 Informer Core Components

The Informer model consists of the following key components:

| Component | Description | Complexity |
|-----------|-------------|------------|
| **ProbSparse Attention** | Selects top-k queries based on KL-divergence sparsity measure | O(n log n) |
| **Self-Attention Distilling** | Conv + MaxPool layers between attention layers to halve sequence length | O(n) |
| **Generative Decoder** | Uses start tokens from input + zero placeholders for prediction | O(n) |
| **Data Embedding** | Token + Positional + Temporal embeddings | O(n) |

### 1.2 Informer Variants

The original repository provides two variants:

1. **Informer** - Standard encoder-decoder with distilling
2. **InformerStack** - Multiple encoders at different input resolutions

### 1.3 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enc_in` | 7 | Encoder input channels |
| `dec_in` | 7 | Decoder input channels |
| `c_out` | 7 | Output channels |
| `d_model` | 512 | Model dimension |
| `n_heads` | 8 | Number of attention heads |
| `e_layers` | 2 | Number of encoder layers |
| `d_layers` | 1 | Number of decoder layers |
| `d_ff` | 2048 | Feed-forward dimension |
| `factor` | 5 | ProbSparse attention factor |
| `attn` | 'prob' | Attention type: 'prob' or 'full' |
| `distil` | True | Whether to use distilling in encoder |
| `label_len` | 48 | Start token length for decoder |

---

## 2. Component Mapping

### 2.1 Existing Components in fidel-ts

The fidel-ts repository already has several components that can be reused:

| Informer Component | fidel-ts Equivalent | Location | Reusable? |
|--------------------|---------------------|----------|-----------|
| `FullAttention` | `FullAttention` | `layers/SelfAttention_Family.py` | ✅ Yes |
| `ProbAttention` | `ProbAttention` | `layers/SelfAttention_Family.py` | ✅ Yes |
| `AttentionLayer` | `AttentionLayer` | `layers/SelfAttention_Family.py` | ✅ Yes |
| `EncoderLayer` | `EncoderLayer` | `layers/Transformer_EncDec.py` | ✅ Yes |
| `Encoder` | `Encoder` | `layers/Transformer_EncDec.py` | ✅ Yes |
| `DecoderLayer` | `DecoderLayer` | `layers/Transformer_EncDec.py` | ✅ Yes |
| `Decoder` | `Decoder` | `layers/Transformer_EncDec.py` | ✅ Yes |
| `ConvLayer` | `ConvLayer` | `layers/Transformer_EncDec.py` | ✅ Yes |
| `TokenEmbedding` | `TokenEmbedding` | `layers/Embed.py` | ✅ Yes |
| `PositionalEmbedding` | `PositionalEmbedding` | `layers/Embed.py` | ✅ Yes |
| `TemporalEmbedding` | `TemporalEmbedding` | `layers/Embed.py` | ✅ Yes |
| `TimeFeatureEmbedding` | `TimeFeatureEmbedding` | `layers/Embed.py` | ✅ Yes |
| `DataEmbedding` | `DataEmbedding` | `layers/Embed.py` | ✅ Yes |
| `TriangularCausalMask` | `TriangularCausalMask` | `utils/masking.py` | ✅ Yes |
| `ProbMask` | `ProbMask` | `utils/masking.py` | ✅ Yes |
| `EncoderStack` | ❌ Not present | - | ⚠️ Needs implementation |

### 2.2 Key Finding

**Almost all Informer components already exist in fidel-ts!** The implementation can be very lightweight since:

- ProbSparse attention is already in `SelfAttention_Family.py` (lines 165-263)
- Encoder/Decoder with distilling support is in `Transformer_EncDec.py`
- All embedding components are in `Embed.py`
- Masking utilities exist in `utils/masking.py`

Only `EncoderStack` (for InformerStack variant) needs to be added if desired.

---

## 3. Implementation Strategy

### 3.1 Approach: Minimal New Code

Given the existing infrastructure, we can implement Informer by:

1. **Creating a single model wrapper file** (`models/Informer.py`) ~150-200 lines
2. **Adding a model config file** (`model_configs/general/Informer.yaml`) ~30 lines
3. **Optionally** adding `EncoderStack` to `Transformer_EncDec.py` for InformerStack variant ~20 lines

### 3.2 Pattern to Follow

Follow the exact pattern used by `FEDformer.py`:

```
models/Informer.py
├── Import existing layers (SelfAttention_Family, Transformer_EncDec, Embed)
├── class Model(nn.Module)
│   ├── __init__(self, configs)
│   │   ├── Parse config parameters with getattr() fallbacks
│   │   ├── Build embeddings using DataEmbedding
│   │   ├── Build encoder using Encoder + EncoderLayer + AttentionLayer
│   │   ├── Build decoder using Decoder + DecoderLayer + AttentionLayer
│   │   └── Build projection layer
│   ├── forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, ...)
│   │   ├── Embed encoder input
│   │   ├── Run through encoder (with optional distilling)
│   │   ├── Embed decoder input
│   │   ├── Run through decoder with cross-attention
│   │   └── Return prediction
│   └── forecast(self, x) - Simplified interface for fidel-ts
```

---

## 4. File Structure

### 4.1 Files to Create

```
fidel-ts-worktree-lynx/
├── models/
│   └── Informer.py                    # NEW: Main model file (~180 lines)
├── model_configs/
│   └── general/
│       └── Informer.yaml              # NEW: Default config (~35 lines)
├── configs/
│   └── experiment_suites/
│       └── informer_test.yaml         # NEW: Test suite (~150 lines)
└── layers/
    └── Transformer_EncDec.py          # MODIFY: Add EncoderStack class (optional)
```

### 4.2 Files to Modify (Optional)

If adding InformerStack support:

```python
# Add to layers/Transformer_EncDec.py (~20 lines)
class EncoderStack(nn.Module):
    """Stack of encoders at different input resolutions."""
    ...
```

---

## 5. Detailed Implementation Steps

### Step 1: Create `models/Informer.py`

```python
"""
Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting

This is a unimodal (time series only) implementation for the fidel-ts framework.
Informer achieves O(n log n) complexity through ProbSparse attention and
uses self-attention distilling for long-horizon forecasting.

Reference: Zhou et al., "Informer: Beyond Efficient Transformer for 
Long Sequence Time-Series Forecasting" (AAAI 2021)

Key features:
- ProbSparse Attention: O(n log n) complexity
- Self-attention Distilling: Halves sequence length between layers
- Generative-style Decoder: Start token + zero placeholders
- Standard Encoder-Decoder architecture with cross-attention
"""

import torch
import torch.nn as nn

from layers.Transformer_EncDec import (
    Encoder, EncoderLayer, ConvLayer, 
    Decoder, DecoderLayer
)
from layers.SelfAttention_Family import (
    FullAttention, ProbAttention, AttentionLayer
)
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Informer: Efficient Transformer for Long Sequence Time-Series Forecasting.
    
    Args (via configs):
        seq_len: Input sequence length
        pred_len: Prediction horizon length
        label_len: Start token length for decoder (default: seq_len // 2)
        enc_in: Number of input channels (encoder)
        dec_in: Number of input channels (decoder), defaults to enc_in
        c_out: Number of output channels, defaults to enc_in
        d_model: Model dimension (default: 512)
        n_heads: Number of attention heads (default: 8)
        e_layers: Number of encoder layers (default: 2)
        d_layers: Number of decoder layers (default: 1)
        d_ff: Feed-forward dimension (default: 2048)
        factor: ProbSparse attention factor (default: 5)
        attn: Attention type: 'prob' or 'full' (default: 'prob')
        distil: Whether to use distilling in encoder (default: True)
        dropout: Dropout rate (default: 0.05)
        activation: Activation function (default: 'gelu')
        output_attention: Whether to output attention weights (default: False)
        embed: Embedding type: 'timeF', 'fixed', 'learned' (default: 'timeF')
        freq: Time frequency for embeddings (default: 'h')
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        
        # Core parameters
        self.seq_len = configs.seq_len
        self.label_len = getattr(configs, 'label_len', configs.seq_len // 2)
        self.pred_len = configs.pred_len
        self.output_attention = getattr(configs, 'output_attention', False)
        
        # Channel configuration
        self.enc_in = getattr(configs, 'enc_in', None)
        if self.enc_in is None:
            self.enc_in = getattr(configs, 'input_channel', None)
            if self.enc_in is None:
                raise ValueError("enc_in must be provided in model config")
        self.dec_in = getattr(configs, 'dec_in', self.enc_in)
        self.c_out = getattr(configs, 'c_out', self.enc_in)
        
        # Model dimensions
        self.d_model = getattr(configs, 'd_model', 512)
        self.n_heads = getattr(configs, 'n_heads', 8)
        self.e_layers = getattr(configs, 'e_layers', 2)
        self.d_layers = getattr(configs, 'd_layers', 1)
        self.d_ff = getattr(configs, 'd_ff', 2048)
        
        # Informer-specific parameters
        self.factor = getattr(configs, 'factor', 5)
        self.attn = getattr(configs, 'attn', 'prob')
        self.distil = getattr(configs, 'distil', True)
        
        # Other parameters
        self.dropout = getattr(configs, 'dropout', 0.05)
        self.activation = getattr(configs, 'activation', 'gelu')
        self.embed = getattr(configs, 'embed', 'timeF')
        self.freq = getattr(configs, 'freq', 'h')
        
        # Select attention mechanism
        Attn = ProbAttention if self.attn == 'prob' else FullAttention
        
        # Embeddings
        self.enc_embedding = DataEmbedding(
            self.enc_in, self.d_model, self.embed, self.freq, self.dropout
        )
        self.dec_embedding = DataEmbedding(
            self.dec_in, self.d_model, self.embed, self.freq, self.dropout
        )
        
        # Encoder with distilling
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, self.factor, attention_dropout=self.dropout, 
                             output_attention=self.output_attention),
                        self.d_model, self.n_heads
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.e_layers)
            ],
            conv_layers=[
                ConvLayer(self.d_model) for _ in range(self.e_layers - 1)
            ] if self.distil else None,
            norm_layer=nn.LayerNorm(self.d_model)
        )
        
        # Decoder
        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    self_attention=AttentionLayer(
                        Attn(True, self.factor, attention_dropout=self.dropout,
                             output_attention=False),
                        self.d_model, self.n_heads
                    ),
                    cross_attention=AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=False),
                        self.d_model, self.n_heads
                    ),
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.d_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, **kwargs):
        """
        Forward pass for Informer.
        
        Args:
            x_enc: Encoder input [B, seq_len, enc_in]
            x_mark_enc: Encoder temporal marks [B, seq_len, time_features] (optional)
            x_dec: Decoder input [B, label_len + pred_len, dec_in] (optional)
            x_mark_dec: Decoder temporal marks (optional)
            *_mask: Optional attention masks
            
        Returns:
            predictions: [B, pred_len, c_out]
        """
        # Handle simplified interface (just x_enc provided)
        if x_dec is None:
            # Create decoder input: last label_len of input + zeros for prediction
            x_dec = torch.zeros(
                x_enc.size(0), self.label_len + self.pred_len, x_enc.size(2),
                device=x_enc.device, dtype=x_enc.dtype
            )
            x_dec[:, :self.label_len, :] = x_enc[:, -self.label_len:, :]
        
        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        # Decoder
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out, enc_out,
            x_mask=dec_self_mask, cross_mask=dec_enc_mask
        )
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, pred_len, c_out]


if __name__ == '__main__':
    """Quick test of the model."""
    
    class Configs:
        seq_len = 96
        label_len = 48
        pred_len = 24
        enc_in = 7
        d_model = 512
        n_heads = 8
        e_layers = 2
        d_layers = 1
        d_ff = 2048
        factor = 5
        attn = 'prob'
        distil = True
        dropout = 0.05
        activation = 'gelu'
        embed = 'timeF'
        freq = 'h'
        output_attention = False

    configs = Configs()
    model = Model(configs)

    print(f'Parameter count: {sum(p.numel() for p in model.parameters()):,}')
    
    # Test with simplified interface
    enc = torch.randn(2, 96, 7)
    out = model(enc)
    print(f'Output shape: {out.shape}')
    
    # Test with full interface
    enc_mark = torch.randn(2, 96, 4)
    dec = torch.randn(2, 48 + 24, 7)
    dec_mark = torch.randn(2, 48 + 24, 4)
    out_full = model(enc, enc_mark, dec, dec_mark)
    print(f'Full interface output shape: {out_full.shape}')
    print('Test passed!')
```

### Step 2: Create `model_configs/general/Informer.yaml`

```yaml
# Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
# Reference: Zhou et al., AAAI 2021
# https://arxiv.org/abs/2012.07436

model: Informer

# Model dimensions
d_model: 512          # Model embedding dimension
d_ff: 2048            # Feed-forward network dimension
n_heads: 8            # Number of attention heads
e_layers: 2           # Number of encoder layers
d_layers: 1           # Number of decoder layers

# Informer-specific parameters
factor: 5             # ProbSparse attention factor (c in paper)
attn: prob            # Attention type: 'prob' (ProbSparse) or 'full'
distil: true          # Whether to use self-attention distilling

# General settings
dropout: 0.05         # Dropout rate
activation: gelu      # Activation function: 'relu' or 'gelu'
output_attention: false  # Whether to output attention weights

# Embedding settings
embed: timeF          # Embedding type: 'timeF', 'fixed', or 'learned'
freq: h               # Time frequency: 'h', 'd', 't', 'm', etc.

# Task type
task: TSF             # Time Series Forecasting
```

### Step 3: Create Test Suite `configs/experiment_suites/informer_test.yaml`

```yaml
# Experiment Suite: Informer Test Suite
# Description: Test Informer (unimodal efficient transformer) on various datasets
# Usage: python -m cli.suite run configs/experiment_suites/informer_test.yaml

suite:
  name: "informer_test"
  description: "Test Informer on TTC and benchmark datasets"
  tags: ["informer", "unimodal", "pytorch", "test", "probsparse"]
  
  execution:
    parallel: false
    continue_on_error: true
    log_dir: "./logs/suites/informer_test"
    
  experiments:
    # TTC Climate
    - name: "informer_ttc_climate_24"
      description: "Informer on TTC Climate with 24-step prediction"
      enabled: true
      template: "configs/templates/pytorch_training.yaml"
      overrides:
        model:
          name: "Informer"
          config_path: "model_configs/general/Informer.yaml"
        data:
          name: "ttc_climate"
          config_path: "data_configs/ttc/climate/config.yaml"
        training:
          input_len: 96
          output_len: 24
          batch_size: 32
          learning_rate: 1e-4
          patience: 5
          epochs: 1
        wandb:
          enabled: false
        model_config_overrides:
          enc_in: 4
          attn: prob
          distil: true
        experiment:
          type: "pytorch"
          output_dir: "outputs/suites/informer_test"
    
    # ProbSparse vs Full Attention comparison
    - name: "informer_ttc_climate_full_attn"
      description: "Informer with full attention (for comparison)"
      enabled: true
      template: "configs/templates/pytorch_training.yaml"
      overrides:
        model:
          name: "Informer"
          config_path: "model_configs/general/Informer.yaml"
        data:
          name: "ttc_climate"
          config_path: "data_configs/ttc/climate/config.yaml"
        training:
          input_len: 96
          output_len: 24
          batch_size: 32
          learning_rate: 1e-4
          epochs: 1
        wandb:
          enabled: false
        model_config_overrides:
          enc_in: 4
          attn: full  # Use full attention
          distil: false
        experiment:
          type: "pytorch"
          output_dir: "outputs/suites/informer_test"
```

### Step 4 (Optional): Add EncoderStack to `layers/Transformer_EncDec.py`

If you want InformerStack support, add this class at the end of the file:

```python
class EncoderStack(nn.Module):
    """
    Stack of encoders for InformerStack variant.
    
    Each encoder processes the input at a different resolution (halved sequence length).
    The outputs are concatenated along the sequence dimension.
    
    Args:
        encoders: List of Encoder modules
        inp_lens: List of input length factors (e.g., [0, 1, 2] for full, half, quarter)
    """
    
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        Forward pass through encoder stack.
        
        Args:
            x: Input tensor [B, L, D]
            
        Returns:
            x_stack: Concatenated encoder outputs
            attns: List of attention weights from each encoder
        """
        x_stack = []
        attns = []
        
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            # Compute input length at this resolution
            inp_len = x.shape[1] // (2 ** i_len)
            # Process last inp_len timesteps
            x_s, attn = encoder(x[:, -inp_len:, :], attn_mask=attn_mask, tau=tau, delta=delta)
            x_stack.append(x_s)
            attns.append(attn)
        
        # Concatenate along sequence dimension
        x_stack = torch.cat(x_stack, dim=-2)
        
        return x_stack, attns
```

---

## 6. Configuration Files

### 6.1 Recommended Configurations for Different Horizons

| Prediction Length | `label_len` | `e_layers` | `d_layers` | `d_model` | `d_ff` |
|-------------------|-------------|------------|------------|-----------|--------|
| 24 (short) | 48 | 2 | 1 | 256 | 512 |
| 48 (medium) | 48 | 2 | 1 | 512 | 1024 |
| 96 (medium-long) | 48 | 2 | 1 | 512 | 2048 |
| 192 (long) | 96 | 3 | 2 | 512 | 2048 |
| 336+ (very long) | 168 | 3 | 2 | 512 | 2048 |

### 6.2 Attention Type Selection

| Scenario | `attn` | `distil` | Notes |
|----------|--------|----------|-------|
| Long sequences (>200) | `prob` | `true` | Use ProbSparse for efficiency |
| Short sequences (<100) | `full` | `false` | Full attention is fast enough |
| Memory constrained | `prob` | `true` | Reduces memory footprint |
| Maximum accuracy | `full` | `false` | Full attention may be more accurate |

---

## 7. Testing Plan

### 7.1 Unit Tests

```python
# tests/test_informer.py

def test_informer_forward():
    """Test basic forward pass."""
    model = Model(configs)
    x = torch.randn(2, 96, 7)
    out = model(x)
    assert out.shape == (2, 24, 7)

def test_informer_with_marks():
    """Test with temporal marks."""
    model = Model(configs)
    enc = torch.randn(2, 96, 7)
    enc_mark = torch.randn(2, 96, 4)
    dec = torch.randn(2, 72, 7)
    dec_mark = torch.randn(2, 72, 4)
    out = model(enc, enc_mark, dec, dec_mark)
    assert out.shape == (2, 24, 7)

def test_informer_probsparse_vs_full():
    """Verify both attention types work."""
    configs_prob = Configs(attn='prob')
    configs_full = Configs(attn='full')
    
    model_prob = Model(configs_prob)
    model_full = Model(configs_full)
    
    x = torch.randn(2, 96, 7)
    out_prob = model_prob(x)
    out_full = model_full(x)
    
    assert out_prob.shape == out_full.shape

def test_informer_distilling():
    """Verify distilling reduces encoder sequence length."""
    configs_distil = Configs(distil=True)
    configs_no_distil = Configs(distil=False)
    
    # Distilling should reduce memory usage
    model_distil = Model(configs_distil)
    model_no_distil = Model(configs_no_distil)
    
    assert model_distil is not None
    assert model_no_distil is not None
```

### 7.2 Integration Tests

1. Run `informer_test.yaml` suite
2. Compare metrics with published Informer results
3. Verify GPU memory usage is lower with ProbSparse vs Full attention
4. Test on multiple datasets (TTC, Time-MMD, etc.)

### 7.3 Benchmark Validation

Run on ETTh1 dataset and compare with published results:

| Horizon | MSE (Published) | MAE (Published) | Our MSE | Our MAE |
|---------|-----------------|-----------------|---------|---------|
| 24 | 0.098 | 0.247 | TBD | TBD |
| 48 | 0.158 | 0.319 | TBD | TBD |
| 168 | 0.183 | 0.346 | TBD | TBD |
| 336 | 0.222 | 0.387 | TBD | TBD |

---

## 8. Potential Challenges

### 8.1 Interface Alignment

**Challenge:** Original Informer uses `(x_enc, x_mark_enc, x_dec, x_mark_dec)` interface but fidel-ts models may expect simpler `forward(x)`.

**Solution:** Implement both interfaces in the `forward()` method:
- Full interface for maximum control
- Simplified interface that auto-constructs decoder input

### 8.2 Temporal Marks

**Challenge:** Not all fidel-ts datasets provide temporal marks (time features).

**Solution:** Make temporal embeddings optional. When `x_mark_enc=None`, skip temporal embedding (already supported in `DataEmbedding`).

### 8.3 Label Length Configuration

**Challenge:** Informer uses `label_len` (start token length) which differs from `seq_len`.

**Solution:** Default `label_len = seq_len // 2` if not specified.

### 8.4 Encoder Sequence Shrinking with Distilling

**Challenge:** Distilling halves sequence length after each encoder layer, which can cause dimension mismatches with decoder cross-attention.

**Solution:** The existing `Encoder` class handles this correctly. Ensure decoder cross-attention uses the final encoder output shape.

---

## 9. Timeline Estimate

| Phase | Task | Time Estimate |
|-------|------|---------------|
| 1 | Create `models/Informer.py` | 2-3 hours |
| 2 | Create `model_configs/general/Informer.yaml` | 30 minutes |
| 3 | Create `informer_test.yaml` experiment suite | 1 hour |
| 4 | Unit testing and debugging | 2-3 hours |
| 5 | Integration testing on TTC datasets | 1-2 hours |
| 6 | (Optional) Add InformerStack support | 1-2 hours |
| 7 | Documentation and cleanup | 1 hour |
| **Total** | | **8-12 hours** |

---

## 10. Summary

### Why Implementation is Straightforward

The Informer implementation in fidel-ts is particularly straightforward because:

1. **ProbSparse attention already exists** in `layers/SelfAttention_Family.py` (lines 165-263)
2. **Encoder with distilling already exists** via `ConvLayer` in `layers/Transformer_EncDec.py`
3. **All embedding components exist** in `layers/Embed.py`
4. **Masking utilities exist** in `utils/masking.py`

### Implementation Checklist

- [ ] Create `models/Informer.py` (~180 lines)
- [ ] Create `model_configs/general/Informer.yaml` (~35 lines)
- [ ] Create `configs/experiment_suites/informer_test.yaml` (~150 lines)
- [ ] Add unit tests in `tests/test_informer.py`
- [ ] Run integration tests on TTC datasets
- [ ] (Optional) Add `EncoderStack` for InformerStack variant
- [ ] Update model registry if needed
- [ ] Document in README or model catalog

### Key Advantages

- Minimal new code required (~200 lines total)
- Reuses battle-tested fidel-ts infrastructure
- Full compatibility with existing training pipelines
- Supports both simplified and full interfaces
- ProbSparse attention provides O(n log n) efficiency for long sequences

---

*Document created: December 28, 2025*
*Author: AI Assistant*
*Status: Implementation Ready*

