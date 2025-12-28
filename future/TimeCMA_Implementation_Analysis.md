# TimeCMA Implementation Analysis & Comparison with TGTSF and Lynx-FiLM-Raw

## Table of Contents
1. [Overview](#overview)
2. [TimeCMA Implementation Details](#timecma-implementation-details)
3. [TGTSF Implementation Details](#tgtsf-implementation-details)
4. [Lynx-FiLM-Raw Implementation Details](#lynx-film-raw-implementation-details)
5. [Architectural Comparison](#architectural-comparison)
6. [Key Differences Summary](#key-differences-summary)
7. [Fidel-TS Port Implementation Plan](#fidel-ts-port-implementation-plan)

---

## Overview

This document provides a detailed analysis of the **TimeCMA** (Time Series Cross-Modality Alignment) model from AAAI 2025, comparing it with **TGTSF** (Text-Guided Time Series Forecasting) and **Lynx-FiLM-Raw** models.

### High-Level Comparison

| Aspect | TimeCMA | TGTSF | Lynx-FiLM-Raw |
|--------|---------|-------|---------------|
| **Paper** | AAAI 2025 | - | - |
| **Text Integration** | Cross-Modal Attention | Text-Temp Cross Attention | FiLM Modulation |
| **Text Source** | LLM (GPT-2) Last Token | Pre-computed Embeddings | Pre-computed Embeddings |
| **Architecture** | Dual-branch Encoder | Patch-based Transformer | iTransformer + FiLM |
| **Normalization** | RevIN | RevIN | RevIN or use_norm |

---

## TimeCMA Implementation Details

### Core Architecture (`models/TimeCMA.py`)

TimeCMA uses a **dual-modality encoding** approach with cross-modal alignment:

```python
class Dual(nn.Module):
    def __init__(self, device, channel, num_nodes, seq_len, pred_len, 
                 dropout_n, d_llm, e_layer, d_layer, d_ff, head):
        # Key components:
        # 1. RevIN Normalization
        self.normalize_layers = Normalize(num_nodes, affine=False)
        
        # 2. Length-to-Feature projection
        self.length_to_feature = nn.Linear(seq_len, channel)
        
        # 3. Time Series Encoder (Transformer)
        self.ts_encoder = nn.TransformerEncoder(...)
        
        # 4. Prompt Encoder (for LLM embeddings)
        self.prompt_encoder = nn.TransformerEncoder(...)
        
        # 5. Cross-Modal Alignment Layer
        self.cross = CrossModal(...)
        
        # 6. Transformer Decoder
        self.decoder = nn.TransformerDecoder(...)
        
        # 7. Output Projection
        self.c_to_length = nn.Linear(channel, pred_len)
```

### Data Flow

```
Input: [B, L, N] (Batch, Seq_Length, Num_Nodes)
       + Pre-computed LLM embeddings [B, E, N] (from GPT-2 last token)

1. RevIN Normalization
   └─► [B, L, N] → [B, L, N] (normalized)

2. Permute & Project
   └─► [B, L, N] → [B, N, L] → [B, N, C] (length_to_feature)

3. Time Series Encoding
   └─► [B, N, C] → [B, N, C] (ts_encoder)
   └─► [B, N, C] → [B, C, N] (permute for cross-attention)

4. Prompt Encoding
   └─► [B, N, E] → [B, N, E] (prompt_encoder)
   └─► [B, N, E] → [B, E, N] (permute for cross-attention)

5. Cross-Modal Alignment
   └─► Q: [B, C, N], KV: [B, E, N] → [B, C, N]
   └─► [B, C, N] → [B, N, C] (permute)

6. Decoder
   └─► [B, N, C] → [B, N, C]

7. Output Projection + Denormalization
   └─► [B, N, C] → [B, N, pred_len] → [B, pred_len, N]
```

### Cross-Modal Alignment (`layers/Cross_Modal_Align.py`)

The core innovation of TimeCMA is cross-modal alignment using a custom attention mechanism:

```python
class CrossModal(nn.Module):
    """
    Retrieves 'disentangled and robust' embeddings by aligning:
    - Time series embeddings (disentangled but weak)
    - LLM prompt embeddings (entangled but robust)
    """
    def forward(self, q, k, v):
        # Q: Time series encoded features [B, C, N]
        # K, V: LLM prompt embeddings [B, E, N]
        # Output: Cross-modality aligned features [B, C, N]
```

### LLM Embedding Generation (`storage/gen_prompt_emb.py`)

TimeCMA uses **GPT-2** to generate prompt embeddings:

```python
class GenPromptEmb(nn.Module):
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)
    
    def _prepare_prompt(self, input_template, in_data, in_data_mark, i, j):
        # Template: "From [t1] to [t2], the values were value1, ..., valuen 
        #            every {frequency}. The total trend value was Trends"
        pass
    
    def generate_embeddings(self, in_data, in_data_mark):
        # Returns: ONLY the last token embedding [B, d_llm, N]
        # Key insight: Most essential temporal info is in the last token
        return last_token_emb
```

**Key Design Decision**: Only the **last token** of the LLM output is used, reducing computational costs while preserving essential temporal information.

---

## TGTSF Implementation Details

### Core Architecture (`models/TGTSF.py`)

TGTSF uses a **patch-based** approach with text-temporal cross-attention:

```python
class Model(nn.Module):
    def __init__(self, configs):
        # 1. Patch-based Time Series Encoder
        self.TS_encoder = TS_encoder(
            embedding_dim=d_model,
            patch_len=patch_len,
            stride=stride,
            ...
        )
        
        # 2. Text Encoder (for news + channel descriptions)
        self.text_encoder = text_encoder(
            cross_layer=configs.cross_layers,
            embedding_dim=configs.text_dim,
            ...
        )
        
        # 3. Text-Temporal Mixer
        self.mixer = text_temp_cross_block(
            text_embedding_dim=configs.text_dim,
            temp_embedding_dim=d_model,
            ...
        )
        
        # 4. Prediction Head
        self.head = nn.Linear(d_model, patch_len)
```

### Data Flow

```
Input: x [B, L, C] + news [B, l, n, D] + channel_desc [B, C, D]

1. RevIN Normalization
   └─► [B, L, C] → [B, L, C]

2. Time Series Encoding (Patch-based)
   └─► [B, L, C] → patches → [B, N_patches, C, d_model]

3. Text Encoding (News + Descriptions)
   └─► news + desc → [B, l, C, text_dim]

4. Text-Temporal Mixing (Cross-Attention)
   └─► text_emb + temp_emb → [B, N_patches, C, d_model]

5. Prediction Head + Patch Reconstruction
   └─► [B, N_patches, C, patch_len] → [B, C, pred_len]

6. Denormalization
   └─► [B, pred_len, C]
```

### Text Encoder (`layers/TGTSF_torch.py`)

```python
class text_encoder(nn.Module):
    """
    Encodes news and channel descriptions using cross-attention.
    
    news_emb: [B, l, n, D] - News embeddings over time segments
    description_emb: [B, l, C, D] - Channel descriptions
    
    Returns: [B, l, C, D] - Channel-specific text embeddings
    """
    def forward(self, news_emb, description_emb):
        # Cross-attention: description queries news
        text_emb = self.cross_encoder(tgt=text_emb, memory=news_emb)
        # Add positional encoding
        text_emb = text_emb + self.W_pos
        return text_emb
```

### Text-Temporal Mixer

```python
class text_temp_cross_block(nn.Module):
    """
    Mixes text and temporal embeddings via cross-attention.
    
    text_emb: [B, l, C, D_text]
    temp_emb: [B, N_patches, C, D_temp]
    """
    def forward(self, text_emb, temp_emb):
        # text as Query, temp as Key/Value
        result = self.cross_encoder(tgt=text_emb, memory=temp_emb)
        result = self.self_encoder(tgt=result, memory=result)
        return result
```

### Patch Reconstruction (Vectorized)

```python
def patch_reconstruction(self, x):
    """
    Reconstructs overlapping patches into continuous sequence.
    Uses torch.nn.functional.fold for GPU-parallel vectorization.
    
    Input: [B, C, N_patches, patch_len]
    Output: [B, C, pred_len]
    """
    # Vectorized using F.fold (significant speedup over loop-based)
    output_sum = F.fold(x_reshaped, output_size=(1, covered_length), ...)
    output_count = F.fold(ones_reshaped, ...)
    output = output_sum / output_count
    return output
```

---

## Lynx-FiLM-Raw Implementation Details

### Core Architecture (`models/lynx_film_raw.py`)

Lynx-FiLM-Raw uses **FiLM (Feature-wise Linear Modulation)** for text conditioning:

```python
class Model(nn.Module):
    def __init__(self, configs):
        # 1. Text Encoder (from TGTSF)
        self.text_encoder = text_encoder(...)
        
        # 2. Main Model: iTransformerFilm
        self.model = iTransformerFilm(model_configs)
        
        # 3. Optional text projection (input_text_dim → text_dim)
        if input_text_dim != text_dim:
            self.text_projection = nn.Linear(input_text_dim, text_dim)
```

### Data Flow

```
Input: x [B, L, C] + news [B, l, n, D] + channel_desc [B, C, D]

1. Normalization (RevIN or use_norm)
   └─► [B, L, C] → [B, L, C] (normalized)

2. Project Text Embeddings (if needed)
   └─► [B, l, n, input_D] → [B, l, n, text_D]

3. Text Encoding
   └─► news + desc → [B, L_text, C, text_dim]
   └─► permute → [B, C, L_text, text_dim]

4. iTransformerFilm (FiLM-modulated)
   └─► x_norm + text_emb → [B, pred_len, C]

5. Denormalization
   └─► [B, pred_len, C]
```

### iTransformerFilm (`layers/lynx_film_layers.py`)

```python
class iTransformerFilm(nn.Module):
    """
    iTransformer with FiLM modulation from text embeddings.
    Inverts the sequence dimension (tokens = channels).
    """
    def __init__(self, configs):
        # Inverted embedding: [B, L, N] → [B, N, d_model]
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model)
        
        # Encoder with FiLM layers
        self.encoder = EncoderFilm(
            attn_layers=[EncoderLayerFilm(...) for _ in range(e_layers)],
            film_generators=[FiLMGenerator(...) for _ in range(e_layers)]
        )
        
        # Output projection
        self.projector = nn.Linear(d_model, pred_len)
```

### FiLM Modulation (`layers/FiLM_layers.py`)

```python
class FiLMGenerator(nn.Module):
    """
    Generates FiLM parameters (gamma, beta) from text embeddings.
    
    Input: text_emb [B, C, L, text_dim]
    Output: gamma1, beta1, gamma2, beta2 for each layer
    """
    def forward(self, x):
        # Flatten: [B, C, L, D] → [B, C, L*D]
        x = x.view(B, C, L * D)
        params = self.net(x)  # MLP
        gamma1, beta1, gamma2, beta2 = torch.chunk(params, 4, dim=-1)
        return gamma1, beta1, gamma2, beta2

class EncoderLayerFilm(nn.Module):
    """
    Encoder layer with FiLM modulation at two points:
    1. Post-attention normalization
    2. Post-FFN normalization
    """
    def forward(self, x, gamma1, beta1, gamma2, beta2, ...):
        # Self-Attention
        x = x + self.dropout(self.attention(x, x, x))
        x = self.norm1(x)
        
        # FiLM Modulation 1 (after attention)
        x = (1 + gamma1) * x + beta1
        
        # Feed-Forward
        y = self.linear2(self.activation(self.linear1(x)))
        x = self.norm2(x + y)
        
        # FiLM Modulation 2 (after FFN)
        x = (1 + gamma2) * x + beta2
        
        return x
```

---

## Architectural Comparison

### 1. Text Integration Strategy

| Model | Strategy | Description |
|-------|----------|-------------|
| **TimeCMA** | Cross-Modal Attention | Retrieves aligned embeddings using Q(TS)×KV(LLM) |
| **TGTSF** | Cross-Attention Mixer | Text queries temporal features, then self-attention |
| **Lynx-FiLM-Raw** | FiLM Modulation | Affine transformation: `(1+γ)x + β` |

### 2. Text Embedding Source

| Model | Source | Processing |
|-------|--------|------------|
| **TimeCMA** | GPT-2 last token | Pre-computed, stored as `.h5` files |
| **TGTSF** | Pre-computed embeddings | News + channel descriptions |
| **Lynx-FiLM-Raw** | Pre-computed embeddings | Same as TGTSF with optional projection |

### 3. Time Series Encoding

| Model | Strategy | Key Feature |
|-------|----------|-------------|
| **TimeCMA** | Length-to-Feature + Transformer | Projects sequence length to channel dimension |
| **TGTSF** | Patch-based Transformer | Unfolds into patches with overlap |
| **Lynx-FiLM-Raw** | iTransformer (Inverted) | Channels as tokens, length as features |

### 4. Normalization

| Model | Method | Implementation |
|-------|--------|----------------|
| **TimeCMA** | RevIN | `Normalize(num_features, affine=False)` |
| **TGTSF** | RevIN | Manual mean/std computation |
| **Lynx-FiLM-Raw** | RevIN or use_norm | Configurable, handled externally |

### 5. Prediction Strategy

| Model | Approach |
|-------|----------|
| **TimeCMA** | Direct projection: `channel → pred_len` |
| **TGTSF** | Patch-wise prediction + reconstruction |
| **Lynx-FiLM-Raw** | Direct projection: `d_model → pred_len` |

---

## Key Differences Summary

### TimeCMA Innovations
1. **Dual-Modality Encoding**: Separate branches for TS (disentangled/weak) and LLM (entangled/robust)
2. **Last Token Focus**: Uses only the last LLM token to reduce computation
3. **Cross-Modal Alignment**: Retrieves "best of both worlds" embeddings
4. **Pre-stored Embeddings**: H5 files for fast training/inference

### TGTSF Innovations
1. **Patch-Based Processing**: Captures local patterns with overlapping patches
2. **Text-Temporal Mixer**: Sophisticated cross-attention mechanism
3. **Vectorized Reconstruction**: Efficient GPU-parallel patch merging
4. **Channel Descriptions**: Per-channel semantic information

### Lynx-FiLM-Raw Innovations
1. **FiLM Modulation**: Lightweight affine conditioning
2. **iTransformer Architecture**: Inverted attention (channels as tokens)
3. **Multi-Point Modulation**: FiLM applied after attention AND FFN
4. **Text Projection**: Handles dimension mismatch gracefully

---

## Code Structure Comparison

```
TimeCMA/
├── models/
│   └── TimeCMA.py          # Dual-branch model
├── layers/
│   ├── Cross_Modal_Align.py # Cross-modal attention
│   └── StandardNorm.py      # RevIN implementation
├── storage/
│   └── gen_prompt_emb.py    # GPT-2 embedding generation
└── data_provider/
    └── data_loader_emb.py   # Loads pre-computed embeddings

TGTSF/
├── models/
│   └── TGTSF.py            # Main model
└── layers/
    └── TGTSF_torch.py      # Text encoder, TS encoder, Mixer

Lynx-FiLM-Raw/
├── models/
│   └── lynx_film_raw.py    # Main model wrapper
└── layers/
    ├── lynx_film_layers.py # iTransformerFilm
    ├── FiLM_layers.py      # FiLM generator & layers
    └── TGTSF_torch.py      # Reuses text_encoder
```

---

## Computational Considerations

| Aspect | TimeCMA | TGTSF | Lynx-FiLM-Raw |
|--------|---------|-------|---------------|
| **LLM Dependency** | GPT-2 (offline) | None | None |
| **Memory** | Medium | High (patches) | Low-Medium |
| **Parallelization** | Good | Excellent (vectorized) | Excellent |
| **Training Speed** | Fast (pre-computed) | Medium | Fast |
| **Inference Speed** | Fast | Medium | Fast |

---

## Fidel-TS Port Implementation Plan

This section provides a detailed implementation plan for porting TimeCMA to fidel-ts as a new model.

### Prerequisites

Before implementing TimeCMA in fidel-ts, the **Local LLM Infrastructure** must be in place. See [future/local_llm.md](./local_llm.md) for the complete plan covering:
- LLM model registry (singleton pattern like `embedder/registry.py`)
- Prompt template system
- LLM embedding caching
- Configuration integration

### Phase 1: Layer Implementation (Week 1)

#### 1.1 Cross-Modal Alignment Layer

**File**: `layers/CrossModal.py`

```python
"""
Cross-Modal Alignment Layer for TimeCMA.

Implements the core innovation: aligning time series embeddings with
LLM prompt embeddings to retrieve 'disentangled and robust' representations.
"""

import torch
import torch.nn as nn
from typing import Optional

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for aligning TS and LLM embeddings.
    
    Q: Time series encoded features (disentangled but weak)
    K, V: LLM prompt embeddings (entangled but robust)
    Output: Cross-modality aligned features
    """
    
    def __init__(self, 
                 d_model: int,
                 n_heads: int = 1,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 pre_norm: bool = True,
                 res_attention: bool = True):
        super().__init__()
        
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5
        
        # Q, K, V projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Normalization and FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm
        self.res_attention = res_attention
    
    def forward(self, 
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            q: Query (TS features) [B, C, N] or [B, seq_len, d_model]
            k: Key (LLM features) [B, E, N] or [B, llm_len, d_model]
            v: Value (LLM features) [B, E, N] or [B, llm_len, d_model]
            attn_mask: Optional attention mask
        
        Returns:
            Cross-modality aligned features, same shape as q
        """
        residual = q
        
        if self.pre_norm:
            q = self.norm1(q)
            k = self.norm1(k)
            v = self.norm1(v)
        
        # Multi-head attention
        B = q.size(0)
        q_proj = self.W_Q(q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_proj = self.W_K(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_proj = self.W_V(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v_proj)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        attn_output = self.out_proj(attn_output)
        
        # Residual connection
        if self.res_attention:
            output = residual + self.dropout(attn_output)
        else:
            output = self.dropout(attn_output)
        
        if not self.pre_norm:
            output = self.norm1(output)
        
        # FFN
        residual = output
        if self.pre_norm:
            output = self.norm2(output)
        
        output = residual + self.dropout(self.ffn(output))
        
        if not self.pre_norm:
            output = self.norm2(output)
        
        return output
```

#### 1.2 StandardNorm (RevIN) Layer

**File**: `layers/StandardNorm.py`

```python
"""
Standard Normalization (RevIN) for TimeCMA.

Provides reversible instance normalization for time series data.
"""

import torch
import torch.nn as nn


class StandardNorm(nn.Module):
    """
    Reversible Instance Normalization for time series.
    
    Same implementation as TimeCMA's Normalize class.
    """
    
    def __init__(self, 
                 num_features: int,
                 eps: float = 1e-5,
                 affine: bool = False,
                 subtract_last: bool = False):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        
        # Statistics (set during normalization)
        self.mean = None
        self.stdev = None
        self.last = None
    
    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Apply normalization or denormalization.
        
        Args:
            x: Input tensor [B, L, N]
            mode: 'norm' or 'denorm'
        
        Returns:
            Normalized or denormalized tensor
        """
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input and store statistics."""
        dim2reduce = tuple(range(1, x.ndim - 1))
        
        if self.subtract_last:
            self.last = x[:, -1:, :]
            x = x - self.last
        else:
            self.mean = x.mean(dim=dim2reduce, keepdim=True).detach()
            x = x - self.mean
        
        self.stdev = torch.sqrt(
            x.var(dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()
        x = x / self.stdev
        
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        
        return x
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize using stored statistics."""
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        
        x = x * self.stdev
        
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        
        return x
```

### Phase 2: Model Implementation (Week 2)

#### 2.1 TimeCMA Model

**File**: `models/TimeCMA.py`

```python
"""
TimeCMA Model for Fidel-TS.

Implements the dual-modality encoding with cross-modal alignment approach
from "TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting
via Cross-Modality Alignment" (AAAI 2025).
"""

import torch
import torch.nn as nn
from layers.CrossModal import CrossModalAttention
from layers.StandardNorm import StandardNorm


class Model(nn.Module):
    """
    TimeCMA: Cross-Modality Alignment for Time Series Forecasting.
    
    Architecture:
    1. Time Series Encoder (Transformer)
    2. Prompt Encoder (for LLM embeddings)
    3. Cross-Modal Alignment Layer
    4. Transformer Decoder
    5. Output Projection
    """
    
    def __init__(self, configs):
        super().__init__()
        
        # Configuration
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_nodes = configs.enc_in  # Number of channels/variables
        self.channel = configs.channel   # Hidden dimension
        self.d_llm = configs.d_llm       # LLM embedding dimension
        self.e_layer = configs.e_layers
        self.d_layer = configs.d_layers
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout
        self.revin = getattr(configs, 'revin', True)
        
        # 1. RevIN Normalization
        if self.revin:
            self.normalize_layers = StandardNorm(self.num_nodes, affine=False)
        
        # 2. Length-to-Feature Projection
        self.length_to_feature = nn.Linear(self.seq_len, self.channel)
        
        # 3. Time Series Encoder
        ts_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel,
            nhead=self.n_heads,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout
        )
        self.ts_encoder = nn.TransformerEncoder(ts_encoder_layer, num_layers=self.e_layer)
        
        # 4. Prompt Encoder (for LLM embeddings)
        prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm,
            nhead=self.n_heads,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout
        )
        self.prompt_encoder = nn.TransformerEncoder(prompt_encoder_layer, num_layers=self.e_layer)
        
        # 5. Cross-Modal Alignment
        self.cross_modal = CrossModalAttention(
            d_model=self.num_nodes,  # Attention over channels
            n_heads=1,
            d_ff=configs.d_ff,
            dropout=self.dropout,
            pre_norm=True,
            res_attention=True
        )
        
        # 6. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.channel,
            nhead=self.n_heads,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.d_layer)
        
        # 7. Output Projection
        self.output_projection = nn.Linear(self.channel, self.pred_len, bias=True)
    
    def forward(self, x, llm_embeddings, **kwargs):
        """
        Forward pass.
        
        Args:
            x: Input time series [B, L, N]
            llm_embeddings: Pre-computed LLM embeddings [B, E, N, 1] or [B, E, N]
                            E = LLM embedding dimension
        
        Returns:
            Predictions [B, pred_len, N]
        """
        # Handle embedding shape
        if llm_embeddings.dim() == 4:
            llm_embeddings = llm_embeddings.squeeze(-1)  # [B, E, N]
        llm_embeddings = llm_embeddings.permute(0, 2, 1)  # [B, N, E]
        
        # 1. RevIN Normalization
        if self.revin:
            x = self.normalize_layers(x, 'norm')
        
        # 2. Project: [B, L, N] → [B, N, L] → [B, N, C]
        x = x.permute(0, 2, 1)  # [B, N, L]
        x = self.length_to_feature(x)  # [B, N, C]
        
        # 3. Time Series Encoding
        enc_out = self.ts_encoder(x)  # [B, N, C]
        enc_out = enc_out.permute(0, 2, 1)  # [B, C, N]
        
        # 4. Prompt Encoding
        prompt_out = self.prompt_encoder(llm_embeddings)  # [B, N, E]
        prompt_out = prompt_out.permute(0, 2, 1)  # [B, E, N]
        
        # 5. Cross-Modal Alignment
        # Q: [B, C, N], KV: [B, E, N] → [B, C, N]
        cross_out = self.cross_modal(enc_out, prompt_out, prompt_out)
        cross_out = cross_out.permute(0, 2, 1)  # [B, N, C]
        
        # 6. Decoder (self-attention)
        dec_out = self.decoder(cross_out, cross_out)  # [B, N, C]
        
        # 7. Output Projection
        output = self.output_projection(dec_out)  # [B, N, pred_len]
        output = output.permute(0, 2, 1)  # [B, pred_len, N]
        
        # 8. Denormalize
        if self.revin:
            output = self.normalize_layers(output, 'denorm')
        
        return output
    
    def move_to_device(self, seq_x, seq_y, x_time, y_time, x_hetero, y_hetero,
                       hetero_x_time, hetero_y_time, hetero_general, hetero_channel, device):
        """
        Move data to device (compatibility with fidel-ts training loop).
        
        Note: hetero_channel contains LLM embeddings in TimeCMA context.
        """
        seq_x = seq_x.float().to(device)
        seq_y = seq_y.float().to(device)
        hetero_channel = hetero_channel.float().to(device)  # LLM embeddings
        
        return (seq_x, seq_y, x_time, y_time, x_hetero, y_hetero,
                hetero_x_time, hetero_y_time, hetero_general, hetero_channel)
```

### Phase 3: Configuration & Data Integration (Week 3)

#### 3.1 Model Configuration

**File**: `model_configs/general/TimeCMA.yaml`

```yaml
# TimeCMA Model Configuration
# Cross-Modality Alignment for Time Series Forecasting (AAAI 2025)

model: TimeCMA

# Model Architecture
channel: 32                # Hidden dimension (C in paper)
d_ff: 128                  # FFN hidden dimension
e_layers: 1                # Time series encoder layers
d_layers: 1                # Decoder layers
n_heads: 8                 # Attention heads
dropout: 0.2

# LLM Embedding Settings
d_llm: 768                 # LLM embedding dimension (GPT-2 base)

# Normalization
revin: true

# Task
task: TimeCMA

# LLM Embedding Configuration (references local_llm infrastructure)
llm_embedding:
  model_name: "gpt2"
  extraction_mode: "last_token"
  prompt_template: "timecma_v1"
  cache_embeddings: true
```

#### 3.2 Data Provider Integration

**File**: `data_provider/timecma_hetero_getter.py`

```python
"""
TimeCMA Heterogeneous Data Getter.

Loads LLM embeddings for TimeCMA model, either from cache or by
generating them using the local LLM infrastructure.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from embedder.llm_registry import LLMRegistry
from embedder.prompt_builder import PromptBuilder
from embedder.llm_cache import LLMEmbeddingCache, LLMEmbeddingMetadata


class TimeCMAHeteroGetter:
    """
    Handles LLM embedding loading/generation for TimeCMA.
    
    Integrates with fidel-ts heterogeneous data system.
    """
    
    def __init__(self,
                 dataset_name: str,
                 data_path: str,
                 llm_config: Dict[str, Any],
                 device: str = 'cpu'):
        
        self.dataset_name = dataset_name
        self.data_path = Path(data_path)
        self.llm_config = llm_config
        self.device = device
        
        # Extract config
        self.model_name = llm_config.get('model_name', 'gpt2')
        self.extraction_mode = llm_config.get('extraction_mode', 'last_token')
        self.prompt_template = llm_config.get('prompt_template', 'timecma_v1')
        self.cache_embeddings = llm_config.get('cache_embeddings', True)
        
        # Initialize components (lazy loading)
        self._llm_model = None
        self._tokenizer = None
        self._prompt_builder = None
        self._cache = None
    
    def get_llm_embeddings(self,
                           values: np.ndarray,
                           timestamps: np.ndarray,
                           split: str,
                           indices: np.ndarray) -> torch.Tensor:
        """
        Get LLM embeddings for given time series samples.
        
        Args:
            values: Time series values [N, seq_len, channels]
            timestamps: Timestamp features [N, seq_len, features]
            split: Data split ('train', 'val', 'test')
            indices: Sample indices
        
        Returns:
            LLM embeddings [N, d_llm, channels]
        """
        # Try loading from cache first
        if self.cache_embeddings:
            cached = self._try_load_cache(split, indices)
            if cached is not None:
                return cached
        
        # Generate embeddings
        embeddings = self._generate_embeddings(values, timestamps)
        
        # Cache if enabled
        if self.cache_embeddings:
            self._save_cache(embeddings, indices, split)
        
        return torch.from_numpy(embeddings).float()
    
    def _generate_embeddings(self,
                             values: np.ndarray,
                             timestamps: np.ndarray) -> np.ndarray:
        """Generate LLM embeddings for samples."""
        # Lazy init
        if self._llm_model is None:
            self._init_llm()
        
        N, L, C = values.shape
        d_llm = LLMRegistry.get_embedding_dim(self.model_name)
        embeddings = np.zeros((N, d_llm, C), dtype=np.float32)
        
        # Generate prompts
        prompts = self._prompt_builder.build_batch_prompts(
            values, timestamps,
            metadata={'freq': 'h'}  # TODO: Get from config
        )
        
        # Process each channel
        for c in range(C):
            channel_prompts = [p[c] for p in prompts]
            
            # Tokenize
            inputs = self._tokenizer(
                channel_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self._llm_model(**inputs)
                
                if self.extraction_mode == 'last_token':
                    # Get last token embedding
                    last_token_emb = outputs.last_hidden_state[:, -1, :]
                else:
                    # Pooled (mean)
                    last_token_emb = outputs.last_hidden_state.mean(dim=1)
                
                embeddings[:, :, c] = last_token_emb.cpu().numpy()
        
        return embeddings
    
    def _init_llm(self):
        """Initialize LLM model and related components."""
        cache_dir = self.llm_config.get('cache_dir', './LLM_cache/')
        
        self._llm_model = LLMRegistry.get_model(
            self.model_name, self.device, cache_dir, self.extraction_mode
        )
        self._tokenizer = LLMRegistry.get_tokenizer(self.model_name, cache_dir)
        
        self._prompt_builder = PromptBuilder(
            template_name=self.prompt_template,
            template_config=self.llm_config.get('prompt_config', {})
        )
        
        self._cache = LLMEmbeddingCache(str(self.data_path), self.dataset_name)
```

### Phase 4: Testing & Validation (Week 4)

#### 4.1 Unit Tests

**File**: `tests/test_timecma.py`

```python
"""Unit tests for TimeCMA model and components."""

import pytest
import torch
import numpy as np
from utils.tools import dotdict
from models.TimeCMA import Model as TimeCMA
from layers.CrossModal import CrossModalAttention
from layers.StandardNorm import StandardNorm


class TestStandardNorm:
    """Tests for StandardNorm (RevIN) layer."""
    
    def test_normalize_denormalize(self):
        """Test that denorm(norm(x)) ≈ x."""
        norm = StandardNorm(num_features=7, affine=False)
        x = torch.randn(4, 96, 7)  # [B, L, N]
        
        x_norm = norm(x, 'norm')
        x_recon = norm(x_norm, 'denorm')
        
        assert torch.allclose(x, x_recon, atol=1e-5)


class TestCrossModalAttention:
    """Tests for CrossModalAttention layer."""
    
    def test_forward_shape(self):
        """Test output shape matches query shape."""
        layer = CrossModalAttention(d_model=7, n_heads=1)
        
        q = torch.randn(4, 32, 7)   # [B, C, N]
        k = torch.randn(4, 768, 7)  # [B, E, N]
        v = torch.randn(4, 768, 7)
        
        output = layer(q, k, v)
        
        assert output.shape == q.shape


class TestTimeCMAModel:
    """Tests for TimeCMA model."""
    
    @pytest.fixture
    def configs(self):
        return dotdict({
            'seq_len': 96,
            'pred_len': 24,
            'enc_in': 7,
            'channel': 32,
            'd_llm': 768,
            'e_layers': 1,
            'd_layers': 1,
            'n_heads': 8,
            'd_ff': 128,
            'dropout': 0.1,
            'revin': True
        })
    
    def test_forward(self, configs):
        """Test forward pass produces correct output shape."""
        model = TimeCMA(configs)
        
        x = torch.randn(4, 96, 7)        # [B, L, N]
        emb = torch.randn(4, 768, 7)     # [B, E, N]
        
        output = model(x, emb)
        
        assert output.shape == (4, 24, 7)  # [B, pred_len, N]
```

### Implementation Checklist

| Phase | Task | Status | File |
|-------|------|--------|------|
| **Prereq** | Local LLM Infrastructure | ⬜ | [local_llm.md](./local_llm.md) |
| **1.1** | CrossModal layer | ⬜ | `layers/CrossModal.py` |
| **1.2** | StandardNorm layer | ⬜ | `layers/StandardNorm.py` |
| **2.1** | TimeCMA model | ⬜ | `models/TimeCMA.py` |
| **3.1** | Model config | ⬜ | `model_configs/general/TimeCMA.yaml` |
| **3.2** | Data integration | ⬜ | `data_provider/timecma_hetero_getter.py` |
| **4.1** | Unit tests | ⬜ | `tests/test_timecma.py` |
| **4.2** | Integration tests | ⬜ | `tests/test_timecma_integration.py` |

### Key Differences from Original TimeCMA

| Aspect | Original TimeCMA | Fidel-TS Port |
|--------|-----------------|---------------|
| **LLM Access** | Pre-computed H5 files | Dynamic via LLMRegistry |
| **Prompt System** | Hardcoded templates | Configurable PromptBuilder |
| **Data Loading** | Custom DataLoader | Fidel-TS HeteroGetter |
| **Config** | Argparse | YAML configs |
| **Training** | Custom loop | Fidel-TS training infrastructure |

---

## Related Documents

- [Local LLM Infrastructure](./local_llm.md) - LLM hosting and prompt system plan
- [Embedding Flow Analysis](../archive/embeddings_centralization/embedding_flow_analysis.md) - Existing embedding system
- [FiLM Architecture](./FiLM.md) - Related FiLM modulation approach

