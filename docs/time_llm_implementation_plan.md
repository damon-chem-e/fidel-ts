# Time-LLM Implementation Plan

> **Paper**: Time-LLM: Time Series Forecasting by Reprogramming Large Language Models (ICLR 2024)
> **Authors**: Jin et al.
> **Reference Implementation**: `benchmark_models/Time-LLM/`

---

## Executive Summary

Time-LLM is a **reprogramming framework** that converts time series into LLM-compatible token representations during the forward pass. This is fundamentally different from the precomputed embedding approach in the existing `fidel-ts` codebase.

**Key Insight**: Time-LLM and fidel-ts use LLMs differently:
- **fidel-ts (TimeCMA)**: Precompute embeddings offline via prompts using `LLMEmbedder`
- **Time-LLM**: Reprogram patches into LLM space during training with dynamic prompts

Both approaches can coexist without conflict, sharing the unified LLM loading infrastructure.

---

## 1. Architecture Overview

### 1.1 Time-LLM Forward Pass

```
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐    ┌──────────┐
│ Time Series │───>│ PatchEmbed   │───>│ ReprogrammingLayer│───>│ LLM      │
│ [B,T,C]     │    │ (Conv1D)     │    │ (Cross-Attention) │    │ (frozen) │
└─────────────┘    └──────────────┘    └──────────────────┘    └──────────┘
       │                                       │
       │                    Uses: Word Embeddings as Keys/Values
       │                                       │
       v                                       v
┌─────────────────────────────────────────────────────────────────────────┐
│ Dynamic Prompt: "<|start_prompt|>Dataset description: {desc}. Task:    │
│ forecast next {pred_len} steps. Stats: min {min}, max {max}, median   │
│ {med}, trend {trend}, lags {lags}<|end_prompt|>"                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Components

| Component | Source File | Description |
|-----------|-------------|-------------|
| **ReprogrammingLayer** | `TimeLLM.py:267-305` | Cross-attention mapping TS patches to LLM vocab space |
| **PatchEmbedding** | `Embed.py:160-186` | Conv1D-based patching (like ViT for images) |
| **FlattenHead** | `TimeLLM.py:15-27` | Output projection from LLM space to predictions |
| **Dynamic Prompts** | `TimeLLM.py:213-230` | Stats-based prompt generation (min/max/median/lags/trend) |
| **Normalize** | `StandardNorm.py:5-68` | RevIN-style instance normalization |

---

## 2. Overlap Analysis with Existing Embedder Infrastructure

### 2.1 Components to REUSE (Existing in `embedder/`)

| fidel-ts Component | Location | Use in Time-LLM |
|--------------------|----------|-----------------|
| `LLMRegistry` | `embedder/llm_registry.py` | **REUSE** - Singleton model loading with quantization |
| `load_model_for_embedding()` | `embedder/llm_utils.py` | **REUSE** - Model loading with quantization configs |
| `get_quantization_config()` | `embedder/llm_utils.py` | **REUSE** - BitsAndBytes 4-bit/8-bit configs |
| `MODEL_SPECS` | `embedder/llm_utils.py` | **REUSE** - Model specs (dim, VRAM requirements) |
| `Data_Provider` | `data_provider/data_factory.py` | **REUSE** - Standard data loading |

### 2.2 Components That Are DIFFERENT (Must Implement)

| Time-LLM Component | Difference from fidel-ts | Action |
|--------------------|--------------------------|--------|
| **ReprogrammingLayer** | Novel cross-attention mechanism | **NEW** |
| **PatchEmbedding** | Conv1D patching (not text prompts) | **NEW** |
| **Dynamic prompt generation** | On-the-fly stats in forward pass | **NEW** |
| **FlattenHead** | Specific output projection | **NEW** |
| **Normalize (RevIN)** | Instance normalization | **NEW** |

### 2.3 Components to KEEP SEPARATE (Different Use Cases)

| Existing Component | Reason |
|--------------------|--------|
| `TSPromptBuilder` (`prompt_builder.py`) | Different use case - precomputation vs online |
| `LLMEmbedder` (`llm_embedder.py`) | Different workflow - offline batch generation |
| `LLMEmbeddingCache` (`llm_cache.py`) | Time-LLM doesn't need embedding cache |
| `LLMEmbeddingProvider` (`llm_embedding_provider.py`) | Time-LLM computes embeddings online |

### 2.4 Key Differences: TimeCMA vs Time-LLM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TIMECMA (Current)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [Precomputation Phase - Offline]                                           │
│  ┌──────────┐    ┌───────────────┐    ┌────────────┐    ┌──────────────────┐│
│  │TimeSeries│───>│TSPromptBuilder│───>│ LLMEmbedder│───>│LLMEmbeddingCache ││
│  │ values   │    │ (template)    │    │(GPT/Qwen)  │    │ (H5 files)       ││
│  └──────────┘    └───────────────┘    └────────────┘    └──────────────────┘│
│                                                                              │
│  [Training Phase - Uses Precomputed]                                        │
│  ┌──────────────────┐    ┌───────────────┐    ┌──────────┐                  │
│  │LLMEmbeddingProvider│─>│ TimeCMA Model │───>│Prediction│                  │
│  │(loads from cache)│    │ (cross-modal) │    │          │                  │
│  └──────────────────┘    └───────────────┘    └──────────┘                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          TIME-LLM (New)                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [Training Phase - All Online]                                              │
│  ┌──────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌──────────┐│
│  │TimeSeries│───>│ PatchEmbedding  │───>│ReprogrammingLayer│───>│LLM (frozen)││
│  │ [B,T,C]  │    │ (Conv1D)        │    │(cross-attention) │    │          ││
│  └──────────┘    └─────────────────┘    └─────────────────┘    └──────────┘│
│        │                                         │                          │
│        v                                         v                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ DynamicPromptBuilder: Compute stats per batch → tokenize → embed       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Implementation Phases

### Phase 1: Core Model Components

#### 3.1 Create `models/time_llm/normalization.py`

```python
"""
RevIN-style Instance Normalization for time series.

Features:
- Normalize at forward pass, denormalize at output
- Optional affine transformation
- Handles per-sample statistics
"""

import torch
import torch.nn as nn


class Normalize(nn.Module):
    """Reversible Instance Normalization."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        """
        Initialize Normalize layer.
        
        Args:
            num_features: Number of channels/features to normalize
            eps: Small constant for numerical stability
            affine: If True, learn scale and shift parameters
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        
        # Statistics are stored per forward pass (not learned)
        self.mean = None
        self.std = None
    
    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Normalize or denormalize input tensor.
        
        Args:
            x: Input tensor [B, T, C]
            mode: 'norm' to normalize, 'denorm' to reverse
        
        Returns:
            Normalized or denormalized tensor [B, T, C]
        """
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'norm' or 'denorm'.")
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Compute and apply normalization."""
        # Compute per-sample statistics over time dimension
        self.mean = x.mean(dim=1, keepdim=True)
        self.std = x.std(dim=1, keepdim=True) + self.eps
        
        # Normalize
        x_norm = (x - self.mean) / self.std
        
        # Apply affine if enabled
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
        
        return x_norm
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse normalization using stored statistics."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Must call forward with mode='norm' before 'denorm'")
        
        # Remove affine if enabled
        if self.affine:
            x = (x - self.affine_bias) / self.affine_weight
        
        # Denormalize
        return x * self.std + self.mean
```

#### 3.2 Create `models/time_llm/patch_embed.py`

```python
"""
Patch Embedding for Time Series.

Key parameters:
- patch_len: Size of each patch (default: 16)
- stride: Stride for patching (default: 8, allows overlap)
- d_model: Embedding dimension for patches

Note: This is DIFFERENT from text tokenization.
Here we treat time series values directly, not as text.
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Convert patches to embeddings using 1D convolution."""
    
    def __init__(self, patch_len: int, d_model: int):
        """
        Initialize TokenEmbedding.
        
        Args:
            patch_len: Length of each patch
            d_model: Output embedding dimension
        """
        super().__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=patch_len,
            out_channels=d_model,
            kernel_size=1,
            padding=0,
            bias=False
        )
        nn.init.kaiming_normal_(self.tokenConv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches to embeddings.
        
        Args:
            x: [B*C, num_patches, patch_len]
        
        Returns:
            Embeddings [B*C, num_patches, d_model]
        """
        # Transpose for Conv1d: [B*C, patch_len, num_patches]
        x = x.permute(0, 2, 1)
        # Apply convolution
        x = self.tokenConv(x)
        # Transpose back: [B*C, num_patches, d_model]
        return x.transpose(1, 2)


class PatchEmbedding(nn.Module):
    """
    Patch embedding for time series.
    
    Converts [B, C, T] time series to [B*C, num_patches, d_model] embeddings.
    """
    
    def __init__(
        self, 
        d_model: int, 
        patch_len: int, 
        stride: int, 
        dropout: float = 0.1
    ):
        """
        Initialize PatchEmbedding.
        
        Args:
            d_model: Embedding dimension for patches
            patch_len: Length of each patch
            stride: Stride between patches (overlap if stride < patch_len)
            dropout: Dropout rate
        """
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        
        # Padding to ensure we can extract patches
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        
        # Convert patches to embeddings
        self.value_embedding = TokenEmbedding(patch_len, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Convert time series to patch embeddings.
        
        Args:
            x: Time series [B, C, T]
        
        Returns:
            Tuple of:
                - embeddings: [B*C, num_patches, d_model]
                - n_vars: Number of variables/channels (C)
        """
        n_vars = x.shape[1]
        
        # Pad sequence
        x = self.padding_patch_layer(x)
        
        # Extract patches: [B, C, T+stride] -> [B, C, num_patches, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # Flatten batch and channel dims: [B*C, num_patches, patch_len]
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        
        # Embed patches: [B*C, num_patches, d_model]
        x = self.value_embedding(x)
        
        return self.dropout(x), n_vars
```

#### 3.3 Create `models/time_llm/reprogramming.py`

```python
"""
Reprogramming Layer - Maps time series patches to LLM vocabulary space.

The KEY innovation of Time-LLM:
- Query: embedded time series patches [B, num_patches, d_model]
- Key/Value: LLM word embeddings [vocab_size, d_llm] → mapped to [num_tokens, d_llm]
- Output: Reprogrammed embeddings in LLM space [B, num_patches, d_llm]
"""

import math
import torch
import torch.nn as nn


class ReprogrammingLayer(nn.Module):
    """
    Cross-attention layer that maps time series patches to LLM vocabulary space.
    
    The reprogramming enables frozen LLMs to process time series by translating
    patch embeddings into the LLM's embedding space via cross-attention with
    word embeddings.
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_keys: int = None, 
        d_llm: int = None, 
        attention_dropout: float = 0.1
    ):
        """
        Initialize ReprogrammingLayer.
        
        Args:
            d_model: Dimension of time series patch embeddings
            n_heads: Number of attention heads
            d_keys: Dimension of keys (defaults to d_model // n_heads)
            d_llm: Dimension of LLM embeddings (hidden size)
            attention_dropout: Dropout rate for attention
        """
        super().__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        
        # Project time series patches to query space
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        
        # Project LLM word embeddings to key/value space
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        
        # Output projection back to LLM space
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self, 
        target_embedding: torch.Tensor, 
        source_embedding: torch.Tensor, 
        value_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Reprogram time series patches using LLM word embeddings.
        
        Args:
            target_embedding: Time series patch embeddings [B, L, d_model]
            source_embedding: LLM word embeddings (mapped) [S, d_llm]
            value_embedding: LLM word embeddings (mapped) [S, d_llm]
        
        Returns:
            Reprogrammed embeddings in LLM space [B, L, d_llm]
        """
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        # Project to multi-head attention space
        target = self.query_projection(target_embedding).view(B, L, H, -1)
        source = self.key_projection(source_embedding).view(S, H, -1)
        value = self.value_projection(value_embedding).view(S, H, -1)

        # Compute reprogramming via cross-attention
        out = self._reprogramming(target, source, value)
        
        # Reshape and project output
        out = out.reshape(B, L, -1)
        return self.out_projection(out)

    def _reprogramming(
        self, 
        target: torch.Tensor, 
        source: torch.Tensor, 
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Core reprogramming via scaled dot-product cross-attention.
        
        Args:
            target: Query [B, L, H, E]
            source: Key [S, H, E]
            value: Value [S, H, E]
        
        Returns:
            Attention output [B, L, H, E]
        """
        B, L, H, E = target.shape
        scale = 1. / math.sqrt(E)
        
        # Compute attention scores: Q @ K^T
        # target: [B, L, H, E], source: [S, H, E]
        # Result: [B, H, L, S]
        scores = torch.einsum("blhe,she->bhls", target, source)
        
        # Apply softmax with scaling
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        
        # Apply attention to values
        # A: [B, H, L, S], value: [S, H, E]
        # Result: [B, L, H, E]
        return torch.einsum("bhls,she->blhe", A, value)
```

#### 3.4 Create `models/time_llm/dynamic_prompt.py`

```python
"""
Dynamic Prompt Generation for Time-LLM.

IMPORTANT: This is DIFFERENT from embedder/prompt_builder.py!

- TSPromptBuilder (existing): Precomputes prompts for offline embedding
- DynamicPromptBuilder (this): Generates prompts ON-THE-FLY in forward pass

Key differences:
1. Computes statistics (min/max/median/lags/trend) per forward pass
2. Includes prediction task description
3. Used during training, not precomputation
4. Returns tokenized embeddings, not just text strings
"""

from typing import List, Optional

import torch
import numpy as np


class DynamicPromptBuilder:
    """
    Generate prompts with time series statistics during inference.
    
    Unlike TSPromptBuilder which is for offline precomputation,
    this class generates prompts on-the-fly during the forward pass
    with dynamic per-batch statistics.
    """
    
    @staticmethod
    def build_prompts(
        x_enc: torch.Tensor,
        description: str,
        pred_len: int,
        seq_len: int,
        lags: torch.Tensor,
    ) -> List[str]:
        """
        Build prompts with dynamic statistics.
        
        Args:
            x_enc: [B*N, T, 1] - flattened time series (B samples, N channels)
            description: Dataset description string
            pred_len: Prediction length
            seq_len: Input sequence length
            lags: [B*N, top_k] - top-k autocorrelation lags
        
        Returns:
            List of prompt strings, one per sample-channel pair
        """
        # Compute statistics along time dimension
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        
        # Compute trend (sum of differences)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompts = []
        for b in range(x_enc.shape[0]):
            # Determine trend direction
            trend_dir = 'upward' if trends[b].item() > 0 else 'downward'
            
            # Format lag values
            lag_values = lags[b].tolist()
            
            # Build prompt with statistics
            prompt = (
                f"<|start_prompt|>Dataset description: {description} "
                f"Task description: forecast the next {pred_len} steps "
                f"given the previous {seq_len} steps information; "
                "Input statistics: "
                f"min value {min_values[b].item():.4f}, "
                f"max value {max_values[b].item():.4f}, "
                f"median value {medians[b].item():.4f}, "
                f"the trend of input is {trend_dir}, "
                f"top 5 lags are: {lag_values}<|end_prompt|>"
            )
            prompts.append(prompt)
        
        return prompts
    
    @staticmethod
    def calculate_lags(x_enc: torch.Tensor, top_k: int = 5) -> torch.Tensor:
        """
        Compute top-k autocorrelation lags via FFT.
        
        Uses FFT-based autocorrelation for efficiency.
        
        Args:
            x_enc: Time series [B*N, T, 1]
            top_k: Number of top lags to return
        
        Returns:
            Top-k lag indices [B*N, top_k]
        """
        # Transpose for FFT: [B*N, 1, T]
        x = x_enc.permute(0, 2, 1)
        
        # FFT-based autocorrelation
        q_fft = torch.fft.rfft(x, dim=-1)
        k_fft = torch.fft.rfft(x, dim=-1)
        
        # Power spectrum (autocorrelation in frequency domain)
        res = q_fft * torch.conj(k_fft)
        
        # Inverse FFT to get autocorrelation
        corr = torch.fft.irfft(res, dim=-1)
        
        # Average over channel dimension (if multi-channel)
        mean_value = torch.mean(corr, dim=1)
        
        # Get top-k lags (exclude lag 0 which is always highest)
        _, lags = torch.topk(mean_value[:, 1:], top_k, dim=-1)
        
        # Adjust indices since we excluded lag 0
        return lags + 1
```

### Phase 2: Main Model Class

#### 3.5 Create `models/time_llm/time_llm_model.py`

```python
"""
Main Time-LLM Model - Reprogramming LLMs for Time Series Forecasting.

DESIGN DECISIONS:
1. LLM backbone is FROZEN (requires_grad=False)
2. Only train: PatchEmbedding, ReprogrammingLayer, OutputProjection
3. Dynamic prompts with statistics generated in forward pass
4. Uses existing LLMRegistry for model loading (no duplication)

INTEGRATION WITH FIDEL-TS:
- Uses LLMRegistry from embedder/ for model loading
- Uses Data_Provider for data loading
- Follows exp/exp_universal.py patterns for training
"""

from typing import Optional

import torch
import torch.nn as nn

# REUSE existing LLM infrastructure
from embedder.llm_registry import LLMRegistry

# Import local components
from .normalization import Normalize
from .patch_embed import PatchEmbedding
from .reprogramming import ReprogrammingLayer
from .dynamic_prompt import DynamicPromptBuilder


class FlattenHead(nn.Module):
    """Output projection from LLM space to predictions."""
    
    def __init__(
        self, 
        n_vars: int, 
        head_nf: int, 
        pred_len: int, 
        dropout: float = 0.1
    ):
        """
        Initialize FlattenHead.
        
        Args:
            n_vars: Number of input variables/channels
            head_nf: Number of features in flattened representation
            pred_len: Prediction length
            dropout: Dropout rate
        """
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(head_nf, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project from LLM space to predictions.
        
        Args:
            x: [B, n_vars, d_ff, num_patches]
        
        Returns:
            Predictions [B, pred_len, n_vars]
        """
        x = self.flatten(x)  # [B, n_vars, d_ff * num_patches]
        x = self.linear(x)   # [B, n_vars, pred_len]
        x = self.dropout(x)
        return x.permute(0, 2, 1)  # [B, pred_len, n_vars]


class TimeLLM(nn.Module):
    """
    Time-LLM: Time Series Forecasting by Reprogramming Large Language Models.
    
    This model reprograms a frozen LLM to process time series by:
    1. Patching time series into embeddings
    2. Cross-attending patches with LLM word embeddings
    3. Passing reprogrammed embeddings through frozen LLM
    4. Projecting LLM output to predictions
    
    Key Design:
    - LLM is FROZEN (no gradients)
    - Only patch embedding, reprogramming, and output projection are trained
    - Dynamic prompts with per-batch statistics
    """
    
    def __init__(
        self,
        llm_model: str = 'gpt2',
        llm_layers: int = 6,
        seq_len: int = 96,
        pred_len: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 16,
        d_ff: int = 32,
        n_heads: int = 8,
        enc_in: int = 7,
        dropout: float = 0.1,
        prompt_domain: bool = False,
        dataset_description: str = "",
        cache_dir: str = './LLM_cache/',
        device: str = 'cuda:0',
        quantization: Optional[str] = None,
    ):
        """
        Initialize Time-LLM model.
        
        Args:
            llm_model: HuggingFace model name or Time-LLM alias ('LLAMA', 'GPT2', 'BERT')
            llm_layers: Number of LLM layers to use (not currently used, full model loaded)
            seq_len: Input sequence length
            pred_len: Prediction sequence length
            patch_len: Length of each time series patch
            stride: Stride between patches
            d_model: Patch embedding dimension
            d_ff: Feed-forward dimension (also used for output projection)
            n_heads: Number of attention heads for reprogramming
            enc_in: Number of input channels/variables
            dropout: Dropout rate
            prompt_domain: If True, use provided dataset_description
            dataset_description: Domain-specific dataset description
            cache_dir: Directory for LLM model weights cache
            device: Target device
            quantization: Quantization mode ('4bit', '8bit', or None)
        """
        super().__init__()
        
        # Store configuration
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_ff = d_ff
        self.patch_len = patch_len
        self.stride = stride
        self.top_k = 5
        
        # Load LLM using existing registry (REUSE)
        self._load_llm(llm_model, cache_dir, device, quantization)
        
        # Get LLM hidden dimension
        self.d_llm = self.llm_model.config.hidden_size
        
        # Freeze LLM weights
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
        # Dataset description for prompts
        self.description = dataset_description if prompt_domain else (
            "The Electricity Transformer Temperature (ETT) is a crucial "
            "indicator in the electric power long-term deployment."
        )
        
        # Patch embedding layer
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, dropout)
        
        # Word embedding mapping (reduce vocabulary for efficiency)
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000  # Reduced vocabulary size
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        
        # Reprogramming layer
        self.reprogramming_layer = ReprogrammingLayer(
            d_model=d_model,
            n_heads=n_heads,
            d_keys=d_ff,
            d_llm=self.d_llm,
            attention_dropout=dropout
        )
        
        # Output projection
        self.patch_nums = int((seq_len - patch_len) / stride + 2)
        self.head_nf = d_ff * self.patch_nums
        self.output_projection = FlattenHead(enc_in, self.head_nf, pred_len, dropout)
        
        # Normalization layers
        self.normalize_layers = Normalize(enc_in, affine=False)
        
        self.dropout = nn.Dropout(dropout)

    def _load_llm(
        self, 
        llm_model: str, 
        cache_dir: str, 
        device: str,
        quantization: Optional[str]
    ):
        """
        Load LLM model using existing LLMRegistry.
        
        Maps Time-LLM aliases to HuggingFace model names.
        
        Args:
            llm_model: Model name or alias
            cache_dir: Cache directory for model weights
            device: Target device
            quantization: Quantization mode
        """
        # Map Time-LLM aliases to HuggingFace model names
        model_map = {
            'LLAMA': 'meta-llama/Llama-3.1-8B-Instruct',
            'LLAMA-7B': 'meta-llama/Llama-3.1-8B-Instruct',  # Updated to 3.1
            'GPT2': 'gpt2',
            'BERT': 'google-bert/bert-base-uncased',
            'QWEN': 'Qwen/Qwen2.5-7B-Instruct',
            'QWEN-72B': 'Qwen/Qwen2.5-72B-Instruct',
        }
        model_name = model_map.get(llm_model.upper(), llm_model)
        
        # Use existing LLMRegistry
        self.llm_model, _ = LLMRegistry.get_model(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            quantization=quantization,
        )
        
        # Get tokenizer
        self.tokenizer = LLMRegistry.get_tokenizer(
            model_name=model_name,
            cache_dir=cache_dir,
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def forward(
        self, 
        x_enc: torch.Tensor, 
        x_mark_enc: torch.Tensor = None, 
        x_dec: torch.Tensor = None, 
        x_mark_dec: torch.Tensor = None, 
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_enc: Input time series [B, T, C]
            x_mark_enc: Time features for encoder (optional)
            x_dec: Decoder input (not used)
            x_mark_dec: Time features for decoder (not used)
            mask: Attention mask (optional)
        
        Returns:
            Predictions [B, pred_len, C]
        """
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]

    def forecast(
        self, 
        x_enc: torch.Tensor, 
        x_mark_enc: torch.Tensor, 
        x_dec: torch.Tensor, 
        x_mark_dec: torch.Tensor
    ) -> torch.Tensor:
        """
        Forecast future values.
        
        Args:
            x_enc: Input time series [B, T, C]
            x_mark_enc: Time features (not used in reprogramming)
            x_dec: Decoder input (not used)
            x_mark_dec: Decoder time features (not used)
        
        Returns:
            Forecasted values [B, pred_len, C]
        """
        # Step 1: Normalize input
        x_enc = self.normalize_layers(x_enc, 'norm')
        
        B, T, N = x_enc.size()
        
        # Step 2: Reshape for per-channel processing
        # [B, T, N] -> [B*N, T, 1]
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        
        # Step 3: Calculate statistics for dynamic prompts
        lags = DynamicPromptBuilder.calculate_lags(x_enc, self.top_k)
        prompts = DynamicPromptBuilder.build_prompts(
            x_enc, self.description, self.pred_len, self.seq_len, lags
        )
        
        # Step 4: Reshape back for patching
        # [B*N, T, 1] -> [B, N, T] -> [B, T, N]
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        
        # Step 5: Tokenize prompts and get embeddings
        prompt_tokens = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048
        ).input_ids.to(x_enc.device)
        
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_tokens)
        
        # Step 6: Map word embeddings to reduced vocabulary
        source_embeddings = self.mapping_layer(
            self.word_embeddings.permute(1, 0)
        ).permute(1, 0)
        
        # Step 7: Patch embedding
        # [B, T, N] -> [B, N, T]
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        
        # Step 8: Reprogramming
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        
        # Step 9: Concatenate with prompt embeddings and pass through LLM
        llm_input = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llm_input).last_hidden_state
        
        # Step 10: Extract relevant dimensions
        dec_out = dec_out[:, :, :self.d_ff]
        
        # Step 11: Reshape for output projection
        # [B*N, seq_len+patches, d_ff] -> [B, N, d_ff, patches]
        dec_out = dec_out.reshape(-1, n_vars, dec_out.shape[-2], dec_out.shape[-1])
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        
        # Step 12: Output projection
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        
        # Step 13: Denormalize
        dec_out = self.normalize_layers(dec_out, 'denorm')
        
        return dec_out
```

### Phase 3: Training Infrastructure

#### 3.6 Create `exp/exp_time_llm.py`

```python
"""
Experiment class for Time-LLM training.

Follows the pattern of exp/exp_universal.py but specialized for Time-LLM:
- Only trains non-frozen parameters
- Uses Accelerate/DeepSpeed for multi-GPU training
- Reuses existing EarlyStopping and LR scheduling
- Integrates with Data_Provider for data loading
"""

import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn

from exp.exp_basic import Exp_Basic
from models import model_init
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from data_provider.data_factory import Data_Provider

warnings.filterwarnings('ignore')


class Experiment(Exp_Basic):
    """
    Experiment orchestrator for Time-LLM model.
    
    Key differences from exp_universal.py:
    - Only optimizes non-frozen parameters (LLM is frozen)
    - May use gradient checkpointing for memory efficiency
    - Supports 4-bit/8-bit quantization of frozen LLM
    """
    
    def __init__(self, args, exp_manager=None):
        """
        Initialize Time-LLM experiment.
        
        Args:
            args: Configuration object with Time-LLM specific settings
            exp_manager: Optional ExperimentManager for tracking
        """
        self.args = args
        self.exp_manager = exp_manager
        self.model = self._build_model()
        self.data_provider = Data_Provider(args, buffer=(not args.disable_buffer))
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        print(f"[ Time-LLM ] Trainable params: {trainable_params:,}")
        print(f"[ Time-LLM ] Frozen params: {frozen_params:,}")

    def _build_model(self):
        """Build Time-LLM model with frozen LLM backbone."""
        # model_init handles device placement
        model = model_init(self.args.model, self.args.model_config, self.args)
        return model

    def _select_optimizer(self):
        """
        Select optimizer for trainable parameters only.
        
        Only optimizes parameters with requires_grad=True,
        which excludes the frozen LLM backbone.
        """
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.args.learning_rate
        )
        
        return optimizer

    def _select_criterion(self):
        """Select loss criterion (MSE for forecasting)."""
        return nn.MSELoss()

    def train(self, setting):
        """
        Train Time-LLM model.
        
        Standard training loop with:
        - Only trainable parameter optimization
        - Early stopping
        - LR scheduling
        - Checkpoint saving
        """
        train_loader = self.data_provider.get_train(return_type='loader')
        val_loader = self.data_provider.get_val(return_type='loader')
        
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        
        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            
            epoch_time = time.time()
            
            for i, batch_data in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Unpack batch (format depends on Data_Provider)
                batch_x, batch_y, batch_x_mark, batch_y_mark = self._process_batch(batch_data)
                
                # Forward pass
                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                
                # Compute loss on prediction horizon
                loss = criterion(
                    outputs[:, -self.args.pred_len:, :],
                    batch_y[:, -self.args.pred_len:, :]
                )
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss.append(loss.item())
            
            # Validation
            val_loss = self.validate(val_loader, criterion)
            
            print(f"Epoch: {epoch+1} | Time: {time.time()-epoch_time:.1f}s | "
                  f"Train Loss: {np.mean(train_loss):.4f} | Val Loss: {val_loss:.4f}")
            
            # Early stopping check
            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
            
            # LR scheduling
            adjust_learning_rate(optimizer, epoch + 1, self.args)
        
        # Load best model
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def validate(self, loader, criterion):
        """Run validation and return loss."""
        self.model.eval()
        total_loss = []
        
        with torch.no_grad():
            for batch_data in loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = self._process_batch(batch_data)
                
                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                
                loss = criterion(
                    outputs[:, -self.args.pred_len:, :],
                    batch_y[:, -self.args.pred_len:, :]
                )
                
                total_loss.append(loss.item())
        
        return np.mean(total_loss)

    def _process_batch(self, batch_data):
        """
        Process batch data from Data_Provider.
        
        Handles the batch format from Data_Provider:
        (sample_ids, seq_x, seq_y, x_time, y_time, ...)
        
        Returns:
            Tuple of (batch_x, batch_y, batch_x_mark, batch_y_mark)
        """
        # Unpack based on batch structure
        # Format: (sample_ids, seq_x, seq_y, x_time, y_time, x_hetero, ...)
        batch_x = batch_data[1].float().to(self.args.device)
        batch_y = batch_data[2].float().to(self.args.device)
        batch_x_mark = batch_data[3].float().to(self.args.device) if len(batch_data) > 3 else None
        batch_y_mark = batch_data[4].float().to(self.args.device) if len(batch_data) > 4 else None
        
        return batch_x, batch_y, batch_x_mark, batch_y_mark

    def test(self, setting):
        """Run test evaluation."""
        test_loader = self.data_provider.get_test(return_type='loader')
        
        self.model.eval()
        preds = []
        trues = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = self._process_batch(batch_data)
                
                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                
                pred = outputs[:, -self.args.pred_len:, :].cpu().numpy()
                true = batch_y[:, -self.args.pred_len:, :].cpu().numpy()
                
                preds.append(pred)
                trues.append(true)
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # Compute metrics
        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)
        
        print(f"Test Results: MSE={mse:.4f}, MAE={mae:.4f}")
        
        return mse, mae
```

---

## 4. File Structure

```
fidel-ts-worktree-lynx/
├── models/
│   └── time_llm/
│       ├── __init__.py                 # Module exports
│       ├── time_llm_model.py           # Main model class
│       ├── reprogramming.py            # ReprogrammingLayer
│       ├── patch_embed.py              # PatchEmbedding, TokenEmbedding
│       ├── normalization.py            # Normalize (RevIN)
│       └── dynamic_prompt.py           # DynamicPromptBuilder
├── exp/
│   └── exp_time_llm.py                 # Training loop (new)
├── model_configs/
│   └── time_llm/
│       ├── default.yaml                # Default configuration
│       ├── gpt2.yaml                   # GPT-2 specific
│       └── qwen_7b.yaml                # Qwen 2.5-7B specific
└── configs/
    └── experiment_suites/
        └── time_llm_test.yaml          # Test experiment suite
```

---

## 5. Configuration Schema

```yaml
# model_configs/time_llm/default.yaml
model:
  name: "TimeLLM"
  task_name: "long_term_forecast"
  
llm:
  backbone: "gpt2"                     # or "Qwen/Qwen2.5-7B-Instruct", etc.
  quantization: null                   # "4bit", "8bit", or null for fp16
  cache_dir: "./LLM_cache/"
  freeze: true                         # Always freeze LLM weights

patching:
  patch_len: 16
  stride: 8

model_dims:
  d_model: 32                          # Patch embedding dim
  d_ff: 128                            # Feed-forward dim
  n_heads: 8                           # Attention heads for reprogramming
  dropout: 0.1

forecasting:
  seq_len: 512
  label_len: 48
  pred_len: 96

training:
  epochs: 100
  batch_size: 24
  learning_rate: 0.01
  patience: 10

prompt:
  domain_specific: true
  # Domain descriptions loaded from dataset/prompt_bank/{dataset}.txt
```

---

## 6. Implementation Priority & Dependencies

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Core Components (Independent - No Dependencies)                │
├─────────────────────────────────────────────────────────────────────────┤
│ 1a. normalization.py (Normalize class)                                  │
│ 1b. patch_embed.py (PatchEmbedding, TokenEmbedding)                     │
│ 1c. reprogramming.py (ReprogrammingLayer)                               │
│ 1d. dynamic_prompt.py (DynamicPromptBuilder)                            │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Main Model (Depends on Phase 1 + LLMRegistry)                  │
├─────────────────────────────────────────────────────────────────────────┤
│ 2a. time_llm_model.py (TimeLLM class)                                   │
│     - Uses: All Phase 1 components                                      │
│     - Uses: embedder/llm_registry.py (LLMRegistry) - EXISTING           │
│     - Uses: embedder/llm_utils.py - EXISTING                            │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Training & Integration (Depends on Phase 2)                    │
├─────────────────────────────────────────────────────────────────────────┤
│ 3a. exp/exp_time_llm.py (Experiment class)                              │
│     - Uses: data_provider/data_factory.py (Data_Provider) - EXISTING    │
│     - Uses: exp/exp_basic.py (Exp_Basic) - EXISTING                     │
│ 3b. model_configs/time_llm/*.yaml                                       │
│ 3c. Register model in models/__init__.py                                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

```python
# tests/test_time_llm_components.py

import pytest
import torch
from models.time_llm.normalization import Normalize
from models.time_llm.patch_embed import PatchEmbedding
from models.time_llm.reprogramming import ReprogrammingLayer
from models.time_llm.dynamic_prompt import DynamicPromptBuilder


def test_normalize():
    """Test RevIN normalization and denormalization."""
    norm = Normalize(num_features=7)
    x = torch.randn(2, 96, 7)
    x_norm = norm(x, 'norm')
    x_denorm = norm(x_norm, 'denorm')
    assert torch.allclose(x, x_denorm, atol=1e-5)


def test_patch_embedding():
    """Test patch embedding output shapes."""
    patch_embed = PatchEmbedding(d_model=32, patch_len=16, stride=8, dropout=0.1)
    x = torch.randn(2, 7, 96)  # [B, C, T]
    out, n_vars = patch_embed(x)
    assert n_vars == 7
    # Expected patches: (96 - 16) / 8 + 2 = 12
    assert out.shape == (2 * 7, 12, 32)


def test_reprogramming_layer():
    """Test reprogramming layer dimensions."""
    reprogram = ReprogrammingLayer(d_model=32, n_heads=8, d_keys=32, d_llm=768)
    target = torch.randn(2, 12, 32)  # [B, num_patches, d_model]
    source = torch.randn(1000, 768)  # [num_tokens, d_llm]
    out = reprogram(target, source, source)
    assert out.shape == (2, 12, 768)


def test_dynamic_prompt():
    """Test dynamic prompt generation."""
    x = torch.randn(4, 96, 1)
    lags = DynamicPromptBuilder.calculate_lags(x, top_k=5)
    assert lags.shape == (4, 5)
    
    prompts = DynamicPromptBuilder.build_prompts(
        x, "Test dataset", pred_len=96, seq_len=96, lags=lags
    )
    assert len(prompts) == 4
    assert "<|start_prompt|>" in prompts[0]
```

### 7.2 Integration Test

```python
# tests/test_time_llm_model.py

import pytest
import torch

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_time_llm_forward():
    """Test end-to-end forward pass with GPT-2 (smaller model)."""
    from models.time_llm.time_llm_model import TimeLLM
    
    model = TimeLLM(
        llm_model='gpt2',
        seq_len=96,
        pred_len=96,
        enc_in=7,
        device='cuda:0',
    )
    
    x = torch.randn(2, 96, 7).to('cuda:0')
    out = model(x, None, None, None)
    
    assert out.shape == (2, 96, 7)


def test_time_llm_trainable_params():
    """Verify only non-LLM params are trainable."""
    from models.time_llm.time_llm_model import TimeLLM
    
    model = TimeLLM(llm_model='gpt2', device='cpu')
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    # LLM should be frozen (GPT-2 has ~124M params)
    assert frozen > 100_000_000
    # Trainable should be small (patches, reprogramming, output)
    assert trainable < 10_000_000
```

### 7.3 Benchmark Comparison

```bash
# Compare with original Time-LLM on ETTh1
python scripts/run_time_llm.py \
    --dataset ETTh1 \
    --seq_len 512 \
    --pred_len 96 \
    --llm_model GPT2 \
    --batch_size 24 \
    --train_epochs 100
```

---

## 8. LLM Dimension Reference

| LLM Model | Hidden Dimension | HuggingFace Name | Min VRAM (4-bit) |
|-----------|------------------|------------------|------------------|
| GPT-2 | 768 | `gpt2` | 0.3 GB |
| Qwen2.5-7B | 3584 | `Qwen/Qwen2.5-7B-Instruct` | 5.0 GB |
| Qwen2.5-14B | 5120 | `Qwen/Qwen2.5-14B-Instruct` | 9.0 GB |
| Qwen2.5-72B | 8192 | `Qwen/Qwen2.5-72B-Instruct` | 42.0 GB |
| LLaMA-3.1-8B | 4096 | `meta-llama/Llama-3.1-8B-Instruct` | 6.0 GB |
| LLaMA-3.1-70B | 8192 | `meta-llama/Llama-3.1-70B-Instruct` | 40.0 GB |

*Note: These specs are from `embedder/llm_utils.py:MODEL_SPECS`*

---

## 9. Key Integration Points

### 9.1 Using Existing LLMRegistry

```python
# In time_llm_model.py - REUSE existing infrastructure
from embedder.llm_registry import LLMRegistry

# Load model with quantization support
self.llm_model, embed_dim = LLMRegistry.get_model(
    model_name='Qwen/Qwen2.5-72B-Instruct',
    device='cuda:0',
    cache_dir='./LLM_cache/',
    quantization='4bit'  # Uses existing BitsAndBytes config
)

# Get tokenizer
self.tokenizer = LLMRegistry.get_tokenizer(
    model_name='Qwen/Qwen2.5-72B-Instruct',
    cache_dir='./LLM_cache/'
)
```

### 9.2 Using Existing Data_Provider

```python
# In exp_time_llm.py - REUSE existing data infrastructure
from data_provider.data_factory import Data_Provider

# Standard data loading
data_provider = Data_Provider(args, buffer=True)
train_loader = data_provider.get_train(return_type='loader')

# Batch format from Data_Provider:
# (sample_ids, seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, ...)
```

### 9.3 NOT Using (Different Paradigm)

```python
# These are for PRECOMPUTED embeddings (TimeCMA), not Time-LLM:
# - LLMEmbedder: Generates embeddings offline
# - LLMEmbeddingCache: Stores precomputed embeddings
# - LLMEmbeddingProvider: Loads embeddings during training
# - TSPromptBuilder: Converts time series to prompts for offline embedding

# Time-LLM does NOT need these because:
# 1. Dynamic prompts are generated per-batch in forward pass
# 2. Reprogramming happens online, not via precomputed embeddings
# 3. LLM processes patches directly, not prompt embeddings
```

---

## 10. Open Questions / Future Work

1. **Memory Optimization**: Can we use gradient checkpointing for the frozen LLM layers to reduce memory during backprop?

2. **Quantization Impact**: Does 4-bit/8-bit quantization of frozen LLM affect reprogramming quality? Need empirical testing.

3. **Multi-modal Extension**: Can we combine Time-LLM's online reprogramming with fidel-ts's precomputed text embeddings for richer context?

4. **Channel Independence**: Should we share reprogramming across channels or keep separate? Original paper uses per-channel processing.

5. **Prompt Caching**: Could dynamic prompts be cached during training for faster iteration, or do per-batch statistics make this impractical?

---

## 11. References

- **Paper**: [Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://arxiv.org/abs/2310.01728)
- **Original Code**: `benchmark_models/Time-LLM/`
- **Related Work**: TimeCMA (uses precomputed embeddings instead of reprogramming)
- **Existing Infrastructure**:
  - `embedder/llm_registry.py`: LLM model loading with quantization
  - `embedder/llm_utils.py`: Model specs and utilities
  - `data_provider/data_factory.py`: Data loading infrastructure
  - `exp/exp_basic.py`: Base experiment class
