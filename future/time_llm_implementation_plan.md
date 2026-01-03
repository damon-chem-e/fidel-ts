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
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        
        self.mean = None
        self.std = None
    
    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'norm' or 'denorm'.")
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = x.mean(dim=1, keepdim=True)
        self.std = x.std(dim=1, keepdim=True) + self.eps
        x_norm = (x - self.mean) / self.std
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
        return x_norm
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Must call forward with mode='norm' before 'denorm'")
        if self.affine:
            x = (x - self.affine_bias) / self.affine_weight
        return x * self.std + self.mean
```

#### 3.2 Create `models/time_llm/patch_embed.py`

```python
"""
Patch Embedding for Time Series.
Converts [B, C, T] time series to [B*C, num_patches, d_model] embeddings.
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, patch_len: int, d_model: int):
        super().__init__()
        self.tokenConv = nn.Conv1d(patch_len, d_model, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.tokenConv(x)
        return x.transpose(1, 2)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model: int, patch_len: int, stride: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.value_embedding = TokenEmbedding(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple:
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        x = self.value_embedding(x)
        return self.dropout(x), n_vars
```

#### 3.3 Create `models/time_llm/reprogramming.py`

```python
"""
Reprogramming Layer - Maps time series patches to LLM vocabulary space.
"""

import math
import torch
import torch.nn as nn


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_keys: int = None, 
                 d_llm: int = None, attention_dropout: float = 0.1):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads
        target = self.query_projection(target_embedding).view(B, L, H, -1)
        source = self.key_projection(source_embedding).view(S, H, -1)
        value = self.value_projection(value_embedding).view(S, H, -1)
        out = self._reprogramming(target, source, value)
        out = out.reshape(B, L, -1)
        return self.out_projection(out)

    def _reprogramming(self, target, source, value):
        B, L, H, E = target.shape
        scale = 1. / math.sqrt(E)
        scores = torch.einsum("blhe,she->bhls", target, source)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum("bhls,she->blhe", A, value)
```

#### 3.4 Create `models/time_llm/dynamic_prompt.py`

```python
"""
Dynamic Prompt Generation for Time-LLM.
Generates prompts ON-THE-FLY in forward pass with per-batch statistics.
"""

from typing import List
import torch


class DynamicPromptBuilder:
    @staticmethod
    def build_prompts(x_enc: torch.Tensor, description: str, 
                      pred_len: int, seq_len: int, lags: torch.Tensor) -> List[str]:
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompts = []
        for b in range(x_enc.shape[0]):
            trend_dir = 'upward' if trends[b].item() > 0 else 'downward'
            prompt = (
                f"<|start_prompt|>Dataset description: {description} "
                f"Task description: forecast the next {pred_len} steps "
                f"given the previous {seq_len} steps information; "
                f"Input statistics: min value {min_values[b].item():.4f}, "
                f"max value {max_values[b].item():.4f}, "
                f"median value {medians[b].item():.4f}, "
                f"the trend of input is {trend_dir}, "
                f"top 5 lags are: {lags[b].tolist()}<|end_prompt|>"
            )
            prompts.append(prompt)
        return prompts
    
    @staticmethod
    def calculate_lags(x_enc: torch.Tensor, top_k: int = 5) -> torch.Tensor:
        x = x_enc.permute(0, 2, 1)
        q_fft = torch.fft.rfft(x, dim=-1)
        res = q_fft * torch.conj(q_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value[:, 1:], top_k, dim=-1)
        return lags + 1
```

### Phase 2: Main Model Class

#### 3.5 Create `models/time_llm/time_llm_model.py`

See existing implementation plan for full model code. Key integration point:

```python
# REUSE existing LLM infrastructure
from embedder.llm_registry import LLMRegistry

class TimeLLM(nn.Module):
    def _load_llm(self, llm_model, cache_dir, device, quantization):
        model_map = {
            'LLAMA': 'meta-llama/Llama-3.1-8B-Instruct',
            'GPT2': 'gpt2',
            'QWEN': 'Qwen/Qwen2.5-7B-Instruct',
        }
        model_name = model_map.get(llm_model.upper(), llm_model)
        
        # Use existing LLMRegistry
        self.llm_model, _ = LLMRegistry.get_model(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            quantization=quantization,
        )
        self.tokenizer = LLMRegistry.get_tokenizer(model_name, cache_dir)
```

---

## 4. Model-Specific Training (NEW: Following LeRet Pattern)

### 4.1 Create `cli/config/model_training.py` - Add TimeLLMTrainingConfig

```python
class TimeLLMTrainingConfig(BaseModel):
    """
    Configuration for Time-LLM training.
    
    Time-LLM uses a frozen LLM backbone and only trains:
    - PatchEmbedding layer
    - ReprogrammingLayer  
    - Word embedding mapping layer
    - Output projection (FlattenHead)
    
    Special considerations:
    - LLM is always frozen (requires_grad=False)
    - Supports quantization for large LLMs (4-bit, 8-bit)
    - Dynamic prompts generated per-batch during forward pass
    
    Example YAML config:
        training:
          time_llm:
            llm_backbone: "gpt2"
            quantization: null
            prompt_domain: true
            dataset_description: "ETT dataset for power transformer monitoring"
    
    Attributes:
        llm_backbone: HuggingFace model name or alias (GPT2, LLAMA, QWEN)
        quantization: Quantization mode for LLM (4bit, 8bit, or null)
        llm_cache_dir: Cache directory for LLM model weights
        prompt_domain: If True, use provided dataset_description
        dataset_description: Domain-specific description for dynamic prompts
        loss: Loss function for training (mse or mae)
    """
    model_config = ConfigDict(extra="forbid")
    
    llm_backbone: str = Field(
        default="gpt2",
        description="HuggingFace model name or alias (GPT2, LLAMA, QWEN, or full HF name)"
    )
    
    quantization: Optional[Literal["4bit", "8bit"]] = Field(
        default=None,
        description="Quantization mode for frozen LLM (4bit, 8bit, or null for fp16)"
    )
    
    llm_cache_dir: str = Field(
        default="./LLM_cache/",
        description="Cache directory for LLM model weights"
    )
    
    prompt_domain: bool = Field(
        default=False,
        description="If True, use dataset_description in dynamic prompts"
    )
    
    dataset_description: str = Field(
        default="",
        description="Domain-specific dataset description for dynamic prompts"
    )
    
    loss: Literal["mse", "mae"] = Field(
        default="mse",
        description="Loss function for forecasting"
    )


# Add to registry
_MODEL_CONFIG_REGISTRY: dict = {
    "LeRet": ("leret", LeRetTrainingConfig),
    "TimeLLM": ("time_llm", TimeLLMTrainingConfig),
}
```

### 4.2 Create `exp/model_specific/time_llm.py`

```python
"""
Time-LLM Model-Specific Training Module.

This module provides training infrastructure for Time-LLM, implementing
both PyTorch Lightning and standard PyTorch training paths.

Key Features:
    - Frozen LLM backbone (only train non-LLM parameters)
    - Supports quantization (4-bit, 8-bit) for large LLMs
    - Dynamic prompts generated per-batch during forward pass

Exports:
    - train_time_llm_lightning: Lightning-based training
    - train_time_llm_pytorch: Standard PyTorch training
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings

from models import model_init
from utils.tools import adjust_learning_rate, EarlyStopping
from cli.config.model_training import TimeLLMTrainingConfig
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

warnings.filterwarnings('ignore')


# =============================================================================
# Shared Utilities
# =============================================================================

def _select_criterion(loss_type: str) -> nn.Module:
    """Select the appropriate loss function."""
    if loss_type == 'mae' or loss_type == 'l1':
        return nn.L1Loss()
    return nn.MSELoss()


def _get_time_llm_config(args) -> TimeLLMTrainingConfig:
    """Extract and validate Time-LLM training config from args."""
    time_llm_config = getattr(args, 'time_llm', None)
    if time_llm_config is None:
        return TimeLLMTrainingConfig()
    elif isinstance(time_llm_config, dict):
        return TimeLLMTrainingConfig(**time_llm_config)
    return time_llm_config


def _log_trainable_params(model, logger):
    """Log trainable vs frozen parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(f"[ Time-LLM ] Trainable params: {trainable:,}")
    logger.info(f"[ Time-LLM ] Frozen params: {frozen:,}")


# =============================================================================
# PyTorch Lightning Training
# =============================================================================

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping as PLEarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    pl = None


if HAS_LIGHTNING:
    class TimeLLMLightningModule(pl.LightningModule):
        """PyTorch Lightning module for Time-LLM with frozen LLM backbone."""
        
        def __init__(self, args, exp_manager=None, 
                     time_llm_config: Optional[TimeLLMTrainingConfig] = None):
            super().__init__()
            self.args = args
            self.exp_manager = exp_manager
            self.time_llm_config = time_llm_config or TimeLLMTrainingConfig()
            
            self.save_hyperparameters(ignore=['args', 'exp_manager', 'time_llm_config'])
            
            # Build the Time-LLM model
            self.model = model_init(self.args.model, self.args.model_config, self.args)
            
            # Loss function
            self.criterion = _select_criterion(self.time_llm_config.loss)
            
            self._epoch_start_time: Optional[float] = None
        
        def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
            sample_ids, batch_x, batch_y, timestamp_x, timestamp_y, \
                batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, \
                hetero_general, hetero_channel = batch
            
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Time-LLM forward pass
            forecast = self.model(batch_x)
            
            return forecast, batch_y
        
        def training_step(self, batch, batch_idx):
            forecast, batch_y = self.forward(batch)
            output = forecast[:, -self.args.output_len:, :]
            loss = self.criterion(output, batch_y)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        
        def validation_step(self, batch, batch_idx):
            forecast, batch_y = self.forward(batch)
            output = forecast[:, -self.args.output_len:, :]
            loss = self.criterion(output, batch_y)
            self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
            return loss
        
        def test_step(self, batch, batch_idx, dataloader_idx=0):
            forecast, batch_y = self.forward(batch)
            output = forecast[:, -self.args.output_len:, :]
            loss = self.criterion(output, batch_y)
            self.log('test_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        
        def on_train_epoch_start(self):
            if self.exp_manager and hasattr(self.exp_manager, 'gpu_monitor') and self.exp_manager.gpu_monitor:
                self.exp_manager.gpu_monitor.mark_epoch_start()
            self._epoch_start_time = time.time()
        
        def on_train_epoch_end(self):
            if self.trainer.sanity_checking:
                return
            current_epoch = self.trainer.current_epoch + 1
            if self._epoch_start_time is not None and self.exp_manager:
                epoch_time = time.time() - self._epoch_start_time
                self.exp_manager.log_file_only(f"Epoch {current_epoch} completed in {epoch_time:.2f}s")
        
        def configure_optimizers(self):
            # Only optimize non-frozen parameters (excludes LLM backbone)
            trainable_params = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=self.args.learning_rate)
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda epoch: adjust_learning_rate(None, epoch, self.args, return_rate=True)
                ),
                'name': 'learning_rate',
                'interval': 'epoch',
                'frequency': 1
            }
            return [optimizer], [lr_scheduler]


def train_time_llm_lightning(args, exp_manager) -> Path:
    """
    Train Time-LLM model using PyTorch Lightning.
    
    Args:
        args: Experiment arguments with time_llm config
        exp_manager: ExperimentManager for tracking
    
    Returns:
        Path to best model checkpoint
    """
    if not HAS_LIGHTNING:
        raise ImportError("PyTorch Lightning is required for Lightning training")
    
    from data_provider.lightning_data_module import TimeSeriesDataModule
    
    time_llm_config = _get_time_llm_config(args)
    
    exp_manager.logger.info("=" * 60)
    exp_manager.logger.info("Time-LLM Training (Lightning)")
    exp_manager.logger.info(f"  LLM Backbone: {time_llm_config.llm_backbone}")
    exp_manager.logger.info(f"  Quantization: {time_llm_config.quantization}")
    exp_manager.logger.info("=" * 60)
    
    model = TimeLLMLightningModule(args, exp_manager, time_llm_config)
    _log_trainable_params(model.model, exp_manager.logger)
    
    data_module = TimeSeriesDataModule(args)
    checkpoint_dir = str(exp_manager.get_checkpoint_dir())
    
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='checkpoint-{epoch:02d}-{val_loss:.6f}',
        save_top_k=1, monitor='val_loss', mode='min', save_last=True
    )
    
    class JobHistoryCallback(pl.Callback):
        def __init__(self, em): self.em = em
        def on_train_epoch_end(self, trainer, pl_module):
            if not trainer.sanity_checking:
                self.em.update_current_epoch(trainer.current_epoch + 1)
    
    trainer = pl.Trainer(
        max_epochs=args.train_epochs,
        accelerator='gpu' if args.use_gpu else 'cpu',
        devices=[args.gpu] if args.use_gpu and not args.use_multi_gpu else (
            args.device_ids if args.use_multi_gpu else None
        ),
        strategy='ddp' if args.use_multi_gpu else 'auto',
        callbacks=[
            checkpoint_cb, 
            PLEarlyStopping(monitor='val_loss', patience=args.patience, mode='min'),
            JobHistoryCallback(exp_manager)
        ],
        logger=TensorBoardLogger(str(exp_manager.get_experiment_dir() / "tb_logs"), name="time_llm"),
        deterministic=True,
        precision=getattr(args, 'precision', 32),
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )
    
    trainer.fit(model, data_module)
    
    best_model_path = checkpoint_cb.best_model_path or checkpoint_cb.last_model_path
    
    # Run final testing
    data_module.setup(stage='test')
    test_loaders = data_module.test_dataloader()
    test_results = {}
    
    for subset_id, loader in test_loaders.items():
        trainer.test(model, dataloaders=loader, ckpt_path=best_model_path)
        test_results[subset_id] = trainer.callback_metrics['test_loss'].item()
    
    if trainer.is_global_zero:
        with open(exp_manager.get_checkpoint_dir() / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
    
    exp_manager.register_job_end(
        end_epoch=trainer.current_epoch + 1, 
        status="completed", 
        checkpoint_path=best_model_path
    )
    
    return Path(best_model_path)


# =============================================================================
# Standard PyTorch Training
# =============================================================================

class TimeLLMPyTorchTrainer:
    """Standard PyTorch trainer for Time-LLM with frozen LLM backbone."""
    
    def __init__(self, args, exp_manager):
        self.args = args
        self.exp_manager = exp_manager
        self.device = self._get_device()
        
        self.model = self._build_model()
        self.model.to(self.device)
        
        _log_trainable_params(self.model, exp_manager.logger)
        
        from data_provider.data_factory import Data_Provider
        console = exp_manager.get_console() if exp_manager else None
        self.data_provider = Data_Provider(args, buffer=(not args.disable_buffer), console=console)
    
    def _get_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            return torch.device(f'cuda:{self.args.gpu}')
        return torch.device('cpu')
    
    def _build_model(self):
        model = model_init(self.args.model, self.args.model_config, self.args)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _forward_step(self, batch):
        sample_ids, batch_x, batch_y, timestamp_x, timestamp_y, \
            batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, \
            hetero_general, hetero_channel, x_time_features, y_time_features = batch
        
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        
        forecast = self.model(x=batch_x)
        return forecast, batch_y, sample_ids
    
    def _train_epoch(self, train_loader, optimizer, criterion, epoch):
        self.model.train()
        total_loss, total_samples = 0.0, 0
        epoch_time = time.time()
        console = self.exp_manager.console
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("• loss: {task.fields[loss]:.7f}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Epoch {epoch}", total=len(train_loader), loss=0.0)
            
            for batch in train_loader:
                optimizer.zero_grad()
                forecast, batch_y, _ = self._forward_step(batch)
                
                output = forecast[:, -self.args.output_len:, :]
                loss = criterion(output, batch_y)
                
                loss.backward()
                optimizer.step()
                
                batch_size = batch_y.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                progress.update(task, advance=1, loss=loss.item())
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        epoch_time_elapsed = time.time() - epoch_time
        return avg_loss, epoch_time_elapsed
    
    def _validate(self, loader, criterion):
        self.model.eval()
        total_loss, total_samples = 0.0, 0
        
        with torch.no_grad():
            for batch in loader:
                forecast, batch_y, _ = self._forward_step(batch)
                output = forecast[:, -self.args.output_len:, :]
                loss = criterion(output, batch_y)
                
                batch_size = batch_y.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        self.model.train()
        return total_loss / total_samples if total_samples > 0 else 0.0
    
    def _test(self, loaders, criterion):
        self.model.eval()
        results = {}
        overall_loss, overall_samples = 0.0, 0
        
        with torch.no_grad():
            for subset_id, loader in loaders.items():
                subset_loss, subset_samples = 0.0, 0
                
                for batch in loader:
                    forecast, batch_y, _ = self._forward_step(batch)
                    output = forecast[:, -self.args.output_len:, :]
                    loss = criterion(output, batch_y)
                    
                    batch_size = batch_y.size(0)
                    subset_loss += loss.item() * batch_size
                    subset_samples += batch_size
                
                avg_loss = subset_loss / subset_samples if subset_samples > 0 else 0.0
                results[subset_id] = avg_loss
                overall_loss += subset_loss
                overall_samples += subset_samples
        
        return results, overall_loss / overall_samples if overall_samples > 0 else 0.0
    
    def train(self, time_llm_config: TimeLLMTrainingConfig) -> Path:
        """Run full training loop."""
        self.exp_manager.logger.info("=" * 60)
        self.exp_manager.logger.info("Time-LLM Training (PyTorch)")
        self.exp_manager.logger.info(f"  Epochs: {self.args.train_epochs}")
        self.exp_manager.logger.info("=" * 60)
        
        train_loader = self.data_provider.get_train(return_type='loader')
        val_loader = self.data_provider.get_val(return_type='loader')
        test_loaders = self.data_provider.get_test(return_type='loader')
        
        self.data_provider.data_buffer.clear()
        
        # Only optimize non-frozen parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=self.args.learning_rate)
        criterion = _select_criterion(time_llm_config.loss)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        checkpoint_dir = self.exp_manager.get_checkpoint_dir()
        best_val_loss = float('inf')
        
        for epoch in range(1, self.args.train_epochs + 1):
            train_loss, epoch_time = self._train_epoch(train_loader, optimizer, criterion, epoch)
            val_loss = self._validate(val_loader, criterion)
            
            self.exp_manager.logger.info(
                f"Epoch {epoch}/{self.args.train_epochs} | "
                f"Train: {train_loss:.7f} | Val: {val_loss:.7f} | Time: {epoch_time:.2f}s"
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), checkpoint_dir / 'checkpoint.pth')
            
            early_stopping(val_loss, self.model, str(checkpoint_dir))
            if early_stopping.early_stop:
                self.exp_manager.logger.info("Early stopping triggered")
                break
            
            adjust_learning_rate(optimizer, epoch, self.args)
            self.exp_manager.update_current_epoch(epoch)
        
        # Load best model for testing
        best_model_path = checkpoint_dir / 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        
        # Final testing
        self.exp_manager.logger.info("Running final testing...")
        test_results, overall_test_loss = self._test(test_loaders, criterion)
        
        self.exp_manager.logger.info(f"Test results: {test_results}")
        self.exp_manager.logger.info(f"Overall test loss: {overall_test_loss:.7f}")
        
        with open(checkpoint_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        self.exp_manager.register_job_end(
            end_epoch=epoch, status="completed",
            checkpoint_path=str(best_model_path),
            final_train_loss=train_loss, final_val_loss=val_loss
        )
        
        return best_model_path


def train_time_llm_pytorch(args, exp_manager) -> Path:
    """
    Train Time-LLM model using standard PyTorch.
    
    Args:
        args: Experiment arguments with time_llm config
        exp_manager: ExperimentManager for tracking
    
    Returns:
        Path to best model checkpoint
    """
    time_llm_config = _get_time_llm_config(args)
    exp_manager.logger.info(f"Time-LLM PyTorch Training")
    
    trainer = TimeLLMPyTorchTrainer(args, exp_manager)
    return trainer.train(time_llm_config)
```

### 4.3 Update `exp/model_specific/__init__.py`

```python
def _register_all_trainers():
    """Register all model-specific trainers."""
    # LeRet: Two-stage training (pretrain + finetune)
    from exp.model_specific.leret import (
        train_leret_lightning,
        train_leret_pytorch
    )
    register_model_trainer("LeRet", "lightning", train_leret_lightning)
    register_model_trainer("LeRet", "pytorch", train_leret_pytorch)
    
    # Time-LLM: Frozen LLM with online reprogramming
    from exp.model_specific.time_llm import (
        train_time_llm_lightning,
        train_time_llm_pytorch
    )
    register_model_trainer("TimeLLM", "lightning", train_time_llm_lightning)
    register_model_trainer("TimeLLM", "pytorch", train_time_llm_pytorch)
```

---

## 5. Updated File Structure

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
│   └── model_specific/
│       ├── __init__.py                 # Trainer registry (update)
│       ├── leret.py                    # LeRet training (existing)
│       └── time_llm.py                 # Time-LLM training (NEW)
├── cli/
│   └── config/
│       └── model_training.py           # Add TimeLLMTrainingConfig
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

## 6. Configuration Schema

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
  
  # Model-specific training config
  time_llm:
    llm_backbone: "gpt2"
    quantization: null
    prompt_domain: true
    dataset_description: "ETT dataset for power transformer monitoring"
    loss: "mse"

prompt:
  domain_specific: true
```

---

## 7. Implementation Priority & Dependencies

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
│ 3a. cli/config/model_training.py - Add TimeLLMTrainingConfig            │
│ 3b. exp/model_specific/time_llm.py - Training functions                 │
│     - train_time_llm_lightning()                                        │
│     - train_time_llm_pytorch()                                          │
│ 3c. exp/model_specific/__init__.py - Register trainers                  │
│ 3d. model_configs/time_llm/*.yaml - Config files                        │
│ 3e. Register model in models/__init__.py                                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Key Integration Points

### 8.1 Using Model-Specific Training Registry

```python
# In training runner (runs/lightning.py or runs/pytorch.py)
from exp.model_specific import has_custom_trainer, get_model_trainer

def train_model(args, exp_manager):
    model_name = args.model
    framework = "lightning"  # or "pytorch"
    
    # Check if model needs custom handling
    if has_custom_trainer(model_name, framework):
        trainer_fn = get_model_trainer(model_name, framework)
        return trainer_fn(args, exp_manager)
    else:
        # Use default training
        return default_train(args, exp_manager)
```

### 8.2 Using Existing LLMRegistry

```python
# In time_llm_model.py - REUSE existing infrastructure
from embedder.llm_registry import LLMRegistry

self.llm_model, embed_dim = LLMRegistry.get_model(
    model_name='Qwen/Qwen2.5-72B-Instruct',
    device='cuda:0',
    cache_dir='./LLM_cache/',
    quantization='4bit'
)
```

### 8.3 NOT Using (Different Paradigm)

```python
# These are for PRECOMPUTED embeddings (TimeCMA), not Time-LLM:
# - LLMEmbedder, LLMEmbeddingCache, LLMEmbeddingProvider, TSPromptBuilder

# Time-LLM does NOT need these because:
# 1. Dynamic prompts are generated per-batch in forward pass
# 2. Reprogramming happens online, not via precomputed embeddings
```

---

## 9. LLM Dimension Reference

| LLM Model | Hidden Dim | HuggingFace Name | Min VRAM (4-bit) |
|-----------|------------|------------------|------------------|
| GPT-2 | 768 | `gpt2` | 0.3 GB |
| Qwen2.5-7B | 3584 | `Qwen/Qwen2.5-7B-Instruct` | 5.0 GB |
| Qwen2.5-14B | 5120 | `Qwen/Qwen2.5-14B-Instruct` | 9.0 GB |
| Qwen2.5-72B | 8192 | `Qwen/Qwen2.5-72B-Instruct` | 42.0 GB |
| LLaMA-3.1-8B | 4096 | `meta-llama/Llama-3.1-8B-Instruct` | 6.0 GB |
| LLaMA-3.1-70B | 8192 | `meta-llama/Llama-3.1-70B-Instruct` | 40.0 GB |

---

## 10. Testing Strategy

### 10.1 Unit Tests

```python
# tests/test_time_llm_components.py
def test_normalize():
    norm = Normalize(num_features=7)
    x = torch.randn(2, 96, 7)
    x_norm = norm(x, 'norm')
    x_denorm = norm(x_norm, 'denorm')
    assert torch.allclose(x, x_denorm, atol=1e-5)

def test_patch_embedding():
    patch_embed = PatchEmbedding(d_model=32, patch_len=16, stride=8, dropout=0.1)
    x = torch.randn(2, 7, 96)
    out, n_vars = patch_embed(x)
    assert n_vars == 7
    assert out.shape == (2 * 7, 12, 32)
```

### 10.2 Integration Test

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_time_llm_forward():
    from models.time_llm.time_llm_model import TimeLLM
    model = TimeLLM(llm_model='gpt2', seq_len=96, pred_len=96, enc_in=7, device='cuda:0')
    x = torch.randn(2, 96, 7).to('cuda:0')
    out = model(x, None, None, None)
    assert out.shape == (2, 96, 7)
```

---

## 11. Open Questions / Future Work

1. **Memory Optimization**: Can we use gradient checkpointing for frozen LLM layers?

2. **Quantization Impact**: Does 4-bit/8-bit quantization affect reprogramming quality?

3. **Multi-modal Extension**: Combine online reprogramming with precomputed text embeddings?

4. **Channel Independence**: Share reprogramming across channels or keep separate?

5. **Prompt Caching**: Cache dynamic prompts during training for faster iteration?

---

## 12. References

- **Paper**: [Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://arxiv.org/abs/2310.01728)
- **Original Code**: `benchmark_models/Time-LLM/`
- **Related Work**: TimeCMA (uses precomputed embeddings instead of reprogramming)
- **LeRet Pattern**: See `exp/model_specific/leret.py` for model-specific training pattern
- **Existing Infrastructure**:
  - `embedder/llm_registry.py`: LLM model loading with quantization
  - `exp/model_specific/__init__.py`: Trainer registration
  - `cli/config/model_training.py`: Model-specific config classes
