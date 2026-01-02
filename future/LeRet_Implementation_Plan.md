# LeRet Model Implementation Plan

## 1. Model Overview

**LeRet** (Language-Enhanced Retention Network for Time Series) is a time series forecasting model that combines:

1. **RetNet (Retention Network)** - A linear-complexity sequence model replacing traditional quadratic-cost self-attention with a retention mechanism
2. **Patch-based Input Processing** - Similar to PatchTST, divides time series into patches for efficient encoding
3. **Language Knowledge Integration** - Cross-attention mechanism to incorporate text embeddings (via fidel-ts embedder) for enhanced forecasting
4. **Configurable Two-Stage Training** - Auto-regressive pretraining followed by forecasting fine-tuning, with flexible stage control

### 1.1 Key Innovations

| Component | Description | Benefit |
|-----------|-------------|---------|
| **Multi-Scale Retention** | Replaces self-attention with exponentially decaying retention | O(n) complexity vs O(n²), better long-range dependencies |
| **Patch Embedding** | Input segmented into patches with linear projection | Reduces sequence length, captures local patterns |
| **RevIN** | Reversible Instance Normalization | Handles distribution shift between train/test |
| **TS-Language Integrator** | Bidirectional cross-attention with language embeddings | Injects domain knowledge into predictions |
| **Dual-Head Architecture** | Auto-regressive + sequence prediction heads | Enables two-stage curriculum learning |

### 1.2 Design Decisions (Resolved)

| Question | Decision | Rationale |
|----------|----------|-----------|
| **MoE Support** | ❌ Not needed | Simplifies implementation; not required for fidel-ts use cases |
| **Language Embeddings** | Use fidel-ts `embedder/` infrastructure | Centralized, cached embeddings with `TextEmbedder` and `FidelTSEmbeddingLoader` |
| **Chunkwise Recurrent** | ❌ Initially omitted (parallel mode only) | Simpler implementation; documented for future extension if needed |
| **Two-Stage Training** | ✅ Fully configurable | User can run pretrain only, finetune only, or both; must be within same experiment |

---

## 2. Architecture Deep Dive

### 2.1 Overall Data Flow

```
Input: [Batch, Seq_Len, Channels]
         │
         ▼
    ┌─────────────┐
    │   RevIN     │ (Instance Normalization)
    │   (norm)    │
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │  Patching   │ (Unfold into patches)
    │             │ → [Batch, Channels, Patch_Num, Patch_Len]
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │ LeRetEncoder│ (RetNet backbone)
    │  - W_P      │ (Linear projection: patch_len → d_model)
    │  - RetNet   │ (Multi-scale retention layers, parallel mode)
    └─────────────┘
         │
         ▼ h: [Batch, Channels, d_model, Patch_Num]
    ┌─────────────────────┬───────────────────────┐
    │                     │                       │
    ▼                     ▼                       │
┌────────────┐    ┌────────────────────┐          │
│ Patch Head │    │TS-Language Integrator│        │
│(auto_y)    │    │  - ts2text         │          │
└────────────┘    │  - text2ts         │          │
    │             └────────────────────┘          │
    │                     │                       │
    │                     ▼                       │
    │             ┌───────────────┐               │
    │             │ Sequence Head │               │
    │             │ (flatten+fc)  │               │
    │             └───────────────┘               │
    │                     │                       │
    │                     ▼                       │
    │             ┌─────────────┐                 │
    │             │   RevIN     │                 │
    │             │  (denorm)   │                 │
    │             └─────────────┘                 │
    │                     │                       │
    ▼                     ▼                       │
  auto_y                 output                   │
```

### 2.2 Core Components

#### 2.2.1 RetNet Backbone (`LeRetEncoder`)

The encoder uses a simplified **RetNetDecoder** with only **parallel mode** retention:

```python
# Key configuration (simplified - no MoE, no chunkwise)
config = RetNetConfig(
    decoder_layers=n_layers,           # Number of retention layers
    decoder_embed_dim=d_model,         # Hidden dimension
    decoder_ffn_embed_dim=256,         # FFN intermediate dimension
    dropout=dropout,
    decoder_retention_heads=n_heads,
    decoder_value_embed_dim=d_model,
    # Disabled features:
    moe_freq=0,                        # No Mixture-of-Experts
    chunkwise_recurrent=False,         # Parallel mode only (see Section 2.2.1.1)
)
```

**Multi-Scale Retention** mechanism formula:
```
Retention(X) = (QK^T ⊙ D) V

where:
- Q, K, V = linear projections of X
- D = exponential decay mask: D_nm = γ^(n-m) for n ≥ m, else 0
- γ = learned per-head decay rate
```

##### 2.2.1.1 Operating Mode: Parallel Only

The original RetNet supports three modes. For this implementation, we use **parallel mode only**:

| Mode | Status | Description |
|------|--------|-------------|
| **Parallel** | ✅ Implemented | Full attention-like computation, O(n²) memory but parallelizable |
| **Recurrent** | 📋 Future | O(1) memory per step, for inference |
| **Chunk-Recurrent** | 📋 Future | Hybrid for long sequences |

> **Note for Future Extension**: Chunkwise recurrent mode can be added by:
> 1. Adding `chunkwise_recurrent` and `recurrent_chunk_size` config options
> 2. Implementing `chunk_recurrent_forward()` in `MultiScaleRetention`
> 3. Updating `RetNetRelPos` to compute chunk-based decay masks
> See original `retnet.py` lines 37-57 and `multiscale_retention.py` lines 114-165 for reference.

#### 2.2.2 TS-Language Integrator (Using fidel-ts Embedder)

Bidirectional cross-attention between time series patches and text embeddings from the **centralized fidel-ts embedder infrastructure**:

```python
# Language embeddings loaded via fidel-ts embedder (not hardcoded .pt file)
from embedder import TextEmbedder, FidelTSEmbeddingLoader

class TSLanguageIntegrator(nn.Module):
    def __init__(self, d_model: int, language_embed_dim: int, ...):
        super().__init__()
        # Project language embeddings to model dimension
        self.text_linear = nn.Linear(language_embed_dim, d_model)
        
        # Cross-attention modules
        self.ts2text = CrossEncoder(d_model, n_heads, ...)
        self.text2ts = CrossEncoder(d_model, n_heads, ...)
    
    def forward(self, z_patch: Tensor, language_embeddings: Tensor) -> Tensor:
        """
        Args:
            z_patch: [bs*nvars, patch_num, d_model] - encoded time series patches
            language_embeddings: [text_num, language_embed_dim] - from fidel-ts embedder
        
        Returns:
            z_output: [bs*nvars, patch_num, d_model] - language-enhanced patches
        """
        # Project language embeddings to model dimension
        lang_proj = self.text_linear(language_embeddings)  # [text_num, d_model]
        
        # Expand for batch processing
        lang_proj = lang_proj.unsqueeze(0).expand(z_patch.size(0), -1, -1)
        
        # Cross-attention: Text → TS
        lang_enhanced = self.ts2text(lang_proj, z_patch, z_patch)
        
        # Cross-attention: TS → Text
        z_output = self.text2ts(z_patch, lang_enhanced, lang_enhanced)
        
        return z_output
```

**Integration with fidel-ts Embedder:**

```python
# In model initialization or data loading:
from embedder import TextEmbedder

# Option 1: Load pre-computed embeddings (recommended for reproducibility)
embedder = TextEmbedder(
    model_name='bert-base-uncased',  # or other supported model
    aggregation_method='cls',
    device='cuda',
    cache_root='./embedding_cache/'
)

# Option 2: Use FidelTSEmbeddingLoader for dataset-specific embeddings
from embedder import FidelTSEmbeddingLoader

loader = FidelTSEmbeddingLoader(
    dataset_name='Bear_room',
    hetero_info=data_config['hetero_info'],
    base_data_path='./data/',
    embed_model_name='bert-base-uncased'
)
dynamic_embeddings, static_embeddings = loader.load_embeddings()
```

#### 2.2.3 Dual Prediction Heads

| Head | Purpose | Architecture |
|------|---------|--------------|
| **Patch Head** | Auto-regressive pretraining (Stage 1) | `Linear(d_model → patch_len) + Flatten` |
| **Sequence Head** | Final forecasting (Stage 2) | `Flatten + Linear(d_model * patch_num → target_window)` |

### 2.3 Configurable Two-Stage Training

LeRet uses a **two-stage curriculum** that is **fully configurable** through the experiment config:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         SAME EXPERIMENT (experiment_id)                    │
│                                                                            │
│   ┌─────────────────────────┐         ┌─────────────────────────┐         │
│   │    Stage 1: Pretrain    │  ───►   │   Stage 2: Finetune     │         │
│   │  (Auto-Regressive Loss) │         │   (Forecasting Loss)    │         │
│   │                         │         │                         │         │
│   │  Loss: MSE(patch_head,  │         │  Loss: MSE(seq_head,    │         │
│   │        y_auto)          │  save   │        target)          │         │
│   │                         │ ckpt    │                         │         │
│   │  Epochs: pretrain_epochs│ ───────►│  Load: pretrain_ckpt    │         │
│   └─────────────────────────┘         └─────────────────────────┘         │
│                                                                            │
│   job_history.json tracks both stages within the same experiment           │
└────────────────────────────────────────────────────────────────────────────┘
```

#### 2.3.1 Training Stage Configuration

```yaml
# In experiment config (configs/experiments/*.yaml)
training:
  # LeRet-specific two-stage training options
  leret:
    training_stage: "both"  # Options: "pretrain", "finetune", "both"
    
    # Stage 1 (Pretraining) settings
    pretrain_epochs: 10
    pretrain_loss: "mse"  # Loss for patch_head output
    
    # Stage 2 (Finetuning) settings - uses standard training.epochs
    finetune_loss: "mse"  # Loss for sequence_head output
    
    # If training_stage is "finetune", must specify pretrain checkpoint:
    # pretrain_checkpoint: null  # Auto-detected from same experiment
    # OR explicit path (rarely needed):
    # pretrain_checkpoint: "/path/to/pretrain_checkpoint.pth"
```

#### 2.3.2 Stage Execution Logic

| `training_stage` | Behavior |
|------------------|----------|
| `"pretrain"` | Run Stage 1 only. Save checkpoint as `pretrain_checkpoint.pth`. Mark stage in job_history. |
| `"finetune"` | Run Stage 2 only. **Requires** pretrain checkpoint from same experiment. Validates experiment_id match. |
| `"both"` | Run Stage 1, save checkpoint, then run Stage 2. Single experiment, two phases. |

#### 2.3.3 Integration with ExperimentManager

Both stages **must** occur within the **same experiment** for consistency and reproducibility:

```python
# In exp/exp_lightning.py or exp/exp_leret.py

class LeRetExperiment:
    def __init__(self, config: ExperimentConfig, exp_manager: ExperimentManager):
        self.config = config
        self.exp_manager = exp_manager
        self.leret_config = config.training.leret
        
    def run(self):
        stage = self.leret_config.training_stage
        
        if stage in ["pretrain", "both"]:
            self._run_pretrain_stage()
        
        if stage in ["finetune", "both"]:
            self._run_finetune_stage()
    
    def _run_pretrain_stage(self):
        """Stage 1: Auto-regressive pretraining."""
        self.exp_manager.logger.info("Starting Stage 1: Auto-regressive pretraining")
        
        # Train with patch_head loss
        for epoch in range(1, self.leret_config.pretrain_epochs + 1):
            train_loss = self._train_epoch_pretrain()
            val_loss = self._validate_pretrain()
            
            # Update job history after each epoch
            self.exp_manager.update_current_epoch(
                epoch=epoch,
                checkpoint_path=None  # Checkpoint saved at end of stage
            )
            self.exp_manager.log_metrics({
                'pretrain/train_loss': train_loss,
                'pretrain/val_loss': val_loss,
            }, step=epoch)
        
        # Save pretrain checkpoint
        pretrain_ckpt_path = self.exp_manager.save_checkpoint(
            checkpoint={
                'epoch': self.leret_config.pretrain_epochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'stage': 'pretrain',
                'experiment_id': self.exp_manager.experiment_id,
            },
            filename='pretrain_checkpoint.pth',
            is_best=False  # Not the final best model
        )
        
        # Update job history with stage completion
        self.exp_manager.job_history['pretrain_completed'] = True
        self.exp_manager.job_history['pretrain_checkpoint'] = str(pretrain_ckpt_path)
        self.exp_manager._save_job_history()
        
        self.exp_manager.logger.info(f"Stage 1 complete. Checkpoint: {pretrain_ckpt_path}")
    
    def _run_finetune_stage(self):
        """Stage 2: Forecasting fine-tuning."""
        # Validate and load pretrain checkpoint
        pretrain_ckpt = self._get_pretrain_checkpoint()
        
        self.exp_manager.logger.info(f"Starting Stage 2: Finetuning from {pretrain_ckpt}")
        
        # Load pretrain checkpoint
        checkpoint = torch.load(pretrain_ckpt)
        
        # Validate experiment_id match
        if checkpoint.get('experiment_id') != self.exp_manager.experiment_id:
            raise ValueError(
                f"Pretrain checkpoint experiment_id mismatch. "
                f"Expected: {self.exp_manager.experiment_id}, "
                f"Got: {checkpoint.get('experiment_id')}. "
                f"Both stages must be part of the same experiment."
            )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Continue epoch numbering from pretrain
        start_epoch = self.leret_config.pretrain_epochs + 1
        total_epochs = start_epoch + self.config.training.epochs - 1
        
        # Train with sequence_head loss
        for epoch in range(start_epoch, total_epochs + 1):
            train_loss = self._train_epoch_finetune()
            val_loss = self._validate_finetune()
            
            self.exp_manager.update_current_epoch(epoch=epoch)
            self.exp_manager.log_metrics({
                'finetune/train_loss': train_loss,
                'finetune/val_loss': val_loss,
            }, step=epoch)
        
        # Save final checkpoint
        self.exp_manager.save_checkpoint(
            checkpoint={
                'epoch': total_epochs,
                'model_state_dict': self.model.state_dict(),
                'stage': 'finetune',
                'experiment_id': self.exp_manager.experiment_id,
            },
            filename='checkpoint.pth',
            is_best=True
        )
        
        self.exp_manager.register_job_end(
            end_epoch=total_epochs,
            status='completed',
            final_val_loss=val_loss
        )
    
    def _get_pretrain_checkpoint(self) -> Path:
        """Get pretrain checkpoint path, validating it exists and matches experiment."""
        # Check if explicit path provided
        if self.leret_config.pretrain_checkpoint:
            ckpt_path = Path(self.leret_config.pretrain_checkpoint)
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Specified pretrain_checkpoint not found: {ckpt_path}"
                )
            return ckpt_path
        
        # Auto-detect from same experiment
        expected_path = self.exp_manager.get_checkpoint_dir() / 'pretrain_checkpoint.pth'
        
        if not expected_path.exists():
            # Check job_history for pretrain completion
            if not self.exp_manager.job_history.get('pretrain_completed'):
                raise ValueError(
                    f"Cannot run finetune stage: pretrain stage not completed. "
                    f"Set training_stage='both' or 'pretrain' first, "
                    f"or provide explicit pretrain_checkpoint path."
                )
            raise FileNotFoundError(
                f"Pretrain checkpoint not found at expected path: {expected_path}. "
                f"Job history indicates pretrain was completed but checkpoint is missing."
            )
        
        return expected_path
```

#### 2.3.4 Job History Structure for Two-Stage Training

```json
{
  "experiment_id": "20260102-143052_abc123def456",
  "total_epochs": 110,  // pretrain_epochs (10) + finetune_epochs (100)
  "jobs": [
    {
      "job_id": "12345",
      "slurm_job_id": "12345",
      "start_time": "2026-01-02T14:30:52",
      "end_time": "2026-01-02T16:45:30",
      "start_epoch": 0,
      "end_epoch": 110,
      "status": "completed",
      "stages_completed": ["pretrain", "finetune"]
    }
  ],
  "current_epoch": 110,
  "pretrain_completed": true,
  "pretrain_checkpoint": "/output/20260102-143052_abc123def456/checkpoints/pretrain_checkpoint.pth",
  "last_checkpoint": "/output/20260102-143052_abc123def456/checkpoints/checkpoint.pth",
  "best_checkpoint": "/output/20260102-143052_abc123def456/checkpoints/checkpoint.pth",
  "wandb_run_id": "abc123xyz"
}
```

---

## 3. Implementation Plan for fidel-ts

### 3.1 File Structure

```
fidel-ts-worktree-lynx/
├── layers/
│   ├── LeRet_backbone.py          # NEW: LeRetEncoder, heads
│   ├── LeRet_layers.py            # NEW: Utility layers (series_decomp, etc.)
│   ├── RetNet/                    # NEW: Retention network components (simplified)
│   │   ├── __init__.py
│   │   ├── config.py              # RetNetConfig (no MoE options)
│   │   ├── retnet.py              # RetNetDecoder, DecoderLayer (parallel only)
│   │   ├── multiscale_retention.py # Parallel mode only, documented for extension
│   │   ├── rms_norm.py
│   │   └── feedforward.py         # GLU feedforward network
│   └── TS_Language_Integrator.py  # NEW: Cross-attention using fidel-ts embedder
│
├── models/
│   └── LeRet.py                   # NEW: Main model class
│
├── model_configs/
│   └── general/
│       └── LeRet.yaml             # NEW: Default hyperparameters
│
├── exp/
│   └── exp_leret.py               # NEW: LeRet-specific experiment runner
│
├── cli/
│   └── config/
│       └── models.py              # UPDATE: Add LeRetConfig for two-stage training
│
└── data_provider/
    └── data_loader.py             # UPDATE: Add auto-regression target (y_auto)
```

### 3.2 Implementation Tasks

#### Phase 1: Core RetNet Components (Priority: HIGH)

| Task | File | Description | Complexity |
|------|------|-------------|------------|
| 1.1 | `layers/RetNet/config.py` | Simplified `RetNetConfig` (no MoE) | Low |
| 1.2 | `layers/RetNet/rms_norm.py` | RMSNorm layer | Low |
| 1.3 | `layers/RetNet/multiscale_retention.py` | Parallel-mode retention with extension docs | Medium |
| 1.4 | `layers/RetNet/feedforward.py` | GLU feedforward network | Low |
| 1.5 | `layers/RetNet/retnet.py` | Simplified RetNetDecoder (no MoE, no chunk) | Medium |

**Simplifications from Original:**
- ❌ Remove `moe_freq`, `moe_expert_count`, `MOELayer` (not needed)
- ❌ Remove `chunkwise_recurrent`, `recurrent_chunk_size` (parallel only, documented)
- ❌ Remove `fairscale` imports (checkpoint_wrapper, wrap)
- ❌ Remove `multiway_network.py` (simplify to standard single-way)
- ✅ Keep core retention mechanism, position encoding, GLU FFN

#### Phase 2: Backbone & Utilities (Priority: HIGH)

| Task | File | Description | Complexity |
|------|------|-------------|------------|
| 2.1 | `layers/LeRet_layers.py` | Utility functions (series_decomp, positional encoding) | Low |
| 2.2 | `layers/LeRet_backbone.py` | LeRetEncoder, Patch_Level_Head, Flatten_Head | Medium |

**Notes:**
- Reuse existing `layers/RevIN.py` (already in fidel-ts)
- Patching logic similar to PatchTST

#### Phase 3: Language Integration via fidel-ts Embedder (Priority: MEDIUM)

| Task | File | Description | Complexity |
|------|------|-------------|------------|
| 3.1 | `layers/TS_Language_Integrator.py` | CrossEncoder using `embedder.TextEmbedder` | Medium |
| 3.2 | Integration with `embedder/` | Use existing `FidelTSEmbeddingLoader` | Low |

**Key Integration Points:**
```python
# In TS_Language_Integrator.py
from embedder import TextEmbedder

class TSLanguageIntegrator(nn.Module):
    """
    Integrates language knowledge into time series representations.
    
    Uses fidel-ts centralized embedder infrastructure for text embeddings,
    supporting multiple embedding models (BERT, etc.) with caching.
    """
    def __init__(self, 
                 d_model: int,
                 n_heads: int,
                 embed_model_name: str = 'bert-base-uncased',
                 aggregation_method: str = 'cls',
                 ...):
        # Language embedding dimension from embedder
        self.embed_model_name = embed_model_name
        # BERT: 768, others vary
        language_embed_dim = self._get_embed_dim(embed_model_name)
        self.text_linear = nn.Linear(language_embed_dim, d_model)
        ...
```

#### Phase 4: Main Model & Config (Priority: HIGH)

| Task | File | Description | Complexity |
|------|------|-------------|------------|
| 4.1 | `models/LeRet.py` | Main Model class with forward pass | Medium |
| 4.2 | `model_configs/general/LeRet.yaml` | Hyperparameter configuration | Low |
| 4.3 | `models/__init__.py` | Register LeRet in model registry | Low |
| 4.4 | `cli/config/models.py` | Add `LeRetTrainingConfig` pydantic model | Medium |

#### Phase 5: Two-Stage Training Support (Priority: HIGH)

| Task | File | Description | Complexity |
|------|------|-------------|------------|
| 5.1 | `exp/exp_leret.py` | LeRet experiment runner with stage control | High |
| 5.2 | `cli/config/models.py` | Add `LeRetConfig` with training_stage options | Medium |
| 5.3 | `data_provider/data_loader.py` | Add auto-regression target (y_auto) | Low |

### 3.3 Configuration Schema

#### 3.3.1 Model Config (`model_configs/general/LeRet.yaml`)

```yaml
# model_configs/general/LeRet.yaml
model_type: LeRet

# Architecture
d_model: 128
n_heads: 8
e_layers: 3
d_ff: 256

# Patching (similar to PatchTST)
patch_len: 16
stride: 8

# RevIN
revin: true
affine: false
subtract_last: false

# Decomposition (optional)
decomposition: false
kernel_size: 25

# Heads
individual: false
head_dropout: 0.0
fc_dropout: 0.05

# General
dropout: 0.05

# Language Integration (uses fidel-ts embedder)
use_language: true
language:
  embed_model_name: "bert-base-uncased"  # Model from embedder registry
  aggregation_method: "cls"              # cls, average, none
  # language_embed_dim auto-detected from model (768 for BERT)
```

#### 3.3.2 Training Config (Experiment-level)

```yaml
# In experiment config (configs/experiments/leret_example.yaml)
training:
  epochs: 100           # Finetune epochs (Stage 2)
  batch_size: 32
  learning_rate: 0.0001
  
  # LeRet-specific two-stage training
  leret:
    training_stage: "both"    # "pretrain" | "finetune" | "both"
    pretrain_epochs: 10       # Stage 1 epochs
    pretrain_loss: "mse"
    finetune_loss: "mse"
    
    # Only needed if training_stage="finetune" and running standalone
    # (auto-detected if within same experiment)
    pretrain_checkpoint: null
```

#### 3.3.3 Pydantic Config Model

```python
# In cli/config/models.py

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal

class LeRetTrainingConfig(BaseModel):
    """Configuration for LeRet two-stage training."""
    
    training_stage: Literal["pretrain", "finetune", "both"] = Field(
        default="both",
        description="Which training stage(s) to run. 'both' runs pretrain then finetune."
    )
    
    pretrain_epochs: int = Field(
        default=10,
        ge=1,
        description="Number of epochs for Stage 1 (auto-regressive pretraining)"
    )
    
    pretrain_loss: Literal["mse", "mae"] = Field(
        default="mse",
        description="Loss function for pretrain stage"
    )
    
    finetune_loss: Literal["mse", "mae"] = Field(
        default="mse",
        description="Loss function for finetune stage"
    )
    
    pretrain_checkpoint: Optional[str] = Field(
        default=None,
        description="Path to pretrain checkpoint. Required if training_stage='finetune' "
                    "and not auto-detected from same experiment."
    )
    
    @field_validator('pretrain_checkpoint')
    @classmethod
    def validate_pretrain_checkpoint(cls, v, info):
        """Validate pretrain_checkpoint is provided when needed."""
        # Validation happens at runtime in experiment runner
        # since we need experiment context
        return v


class TrainingConfig(BaseModel):
    """Extended training config with LeRet support."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.0001
    # ... other fields ...
    
    # LeRet-specific (optional, only used for LeRet model)
    leret: Optional[LeRetTrainingConfig] = None
```

### 3.4 Dependency Considerations

**Required (already in fidel-ts):**
- PyTorch
- numpy
- transformers (for embedder)

**Removed Dependencies:**
- ❌ fairscale (not needed without MoE/FSDP)

---

## 4. Code Snippets

### 4.1 Simplified MultiScaleRetention (Parallel Mode Only)

```python
class MultiScaleRetention(nn.Module):
    """
    Multi-Scale Retention mechanism (Parallel Mode).
    
    Replaces self-attention with O(n) complexity retention.
    Currently implements parallel mode only for training efficiency.
    
    Future Extension: Chunkwise recurrent mode can be added for long sequences.
    See original implementation for reference:
    - layers/torchscalefly/component/multiscale_retention.py:114-165
    - Requires: chunkwise_recurrent config flag, chunk_recurrent_forward() method,
      updated RetNetRelPos for chunk-based decay masks.
    """
    def __init__(self, embed_dim: int, value_dim: int, num_heads: int,
                 gate_fn: str = "swish", layernorm_eps: float = 1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = value_dim // num_heads
        self.key_dim = embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        
        # Activation for gating
        self.gate_fn = F.silu if gate_fn == "swish" else F.gelu
        
        # Projections (no bias, following original)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, value_dim, bias=False)
        self.g_proj = nn.Linear(embed_dim, value_dim, bias=False)
        self.out_proj = nn.Linear(value_dim, embed_dim, bias=False)
        
        # Group normalization per head
        self.group_norm = RMSNorm(self.head_dim, eps=layernorm_eps)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters with scaled Xavier."""
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(self, x: Tensor, rel_pos: Tuple) -> Tensor:
        """
        Forward pass using parallel retention.
        
        Args:
            x: [batch, seq_len, embed_dim] - input sequence
            rel_pos: ((sin, cos), decay_mask) from RetNetRelPos
        
        Returns:
            output: [batch, seq_len, embed_dim] - retained sequence
        
        Note:
            This implementation uses parallel mode only. For recurrent or
            chunk-recurrent modes (useful for very long sequences or inference),
            see the original TorchScale implementation.
        """
        bsz, tgt_len, _ = x.size()
        (sin, cos), mask = rel_pos
        
        # Project Q, K, V, G
        q = self.q_proj(x)
        k = self.k_proj(x) * self.scaling
        v = self.v_proj(x)
        g = self.g_proj(x)
        
        # Reshape for multi-head: [B, L, H, D] -> [B, H, L, D]
        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        
        # Apply rotary position encoding (theta shift)
        qr = self._theta_shift(q, sin, cos)
        kr = self._theta_shift(k, sin, cos)
        
        # Parallel retention computation
        output = self._parallel_retention(qr, kr, v, mask)
        
        # Group norm + gating
        output = self.group_norm(output).reshape(bsz, tgt_len, -1)
        output = self.gate_fn(g) * output
        
        return self.out_proj(output)
    
    def _theta_shift(self, x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
        """Apply rotary position encoding."""
        return (x * cos) + (self._rotate_every_two(x) * sin)
    
    @staticmethod
    def _rotate_every_two(x: Tensor) -> Tensor:
        """Rotate pairs of dimensions for rotary encoding."""
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)
    
    def _parallel_retention(self, qr: Tensor, kr: Tensor, v: Tensor, 
                            mask: Tensor) -> Tensor:
        """
        Compute retention in parallel mode.
        
        This is equivalent to attention but with exponential decay mask
        instead of causal mask.
        
        Args:
            qr: [B, H, L, D_k] - rotary-encoded queries
            kr: [B, H, L, D_k] - rotary-encoded keys
            v: [B, L, D_v] - values
            mask: [H, L, L] - exponential decay mask
        
        Returns:
            output: [B, L, H, D_h] - retained values
        """
        bsz, tgt_len, _ = v.size()
        vr = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # QK^T with decay mask: [B, H, L, L]
        qk_mat = qr @ kr.transpose(-1, -2) * mask
        
        # Normalize for stability
        qk_mat = qk_mat / qk_mat.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1)
        
        # Apply to values: [B, H, L, D_h] -> [B, L, H, D_h]
        output = torch.matmul(qk_mat, vr).transpose(1, 2)
        return output
    
    # =========================================================================
    # FUTURE EXTENSION: Chunkwise Recurrent Mode
    # =========================================================================
    # def chunk_recurrent_forward(self, qr, kr, v, inner_mask):
    #     """
    #     Chunk-recurrent retention for long sequences.
    #     
    #     Divides sequence into chunks, computes retention within chunks,
    #     and propagates state between chunks for O(L/C) memory where C is chunk size.
    #     
    #     To enable:
    #     1. Add chunkwise_recurrent and recurrent_chunk_size to config
    #     2. Update RetNetRelPos to compute chunk-based masks
    #     3. Implement this method following original:
    #        layers/torchscalefly/component/multiscale_retention.py:114-165
    #     """
    #     raise NotImplementedError(
    #         "Chunkwise recurrent mode not yet implemented. "
    #         "See docstring for extension instructions."
    #     )
```

### 4.2 Model Forward Pass

```python
class Model(nn.Module):
    """
    LeRet: Language-Enhanced Retention Network for Time Series.
    
    A time series forecasting model that combines:
    - RetNet backbone for O(n) complexity sequence modeling
    - Patch-based input processing
    - Language knowledge integration via fidel-ts embedder
    - Two-stage training (auto-regressive pretrain + forecasting finetune)
    """
    
    def forward(self, x: Tensor, language_embeddings: Optional[Tensor] = None
               ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through LeRet model.
        
        Args:
            x: [Batch, Seq_Len, Channels] - input time series
            language_embeddings: [text_num, embed_dim] - optional language embeddings
                                 from fidel-ts embedder. If None, skips language integration.
        
        Returns:
            output: [Batch, Pred_Len, Channels] - forecasting prediction (Stage 2)
            auto_y: [Batch, Seq_Len, Channels] - auto-regressive output (Stage 1)
        """
        # 1. Transpose for channel-independent processing
        x = x.permute(0, 2, 1)  # [B, C, S]
        
        # 2. RevIN normalization
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)
        
        # 3. Patching: unfold into patches
        x = x.unfold(-1, self.patch_len, self.stride)  # [B, C, P, L]
        x = x.permute(0, 1, 3, 2)  # [B, C, L, P]
        
        # 4. Backbone encoding (RetNet)
        h = self.backbone(x)  # [B, C, D, P]
        
        # 5. Auto-regression head (for Stage 1 pretraining)
        auto_y = self.patch_head(h)  # [B, C, P*L]
        
        # 6. Language integration (if embeddings provided)
        if self.use_language and language_embeddings is not None:
            z = self.ts_language_integrator(h, language_embeddings)  # [B, C, D, P]
        else:
            z = h
        
        # 7. Sequence prediction head (for Stage 2 finetuning)
        z = self.sequence_head(z)  # [B, C, Pred_Len]
        
        # 8. RevIN denormalization
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        
        # 9. Transpose back to standard format
        output = z.permute(0, 2, 1)  # [B, Pred_Len, C]
        auto_y = auto_y.permute(0, 2, 1)  # [B, P*L, C]
        
        return output, auto_y
```

---

## 5. Testing & Validation

### 5.1 Unit Tests

| Test | Description |
|------|-------------|
| `test_retnet_forward` | Verify RetNet encoder produces correct output shapes |
| `test_multiscale_retention_parallel` | Test parallel retention mode |
| `test_language_integrator_with_embedder` | Verify integration with `TextEmbedder` |
| `test_end_to_end` | Full model forward pass with dummy data |
| `test_two_stage_config_validation` | Validate LeRetTrainingConfig constraints |

### 5.2 Integration Tests

| Test | Description |
|------|-------------|
| `test_lightning_training_both_stages` | Training loop with `training_stage="both"` |
| `test_pretrain_only` | Run pretrain stage, verify checkpoint saved |
| `test_finetune_from_pretrain` | Load pretrain checkpoint, run finetune |
| `test_experiment_id_validation` | Verify finetune rejects mismatched experiment_id |
| `test_job_history_two_stage` | Verify job_history tracks both stages |

### 5.3 Benchmarks

Run on standard datasets to validate implementation:
- ETTh1, ETTh2, ETTm1, ETTm2
- Weather
- Electricity

---

## 6. Timeline Estimate

| Phase | Tasks | Duration | Dependencies |
|-------|-------|----------|--------------|
| Phase 1 | RetNet Components (simplified) | 2-3 days | None |
| Phase 2 | Backbone & Utilities | 2 days | Phase 1 |
| Phase 3 | Language Integration (fidel-ts embedder) | 1-2 days | Phase 2 |
| Phase 4 | Main Model & Config | 1-2 days | Phase 2, 3 |
| Phase 5 | Two-Stage Training Support | 2-3 days | Phase 4 |
| Testing | Unit + Integration | 2-3 days | Phase 5 |

**Total Estimate: 10-15 days**

---

## 7. References

1. **RetNet Paper**: "Retentive Network: A Successor to Transformer for Large Language Models" (Sun et al., 2023)
2. **PatchTST**: "A Time Series is Worth 64 Words" (Nie et al., 2023)
3. **RevIN**: "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift" (Kim et al., 2022)
4. **fidel-ts Embedder**: See `embedder/__init__.py` for centralized text embedding infrastructure

