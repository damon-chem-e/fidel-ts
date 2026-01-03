# Time-LLM Implementation Plan

> **Paper**: Time-LLM: Time Series Forecasting by Reprogramming Large Language Models (ICLR 2024)
> **Authors**: Jin et al.
> **Reference Implementation**: `benchmark_models/Time-LLM/`

---

## Executive Summary

Time-LLM is a **reprogramming framework** that converts time series into LLM-compatible token representations during the forward pass. This is fundamentally different from the precomputed embedding approach in the existing `fidel-ts` codebase.

**Key Insight**: Time-LLM and fidel-ts use LLMs differently:
- **fidel-ts**: Precompute embeddings offline via prompts
- **Time-LLM**: Reprogram patches into LLM space during training with dynamic prompts

Both approaches can coexist without conflict, sharing only the low-level LLM loading infrastructure.

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

## 2. Overlap Analysis with Existing Code

### 2.1 Components to REUSE (No Duplication)

| fidel-ts Component | Time-LLM Equivalent | Action |
|--------------------|---------------------|--------|
| `embedder/llm_utils.py` (LLMRegistry) | LLM loading in TimeLLM.py | **REUSE** |
| `data_provider/` | Nearly identical structure | **REUSE** |
| `utils/timefeatures.py` | Identical functionality | **REUSE** |

### 2.2 Components That Are DIFFERENT (Must Implement)

| Time-LLM Component | Difference from fidel-ts | Action |
|--------------------|--------------------------|--------|
| **ReprogrammingLayer** | Novel cross-attention mechanism | **NEW** |
| **PatchEmbedding** | Conv1D patching (not text prompts) | **NEW** |
| **Dynamic prompt generation** | On-the-fly stats in forward pass | **NEW** |
| **FlattenHead** | Specific output projection | **NEW** |
| **Normalize (RevIN)** | Instance normalization | **NEW** |

### 2.3 Components to KEEP SEPARATE

| Existing Component | Reason |
|--------------------|--------|
| `embedder/prompt_builder.py` | Different use case - precomputation vs online |
| `embedder/llm_embedder.py` | Different workflow - offline vs training-time |

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

class Normalize(nn.Module):
    """Reversible Instance Normalization."""
    
    def __init__(self, num_features: int, eps=1e-5, affine=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
    def forward(self, x, mode: str):
        """
        Args:
            x: Input tensor [B, T, C]
            mode: 'norm' or 'denorm'
        """
        if mode == 'norm':
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
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

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))
        self.value_embedding = TokenEmbedding(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, T]
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

The KEY innovation of Time-LLM:
- Query: embedded time series patches [B, num_patches, d_model]
- Key/Value: LLM word embeddings [vocab_size, d_llm] → mapped to [num_tokens, d_llm]
- Output: Reprogrammed embeddings in LLM space [B, num_patches, d_llm]
"""

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        """
        Args:
            target_embedding: [B, L, d_model] - TS patch embeddings
            source_embedding: [S, d_llm] - Word embeddings (mapped)
            value_embedding: [S, d_llm] - Word embeddings (mapped)
        Returns:
            Reprogrammed embeddings [B, L, d_llm]
        """
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

IMPORTANT: This is DIFFERENT from embedder/prompt_builder.py!

- TSPromptBuilder (existing): Precomputes prompts for offline embedding
- DynamicPromptBuilder (this): Generates prompts ON-THE-FLY in forward pass

Key differences:
1. Computes statistics (min/max/median/lags/trend) per forward pass
2. Includes prediction task description
3. Used during training, not precomputation
"""

class DynamicPromptBuilder:
    """Generate prompts with time series statistics during inference."""
    
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
            x_enc: [B*N, T, 1] - flattened time series
            description: Dataset description string
            pred_len: Prediction length
            seq_len: Input sequence length
            lags: [B*N, top_k] - top-k autocorrelation lags
        
        Returns:
            List of prompt strings, one per sample
        """
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompts = []
        for b in range(x_enc.shape[0]):
            prompt = (
                f"<|start_prompt|>Dataset description: {description}"
                f"Task description: forecast the next {pred_len} steps "
                f"given the previous {seq_len} steps information; "
                "Input statistics: "
                f"min value {min_values[b].item():.4f}, "
                f"max value {max_values[b].item():.4f}, "
                f"median value {medians[b].item():.4f}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags[b].tolist()}<|<end_prompt>|>"
            )
            prompts.append(prompt)
        return prompts
    
    @staticmethod
    def calculate_lags(x_enc: torch.Tensor, top_k: int = 5) -> torch.Tensor:
        """Compute top-k autocorrelation lags via FFT."""
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, top_k, dim=-1)
        return lags
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
"""

from embedder.llm_utils import LLMRegistry  # REUSE existing

class TimeLLM(nn.Module):
    def __init__(
        self,
        llm_model: str = 'LLAMA',
        llm_dim: int = 4096,
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
    ):
        super().__init__()
        
        # Store config
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_ff = d_ff
        self.d_llm = llm_dim
        self.patch_len = patch_len
        self.stride = stride
        self.top_k = 5
        
        # Load LLM using existing registry (REUSE)
        self._load_llm(llm_model, llm_dim, llm_layers, cache_dir, device)
        
        # Freeze LLM weights
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
        # Dataset description for prompts
        self.description = dataset_description if prompt_domain else (
            "The Electricity Transformer Temperature (ETT) is a crucial "
            "indicator in the electric power long-term deployment."
        )
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, dropout)
        
        # Word embedding mapping
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        
        # Reprogramming layer
        self.reprogramming_layer = ReprogrammingLayer(d_model, n_heads, d_ff, llm_dim)
        
        # Output projection
        self.patch_nums = int((seq_len - patch_len) / stride + 2)
        self.head_nf = d_ff * self.patch_nums
        self.output_projection = FlattenHead(enc_in, self.head_nf, pred_len, dropout)
        
        # Normalization
        self.normalize_layers = Normalize(enc_in, affine=False)
        
        self.dropout = nn.Dropout(dropout)

    def _load_llm(self, llm_model, llm_dim, llm_layers, cache_dir, device):
        """Load LLM model using existing LLMRegistry."""
        # Map Time-LLM names to HuggingFace names
        model_map = {
            'LLAMA': 'huggyllama/llama-7b',
            'GPT2': 'openai-community/gpt2',
            'BERT': 'google-bert/bert-base-uncased',
        }
        model_name = model_map.get(llm_model, llm_model)
        
        # Use existing registry
        self.llm_model, _ = LLMRegistry.get_model(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
        )
        self.tokenizer = LLMRegistry.get_tokenizer(
            model_name=model_name,
            cache_dir=cache_dir,
        )
        
        # Configure tokenizer padding
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass.
        
        Args:
            x_enc: [B, T, C] - input time series
            x_mark_enc: [B, T, F] - time features (optional)
            x_dec: decoder input (not used)
            x_mark_dec: decoder time features (not used)
        """
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalize input
        x_enc = self.normalize_layers(x_enc, 'norm')
        
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        
        # Calculate statistics for dynamic prompts
        lags = DynamicPromptBuilder.calculate_lags(x_enc, self.top_k)
        prompts = DynamicPromptBuilder.build_prompts(
            x_enc, self.description, self.pred_len, self.seq_len, lags
        )
        
        # Reshape back
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        
        # Tokenize prompts
        prompt_tokens = self.tokenizer(
            prompts, return_tensors="pt", padding=True, 
            truncation=True, max_length=2048
        ).input_ids.to(x_enc.device)
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_tokens)
        
        # Map word embeddings to reduced vocabulary
        source_embeddings = self.mapping_layer(
            self.word_embeddings.permute(1, 0)
        ).permute(1, 0)
        
        # Patch embedding
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        
        # Reprogramming
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        
        # Concatenate with prompt embeddings and pass through LLM
        llm_input = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llm_input).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        
        # Reshape and project to output
        dec_out = dec_out.reshape(-1, n_vars, dec_out.shape[-2], dec_out.shape[-1])
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        
        # Denormalize
        dec_out = self.normalize_layers(dec_out, 'denorm')
        
        return dec_out
```

### Phase 3: Training Infrastructure

#### 3.6 Create `training/time_llm_trainer.py`

```python
"""
Training loop for Time-LLM.

Design decisions:
- Only train non-frozen parameters
- Use Accelerate/DeepSpeed for multi-GPU training
- Reuse existing EarlyStopping and LR scheduling patterns
"""

class TimeLLMTrainer:
    def __init__(self, model, config, accelerator):
        self.model = model
        self.config = config
        self.accelerator = accelerator
        
        # Only optimize non-frozen parameters
        self.trained_parameters = [
            p for p in model.parameters() if p.requires_grad
        ]
        self.optimizer = torch.optim.Adam(
            self.trained_parameters, 
            lr=config.learning_rate
        )
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = []
        
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
            
            # Compute loss
            loss = F.mse_loss(outputs[:, -self.config.pred_len:], 
                             batch_y[:, -self.config.pred_len:])
            
            # Backward pass
            self.accelerator.backward(loss)
            self.optimizer.step()
            
            total_loss.append(loss.item())
        
        return np.mean(total_loss)
```

---

## 4. File Structure

```
fidel-ts-worktree-lynx/
├── models/
│   └── time_llm/
│       ├── __init__.py
│       ├── time_llm_model.py      # Main model class
│       ├── reprogramming.py       # ReprogrammingLayer
│       ├── patch_embed.py         # PatchEmbedding
│       ├── normalization.py       # Normalize (RevIN)
│       └── dynamic_prompt.py      # DynamicPromptBuilder
├── training/
│   └── time_llm_trainer.py        # Training loop
├── model_configs/
│   └── time_llm/
│       ├── default.yaml
│       ├── llama_7b.yaml
│       └── gpt2.yaml
└── scripts/
    └── run_time_llm.py            # CLI entry point
```

---

## 5. Configuration Schema

```yaml
# model_configs/time_llm/default.yaml
model:
  name: "TimeLLM"
  task_name: "long_term_forecast"
  
llm:
  backbone: "LLAMA"  # or "GPT2", "BERT"
  model_path: "huggyllama/llama-7b"
  hidden_dim: 4096  # LLAMA:4096, GPT2:768, BERT:768
  num_layers: 6     # Number of LLM layers to use
  freeze: true      # Always freeze LLM weights

patching:
  patch_len: 16
  stride: 8

model_dims:
  d_model: 32       # Patch embedding dim
  d_ff: 128         # Feed-forward dim
  n_heads: 8        # Attention heads for reprogramming
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
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 1: Core Components (Independent - No Dependencies)           │
├─────────────────────────────────────────────────────────────────────┤
│ 1a. normalization.py (Normalize class)                              │
│ 1b. patch_embed.py (PatchEmbedding, TokenEmbedding)                 │
│ 1c. reprogramming.py (ReprogrammingLayer)                           │
│ 1d. dynamic_prompt.py (DynamicPromptBuilder)                        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 2: Main Model (Depends on Phase 1 + LLMRegistry)              │
├─────────────────────────────────────────────────────────────────────┤
│ 2a. time_llm_model.py (TimeLLM class)                               │
│     - Uses: All Phase 1 components                                  │
│     - Uses: embedder/llm_utils.py (LLMRegistry) - EXISTING          │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 3: Training & CLI (Depends on Phase 2)                        │
├─────────────────────────────────────────────────────────────────────┤
│ 3a. time_llm_trainer.py                                             │
│ 3b. model_configs/time_llm/*.yaml                                   │
│ 3c. scripts/run_time_llm.py                                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

```python
# tests/test_time_llm_components.py

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

def test_time_llm_forward():
    """Test end-to-end forward pass with GPT-2 (smaller model)."""
    model = TimeLLM(
        llm_model='GPT2',
        llm_dim=768,
        llm_layers=6,
        seq_len=96,
        pred_len=96,
        enc_in=7,
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
    )
    
    x = torch.randn(2, 96, 7)
    out = model(x, None, None, None)
    
    assert out.shape == (2, 96, 7)
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

| LLM Model | Hidden Dimension | HuggingFace Name |
|-----------|------------------|------------------|
| LLaMA-7B | 4096 | `huggyllama/llama-7b` |
| GPT-2 | 768 | `openai-community/gpt2` |
| BERT-base | 768 | `google-bert/bert-base-uncased` |
| Qwen2.5-7B | 3584 | `Qwen/Qwen2.5-7B-Instruct` |
| Qwen2.5-72B | 8192 | `Qwen/Qwen2.5-72B-Instruct` |

---

## 9. Open Questions / Future Work

1. **Memory Optimization**: Can we use gradient checkpointing for the frozen LLM?
2. **Quantization**: Does 4-bit/8-bit quantization of frozen LLM affect reprogramming quality?
3. **Multi-modal Extension**: Can we combine Time-LLM's online reprogramming with fidel-ts's precomputed text embeddings?
4. **Channel Independence**: Should we share reprogramming across channels or keep separate?

---

## 10. References

- **Paper**: [Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://arxiv.org/abs/2310.01728)
- **Original Code**: `benchmark_models/Time-LLM/`
- **Related Work**: TimeCMA (uses precomputed embeddings instead of reprogramming)

