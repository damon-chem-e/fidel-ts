# Embedder Module Documentation

The embedder module provides centralized text and LLM embedding functionality for Fidel-TS. It supports two distinct embedding paradigms with different input sources and use cases.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Types](#dataset-types)
3. [Directory Structure](#directory-structure)
4. [Text Embeddings (BERT-style)](#text-embeddings-bert-style)
5. [LLM Embeddings (GPT-style)](#llm-embeddings-gpt-style)
6. [LLM Input Sources & Extensibility](#llm-input-sources--extensibility)
7. [Comparison: Text vs LLM Embeddings](#comparison-text-vs-llm-embeddings)
8. [Configuration](#configuration)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

### Two Embedding Paradigms

The embedder module supports two fundamentally different approaches to generating embeddings:

| Paradigm | Model Type | Input | Output | Use Case |
|----------|------------|-------|--------|----------|
| **Text Embeddings** | Encoder (BERT) | Pre-existing text | CLS/pooled token | Multimodal forecasting |
| **LLM Embeddings** | Decoder (GPT-2, Qwen) | Any text (prompts, raw) | Hidden state | TimeCMA, future models |

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Fidel-TS Embedder Module                         │
├────────────────────────────────────┬────────────────────────────────────────┤
│     Text Embeddings (BERT-style)   │      LLM Embeddings (GPT-style)        │
├────────────────────────────────────┼────────────────────────────────────────┤
│                                    │                                        │
│  Input: Pre-existing text          │  Input: ANY TEXT SOURCE                │
│  (weather reports, news)           │    ├── Time series → Prompts           │
│                                    │    ├── Raw text (future)               │
│  ┌──────────────────────────────┐  │    └── Custom adapters (future)        │
│  │  EmbeddingModelRegistry      │  │                                        │
│  │  (BERT, RoBERTa, etc.)       │  │  ┌──────────────────────────────────┐  │
│  └──────────────┬───────────────┘  │  │     Input Adapters               │  │
│                 │                  │  │  ┌─────────────────────────────┐ │  │
│  ┌──────────────▼───────────────┐  │  │  │ TSPromptBuilder (TS→Text) │ │  │
│  │  TextEmbedder                │  │  │  │ Raw Text (direct)          │ │  │
│  │  - embed_texts()             │  │  │  │ Custom Adapters (future)   │ │  │
│  │  - embed_text_dict()         │  │  │  └─────────────────────────────┘ │  │
│  └──────────────┬───────────────┘  │  └──────────────┬───────────────────┘  │
│                 │                  │                 │                      │
│                 │                  │  ┌──────────────▼───────────────────┐  │
│                 │                  │  │  LLMEmbedder                     │  │
│                 │                  │  │  - embed_texts() [CORE]          │  │
│                 │                  │  │  - generate_ts_embeddings()      │  │
│                 │                  │  └──────────────┬───────────────────┘  │
│                 │                  │                 │                      │
│  ┌──────────────▼───────────────┐  │  ┌──────────────▼───────────────────┐  │
│  │  EmbeddingCacheManager       │  │  │  LLMEmbeddingCache               │  │
│  │  embeddings_{hash}/          │  │  │  llm_{hash}/                     │  │
│  └──────────────────────────────┘  │  └──────────────────────────────────┘  │
│                                    │                                        │
│  Training: On-the-fly or cached    │  Training: MUST precompute             │
│                                    │                                        │
└────────────────────────────────────┴────────────────────────────────────────┘
```

### Key Insight: LLMEmbedder is Text-Agnostic

The `LLMEmbedder` operates on **text strings** - it doesn't care where they come from:

```python
# These all use the same core embed_texts() method:

# 1. Time series → prompts (current, via TSPromptBuilder)
embedder.generate_ts_embeddings(dataset='ETTh1', split='train')

# 2. Raw text (future)
embedder.embed_texts(["Weather report...", "News article..."])

# 3. Custom format (future)
embedder.embed_texts(my_custom_adapter.to_text(data))
```

---

## Dataset Types

Fidel-TS works with two types of datasets that have different directory structures:

### Time-MMD Datasets

Standard time series datasets with simple structure.

**Examples:** ETTh1, ETTm1, Weather, Electricity, Traffic

**Structure:**
```
data/
└── ETTh1/
    ├── ETTh1.csv              # Raw time series data
    ├── embeddings_cache/      # Text embeddings (if using hetero)
    │   └── embeddings_{hash}/
    └── llm_embeddings/        # LLM embeddings
        └── llm_{hash}/
            ├── metadata.json
            ├── train/
            ├── val/
            └── test/
```

**Characteristics:**
- Simple CSV with timestamp + features
- Standard data loading via `data_provider`
- Embeddings indexed by sample index

### Fidel-TS Datasets

Complex multimodal datasets with nested directory structures.

**Examples:** Bear_room, California_ISO, Canada_photovoltaics_plants, Germany_Renewable_Power_Grid, Jena_Atmospheric_Physics, NYC_traffic_speed

**Structure:**
```
data/
└── Bear_room/
    ├── raw_data/                              # Time series
    │   └── ...
    ├── hetero/                                # Text data sources
    │   ├── weather/
    │   │   ├── weather_report/
    │   │   │   └── formal_report/
    │   │   │       └── wm_messages_v3.json    # Raw text
    │   │   └── report_embedding/
    │   │       └── formal_report/
    │   │           ├── wm_messages_v3.pkl     # Legacy BERT embeddings
    │   │           └── static_embeddings.pkl
    │   └── room/
    │       └── ...
    ├── embeddings_cache/                      # NEW: Hash-based BERT cache
    │   └── weather/
    │       └── report_embedding/
    │           └── formal_report/
    │               └── embeddings_{hash}/
    ├── llm_embeddings/                        # NEW: LLM embeddings
    │   └── llm_{hash}/
    │       ├── metadata.json
    │       └── train/, val/, test/
    └── static_info.json                       # Channel descriptions
```

**Characteristics:**
- Complex nested directories per subdataset
- Hardcoded path resolution via `FidelTSPathResolver`
- Each subdataset has unique structure (year files, source types, etc.)
- Separate text embeddings in `hetero/` and LLM embeddings in `llm_embeddings/`

### Path Resolution for Fidel-TS

The `FidelTSPathResolver` handles the complex path logic:

```python
from embedder import FidelTSPathResolver

resolver = FidelTSPathResolver(
    dataset_name='Bear_room',
    hetero_info=config['hetero_info'],
    base_data_path='./data/'
)

paths = resolver.resolve_paths()
# Returns:
# {
#     'old_embedding_path': Path to legacy .pkl
#     'old_text_path': Path to raw text JSON
#     'cache_base': Path for new embeddings cache
#     'source_type': 'weather' or 'room'
# }
```

---

## Directory Structure

### Module Files

```
embedder/
├── __init__.py                 # Module exports
│
├── # Text Embeddings (BERT-style)
├── registry.py                 # EmbeddingModelRegistry (singleton)
├── embedder.py                 # TextEmbedder (main interface)
├── aggregation.py              # CLS, average, none methods
├── metadata.py                 # EmbeddingMetadata class
├── cache_manager.py            # EmbeddingCacheManager
├── fidel_ts_path_resolver.py   # Fidel-TS specific path resolution
├── fidel_ts_embedder.py        # FidelTSEmbeddingLoader
│
├── # LLM Embeddings (GPT-style)
├── llm_registry.py             # LLMRegistry (singleton)
├── llm_utils.py                # Memory estimation, quantization
├── prompt_builder.py           # Time series → prompt conversion
├── llm_cache.py                # LLMEmbeddingCache
└── llm_embedder.py             # LLMEmbedder (main interface)
```

### Cache Directory Comparison

| Cache Type | Location | Format | Indexed By |
|------------|----------|--------|------------|
| Text (BERT) | `embeddings_cache/embeddings_{hash}/` | `.pkl` | Timestamp (string) |
| LLM (GPT) | `llm_embeddings/llm_{hash}/` | `.h5` | Sample index (int) |

These caches are **independent** and can coexist for the same dataset.

---

## Text Embeddings (BERT-style)

Text embeddings convert pre-existing text (weather reports, news articles, channel descriptions) into dense vector representations using encoder models like BERT.

### Components

#### 1. EmbeddingModelRegistry (`registry.py`)

Singleton registry for sharing BERT models.

```python
from embedder import EmbeddingModelRegistry

model = EmbeddingModelRegistry.get_model(
    model_name='bert-base-uncased',
    device='cuda:0',
    hf_cache_dir='./HF_cache/'
)
```

#### 2. TextEmbedder (`embedder.py`)

Main interface for text embedding.

```python
from embedder import TextEmbedder

embedder = TextEmbedder(
    model_name='bert-base-uncased',
    aggregation_method='cls',      # 'cls', 'average', or 'none'
    device='cuda:0'
)

# Embed list of texts
embeddings = embedder.embed_texts(texts, text_keys=timestamps)

# Embed dictionary (timestamp → text)
embeddings_dict = embedder.embed_text_dict(text_dict)
```

#### 3. FidelTSEmbeddingLoader (`fidel_ts_embedder.py`)

Specialized loader for Fidel-TS datasets with hardcoded path resolution.

```python
from embedder import FidelTSEmbeddingLoader

loader = FidelTSEmbeddingLoader(
    dataset_name='Bear_room',
    hetero_info=config['hetero_info'],
    base_data_path='./data/',
    use_old_embeddings=False  # Use new cache system
)

dynamic_emb, static_emb = loader.load_embeddings()
```

#### 4. Aggregation Methods

| Method | Output Shape | Description |
|--------|--------------|-------------|
| `cls` | `[B, hidden_dim]` | First token (CLS) embedding |
| `average` | `[B, hidden_dim]` | Mean pooling over tokens |
| `none` | `[B, seq_len, hidden_dim]` | Full sequence |

---

## LLM Embeddings (GPT-style)

LLM embeddings extract hidden states from decoder models. The key difference from text embeddings is that LLMs can process **any text input**, not just pre-existing documents.

### ⚠️ Critical: Precomputation Required

**LLM embeddings must be precomputed before training.** No on-the-fly inference.

```bash
# Precompute before training
python -m cli.inference generate ETTh1 --model gpt2

# Then train
python -m cli.train run experiment.yaml
```

### Components

#### 1. LLMRegistry (`llm_registry.py`)

Singleton registry for LLM models with quantization support.

```python
from embedder import LLMRegistry

# Standard model
model, embed_dim = LLMRegistry.get_model(
    model_name='gpt2',
    device='cuda:0',
    cache_dir='./LLM_cache/'
)

# Large model with quantization
model, embed_dim = LLMRegistry.get_model(
    model_name='Qwen/Qwen2.5-72B-Instruct',
    device='cuda:0',
    cache_dir='./LLM_cache/',
    quantization='4bit'
)
```

#### 2. LLMEmbedder (`llm_embedder.py`)

Main interface for LLM embedding. Supports multiple input sources.

```python
from embedder import LLMEmbedder

# For time series (with TSPromptBuilder)
embedder = LLMEmbedder(
    model_name='gpt2',
    prompt_template='timecma_v1'
)
embeddings = embedder.generate_ts_embeddings('ETTh1', 'train')

# For raw text (without TSPromptBuilder)
embedder = LLMEmbedder.for_raw_text(model_name='gpt2')
embeddings = embedder.embed_texts(["Text 1", "Text 2"])
```

#### 3. TSPromptBuilder (`prompt_builder.py`)

Converts time series to text prompts. **One of several possible input adapters.**

```python
from embedder import TSPromptBuilder

builder = TSPromptBuilder(template_name='timecma_v1')
prompts = builder.build_flat_prompts(values, timestamps, metadata)
# Result: "From 01/01 00:00 to 01/04 23:00, the values were 1, 3, 5, ... every hour."
```

#### 4. LLMEmbeddingCache (`llm_cache.py`)

Hash-based cache with full traceability.

```python
from embedder import LLMEmbeddingCache, LLMEmbeddingMetadata

cache = LLMEmbeddingCache('./data/', 'ETTh1')
embeddings = cache.load_embeddings(metadata, 'train')
```

### CLI Reference

```bash
# Generate embeddings
python -m cli.inference generate <dataset> [OPTIONS]
  --model, -m       Model name (e.g., 'gpt2', 'Qwen/Qwen2.5-7B-Instruct')
  --quantization    '4bit', '8bit', or None
  --splits          train,val,test
  --force           Regenerate existing cache

# Verify cache
python -m cli.inference verify <dataset>

# Estimate memory
python -m cli.inference estimate-memory <model> --quantization 4bit

# List models
python -m cli.inference list-models

# GPU info
python -m cli.inference gpu-info
```

---

## LLM Input Sources & Extensibility

### Current Implementation

The LLMEmbedder currently supports **time series → prompts** via TSPromptBuilder:

```python
# TimeCMA-style: time series values → text prompt → LLM → embedding
embedder = LLMEmbedder(prompt_template='timecma_v1')
embeddings = embedder.generate_ts_embeddings('ETTh1', 'train')
```

### Future Extensions

The architecture is designed for multiple input sources:

#### 1. Raw Text Input (Ready to Use)

Pass pre-existing text directly to LLM:

```python
# For news, weather reports, or any text
embedder = LLMEmbedder.for_raw_text(model_name='gpt2')
embeddings = embedder.embed_texts([
    "Weather forecast: Heavy rain expected...",
    "Breaking news: Market volatility increases..."
])
```

This infrastructure is **already implemented** - just call `embed_texts()` or `embed_text_dict()`.

#### 2. Integration with Fidel-TS Text Sources (Future)

The Fidel-TS datasets have raw text in `hetero/` directories. Future work:

```python
# Future: Load weather reports and embed via LLM instead of BERT
raw_texts = load_fidel_ts_text(dataset='Bear_room', source='weather')
embeddings = llm_embedder.embed_text_dict(raw_texts)
```

This would provide **LLM-quality understanding** of text data, potentially better than BERT embeddings.

#### 3. Custom Input Adapters (Future)

Create domain-specific adapters:

```python
# Future: Custom adapter for structured data
class FinancialNewsAdapter:
    def to_texts(self, data) -> List[str]:
        """Convert financial news to LLM-ready prompts."""
        return [f"Market update: {item['headline']}. {item['summary']}" 
                for item in data]

adapter = FinancialNewsAdapter()
embeddings = embedder.embed_texts(adapter.to_texts(news_data))
```

#### 4. Multimodal Combinations (Future)

Combine time series context with external text:

```python
# Future: Combined prompt with TS and external context
class MultimodalPromptBuilder:
    def format(self, ts_values, weather_text, channel_desc):
        return (
            f"Channel: {channel_desc}. "
            f"Weather: {weather_text}. "
            f"Values: {ts_values}. "
            f"Predict next values."
        )
```

### Extension Points

| Extension | Status | How to Add |
|-----------|--------|------------|
| Time series → prompts | ✅ Implemented | Use `TSPromptBuilder` |
| Raw text embedding | ✅ Ready | Call `embed_texts()` directly |
| Fidel-TS text → LLM | 🔮 Future | Create loader + call `embed_text_dict()` |
| Custom adapters | 🔮 Future | Implement adapter → call `embed_texts()` |
| Multimodal prompts | 🔮 Future | Create combined `PromptTemplate` |

---

## Comparison: Text vs LLM Embeddings

| Feature | Text Embeddings | LLM Embeddings |
|---------|-----------------|----------------|
| **Model Type** | Encoder (BERT) | Decoder (GPT-2, Qwen) |
| **Input Source** | Pre-existing text only | Any text (prompts, raw, custom) |
| **Registry** | `EmbeddingModelRegistry` | `LLMRegistry` |
| **Cache Location** | `embeddings_cache/` | `llm_embeddings/` |
| **Cache Format** | `.pkl` | `.h5` |
| **Index Type** | Timestamp (string) | Sample index (int) |
| **Training Mode** | On-the-fly or cached | **Precomputed only** |
| **Quantization** | N/A | 4-bit, 8-bit |
| **CLI** | N/A | `cli/inference.py` |
| **Extensibility** | Fixed (text only) | Flexible (any input) |

### When to Use Which

| Use Case | Recommended |
|----------|-------------|
| Weather/news text → forecasting | Text Embeddings (current) |
| TimeCMA-style time series prompts | LLM Embeddings |
| Deep text understanding (future) | LLM Embeddings |
| Large models (70B+) | LLM Embeddings (precomputed) |
| Real-time inference | Text Embeddings |

---

## Configuration

### Text Embedding Configuration

Via experiment config or direct initialization:

```yaml
hetero_info:
  embed_model_name: "bert-base-uncased"
  aggregation_method: "cls"
  device: "cuda:0"
  hf_cache_dir: "./HF_cache/"
```

### LLM Embedding Configuration

Located in `model_configs/llm_embedding/`:

```yaml
# model_configs/llm_embedding/default.yaml
llm_embedding:
  model_name: "gpt2"
  cache_dir: "./LLM_cache/"
  device: "cuda:0"
  quantization: null
  extraction_mode: "last_token"
  max_length: 512
  prompt_template: "timecma_v1"
  prompt_config:
    value_format: "integer"
    include_timestamps: true
  data_root: "./data/"
  batch_size: 32
```

---

## Best Practices

### For Time-MMD Datasets

1. Use simple data loading via `data_provider`
2. Embeddings cache in `data/{dataset}/llm_embeddings/`
3. Standard train/val/test splits

### For Fidel-TS Datasets

1. Use `FidelTSPathResolver` for path resolution
2. Don't manually construct paths - they're hardcoded per subdataset
3. Text embeddings stay in `hetero/` or `embeddings_cache/`
4. LLM embeddings go to `llm_embeddings/` (separate directory)
5. Static embeddings handled via `static_info.json`

### For LLM Embeddings

1. **Always precompute** before training
2. **Verify cache** with `cli/inference verify`
3. Start with GPT-2 for testing
4. Use quantization for large models
5. Check memory with `estimate-memory` command

### Hardware Recommendations

| Model | GPU | Quantization | Batch Size |
|-------|-----|--------------|------------|
| GPT-2 | Any ≥8GB | None | 32-64 |
| Qwen 7B | RTX 4090 | None or 4bit | 16-32 |
| Qwen 14B | L40S / A100 | None or 4bit | 8-16 |
| Qwen 72B | H200 / 2×A100 | 4bit | 4-8 |

---

## Troubleshooting

### "No cached embeddings found"

```bash
python -m cli.inference generate <dataset>
```

### "CUDA out of memory"

```bash
python -m cli.inference generate <dataset> --quantization 4bit --batch-size 8
```

### "Unknown Fidel-TS dataset"

Check that dataset name matches exactly:
- `Bear_room` (not `bear_room` or `BearRoom`)
- `California_ISO` (not `California-ISO`)

See `FidelTSPathResolver.DATASET_NAMES` for valid names.

### Different cache for same config?

Different configs → different hash → different cache. Check `metadata.json` to compare.

---

## Related Documentation

- [Job Resumption](job_resumption.md) - Resuming experiments
- [Config State](config_state.md) - Configuration management
- [TimeCMA Implementation](../future/TimeCMA_Implementation_Analysis.md) - TimeCMA plan
- [Local LLM Infrastructure](../future/local_llm.md) - LLM design doc
