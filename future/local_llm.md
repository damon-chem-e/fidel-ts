# Local LLM Infrastructure for Fidel-TS

## Overview

This document outlines a comprehensive plan for integrating local Large Language Model (LLM) hosting into fidel-ts. The system enables TimeCMA-style prompt-to-embedding workflows while maintaining efficient resource usage through a singleton registry pattern (similar to `embedder/registry.py`).

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [LLM Registry Implementation](#llm-registry-implementation)
3. [Prompt System Design](#prompt-system-design)
4. [Prompt Template Module](#prompt-template-module)
5. [Configuration System](#configuration-system)
6. [Caching Strategy](#caching-strategy)
7. [Implementation Plan](#implementation-plan)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Fidel-TS LLM Infrastructure                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────┐│
│  │  Prompt Template │     │   LLM Registry   │     │  Embedding Cache     ││
│  │      Module      │────►│   (Singleton)    │────►│    Manager           ││
│  └────────┬─────────┘     └────────┬─────────┘     └──────────────────────┘│
│           │                        │                                        │
│           ▼                        ▼                                        │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────┐│
│  │  Prompt Builder  │     │   Model Loader   │     │  H5/Pickle Storage   ││
│  │  (TS → Prompt)   │     │   (GPT2/Llama)   │     │                      ││
│  └──────────────────┘     └──────────────────┘     └──────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `LLMRegistry` | `embedder/llm_registry.py` | Singleton registry for LLM model sharing |
| `PromptBuilder` | `embedder/prompt_builder.py` | Converts time series to prompts |
| `PromptTemplate` | `prompt_templates/llm/` | Configurable prompt templates |
| `LLMEmbeddingCache` | `embedder/llm_cache.py` | Hash-based embedding storage |
| `LLMConfig` | `model_configs/llm_embedding/` | Per-model LLM configuration |

---

## LLM Registry Implementation

### Design Goals

1. **Singleton Pattern**: Only one instance of each LLM model in memory
2. **Thread-Safe**: Safe concurrent access from DataLoader workers
3. **Device-Aware**: Support multi-GPU environments
4. **Lazy Loading**: Load models only when first requested
5. **Graceful Cleanup**: Proper memory management

### Implementation: `embedder/llm_registry.py`

```python
"""
Local LLM Registry for Fidel-TS.

Singleton registry that manages LLM instances (GPT-2, LLaMA, etc.) with:
- Thread-safe access
- Local caching
- Efficient memory management (no duplicate model loading)

Modeled after embedder/registry.py for embedding models.
"""

import os
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Model


class LLMRegistry:
    """
    Singleton registry for shared LLM models and tokenizers.
    
    Similar to EmbeddingModelRegistry, but for decoder-only or
    encoder-decoder LLMs used for prompt-to-embedding workflows.
    
    Keys: (model_name, device, cache_dir, extraction_mode)
    - extraction_mode: 'last_token', 'pooled', 'all_tokens'
    """
    
    # Class-level storage
    _models: Dict[Tuple, Any] = {}
    _tokenizers: Dict[Tuple, Any] = {}
    _lock = threading.Lock()
    
    # Supported model types and their loader classes
    MODEL_LOADERS = {
        'gpt2': GPT2Model,
        'gpt2-medium': GPT2Model,
        'gpt2-large': GPT2Model,
        'gpt2-xl': GPT2Model,
        'default': AutoModelForCausalLM  # For other models (LLaMA, etc.)
    }
    
    @classmethod
    def get_model(cls, 
                  model_name: str, 
                  device: str, 
                  cache_dir: str,
                  extraction_mode: str = 'last_token') -> Any:
        """
        Get or create a shared LLM model instance.
        
        Args:
            model_name: HuggingFace model name (e.g., 'gpt2', 'meta-llama/Llama-2-7b')
            device: Target device ('cpu', 'cuda:0', etc.)
            cache_dir: Local cache directory for model weights
            extraction_mode: How to extract embeddings ('last_token', 'pooled', 'all_tokens')
        
        Returns:
            Shared model instance on target device
        """
        key = (model_name, device, cache_dir, extraction_mode)
        
        # Fast path: model already exists
        if key in cls._models:
            return cls._models[key]
        
        # Slow path: need to load model (with locking)
        with cls._lock:
            # Double-check after acquiring lock
            if key not in cls._models:
                os.makedirs(cache_dir, exist_ok=True)
                
                # Determine loader class
                loader_class = cls.MODEL_LOADERS.get(
                    model_name, 
                    cls.MODEL_LOADERS['default']
                )
                
                # Check cache status
                model_name_sanitized = model_name.replace('/', '--')
                cache_model_path = Path(cache_dir) / f"models--{model_name_sanitized}"
                in_cache = cache_model_path.exists() and any(cache_model_path.iterdir())
                
                if in_cache:
                    print(f'[ LLM ] Loading {model_name} from local cache: {cache_model_path}')
                else:
                    print(f'[ LLM ] Downloading {model_name} from HuggingFace...')
                
                # Load model
                model = loader_class.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                ).to(device)
                
                model.eval()
                cls._models[key] = model
                
                print(f'[ LLM ] Model {model_name} ready on {device} (extraction: {extraction_mode})')
        
        return cls._models[key]
    
    @classmethod
    def get_tokenizer(cls, model_name: str, cache_dir: str) -> Any:
        """Get or create a shared tokenizer instance."""
        key = (model_name, cache_dir)
        
        if key in cls._tokenizers:
            return cls._tokenizers[key]
        
        with cls._lock:
            if key not in cls._tokenizers:
                os.makedirs(cache_dir, exist_ok=True)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
                cls._tokenizers[key] = tokenizer
        
        return cls._tokenizers[key]
    
    @classmethod
    def get_embedding_dim(cls, model_name: str) -> int:
        """Get the embedding dimension for a model."""
        DIM_MAP = {
            'gpt2': 768,
            'gpt2-medium': 1024,
            'gpt2-large': 1280,
            'gpt2-xl': 1600,
        }
        return DIM_MAP.get(model_name, 768)  # Default to 768
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached models and tokenizers."""
        with cls._lock:
            # Explicit cleanup for GPU memory
            for model in cls._models.values():
                del model
            cls._models.clear()
            cls._tokenizers.clear()
            
            # Force garbage collection
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    @classmethod
    def list_loaded_models(cls) -> list:
        """List all currently loaded models."""
        return list(cls._models.keys())
```

---

## Prompt System Design

### Prompt Storage Structure

```
data/
├── {dataset_name}/
│   ├── raw_data/
│   │   └── ...
│   └── prompts/
│       ├── metadata.json           # Prompt system metadata
│       ├── timecma_v1/             # Prompt template version
│       │   ├── train/
│       │   │   ├── 0.txt           # Prompt for sample 0
│       │   │   ├── 1.txt
│       │   │   └── ...
│       │   ├── val/
│       │   └── test/
│       └── custom_v1/
│           └── ...
```

### Prompt Metadata Schema

```json
{
    "template_name": "timecma_v1",
    "template_version": "1.0.0",
    "created_at": "2025-01-15T10:30:00Z",
    "dataset_name": "ETTh1",
    "parameters": {
        "include_values": true,
        "include_trends": true,
        "include_timestamps": true,
        "value_format": "integer",
        "frequency_label": "hour"
    },
    "llm_config": {
        "model_name": "gpt2",
        "extraction_mode": "last_token"
    }
}
```

---

## Prompt Template Module

### Location: `embedder/prompt_builder.py`

```python
"""
Prompt Builder for Time Series → LLM Prompt conversion.

Converts time series data to textual prompts using configurable templates.
Inspired by TimeCMA's prompt generation but with flexible template system.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
import torch


class PromptTemplate:
    """
    Base class for prompt templates.
    
    Templates define how time series data is converted to text prompts
    for LLM processing.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
    
    def format(self, 
               values: np.ndarray,
               timestamps: Optional[np.ndarray] = None,
               channel_idx: int = 0,
               metadata: Optional[Dict] = None) -> str:
        """
        Format time series data into a prompt string.
        
        Args:
            values: Time series values [seq_len] or [seq_len, channels]
            timestamps: Optional timestamp features [seq_len, features]
            channel_idx: Channel index for multi-channel data
            metadata: Optional metadata (dataset info, etc.)
        
        Returns:
            Formatted prompt string
        """
        raise NotImplementedError


class TimeCMATemplate(PromptTemplate):
    """
    TimeCMA-style prompt template.
    
    Format: "From [t1] to [t2], the values were v1, v2, ..., vn every {freq}.
             The total trend value was {trend}"
    """
    
    FREQUENCY_MAP = {
        'h': 'hour',
        't': '15 minutes',
        'd': 'day',
        'w': 'week',
        'm': 'month',
        '10min': '10 minutes'
    }
    
    def format(self,
               values: np.ndarray,
               timestamps: Optional[np.ndarray] = None,
               channel_idx: int = 0,
               metadata: Optional[Dict] = None) -> str:
        
        # Extract values for this channel
        if values.ndim == 2:
            channel_values = values[:, channel_idx]
        else:
            channel_values = values
        
        # Format values as integers (like TimeCMA)
        if self.config.get('value_format', 'integer') == 'integer':
            values_str = ", ".join([str(int(v)) for v in channel_values])
        else:
            values_str = ", ".join([f"{v:.2f}" for v in channel_values])
        
        # Compute trend
        trend = np.sum(np.diff(channel_values))
        trend_str = f"{trend:.0f}"
        
        # Format timestamps
        if timestamps is not None:
            start_date = self._format_timestamp(timestamps[0], metadata)
            end_date = self._format_timestamp(timestamps[-1], metadata)
        else:
            start_date = "[t1]"
            end_date = "[t2]"
        
        # Get frequency label
        freq = metadata.get('freq', 'h') if metadata else 'h'
        freq_label = self.FREQUENCY_MAP.get(freq, freq)
        
        # Build prompt
        prompt = (
            f"From {start_date} to {end_date}, "
            f"the values were {values_str} every {freq_label}. "
            f"The total trend value was {trend_str}"
        )
        
        return prompt
    
    def _format_timestamp(self, ts_features: np.ndarray, metadata: Optional[Dict]) -> str:
        """Format timestamp features into readable date string."""
        # Assuming features: [year, month, day, weekday, hour, minute]
        if len(ts_features) >= 5:
            year, month, day = int(ts_features[0]), int(ts_features[1]), int(ts_features[2])
            hour = int(ts_features[4]) if len(ts_features) > 4 else 0
            minute = int(ts_features[5]) if len(ts_features) > 5 else 0
            
            if minute > 0:
                return f"{day:02d}/{month:02d}/{year:04d} {hour:02d}:{minute:02d}"
            elif hour > 0:
                return f"{day:02d}/{month:02d}/{year:04d} {hour:02d}:00"
            else:
                return f"{day:02d}/{month:02d}/{year:04d}"
        return "[unknown]"


class PromptBuilder:
    """
    Main interface for building prompts from time series data.
    
    Manages templates and provides batch prompt generation.
    """
    
    TEMPLATES = {
        'timecma_v1': TimeCMATemplate,
        # Add more templates here
    }
    
    def __init__(self, 
                 template_name: str = 'timecma_v1',
                 template_config: Optional[Dict] = None,
                 prompt_cache_dir: Optional[str] = None):
        """
        Initialize prompt builder.
        
        Args:
            template_name: Name of prompt template to use
            template_config: Configuration for the template
            prompt_cache_dir: Directory to cache generated prompts
        """
        self.template_name = template_name
        self.template_config = template_config or {}
        self.prompt_cache_dir = Path(prompt_cache_dir) if prompt_cache_dir else None
        
        # Initialize template
        template_class = self.TEMPLATES.get(template_name)
        if template_class is None:
            raise ValueError(f"Unknown template: {template_name}. "
                           f"Available: {list(self.TEMPLATES.keys())}")
        
        self.template = template_class(template_name, self.template_config)
    
    def build_prompt(self,
                     values: np.ndarray,
                     timestamps: Optional[np.ndarray] = None,
                     channel_idx: int = 0,
                     metadata: Optional[Dict] = None) -> str:
        """Build a single prompt from time series data."""
        return self.template.format(values, timestamps, channel_idx, metadata)
    
    def build_batch_prompts(self,
                            batch_values: np.ndarray,
                            batch_timestamps: Optional[np.ndarray] = None,
                            metadata: Optional[Dict] = None) -> List[List[str]]:
        """
        Build prompts for a batch of time series.
        
        Args:
            batch_values: [B, seq_len, channels]
            batch_timestamps: [B, seq_len, features] (optional)
            metadata: Shared metadata for all samples
        
        Returns:
            List of lists: [[prompts for channels] for each sample]
        """
        B, L, C = batch_values.shape
        all_prompts = []
        
        for b in range(B):
            sample_prompts = []
            for c in range(C):
                ts = batch_timestamps[b] if batch_timestamps is not None else None
                prompt = self.build_prompt(
                    batch_values[b], ts, c, metadata
                )
                sample_prompts.append(prompt)
            all_prompts.append(sample_prompts)
        
        return all_prompts
    
    def save_prompts(self, 
                     prompts: List[List[str]], 
                     split: str,
                     start_idx: int = 0):
        """Save generated prompts to cache directory."""
        if self.prompt_cache_dir is None:
            raise ValueError("prompt_cache_dir not set")
        
        split_dir = self.prompt_cache_dir / self.template_name / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for i, sample_prompts in enumerate(prompts):
            idx = start_idx + i
            # Save all channel prompts as JSON
            prompt_file = split_dir / f"{idx}.json"
            with open(prompt_file, 'w') as f:
                json.dump(sample_prompts, f, indent=2)
    
    def load_prompts(self, split: str, indices: List[int]) -> List[List[str]]:
        """Load cached prompts for given indices."""
        if self.prompt_cache_dir is None:
            raise ValueError("prompt_cache_dir not set")
        
        split_dir = self.prompt_cache_dir / self.template_name / split
        prompts = []
        
        for idx in indices:
            prompt_file = split_dir / f"{idx}.json"
            if prompt_file.exists():
                with open(prompt_file, 'r') as f:
                    prompts.append(json.load(f))
            else:
                raise FileNotFoundError(f"Prompt not found: {prompt_file}")
        
        return prompts
```

---

## Configuration System

### Model Config Extension: `model_configs/llm_embedding/default.yaml`

```yaml
# LLM Embedding Configuration
# Used by models that require LLM-generated embeddings (e.g., TimeCMA)

llm_embedding:
  # LLM Model Settings
  model_name: "gpt2"                    # HuggingFace model name
  cache_dir: "./LLM_cache/"             # Local model cache directory
  device: "cuda:0"                      # Device for LLM inference
  
  # Extraction Settings
  extraction_mode: "last_token"         # 'last_token', 'pooled', 'all_tokens'
  embedding_dim: 768                    # Output embedding dimension
  
  # Prompt Settings
  prompt_template: "timecma_v1"         # Prompt template name
  prompt_config:
    include_values: true
    include_trends: true
    include_timestamps: true
    value_format: "integer"             # 'integer' or 'float'
  
  # Caching Settings
  cache_embeddings: true                # Cache LLM embeddings to disk
  embedding_cache_dir: null             # Auto-set based on dataset
  force_recompute: false                # Force recomputation of embeddings
  
  # Batch Processing
  batch_size: 32                        # Batch size for LLM inference
  max_length: 512                       # Max prompt token length
```

### Integration with Existing Configs

Add to `model_configs/general/TimeCMA.yaml`:

```yaml
model: TimeCMA

# Model Architecture
channel: 32
num_nodes: 7
e_layer: 1
d_layer: 1
d_ff: 32
head: 8
dropout_n: 0.2

# Include LLM embedding config
llm_embedding:
  model_name: "gpt2"
  extraction_mode: "last_token"
  prompt_template: "timecma_v1"
  cache_embeddings: true

# Normalization
revin: true

# Task
task: TimeCMA
```

---

## Caching Strategy

### LLM Embedding Cache: `embedder/llm_cache.py`

```python
"""
LLM Embedding Cache Manager.

Caches LLM-generated embeddings using a hash-based system similar to
EmbeddingCacheManager, but specifically for LLM last-token embeddings.
"""

import os
import json
import hashlib
import h5py
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class LLMEmbeddingMetadata:
    """Metadata for cached LLM embeddings."""
    model_name: str
    extraction_mode: str
    embedding_dim: int
    prompt_template: str
    prompt_config: Dict[str, Any]
    dataset_name: str
    split: str
    created_at: str
    num_samples: int
    
    def compute_hash(self) -> str:
        """Compute deterministic hash for this metadata configuration."""
        hash_dict = {
            'model_name': self.model_name,
            'extraction_mode': self.extraction_mode,
            'embedding_dim': self.embedding_dim,
            'prompt_template': self.prompt_template,
            'prompt_config': json.dumps(self.prompt_config, sort_keys=True)
        }
        hash_str = json.dumps(hash_dict, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]


class LLMEmbeddingCache:
    """
    Manages caching of LLM-generated embeddings.
    
    Storage Format (H5):
    - embeddings.h5
      ├── embeddings: [N, num_channels, embedding_dim]
      └── indices: [N] (original sample indices)
    
    Storage Format (metadata):
    - metadata.json
    """
    
    def __init__(self, cache_root: str, dataset_name: str):
        self.cache_root = Path(cache_root)
        self.dataset_name = dataset_name
    
    def get_cache_dir(self, metadata: LLMEmbeddingMetadata) -> Path:
        """Get cache directory for given metadata configuration."""
        hash_str = metadata.compute_hash()
        return self.cache_root / self.dataset_name / f"llm_embeddings_{hash_str}"
    
    def cache_exists(self, metadata: LLMEmbeddingMetadata, split: str) -> bool:
        """Check if cache exists for given configuration and split."""
        cache_dir = self.get_cache_dir(metadata)
        embeddings_file = cache_dir / split / "embeddings.h5"
        return embeddings_file.exists()
    
    def save_embeddings(self,
                        embeddings: np.ndarray,
                        indices: np.ndarray,
                        metadata: LLMEmbeddingMetadata,
                        split: str):
        """
        Save embeddings to cache.
        
        Args:
            embeddings: [N, num_channels, embedding_dim]
            indices: [N] original sample indices
            metadata: Embedding metadata
            split: Data split ('train', 'val', 'test')
        """
        cache_dir = self.get_cache_dir(metadata) / split
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_path = cache_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        # Save embeddings as H5
        embeddings_path = cache_dir / "embeddings.h5"
        with h5py.File(embeddings_path, 'w') as hf:
            hf.create_dataset('embeddings', data=embeddings, compression='gzip')
            hf.create_dataset('indices', data=indices)
        
        print(f"[ LLM Cache ] Saved {len(embeddings)} embeddings to {cache_dir}")
    
    def load_embeddings(self, 
                        metadata: LLMEmbeddingMetadata, 
                        split: str) -> Dict[str, np.ndarray]:
        """
        Load embeddings from cache.
        
        Returns:
            Dictionary with 'embeddings' and 'indices' arrays
        """
        cache_dir = self.get_cache_dir(metadata) / split
        embeddings_path = cache_dir / "embeddings.h5"
        
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Cache not found: {embeddings_path}")
        
        with h5py.File(embeddings_path, 'r') as hf:
            result = {
                'embeddings': hf['embeddings'][:],
                'indices': hf['indices'][:]
            }
        
        print(f"[ LLM Cache ] Loaded {len(result['embeddings'])} embeddings from {cache_dir}")
        return result
```

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)

| Task | File | Description |
|------|------|-------------|
| 1.1 | `embedder/llm_registry.py` | LLM model registry (singleton) |
| 1.2 | `embedder/prompt_builder.py` | Prompt template system |
| 1.3 | `embedder/llm_cache.py` | Embedding cache manager |
| 1.4 | `model_configs/llm_embedding/` | Configuration schemas |

### Phase 2: Integration (Week 2-3)

| Task | File | Description |
|------|------|-------------|
| 2.1 | `data_provider/llm_hetero_getter.py` | Data loader with LLM embeddings |
| 2.2 | `embedder/llm_embedder.py` | End-to-end embedding generator |
| 2.3 | Update `data_provider/data_loader.py` | Integration hooks |

### Phase 3: Testing & Optimization (Week 3-4)

| Task | Description |
|------|-------------|
| 3.1 | Unit tests for registry, builder, cache |
| 3.2 | Integration tests with sample datasets |
| 3.3 | Memory profiling and optimization |
| 3.4 | Multi-GPU support validation |

### Phase 4: Documentation (Week 4)

| Task | Description |
|------|-------------|
| 4.1 | Usage documentation |
| 4.2 | Example notebooks |
| 4.3 | Configuration reference |

---

## File Structure Summary

```
fidel-ts/
├── embedder/
│   ├── __init__.py              # Updated exports
│   ├── registry.py              # Existing (embedding models)
│   ├── llm_registry.py          # NEW: LLM model registry
│   ├── prompt_builder.py        # NEW: TS → Prompt conversion
│   ├── llm_cache.py             # NEW: LLM embedding cache
│   └── llm_embedder.py          # NEW: End-to-end LLM embedder
├── model_configs/
│   └── llm_embedding/
│       ├── default.yaml         # NEW: Default LLM config
│       └── gpt2.yaml            # NEW: GPT-2 specific config
├── prompt_templates/
│   └── llm/                     # NEW: LLM prompt templates
│       ├── timecma_v1.template
│       └── custom.template
└── data/
    └── {dataset}/
        └── prompts/             # NEW: Cached prompts
            └── {template_name}/
```

---

## Usage Example

```python
from embedder.llm_registry import LLMRegistry
from embedder.prompt_builder import PromptBuilder
from embedder.llm_cache import LLMEmbeddingCache, LLMEmbeddingMetadata

# 1. Get LLM model (singleton - loaded once, shared)
model = LLMRegistry.get_model(
    model_name='gpt2',
    device='cuda:0',
    cache_dir='./LLM_cache/'
)
tokenizer = LLMRegistry.get_tokenizer('gpt2', './LLM_cache/')

# 2. Build prompts from time series
builder = PromptBuilder(
    template_name='timecma_v1',
    template_config={'value_format': 'integer'}
)
prompts = builder.build_batch_prompts(batch_values, batch_timestamps)

# 3. Generate embeddings
inputs = tokenizer(prompts, return_tensors='pt', padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, -1, :]  # Last token

# 4. Cache embeddings
cache = LLMEmbeddingCache('./data/', 'ETTh1')
metadata = LLMEmbeddingMetadata(
    model_name='gpt2',
    extraction_mode='last_token',
    embedding_dim=768,
    ...
)
cache.save_embeddings(embeddings.numpy(), indices, metadata, 'train')
```

---

## Related Documents

- [TimeCMA Implementation Analysis](./TimeCMA_Implementation_Analysis.md) - TimeCMA port plan
- [Embedding Flow Analysis](../archive/embeddings_centralization/embedding_flow_analysis.md) - Existing embedding system

