# Embeddings Centralization Implementation Plan

## Overview
This plan outlines the centralization of text embedding functionality into a dedicated `embedder` module. This will consolidate embedding logic currently scattered across `data_provider`, `data_loader`, and `time_mmd_dataset`, provide a unified interface, support multiple aggregation methods (CLS token, average pooling, no pooling), and implement a robust caching system with metadata.

## Goals

1. **Centralize embedding logic** in a new `embedder/` module (subfolder of root)
2. **Support multiple aggregation methods**: CLS token (default), average pooling, no pooling (sequence dimension preserved)
3. **Unified caching system** with metadata tracking
4. **Metadata-driven embedding storage** using hash-based folder names
5. **Backward compatibility** with existing code

## Current State

### Embedding Logic Locations

1. **`data_provider/time_mmd_dataset.py`**:
   - `TimeMMD_HeteroGetter._embed_text_corpus()`: Embeds text using BERT with CLS token
   - `TimeMMD_HeteroGetter._embed_single_text()`: Embeds single text
   - `TimeMMD_HeteroGetter._load_or_create_embeddings()`: Loads from `.pkl` or creates
   - Uses `EmbeddingModelRegistry` from `utils/embedding_model_registry.py`

2. **`data_provider/data_loader.py`**:
   - `Heterogeneous_Dataset.convert_plain_text_to_embeddings()`: CLS token embedding
   - `Heterogeneous_Dataset.convert_df_text_to_embeddings()`: Batch CLS token embedding
   - Loads models directly (not using registry)

3. **`utils/embedding_model_registry.py`**:
   - `EmbeddingModelRegistry`: Singleton registry for shared models/tokenizers
   - Used by TimeMMD datasets

### Current Embedding Storage

**Time-MMD datasets**:
- Location: `{root_path}/{base_filename}.pkl`
- Format: Dictionary mapping timestamp strings to embedding arrays
- No metadata: Just pickle file with embeddings

**Fidel-TS datasets**:
- Pre-computed embeddings from HuggingFace
- Location: Various (need to check structure)
- No metadata about embedding configuration

## Target Architecture

### Module Structure

```
embedder/
├── __init__.py
├── registry.py          # EmbeddingModelRegistry (moved from utils/)
├── embedder.py          # Main embedding functionality
├── metadata.py          # Metadata management
├── cache_manager.py     # Caching logic with hash-based folders
└── aggregation.py       # Aggregation methods (CLS, average, none)
```

### Key Components

1. **EmbeddingModelRegistry** (moved from `utils/embedding_model_registry.py`)
2. **TextEmbedder**: Main class for embedding text
3. **EmbeddingMetadata**: Metadata structure and validation
4. **EmbeddingCacheManager**: Manages cache directories and metadata
5. **Aggregation functions**: CLS, average pooling, no pooling

## Implementation Details

### Phase 1: Create Embedder Module Structure

#### 1.1 Create `embedder/` Directory

Create new directory at root level: `embedder/`

#### 1.2 Move EmbeddingModelRegistry

**Source**: `utils/embedding_model_registry.py`
**Destination**: `embedder/registry.py`

**Changes**:
- Keep existing functionality
- Update imports in all files that use it
- Add to `embedder/__init__.py` exports

#### 1.3 Create Aggregation Module

**File**: `embedder/aggregation.py`

```python
"""
Text embedding aggregation methods.

Supports:
- CLS token: Extract first token embedding (default, current behavior)
- Average pooling: Average over all non-padding tokens
- None: Return full sequence (preserve sequence dimension)
"""

import torch


def aggregate_cls_token(last_hidden_state, attention_mask=None):
    """
    Extract CLS token embedding (first token).
    
    Args:
        last_hidden_state: [B, seq_len, hidden_dim] token embeddings
        attention_mask: Optional [B, seq_len] mask (unused, kept for API consistency)
    
    Returns:
        embeddings: [B, hidden_dim] CLS token embeddings
    """
    return last_hidden_state[:, 0, :]


def aggregate_average_pooling(last_hidden_state, attention_mask):
    """
    Average pooling over tokens (excluding padding).
    
    Args:
        last_hidden_state: [B, seq_len, hidden_dim] token embeddings
        attention_mask: [B, seq_len] mask (1 for real tokens, 0 for padding)
    
    Returns:
        embeddings: [B, hidden_dim] averaged token embeddings
    """
    # Mask out padding tokens
    masked_embeddings = last_hidden_state * attention_mask.unsqueeze(-1)  # [B, seq_len, hidden_dim]
    
    # Sum over sequence dimension
    sum_embeddings = masked_embeddings.sum(dim=1)  # [B, hidden_dim]
    
    # Count non-padding tokens
    sum_mask = attention_mask.sum(dim=1, keepdim=True)  # [B, 1]
    
    # Avoid division by zero
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    
    # Average
    embeddings = sum_embeddings / sum_mask  # [B, hidden_dim]
    
    return embeddings


def aggregate_none(last_hidden_state, attention_mask=None):
    """
    Return full sequence (no aggregation).
    
    Args:
        last_hidden_state: [B, seq_len, hidden_dim] token embeddings
        attention_mask: Optional [B, seq_len] mask (returned as-is for downstream use)
    
    Returns:
        embeddings: [B, seq_len, hidden_dim] full sequence embeddings
        attention_mask: [B, seq_len] attention mask (if provided)
    """
    if attention_mask is not None:
        return last_hidden_state, attention_mask
    return last_hidden_state


# Registry of aggregation methods
AGGREGATION_METHODS = {
    'cls': aggregate_cls_token,
    'average': aggregate_average_pooling,
    'none': aggregate_none,
}


def get_aggregation_function(method):
    """
    Get aggregation function by name.
    
    Args:
        method: 'cls', 'average', or 'none'
    
    Returns:
        Aggregation function
    """
    if method not in AGGREGATION_METHODS:
        raise ValueError(f"Unknown aggregation method: {method}. Must be one of: {list(AGGREGATION_METHODS.keys())}")
    return AGGREGATION_METHODS[method]
```

### Phase 2: Metadata System

#### 2.1 EmbeddingMetadata Structure

**File**: `embedder/metadata.py`

```python
"""
Embedding metadata management.

Metadata tracks:
- Creation date/time
- Tokenizer name
- Embedding model name
- Aggregation method (cls/average/none)
- Sequence length (if aggregation='none')
- Embedding dimension
- Other configuration parameters
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
import hashlib


class EmbeddingMetadata:
    """
    Metadata for cached embeddings.
    
    Stores all information needed to identify and validate embedding cache.
    """
    
    def __init__(self, 
                 tokenizer_name: str,
                 model_name: str,
                 aggregation_method: str,
                 embedding_dim: int,
                 sequence_length: Optional[int] = None,
                 max_length: int = 512,
                 created_at: Optional[str] = None,
                 **extra_config):
        """
        Initialize embedding metadata.
        
        Args:
            tokenizer_name: Name of tokenizer (e.g., 'bert-base-uncased')
            model_name: Name of embedding model (e.g., 'bert-base-uncased')
            aggregation_method: 'cls', 'average', or 'none'
            embedding_dim: Dimension of embeddings
            sequence_length: Max sequence length (if aggregation='none', this is the padded length)
            max_length: Maximum tokenization length
            created_at: ISO format timestamp (auto-generated if None)
            **extra_config: Additional configuration parameters
        """
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.aggregation_method = aggregation_method
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.max_length = max_length
        self.created_at = created_at or datetime.now().isoformat()
        self.extra_config = extra_config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'tokenizer_name': self.tokenizer_name,
            'model_name': self.model_name,
            'aggregation_method': self.aggregation_method,
            'embedding_dim': self.embedding_dim,
            'sequence_length': self.sequence_length,
            'max_length': self.max_length,
            'created_at': self.created_at,
            **self.extra_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingMetadata':
        """Create metadata from dictionary."""
        # Extract known fields
        known_fields = {
            'tokenizer_name', 'model_name', 'aggregation_method', 
            'embedding_dim', 'sequence_length', 'max_length', 'created_at'
        }
        kwargs = {k: data.pop(k) for k in list(data.keys()) if k in known_fields}
        # Remaining fields go to extra_config
        kwargs['extra_config'] = data
        return cls(**kwargs)
    
    def to_json(self) -> str:
        """Serialize metadata to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EmbeddingMetadata':
        """Deserialize metadata from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def save(self, file_path: Path):
        """Save metadata to file."""
        with open(file_path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, file_path: Path) -> 'EmbeddingMetadata':
        """Load metadata from file."""
        with open(file_path, 'r') as f:
            return cls.from_json(f.read())
    
    def compute_hash(self) -> str:
        """
        Compute hash identifier for this metadata configuration.
        
        Uses a subset of metadata fields that determine the embedding characteristics.
        """
        # Fields that affect embedding computation
        hash_fields = {
            'tokenizer_name': self.tokenizer_name,
            'model_name': self.model_name,
            'aggregation_method': self.aggregation_method,
            'embedding_dim': self.embedding_dim,
            'max_length': self.max_length,
        }
        
        # Include sequence_length only if aggregation='none'
        if self.aggregation_method == 'none':
            hash_fields['sequence_length'] = self.sequence_length
        
        # Sort for consistent hashing
        hash_str = json.dumps(hash_fields, sort_keys=True)
        
        # Compute hash
        hash_obj = hashlib.sha256(hash_str.encode())
        return hash_obj.hexdigest()[:16]  # Use first 16 chars for folder name
    
    def matches(self, other: 'EmbeddingMetadata') -> bool:
        """
        Check if this metadata matches another (for cache lookup).
        
        Compares fields that affect embedding computation.
        """
        return (
            self.tokenizer_name == other.tokenizer_name and
            self.model_name == other.model_name and
            self.aggregation_method == other.aggregation_method and
            self.embedding_dim == other.embedding_dim and
            self.max_length == other.max_length and
            (self.aggregation_method != 'none' or self.sequence_length == other.sequence_length)
        )
```

#### 2.2 Metadata File Format

**File**: `metadata.json` in each embedding cache folder

```json
{
  "tokenizer_name": "bert-base-uncased",
  "model_name": "bert-base-uncased",
  "aggregation_method": "average",
  "embedding_dim": 768,
  "sequence_length": null,
  "max_length": 512,
  "created_at": "2024-01-15T10:30:00.123456",
  "hf_cache_dir": "./HF_cache/",
  "device": "cpu"
}
```

### Phase 3: Cache Manager

#### 3.1 EmbeddingCacheManager

**File**: `embedder/cache_manager.py`

```python
"""
Embedding cache management with hash-based folders and metadata.

Cache structure:
{dataset_root}/{dataset_path}/embeddings_{hash}/
  ├── metadata.json
  └── embeddings.pkl (or embeddings.pt, etc.)
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import joblib

from .metadata import EmbeddingMetadata


class EmbeddingCacheManager:
    """
    Manages embedding cache directories and metadata.
    
    Uses hash-based folder names (embeddings_{hash}) to identify embedding configurations.
    """
    
    def __init__(self, dataset_root: str, dataset_path: str):
        """
        Initialize cache manager.
        
        Args:
            dataset_root: Root directory for dataset (e.g., 'data/time_mmd')
            dataset_path: Dataset-specific path (e.g., subdataset name, or '' for single file)
        """
        self.dataset_root = Path(dataset_root)
        self.dataset_path = Path(dataset_path) if dataset_path else Path()
        self.cache_base = self.dataset_root / self.dataset_path
    
    def get_cache_dir(self, metadata: EmbeddingMetadata) -> Path:
        """
        Get cache directory path for given metadata.
        
        Args:
            metadata: EmbeddingMetadata object
        
        Returns:
            Path to cache directory (embeddings_{hash}/)
        """
        hash_id = metadata.compute_hash()
        return self.cache_base / f"embeddings_{hash_id}"
    
    def find_existing_cache(self, target_metadata: EmbeddingMetadata) -> Optional[Path]:
        """
        Find existing cache directory matching target metadata.
        
        Searches for embeddings_{hash} directories and checks metadata.
        
        Args:
            target_metadata: Metadata to match
        
        Returns:
            Path to matching cache directory, or None if not found
        """
        if not self.cache_base.exists():
            return None
        
        # Look for embeddings_* directories
        for cache_dir in self.cache_base.iterdir():
            if cache_dir.is_dir() and cache_dir.name.startswith('embeddings_'):
                metadata_path = cache_dir / 'metadata.json'
                if metadata_path.exists():
                    try:
                        existing_metadata = EmbeddingMetadata.load(metadata_path)
                        if existing_metadata.matches(target_metadata):
                            return cache_dir
                    except Exception as e:
                        # Skip invalid metadata files
                        print(f"[ warning ] Invalid metadata in {cache_dir}: {e}")
                        continue
        
        return None
    
    def create_cache_dir(self, metadata: EmbeddingMetadata, force: bool = False) -> Path:
        """
        Create cache directory for given metadata.
        
        Args:
            metadata: EmbeddingMetadata object
            force: If True, overwrite existing cache
        
        Returns:
            Path to cache directory
        """
        cache_dir = self.get_cache_dir(metadata)
        
        if cache_dir.exists() and not force:
            raise FileExistsError(f"Cache directory already exists: {cache_dir}. Use force=True to overwrite.")
        
        cache_dir.mkdir(parents=True, exist_ok=force)
        
        # Save metadata
        metadata_path = cache_dir / 'metadata.json'
        metadata.save(metadata_path)
        
        return cache_dir
    
    def load_embeddings(self, cache_dir: Path) -> dict:
        """
        Load embeddings from cache directory.
        
        Args:
            cache_dir: Path to cache directory
        
        Returns:
            Dictionary of embeddings (format depends on dataset)
        """
        # Try different file formats
        for ext in ['.pkl', '.pt', '.npz']:
            emb_path = cache_dir / f'embeddings{ext}'
            if emb_path.exists():
                if ext == '.pkl':
                    return joblib.load(emb_path)
                elif ext == '.pt':
                    import torch
                    return torch.load(emb_path)
                elif ext == '.npz':
                    import numpy as np
                    return dict(np.load(emb_path))
        
        raise FileNotFoundError(f"Embeddings file not found in {cache_dir}")
    
    def save_embeddings(self, embeddings: dict, cache_dir: Path, format: str = 'pkl'):
        """
        Save embeddings to cache directory.
        
        Args:
            embeddings: Dictionary of embeddings
            cache_dir: Path to cache directory
            format: File format ('pkl', 'pt', 'npz')
        """
        if format == 'pkl':
            emb_path = cache_dir / 'embeddings.pkl'
            joblib.dump(embeddings, emb_path)
        elif format == 'pt':
            import torch
            emb_path = cache_dir / 'embeddings.pt'
            torch.save(embeddings, emb_path)
        elif format == 'npz':
            import numpy as np
            emb_path = cache_dir / 'embeddings.npz'
            np.savez_compressed(emb_path, **embeddings)
        else:
            raise ValueError(f"Unknown format: {format}")
```

### Phase 4: Main Embedder Class

#### 4.1 TextEmbedder

**File**: `embedder/embedder.py`

```python
"""
Main text embedding functionality.

Unified interface for embedding text with support for:
- Multiple aggregation methods (CLS, average, none)
- Caching with metadata
- Batch processing
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path

from .registry import EmbeddingModelRegistry
from .aggregation import get_aggregation_function
from .metadata import EmbeddingMetadata
from .cache_manager import EmbeddingCacheManager


class TextEmbedder:
    """
    Unified text embedder with caching and multiple aggregation methods.
    """
    
    def __init__(self,
                 model_name: str = 'bert-base-uncased',
                 aggregation_method: str = 'cls',
                 device: str = 'cpu',
                 hf_cache_dir: str = './HF_cache/',
                 max_length: int = 512,
                 batch_size: int = 32,
                 cache_root: Optional[str] = None,
                 cache_path: Optional[str] = None,
                 force_reembed: bool = False):
        """
        Initialize text embedder.
        
        Args:
            model_name: HuggingFace model name
            aggregation_method: 'cls', 'average', or 'none'
            device: Device for model ('cpu', 'cuda:0', etc.)
            hf_cache_dir: HuggingFace cache directory
            max_length: Maximum tokenization length
            batch_size: Batch size for embedding computation
            cache_root: Root directory for embedding cache (if None, caching disabled)
            cache_path: Dataset-specific path within cache_root
            force_reembed: If True, recompute embeddings even if cache exists
        """
        self.model_name = model_name
        self.aggregation_method = aggregation_method
        self.device = device
        self.hf_cache_dir = hf_cache_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.force_reembed = force_reembed
        
        # Get model and tokenizer from registry
        self.model = EmbeddingModelRegistry.get_model(model_name, device, hf_cache_dir)
        self.tokenizer = EmbeddingModelRegistry.get_tokenizer(model_name, hf_cache_dir)
        
        # Get aggregation function
        self.aggregate_fn = get_aggregation_function(aggregation_method)
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        
        # Cache manager
        if cache_root is not None:
            self.cache_manager = EmbeddingCacheManager(cache_root, cache_path or '')
        else:
            self.cache_manager = None
        
        # Determine sequence_length for aggregation='none'
        self.sequence_length = None
        if aggregation_method == 'none':
            self.sequence_length = max_length  # All sequences padded to max_length
    
    def create_metadata(self, **extra_config) -> EmbeddingMetadata:
        """Create metadata object for current configuration."""
        return EmbeddingMetadata(
            tokenizer_name=self.model_name,
            model_name=self.model_name,
            aggregation_method=self.aggregation_method,
            embedding_dim=self.embedding_dim,
            sequence_length=self.sequence_length,
            max_length=self.max_length,
            **extra_config
        )
    
    def embed_texts(self, texts: List[str], 
                   return_metadata: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, EmbeddingMetadata]]:
        """
        Embed list of texts.
        
        Args:
            texts: List of text strings
            return_metadata: If True, also return metadata
        
        Returns:
            embeddings: numpy array of embeddings
                - If aggregation='cls' or 'average': [N, embedding_dim]
                - If aggregation='none': [N, sequence_length, embedding_dim]
            metadata: EmbeddingMetadata (if return_metadata=True)
        """
        # Check cache
        if self.cache_manager and not self.force_reembed:
            metadata = self.create_metadata()
            cache_dir = self.cache_manager.find_existing_cache(metadata)
            if cache_dir is not None:
                # Load from cache
                cached_embeddings = self.cache_manager.load_embeddings(cache_dir)
                # TODO: Match texts to cached embeddings (need text keys or index)
                # For now, assume cache contains all texts in same order
                # This needs refinement based on actual use case
        
        # Compute embeddings
        all_embeddings = []
        all_attention_masks = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state  # [B, seq_len, hidden_dim]
            
            # Aggregate
            if self.aggregation_method == 'none':
                embeddings, masks = self.aggregate_fn(last_hidden_state, attention_mask)
                all_embeddings.append(embeddings.cpu().numpy())
                all_attention_masks.append(masks.cpu().numpy())
            else:
                embeddings = self.aggregate_fn(last_hidden_state, attention_mask)
                all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate
        if self.aggregation_method == 'none':
            result = np.concatenate(all_embeddings, axis=0)
            attention_masks = np.concatenate(all_attention_masks, axis=0)
            # Return both embeddings and masks
            # TODO: Handle return format
        else:
            result = np.concatenate(all_embeddings, axis=0)
        
        # Save to cache if enabled
        if self.cache_manager:
            metadata = self.create_metadata()
            cache_dir = self.cache_manager.find_existing_cache(metadata)
            if cache_dir is None:
                cache_dir = self.cache_manager.create_cache_dir(metadata)
                # Save embeddings
                # TODO: Determine save format based on use case
                embeddings_dict = {i: emb for i, emb in enumerate(result)}
                self.cache_manager.save_embeddings(embeddings_dict, cache_dir, format='pkl')
        
        if return_metadata:
            metadata = self.create_metadata()
            return result, metadata
        return result
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed single text."""
        result = self.embed_texts([text])
        return result[0]
```

### Phase 5: Update Existing Code

#### 5.1 Update TimeMMD_Dataset

**File**: `data_provider/time_mmd_dataset.py`

**Changes**:
1. Import `TextEmbedder` from `embedder`
2. Replace `TimeMMD_HeteroGetter._embed_text_corpus()` with `TextEmbedder.embed_texts()`
3. Replace `TimeMMD_HeteroGetter._load_or_create_embeddings()` with cache manager logic
4. Update embedding path logic to use hash-based folders

**New structure**:
```python
from embedder import TextEmbedder, EmbeddingCacheManager, EmbeddingMetadata

class TimeMMD_HeteroGetter:
    def __init__(self, ...):
        # ...
        # Initialize embedder
        self.embedder = TextEmbedder(
            model_name=self.embed_model_name,
            aggregation_method='cls',  # Default for backward compatibility
            device=self.device,
            hf_cache_dir=self.hf_cache_dir,
            cache_root=self.root_path,
            cache_path=self.data_path.parent.name if hasattr(self.data_path, 'parent') else '',
            force_reembed=self.force_reembed
        )
    
    def _load_or_create_embeddings(self):
        """Load or create embeddings using centralized embedder."""
        # Use embedder's caching logic
        # ...
```

#### 5.2 Update Heterogeneous_Dataset

**File**: `data_provider/data_loader.py`

**Changes**:
1. Import `TextEmbedder` from `embedder`
2. Replace `convert_plain_text_to_embeddings()` and `convert_df_text_to_embeddings()` with `TextEmbedder`

#### 5.3 Update Data Factory

**File**: `data_provider/data_factory.py`

**Changes**:
1. Pass embedding configuration from data config to datasets
2. Ensure embedding config includes aggregation method

### Phase 6: Embeddings Configuration

#### 6.1 Embeddings Config as Subconfig

Add embeddings configuration section to data configs:

```yaml
# Example: data_configs/time_mmd/example.yaml

embeddings:
  model_name: "bert-base-uncased"
  aggregation_method: "cls"  # Options: cls, average, none
  max_length: 512
  hf_cache_dir: "./HF_cache/"
  force_reembed: false
  # Other embedding parameters
```

#### 6.2 Config Structure

**Embeddings config fields**:
- `model_name`: HuggingFace model name (required)
- `aggregation_method`: 'cls' (default), 'average', or 'none' (optional)
- `max_length`: Maximum tokenization length (optional, default: 512)
- `hf_cache_dir`: HuggingFace cache directory (optional)
- `force_reembed`: Force recomputation (optional, default: false)
- `sequence_length`: For aggregation='none', padded sequence length (optional)

This config will be:
1. Used to create `TextEmbedder` instances
2. Saved in `metadata.json` when caching embeddings
3. Used to match existing caches

### Phase 7: File Structure for Different Datasets

#### 7.1 Time-MMD Datasets

**Current**: `data/time_mmd/{subdataset}/{filename}.pkl`

**New**: `data/time_mmd/{subdataset}/embeddings_{hash}/`
- `metadata.json`
- `embeddings.pkl`

**Migration**:
- Keep old `.pkl` files for backward compatibility during transition
- New embeddings go to hash-based folders
- Old embeddings can be migrated by reading and re-saving with metadata

#### 7.2 Fidel-TS Datasets

**Current**: Pre-computed embeddings from HuggingFace, various locations

**New**: Based on `data_structure.json` analysis:
- For each dataset (e.g., `Bear_room`), create embeddings folders
- Structure: `data/{dataset_name}/embeddings_{hash}/`
- `metadata.json` should note that embeddings are pre-computed from HuggingFace
- Include source information in metadata

**Metadata for pre-computed embeddings**:
```json
{
  "tokenizer_name": "bert-base-uncased",
  "model_name": "bert-base-uncased",
  "aggregation_method": "cls",
  "embedding_dim": 768,
  "sequence_length": null,
  "max_length": 512,
  "created_at": "2024-01-15T10:30:00.123456",
  "source": "huggingface_precomputed",
  "source_url": "https://huggingface.co/...",
  "original_location": "data/Bear_room/.cache/huggingface/download/..."
}
```

#### 7.3 TTC Datasets

Similar to Time-MMD structure.

### Phase 8: Integration with Data Providers

#### 8.1 Data Provider Updates

**File**: `data_provider/data_factory.py`

**Changes**:
1. Read embeddings config from data config
2. Pass embeddings config to dataset classes
3. Ensure cache paths are set correctly

#### 8.2 Dataset Updates

All dataset classes that use embeddings should:
1. Accept embeddings config
2. Initialize `TextEmbedder` with config
3. Use embedder for all embedding operations
4. Use cache manager for loading/saving

### Phase 9: Testing and Migration

#### 9.1 Unit Tests

1. **Aggregation functions**:
   - Test CLS token extraction
   - Test average pooling (with padding)
   - Test no pooling (sequence preservation)

2. **Metadata**:
   - Test metadata creation, serialization, loading
   - Test hash computation (deterministic)
   - Test matching logic

3. **Cache manager**:
   - Test cache directory creation
   - Test cache lookup
   - Test embedding save/load

4. **TextEmbedder**:
   - Test embedding computation
   - Test caching
   - Test batch processing

#### 9.2 Integration Tests

1. **Time-MMD integration**:
   - Test embedding computation with new system
   - Test cache loading
   - Test backward compatibility

2. **Data loader integration**:
   - Test heterogeneous dataset embedding
   - Test embedding format compatibility

#### 9.3 Migration Strategy

1. **Phase 1**: Implement new embedder module (non-breaking)
2. **Phase 2**: Update one dataset class to use new system (test)
3. **Phase 3**: Migrate all dataset classes
4. **Phase 4**: Update data configs to include embeddings config
5. **Phase 5**: Remove old embedding code

**Backward compatibility**:
- Keep old embedding paths working during transition
- Provide migration script to convert old `.pkl` files to new structure
- Support loading from both old and new formats

### Phase 10: Documentation

#### 10.1 Code Documentation

- Docstrings for all classes and functions
- Type hints
- Usage examples

#### 10.2 User Documentation

- How to configure embeddings
- How to use different aggregation methods
- How cache system works
- Migration guide

## Implementation Order

### Week 1: Foundation
1. Create `embedder/` directory structure
2. Move `EmbeddingModelRegistry` to `embedder/registry.py`
3. Create `aggregation.py` with aggregation functions
4. Create `metadata.py` with metadata classes
5. Unit tests for aggregation and metadata

### Week 2: Cache System
6. Create `cache_manager.py`
7. Implement hash-based folder system
8. Implement metadata-based cache lookup
9. Unit tests for cache manager

### Week 3: Main Embedder
10. Create `TextEmbedder` class
11. Integrate aggregation, caching, metadata
12. Unit tests for TextEmbedder

### Week 4: Integration
13. Update `TimeMMD_Dataset` to use new embedder
14. Update `Heterogeneous_Dataset` to use new embedder
15. Update `data_factory.py` for embeddings config
16. Integration tests

### Week 5: Migration and Cleanup
17. Migrate existing embeddings (optional)
18. Update all data configs
19. Remove old embedding code
20. Documentation

## Key Design Decisions

1. **Hash-based cache folders**: Encode metadata in folder name for easy lookup
2. **Metadata-driven**: All embedding characteristics tracked in metadata.json
3. **Multiple aggregation methods**: Support CLS, average, none for flexibility
4. **Backward compatibility**: Support old format during transition
5. **Centralized logic**: Single source of truth for embedding operations
6. **Config-driven**: Embedding configuration in data configs, saved in metadata

## Potential Challenges

1. **Cache migration**: Converting old `.pkl` files to new structure
2. **Text key matching**: How to match texts to cached embeddings (may need text hashing)
3. **Sequence length handling**: For aggregation='none', ensuring consistent padding
4. **Memory efficiency**: Large batch processing for embeddings
5. **Fidel-TS structure**: Understanding exact structure of pre-computed embeddings

## Notes

- Hash length (16 chars) can be adjusted if collisions become an issue
- Embedding file format (pkl/pt/npz) can be configurable
- Sequence dimension preservation (aggregation='none') adds complexity but needed for some models
- Metadata should include enough information to reproduce embeddings exactly

