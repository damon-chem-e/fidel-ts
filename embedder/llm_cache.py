"""
LLM Embedding Cache Manager.

Caches LLM-generated embeddings using a hash-based system similar to
EmbeddingCacheManager, but specifically for LLM hidden state embeddings.

CACHE STRUCTURE:
================

    data/{dataset_name}/llm_embeddings/llm_{hash}/
        ├── metadata.json          # Full metadata including prompt info
        ├── train/
        │   └── embeddings.h5      # [N, embed_dim, C] HDF5 format
        ├── val/
        │   └── embeddings.h5
        └── test/
            └── embeddings.h5

The metadata includes complete configuration and an example prompt for
full traceability - you can inspect any cached embedding and see exactly
what configuration and prompt format produced it.

COMPARISON WITH TEXT EMBEDDING CACHE:
=====================================

There are TWO separate embedding cache systems:

1. Text Embedding Cache (EmbeddingCacheManager):
   - Location: data/{dataset}/embeddings_cache/embeddings_{hash}/
   - Format: .pkl (joblib)
   - Indexed by: timestamp (string keys)
   - Content: BERT/encoder model embeddings of pre-existing text
   - Used by: Lynx, TGTSF, and other multimodal models
   
2. LLM Embedding Cache (this module):
   - Location: data/{dataset}/llm_embeddings/llm_{hash}/
   - Format: .h5 (HDF5 compressed)
   - Indexed by: sample index (integer)
   - Content: GPT/decoder model hidden states
   - Used by: TimeCMA and LLM-based models

These caches are INDEPENDENT and can coexist. A dataset can have both
BERT embeddings (for text inputs) and LLM embeddings (for time series
prompts or other LLM-processed inputs).

DATASET CONSIDERATIONS:
=======================

For Time-MMD datasets (ETTh1, Weather, etc.):
    - Simple structure: data/{dataset}/llm_embeddings/
    - Standard data loading

For Fidel-TS datasets (Bear_room, California_ISO, etc.):
    - Complex nested structures handled by FidelTSPathResolver
    - LLM embeddings still stored in: data/{dataset}/llm_embeddings/
    - Separate from hetero/ directory which contains text data

STATIC DATA NOTE:
=================

The hash is computed ONLY from configuration parameters, NOT from
the underlying dataset data. This is intentional for benchmark datasets
which are static and immutable.

FUTURE EXTENSION: If dynamic datasets with different versions/vintages
are introduced, the cache key must include a data fingerprint (e.g.,
hash of first/last samples, dataset version ID, or data checksum) to
ensure cache invalidation when data changes.

EXTENSIBILITY:
==============

This cache system supports multiple input sources to LLMEmbedder:

1. Time series → prompts (current): Cached with prompt_template in metadata
2. Raw text → embeddings (future): Can use prompt_template='raw_text'
3. Custom sources (future): Extend metadata schema as needed

The cache key (hash) is based on:
- model_name, extraction_mode, embedding_dim
- prompt_template, prompt_config
- quantization

Different input sources will produce different hashes and thus
different cache directories.
"""

import json
import hashlib
import h5py
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field

import numpy as np


@dataclass
class LLMEmbeddingMetadata:
    """
    Metadata for cached LLM embeddings.
    
    Includes complete prompt information for full traceability - you can
    inspect any cached embedding and see exactly what prompt was used.
    """
    # Model configuration
    model_name: str
    extraction_mode: str  # 'last_token' or 'pooled'
    embedding_dim: int
    quantization: Optional[str] = None
    
    # Prompt configuration (full traceability)
    prompt_template: str = 'timecma_v1'
    prompt_template_version: str = '1.0.0'
    prompt_config: Dict[str, Any] = field(default_factory=dict)
    prompt_example: str = ''  # Example prompt for first sample (for inspection)
    
    # Dataset information
    dataset_name: str = ''
    split: str = ''
    num_samples: int = 0
    num_channels: int = 0
    seq_len: int = 0
    
    # Timestamps
    created_at: str = ''
    generation_time_seconds: float = 0.0
    
    # Hardware info (for reproducibility)
    device: str = ''
    torch_version: str = ''
    transformers_version: str = ''
    
    def __post_init__(self):
        """Set default values after initialization."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        
        if not self.torch_version:
            try:
                import torch
                self.torch_version = torch.__version__
            except ImportError:
                self.torch_version = 'unknown'
        
        if not self.transformers_version:
            try:
                import transformers
                self.transformers_version = transformers.__version__
            except ImportError:
                self.transformers_version = 'unknown'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMEmbeddingMetadata':
        """Create from dictionary."""
        # Handle missing fields gracefully
        return cls(
            model_name=data.get('model_name', ''),
            extraction_mode=data.get('extraction_mode', 'last_token'),
            embedding_dim=data.get('embedding_dim', 768),
            quantization=data.get('quantization'),
            prompt_template=data.get('prompt_template', 'timecma_v1'),
            prompt_template_version=data.get('prompt_template_version', '1.0.0'),
            prompt_config=data.get('prompt_config', {}),
            prompt_example=data.get('prompt_example', ''),
            dataset_name=data.get('dataset_name', ''),
            split=data.get('split', ''),
            num_samples=data.get('num_samples', 0),
            num_channels=data.get('num_channels', 0),
            seq_len=data.get('seq_len', 0),
            created_at=data.get('created_at', ''),
            generation_time_seconds=data.get('generation_time_seconds', 0.0),
            device=data.get('device', ''),
            torch_version=data.get('torch_version', ''),
            transformers_version=data.get('transformers_version', ''),
        )
    
    def compute_hash(self) -> str:
        """
        Compute deterministic hash for this metadata configuration.
        
        NOTE: The hash is computed ONLY from configuration parameters, NOT from
        the underlying dataset data. This is intentional for benchmark datasets
        which are static and immutable.
        
        FUTURE EXTENSION: If dynamic datasets with different versions/vintages
        are introduced, the cache key must include a data fingerprint (e.g.,
        hash of first/last samples, dataset version ID, or data checksum) to
        ensure cache invalidation when data changes.
        """
        hash_dict = {
            'model_name': self.model_name,
            'extraction_mode': self.extraction_mode,
            'embedding_dim': self.embedding_dim,
            'quantization': self.quantization,
            'prompt_template': self.prompt_template,
            'prompt_template_version': self.prompt_template_version,
            'prompt_config': json.dumps(self.prompt_config, sort_keys=True),
        }
        hash_str = json.dumps(hash_dict, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]
    
    def save(self, file_path: Path):
        """Save metadata to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, file_path: Path) -> 'LLMEmbeddingMetadata':
        """Load metadata from JSON file."""
        with open(file_path, 'r') as f:
            return cls.from_dict(json.load(f))


class LLMEmbeddingCache:
    """
    Manages caching of LLM-generated embeddings.
    
    Storage Format (H5):
    - embeddings.h5
      └── embeddings: [N, embed_dim, C]
    
    Storage Format (metadata):
    - metadata.json
    
    Example:
        # Use the actual data directory (root_path from config)
        cache = LLMEmbeddingCache('./data/time_mmd/Climate', 'time_mmd_climate')
        
        # Check if cache exists
        if cache.cache_exists(metadata, 'train'):
            embeddings = cache.load_embeddings(metadata, 'train')
        else:
            # Generate and save
            cache.save_embeddings(embeddings, metadata, 'train')
        
        # Embeddings are stored in: ./data/time_mmd/Climate/llm_embeddings/llm_{hash}/
    """
    
    def __init__(self, data_dir: str, dataset_name: str = None):
        """
        Initialize cache manager.
        
        Args:
            data_dir: Path to the actual data directory (e.g., './data/time_mmd/Climate').
                      This is typically the 'root_path' from the dataset config file.
                      LLM embeddings will be stored in {data_dir}/llm_embeddings/
            dataset_name: Dataset name (optional, used for logging/metadata only)
        """
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name or self.data_dir.name
        self.cache_base = self.data_dir / 'llm_embeddings'
    
    def get_cache_dir(self, metadata: LLMEmbeddingMetadata) -> Path:
        """
        Get cache directory for given metadata configuration.
        
        Args:
            metadata: LLMEmbeddingMetadata object
        
        Returns:
            Path to cache directory (llm_{hash}/)
        """
        hash_str = metadata.compute_hash()
        return self.cache_base / f"llm_{hash_str}"
    
    def cache_exists(self, metadata: LLMEmbeddingMetadata, split: str) -> bool:
        """
        Check if cache exists for given configuration and split.
        
        Args:
            metadata: LLMEmbeddingMetadata object
            split: Data split ('train', 'val', 'test')
        
        Returns:
            True if embeddings.h5 exists for this split
        """
        cache_dir = self.get_cache_dir(metadata)
        embeddings_file = cache_dir / split / "embeddings.h5"
        return embeddings_file.exists()
    
    def find_existing_cache(self, target_metadata: LLMEmbeddingMetadata) -> Optional[Path]:
        """
        Find existing cache directory matching target metadata.
        
        Searches for llm_* directories and checks metadata.
        
        Args:
            target_metadata: Metadata to match
        
        Returns:
            Path to matching cache directory, or None if not found
        """
        if not self.cache_base.exists():
            return None
        
        target_hash = target_metadata.compute_hash()
        
        for cache_dir in self.cache_base.iterdir():
            if cache_dir.is_dir() and cache_dir.name.startswith('llm_'):
                # Check if hash matches
                if cache_dir.name == f"llm_{target_hash}":
                    return cache_dir
        
        return None
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: LLMEmbeddingMetadata,
        split: str,
    ):
        """
        Save embeddings to cache.
        
        Args:
            embeddings: [N, embed_dim, C] numpy array
            metadata: LLMEmbeddingMetadata object (will be saved alongside)
            split: Data split ('train', 'val', 'test')
        """
        cache_dir = self.get_cache_dir(metadata)
        split_dir = cache_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Update metadata with split info
        metadata.split = split
        metadata.num_samples = embeddings.shape[0]
        metadata.embedding_dim = embeddings.shape[1]
        metadata.num_channels = embeddings.shape[2]
        
        # Save metadata (in cache root, not split dir)
        metadata_path = cache_dir / "metadata.json"
        metadata.save(metadata_path)
        
        # Save embeddings as H5 (compressed)
        embeddings_path = split_dir / "embeddings.h5"
        with h5py.File(embeddings_path, 'w') as hf:
            hf.create_dataset(
                'embeddings', 
                data=embeddings, 
                compression='gzip',
                compression_opts=4
            )
        
        print(f"[ LLM Cache ] Saved {embeddings.shape[0]} embeddings to {split_dir}")
    
    def load_embeddings(
        self, 
        metadata: LLMEmbeddingMetadata, 
        split: str,
        quiet: bool = False,
    ) -> np.ndarray:
        """
        Load embeddings from cache.
        
        Args:
            metadata: LLMEmbeddingMetadata object (used to find cache dir)
            split: Data split ('train', 'val', 'test')
            quiet: If True, suppress print statements
        
        Returns:
            Embeddings array [N, embed_dim, C]
        
        Raises:
            FileNotFoundError: If cache not found
        """
        cache_dir = self.get_cache_dir(metadata)
        embeddings_path = cache_dir / split / "embeddings.h5"
        
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Cache not found: {embeddings_path}")
        
        with h5py.File(embeddings_path, 'r') as hf:
            embeddings = hf['embeddings'][:]
        
        if not quiet:
            print(f"[ LLM Cache ] Loaded {embeddings.shape[0]} embeddings from {cache_dir / split}")
        return embeddings
    
    def load_metadata(self, metadata: LLMEmbeddingMetadata) -> LLMEmbeddingMetadata:
        """
        Load metadata from cache.
        
        Args:
            metadata: LLMEmbeddingMetadata object (used to find cache dir)
        
        Returns:
            Loaded LLMEmbeddingMetadata from disk
        """
        cache_dir = self.get_cache_dir(metadata)
        metadata_path = cache_dir / "metadata.json"
        return LLMEmbeddingMetadata.load(metadata_path)
    
    def verify_all_splits(
        self, 
        metadata: LLMEmbeddingMetadata,
        splits: List[str] = None
    ) -> Dict[str, Any]:
        """
        Verify that all splits have valid embeddings.
        
        Args:
            metadata: LLMEmbeddingMetadata object
            splits: List of splits to check (default: ['train', 'val', 'test'])
        
        Returns:
            Dictionary with:
                - valid: bool - True if all splits valid
                - issues: List[str] - List of issues found
                - splits: Dict[str, bool] - Per-split validity
        """
        if splits is None:
            splits = ['train', 'val', 'test']
        
        issues = []
        split_status = {}
        
        cache_dir = self.get_cache_dir(metadata)
        
        # Check metadata exists
        metadata_path = cache_dir / "metadata.json"
        if not metadata_path.exists():
            issues.append(f"Metadata file not found: {metadata_path}")
        
        # Check each split
        for split in splits:
            embeddings_path = cache_dir / split / "embeddings.h5"
            
            if not embeddings_path.exists():
                issues.append(f"Missing embeddings for split: {split}")
                split_status[split] = False
                continue
            
            # Try to open and verify H5 file
            try:
                with h5py.File(embeddings_path, 'r') as hf:
                    if 'embeddings' not in hf:
                        issues.append(f"Invalid H5 structure for split: {split}")
                        split_status[split] = False
                    else:
                        split_status[split] = True
            except Exception as e:
                issues.append(f"Error reading {split}: {str(e)}")
                split_status[split] = False
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'splits': split_status,
            'cache_dir': str(cache_dir),
        }
    
    def list_cached_configs(self) -> List[Dict[str, Any]]:
        """
        List all cached configurations for this dataset.
        
        Returns:
            List of metadata dictionaries for each cached config
        """
        if not self.cache_base.exists():
            return []
        
        configs = []
        for cache_dir in self.cache_base.iterdir():
            if cache_dir.is_dir() and cache_dir.name.startswith('llm_'):
                metadata_path = cache_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        metadata = LLMEmbeddingMetadata.load(metadata_path)
                        configs.append({
                            'hash': cache_dir.name.replace('llm_', ''),
                            'model_name': metadata.model_name,
                            'prompt_template': metadata.prompt_template,
                            'created_at': metadata.created_at,
                            'cache_dir': str(cache_dir),
                        })
                    except Exception as e:
                        print(f"[ warning ] Error reading metadata from {cache_dir}: {e}")
        
        return configs

