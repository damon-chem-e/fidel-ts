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


class StreamingEmbeddingWriter:
    """
    Write embeddings to HDF5 incrementally without holding all in memory.
    
    This class enables memory-efficient embedding generation for large datasets
    by writing embeddings to disk as they are generated, rather than accumulating
    them all in memory before writing.
    
    The HDF5 file is created with a pre-allocated dataset of known dimensions,
    and embeddings are written in chunks as they are processed.
    
    Usage:
        with StreamingEmbeddingWriter(cache_dir, split, N, E, C) as writer:
            for batch_embeddings in generate_embeddings():
                writer.write_batch(batch_embeddings)
    
    Args:
        cache_dir: Directory to store the embeddings
        split: Data split name ('train', 'val', 'test')
        num_samples: Total number of samples (N)
        embed_dim: Embedding dimension (E)
        num_channels: Number of channels (C)
        hdf5_chunk_size: Chunk size for HDF5 storage (for efficient I/O)
    """
    
    def __init__(
        self,
        cache_dir: Path,
        split: str,
        num_samples: int,
        embed_dim: int,
        num_channels: int,
        hdf5_chunk_size: int = 100,
    ):
        """
        Initialize the streaming writer.
        
        Creates the HDF5 file with a pre-allocated dataset.
        
        Args:
            cache_dir: Path to cache directory (e.g., llm_embeddings/llm_{hash}/)
            split: Data split ('train', 'val', 'test')
            num_samples: Total number of samples to write
            embed_dim: LLM embedding dimension
            num_channels: Number of data channels
            hdf5_chunk_size: Chunk size for HDF5 compression/access
        """
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.num_samples = num_samples
        self.embed_dim = embed_dim
        self.num_channels = num_channels
        
        # Create split directory
        self.split_dir = self.cache_dir / split
        self.split_dir.mkdir(parents=True, exist_ok=True)
        
        self.h5_path = self.split_dir / "embeddings.h5"
        
        # Calculate optimal chunk size (don't exceed num_samples)
        effective_chunk_size = min(hdf5_chunk_size, num_samples)
        
        # Create HDF5 file with pre-allocated dataset
        self.h5_file = h5py.File(self.h5_path, 'w')
        self.dataset = self.h5_file.create_dataset(
            'embeddings',
            shape=(num_samples, embed_dim, num_channels),
            dtype=np.float32,
            chunks=(effective_chunk_size, embed_dim, num_channels),
            compression='gzip',
            compression_opts=4,
        )
        
        # Track write position
        self.write_idx = 0
        self.is_closed = False
    
    def write_batch(self, embeddings: np.ndarray):
        """
        Write a batch of embeddings to disk.
        
        Args:
            embeddings: Array of shape [batch_size, embed_dim, num_channels]
        
        Raises:
            ValueError: If write would exceed allocated space
            RuntimeError: If writer is already closed
        """
        if self.is_closed:
            raise RuntimeError("Cannot write to closed StreamingEmbeddingWriter")
        
        batch_size = embeddings.shape[0]
        
        # Validate dimensions
        if embeddings.shape[1] != self.embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: got {embeddings.shape[1]}, "
                f"expected {self.embed_dim}"
            )
        if embeddings.shape[2] != self.num_channels:
            raise ValueError(
                f"Channel count mismatch: got {embeddings.shape[2]}, "
                f"expected {self.num_channels}"
            )
        
        # Check bounds
        if self.write_idx + batch_size > self.num_samples:
            raise ValueError(
                f"Write would exceed allocated space: "
                f"position {self.write_idx} + batch {batch_size} > total {self.num_samples}"
            )
        
        # Write to HDF5
        self.dataset[self.write_idx:self.write_idx + batch_size] = embeddings.astype(np.float32)
        self.write_idx += batch_size
    
    def write_sample_embeddings(self, sample_idx: int, embeddings: np.ndarray):
        """
        Write embeddings for a specific sample index.
        
        Useful when processing samples out of order or in chunks.
        
        Args:
            sample_idx: Index of the sample in the dataset
            embeddings: Array of shape [embed_dim, num_channels]
        """
        if self.is_closed:
            raise RuntimeError("Cannot write to closed StreamingEmbeddingWriter")
        
        if sample_idx >= self.num_samples:
            raise ValueError(f"Sample index {sample_idx} >= num_samples {self.num_samples}")
        
        self.dataset[sample_idx] = embeddings.astype(np.float32)
    
    @property
    def samples_written(self) -> int:
        """Return the number of samples written so far."""
        return self.write_idx
    
    @property
    def progress(self) -> float:
        """Return progress as a fraction (0.0 to 1.0)."""
        return self.write_idx / self.num_samples if self.num_samples > 0 else 0.0
    
    def flush(self):
        """Flush pending writes to disk."""
        if not self.is_closed:
            self.h5_file.flush()
    
    def close(self):
        """Close the HDF5 file and finalize writes."""
        if not self.is_closed:
            self.h5_file.close()
            self.is_closed = True
    
    def __enter__(self) -> 'StreamingEmbeddingWriter':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close the file."""
        self.close()
        return False  # Don't suppress exceptions
    
    def __repr__(self) -> str:
        return (
            f"StreamingEmbeddingWriter("
            f"split='{self.split}', "
            f"samples={self.write_idx}/{self.num_samples}, "
            f"shape=[{self.num_samples}, {self.embed_dim}, {self.num_channels}])"
        )
