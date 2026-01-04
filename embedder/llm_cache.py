"""
LLM Embedding Cache Manager.

Caches LLM-generated embeddings using a hash-based system similar to
EmbeddingCacheManager, but specifically for LLM hidden state embeddings.

CACHE STRUCTURE:
================

    data/{dataset_name}/llm_embeddings/llm_{hash}/
        ├── metadata.json          # Full metadata including prompt info
        ├── progress.json          # Per-split progress tracking for resume
        ├── train/
        │   └── embeddings.h5      # [N, embed_dim, C] HDF5 format
        ├── val/
        │   └── embeddings.h5
        └── test/
            └── embeddings.h5

CHECKPOINTING & RESUME:
=======================

The progress.json file tracks per-split embedding progress, enabling:
1. Resume from interruption (timeout, OOM, manual stop)
2. Atomic progress updates (write temp file, then rename)
3. Integrity verification on restart

Progress states per split:
- 'pending': Not started
- 'in_progress': Currently processing (has samples_written count)
- 'completed': Finished successfully  
- 'failed': Error occurred (has error message)

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
import os
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum

import numpy as np


class ProgressState(Enum):
    """State of embedding generation for a split."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SplitProgress:
    """
    Progress tracking for a single split (train/val/test).
    
    Tracks the state and sample count to enable resume from interruption.
    """
    state: str = "pending"
    total_samples: int = 0
    samples_written: int = 0
    last_chunk_idx: int = 0
    started_at: str = ""
    completed_at: str = ""
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SplitProgress':
        """Create from dictionary."""
        return cls(
            state=data.get('state', 'pending'),
            total_samples=data.get('total_samples', 0),
            samples_written=data.get('samples_written', 0),
            last_chunk_idx=data.get('last_chunk_idx', 0),
            started_at=data.get('started_at', ''),
            completed_at=data.get('completed_at', ''),
            error_message=data.get('error_message', ''),
        )


@dataclass
class EmbeddingProgress:
    """
    Progress tracking for embedding generation across all splits.
    
    Enables resuming from interruption by tracking:
    - Per-split progress state (pending, in_progress, completed, failed)
    - Number of samples written per split
    - Last successfully processed chunk index
    
    Progress is saved atomically (write to temp, then rename) to prevent
    corruption if the process is killed during a write.
    
    Example:
        progress = EmbeddingProgress.load_or_create(cache_dir)
        progress.start_split('train', total_samples=1000000)
        
        for chunk_idx, chunk in enumerate(chunks):
            # Process chunk...
            progress.update_progress('train', samples_written=chunk_idx * chunk_size)
            progress.save(cache_dir)  # Atomic save
        
        progress.complete_split('train')
        progress.save(cache_dir)
    """
    splits: Dict[str, SplitProgress] = field(default_factory=dict)
    created_at: str = ""
    last_updated: str = ""
    
    def __post_init__(self):
        """Set default values after initialization."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.last_updated = datetime.now().isoformat()
    
    def get_split(self, split: str) -> SplitProgress:
        """Get or create progress for a split."""
        if split not in self.splits:
            self.splits[split] = SplitProgress()
        return self.splits[split]
    
    def start_split(self, split: str, total_samples: int):
        """Mark a split as starting (in_progress)."""
        self.splits[split] = SplitProgress(
            state=ProgressState.IN_PROGRESS.value,
            total_samples=total_samples,
            samples_written=0,
            last_chunk_idx=0,
            started_at=datetime.now().isoformat(),
        )
        self.last_updated = datetime.now().isoformat()
    
    def update_progress(self, split: str, samples_written: int, last_chunk_idx: int = 0):
        """Update progress for a split (call after each chunk)."""
        if split not in self.splits:
            raise ValueError(f"Split '{split}' not started. Call start_split first.")
        self.splits[split].samples_written = samples_written
        self.splits[split].last_chunk_idx = last_chunk_idx
        self.last_updated = datetime.now().isoformat()
    
    def complete_split(self, split: str):
        """Mark a split as completed."""
        if split not in self.splits:
            raise ValueError(f"Split '{split}' not started.")
        self.splits[split].state = ProgressState.COMPLETED.value
        self.splits[split].completed_at = datetime.now().isoformat()
        self.last_updated = datetime.now().isoformat()
    
    def fail_split(self, split: str, error_message: str):
        """Mark a split as failed with error message."""
        if split not in self.splits:
            self.splits[split] = SplitProgress()
        self.splits[split].state = ProgressState.FAILED.value
        self.splits[split].error_message = error_message
        self.last_updated = datetime.now().isoformat()
    
    def is_split_completed(self, split: str) -> bool:
        """Check if a split is completed."""
        return (split in self.splits and 
                self.splits[split].state == ProgressState.COMPLETED.value)
    
    def is_split_resumable(self, split: str) -> bool:
        """Check if a split can be resumed (was in_progress with some samples written)."""
        if split not in self.splits:
            return False
        sp = self.splits[split]
        return (sp.state == ProgressState.IN_PROGRESS.value and 
                sp.samples_written > 0)
    
    def get_resume_info(self, split: str) -> Optional[Tuple[int, int]]:
        """
        Get resume information for a split.
        
        Returns:
            Tuple of (samples_written, last_chunk_idx) or None if not resumable
        """
        if not self.is_split_resumable(split):
            return None
        sp = self.splits[split]
        return (sp.samples_written, sp.last_chunk_idx)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            'splits': {k: v.to_dict() for k, v in self.splits.items()},
            'created_at': self.created_at,
            'last_updated': self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingProgress':
        """Create from dictionary."""
        splits_data = data.get('splits', {})
        splits = {k: SplitProgress.from_dict(v) for k, v in splits_data.items()}
        return cls(
            splits=splits,
            created_at=data.get('created_at', ''),
            last_updated=data.get('last_updated', ''),
        )
    
    def save(self, cache_dir: Path):
        """
        Save progress atomically (write temp file, then rename).
        
        This ensures the progress file is never corrupted, even if the
        process is killed during the write.
        
        Args:
            cache_dir: Directory containing the progress.json file
        """
        progress_path = Path(cache_dir) / "progress.json"
        
        # Write to temp file in same directory (for atomic rename)
        fd, temp_path = tempfile.mkstemp(
            suffix='.json.tmp',
            dir=str(cache_dir)
        )
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            # Atomic rename (on same filesystem)
            shutil.move(temp_path, str(progress_path))
        except Exception:
            # Clean up temp file if rename failed
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
    
    @classmethod
    def load(cls, cache_dir: Path) -> 'EmbeddingProgress':
        """Load progress from file."""
        progress_path = Path(cache_dir) / "progress.json"
        if not progress_path.exists():
            raise FileNotFoundError(f"Progress file not found: {progress_path}")
        with open(progress_path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    @classmethod
    def load_or_create(cls, cache_dir: Path) -> 'EmbeddingProgress':
        """Load existing progress or create new."""
        progress_path = Path(cache_dir) / "progress.json"
        if progress_path.exists():
            try:
                return cls.load(cache_dir)
            except Exception as e:
                print(f"[ warning ] Failed to load progress.json: {e}. Creating new.")
        return cls()


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
    
    def get_cache_info(
        self,
        metadata: LLMEmbeddingMetadata,
        split: str,
    ) -> Optional[Tuple[int, int, int]]:
        """
        Get cache info without loading full embeddings into memory.
        
        Args:
            metadata: LLMEmbeddingMetadata object
            split: Data split ('train', 'val', 'test')
        
        Returns:
            Tuple of (num_samples, embed_dim, num_channels) or None if not cached
        """
        cache_dir = self.get_cache_dir(metadata)
        embeddings_path = cache_dir / split / "embeddings.h5"
        
        if not embeddings_path.exists():
            return None
        
        with h5py.File(embeddings_path, 'r') as hf:
            shape = hf['embeddings'].shape  # Just reads shape, doesn't load data!
        
        return shape  # (num_samples, embed_dim, num_channels)
    
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
        Verify that all splits have valid and COMPLETE embeddings.
        
        Uses verify_integrity() for each split to check:
        - HDF5 file exists and is readable
        - Dataset has correct structure
        - Progress tracking indicates completion (not partial)
        
        Args:
            metadata: LLMEmbeddingMetadata object
            splits: List of splits to check (default: ['train', 'val', 'test'])
        
        Returns:
            Dictionary with:
                - valid: bool - True if ALL splits are fully complete
                - issues: List[str] - List of issues found
                - splits: Dict[str, bool] - Per-split validity
                - resumable: Dict[str, int] - Splits that can be resumed (with sample count)
        """
        if splits is None:
            splits = ['train', 'val', 'test']
        
        issues = []
        split_status = {}
        resumable_splits = {}
        
        cache_dir = self.get_cache_dir(metadata)
        
        # Check metadata exists
        metadata_path = cache_dir / "metadata.json"
        if not metadata_path.exists():
            issues.append(f"Metadata file not found: {metadata_path}")
        
        # Check each split using verify_integrity
        for split in splits:
            integrity = self.verify_integrity(metadata, split)
            
            if integrity['valid']:
                split_status[split] = True
            else:
                split_status[split] = False
                
                # Check if resumable
                if integrity['can_resume']:
                    resumable_splits[split] = integrity['samples_complete']
                    issues.append(
                        f"Partial {split}: {integrity['samples_complete']:,} samples "
                        f"(resumable)"
                    )
                else:
                    # Add all issues for this split
                    for issue in integrity['issues']:
                        issues.append(f"{split}: {issue}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'splits': split_status,
            'resumable': resumable_splits,
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
    
    def verify_integrity(
        self,
        metadata: LLMEmbeddingMetadata,
        split: str,
    ) -> Dict[str, Any]:
        """
        Verify integrity of cached embeddings for a split.
        
        Checks:
        1. HDF5 file exists and is readable
        2. Dataset has correct shape and dimensions
        3. No NaN or Inf values (optional, can be slow for large files)
        4. Progress tracking is consistent
        
        Args:
            metadata: LLMEmbeddingMetadata object
            split: Data split ('train', 'val', 'test')
        
        Returns:
            Dictionary with:
                - valid: bool - True if cache is valid
                - issues: List[str] - List of issues found
                - shape: Tuple - Dataset shape if readable
                - can_resume: bool - True if partial and can resume
                - samples_complete: int - Number of valid samples
        """
        cache_dir = self.get_cache_dir(metadata)
        embeddings_path = cache_dir / split / "embeddings.h5"
        progress_path = cache_dir / "progress.json"
        
        result = {
            'valid': False,
            'issues': [],
            'shape': None,
            'can_resume': False,
            'samples_complete': 0,
        }
        
        # Check if file exists
        if not embeddings_path.exists():
            result['issues'].append(f"HDF5 file not found: {embeddings_path}")
            
            # Check if we can resume from progress
            if progress_path.exists():
                try:
                    progress = EmbeddingProgress.load(cache_dir)
                    if progress.is_split_resumable(split):
                        resume_info = progress.get_resume_info(split)
                        result['can_resume'] = True
                        result['samples_complete'] = resume_info[0] if resume_info else 0
                        result['issues'].append(
                            f"Partial progress found: {result['samples_complete']} samples"
                        )
                except Exception as e:
                    result['issues'].append(f"Failed to load progress: {e}")
            
            return result
        
        # Try to open and verify HDF5 file
        try:
            with h5py.File(embeddings_path, 'r') as hf:
                if 'embeddings' not in hf:
                    result['issues'].append("HDF5 file missing 'embeddings' dataset")
                    return result
                
                dataset = hf['embeddings']
                result['shape'] = dataset.shape
                result['samples_complete'] = dataset.shape[0]
                
                # Verify dimensions match metadata expectations
                expected_dim = metadata.embedding_dim
                if dataset.shape[1] != expected_dim:
                    result['issues'].append(
                        f"Embedding dim mismatch: got {dataset.shape[1]}, "
                        f"expected {expected_dim}"
                    )
                
                # Check for completion via progress.json
                if progress_path.exists():
                    try:
                        progress = EmbeddingProgress.load(cache_dir)
                        if progress.is_split_completed(split):
                            result['valid'] = True
                        elif progress.is_split_resumable(split):
                            result['can_resume'] = True
                            resume_info = progress.get_resume_info(split)
                            if resume_info:
                                result['samples_complete'] = resume_info[0]
                                result['issues'].append(
                                    f"Incomplete: {resume_info[0]}/{dataset.shape[0]} samples"
                                )
                    except Exception as e:
                        result['issues'].append(f"Progress file error: {e}")
                        # Fall back to checking if file looks complete
                        result['valid'] = len(result['issues']) == 0
                else:
                    # No progress file - assume complete if no other issues
                    result['valid'] = len(result['issues']) == 0
                    
        except Exception as e:
            result['issues'].append(f"Failed to read HDF5: {str(e)}")
            
            # Check if resumable
            if progress_path.exists():
                try:
                    progress = EmbeddingProgress.load(cache_dir)
                    if progress.is_split_resumable(split):
                        result['can_resume'] = True
                except Exception:
                    pass
        
        return result
    
    def get_resume_info(
        self,
        metadata: LLMEmbeddingMetadata,
        split: str,
    ) -> Optional[Tuple[int, int, int]]:
        """
        Get information needed to resume interrupted embedding generation.
        
        Args:
            metadata: LLMEmbeddingMetadata object
            split: Data split ('train', 'val', 'test')
        
        Returns:
            Tuple of (samples_written, last_chunk_idx, total_samples) or None
            if not resumable
        """
        cache_dir = self.get_cache_dir(metadata)
        progress_path = cache_dir / "progress.json"
        
        if not progress_path.exists():
            return None
        
        try:
            progress = EmbeddingProgress.load(cache_dir)
            
            if progress.is_split_completed(split):
                return None  # Already complete, no need to resume
            
            if progress.is_split_resumable(split):
                sp = progress.get_split(split)
                return (sp.samples_written, sp.last_chunk_idx, sp.total_samples)
            
            return None
        except Exception:
            return None


class StreamingEmbeddingWriter:
    """
    Write embeddings to HDF5 incrementally without holding all in memory.
    
    This class enables memory-efficient embedding generation for large datasets
    by writing embeddings to disk as they are generated, rather than accumulating
    them all in memory before writing.
    
    The HDF5 file is created with a pre-allocated dataset of known dimensions,
    and embeddings are written in chunks as they are processed.
    
    CHECKPOINTING:
    - Periodic flushing to disk (configurable via flush_every)
    - Progress tracking via EmbeddingProgress for resume capability
    - Supports resume from partial completion
    
    Usage:
        progress = EmbeddingProgress.load_or_create(cache_dir)
        with StreamingEmbeddingWriter(
            cache_dir, split, N, E, C,
            progress=progress,
            flush_every=10000,
        ) as writer:
            for batch_embeddings in generate_embeddings():
                writer.write_batch(batch_embeddings)
    
    Args:
        cache_dir: Directory to store the embeddings
        split: Data split name ('train', 'val', 'test')
        num_samples: Total number of samples (N)
        embed_dim: Embedding dimension (E)
        num_channels: Number of channels (C)
        hdf5_chunk_size: Chunk size for HDF5 storage (for efficient I/O)
        progress: EmbeddingProgress tracker for resume capability
        flush_every: Flush to disk every N samples (default: 10000)
        resume_from: Resume from this sample index (skip writing before this)
    """
    
    def __init__(
        self,
        cache_dir: Path,
        split: str,
        num_samples: int,
        embed_dim: int,
        num_channels: int,
        hdf5_chunk_size: int = 100,
        progress: Optional['EmbeddingProgress'] = None,
        flush_every: int = 10000,
        resume_from: int = 0,
    ):
        """
        Initialize the streaming writer.
        
        Creates the HDF5 file with a pre-allocated dataset, or opens existing
        file for resume.
        
        Args:
            cache_dir: Path to cache directory (e.g., llm_embeddings/llm_{hash}/)
            split: Data split ('train', 'val', 'test')
            num_samples: Total number of samples to write
            embed_dim: LLM embedding dimension
            num_channels: Number of data channels
            hdf5_chunk_size: Chunk size for HDF5 compression/access
            progress: EmbeddingProgress tracker (optional, for resume)
            flush_every: Flush HDF5 to disk every N samples written
            resume_from: Resume from this sample index (for interrupted jobs)
        """
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.num_samples = num_samples
        self.embed_dim = embed_dim
        self.num_channels = num_channels
        self.flush_every = flush_every
        self.progress = progress
        self.resume_from = resume_from
        
        # Create split directory
        self.split_dir = self.cache_dir / split
        self.split_dir.mkdir(parents=True, exist_ok=True)
        
        self.h5_path = self.split_dir / "embeddings.h5"
        
        # Calculate optimal chunk size (don't exceed num_samples)
        effective_chunk_size = min(hdf5_chunk_size, num_samples)
        
        # Check if resuming from existing file
        if resume_from > 0 and self.h5_path.exists():
            # Open existing file for appending
            self.h5_file = h5py.File(self.h5_path, 'r+')
            self.dataset = self.h5_file['embeddings']
            # Verify shape matches
            if self.dataset.shape != (num_samples, embed_dim, num_channels):
                raise ValueError(
                    f"Resume shape mismatch: existing {self.dataset.shape} vs "
                    f"expected ({num_samples}, {embed_dim}, {num_channels})"
                )
            self.write_idx = resume_from
            self._samples_since_flush = 0
        else:
            # Create new HDF5 file with pre-allocated dataset
            self.h5_file = h5py.File(self.h5_path, 'w')
            self.dataset = self.h5_file.create_dataset(
                'embeddings',
                shape=(num_samples, embed_dim, num_channels),
                dtype=np.float32,
                chunks=(effective_chunk_size, embed_dim, num_channels),
                compression='gzip',
                compression_opts=4,
            )
            self.write_idx = 0
            self._samples_since_flush = 0
        
        # Track state
        self.is_closed = False
        self._last_flush_idx = self.write_idx
        self._chunk_idx = 0
    
    def write_batch(self, embeddings: np.ndarray, chunk_idx: int = None):
        """
        Write a batch of embeddings to disk with periodic flushing.
        
        Automatically flushes to disk every `flush_every` samples and updates
        progress tracking for resume capability.
        
        Args:
            embeddings: Array of shape [batch_size, embed_dim, num_channels]
            chunk_idx: Optional chunk index for progress tracking
        
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
        self._samples_since_flush += batch_size
        
        # Track chunk index
        if chunk_idx is not None:
            self._chunk_idx = chunk_idx
        
        # Periodic flush for durability
        if self._samples_since_flush >= self.flush_every:
            self._flush_with_progress()
    
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
    def progress_fraction(self) -> float:
        """Return progress as a fraction (0.0 to 1.0)."""
        return self.write_idx / self.num_samples if self.num_samples > 0 else 0.0
    
    def _flush_with_progress(self):
        """
        Flush HDF5 and atomically update progress tracking.
        
        This ensures that progress.json is always consistent with the
        HDF5 file - we flush HDF5 first, then update progress.
        """
        # Flush HDF5 first (ensure data is on disk)
        self.h5_file.flush()
        
        # Update progress tracker (atomic save)
        if self.progress is not None:
            self.progress.update_progress(
                self.split,
                samples_written=self.write_idx,
                last_chunk_idx=self._chunk_idx,
            )
            self.progress.save(self.cache_dir)
        
        # Reset flush counter
        self._samples_since_flush = 0
        self._last_flush_idx = self.write_idx
    
    def flush(self):
        """
        Flush pending writes to disk with progress update.
        
        Always updates progress.json atomically after flushing HDF5.
        """
        if not self.is_closed:
            self._flush_with_progress()
    
    def close(self):
        """
        Close the HDF5 file and finalize writes.
        
        Flushes any pending data and updates progress before closing.
        """
        if not self.is_closed:
            # Final flush before close
            if self._samples_since_flush > 0:
                self._flush_with_progress()
            self.h5_file.close()
            self.is_closed = True
    
    def __enter__(self) -> 'StreamingEmbeddingWriter':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - flush and close the file.
        
        On exception, marks the split as failed in progress tracking.
        """
        if exc_type is not None and self.progress is not None:
            # Exception occurred - mark as failed but still save progress
            self.progress.fail_split(
                self.split,
                error_message=f"{exc_type.__name__}: {exc_val}"
            )
            self.progress.save(self.cache_dir)
        
        self.close()
        return False  # Don't suppress exceptions
    
    def force_flush_progress(self):
        """
        Force immediate flush of HDF5 and progress.
        
        Call this from signal handlers or before potential interruption.
        """
        if not self.is_closed:
            self.h5_file.flush()
            if self.progress is not None:
                self.progress.update_progress(
                    self.split,
                    samples_written=self.write_idx,
                    last_chunk_idx=self._chunk_idx,
                )
                self.progress.save(self.cache_dir)
    
    def __repr__(self) -> str:
        return (
            f"StreamingEmbeddingWriter("
            f"split='{self.split}', "
            f"samples={self.write_idx}/{self.num_samples}, "
            f"shape=[{self.num_samples}, {self.embed_dim}, {self.num_channels}])"
        )
