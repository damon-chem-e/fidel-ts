"""
Embedding cache management with hash-based folders and metadata.

Cache structure:
{dataset_root}/{dataset_path}/embeddings_{hash}/
  ├── metadata.json
  └── embeddings.pkl (or embeddings.pt, etc.)
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import joblib

from .metadata import EmbeddingMetadata


class EmbeddingCacheManager:
    """
    Manages embedding cache directories and metadata.
    
    Uses hash-based folder names (embeddings_{hash}) to identify embedding configurations.
    """
    
    def __init__(self, dataset_root: str, dataset_path: str = ''):
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
        
        Raises:
            FileExistsError: If cache exists and force=False
        """
        cache_dir = self.get_cache_dir(metadata)
        
        if cache_dir.exists() and not force:
            raise FileExistsError(
                f"Cache directory already exists: {cache_dir}. "
                f"Use force=True to overwrite."
            )
        
        cache_dir.mkdir(parents=True, exist_ok=force)
        
        # Save metadata
        metadata_path = cache_dir / 'metadata.json'
        metadata.save(metadata_path)
        
        return cache_dir
    
    def load_embeddings(self, cache_dir: Path) -> Dict[str, Any]:
        """
        Load embeddings from cache directory.
        
        Args:
            cache_dir: Path to cache directory
        
        Returns:
            Dictionary of embeddings (format depends on dataset)
        
        Raises:
            FileNotFoundError: If embeddings file not found
        """
        # Try different file formats
        for ext in ['.pkl', '.pt', '.npz']:
            emb_path = cache_dir / f'embeddings{ext}'
            if emb_path.exists():
                if ext == '.pkl':
                    return joblib.load(emb_path)
                elif ext == '.pt':
                    import torch
                    return torch.load(emb_path, map_location='cpu')
                elif ext == '.npz':
                    import numpy as np
                    return dict(np.load(emb_path))
        
        raise FileNotFoundError(f"Embeddings file not found in {cache_dir}")
    
    def save_embeddings(self, embeddings: Dict[str, Any], cache_dir: Path, format: str = 'pkl'):
        """
        Save embeddings to cache directory.
        
        Args:
            embeddings: Dictionary of embeddings
            cache_dir: Path to cache directory
            format: File format ('pkl', 'pt', 'npz')
        
        Raises:
            ValueError: If format is not supported
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
            raise ValueError(f"Unknown format: {format}. Must be one of: 'pkl', 'pt', 'npz'")
    
    @staticmethod
    def find_fidel_ts_cache(cache_base: Path, target_metadata: EmbeddingMetadata) -> Optional[Path]:
        """
        Find existing Fidel-TS cache directory matching target metadata.
        
        Static method that works directly with cache_base path.
        
        Args:
            cache_base: Base path for Fidel-TS cache
            target_metadata: Metadata to match
        
        Returns:
            Path to matching cache directory, or None if not found
        """
        if not cache_base.exists():
            return None
        
        # Look for embeddings_* directories
        for cache_dir in cache_base.iterdir():
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
    
    @staticmethod
    def create_fidel_ts_cache_dir(cache_base: Path, metadata: EmbeddingMetadata, force: bool = False) -> Path:
        """
        Create Fidel-TS cache directory for given metadata.
        
        Static method that works directly with cache_base path.
        
        Args:
            cache_base: Base path for Fidel-TS cache
            metadata: EmbeddingMetadata object
            force: If True, overwrite existing cache
        
        Returns:
            Path to cache directory
        
        Raises:
            FileExistsError: If cache exists and force=False
        """
        hash_id = metadata.compute_hash()
        cache_dir = cache_base / f"embeddings_{hash_id}"
        
        if cache_dir.exists() and not force:
            raise FileExistsError(
                f"Cache directory already exists: {cache_dir}. "
                f"Use force=True to overwrite."
            )
        
        cache_dir.mkdir(parents=True, exist_ok=force)
        
        # Save metadata
        metadata_path = cache_dir / 'metadata.json'
        metadata.save(metadata_path)
        
        return cache_dir
    
    @staticmethod
    def load_fidel_ts_embeddings(cache_dir: Path) -> Dict[str, Any]:
        """
        Load Fidel-TS embeddings from cache directory.
        
        Loads both dynamic and static embeddings from the same cache directory.
        
        Args:
            cache_dir: Path to cache directory
        
        Returns:
            Dictionary with keys:
                - 'dynamic': Dynamic embeddings dict
                - 'static': Static embeddings dict (or None if not present)
        
        Raises:
            FileNotFoundError: If dynamic_embeddings.pkl not found
        """
        dynamic_path = cache_dir / 'dynamic_embeddings.pkl'
        static_path = cache_dir / 'static_embeddings.pkl'
        
        if not dynamic_path.exists():
            raise FileNotFoundError(f"Dynamic embeddings not found in {cache_dir}")
        
        result = {
            'dynamic': joblib.load(dynamic_path)
        }
        
        if static_path.exists():
            result['static'] = joblib.load(static_path)
        else:
            result['static'] = None
        
        return result
    
    @staticmethod
    def save_fidel_ts_embeddings(dynamic_embeddings: Dict[str, Any], 
                                  cache_dir: Path, static_embeddings: Optional[Dict[str, Any]] = None):
        """
        Save Fidel-TS embeddings to cache directory.
        
        Saves both dynamic and static embeddings to the same cache directory.
        
        Args:
            dynamic_embeddings: Dictionary of dynamic embeddings (timestamp-keyed)
            cache_dir: Path to cache directory (must already exist with metadata.json)
            static_embeddings: Optional dictionary of static embeddings
        """
        # Save dynamic embeddings
        dynamic_path = cache_dir / 'dynamic_embeddings.pkl'
        joblib.dump(dynamic_embeddings, dynamic_path)
        
        # Save static embeddings if provided
        if static_embeddings is not None:
            static_path = cache_dir / 'static_embeddings.pkl'
            joblib.dump(static_embeddings, static_path)

