"""
Tensor Cache for Amortized Data Loading

This module provides pre-computation of all CPU-intensive dataloader operations,
storing results as memory-mapped numpy arrays for ultra-fast training data loading.

The tensor cache eliminates per-sample overhead by:
1. Pre-computing all temporal matching
2. Pre-looking up all embeddings
3. Pre-building all arrays
4. Storing everything in memory-mapped format

Usage:
    # Generate cache (run once, CPU-only job)
    generator = TensorCacheGenerator(data_provider, cache_dir, config)
    generator.generate()

    # Use cache during training (ultra-fast)
    dataset = TensorCacheDataset(cache_dir, 'train')
    loader = DataLoader(dataset, batch_size=768, num_workers=8)

See context/performance_optimization/training_optimization_plan.md for details.
"""

import os
import json
import hashlib
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, TYPE_CHECKING
from datetime import datetime
from tqdm import tqdm
import warnings

if TYPE_CHECKING:
    from data_provider.data_factory import Data_Provider

logger = logging.getLogger(__name__)


def compute_config_hash(config: dict) -> str:
    """
    Compute hash of config parameters that affect cache validity.

    Cache must be regenerated if ANY of these change:
    - input_len / output_len
    - hetero_stride
    - normalize settings
    - split settings
    - truncate_train_for_purge
    - Entity filtering settings
    """
    relevant_keys = [
        'input_len', 'output_len', 'hetero_stride',
        'scale', 'split', 'truncate_train_for_purge',
        'downsample', 'hetero_type', 'data_name',
        'missing_value_strategy'
    ]

    relevant_config = {}
    for key in relevant_keys:
        if key in config:
            relevant_config[key] = config[key]

    config_str = json.dumps(relevant_config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class TensorCacheMetadata:
    """Metadata for tensor cache validation and info."""

    VERSION = "1.0.0"

    def __init__(
        self,
        config_hash: str,
        data_config: dict,
        shapes: dict,
        dtypes: dict,
        scaler_params: Optional[dict] = None,
        entity_info: Optional[dict] = None,
        created_at: Optional[str] = None
    ):
        self.version = self.VERSION
        self.config_hash = config_hash
        self.data_config = data_config
        self.shapes = shapes
        self.dtypes = dtypes
        self.scaler_params = scaler_params or {}
        self.entity_info = entity_info or {}
        self.created_at = created_at or datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            'version': self.version,
            'config_hash': self.config_hash,
            'created_at': self.created_at,
            'data_config': self.data_config,
            'shapes': self.shapes,
            'dtypes': self.dtypes,
            'scaler_params': self.scaler_params,
            'entity_info': self.entity_info
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TensorCacheMetadata':
        return cls(
            config_hash=data['config_hash'],
            data_config=data['data_config'],
            shapes=data['shapes'],
            dtypes=data['dtypes'],
            scaler_params=data.get('scaler_params', {}),
            entity_info=data.get('entity_info', {}),
            created_at=data.get('created_at')
        )

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> 'TensorCacheMetadata':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class TensorCacheGenerator:
    """
    Generates pre-computed tensor cache for fast training data loading.

    This class processes all samples from a Data_Provider, computing all
    CPU-intensive operations once and storing results as memory-mapped arrays.
    """

    # Array names and their expected tuple indices from __getitem__
    ARRAY_SPECS = {
        'sample_ids': {'index': 0, 'dtype': 'U64'},  # String sample IDs
        'seq_x': {'index': 1, 'dtype': 'float32'},
        'seq_y': {'index': 2, 'dtype': 'float32'},
        'x_time': {'index': 3, 'dtype': 'int64'},
        'y_time': {'index': 4, 'dtype': 'int64'},
        'hetero_x': {'index': 5, 'dtype': 'float32'},
        'hetero_y': {'index': 6, 'dtype': 'float32'},
        'hetero_x_time': {'index': 7, 'dtype': 'float32'},
        'hetero_y_time': {'index': 8, 'dtype': 'float32'},
        'hetero_general': {'index': 9, 'dtype': 'float32'},
        'hetero_channel': {'index': 10, 'dtype': 'float32'},
        'x_time_features': {'index': 11, 'dtype': 'float32'},
        'y_time_features': {'index': 12, 'dtype': 'float32'},
    }

    def __init__(
        self,
        data_provider: 'Data_Provider',
        cache_dir: Union[str, Path],
        config: dict,
        chunk_size: int = 10000,
        verbose: bool = True
    ):
        """
        Initialize tensor cache generator.

        Args:
            data_provider: Data_Provider instance with datasets configured
            cache_dir: Directory to store cache files
            config: Experiment config dict (for hash computation)
            chunk_size: Number of samples to process at once (memory management)
            verbose: Whether to show progress bars
        """
        self.data_provider = data_provider
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.chunk_size = chunk_size
        self.verbose = verbose

        # Compute config hash for cache validation
        self.config_hash = compute_config_hash(self._extract_cache_config())

    def _extract_cache_config(self) -> dict:
        """Extract relevant config for cache hash."""
        args = self.data_provider.args
        return {
            'input_len': getattr(args, 'input_len', None),
            'output_len': getattr(args, 'output_len', None),
            'scale': getattr(args, 'scale', True),
            'truncate_train_for_purge': getattr(args, 'truncate_train_for_purge', False),
            'downsample': getattr(args, 'downsample', None),
            'data_name': getattr(args, 'data', 'unknown'),
        }

    def generate(self, flags: List[str] = None) -> Path:
        """
        Generate cache for specified data splits.

        Args:
            flags: List of splits to generate ('train', 'val', 'test').
                   Defaults to all three.

        Returns:
            Path to cache directory
        """
        if flags is None:
            flags = ['train', 'val', 'test']

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata
        shapes = {}
        entity_info = {'entity_ids': [], 'samples_per_entity': {}}

        for flag in flags:
            logger.info(f"Generating cache for {flag} split...")
            split_shapes, split_entity_info = self._generate_split(flag)
            shapes[flag] = split_shapes

            # Merge entity info
            for eid in split_entity_info.get('entity_ids', []):
                if eid not in entity_info['entity_ids']:
                    entity_info['entity_ids'].append(eid)
            entity_info['samples_per_entity'].update(
                split_entity_info.get('samples_per_entity', {})
            )

        # Save metadata
        metadata = TensorCacheMetadata(
            config_hash=self.config_hash,
            data_config=self._extract_cache_config(),
            shapes=shapes,
            dtypes={name: spec['dtype'] for name, spec in self.ARRAY_SPECS.items()},
            entity_info=entity_info
        )
        metadata.save(self.cache_dir / 'metadata.json')

        logger.info(f"Cache generated at: {self.cache_dir}")
        return self.cache_dir

    def _generate_split(self, flag: str) -> Tuple[dict, dict]:
        """
        Generate cache for a single split.

        Returns:
            Tuple of (shapes dict, entity_info dict)
        """
        split_dir = self.cache_dir / flag
        split_dir.mkdir(exist_ok=True)

        # Get datasets (dict of entity_id -> dataset)
        datasets = self.data_provider.get_datasets(flag)

        if not datasets:
            logger.warning(f"No datasets found for {flag} split")
            return {}, {}

        # Calculate total samples
        total_samples = sum(len(ds) for ds in datasets.values())
        logger.info(f"Total samples for {flag}: {total_samples:,}")

        if total_samples == 0:
            logger.warning(f"No samples in {flag} split")
            return {}, {}

        # Get sample shape from first dataset
        first_dataset = next(iter(datasets.values()))
        sample = first_dataset[0]

        # Determine array shapes
        array_shapes = self._get_array_shapes(sample, total_samples)

        # Create memory-mapped arrays
        arrays = self._create_mmap_arrays(split_dir, array_shapes)

        # Process all datasets
        current_idx = 0
        entity_info = {'entity_ids': [], 'samples_per_entity': {}}

        dataset_iter = tqdm(datasets.items(), desc=f"Entities ({flag})", disable=not self.verbose)
        for entity_id, dataset in dataset_iter:
            entity_samples = len(dataset)
            if entity_samples == 0:
                continue

            entity_info['entity_ids'].append(entity_id)
            entity_info['samples_per_entity'][entity_id] = entity_samples

            # Process in chunks
            for chunk_start in range(0, entity_samples, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, entity_samples)
                chunk_indices = range(chunk_start, chunk_end)

                # Get samples
                chunk_samples = [dataset[i] for i in chunk_indices]

                # Write to arrays
                write_start = current_idx + chunk_start
                write_end = current_idx + chunk_end
                self._write_chunk(arrays, chunk_samples, write_start, write_end)

            current_idx += entity_samples

        # Flush arrays
        for arr in arrays.values():
            if hasattr(arr, 'flush'):
                arr.flush()

        # Build shapes dict for metadata
        shapes = {name: list(arr.shape) for name, arr in arrays.items()}

        return shapes, entity_info

    def _get_array_shapes(self, sample: tuple, total_samples: int) -> Dict[str, tuple]:
        """Determine array shapes from a sample."""
        shapes = {}

        for name, spec in self.ARRAY_SPECS.items():
            idx = spec['index']
            item = sample[idx]

            if item is None:
                # Skip None items
                continue

            if isinstance(item, str):
                # String sample ID
                shapes[name] = (total_samples,)
            elif isinstance(item, np.ndarray):
                if item.size == 0:
                    # Skip empty arrays
                    continue
                shapes[name] = (total_samples,) + item.shape
            elif isinstance(item, (int, float)):
                shapes[name] = (total_samples,)
            else:
                # Try to convert to numpy
                try:
                    arr = np.asarray(item)
                    if arr.size > 0:
                        shapes[name] = (total_samples,) + arr.shape
                except Exception:
                    pass

        return shapes

    def _create_mmap_arrays(
        self,
        split_dir: Path,
        shapes: Dict[str, tuple]
    ) -> Dict[str, np.memmap]:
        """Create memory-mapped arrays for cache storage."""
        arrays = {}

        for name, shape in shapes.items():
            dtype = self.ARRAY_SPECS[name]['dtype']
            filepath = split_dir / f"{name}.npy"

            # Create memory-mapped array
            arrays[name] = np.memmap(
                filepath,
                dtype=dtype,
                mode='w+',
                shape=shape
            )

            logger.debug(f"Created mmap array: {name}, shape={shape}, dtype={dtype}")

        return arrays

    def _write_chunk(
        self,
        arrays: Dict[str, np.memmap],
        samples: List[tuple],
        start_idx: int,
        end_idx: int
    ):
        """Write a chunk of samples to memory-mapped arrays."""
        for name, arr in arrays.items():
            spec = self.ARRAY_SPECS[name]
            idx = spec['index']

            # Extract items from samples
            items = [s[idx] for s in samples]

            # Handle different types
            if name == 'sample_ids':
                # String array
                arr[start_idx:end_idx] = items
            else:
                # Convert to numpy array
                try:
                    chunk_data = np.stack([
                        np.asarray(item) if item is not None else np.zeros(arr.shape[1:])
                        for item in items
                    ])
                    arr[start_idx:end_idx] = chunk_data
                except Exception as e:
                    logger.warning(f"Failed to write {name}: {e}")


class TensorCacheDataset(Dataset):
    """
    Ultra-fast dataset that loads from pre-computed tensor cache.

    __getitem__ is just array indexing - no computation required.
    This provides 100-1000x speedup over on-demand data loading.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        flag: str = 'train',
        preload_to_ram: bool = False
    ):
        """
        Initialize tensor cache dataset.

        Args:
            cache_dir: Path to tensor cache directory
            flag: Data split ('train', 'val', 'test')
            preload_to_ram: If True, load all data to RAM (faster but uses more memory)
        """
        self.cache_dir = Path(cache_dir)
        self.split_dir = self.cache_dir / flag
        self.flag = flag
        self.preload_to_ram = preload_to_ram

        # Load metadata
        self.metadata = TensorCacheMetadata.load(self.cache_dir / 'metadata.json')

        # Load arrays
        self.arrays = self._load_arrays()

        # Get number of samples from first array
        first_array = next(iter(self.arrays.values()))
        self.n_samples = first_array.shape[0]

        logger.info(
            f"TensorCacheDataset initialized: {flag}, "
            f"{self.n_samples:,} samples, "
            f"preload={preload_to_ram}"
        )

    def _load_arrays(self) -> Dict[str, np.ndarray]:
        """Load memory-mapped or RAM arrays."""
        arrays = {}

        for name in TensorCacheGenerator.ARRAY_SPECS.keys():
            filepath = self.split_dir / f"{name}.npy"

            if not filepath.exists():
                continue

            if self.preload_to_ram:
                # Load fully into RAM
                arrays[name] = np.load(filepath, mmap_mode=None)
            else:
                # Memory-mapped (lazy loading)
                arrays[name] = np.load(filepath, mmap_mode='r')

        return arrays

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> tuple:
        """
        Ultra-fast sample retrieval - just array indexing.

        Time complexity: O(1) with memory-mapped I/O
        No computation, no dict lookups, no temporal matching.
        """
        # Build return tuple matching Universal_Dataset format
        return (
            self._get_item('sample_ids', index, default=''),
            self._get_item('seq_x', index),
            self._get_item('seq_y', index),
            self._get_item('x_time', index),
            self._get_item('y_time', index),
            self._get_item('hetero_x', index),
            self._get_item('hetero_y', index),
            self._get_item('hetero_x_time', index),
            self._get_item('hetero_y_time', index),
            self._get_item('hetero_general', index),
            self._get_item('hetero_channel', index),
            self._get_item('x_time_features', index),
            self._get_item('y_time_features', index),
        )

    def _get_item(self, name: str, index: int, default=None):
        """Get item from array or return default."""
        if name in self.arrays:
            return self.arrays[name][index]
        return default if default is not None else np.zeros((1,), dtype=np.float32)

    def get_scaler_params(self) -> Optional[dict]:
        """Get scaler parameters for inverse transform."""
        return self.metadata.scaler_params


def validate_cache(cache_dir: Union[str, Path], config: dict) -> Tuple[bool, str]:
    """
    Validate that a cache exists and matches the given config.

    Args:
        cache_dir: Path to cache directory
        config: Experiment config dict

    Returns:
        Tuple of (is_valid, message)
    """
    cache_dir = Path(cache_dir)

    # Check directory exists
    if not cache_dir.exists():
        return False, f"Cache directory does not exist: {cache_dir}"

    # Check metadata exists
    metadata_path = cache_dir / 'metadata.json'
    if not metadata_path.exists():
        return False, f"Metadata file not found: {metadata_path}"

    # Load and validate metadata
    try:
        metadata = TensorCacheMetadata.load(metadata_path)
    except Exception as e:
        return False, f"Failed to load metadata: {e}"

    # Check version
    if metadata.version != TensorCacheMetadata.VERSION:
        return False, f"Cache version mismatch: {metadata.version} != {TensorCacheMetadata.VERSION}"

    # Check config hash
    current_hash = compute_config_hash(config)
    if metadata.config_hash != current_hash:
        return False, (
            f"Config hash mismatch (cache may be stale). "
            f"Expected: {current_hash}, Got: {metadata.config_hash}"
        )

    # Check that split directories exist with data
    for flag in ['train', 'val', 'test']:
        split_dir = cache_dir / flag
        if split_dir.exists():
            seq_x_file = split_dir / 'seq_x.npy'
            if not seq_x_file.exists():
                return False, f"Missing seq_x.npy in {flag} split"

    return True, "Cache is valid"


def get_tensor_cache_dataloader(
    cache_dir: Union[str, Path],
    flag: str,
    batch_size: int,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    shuffle: bool = None,
    preload_to_ram: bool = False
) -> torch.utils.data.DataLoader:
    """
    Create an optimized DataLoader from tensor cache.

    Args:
        cache_dir: Path to tensor cache directory
        flag: Data split ('train', 'val', 'test')
        batch_size: Batch size
        num_workers: Number of worker processes
        prefetch_factor: Number of batches to prefetch per worker
        shuffle: Whether to shuffle (default: True for train, False otherwise)
        preload_to_ram: Whether to preload data to RAM

    Returns:
        Configured DataLoader
    """
    if shuffle is None:
        shuffle = (flag == 'train')

    dataset = TensorCacheDataset(
        cache_dir=cache_dir,
        flag=flag,
        preload_to_ram=preload_to_ram
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=(flag == 'train')
    )
