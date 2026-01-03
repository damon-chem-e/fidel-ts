"""
LLM Embedding Provider for Training

Loads precomputed LLM embeddings from cache and provides them
to the data loader during training.

Supports all dataset types: Time-MMD, TTC, and Fidel-TS.

Usage:
    provider = LLMEmbeddingProvider.from_experiment_config(config, split='train')
    embedding = provider[sample_index]  # Returns [embed_dim, num_channels]

The provider is created by Data_Provider when an experiment has an
llm_embedding section in its configuration.
"""

import numpy as np
from typing import Dict, Any, List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from embedder.llm_cache import LLMEmbeddingMetadata


class LLMEmbeddingProvider:
    """
    Provides precomputed LLM embeddings for training.
    
    Loads embeddings from the LLM cache and provides them by sample index.
    Implements the interface expected by the data loader for hetero_channel.
    
    Supports:
        - Time-MMD datasets (time_mmd_*)
        - TTC datasets (ttc_*)
        - Fidel-TS datasets (fidel_*)
    
    Attributes:
        embeddings: Preloaded embeddings array [N, embed_dim, C]
        metadata: Cache metadata for validation
        num_samples: Number of samples in the split
        embed_dim: LLM embedding dimension (e.g., 768 for GPT-2)
        num_channels: Number of channels in the data
    """
    
    def __init__(self, embeddings: np.ndarray, metadata: 'LLMEmbeddingMetadata'):
        """
        Initialize provider with preloaded embeddings.
        
        Args:
            embeddings: Preloaded embeddings array [N, embed_dim, C]
            metadata: Cache metadata for validation
        """
        self.embeddings = embeddings  # [N, embed_dim, C]
        self.metadata = metadata
        self.num_samples = embeddings.shape[0]
        self.embed_dim = embeddings.shape[1]
        self.num_channels = embeddings.shape[2]
    
    @classmethod
    def from_experiment_config(
        cls, 
        experiment_config: Dict[str, Any], 
        split: str,
        validate: bool = True,
        quiet: bool = False,
    ) -> 'LLMEmbeddingProvider':
        """
        Create provider from experiment configuration.
        
        This method mirrors the hash computation from cli.inference generate
        to find the correct cache directory.
        
        Args:
            experiment_config: Experiment configuration dict containing:
                - data.name: Dataset name (e.g., 'time_mmd_traffic')
                - training.input_len, training.output_len
                - llm_embedding: LLM embedding configuration
            split: Data split ('train', 'val', 'test')
            validate: Whether to validate embeddings after loading
            quiet: If True, suppress print statements
        
        Returns:
            LLMEmbeddingProvider instance
        
        Raises:
            FileNotFoundError: If embeddings not found in cache
            ValueError: If embeddings don't match expected configuration
        """
        from embedder.llm_embedder import LLMEmbedder
        from embedder.llm_cache import LLMEmbeddingCache
        
        # Extract configuration
        dataset_name = experiment_config.get('data', {}).get('name')
        if not dataset_name:
            raise ValueError("Experiment config missing data.name")
        
        llm_config = experiment_config.get('llm_embedding', {})
        if not llm_config:
            raise ValueError("Experiment config missing llm_embedding section")
        
        training = experiment_config.get('training', {})
        input_len = training.get('input_len', 96)
        output_len = training.get('output_len', 96)
        
        # Create embedder to resolve paths and build metadata
        # Uses same logic as cli.inference generate
        embedder = LLMEmbedder(
            model_name=llm_config.get('model_name', 'gpt2'),
            cache_dir=llm_config.get('cache_dir', './LLM_cache/'),
            data_root=experiment_config.get('base_data_path', './data/'),
            prompt_template=llm_config.get('prompt_template', 'timecma_v1'),
            prompt_config=llm_config.get('prompt_config', {
                'value_format': 'integer', 
                'include_timestamps': True
            }),
            input_len=input_len,
            output_len=output_len,
            scale=training.get('scale', True),
            data_config_path=experiment_config.get('data', {}).get('config_path'),
        )
        
        # Resolve data directory (handles time_mmd_, ttc_, fidel_ prefixes)
        data_dir = embedder._get_data_directory(dataset_name)
        
        # Build metadata for cache lookup
        metadata = embedder._build_metadata(dataset_name)
        
        # Create cache and check if embeddings exist
        cache = LLMEmbeddingCache(str(data_dir), dataset_name)
        
        if not cache.cache_exists(metadata, split):
            cache_path = cache.get_cache_dir(metadata)
            raise FileNotFoundError(
                f"LLM embeddings not found for '{dataset_name}/{split}'.\n"
                f"Expected cache at: {cache_path}\n"
                f"Hash: {metadata.compute_hash()}\n\n"
                f"Generate embeddings with:\n"
                f"  python -m cli.inference generate-suite <suite_config.yaml>\n"
                f"Or:\n"
                f"  python -m cli.inference generate <experiment_config.yaml>"
            )
        
        # Load embeddings
        if not quiet:
            print(f"[ LLM Provider ] Loading embeddings for {dataset_name}/{split}")
        
        embeddings = cache.load_embeddings(metadata, split, quiet=True)
        
        if not quiet:
            print(f"[ LLM Provider ] Loaded {embeddings.shape[0]} embeddings, "
                  f"shape: {embeddings.shape}")
        
        provider = cls(embeddings, metadata)
        
        if validate:
            provider._validate_config(experiment_config)
        
        return provider
    
    def _validate_config(self, experiment_config: Dict[str, Any]):
        """
        Validate loaded embeddings match experiment configuration.
        
        Args:
            experiment_config: Experiment configuration dict
        
        Raises:
            ValueError: If embeddings don't match model configuration
        """
        # Validate embedding dimension matches model config
        model_overrides = experiment_config.get('model_config_overrides', {})
        expected_d_llm = model_overrides.get('d_llm', 768)
        
        if self.embed_dim != expected_d_llm:
            raise ValueError(
                f"LLM embedding dimension mismatch: "
                f"cached embeddings have dim={self.embed_dim}, "
                f"but model expects d_llm={expected_d_llm}. "
                f"Regenerate embeddings with matching LLM model."
            )
    
    def validate_sample_count(self, dataset_length: int):
        """
        Validate embedding count matches dataset length.
        
        Call this after dataset is created to ensure alignment.
        
        Args:
            dataset_length: Number of samples in the dataset
        
        Raises:
            ValueError: If counts don't match
        """
        if self.num_samples != dataset_length:
            raise ValueError(
                f"LLM embedding count mismatch: "
                f"embeddings={self.num_samples}, dataset={dataset_length}. "
                f"This can happen if:\n"
                f"  1. Dataset was modified after embedding generation\n"
                f"  2. input_len/output_len changed\n"
                f"  3. Split ratios changed\n"
                f"Regenerate embeddings with 'cli.inference generate-suite'."
            )
    
    def __getitem__(self, index: int) -> np.ndarray:
        """
        Return embedding for sample index.
        
        Args:
            index: Sample index (0 to N-1)
        
        Returns:
            Embedding array with shape [embed_dim, num_channels]
        """
        if index < 0 or index >= self.num_samples:
            raise IndexError(
                f"Sample index {index} out of range [0, {self.num_samples})"
            )
        return self.embeddings[index]
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples
    
    def get_for_batch(self, indices: Union[List[int], np.ndarray]) -> np.ndarray:
        """
        Return embeddings for batch of indices.
        
        Args:
            indices: List or array of sample indices
        
        Returns:
            Embeddings array with shape [B, embed_dim, C]
        """
        return self.embeddings[indices]
    
    @property
    def shape(self) -> tuple:
        """Return embedding array shape: (N, embed_dim, C)."""
        return self.embeddings.shape
    
    def __repr__(self) -> str:
        return (
            f"LLMEmbeddingProvider("
            f"samples={self.num_samples}, "
            f"embed_dim={self.embed_dim}, "
            f"channels={self.num_channels}, "
            f"model='{self.metadata.model_name}')"
        )

