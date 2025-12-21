"""
Shared embedding model registry for text embeddings.

This module provides a singleton registry to share embedding models (e.g., BERT)
across all embedding instances, preventing multiple model copies from
being loaded into GPU memory.

Thread-safe implementation ensures safe concurrent access from DataLoader workers.
"""

import os
import threading
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from typing import Optional


class EmbeddingModelRegistry:
    """
    Singleton registry for shared embedding models and tokenizers.
    
    Provides thread-safe access to shared model instances, keyed by
    (model_name, device, hf_cache_dir) combination. Ensures only one
    model instance exists per unique combination.
    
    Thread-safe: Uses locks to prevent race conditions when multiple
    DataLoader workers try to load the same model simultaneously.
    """
    
    # Class-level storage for shared models and tokenizers
    _models = {}
    _tokenizers = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_model(cls, model_name: str, device: str, hf_cache_dir: str):
        """
        Get or create a shared embedding model instance.
        
        If the model doesn't exist for the given (model_name, device, hf_cache_dir)
        combination, it will be loaded and cached. Subsequent calls with the same
        parameters return the cached instance.
        
        Args:
            model_name: HuggingFace model name (e.g., 'bert-base-uncased')
            device: Target device ('cpu', 'cuda:0', etc.)
            hf_cache_dir: HuggingFace cache directory path
        
        Returns:
            The shared model instance (already on the target device)
        """
        key = (model_name, device, hf_cache_dir)
        
        # Fast path: model already exists
        if key in cls._models:
            return cls._models[key]
        
        # Slow path: need to load model (with locking)
        with cls._lock:
            # Double-check after acquiring lock (prevent race condition)
            if key not in cls._models:
                # Ensure cache directory exists
                os.makedirs(hf_cache_dir, exist_ok=True)
                
                # Check if model exists in cache before loading
                model_name_sanitized = model_name.replace('/', '--')
                cache_model_path = Path(hf_cache_dir) / f"models--{model_name_sanitized}"
                model_in_cache = cache_model_path.exists() and any(cache_model_path.iterdir())
                
                if model_in_cache:
                    print(f'[ info ] Loading {model_name} from local cache: {cache_model_path}')
                else:
                    print(f'[ info ] Downloading {model_name} from HuggingFace (will cache to: {cache_model_path})')
                
                # Load model and move to target device
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=hf_cache_dir
                ).to(device)
                
                model.eval()  # Set to evaluation mode
                cls._models[key] = model
                
                if not model_in_cache and cache_model_path.exists() and any(cache_model_path.iterdir()):
                    print(f'[ info ] Model cached successfully to: {cache_model_path}')
        
        return cls._models[key]
    
    @classmethod
    def get_tokenizer(cls, model_name: str, hf_cache_dir: str):
        """
        Get or create a shared tokenizer instance.
        
        Tokenizers are device-agnostic, so they're keyed only by
        (model_name, hf_cache_dir) combination.
        
        Args:
            model_name: HuggingFace model name (e.g., 'bert-base-uncased')
            hf_cache_dir: HuggingFace cache directory path
        
        Returns:
            The shared tokenizer instance
        """
        key = (model_name, hf_cache_dir)
        
        # Fast path: tokenizer already exists
        if key in cls._tokenizers:
            return cls._tokenizers[key]
        
        # Slow path: need to load tokenizer (with locking)
        with cls._lock:
            # Double-check after acquiring lock
            if key not in cls._tokenizers:
                # Ensure cache directory exists
                os.makedirs(hf_cache_dir, exist_ok=True)
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=hf_cache_dir
                )
                
                cls._tokenizers[key] = tokenizer
        
        return cls._tokenizers[key]
    
    @classmethod
    def clear_cache(cls):
        """
        Clear all cached models and tokenizers.
        
        Useful for testing or freeing memory. Note: This will cause
        models to be reloaded on next access.
        """
        with cls._lock:
            cls._models.clear()
            cls._tokenizers.clear()
    
    @classmethod
    def get_model_info(cls, model_name: str, device: str, hf_cache_dir: str) -> Optional[dict]:
        """
        Get information about a cached model (if it exists).
        
        Args:
            model_name: HuggingFace model name
            device: Target device
            hf_cache_dir: HuggingFace cache directory path
        
        Returns:
            Dictionary with model info (hidden_size, etc.) or None if not loaded
        """
        key = (model_name, device, hf_cache_dir)
        if key in cls._models:
            model = cls._models[key]
            return {
                'hidden_size': model.config.hidden_size,
                'device': str(next(model.parameters()).device),
                'dtype': str(next(model.parameters()).dtype)
            }
        return None

