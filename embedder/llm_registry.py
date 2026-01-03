"""
LLM Registry for Fidel-TS.

Singleton registry that manages LLM instances (GPT-2, Qwen, LLaMA, etc.) with:
- Thread-safe access for loading models
- Local caching of model weights
- Efficient memory management (no duplicate model loading)
- Support for quantization (4-bit, 8-bit) for large models

DESIGN PHILOSOPHY:
==================

This registry provides LLM models for extracting hidden state embeddings.
Unlike text embeddings (BERT-style) which use encoder models, this registry
serves decoder models that process text and expose internal representations.

The registry is INPUT-AGNOSTIC - it provides models, not input processing.
Input sources are handled by LLMEmbedder and its adapters:
- PromptBuilder: time series → text prompts
- Direct text: pass strings directly to embed_texts()
- Custom adapters: any data → text

USAGE:
======

The registry is designed for PRECOMPUTATION of embeddings only.
It should NOT be used during training (use precomputed embeddings instead).
The singleton pattern works for single-process precomputation but would fail
with multiprocessing DataLoader workers.

SUPPORTED MODELS:
=================

- GPT-2 family: gpt2, gpt2-medium, gpt2-large, gpt2-xl
- Qwen 2.5 family: 0.5B to 72B (with quantization for large models)
- LLaMA 3.1 family: 8B, 70B
- Mistral family: 7B

To add new models:
1. Add to EMBED_DIM_MAP for embedding dimension lookup
2. Add to llm_utils.MODEL_SPECS for memory estimation
3. Test with verify_model_cache() before production use

Modeled after embedder/registry.py for embedding models.
"""

import os
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import torch


class LLMRegistry:
    """
    Singleton registry for shared LLM models and tokenizers.
    
    Similar to EmbeddingModelRegistry, but for decoder-only or
    encoder-decoder LLMs used for prompt-to-embedding workflows.
    
    Keys: (model_name, device, cache_dir, quantization)
    
    Example:
        model, embed_dim = LLMRegistry.get_model(
            model_name='gpt2',
            device='cuda:0',
            cache_dir='./LLM_cache/',
            quantization=None
        )
    """
    
    # Class-level storage for shared models and tokenizers
    _models: Dict[Tuple, Any] = {}
    _tokenizers: Dict[Tuple, Any] = {}
    _embed_dims: Dict[str, int] = {}
    _lock = threading.Lock()
    
    # Embedding dimensions for known models
    EMBED_DIM_MAP = {
        # GPT-2 family
        'gpt2': 768,
        'gpt2-medium': 1024,
        'gpt2-large': 1280,
        'gpt2-xl': 1600,
        # Qwen 2.5 family
        'Qwen/Qwen2.5-0.5B-Instruct': 896,
        'Qwen/Qwen2.5-1.5B-Instruct': 1536,
        'Qwen/Qwen2.5-3B-Instruct': 2048,
        'Qwen/Qwen2.5-7B-Instruct': 3584,
        'Qwen/Qwen2.5-14B-Instruct': 5120,
        'Qwen/Qwen2.5-32B-Instruct': 5120,
        'Qwen/Qwen2.5-72B-Instruct': 8192,
        # LLaMA 3.1 family
        'meta-llama/Llama-3.1-8B-Instruct': 4096,
        'meta-llama/Llama-3.1-70B-Instruct': 8192,
        # Mistral family
        'mistralai/Mistral-7B-Instruct-v0.3': 4096,
    }
    
    @classmethod
    def get_model(
        cls, 
        model_name: str, 
        device: str, 
        cache_dir: str,
        quantization: Optional[str] = None,
        use_flash_attention: bool = True
    ) -> Tuple[Any, int]:
        """
        Get or create a shared LLM model instance.
        
        Args:
            model_name: HuggingFace model name (e.g., 'gpt2', 'Qwen/Qwen2.5-72B-Instruct')
            device: Target device ('cpu', 'cuda:0', etc.)
            cache_dir: Local cache directory for model weights
            quantization: Quantization mode ('4bit', '8bit', or None for fp16)
            use_flash_attention: Enable Flash Attention 2 (recommended for speed)
        
        Returns:
            Tuple of (model, embedding_dim)
        
        Example:
            model, embed_dim = LLMRegistry.get_model(
                'Qwen/Qwen2.5-72B-Instruct',
                'cuda:0',
                './LLM_cache/',
                quantization='4bit'
            )
        """
        key = (model_name, device, cache_dir, quantization)
        
        # Fast path: model already exists
        if key in cls._models:
            return cls._models[key], cls._embed_dims.get(model_name, 768)
        
        # Slow path: need to load model (with locking)
        with cls._lock:
            # Double-check after acquiring lock (prevent race condition)
            if key not in cls._models:
                from .llm_utils import load_model_for_embedding
                
                os.makedirs(cache_dir, exist_ok=True)
                
                # Check cache status for logging
                model_name_sanitized = model_name.replace('/', '--')
                cache_model_path = Path(cache_dir) / f"models--{model_name_sanitized}"
                in_cache = cache_model_path.exists() and any(cache_model_path.iterdir())
                
                if in_cache:
                    print(f'[ LLM ] Loading {model_name} from local cache: {cache_model_path}')
                else:
                    print(f'[ LLM ] Downloading {model_name} from HuggingFace...')
                
                # Load model using utility function
                model, embed_dim = load_model_for_embedding(
                    model_name=model_name,
                    device=device,
                    cache_dir=cache_dir,
                    quantization=quantization,
                    use_flash_attention=use_flash_attention
                )
                
                cls._models[key] = model
                cls._embed_dims[model_name] = embed_dim
                
                print(f'[ LLM ] Model {model_name} ready on {device} '
                      f'(quantization={quantization}, embed_dim={embed_dim})')
        
        return cls._models[key], cls._embed_dims.get(model_name, 768)
    
    @classmethod
    def get_tokenizer(cls, model_name: str, cache_dir: str) -> Any:
        """
        Get or create a shared tokenizer instance.
        
        Args:
            model_name: HuggingFace model name (e.g., 'gpt2')
            cache_dir: HuggingFace cache directory path
        
        Returns:
            The shared tokenizer instance (with pad_token set)
        """
        from transformers import AutoTokenizer
        
        key = (model_name, cache_dir)
        
        # Fast path: tokenizer already exists
        if key in cls._tokenizers:
            return cls._tokenizers[key]
        
        # Slow path: need to load tokenizer (with locking)
        with cls._lock:
            # Double-check after acquiring lock
            if key not in cls._tokenizers:
                os.makedirs(cache_dir, exist_ok=True)
                
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
                
                # CRITICAL: GPT-2 and many decoder-only models lack a pad token.
                # Set pad_token = eos_token to enable batched tokenization with padding.
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                
                cls._tokenizers[key] = tokenizer
        
        return cls._tokenizers[key]
    
    @classmethod
    def get_embedding_dim(cls, model_name: str) -> int:
        """
        Get the embedding dimension for a model.
        
        Args:
            model_name: HuggingFace model name
        
        Returns:
            Embedding dimension (hidden size)
        """
        # Check if already loaded
        if model_name in cls._embed_dims:
            return cls._embed_dims[model_name]
        
        # Check known models
        if model_name in cls.EMBED_DIM_MAP:
            return cls.EMBED_DIM_MAP[model_name]
        
        # Default fallback
        return 768
    
    @classmethod
    def verify_model_cache(cls, model_name: str, cache_dir: str) -> Dict[str, Any]:
        """
        Verify that a model is properly cached and not corrupted.
        
        Performs basic checks:
        - Cache directory exists
        - Model config file exists and is valid JSON
        - Model weights file exists (checks for safetensors or pytorch_model.bin)
        
        Args:
            model_name: HuggingFace model name
            cache_dir: Local cache directory
        
        Returns:
            Dictionary with:
                - 'valid': bool - True if cache appears valid
                - 'cached': bool - True if any cache exists
                - 'issues': List[str] - List of any issues found
        
        NOTE: This is a minimal verification. Future extensions could include:
        - Checksum verification against HuggingFace
        - Retry logic with exponential backoff for failed downloads
        - Timeout handling for network operations
        These are documented here for future implementation.
        """
        issues = []
        model_name_sanitized = model_name.replace('/', '--')
        cache_path = Path(cache_dir) / f"models--{model_name_sanitized}"
        
        if not cache_path.exists():
            return {'valid': False, 'cached': False, 'issues': ['Cache directory does not exist']}
        
        # Check for snapshots directory
        snapshots_dir = cache_path / 'snapshots'
        if not snapshots_dir.exists():
            issues.append('No snapshots directory found')
            return {'valid': False, 'cached': True, 'issues': issues}
        
        # Find latest snapshot
        snapshots = list(snapshots_dir.iterdir())
        if not snapshots:
            issues.append('Snapshots directory is empty')
            return {'valid': False, 'cached': True, 'issues': issues}
        
        latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
        
        # Check config file
        config_file = latest_snapshot / 'config.json'
        if not config_file.exists():
            issues.append('config.json not found')
        else:
            try:
                with open(config_file, 'r') as f:
                    json.load(f)
            except json.JSONDecodeError:
                issues.append('config.json is not valid JSON')
        
        # Check for model weights
        has_weights = (
            (latest_snapshot / 'model.safetensors').exists() or
            (latest_snapshot / 'pytorch_model.bin').exists() or
            any(latest_snapshot.glob('model-*.safetensors')) or
            any(latest_snapshot.glob('pytorch_model-*.bin'))
        )
        
        if not has_weights:
            issues.append('No model weight files found')
        
        return {
            'valid': len(issues) == 0,
            'cached': True,
            'issues': issues,
            'snapshot_path': str(latest_snapshot),
        }
    
    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """
        List all supported LLM models with their specifications.
        
        Returns:
            List of model info dictionaries
        """
        from .llm_utils import MODEL_SPECS
        
        result = []
        for name, specs in MODEL_SPECS.items():
            params_b, embed_dim, vram_fp16, vram_4bit = specs
            result.append({
                'name': name,
                'params': f"{params_b:.1f}B" if params_b >= 1 else f"{int(params_b*1000)}M",
                'embed_dim': embed_dim,
                'min_vram_fp16': vram_fp16,
                'min_vram_4bit': vram_4bit,
            })
        return result
    
    @classmethod
    def clear_cache(cls):
        """
        Clear all cached models and tokenizers.
        
        Useful for testing or freeing memory. Note: This will cause
        models to be reloaded on next access.
        """
        with cls._lock:
            # Explicit cleanup for GPU memory
            for model in cls._models.values():
                del model
            cls._models.clear()
            cls._tokenizers.clear()
            cls._embed_dims.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    @classmethod
    def list_loaded_models(cls) -> List[Tuple]:
        """List all currently loaded model keys."""
        return list(cls._models.keys())

