"""
LLM Utilities for large model inference.

Best practices for running large models (up to 70B) locally:
- Hardware: L40S (48GB), A100 (40/80GB), H200 (141GB)
- Quantization: 4-bit (bitsandbytes) for memory efficiency
- Flash Attention 2: For faster inference
- Gradient checkpointing: Not needed for inference-only

Supported models for hidden state extraction:
- Qwen 2.5 series: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B
- LLaMA 3.1 series: 8B, 70B
- Mistral series: 7B
- GPT-2 series: 124M, 355M, 774M, 1.5B

Hardware recommendations:
- L40S (48GB): 4bit for 70B, fp16 for ≤14B
- A100-40GB: 4bit for 70B, 8bit for 32B, fp16 for ≤14B
- A100-80GB: 4bit for 70B, fp16 for ≤32B
- H200 (141GB): fp16 for all models up to 70B
"""

import os
from typing import Optional, Dict, Any, Tuple

import torch


# Model specifications: (params_billions, embed_dim, min_vram_fp16_gb, min_vram_4bit_gb)
MODEL_SPECS = {
    # GPT-2 family
    'gpt2': (0.124, 768, 0.5, 0.3),
    'gpt2-medium': (0.355, 1024, 1.5, 0.8),
    'gpt2-large': (0.774, 1280, 3.0, 1.5),
    'gpt2-xl': (1.5, 1600, 6.0, 3.0),
    
    # Qwen 2.5 family 
    'Qwen/Qwen2.5-0.5B-Instruct': (0.5, 896, 1.0, 0.5),
    'Qwen/Qwen2.5-1.5B-Instruct': (1.5, 1536, 3.0, 1.5),
    'Qwen/Qwen2.5-3B-Instruct': (3.0, 2048, 6.0, 2.5),
    'Qwen/Qwen2.5-7B-Instruct': (7.0, 3584, 14.0, 5.0),
    'Qwen/Qwen2.5-14B-Instruct': (14.0, 5120, 28.0, 9.0),
    'Qwen/Qwen2.5-32B-Instruct': (32.0, 5120, 64.0, 20.0),
    'Qwen/Qwen2.5-72B-Instruct': (72.0, 8192, 144.0, 42.0),
    
    # LLaMA 3.1 family
    'meta-llama/Llama-3.1-8B-Instruct': (8.0, 4096, 16.0, 6.0),
    'meta-llama/Llama-3.1-70B-Instruct': (70.0, 8192, 140.0, 40.0),
    
    # Mistral family
    'mistralai/Mistral-7B-Instruct-v0.3': (7.0, 4096, 14.0, 5.0),
}


def get_quantization_config(quantization: Optional[str]):
    """
    Get BitsAndBytes quantization config for memory-efficient loading.
    
    Args:
        quantization: '4bit', '8bit', or None for fp16
    
    Returns:
        BitsAndBytesConfig or None
    
    Hardware recommendations:
        - L40S (48GB): 4bit for 70B, fp16 for ≤14B
        - A100-40GB: 4bit for 70B, 8bit for 32B, fp16 for ≤14B
        - A100-80GB: 4bit for 70B, fp16 for ≤32B
        - H200 (141GB): fp16 for all models up to 70B
    """
    if quantization is None:
        return None
    
    from transformers import BitsAndBytesConfig
    
    if quantization == '4bit':
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,  # Nested quantization for extra savings
        )
    elif quantization == '8bit':
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        raise ValueError(f"Unknown quantization: {quantization}. Use '4bit', '8bit', or None.")


def load_model_for_embedding(
    model_name: str,
    device: str,
    cache_dir: str,
    quantization: Optional[str] = None,
) -> Tuple[Any, int]:
    """
    Load a model configured for embedding extraction (hidden state access).
    
    This function loads models specifically for extracting the last hidden state,
    NOT for text generation. Uses PyTorch's default attention implementation.
    
    Args:
        model_name: HuggingFace model name
        device: Target device ('cuda:0', etc.)
        cache_dir: Local cache directory
        quantization: '4bit', '8bit', or None
    
    Returns:
        Tuple of (model, embedding_dim)
    
    Example:
        model, embed_dim = load_model_for_embedding(
            "Qwen/Qwen2.5-72B-Instruct",
            device="cuda:0",
            cache_dir="./LLM_cache/",
            quantization="4bit",
        )
    """
    from transformers import AutoModelForCausalLM, AutoConfig
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Get quantization config
    quant_config = get_quantization_config(quantization)
    
    # Load config first to get embedding dimension
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    embed_dim = config.hidden_size
    
    # Build model kwargs
    model_kwargs = {
        'cache_dir': cache_dir,
        'dtype': torch.float16,
    }
    
    # Device mapping for quantization (required by bitsandbytes)
    if quantization:
        model_kwargs['device_map'] = device
        model_kwargs['quantization_config'] = quant_config
    
    # Load model (uses PyTorch's default attention implementation)
    print(f"[ LLM ] Loading {model_name} (quantization={quantization})")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Move to device if not using quantization (quantization handles device_map)
    if not quantization:
        model = model.to(device)
    
    model.eval()
    print(f"[ LLM ] Model ready. Embedding dim: {embed_dim}")
    
    return model, embed_dim


def estimate_model_memory(model_name: str, quantization: Optional[str] = None) -> Dict[str, Any]:
    """
    Estimate GPU memory requirements for a model.
    
    Args:
        model_name: HuggingFace model name
        quantization: '4bit', '8bit', or None
    
    Returns:
        Dictionary with:
            - params_billions: Number of parameters in billions
            - vram_gb: Estimated VRAM in GB
            - recommended_gpu: Suggested GPU for this config
            - embed_dim: Embedding dimension
    """
    specs = MODEL_SPECS.get(model_name)
    
    if specs is None:
        # Unknown model - return defaults with warning
        return {
            'params_billions': 0,
            'vram_gb': 0,
            'recommended_gpu': 'Unknown model - check manually',
            'embed_dim': 768,
        }
    
    params_b, embed_dim, vram_fp16, vram_4bit = specs
    
    # Calculate VRAM based on quantization
    if quantization == '4bit':
        vram = vram_4bit
    elif quantization == '8bit':
        vram = vram_fp16 * 0.6  # Rough estimate
    else:
        vram = vram_fp16
    
    # Determine recommended GPU
    if vram <= 24:
        gpu = "RTX 4090 / L4 / A10"
    elif vram <= 48:
        gpu = "L40S / A40"
    elif vram <= 80:
        gpu = "A100-80GB"
    elif vram <= 141:
        gpu = "H200"
    else:
        gpu = "Multi-GPU required"
    
    return {
        'params_billions': params_b,
        'vram_gb': vram,
        'recommended_gpu': gpu,
        'embed_dim': embed_dim,
    }


def get_optimal_batch_size(
    model_name: str,
    available_vram_gb: float,
    max_seq_len: int = 512,
    quantization: Optional[str] = None,
) -> int:
    """
    Estimate optimal batch size based on available VRAM.
    
    This is a simple heuristic - actual optimal batch size depends on
    model architecture, sequence length, and CUDA memory fragmentation.
    
    Args:
        model_name: Model name for lookup
        available_vram_gb: Available GPU VRAM in GB
        max_seq_len: Maximum sequence length for prompts
        quantization: Quantization mode
    
    Returns:
        Recommended batch size (conservative estimate)
    
    NOTE: This is a utility function for convenience. For production use,
    consider implementing OOM detection with automatic batch size reduction.
    This is a potential future extension.
    """
    memory_info = estimate_model_memory(model_name, quantization)
    model_vram = memory_info['vram_gb']
    
    # Available memory after model loading
    free_vram = available_vram_gb - model_vram
    
    if free_vram <= 0:
        return 1  # Minimum batch size
    
    # Rough estimate: ~0.5GB per sample for 512 token seq len
    bytes_per_sample = 0.5 * (max_seq_len / 512)
    
    batch_size = int(free_vram / bytes_per_sample)
    
    # Clamp to reasonable range
    return max(1, min(batch_size, 64))


def get_gpu_memory_info(device: str = 'cuda:0') -> Dict[str, float]:
    """
    Get current GPU memory usage information.
    
    Args:
        device: CUDA device string
    
    Returns:
        Dictionary with:
            - total_gb: Total VRAM in GB
            - used_gb: Currently used VRAM in GB
            - free_gb: Available VRAM in GB
    """
    if not torch.cuda.is_available():
        return {'total_gb': 0, 'used_gb': 0, 'free_gb': 0}
    
    device_idx = int(device.split(':')[1]) if ':' in device else 0
    
    total = torch.cuda.get_device_properties(device_idx).total_memory
    reserved = torch.cuda.memory_reserved(device_idx)
    allocated = torch.cuda.memory_allocated(device_idx)
    
    total_gb = total / (1024**3)
    used_gb = allocated / (1024**3)
    free_gb = (total - reserved) / (1024**3)
    
    return {
        'total_gb': total_gb,
        'used_gb': used_gb,
        'free_gb': free_gb,
    }

