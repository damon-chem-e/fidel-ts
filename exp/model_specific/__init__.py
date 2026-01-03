"""
Model-specific experiment runners and training dispatchers.

This module provides a clean abstraction for models that require special
training logic beyond the standard Lightning or PyTorch training loops.

Examples:
- LeRet: Two-stage training (pretrain + finetune)
- Future models with special requirements

Usage:
    from exp.model_specific import get_model_trainer, has_custom_trainer

    # Check if model needs custom handling
    if has_custom_trainer(model_name, framework="lightning"):
        trainer_fn = get_model_trainer(model_name, framework="lightning")
        return trainer_fn(args, exp_manager)
    else:
        # Use default training
        return train_lightning_model(args, exp_manager)
"""

from typing import Callable, Optional, Dict, Any
from pathlib import Path


# =============================================================================
# Model Trainer Registry
# =============================================================================
# Maps (model_name, framework) -> training function
# Framework can be "lightning" or "pytorch"

_MODEL_TRAINERS: Dict[str, Dict[str, Callable]] = {}


def register_model_trainer(
    model_name: str, 
    framework: str, 
    trainer_fn: Callable
) -> None:
    """
    Register a custom training function for a specific model and framework.
    
    Args:
        model_name: Name of the model (e.g., "LeRet")
        framework: Training framework ("lightning" or "pytorch")
        trainer_fn: Training function with signature (args, exp_manager) -> Path
    """
    if model_name not in _MODEL_TRAINERS:
        _MODEL_TRAINERS[model_name] = {}
    _MODEL_TRAINERS[model_name][framework] = trainer_fn


def has_custom_trainer(model_name: str, framework: str = "lightning") -> bool:
    """
    Check if a model has a custom trainer registered for the given framework.
    
    Args:
        model_name: Name of the model
        framework: Training framework ("lightning" or "pytorch")
    
    Returns:
        True if custom trainer exists, False otherwise
    """
    return (
        model_name in _MODEL_TRAINERS and 
        framework in _MODEL_TRAINERS[model_name]
    )


def get_model_trainer(
    model_name: str, 
    framework: str = "lightning"
) -> Optional[Callable]:
    """
    Get the custom training function for a model and framework.
    
    Args:
        model_name: Name of the model
        framework: Training framework ("lightning" or "pytorch")
    
    Returns:
        Training function if registered, None otherwise
    """
    if has_custom_trainer(model_name, framework):
        return _MODEL_TRAINERS[model_name][framework]
    return None


def list_custom_models() -> Dict[str, list]:
    """
    List all models with custom trainers and their supported frameworks.
    
    Returns:
        Dict mapping model names to list of supported frameworks
    """
    return {
        model: list(frameworks.keys()) 
        for model, frameworks in _MODEL_TRAINERS.items()
    }


# =============================================================================
# Register Model-Specific Trainers
# =============================================================================

def _register_all_trainers():
    """
    Register all model-specific trainers.
    
    Called at module import time to populate the registry.
    """
    # LeRet: Two-stage training (pretrain + finetune)
    from exp.model_specific.leret import (
        train_leret_lightning,
        train_leret_pytorch
    )
    
    register_model_trainer("LeRet", "lightning", train_leret_lightning)
    register_model_trainer("LeRet", "pytorch", train_leret_pytorch)
    
    # Time-LLM: Frozen LLM backbone with reprogramming
    from exp.model_specific.time_llm import (
        train_time_llm_lightning,
        train_time_llm_pytorch
    )
    
    register_model_trainer("TimeLLM", "lightning", train_time_llm_lightning)
    register_model_trainer("TimeLLM", "pytorch", train_time_llm_pytorch)


# Auto-register trainers when module is imported
_register_all_trainers()


# =============================================================================
# Convenience Exports
# =============================================================================

__all__ = [
    'register_model_trainer',
    'has_custom_trainer', 
    'get_model_trainer',
    'list_custom_models',
]

