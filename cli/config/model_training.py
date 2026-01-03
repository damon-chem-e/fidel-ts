"""
Model-specific training configuration classes.

This module contains Pydantic models for model-specific training configurations
that are too specialized to include in the general TrainingConfig class.

Each model with unique training requirements (e.g., multi-stage training,
special loss functions, custom optimizers) should have its config defined here.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class LeRetTrainingConfig(BaseModel):
    """
    Configuration for LeRet two-stage training.
    
    LeRet (Language-Enhanced Retention Network) uses a curriculum learning 
    approach with two distinct training stages:
    
    Stage 1 (Pretrain): Auto-regressive pretraining
        - Uses patch_head output for reconstruction loss
        - Trains the model to reconstruct input patches
        - Captures local temporal patterns
        - Loss computed between auto_y and patchified input
    
    Stage 2 (Finetune): Forecasting finetuning
        - Uses sequence_head output for prediction loss
        - Trains for the actual forecasting task
        - Leverages pretrained representations
        - Loss computed between forecast and ground truth
    
    Both stages must occur within the same experiment for consistency and 
    reproducibility. The training_stage parameter controls which stage(s) to run.
    
    Example YAML config:
        training:
          leret:
            training_stage: "both"
            pretrain_epochs: 10
            pretrain_loss: "mse"
            finetune_loss: "mse"
    
    Attributes:
        training_stage: Which stage(s) to run
        pretrain_epochs: Number of epochs for Stage 1
        pretrain_loss: Loss function for pretrain stage
        finetune_loss: Loss function for finetune stage
        pretrain_checkpoint: Explicit path to pretrain checkpoint (optional)
        pretrain_learning_rate: Optional different LR for pretrain stage
    """
    model_config = ConfigDict(extra="forbid")
    
    training_stage: Literal["pretrain", "finetune", "both"] = Field(
        default="both",
        description=(
            "Which training stage(s) to run: "
            "'pretrain' (Stage 1 only - auto-regressive pretraining), "
            "'finetune' (Stage 2 only - requires pretrain checkpoint), "
            "'both' (run pretrain then finetune sequentially)"
        )
    )
    
    pretrain_epochs: int = Field(
        default=10,
        ge=1,
        description="Number of epochs for Stage 1 (auto-regressive pretraining)"
    )
    
    pretrain_loss: Literal["mse", "mae"] = Field(
        default="mse",
        description="Loss function for pretrain stage (applied to patch_head output)"
    )
    
    finetune_loss: Literal["mse", "mae"] = Field(
        default="mse",
        description="Loss function for finetune stage (applied to sequence_head output)"
    )
    
    pretrain_checkpoint: Optional[str] = Field(
        default=None,
        description=(
            "Path to pretrain checkpoint. Required if training_stage='finetune' "
            "and checkpoint is not auto-detected from same experiment. "
            "If within same experiment, auto-detected from checkpoints/pretrain_checkpoint.ckpt"
        )
    )
    
    pretrain_learning_rate: Optional[float] = Field(
        default=None,
        gt=0,
        description="Learning rate for pretrain stage. If None, uses training.learning_rate"
    )


# =============================================================================
# Future model-specific configs can be added here
# =============================================================================
# 
# class SomeOtherModelConfig(BaseModel):
#     """Config for another model with special training requirements."""
#     ...


# =============================================================================
# Model-Specific Config Registry and Utilities
# =============================================================================

# Registry mapping model names to their training config attribute names
# Format: {model_name: (attribute_name_in_TrainingConfig, ConfigClass)}
_MODEL_CONFIG_REGISTRY: dict = {
    "LeRet": ("leret", LeRetTrainingConfig),
    # Future models:
    # "SomeModel": ("some_model", SomeModelConfig),
}


def get_model_config_attribute(model_name: str) -> Optional[str]:
    """
    Get the TrainingConfig attribute name for a model's config.
    
    Args:
        model_name: Name of the model (e.g., "LeRet")
    
    Returns:
        Attribute name (e.g., "leret") or None if no special config
    """
    if model_name in _MODEL_CONFIG_REGISTRY:
        return _MODEL_CONFIG_REGISTRY[model_name][0]
    return None


def has_model_specific_config(model_name: str) -> bool:
    """
    Check if a model has model-specific training configuration.
    
    Args:
        model_name: Name of the model
    
    Returns:
        True if model has special training config
    """
    return model_name in _MODEL_CONFIG_REGISTRY


def extract_model_config(training_config, model_name: str) -> Optional[dict]:
    """
    Extract model-specific configuration from a TrainingConfig.
    
    This function provides a clean abstraction for extracting model-specific
    configs without hardcoding model names in the training runners.
    
    Args:
        training_config: TrainingConfig instance (from ExperimentConfig.training)
        model_name: Name of the model (e.g., "LeRet")
    
    Returns:
        Dictionary of model-specific config values, or None if no config exists
    
    Example:
        >>> model_config = extract_model_config(config.training, "LeRet")
        >>> if model_config:
        >>>     args.leret = model_config
    """
    attr_name = get_model_config_attribute(model_name)
    if attr_name is None:
        return None
    
    # Get the config from TrainingConfig
    model_config = getattr(training_config, attr_name, None)
    if model_config is None:
        return None
    
    # Convert to dict if it's a Pydantic model
    if hasattr(model_config, 'model_dump'):
        return model_config.model_dump()
    elif isinstance(model_config, dict):
        return model_config
    
    return None


def apply_model_configs_to_args(args, training_config, model_name: str) -> None:
    """
    Apply all model-specific configs from TrainingConfig to args object.
    
    This modifies args in-place, adding model-specific config attributes.
    
    Args:
        args: Arguments object (dotdict) to modify
        training_config: TrainingConfig instance
        model_name: Name of the model
    
    Example:
        >>> apply_model_configs_to_args(args, config.training, config.model.name)
        >>> # args.leret is now set if model is LeRet
    """
    attr_name = get_model_config_attribute(model_name)
    if attr_name:
        config_dict = extract_model_config(training_config, model_name)
        setattr(args, attr_name, config_dict)


def list_models_with_configs() -> list:
    """
    List all models that have special training configurations.
    
    Returns:
        List of model names with registered configs
    """
    return list(_MODEL_CONFIG_REGISTRY.keys())

