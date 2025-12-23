"""
Pydantic models for configuration structure.

This module defines the complete type-safe configuration structure using Pydantic,
replacing the previous dotdict-based system.
"""

from typing import Optional, Union, List, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import yaml


class ModelConfig(BaseModel):
    """Model configuration section."""
    model_config = ConfigDict(extra="forbid")  # Don't allow extra fields
    
    name: str = Field(..., description="Model name (e.g., DLinear, TGTSF)")
    config_path: str = Field(..., description="Path to model-specific config file")


class DataConfig(BaseModel):
    """Data configuration section."""
    model_config = ConfigDict(extra="forbid")
    
    name: str = Field(..., description="Dataset name")
    config_path: str = Field(..., description="Path to data-specific config file")


class TrainingConfig(BaseModel):
    """Training configuration section."""
    # Experiment output directory (base directory for experiment outputs)
    experiment_output: Optional[str] = Field(default=None, description="Base directory for experiment outputs")
    last_ckpt: Optional[str] = Field(default=None, description="Path to last checkpoint for resuming")
    
    # Training hyperparameters
    epochs: int = Field(default=20, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=96, ge=1, description="Batch size")
    learning_rate: float = Field(default=5e-4, gt=0, description="Initial learning rate")
    patience: int = Field(default=3, ge=1, description="Early stopping patience")
    loss: str = Field(default="mse", description="Loss function (mse, l1)")
    lradj: str = Field(default="type3", description="Learning rate adjustment strategy")
    
    # Task definition (mutually exclusive with input_len/output_len)
    ahead: Optional[str] = Field(default=None, description="Ahead task (day/week/month)")
    input_len: Optional[Union[int, str]] = Field(default=None, description="Input sequence length")
    output_len: Optional[int] = Field(default=None, description="Output sequence length")
    
    # Data loading
    scale: bool = Field(default=True, description="Whether to scale the data")
    disable_buffer: bool = Field(default=False, description="Disable data buffer")
    preload_hetero: bool = Field(default=False, description="Preload heterogeneous data")
    prefetch_factor: int = Field(default=2, ge=1, description="Dataloader prefetch factor")
    num_workers: int = Field(default=0, ge=0, description="Number of dataloader workers")
    noise: float = Field(default=0.0, ge=0, description="Noise level for data augmentation")
    downsample: Optional[int] = Field(default=None, description="Downsampling factor")
    
    # LLM-specific
    filtered_samples: Optional[str] = Field(default=None, description="Path to filtered samples JSON")
    sample_step: int = Field(default=24, ge=1, description="Sample step for LLM inference")
    no_parallel: bool = Field(default=False, description="Disable parallel processing for LLM")
    valisets: str = Field(default="full", description="Validation set configuration")
    
    # Lightning-specific
    precision: Optional[str] = Field(default=None, description="Training precision (32/16/bf16)")
    gradient_clip_val: Optional[float] = Field(default=None, ge=0, description="Gradient clipping value")
    test_after_epoch: bool = Field(default=False, description="Run test after each epoch")
    
    # FM-specific
    individual: Optional[bool] = Field(default=None, description="Use individual parameters per channel")
    
    # Tracking per sample metrics
    track_per_sample: bool = Field(default=False, description="Whether to track per-sample metrics (Parquet)")
    
    # Evaluation during training
    evaluate_test_during_training: bool = Field(default=False, description="Whether to evaluate on test set during training epochs (default: False to hold out test)")
    
    model_config = ConfigDict(extra="allow")  # Allow extra fields for flexibility
    
    @model_validator(mode='after')
    def validate_task_definition(self):
        """Ensure either ahead or input_len/output_len is specified."""
        if self.ahead is None and (self.input_len is None or self.output_len is None):
            # This is okay - some models might have defaults
            pass
        return self


class DeviceConfig(BaseModel):
    """Device configuration section."""
    model_config = ConfigDict(extra="forbid")
    
    use_gpu: bool = Field(default=True, description="Whether to use GPU")
    gpu: int = Field(default=0, ge=0, description="GPU device ID")
    use_multi_gpu: bool = Field(default=False, description="Whether to use multiple GPUs")
    devices: str = Field(default="0", description="Comma-separated GPU device IDs")


class WandBConfig(BaseModel):
    """Weights & Biases (wandb) configuration section."""
    model_config = ConfigDict(extra="forbid")
    
    project: str = Field(default="fidel-ts", description="WandB project name")
    entity: Optional[str] = Field(default=None, description="WandB entity/team name (optional)")
    run_name: Optional[str] = Field(default=None, description="WandB run name (defaults to experiment_id if not specified)")
    run_id: Optional[str] = Field(default=None, description="WandB run ID for resuming/overwriting runs")
    tags: List[str] = Field(default_factory=list, description="Tags for experiment organization")
    notes: Optional[str] = Field(default=None, description="Notes/description for the experiment")
    enabled: bool = Field(default=True, description="Whether to enable wandb logging")
    mode: str = Field(default="online", description="WandB mode: online, offline, or disabled")


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""
    model: ModelConfig = Field(..., description="Model configuration")
    data: DataConfig = Field(..., description="Data configuration")
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="Training configuration")
    device: DeviceConfig = Field(default_factory=DeviceConfig, description="Device configuration")
    wandb: WandBConfig = Field(default_factory=WandBConfig, description="WandB configuration")
    
    # Optional fields
    hf_mirror: bool = Field(default=False, description="Use HuggingFace mirror")
    hf_offline: bool = Field(default=False, description="Use HuggingFace offline mode")
    base_data_path: Optional[str] = Field(default=None, description="Base path to replace './data' in all data config paths")
    
    # Optional nested configs (as string paths)
    plotting: Optional[str] = Field(default=None, description="Path to plotting config")
    evaluation: Optional[str] = Field(default=None, description="Path to evaluation config")
    
    # Model config overrides (optional, allows overriding parameters from model config YAML)
    # These override values in the model config file specified by model.config_path
    # Example: {"enc_in": 4, "input_text_dim": 768} to override enc_in and input_text_dim
    model_config_overrides: Optional[Dict[str, Any]] = Field(default=None, description="Model configuration parameter overrides (e.g., enc_in, input_text_dim)")
    
    # Experiment metadata (optional, can be set by ExperimentManager)
    experiment_name: Optional[str] = Field(default=None, description="Experiment name")
    job_id: Optional[str] = Field(default=None, description="Job ID for cluster/scheduler")
    job_name: Optional[str] = Field(default=None, description="Job name for cluster/scheduler")
    random_seed: int = Field(default=2021, description="Random seed for reproducibility")
    resume_experiment_id: Optional[str] = Field(default=None, description="Experiment ID to resume (must match exactly, including timestamp)")
    resume_suite_id: Optional[str] = Field(default=None, description="Suite directory name to resume (required when resuming suite experiments, e.g., 'tgtsf_test_20251207_145025')")
    mark_last_job_complete: bool = Field(default=False, description="Mark the last running job as complete/timeout (required when resuming if previous job timed out)")
    
    model_config = ConfigDict(extra="allow")  # Allow extra fields for nested configs

    @model_validator(mode="before")
    @classmethod
    def forbid_legacy_model_config_key(cls, data: Any) -> Any:
        """
        Forbid the legacy top-level key 'model_config'.

        In Pydantic v2, 'model_config' is a reserved attribute used to configure model behavior.
        Historically, this repo used a user-provided YAML key named 'model_config' to mean
        "model config overrides", but that name collides with Pydantic internals and is unsafe.

        Users must use 'model_config_overrides' instead.
        """
        if isinstance(data, dict) and "model_config" in data:
            raise ValueError(
                "Legacy config key 'model_config' is not supported. "
                "Use 'model_config_overrides' for model YAML overrides."
            )
        return data
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'ExperimentConfig':
        """
        Load experiment configuration from YAML file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            ExperimentConfig instance
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            raise ValueError(f"Config file is empty: {config_path}")
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump(exclude_none=True)
    
    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """
        Save config to YAML file.
        
        Args:
            output_path: Path to output YAML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

