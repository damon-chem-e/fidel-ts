"""
Pydantic models for configuration structure.

This module defines the complete type-safe configuration structure using Pydantic,
replacing the previous dotdict-based system.

Model-specific training configs are in cli/config/model_training.py
"""

from typing import Optional, Union, List, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field, model_validator, ConfigDict
import yaml

# Import model-specific training configs from dedicated module
from cli.config.model_training import LeRetTrainingConfig, TimeLLMTrainingConfig


# =============================================================================
# Base Configuration Models
# =============================================================================

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
    
    # PyTorch Compile (PyTorch 2.0+ optimization)
    torch_compile: bool = Field(default=False, description="Enable torch.compile for faster training (PyTorch 2.0+)")
    compile_mode: str = Field(default="reduce-overhead", description="torch.compile mode (default/reduce-overhead/max-autotune)")
    
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
    
    # Purge period truncation to remove lookahead bias
    truncate_train_for_purge: bool = Field(
        default=False,
        description=(
            "If True, truncate training data by pred_len to remove lookahead bias. "
            "This ensures training predictions don't overlap with validation data, "
            "making validation loss reliable for hyperparameter tuning. "
            "See docs/train_val_test_purge_period_issue.md for details."
        )
    )

    # Tensor cache for fast data loading
    use_tensor_cache: bool = Field(
        default=False,
        description=(
            "If True, use pre-generated tensor cache for ultra-fast data loading. "
            "Cache must be generated first with: python -m cli.tensor_cache generate <config>. "
            "See context/performance_optimization/training_optimization_plan.md for details."
        )
    )
    tensor_cache_dir: Optional[str] = Field(
        default=None,
        description=(
            "Path to tensor cache directory. If None, auto-generated based on dataset and task config. "
            "Default location: ./tensor_cache/{data_name}_{input_len}_{output_len}/"
        )
    )
    
    # Text embedding stride for multimodal models (lynx, lynx_film, TGTSF, etc.)
    text_embedding_stride: Optional[Union[str, int]] = Field(
        default=None,
        description=(
            "Controls text embedding temporal resolution. Options:\n"
            "  - None/'aligned' (default): Use model_config.stride if hetero_align_stride=True, else 1\n"
            "  - 'full': Always use stride=1 (full resolution, recommended for lynx_film_raw)\n"
            "  - <int>: Explicit stride value (e.g., 3 for every 3rd timestep)\n"
            "Full resolution preserves all text information but uses more memory (~15MB/batch extra).\n"
            "See docs/planning/hetero_stride_optional_plan.md for details."
        )
    )

    # LeRet-specific two-stage training configuration
    leret: Optional[LeRetTrainingConfig] = Field(
        default=None,
        description=(
            "LeRet-specific two-stage training configuration. "
            "Only used when model is LeRet. Controls pretrain/finetune stages."
        )
    )
    
    # Time-LLM specific training configuration
    time_llm: Optional[TimeLLMTrainingConfig] = Field(
        default=None,
        description=(
            "Time-LLM specific training configuration. "
            "Only used when model is TimeLLM. Controls LLM backbone and quantization."
        )
    )
    
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

    # System monitoring (GPU, CPU, RAM)
    system_monitoring: bool = Field(
        default=True,
        description="Enable GPU, CPU, and RAM monitoring"
    )
    system_log_interval_s: float = Field(
        default=30.0,
        ge=1.0,
        description="Interval between system metric logs (seconds)"
    )
    system_sample_interval_s: float = Field(
        default=0.5,
        ge=0.1,
        description="Interval between system metric samples (seconds)"
    )

    # Batch logging
    batch_log_interval: int = Field(
        default=10,
        ge=1,
        description="Log batch loss and gradient norm every N batches (1=all, 10=every 10th)"
    )


class LLMEmbeddingConfig(BaseModel):
    """
    LLM embedding configuration for experiments that use LLM-based embeddings.

    This configures how time series data is converted to text prompts and then
    embedded using an LLM (e.g., GPT-2, Qwen) for models like TimeCMA.

    The embeddings are generated using the experiment's input_len/output_len
    to ensure consistency between training and embedding generation.

    Note: This config allows extra fields for inference-specific parameters
    (e.g., memory_efficient, chunk_size, flush_every) that are used by
    cli.inference but ignored during training.
    """
    model_config = ConfigDict(extra="allow")
    
    # LLM Model Settings
    model_name: str = Field(default="gpt2", description="HuggingFace model name (e.g., 'gpt2', 'Qwen/Qwen2.5-7B-Instruct')")
    cache_dir: str = Field(default="./LLM_cache/", description="Directory for LLM model weights cache")
    quantization: Optional[str] = Field(default=None, description="Quantization mode: '4bit', '8bit', or None for fp16")
    
    # Extraction Settings
    extraction_mode: str = Field(default="last_token", description="Embedding extraction: 'last_token' or 'pooled'")
    max_length: int = Field(default=512, ge=1, description="Maximum token length for LLM input")
    
    # Prompt Settings
    prompt_template: str = Field(default="timecma_v1", description="Prompt template name: 'timecma_v1' or 'simple'")
    prompt_config: Dict[str, Any] = Field(default_factory=lambda: {"value_format": "integer", "include_timestamps": True}, description="Prompt template configuration")
    
    # Batch Processing
    batch_size: int = Field(default=64, ge=1, description="Batch size for LLM inference")


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""
    model: ModelConfig = Field(..., description="Model configuration")
    data: DataConfig = Field(..., description="Data configuration")
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="Training configuration")
    device: DeviceConfig = Field(default_factory=DeviceConfig, description="Device configuration")
    wandb: WandBConfig = Field(default_factory=WandBConfig, description="WandB configuration")
    llm_embedding: Optional[LLMEmbeddingConfig] = Field(default=None, description="LLM embedding configuration (for TimeCMA-style models)")
    
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

