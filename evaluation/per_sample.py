"""
Per-sample metrics evaluation module for trained models.

This module provides functionality to compute per-sample metrics on trained model
checkpoints without the overhead of training. Useful for analyzing final model
performance after training completes.
"""

import torch
import os
import json
import glob
import yaml
from typing import Optional
from models import model_init
from data_provider.data_factory import Data_Provider
from utils.tools import dotdict, general_move_to_device
from utils.per_sample_metrics import PerSampleMetricsTracker
from utils.loss_utils import compute_per_sample_loss, compute_per_sample_per_channel_loss
from evaluation.standard import find_checkpoint
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from rich.console import Console


def _forward_step_standalone(iter_data, model, config, device):
    """
    Execute a single forward pass through the model (standalone version).
    
    This function mimics the _forward_step method from Experiment class but works
    without requiring an Experiment instance. Handles all task types (TSF, TGTSF, MTSF).
    
    Args:
        iter_data: Batch tuple containing:
            - sample_ids: Sample identifiers
            - batch_x: Input sequences
            - batch_y: Target sequences
            - timestamp_x, timestamp_y: Timestamps
            - batch_x_hetero, batch_y_hetero: Heterogeneous data
            - hetero_x_time, hetero_y_time: Heterogeneous timestamps
            - hetero_general: General heterogeneous info
            - hetero_channel: Channel-specific heterogeneous info
        model: Trained model instance
        config: Configuration object
        device: PyTorch device
        
    Returns:
        tuple: (predictions, ground_truth, sample_ids)
    """
    # Unpack batch tuple
    sample_ids, batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = iter_data
    
    # Convert to tensors if needed
    if not isinstance(batch_x, torch.Tensor):
        batch_x = torch.tensor(batch_x)
    if not isinstance(batch_y, torch.Tensor):
        batch_y = torch.tensor(batch_y)
    
    # Move tensors to device (use model's move_to_device if available, otherwise use general function)
    if hasattr(model, 'move_to_device'):
        batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = model.move_to_device(
            batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, 
            hetero_x_time, hetero_y_time, hetero_general, hetero_channel, device
        )
    else:
        # Use general move_to_device for standard models
        batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = general_move_to_device(
            batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero,
            hetero_x_time, hetero_y_time, hetero_general, hetero_channel, device
        )
    
    # Handle different task types (data already moved to device by move_to_device)
    if config.task == 'TSF':
        # Time series forecasting only
        prediction = model(x=batch_x)
    elif config.task == 'TGTSF':
        # Text-guided time series forecasting
        prediction = model(x=batch_x, news=batch_y_hetero, channel_description=hetero_channel)
    elif config.task == 'MTSF':
        # Multimodal time series forecasting
        prediction = model(x=batch_x, historical_events=batch_x_hetero)
    else:
        # Default: try full model signature (most flexible)
        prediction = model(
            x=batch_x,
            historical_events=batch_x_hetero,
            news=batch_y_hetero,
            dataset_description=hetero_general,
            channel_description=hetero_channel
        )
    
    # Extract output sequence (last output_len timesteps)
    prediction = prediction[:, -config.output_len:, :]
    
    return prediction, batch_y, sample_ids


def _get_channel_names(num_channels, config):
    """
    Get channel names for per-channel metrics tracking.
    
    Args:
        num_channels: Number of channels/variables
        config: Configuration object
        
    Returns:
        List of channel names (strings)
    """
    # Try to get channel names from data config
    if hasattr(config, 'data_config') and config.data_config:
        if isinstance(config.data_config, dict):
            target = config.data_config.get('target', None)
            if target and isinstance(target, list):
                if len(target) == num_channels:
                    return target
    
    # Fallback: use integer indices
    return [str(i) for i in range(num_channels)]


def _process_batch(iter_data, model, config, device, split_name, epoch, entity_id,
                   metrics_tracker, criterion):
    """
    Process a single batch and add metrics to tracker.
    
    Args:
        iter_data: Batch data from DataLoader
        model: Trained model instance
        config: Configuration object
        device: PyTorch device
        split_name: Split name
        epoch: Epoch number
        entity_id: Entity ID (None for train/val)
        metrics_tracker: PerSampleMetricsTracker instance
        criterion: Loss function
        
    Returns:
        tuple: (batch_loss, batch_size) for loss aggregation
    """
    # Forward pass
    output, gt, sample_ids = _forward_step_standalone(iter_data, model, config, device)
    current_batch_size = gt.size(0)
    
    # Compute batch loss
    loss = criterion(output, gt)
    batch_loss_value = loss.item() * current_batch_size
    
    # Compute per-sample metrics
    per_sample_loss = compute_per_sample_loss(output, gt, criterion)
    per_sample_per_channel_loss = compute_per_sample_per_channel_loss(output, gt, criterion)
    
    # Extract timestamps from batch tuple (index 3 after sample_ids)
    timestamps = iter_data[3] if len(iter_data) > 3 else None
    num_channels = per_sample_per_channel_loss.shape[1]
    channel_names = _get_channel_names(num_channels, config)
    
    # Add to metrics tracker
    metrics_tracker.add_batch(
        epoch=epoch,
        split=split_name,
        entity_id=entity_id,
        sample_ids=sample_ids,
        timestamps=timestamps[:, 0] if timestamps is not None else None,
        losses=per_sample_loss,
        channel_ids=channel_names,
        per_channel_losses=per_sample_per_channel_loss
    )
    
    return batch_loss_value, current_batch_size


def process_split_with_per_sample_metrics(loader, model, config, device, split_name, 
                                         metrics_tracker, criterion, epoch=0, 
                                         entity_id=None):
    """
    Process a single data split and compute per-sample metrics.
    
    Handles both single DataLoader (train/val) and dict of loaders (test).
    
    Args:
        loader: DataLoader (for train/val) or dict of DataLoaders (for test)
        model: Trained model instance
        config: Configuration object
        device: PyTorch device
        split_name: Split name ('train', 'val', or 'test')
        metrics_tracker: PerSampleMetricsTracker instance
        criterion: Loss function
        epoch: Epoch number (default 0 for final model)
        entity_id: Entity ID for test split (None for train/val)
        
    Returns:
        float: Average loss for the split
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    
    console = Console()
    
    # Create progress bar components (shared for both cases)
    progress_components = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ]
    
    with torch.no_grad():
        # Handle test split (dict of loaders per entity)
        if isinstance(loader, dict):
            # Test split: process each entity separately
            with Progress(*progress_components, console=console) as progress:
                entity_task = progress.add_task(f"Processing {split_name}", total=len(loader))
                
                for entity_name, entity_loader in loader.items():
                    # Inner progress bar for samples within current entity
                    sample_task = progress.add_task(f"  └─ {entity_name}", total=len(entity_loader))
                    
                    for iter_data in entity_loader:
                        batch_loss, batch_size = _process_batch(
                            iter_data, model, config, device, split_name, epoch,
                            entity_name, metrics_tracker, criterion
                        )
                        running_loss += batch_loss
                        total_samples += batch_size
                        progress.update(sample_task, advance=1)
                    
                    progress.remove_task(sample_task)
                    progress.update(entity_task, advance=1)
        
        else:
            # Train/Val split: single DataLoader
            with Progress(*progress_components, console=console) as progress:
                task = progress.add_task(f"Processing {split_name}", total=len(loader))
                
                for iter_data in loader:
                    batch_loss, batch_size = _process_batch(
                        iter_data, model, config, device, split_name, epoch,
                        entity_id, metrics_tracker, criterion
                    )
                    running_loss += batch_loss
                    total_samples += batch_size
                    progress.update(task, advance=1)
    
    # Return average loss
    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss


def _resolve_config_path(config_path_str):
    """
    Resolve a config file path (absolute or relative to current working directory).
    
    Args:
        config_path_str: Config file path (may be absolute or relative)
        
    Returns:
        str: Resolved absolute path
    """
    if os.path.isabs(config_path_str):
        return config_path_str
    else:
        return os.path.join(os.getcwd(), config_path_str)


def _load_config_file(config_path_str, file_type="config"):
    """
    Load a config file (YAML) and return as dotdict.
    
    Args:
        config_path_str: Path to config file
        file_type: Type of config file (for error messages)
        
    Returns:
        dotdict: Loaded configuration
    """
    resolved_path = _resolve_config_path(config_path_str)
    
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(
            f"{file_type.capitalize()} file not found: {config_path_str}\n"
            f"  Resolved to: {resolved_path}\n"
            f"  Current working directory: {os.getcwd()}\n"
            f"  Note: Use absolute paths in configs for more robust evaluation."
        )
    
    with open(resolved_path, 'r') as f:
        loaded_config = yaml.safe_load(f) or {}
        return dotdict(loaded_config)


def _load_yaml_config(ckpt_path):
    """
    Load experiment config from YAML format.
    
    Args:
        ckpt_path: Path to checkpoint directory
        
    Returns:
        dict: Parsed YAML configuration
    """
    config_path = os.path.join(ckpt_path, 'configs', 'experiment_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _load_legacy_config(ckpt_path):
    """
    Load experiment config from legacy JSON format.
    
    Args:
        ckpt_path: Path to checkpoint directory
        
    Returns:
        dotdict: Parsed configuration
    """
    config_path = os.path.join(ckpt_path, 'args.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found in {ckpt_path}\n"
            f"  Expected: configs/experiment_config.yaml or args.json"
        )
    checkpoint_config = dotdict(json.load(open(config_path)))
    checkpoint_config.model_config = dotdict(checkpoint_config.model_config)
    return checkpoint_config


def _extract_training_params(training_config):
    """
    Extract training parameters with defaults.
    
    Args:
        training_config: Training config dictionary
        
    Returns:
        dict: Training parameters with defaults applied
    """
    return {
        'loss': training_config.get('loss', 'mse'),
        'input_len': training_config.get('input_len', 360),
        'output_len': training_config.get('output_len', 24),
        'batch_size': training_config.get('batch_size', 128),
        'noise': training_config.get('noise', 0.0),
        'scale': training_config.get('scale', True),
        'disable_buffer': training_config.get('disable_buffer', False),
        'preload_hetero': training_config.get('preload_hetero', False),
        'prefetch_factor': training_config.get('prefetch_factor', 2),
        'num_workers': training_config.get('num_workers', 0),
        'sample_step': training_config.get('sample_step', 24),
        'hetero_align_stride': training_config.get('hetero_align_stride', True),
    }


def _load_nested_config(config_value):
    """
    Load a nested config file if config_value is a path.
    
    Args:
        config_value: Either a string path, dict with 'config_path', or dict config
        
    Returns:
        dotdict or str: Loaded config or original value
    """
    # Extract path if it's a dict with config_path
    if isinstance(config_value, dict) and 'config_path' in config_value:
        config_path = config_value.get('config_path')
    elif isinstance(config_value, str):
        config_path = config_value
    else:
        # Already a dict config, wrap in dotdict
        return dotdict(config_value) if config_value else {}
    
    # If we have a path, load the file
    if config_path:
        return _load_config_file(config_path, "config")
    
    return dotdict({})


def _merge_model_config_overrides(checkpoint_config, config_dict):
    """
    Merge model_config overrides from experiment_config.yaml into checkpoint_config.
    
    Args:
        checkpoint_config: Checkpoint configuration to update
        config_dict: Experiment config dictionary
    """
    if 'model_config' not in config_dict or not isinstance(config_dict['model_config'], dict):
        return
    
    model_config_overrides = config_dict['model_config']
    
    # Ensure model_config is a dotdict for attribute access
    if not isinstance(checkpoint_config.model_config, dotdict):
        if isinstance(checkpoint_config.model_config, dict):
            checkpoint_config.model_config = dotdict(checkpoint_config.model_config)
        else:
            checkpoint_config.model_config = dotdict({})
    
    # Merge overrides (only non-empty values)
    for key, value in model_config_overrides.items():
        if value:  # Only override if value is not empty/None/empty string
            checkpoint_config.model_config[key] = value


def _validate_pretrained_model_path(checkpoint_config):
    """
    Validate and resolve pretrained_model_path if present in model config.
    
    Args:
        checkpoint_config: Checkpoint configuration to validate
    """
    if not hasattr(checkpoint_config.model_config, 'pretrained_model_path'):
        return
    
    pretrained_path = checkpoint_config.model_config.pretrained_model_path
    if not pretrained_path:
        return
    
    if not os.path.isabs(pretrained_path):
        print(f"[Warning] pretrained_model_path is not absolute: {pretrained_path}")
        print("  It is much more robust to use absolute paths for pretrained_model_path in configs.")
        print(f"  Will only look relative to current working directory: {os.getcwd()}")
    
    resolved_path = _resolve_config_path(pretrained_path)
    
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(
            f"pretrained_model_path not found: {pretrained_path}\n"
            f"  Resolved to: {resolved_path}\n"
            f"  Current working directory: {os.getcwd()}\n"
            f"  Note: Use absolute paths in model config for more robust evaluation."
        )
    
    # Update the path to the resolved absolute path
    checkpoint_config.model_config.pretrained_model_path = os.path.abspath(resolved_path)


def _load_checkpoint_config(ckpt_path, eval_config, config):
    """
    Load and prepare configuration from checkpoint directory.
    
    Args:
        ckpt_path: Path to checkpoint directory
        eval_config: Evaluation configuration
        config: Original config (may contain data_config override)
        
    Returns:
        dotdict: Prepared checkpoint configuration
    """
    # Load configuration from checkpoint folder
    # Try new format first (configs/experiment_config.yaml), then legacy (args.json)
    config_path = os.path.join(ckpt_path, 'configs', 'experiment_config.yaml')
    if os.path.exists(config_path):
        # New format: YAML config
        config_dict = _load_yaml_config(ckpt_path)
        
        # Extract model and data config paths
        model_config_path = config_dict.get('model', {}).get('config_path', {})
        data_config_path = config_dict.get('data', {}).get('config_path', {})
        
        # Initialize config values (will be loaded if they're paths)
        model_config_val = model_config_path if isinstance(model_config_path, str) else (
            dotdict(model_config_path) if isinstance(model_config_path, dict) else {}
        )
        data_config_val = data_config_path if isinstance(data_config_path, str) else (
            dotdict(data_config_path) if isinstance(data_config_path, dict) else {}
        )
        
        # Extract training config with defaults
        training_config = config_dict.get('training', {})
        training_params = _extract_training_params(training_config)
        
        # Create base checkpoint config
        checkpoint_config = dotdict({
            'model': config_dict.get('model', {}).get('name', 'unknown'),
            'model_config': model_config_val,
            'data': config_dict.get('data', {}).get('name', 'unknown'),
            'data_config': data_config_val,
            'task': config_dict.get('experiment', {}).get('type', 'TSF'),
            **training_params
        })
        
        # Load actual model and data configs if they're paths
        checkpoint_config.model_config = _load_nested_config(checkpoint_config.model_config)
        checkpoint_config.data_config = _load_nested_config(checkpoint_config.data_config)
        
        # Merge model_config overrides from experiment_config.yaml
        _merge_model_config_overrides(checkpoint_config, config_dict)
    else:
        # Legacy format: args.json
        checkpoint_config = _load_legacy_config(ckpt_path)
    
    # Use provided data_config override if available, otherwise use checkpoint's data_config
    if hasattr(config, 'data_config') and config.data_config:
        checkpoint_config.data_config = _load_config_file(config.data_config, "data config")
    else:
        checkpoint_config.data_config = dotdict(checkpoint_config.data_config)
    
    # Validate pretrained_model_path if present
    _validate_pretrained_model_path(checkpoint_config)
    
    # Set evaluation-specific config values
    checkpoint_config.gpu = eval_config.device
    checkpoint_config.num_workers = 0  # Disable multiprocessing for evaluation
    checkpoint_config.task = eval_config.task
    checkpoint_config.batch_size = eval_config.batch_size
    
    return checkpoint_config


def _setup_device(eval_config):
    """
    Setup and return the appropriate device for evaluation.
    
    Args:
        eval_config: Evaluation configuration
        
    Returns:
        torch.device: Device to use for evaluation
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{eval_config.device}")
    else:
        device = torch.device("cpu")
        print("[Warning] CUDA is not available, use CPU instead.")
    print(f"[Info] Running on device: {device}")
    return device


def _load_model_from_checkpoint(ckpt_path, checkpoint_config, device):
    """
    Load model and weights from checkpoint.
    
    Args:
        ckpt_path: Path to checkpoint directory
        checkpoint_config: Model configuration
        device: Target device
        
    Returns:
        torch.nn.Module: Loaded model in eval mode
    """
    # Initialize model
    model = model_init(checkpoint_config.model, checkpoint_config.model_config, checkpoint_config).to(device)
    
    # Find checkpoint file in checkpoints subdirectory (filter out directories, only keep files)
    # Prefer best_checkpoint.* if it exists, otherwise use checkpoint.*
    checkpoints_dir = os.path.join(ckpt_path, 'checkpoints')
    best_ckpt_files = [f for f in glob.glob(os.path.join(checkpoints_dir, 'best_checkpoint*')) if os.path.isfile(f)]
    if best_ckpt_files:
        ckpt_file_path = best_ckpt_files[0]
    else:
        ckpt_files = [f for f in glob.glob(os.path.join(checkpoints_dir, 'checkpoint*')) if os.path.isfile(f)]
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint file (e.g., 'checkpoint.pth' or 'best_checkpoint.pth') found in {checkpoints_dir}")
        ckpt_file_path = ckpt_files[0]
    
    print(f"[Info] Loading model from: {ckpt_file_path}")
    checkpoint = torch.load(ckpt_file_path, map_location=device)
    
    # Handle different checkpoint formats
    if ckpt_file_path.endswith('.ckpt') and 'state_dict' in checkpoint:
        # Lightning checkpoint
        if checkpoint_config.model == "PatchTST":
            state_dict = {key.replace("model.model.", "model."): value for key, value in checkpoint['state_dict'].items()}
        else:
            state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Checkpoint format with model_state_dict key (common in training checkpoints)
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # Standard PyTorch checkpoint with state_dict key
        state_dict = checkpoint['state_dict']
    else:
        # Direct state dict (uncommon)
        state_dict = checkpoint
    
    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    print(f'[Info] Successfully loaded model: {checkpoint_config.model}')
    
    return model


def _initialize_loss_function(checkpoint_config):
    """
    Initialize loss function based on checkpoint configuration.
    
    Args:
        checkpoint_config: Checkpoint configuration
        
    Returns:
        torch.nn.Module: Loss function (criterion)
    """
    loss_name = getattr(checkpoint_config, 'loss', 'mse').lower()
    if loss_name == 'mse' or loss_name == 'mean_squared_error':
        criterion = torch.nn.MSELoss()
    elif loss_name == 'l1' or loss_name == 'mean_absolute_error' or loss_name == 'mae':
        criterion = torch.nn.L1Loss()
    else:
        print(f"[Warning] Unknown loss function '{loss_name}', defaulting to MSE")
        criterion = torch.nn.MSELoss()
    return criterion


def _get_split_loader(data_provider, split_name):
    """
    Get data loader for specified split.
    
    Args:
        data_provider: Data_Provider instance
        split_name: Name of split ('train', 'val', or 'test')
        
    Returns:
        DataLoader or dict: Loader(s) for the split
        
    Raises:
        ValueError: If split_name is not recognized
    """
    if split_name == 'train':
        return data_provider.get_train(return_type='loader')
    elif split_name == 'val':
        return data_provider.get_val(return_type='loader')
    elif split_name == 'test':
        return data_provider.get_test(return_type='loader')
    else:
        raise ValueError(f"Unknown split '{split_name}'. Must be 'train', 'val', or 'test'.")


def _process_all_splits(splits_to_process, data_provider, model, checkpoint_config, 
                        device, metrics_tracker, criterion):
    """
    Process all requested splits and compute per-sample metrics.
    
    Args:
        splits_to_process: List of split names to process
        data_provider: Data_Provider instance
        model: Trained model
        checkpoint_config: Configuration object
        device: Target device
        metrics_tracker: PerSampleMetricsTracker instance
        criterion: Loss function
        
    Returns:
        dict: Mapping of split names to average losses
    """
    results = {}
    print(f"\n[Info] Processing splits: {splits_to_process}")
    
    for split_name in splits_to_process:
        print(f"\n[Info] Processing {split_name} split...")
        
        try:
            # Get loader for this split
            loader = _get_split_loader(data_provider, split_name)
        except ValueError as e:
            print(f"[Warning] {e}, skipping...")
            continue
        
        # Process split and compute per-sample metrics
        avg_loss = process_split_with_per_sample_metrics(
            loader=loader,
            model=model,
            config=checkpoint_config,
            device=device,
            split_name=split_name,
            metrics_tracker=metrics_tracker,
            criterion=criterion,
            epoch=0  # Use epoch 0 to represent "final model"
        )
        
        results[split_name] = avg_loss
        print(f"[Info] {split_name} split - Average loss: {avg_loss:.7f}")
    
    return results


def _print_summary(results, metrics_output_dir):
    """
    Print evaluation summary and metrics save location.
    
    Args:
        results: Dictionary mapping split names to average losses
        metrics_output_dir: Directory where metrics were saved
    """
    print("\n" + "="*50)
    print(" " * 15 + "Per-Sample Metrics Summary")
    print("="*50)
    for split_name, loss in results.items():
        print(f"  {split_name:>6} Loss: {loss:.7f}")
    print("="*50)
    print(f"[Info] Per-sample metrics saved to: {os.path.join(metrics_output_dir, 'per_sample', 'epoch_000.parquet')}")


def evaluate_per_sample(config, checkpoint_path: Optional[str] = None):
    """
    Evaluate a trained model checkpoint and compute per-sample metrics.
    
    This function loads a trained model checkpoint, runs inference on train/val/test
    splits, and computes per-sample metrics without any training overhead. Metrics
    are saved to parquet files in the checkpoint directory.
    
    Args:
        config: dotdict containing evaluation configuration with structure:
            - evaluation: {model, data, version, input_len, output_len, checkpoint_base,
                          batch_size, task, device, splits}
            - data_config: Optional path to override data config
            - splits: Optional list of splits to process (default: ['train', 'val', 'test'])
        checkpoint_path: Optional direct path to checkpoint directory. If provided,
                        bypasses find_checkpoint and uses this path directly.
    
    Returns:
        dict: Mapping of split names to average losses
    
    Example:
        >>> from cli.config.loader import load_config_with_nested
        >>> config_hierarchy = load_config_with_nested("configs/experiments/dlinear_solar.yaml")
        >>> evaluate_per_sample(config_hierarchy['primary'])
        
        >>> # Or with direct checkpoint path
        >>> evaluate_per_sample(config, checkpoint_path="/path/to/experiment")
    """
    eval_config = config.evaluation
    
    # Find checkpoint path (use direct path if provided, otherwise search)
    if checkpoint_path:
        ckpt_path = checkpoint_path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")
    else:
        ckpt_path = find_checkpoint(eval_config)
    print(f"[Info] Using checkpoint path: {ckpt_path}")
    checkpoint_config = _load_checkpoint_config(ckpt_path, eval_config, config)
    
    # Determine which splits to process
    splits_to_process = getattr(eval_config, 'splits', ['train', 'val', 'test'])
    if isinstance(splits_to_process, str):
        splits_to_process = [splits_to_process]
    
    # Setup device and load model
    device = _setup_device(eval_config)
    model = _load_model_from_checkpoint(ckpt_path, checkpoint_config, device)
    
    # Initialize loss function and metrics tracker
    criterion = _initialize_loss_function(checkpoint_config)
    metrics_output_dir = os.path.join(ckpt_path, 'metrics')
    metrics_tracker = PerSampleMetricsTracker(metrics_output_dir)
    print(f"[Info] Per-sample metrics will be saved to: {metrics_output_dir}/per_sample/")
    
    # Load data and process splits
    data_provider = Data_Provider(checkpoint_config, buffer=False)
    results = _process_all_splits(
        splits_to_process, data_provider, model, checkpoint_config,
        device, metrics_tracker, criterion
    )
    
    # Save metrics and print summary
    print("\n[Info] Saving per-sample metrics...")
    metrics_tracker.save_epoch(0)  # Save as epoch 0 (final model)
    _print_summary(results, metrics_output_dir)
    
    return results

