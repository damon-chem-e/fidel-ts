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
from tqdm import tqdm
from models import model_init
from data_provider.data_factory import Data_Provider
from utils.tools import dotdict, general_move_to_device
from utils.per_sample_metrics import PerSampleMetricsTracker
from utils.loss_utils import compute_per_sample_loss, compute_per_sample_per_channel_loss
from evaluation.standard import find_checkpoint


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


def process_split_with_per_sample_metrics(loader, model, config, device, split_name, 
                                         metrics_tracker, criterion, epoch=0, 
                                         entity_id=None, progress_bar=None):
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
        progress_bar: Optional tqdm progress bar
        
    Returns:
        float: Average loss for the split
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        # Handle test split (dict of loaders per entity)
        if isinstance(loader, dict):
            # Test split: process each entity separately
            for entity_name, entity_loader in loader.items():
                entity_running_loss = 0.0
                entity_total_samples = 0
                
                for iter_data in tqdm(entity_loader, desc=f"  {split_name}/{entity_name}", 
                                     disable=(progress_bar is not None), leave=False):
                    # Forward pass
                    output, gt, sample_ids = _forward_step_standalone(iter_data, model, config, device)
                    current_batch_size = gt.size(0)
                    
                    # Compute batch loss
                    loss = criterion(output, gt)
                    entity_running_loss += loss.item() * current_batch_size
                    entity_total_samples += current_batch_size
                    running_loss += loss.item() * current_batch_size
                    total_samples += current_batch_size
                    
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
                        entity_id=entity_name,  # Use entity name from dict key
                        sample_ids=sample_ids,
                        timestamps=timestamps[:, 0] if timestamps is not None else None,
                        losses=per_sample_loss,
                        channel_ids=channel_names,
                        per_channel_losses=per_sample_per_channel_loss
                    )
        
        else:
            # Train/Val split: single DataLoader
            for iter_data in tqdm(loader, desc=f"  {split_name}", 
                                 disable=(progress_bar is not None), leave=False):
                # Forward pass
                output, gt, sample_ids = _forward_step_standalone(iter_data, model, config, device)
                current_batch_size = gt.size(0)
                
                # Compute batch loss
                loss = criterion(output, gt)
                running_loss += loss.item() * current_batch_size
                total_samples += current_batch_size
                
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
                    entity_id=entity_id,  # None for train/val (will extract from sample_ids)
                    sample_ids=sample_ids,
                    timestamps=timestamps[:, 0] if timestamps is not None else None,
                    losses=per_sample_loss,
                    channel_ids=channel_names,
                    per_channel_losses=per_sample_per_channel_loss
                )
    
    # Return average loss
    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss


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
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        # Extract model and data configs
        # Handle config_path which may be a string (path) or dict
        model_config_path = config_dict.get('model', {}).get('config_path', {})
        data_config_path = config_dict.get('data', {}).get('config_path', {})
        
        # Only wrap in dotdict if it's a dict, otherwise keep as string (will be loaded later)
        model_config_val = model_config_path if isinstance(model_config_path, str) else (dotdict(model_config_path) if isinstance(model_config_path, dict) else {})
        data_config_val = data_config_path if isinstance(data_config_path, str) else (dotdict(data_config_path) if isinstance(data_config_path, dict) else {})
        
        checkpoint_config = dotdict({
            'model': config_dict.get('model', {}).get('name', 'unknown'),
            'model_config': model_config_val,
            'data': config_dict.get('data', {}).get('name', 'unknown'),
            'data_config': data_config_val,
            'task': config_dict.get('experiment', {}).get('type', 'TSF'),
            'loss': config_dict.get('training', {}).get('loss', 'mse'),
            'input_len': config_dict.get('training', {}).get('input_len', 360),
            'output_len': config_dict.get('training', {}).get('output_len', 24),
            'batch_size': config_dict.get('training', {}).get('batch_size', 128),
        })
        # Load actual model and data configs if they're paths
        if isinstance(checkpoint_config.model_config, str) or (isinstance(checkpoint_config.model_config, dict) and 'config_path' in checkpoint_config.model_config):
            model_config_path = checkpoint_config.model_config if isinstance(checkpoint_config.model_config, str) else checkpoint_config.model_config.get('config_path')
            if model_config_path:
                # Resolve path: check if absolute, otherwise look relative to current working directory
                if os.path.isabs(model_config_path):
                    resolved_path = model_config_path
                else:
                    resolved_path = os.path.join(os.getcwd(), model_config_path)
                
                if os.path.exists(resolved_path):
                    with open(resolved_path, 'r') as f:
                        loaded_model_config = yaml.safe_load(f) or {}
                        checkpoint_config.model_config = dotdict(loaded_model_config)
                else:
                    raise FileNotFoundError(
                        f"Model config file not found: {model_config_path}\n"
                        f"  Resolved to: {resolved_path}\n"
                        f"  Current working directory: {os.getcwd()}\n"
                        f"  Note: Use absolute paths in configs for more robust evaluation."
                    )
        
        # Merge any model_config overrides from experiment_config.yaml
        # (e.g., pretrained_model_path may be set in experiment_config.yaml's model_config section)
        # This ensures that overrides from suite configs are properly applied
        if 'model_config' in config_dict and isinstance(config_dict['model_config'], dict):
            model_config_overrides = config_dict['model_config']
            # Ensure model_config is a dotdict for attribute access
            if not isinstance(checkpoint_config.model_config, dotdict):
                if isinstance(checkpoint_config.model_config, dict):
                    checkpoint_config.model_config = dotdict(checkpoint_config.model_config)
                else:
                    checkpoint_config.model_config = dotdict({})
            # Merge overrides
            for key, value in model_config_overrides.items():
                if value:  # Only override if value is not empty/None/empty string
                    checkpoint_config.model_config[key] = value
        
        if isinstance(checkpoint_config.data_config, str) or (isinstance(checkpoint_config.data_config, dict) and 'config_path' in checkpoint_config.data_config):
            data_config_path = checkpoint_config.data_config if isinstance(checkpoint_config.data_config, str) else checkpoint_config.data_config.get('config_path')
            if data_config_path:
                # Resolve path: check if absolute, otherwise look relative to current working directory
                if os.path.isabs(data_config_path):
                    resolved_path = data_config_path
                else:
                    resolved_path = os.path.join(os.getcwd(), data_config_path)
                
                if os.path.exists(resolved_path):
                    with open(resolved_path, 'r') as f:
                        checkpoint_config.data_config = dotdict(yaml.safe_load(f))
                else:
                    raise FileNotFoundError(
                        f"Data config file not found: {data_config_path}\n"
                        f"  Resolved to: {resolved_path}\n"
                        f"  Current working directory: {os.getcwd()}\n"
                        f"  Note: Use absolute paths in configs for more robust evaluation."
                    )
    else:
        # Legacy format: args.json
        config_path = os.path.join(ckpt_path, 'args.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found in {ckpt_path}\n"
                f"  Expected: configs/experiment_config.yaml or args.json"
            )
        checkpoint_config = dotdict(json.load(open(config_path)))
        checkpoint_config.model_config = dotdict(checkpoint_config.model_config)
    
    # Use provided data_config override if available, otherwise use checkpoint's data_config
    if hasattr(config, 'data_config') and config.data_config:
        checkpoint_config.data_config = dotdict(yaml.safe_load(open(config.data_config, 'r')))
    else:
        checkpoint_config.data_config = dotdict(checkpoint_config.data_config)
    
    # Check pretrained_model_path if present in model config (e.g., for LYNX models)
    if hasattr(checkpoint_config.model_config, 'pretrained_model_path') and checkpoint_config.model_config.pretrained_model_path:
        pretrained_path = checkpoint_config.model_config.pretrained_model_path
        if not os.path.isabs(pretrained_path):
            print(f"[Warning] pretrained_model_path is not absolute: {pretrained_path}")
            print("  It is much more robust to use absolute paths for pretrained_model_path in configs.")
            print(f"  Will only look relative to current working directory: {os.getcwd()}")
        
        # Resolve path: check if absolute, otherwise look relative to current working directory
        if os.path.isabs(pretrained_path):
            resolved_pretrained_path = pretrained_path
        else:
            resolved_pretrained_path = os.path.join(os.getcwd(), pretrained_path)
        
        if not os.path.exists(resolved_pretrained_path):
            raise FileNotFoundError(
                f"pretrained_model_path not found: {pretrained_path}\n"
                f"  Resolved to: {resolved_pretrained_path}\n"
                f"  Current working directory: {os.getcwd()}\n"
                f"  Note: Use absolute paths in model config for more robust evaluation."
            )
        # Update the path to the resolved absolute path
        checkpoint_config.model_config.pretrained_model_path = os.path.abspath(resolved_pretrained_path)
    
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
    
    # Find checkpoint file (filter out directories, only keep files)
    # Prefer best_checkpoint.* if it exists, otherwise use checkpoint.*
    best_ckpt_files = [f for f in glob.glob(os.path.join(ckpt_path, 'best_checkpoint*')) if os.path.isfile(f)]
    if best_ckpt_files:
        ckpt_file_path = best_ckpt_files[0]
    else:
        ckpt_files = [f for f in glob.glob(os.path.join(ckpt_path, 'checkpoint*')) if os.path.isfile(f)]
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint file (e.g., 'checkpoint.pth' or 'best_checkpoint.pth') found in {ckpt_path}")
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
    else:
        # Standard PyTorch checkpoint
        state_dict = checkpoint if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint else checkpoint.get('state_dict', checkpoint)
    
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

