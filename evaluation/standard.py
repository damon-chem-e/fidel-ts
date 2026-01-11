"""
Standard evaluation module for PyTorch-trained models.

This module provides evaluation functionality for models trained with standard PyTorch,
migrated from criterias.py.
"""

import torch
import os
import json
import glob
import yaml
from tqdm import tqdm
from models import model_init
from data_provider.data_factory import Data_Provider
from utils.tools import dotdict


def evaluate_full_dataset(loader, model, config, device, indexes, channel_wise, dataset=None):
    """
    Evaluates all samples in a dataset using a DataLoader for efficient batch processing.
    Calculates both normalized and denormalized MSE and MAE over the entire dataset.
    
    Args:
        loader: DataLoader for dataset
        model: Trained model
        config: Configuration object
        device: PyTorch device
        indexes: List of sample indexes to evaluate (None for all)
        channel_wise: Whether to compute channel-wise metrics
        dataset: Optional dataset object to access scaler for denormalization
    
    Returns:
        If channel_wise: (channel_mse_norm, channel_mae_norm, channel_mse_denorm, channel_mae_denorm, channel_counts)
        Otherwise: (total_mse_norm, total_mae_norm, total_mse_denorm, total_mae_denorm, num_samples)
    """
    # Normalized metrics
    total_mse_norm, total_mae_norm, num_samples = 0.0, 0.0, 0
    channel_mse_norm, channel_mae_norm, channel_counts = None, None, None
    
    # Denormalized metrics
    total_mse_denorm, total_mae_denorm = 0.0, 0.0
    channel_mse_denorm, channel_mae_denorm = None, None
    
    # Check if scaler is available for denormalization
    has_scaler = False
    scaler = None
    if dataset is not None and hasattr(dataset, 'scaler') and dataset.scaler is not None:
        has_scaler = True
        scaler = dataset.scaler

    for i, iter_data in tqdm(enumerate(loader), total=len(loader), desc="Running tests"):
        if indexes is not None and i not in indexes:
            continue
        with torch.no_grad():
            # Unpack batch: sample_id, seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, 
            # hetero_x_time, hetero_y_time, hetero_general, hetero_channel, x_time_features, y_time_features
            sample_ids, batch_x, batch_y, _, _, x_hetero, y_hetero, _, _, _, hetero_channel, _, _ = iter_data

            batch_x = torch.tensor(batch_x).to(device)
            batch_y = torch.tensor(batch_y).to(device)
            
            if config.task == 'TSF':
                prediction = model(x=batch_x)
            elif config.task == 'TGTSF':
                y_hetero = torch.tensor(y_hetero).to(device)
                hetero_channel = torch.tensor(hetero_channel).to(device)
                prediction = model(x=batch_x, news=y_hetero, channel_description=hetero_channel)
            elif config.task == 'MTSF':
                x_hetero = torch.tensor(x_hetero).to(device)
                prediction = model(x=batch_x, historical_events=x_hetero)
            else:
                # todo
                pass
            
            prediction = prediction[:, -config.output_len:, :]

            if channel_wise:
                if channel_mse_norm is None:
                    C = prediction.shape[2]
                    channel_mse_norm = [0.0] * C
                    channel_mae_norm = [0.0] * C
                    channel_mse_denorm = [0.0] * C
                    channel_mae_denorm = [0.0] * C
                    channel_counts = [0] * C
                for k in range(prediction.shape[2]):
                    # Normalized metrics
                    mse_loss_norm = torch.nn.MSELoss()(prediction[:, :, k], batch_y[:, :, k])
                    mae_loss_norm = torch.nn.L1Loss()(prediction[:, :, k], batch_y[:, :, k])
                    channel_mse_norm[k] += mse_loss_norm.item() * batch_y.size(0)
                    channel_mae_norm[k] += mae_loss_norm.item() * batch_y.size(0)
                    
                    # Denormalized metrics
                    if has_scaler:
                        pred_denorm = scaler.inverse_transform(prediction[:, :, k].cpu().numpy().reshape(-1, 1))
                        target_denorm = scaler.inverse_transform(batch_y[:, :, k].cpu().numpy().reshape(-1, 1))
                        pred_denorm_tensor = torch.tensor(pred_denorm.flatten(), device=device).reshape(prediction[:, :, k].shape)
                        target_denorm_tensor = torch.tensor(target_denorm.flatten(), device=device).reshape(batch_y[:, :, k].shape)
                        mse_loss_denorm = torch.nn.MSELoss()(pred_denorm_tensor, target_denorm_tensor)
                        mae_loss_denorm = torch.nn.L1Loss()(pred_denorm_tensor, target_denorm_tensor)
                        channel_mse_denorm[k] += mse_loss_denorm.item() * batch_y.size(0)
                        channel_mae_denorm[k] += mae_loss_denorm.item() * batch_y.size(0)
                    
                    channel_counts[k] += batch_y.size(0)
            else:
                # Normalized metrics
                mse_loss_norm = torch.nn.MSELoss()(prediction, batch_y)
                mae_loss_norm = torch.nn.L1Loss()(prediction, batch_y)
                total_mae_norm += mae_loss_norm.item() * batch_y.size(0)
                total_mse_norm += mse_loss_norm.item() * batch_y.size(0)
                
                # Denormalized metrics
                if has_scaler:
                    # Reshape for scaler: (batch, seq, features) -> (batch*seq, features)
                    batch_size, seq_len, num_features = prediction.shape
                    pred_flat = prediction.cpu().numpy().reshape(-1, num_features)
                    target_flat = batch_y.cpu().numpy().reshape(-1, num_features)
                    
                    # Denormalize - scaler expects (n_samples, n_features)
                    pred_denorm = scaler.inverse_transform(pred_flat)
                    target_denorm = scaler.inverse_transform(target_flat)
                    
                    # Convert back to tensors and compute metrics
                    pred_denorm_tensor = torch.tensor(pred_denorm, device=device, dtype=torch.float32).reshape(batch_size, seq_len, num_features)
                    target_denorm_tensor = torch.tensor(target_denorm, device=device, dtype=torch.float32).reshape(batch_size, seq_len, num_features)
                    mse_loss_denorm = torch.nn.MSELoss()(pred_denorm_tensor, target_denorm_tensor)
                    mae_loss_denorm = torch.nn.L1Loss()(pred_denorm_tensor, target_denorm_tensor)
                    total_mse_denorm += mse_loss_denorm.item() * batch_y.size(0)
                    total_mae_denorm += mae_loss_denorm.item() * batch_y.size(0)
                
                num_samples += batch_y.size(0)

    if channel_wise:
        return channel_mse_norm, channel_mae_norm, channel_mse_denorm, channel_mae_denorm, channel_counts
    else:
        return total_mse_norm, total_mae_norm, total_mse_denorm, total_mae_denorm, num_samples


def find_checkpoint(eval_config):
    """
    Find checkpoint path based on evaluation configuration (legacy format only).
    
    This function is kept for backward compatibility with old evaluation configs
    that use pattern matching instead of experiment directory structure.
    
    Args:
        eval_config: Evaluation configuration with checkpoint_base, model, data, input_len, output_len
    
    Returns:
        Path to checkpoint directory
    """
    from evaluation.checkpoint_finder import find_checkpoint_legacy
    import warnings
    warnings.warn(
        "Using legacy checkpoint finding with pattern matching. "
        "Consider migrating to experiment directory structure with resume_experiment_id.",
        DeprecationWarning,
        stacklevel=2
    )
    return find_checkpoint_legacy(eval_config)


def _determine_experiment_dir(eval_config):
    """
    Determine experiment directory from evaluation config.
    
    Args:
        eval_config: Evaluation configuration
        
    Returns:
        Path: Path to experiment directory
    """
    from pathlib import Path
    
    if hasattr(eval_config, 'experiment_dir'):
        return Path(eval_config.experiment_dir)
    elif (Path(eval_config.checkpoint_base) / "checkpoints").exists():
        return Path(eval_config.checkpoint_base)
    else:
        # Legacy format: use old find_checkpoint logic
        ckpt_path = find_checkpoint(eval_config)
        experiment_dir = Path(ckpt_path)
        print(f"[Info] Using checkpoint path (legacy): {ckpt_path}")
        return experiment_dir


def _load_checkpoint_config(experiment_dir, eval_config, config):
    """
    Load and prepare checkpoint configuration from experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        eval_config: Evaluation configuration
        config: Original config (may contain data_config override)
        
    Returns:
        dotdict: Prepared checkpoint configuration
    """
    from pathlib import Path
    from evaluation.config_builder import load_checkpoint_config
    
    # Load checkpoint config
    checkpoint_config_dict = load_checkpoint_config(experiment_dir)
    
    # Convert to dotdict (new format: experiment_config.yaml)
    model_config_path = experiment_dir / "configs" / "model_config.yaml"
    data_config_path = experiment_dir / "configs" / "data_config.yaml"
    
    checkpoint_config = dotdict()
    checkpoint_config.model = checkpoint_config_dict.get('model', {}).get('name', 'unknown')
    checkpoint_config.data = checkpoint_config_dict.get('data', {}).get('name', 'unknown')
    training_config = checkpoint_config_dict.get('training', {})
    checkpoint_config.input_len = training_config.get('input_len')
    checkpoint_config.output_len = training_config.get('output_len')
    checkpoint_config.batch_size = training_config.get('batch_size', 128)
    
    # Load model config
    if model_config_path.exists():
        with open(model_config_path, 'r', encoding='utf-8') as f:
            checkpoint_config.model_config = dotdict(yaml.safe_load(f))
    else:
        # Fallback: try to load from original path
        model_cfg_path = checkpoint_config_dict.get('model', {}).get('config_path')
        if model_cfg_path:
            with open(model_cfg_path, 'r', encoding='utf-8') as f:
                checkpoint_config.model_config = dotdict(yaml.safe_load(f))
        else:
            raise FileNotFoundError(f"Model config not found: {model_config_path}")
    
    # Load data config
    if data_config_path.exists():
        with open(data_config_path, 'r', encoding='utf-8') as f:
            checkpoint_config.data_config = dotdict(yaml.safe_load(f))
    else:
        # Fallback: try to load from original path
        data_cfg_path = checkpoint_config_dict.get('data', {}).get('config_path')
        if data_cfg_path:
            with open(data_cfg_path, 'r', encoding='utf-8') as f:
                checkpoint_config.data_config = dotdict(yaml.safe_load(f))
        else:
            raise FileNotFoundError(f"Data config not found: {data_config_path}")
    
    # Use provided data_config override if available
    if hasattr(config, 'data_config') and config.data_config:
        checkpoint_config.data_config = dotdict(yaml.safe_load(open(config.data_config, 'r')))
    
    # Set evaluation-specific config values
    checkpoint_config.gpu = eval_config.device
    checkpoint_config.num_workers = 0
    checkpoint_config.task = eval_config.task
    checkpoint_config.batch_size = 1 if eval_config.filtered_samples is not None else eval_config.batch_size
    
    return checkpoint_config


def _load_model_and_checkpoint(experiment_dir, eval_config, checkpoint_config):
    """
    Load model and checkpoint from experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        eval_config: Evaluation configuration
        checkpoint_config: Checkpoint configuration
        
    Returns:
        tuple: (model, device)
    """
    from pathlib import Path
    from evaluation.checkpoint_finder import find_checkpoint_from_experiment_dir
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{eval_config.device}")
    else:
        device = torch.device("cpu")
        print("[Warning] CUDA is not available, use CPU instead.")
    print(f"[Info] Running on device: {device}")
    
    # Initialize model
    model = model_init(checkpoint_config.model, checkpoint_config.model_config, checkpoint_config).to(device)
    
    # Find checkpoint file
    if hasattr(eval_config, 'experiment_dir') or (Path(eval_config.checkpoint_base) / "checkpoints").exists():
        # New format: use experiment directory structure
        ckpt_file_path = find_checkpoint_from_experiment_dir(experiment_dir, version=eval_config.version)
    else:
        # Legacy format: look in checkpoint directory
        ckpt_file = glob.glob(os.path.join(str(experiment_dir), 'checkpoint*'))
        if not ckpt_file:
            raise FileNotFoundError(f"No checkpoint file (e.g., 'checkpoint.pth') found in {experiment_dir}")
        ckpt_file_path = Path(ckpt_file[0])
    
    # Load checkpoint
    print(f"[Info] Loading model from: {ckpt_file_path}")
    checkpoint = torch.load(str(ckpt_file_path), map_location=device)
    
    # Extract state dict
    if str(ckpt_file_path).endswith('.ckpt') and isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict into model
    model.load_state_dict(state_dict)
    model.eval()
    print(f'[Info] Successfully loaded model: {checkpoint_config.model}')
    
    return model, device


def _run_evaluation_loop(loaders_dict, datasets_dict, model, checkpoint_config, device, eval_config):
    """
    Run evaluation loop over train, val, and test datasets separately.
    
    Args:
        loaders_dict: Dictionary mapping split names ('train', 'val', 'test') to DataLoaders
        datasets_dict: Dictionary mapping split names to dataset objects (for scaler access)
        model: Trained model
        checkpoint_config: Checkpoint configuration
        device: PyTorch device
        eval_config: Evaluation configuration
        
    Returns:
        dict: Results dictionary with keys for each split, containing normalized and denormalized metrics
    """
    results = {}
    
    # Load filtered samples if provided
    filtered_samples = None
    if eval_config.filtered_samples is not None:
        filtered_samples = json.load(open(eval_config.filtered_samples))
        print(f"[Info] Using filtered samples from: {eval_config.filtered_samples}")
    
    # Evaluate each split (train, val, test)
    for split_name in ['train', 'val', 'test']:
        if split_name not in loaders_dict:
            continue
            
        loader = loaders_dict[split_name]
        dataset = datasets_dict.get(split_name, None)
        
        # Handle case where loader is a dictionary (multiple entities)
        if isinstance(loader, dict):
            # Aggregate results across all entities
            total_mse_norm, total_mae_norm = 0.0, 0.0
            total_mse_denorm, total_mae_denorm = 0.0, 0.0
            total_samples = 0
            
            for entity_name, entity_loader in loader.items():
                print(f"\n[Info] Testing on {split_name} dataset - entity: {entity_name}")
                
                # Get dataset for this entity if available
                entity_dataset = None
                if isinstance(dataset, dict) and entity_name in dataset:
                    entity_dataset = dataset[entity_name]
                elif not isinstance(dataset, dict):
                    entity_dataset = dataset
                
                # Get indexes for this entity
                if filtered_samples is not None:
                    indexes = filtered_samples.get(f"{split_name}_{entity_name}", filtered_samples.get(split_name, []))
                else:
                    indexes = None
                
                # Run evaluation for this entity
                result = evaluate_full_dataset(entity_loader, model, checkpoint_config, device, indexes, 
                                              eval_config.channel_wise, dataset=entity_dataset)
                
                if not eval_config.channel_wise:
                    mse_norm, mae_norm, mse_denorm, mae_denorm, num_samples = result
                    total_mse_norm += mse_norm
                    total_mae_norm += mae_norm
                    if mse_denorm > 0:
                        total_mse_denorm += mse_denorm
                        total_mae_denorm += mae_denorm
                    total_samples += num_samples
            
            # Aggregate results
            if total_samples > 0:
                avg_mse_norm = total_mse_norm / total_samples
                avg_mae_norm = total_mae_norm / total_samples
                avg_mse_denorm = total_mse_denorm / total_samples if total_mse_denorm > 0 else None
                avg_mae_denorm = total_mae_denorm / total_samples if total_mae_denorm > 0 else None
                
                print(f"\n-> Results for '{split_name}' (normalized): MSE = {avg_mse_norm:.7f}, MAE = {avg_mae_norm:.7f}")
                if avg_mse_denorm is not None:
                    print(f"-> Results for '{split_name}' (denormalized): MSE = {avg_mse_denorm:.7f}, MAE = {avg_mae_denorm:.7f}")
                else:
                    print(f"-> Results for '{split_name}' (denormalized): N/A (scaler not available)")
                
                results[split_name] = {
                    'mse_norm': avg_mse_norm,
                    'mae_norm': avg_mae_norm,
                    'mse_denorm': avg_mse_denorm,
                    'mae_denorm': avg_mae_denorm,
                    'num_samples': total_samples
                }
            else:
                results[split_name] = None
        else:
            # Single loader (single entity or already aggregated)
            print(f"\n[Info] Testing on {split_name} dataset")
            
            # Get indexes for this dataset
            if filtered_samples is not None:
                indexes = filtered_samples.get(split_name, [])
                print(f"[Info] Using {len(indexes)} filtered samples for testing.")
                print(f"[Info] Sample indexes: {indexes}")
            else:
                indexes = None
                print("[Info] Using all samples for testing.")
            
            # Run evaluation
            result = evaluate_full_dataset(loader, model, checkpoint_config, device, indexes, 
                                          eval_config.channel_wise, dataset=dataset)
        
        # Process results
        if eval_config.channel_wise:
            channel_mse_norm, channel_mae_norm, channel_mse_denorm, channel_mae_denorm, channel_counts = result
            if channel_mse_norm is None or sum(channel_counts) == 0:
                print(f"-> No valid samples found in '{split_name}'")
                results[split_name] = None
            else:
                avg_ch_mse_norm = [m / count if count > 0 else 0 for m, count in zip(channel_mse_norm, channel_counts)]
                avg_ch_mae_norm = [m / count if count > 0 else 0 for m, count in zip(channel_mae_norm, channel_counts)]
                avg_ch_mse_denorm = [m / count if count > 0 else 0 for m, count in zip(channel_mse_denorm, channel_counts)]
                avg_ch_mae_denorm = [m / count if count > 0 else 0 for m, count in zip(channel_mae_denorm, channel_counts)]
                
                print(f"-> Results for '{split_name}' (normalized): Channel-wise MSE = {avg_ch_mse_norm}, Channel-wise MAE = {avg_ch_mae_norm}")
                print(f"-> Results for '{split_name}' (denormalized): Channel-wise MSE = {avg_ch_mse_denorm}, Channel-wise MAE = {avg_ch_mae_denorm}")
                
                results[split_name] = {
                    'mse_norm': sum(avg_ch_mse_norm) / len(avg_ch_mse_norm),
                    'mae_norm': sum(avg_ch_mae_norm) / len(avg_ch_mae_norm),
                    'mse_denorm': sum(avg_ch_mse_denorm) / len(avg_ch_mse_denorm),
                    'mae_denorm': sum(avg_ch_mae_denorm) / len(avg_ch_mae_denorm),
                    'num_samples': sum(channel_counts)
                }
        else:
            total_mse_norm, total_mae_norm, total_mse_denorm, total_mae_denorm, num_samples = result
            if num_samples > 0:
                avg_mse_norm = total_mse_norm / num_samples
                avg_mae_norm = total_mae_norm / num_samples
                avg_mse_denorm = total_mse_denorm / num_samples if total_mse_denorm > 0 else None
                avg_mae_denorm = total_mae_denorm / num_samples if total_mae_denorm > 0 else None
                
                print(f"-> Results for '{split_name}' (normalized): MSE = {avg_mse_norm:.7f}, MAE = {avg_mae_norm:.7f}")
                if avg_mse_denorm is not None:
                    print(f"-> Results for '{split_name}' (denormalized): MSE = {avg_mse_denorm:.7f}, MAE = {avg_mae_denorm:.7f}")
                else:
                    print(f"-> Results for '{split_name}' (denormalized): N/A (scaler not available)")
                
                results[split_name] = {
                    'mse_norm': avg_mse_norm,
                    'mae_norm': avg_mae_norm,
                    'mse_denorm': avg_mse_denorm,
                    'mae_denorm': avg_mae_denorm,
                    'num_samples': num_samples
                }
            else:
                print(f"-> No valid samples found in '{split_name}'")
                results[split_name] = None
    
    return results


def _print_summary(results, eval_config):
    """
    Print evaluation summary for train, val, and test splits.
    
    Args:
        results: Dictionary mapping split names to result dictionaries
        eval_config: Evaluation configuration
    """
    print("\n" + "="*50)
    print(" " * 15 + "Evaluation Summary")
    print("="*50)
    
    for split_name in ['train', 'val', 'test']:
        if split_name not in results or results[split_name] is None:
            continue
        
        result = results[split_name]
        print(f"\n{split_name.upper()} Set:")
        print(f"  Normalized   - MSE: {result['mse_norm']:.7f}, MAE: {result['mae_norm']:.7f}")
        if result['mse_denorm'] is not None:
            print(f"  Denormalized - MSE: {result['mse_denorm']:.7f}, MAE: {result['mae_denorm']:.7f}")
        else:
            print(f"  Denormalized - N/A (scaler not available)")
        print(f"  Samples: {result['num_samples']}")
    
    print("="*50)


def evaluate(config):
    """
    Evaluate standard PyTorch-trained models.
    
    This function loads a trained model checkpoint, runs inference on test datasets,
    and calculates evaluation metrics (MSE, MAE) per subset and overall.
    
    Args:
        config: dotdict containing evaluation configuration with structure:
            - evaluation: {model, data, version, input_len, output_len, checkpoint_base,
                          batch_size, task, filtered_samples, device, channel_wise}
            - data_config: Optional path to override data config
    
    Example:
        >>> from cli.config.loader import load_config_with_nested
        >>> config_hierarchy = load_config_with_nested("configs/experiments/dlinear_solar.yaml")
        >>> evaluate(config_hierarchy['primary'])
    """
    eval_config = config.evaluation
    
    # Determine experiment directory
    experiment_dir = _determine_experiment_dir(eval_config)
    print(f"[Info] Loading config from experiment directory: {experiment_dir}")
    
    # Load checkpoint configuration
    checkpoint_config = _load_checkpoint_config(experiment_dir, eval_config, config)
    
    # Load model and checkpoint
    model, device = _load_model_and_checkpoint(experiment_dir, eval_config, checkpoint_config)
    
    # Load data - get train, val, and test separately
    data_provider = Data_Provider(checkpoint_config)
    
    # Get loaders and datasets for each split
    loaders_dict = {}
    datasets_dict = {}
    
    train_loader = data_provider.get_train("loader")
    if train_loader is not None:
        loaders_dict['train'] = train_loader
        train_dataset = data_provider.get_train("set")
        if train_dataset is not None:
            # Handle case where get_train returns a dict of datasets
            if isinstance(train_dataset, dict):
                # Use first dataset's scaler (assuming all have same scaler)
                first_dataset = next(iter(train_dataset.values()))
                datasets_dict['train'] = first_dataset
            else:
                datasets_dict['train'] = train_dataset
    
    val_loader = data_provider.get_val("loader")
    if val_loader is not None:
        loaders_dict['val'] = val_loader
        val_dataset = data_provider.get_val("set")
        if val_dataset is not None:
            if isinstance(val_dataset, dict):
                first_dataset = next(iter(val_dataset.values()))
                datasets_dict['val'] = first_dataset
            else:
                datasets_dict['val'] = val_dataset
    
    test_loader = data_provider.get_test("loader")
    if test_loader is not None:
        loaders_dict['test'] = test_loader
        test_dataset = data_provider.get_test("set")
        if test_dataset is not None:
            if isinstance(test_dataset, dict):
                first_dataset = next(iter(test_dataset.values()))
                datasets_dict['test'] = first_dataset
            else:
                datasets_dict['test'] = test_dataset
    
    # Run evaluation loop
    results = _run_evaluation_loop(
        loaders_dict, datasets_dict, model, checkpoint_config, device, eval_config
    )
    
    # Print summary
    _print_summary(results, eval_config)

