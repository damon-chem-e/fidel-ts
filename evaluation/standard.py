"""
Standard evaluation module for PyTorch-trained models.

This module provides evaluation functionality for models trained with standard PyTorch,
migrated from criterias.py.
"""

import torch
import os
import json
import glob
import numpy as np
from tqdm import tqdm
from models import model_init
from data_provider.data_factory import Data_Provider
from evaluation.config_builder import build_eval_args_from_experiment_dir


def _compute_per_sample_losses(prediction, target):
    """
    Compute per-sample MSE and MAE losses.
    
    Args:
        prediction: Model predictions, shape (batch_size, seq_len, features)
        target: Ground truth targets, shape (batch_size, seq_len, features)
        
    Returns:
        tuple: (mse_per_sample, mae_per_sample) both of shape (batch_size,)
    """
    # Compute per-sample, per-timestep losses
    mse_loss_per_sample = torch.nn.MSELoss(reduction='none')(prediction, target)
    mae_loss_per_sample = torch.nn.L1Loss(reduction='none')(prediction, target)
    
    # Reduce across sequence and feature dimensions to get per-sample loss
    # Shape: (batch_size, seq_len, features) -> (batch_size,)
    mse_per_sample = mse_loss_per_sample.mean(dim=(1, 2))
    mae_per_sample = mae_loss_per_sample.mean(dim=(1, 2))
    
    return mse_per_sample, mae_per_sample


def _filter_nan_samples(prediction, target, mse_per_sample, mae_per_sample):
    """
    Filter out samples that have NaN in their losses.
    
    Args:
        prediction: Model predictions, shape (batch_size, seq_len, features)
        target: Ground truth targets, shape (batch_size, seq_len, features)
        mse_per_sample: Per-sample MSE losses, shape (batch_size,)
        mae_per_sample: Per-sample MAE losses, shape (batch_size,)
        
    Returns:
        tuple: (prediction_valid, target_valid, valid_mask, num_valid_samples, num_skipped)
            - prediction_valid: Filtered predictions (only valid samples)
            - target_valid: Filtered targets (only valid samples)
            - valid_mask: Boolean mask indicating valid samples, shape (batch_size,)
            - num_valid_samples: Number of valid samples
            - num_skipped: Number of skipped samples
    """
    # Identify valid samples (no NaN in either loss)
    valid_mask = ~(torch.isnan(mse_per_sample) | torch.isnan(mae_per_sample))
    num_valid_samples = valid_mask.sum().item()
    num_skipped = prediction.size(0) - num_valid_samples
    
    if num_valid_samples > 0:
        # Filter to valid samples only
        prediction_valid = prediction[valid_mask]
        target_valid = target[valid_mask]
    else:
        # All samples are NaN
        prediction_valid = None
        target_valid = None
    
    return prediction_valid, target_valid, valid_mask, num_valid_samples, num_skipped


def evaluate_full_dataset(loader, model, config, device, indexes, channel_wise, dataset=None, filter_nan_samples=False):
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
        filter_nan_samples: If True, filter out individual samples with NaN losses (less performant)
    
    Returns:
        If channel_wise: (channel_mse_norm, channel_mae_norm, channel_mse_denorm, channel_mae_denorm, channel_counts)
        Otherwise: (total_mse_norm, total_mae_norm, total_mse_denorm, total_mae_denorm, num_samples)
    """
    # Log performance warning if filtering is enabled
    if filter_nan_samples:
        print("[Warning] filter_nan_samples is enabled. This requires per-sample loss computation "
              "which is less performant than batch-averaged losses. Evaluation will be slower.")
    
    # Normalized metrics
    total_mse_norm, total_mae_norm, num_samples = 0.0, 0.0, 0
    channel_mse_norm, channel_mae_norm, channel_counts = None, None, None
    
    # Denormalized metrics
    total_mse_denorm, total_mae_denorm = 0.0, 0.0
    channel_mse_denorm, channel_mae_denorm = None, None
    
    # Track skipped samples when filtering is enabled
    skipped_samples = 0
    
    # Check if scaler is available and fitted for denormalization
    has_scaler = False
    scaler = None
    
    # Try to get dataset from loader if not provided
    if dataset is None and hasattr(loader, 'dataset'):
        dataset = loader.dataset
    
    if dataset is not None:
        # Handle case where dataset is a dictionary (multiple entities)
        if isinstance(dataset, dict):
            # Try to get first dataset from dict
            dataset = next(iter(dataset.values())) if dataset else None
        
        # Handle ConcatDataset - extract first underlying dataset to access scaler
        if hasattr(dataset, 'datasets') and isinstance(dataset.datasets, (list, tuple)):
            # This is a ConcatDataset, get the first underlying dataset
            if len(dataset.datasets) > 0:
                dataset = dataset.datasets[0]
        
        if dataset is not None and hasattr(dataset, 'scaler') and dataset.scaler is not None:
            # Check if scaler has been fitted (has mean_ attribute)
            try:
                if hasattr(dataset.scaler, 'mean_') and dataset.scaler.mean_ is not None:
                    has_scaler = True
                    scaler = dataset.scaler
                else:
                    # Scaler exists but not fitted
                    pass  # Will skip denormalized metrics
            except Exception:
                # Scaler exists but error accessing it
                pass  # Will skip denormalized metrics

    for i, iter_data in tqdm(enumerate(loader), total=len(loader), desc="Running tests"):
        if indexes is not None and i not in indexes:
            continue
        with torch.no_grad():
            # Unpack batch: sample_id, seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, 
            # hetero_x_time, hetero_y_time, hetero_general, hetero_channel, x_time_features, y_time_features
            sample_ids, batch_x, batch_y, _, _, x_hetero, y_hetero, _, _, _, hetero_channel, _, _ = iter_data

            # Convert to tensors if not already (handles numpy arrays and tensors)
            if not isinstance(batch_x, torch.Tensor):
                batch_x = torch.as_tensor(batch_x, dtype=torch.float32)
            batch_x = batch_x.to(device)
            
            if not isinstance(batch_y, torch.Tensor):
                batch_y = torch.as_tensor(batch_y, dtype=torch.float32)
            batch_y = batch_y.to(device)
            
            if config.task == 'TSF':
                prediction = model(x=batch_x)
            elif config.task == 'TGTSF':
                # Convert y_hetero (news) to tensor
                if not isinstance(y_hetero, torch.Tensor):
                    y_hetero = torch.as_tensor(y_hetero, dtype=torch.float32)
                y_hetero = y_hetero.to(device)
                
                # Convert hetero_channel to tensor
                if not isinstance(hetero_channel, torch.Tensor):
                    hetero_channel = torch.as_tensor(hetero_channel, dtype=torch.float32)
                hetero_channel = hetero_channel.to(device)
                
                # IMPORTANT: Also pass x_hetero (historical_events) for models that use timestamp_semantics
                # Models like LYNX internally select between news and historical_events based on timestamp_semantics
                # - timestamp_semantics='t_about': uses news (y_hetero)
                # - timestamp_semantics='t_known': uses historical_events (x_hetero)
                if x_hetero is not None:
                    if not isinstance(x_hetero, torch.Tensor):
                        x_hetero = torch.as_tensor(x_hetero, dtype=torch.float32)
                    x_hetero = x_hetero.to(device)
                    # Pass both news and historical_events - let model decide which to use
                    prediction = model(x=batch_x, news=y_hetero, channel_description=hetero_channel, 
                                     historical_events=x_hetero)
                else:
                    # Standard TGTSF: only pass news
                    prediction = model(x=batch_x, news=y_hetero, channel_description=hetero_channel)
            elif config.task == 'MTSF':
                if not isinstance(x_hetero, torch.Tensor):
                    x_hetero = torch.as_tensor(x_hetero, dtype=torch.float32)
                x_hetero = x_hetero.to(device)
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
                
                # For channel-wise, filter at sample level (if any channel has NaN, filter entire sample)
                if filter_nan_samples:
                    # Compute per-sample losses across all channels
                    # Check if any channel has NaN for each sample
                    sample_has_nan = torch.zeros(prediction.size(0), dtype=torch.bool, device=device)
                    for k in range(prediction.shape[2]):
                        mse_per_sample, mae_per_sample = _compute_per_sample_losses(
                            prediction[:, :, k], batch_y[:, :, k]
                        )
                        sample_has_nan |= (torch.isnan(mse_per_sample) | torch.isnan(mae_per_sample))
                    
                    valid_mask = ~sample_has_nan
                    num_valid = valid_mask.sum().item()
                    skipped_samples += (prediction.size(0) - num_valid)
                    
                    if num_valid == 0:
                        # All samples filtered, skip this batch
                        continue
                    
                    # Filter to valid samples
                    prediction = prediction[valid_mask]
                    batch_y = batch_y[valid_mask]
                
                for k in range(prediction.shape[2]):
                    # Normalized metrics
                    mse_loss_norm = torch.nn.MSELoss()(prediction[:, :, k], batch_y[:, :, k])
                    mae_loss_norm = torch.nn.L1Loss()(prediction[:, :, k], batch_y[:, :, k])
                    channel_mse_norm[k] += mse_loss_norm.item() * batch_y.size(0)
                    channel_mae_norm[k] += mae_loss_norm.item() * batch_y.size(0)
                    
                    # Denormalized metrics
                    if has_scaler:
                        try:
                            pred_denorm = scaler.inverse_transform(prediction[:, :, k].cpu().numpy().reshape(-1, 1))
                            target_denorm = scaler.inverse_transform(batch_y[:, :, k].cpu().numpy().reshape(-1, 1))
                            pred_denorm_tensor = torch.tensor(pred_denorm.flatten(), device=device).reshape(prediction[:, :, k].shape)
                            target_denorm_tensor = torch.tensor(target_denorm.flatten(), device=device).reshape(batch_y[:, :, k].shape)
                            mse_loss_denorm = torch.nn.MSELoss()(pred_denorm_tensor, target_denorm_tensor)
                            mae_loss_denorm = torch.nn.L1Loss()(pred_denorm_tensor, target_denorm_tensor)
                            channel_mse_denorm[k] += mse_loss_denorm.item() * batch_y.size(0)
                            channel_mae_denorm[k] += mae_loss_denorm.item() * batch_y.size(0)
                        except Exception:
                            # Scaler not fitted or error during inverse transform
                            has_scaler = False  # Disable for remaining batches
                            pass
                    
                    channel_counts[k] += batch_y.size(0)
            else:
                # Normalized metrics
                prediction_valid = None
                batch_y_valid = None
                num_valid = batch_y.size(0)  # Default: all samples valid
                
                if filter_nan_samples:
                    # Compute per-sample losses and filter NaN samples
                    mse_per_sample, mae_per_sample = _compute_per_sample_losses(prediction, batch_y)
                    prediction_valid, batch_y_valid, valid_mask, num_valid, num_skipped_batch = _filter_nan_samples(
                        prediction, batch_y, mse_per_sample, mae_per_sample
                    )
                    skipped_samples += num_skipped_batch
                    
                    if num_valid > 0:
                        # Compute losses on valid samples only
                        mse_loss_norm = torch.nn.MSELoss()(prediction_valid, batch_y_valid)
                        mae_loss_norm = torch.nn.L1Loss()(prediction_valid, batch_y_valid)
                        total_mae_norm += mae_loss_norm.item() * num_valid
                        total_mse_norm += mse_loss_norm.item() * num_valid
                        num_samples += num_valid
                else:
                    # Standard batch-averaged losses (current behavior)
                    mse_loss_norm = torch.nn.MSELoss()(prediction, batch_y)
                    mae_loss_norm = torch.nn.L1Loss()(prediction, batch_y)
                    total_mae_norm += mae_loss_norm.item() * batch_y.size(0)
                    total_mse_norm += mse_loss_norm.item() * batch_y.size(0)
                    num_samples += batch_y.size(0)
                
                # Denormalized metrics
                if has_scaler and num_valid > 0:
                    try:
                        # Use filtered predictions/targets if filtering is enabled
                        pred_for_denorm = prediction_valid if filter_nan_samples else prediction
                        target_for_denorm = batch_y_valid if filter_nan_samples else batch_y
                        
                        # Reshape for scaler: (batch, seq, features) -> (batch*seq, features)
                        batch_size, seq_len, num_features = pred_for_denorm.shape
                        pred_flat = pred_for_denorm.cpu().numpy().reshape(-1, num_features)
                        target_flat = target_for_denorm.cpu().numpy().reshape(-1, num_features)
                        
                        # Denormalize - scaler expects (n_samples, n_features)
                        pred_denorm = scaler.inverse_transform(pred_flat)
                        target_denorm = scaler.inverse_transform(target_flat)
                        
                        # Convert back to tensors and compute metrics
                        pred_denorm_tensor = torch.tensor(pred_denorm, device=device, dtype=torch.float32).reshape(batch_size, seq_len, num_features)
                        target_denorm_tensor = torch.tensor(target_denorm, device=device, dtype=torch.float32).reshape(batch_size, seq_len, num_features)
                        
                        if filter_nan_samples:
                            # Check for NaN in denormalized losses too
                            mse_denorm_per_sample, mae_denorm_per_sample = _compute_per_sample_losses(
                                pred_denorm_tensor, target_denorm_tensor
                            )
                            pred_denorm_valid, target_denorm_valid, _, num_valid_denorm, _ = _filter_nan_samples(
                                pred_denorm_tensor, target_denorm_tensor, 
                                mse_denorm_per_sample, mae_denorm_per_sample
                            )
                            
                            if num_valid_denorm > 0:
                                mse_loss_denorm = torch.nn.MSELoss()(pred_denorm_valid, target_denorm_valid)
                                mae_loss_denorm = torch.nn.L1Loss()(pred_denorm_valid, target_denorm_valid)
                                total_mse_denorm += mse_loss_denorm.item() * num_valid_denorm
                                total_mae_denorm += mae_loss_denorm.item() * num_valid_denorm
                        else:
                            mse_loss_denorm = torch.nn.MSELoss()(pred_denorm_tensor, target_denorm_tensor)
                            mae_loss_denorm = torch.nn.L1Loss()(pred_denorm_tensor, target_denorm_tensor)
                            total_mse_denorm += mse_loss_denorm.item() * batch_size
                            total_mae_denorm += mae_loss_denorm.item() * batch_size
                    except Exception:
                        # Scaler not fitted or error during inverse transform
                        has_scaler = False  # Disable for remaining batches
                        pass

    if channel_wise:
        return channel_mse_norm, channel_mae_norm, channel_mse_denorm, channel_mae_denorm, channel_counts
    else:
        return total_mse_norm, total_mae_norm, total_mse_denorm, total_mae_denorm, num_samples


def _evaluate_single_entity(entity_name, entity_loader, dataset, split_name, filtered_samples,
                            model, checkpoint_config, device, channel_wise, filter_nan_samples=False):
    """
    Evaluate a single entity and return its metrics.
    
    Args:
        entity_name: Name of the entity being evaluated
        entity_loader: DataLoader for this entity
        dataset: Dataset object (may be dict or single dataset)
        split_name: Name of the split ('train', 'val', 'test')
        filtered_samples: Optional dict of filtered sample indexes
        model: Trained model
        checkpoint_config: Checkpoint configuration
        device: PyTorch device
        channel_wise: Whether to compute channel-wise metrics
        filter_nan_samples: If True, filter out individual samples with NaN losses
        
    Returns:
        tuple: (entity_name, mse_norm, mae_norm, mse_denorm, mae_denorm, num_samples)
    """
    print(f"\n[Info] Testing on {split_name} dataset - entity: {entity_name}")
    
    # Get dataset for this entity - prefer from loader, fallback to provided dataset
    entity_dataset = None
    if hasattr(entity_loader, 'dataset'):
        entity_dataset = entity_loader.dataset
    elif isinstance(dataset, dict) and entity_name in dataset:
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
                                  channel_wise, dataset=entity_dataset, filter_nan_samples=filter_nan_samples)
    
    if not channel_wise:
        mse_norm, mae_norm, mse_denorm, mae_denorm, num_samples = result
        return (entity_name, mse_norm, mae_norm, mse_denorm, mae_denorm, num_samples)
    else:
        # For channel_wise, return the raw result tuple with entity_name prepended
        return (entity_name, result)


def _aggregate_entity_metrics(entity_metrics, split_name, nan_aware=False):
    """
    Aggregate metrics from multiple entities into final results.
    
    Args:
        entity_metrics: List of (entity_name, mse_norm, mae_norm, mse_denorm, mae_denorm, num_samples) tuples
        split_name: Name of the split (for logging)
        nan_aware: If True, exclude entities with NaN metrics and warn about them
        
    Returns:
        dict: Aggregated results with keys 'mse_norm', 'mae_norm', 'mse_denorm', 'mae_denorm', 
              'num_samples', 'is_concat', or None if no valid metrics
    """
    if not entity_metrics:
        return None
    
    # If NaN-aware, filter out entities with NaN values
    if nan_aware:
        valid_metrics = []
        entities_with_nan = []
        
        for entity_name, mse_norm, mae_norm, mse_denorm, mae_denorm, num_samples in entity_metrics:
            has_nan = (np.isnan(mse_norm) or np.isnan(mae_norm) or 
                      np.isnan(mse_denorm) or np.isnan(mae_denorm))
            if has_nan:
                entities_with_nan.append(entity_name)
                print(f"[Warning] Entity '{entity_name}' produced NaN metrics - excluded from aggregation")
            else:
                valid_metrics.append((entity_name, mse_norm, mae_norm, mse_denorm, mae_denorm, num_samples))
        
        if entities_with_nan:
            print(f"\n[Warning] {len(entities_with_nan)} entity/entities with NaN metrics (excluded): {', '.join(entities_with_nan)}")
        
        if not valid_metrics:
            print(f"\n[Error] No valid entity metrics found for '{split_name}' (all entities produced NaN)")
            return None
        
        entity_metrics = valid_metrics
    
    # Aggregate metrics: sum totals and divide by total samples
    total_mse_norm = sum(m[1] for m in entity_metrics)
    total_mae_norm = sum(m[2] for m in entity_metrics)
    total_mse_denorm = sum(m[3] for m in entity_metrics if m[3] > 0)
    total_mae_denorm = sum(m[4] for m in entity_metrics if m[4] > 0)
    total_samples = sum(m[5] for m in entity_metrics)
    
    if total_samples == 0:
        return None
    
    avg_mse_norm = total_mse_norm / total_samples
    avg_mae_norm = total_mae_norm / total_samples
    avg_mse_denorm = total_mse_denorm / total_samples if total_mse_denorm > 0 else None
    avg_mae_denorm = total_mae_denorm / total_samples if total_mse_denorm > 0 else None
    
    # Print results
    print(f"\n-> Results for '{split_name}' (normalized): MSE = {avg_mse_norm:.7f}, MAE = {avg_mae_norm:.7f}")
    if avg_mse_denorm is not None:
        print(f"-> Results for '{split_name}' (denormalized): MSE = {avg_mse_denorm:.7f}, MAE = {avg_mae_denorm:.7f}")
    else:
        print(f"-> Results for '{split_name}' (denormalized): N/A (scaler not available)")
    
    return {
        'mse_norm': avg_mse_norm,
        'mae_norm': avg_mae_norm,
        'mse_denorm': avg_mse_denorm,
        'mae_denorm': avg_mae_denorm,
        'num_samples': total_samples,
        'is_concat': False
    }


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
    # Rebuild training-equivalent args from saved configs
    checkpoint_config = build_eval_args_from_experiment_dir(
        experiment_dir,
        data_config_override=getattr(config, 'data_config', None)
    )

    # Set evaluation-specific config values
    checkpoint_config.gpu = eval_config.device
    checkpoint_config.use_gpu = torch.cuda.is_available()  # Use GPU if available
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
    
    # Handle torch.compile checkpoints (state dict keys have "_orig_mod." prefix)
    # Check if this is a compiled model checkpoint
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        print("[Info] Detected torch.compile checkpoint - stripping '_orig_mod.' prefix from state dict keys")
        state_dict = {key.replace('_orig_mod.', ''): value for key in state_dict.keys() for value in [state_dict[key]]}
    
    # Load state dict into model
    model.load_state_dict(state_dict)
    model.eval()
    print(f'[Info] Successfully loaded model: {checkpoint_config.model}')
    
    # Apply torch.compile if enabled in evaluation config (PyTorch 2.0+)
    # Check for torch_compile flag in eval_config (backward compatible - defaults to False)
    if getattr(eval_config, 'torch_compile', False):
        if hasattr(torch, 'compile'):
            from utils.tools import compilation_spinner

            compile_mode = getattr(eval_config, 'compile_mode', 'reduce-overhead')

            with compilation_spinner(
                f"Compiling model for evaluation with torch.compile (mode={compile_mode})...",
                logger=None
            ):
                model = torch.compile(
                    model,
                    mode=compile_mode,
                    fullgraph=False  # More compatible with dynamic models
                )
                # Warm-up compilation with a dummy forward pass
                print("[Info] Warming up torch.compile...")
                try:
                    with torch.no_grad():
                        dummy_x = torch.randn(1, checkpoint_config.input_len, checkpoint_config.model_config.enc_in).to(device)
                        _ = model(x=dummy_x)
                    print("[Info] Compilation warm-up complete")
                except Exception as e:
                    print(f"[Warning] Warm-up forward pass failed (model will compile on first real batch): {e}")
        else:
            print("[Warning] torch.compile requested but not available (requires PyTorch 2.0+)")
    
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
    
    # Determine which splits to evaluate
    # Default to test_only=True if not specified (backward compatibility)
    test_only = getattr(eval_config, 'test_only', True)
    
    # Evaluate each split (train, val, test)
    splits_to_evaluate = ['test'] if test_only else ['train', 'val', 'test']
    
    for split_name in splits_to_evaluate:
        if split_name not in loaders_dict:
            continue
            
        loader = loaders_dict[split_name]
        dataset = datasets_dict.get(split_name, None)
        
        # Handle case where loader is a dictionary (multiple entities)
        if isinstance(loader, dict):
            # Check if NaN-aware aggregation is enabled (default: False - propagate NaN as before)
            nan_aware_aggregation = getattr(eval_config, 'nan_aware_aggregation', False)
            
            # Evaluate each entity and collect metrics (single loop, no duplication)
            if not eval_config.channel_wise:
                filter_nan_samples = getattr(eval_config, 'filter_nan_samples', False)
                entity_metrics = []
                for entity_name, entity_loader in loader.items():
                    entity_result = _evaluate_single_entity(
                        entity_name, entity_loader, dataset, split_name, filtered_samples,
                        model, checkpoint_config, device, eval_config.channel_wise,
                        filter_nan_samples=filter_nan_samples
                    )
                    entity_metrics.append(entity_result)
                
                # Aggregate with optional NaN filtering
                results[split_name] = _aggregate_entity_metrics(
                    entity_metrics, split_name, nan_aware=nan_aware_aggregation
                )
            # Skip to next split - dict loaders are fully handled above
            continue
        
        # Single loader (single entity or ConcatDataset)
        print(f"\n[Info] Testing on {split_name} dataset")
        
        # Check if this is a ConcatDataset (multiple entities concatenated)
        is_concat_dataset = False
        if hasattr(loader, 'dataset'):
            loader_dataset = loader.dataset
            if hasattr(loader_dataset, 'datasets') and isinstance(loader_dataset.datasets, (list, tuple)):
                is_concat_dataset = len(loader_dataset.datasets) > 1
        
        # Try to get dataset from loader if not provided
        if dataset is None and hasattr(loader, 'dataset'):
            dataset = loader.dataset
        
        # Get indexes for this dataset
        if filtered_samples is not None:
            indexes = filtered_samples.get(split_name, [])
            print(f"[Info] Using {len(indexes)} filtered samples for testing.")
            print(f"[Info] Sample indexes: {indexes}")
        else:
            indexes = None
            print("[Info] Using all samples for testing.")
        
        # Run evaluation
        filter_nan_samples = getattr(eval_config, 'filter_nan_samples', False)
        result = evaluate_full_dataset(loader, model, checkpoint_config, device, indexes, 
                                      eval_config.channel_wise, dataset=dataset, filter_nan_samples=filter_nan_samples)
        
        # Process results for single loader
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
                    'num_samples': sum(channel_counts),
                    'is_concat': is_concat_dataset
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
                    'num_samples': num_samples,
                    'is_concat': is_concat_dataset
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
    
    # Determine which splits to show (based on what was evaluated)
    test_only = getattr(eval_config, 'test_only', True)
    splits_to_show = ['test'] if test_only else ['train', 'val', 'test']
    
    # Track if any split has concat dataset with denormalized metrics
    concat_splits_with_denorm = []
    
    for split_name in splits_to_show:
        if split_name not in results or results[split_name] is None:
            continue
        
        result = results[split_name]
        is_concat = result.get('is_concat', False)
        
        print(f"\n{split_name.upper()} Set:")
        print(f"  Normalized   - MSE: {result['mse_norm']:.7f}, MAE: {result['mae_norm']:.7f}")
        if result['mse_denorm'] is not None:
            print(f"  Denormalized - MSE: {result['mse_denorm']:.7f}, MAE: {result['mae_denorm']:.7f}")
            # Track concat splits that show denormalized metrics
            if is_concat:
                concat_splits_with_denorm.append(split_name.upper())
        else:
            print("  Denormalized - N/A (scaler not available)")
        print(f"  Samples: {result['num_samples']}")
    
    print("="*50)
    
    # Print warning if any concat dataset showed denormalized metrics
    if concat_splits_with_denorm:
        print("\n" + "!"*50)
        print("  WARNING: Denormalized metrics may be inaccurate")
        print("!"*50)
        print(f"\nThe following splits use ConcatDataset (multiple entities "
              f"concatenated): {', '.join(concat_splits_with_denorm)}")
        print("\nWhy this matters:")
        print("  - Each entity has its own scaler (mean/std) fitted on its data")
        print("  - ConcatDataset loses track of which sample came from which entity")
        print("  - Denormalized metrics use only the first entity's scaler")
        print("  - This gives INCORRECT results for samples from other entities")
        print("\nNormalized metrics are consistent with training but mix")
        print("different real-world scales across entities.")
        print("\nSee: docs/planning/robust_evaluation_metrics_plan.md")
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
    
    # Determine which splits to load (default to test_only=True if not specified)
    test_only = getattr(eval_config, 'test_only', True)
    
    # Get loaders and datasets for each split
    # Note: When tensor cache is enabled, return_type='set' is not supported.
    # We get datasets from loader.dataset instead (handled in evaluate_full_dataset).
    loaders_dict = {}
    datasets_dict = {}
    
    # Only load train/val if not test_only (skip to save time and memory)
    if not test_only:
        print("[Info] Loading train and val datasets (test_only=False)")
        train_loader = data_provider.get_train("loader")
        if train_loader is not None:
            loaders_dict['train'] = train_loader
            # Try to get dataset from loader if available (for scaler access)
            # This works for both regular datasets and tensor cache datasets
            if hasattr(train_loader, 'dataset'):
                train_dataset = train_loader.dataset
                # Handle case where dataset is a dictionary (multiple entities)
                if isinstance(train_dataset, dict):
                    # Use first dataset's scaler (assuming all have same scaler)
                    first_dataset = next(iter(train_dataset.values()))
                    datasets_dict['train'] = first_dataset
                else:
                    datasets_dict['train'] = train_dataset
        
        val_loader = data_provider.get_val("loader")
        if val_loader is not None:
            loaders_dict['val'] = val_loader
            # Try to get dataset from loader if available (for scaler access)
            if hasattr(val_loader, 'dataset'):
                val_dataset = val_loader.dataset
                if isinstance(val_dataset, dict):
                    first_dataset = next(iter(val_dataset.values()))
                    datasets_dict['val'] = first_dataset
                else:
                    datasets_dict['val'] = val_dataset
    
    # Always load test set
    if test_only:
        print("[Info] Loading test dataset only (test_only=True)")
    test_loader = data_provider.get_test("loader")
    if test_loader is not None:
        loaders_dict['test'] = test_loader
        # Try to get dataset from loader if available (for scaler access)
        if hasattr(test_loader, 'dataset'):
            test_dataset = test_loader.dataset
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

