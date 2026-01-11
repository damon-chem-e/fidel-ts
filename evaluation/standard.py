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
from pathlib import Path
from tqdm import tqdm
from models import model_init
from data_provider.data_factory import Data_Provider
from utils.tools import dotdict
from utils.metrics import MAE, MSE


def evaluate_full_dataset(loader, model, config, device, indexes, channel_wise):
    """
    Evaluates all samples in a dataset using a DataLoader for efficient batch processing.
    Calculates the true MSE and MAE over the entire dataset, with an option for channel-wise evaluation.
    
    Args:
        loader: DataLoader for test dataset
        model: Trained model
        config: Configuration object
        device: PyTorch device
        indexes: List of sample indexes to evaluate (None for all)
        channel_wise: Whether to compute channel-wise metrics
    
    Returns:
        If channel_wise: (channel_mse, channel_mae, channel_counts)
        Otherwise: (total_mse, total_mae, num_samples)
    """
    total_mse, total_mae, num_samples = 0.0, 0.0, 0
    channel_mse, channel_mae, channel_counts = None, None, None

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
                if channel_mse is None:
                    C = prediction.shape[2]
                    channel_mse = [0.0] * C
                    channel_mae = [0.0] * C
                    channel_counts = [0] * C
                for k in range(prediction.shape[2]):
                    mse_loss = torch.nn.MSELoss()(prediction[:, :, k], batch_y[:, :, k])
                    mae_loss = torch.nn.L1Loss()(prediction[:, :, k], batch_y[:, :, k])
                    channel_mse[k] += mse_loss.item() * batch_y.size(0)
                    channel_mae[k] += mae_loss.item() * batch_y.size(0)
                    channel_counts[k] += batch_y.size(0)
            else:
                mse_loss = torch.nn.MSELoss()(prediction, batch_y)
                mae_loss = torch.nn.L1Loss()(prediction, batch_y)
                total_mae += mae_loss.item() * batch_y.size(0)
                total_mse += mse_loss.item() * batch_y.size(0)
                num_samples += batch_y.size(0)

    if channel_wise:
        return channel_mse, channel_mae, channel_counts
    else:
        return total_mse, total_mae, num_samples


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


def _run_evaluation_loop(fullloader, model, checkpoint_config, device, eval_config):
    """
    Run evaluation loop over all datasets.
    
    Args:
        fullloader: Dictionary of dataset loaders
        model: Trained model
        checkpoint_config: Checkpoint configuration
        device: PyTorch device
        eval_config: Evaluation configuration
        
    Returns:
        tuple: (all_mse, all_mae, all_sample_num) - aggregated results
    """
    # Initialize result containers
    if eval_config.channel_wise:
        all_mae = {}
        all_mse = {}
        all_sample_num = {}
    else:
        all_mae = 0.0
        all_mse = 0.0
        all_sample_num = 0
    
    # Load filtered samples if provided
    filtered_samples = None
    if eval_config.filtered_samples is not None:
        filtered_samples = json.load(open(eval_config.filtered_samples))
        print(f"[Info] Using filtered samples from: {eval_config.filtered_samples}")
    
    # Evaluate each dataset
    for name, loader in fullloader.items():
        print(f"\n[Info] Testing on dataset: {name}")
        
        # Get indexes for this dataset
        if filtered_samples is not None:
            indexes = filtered_samples.get(name, [])
            print(f"[Info] Using {len(indexes)} filtered samples for testing.")
            print(f"[Info] Sample indexes: {indexes}")
        else:
            indexes = None
            print("[Info] Using all samples for testing.")
        
        # Run evaluation
        result = evaluate_full_dataset(loader, model, checkpoint_config, device, indexes, eval_config.channel_wise)
        
        # Process results
        if eval_config.channel_wise:
            channel_mse, channel_mae, channel_counts = result
            if channel_mse is None or sum(channel_counts) == 0:
                print(f"-> No valid samples found in '{name}'")
            else:
                all_mse[name] = channel_mse
                all_mae[name] = channel_mae
                all_sample_num[name] = channel_counts
                avg_ch_mse = [m / count if count > 0 else 0 for m, count in zip(channel_mse, channel_counts)]
                avg_ch_mae = [m / count if count > 0 else 0 for m, count in zip(channel_mae, channel_counts)]
                print(f"-> Results for '{name}': Channel-wise MSE = {avg_ch_mse}, Channel-wise MAE = {avg_ch_mae}")
                print(f"-> Results for '{name}': Overall Channel MSE = {sum(avg_ch_mse) / len(avg_ch_mse):.7f}, Overall Channel MAE = {sum(avg_ch_mae) / len(avg_ch_mae):.7f}")
        else:
            total_mse, total_mae, num_samples = result
            if num_samples > 0:
                avg_mse = total_mse / num_samples
                avg_mae = total_mae / num_samples
                print(f"-> Results for '{name}': MSE = {avg_mse:.7f}, MAE = {avg_mae:.7f}")
                all_mse += total_mse
                all_mae += total_mae
                all_sample_num += num_samples
            else:
                print(f"-> No valid samples found in '{name}'")
    
    return all_mse, all_mae, all_sample_num


def _print_summary(all_mse, all_mae, all_sample_num, eval_config):
    """
    Print evaluation summary.
    
    Args:
        all_mse: Aggregated MSE results
        all_mae: Aggregated MAE results
        all_sample_num: Aggregated sample counts
        eval_config: Evaluation configuration
    """
    print("\n" + "="*50)
    print(" " * 15 + "Overall Test Summary")
    
    if eval_config.channel_wise:
        if not all_mse:
            print("-> No results to summarize.")
        else:
            sum_mse = [sum(m) for m in zip(*all_mse.values())]
            sum_mae = [sum(m) for m in zip(*all_mae.values())]
            sum_counts = [sum(c) for c in zip(*all_sample_num.values())]
            overall_mse_list = [m / c if c > 0 else 0 for m, c in zip(sum_mse, sum_counts)]
            overall_mae_list = [m / c if c > 0 else 0 for m, c in zip(sum_mae, sum_counts)]
            print(f"-> Overall Results (All Subsets): Channel-wise MSE = {overall_mse_list}, Channel-wise MAE = {overall_mae_list}")
            overall_mse = sum(overall_mse_list) / len(overall_mse_list) if overall_mse_list else 0
            overall_mae = sum(overall_mae_list) / len(overall_mae_list) if overall_mae_list else 0
            print(f"-> Overall Results (All Subsets): MSE = {overall_mse:.7f}, MAE = {overall_mae:.7f}")
    else:
        if all_sample_num > 0:
            print(f"-> Overall Results (All Subsets): MSE = {all_mse / all_sample_num:.7f}, MAE = {all_mae / all_sample_num:.7f}")
        else:
            print("-> No samples were processed.")
    
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
    
    # Load data
    data_provider = Data_Provider(checkpoint_config)
    fullloader = data_provider.get_test("loader")
    
    # Run evaluation loop
    all_mse, all_mae, all_sample_num = _run_evaluation_loop(
        fullloader, model, checkpoint_config, device, eval_config
    )
    
    # Print summary
    _print_summary(all_mse, all_mae, all_sample_num, eval_config)

