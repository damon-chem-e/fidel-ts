"""
Lightning evaluation module for PyTorch Lightning-trained models.

This module provides evaluation functionality for models trained with PyTorch Lightning,
migrated from criterias_lightning.py.
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


def run_test(loader, model, config, device, indexes, channel_wise):
    """
    Run evaluation test on a data loader.
    
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
    total_mse, total_mae = 0.0, 0.0
    num_samples = 0

    channel_mse = None
    channel_mae = None
    channel_counts = None

    for i, iter_data in tqdm(enumerate(loader), total=len(loader), desc="Running tests"):
        if indexes is not None and i not in indexes:
            continue
        with torch.no_grad():
            batch_x, batch_y, _, _, _, y_hetero, _, _, _, hetero_channel = iter_data

            batch_x = torch.tensor(batch_x).to(device)
            batch_y = torch.tensor(batch_y).to(device)
            y_hetero = torch.tensor(y_hetero).to(device)
            hetero_channel = torch.tensor(hetero_channel).to(device)

            prediction = model(x=batch_x) if config.task == 'TSF' else model(x=batch_x, news=y_hetero, channel_description=hetero_channel)
            prediction = prediction[:, -config.output_len:, :]  # [B, L, C]

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
    Find Lightning checkpoint path based on evaluation configuration.
    
    Args:
        eval_config: Evaluation configuration with model, data, version, input_len, output_len, checkpoint_base
    
    Returns:
        Path to checkpoint directory
    """
    ckpt_id = f'_{eval_config.model}_{eval_config.data}_{eval_config.output_len}_{eval_config.input_len}_pl'
    
    if eval_config.version == 'latest':
        ckpt_paths = [os.path.join(eval_config.checkpoint_base, i) for i in os.listdir(eval_config.checkpoint_base) if ckpt_id in i]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found for pattern: *{ckpt_id}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]
    elif eval_config.version == 'oldest':
        ckpt_paths = [os.path.join(eval_config.checkpoint_base, i) for i in os.listdir(eval_config.checkpoint_base) if ckpt_id in i]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found for pattern: *{ckpt_id}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[0]
    else:
        pattern = os.path.join(eval_config.checkpoint_base, eval_config.version + ckpt_id)
        ckpt_paths = glob.glob(pattern)
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found for pattern: {pattern}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]
    
    return ckpt_path


def evaluate(config):
    """
    Evaluate PyTorch Lightning-trained models.
    
    This function loads a Lightning checkpoint, runs inference on test datasets,
    and calculates evaluation metrics. Handles Lightning checkpoint format.
    
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
    
    # Find checkpoint
    ckpt_path = find_checkpoint(eval_config)
    print(f'[Info] Using checkpoint path: {ckpt_path}')

    config_path = os.path.join(ckpt_path, 'args.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"args.json not found in {ckpt_path}")
    
    checkpoint_config = dotdict(json.load(open(config_path)))
    checkpoint_config.model_config = dotdict(checkpoint_config.model_config)
    checkpoint_config.data_config = dotdict(checkpoint_config.data_config) if not hasattr(config, 'data_config') or config.data_config is None else dotdict(yaml.safe_load(open(config.data_config, 'r')))
    checkpoint_config.devices = eval_config.device
    checkpoint_config.num_workers = 0
    checkpoint_config.batch_size = 1 if eval_config.filtered_samples is not None else eval_config.batch_size
    checkpoint_config.task = eval_config.task

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{eval_config.device}")
    else:
        device = torch.device("cpu")
        print("[Warning] CUDA is not available, use CPU instead.")
    print(f"[Info] Running on device: {device}")

    model = model_init(checkpoint_config.model, checkpoint_config.model_config, checkpoint_config).to(device)
    
    ckpt_file = glob.glob(os.path.join(ckpt_path, 'checkpoint*'))
    if not ckpt_file:
        raise FileNotFoundError(f"Checkpoint file not found in {ckpt_path}")
    ckpt_file = ckpt_file[0]
    
    print(f"[Info] Loading model from: {ckpt_file}")
    checkpoint = torch.load(ckpt_file, map_location=device)

    if ckpt_file.endswith('.ckpt'):
        if eval_config.model == "PatchTST":
            state_dict = {key.replace("model.model.", "model."): value for key, value in checkpoint['state_dict'].items()} 
        else:
            state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()} 

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print(f'[Info] Successfully loaded model: {checkpoint_config.model}')

    # Prepare datasets
    id_data = Data_Provider(checkpoint_config)
    fullloader = id_data.get_test('loader')
    print(f'[Info] Found {len(fullloader)} datasets to test: {list(fullloader.keys())}')

    # Handle filtered samples if provided
    if eval_config.filtered_samples is not None:
        filtered_samples = json.load(open(eval_config.filtered_samples))
        print(f"[Info] Using filtered samples from: {eval_config.filtered_samples}")

    all_mae = 0.0 if not eval_config.channel_wise else {}
    all_mse = 0.0 if not eval_config.channel_wise else {}
    all_sample_num = 0 if not eval_config.channel_wise else {}

    for name, loader in fullloader.items():
        print(f"\n[Info] Testing on dataset: {name}")

        if eval_config.filtered_samples is not None:
            indexes = filtered_samples[name]
            print(f"[Info] Using {len(indexes)} filtered samples for testing.")
            print(f"[Info] Sample indexes: {indexes}")
        else:
            indexes = None
            print("[Info] Using all samples for testing.")
        
        result = run_test(loader, model, checkpoint_config, device, indexes, eval_config.channel_wise)
        if eval_config.channel_wise:
            channel_mse, channel_mae, channel_counts = result
            if sum(channel_counts) == 0:
                print(f"-> No index found in '{name}'")
            else:
                all_mse[name] = channel_mse
                all_mae[name] = channel_mae
                all_sample_num[name] = channel_counts
                avg_ch_mse = [m / count if count > 0 else 0 for m, count in zip(channel_mse, channel_counts)]
                avg_ch_mae = [m / count if count > 0 else 0 for m, count in zip(channel_mae, channel_counts)]
                print(f"-> Results for '{name}': Channel-wise MSE = {avg_ch_mse}, MAE = {avg_ch_mae}")
                print(f"-> Results for '{name}': All channel MSE = {sum(avg_ch_mse) / len(avg_ch_mse):.7f}, MAE = {sum(avg_ch_mae) / len(avg_ch_mae):.7f}")
        
        else:
            total_mse, total_mae, num_samples = result
            if num_samples > 0:
                all_mse += total_mse
                all_mae += total_mae
                all_sample_num += num_samples
                avg_mse = total_mse / num_samples
                avg_mae = total_mae / num_samples
                print(f"-> Results for '{name}': MSE = {avg_mse:.7f}, MAE = {avg_mae:.7f}")
            else:
                print(f"-> No index found in '{name}'")
            
    print("\n" + "="*50)
    print(" " * 15 + "Overall Test Summary")
    if eval_config.channel_wise:
        sum_mse = [sum(m) for m in zip(*all_mse.values())]
        sum_mae = [sum(m) for m in zip(*all_mae.values())]
        sum_counts = [sum(c) for c in zip(*all_sample_num.values())]
        overall_mse = [m / c if c > 0 else 0 for m, c in zip(sum_mse, sum_counts)]
        overall_mae = [m / c if c > 0 else 0 for m, c in zip(sum_mae, sum_counts)]
        print(f"-> Results for all subsets: channel-wise MSE = {overall_mse}, channel-wise MAE = {overall_mae}")
        print(f"-> Results for all subsets: All channel MSE = {sum(overall_mse) / len(overall_mse):.7f}, MAE = {sum(overall_mae) / len(overall_mae):.7f}")
    else:
        print(f"-> Results for all subsets: MSE = {all_mse / all_sample_num:.7f}, MAE = {all_mae / all_sample_num:.7f}")
    print("="*50)

