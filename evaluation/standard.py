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
            batch_x, batch_y, _, _, x_hetero, y_hetero, _, _, _, hetero_channel = iter_data

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
    Find checkpoint path based on evaluation configuration.
    
    Args:
        eval_config: Evaluation configuration with model, data, version, input_len, output_len, checkpoint_base
    
    Returns:
        Path to checkpoint directory
    """
    ckpt_pattern = f'_{eval_config.model}_{eval_config.data}_{eval_config.output_len}_{eval_config.input_len}'
    
    if eval_config.version == 'latest':
        ckpt_paths = [os.path.join(eval_config.checkpoint_base, d) for d in os.listdir(eval_config.checkpoint_base) if ckpt_pattern in d]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found with pattern: *{ckpt_pattern}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]
    elif eval_config.version == 'oldest':
        ckpt_paths = [os.path.join(eval_config.checkpoint_base, d) for d in os.listdir(eval_config.checkpoint_base) if ckpt_pattern in d]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found with pattern: *{ckpt_pattern}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[0]
    else:
        pattern = os.path.join(eval_config.checkpoint_base, eval_config.version + ckpt_pattern)
        ckpt_paths = glob.glob(pattern)
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found for pattern: {pattern}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]
    
    return ckpt_path


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
    
    # Find checkpoint
    ckpt_path = find_checkpoint(eval_config)
    print(f"[Info] Using checkpoint path: {ckpt_path}")

    # Load configuration from checkpoint folder
    config_path = os.path.join(ckpt_path, 'args.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file 'args.json' not found in {ckpt_path}")
    
    checkpoint_config = dotdict(json.load(open(config_path)))
    checkpoint_config.model_config = dotdict(checkpoint_config.model_config)
    
    # Use provided data_config override if available, otherwise use checkpoint's data_config
    if hasattr(config, 'data_config') and config.data_config:
        checkpoint_config.data_config = dotdict(yaml.safe_load(open(config.data_config, 'r')))
    else:
        checkpoint_config.data_config = dotdict(checkpoint_config.data_config)
    
    checkpoint_config.gpu = eval_config.device
    checkpoint_config.num_workers = 0
    checkpoint_config.task = eval_config.task
    checkpoint_config.batch_size = 1 if eval_config.filtered_samples is not None else eval_config.batch_size
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{eval_config.device}")
    else:
        device = torch.device("cpu")
        print("[Warning] CUDA is not available, use CPU instead.")
    print(f"[Info] Running on device: {device}")

    # Initialize and load model
    model = model_init(checkpoint_config.model, checkpoint_config.model_config, checkpoint_config).to(device)
    
    ckpt_file = glob.glob(os.path.join(ckpt_path, 'checkpoint*'))
    if not ckpt_file:
        raise FileNotFoundError(f"No checkpoint file (e.g., 'checkpoint.pth') found in {ckpt_path}")
    
    ckpt_file_path = ckpt_file[0]
    print(f"[Info] Loading model from: {ckpt_file_path}")
    checkpoint = torch.load(ckpt_file_path, map_location=device)

    if ckpt_file_path.endswith('.ckpt') and 'state_dict' in checkpoint:
        state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    print(f'[Info] Successfully loaded model: {checkpoint_config.model}')

    # Load data
    data_provider = Data_Provider(checkpoint_config)
    fullloader = data_provider.get_test("loader")

    # Run evaluation
    if eval_config.channel_wise:
        all_mae = {}
        all_mse = {}
        all_sample_num = {}
    else:
        all_mae = 0.0
        all_mse = 0.0
        all_sample_num = 0

    if eval_config.filtered_samples is not None:
        filtered_samples = json.load(open(eval_config.filtered_samples))
        print(f"[Info] Using filtered samples from: {eval_config.filtered_samples}")
    
    for name, loader in fullloader.items():
        print(f"\n[Info] Testing on dataset: {name}")

        if eval_config.filtered_samples is not None:
            indexes = filtered_samples.get(name, [])  # Use .get for safety
            print(f"[Info] Using {len(indexes)} filtered samples for testing.")
            print(f"[Info] Sample indexes: {indexes}")
        else:
            indexes = None
            print("[Info] Using all samples for testing.")
        
        result = evaluate_full_dataset(loader, model, checkpoint_config, device, indexes, eval_config.channel_wise)

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

    print("\n" + "="*50)
    print(" " * 15 + "Overall Test Summary")

    if eval_config.channel_wise:
        # Check if there are any results to summarize
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

