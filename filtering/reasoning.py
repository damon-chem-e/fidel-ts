"""
Reasoning sample filtering module.

This module provides functionality for filtering reasoning samples by comparing
TST vs TGTSF performance, migrated from filter.py and filter_without_inference.py.

The filtering identifies samples where TGTSF significantly outperforms or
underperforms compared to TST, generating filtered sample indexes for use in
LLM reasoning training.
"""

import torch
import pandas as pd
import numpy as np
import os
import json
import glob
import random
from tqdm import tqdm
from models import model_init
from data_provider.data_factory import Data_Provider
from utils.tools import dotdict


def get_reasoning_samples(lossdf, sampling_rate):
    """
    Extract reasoning samples from loss dataframe.
    
    Samples are selected based on mutual loss (relative improvement of TGTSF over TST).
    Positive samples: TGTSF performs better (loss_mutual > 0)
    Negative samples: TGTSF performs worse (loss_mutual < 0)
    
    Args:
        lossdf: DataFrame with columns ['loss_TST', 'loss_TGTSF', 'loss_mutual']
        sampling_rate: Fraction of samples to select from each category
    
    Returns:
        Tuple of (sample_pos, sample_neg) - lists of sample indexes
    """
    lossdf_ok = lossdf[lossdf.loss_mutual > -10]
    lossdf_pos = lossdf_ok[lossdf_ok.loss_mutual > 0]
    lossdf_neg = lossdf_ok[lossdf_ok.loss_mutual < 0]
    
    prob_pos = lossdf_pos['loss_mutual']
    prob_neg = lossdf_neg['loss_mutual'].abs()
    
    sample_num_pos = int(len(lossdf_pos) * sampling_rate)
    sample_num_neg = int(len(lossdf_neg) * sampling_rate)
    
    sample_num_pos = min(sample_num_pos, len(lossdf_pos))
    sample_num_neg = min(sample_num_neg, len(lossdf_neg))
    
    if sample_num_pos > 0:
        sample_pos = lossdf_pos.sample(
            n=sample_num_pos, 
            weights='loss_mutual', 
            replace=False  # Without replacement
        ).index.tolist()
    else:
        sample_pos = []
    
    if sample_num_neg > 0:
        sample_neg = lossdf_neg.sample(
            n=sample_num_neg, 
            weights=prob_neg, 
            replace=False  # Without replacement
        ).index.tolist()
    else:
        sample_neg = []
    
    return sample_pos, sample_neg


def get_lossdf(dataset, model_TST, model_TGTSF, stride, config, device):
    """
    Compute loss dataframe by comparing TST and TGTSF predictions.
    
    Args:
        dataset: Test dataset
        model_TST: Time Series Transformer model
        model_TGTSF: Text-Grounded Time Series Forecasting model
        stride: Stride for sampling (e.g., output_len / 2)
        config: Configuration object
        device: PyTorch device
    
    Returns:
        DataFrame with columns ['loss_TST', 'loss_TGTSF', 'loss_mutual']
    """
    losslist = {}
    for sample_num in tqdm(range(0, len(dataset), stride), desc="Processing samples"):
        with torch.no_grad():
            batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = dataset[sample_num]

            batch_x = torch.tensor(batch_x).unsqueeze(0).float().to(device)
            batch_y = torch.tensor(batch_y).unsqueeze(0).float().to(device)
            batch_y_hetero = torch.tensor(batch_y_hetero).unsqueeze(0).float().to(device)
            hetero_channel = torch.tensor(hetero_channel).unsqueeze(0).float().to(device)

            output_TST = model_TST(x=batch_x)
            output_TST = output_TST[:, -config.output_len:, :]

            output_TGTSF = model_TGTSF(x=batch_x, historical_events=batch_x_hetero, news=batch_y_hetero, dataset_description=hetero_general, channel_description=hetero_channel)
            output_TGTSF = output_TGTSF[:, -config.output_len:, :]

            # Calculate the loss
            loss_TST = torch.nn.MSELoss()(output_TST, batch_y)
            loss_TGTSF = torch.nn.MSELoss()(output_TGTSF, batch_y)

        losslist[sample_num] = {
            'loss_TST': loss_TST.item(),
            'loss_TGTSF': loss_TGTSF.item(),
            'loss_mutual': (loss_TST.item() - loss_TGTSF.item()) / loss_TST.item()
        }
    lossdf = pd.DataFrame.from_dict(losslist, orient='index')
    return lossdf


def find_checkpoint(checkpoint_base, model, data, output_len, input_len, version):
    """
    Find checkpoint path based on model configuration.
    
    Args:
        checkpoint_base: Base directory for checkpoints
        model: Model name
        data: Dataset name
        output_len: Output sequence length
        input_len: Input sequence length
        version: Version string ('latest', 'oldest', or specific version)
    
    Returns:
        Path to checkpoint directory
    """
    ckpt_id = f'_{model}_{data}_{output_len}_{input_len}'
    
    if version == 'latest':
        ckpt_paths = [os.path.join(checkpoint_base, i) for i in os.listdir(checkpoint_base) if ckpt_id in i]
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]
    elif version == 'oldest':
        ckpt_paths = [os.path.join(checkpoint_base, i) for i in os.listdir(checkpoint_base) if ckpt_id in i]
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[0]
    else:
        ckpt_path = version + ckpt_id
        ckpt_path = os.path.join(checkpoint_base, ckpt_path)
    
    return ckpt_path


def get_ahead_string(output_len):
    """
    Convert output_len to ahead string identifier.
    
    Args:
        output_len: Output sequence length
    
    Returns:
        String identifier ('day', 'week', 'hour', 'half_a_day', or 'none')
    """
    if output_len == 24:
        return 'day'
    elif output_len == 168:
        return 'week'
    elif output_len == 12:
        return 'hour'
    elif output_len == 144:
        return 'half_a_day'
    else:
        return 'none'


def filter_samples(config):
    """
    Filter reasoning samples by comparing TST vs TGTSF performance.
    
    This function loads both TST and TGTSF models, runs inference on test datasets,
    computes mutual loss, and generates filtered sample indexes saved as JSON files.
    
    Args:
        config: dotdict containing filtering configuration with structure:
            - filtering: {data, baseline_model, version, input_len, output_len,
                         checkpoint_base, sampling_rate, sample_root, device}
    
    Example:
        >>> from cli.config.loader import load_config_with_nested
        >>> config_hierarchy = load_config_with_nested("configs/experiments/filter_reasoning.yaml")
        >>> filter_samples(config_hierarchy['primary'])
    """
    filter_config = config.filtering
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{filter_config.device}")
    else:
        device = torch.device("cpu")
        print("[Warning] CUDA is not available, use CPU instead.")
    print(f"[Info] Running on device: {device}")

    # Load TST model
    ckpt_path_TST = find_checkpoint(
        filter_config.checkpoint_base,
        filter_config.baseline_model,
        filter_config.data,
        filter_config.output_len,
        filter_config.input_len,
        filter_config.version
    )
    print(f'[Info] Using TST checkpoint path: {ckpt_path_TST}')

    config_TST = dotdict(json.load(open(os.path.join(ckpt_path_TST, 'args.json'))))
    config_TST.model_config = dotdict(config_TST.model_config)
    config_TST.data_config = dotdict(config_TST.data_config)
    config_TST.batch_size = 1
    config_TST.devices = filter_config.device

    model_TST = model_init(config_TST.model, config_TST.model_config, config_TST).to(device)
    ckpt = glob.glob(os.path.join(ckpt_path_TST, 'checkpoint*'))[0]
    checkpoint = torch.load(ckpt)

    if ckpt.endswith('.ckpt'):
        state_dict = {key.replace("model.model.", "model."): value for key, value in checkpoint['state_dict'].items()}
    else:
        state_dict = checkpoint

    model_TST.load_state_dict(state_dict) 
    model_TST.eval()
    print(f'[Info] Using TST model: {config_TST.model}')

    # Load TGTSF model
    TG_model = 'TGTSF'
    ckpt_path_TGTSF = find_checkpoint(
        filter_config.checkpoint_base,
        TG_model,
        filter_config.data,
        filter_config.output_len,
        filter_config.input_len,
        filter_config.version
    )
    print(f'[Info] Using TGTSF checkpoint path: {ckpt_path_TGTSF}')

    config_TGTSF = dotdict(json.load(open(os.path.join(ckpt_path_TGTSF, 'args.json'))))
    config_TGTSF.model_config = dotdict(config_TGTSF.model_config)
    config_TGTSF.data_config = dotdict(config_TGTSF.data_config)
    config_TGTSF.batch_size = 1
    config_TGTSF.devices = filter_config.device

    model_TGTSF = model_init(config_TGTSF.model, config_TGTSF.model_config, config_TGTSF).to(device)
    ckpt = glob.glob(os.path.join(ckpt_path_TGTSF, 'checkpoint*'))[0]
    checkpoint = torch.load(ckpt)

    if ckpt.endswith('.ckpt'):
        state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
    else:
        state_dict = checkpoint

    model_TGTSF.load_state_dict(state_dict)
    model_TGTSF.eval()
    print(f'[Info] Using TGTSF model: {config_TGTSF.model}')

    # Load data
    id_data = Data_Provider(config_TGTSF)
    fullsets = id_data.get_test('set')
    print(f'[Info] fullset keys: {fullsets.keys()}')

    ahead = get_ahead_string(filter_config.output_len)

    # Check for existing samples
    try:
        existing_path = os.path.join(filter_config.sample_root, f'{filter_config.data}_sample_{ahead}.json')
        existing = json.load(open(existing_path))
        existing = existing.keys()
    except:
        existing = []

    # Set seeds for reproducibility
    np.random.seed(114514)
    random.seed(114514)

    sample_dict = {}
    total_pos_samples, total_neg_samples = 0, 0

    for i in fullsets.keys():
        if i in existing:
            continue
        
        dataset = fullsets[i]
        print(f"[Info] handling {i}")

        lossdf = get_lossdf(dataset, model_TST, model_TGTSF, int(filter_config.output_len / 2), config_TGTSF, device)
        lossdf.to_csv(os.path.join(ckpt_path_TGTSF, f'lossdf_{i}.csv'))

        try:
            sample_pos, sample_neg = get_reasoning_samples(lossdf, filter_config.sampling_rate)
            
            print(f"[Info]: selected {len(sample_pos)} pos samples and {len(sample_neg)} neg samples in subset {i}")
            total_pos_samples += len(sample_pos)
            total_neg_samples += len(sample_neg)

            samples = sample_pos + sample_neg

        except Exception as e:
            print(f'[Error] on {i}: {e}')
            continue

        print(f"[Info] generated: {samples}")

        sample_dict[i] = samples
        os.makedirs(filter_config.sample_root, exist_ok=True)
        with open(os.path.join(filter_config.sample_root, f'{filter_config.data}_sample_{ahead}.json'), 'w') as f:
            json.dump(sample_dict, f)

    print("\n" + "="*50)
    print("Final Statistics Summary")
    print("="*50)
    print(f"Total Positive Candidates Across All Processed Subsets: {total_pos_samples}")
    print(f"Total Negative Candidates Across All Processed Subsets: {total_neg_samples}")
    print("="*50)


def filter_from_csv(config):
    """
    Filter reasoning samples from pre-computed loss CSV files.
    
    This function reads pre-computed loss dataframes stored as CSV files and
    generates filtered sample indexes. More efficient than filter_samples() for
    re-filtering with different sampling rates.
    
    Args:
        config: dotdict containing filtering configuration with structure:
            - filtering: {data, baseline_model, version, input_len, output_len,
                         checkpoint_base, sampling_rate, sample_root}
    
    Example:
        >>> from cli.config.loader import load_config_with_nested
        >>> config_hierarchy = load_config_with_nested("configs/experiments/filter_reasoning.yaml")
        >>> filter_from_csv(config_hierarchy['primary'])
    """
    filter_config = config.filtering

    TG_model = 'TGTSF'
    ckpt_path = find_checkpoint(
        filter_config.checkpoint_base,
        TG_model,
        filter_config.data,
        filter_config.output_len,
        filter_config.input_len,
        filter_config.version
    )
    print(f'[Info] Using checkpoint path: {ckpt_path}')

    config_obj = dotdict(json.load(open(os.path.join(ckpt_path, 'args.json'))))
    config_obj.model_config = dotdict(config_obj.model_config)
    config_obj.data_config = dotdict(config_obj.data_config)

    id_data = Data_Provider(config_obj)
    fullsets = id_data.get_test('set')
    print(f'[Info] fullset keys: {fullsets.keys()}')

    # Set seeds for reproducibility
    np.random.seed(114514)
    random.seed(114514)

    ahead = get_ahead_string(filter_config.output_len)

    # Check for existing samples
    try:
        existing_path = os.path.join(filter_config.sample_root, f'{filter_config.data}_sample_{ahead}.json')
        existing = json.load(open(existing_path))
        existing = existing.keys()
    except:
        existing = []

    sample_dict = {}
    total_pos_samples, total_neg_samples = 0, 0

    for i in fullsets.keys():
        if i in existing:
            continue

        dataset = fullsets[i]
        print(f"[Info] handling {i}")

        csv_path = os.path.join(ckpt_path, f'lossdf_{i}.csv')
        if not os.path.exists(csv_path):
            print(f"[Warning] CSV file not found: {csv_path}, skipping subset {i}")
            continue

        lossdf = pd.read_csv(csv_path, index_col=0)

        try:
            sample_pos, sample_neg = get_reasoning_samples(lossdf, filter_config.sampling_rate)
            
            print(f"[Info]: selected {len(sample_pos)} pos samples and {len(sample_neg)} neg samples in subset {i}")
            total_pos_samples += len(sample_pos)
            total_neg_samples += len(sample_neg)

            samples = sample_pos + sample_neg

        except Exception as e:
            print(f'[Error] on {i}: {e}')
            continue

        print(f"[Info] generated: {samples}")

        sample_dict[i] = samples
        os.makedirs(filter_config.sample_root, exist_ok=True)
        with open(os.path.join(filter_config.sample_root, f'{filter_config.data}_sample_{ahead}.json'), 'w') as f:
            json.dump(sample_dict, f)
        

    print("\n" + "="*50)
    print("Final Statistics Summary")
    print("="*50)
    print(f"Total Positive Candidates Across All Processed Subsets: {total_pos_samples}")
    print(f"Total Negative Candidates Across All Processed Subsets: {total_neg_samples}")
    print("="*50)

