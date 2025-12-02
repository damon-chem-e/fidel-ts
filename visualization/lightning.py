"""
Lightning visualization module.

This module provides visualization functionality for PyTorch Lightning-trained models,
migrated from visualize_lightning.py.
"""

import torch
import pandas as pd
import numpy as np
import os
import json
import glob
import matplotlib.pyplot as plt
from models import model_init
from data_provider.data_factory import Data_Provider
from utils.tools import dotdict


def plot_prediction(indate, input_data, outdate, output_data, prediction_data,
                   dataset_name, model_name, subset_name, sample_id, channel_id,
                   img_path, figsize=(15, 7), dpi=300):
    """
    Create prediction visualization plot for Lightning models.
    
    Args:
        indate: Input timestamps
        input_data: Input time series data
        outdate: Output timestamps
        output_data: Ground truth values
        prediction_data: Model predictions
        dataset_name: Name of the dataset
        model_name: Name of the model
        subset_name: Data subset name
        sample_id: Sample index
        channel_id: Channel ID or 'all'
        img_path: Path to save the image
        figsize: Figure size tuple (width, height)
        dpi: Resolution in dots per inch
    """
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(indate, input_data, label='Input History')
    plt.plot(outdate, output_data, label='Ground Truth')
    plt.plot(outdate, prediction_data, label='Prediction', linestyle='--')
    plt.title(f'Prediction Visualization for {dataset_name}: {subset_name}: channel {channel_id}, Model: {model_name}, Sample: {sample_id}')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()

    img_dir = os.path.dirname(img_path)
    if img_dir and not os.path.exists(img_dir):
        os.makedirs(img_dir)
        print(f"Created directory: {img_dir}")
    
    plt.savefig(img_path)
    plt.close()
    print(f"Prediction plot saved to {img_path}")


def visualize(config, plotting_config=None):
    """
    Visualize PyTorch Lightning-trained model predictions.
    
    This function loads a Lightning checkpoint, runs inference on a test sample,
    and generates a visualization. Supports flexible checkpoint selection (best/last).
    
    Args:
        config: dotdict containing visualization configuration with structure:
            - visualization: {checkpoint_base, version, input_len, output_len, checkpoint_file,
                            device, task, channel_id, vis_subset, vis_sample_id, vis_save_path, fig_name}
            - model: {name}
            - data: {name}
        plotting_config: Optional dotdict with plotting parameters:
            - figure: {figsize, dpi}
            - channels: {default} (channel_id override)
    
    Example:
        >>> from cli.config.loader import load_config_with_nested
        >>> config_hierarchy = load_config_with_nested("configs/experiments/dlinear_solar.yaml")
        >>> visualize(config_hierarchy['primary'], config_hierarchy['nested'].get('plotting'))
    """
    vis_config = config.visualization
    
    # Extract plotting parameters from subconfig if available
    if plotting_config:
        figsize = tuple(plotting_config.figure.get('figsize', [15, 7]))
        dpi = plotting_config.figure.get('dpi', 300)
        channel_id = plotting_config.channels.get('default', vis_config.get('channel_id', 'all'))
    else:
        figsize = (15, 7)
        dpi = 300
        channel_id = vis_config.get('channel_id', 'all')
    
    data = config.data.name
    model = config.model.name
    version = vis_config.version
    input_len = vis_config.input_len
    output_len = vis_config.output_len
    ckpt_base = vis_config.checkpoint_base

    ckpt_id = f'_{model}_{data}_{output_len}_{input_len}'

    if version in ['latest', 'newest']:
        ckpt_paths = [i for i in os.listdir(ckpt_base) if ckpt_id in i]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found for pattern: *{ckpt_id}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]
        fig_name = ckpt_path
        ckpt_path = os.path.join(ckpt_base, ckpt_path)
    else:
        ckpt_path = version + ckpt_id
        ckpt_path = os.path.join(ckpt_base, ckpt_path)
        fig_name = vis_config.get('fig_name', ckpt_path)

    print(f'[Info] Using checkpoint path: {ckpt_path}')

    config_path = os.path.join(ckpt_path, 'args.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"args.json not found in {ckpt_path}")
    
    checkpoint_config = dotdict(json.load(open(config_path)))
    checkpoint_config.model_config = dotdict(checkpoint_config.model_config)
    checkpoint_config.data_config = dotdict(checkpoint_config.data_config)
    checkpoint_config.model = model
    checkpoint_config.data = data
    checkpoint_config.batch_size = 1

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{vis_config.device}")
    else:
        device = torch.device("cpu")
        print("[Warning] CUDA is not available, use CPU instead.")
    print(f"[Info] Running on device: {device}")

    model_obj = model_init(checkpoint_config.model, checkpoint_config.model_config, checkpoint_config).to(device)
    
    checkpoint_file = vis_config.get('checkpoint_file', 'best')
    ckpt_file_pattern = os.path.join(ckpt_path, 'checkpoint*') if checkpoint_file == "best" else os.path.join(ckpt_path, 'last*')
    
    ckpt_files = glob.glob(ckpt_file_pattern)
    if not ckpt_files:
        raise FileNotFoundError(f"Checkpoint file not found in {ckpt_path} with pattern {ckpt_file_pattern}")
    ckpt_file = ckpt_files[0]
    
    print(f"[Info] Loading model from: {ckpt_file}")
    checkpoint = torch.load(ckpt_file, map_location=device)

    if 'state_dict' in checkpoint:
        if vis_config.task == 'TGTSF':
            state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
        elif vis_config.task == 'TSF':
            state_dict = {key.replace("model.model.", "model."): value for key, value in checkpoint['state_dict'].items() if 'news' not in key}
    else:
        state_dict = checkpoint

    model_obj.load_state_dict(state_dict)
    model_obj.eval()

    print(f'[Info] Successfully loaded model: {checkpoint_config.model}')

    id_data = Data_Provider(checkpoint_config)
    fullsets = id_data.get_test('set')
    print(f'[Info] Found {len(fullsets)} datasets to test: {list(fullsets.keys())}')

    # Run visualization
    subset_name = vis_config.vis_subset
    sample_id = vis_config.vis_sample_id

    print(f"[Info] Running visualization for subset '{subset_name}', sample ID {sample_id}")

    if subset_name not in fullsets:
        raise KeyError(f"Subset '{subset_name}' not found. Available subsets: {list(fullsets.keys())}")
    
    dataset_to_vis = fullsets[subset_name]

    if not (0 <= sample_id < len(dataset_to_vis)):
        raise IndexError(f"Sample ID {sample_id} is out of bounds for subset '{subset_name}' which has {len(dataset_to_vis)} samples.")

    seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = dataset_to_vis[sample_id]

    input_tensor = torch.tensor(seq_x).to(device).float().unsqueeze(0)
    y_hetero = torch.tensor(y_hetero).to(device).float().unsqueeze(0)
    hetero_channel = torch.tensor(hetero_channel).to(device).float().unsqueeze(0)

    with torch.no_grad():
        prediction_tensor = model_obj(x=input_tensor) if vis_config.task == 'TSF' else model_obj(x=input_tensor, news=y_hetero, channel_description=hetero_channel)
        prediction_tensor = prediction_tensor[:, -checkpoint_config.output_len:, :]

    indate_dt = pd.to_datetime([str(i) for i in x_time], format='%Y%m%d%H%M%S', errors='coerce')
    outdate_dt = pd.to_datetime([str(i) for i in y_time], format='%Y%m%d%H%M%S', errors='coerce')

    input_np = input_tensor.cpu().numpy().squeeze()
    output_np = np.array(seq_y).squeeze()
    prediction_np = prediction_tensor.cpu().numpy().squeeze()
    
    save_path = os.path.join(vis_config.vis_save_path, vis_config.task, f"{fig_name}_subset-{subset_name}_sample-{sample_id}.png")

    input_np = input_np if channel_id == 'all' else input_np[:, int(channel_id)]
    output_np = output_np if channel_id == 'all' else output_np[:, int(channel_id)]
    prediction_np = prediction_np if channel_id == 'all' else prediction_np[:, int(channel_id)]

    plot_prediction(
        indate=indate_dt,
        input_data=input_np,
        outdate=outdate_dt,
        output_data=output_np,
        prediction_data=prediction_np,
        dataset_name=data,
        model_name=model,
        subset_name=subset_name,
        sample_id=sample_id,
        channel_id=channel_id,
        img_path=save_path,
        figsize=figsize,
        dpi=dpi
    )

