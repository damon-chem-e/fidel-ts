"""
TSF/TGTSF visualization module.

This module provides visualization functionality for standard time series
forecasting models, migrated from visualize.py.
"""

import torch
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from models import model_init
from data_provider.data_factory import Data_Provider
from utils.tools import dotdict


def plot_prediction(indate, input_data, outdate, output_data, prediction_data, 
                   data_id, dataset_name, model_name, sample_id, channel_id, 
                   img_path, figsize=(15, 7), dpi=300):
    """
    Create prediction visualization plot.
    
    Args:
        indate: Input timestamps
        input_data: Input time series data
        outdate: Output timestamps
        output_data: Ground truth values
        prediction_data: Model predictions
        data_id: Data subset ID
        dataset_name: Name of the dataset
        model_name: Name of the model
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
    plt.title(f'Prediction Visualization for {dataset_name}: {data_id}: channel {channel_id}, Model: {model_name}, Sample: {sample_id}')
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
    Visualize TSF/TGTSF model predictions.
    
    This function loads a trained model checkpoint, runs inference on a test sample,
    and generates a visualization comparing input history, ground truth, and predictions.
    
    Args:
        config: dotdict containing visualization configuration with structure:
            - visualization: {ckpt_base, ckpt_id, data_id, sample_id, img_path, task, device}
            - model: {name} (for display)
            - data: {name} (for display)
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
    
    # Load config and checkpoint
    ckpt_path = os.path.join(vis_config.ckpt_base, vis_config.ckpt_id)
    print(f"Loading checkpoint from: {ckpt_path}")

    checkpoint_config = dotdict(json.load(open(os.path.join(ckpt_path, 'args.json'))))
    checkpoint_config.model_config = dotdict(checkpoint_config.model_config)
    checkpoint_config.data_config = dotdict(checkpoint_config.data_config)
    checkpoint_config.gpu = vis_config.device
    checkpoint_config.batch_size = 1

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{vis_config.device}")
    else:
        device = torch.device("cpu")
        print("[Warning] CUDA is not available, use CPU instead.")
    print(f"[Info] Running on device: {device}")

    # Load data
    D = Data_Provider(checkpoint_config)
    test_set = D.get_test("set")

    # Initialize and load model
    model = model_init(checkpoint_config.model, checkpoint_config.model_config, checkpoint_config).to(device)
    model.load_state_dict(torch.load(os.path.join(ckpt_path, 'checkpoint.pth'), map_location=device))
    model.eval()

    print(f"--- Perform on ID {vis_config.data_id}, sample {vis_config.sample_id} ---")
    seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = test_set[vis_config.data_id][vis_config.sample_id]

    input_tensor = torch.tensor(seq_x).to(device).float().unsqueeze(0)
    output_tensor = torch.tensor(seq_y).to(device).float().unsqueeze(0)
    y_hetero = torch.tensor(y_hetero).to(device).float().unsqueeze(0)
    hetero_channel = torch.tensor(hetero_channel).to(device).float().unsqueeze(0)

    with torch.inference_mode():
        with torch.no_grad():
            if vis_config.task == 'TSF':
                # For TSF, we only need seq_x
                prediction_tensor = model(x=input_tensor)
                prediction_tensor = prediction_tensor[:, -checkpoint_config.output_len:, :]
            elif vis_config.task == 'TGTSF':
                # For TGTSF, we need to pass news and channel description
                prediction_tensor = model(x=input_tensor, news=y_hetero, channel_description=hetero_channel)
                prediction_tensor = prediction_tensor[:, -checkpoint_config.output_len:, :]
            else:
                raise ValueError("Task type must be either 'TSF' or 'TGTSF'.")

    # Visualization
    indate_dt = pd.to_datetime([str(i) for i in x_time], format='%Y%m%d%H%M%S')
    outdate_dt = pd.to_datetime([str(i) for i in y_time], format='%Y%m%d%H%M%S')

    input_np = input_tensor.cpu().numpy().squeeze()
    output_np = output_tensor.cpu().numpy().squeeze()
    prediction_np = prediction_tensor.cpu().numpy().squeeze()

    input_np = input_np if channel_id == 'all' else input_np[:, int(channel_id)]
    output_np = output_np if channel_id == 'all' else output_np[:, int(channel_id)]
    prediction_np = prediction_np if channel_id == 'all' else prediction_np[:, int(channel_id)]

    img_path = os.path.join(vis_config.img_path, vis_config.task, f"{vis_config.ckpt_id}_subset-{vis_config.data_id}_sample-{vis_config.sample_id}.png")
    
    plot_prediction(indate=indate_dt,
                    input_data=input_np,
                    outdate=outdate_dt,
                    output_data=output_np,
                    prediction_data=prediction_np,
                    data_id=vis_config.data_id,
                    sample_id=vis_config.sample_id, 
                    channel_id=channel_id,
                    img_path=img_path,
                    figsize=figsize,
                    dpi=dpi)

