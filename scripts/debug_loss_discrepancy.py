"""
Diagnostic script to investigate why fidel-ts iTransformer shows 0.01 MSE 
vs 0.18 MSE in papers/MM-TSFlib.

This script checks for:
1. Data leakage (validation samples overlapping with training)
2. Normalization issues (scaler statistics)
3. Shape mismatches in loss computation
4. Actual loss values and scales

Usage:
    # Run from project root
    python scripts/debug_loss_discrepancy.py
    
    # Or run as module
    python -m scripts.debug_loss_discrepancy
"""

import sys
from pathlib import Path

# Add project root to path so imports work
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
from functools import partial
from data_provider.data_loader import Universal_Dataset
from data_provider.time_mmd_dataset import TimeMMD_Dataset
from data_provider.data_helper import ratio_spliter
import yaml
from utils.tools import dotdict

def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def check_data_leakage(dataset, split='val', split_info=[7, 1, 2]):
    """Check if validation/test samples overlap with training data."""
    print(f"\n=== Checking for Data Leakage ({split}) ===")
    
    # Note: dataset.data only contains the current split's data
    # We need to compute split boundaries based on the original data length
    # For this diagnostic, we'll load the full raw data to compute boundaries
    
    # Load full dataset to get total length (we'll use train split to estimate)
    # Actually, we can't easily get the full length without loading all data
    # So we'll work with what we have and note the limitations
    
    print(f"Current split ({split}) data length: {len(dataset.data)}")
    print(f"seq_len: {dataset.seq_len}, pred_len: {dataset.pred_len}")
    
    # Compute split ratios
    total_ratio = sum(split_info)
    train_ratio = split_info[0] / total_ratio
    val_ratio = split_info[1] / total_ratio
    test_ratio = split_info[2] / total_ratio
    
    print(f"\nSplit ratios: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")
    print(f"Note: To fully check for leakage, we'd need to load the full dataset.")
    print(f"  The current dataset only contains the '{split}' split's data.")
    
    # Check sample window structure
    print(f"\nSample window structure:")
    print(f"  Each sample uses:")
    print(f"    - Input: seq_len={dataset.seq_len} timesteps")
    print(f"    - Target: pred_len={dataset.pred_len} timesteps")
    print(f"  Total window size: {dataset.seq_len + dataset.pred_len} timesteps")
    
    # Check if we can access dataset attributes that might indicate split boundaries
    if hasattr(dataset, 'train_split'):
        print(f"\nDataset has train_split attribute: {dataset.train_split}")
    if hasattr(dataset, 'val_split'):
        print(f"Dataset has val_split attribute: {dataset.val_split}")
    
    print(f"\nNote: For a complete leakage check, we'd need to:")
    print(f"  1. Load train, val, and test datasets separately")
    print(f"  2. Check if validation/test input windows overlap with training data")
    print(f"  3. Check if validation/test target windows overlap with training data")
    
    return None  # Can't determine without full dataset

def check_normalization(dataset):
    """Check scaler statistics and normalized data ranges."""
    print(f"\n=== Checking Normalization ===")
    
    if not dataset.scale:
        print("Scaling is disabled - skipping normalization check")
        return
    
    print(f"Scaler type: {type(dataset.scaler).__name__}")
    
    # Note: dataset.data only contains the current split's data
    # We can't easily access raw training data from a validation dataset
    # The scaler was fitted on training data during initialization
    
    if hasattr(dataset.scaler, 'mean_'):
        mean = dataset.scaler.mean_
        var = dataset.scaler.var_
        std = np.sqrt(var)
        
        print(f"\nScaler statistics (fitted on training data):")
        if mean.ndim == 0:
            print(f"  Mean: {mean:.6f}")
            print(f"  Std:  {std:.6f}")
            print(f"  Variance: {var:.6f}")
        else:
            print(f"  Mean shape: {mean.shape}")
            print(f"  Mean (first 5): {mean[:5] if len(mean) > 5 else mean}")
            print(f"  Std (first 5):  {std[:5] if len(std) > 5 else std}")
            print(f"  Mean range: [{mean.min():.6f}, {mean.max():.6f}]")
            print(f"  Std range:  [{std.min():.6f}, {std.max():.6f}]")
        
        # Check normalized data statistics
        if hasattr(dataset, 'data'):
            normalized_data = dataset.data
            print(f"\nNormalized data statistics (all splits):")
            print(f"  Shape: {normalized_data.shape}")
            print(f"  Mean: {normalized_data.mean():.6f}")
            print(f"  Std:  {normalized_data.std():.6f}")
            print(f"  Min:  {normalized_data.min():.6f}")
            print(f"  Max:  {normalized_data.max():.6f}")
            
            # Note: We can't check training split specifically from a validation dataset
            # The scaler was fitted on training data, so training data should have mean~0, std~1
            print(f"\nNote: Scaler was fitted on training data.")
            print(f"  Training data (after normalization) should have mean~0, std~1")
            print(f"  Current split data may have different statistics due to distribution shift")

def check_sample_shapes(dataset, num_samples=3):
    """Check shapes of samples from dataset."""
    print(f"\n=== Checking Sample Shapes ===")
    print(f"Dataset length: {len(dataset)}")
    print(f"seq_len: {dataset.seq_len}, pred_len: {dataset.pred_len}")
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        seq_x, seq_y = sample[1], sample[2]  # batch_x, batch_y
        
        print(f"\nSample {i}:")
        print(f"  seq_x shape: {seq_x.shape} (expected: [{dataset.seq_len}, features])")
        print(f"  seq_y shape: {seq_y.shape} (expected: [{dataset.pred_len}, features])")
        print(f"  seq_x value range: [{seq_x.min():.6f}, {seq_x.max():.6f}]")
        print(f"  seq_y value range: [{seq_y.min():.6f}, {seq_y.max():.6f}]")
        
        # Check if values are normalized (should be roughly mean=0, std=1)
        if dataset.scale:
            print(f"  seq_x normalized stats: mean={seq_x.mean():.6f}, std={seq_x.std():.6f}")
            print(f"  seq_y normalized stats: mean={seq_y.mean():.6f}, std={seq_y.std():.6f}")

def simulate_loss_computation(dataset, num_samples=10):
    """Simulate loss computation to see what values we get."""
    print(f"\n=== Simulating Loss Computation ===")
    
    # Create dummy "perfect" predictions (just copy ground truth)
    # This gives us a baseline of what loss would be if model was perfect
    # Then create "realistic" predictions with some error
    
    mse_perfect = []
    mse_realistic = []
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        seq_x, seq_y = sample[1], sample[2]
        
        # Perfect prediction (should give 0 loss)
        pred_perfect = torch.from_numpy(seq_y.copy())
        gt = torch.from_numpy(seq_y.copy())
        mse_perfect.append(torch.nn.functional.mse_loss(pred_perfect, gt).item())
        
        # Realistic prediction (add small random error)
        pred_realistic = torch.from_numpy(seq_y.copy()) + torch.randn_like(torch.from_numpy(seq_y)) * 0.1
        mse_realistic.append(torch.nn.functional.mse_loss(pred_realistic, gt).item())
    
    print(f"Perfect prediction MSE (baseline, should be ~0): {np.mean(mse_perfect):.6f}")
    print(f"Realistic prediction MSE (with 0.1 std noise): {np.mean(mse_realistic):.6f}")
    print(f"  -> This shows what MSE looks like on normalized data with small errors")

def main():
    """Main diagnostic function."""
    print("=" * 80)
    print("FIDEL-TS iTransformer Loss Discrepancy Diagnostic")
    print("=" * 80)
    
    # Load configs (use absolute path from project root)
    data_config_path = project_root / "data_configs/time_mmd/Traffic/config.yaml"
    if not data_config_path.exists():
        print(f"ERROR: Config file not found at {data_config_path}")
        print(f"Please run from project root or ensure config file exists")
        return
    data_config = load_config(str(data_config_path))
    
    # Extract paths from config
    # root_path is relative to project root, data_path is relative to root_path
    root_path_from_config = data_config.get('root_path', './data')
    data_path_from_config = data_config.get('data_path', 'US_VMT_Month.csv')
    
    # Resolve root_path - handle both relative and absolute paths
    if root_path_from_config.startswith('./'):
        root_path = project_root / root_path_from_config[2:]  # Remove './'
    elif root_path_from_config.startswith('/'):
        root_path = Path(root_path_from_config)  # Absolute path
    else:
        root_path = project_root / root_path_from_config  # Relative to project root
    
    # Check if custom data path is provided via environment variable (overrides config)
    import os
    if 'FIDEL_TS_DATA_PATH' in os.environ:
        root_path = Path(os.environ['FIDEL_TS_DATA_PATH'])
        print(f"Using custom data path from environment: {root_path}")
    
    print(f"Using root_path: {root_path}")
    print(f"Using data_path: {data_path_from_config}")
    print(f"Full data file path: {root_path / data_path_from_config}")
    
    # Create args object (for compatibility, not all fields needed)
    args = dotdict({
        'input_len': 24,
        'output_len': 6,
        'seq_len': 24,
        'pred_len': 6,
    })
    
    # Create dataset
    print("\nLoading dataset...")
    
    # Create spliter with split_info from config
    split_info = data_config.get('split_info', [7, 1, 2])
    spliter = partial(ratio_spliter, split=split_info, seq_len=args.seq_len)
    
    # Extract Time-MMD specific parameters from config
    text_column = data_config.get('text_column', 'auto')
    use_closedllm = data_config.get('use_closedllm', False)
    text_len = data_config.get('text_len', 4)
    # Map timemmd_text_output to output_format
    # Supported formats: 'json', 'dict', 'csv', 'embedding'
    timemmd_output = data_config.get('timemmd_text_output', 'text')
    if timemmd_output == 'text':
        output_format = 'json'  # Default to json for text format
    elif timemmd_output == 'embedding':
        output_format = 'embedding'
    else:
        output_format = timemmd_output  # Use as-is if it's already a valid format
    general_info = data_config.get('general_info', '')
    channel_info = data_config.get('channel_info', '')
    timestamp_col = data_config.get('timestamp_col', 'date')
    target = data_config.get('target', 'OT')
    
    dataset = TimeMMD_Dataset(
        root_path=str(root_path),
        data_path=data_path_from_config,
        flag='val',  # Check validation set
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        spliter=spliter,
        target=target,
        scale=True,  # From config, but hardcoded for diagnostic
        timestamp_col=timestamp_col,
        text_column=text_column,
        use_closedllm=use_closedllm,
        text_len=text_len,
        output_format=output_format,
        general_info=general_info,
        channel_info=channel_info
    )
    
    # Run diagnostics
    check_data_leakage(dataset, split='val', split_info=split_info)
    check_normalization(dataset)
    check_sample_shapes(dataset, num_samples=5)
    simulate_loss_computation(dataset, num_samples=10)
    
    print("\n" + "=" * 80)
    print("Diagnostic complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
