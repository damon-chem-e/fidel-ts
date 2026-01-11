"""
Diagnostic script to investigate why fidel-ts iTransformer shows 0.01 MSE 
vs 0.18 MSE in papers/MM-TSFlib.

This script checks for:
1. Data leakage (validation samples overlapping with training)
2. Normalization issues (scaler statistics)
3. Shape mismatches in loss computation
4. Actual loss values and scales
"""

import numpy as np
import torch
from data_provider.data_loader import Universal_Dataset
from data_provider.time_mmd_dataset import TimeMMD_Dataset
import yaml
from utils.tools import dotdict

def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def check_data_leakage(dataset, split='val'):
    """Check if validation/test samples overlap with training data."""
    print(f"\n=== Checking for Data Leakage ({split}) ===")
    
    # Get data splits
    train_data = dataset.data[0:dataset.train_split]
    if split == 'val':
        split_data = dataset.data[dataset.train_split-dataset.seq_len:dataset.val_split]
        split_name = "validation"
    else:
        split_data = dataset.data[dataset.val_split-dataset.seq_len:]
        split_name = "test"
    
    print(f"Training data range: [0, {len(train_data)})")
    if split == 'val':
        print(f"{split_name.capitalize()} data range: [{dataset.train_split-dataset.seq_len}, {dataset.val_split})")
    else:
        print(f"{split_name.capitalize()} data range: [{dataset.val_split-dataset.seq_len}, {len(dataset.data)})")
    
    # Check for overlap in sample windows
    train_end = len(train_data)
    if split == 'val':
        split_start = dataset.train_split - dataset.seq_len
        split_end = dataset.val_split
    else:
        split_start = dataset.val_split - dataset.seq_len
        split_end = len(dataset.data)
    
    # Check if validation/test windows can access training data
    overlap = train_end > split_start
    print(f"Overlap check: Training ends at {train_end}, {split_name} starts at {split_start}")
    print(f"  -> Overlap exists: {overlap} (this is EXPECTED for seq_len overlap)")
    
    # Check actual sample indices
    print(f"\nSample index ranges:")
    print(f"  Training samples: [0, {len(train_data) - dataset.seq_len - dataset.pred_len + 1})")
    if split == 'val':
        val_start_idx = len(train_data) - dataset.seq_len
        val_end_idx = dataset.val_split - dataset.seq_len - dataset.pred_len + 1
        print(f"  {split_name.capitalize()} samples: [{val_start_idx}, {val_end_idx})")
        
        # Check if any validation sample's input window overlaps with training
        first_val_sample_input_start = val_start_idx
        first_val_sample_input_end = val_start_idx + dataset.seq_len
        print(f"\n  First {split_name} sample input window: [{first_val_sample_input_start}, {first_val_sample_input_end})")
        print(f"  Training data ends at: {train_end}")
        print(f"  -> Input window overlaps with training: {first_val_sample_input_end > train_end}")
        
        # Check if any validation sample's target overlaps with training
        first_val_sample_target_start = first_val_sample_input_end
        first_val_sample_target_end = first_val_sample_target_start + dataset.pred_len
        print(f"  First {split_name} sample target window: [{first_val_sample_target_start}, {first_val_sample_target_end})")
        print(f"  -> Target window overlaps with training: {first_val_sample_target_end > train_end}")
    else:
        test_start_idx = dataset.val_split - dataset.seq_len
        test_end_idx = len(dataset.data) - dataset.seq_len - dataset.pred_len + 1
        print(f"  {split_name.capitalize()} samples: [{test_start_idx}, {test_end_idx})")
    
    return overlap

def check_normalization(dataset):
    """Check scaler statistics and normalized data ranges."""
    print(f"\n=== Checking Normalization ===")
    
    if not dataset.scale:
        print("Scaling is disabled - skipping normalization check")
        return
    
    print(f"Scaler type: {type(dataset.scaler).__name__}")
    
    # Get training data (before normalization)
    train_data_raw = dataset.data[0:dataset.train_split] if hasattr(dataset, 'train_split') else None
    
    if train_data_raw is not None and hasattr(dataset.scaler, 'mean_'):
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
            
            # Check training split specifically
            if hasattr(dataset, 'train_split'):
                train_normalized = normalized_data[0:dataset.train_split]
                print(f"\nNormalized training data statistics:")
                print(f"  Mean: {train_normalized.mean():.6f} (should be ~0)")
                print(f"  Std:  {train_normalized.std():.6f} (should be ~1)")

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
    
    # Load configs
    data_config_path = "data_configs/time_mmd/Traffic/config.yaml"
    data_config = load_config(data_config_path)
    
    # Create args object
    args = dotdict({
        'data_path': './data',
        'data': 'time_mmd_traffic',
        'features': 'S',  # Single target
        'target': 'OT',
        'scale': True,
        'inverse': False,
        'freq': 'h',
        'embed': 'timeF',
        'input_len': 24,
        'output_len': 6,
        'seq_len': 24,
        'pred_len': 6,
        'data_config': dotdict(data_config)
    })
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = TimeMMD_Dataset(
        root_path=args.data_path,
        data_path=args.data_config.data_path,
        flag='val',  # Check validation set
        size=[args.input_len, 0, args.output_len],
        features=args.features,
        target=args.target,
        scale=args.scale,
        timeenc=0,
        freq=args.freq,
        data_config=args.data_config
    )
    
    # Run diagnostics
    check_data_leakage(dataset, split='val')
    check_normalization(dataset)
    check_sample_shapes(dataset, num_samples=5)
    simulate_loss_computation(dataset, num_samples=10)
    
    print("\n" + "=" * 80)
    print("Diagnostic complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
