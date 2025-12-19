#!/usr/bin/env python3
"""
Quick utility to analyze medical patient CSV files (TTC Medical) and report size distribution.

Reports the number of timestamps (rows) in each patient file to help determine
if entity-based splitting is needed or if time-based splitting is sufficient.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_medical_files(data_dir='./data/ttc/medical'):
    """
    Analyze all CSV files in the medical data directory.
    
    Args:
        data_dir: Path to directory containing patient CSV files
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return
    
    # Find all CSV files matching patient pattern
    csv_files = sorted(data_path.glob('patient_*.csv'))
    
    if not csv_files:
        print(f"No patient_*.csv files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} patient CSV files\n")
    print("=" * 60)
    
    # Load and count rows for each file
    file_sizes = []
    file_details = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            num_rows = len(df)
            file_sizes.append(num_rows)
            file_details.append({
                'file': csv_file.name,
                'rows': num_rows
            })
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")
            continue
    
    if not file_sizes:
        print("No files could be read")
        return
    
    # Calculate statistics
    sizes = np.array(file_sizes)
    
    print("\nSize Distribution Summary:")
    print(f"  Total files: {len(file_sizes)}")
    print(f"  Min rows: {sizes.min()}")
    print(f"  Max rows: {sizes.max()}")
    print(f"  Mean rows: {sizes.mean():.1f}")
    print(f"  Median rows: {np.median(sizes):.1f}")
    print(f"  Std dev: {sizes.std():.1f}")
    
    # Show files that might be too small for seq_len + pred_len
    print(f"\n{'=' * 60}")
    print("Files by size (smallest first):")
    print(f"{'File':<30} {'Rows':<10}")
    print("-" * 60)
    
    sorted_details = sorted(file_details, key=lambda x: x['rows'])
    for detail in sorted_details:
        print(f"{detail['file']:<30} {detail['rows']:<10}")
    
    # Check against common sequence requirements with 70/10/20 time-based split
    print(f"\n{'=' * 60}")
    print("70/10/20 Time-Based Split Analysis:")
    print("(Checking if each split has enough data for seq_len + pred_len)")
    
    def check_split_sizes(total_rows, seq_len, pred_len, split_ratios=(7, 1, 2)):
        """Calculate split sizes and check if each has enough data."""
        # Calculate split boundaries (same logic as ratio_spliter)
        train_ratio = split_ratios[0] / sum(split_ratios)  # 0.7
        val_ratio = split_ratios[1] / sum(split_ratios) + train_ratio  # 0.8
        
        train_split_idx = int(total_rows * train_ratio)
        val_split_idx = int(total_rows * val_ratio)
        
        # Calculate actual split lengths (accounting for seq_len overlap)
        train_len = train_split_idx
        val_len = val_split_idx - (train_split_idx - seq_len)  # val_split_idx - train_start + seq_len
        test_len = total_rows - (val_split_idx - seq_len)  # total - val_start + seq_len
        
        # Check if each split has enough for sequences
        min_needed = seq_len + pred_len
        train_ok = train_len >= min_needed
        val_ok = val_len >= min_needed
        test_ok = test_len >= min_needed
        
        return {
            'train_len': train_len,
            'val_len': val_len,
            'test_len': test_len,
            'train_ok': train_ok,
            'val_ok': val_ok,
            'test_ok': test_ok,
            'all_ok': train_ok and val_ok and test_ok
        }
    
    for seq_len, pred_len in [(7, 7), (14, 7), (24, 24)]:
        print(f"\n  For seq_len={seq_len}, pred_len={pred_len}:")
        
        all_ok_count = 0
        train_ok_count = 0
        val_ok_count = 0
        test_ok_count = 0
        problem_files = []
        
        for detail in file_details:
            result = check_split_sizes(detail['rows'], seq_len, pred_len)
            
            if result['all_ok']:
                all_ok_count += 1
            if result['train_ok']:
                train_ok_count += 1
            if result['val_ok']:
                val_ok_count += 1
            if result['test_ok']:
                test_ok_count += 1
            
            if not result['all_ok']:
                issues = []
                if not result['train_ok']:
                    issues.append(f"train({result['train_len']})")
                if not result['val_ok']:
                    issues.append(f"val({result['val_len']})")
                if not result['test_ok']:
                    issues.append(f"test({result['test_len']})")
                problem_files.append({
                    'file': detail['file'],
                    'rows': detail['rows'],
                    'issues': issues,
                    'train_len': result['train_len'],
                    'val_len': result['val_len'],
                    'test_len': result['test_len']
                })
        
        print(f"    Files with sufficient data for ALL splits: {all_ok_count}/{len(file_details)}")
        print(f"    Files with sufficient data for train: {train_ok_count}/{len(file_details)}")
        print(f"    Files with sufficient data for val: {val_ok_count}/{len(file_details)}")
        print(f"    Files with sufficient data for test: {test_ok_count}/{len(file_details)}")
        
        if problem_files:
            print(f"\n    Files with insufficient data in some splits ({len(problem_files)}):")
            for pf in problem_files[:10]:  # Show first 10
                print(f"      {pf['file']}: {pf['rows']} rows total, missing: {', '.join(pf['issues'])}")
            if len(problem_files) > 10:
                print(f"      ... and {len(problem_files) - 10} more")

if __name__ == '__main__':
    import sys
    
    # Allow data directory to be specified as command line argument
    data_dir = sys.argv[1] if len(sys.argv) > 1 else './data/ttc/medical'
    
    analyze_medical_files(data_dir)
