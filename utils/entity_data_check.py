"""
Utility functions for checking if entities have sufficient data for train/val/test splits.

This module provides functions to:
- Calculate split sizes for time series data
- Check if an entity has enough data points for required splits
- Get entity data file sizes efficiently
"""

import os
import pandas as pd
from typing import Tuple, Dict, Optional, List, Union


def calculate_split_sizes(
    total_rows: int,
    seq_len: int,
    pred_len: int,
    split_ratios: Union[Tuple[int, int, int], List[int]] = (7, 1, 2),
    split_type: str = 'ratio'
) -> Dict[str, int]:
    """
    Calculate actual split sizes accounting for sequence overlap.
    
    Uses the same logic as ratio_spliter to determine how many rows
    each split (train/val/test) will actually contain.
    
    Args:
        total_rows: Total number of rows in the entity's data
        seq_len: Input sequence length (used for overlap between splits)
        pred_len: Prediction length (minimum needed per split)
        split_ratios: Tuple/list of (train, val, test) ratios, e.g., (7, 1, 2)
        split_type: Type of split ('ratio' or 'timestamp')
    
    Returns:
        Dictionary with keys: 'train_len', 'val_len', 'test_len'
        Each value is the actual number of rows in that split
    """
    if split_type != 'ratio':
        # For timestamp splits, we can't calculate without actual timestamps
        # Return None values to indicate this needs actual data
        return {
            'train_len': None,
            'val_len': None,
            'test_len': None
        }
    
    # Calculate split boundaries (same logic as ratio_spliter)
    train_ratio = split_ratios[0] / sum(split_ratios)  # e.g., 0.7 for (7,1,2)
    val_ratio = split_ratios[1] / sum(split_ratios) + train_ratio  # e.g., 0.8
    
    train_split_idx = int(total_rows * train_ratio)
    val_split_idx = int(total_rows * val_ratio)
    
    # Calculate actual split lengths (accounting for seq_len overlap)
    # This matches the logic in ratio_spliter and analyze_medical_files.py:
    # train_data = df[0:train_split] → length = train_split_idx
    # val_data = df[train_split-seq_len:val_split] → length = val_split_idx - (train_split_idx - seq_len)
    # test_data = df[val_split-seq_len:] → length = total_rows - (val_split_idx - seq_len)
    train_len = train_split_idx
    val_len = val_split_idx - (train_split_idx - seq_len)  # val_split_idx - train_start + seq_len
    test_len = total_rows - (val_split_idx - seq_len)  # total - val_start + seq_len
    
    return {
        'train_len': train_len,
        'val_len': val_len,
        'test_len': test_len
    }


def check_entity_sufficient(
    total_rows: int,
    seq_len: int,
    pred_len: int,
    split_ratios: Union[Tuple[int, int, int], List[int]] = (7, 1, 2),
    split_type: str = 'ratio',
    require_all_splits: bool = True
) -> Tuple[bool, Dict[str, Union[bool, int]]]:
    """
    Check if an entity has sufficient data for required splits.
    
    Args:
        total_rows: Total number of rows in the entity's data
        seq_len: Input sequence length
        pred_len: Prediction length (minimum needed per split)
        split_ratios: Tuple/list of (train, val, test) ratios
        split_type: Type of split ('ratio' or 'timestamp')
        require_all_splits: If True, all splits must be sufficient.
                          If False, at least one split must be sufficient.
    
    Returns:
        Tuple of (is_sufficient, details_dict)
        details_dict contains:
        - 'train_ok': bool - whether train split has enough data
        - 'val_ok': bool - whether val split has enough data
        - 'test_ok': bool - whether test split has enough data
        - 'train_len': int - actual train split length
        - 'val_len': int - actual val split length
        - 'test_len': int - actual test split length
        - 'min_needed': int - minimum rows needed per split (seq_len + pred_len)
    """
    min_needed = seq_len + pred_len
    
    # For timestamp splits, we can't check without actual data
    # Return False to be safe
    if split_type != 'ratio':
        return False, {
            'train_ok': False,
            'val_ok': False,
            'test_ok': False,
            'train_len': None,
            'val_len': None,
            'test_len': None,
            'min_needed': min_needed
        }
    
    # Calculate split sizes
    split_sizes = calculate_split_sizes(total_rows, seq_len, pred_len, split_ratios, split_type)
    
    train_len = split_sizes['train_len']
    val_len = split_sizes['val_len']
    test_len = split_sizes['test_len']
    
    # Check each split
    train_ok = train_len >= min_needed if train_len is not None else False
    val_ok = val_len >= min_needed if val_len is not None else False
    test_ok = test_len >= min_needed if test_len is not None else False
    
    # Determine overall sufficiency
    if require_all_splits:
        is_sufficient = train_ok and val_ok and test_ok
    else:
        is_sufficient = train_ok or val_ok or test_ok
    
    return is_sufficient, {
        'train_ok': train_ok,
        'val_ok': val_ok,
        'test_ok': test_ok,
        'train_len': train_len,
        'val_len': val_len,
        'test_len': test_len,
        'min_needed': min_needed
    }


def get_entity_data_size(
    root_path: str,
    data_path: Optional[str],
    formatter: str,
    entity_id: str,
    data_buffer: Optional[object] = None
) -> Tuple[Optional[int], Optional[str]]:
    """
    Get the number of rows in an entity's data file without fully loading it.
    
    Args:
        root_path: Root directory path for data files
        data_path: Direct data path (if single-file dataset), or None
        formatter: Formatter string with {i} placeholder (e.g., 'patient_{i}.csv')
        entity_id: Entity ID to substitute into formatter
        data_buffer: Optional data buffer for caching (data_helper.data_buffer instance)
    
    Returns:
        Tuple of (num_rows, file_path) or (None, None) if file doesn't exist or error
    """
    # Determine the actual file path
    if data_path is not None and data_path != 'null':
        # Single-file dataset: use data_path directly (same file for all entities)
        file_path = os.path.join(root_path, data_path)
    else:
        # Multi-file dataset: use formatter
        if '{i}' in formatter:
            formatted_path = formatter.format(i=entity_id)
        else:
            formatted_path = formatter
        file_path = os.path.join(root_path, formatted_path)
    
    # Check if file exists
    if not os.path.exists(file_path):
        return None, file_path
    
    # Try to use data_buffer if available (may have cached row count)
    if data_buffer is not None and hasattr(data_buffer, 'buffer'):
        if file_path in data_buffer.buffer:
            df = data_buffer.buffer[file_path]
            return len(df), file_path
    
    # Read file to count rows
    # Use pandas to match how the actual dataset loader counts rows
    try:
        # For CSV files, use pandas to get accurate row count (matches dataset loading)
        if file_path.endswith('.csv'):
            # Use pandas to count rows (matches how datasets are loaded)
            df = pd.read_csv(file_path)
            num_rows = len(df)
        elif file_path.endswith('.parquet'):
            # For parquet, we need to read metadata or use pandas
            # Use pandas for reliability
            df = pd.read_parquet(file_path, engine='pyarrow')
            num_rows = len(df)
        else:
            # Unknown format, try pandas
            df = pd.read_csv(file_path)
            num_rows = len(df)
        
        return num_rows, file_path
    except Exception as e:
        # Return None on any error
        return None, file_path
