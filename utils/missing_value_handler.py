"""
Utility functions for handling missing values in time series data.

Provides strategies for dealing with NaN values, including forward fill
with indicator variables to help models learn patterns around missing data.
"""

import pandas as pd
import numpy as np
import os
import re
from typing import Tuple, List, Optional, Any


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'none',
    exclude_cols: List[str] = None,
    required_indicators: List[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Handle missing values in DataFrame according to specified strategy.
    
    Args:
        df: Input DataFrame with potential missing values
        strategy: Strategy for handling missing values:
            - 'none': No handling (return DataFrame as-is)
            - 'forward_fill_indicators': Forward fill missing values and create
              indicator variables (1 where original was NaN, 0 otherwise)
        exclude_cols: List of column names to exclude from missing value processing
            (e.g., timestamp, text columns, metadata). These columns are passed through unchanged.
        required_indicators: List of column names that MUST have indicator columns created,
            even if this specific DataFrame doesn't have missing values in those columns.
            This ensures consistent feature dimensions across multiple DataFrames/entities.
            Indicator names will be 'missing_{col}' for each column in this list.
    
    Returns:
        Tuple of (processed_df, indicator_columns):
        - processed_df: DataFrame with missing values handled
        - indicator_columns: List of indicator column names created (empty if strategy='none')
    
    Example:
        ```python
        df_processed, indicators = handle_missing_values(
            df, 
            strategy='forward_fill_indicators',
            exclude_cols=['date', 'text'],
            required_indicators=['Heart_Rate', 'Respiratory_Rate']  # Always create these indicators
        )
        # df_processed now has forward-filled values and new columns like 'missing_Heart_Rate'
        # indicators = ['missing_Heart_Rate', 'missing_Respiratory_Rate'] (always created)
        ```
    """
    if exclude_cols is None:
        exclude_cols = []
    
    if required_indicators is None:
        required_indicators = []
    
    if strategy == 'none':
        # No processing, return as-is
        return df.copy(), []
    
    if strategy == 'forward_fill_indicators':
        df_processed = df.copy()
        indicator_columns = []
        
        # Process only the required indicator columns (already identified via scan)
        # No need to search for columns with missing values - we already know which ones need indicators
        for col in required_indicators:
            if col in exclude_cols:
                continue  # Skip explicitly excluded columns (timestamp, text, metadata)
            
            # Create indicator column name
            indicator_name = f'missing_{col}'
            
            # Ensure indicator name doesn't conflict with existing columns
            if indicator_name in df_processed.columns:
                # If conflict, append number
                counter = 1
                while f'{indicator_name}_{counter}' in df_processed.columns:
                    counter += 1
                indicator_name = f'{indicator_name}_{counter}'
            
            # Check if the column exists in this DataFrame
            if col in df.columns:
                # Column exists: forward fill missing values and create indicator
                # Verify it's numeric before processing
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Create indicator: 1 where original was NaN, 0 otherwise
                    indicator = df[col].isna().astype(np.float32)
                    df_processed[indicator_name] = indicator
                    indicator_columns.append(indicator_name)
                    
                    # Forward fill missing values in original column
                    # Use ffill() which propagates last valid observation forward
                    df_processed[col] = df_processed[col].ffill()
                    
                    # Handle case where first value(s) are NaN (forward fill can't fill these)
                    # Fill remaining leading NaNs with 0 (or could use backward fill, but 0 is safer)
                    if df_processed[col].isna().any():
                        df_processed[col] = df_processed[col].fillna(0)
            else:
                # Column doesn't exist in this DataFrame: create indicator with all zeros
                # This ensures consistent feature dimensions across all entities
                indicator = np.zeros(len(df), dtype=np.float32)
                df_processed[indicator_name] = indicator
                indicator_columns.append(indicator_name)
        
        return df_processed, indicator_columns
    
    else:
        raise ValueError(f"Unknown missing value strategy: {strategy}. "
                        f"Supported strategies: 'none', 'forward_fill_indicators'")


def scan_missing_value_columns(
    id_list: List[str],
    root_path: str,
    timestamp_col: str,
    data_path: Optional[str] = None,
    formatter: str = 'id_{i}.parquet',
    data_buffer: Optional[Any] = None
) -> List[str]:
    """
    Scan all entities to determine which columns have missing values across ANY entity.
    
    This ensures consistent feature dimensions across all entities by creating
    indicator columns for all columns that have missing values in any entity,
    even if a specific entity doesn't have missing values in that column.
    
    Args:
        id_list: List of entity IDs to scan
        root_path: Root path where data files are located
        timestamp_col: Name of the timestamp column (to exclude from scanning)
        data_path: Optional data path for single-file datasets. If None, uses formatter for multi-file datasets
        formatter: String formatter for file naming (e.g., 'patient_{i}.csv')
        data_buffer: Optional data buffer for caching loaded DataFrames
    
    Returns:
        List[str]: Sorted list of column names that need indicator columns created
    
    Example:
        ```python
        missing_cols = scan_missing_value_columns(
            id_list=['10576', '10910', '10977'],
            root_path='./data/ttc/medical',
            timestamp_col='date',
            formatter='patient_{i}.csv'
        )
        # Returns: ['FiO2', 'Respiratory_Rate'] if these columns have missing values in any entity
        ```
    """
    missing_columns = set()
    all_numeric_columns = set()  # Track all numeric columns across all entities
    
    # Handle case where YAML has 'null' as string
    if data_path == 'null':
        data_path = None
    
    # Debug: Print scan start
    print(f'[ DEBUG ] Scanning {len(id_list)} entities for missing value columns...')
    # END DEBUG
    
    # Scan all entities to find columns with missing values
    for entity_id in id_list:
        # Determine file path for this entity
        if data_path is not None:
            # Single-file dataset: use data_path directly
            file_path = os.path.join(root_path, data_path)
        else:
            # Multi-file dataset: use formatter
            file_path = os.path.join(root_path, formatter.format(i=entity_id))
        
        # Skip if file doesn't exist
        if not os.path.exists(file_path):
            continue
        
        try:
            # Load DataFrame
            if data_buffer is not None:
                # Use buffer if available (data_buffer is callable, takes file_path)
                df = data_buffer(file_path)
            else:
                df = pd.read_csv(file_path)
            
            # Determine columns to exclude (timestamp, text, metadata)
            exclude_cols = [timestamp_col]
            for col in df.columns:
                if (re.match(r'Final_Search_\d+', col) or 
                    col == 'Final_Output' or 
                    col == 'text'):
                    if col not in exclude_cols:
                        exclude_cols.append(col)
            
            # Check each numeric column for missing values
            for col in df.columns:
                if col in exclude_cols:
                    continue
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                
                # Track all numeric columns (for debugging)
                all_numeric_columns.add(col)
                
                # Check if column has missing values
                if df[col].isna().any():
                    missing_columns.add(col)
                    print(f'[ DEBUG ] Found missing values in column "{col}" for entity {entity_id}')
        except Exception as e:
            # Skip entities that can't be loaded
            print(f'[ DEBUG ] Error loading entity {entity_id}: {e}')
            continue
    
    # Debug: Print scan results
    print(f'[ DEBUG ] Scan complete: Found {len(missing_columns)} columns with missing values: {sorted(list(missing_columns))}')
    print(f'[ DEBUG ] All numeric columns found: {sorted(list(all_numeric_columns))}')
    
    return sorted(list(missing_columns))
