"""
Utility functions for handling missing values in time series data.

Provides strategies for dealing with NaN values, including forward fill
with indicator variables to help models learn patterns around missing data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'none',
    exclude_cols: List[str] = None
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
    
    Returns:
        Tuple of (processed_df, indicator_columns):
        - processed_df: DataFrame with missing values handled
        - indicator_columns: List of indicator column names created (empty if strategy='none')
    
    Example:
        ```python
        df_processed, indicators = handle_missing_values(
            df, 
            strategy='forward_fill_indicators',
            exclude_cols=['date', 'text']
        )
        # df_processed now has forward-filled values and new columns like 'missing_Heart_Rate'
        # indicators = ['missing_Heart_Rate', 'missing_Respiratory_Rate', ...]
        ```
    """
    if exclude_cols is None:
        exclude_cols = []
    
    if strategy == 'none':
        # No processing, return as-is
        return df.copy(), []
    
    if strategy == 'forward_fill_indicators':
        df_processed = df.copy()
        indicator_columns = []
        
        # Process each column (excluding specified columns and non-numeric columns)
        for col in df.columns:
            if col in exclude_cols:
                continue  # Skip explicitly excluded columns (timestamp, text, metadata)
            
            # Only process numeric columns - skip text/object columns
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue  # Skip non-numeric columns (text, object, etc.)
            
            # Check if column has any missing values
            if df[col].isna().any():
                # Create indicator column: 1 where original was NaN, 0 otherwise
                indicator_name = f'missing_{col}'
                
                # Ensure indicator name doesn't conflict with existing columns
                if indicator_name in df_processed.columns:
                    # If conflict, append number
                    counter = 1
                    while f'{indicator_name}_{counter}' in df_processed.columns:
                        counter += 1
                    indicator_name = f'{indicator_name}_{counter}'
                
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
        
        return df_processed, indicator_columns
    
    else:
        raise ValueError(f"Unknown missing value strategy: {strategy}. "
                        f"Supported strategies: 'none', 'forward_fill_indicators'")
