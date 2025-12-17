"""
Time-MMD Dataset Adapter for fidel-ts

This module provides integration with MM-TSFlib (Time-MMD) datasets, adapting
their CSV format (with embedded text columns) to fidel-ts's expected data format.
"""

import os
import numpy as np
import pandas as pd
from functools import partial
from sklearn.preprocessing import StandardScaler
from .data_loader import Universal_Dataset
from .data_helper import ratio_spliter, data_buffer


class TimeMMD_HeteroGetter:
    """
    Heterogeneous data getter for Time-MMD datasets.
    
    Adapts MM-TSFlib text format (single text at sequence end point) to
    fidel-ts format (text aligned to timestamps via hetero_data_getter interface).
    
    Args:
        text_data (pd.Series): Text data indexed by timestamp (as int64 YYYYMMDDHHMMSS)
        timestamps (np.ndarray): All timestamps in the dataset (int64 format)
        general_info (str): General dataset information
        channel_info (str): Channel-specific information
        output_format (str): Output format for text ('json', 'dict', 'csv', 'embedding')
    """
    
    def __init__(self, text_data, timestamps, general_info='', channel_info='', output_format='json'):
        """
        Initialize TimeMMD_HeteroGetter.
        
        Args:
            text_data: Series with text indexed by timestamp (int64 YYYYMMDDHHMMSS)
            timestamps: All timestamps in dataset (int64 array)
            general_info: General dataset description (empty string if not provided)
            channel_info: Channel-specific description (empty string if not provided)
            output_format: Format for output_dynamic ('json', 'dict', 'csv', 'embedding')
        """
        self.text_data = text_data
        self.timestamps = timestamps
        self.general_info = general_info if general_info else ''
        self.channel_info = channel_info if channel_info else ''
        self.output_format = output_format
        
        # Create a mapping for fast lookup
        self.text_dict = text_data.to_dict()
    
    def __call__(self, timestamps):
        """
        Get heterogeneous data for given timestamps.
        
        Implements the hetero_data_getter interface expected by Universal_Dataset.
        Uses backward matching (text at or before timestamp) since MM-TSFlib
        stores text at sequence end point.
        
        Args:
            timestamps: Array of timestamps (int64 YYYYMMDDHHMMSS format)
            
        Returns:
            tuple: (matched_times, general_info, channel_info, output_dynamic)
                - matched_times: List of matched timestamps as strings
                - general_info: General dataset info (string)
                - channel_info: Channel-specific info (string)
                - output_dynamic: Text data in requested format
        """
        matched_times = []
        matched_texts = []
        
        # Convert timestamps to list if numpy array
        if isinstance(timestamps, np.ndarray):
            timestamps = timestamps.tolist()
        
        # Match each timestamp using backward matching (text at or before timestamp)
        for ts in timestamps:
            # Find text at or before this timestamp
            matched_ts = None
            matched_text = ''
            
            # Try exact match first
            if ts in self.text_dict:
                matched_ts = ts
                matched_text = self.text_dict[ts]
            else:
                # Backward matching: find closest timestamp <= ts
                valid_timestamps = [t for t in self.text_dict.keys() if t <= ts]
                if valid_timestamps:
                    matched_ts = max(valid_timestamps)
                    matched_text = self.text_dict[matched_ts]
            
            if matched_ts is not None:
                matched_times.append(str(matched_ts))
                matched_texts.append(matched_text if matched_text else '')
            else:
                # No match found, use empty string
                matched_times.append(str(ts))
                matched_texts.append('')
        
        # Format output according to output_format
        if self.output_format == 'json':
            import json
            output_dynamic = [json.dumps({'text': text}) for text in matched_texts]
        elif self.output_format == 'dict':
            output_dynamic = [{'text': text} for text in matched_texts]
        elif self.output_format == 'csv':
            # Convert to CSV string format
            df = pd.DataFrame({'text': matched_texts})
            output_dynamic = df.to_csv(index=False)
        elif self.output_format == 'embedding':
            # For embedding format, return zero array (embeddings should be pre-computed)
            # Shape: (num_timesteps, embedding_dim)
            # Note: This requires embedding_dim to be specified, defaulting to 768
            embedding_dim = 768  # Default BERT dimension, should be configurable
            output_dynamic = np.zeros((len(matched_texts), embedding_dim), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported output_format: {self.output_format}")
        
        return matched_times, self.general_info, self.channel_info, output_dynamic


class TimeMMD_Dataset(Universal_Dataset):
    """
    Dataset adapter for MM-TSFlib (Time-MMD) format.
    
    Loads CSV files containing time series data with embedded text columns
    and adapts them to fidel-ts's Universal_Dataset interface.
    
    Key differences from standard Universal_Dataset:
    - Text data is stored in CSV columns (Final_Search_* or Final_Output)
    - Text is aligned to sequence end point (s_end) rather than individual timestamps
    - Provides text via hetero_data_getter interface compatible with fidel-ts
    
    Args:
        text_column (str): Text column name, or 'auto' to auto-detect
        use_closedllm (bool): Whether to use Final_Output column (closed-source LLM)
        text_len (int): Text length for Final_Search_{text_len} detection
        output_format (str): Format for text output ('json', 'dict', 'csv', 'embedding')
        general_info (str): General dataset description
        channel_info (str): Channel-specific description
        All other args: Same as Universal_Dataset
    """
    
    def __init__(self, root_path, flag='train', data_path='ETTh1.csv',
                 seq_len=24, pred_len=24, spliter=ratio_spliter, timestamp_col='date',
                 target='OT', scale=True, data_buffer=None, hetero_data_getter=None,
                 preload_hetero=False, hetero_stride=1, task=None, custom_input=None,
                 timezone=None, downsample=None, entity_id=None,
                 text_column='auto', use_closedllm=False, text_len=4,
                 output_format='json', general_info='', channel_info=''):
        """
        Initialize TimeMMD_Dataset.
        
        Args:
            text_column: Text column name ('auto' to auto-detect)
            use_closedllm: Use Final_Output column if True, else Final_Search_{text_len}
            text_len: Text length for Final_Search_{text_len} pattern
            output_format: Format for text output
            general_info: General dataset description
            channel_info: Channel-specific description
            All other args: Same as Universal_Dataset
        """
        # Store Time-MMD specific parameters
        self.text_column = text_column
        self.use_closedllm = use_closedllm
        self.text_len = text_len
        self.output_format = output_format
        self.general_info = general_info
        self.channel_info = channel_info
        
        # Initialize parent class with hetero_data_getter=None initially
        # We'll set it up after reading data
        super().__init__(
            root_path=root_path,
            flag=flag,
            data_path=data_path,
            seq_len=seq_len,
            pred_len=pred_len,
            spliter=spliter,
            timestamp_col=timestamp_col,
            target=target,
            scale=scale,
            data_buffer=data_buffer,
            hetero_data_getter=None,  # Will be set after data loading
            preload_hetero=preload_hetero,
            hetero_stride=hetero_stride,
            task=task,
            custom_input=custom_input,
            timezone=timezone,
            downsample=downsample,
            entity_id=entity_id
        )
        
        # Setup text getter after data is loaded
        self._setup_text_getter()
    
    def _detect_text_column(self, df_raw):
        """
        Detect text column name from CSV.
        
        Args:
            df_raw: Raw DataFrame loaded from CSV
            
        Returns:
            str or None: Column name if found, None otherwise
        """
        if self.text_column != 'auto':
            if self.text_column in df_raw.columns:
                return self.text_column
            else:
                print(f'[ warning ] Specified text column "{self.text_column}" not found, trying auto-detect')
        
        # Auto-detect logic
        if self.use_closedllm:
            if 'Final_Output' in df_raw.columns:
                return 'Final_Output'
        else:
            # Look for Final_Search_{text_len} pattern
            pattern = f'Final_Search_{self.text_len}'
            if pattern in df_raw.columns:
                return pattern
            # Try common text_len values
            for text_len in [2, 4, 6]:
                pattern = f'Final_Search_{text_len}'
                if pattern in df_raw.columns:
                    print(f'[ info ] Found text column: {pattern} (requested: Final_Search_{self.text_len})')
                    return pattern
        
        return None
    
    def _setup_text_getter(self):
        """
        Setup text getter after data is loaded.
        
        Creates TimeMMD_HeteroGetter and sets it as hetero_data_getter.
        """
        # Check if text column was found and text data exists
        if not hasattr(self, '_text_column_name') or self._text_column_name is None:
            print('[ info ] No text column found, dataset will work as time-series-only')
            return
        
        if not hasattr(self, '_text_data') or self._text_data is None:
            print('[ info ] No text data available, dataset will work as time-series-only')
            return
        
        # Create text getter
        text_getter = TimeMMD_HeteroGetter(
            text_data=self._text_data,
            timestamps=self.timestamp,
            general_info=self.general_info,
            channel_info=self.channel_info,
            output_format=self.output_format
        )
        
        # Set as hetero_data_getter
        self.hetero_data_getter = text_getter
    
    def __read_data__(self):
        """
        Override __read_data__ to handle MM-TSFlib CSV format.
        
        Loads CSV with embedded text columns and extracts both time series
        and text data, then calls parent's data processing logic.
        """
        self.scaler = StandardScaler()
        
        # Load CSV file
        if self.data_buffer is None:
            if self.data_path.endswith('.csv'):
                df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
            elif self.data_path.endswith('.parquet'):
                df_raw = pd.read_parquet(os.path.join(self.root_path, self.data_path))
            else:
                raise NotImplementedError('Only .csv and .parquet data are supported')
        elif isinstance(self.data_buffer, data_buffer):
            df_raw = self.data_buffer(os.path.join(self.root_path, self.data_path))
        
        # Convert timestamp to datetime
        df_raw[self.timestamp_col] = pd.to_datetime(df_raw[self.timestamp_col])
        
        # Handle timezone if present
        if df_raw[self.timestamp_col][0].tz is not None:
            if self.timezone is not None:
                print(f'[ info ] The timestamp column has timezone, converting to {self.timezone}')
                df_raw[self.timestamp_col] = pd.to_datetime(df_raw[self.timestamp_col], utc=True).dt.tz_convert(self.timezone).dt.tz_localize(None)
            else:
                print('[ info ] The timestamp column has timezone, forcing UTC')
                df_raw[self.timestamp_col] = pd.to_datetime(df_raw[self.timestamp_col], utc=True).dt.tz_convert('UTC').dt.tz_localize(None)
        
        # Detect and extract text column
        self._text_column_name = self._detect_text_column(df_raw)
        
        if self._text_column_name is not None:
            # Extract text data before splitting
            # Store text indexed by timestamp (will be converted to int64 format)
            text_series = df_raw[[self.timestamp_col, self._text_column_name]].copy()
            text_series[self.timestamp_col] = text_series[self.timestamp_col].dt.strftime('%Y%m%d%H%M%S').astype(np.int64)
            text_series = text_series.set_index(self.timestamp_col)[self._text_column_name]
            # Handle NaN values
            text_series = text_series.fillna('')
            self._text_data = text_series
            print(f'[ info ] Found text column: {self._text_column_name} with {len(text_series)} entries')
        else:
            self._text_data = None
            print('[ info ] No text column found in CSV')
        
        # Apply data splitting
        train_data, val_data, test_data = self.spliter(df=df_raw)
        
        if self.set_type == 'train':
            self.data = train_data
        elif self.set_type == 'val':
            self.data = val_data
        elif self.set_type == 'test':
            self.data = test_data
        
        # Convert timestamp to int64 format (YYYYMMDDHHMMSS)
        self.data[self.timestamp_col] = self.data[self.timestamp_col].dt.strftime('%Y%m%d%H%M%S')
        self.data[self.timestamp_col] = self.data[self.timestamp_col].astype(np.int64)
        
        self.timestamp = self.data[self.timestamp_col].values.copy()
        
        # Extract time series data
        if self.target == 'all':
            self.data = self.data.drop(columns=[self.timestamp_col])
            self.data = self.data.values.astype(np.float32).copy()
            train_data = train_data.drop(columns=[self.timestamp_col])
            train_data = train_data.values.astype(np.float32).copy()
        else:
            self.data = self.data[self.target].values.astype(np.float32).copy()
            train_data = train_data[self.target].values.astype(np.float32).copy()
        
        # Normalize if requested
        if self.scale:
            self.scaler.fit(train_data)
            self.data = self.scaler.transform(self.data).astype(np.float32).copy()
        
        # Apply downsampling if requested
        if self.downsample is not None:
            self.data = self.data[::self.downsample]
            self.timestamp = self.timestamp[::self.downsample]
            
            # Also downsample text data if it exists
            if hasattr(self, '_text_data') and self._text_data is not None:
                # Reindex text data to match downsampled timestamps
                self._text_data = self._text_data.reindex(self.timestamp, method='nearest')

