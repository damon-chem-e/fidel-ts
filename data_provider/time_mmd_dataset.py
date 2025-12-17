"""
Time-MMD Dataset Adapter for fidel-ts

This module provides integration with MM-TSFlib (Time-MMD) datasets, adapting
their CSV format (with embedded text columns) to fidel-ts's expected data format.
"""

import os
import re
import numpy as np
import pandas as pd
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
    
    def _match_timestamps(self, timestamps):
        """
        Match timestamps to text data using backward matching.
        
        Uses backward matching (text at or before timestamp) since MM-TSFlib
        stores text at sequence end point. This ensures text describes context
        up to that point without lookahead bias.
        
        Args:
            timestamps: Array or list of timestamps (int64 YYYYMMDDHHMMSS format)
            
        Returns:
            tuple: (matched_times, matched_texts)
                - matched_times: List of matched timestamps as strings
                - matched_texts: List of matched text strings
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
        
        return matched_times, matched_texts
    
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
        matched_times, matched_texts = self._match_timestamps(timestamps)
        
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
            # 
            # Note: 'embedding' format expects pre-computed embeddings from files (.pkl).
            # Time-MMD datasets have text in CSV columns, not pre-computed embeddings.
            # 
            # When embeddings are needed:
            # - Use output_format='json' or 'dict' and let models handle text
            # - Or use postemb (future enhancement) to create embeddings during initialization
            # - Or pre-compute embeddings separately and use Heterogeneous_Dataset
            #
            # Returning zeros here is correct for the 'embedding' format expectation,
            # but Time-MMD datasets should typically use 'json' or 'dict' format.
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
        
        Strict detection with no fallbacks:
        - If text_column != 'auto', returns that column or None (no fallback)
        - If text_column == 'auto', detects based on use_closedllm and text_len
        - No fallback to different text_len values
        
        Args:
            df_raw: Raw DataFrame loaded from CSV
            
        Returns:
            str or None: Column name if found, None otherwise
        """
        if self.text_column != 'auto':
            # Strict: return specified column or None
            if self.text_column in df_raw.columns:
                return self.text_column
            else:
                raise ValueError(f'Specified text column "{self.text_column}" not found in CSV columns: {list(df_raw.columns)}')
        
        # Auto-detect logic (no fallbacks)
        if self.use_closedllm:
            if 'Final_Output' in df_raw.columns:
                return 'Final_Output'
            else:
                return None
        elif self.text_column == 'auto':
            # Look for Final_Search_{text_len} pattern (strict, no fallback)
            pattern = f'Final_Search_{self.text_len}'
            if pattern in df_raw.columns:
                return pattern
            else:
                return None

        else:
            raise ValueError(f'Invalid text column detection logic: text_column={self.text_column}, use_closedllm={self.use_closedllm}, text_len={self.text_len}')
    
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
        
        # Columns to exclude from time series data
        exclude_cols = [self.timestamp_col]
        
        # Exclude ALL Final_Search_* and Final_Output columns (even if not the one we're using)
        for col in df_raw.columns:
            if re.match(r'Final_Search_\d+', col) or col == 'Final_Output':
                if col not in exclude_cols:
                    exclude_cols.append(col)
        
        # Also exclude common metadata columns that shouldn't be in time series
        metadata_cols = ['start_date', 'end_date']
        for col in metadata_cols:
            if col in df_raw.columns and col not in exclude_cols:
                exclude_cols.append(col)
        
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
        
        # Extract time series data (excluding text and metadata columns)
        if self.target == 'all':
            # Drop timestamp, text, and metadata columns
            self.data = self.data.drop(columns=exclude_cols)
            self.data = self.data.values.astype(np.float32).copy()
            train_data = train_data.drop(columns=exclude_cols)
            train_data = train_data.values.astype(np.float32).copy()
        else:
            self.data = self.data[self.target].values.astype(np.float32).copy()
            train_data = train_data[self.target].values.astype(np.float32).copy()
        
        # Normalize if requested
        if self.scale:
            self.scaler.fit(train_data)
            self.data = self.scaler.transform(self.data).astype(np.float32).copy()
        
        # Apply downsampling if requested
        # Use simple indexing (no lookahead bias) - same as Universal_Dataset
        if self.downsample is not None:
            self.data = self.data[::self.downsample]
            self.timestamp = self.timestamp[::self.downsample]
            
            # Also downsample text data if it exists
            # Use reindex with forward fill (ffill) to match downsampled timestamps
            # ffill uses the last valid observation (backward in time, no lookahead bias)
            if hasattr(self, '_text_data') and self._text_data is not None:
                # Reindex to downsampled timestamps, using forward fill (previous value)
                # This ensures we use the most recent text value without lookahead bias
                self._text_data = self._text_data.reindex(self.timestamp, method='ffill')
                # Fill any remaining NaN values at the beginning with empty string
                self._text_data = self._text_data.fillna('')
