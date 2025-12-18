"""
Time-MMD Dataset Adapter for fidel-ts

This module provides integration with MM-TSFlib (Time-MMD) datasets, adapting
their CSV format (with embedded text columns) to fidel-ts's expected data format.
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
from .data_loader import Universal_Dataset
from .data_helper import ratio_spliter, data_buffer

from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn


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
    
    def __init__(self, text_data, timestamps, general_info='', channel_info='', 
                 output_format='json', embed_model_name='bert-base-uncased', 
                 embed_dim=768, force_reembed=False, hf_cache_dir='./HF_cache/',
                 root_path=None, data_path=None, device='cpu'):
        """
        Initialize TimeMMD_HeteroGetter.
        
        Args:
            text_data: Series with text indexed by timestamp (int64 YYYYMMDDHHMMSS)
            timestamps: All timestamps in dataset (int64 array)
            general_info: General dataset description (empty string if not provided)
            channel_info: Channel-specific description (empty string if not provided)
            output_format: Format for output_dynamic ('json', 'dict', 'csv', 'embedding')
            embed_model_name: HuggingFace model name for text embedding (default: bert-base-uncased)
            embed_dim: Embedding dimension (default: 768 for BERT)
            force_reembed: If True, recompute embeddings even if .pkl exists
            hf_cache_dir: Local directory for caching HF models (default: ./HF_cache/)
            root_path: Dataset root path (for embedding file location)
            data_path: Data file path (for embedding file location)
            device: Device for embedding model (default: 'cpu')
        """
        self.text_data = text_data
        self.timestamps = timestamps
        self.general_info = general_info if general_info else ''
        self.channel_info = channel_info if channel_info else ''
        self.output_format = output_format
        self.embed_model_name = embed_model_name
        self.embed_dim = embed_dim
        self.force_reembed = force_reembed
        self.hf_cache_dir = hf_cache_dir
        self.root_path = root_path
        self.data_path = data_path
        self.device = device
        
        # Create a mapping for fast lookup
        self.text_dict = text_data.to_dict()
        
        # Initialize embedding-related attributes (lazy loading)
        self.embeddings = None
        self.tokenizer = None
        self.model = None
    
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
    
    def _get_embedding_path(self):
        """
        Get the path to the embedding .pkl file.
        
        Returns absolute path to ensure proper file access regardless of working directory.
        
        Returns:
            str: Absolute path to embedding file
        """
        if self.root_path is None or self.data_path is None:
            raise ValueError("root_path and data_path must be provided for embedding mode")
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(self.data_path))[0]
        pkl_path = os.path.join(self.root_path, f"{base_name}.pkl")
        # Convert to absolute path to avoid permission issues with relative paths
        return os.path.abspath(pkl_path)
    
    def _load_embedding_model(self):
        """
        Load tokenizer and model, using local cache if available.
        
        Only loads once; subsequent calls reuse cached model.
        Logs whether model is loaded from cache or downloaded from cloud.
        """
        if self.tokenizer is not None and self.model is not None:
            return  # Already loaded
        
        # Ensure cache directory exists
        os.makedirs(self.hf_cache_dir, exist_ok=True)
        
        # Check if model exists in cache before loading
        # HuggingFace stores models in: {cache_dir}/models--{model_name_sanitized}/
        model_name_sanitized = self.embed_model_name.replace('/', '--')
        cache_model_path = Path(self.hf_cache_dir) / f"models--{model_name_sanitized}"
        model_in_cache = cache_model_path.exists() and any(cache_model_path.iterdir())
        
        if model_in_cache:
            print(f'[ info ] Loading {self.embed_model_name} from local cache: {cache_model_path}')
        else:
            print(f'[ info ] Downloading {self.embed_model_name} from HuggingFace (will cache to: {cache_model_path})')
        
        # Use cache_dir parameter to store models locally
        # First call downloads and caches; subsequent calls use cache
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.embed_model_name,
            cache_dir=self.hf_cache_dir
        )
        
        # Check again after tokenizer load to see if it was actually cached
        if not model_in_cache and cache_model_path.exists() and any(cache_model_path.iterdir()):
            print('[ info ] Tokenizer cached successfully')
        
        self.model = AutoModel.from_pretrained(
            self.embed_model_name,
            cache_dir=self.hf_cache_dir
        ).to(self.device)
        
        # Check again after model load
        if not model_in_cache and cache_model_path.exists() and any(cache_model_path.iterdir()):
            print(f'[ info ] Model cached successfully to: {cache_model_path}')
        
        self.model.eval()
        print(f'[ info ] {self.embed_model_name} loaded and ready for embedding')
    
    def _embed_text_corpus(self):
        """
        Embed all text in text_data using the embedding model.
        
        Returns:
            dict: Dictionary mapping timestamp strings to embedding arrays
                Format: {"YYYYMMDDHHMMSS": np.ndarray(shape=(1, embed_dim), dtype=np.float32)}
        """
        # Load model if not already loaded
        self._load_embedding_model()
        
        embeddings_dict = {}
        batch_size = 32  # Process in batches for efficiency
        
        # Get all timestamps and texts
        all_timestamps = list(self.text_data.index)
        all_texts = list(self.text_data.values)
        total_batches = (len(all_texts) + batch_size - 1) // batch_size
        
        # Use Rich progress bar for embedding computation
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[progress.completed]{task.completed}/{task.total} batches"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                f"Computing embeddings for {len(all_texts)} texts",
                total=total_batches
            )
            
            # Process in batches
            for i in range(0, len(all_texts), batch_size):
                batch_texts = all_texts[i:i+batch_size]
                batch_timestamps = all_timestamps[i:i+batch_size]
                
                # Tokenize batch
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    # Use [CLS] token embedding (first token)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Store embeddings with timestamp keys
                for ts, emb in zip(batch_timestamps, batch_embeddings):
                    # Reshape to (1, embed_dim) for consistency with expected format
                    embeddings_dict[str(ts)] = emb.reshape(1, -1).astype(np.float32)
                
                # Update progress
                progress.update(task, advance=1)
        
        return embeddings_dict
    
    def _load_or_create_embeddings(self):
        """
        Load embeddings from .pkl file or create them on-the-fly.
        
        If .pkl exists and force_reembed=False, loads from file.
        Otherwise, computes embeddings and saves to .pkl.
        """
        pkl_path = self._get_embedding_path()
        
        if os.path.exists(pkl_path) and not self.force_reembed:
            # Load precomputed embeddings
            print(f'[ info ] Loading embeddings from {pkl_path}')
            self.embeddings = joblib.load(pkl_path)
        else:
            # Compute embeddings on-the-fly
            print('[ info ] Computing embeddings on-the-fly (this may take a while)...')
            self.embeddings = self._embed_text_corpus()
            
            # Save to .pkl
            try:
                os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
                joblib.dump(self.embeddings, pkl_path)
                print(f'[ info ] Saved embeddings to {pkl_path}')
            except Exception as e:
                print(f'[ warning ] Could not save embeddings to {pkl_path}: {e}')
                print('[ info ] Embeddings will be recomputed on next run')
    
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
            # Ensure embeddings are loaded/created
            if self.embeddings is None:
                self._load_or_create_embeddings()
            
            # Fetch embeddings for matched timestamps
            embedding_list = []
            for ts in matched_times:
                # ts is string 'YYYYMMDDHHMMSS'
                if ts in self.embeddings:
                    emb = self.embeddings[ts]  # shape: (1, embed_dim)
                else:
                    # No embedding found, use zero vector
                    emb = np.zeros((1, self.embed_dim), dtype=np.float32)
                embedding_list.append(emb)
            
            # Stack to shape: (num_timesteps, 1, embed_dim)
            # This matches expected format: (seq_len, news_num, embed_dim)
            # where news_num=1 for Time-MMD (single text per timestamp)
            output_dynamic = np.stack(embedding_list, axis=0)  # (num_timesteps, 1, embed_dim)
            # Squeeze middle dimension to match expected shape: (num_timesteps, embed_dim)
            # But we need to keep it as (num_timesteps, 1, embed_dim) for compatibility
            # Actually, let's keep it as (num_timesteps, 1, embed_dim) to match TGTSF expectation
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
                 output_format='json', general_info='', channel_info='',
                 embed_model_name='bert-base-uncased', embed_dim=768,
                 force_reembed=False, hf_cache_dir='./HF_cache/', device='cpu'):
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
        self.embed_model_name = embed_model_name
        self.embed_dim = embed_dim
        self.force_reembed = force_reembed
        self.hf_cache_dir = hf_cache_dir
        self.device = device
        
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
            output_format=self.output_format,
            embed_model_name=self.embed_model_name,
            embed_dim=self.embed_dim,
            force_reembed=self.force_reembed,
            hf_cache_dir=self.hf_cache_dir,
            root_path=self.root_path,
            data_path=self.data_path,
            device=self.device
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
            # Single column target - extract as 1D array, then reshape to 2D for scaler
            self.data = self.data[self.target].values.astype(np.float32).copy()
            train_data = train_data[self.target].values.astype(np.float32).copy()
            # Reshape to 2D (samples, features) for StandardScaler
            if train_data.ndim == 1:
                train_data = train_data.reshape(-1, 1)
            if self.data.ndim == 1:
                self.data = self.data.reshape(-1, 1)
        
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
