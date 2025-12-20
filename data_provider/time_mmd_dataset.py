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
from utils.embedding_model_registry import EmbeddingModelRegistry
from utils.missing_value_handler import handle_missing_values

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
                 root_path=None, data_path=None, device='cpu', num_channels=None, channel_names=None):
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
            num_channels: Number of channels/variables in the dataset (for expanding channel_info embedding)
            channel_names: List of actual channel/column names from CSV (for per-channel embeddings).
                If provided and multiple channels exist, each channel gets a unique embedding combining
                channel_info with its name (e.g., "Weather variables: temperature" for channel "temperature")
        """
        self.text_data = text_data
        self.timestamps = timestamps
        self.general_info = general_info if general_info else ''
        self.channel_info = channel_info if channel_info else ''
        self.output_format = output_format
        self.embed_model_name = embed_model_name
        self.embed_dim = embed_dim  # Kept for backward compatibility, but actual dimension comes from model
        self.force_reembed = force_reembed
        self.hf_cache_dir = hf_cache_dir
        self.root_path = root_path
        self.data_path = data_path
        self.device = device
        self.bert_dim = None  # Will be set when model is loaded
        self.num_channels = num_channels  # Number of channels for expanding channel_info embedding
        self.channel_names = channel_names  # List of channel/column names for per-channel embeddings
        
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
        Load tokenizer and model using shared registry.
        
        Uses EmbeddingModelRegistry to ensure only one model instance exists
        per (model_name, device, hf_cache_dir) combination, preventing
        multiple copies from being loaded into GPU memory.
        
        Thread-safe: Safe to call from multiple DataLoader workers.
        """
        if self.tokenizer is not None and self.model is not None:
            return  # Already loaded
        
        # Get shared tokenizer and model from registry
        # Registry handles caching, device placement, and thread safety
        self.tokenizer = EmbeddingModelRegistry.get_tokenizer(
            self.embed_model_name,
            self.hf_cache_dir
        )
        
        self.model = EmbeddingModelRegistry.get_model(
            self.embed_model_name,
            self.device,
            self.hf_cache_dir
        )
        
        # Get BERT output dimension (typically 768 for bert-base-uncased)
        # Check the model's config to get the hidden size
        self.bert_dim = self.model.config.hidden_size
    
    def _embed_text_corpus(self):
        """
        Embed all text in text_data using the embedding model.
        
        Returns:
            dict: Dictionary mapping timestamp strings to embedding arrays
                Format: {"YYYYMMDDHHMMSS": np.ndarray(shape=(1, bert_dim), dtype=np.float32)}
                where bert_dim is BERT's output dimension (typically 768)
        """
        # Load model only when we need to compute embeddings
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
                    # Output in BERT's native dimension (768) - models will project if needed
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Store embeddings with timestamp keys
                for ts, emb in zip(batch_timestamps, batch_embeddings):
                    # Reshape to (1, bert_dim) for consistency with expected format
                    embeddings_dict[str(ts)] = emb.reshape(1, -1).astype(np.float32)
                
                # Update progress
                progress.update(task, advance=1)
        
        return embeddings_dict
    
    def _embed_single_text(self, text):
        """
        Embed a single text string using the embedding model.
        
        Args:
            text: Text string to embed
            
        Returns:
            np.ndarray: Embedding vector of shape (1, bert_dim) where bert_dim is BERT's output dimension (typically 768)
        """
        # Only load model if we need to compute embeddings (not if embeddings already exist)
        if self.tokenizer is None or self.model is None:
            self._load_embedding_model()
        
        # Tokenize
        encoded = self.tokenizer(
            text if text else '',  # Handle empty strings
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Get embedding
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            # Use [CLS] token embedding (first token)
            # Output in BERT's native dimension (768) - models will project if needed
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # Reshape to (1, bert_dim) for consistency
        return embedding.reshape(1, -1).astype(np.float32)
    
    def _load_or_create_embeddings(self):
        """
        Load embeddings from .pkl file or create them on-the-fly.
        
        If .pkl exists and force_reembed=False, loads from file (no need to load BERT model).
        Otherwise, computes embeddings and saves to .pkl (loads BERT model only when needed).
        """
        pkl_path = self._get_embedding_path()

        if os.path.exists(pkl_path) and not self.force_reembed:
            # Load precomputed embeddings (no need to load BERT model)
            self.embeddings = joblib.load(pkl_path)
            # Set bert_dim from a sample embedding if not already set
            if self.bert_dim is None and self.embeddings:
                sample_key = next(iter(self.embeddings.keys()))
                sample_emb = self.embeddings[sample_key]
                if hasattr(sample_emb, 'shape'):
                    self.bert_dim = sample_emb.shape[-1]
        else:
            # Compute embeddings on-the-fly (load BERT model only when needed)
            print('[ info ] Computing embeddings on-the-fly (this may take a while)...')
            self.embeddings = self._embed_text_corpus()
            
            # Save to .pkl
            pkl_dir = os.path.dirname(pkl_path)
            try:
                # Ensure directory exists
                os.makedirs(pkl_dir, exist_ok=True)
                    
                joblib.dump(self.embeddings, pkl_path)
                print(f'[ info ] Saved embeddings to {pkl_path}')
            except PermissionError as e:
                print(f'[ error ] Permission denied saving embeddings to {pkl_path}')
                print(f'[ error ] Error details: {e}')
                print(f'[ error ] Current working directory: {os.getcwd()}')
                print(f'[ error ] Directory permissions: {oct(os.stat(pkl_dir).st_mode) if os.path.exists(pkl_dir) else "N/A"}')
                print('[ info ] Embeddings will be recomputed on next run')
            except Exception as e:
                print(f'[ error ] Could not save embeddings to {pkl_path}: {type(e).__name__}: {e}')
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
            # Only load/create if not already loaded (avoids loading BERT if embeddings exist)
            if self.embeddings is None:
                self._load_or_create_embeddings()
            
            # Fetch embeddings for matched timestamps
            embedding_list = []
            for ts in matched_times:
                # ts is string 'YYYYMMDDHHMMSS'
                if ts in self.embeddings:
                    emb = self.embeddings[ts]  # shape: (1, bert_dim) where bert_dim is BERT's output (typically 768)
                else:
                    # No embedding found, use zero vector
                    # Use BERT's native dimension (768) - models will project if needed
                    emb = np.zeros((1, self.bert_dim), dtype=np.float32)
                embedding_list.append(emb)
            
            # Stack to shape: (num_timesteps, 1, bert_dim)
            # This matches expected format: (seq_len, news_num, bert_dim)
            # where news_num=1 for Time-MMD (single text per timestamp)
            # Models will project bert_dim -> text_dim if needed
            output_dynamic = np.stack(embedding_list, axis=0)  # (num_timesteps, 1, bert_dim)
            # Squeeze middle dimension to match expected shape: (num_timesteps, embed_dim)
            # But we need to keep it as (num_timesteps, 1, embed_dim) for compatibility
            # Actually, let's keep it as (num_timesteps, 1, embed_dim) to match TGTSF expectation
        else:
            raise ValueError(f"Unsupported output_format: {self.output_format}")
        
        # Handle general_info and channel_info based on output_format
        # When output_format='embedding', these should also be embeddings for models like TGTSF
        if self.output_format == 'embedding':
            # Embed general_info and channel_info if they're strings
            # TGTSF expects channel_description: [bs, nvars, d_model] (before unsqueeze in forward)
            # So per sample: [nvars, d_model], which collates to [batch_size, nvars, d_model]
            # For Time-MMD with single channel: [1, embed_dim] per sample
            if isinstance(self.general_info, str):
                general_info_emb = self._embed_single_text(self.general_info)  # (1, embed_dim)
            else:
                general_info_emb = self.general_info
            
            if isinstance(self.channel_info, str):
                # Create per-channel embeddings based on actual channel names if available
                # This allows each channel (e.g., 'temperature', 'humidity', 'pressure') to have
                # its own unique embedding derived from its name, rather than repeating the same
                # generic channel_info string for all channels
                
                if self.channel_names is not None and len(self.channel_names) > 1:
                    # Create unique embedding for each channel using its name
                    # Format: combine channel_info (dataset-level description) with channel name
                    channel_info_emb_list = []
                    for channel_name in self.channel_names:
                        # Combine dataset-level channel_info with per-channel name
                        # e.g., "Weather variables: temperature" if channel_info="Weather variables:" and channel_name="temperature"
                        combined_text = f"{self.channel_info} {channel_name}" if self.channel_info else channel_name
                        channel_emb = self._embed_single_text(combined_text)  # (1, embed_dim)
                        channel_info_emb_list.append(channel_emb)
                    # Stack to create (num_channels, embed_dim) array
                    channel_info_emb = np.vstack(channel_info_emb_list)  # (num_channels, embed_dim)
                elif self.num_channels is not None and self.num_channels > 1:
                    # Fallback: if we have num_channels but not channel names, repeat the embedding
                    # This happens when channel names aren't available (shouldn't happen for TTC)
                    channel_info_emb = self._embed_single_text(self.channel_info)  # (1, embed_dim)
                    channel_info_emb = np.repeat(channel_info_emb, self.num_channels, axis=0)  # (num_channels, embed_dim)
                else:
                    # Single channel case: (1, embed_dim) is correct as-is
                    channel_info_emb = self._embed_single_text(self.channel_info)  # (1, embed_dim)
                
                # DataLoader will collate to [batch_size, nvars, embed_dim]
                # TGTSF forward expects [bs, nvars, d_model] before unsqueeze
            else:
                # channel_info is already an array/list - use as-is (should already have correct shape)
                channel_info_emb = self.channel_info
            
            return matched_times, general_info_emb, channel_info_emb, output_dynamic
        else:
            # For text formats, return strings as-is
            return matched_times, self.general_info, self.channel_info, output_dynamic


class TimeMMD_Dataset(Universal_Dataset):
    """
    Dataset adapter for MM-TSFlib (Time-MMD) and TTC (Time Text Corpus) formats.
    
    Loads CSV files containing time series data with embedded text columns
    and adapts them to fidel-ts's Universal_Dataset interface.
    
    Supports two dataset formats:
    1. **Time-MMD format** (MM-TSFlib): Uses Final_Search_{text_len} or Final_Output columns
    2. **TTC format**: Uses a simpler 'text' column name for embedded text descriptions
    
    Key differences from standard Universal_Dataset:
    - Text data is stored in CSV columns (Final_Search_*, Final_Output, or 'text')
    - Text is aligned to sequence end point (s_end) rather than individual timestamps
    - Provides text via hetero_data_getter interface compatible with fidel-ts
    
    TTC Support:
    This class was extended to support TTC datasets (climate and medical data) which use
    a simple 'text' column instead of the more complex Final_Search_* naming convention.
    The modifications include:
    - Text column detection: When text_column='auto', falls back to checking for 'text'
      column after trying Time-MMD patterns (Final_Search_*, Final_Output)
    - Column exclusion: The 'text' column is automatically excluded from time series data,
      similar to how Final_Search_* and Final_Output columns are handled
    - Backward compatibility: All existing Time-MMD datasets continue to work without changes
    
    Args:
        text_column (str): Text column name, or 'auto' to auto-detect
            - For Time-MMD: 'auto' detects Final_Search_{text_len} or Final_Output
            - For TTC: Specify 'text' explicitly or use 'auto' (which will detect 'text' as fallback)
        use_closedllm (bool): Whether to use Final_Output column (closed-source LLM, Time-MMD only)
        text_len (int): Text length for Final_Search_{text_len} detection (Time-MMD only)
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
                 force_reembed=False, hf_cache_dir='./HF_cache/', device='cpu',
                 missing_value_strategy='none'):
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
        self.missing_value_strategy = missing_value_strategy
        
        # Initialize missing value indicator tracking (will be populated in __read_data__)
        self.missing_indicators = []
        self.target_columns = None
        
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
            entity_id=entity_id,
            missing_value_strategy=missing_value_strategy
        )
        
        # Setup text getter after data is loaded (after parent.__init__ which loads data)
        self._setup_text_getter()
    
    def _detect_text_column(self, df_raw):
        """
        Detect text column name from CSV.
        
        Supports multiple text column formats:
        - Explicit specification: If text_column != 'auto', returns that column (or raises error)
        - Time-MMD auto-detection: Detects Final_Search_{text_len} or Final_Output columns
        - TTC fallback: When text_column='auto' and no Time-MMD columns found, checks for 'text' column
        
        Detection priority (when text_column='auto'):
        1. Final_Output (if use_closedllm=True)
        2. Final_Search_{text_len} (Time-MMD format)
        3. 'text' column (TTC format fallback)
        
        This priority order maintains backward compatibility with Time-MMD datasets while
        adding support for TTC datasets that use the simpler 'text' column name.
        
        Args:
            df_raw: Raw DataFrame loaded from CSV
            
        Returns:
            str or None: Column name if found, None otherwise
        """
        # If explicitly specified, use that column (or raise error if not found)
        if self.text_column != 'auto':
            if self.text_column in df_raw.columns:
                return self.text_column
            else:
                raise ValueError(
                    f'Specified text column "{self.text_column}" not found in CSV columns: '
                    f'{list(df_raw.columns)}'
                )
        
        # Auto-detect logic: Try Time-MMD patterns first, then TTC 'text' column as fallback
        if self.use_closedllm:
            if 'Final_Output' in df_raw.columns:
                return 'Final_Output'
            else:
                # No Final_Output found, fall through to check for 'text' column (TTC format)
                pass
        
        # Look for Final_Search_{text_len} pattern (Time-MMD format)
        pattern = f'Final_Search_{self.text_len}'
        if pattern in df_raw.columns:
            return pattern
        
        # Fallback: Check for 'text' column (TTC format support)
        # This allows TTC datasets to work with text_column='auto' without explicit specification
        if 'text' in df_raw.columns:
            return 'text'
        
        # No text column found
        return None
    
    def _setup_text_getter(self):
        """
        Setup text getter after data is loaded.
        
        Creates TimeMMD_HeteroGetter and sets it as hetero_data_getter.
        """
        # Check if text column was found and text data exists
        if not hasattr(self, '_text_column_name') or self._text_column_name is None:
            return
        
        if not hasattr(self, '_text_data') or self._text_data is None:
            print('[ info ] No text data available, dataset will work as time-series-only')
            return
        
        # Get channel information for per-channel embeddings
        # Note: self.data is already filtered at this point (text/metadata columns excluded in __read_data__)
        # The filtering happens in __read_data__() at lines 720-721 (for target='all') or 726-732 (for single target)
        # After filtering, self.data is a numpy array with shape (num_samples, num_channels)
        # where num_channels is the number of actual time series channels (excluding text/metadata)
        num_channels = self.data.shape[1] if hasattr(self, 'data') and self.data is not None else None
        channel_names = getattr(self, '_channel_names', None)  # Per-channel names from CSV columns
        
        # Create text getter with channel info for per-channel embedding creation
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
            device=self.device,
            num_channels=num_channels,
            channel_names=channel_names  # Pass actual channel names for per-channel embeddings
        )
        
        # Set as hetero_data_getter
        self.hetero_data_getter = text_getter
    
    def __read_data__(self):
        """
        Override __read_data__ to handle MM-TSFlib (Time-MMD) and TTC CSV formats.
        
        Loads CSV with embedded text columns and extracts both time series
        and text data, then calls parent's data processing logic.
        
        Supports:
        - Time-MMD format: Final_Search_{text_len} or Final_Output columns
        - TTC format: 'text' column (climate and medical datasets)
        
        Both formats are handled identically: text columns are excluded from
        time series data and extracted for use via the hetero_data_getter interface.
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
        
        # Exclude text columns from time series data
        # Supports both Time-MMD format (Final_Search_*, Final_Output) and TTC format ('text')
        for col in df_raw.columns:
            if (re.match(r'Final_Search_\d+', col) or 
                col == 'Final_Output' or 
                col == 'text'):  # TTC format support
                if col not in exclude_cols:
                    exclude_cols.append(col)
        
        # Also exclude common metadata columns that shouldn't be in time series
        metadata_cols = ['start_date', 'end_date']
        for col in metadata_cols:
            if col in df_raw.columns and col not in exclude_cols:
                exclude_cols.append(col)
        
        # Handle missing values before splitting (ensures consistent processing across train/val/test)
        # Exclude timestamp, text, and metadata columns from missing value processing
        df_raw, missing_indicators = handle_missing_values(
            df_raw,
            strategy=self.missing_value_strategy,
            exclude_cols=exclude_cols
        )
        
        # Store indicator column names (needed for target column tracking)
        self.missing_indicators = missing_indicators
        
        if missing_indicators:
            print(f'[ info ] Created {len(missing_indicators)} missing value indicator columns: {missing_indicators[:5]}{"..." if len(missing_indicators) > 5 else ""}')
            # Note: Indicators are scaled along with other data. Performance may improve if indicators
            # are left unscaled (they're binary 0/1 by design), but scaling is simpler for now.
        
        if self._text_column_name is not None:
            # Extract text data before splitting
            # Store text indexed by timestamp (will be converted to int64 format)
            text_series = df_raw[[self.timestamp_col, self._text_column_name]].copy()
            text_series[self.timestamp_col] = text_series[self.timestamp_col].dt.strftime('%Y%m%d%H%M%S').astype(np.int64)
            text_series = text_series.set_index(self.timestamp_col)[self._text_column_name]
            # Handle NaN values
            text_series = text_series.fillna('')
            self._text_data = text_series
        else:
            self._text_data = None
        
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
        # Store channel names BEFORE filtering (needed for per-channel embeddings)
        if self.target == 'all':
            # Get column names that will become channels (exclude text/metadata/timestamp columns)
            channel_columns = [col for col in self.data.columns if col not in exclude_cols]
            self._channel_names = channel_columns  # Store for per-channel embedding creation
            
            # Track target columns (exclude indicator columns - indicators are covariates, not targets)
            # Note: missing_indicators is set in parent class, but we need to filter them here
            all_data_columns = [col for col in self.data.columns if col not in exclude_cols]
            self.target_columns = [col for col in all_data_columns if col not in self.missing_indicators]
            
            # Drop timestamp, text, and metadata columns
            self.data = self.data.drop(columns=exclude_cols)
            self.data = self.data.values.astype(np.float32).copy()
            train_data = train_data.drop(columns=exclude_cols)
            train_data = train_data.values.astype(np.float32).copy()
        else:
            # Single column target - extract as 1D array, then reshape to 2D for scaler
            self._channel_names = [self.target]  # Store channel name for single-target case
            self.target_columns = [self.target] if isinstance(self.target, str) else self.target
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
