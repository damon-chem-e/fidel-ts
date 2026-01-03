import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import warnings
from .data_helper import ratio_spliter, data_buffer
from time import time
import json
from functools import partial
import logging
from utils.missing_value_handler import handle_missing_values
from embedder import FidelTSEmbeddingLoader, FidelTSPathResolver
from typing import Optional, Dict, Any
from utils.timefeatures import time_features

warnings.filterwarnings('ignore')

# Set up logger for dataset operations
logger = logging.getLogger(__name__)

class Universal_Dataset(Dataset):
    """
    PyTorch Dataset for universal time series forecasting with cross-modal data support.
    
    This dataset class handles both traditional time series data and heterogeneous cross-modal
    information (text, events, etc.) for enhanced forecasting. It supports multiple task types
    including standard time series forecasting (TSF), text-guided forecasting (TGTSF), and 
    multi-modal forecasting (MTSF).
    
    Args:
        root_path (str): Root directory path containing data files
        flag (str): Dataset split identifier ('train', 'val', 'test')
        data_path (str): Relative path to the specific data file (e.g., 'ETTh1.csv')
        seq_len (int): Length of input sequences for modeling
        pred_len (int): Length of prediction/forecast sequences  
        spliter (callable): Function to split data into train/val/test sets
        timestamp_col (str): Name of the timestamp column in the data
        target (str): Target column name(s) for prediction. Use 'all' for all columns
        scale (bool): Whether to apply z-score normalization to the data
        data_buffer (data_buffer, optional): Buffer object for caching loaded data
        hetero_data_getter (callable, optional): Function to retrieve heterogeneous data
        preload_hetero (bool): Whether to preload all heterogeneous data into memory
        hetero_stride (int): Stride for heterogeneous data alignment
        task (str, optional): Task type ('TSF', 'TGTSF', 'MTSF', 'Reasoning')
        custom_input (str, optional): Custom input specification overriding task defaults
        timezone (str, optional): Timezone for timestamp conversion
        downsample (int, optional): Downsampling factor for data reduction
    
    Attributes:
        data: Processed time series data as numpy array
        timestamp: Processed timestamps as numpy array
        scaler: StandardScaler for data normalization
        custom_input: List of input components to include in batches
    
    Example:
        ```python
        dataset = Universal_Dataset(
            root_path='./data',
            data_path='stock_data.csv',
            flag='train',
            seq_len=96,
            pred_len=24,
            target='close_price',
            scale=True,
            task='TSF'
        )
        ```
    """
    def __init__(self, root_path, flag='train', data_path='ETTh1.csv',
                 seq_len=24, pred_len=24, spliter=ratio_spliter, timestamp_col='date',
                 target='OT', scale=True, data_buffer=None, hetero_data_getter=None, 
                 preload_hetero=False, hetero_stride=1, task=None, custom_input=None, 
                 timezone=None, downsample=None, entity_id=None, 
                 missing_value_strategy='none', required_indicators=None,
                 generate_time_features=False, time_feature_freq='h',
                 llm_embedding_provider=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = seq_len
        self.pred_len = pred_len
        # init
        self.spliter = spliter
        self.set_type = flag

        self.target = target
        self.scale = scale
        self.data_buffer = data_buffer
        self.missing_value_strategy = missing_value_strategy
        self.required_indicators = required_indicators if required_indicators is not None else []

        self.timestamp_col = timestamp_col

        self.root_path = root_path
        self.data_path = data_path
        
        # Store entity_id for sample_id generation
        self.entity_id = str(entity_id) if entity_id is not None else None
        
        # Initialize missing value indicator tracking
        self.missing_indicators = []  # Will be populated in __read_data__ if indicators are created
        self.target_columns = None  # Will be set to track which columns are targets (excludes indicators)

        self.hetero_data_getter = (lambda x: x) if hetero_data_getter is None else hetero_data_getter # return the timestamp
        self.timezone = timezone
        self.downsample = downsample
        
        # Time feature generation (for FEDformer and similar models)
        self.generate_time_features = generate_time_features
        self.time_feature_freq = time_feature_freq
        
        # LLM Embedding Provider (for TimeCMA-style models)
        # When provided, LLM embeddings are used as hetero_channel
        self.llm_embedding_provider = llm_embedding_provider

        self.__read_data__()
        self.preload_hetero = preload_hetero
        self.hetero_stride = hetero_stride

        if self.preload_hetero:
            self.__preload_hetero__()

        self.task = task
        self.custom_input = custom_input


        self.__input_format_parser__()

    def __input_format_parser__(self):
        """
        Parses and configures the input format based on task type or custom specification.
        
        Different tasks require different input components:
        - TSF: Basic time series forecasting (seq_x, seq_y, x_time, y_time)
        - TGTSF: Text-guided forecasting (adds future heterogeneous data)
        - MTSF: Multi-modal forecasting (adds historical heterogeneous data)
        - Reasoning/all: Full multi-modal input with all components
        
        Sets self.custom_input to list of required input component names.
        """
        if self.custom_input is not None:
            print('[ warning ] Custom input set, overriding task defined input as: {}'.format(self.custom_input))
            self.custom_input = self.custom_input.strip().split(',')
            # assert all the input is in the list of ['seq_x', 'seq_y', 'x_time', 'y_time', 'hetero_x_time', 'x_hetero', 'hetero_y_time', 'y_hetero', 'hetero_general', 'hetero_channel']
            assert all([i in ['seq_x', 'seq_y', 'x_time', 'y_time', 'hetero_x_time', 'x_hetero', 'hetero_y_time', 'y_hetero', 'hetero_general', 'hetero_channel'] for i in self.custom_input]), "Custom input should be a comma split string of ['seq_x', 'seq_y', 'x_time', 'y_time', 'hetero_x_time', 'x_hetero', 'hetero_y_time', 'y_hetero', 'hetero_general', 'hetero_channel']"
        else:
            if self.task is None:
                print('[ warning ] No task defined, using default all input')
                self.custom_input = ['seq_x', 'seq_y', 'x_time', 'y_time', 'hetero_x_time', 'x_hetero', 'hetero_y_time', 'y_hetero', 'hetero_general', 'hetero_channel']
            elif self.task == 'TSF':
                self.custom_input = ['seq_x', 'seq_y', 'x_time', 'y_time']
            elif self.task == 'TGTSF':
                self.custom_input = ['seq_x', 'seq_y', 'x_time', 'y_time', 'hetero_y_time', 'y_hetero', 'hetero_general', 'hetero_channel']
            elif self.task == 'MTSF':
                self.custom_input = ['seq_x', 'seq_y', 'x_time', 'y_time', 'hetero_x_time', 'x_hetero', 'hetero_general', 'hetero_channel']
            elif self.task == 'Reasoning' or 'all':
                self.custom_input = ['seq_x', 'seq_y', 'x_time', 'y_time', 'hetero_x_time', 'x_hetero', 'hetero_y_time', 'y_hetero', 'hetero_general', 'hetero_channel']
            else:
                raise NotImplementedError('Task not supported, please use custom input to override')
            

    # @profile
    def __read_data__(self):
        self.scaler = StandardScaler()
        if self.data_buffer is None:
            if self.data_path.endswith('.csv'):
                df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
            elif self.data_path.endswith('.parquet'):
                df_raw = pd.read_parquet(os.path.join(self.root_path, self.data_path))
            else:
                raise NotImplementedError('Only .csv and .parquet data are supported, implement more if needed')
        elif isinstance(self.data_buffer, data_buffer):
            df_raw = self.data_buffer(os.path.join(self.root_path, self.data_path))

        # convert the timestamp to datetime
        df_raw[self.timestamp_col] = pd.to_datetime(df_raw[self.timestamp_col])

        # Check if timestamp_col has timezone
        if df_raw[self.timestamp_col][0].tz is not None:
            if self.timezone is not None:
                print('[ info ] The timestamp column has timezone, converting to {}'.format(self.timezone))
                df_raw[self.timestamp_col] = pd.to_datetime(df_raw[self.timestamp_col], utc=True).dt.tz_convert(self.timezone).dt.tz_localize(None)
            else:
                print('[ info ] The timestamp column has timezone, forcing UTC')
                df_raw[self.timestamp_col] = pd.to_datetime(df_raw[self.timestamp_col], utc=True).dt.tz_convert('UTC').dt.tz_localize(None)

        # Handle missing values before splitting (ensures consistent processing across train/val/test)
        # Exclude timestamp column from missing value processing
        exclude_cols = [self.timestamp_col]
        df_raw, self.missing_indicators = handle_missing_values(
            df_raw,
            strategy=self.missing_value_strategy,
            exclude_cols=exclude_cols,
            required_indicators=self.required_indicators
        )
        # Note: Logging is aggregated in data_factory.py get_datasets() method

        # apply the spliter
        train_data, val_data, test_data = self.spliter(df=df_raw)

        if self.set_type == 'train':
            self.data = train_data
        elif self.set_type == 'val':
            self.data = val_data
        elif self.set_type == 'test':
            self.data = test_data

        # Use debug level to avoid cluttering output during data loading
        logger.debug(f"Length of {self.set_type}: {self.data.shape[0]}")

        # convert the self.timestamp_col to yyyymmddHHMMSS int
        self.data[self.timestamp_col] = self.data[self.timestamp_col].dt.strftime('%Y%m%d%H%M%S')
        # convert to int
        self.data[self.timestamp_col] = self.data[self.timestamp_col].astype(np.int64)
        

        self.timestamp = self.data[self.timestamp_col].values.copy()
        if self.target == 'all':
            # Store column names before converting to numpy array
            # This allows tracking which columns are targets vs indicators
            all_columns = [col for col in self.data.columns if col != self.timestamp_col]
            # Target columns exclude indicator columns (indicators are covariates, not targets)
            self.target_columns = [col for col in all_columns if col not in self.missing_indicators]
            
            self.data = self.data.drop(columns=[self.timestamp_col])
            self.data = self.data.values.astype(np.float32).copy()
            train_data = train_data.drop(columns=[self.timestamp_col])
            train_data = train_data.values.astype(np.float32).copy()
        else:
            # Single target case: target column is the specified column
            self.target_columns = [self.target] if isinstance(self.target, str) else self.target
            self.data = self.data[self.target].values.astype(np.float32).copy()
            train_data = train_data[self.target].values.astype(np.float32).copy()

        if self.scale:
            self.scaler.fit(train_data)
            # Use debug level to avoid cluttering output during data loading
            logger.debug(f"mean and std (on train) of {self.data_path}: mean {self.scaler.mean_}, std {self.scaler.var_}")
            self.data = self.scaler.transform(self.data).astype(np.float32).copy()

        if self.downsample is not None:

            self.data = self.data[::self.downsample]
            self.timestamp = self.timestamp[::self.downsample]


    def __preload_hetero__(self):
        """
        Preloads all heterogeneous data into memory for efficient batch processing.
        
        When enabled, this method loads the complete heterogeneous dataset at initialization
        time rather than loading data on-demand during training. This improves training
        speed at the cost of increased memory usage.
        """
        if self.preload_hetero:
            print('[ info ] Preloading the full heterogeneous data')
            _ = time()
            self.hetero_time, self.hetero_general, self.hetero_channel, self.full_hetero = self.hetero_data_getter(self.timestamp)
            print('[ info ] Preload the full heterogeneous data successfully, cost time: {:.2f}s'.format(time() - _))
            del self.hetero_data_getter
            
    def __getitem__(self, index):
        """
        Retrieves a single data sample with all associated modalities.
        
        Constructs a complete training/inference sample containing time series data,
        timestamps, and corresponding heterogeneous cross-modal information. Handles
        both preloaded and on-demand heterogeneous data loading based on configuration.
        
        Args:
            index (int): Sample index in the dataset
        
        Returns:
            tuple: Complete data sample containing:
                - seq_x: Input time series sequence (seq_len, features)
                - seq_y: Target time series sequence (pred_len, features)  
                - x_time, y_time: Corresponding timestamps
                - x_hetero, y_hetero: Heterogeneous data (text, events, etc.)
                - hetero_x_time, hetero_y_time: Heterogeneous data timestamps
                - hetero_general: General heterogeneous information
                - hetero_channel: Channel-specific heterogeneous information
        """
        
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        x_time = self.timestamp[s_begin:s_end]
        y_time = self.timestamp[r_begin:r_end]
        
        # Generate deterministic sample_id: entity_id|timestamp_start|sequence_index
        # timestamp_start is the first timestamp in the input sequence
        timestamp_start = x_time[0] if len(x_time) > 0 else 0
        sequence_index = index
        
        # Format timestamp as string (it's already an int64 from __read_data__)
        timestamp_str = str(timestamp_start)
        
        # Generate sample_id with format: entity_id|timestamp|sequence_index
        if self.entity_id is not None:
            sample_id = f"{self.entity_id}|{timestamp_str}|{sequence_index}"
        else:
            # Fallback if entity_id not provided (backward compatibility)
            sample_id = f"unknown|{timestamp_str}|{sequence_index}"

        # Initialize defaults
        x_hetero = np.zeros((1), dtype=np.float32)
        y_hetero = np.zeros((1), dtype=np.float32)
        hetero_x_time = np.zeros((1), dtype=np.float32)
        hetero_y_time = np.zeros((1), dtype=np.float32)
        hetero_general = np.zeros((1), dtype=np.float32)
        hetero_channel = np.zeros((1), dtype=np.float32)
        
        # Heterogeneous data loading - explicit paths, no fallback
        # Path 1: LLM Embedding Provider (TimeCMA-style models)
        # Path 2: Traditional hetero_data_getter (Fidel-TS, Time-MMD text embeddings)
        
        if self.llm_embedding_provider is not None:
            # LLM embeddings mode: hetero_channel comes from precomputed LLM cache
            # Other hetero fields (x_hetero, y_hetero, etc.) are not used in this mode
            hetero_channel = self.llm_embedding_provider[index]
            
        elif self.preload_hetero:
            # Preloaded hetero mode: all hetero data was loaded at init
            hetero_general = self.hetero_general
            hetero_channel = self.hetero_channel

            if 'x_hetero' in self.custom_input:
                hetero_x_time = self.hetero_time[s_begin:s_end:self.hetero_stride]
                x_hetero = self.full_hetero[s_begin:s_end:self.hetero_stride]
            if 'y_hetero' in self.custom_input:
                hetero_y_time = self.hetero_time[r_begin:r_end:self.hetero_stride]
                y_hetero = self.full_hetero[r_begin:r_end:self.hetero_stride]
            
        else:
            # On-demand hetero mode: fetch hetero data per sample via getter
            if 'x_hetero' in self.custom_input:
                x_hetero = self.hetero_data_getter(x_time[::self.hetero_stride])
                hetero_x_time = x_hetero[0]
                hetero_general = x_hetero[1]
                hetero_channel = x_hetero[2]
                x_hetero = x_hetero[3]

            if 'y_hetero' in self.custom_input:
                y_hetero = self.hetero_data_getter(y_time[::self.hetero_stride])
                hetero_y_time = y_hetero[0]
                hetero_general = y_hetero[1]
                hetero_channel = y_hetero[2]
                y_hetero = y_hetero[3]
        
        # Generate time features if enabled (for FEDformer and similar models)
        x_time_features = None
        y_time_features = None
        if self.generate_time_features:
            x_time_features = time_features(x_time, freq=self.time_feature_freq)
            y_time_features = time_features(y_time, freq=self.time_feature_freq)
        
        # Return sample_id as first element for consistent sample tracking across models
        # still return everything for compatibility, but unwanted set as 0 for efficiency
        return sample_id, seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, x_time_features, y_time_features

    def __len__(self):
        """
        Returns the total number of valid samples in the dataset.
        
        Calculates the number of complete sequences that can be generated
        given the sequence length and prediction length constraints.
        
        Returns:
            int: Number of valid data samples (non-negative, returns 0 if dataset is too small)
        """
        length = len(self.data) - self.seq_len - self.pred_len + 1
        return max(0, length)

    def inverse_transform(self, data):
        """
        Reverses the normalization transformation applied to data.
        
        Converts normalized data back to original scale using the fitted scaler.
        Essential for interpreting model predictions in their original units.
        
        Args:
            data (np.ndarray): Normalized data to transform back
        
        Returns:
            np.ndarray: Data in original scale
        """
        return self.scaler.inverse_transform(data)



class Heterogeneous_Dataset(Dataset):
    """
    Specialized dataset for managing heterogeneous cross-modal data sources, 
    currently specialized for Fidel-TS datasets. For Time-MMD and TTC datasets 
    use the TimeMMD_Dataset class.
    
    This class handles diverse data types including text (news, events), images, 
    and other modalities that complement time series forecasting. It provides
    efficient data loading, temporal alignment, and embedding generation for
    cross-modal forecasting tasks.
    
    Usage Pattern:
    -------------
    Heterogeneous_Dataset is instantiated once by Data_Provider.__init__() when
    hetero_info is configured. It's then used to create hetero_data_getter functions
    via init_hetero_data(id) for each entity ID. These functions are passed to
    Universal_Dataset instances, which call them during __getitem__() to fetch
    text/embedding data for input/target sequences.
    
    Key Features:
        - Support for multiple heterogeneous data formats (JSON, text, images, embeddings)
        - Temporal alignment between time series and heterogeneous data
        - Downtime handling (replaces data with downtime prompts during sensor failures)
        - Memory-efficient data loading and caching via new embedding system
        - Flexible matching strategies (nearest, forward, backward, single)
        - Support for two data structures: 'all_for_one' (shared time series) and
          'each_subset' (independent time series per entity)
    
    Args:
        root_path (str): Root directory for heterogeneous data files
        formatter (str): File naming pattern for data files
        id_info (dict): Mapping of dataset IDs to metadata (includes downtime info)
        static_path (str, optional): Path to static/constant heterogeneous data JSON file
        matching (str): Strategy for temporal alignment ('nearest', 'forward', 'backward', 'single')
        output_format (str): Format for heterogeneous data output ('json', 'dict', 'csv', 'embedding')
        timezone (str, optional): Timezone for timestamp alignment
        noise (float): Noise level for data augmentation
        hetero_type (str): Data structure type - 'all_for_one' (single DataFrame) or
                          'each_subset' (dict of DataFrames keyed by ID)
        id_list (list, optional): Specific IDs to process
        postemb (str, optional): Positional embedding configuration
        postemb_model (str, optional): Model for generating positional embeddings
        postemb_max_len (int, optional): Maximum sequence length for embeddings
        postemb_d (int, optional): Dimensionality of positional embeddings
        postemb_batch_size (int): Batch size for embedding generation
        postemb_handle_downtime (str, optional): Strategy for handling data gaps (deprecated)
        device (str): Computing device ('cpu' or cuda device id)
        embedding_config (dict, optional): Configuration for new embedding system
        use_old_embeddings (bool): Whether to use old .pkl embedding files (default: False)
        base_data_path (str, optional): Base path for data directory (default: './data')
    
    Example:
        ```python
        # In Data_Provider.__init__():
        hetero_dataset = Heterogeneous_Dataset(
            root_path='./data/Bear_room/hetero',
            formatter='dynamic_aggregate_text_v3.json',
            id_info=id_info_dict,
            matching='backward',
            output_format='embedding',
            hetero_type='each_subset',
            embedding_config={'model_name': 'bert-base-uncased', 'aggregation_method': 'average'}
        )
        
        # Later in get_datasets():
        getter = hetero_dataset.init_hetero_data(entity_id)  # Returns callable
        dataset = Universal_Dataset(..., hetero_data_getter=getter)
        ```
    """
    def __init__(self, root_path, formatter, id_info, static_path=None, matching='nearest', output_format='json', 
                 timezone=None, noise = 0.0, hetero_type='all_for_one', id_list=None, postemb=None, postemb_model=None, 
                 postemb_max_len=None, postemb_d=None, postemb_batch_size=200, postemb_handle_downtime=None, device='cpu', 
                 embedding_config=None, use_old_embeddings=False, base_data_path=None, console=None):
        super().__init__()

        self.hetero_type = hetero_type
        self.root_path = root_path
        self.formatter = formatter
        self.id_list = id_list
        self.device = torch.device('cpu') if device == 'cpu' else torch.device(f'cuda:{device}')
        self.postemb = postemb
        self.postemb_model = postemb_model
        self.postemb_max_len = int(postemb_max_len) if postemb_max_len is not None else postemb_max_len
        self.postemb_d = int(postemb_d) if postemb_d is not None else postemb_d
        self.postemb_batch_size = int(postemb_batch_size) if postemb_batch_size is not None else postemb_batch_size
        self.postemb_handle_downtime = postemb_handle_downtime

        self.tokenizer = AutoTokenizer.from_pretrained(self.postemb_model) if postemb is not None else None
        self.model = AutoModel.from_pretrained(self.postemb_model).to(self.device) if postemb is not None else None

        self.id_info = id_info
        self.static_path = static_path
        assert matching in ['nearest', 'forward', 'backward', 'single'], "The matching method should be one of ['nearest', 'forward', 'backward', 'single']"
        self.matching = matching
        assert output_format in ['dict','json', 'csv', 'embedding'], "The output format should be one of ['dict','json', 'csv', 'embedding']"
        self.output_format = output_format
        self.timezone = timezone
        
        # New embedding system parameters
        self.embedding_config = embedding_config or {}
        self.use_old_embeddings = use_old_embeddings
        self.base_data_path = base_data_path or './data'  # Default to './data' if not provided
        self.console = console  # Rich Console for progress bars
        
        if self.output_format == 'embedding':
            assert self.formatter is not None, "The embedding formatter should be provided if the output format is embedding"
            self.load_embedding(id_list=self.id_list)
        else:
            # Text formats (json/dict/csv) are not supported for Fidel-TS datasets.
            # Heterogeneous_Dataset is now exclusively for Fidel-TS which always uses embedding format.
            # For text-based models with Time-MMD/TTC datasets, use TimeMMD_Dataset instead.
            raise NotImplementedError(
                f"Heterogeneous_Dataset only supports output_format='embedding' for Fidel-TS datasets. "
                f"Got output_format='{self.output_format}'. "
                f"For text formats (json/dict/csv), use TimeMMD_Dataset with Time-MMD or TTC datasets instead."
            )
        self.noise = noise

    def __addnoise__(self, x):
        x = x * (1 - self.noise) + np.random.randn(*x.shape) * self.noise
        # normalize
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        return x
    
    def _detect_fidel_ts_dataset(self) -> Optional[str]:
        """
        Detect if this is a Fidel-TS dataset and return dataset name.
        
        Returns:
            Dataset name if Fidel-TS dataset detected, None otherwise
        """
        # Check if root_path contains any known Fidel-TS dataset name
        root_path_str = str(self.root_path)
        for dataset_name in FidelTSPathResolver.DATASET_NAMES:
            if dataset_name in root_path_str:
                return dataset_name
        return None
    
    def load_embedding(self, id_list=None):
        """
        Load embeddings for Fidel-TS dataset.
        
        Heterogeneous_Dataset is only for Fidel-TS datasets. All embedding loading
        is handled by FidelTSEmbeddingLoader (which supports both new cache and old .pkl formats).
        """
        # Detect Fidel-TS dataset (should always succeed since this class is only for Fidel-TS)
        dataset_name = self._detect_fidel_ts_dataset()
        if not dataset_name:
            raise ValueError(
                "Heterogeneous_Dataset is only for Fidel-TS datasets. "
                "For Time-MMD/TTC datasets, use TimeMMD_Dataset instead."
            )
        
        # Use Fidel-TS embedding system (handles both new cache and old .pkl formats)
        self._load_embedding_fidel_ts(dataset_name)
    
    def _load_embedding_fidel_ts(self, dataset_name: str):
        """
        Load embeddings using new Fidel-TS embedding system that supports
        metadata, multiple embedding types, and manages complicated and 
        unique file structures within each Fidel-TS subdataset.
        
        Args:
            dataset_name: Name of Fidel-TS dataset
        """
        # Prepare hetero_info dict for FidelTSEmbeddingLoader
        hetero_info = {
            'root_path': self.root_path,
            'formatter': self.formatter,
            'static_path': self.static_path
        }
        
        # Get embedding config parameters
        embed_model_name = self.embedding_config.get('model_name', 'bert-base-uncased')
        aggregation_method = self.embedding_config.get('aggregation_method', 'cls')
        force_reembed = self.embedding_config.get('force_reembed', False)
        hf_cache_dir = self.embedding_config.get('hf_cache_dir', './HF_cache/')
        
        # Create embedding loader
        loader = FidelTSEmbeddingLoader(
            dataset_name=dataset_name,
            hetero_info=hetero_info,
            base_data_path=self.base_data_path,
            embed_model_name=embed_model_name,
            aggregation_method=aggregation_method,
            device=str(self.device) if hasattr(self.device, 'index') else 'cpu',
            hf_cache_dir=hf_cache_dir,
            force_reembed=force_reembed,
            use_old_embeddings=self.use_old_embeddings,
            console=self.console
        )
        
        # Load embeddings
        dynamic_embeddings, static_embeddings = loader.load_embeddings()
        
        # Store embeddings in same format as old system
        self.embeddings = dynamic_embeddings
        self.static_data = static_embeddings
        
        # Create dynamic data for timestamp matching (shared helper method)
        if self.hetero_type == 'all_for_one':
            self._create_dynamic_data_from_embeddings()
        else:
            # each_subset - not yet supported for new system
            raise NotImplementedError(
                'New embedding system currently only supports hetero_type="all_for_one". '
                'Set use_old_embeddings=True to use old system for each_subset.'
            )
    
    def _create_dynamic_data_from_embeddings(self):
        """
        Create dynamic_data DataFrame from embeddings keys (timestamps).
        
        This method creates a "placeholder" DataFrame that serves as an index for temporal
        matching. The actual embeddings are stored in self.embeddings (dict), but the
        time_matcher() method needs a sorted pandas DatetimeIndex to efficiently perform
        nearest/forward/backward timestamp matching operations.
        
        Why it's needed:
        - self.embeddings is a dict {timestamp_str: embedding_array} - good for exact lookup
        - time_matcher() needs a sorted DatetimeIndex to use pandas searchsorted() for
          efficient temporal queries (finding nearest timestamp, etc.)
        - The DataFrame values are dummy (just 0) - only the index matters for matching
        - Also handles timezone conversion and sorting of timestamps
        
        Used for hetero_type='all_for_one'.
        """
        if self.hetero_type != 'all_for_one':
            raise ValueError(f"_create_dynamic_data_from_embeddings() only supports hetero_type='all_for_one', got '{self.hetero_type}'")
        
        self.dynamic_data = pd.DataFrame.from_dict({k: 0 for k in self.embeddings.keys()}, orient='index')
        self.dynamic_data['time'] = self.dynamic_data.index
        self.dynamic_data.index = pd.to_datetime(self.dynamic_data.index)
        
        # Time zone processing
        if self.dynamic_data.index.tz is not None:
            if self.timezone is not None:
                print('[ info ] The index has timezone, converting to {}'.format(self.timezone))
                self.dynamic_data.index = self.dynamic_data.index.tz_convert(self.timezone).tz_localize(None)
            else:
                print('[ Warning ] The index has timezone, forcing UTC')
                self.dynamic_data.index = self.dynamic_data.index.tz_convert('UTC').tz_localize(None)
        
        self.dynamic_data.sort_index(inplace=True)
    
    def _create_dynamic_data_from_embeddings_dict(self, embeddings_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Create dynamic_data DataFrame from embeddings dict for a single ID.
        
        Shared helper method for hetero_type='each_subset'.
        
        Args:
            embeddings_dict: Dictionary mapping timestamps to embeddings
        
        Returns:
            DataFrame with timestamps as index
        """
        df = pd.DataFrame.from_dict({k: 0 for k in embeddings_dict.keys()}, orient='index')
        df['time'] = df.index
        df.index = pd.to_datetime(df.index)
        
        # Time zone processing
        if df.index.tz is not None:
            if self.timezone is not None:
                df.index = df.index.tz_convert(self.timezone).tz_localize(None)
            else:
                df.index = df.index.tz_convert('UTC').tz_localize(None)
        
        df.sort_index(inplace=True)
        return df

    def init_hetero_data(self, id):
        """
        Factory method that creates a callable hetero_data_getter function for a specific entity ID.
        
        This method prepares entity-specific parameters (downtime ranges, static info) and creates
        a partially applied version of `get_hetero_data` that can be called with just timestamps.
        The returned function implements the hetero_data_getter interface expected by Universal_Dataset.
        
        Where It's Used:
        ----------------
        Called by Data_Provider.get_datasets() for each entity ID when creating Universal_Dataset
        instances for Fidel-TS datasets (non-Time-MMD datasets with hetero_info configured).
        The returned function is passed as the `hetero_data_getter` parameter to Universal_Dataset.
        
        Args:
            id: Entity/channel ID for which to create the hetero_data_getter function
        
        Returns:
            callable: A partially applied function that takes timestamps and returns
                (matched_times, general_info, channel_info, output_dynamic) tuple.
                This function can be called as: hetero_data_getter(timestamps)
        
        Process:
        --------
        1. Extracts downtime ranges from id_info for the given entity ID
        2. Converts downtime to pandas IntervalIndex with timezone handling
        3. Retrieves entity-specific static data (general_info, channel_info, downtime_prompt)
        4. Returns a partial function that binds these parameters to get_hetero_data()
        
        Note:
        -----
        This method is entity-specific - each entity ID gets its own hetero_data_getter
        with its own downtime ranges and channel_info. For Time-MMD datasets, use
        TimeMMD_HeteroGetter instead, which handles text directly from CSV columns.
        """
        down_time = self.id_info[id]['sensor_downtime']
        down_time = [down_time[k]['time'] for k in down_time.keys()]
        down_time = [[pd.to_datetime(t[0]), pd.to_datetime(t[1])] for t in down_time]

        # check if all the downtime have timezone
        if any([t[0].tz is not None for t in down_time]):
            if self.timezone is not None:
                print('[ info ] The downtime has timezone, converting to {}'.format(self.timezone))
                down_time = [[t[0].tz_convert(self.timezone).tz_localize(None), t[1].tz_convert(self.timezone).tz_localize(None)] for t in down_time]
            else:
                print('[ Warning ] The downtime has timezone, forcing UTC')
                down_time = [[t[0].tz_convert('UTC').tz_localize(None), t[1].tz_convert('UTC').tz_localize(None)] for t in down_time]
            # print('[ info ] The downtime has timezone, converting to naive datetime, if need to keep timezone, please implement alignment using UDT')
            # down_time = [[t[0].tz_localize(None), t[1].tz_localize(None)] for t in down_time]

        general_info = self.static_data['general_info']
        channel_info = self.static_data['channel_info'][id]
        
        # Normalize channel_info shape to (1, embedding_dim) if it's an embedding
        # This ensures it works with TGTSF projection which expects 3D/4D inputs [B, 1, D] or [B, C, D]
        if isinstance(channel_info, np.ndarray):
            channel_info = self._normalize_embedding_shape(channel_info)

        # channel_info = channel_info.reshape(1, 256) if channel_info.shape == (256,) else channel_info
        
        downtime_prompt = self.static_data['downtime_prompt']
        # Convert downtime ranges to IntervalIndex using from_arrays
        start_times = [t[0] for t in down_time]
        end_times = [t[1] for t in down_time]
        downtime_ranges = pd.IntervalIndex.from_arrays(start_times, end_times)

        return partial(self.get_hetero_data, downtime_ranges, general_info, channel_info, downtime_prompt, id)
            

    def time_matcher(self, timestamps, id=None):
        # Convert timestamps to datetime
        timestamps = pd.to_datetime(timestamps.astype(str))

        id_specific_df = self.dynamic_data[id] if self.hetero_type == 'each_subset' else self.dynamic_data

        # Match times using vectorized operations on the correct DataFrame
        matched_indices = id_specific_df.index.searchsorted(timestamps)
        if self.matching == 'nearest':
            prev_indices = np.maximum(matched_indices - 1, 0)
            next_indices = np.minimum(matched_indices, len(id_specific_df.index) - 1)
            prev_deltas = (timestamps - id_specific_df.index[prev_indices]).total_seconds()
            next_deltas = (id_specific_df.index[next_indices] - timestamps).total_seconds()
            matched_indices = np.where(prev_deltas <= next_deltas, prev_indices, next_indices)
        elif self.matching == 'forward':
            matched_indices = np.minimum(matched_indices, len(id_specific_df.index) - 1)
        elif self.matching in ['backward', 'single']:
            matched_indices = np.maximum(matched_indices - 1, 0)

        matched_times = id_specific_df.index[matched_indices]

        if self.matching == 'single':
            _, unique_indices = np.unique(matched_times, return_index=True)
            matched_times = matched_times[unique_indices]
            timestamps = timestamps[unique_indices]

        return matched_times

    def downtime_checker(self, timestamps, downtime_ranges):
        # try:
            

            # Check downtime using vectorized operations
        is_downtime = np.array([any(downtime_ranges.contains(ts)) for ts in timestamps])
        # except TypeError:
        #     print(timestamps, type(timestamps), downtime_ranges, type(downtime_ranges))

        return is_downtime
    
    def _normalize_embedding_shape(self, emb: np.ndarray) -> np.ndarray:
        """
        Normalize embedding shape to expected format for Fidel-TS compatibility.
        
        Ensures embeddings have shape (1, embedding_dim) for CLS/average aggregation
        to match the expected format for downtime concatenation in get_hetero_data().
        
        Args:
            emb: Embedding array (may have shape (embedding_dim,) or (1, embedding_dim))
        
        Returns:
            Normalized embedding array with shape (1, embedding_dim)
        """
        if len(emb.shape) == 1:
            # 1D array: (embedding_dim,) -> reshape to (1, embedding_dim)
            emb = emb.reshape(1, -1)
        elif len(emb.shape) == 2:
            if emb.shape[0] != 1:
                # If (embedding_dim, 1) or other shape, reshape to (1, embedding_dim)
                emb = emb.reshape(1, -1)
            # else: already (1, embedding_dim) - correct shape
        else:
            raise ValueError(
                f"Unexpected embedding shape: {emb.shape}. "
                f"Expected 1D (embedding_dim,) or 2D (1, embedding_dim)"
            )
        
        return emb.astype(np.float32)

    # @profile
    def get_hetero_data(self, downtime_ranges, general_info, channel_info, downtime_prompt, id, timestamp):
        """
        Retrieve heterogeneous (text/embedding) data for given timestamps.
        
        This is the core method that fetches and formats heterogeneous data (embeddings or text)
        for a sequence of timestamps. It handles temporal matching, downtime detection, and
        output formatting according to the configured output_format.
        
        Where It's Used:
        ----------------
        Called indirectly via hetero_data_getter functions created by init_hetero_data().
        The hetero_data_getter is invoked by Universal_Dataset.__getitem__() during training/inference
        to fetch text/embedding data for input and/or target sequences.
        
        Args:
            downtime_ranges (pd.IntervalIndex): Time intervals when sensors were down
            general_info: General dataset information (string or embedding array)
            channel_info: Channel/entity-specific information (string or embedding array)
            downtime_prompt: Prompt/embedding to use during downtime periods
            id: Entity/channel ID (used only for hetero_type='each_subset')
            timestamp: Array of timestamps to retrieve data for (int64 format: YYYYMMDDHHMMSS)
        
        Returns:
            tuple: (matched_times, general_info, channel_info, output_dynamic)
                - matched_times: List of matched timestamps as strings (YYYYMMDDHHMMSS)
                - general_info: General dataset info (format depends on output_format)
                - channel_info: Channel-specific info (format depends on output_format)
                - output_dynamic: Dynamic heterogeneous data in requested format:
                    - For 'embedding': numpy array of shape (num_timesteps, num_items, embedding_dim)
                      where num_items depends on hetero_type and downtime handling
                    - For 'json'/'dict'/'csv': Formatted text data
        
        Process:
        --------
        1. Matches input timestamps to available heterogeneous data using time_matcher()
        2. Checks which matched timestamps fall within downtime periods
        3. Retrieves dynamic embeddings/text for matched timestamps from self.embeddings or self.dynamic_data
        4. Handles downtime: replaces data with downtime_prompt or zero vectors
        5. Formats output according to output_format ('embedding', 'json', 'dict', 'csv')
        6. Optionally applies noise augmentation if self.noise > 0
        
        Hetero Type Differences:
        ------------------------
        - all_for_one: Uses single self.dynamic_data DataFrame and self.embeddings dict
                       (all entities share the same time series)
        - each_subset: Uses self.dynamic_data[id] DataFrame and self.embeddings[id] dict
                       (each entity has independent time series data)
        
        Note:
        -----
        This method is typically not called directly. Instead, use init_hetero_data(id) to
        create a callable function that binds entity-specific parameters (downtime_ranges,
        general_info, channel_info, downtime_prompt, id) to this method, leaving only
        timestamp as the argument.
        """

        if self.hetero_type == 'all_for_one':
            # Match times
            matched_times = self.time_matcher(timestamp)

            # Check downtime
            if len(downtime_ranges) == 0:
                is_downtime = np.zeros(len(matched_times), dtype=bool)
            else:  
                is_downtime = self.downtime_checker(matched_times, downtime_ranges)

            if self.output_format == 'embedding':
                matched_dynamic = self.dynamic_data.loc[matched_times]['time'].values
                # Normalize embedding shapes to (1, embedding_dim) before batching
                # This ensures compatibility with downtime concatenation logic
                normalized_embeddings = [self._normalize_embedding_shape(self.embeddings[time]) for time in matched_dynamic]
                output_dynamic_ = np.array(normalized_embeddings, dtype=np.float32)  # Shape: (batch, 1, embedding_dim)
                
                # Normalize downtime_prompt to (1, embedding_dim)
                downtime_prompt_norm = self._normalize_embedding_shape(downtime_prompt)
                
                # Downtime data: shape (batch, 1, embedding_dim)
                downtime_data_ = np.array([downtime_prompt_norm if is_down else np.zeros_like(downtime_prompt_norm) 
                                        for is_down in is_downtime], dtype=np.float32)
                
                # Concatenate dynamic embeddings and downtime indicators along num_items dimension
                # Final shape: (batch, 2, embedding_dim) where 2 = num_items (dynamic + downtime)
                output_dynamic = np.concatenate([output_dynamic_, downtime_data_], axis=1)

                if self.noise > 0:
                    output_dynamic = self.__addnoise__(output_dynamic)

            else:
                if self.postemb is not None:
                    matched_dynamic = self.dynamic_data.loc[matched_times].copy()
                    matched_embed = self.dynamic_embed.loc[matched_times].copy() # .to_dict(orient='records')
                    # handling downtime
                    if self.postemb_handle_downtime is not None:
                        downtime_indices = np.where(is_downtime)[0]
                        for i in downtime_indices:
                            row_data = matched_dynamic.iloc[i].drop('time')
                            concatenated_row = ' '.join(str(x) for x in row_data.values) + " " + downtime_prompt
                            embedding_add_downtime = self.convert_plain_text_to_embeddings(concatenated_row)
                            matched_embed.iloc[i, matched_embed.columns.get_loc('embeddings')] = embedding_add_downtime
                    matched_embed = matched_embed.to_dict(orient='records')
                    output_dynamic = torch.stack([matched_df['embeddings'] for matched_df in matched_embed], dim=0)
                    # handling different length
                    if output_dynamic.size(0) < self.postemb_max_len:
                        padding_output = torch.zeros(self.postemb_max_len, self.postemb_d)
                        padding_output[:output_dynamic.size(0)] = output_dynamic
                        output_dynamic = padding_output
                    elif output_dynamic.size(0) > self.postemb_max_len:
                        output_dynamic = output_dynamic[:self.postemb_max_len]
                else:
                    matched_dynamic = self.dynamic_data.loc[matched_times].copy()
                    matched_dynamic['note'] = np.where(is_downtime, downtime_prompt, '')
                    matched_dynamic = matched_dynamic.to_dict(orient='records')
                    # remove the time from the dicts
                    for record in matched_dynamic:
                        record.pop('time', None)
                    
                    if self.output_format == 'dict':
                        output_dynamic = matched_dynamic
                    elif self.output_format == 'json':
                        output_dynamic = [json.dumps(record) for record in matched_dynamic]
                    elif self.output_format == 'csv':
                        output_dynamic = matched_dynamic.to_csv(index=False)
                    else:
                        raise NotImplementedError('Output format is not implemented yet')

            matched_times = matched_times.strftime('%Y%m%d%H%M%S').tolist()
            return matched_times, general_info, channel_info, output_dynamic

        elif self.hetero_type == 'each_subset':
            # Match times using the correct id
            matched_times = self.time_matcher(timestamp, id)

            if len(downtime_ranges) == 0:
                is_downtime = np.zeros(len(matched_times), dtype=bool)
            else:  
                is_downtime = self.downtime_checker(matched_times, downtime_ranges)

            if self.output_format == 'embedding':
                # Get the matched dynamic data for the specific subset id
                matched_dynamic = self.dynamic_data[id].loc[matched_times]['time'].values
                id_specific_embeddings = self.embeddings[id]
                # Normalize embedding shapes to (1, embedding_dim) before batching
                # This ensures compatibility with downtime concatenation logic
                normalized_embeddings = [self._normalize_embedding_shape(id_specific_embeddings[time]) for time in matched_dynamic]
                output_dynamic_ = np.array(normalized_embeddings, dtype=np.float32)  # Shape: (batch, 1, embedding_dim)
                
                # Normalize downtime_prompt to (1, embedding_dim)
                downtime_prompt_norm = self._normalize_embedding_shape(downtime_prompt)
                
                # Downtime data: shape (batch, 1, embedding_dim)
                downtime_data_ = np.array([downtime_prompt_norm if is_down else np.zeros_like(downtime_prompt_norm) 
                                        for is_down in is_downtime], dtype=np.float32)
                
                # Concatenate dynamic embeddings and downtime indicators along num_items dimension
                # Final shape: (batch, 2, embedding_dim) where 2 = num_items (dynamic + downtime)
                output_dynamic = np.concatenate([output_dynamic_, downtime_data_], axis=1)
                if self.noise > 0:
                    output_dynamic = self.__addnoise__(output_dynamic)
            else:
                # handling postemb
                if self.postemb is not None:
                    matched_dynamic = self.dynamic_data[id].loc[matched_times].copy()
                    matched_embed = self.dynamic_embed[id].loc[matched_times].copy() # .to_dict(orient='records')
                    # handling downtime
                    if self.postemb_handle_downtime is not None:
                        downtime_indices = np.where(is_downtime)[0]
                        for i in downtime_indices:
                            row_data = matched_dynamic.iloc[i].drop('time')
                            concatenated_row = ' '.join(str(x) for x in row_data.values) + " " + downtime_prompt
                            embedding_add_downtime = self.convert_plain_text_to_embeddings(concatenated_row)
                            matched_embed.iloc[i, matched_embed.columns.get_loc('embeddings')] = embedding_add_downtime
                    matched_embed = matched_embed.to_dict(orient='records')
                    output_dynamic = torch.stack([matched_df['embeddings'] for matched_df in matched_embed], dim=0)
                    # handling different length
                    if output_dynamic.size(0) < self.postemb_max_len:
                        padding_output = torch.zeros(self.postemb_max_len, self.postemb_d)
                        padding_output[:output_dynamic.size(0)] = output_dynamic
                        output_dynamic = padding_output
                    elif output_dynamic.size(0) > self.postemb_max_len:
                        output_dynamic = output_dynamic[:self.postemb_max_len]
                else:
                    matched_dynamic = self.dynamic_data[id].loc[matched_times].copy()
                    matched_dynamic['note'] = np.where(is_downtime, downtime_prompt, '')
                    matched_dynamic = matched_dynamic.to_dict(orient='records')
                    # remove the time from the dicts
                    for record in matched_dynamic:
                        record.pop('time', None)
                    
                    if self.output_format == 'dict':
                        output_dynamic = matched_dynamic
                    elif self.output_format == 'json':
                        output_dynamic = [json.dumps(record) for record in matched_dynamic]
                    elif self.output_format == 'csv':
                        output_dynamic = matched_dynamic.to_csv(index=False)
                    else:
                        raise NotImplementedError('Output format is not implemented yet')

            matched_times = matched_times.strftime('%Y%m%d%H%M%S').tolist()
            return matched_times, general_info, channel_info, output_dynamic
        
        else:
            raise NotImplementedError('Only all_for_one and each_subset hetero type are supported, implement more if needed')
