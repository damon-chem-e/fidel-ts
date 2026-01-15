
from data_provider.data_loader import Universal_Dataset, Heterogeneous_Dataset
from torch.utils.data import DataLoader
import json
import torch
import os
import re
from utils.tools import dotdict
from functools import partial
from .data_helper import timestamp_spliter, ratio_spliter, data_buffer
from typing import Optional, Any, Dict
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from utils.entity_data_check import check_entity_sufficient, get_entity_data_size
from utils.missing_value_handler import scan_missing_value_columns

class Data_Provider(object):
    """
    Central data management class for the Universal Cross-Modal Time Series Forecasting Pipeline.
    
    This class orchestrates data loading, preprocessing, and provides unified access to training,
    validation, and test datasets. It supports both traditional time series data and heterogeneous
    cross-modal data sources (text, events, etc.) for enhanced forecasting capabilities.
    
    Args:
        args: Configuration object containing all data and model parameters including:
            - data_config: Dataset configuration with root_path, id_info, target, etc.
            - batch_size: Batch size for data loaders
            - input_len: Length of input sequences
            - output_len: Length of prediction sequences
            - scale: Whether to apply standardization
            - noise: Noise injection settings
            - num_workers: Number of workers for data loading
            - prefetch_factor: Prefetch factor for data loaders
        buffer (bool): Whether to enable data buffering for improved performance.
            When True, loaded data files are cached in memory to avoid repeated I/O.
    
    Attributes:
        id_list: List of dataset IDs to process
        formatter: String formatter for dataset file naming (e.g., 'id_{i}.parquet')
        spliter: Function for splitting data into train/val/test sets
        data_buffer: Optional data buffer for caching
        hetero_dataset: Heterogeneous dataset handler for cross-modal data
    
    Example:
        ```python
        data_provider = Data_Provider(args, buffer=True)
        train_loader = data_provider.get_train(return_type='loader')
        val_loader = data_provider.get_val(return_type='loader')
        ```
    """
    def __init__(self, args, buffer=False, console: Optional[Any] = None):
        """
        Initialize Data_Provider.
        
        Args:
            args: Configuration object containing all data and model parameters
            buffer: Whether to enable data buffering for improved performance
            console: Optional Rich Console instance for progress bar display
        """
        self.args = args
        self.buffer = buffer
        self.batch_size = args.batch_size
        self.console = console

        self.dataset_config = args.data_config

        # Load or create id_info.json
        self.id_info = self._load_or_create_id_info()

        if self.dataset_config.id == 'all':
            self.id_list = list(self.id_info.keys())
        else:
            self.id_list = self.dataset_config.id
            # check if all the id in the list is in the id_info
            for id in self.dataset_config.id:
                assert id in self.id_info.keys(), "The id {} is not in the id_info".format(id)

        self.formatter = self.dataset_config.get('formatter', 'id_{i}.parquet')
        self.spliter = self.get_spliter()

        if buffer:
            self.data_buffer = data_buffer()
        else:
            self.data_buffer = None
        
        # Filter entities with insufficient data if enabled
        # Must be called after data_buffer is initialized
        filter_insufficient = self.dataset_config.get('filter_insufficient_entities', False)
        if filter_insufficient:
            self._filter_insufficient_entities()
        
        # Scan entities to determine which columns need missing value indicators
        # This ensures consistent feature dimensions across all entities
        # Only do this if missing value strategy is enabled
        missing_value_strategy = self.dataset_config.get('missing_value_strategy', 'none')
        if missing_value_strategy == 'forward_fill_indicators':
            # Get data_path (may be None or 'null' for multi-file datasets)
            data_path = self.dataset_config.get('data_path', None)
            self.required_indicator_columns = scan_missing_value_columns(
                id_list=self.id_list,
                root_path=self.dataset_config.root_path,
                timestamp_col=self.dataset_config.timestamp_col,
                data_path=data_path,
                formatter=self.formatter,
                data_buffer=self.data_buffer
            )
        else:
            self.required_indicator_columns = []

        if args.data_config.hetero_info is not None:
            hetero_info = dotdict(args.data_config.hetero_info)
            if hetero_info.root_path is None:
                hetero_info.root_path = args.data_config.root_path
            
            # Parse embedding config (for new embedding system)
            embedding_config = getattr(hetero_info, 'embedding_config', None)
            if embedding_config is not None:
                embedding_config = dotdict(embedding_config) if not isinstance(embedding_config, dict) else embedding_config
            
            # Parse use_old_embeddings flag (default: False, use new system)
            use_old_embeddings = getattr(hetero_info, 'use_old_embeddings', False)
            
            # Get base_data_path from args (if provided in config)
            base_data_path = getattr(self.args, 'base_data_path', None)
            
            # Create Heterogeneous_Dataset instance for managing Fidel-TS embedding/text data
            # 
            # This is NOT the dataset returned to users. Instead, it's a helper class that:
            # 1. Loads/computes embeddings or text data for Fidel-TS datasets during __init__
            # 2. Provides init_hetero_data(id) method to create hetero_data_getter functions
            # 3. These functions are passed to Universal_Dataset instances in get_datasets()
            # 4. Universal_Dataset calls hetero_data_getter(timestamps) during __getitem__()
            #    to fetch text/embedding data for input/target sequences
            #
            # The output_format is determined by hetero_info.input_format:
            # - 'embedding': Uses new embedding system (FidelTSEmbeddingLoader) for embedding-based models
            # - 'json'/'dict'/'csv': Uses text format for prompting-based models (deprecated path via load_data)
            #
            # Note: This is ONLY used for non-Time-MMD datasets. Time-MMD datasets use
            # TimeMMD_HeteroGetter instead, which is created internally by TimeMMD_Dataset.
            self.hetero_dataset = Heterogeneous_Dataset(root_path=hetero_info.root_path, 
                                                        formatter=hetero_info.formatter, 
                                                        id_info=self.id_info, 
                                                        matching=hetero_info.matching, 
                                                        output_format=hetero_info.input_format, 
                                                        static_path=hetero_info.static_path, 
                                                        timezone=self.dataset_config.time_zone, 
                                                        noise=self.args.noise, 
                                                        hetero_type=hetero_info.hetero_type, 
                                                        id_list=self.id_list, 
                                                        postemb=hetero_info.postemb, 
                                                        postemb_model=hetero_info.postemb_model, 
                                                        postemb_max_len=hetero_info.postemb_max_len, 
                                                        postemb_d=hetero_info.postemb_d, 
                                                        postemb_batch_size=hetero_info.postemb_batch_size, 
                                                        postemb_handle_downtime=hetero_info.postemb_handle_downtime, 
                                                        device=self.args.gpu if self.args.use_gpu else 'cpu',
                                                        embedding_config=embedding_config,
                                                        use_old_embeddings=use_old_embeddings,
                                                        base_data_path=base_data_path,
                                                        console=self.console)
        
        # LLM Embedding Provider - for TimeCMA and similar models
        # Loaded lazily per-split in get_datasets() when llm_embedding config is present
        self.llm_embedding_config = getattr(args, 'llm_embedding', None)
        self._llm_embedding_providers: Dict[str, Any] = {}  # Cache providers per split

        # Tensor cache for fast data loading
        self.use_tensor_cache = getattr(args, 'use_tensor_cache', False)
        self.tensor_cache_dir = self._resolve_tensor_cache_dir()
        self._tensor_cache_validated = False  # Track if validation was done
        self._tensor_cache_valid = False  # Result of validation

    def _get_llm_embedding_provider(self, flag: str):
        """
        Get or create LLM embedding provider for a split.
        
        Loads precomputed LLM embeddings from cache for models like TimeCMA
        that use GPT-2/LLM embeddings of time series data.
        
        Args:
            flag: Dataset split ('train', 'val', 'test')
        
        Returns:
            LLMEmbeddingProvider instance, or None if not configured
        
        Raises:
            FileNotFoundError: If embeddings not found (user should run generate-suite)
        """
        # Return None if no LLM embedding config
        if self.llm_embedding_config is None:
            return None
        
        # Return cached provider if already loaded for this split
        if flag in self._llm_embedding_providers:
            return self._llm_embedding_providers[flag]
        
        # Load provider for this split
        from embedder.llm_embedding_provider import LLMEmbeddingProvider
        
        # Build experiment config from args
        experiment_config = self._build_experiment_config_for_llm()
        
        try:
            provider = LLMEmbeddingProvider.from_experiment_config(
                experiment_config, 
                split=flag,
                validate=True,
                quiet=False,
            )
            self._llm_embedding_providers[flag] = provider
            return provider
        except FileNotFoundError as e:
            # Re-raise with helpful context
            raise FileNotFoundError(
                f"LLM embeddings required but not found for '{flag}' split.\n"
                f"The experiment config has 'llm_embedding' section, indicating "
                f"this model requires precomputed LLM embeddings.\n\n"
                f"Generate them with:\n"
                f"  python -m cli.inference generate-suite <suite_config.yaml>\n"
                f"Or:\n"
                f"  python -m cli.inference generate <experiment_config.yaml>\n\n"
                f"Original error: {e}"
            ) from e
    
    def _build_experiment_config_for_llm(self) -> Dict[str, Any]:
        """
        Build experiment config dict for LLMEmbeddingProvider.
        
        Extracts relevant configuration from args to match the format
        expected by LLMEmbeddingProvider.from_experiment_config().
        
        Returns:
            Dict with data, training, llm_embedding, and other config
        """
        # Get dataset name - handle different config structures
        dataset_name = None
        if hasattr(self.dataset_config, 'name'):
            dataset_name = self.dataset_config.name
        elif hasattr(self.args, 'data_name'):
            dataset_name = self.args.data_name
        
        if dataset_name is None:
            raise ValueError(
                "Cannot determine dataset name for LLM embedding loading. "
                "Ensure data config has 'name' field."
            )
        
        # Build config dict
        return {
            'data': {
                'name': dataset_name,
                'config_path': getattr(self.dataset_config, 'config_path', None),
            },
            'training': {
                'input_len': self.args.input_len,
                'output_len': self.args.output_len,
                'scale': getattr(self.args, 'scale', True),
            },
            'llm_embedding': self.llm_embedding_config,
            'base_data_path': getattr(self.args, 'base_data_path', './data/'),
            'model_config_overrides': getattr(self.args, 'model_config_overrides', {}),
        }

    def _get_effective_hetero_stride(self) -> int:
        """
        Get the effective hetero_stride for text embedding temporal resolution.
        
        This centralizes the hetero_stride computation logic, supporting:
        - Pre-computed hetero_stride from experiment_config_builder
        - Legacy fallback to model_config.stride if hetero_align_stride=True
        
        The stride determines how many text embeddings are loaded:
        - stride=1: Full resolution (all timesteps)
        - stride=3: Every 3rd timestep (reduces from 24 to 8 for input_len=24)
        
        See docs/planning/hetero_stride_optional_plan.md for details.
        
        Returns:
            Effective hetero_stride (1 = full resolution, >1 = strided)
        """
        # Check if pre-computed by build_experiment_args (preferred path)
        if hasattr(self.args, 'hetero_stride'):
            return self.args.hetero_stride
        
        # Legacy fallback: compute from model_config
        # This path is used by:
        # 1. Old training code that doesn't use build_experiment_args
        # 2. Embedding generation scripts (llm_embedder.py) - they should use stride=1
        if not hasattr(self.args, 'model_config'):
            # No model_config - this is likely embedding generation or testing
            # Default to full resolution (stride=1)
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "Computing hetero_stride without model_config. "
                "Defaulting to stride=1 (full resolution). "
                "This is expected for embedding generation but unusual for training."
            )
            return 1
            
        model_config = self.args.model_config
        hetero_align_stride = getattr(model_config, 'hetero_align_stride', True)
        
        if hetero_align_stride:
            return getattr(model_config, 'stride', 1) or 1
        else:
            return 1

    def _resolve_tensor_cache_dir(self) -> Optional[str]:
        """
        Resolve tensor cache directory path.

        If tensor_cache_dir is explicitly set in config, use it.
        Otherwise, auto-generate based on dataset root_path and config hash.

        The auto-generated path follows the pattern:
            {dataset_root_path}/tensor_cache/{config_hash}/

        This co-locates caches with dataset data and enables automatic reuse
        when the same config hash is encountered again.

        For time_mmd datasets with root_path='data/time_mmd/Traffic':
            -> Cache: data/time_mmd/Traffic/tensor_cache/<hash>/

        For other datasets with root_path='data/fidel-ts/germany_renewable/time_series':
            -> Cache: data/fidel-ts/germany_renewable/time_series/tensor_cache/<hash>/

        Returns:
            Path to tensor cache directory, or None if not using tensor cache
        """
        if not self.use_tensor_cache:
            return None

        # Check if explicitly set in args
        explicit_dir = getattr(self.args, 'tensor_cache_dir', None)
        if explicit_dir is not None:
            return explicit_dir

        # Auto-generate path based on dataset location and config hash
        # Cache lives directly under root_path: {root_path}/tensor_cache/<hash>/
        from data_provider.tensor_cache import compute_config_hash

        # Build config dict for hash computation
        config = self._build_tensor_cache_config()
        config_hash = compute_config_hash(config)

        # Get dataset root path - cache goes directly under this directory
        root_path = self.dataset_config.root_path
        dataset_dir = root_path.rstrip('/\\')

        cache_dir = os.path.join(dataset_dir, 'tensor_cache', config_hash)
        return cache_dir

    def _build_tensor_cache_config(self) -> dict:
        """
        Build config dict for tensor cache hash computation.

        These are the parameters that affect cache validity - if any change,
        the cache must be regenerated.

        IMPORTANT: This must match the config built by utils.experiment_config_builder.build_cache_config()
        to ensure consistent hash computation across CLI and runtime.

        Returns:
            Dict of config parameters for hash computation
        """
        return {
            'input_len': getattr(self.args, 'input_len', None),
            'output_len': getattr(self.args, 'output_len', None),
            'scale': getattr(self.args, 'scale', True),
            'truncate_train_for_purge': getattr(self.args, 'truncate_train_for_purge', False),
            'downsample': getattr(self.args, 'downsample', None),
            'data_name': getattr(self.dataset_config, 'name', 'unknown'),
            'hetero_stride': self._get_effective_hetero_stride(),
            'hetero_type': self.dataset_config.hetero_info.get('hetero_type') if self.dataset_config.get('hetero_info') else None,
            'timemmd_text_output': self.dataset_config.get('timemmd_text_output'),  # Critical for time_mmd datasets!
            'missing_value_strategy': self.dataset_config.get('missing_value_strategy', 'none'),
            'split_info': str(self.dataset_config.get('split_info', '')),
        }

    def _validate_tensor_cache(self) -> bool:
        """
        Validate that tensor cache exists and matches current config.

        Performs validation once and caches the result.

        Returns:
            True if cache is valid and can be used, False otherwise

        Raises:
            FileNotFoundError: If use_tensor_cache is True but cache is invalid
        """
        # Only validate once
        if self._tensor_cache_validated:
            return self._tensor_cache_valid

        self._tensor_cache_validated = True

        if not self.use_tensor_cache or self.tensor_cache_dir is None:
            self._tensor_cache_valid = False
            return False

        # Build config dict for validation (same as used for hash)
        from data_provider.tensor_cache import validate_cache

        config = self._build_tensor_cache_config()
        is_valid, message = validate_cache(self.tensor_cache_dir, config)

        if not is_valid:
            raise FileNotFoundError(
                f"Tensor cache requested but invalid: {message}\n"
                f"Cache directory: {self.tensor_cache_dir}\n\n"
                f"Generate the cache with:\n"
                f"  python -m cli.tensor_cache generate <config.yaml>\n\n"
                f"Or disable tensor cache by setting:\n"
                f"  training.use_tensor_cache: false"
            )

        self._tensor_cache_valid = True
        print(f"[ info ] Using tensor cache from: {self.tensor_cache_dir}")
        return True

    def _get_tensor_cache_dataloader(self, flag: str, shuffle: bool, drop_last: bool):
        """
        Get a DataLoader from tensor cache.

        Args:
            flag: Data split ('train', 'val', 'test')
            shuffle: Whether to shuffle the data
            drop_last: Whether to drop the last incomplete batch

        Returns:
            DataLoader configured for tensor cache
        """
        from data_provider.tensor_cache import get_tensor_cache_dataloader
        
        # BEGIN DEBUG
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 ** 3)
        print(f"[DEBUG MEM] Before creating {flag} DataLoader: RSS={mem_before:.2f}GB")
        # END DEBUG

        # Log DataLoader creation start (this can take time with many workers)
        print(f"[ info ] Creating {flag} DataLoader from tensor cache (num_workers={self.args.num_workers})...")
        
        dataloader = get_tensor_cache_dataloader(
            cache_dir=self.tensor_cache_dir,
            flag=flag,
            batch_size=self.batch_size,
            num_workers=self.args.num_workers,
            prefetch_factor=self.args.prefetch_factor,
            shuffle=shuffle,
            preload_to_ram=False  # Memory-mapped is usually best
        )
        
        # BEGIN DEBUG
        mem_after = process.memory_info().rss / (1024 ** 3)
        mem_delta = mem_after - mem_before
        print(f"[DEBUG MEM] After creating {flag} DataLoader: RSS={mem_after:.2f}GB (delta={mem_delta:+.2f}GB)")
        # END DEBUG
        
        print(f"[ info ] {flag} DataLoader created successfully")
        return dataloader

    def get_spliter(self):
        """
        Creates and returns a data splitting function based on configuration.
        
        Supports two splitting strategies:
        - 'timestamp': Split data based on specific timestamp boundaries
        - 'ratio': Split data based on proportional ratios (e.g., 7:1:2 for train:val:test)
        
        Returns:
            callable: Configured splitting function that takes a DataFrame and returns
                     (train_data, val_data, test_data) tuple
        """
        # Check if we're in embedding generation mode (suppress verbose messages)
        quiet = getattr(self.args, 'embedding_generation_mode', False)
        
        if self.dataset_config.spliter == 'timestamp':
            spliter = partial(timestamp_spliter, split=self.dataset_config.split_info, seq_len=self.args.input_len, timestamp_col=self.dataset_config.timestamp_col, quiet=quiet)
        elif self.dataset_config.spliter == 'ratio':
            spliter = partial(ratio_spliter, split=self.dataset_config.split_info, seq_len=self.args.input_len)
        else:
            print('no split method specified, use ratio of 7:1:2 as default')
            spliter = partial(ratio_spliter, split=(7,1,2), seq_len=self.args.input_len)
        return spliter
    
    def _filter_insufficient_entities(self):
        """
        Filter out entities that don't have sufficient data for train/val/test splits.
        
        Checks each entity's data file size and removes entities that don't have
        enough data points for the required sequence lengths and split ratios.
        Logs a warning for each filtered entity.
        """
        # Get split parameters
        seq_len = self.args.input_len
        pred_len = self.args.output_len
        split_type = self.dataset_config.spliter if hasattr(self.dataset_config, 'spliter') else 'ratio'
        
        # Get split ratios from config
        if hasattr(self.dataset_config, 'split_info') and self.dataset_config.split_info is not None:
            split_ratios = self.dataset_config.split_info
            if isinstance(split_ratios, str):
                # Handle string format "x:y:z"
                split_ratios = [int(x) for x in split_ratios.split(':')]
        else:
            # Default to 7:1:2
            split_ratios = (7, 1, 2)
        
        # Get data_path (may be None or 'null' for multi-file datasets)
        data_path = self.dataset_config.get('data_path', None)
        # Handle case where YAML has 'null' as string
        if data_path == 'null':
            data_path = None
        
        # Filter entities
        filtered_ids = []
        sufficient_ids = []
        
        for entity_id in self.id_list:
            # Get entity data size
            num_rows, file_path = get_entity_data_size(
                root_path=self.dataset_config.root_path,
                data_path=data_path,
                formatter=self.formatter,
                entity_id=entity_id,
                data_buffer=self.data_buffer
            )
            
            # Check if file exists and has data
            if num_rows is None:
                print(f"[ warning ] Filtered entity '{entity_id}': file not found or error reading ({file_path})")
                filtered_ids.append(entity_id)
                continue
            
            # Check if entity has sufficient data
            is_sufficient, details = check_entity_sufficient(
                total_rows=num_rows,
                seq_len=seq_len,
                pred_len=pred_len,
                split_ratios=split_ratios,
                split_type=split_type,
                require_all_splits=True
            )
            
            if not is_sufficient:
                # Build list of insufficient splits
                insufficient_splits = []
                if not details['train_ok']:
                    insufficient_splits.append('train')
                if not details['val_ok']:
                    insufficient_splits.append('val')
                if not details['test_ok']:
                    insufficient_splits.append('test')
                
                # Format split names (e.g., "train", "train and val", "train, val, and test")
                if len(insufficient_splits) == 1:
                    split_str = insufficient_splits[0]
                elif len(insufficient_splits) == 2:
                    split_str = f"{insufficient_splits[0]} and {insufficient_splits[1]}"
                else:
                    split_str = ', '.join(insufficient_splits[:-1]) + f', and {insufficient_splits[-1]}'
                
                print(f"[ warning ] Filtered entity '{entity_id}': insufficient data in {split_str} split")
                filtered_ids.append(entity_id)
            else:
                sufficient_ids.append(entity_id)
        
        # Update id_list to only include sufficient entities
        original_count = len(self.id_list)
        self.id_list = sufficient_ids
        filtered_count = len(filtered_ids)
        
        # Log summary
        if filtered_count > 0:
            print(f"[ info ] Entity filtering: removed {filtered_count} entities, using {len(sufficient_ids)} entities")
        else:
            print(f"[ info ] Entity filtering: all {original_count} entities have sufficient data")
    
    def _is_time_mmd_dataset(self):
        """
        Check if the current dataset configuration is for a Time-MMD dataset.
        
        Returns:
            bool: True if dataset_type is 'time_mmd', False otherwise
        """
        return self.dataset_config.get('dataset_type', None) == 'time_mmd'
    
    def _load_or_create_id_info(self):
        """
        Load or create id_info.json for the dataset.
        
        For all datasets, if id_info.json doesn't exist, attempts to create it by:
        1. For single-file datasets (data_path is set): create simple id_info with 'all'
        2. For multi-file datasets (formatter has {i}): scan directory for matching files
        
        This works for both Time-MMD and standard fidel-ts datasets.
        
        Returns:
            dict: id_info dictionary
        """
        # Get id_info filename (default: 'id_info.json')
        id_info_filename = getattr(self.dataset_config, 'id_info', 'id_info.json')
        
        # For fidel-ts datasets, id_info.json is at the dataset root (parent of time_series/)
        # For Time-MMD datasets, id_info.json is at root_path itself
        # Try both locations: parent directory first, then root_path
        parent_id_info_path = os.path.join(os.path.dirname(self.dataset_config.root_path), id_info_filename)
        root_id_info_path = os.path.join(self.dataset_config.root_path, id_info_filename)
        
        # Check parent directory first (fidel-ts pattern)
        if os.path.exists(parent_id_info_path):
            try:
                with open(parent_id_info_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f'[ warning ] Failed to load existing id_info from {parent_id_info_path}: {e}')
        
        # Check root_path directory (Time-MMD pattern)
        if os.path.exists(root_id_info_path):
            try:
                with open(root_id_info_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f'[ warning ] Failed to load existing id_info from {root_id_info_path}: {e}')
                print(f'[ info ] Will attempt to auto-create id_info')
        
        # Neither location has id_info - will auto-create
        # Use root_path as the default location for auto-created files
        id_info_path = root_id_info_path
        
        # id_info doesn't exist - try to auto-create it
        print(f'[ info ] id_info file not found at {id_info_path}')
        print(f'[ info ] Attempting to auto-create by scanning directory...')
        
        # Check if root_path exists
        if not os.path.exists(self.dataset_config.root_path):
            raise FileNotFoundError(
                f"Cannot create id_info: root_path does not exist: {self.dataset_config.root_path}\n"
                f"Please ensure the dataset has been downloaded and is in the correct location."
            )
        
        # Check if this is a single-file dataset (data_path is set)
        data_path = self.dataset_config.get('data_path', None)
        if data_path is not None:
            # Single-file dataset: create minimal id_info with 'all'
            dataset_type = 'Time-MMD' if self._is_time_mmd_dataset() else 'Standard'
            id_info = {'all': {'description': f'{dataset_type} single-file dataset'}}
            self._save_id_info(id_info_path, id_info)
            return id_info
        
        # Multi-file case: scan directory for files matching formatter pattern
        # Get formatter (default varies by dataset type)
        formatter = self.dataset_config.get('formatter', 'id_{i}.parquet')
        
        # Check if formatter has {i} placeholder (indicating multi-file pattern)
        if '{i}' not in formatter:
            # Formatter doesn't have {i}, treat as single-file case
            dataset_type = 'Time-MMD' if self._is_time_mmd_dataset() else 'Standard'
            id_info = {'all': {'description': f'{dataset_type} dataset (no {{i}} in formatter)'}}
            self._save_id_info(id_info_path, id_info)
            return id_info
        
        # Scan directory for files matching the formatter pattern
        id_info = self._scan_directory_for_ids(formatter, self.dataset_config.root_path)
        
        if not id_info:
            # No files found matching pattern
            pattern = formatter.replace('{i}', '*')
            raise FileNotFoundError(
                f"Cannot create id_info: no files matching pattern '{pattern}' found in {self.dataset_config.root_path}\n"
                f"Expected files like: {formatter.replace('{i}', '123')}\n"
                f"Please ensure the dataset has been downloaded and files are in the correct format."
            )
        
        # Successfully created id_info - save it
        pattern = formatter.replace('{i}', '*')
        print(f'[ info ] Discovered {len(id_info)} files matching pattern "{pattern}"')
        self._save_id_info(id_info_path, id_info)
        return id_info
    
    def _scan_directory_for_ids(self, formatter, root_path):
        """
        Scan directory for files matching the formatter pattern and extract IDs.
        
        Args:
            formatter: File pattern with {i} placeholder (e.g., 'id_{i}.parquet', '{i}.parquet')
            root_path: Directory to scan
            
        Returns:
            dict: id_info dictionary with discovered IDs
        """
        # Convert formatter to regex pattern, escaping special regex characters
        # Example: 'id_{i}.parquet' -> r'^id_(\d+)\.parquet$'
        # Example: '{i}.parquet' -> r'^(\d+)\.parquet$'
        regex_escaped = re.escape(formatter)
        # Replace the escaped {i} placeholder (which is now \{i\}) with a capture group for digits
        regex_escaped = regex_escaped.replace('\\{i\\}', r'(\d+)')
        # Anchor the pattern to match the entire filename
        regex_pattern = '^' + regex_escaped + '$'
        
        id_info = {}
        
        # List all files in the directory
        for filename in os.listdir(root_path):
            # Skip directories, only process files
            file_path = os.path.join(root_path, filename)
            if not os.path.isfile(file_path):
                continue
            
            # Check if filename matches the pattern
            match = re.match(regex_pattern, filename)
            if match:
                # Extract the ID (the number part)
                file_id = match.group(1)
                id_info[file_id] = {'description': f'Dataset entry: {filename}'}
        
        return id_info
    
    def _save_id_info(self, id_info_path, id_info):
        """
        Save id_info dictionary to JSON file.
        
        Args:
            id_info_path: Full path to id_info.json file
            id_info: Dictionary to save
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(id_info_path), exist_ok=True)
            
            with open(id_info_path, 'w') as f:
                json.dump(id_info, f, indent=2)
            print(f'[ info ] Created id_info file at {id_info_path} with {len(id_info)} entries')
        except Exception as e:
            print(f'[ warning ] Could not create id_info.json file: {e}')
            print(f'[ info ] Using in-memory id_info (this is OK)')
    
    def _create_time_mmd_dataset(self, i, flag, llm_embedding_provider=None):
        """
        Create a TimeMMD_Dataset instance for the given ID and flag.
        
        This method handles all Time-MMD dataset instantiation logic in one place
        to avoid code duplication.
        
        Args:
            i: Dataset ID
            flag: Dataset split identifier ('train', 'val', 'test')
            llm_embedding_provider: Optional LLMEmbeddingProvider for TimeCMA-style models
            
        Returns:
            TimeMMD_Dataset: Configured TimeMMD_Dataset instance
        """
        from data_provider.time_mmd_dataset import TimeMMD_Dataset
        
        # For Time-MMD, use data_path if specified and not None, otherwise formatter
        if 'data_path' in self.dataset_config and self.dataset_config.data_path is not None:
            data_path = self.dataset_config.data_path
        else:
            data_path = self.formatter.format(i=i) if '{i}' in self.formatter else self.formatter
        
        # Determine output_format based on timemmd_text_output config
        # Priority: timemmd_text_output > hetero_info.input_format > default 'json'
        timemmd_text_output = self.dataset_config.get('timemmd_text_output', None)
        if timemmd_text_output is not None:
            # Map timemmd_text_output to output_format
            if timemmd_text_output == 'text':
                output_format = 'json'  # Use JSON format for text mode
            elif timemmd_text_output == 'embedding':
                output_format = 'embedding'
            else:
                raise ValueError(f"Invalid timemmd_text_output: {timemmd_text_output}. Must be 'text' or 'embedding'")
        else:
            # Fall back to hetero_info if available
            output_format = 'json'  # default
            if self.args.data_config.hetero_info is not None:
                output_format = self.args.data_config.hetero_info.get('input_format', 'json')
        
        # Get embedding parameters from config (with defaults)
        embed_model_name = self.dataset_config.get('timemmd_embed_model', 'bert-base-uncased')
        # For embedding format, use BERT's native dimension (768)
        # Models will handle dimension conversion with learned projections if needed
        embed_dim = self.dataset_config.get('timemmd_embed_dim', 768)
        force_reembed = self.dataset_config.get('timemmd_force_reembed', False)
        # Get aggregation method from embeddings config if available, default to 'cls'
        aggregation_method = self.dataset_config.get('aggregation_method', 'cls')
        # hf_cache_dir is data-agnostic; prefer global args.hf_cache_dir, fall back to dataset config, then default
        hf_cache_dir = getattr(self.args, 'hf_cache_dir', None) or self.dataset_config.get('hf_cache_dir', './HF_cache/')
        
        # Get device from args (default to 'cpu' if not available)
        device = getattr(self.args, 'device', 'cpu')
        if hasattr(self.args, 'gpu') and self.args.gpu is not None:
            device = f'cuda:{int(self.args.gpu)}'
        
        # Get missing value strategy from config (default: 'none')
        missing_value_strategy = self.dataset_config.get('missing_value_strategy', 'none')
        
        # Get required indicator columns (ensures consistent feature dimensions)
        required_indicators = getattr(self, 'required_indicator_columns', [])
        
        # Determine if time features should be generated (for FEDformer, Informer, etc.)
        model_name = getattr(self.args, 'model', '').lower()
        generate_time_features = model_name in ['fedformer', 'informer', 'autoformer']
        time_feature_freq = getattr(self.args.model_config, 'freq', 'h') if hasattr(self.args, 'model_config') else 'h'
        
        return TimeMMD_Dataset(
            root_path=self.dataset_config.root_path,
            data_path=data_path,
            flag=flag,
            seq_len=self.args.input_len,
            pred_len=self.args.output_len,
            spliter=self.spliter,
            timestamp_col=self.dataset_config.timestamp_col,
            target=self.dataset_config.target,
            scale=self.args.scale,
            data_buffer=self.data_buffer,
            preload_hetero=self.args.preload_hetero,
            hetero_stride=self._get_effective_hetero_stride(),
            task=self.args.model_config.task,
            custom_input=self.args.model_config.custom_input,
            timezone=self.dataset_config.time_zone,
            downsample=self.dataset_config.downsample,
            entity_id=i,
            text_column=self.dataset_config.get('text_column', 'auto'),
            use_closedllm=self.dataset_config.get('use_closedllm', False),
            text_len=self.dataset_config.get('text_len', 4),
            output_format=output_format,
            general_info=self.dataset_config.get('general_info', ''),
            channel_info=self.dataset_config.get('channel_info', ''),
            embed_model_name=embed_model_name,
            embed_dim=embed_dim,
            force_reembed=force_reembed,
            hf_cache_dir=hf_cache_dir,
            device=device,
            missing_value_strategy=missing_value_strategy,
            required_indicators=required_indicators,
            aggregation_method=aggregation_method,
            generate_time_features=generate_time_features,
            time_feature_freq=time_feature_freq,
            llm_embedding_provider=llm_embedding_provider,
            truncate_train_for_purge=getattr(self.args, 'truncate_train_for_purge', False),
            console=self.console,
        )
    
    def get_train(self, return_type='loader'):
        """
        Creates and returns training data in the specified format.

        Args:
            return_type (str): Format of returned data. Options:
                - 'set': Returns dataset objects only
                - 'loader': Returns DataLoader objects only
                - 'both': Returns tuple of (dataset, dataloader)

        Returns:
            Dataset/DataLoader/tuple: Training data in requested format

        Note:
            If use_tensor_cache is True and cache is valid, returns tensor cache
            dataloader instead of computing from scratch. The 'set' return type
            is not supported with tensor cache (will raise ValueError).
        """
        assert return_type in ['set', 'loader', 'both'], 'return type not supported, only support set, loader, both'

        # Check if tensor cache should be used
        if self.use_tensor_cache and self._validate_tensor_cache():
            if return_type == 'set':
                raise ValueError(
                    "return_type='set' is not supported with tensor cache. "
                    "Use 'loader' instead, or disable tensor cache."
                )
            loader = self._get_tensor_cache_dataloader('train', shuffle=True, drop_last=True)
            if return_type == 'loader':
                return loader
            else:  # 'both' - return None for dataset since we don't have it
                return None, loader

        self.train_dataset=self.get_datasets('train')
        if return_type == 'set':
            return self.train_dataset
        elif return_type == 'loader':
            return self.get_dataloader(self.train_dataset, True, True, True)
        else:
            return self.train_dataset, self.get_dataloader(self.train_dataset, True, True, True)
    
    def get_val(self, return_type='loader'):
        """
        Creates and returns validation data in the specified format.

        Args:
            return_type (str): Format of returned data. Options:
                - 'set': Returns dataset objects only
                - 'loader': Returns DataLoader objects only
                - 'both': Returns tuple of (dataset, dataloader)

        Returns:
            Dataset/DataLoader/tuple: Validation data in requested format

        Note:
            If use_tensor_cache is True and cache is valid, returns tensor cache
            dataloader instead of computing from scratch.
        """
        # Check if tensor cache should be used
        if self.use_tensor_cache and self._validate_tensor_cache():
            if return_type == 'set':
                raise ValueError(
                    "return_type='set' is not supported with tensor cache. "
                    "Use 'loader' instead, or disable tensor cache."
                )
            loader = self._get_tensor_cache_dataloader('val', shuffle=False, drop_last=False)
            if return_type == 'loader':
                return loader
            else:  # 'both'
                return None, loader

        self.val_dataset = self.get_datasets('val')
        if return_type == 'set':
            return self.val_dataset
        elif return_type == 'loader':
            return self.get_dataloader(self.val_dataset, True, False, True)
        else:
            return self.val_dataset, self.get_dataloader(self.val_dataset, True, False, True)

    
    def get_test(self, return_type='loader'):
        """
        Creates and returns test data in the specified format.

        Args:
            return_type (str): Format of returned data. Options:
                - 'set': Returns dataset objects only
                - 'loader': Returns DataLoader objects only
                - 'both': Returns tuple of (dataset, dataloader)

        Returns:
            Dataset/DataLoader/tuple: Test data in requested format

        Note:
            If use_tensor_cache is True and cache is valid, returns tensor cache
            dataloader instead of computing from scratch.
        """
        # Check if tensor cache should be used
        if self.use_tensor_cache and self._validate_tensor_cache():
            if return_type == 'set':
                raise ValueError(
                    "return_type='set' is not supported with tensor cache. "
                    "Use 'loader' instead, or disable tensor cache."
                )
            loader = self._get_tensor_cache_dataloader('test', shuffle=False, drop_last=False)
            if return_type == 'loader':
                return loader
            else:  # 'both'
                return None, loader

        self.test_dataset = self.get_datasets('test')
        if return_type == 'set':
            return self.test_dataset
        elif return_type == 'loader':
            return self.get_dataloader(self.test_dataset, False, False, False)
        else:
            return self.test_dataset, self.get_dataloader(self.test_dataset, False, False, False)

    def get_datasets(self, flag):
        """
        Creates Universal_Dataset or TimeMMD_Dataset instances for all configured data IDs.
        
        This method handles two types of datasets:
        1. **Time-MMD datasets**: Creates TimeMMD_Dataset instances (which handle their own
           text embedding via TimeMMD_HeteroGetter internally)
        2. **Fidel-TS and other datasets**: Creates Universal_Dataset instances, optionally
           with hetero_data_getter from self.hetero_dataset if hetero_info is configured
        
        Heterogeneous Dataset Usage:
        ----------------------------
        If hetero_info is configured in the data config (checked during __init__), then
        self.hetero_dataset is a Heterogeneous_Dataset instance that manages Fidel-TS
        embedding/text data. In get_datasets(), for each entity ID:
        
        - If hetero_info is not None: Creates a hetero_data_getter function via
          self.hetero_dataset.init_hetero_data(i), which returns a callable that
          implements the hetero_data_getter interface expected by Universal_Dataset
        - If hetero_info is None: Sets hetero_data_getter to None (no heterogeneous data)
        
        Note: self.hetero_dataset is ONLY used for non-Time-MMD datasets. Time-MMD datasets
        use TimeMMD_HeteroGetter instead, which is created internally by TimeMMD_Dataset.
        
        Uses Rich Progress for clean progress bar display that doesn't interfere with logging.
        Aggregates missing value indicator logging to reduce clutter.
        
        Args:
            flag (str): Dataset split identifier ('train', 'val', 'test')
        
        Returns:
            dict: Dictionary mapping data IDs to their corresponding dataset instances
                (Universal_Dataset for Fidel-TS/standard datasets, TimeMMD_Dataset for Time-MMD)
        """
        datasets = {}
        
        # Track missing value indicators across all entities for aggregated logging
        all_indicator_columns = set()
        
        # Load LLM embedding provider if configured (for TimeCMA-style models)
        llm_embedding_provider = self._get_llm_embedding_provider(flag)
        
        # Use Rich Progress if console is available, otherwise fall back to simple iteration
        if self.console is not None:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("[progress.completed]{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=self.console,
                transient=False,
            ) as progress:
                task = progress.add_task(f"Loading {flag} datasets", total=len(self.id_list))
                for i in self.id_list:
                    if self._is_time_mmd_dataset():
                        dataset = self._create_time_mmd_dataset(i, flag, llm_embedding_provider)
                    else:
                        # Use standard Universal_Dataset
                        if self.args.data_config.hetero_info is not None:
                            get_hetero_data = self.hetero_dataset.init_hetero_data(i)
                        else:
                            get_hetero_data = None

                        data_path = self.formatter.format(i=i)
                        # Get missing value strategy from config (default: 'none')
                        missing_value_strategy = self.dataset_config.get('missing_value_strategy', 'none')
                        # Get required indicator columns (ensures consistent feature dimensions)
                        required_indicators = getattr(self, 'required_indicator_columns', [])
                        
                        # Determine if time features should be generated (for FEDformer, Informer, etc.)
                        # Check if model requires temporal marks
                        model_name = getattr(self.args, 'model', '').lower()
                        generate_time_features = model_name in ['fedformer', 'informer', 'autoformer']
                        time_feature_freq = getattr(self.args.model_config, 'freq', 'h') if hasattr(self.args, 'model_config') else 'h'
                        dataset = Universal_Dataset(root_path=self.dataset_config.root_path, data_path=data_path, 
                                                    flag=flag, seq_len=self.args.input_len, pred_len=self.args.output_len, 
                                                    spliter=self.spliter, timestamp_col=self.dataset_config.timestamp_col, 
                                                    target=self.dataset_config.target, scale=self.args.scale, 
                                                    data_buffer=self.data_buffer, hetero_data_getter=get_hetero_data, preload_hetero=self.args.preload_hetero, 
                                                    hetero_stride=self._get_effective_hetero_stride(),
                                                    task=self.args.model_config.task, custom_input=self.args.model_config.custom_input,
                                                    timezone=self.dataset_config.time_zone, downsample=self.dataset_config.downsample,
                                                    entity_id=i, missing_value_strategy=missing_value_strategy, required_indicators=required_indicators,
                                                    generate_time_features=generate_time_features, time_feature_freq=time_feature_freq,
                                                    llm_embedding_provider=llm_embedding_provider,
                                                    truncate_train_for_purge=getattr(self.args, 'truncate_train_for_purge', False),
                                                    console=self.console)
                    datasets[i] = dataset
                    # Collect indicator columns for aggregated logging
                    if hasattr(dataset, 'missing_indicators') and dataset.missing_indicators:
                        all_indicator_columns.update(dataset.missing_indicators)
                    progress.update(task, advance=1)
        else:
            # Fallback: simple iteration without progress bar
            for i in self.id_list:
                if self._is_time_mmd_dataset():
                    dataset = self._create_time_mmd_dataset(i, flag, llm_embedding_provider)
                else:
                    # Use standard Universal_Dataset
                    if self.args.data_config.hetero_info is not None:
                        get_hetero_data = self.hetero_dataset.init_hetero_data(i)
                    else:
                        get_hetero_data = None

                    data_path = self.formatter.format(i=i)
                    # Get missing value strategy from config (default: 'none')
                    missing_value_strategy = self.dataset_config.get('missing_value_strategy', 'none')
                    # Get required indicator columns (ensures consistent feature dimensions)
                    required_indicators = getattr(self, 'required_indicator_columns', [])
                    
                    # Determine if time features should be generated (for FEDformer, Informer, etc.)
                    # Check if model requires temporal marks
                    model_name = getattr(self.args, 'model', '').lower()
                    generate_time_features = model_name in ['fedformer', 'informer', 'autoformer']
                    time_feature_freq = getattr(self.args.model_config, 'freq', 'h') if hasattr(self.args, 'model_config') else 'h'
                    dataset = Universal_Dataset(root_path=self.dataset_config.root_path, data_path=data_path,
                                                flag=flag, seq_len=self.args.input_len, pred_len=self.args.output_len, 
                                                spliter=self.spliter, timestamp_col=self.dataset_config.timestamp_col, 
                                                target=self.dataset_config.target, scale=self.args.scale, 
                                                data_buffer=self.data_buffer, hetero_data_getter=get_hetero_data, preload_hetero=self.args.preload_hetero, 
                                                hetero_stride=self._get_effective_hetero_stride(),
                                                task=self.args.model_config.task, custom_input=self.args.model_config.custom_input,
                                                timezone=self.dataset_config.time_zone, downsample=self.dataset_config.downsample,
                                                entity_id=i, missing_value_strategy=missing_value_strategy, required_indicators=required_indicators,
                                                generate_time_features=generate_time_features, time_feature_freq=time_feature_freq,
                                                llm_embedding_provider=llm_embedding_provider,
                                                truncate_train_for_purge=getattr(self.args, 'truncate_train_for_purge', False),
                                                console=self.console)
                datasets[i] = dataset
                # Collect indicator columns for aggregated logging
                if hasattr(dataset, 'missing_indicators') and dataset.missing_indicators:
                    all_indicator_columns.update(dataset.missing_indicators)
        
        return datasets

    def get_dataloader(self, datasets, shuffle, drop_last, concat=False):
        """
        Creates PyTorch DataLoader instances from datasets.
        
        Args:
            datasets (dict): Dictionary of dataset instances mapped by ID
            shuffle (bool): Whether to shuffle data during loading
            drop_last (bool): Whether to drop the last incomplete batch
            concat (bool): Whether to concatenate all datasets into a single loader
                          or return separate loaders for each dataset
        
        Returns:
            DataLoader or dict: Single DataLoader if concat=True, 
                               dict of DataLoaders mapped by ID if concat=False
        """
        if concat:
            # Check for datasets with invalid length before concatenation
            for i in datasets.keys():
                dataset = datasets[i]
                dataset_len = len(dataset)
                if dataset_len <= 0:
                    raise ValueError(
                        f"Dataset '{i}' has invalid length {dataset_len}. "
                        f"Dataset must have at least seq_len ({getattr(dataset, 'seq_len', '?')}) + "
                        f"pred_len ({getattr(dataset, 'pred_len', '?')}) data points to create valid sequences. "
                        f"Current dataset has {len(getattr(dataset, 'data', []))} data points."
                    )
            
            data_set = torch.utils.data.ConcatDataset([datasets[i] for i in datasets.keys()])
            data_loader = DataLoader(data_set,
                                    batch_size=self.batch_size,
                                    shuffle=shuffle,
                                    drop_last=drop_last,
                                    num_workers=self.args.num_workers,
                                    persistent_workers=(self.args.num_workers > 1),
                                    pin_memory=True,
                                    prefetch_factor=self.args.prefetch_factor if self.args.num_workers > 1 else None,
                                    )
            return data_loader
        else:
            data_loader = {}
            for i in datasets.keys():
                data_loader[i] = DataLoader(datasets[i],
                                    batch_size=self.batch_size,
                                    shuffle=shuffle,
                                    drop_last=drop_last,
                                    num_workers=self.args.num_workers,
                                    persistent_workers=(self.args.num_workers > 1),
                                    pin_memory=True,
                                    prefetch_factor=self.args.prefetch_factor if self.args.num_workers > 1 else None,
                                    )

            return data_loader


