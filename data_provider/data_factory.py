
from data_provider.data_loader import Universal_Dataset, Heterogeneous_Dataset
from torch.utils.data import DataLoader
import json
import torch
import os
import re
from utils.tools import dotdict
from functools import partial
from .data_helper import timestamp_spliter, ratio_spliter, data_buffer
from typing import Optional, Any
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from utils.entity_data_check import check_entity_sufficient, get_entity_data_size

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

        if args.data_config.hetero_info is not None:
            hetero_info = dotdict(args.data_config.hetero_info)
            if hetero_info.root_path is None:
                hetero_info.root_path = args.data_config.root_path
            
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
                                                        device=self.args.gpu if self.args.use_gpu else 'cpu')

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
        if self.dataset_config.spliter == 'timestamp':
            spliter = partial(timestamp_spliter, split=self.dataset_config.split_info, seq_len=self.args.input_len, timestamp_col=self.dataset_config.timestamp_col)
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
                # Build issue description
                issues = []
                if not details['train_ok']:
                    issues.append(f"train({details['train_len']})")
                if not details['val_ok']:
                    issues.append(f"val({details['val_len']})")
                if not details['test_ok']:
                    issues.append(f"test({details['test_len']})")
                
                issue_str = ', '.join(issues)
                print(f"[ warning ] Filtered entity '{entity_id}': insufficient data ({num_rows} rows, need {details['min_needed']} for seq_len={seq_len}, pred_len={pred_len}, missing: {issue_str})")
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
        
        For Time-MMD datasets, creates id_info.json by scanning directory for files
        matching the formatter pattern if it doesn't exist.
        For standard datasets, loads the specified id_info.json file.
        
        Returns:
            dict: id_info dictionary
        """
        if self._is_time_mmd_dataset():
            # For Time-MMD, handle two cases:
            # 1. Single-file: data_path is set -> create simple id_info with 'all'
            # 2. Multi-file (TTC medical): data_path is null and formatter has {i} -> scan directory
            id_info_filename = getattr(self.dataset_config, 'id_info', 'id_info.json')
            id_info_path = os.path.join(self.dataset_config.root_path, id_info_filename)
            
            if os.path.exists(id_info_path):
                # Load existing id_info if present
                return json.load(open(id_info_path))
            
            # Check if this is a single-file dataset (data_path is set)
            data_path = self.dataset_config.get('data_path', None)
            if data_path is not None:
                # Single-file Time-MMD dataset: create minimal id_info with 'all'
                id_info = {'all': {'description': 'Time-MMD single-file dataset'}}
                # Optionally create the file for future use
                try:
                    with open(id_info_path, 'w') as f:
                        json.dump(id_info, f, indent=2)
                    print(f'[ info ] Created id_info.json at {id_info_path} for single-file dataset')
                except Exception as e:
                    print(f'[ warning ] Could not create id_info.json file: {e}')
                    print('[ info ] Using in-memory id_info (this is OK)')
                return id_info
            
            # Multi-file case (TTC): data_path is null, scan directory for files matching formatter
            # Only scan if formatter has {i} placeholder (indicating multi-file pattern)
            formatter = self.dataset_config.get('formatter', 'id_{i}.parquet')
            if '{i}' not in formatter:
                # Formatter doesn't have {i}, treat as single-file case
                id_info = {'all': {'description': 'Time-MMD dataset (no {i} in formatter)'}}
                try:
                    with open(id_info_path, 'w') as f:
                        json.dump(id_info, f, indent=2)
                    print(f'[ info ] Created id_info.json at {id_info_path} (formatter has no {{i}} placeholder)')
                except Exception as e:
                    print(f'[ warning ] Could not create id_info.json file: {e}')
                return id_info
            
            # Multi-file case: scan directory for files matching the formatter pattern
            # Extract IDs from filenames (e.g., patient_10576.csv -> 10576)
            id_info = {}
            
            # Extract pattern from formatter (e.g., 'patient_{i}.csv' -> 'patient_*.csv')
            # Replace {i} with * for glob pattern matching (for display purposes)
            pattern = formatter.replace('{i}', '*')
            
            # Convert formatter to regex pattern, escaping special regex characters
            # Example: 'patient_{i}.csv' -> r'^patient_(\d+)\.csv$'
            # Escape all regex special characters (this will escape {, }, and .)
            regex_escaped = re.escape(formatter)
            # Replace the escaped {i} placeholder (which is now \{i\}) with a capture group for digits
            # re.escape turns {i} into \{i\}, so we need to match the escaped version
            regex_escaped = regex_escaped.replace('\\{i\\}', r'(\d+)')
            # Anchor the pattern to match the entire filename
            regex_pattern = '^' + regex_escaped + '$'
            
            if os.path.exists(self.dataset_config.root_path):
                # List all files in the directory
                for filename in os.listdir(self.dataset_config.root_path):
                    # Skip directories, only process files
                    file_path = os.path.join(self.dataset_config.root_path, filename)
                    if not os.path.isfile(file_path):
                        continue
                    # Check if filename matches the pattern
                    match = re.match(regex_pattern, filename)
                    if match:
                        # Extract the ID (the number part)
                        patient_id = match.group(1)
                        id_info[patient_id] = {'description': f'Time-MMD dataset entry: {filename}'}
            
            if not id_info:
                # Fallback: if no files found, create minimal entry
                # This should not happen in normal operation
                id_info = {'all': {'description': 'Time-MMD dataset (no files found)'}}
                print(f'[ warning ] No files matching pattern "{pattern}" found in {self.dataset_config.root_path}')
            else:
                print(f'[ info ] Discovered {len(id_info)} files matching pattern "{pattern}"')
            
            # Optionally create the file for future use
            try:
                with open(id_info_path, 'w') as f:
                    json.dump(id_info, f, indent=2)
                print(f'[ info ] Created id_info.json at {id_info_path} with {len(id_info)} entries')
            except Exception as e:
                print(f'[ warning ] Could not create id_info.json file: {e}')
                print('[ info ] Using in-memory id_info (this is OK)')
            return id_info
        else:
            # Standard datasets require id_info
            id_info_path = os.path.join(self.dataset_config.root_path, self.dataset_config.id_info)
            return json.load(open(id_info_path))
    
    def _create_time_mmd_dataset(self, i, flag):
        """
        Create a TimeMMD_Dataset instance for the given ID and flag.
        
        This method handles all Time-MMD dataset instantiation logic in one place
        to avoid code duplication.
        
        Args:
            i: Dataset ID
            flag: Dataset split identifier ('train', 'val', 'test')
            
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
        # hf_cache_dir is data-agnostic; prefer global args.hf_cache_dir, fall back to dataset config, then default
        hf_cache_dir = getattr(self.args, 'hf_cache_dir', None) or self.dataset_config.get('hf_cache_dir', './HF_cache/')
        
        # Get device from args (default to 'cpu' if not available)
        device = getattr(self.args, 'device', 'cpu')
        if hasattr(self.args, 'gpu') and self.args.gpu is not None:
            device = f'cuda:{int(self.args.gpu)}'
        
        # Get missing value strategy from config (default: 'none')
        missing_value_strategy = self.dataset_config.get('missing_value_strategy', 'none')
        
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
            hetero_stride=self.args.model_config.stride if self.args.model_config.hetero_align_stride else 1,
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
            missing_value_strategy=missing_value_strategy
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
        """
        assert return_type in ['set', 'loader', 'both'], 'return type not supported, only support set, loader, both'
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
        """
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
        """
        self.test_dataset = self.get_datasets('test')
        if return_type == 'set':
            return self.test_dataset
        elif return_type == 'loader':
            return self.get_dataloader(self.test_dataset, False, False, False)
        else:
            return self.test_dataset, self.get_dataloader(self.test_dataset, False, False, False)

    def get_datasets(self, flag):
        """
        Creates Universal_Dataset instances for all configured data IDs.
        
        Uses Rich Progress for clean progress bar display that doesn't interfere with logging.
        
        Args:
            flag (str): Dataset split identifier ('train', 'val', 'test')
        
        Returns:
            dict: Dictionary mapping data IDs to their corresponding Universal_Dataset instances
        """
        datasets = {}
        
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
                        dataset = self._create_time_mmd_dataset(i, flag)
                    else:
                        # Use standard Universal_Dataset
                        if self.args.data_config.hetero_info is not None:
                            get_hetero_data = self.hetero_dataset.init_hetero_data(i)
                        else:
                            get_hetero_data = None

                        data_path = self.formatter.format(i=i)
                        # Get missing value strategy from config (default: 'none')
                        missing_value_strategy = self.dataset_config.get('missing_value_strategy', 'none')
                        dataset = Universal_Dataset(root_path=self.dataset_config.root_path, data_path=data_path, 
                                                    flag=flag, seq_len=self.args.input_len, pred_len=self.args.output_len, 
                                                    spliter=self.spliter, timestamp_col=self.dataset_config.timestamp_col, 
                                                    target=self.dataset_config.target, scale=self.args.scale, 
                                                    data_buffer=self.data_buffer, hetero_data_getter=get_hetero_data, preload_hetero=self.args.preload_hetero, 
                                                    hetero_stride=self.args.model_config.stride if self.args.model_config.hetero_align_stride else 1,
                                                    task=self.args.model_config.task, custom_input=self.args.model_config.custom_input,
                                                    timezone=self.dataset_config.time_zone, downsample=self.dataset_config.downsample,
                                                    entity_id=i, missing_value_strategy=missing_value_strategy)  # Pass entity_id and missing_value_strategy
                    datasets[i] = dataset
                    progress.update(task, advance=1)
        else:
            # Fallback: simple iteration without progress bar
            for i in self.id_list:
                if self._is_time_mmd_dataset():
                    dataset = self._create_time_mmd_dataset(i, flag)
                else:
                    # Use standard Universal_Dataset
                    if self.args.data_config.hetero_info is not None:
                        get_hetero_data = self.hetero_dataset.init_hetero_data(i)
                    else:
                        get_hetero_data = None

                    data_path = self.formatter.format(i=i)
                    # Get missing value strategy from config (default: 'none')
                    missing_value_strategy = self.dataset_config.get('missing_value_strategy', 'none')
                    dataset = Universal_Dataset(root_path=self.dataset_config.root_path, data_path=data_path,
                                                flag=flag, seq_len=self.args.input_len, pred_len=self.args.output_len, 
                                                spliter=self.spliter, timestamp_col=self.dataset_config.timestamp_col, 
                                                target=self.dataset_config.target, scale=self.args.scale, 
                                                data_buffer=self.data_buffer, hetero_data_getter=get_hetero_data, preload_hetero=self.args.preload_hetero, 
                                                hetero_stride=self.args.model_config.stride if self.args.model_config.hetero_align_stride else 1,
                                                task=self.args.model_config.task, custom_input=self.args.model_config.custom_input,
                                                timezone=self.dataset_config.time_zone, downsample=self.dataset_config.downsample,
                                                entity_id=i, missing_value_strategy=missing_value_strategy)  # Pass entity_id and missing_value_strategy
                datasets[i] = dataset
        
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


