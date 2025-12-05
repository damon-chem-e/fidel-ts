from exp.exp_basic import Exp_Basic
from models import model_init

import numpy as np
import torch
import torch.nn as nn

import os
import time
import warnings
import logging

import json
# Rich imports
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.console import Console

from data_provider.data_factory import Data_Provider

from utils.tools import general_move_to_device


class Experiment(Exp_Basic):
    """
    Experiment orchestrator for Foundation Model (FM) based time series forecasting.
    
    This class specializes the base experiment framework for pre-trained foundation models
    that have been designed for time series forecasting tasks. Foundation models leverage
    large-scale pre-training on diverse time series data to provide strong generalization
    capabilities across different domains and datasets.
    
    Key Features:
        - Foundation model initialization with pre-trained weights
        - Efficient fine-tuning and adaptation workflows
        - Memory-optimized device management for large models
        - Support for cross-modal and heterogeneous data inputs
        - Specialized inference procedures for foundation models
    
    Args:
        args: Configuration object containing FM-specific parameters including:
            - model_config: Foundation model architecture and parameter settings
            - fine_tuning: Adaptation strategy and learning rate schedules
            - inference: Batch processing and memory optimization settings
            - data: Cross-modal data configuration and preprocessing
    
    Example:
        ```python
        exp = Experiment(args)
        model = exp.train(setting='fm_adaptation_v1') 
        results = exp.test(setting='fm_adaptation_v1')
        ```
    """
    
    def __init__(self, args, exp_manager=None):
        super(Experiment, self).__init__(args, exp_manager)
        
    def _build_model(self):
        """
        Constructs and returns a Foundation Model for time series forecasting.
        
        Initializes pre-trained foundation models with specialized configurations
        for time series forecasting tasks. The model is built with FM-specific
        optimizations and adaptation capabilities.
        
        Returns:
            torch.nn.Module: Initialized foundation model configured for time series forecasting
        """
        model = model_init(self.args.model, self.args.model_config, self.args, is_FM=True)
        return model

    def _get_data(self, flag, return_type='loader'):
        """
        Retrieves data in the specified format for foundation model experiments.
        
        Args:
            flag (str): Dataset split identifier ('train', 'val', 'test')
            return_type (str): Format of returned data ('loader', 'set', 'both')
        
        Returns:
            DataLoader/Dataset/tuple: Data in requested format for foundation model training/inference
        """
        if flag == 'train':
            data_loader = self.data_provider.get_train(return_type=return_type)
        elif flag == 'val':
            data_loader = self.data_provider.get_val(return_type=return_type)
        elif flag == 'test':
            data_loader = self.data_provider.get_test(return_type=return_type)

        return data_loader

    def _forward_step(self, iter):
        """
        Executes a single forward pass for foundation model inference.
        
        Handles device placement and data formatting specific to foundation models,
        which may have different memory requirements and optimization strategies
        compared to standard models.
        
        Args:
            iter: Data batch containing:
                - sample_ids: Sample identifiers (first element)
                - batch_x, batch_y: Time series sequences
                - timestamp_x, timestamp_y: Timestamps
                - batch_x_hetero, batch_y_hetero: Heterogeneous data
                - hetero_x_time, hetero_y_time: Heterogeneous timestamps
                - hetero_general, hetero_channel: General and channel info
        
        Returns:
            tuple: (predictions, ground_truth, sample_ids) for foundation model evaluation
        """
        # iteration: sample_ids, seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel

        sample_ids, batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = iter

        if hasattr(self.model, 'move_to_device'):
            # move only the ones needed to device according to model's definition to save VRAM
            batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = self.model.move_to_device(batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, self.device)
        else:
            # only move batch_x, batch_y to device for TSF models
            batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = general_move_to_device(batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, self.device)
        
        # Require logger to be available via exp_manager
        logger = self.exp_manager.logger if self.exp_manager else None

        if self.args.individual:
            num_channels = batch_x.size(-1)
            outputs = []
            for c in range(num_channels):
                # batch_x: [batch_size, seq_len, num_channels]
                channel_x = batch_x[:, :, c:c+1].squeeze(-1)  # channel_x: [batch_size, seq_len]

                if self.args.task == 'TSF':
                    channel_output = self.model.forward(x=channel_x)  # channel_output: [batch_size, output_len, 1]
                elif self.args.task == 'TGTSF':
                    channel_output = self.model.forward(x=channel_x, batch_y_hetero=batch_y_hetero, hetero_general=hetero_general, hetero_channel=hetero_channel)
                else:
                    raise ValueError(f"Unsupported task type: {self.args.task}")
                
                if channel_output is None:
                    msg = f"[ Warning ]: Model returned None for channel {c}."
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                    return None, None, sample_ids
                elif torch.isnan(channel_output).any():
                    msg = f"[ Warning ]: NaN detected in channel {c} output"
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                    return None, None, sample_ids
                else:
                    channel_output = channel_output.unsqueeze(-1)
                    # print(f"Channel {c} output shape: {channel_output}") # Commented out verbose print
                
                outputs.append(channel_output)
            
            final_output = torch.cat(outputs, dim=-1)  # final_output: [batch_size, output_len, num_channels]
        
        else:
            if self.args.task == 'TSF':
                final_output = self.model.forward(x = batch_x)
            elif self.args.task == 'TGTSF':
                final_output = self.model.forward(x=channel_x, batch_y_hetero=batch_y_hetero, hetero_general=hetero_general, hetero_channel=hetero_channel)
            else:
                raise ValueError(f"Unsupported task type: {self.args.task}")
            
            if final_output is None:
                msg = "[ Warning ]: Model returned None."
                if logger:
                    logger.warning(msg)
                else:
                    print(msg)
                return None, None, sample_ids
            elif torch.isnan(final_output).any(): # Fixed variable name from channel_output to final_output
                msg = "[ Warning ]: NaN detected in model output"
                if logger:
                    logger.warning(msg)
                else:
                    print(msg)
                return None, None, sample_ids

        gt = batch_y  # batch_y: [batch_size, output_len, num_channels]

        return final_output, gt, sample_ids

    def test(self, savepath=None):
        """
        Validate the model on the validation dataset.
        
        Args:
            savepath: Optional path for saving results. If None and exp_manager is available,
                     uses exp_manager's checkpoint directory.
        """
        # Require exp_manager
        if self.exp_manager is None:
            # Fallback only if savepath provided, but logging will be issue
            if savepath is None:
                raise ValueError("exp_manager or savepath is required")
            # If we strictly require exp_manager, we should raise here.
            # But let's allow it if savepath is present, just no logging?
            # User said "require logger and console to exist".
            # So assuming self.exp_manager is present.
            if savepath is None:
                 raise ValueError("savepath must be provided if exp_manager is not available")
        
        if self.exp_manager:
            path = str(self.exp_manager.get_checkpoint_dir())
            logger = self.exp_manager.logger
            console = self.exp_manager.console
        else:
            # Legacy path? Or should we crash?
            path = savepath
            logger = None # Will crash if we use it without check
            console = None
            
        # Since user demanded no fallback and "require logger", I will enforce exp_manager
        if not self.exp_manager:
             raise ValueError("ExperimentManager is required for testing")
             
        path = str(self.exp_manager.get_checkpoint_dir()) if savepath is None else savepath
        logger = self.exp_manager.logger
        console = self.exp_manager.console
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        all_metrics_filename = "all_test_metrics.json"
        final_filename = "final_test_result.json"
        error_filename = "overall_error.json"
        
        criterion = self._select_criterion()
        loaders = self._get_data(flag='test')

        overall_running_loss = 0.0
        overall_total_samples = 0
        overall_error = 0
        
        if self.args.filtered_samples is not None:
            filtered_samples = json.load(open(self.args.filtered_samples))
            logger.info(f"[Info] Using filtered samples from: {self.args.filtered_samples}")

        self.model.eval()

        with torch.inference_mode():
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                # Outer progress bar for entities
                entity_task = progress.add_task("Testing entities", total=len(loaders))
                
                for info, loader in loaders.items():
                    info_running_loss = 0.0
                    info_total_samples = 0
                    info_error = 0
                    
                    if self.args.filtered_samples is not None:
                        filter_index = filtered_samples[info]
                    
                    # Inner progress bar for samples within current entity
                    sample_task = progress.add_task(f"  └─ {info}", total=len(loader))
                    
                    for i, iter_data in enumerate(loader):
                        if self.args.filtered_samples is not None and i in filter_index:

                            logger.info(f"[ Info ]: Testing on sample {i}, total: {len(filter_index)}")

                            output, gt, sample_ids = self._forward_step(iter_data)

                            if output is None and gt is None:
                                logger.warning(f"[ Warning ]: Model returned None for sample {i}. Skipping this sample.")
                                info_error += 1
                                overall_error += 1
                                progress.update(sample_task, advance=1)
                                continue
                            
                            current_batch_size = gt.size(0)
                            loss = criterion(output, gt)

                            info_running_loss += loss.item() * current_batch_size
                            info_total_samples += current_batch_size
                            
                            overall_running_loss += loss.item() * current_batch_size
                            overall_total_samples += current_batch_size
                        
                        elif self.args.filtered_samples is None:
                            # Verbose logging for every batch might be too much, consider removing or lowering level
                            # logger.info(f"[ Info ]: Testing on all samples") 

                            output, gt, sample_ids = self._forward_step(iter_data)

                            if output is None and gt is None:
                                logger.warning(f"[ Warning ]: Model returned None for sample {i}. Skipping this sample.")
                                info_error += 1
                                overall_error += 1
                                progress.update(sample_task, advance=1)
                                continue
                            
                            current_batch_size = gt.size(0)
                            loss = criterion(output, gt)

                            info_running_loss += loss.item() * current_batch_size
                            info_total_samples += current_batch_size
                            
                            overall_running_loss += loss.item() * current_batch_size
                            overall_total_samples += current_batch_size
                        
                        progress.update(sample_task, advance=1)
                    
                    # Remove the sample task when done with this entity
                    progress.remove_task(sample_task)
                    
                    if info_total_samples > 0:
                        info_epoch_loss = info_running_loss / info_total_samples
                        # Log to file only (not console) to avoid interfering with progress bar display
                        self.exp_manager.log_file_only(f"Test loss for {info}: {info_epoch_loss:.7f}")
                    else:
                        # Log to file only (not console) to avoid interfering with progress bar display
                        self.exp_manager.log_file_only(f"Test loss for {info}: N/A (no samples processed)", level=logging.WARNING)
                        info_epoch_loss = None

                    logger.info(f"Total Errors: {info_error}")
                    
                    # Update entity progress
                    progress.update(entity_task, advance=1)
            
            # Save metrics
            try:
                with open(os.path.join(path, all_metrics_filename), 'r') as f:
                    existing_data = json.load(f)
            except FileNotFoundError:
                existing_data = {}

            existing_data.update({info: info_epoch_loss})
            with open(os.path.join(path, all_metrics_filename), 'w') as f:
                json.dump(existing_data, f, indent=4)

        total_epoch_loss = overall_running_loss / overall_total_samples if overall_total_samples > 0 else 0.0
        logger.info(f"Overall test loss: {total_epoch_loss:.7f}")

        # Save results
        final_result = {"final_res": total_epoch_loss}
        with open(os.path.join(path, final_filename), 'w') as f:
            json.dump(final_result, f, indent=4)
        logger.info(f"Saved final result to {final_filename}")

        # Save overall errors
        overall_error_result = {"overall_error": overall_error}
        with open(os.path.join(path, error_filename), 'w') as f:
            json.dump(overall_error_result, f, indent=4)
        logger.info(f"Saved overall error info to {error_filename}")
