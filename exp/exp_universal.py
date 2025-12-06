from exp.exp_basic import Exp_Basic
from models import model_init

from utils.tools import EarlyStopping, adjust_learning_rate, general_move_to_device
from utils.per_sample_metrics import PerSampleMetricsTracker
from utils.loss_utils import compute_per_sample_loss, compute_per_sample_per_channel_loss

import torch
import torch.nn as nn

import os
import time
import warnings

import json

from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

warnings.filterwarnings('ignore')

class Experiment(Exp_Basic):
    """
    Main experiment orchestrator for universal time series forecasting models.
    
    This class extends Exp_Basic to provide a complete experiment framework for training,
    validating, and testing time series forecasting models. It supports both traditional
    and cross-modal forecasting approaches with comprehensive model management, training
    optimization, and evaluation capabilities.
    
    Key Features:
        - Multi-GPU training support via DataParallel
        - Early stopping and learning rate scheduling
        - Cross-modal data handling (text, events, etc.)
        - Comprehensive model checkpointing
        - Progress tracking with detailed metrics
    
    Args:
        args: Configuration object inherited from Exp_Basic containing all experiment
              parameters including model config, data config, training settings, etc.
    
    Example:
        ```python
        exp = Experiment(args)
        model = exp.train(setting='experiment_1')
        test_results = exp.test(setting='experiment_1')
        ```
    """
    def __init__(self, args, exp_manager=None):
        """
        Initialize Experiment.
        
        Args:
            args: Configuration object (dotdict or argparse-like)
            exp_manager: Optional ExperimentManager for experiment tracking
        """
        super(Experiment, self).__init__(args, exp_manager)
        
        # Initialize per-sample metrics tracker if enabled
        track_per_sample = getattr(args, 'track_per_sample', False)
        if track_per_sample:
            if exp_manager:
                output_dir = str(exp_manager.get_experiment_dir() / "metrics")
            else:
                # Fallback for backward compatibility
                output_dir = os.path.join(args.checkpoints, "..", "metrics")
                output_dir = os.path.abspath(output_dir)
            self.metrics_tracker = PerSampleMetricsTracker(output_dir)
        else:
            self.metrics_tracker = None
    
    def _get_channel_names(self, num_channels):
        """
        Get channel names from data config if available, otherwise use integer indices.
        
        Uses the 'target' list from data config as channel names, since target specifies
        which columns are used as channels. Falls back to explicit 'channel_names' if
        provided, then to integer indices.
        
        Args:
            num_channels: Number of channels/features
            
        Returns:
            List of channel identifiers (names or indices as strings)
        """
        # First try to get channel names from explicit channel_names field
        if hasattr(self.args, 'data_config') and hasattr(self.args.data_config, 'channel_names'):
            channel_names = self.args.data_config.channel_names
            if isinstance(channel_names, list) and len(channel_names) == num_channels:
                return [str(name) for name in channel_names]
        
        # Fallback: use 'target' list from data config as channel names
        # The target list specifies which columns are used, so they should be the channel names
        if hasattr(self.args, 'data_config') and hasattr(self.args.data_config, 'target'):
            target = self.args.data_config.target
            # Handle both list and single string cases
            if isinstance(target, list):
                if len(target) == num_channels:
                    return [str(name) for name in target]
            elif isinstance(target, str) and target != 'all':
                # Single target column
                if num_channels == 1:
                    return [str(target)]
        
        # Final fallback: use integer indices
        return [str(i) for i in range(num_channels)]

    def _build_model(self):
        """
        Constructs the forecasting model with optional multi-GPU support.
        
        Initializes the model using the model factory based on configuration,
        then wraps it with DataParallel if multi-GPU training is enabled.
        
        Returns:
            torch.nn.Module: Configured model (potentially wrapped with DataParallel)
        """
        model = model_init(self.args.model, self.args.model_config, self.args)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """
        Retrieves data loaders for the specified dataset split.
        
        Args:
            flag (str): Dataset split identifier ('train', 'val', 'test')
        
        Returns:
            DataLoader or dict: Data loader(s) for the specified split.
                               May return dict of loaders if multiple datasets are configured.
        """
        if flag == 'train':
            data_loader = self.data_provider.get_train(return_type='loader')
        elif flag == 'val':
            data_loader = self.data_provider.get_val(return_type='loader')
        elif flag == 'test':
            data_loader = self.data_provider.get_test(return_type='loader')

        return data_loader

    def _forward_step(self, iter):
        """
        Executes a single forward pass through the model with cross-modal data handling.
        
        This method processes a batch of data containing both time series and heterogeneous
        cross-modal information (text, events, etc.), performs device movement optimization,
        and executes the forward pass to generate predictions.
        
        Args:
            iter: Data batch tuple containing:
                - sample_ids: Sample identifiers for consistent tracking across models
                - batch_x: Input time series sequences
                - batch_y: Target time series sequences  
                - timestamp_x, timestamp_y: Corresponding timestamps
                - batch_x_hetero, batch_y_hetero: Heterogeneous data (text, events)
                - hetero_x_time, hetero_y_time: Heterogeneous data timestamps
                - hetero_general: General heterogeneous information
                - hetero_channel: Channel-specific heterogeneous information
        
        Returns:
            tuple: (predictions, ground_truth, sample_ids) all as torch tensors/arrays
                  - predictions: Model output for the prediction horizon
                  - ground_truth: True target values for comparison
                  - sample_ids: Sample identifiers for metrics tracking
        """
        # iteration: sample_ids, seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel

        sample_ids, batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = iter

        if hasattr(self.model, 'move_to_device'):
            batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = self.model.move_to_device(batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, self.device) # move only the ones needed to device according to model's definition to save VRAM
        else:
            # only move batch_x, batch_y to device for TSF models
            batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = general_move_to_device(batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, self.device)

        output = self.model(x=batch_x, historical_events =batch_x_hetero, news = batch_y_hetero, dataset_description=hetero_general, channel_description=hetero_channel)

        output = output[:, -self.args.output_len:, :]
        gt = batch_y

        return output, gt, sample_ids

    def _setup_training(self):
        """
        Initialize all components needed for training.
        
        Sets up data loaders, checkpoint paths, early stopping, optimizer,
        criterion, and per-sample tracking configuration.
        
        Returns:
            tuple: (train_loader, vali_loader, test_loader, path, early_stopping,
                   model_optim, criterion, track_per_sample)
        """
        # Get data loaders for all splits
        train_loader = self._get_data(flag='train')
        vali_loader = self._get_data(flag='val')
        test_loader = self._get_data(flag='test')
        
        # Release raw file buffer to save memory
        self.data_provider.data_buffer.clear()
        
        # Determine checkpoint directory path
        if self.exp_manager is not None:
            path = str(self.exp_manager.get_checkpoint_dir())
        else:
            # Fallback for backward compatibility
            path = self.args.checkpoints
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save args.json for backward compatibility (if not using exp_manager)
        if self.exp_manager is None:
            with open(os.path.join(path, 'args.json'), 'w') as f:
                json.dump(self.args.__dict__, f)
        
        # Initialize early stopping mechanism
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        # Initialize optimizer and loss function
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        # Check if per-sample tracking is enabled
        track_per_sample = getattr(self.args, 'track_per_sample', False)
        
        return train_loader, vali_loader, test_loader, path, early_stopping, \
               model_optim, criterion, track_per_sample

    def _format_eta(self, seconds):
        """
        Format ETA time in a human-readable format.
        
        Shows minutes when above 60 seconds, otherwise shows seconds.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            str: Formatted time string (e.g., "2.5m" or "45.2s")
        """
        if seconds >= 60:
            minutes = seconds / 60.0
            return f"{minutes:.1f}m"
        else:
            return f"{seconds:.1f}s"
    
    def _create_training_progress_bar(self, epoch, train_loader):
        """
        Create and configure progress bar for training epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            train_loader: Training data loader
            
        Returns:
            tuple: (progress, task, logger, console) for progress tracking
        """
        # Get logger and console from exp_manager
        logger = self.exp_manager.logger
        console = self.exp_manager.console
        
        # Define progress bar columns with metrics
        # Use a custom TextColumn that formats ETA dynamically
        progress_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("loss: {task.fields[loss]:.7f}"),
            TextColumn("•"),
            TextColumn("speed: {task.fields[speed]:.4f}s/iter"),
            TextColumn("•"),
            TextColumn("ETA: {task.fields[eta_formatted]}"),
            TimeElapsedColumn(),
        ]
        
        # Create progress bar and task
        progress = Progress(*progress_columns, console=console)
        task = progress.add_task(
            f"Epoch {epoch + 1}/{self.args.train_epochs}",
            total=len(train_loader),
            loss=0.0,
            speed=0.0,
            eta=0.0,
            eta_formatted="0.0s"
        )
        
        return progress, task, logger, console

    def _train_single_batch(self, iter, model_optim, criterion, track_per_sample):
        """
        Execute a single training batch: forward pass, loss, backward, update.
        
        Args:
            iter: Batch data tuple from data loader
            model_optim: Optimizer for parameter updates
            criterion: Loss function
            track_per_sample: Whether to track per-sample metrics
            
        Returns:
            tuple: (loss_value, batch_size, sample_ids, output, gt)
        """
        # Zero gradients before forward pass
        model_optim.zero_grad()
        
        # Forward pass through model
        output, gt, sample_ids = self._forward_step(iter)
        
        # Compute loss
        loss = criterion(output, gt)
        
        # Backward pass and optimizer step
        loss.backward()
        model_optim.step()
        
        # Get batch size for loss accumulation
        current_batch_size = gt.size(0)
        loss_value = loss.item()
        
        # Track per-sample metrics if enabled
        if track_per_sample and self.metrics_tracker:
            per_sample_loss = compute_per_sample_loss(output, gt, criterion)
            per_sample_per_channel_loss = compute_per_sample_per_channel_loss(output, gt, criterion)
            timestamps = iter[3]  # timestamp_x from batch tuple (index 3 after sample_ids)
            num_channels = per_sample_per_channel_loss.shape[1]
            channel_names = self._get_channel_names(num_channels)
            
            self.metrics_tracker.add_batch(
                epoch=self.current_epoch,
                split='train',
                entity_id=None,  # Will extract from sample_ids
                sample_ids=sample_ids,
                timestamps=timestamps[:, 0] if timestamps is not None else None,  # First timestamp of each sequence
                losses=per_sample_loss,
                channel_ids=channel_names,
                per_channel_losses=per_sample_per_channel_loss
            )
        
        return loss_value, current_batch_size, sample_ids, output, gt

    def _update_training_progress(self, progress, task, time_now, iter_count, 
                                   epoch, train_steps, current_iter, loss_value):
        """
        Update progress bar with current training metrics.
        
        Calculates speed and ETA, then updates the progress bar display.
        
        Args:
            progress: Rich Progress object
            task: Progress task ID
            time_now: Start time for current iteration
            iter_count: Current iteration count
            epoch: Current epoch number (0-indexed)
            train_steps: Total number of training steps per epoch
            current_iter: Current iteration index in epoch
            loss_value: Current batch loss value
            
        Returns:
            tuple: (updated_time_now, reset_iter_count) where iter_count is reset to 0
        """
        # Calculate speed (time per iteration)
        speed = (time.time() - time_now) / iter_count
        
        # Calculate estimated time remaining
        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - current_iter)
        
        # Format ETA for display (minutes if >= 60 seconds, otherwise seconds)
        eta_formatted = self._format_eta(left_time)
        
        # Update progress bar with current metrics
        progress.update(
            task,
            advance=1,
            loss=loss_value,
            speed=speed,
            eta=left_time,
            eta_formatted=eta_formatted
        )
        
        # Reset iteration counter and update time for next iteration
        return time.time(), 0

    def _train_single_epoch(self, epoch, train_loader, model_optim, criterion, 
                           track_per_sample, train_steps):
        """
        Execute a complete training epoch with progress tracking.
        
        Args:
            epoch: Current epoch number (0-indexed)
            train_loader: Training data loader
            model_optim: Optimizer for parameter updates
            criterion: Loss function
            track_per_sample: Whether to track per-sample metrics
            train_steps: Total number of training steps per epoch
            
        Returns:
            tuple: (train_loss, epoch_time_elapsed, total_samples)
        """
        # Set model to training mode and initialize tracking variables
        self.model.train()
        epoch_loss, total_samples, iter_count = 0.0, 0, 0
        epoch_time = time_now = time.time()
        
        # Mark epoch start in GPU monitor for epoch-level GPU utilization tracking
        if self.exp_manager and hasattr(self.exp_manager, 'gpu_monitor') and self.exp_manager.gpu_monitor:
            self.exp_manager.gpu_monitor.mark_epoch_start()
        
        # Create progress bar for this epoch (includes logger)
        progress, task, logger, console = self._create_training_progress_bar(epoch, train_loader)
        
        # Training loop over all batches
        with progress:
            for i, iter in enumerate(train_loader):
                iter_count += 1
                # Train on single batch and accumulate metrics
                loss_value, batch_size, _, _, _ = \
                    self._train_single_batch(iter, model_optim, criterion, track_per_sample)
                epoch_loss += loss_value * batch_size
                total_samples += batch_size
                # Update progress bar
                time_now, iter_count = self._update_training_progress(
                    progress, task, time_now, iter_count, epoch, train_steps, i, loss_value
                )
        
        # Calculate and log epoch statistics
        epoch_time_elapsed = time.time() - epoch_time
        train_loss = epoch_loss / total_samples if total_samples > 0 else 0.0
        logger.info(f"Epoch: {epoch + 1} cost time: {epoch_time_elapsed:.2f}s")
        
        return train_loss, epoch_time_elapsed, total_samples

    def _evaluate_epoch(self, epoch, train_loss, total_samples, vali_loader, 
                       test_loader, criterion, early_stopping, path, 
                       model_optim, track_per_sample, train_steps, epoch_time_elapsed=None):
        """
        Evaluate model on validation and test sets, log metrics, check early stopping.
        
        Args:
            epoch: Current epoch number (0-indexed)
            train_loss: Average training loss for this epoch
            total_samples: Total number of training samples processed
            vali_loader: Validation data loader
            test_loader: Test data loader
            criterion: Loss function
            early_stopping: EarlyStopping callback instance
            path: Checkpoint directory path
            model_optim: Optimizer (for learning rate logging)
            track_per_sample: Whether per-sample tracking is enabled
            train_steps: Total number of training steps per epoch
            epoch_time_elapsed: Time taken for the epoch in seconds (optional)
            
        Returns:
            tuple: (vali_loss, test_loss, should_stop) where should_stop is boolean
        """
        logger = self.exp_manager.logger
        
        # Run validation and testing (metrics accumulate with training metrics)
        vali_loss = self.vali(vali_loader, criterion)
        test_loss = self.test(test_loader, criterion)
        
        # Save all accumulated metrics (train + val + test) for this epoch together
        if track_per_sample and self.metrics_tracker:
            self.metrics_tracker.save_epoch(self.current_epoch)
        
        # Log epoch time to file only (not console)
        if epoch_time_elapsed is not None:
            self.exp_manager.log_file_only(f"Epoch {epoch + 1} completed in {epoch_time_elapsed:.2f}s")
        
        # Get and log full epoch GPU summary to file only (not console) using GpuMonitor's formatting method
        if self.exp_manager and hasattr(self.exp_manager, 'gpu_monitor') and self.exp_manager.gpu_monitor:
            gpu_summary_str = self.exp_manager.gpu_monitor.format_epoch_summary_for_log()
            if gpu_summary_str:
                self.exp_manager.log_file_only(f"Epoch {epoch + 1}: {gpu_summary_str}")
        
        # Log epoch results to console
        logger.info(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
        
        # Log metrics to ExperimentManager (and wandb if enabled)
        if self.exp_manager is not None:
            current_lr = model_optim.param_groups[0]['lr']
            self.exp_manager.log_metrics({
                'train_loss': train_loss,
                'val_loss': vali_loss,
                'test_loss': test_loss,
                'learning_rate': current_lr,
            }, step=epoch + 1)
        
        # Check early stopping condition
        early_stopping(vali_loss, self.model, path)
        should_stop = early_stopping.early_stop
        
        if should_stop:
            print("Early stopping")
            # Log final metrics if early stopping
            if self.exp_manager is not None:
                self.exp_manager.log_metrics({
                    'best_epoch': epoch + 1,
                    'final_train_loss': train_loss,
                    'final_val_loss': vali_loss,
                    'final_test_loss': test_loss,
                })
        else:
            # Adjust learning rate if not stopping
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        return vali_loss, test_loss, should_stop

    def _finalize_training(self, path, model_optim, train_loss, vali_loss, test_loss):
        """
        Load best model checkpoint and finalize experiment tracking.
        
        Args:
            path: Checkpoint directory path
            model_optim: Optimizer (for checkpoint saving)
            train_loss: Final training loss
            vali_loss: Final validation loss
            test_loss: Final test loss
        """
        # Load best model from checkpoint
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        
        # Save checkpoint to ExperimentManager if available
        if self.exp_manager is not None:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': model_optim.state_dict(),
                'epoch': self.args.train_epochs,
                'train_loss': train_loss,
                'val_loss': vali_loss,
                'test_loss': test_loss,
            }
            self.exp_manager.save_checkpoint(
                checkpoint, 
                filename="best_checkpoint.pth",
                is_best=True
            )
            
            # End experiment and finalize wandb
            final_metrics = {
                'best_epoch': self.args.train_epochs,
                'final_train_loss': train_loss,
                'final_val_loss': vali_loss,
                'final_test_loss': test_loss,
            }
            self.exp_manager.end_experiment(final_metrics)

    def train(self):
        """
        Executes the complete model training pipeline with comprehensive monitoring.
        
        This method orchestrates the entire training process including data loading,
        model optimization, validation monitoring, early stopping, and checkpointing.
        Supports advanced features like learning rate scheduling and progress tracking.
        
        Returns:
            torch.nn.Module: Trained model loaded from the best checkpoint
        
        Training Features:
            - Automatic early stopping based on validation loss
            - Learning rate scheduling with multiple strategies  
            - Progress bars with detailed metrics (loss, speed, time estimates)
            - Model checkpointing with best model preservation
            - Memory optimization with buffer clearing
            - Comprehensive logging of training/validation/test performance
        
        Example:
            ```python
            exp = Experiment(args, exp_manager=exp_manager)
            trained_model = exp.train()
            ```
        """
        # Setup phase: initialize all training components
        train_loader, vali_loader, test_loader, path, early_stopping, \
            model_optim, criterion, track_per_sample = self._setup_training()
        
        train_steps = len(train_loader)
        
        # Training loop over all epochs
        for epoch in range(self.args.train_epochs):
            self.current_epoch = epoch + 1
            
            # Train one complete epoch
            train_loss, epoch_time, total_samples = self._train_single_epoch(
                epoch, train_loader, model_optim, criterion, 
                track_per_sample, train_steps
            )
            
            # Evaluate epoch: validation, testing, logging, early stopping check
            vali_loss, test_loss, should_stop = self._evaluate_epoch(
                epoch, train_loss, total_samples, vali_loader, test_loader,
                criterion, early_stopping, path, model_optim, track_per_sample, train_steps, epoch_time
            )
            
            # Break if early stopping triggered
            if should_stop:
                break
        
        # Finalize training: load best model and save final checkpoint
        self._finalize_training(path, model_optim, train_loss, vali_loss, test_loss)
        
        return self.model

    def vali(self, loader, criterion):
        """
        Validates the model on the validation dataset and computes average loss.
        
        Performs inference on the validation set without gradient computation
        to evaluate model performance during training. Used for early stopping
        and learning rate scheduling decisions.
        
        Args:
            loader: Validation data loader (can be single loader or dict of loaders)
            criterion: Loss function for evaluation
        
        Returns:
            float: Average validation loss across all validation samples
        """
        running_loss = 0.0
        total_samples = 0

        self.model.eval()
        
        # Check if per-sample tracking is enabled
        track_per_sample = getattr(self.args, 'track_per_sample', False)

        console = self.exp_manager.get_console() if self.exp_manager else None
        
        with torch.inference_mode():
            with torch.no_grad():
                # NOTE: Only handling standard single DataLoader for validation for now.
                # Per-sample tracking for validation requires handling dict loaders here
                # similar to test(), but is deferred to avoid complexity.
                if console:
                    with Progress(
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        TimeElapsedColumn(),
                        console=console
                    ) as progress:
                        task = progress.add_task("Validating...", total=len(loader))
                        for i, iter_data in enumerate(loader):
                            output, gt, sample_ids = self._forward_step(iter_data)
                            current_batch_size = gt.size(0)
                            loss = criterion(output, gt)
                            running_loss += loss.item() * current_batch_size
                            total_samples += current_batch_size
                            
                            # Track per-sample metrics if enabled
                            if track_per_sample and self.metrics_tracker:
                                per_sample_loss = compute_per_sample_loss(output, gt, criterion)
                                per_sample_per_channel_loss = compute_per_sample_per_channel_loss(output, gt, criterion)
                                timestamps = iter_data[3]  # timestamp_x from batch tuple (index 3 after sample_ids)
                                num_channels = per_sample_per_channel_loss.shape[1]
                                channel_names = self._get_channel_names(num_channels)
                                
                                self.metrics_tracker.add_batch(
                                    epoch=self.current_epoch,  # Associate validation metrics with current epoch
                                    split='val',
                                    entity_id=None,  # Will extract from sample_ids
                                    sample_ids=sample_ids,
                                    timestamps=timestamps[:, 0] if timestamps is not None else None,  # First timestamp of each sequence
                                    losses=per_sample_loss,
                                    channel_ids=channel_names,
                                    per_channel_losses=per_sample_per_channel_loss
                                )
                            
                            progress.update(task, advance=1)
                else:
                    for i, iter_data in enumerate(loader):
                        output, gt, sample_ids = self._forward_step(iter_data)
                        current_batch_size = gt.size(0)
                        loss = criterion(output, gt)
                        running_loss += loss.item() * current_batch_size
                        total_samples += current_batch_size
                        
                        # Track per-sample metrics if enabled
                        if track_per_sample and self.metrics_tracker:
                            per_sample_loss = compute_per_sample_loss(output, gt, criterion)
                            per_sample_per_channel_loss = compute_per_sample_per_channel_loss(output, gt, criterion)
                            timestamps = iter_data[3]  # timestamp_x from batch tuple (index 3 after sample_ids)
                            num_channels = per_sample_per_channel_loss.shape[1]
                            channel_names = self._get_channel_names(num_channels)
                            
                            self.metrics_tracker.add_batch(
                                epoch=self.current_epoch,  # Associate validation metrics with current epoch
                                split='val',
                                entity_id=None,  # Will extract from sample_ids
                                sample_ids=sample_ids,
                                timestamps=timestamps[:, 0] if timestamps is not None else None,  # First timestamp of each sequence
                                losses=per_sample_loss,
                                channel_ids=channel_names,
                                per_channel_losses=per_sample_per_channel_loss
                            )

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        
        self.model.train()
        return epoch_loss
    
    def test(self, loaders, criterion, valinum='full'):
        """
        Validate the model on the validation dataset.
        """
        overall_running_loss = 0.0
        overall_total_samples = 0
        self.model.eval()
        
        # Determine tracking mode
        track_per_sample = getattr(self.args, 'track_per_sample', False)

        console = self.exp_manager.console
        
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
                    
                    # Inner progress bar for samples within current entity
                    sample_task = progress.add_task(f"  └─ {info}", total=len(loader))
                    
                    for i, iter_data in enumerate(loader):
                        output, gt, sample_ids = self._forward_step(iter_data)
                        current_batch_size = gt.size(0)
                        loss = criterion(output, gt)
                        info_running_loss += loss.item() * current_batch_size
                        info_total_samples += current_batch_size
                        overall_running_loss += loss.item() * current_batch_size
                        overall_total_samples += current_batch_size
                        
                        # Track per-sample metrics if enabled
                        if track_per_sample and self.metrics_tracker:
                            per_sample_loss = compute_per_sample_loss(output, gt, criterion)
                            per_sample_per_channel_loss = compute_per_sample_per_channel_loss(output, gt, criterion)
                            timestamps = iter_data[3]  # timestamp_x from batch tuple (index 3 after sample_ids)
                            num_channels = per_sample_per_channel_loss.shape[1]
                            channel_names = self._get_channel_names(num_channels)
                            
                            self.metrics_tracker.add_batch(
                                epoch=self.current_epoch,  # Associate test metrics with current epoch
                                split='test',
                                entity_id=info,  # Entity ID known for test
                                sample_ids=sample_ids,
                                timestamps=timestamps[:, 0] if timestamps is not None else None,  # First timestamp of each sequence
                                losses=per_sample_loss,
                                channel_ids=channel_names,
                                per_channel_losses=per_sample_per_channel_loss
                            )

                        progress.update(sample_task, advance=1)
                    
                    # Remove the sample task when done with this entity
                    progress.remove_task(sample_task)
                    
                    if valinum != 'full':
                        # The original logic `i == valinum` processes `valinum` batches (indices 0 to valinum-1).
                        if i + 1 >= valinum:
                            break
                    
                    # Update entity progress
                    progress.update(entity_task, advance=1)
                    
                    # Note: Per-entity test loss is tracked in per-sample metrics (parquet files)
                    # No need to log individual entity losses here

        total_epoch_loss = overall_running_loss / overall_total_samples if overall_total_samples > 0 else 0.0
        
        self.model.train()
        return total_epoch_loss
