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
from utils.progress_utils import ProgressWrapper

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
        
        NOTE: torch.compile is applied in Exp_Basic after moving to device,
        not here. For multi-GPU, compilation happens per-GPU worker automatically.
        
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
        # iteration: sample_ids, seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, x_time_features, y_time_features

        sample_ids, batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, x_time_features, y_time_features = iter

        if hasattr(self.model, 'move_to_device'):
            batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = self.model.move_to_device(batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, self.device) # move only the ones needed to device according to model's definition to save VRAM
        else:
            # only move batch_x, batch_y to device for TSF models
            batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = general_move_to_device(batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, self.device)
        
        # Handle time features for models that require them (FEDformer, Informer, etc.)
        x_mark_enc, x_mark_dec = self._prepare_temporal_marks(
            x_time_features, y_time_features, batch_x.shape[0]
        )
        
        # Call model with appropriate interface based on temporal marks availability
        if x_mark_enc is not None and x_mark_dec is not None:
            output = self.model(x=batch_x, x_mark_enc=x_mark_enc, x_mark_dec=x_mark_dec, 
                              historical_events=batch_x_hetero, news=batch_y_hetero, 
                              dataset_description=hetero_general, channel_description=hetero_channel)
        else:
            # Standard framework interface
            output = self.model(x=batch_x, historical_events =batch_x_hetero, news = batch_y_hetero, dataset_description=hetero_general, channel_description=hetero_channel)

        output = output[:, -self.args.output_len:, :]
        gt = batch_y

        return output, gt, sample_ids

    def _prepare_temporal_marks(self, x_time_features, y_time_features, batch_size):
        """
        Prepare temporal marks (x_mark_enc, x_mark_dec) for models that require them.
        
        Converts time feature arrays (numpy or tensor) to tensors, moves them to device, and constructs
        the proper format for decoder temporal marks (label_len + pred_len) for models like
        FEDformer, Informer, and Autoformer.
        
        Args:
            x_time_features: Encoder time features [B, seq_len, time_features] as numpy array, 
                            torch.Tensor, or None. Note: DataLoader's default_collate may convert
                            numpy arrays to tensors automatically.
            y_time_features: Target time features [B, pred_len, time_features] as numpy array,
                            torch.Tensor, or None
            batch_size: Batch size for validation (not currently used but available for future use)
        
        Returns:
            tuple: (x_mark_enc, x_mark_dec) as torch.Tensor or (None, None) if not applicable
                   - x_mark_enc: [B, seq_len, time_features] encoder temporal marks
                   - x_mark_dec: [B, label_len + pred_len, time_features] decoder temporal marks
        """
        # Check if model requires temporal marks
        # Direct models: fedformer, informer, autoformer
        # Wrapped models: MMTSFlib with FEDformer/Informer/Autoformer as unimodal_model_type
        model_name = getattr(self.args, 'model', '').lower()
        
        # Check if this is a model that directly needs temporal marks
        needs_temporal_marks = model_name in ['fedformer', 'informer', 'autoformer']
        
        # Check if MMTSFlib is wrapping a model that needs temporal marks
        if not needs_temporal_marks and model_name == 'mmtsflib':
            # Check if MMTSFlib has a unimodal_model_type that requires temporal marks
            if hasattr(self.model, 'unimodal_model_type'):
                unimodal_type = self.model.unimodal_model_type
                if unimodal_type in ['FEDformer', 'Informer', 'Autoformer']:
                    needs_temporal_marks = True
            # Also check args/config for unimodal_model_type
            elif hasattr(self.args, 'unimodal_model_type'):
                unimodal_type = getattr(self.args, 'unimodal_model_type', '').lower()
                if unimodal_type in ['fedformer', 'informer', 'autoformer']:
                    needs_temporal_marks = True
        
        if not needs_temporal_marks:
            return None, None
        
        # If time features are not provided, return None (model will handle error)
        if x_time_features is None or y_time_features is None:
            return None, None
        
        # Convert to tensors and move to device
        # DataLoader's default_collate may already convert numpy arrays to tensors
        # Handle both cases: numpy arrays and tensors
        if isinstance(x_time_features, torch.Tensor):
            x_mark_enc = x_time_features.float().to(self.device)
        else:
            x_mark_enc = torch.from_numpy(x_time_features).float().to(self.device)
        
        if isinstance(y_time_features, torch.Tensor):
            x_mark_dec = y_time_features.float().to(self.device)
        else:
            x_mark_dec = torch.from_numpy(y_time_features).float().to(self.device)
        
        # For FEDformer and similar models, x_mark_dec needs to cover label_len + pred_len
        # y_time_features only covers pred_len, so we need to extend it by taking
        # the last label_len timestamps from x_mark_enc and prepending to x_mark_dec
        # Get label_len from model, args, or calculate default (seq_len // 2)
        label_len = None
        
        # Try to get label_len from model (check wrapped model if MMTSFlib)
        if hasattr(self.model, 'label_len') and self.model.label_len is not None:
            label_len = self.model.label_len
        # Check if MMTSFlib wraps a model with label_len
        elif hasattr(self.model, 'ts_model') and hasattr(self.model.ts_model, 'label_len') and self.model.ts_model.label_len is not None:
            label_len = self.model.ts_model.label_len
        # Check args/config
        elif hasattr(self.args, 'label_len') and self.args.label_len is not None:
            label_len = self.args.label_len
        # Try to get seq_len from model (check wrapped model if MMTSFlib)
        elif hasattr(self.model, 'seq_len') and self.model.seq_len is not None:
            # Default: label_len = seq_len // 2 (matching FEDformer/Informer default)
            label_len = self.model.seq_len // 2
        # Check wrapped model's seq_len if MMTSFlib
        elif hasattr(self.model, 'ts_model') and hasattr(self.model.ts_model, 'seq_len') and self.model.ts_model.seq_len is not None:
            label_len = self.model.ts_model.seq_len // 2
        # Check args/config
        elif hasattr(self.args, 'seq_len') and self.args.seq_len is not None:
            label_len = self.args.seq_len // 2
        elif hasattr(self.args, 'input_len') and self.args.input_len is not None:
            # Some configs use input_len instead of seq_len
            label_len = self.args.input_len // 2
        
        # Only extend x_mark_dec if we have a valid label_len
        if label_len is not None and label_len > 0:
            # Take last label_len time features from encoder
            x_mark_dec_label = x_mark_enc[:, -label_len:, :]
            # Concatenate with prediction horizon time features
            x_mark_dec = torch.cat([x_mark_dec_label, x_mark_dec], dim=1)
        
        return x_mark_enc, x_mark_dec

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
        # Note: DataLoader creation can take 10-30s with many workers
        print("[ info ] Initializing data loaders (this may take a moment)...")
        train_loader = self._get_data(flag='train')
        vali_loader = self._get_data(flag='val')
        test_loader = self._get_data(flag='test')
        print("[ info ] All data loaders initialized successfully")
        
        # Release raw file buffer to save memory
        self.data_provider.data_buffer.clear()
        if os.environ.get('FIDEL_DEBUG', '0') == '1':
            print("[ info ] Buffer cleared")
        
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
            tuple: (loss_value, batch_size, sample_ids, output, gt, grad_norm)
        """
        # Zero gradients before forward pass
        model_optim.zero_grad()

        # Forward pass through model
        output, gt, sample_ids = self._forward_step(iter)

        # Compute loss
        loss = criterion(output, gt)

        # Backward pass
        loss.backward()


        # Apply gradient clipping for TimeLLM to prevent gradient explosion
        if self.args.model == 'TimeLLM':
            # Read grad_clip_max_norm from model config, default to 1.0
            grad_clip_max_norm = getattr(self.args.model_config, 'grad_clip_max_norm', 1.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=grad_clip_max_norm
            ).item()
        # Other models: just calculate gradient norm before optimizer step (no clipping)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=float('inf')  # Don't actually clip, just compute norm
            ).item()

        model_optim.step()
        
        # Get batch size for loss accumulation
        current_batch_size = gt.size(0)

        # Check for NaN/Inf in training loss (per-batch detection)
        # Check tensor before calling .item() to catch NaN early
        # Use .cpu() to ensure synchronization if loss is on GPU
        if torch.isnan(loss).any().item() or torch.isinf(loss).any().item():
            loss_value = loss.item()
            error_msg = f"Training loss became NaN/Inf at epoch {self.current_epoch}, batch {getattr(self, '_current_batch_idx', 'unknown')}: {loss_value}"
            logger = self.exp_manager.logger if self.exp_manager else None
            if logger:
                logger.error(error_msg)
            print(f"\n[ CRITICAL ] {error_msg}")  # Print to console to ensure visibility
            raise ValueError(error_msg)

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

        return loss_value, current_batch_size, sample_ids, output, gt, grad_norm

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
                           track_per_sample, train_steps, vali_loader=None):
        """
        Execute a complete training epoch with progress tracking.

        Args:
            epoch: Current epoch number (0-indexed)
            train_loader: Training data loader
            model_optim: Optimizer for parameter updates
            criterion: Loss function
            track_per_sample: Whether to track per-sample metrics
            train_steps: Total number of training steps per epoch
            vali_loader: Validation data loader (optional, for batch-level validation)

        Returns:
            tuple: (train_loss, epoch_time_elapsed, total_samples)
        """
        # Set model to training mode and initialize tracking variables
        self.model.train()
        epoch_loss, total_samples, iter_count = 0.0, 0, 0
        epoch_time = time_now = time.time()

        # Batch logging configuration
        batch_log_interval = getattr(
            self.args.wandb, 'batch_log_interval', 10
        ) if hasattr(self.args, 'wandb') else 10
        
        # Validation batch interval configuration (must be multiple of batch_log_interval)
        validate_every_n_batches = None
        if hasattr(self.args, 'wandb') and hasattr(self.args.wandb, 'validate_every_n_batches'):
            validate_every_n_batches = getattr(self.args.wandb, 'validate_every_n_batches', None)
        
        total_batches = len(train_loader)

        # Mark epoch start in GPU monitor for epoch-level GPU utilization tracking
        if self.exp_manager and hasattr(self.exp_manager, 'gpu_monitor') and self.exp_manager.gpu_monitor:
            self.exp_manager.gpu_monitor.mark_epoch_start()

        # Create progress bar for this epoch (includes logger)
        progress, task, logger, console = self._create_training_progress_bar(epoch, train_loader)

        # Training loop over all batches
        with progress:
            for i, iter in enumerate(train_loader):
                iter_count += 1
                # Store current batch index for error reporting
                self._current_batch_idx = i
                # Train on single batch and accumulate metrics
                loss_value, batch_size, _, _, _, grad_norm = \
                    self._train_single_batch(iter, model_optim, criterion, track_per_sample)
                epoch_loss += loss_value * batch_size
                total_samples += batch_size

                # Log batch loss and gradient norm to wandb every N batches
                if self.exp_manager and self.exp_manager.wandb_run:
                    should_log = (i % batch_log_interval == 0) or (i == total_batches - 1)

                    if should_log:
                        global_step = epoch * total_batches + i
                        batch_metrics = {
                            'batch_loss': loss_value,
                            'batch_grad_norm': grad_norm
                        }
                        
                        # Optionally run validation at batch level if configured
                        # Validation only runs when batch logging occurs and batch index matches validation interval
                        if (validate_every_n_batches is not None and 
                            vali_loader is not None and 
                            i % validate_every_n_batches == 0):
                            # Run validation and log loss at batch level
                            vali_loss = self.vali(vali_loader, criterion)
                            batch_metrics['val_loss'] = vali_loss
                            # Set model back to training mode after validation
                            self.model.train()
                        
                        self.exp_manager.log_batch_metrics(batch_metrics, batch_step=global_step)

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
        
        # Check if test evaluation is enabled during training
        evaluate_test = getattr(self.args, 'evaluate_test_during_training', False)
        
        # Run validation (always)
        vali_loss = self.vali(vali_loader, criterion)
        
        # Conditionally run test evaluation
        if evaluate_test:
            test_loss = self.test(test_loader, criterion)
        else:
            test_loss = None  # Test held out until final evaluation
        
        # Save all accumulated metrics (train + val + test if enabled) for this epoch together
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
        if test_loss is not None:
            logger.info(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
        else:
            logger.info(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")
        
        # Log metrics to ExperimentManager (and wandb if enabled)
        if self.exp_manager is not None:
            current_lr = model_optim.param_groups[0]['lr']
            metrics_dict = {
                'train_loss': train_loss,
                'val_loss': vali_loss,
                'learning_rate': current_lr,
            }
            # Only log test loss if it was computed
            if test_loss is not None:
                metrics_dict['test_loss'] = test_loss
            self.exp_manager.log_epoch_metrics(metrics_dict, epoch=epoch + 1)
        
        # Check early stopping condition (only uses validation loss, not test)
        # Pass epoch+1 (1-indexed) so EarlyStopping tracks best_epoch
        early_stopping(vali_loss, self.model, path, epoch=epoch + 1, train_loss=train_loss)
        should_stop = early_stopping.early_stop

        if should_stop:
            print("Early stopping")
            # Log final metrics if early stopping
            if self.exp_manager is not None:
                # Set completion reason for sweep system
                self.exp_manager.set_completion_reason("early_stopping")
                final_metrics = {
                    'best_epoch': early_stopping.best_epoch,
                    'final_train_loss': train_loss,
                    'final_val_loss': vali_loss,
                }
                # Only log test loss if it was computed
                if test_loss is not None:
                    final_metrics['final_test_loss'] = test_loss
                self.exp_manager.log_metrics(final_metrics)
        else:
            # Adjust learning rate if not stopping
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        return vali_loss, test_loss, should_stop

    def _load_resume_checkpoint(self, model_optim) -> int:
        """
        Load checkpoint if resuming from a previous job.
        
        Args:
            model_optim: Optimizer instance (for loading optimizer state)
        
        Returns:
            Start epoch (0-indexed) to begin training from
        """
        if not self.exp_manager:
            return 0
        
        resume_info = self.exp_manager.get_resume_info()
        if not resume_info:
            return 0
        
        start_epoch = resume_info["start_epoch"] - 1  # Convert to 0-indexed
        checkpoint_path = resume_info.get("checkpoint_path")
        
        # Resolve checkpoint path relative to experiment directory if it's a relative path
        if checkpoint_path:
            if not os.path.isabs(checkpoint_path):
                # Relative path - resolve relative to experiment directory
                exp_dir = self.exp_manager.get_experiment_dir()
                checkpoint_path = str(exp_dir / checkpoint_path)
        
        # Load checkpoint if available
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.exp_manager.logger.info(f"Loading checkpoint from: {checkpoint_path}")
            try:
                import torch
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Extract model state dict
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    # Try loading directly (for Lightning checkpoints converted to PyTorch format)
                    state_dict = checkpoint
                
                # Handle torch.compile checkpoint loading
                # Check if model is compiled and adjust checkpoint keys accordingly
                is_data_parallel = hasattr(self.model, 'module')
                model_is_compiled = hasattr(self.model, '_orig_mod') or (
                    is_data_parallel and hasattr(self.model.module, '_orig_mod')
                )
                checkpoint_has_prefix = any(
                    key.startswith('_orig_mod.') or key.startswith('module._orig_mod.') 
                    for key in state_dict.keys()
                )
                
                if checkpoint_has_prefix and not model_is_compiled:
                    # Checkpoint has prefix but model is not compiled - strip prefix
                    self.exp_manager.logger.info("Detected torch.compile checkpoint - stripping '_orig_mod.' prefix from state dict keys")
                    # Strip both possible prefixes
                    state_dict = {
                        key.replace('module._orig_mod.', 'module.' if is_data_parallel else '').replace('_orig_mod.', ''): value 
                        for key, value in state_dict.items()
                    }
                elif not checkpoint_has_prefix and model_is_compiled:
                    # Checkpoint doesn't have prefix but model is compiled - add appropriate prefix
                    self.exp_manager.logger.info("Model is compiled but checkpoint lacks prefix - adding appropriate prefix to checkpoint keys")
                    if is_data_parallel:
                        # DataParallel + compiled: need 'module._orig_mod.' prefix
                        state_dict = {f'module._orig_mod.{key}': value for key, value in state_dict.items()}
                    else:
                        # Compiled but not DataParallel: need '_orig_mod.' prefix
                        state_dict = {f'_orig_mod.{key}': value for key, value in state_dict.items()}
                
                # Load model state
                self.model.load_state_dict(state_dict)
                
                # Load optimizer state if available
                if 'optimizer_state_dict' in checkpoint and model_optim is not None:
                    model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
                
                self.exp_manager.logger.info(f"Resumed from checkpoint, starting at epoch {start_epoch + 1}")
            except Exception as e:
                self.exp_manager.logger.warning(f"Failed to load checkpoint: {e}. Starting from scratch.")
                return 0
        else:
            self.exp_manager.logger.info(f"Resuming from epoch {start_epoch + 1} (no checkpoint to load)")
        
        return start_epoch
    
    def _update_job_history_after_epoch(self, path: str) -> bool:
        """
        Update job history after each epoch completes.
        
        This method updates the epoch progress and checks for external stop signals
        (e.g., Hyperband pruning). Returns True if training should stop.
        
        Args:
            path: Checkpoint directory path
            
        Returns:
            True if training should stop (e.g., Hyperband pruning), False otherwise
        """
        if not self.exp_manager:
            return False
        
        # Get checkpoint path if available (store as absolute path)
        checkpoint_path = None
        best_checkpoint = os.path.join(path, 'checkpoint.pth')
        if os.path.exists(best_checkpoint):
            # Convert to absolute path for storage
            checkpoint_path = os.path.abspath(best_checkpoint)
        
        # Update epoch and check for stop signals (Hyperband, etc.)
        return self.exp_manager.update_current_epoch(
            epoch=self.current_epoch,
            checkpoint_path=checkpoint_path
        )
    
    def _register_job_end(self, path: str, train_loss: float, vali_loss: float) -> None:
        """
        Register job end in job history.
        
        Args:
            path: Checkpoint directory path
            train_loss: Final training loss
            vali_loss: Final validation loss
        """
        if not self.exp_manager:
            return
        
        final_checkpoint = os.path.join(path, 'checkpoint.pth')
        if os.path.exists(final_checkpoint):
            # Convert to absolute path for storage
            checkpoint_path = os.path.abspath(final_checkpoint)
        else:
            checkpoint_path = None
        
        # Determine job status (completed if reached end, otherwise timeout)
        final_epoch = self.current_epoch
        if final_epoch >= self.args.train_epochs:
            status = "completed"
        else:
            status = "timeout"  # Job likely timed out before completing all epochs
        
        self.exp_manager.register_job_end(
            end_epoch=final_epoch,
            status=status,
            checkpoint_path=checkpoint_path,
            final_train_loss=train_loss,
            final_val_loss=vali_loss
        )
    
    def _finalize_training(self, path, model_optim, train_loss, vali_loss, test_loss,
                          best_epoch=None, test_metrics=None):
        """
        Load best model checkpoint and finalize experiment tracking.

        Loads the best checkpoint (checkpoint.pth) which contains model state and metrics
        from the best epoch. Uses these saved metrics for final reporting to ensure
        consistency across train/val/test splits (all from best epoch).

        Args:
            path: Checkpoint directory path
            model_optim: Optimizer (unused, kept for compatibility)
            train_loss: Final training loss (unused - loaded from checkpoint)
            vali_loss: Final validation loss (unused - loaded from checkpoint)
            test_loss: Final test loss (unused - loaded from checkpoint)
            best_epoch: Best epoch from early stopping (loaded from checkpoint if None)
            test_metrics: Test metrics dict from final evaluation (default: None)
        """
        # Load best model checkpoint (saved by EarlyStopping)
        best_model_path = os.path.join(path, 'checkpoint.pth')
        checkpoint = torch.load(best_model_path)

        # Extract metrics from checkpoint
        # Checkpoint structure: {model_state_dict, epoch, train_loss, val_loss}
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format: checkpoint is a dict with metadata
            model_state_dict = checkpoint['model_state_dict']
            best_epoch = checkpoint.get('epoch', best_epoch or self.args.train_epochs)
            train_loss = checkpoint.get('train_loss', train_loss)
            vali_loss = checkpoint.get('val_loss', vali_loss)
        else:
            # Old format: checkpoint is bare state_dict (backward compatibility)
            model_state_dict = checkpoint
            if best_epoch is None:
                best_epoch = self.args.train_epochs

        # Handle torch.compile checkpoint loading
        # Check if model is compiled and adjust checkpoint keys accordingly
        is_data_parallel = hasattr(self.model, 'module')
        model_is_compiled = hasattr(self.model, '_orig_mod') or (
            is_data_parallel and hasattr(self.model.module, '_orig_mod')
        )
        checkpoint_has_prefix = isinstance(model_state_dict, dict) and any(
            key.startswith('_orig_mod.') or key.startswith('module._orig_mod.') for key in model_state_dict.keys()
        )

        if checkpoint_has_prefix and not model_is_compiled:
            # Checkpoint has prefix but model is not compiled - strip prefix
            if self.exp_manager:
                self.exp_manager.logger.info("Detected torch.compile checkpoint - stripping '_orig_mod.' prefix from state dict keys")
            # Strip both possible prefixes
            model_state_dict = {
                key.replace('module._orig_mod.', 'module.' if is_data_parallel else '').replace('_orig_mod.', ''): value
                for key, value in model_state_dict.items()
            }
        elif not checkpoint_has_prefix and model_is_compiled:
            # Checkpoint doesn't have prefix but model is compiled - add appropriate prefix
            if self.exp_manager:
                self.exp_manager.logger.info("Model is compiled but checkpoint lacks prefix - adding appropriate prefix to checkpoint keys")
            if is_data_parallel:
                # DataParallel + compiled: need 'module._orig_mod.' prefix
                model_state_dict = {f'module._orig_mod.{key}': value for key, value in model_state_dict.items()}
            else:
                # Compiled but not DataParallel: need '_orig_mod.' prefix
                model_state_dict = {f'_orig_mod.{key}': value for key, value in model_state_dict.items()}

        self.model.load_state_dict(model_state_dict)

        # Optionally update checkpoint with test metrics if available
        # This makes checkpoint.pth a complete record of all metrics from best epoch
        if test_metrics is not None and isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Update the loaded checkpoint dict with test metrics
            checkpoint['test_mse_normalized'] = test_metrics['overall']['mse_normalized']
            checkpoint['test_mae_normalized'] = test_metrics['overall']['mae_normalized']
            checkpoint['test_num_samples'] = test_metrics['overall']['num_samples']

            if 'mse_denormalized' in test_metrics['overall']:
                checkpoint['test_mse_denormalized'] = test_metrics['overall']['mse_denormalized']
                checkpoint['test_mae_denormalized'] = test_metrics['overall']['mae_denormalized']

            # Save updated checkpoint back to file
            try:
                torch.save(checkpoint, best_model_path)
                if self.exp_manager:
                    self.exp_manager.logger.info(f"Updated checkpoint with test metrics at {best_model_path}")
            except Exception as e:
                if self.exp_manager:
                    self.exp_manager.logger.warning(f"Failed to update checkpoint with test metrics: {e}")

        # End experiment and finalize wandb
        if self.exp_manager is not None:
            # All metrics are from best epoch for consistency
            # train_loss and vali_loss loaded from checkpoint (saved by EarlyStopping)
            # test metrics come from final_test_evaluation (if provided)
            final_metrics = {
                'best_epoch': best_epoch,
                'final_train_loss': train_loss,
                'final_val_loss': vali_loss,
            }

            # Add test metrics if available
            if test_metrics is not None:
                final_metrics.update({
                    'test/mse_normalized': test_metrics['overall']['mse_normalized'],
                    'test/mae_normalized': test_metrics['overall']['mae_normalized'],
                    'test/num_samples': test_metrics['overall']['num_samples'],
                })

                # Add denormalized metrics if available
                if 'mse_denormalized' in test_metrics['overall']:
                    final_metrics['test/mse_denormalized'] = test_metrics['overall']['mse_denormalized']
                    final_metrics['test/mae_denormalized'] = test_metrics['overall']['mae_denormalized']

            self.exp_manager.end_experiment(final_metrics)

    def _run_final_test_evaluation(self, test_loader, best_epoch, checkpoint_path):
        """
        Run comprehensive test evaluation at end of training.

        Uses the same forward pass logic as self.test() but computes comprehensive
        metrics (MSE/MAE normalized and denormalized) for all test subsets.
        Saves results to metrics/test_results.json and logs to WandB.

        Args:
            test_loader: Test data loader (dict of loaders by entity)
            best_epoch: The epoch number of the best checkpoint
            checkpoint_path: Path to the checkpoint directory

        Returns:
            dict: Test metrics with overall, per_entity, and metadata sections,
                  or None if evaluation fails or no test data
        """
        from utils.tools import format_test_results
        from pathlib import Path

        if test_loader is None:
            if self.exp_manager:
                self.exp_manager.logger.info("No test data available - skipping final test evaluation")
            return None

        self.model.eval()

        # Collect per-entity metrics
        per_entity_metrics = {}

        # Ensure test_loader is a dict
        if not isinstance(test_loader, dict):
            test_loader = {'default': test_loader}

        total_entities = len(test_loader)

        # Get console for progress bar
        console = self.exp_manager.console if self.exp_manager else None

        with torch.inference_mode():
            with ProgressWrapper(console, show_metrics=True) as progress:
                # Outer progress bar for entities
                entity_task = progress.add_task("Final test evaluation", total=len(test_loader), metrics="")

                for idx, (entity_id, loader) in enumerate(test_loader.items()):
                    # Inner progress bar for batches within entity
                    sample_task = progress.add_task(f"  └─ {entity_id}", total=len(loader), metrics="")

                    # Get dataset from loader for scaler access
                    dataset = loader.dataset if hasattr(loader, 'dataset') else None
                    scaler = getattr(dataset, 'scaler', None) if dataset else None

                    # Accumulators for this entity
                    total_mse_norm = 0.0
                    total_mae_norm = 0.0
                    total_mse_denorm = 0.0
                    total_mae_denorm = 0.0
                    num_samples = 0

                    # Tracking for progress bar metrics
                    iter_count = 0
                    time_now = time.time()

                    try:
                        for iter_data in loader:
                            iter_count += 1
                            # Use the same forward pass as training/validation
                            output, gt, _ = self._forward_step(iter_data)
                            batch_size = gt.size(0)

                            # Compute normalized metrics (MSE and MAE)
                            mse = nn.MSELoss()(output, gt)
                            mae = nn.L1Loss()(output, gt)

                            total_mse_norm += mse.item() * batch_size
                            total_mae_norm += mae.item() * batch_size
                            num_samples += batch_size

                            # Compute denormalized metrics if scaler available
                            if scaler is not None and hasattr(scaler, 'inverse_transform'):
                                try:
                                    # Reshape for scaler: (batch, seq, features) -> (batch*seq, features)
                                    pred_shape = output.shape
                                    pred_flat = output.cpu().numpy().reshape(-1, pred_shape[-1])
                                    gt_flat = gt.cpu().numpy().reshape(-1, pred_shape[-1])

                                    # Inverse transform
                                    pred_denorm = scaler.inverse_transform(pred_flat)
                                    gt_denorm = scaler.inverse_transform(gt_flat)

                                    # Reshape back and compute metrics
                                    pred_denorm = torch.from_numpy(pred_denorm.reshape(pred_shape)).to(self.device)
                                    gt_denorm = torch.from_numpy(gt_denorm.reshape(pred_shape)).to(self.device)

                                    mse_denorm = nn.MSELoss()(pred_denorm, gt_denorm)
                                    mae_denorm = nn.L1Loss()(pred_denorm, gt_denorm)

                                    total_mse_denorm += mse_denorm.item() * batch_size
                                    total_mae_denorm += mae_denorm.item() * batch_size
                                except Exception:
                                    # Scaler failed, skip denormalized metrics
                                    scaler = None

                            # Calculate metrics for progress bar display
                            running_mse = total_mse_norm / num_samples if num_samples > 0 else 0.0
                            running_mae = total_mae_norm / num_samples if num_samples > 0 else 0.0
                            speed = (time.time() - time_now) / iter_count if iter_count > 0 else 0.0

                            # Format metrics string
                            metrics_str = f"MSE: {running_mse:.7f} • MAE: {running_mae:.7f} • speed: {speed:.4f}s/iter"

                            # Update progress bar with metrics
                            progress.update(sample_task, advance=1, metrics=metrics_str)

                        if num_samples > 0:
                            entity_metrics = {
                                'mse_normalized': total_mse_norm / num_samples,
                                'mae_normalized': total_mae_norm / num_samples,
                                'num_samples': num_samples
                            }

                            # Add denormalized metrics if computed
                            if total_mse_denorm > 0:
                                entity_metrics['mse_denormalized'] = total_mse_denorm / num_samples
                                entity_metrics['mae_denormalized'] = total_mae_denorm / num_samples

                            per_entity_metrics[entity_id] = entity_metrics

                            if self.exp_manager:
                                self.exp_manager.logger.info(
                                    f"  {entity_id}: MSE={entity_metrics['mse_normalized']:.7f}, "
                                    f"MAE={entity_metrics['mae_normalized']:.7f}"
                                )
                        else:
                            if self.exp_manager:
                                self.exp_manager.logger.warning(f"  {entity_id}: No valid samples")

                    except Exception as e:
                        if self.exp_manager:
                            self.exp_manager.logger.error(f"  {entity_id}: Evaluation failed - {e}")
                    finally:
                        # Clean up progress bars
                        progress.remove_task(sample_task)
                        progress.update(entity_task, advance=1)

        if not per_entity_metrics:
            if self.exp_manager:
                self.exp_manager.logger.warning("Final test evaluation: No valid metrics collected")
            return None

        # Format results using helper function
        final_test_metrics = format_test_results(
            per_entity_metrics=per_entity_metrics,
            best_epoch=best_epoch,
            checkpoint_path=os.path.join(checkpoint_path, 'checkpoint.pth')
        )

        if final_test_metrics is None:
            return None

        # Save to metrics/test_results.json
        try:
            if self.exp_manager:
                metrics_dir = Path(self.exp_manager.experiment_dir) / "metrics"
                metrics_dir.mkdir(parents=True, exist_ok=True)
                results_path = metrics_dir / "test_results.json"

                with open(results_path, 'w') as f:
                    json.dump(final_test_metrics, f, indent=2)

                self.exp_manager.logger.info(f"Test results saved to {results_path}")
        except Exception as e:
            if self.exp_manager:
                self.exp_manager.logger.error(f"Failed to save test results: {e}")

        # Note: WandB logging now happens in _finalize_training() to ensure
        # metrics are logged before wandb.finish() is called

        return final_test_metrics

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
        
        # Load resume checkpoint if available
        start_epoch = self._load_resume_checkpoint(model_optim)
        
        # Log training configuration
        print(f"[ info ] Starting training: {train_steps} steps/epoch, epochs {start_epoch+1}-{self.args.train_epochs}")
        
        # Training loop over all epochs
        for epoch in range(start_epoch, self.args.train_epochs):
            self.current_epoch = epoch + 1
            
            # Train one complete epoch
            train_loss, epoch_time, total_samples = self._train_single_epoch(
                epoch, train_loader, model_optim, criterion, 
                track_per_sample, train_steps, vali_loader=vali_loader
            )
            
            # Evaluate epoch: validation, testing, logging, early stopping check
            vali_loss, test_loss, should_stop = self._evaluate_epoch(
                epoch, train_loss, total_samples, vali_loader, test_loader,
                criterion, early_stopping, path, model_optim, track_per_sample, train_steps, epoch_time
            )
            
            # Update job history after each epoch
            # This also checks for Hyperband pruning and other stop signals
            should_stop_epoch = self._update_job_history_after_epoch(path)
            
            # Break if early stopping triggered OR external stop signal (Hyperband, etc.)
            if should_stop or should_stop_epoch:
                break

        # Determine best epoch BEFORE test evaluation and finalization
        # Use best_epoch tracked by EarlyStopping (set when checkpoint was saved)
        best_epoch = early_stopping.best_epoch if early_stopping.best_epoch is not None else self.current_epoch

        # Run final comprehensive test evaluation on best checkpoint BEFORE wandb finalization
        # This ensures test metrics are available for logging to wandb
        final_test_metrics = None
        if test_loader is not None:
            try:
                if self.exp_manager:
                    self.exp_manager.logger.info("Running final test evaluation on best checkpoint...")
                final_test_metrics = self._run_final_test_evaluation(test_loader, best_epoch, path)

                if final_test_metrics:
                    if self.exp_manager:
                        self.exp_manager.logger.info(
                            f"Final test MSE (normalized): {final_test_metrics['overall']['mse_normalized']:.7f}"
                        )
                        self.exp_manager.logger.info(
                            f"Final test MAE (normalized): {final_test_metrics['overall']['mae_normalized']:.7f}"
                        )

                        if 'mse_denormalized' in final_test_metrics['overall']:
                            self.exp_manager.logger.info(
                                f"Final test MSE (denormalized): {final_test_metrics['overall']['mse_denormalized']:.4f}"
                            )
                            self.exp_manager.logger.info(
                                f"Final test MAE (denormalized): {final_test_metrics['overall']['mae_denormalized']:.4f}"
                            )
            except Exception as e:
                if self.exp_manager:
                    self.exp_manager.logger.error(f"Failed to run final test evaluation: {e}")
                import traceback
                traceback.print_exc()

        # Finalize training: load best model, save final checkpoint, and finalize wandb
        # This happens AFTER test evaluation so test metrics can be logged to wandb
        self._finalize_training(
            path,
            model_optim,
            train_loss,
            vali_loss,
            test_loss,
            best_epoch=best_epoch,
            test_metrics=final_test_metrics
        )

        # Register job end in job history
        self._register_job_end(path, train_loss, vali_loss)

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

        # Check for NaN/Inf in validation loss
        import math
        if math.isnan(epoch_loss) or math.isinf(epoch_loss):
            error_msg = f"Validation loss became NaN/Inf at epoch {self.current_epoch}: {epoch_loss}"
            logger = self.exp_manager.logger if self.exp_manager else None
            if logger:
                logger.error(error_msg)
            print(f"\n[ CRITICAL ] {error_msg}")  # Print to console to ensure visibility
            raise ValueError(error_msg)

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
