from exp.exp_basic import Exp_Basic
from models import model_init

from utils.tools import EarlyStopping, adjust_learning_rate, general_move_to_device
from utils.per_sample_metrics import PerSampleMetricsTracker
from utils.loss_utils import compute_per_sample_loss

import torch
import torch.nn as nn

import os
import time
import warnings
import logging

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
            output_dir = str(exp_manager.get_checkpoint_dir()) if exp_manager else args.checkpoints
            self.metrics_tracker = PerSampleMetricsTracker(output_dir)
        else:
            self.metrics_tracker = None

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
                - batch_x: Input time series sequences
                - batch_y: Target time series sequences  
                - timestamp_x, timestamp_y: Corresponding timestamps
                - batch_x_hetero, batch_y_hetero: Heterogeneous data (text, events)
                - hetero_x_time, hetero_y_time: Heterogeneous data timestamps
                - hetero_general: General heterogeneous information
                - hetero_channel: Channel-specific heterogeneous information
        
        Returns:
            tuple: (predictions, ground_truth) both as torch tensors
                  - predictions: Model output for the prediction horizon
                  - ground_truth: True target values for comparison
        """
        # iteration: seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel

        batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = iter

        if hasattr(self.model, 'move_to_device'):
            batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = self.model.move_to_device(batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, self.device) # move only the ones needed to device according to model's definition to save VRAM
        else:
            # only move batch_x, batch_y to device for TSF models
            batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = general_move_to_device(batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, self.device)

        output = self.model(x=batch_x, historical_events =batch_x_hetero, news = batch_y_hetero, dataset_description=hetero_general, channel_description=hetero_channel)

        output = output[:, -self.args.output_len:, :]
        gt = batch_y

        return output, gt


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
        train_loader = self._get_data(flag='train')
        vali_loader = self._get_data(flag='val')
        test_loader = self._get_data(flag='test')
        self.data_provider.data_buffer.clear() # release the raw file in buffer
        # self._get_profile(self.model)
        # print('Trainable parameters: ', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        # Use experiment_id from exp_manager if available, otherwise fall back to args.checkpoints
        if self.exp_manager is not None:
            path = str(self.exp_manager.get_checkpoint_dir())
        else:
            # Fallback for backward compatibility
            path = self.args.checkpoints
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save args.json for backward compatibility (if not using exp_manager)
        if self.exp_manager is None:
            with open(os.path.join(path, 'args.json'), 'w') as f:
                json.dump(self.args.__dict__, f)
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        # Check if per-sample tracking is enabled
        track_per_sample = getattr(self.args, 'track_per_sample', False)

        # self.model.eval()

        # vali_loss = self.vali(test_data, test_loader, criterion, valinum='full')
        # print("Initial test loss: ", vali_loss)

        for epoch in range(self.args.train_epochs):
            self.current_epoch = epoch + 1
            iter_count = 0

            epoch_loss = 0.0
            total_samples = 0

            self.model.train()
            epoch_time = time.time()

            # Define logger at this scope
            logger = self.exp_manager.logger
            # Get console from exp_manager
            console = self.exp_manager.console

            progress_columns = [
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("loss: {task.fields[loss]:.7f}"),
                TextColumn("•"),
                TextColumn("speed: {task.fields[speed]:.4f}s/iter"),
                TextColumn("•"),
                TextColumn("ETA: {task.fields[eta]:.1f}s"),
                TimeElapsedColumn(),
            ]
            with Progress(*progress_columns, console=console) as progress:
                task = progress.add_task(
                    f"Epoch {epoch + 1}/{self.args.train_epochs}",
                    total=len(train_loader),
                    loss=0.0,
                    speed=0.0,
                    eta=0.0
                )
                for i, iter in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()
                        
                    output, gt = self._forward_step(iter)

                    loss = criterion(output, gt)

                    loss.backward()

                    model_optim.step()

                    current_batch_size = gt.size(0)
                    epoch_loss += loss.item() * current_batch_size
                    total_samples += current_batch_size
                    
                    # Track per-sample metrics if enabled
                    if track_per_sample and self.metrics_tracker:
                        per_sample_loss = compute_per_sample_loss(output, gt, criterion)
                        timestamps = iter[2]  # timestamp_x from batch tuple
                        self.metrics_tracker.add_batch(
                            epoch=self.current_epoch,
                            split='train',
                            entity_id=None,  # Train/val use concatenated loaders
                            timestamps=timestamps,
                            losses=per_sample_loss
                        )

                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    progress.update(
                        task,
                        advance=1,
                        loss=loss.item(),
                        speed=speed,
                        eta=left_time
                    )
                    iter_count = 0
                    time_now = time.time()

            epoch_time_elapsed = time.time() - epoch_time
            logger.info(f"Epoch: {epoch + 1} cost time: {epoch_time_elapsed:.2f}s")
            
            train_loss = epoch_loss / total_samples if total_samples > 0 else 0.0
            
            # Save training metrics after training loop (before vali/test)
            if track_per_sample and self.metrics_tracker:
                self.metrics_tracker.save_epoch(self.current_epoch)
            
            vali_loss = self.vali(vali_loader, criterion)  # vali() saves its own metrics
            test_loss = self.test(test_loader, criterion)  # test() saves its own metrics

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
            
            # EarlyStopping expects a directory path, ensure it exists
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                # Log final metrics if early stopping
                if self.exp_manager is not None:
                    self.exp_manager.log_metrics({
                        'best_epoch': epoch + 1,
                        'final_train_loss': train_loss,
                        'final_val_loss': vali_loss,
                        'final_test_loss': test_loss,
                    })
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

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
                            output, gt = self._forward_step(iter_data)
                            current_batch_size = gt.size(0)
                            loss = criterion(output, gt)
                            running_loss += loss.item() * current_batch_size
                            total_samples += current_batch_size
                            
                            # Track per-sample metrics if enabled
                            if track_per_sample and self.metrics_tracker:
                                per_sample_loss = compute_per_sample_loss(output, gt, criterion)
                                timestamps = iter_data[2]  # timestamp_x from batch tuple
                                self.metrics_tracker.add_batch(
                                    epoch=None,  # Epoch is null for validation/test folds
                                    split='val',
                                    entity_id=None,  # Entity is null for train/val folds
                                    timestamps=timestamps,
                                    losses=per_sample_loss
                                )
                            
                            progress.update(task, advance=1)
                else:
                    for i, iter_data in enumerate(loader):
                        output, gt = self._forward_step(iter_data)
                        current_batch_size = gt.size(0)
                        loss = criterion(output, gt)
                        running_loss += loss.item() * current_batch_size
                        total_samples += current_batch_size
                        
                        # Track per-sample metrics if enabled
                        if track_per_sample and self.metrics_tracker:
                            per_sample_loss = compute_per_sample_loss(output, gt, criterion)
                            timestamps = iter_data[2]  # timestamp_x from batch tuple
                            self.metrics_tracker.add_batch(
                                epoch=None,  # Epoch is null for validation/test folds
                                split='val',
                                entity_id=None,  # Entity is null for train/val folds
                                timestamps=timestamps,
                                losses=per_sample_loss
                            )

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        
        # Save per-sample metrics for validation if enabled
        if track_per_sample and self.metrics_tracker:
            self.metrics_tracker.save_split('val')
        
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
                        output, gt = self._forward_step(iter_data)
                        current_batch_size = gt.size(0)
                        loss = criterion(output, gt)
                        info_running_loss += loss.item() * current_batch_size
                        info_total_samples += current_batch_size
                        overall_running_loss += loss.item() * current_batch_size
                        overall_total_samples += current_batch_size
                        
                        # Track per-sample metrics if enabled
                        if track_per_sample and self.metrics_tracker:
                            per_sample_loss = compute_per_sample_loss(output, gt, criterion)
                            timestamps = iter_data[2]
                            
                            self.metrics_tracker.add_batch(
                                epoch=None,  # Epoch is null for validation/test folds
                                split='test',
                                entity_id=info,
                                timestamps=timestamps,
                                losses=per_sample_loss
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
                    
                    # Log results to file only
                    if info_total_samples > 0:
                        info_epoch_loss = info_running_loss / info_total_samples
                        self.exp_manager.log_file_only(f"Test loss for entity {info}: {info_epoch_loss:.7f}")
                    else:
                        self.exp_manager.log_file_only(f"Test loss for entity {info}: N/A (no samples processed)", level=logging.WARNING)
        
        # Save per-sample metrics for test if enabled
        if track_per_sample and self.metrics_tracker:
            self.metrics_tracker.save_split('test')

        total_epoch_loss = overall_running_loss / overall_total_samples if overall_total_samples > 0 else 0.0
        
        self.model.train()
        return total_epoch_loss
