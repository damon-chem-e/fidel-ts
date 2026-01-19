import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn as nn
from models import model_init
from utils.tools import general_move_to_device, adjust_learning_rate
import json
import time
import logging
from typing import Optional
# Rich imports for progress bars
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.console import Console

import warnings
import inspect

warnings.filterwarnings('ignore')


class TimeSeriesLightningModel(pl.LightningModule):
    """
    PyTorch Lightning module for time series forecasting.
    Wraps the existing model implementations and training logic.
    """
    def __init__(self, args, exp_manager=None):
        super(TimeSeriesLightningModel, self).__init__()
        self.args = args
        self.exp_manager = exp_manager
        self.save_hyperparameters(ignore=['args', 'exp_manager'])
        
        # Build model
        self.model = model_init(self.args.model, self.args.model_config, self.args)
        
        # Apply torch.compile if enabled (PyTorch 2.0+)
        # Check for torch_compile flag in args (backward compatible - defaults to False)
        if getattr(args, 'torch_compile', False):
            if hasattr(torch, 'compile'):
                from utils.tools import compilation_spinner

                compile_mode = getattr(args, 'compile_mode', 'reduce-overhead')
                logger = exp_manager.logger if exp_manager else None

                with compilation_spinner(
                    f"Compiling model with torch.compile (mode={compile_mode})...",
                    logger=logger
                ):
                    self.model = torch.compile(
                        self.model,
                        mode=compile_mode,
                        fullgraph=False  # More compatible with dynamic models
                    )
            else:
                warning_msg = "torch.compile requested but not available (requires PyTorch 2.0+)"
                if exp_manager:
                    exp_manager.logger.warning(f"WARNING: {warning_msg}")
                else:
                    print(f"WARNING: {warning_msg}")
        
        # Loss function
        # --- MODIFICATION START ---
        # The criterion itself is correct (reduction='mean' is fine for training steps).
        # The error was in how the results were aggregated during testing.
        self.criterion = self._select_criterion()
        # --- MODIFICATION END ---
        
        # Configure automatic optimization if needed
        self.automatic_optimization = True

        # --- MODIFICATION START ---
        # We need to store both the total loss and the number of samples for each batch 
        # to calculate the correct average loss at the end of the test epoch.
        # Storing just the average loss of each batch (loss.item()) is incorrect.
        self.test_total_loss = []
        self.test_total_samples = []
        # --- MODIFICATION END ---
        
        # Track epoch timing for logging
        self._epoch_start_time: Optional[float] = None
        
    def _select_criterion(self):
        """Select the loss function."""
        if self.args.loss == 'l1':
            return nn.L1Loss()
        elif self.args.loss == 'mse':
            return nn.MSELoss()
        else:
            return nn.MSELoss()  # Default
    
    def forward(self, batch):
        """Forward pass."""
        # Batch structure: sample_ids, batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel
        sample_ids, batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = batch
        
        if hasattr(self.model, 'move_to_device'):
            batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel = self.model.move_to_device(
                batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, 
                hetero_x_time, hetero_y_time, hetero_general, hetero_channel, self.device
            )
        else:
            # Only move batch_x, batch_y to device for TSF models
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
        
        # Inspect model's forward method signature to determine what inputs to provide
        model_params = list(inspect.signature(self.model.forward).parameters.keys())
        
        # Build input dictionary based on available parameters
        forward_kwargs = {}
        if 'x' in model_params:
            forward_kwargs['x'] = batch_x
        if 'historical_events' in model_params and 'historical_events' not in forward_kwargs:
            forward_kwargs['historical_events'] = batch_x_hetero
        if 'news' in model_params and 'news' not in forward_kwargs:
            forward_kwargs['news'] = batch_y_hetero
        if 'dataset_description' in model_params and 'dataset_description' not in forward_kwargs:
            forward_kwargs['dataset_description'] = hetero_general
        if 'channel_description' in model_params and 'channel_description' not in forward_kwargs:
            forward_kwargs['channel_description'] = hetero_channel
        
        # Call model with appropriate arguments
        output = self.model(**forward_kwargs)
        
        output = output[:, -self.args.output_len:, :]
        return output, batch_y
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        output, gt = self.forward(batch)
        loss = self.criterion(output, gt)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        output, gt = self.forward(batch)
        loss = self.criterion(output, gt)
        
        # Log metrics
        # Pytorch-Lightning's default on_epoch=True aggregation is a weighted average, which is correct.
        # So no change is needed here. The error was in the manual test loops.
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        # Mark epoch start in GPU monitor for epoch-level GPU utilization tracking
        if self.exp_manager and hasattr(self.exp_manager, 'gpu_monitor') and self.exp_manager.gpu_monitor:
            self.exp_manager.gpu_monitor.mark_epoch_start()
        
        # Record epoch start time for timing
        self._epoch_start_time = time.time()
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        if self.trainer.sanity_checking:
            return
        
        # Get current epoch number (0-indexed in Lightning, so add 1 for display)
        current_epoch = self.trainer.current_epoch + 1
        
        # Calculate and log epoch time to file only (not console)
        if self._epoch_start_time is not None and self.exp_manager:
            epoch_time_elapsed = time.time() - self._epoch_start_time
            self.exp_manager.log_file_only(f"Epoch {current_epoch} completed in {epoch_time_elapsed:.2f}s")
        
        # Get and log full epoch GPU summary to file only (not console) using GpuMonitor's formatting method
        if self.exp_manager and hasattr(self.exp_manager, 'gpu_monitor') and self.exp_manager.gpu_monitor:
            gpu_summary_str = self.exp_manager.gpu_monitor.format_epoch_summary_for_log()
            if gpu_summary_str:
                self.exp_manager.log_file_only(f"Epoch {current_epoch}: {gpu_summary_str}")
    
    def on_validation_epoch_end(self):
        """After validation completes, run test on all subsets."""
        # Only run test during training, not during sanity check
        if self.trainer.sanity_checking:
            return
        
        # Manually run test on all subsets
        if self.args.test_after_epoch:
            self._run_epoch_test()
        
    def _run_epoch_test(self):
        """Run test on all subsets and print results."""
        # Skip if we don't have access to datamodule or if test dataset isn't set up yet
        if not hasattr(self.trainer.datamodule, 'test_dataset'):
            self.trainer.datamodule.setup(stage='test')
        
        # Assume exp_manager and logger exist as they are required
        logger = self.exp_manager.logger
        console = self.exp_manager.console
        
        logger.info("\n\n------- Testing on Epoch {} -------".format(self.current_epoch + 1))
        
        # Save current state
        self.model.eval()
        test_loaders = self.trainer.datamodule.test_dataloader()
        
        # Track total loss and total samples to calculate the true average loss,
        # avoiding the "mean of means" error.
        subset_losses = {}
        overall_total_loss = 0.0
        overall_total_samples = 0
        
        with torch.no_grad():
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                # Outer progress bar for entities
                entity_task = progress.add_task("Testing entities", total=len(test_loaders))
                
                for subset_id, loader in test_loaders.items():
                    subset_total_loss = 0.0
                    subset_total_samples = 0
                    
                    # Inner progress bar for samples within current entity
                    sample_task = progress.add_task(f"  └─ {subset_id}", total=len(loader))
                    
                    for i, batch in enumerate(loader):
                        output, gt = self.forward(batch)
                        loss = self.criterion(output, gt)
                        
                        # The criterion calculates the mean loss for the batch. To get the total
                        # loss for the batch, we multiply by the number of elements.
                        num_samples_in_batch = gt.numel()
                        total_batch_loss = loss.item() * num_samples_in_batch
                        
                        subset_total_loss += total_batch_loss
                        subset_total_samples += num_samples_in_batch
                        progress.update(sample_task, advance=1)
                    
                    # Remove the sample task when done with this entity
                    progress.remove_task(sample_task)
                    
                    # Calculate average for this subset
                    if subset_total_samples > 0:
                        avg_loss = subset_total_loss / subset_total_samples
                        subset_losses[subset_id] = avg_loss
                        overall_total_loss += subset_total_loss
                        overall_total_samples += subset_total_samples
                        
                        # Note: Per-entity test loss is tracked in per-sample metrics (parquet files)
                        # No need to log individual entity losses here
                    
                    # Update entity progress
                    progress.update(entity_task, advance=1)
        
        # Calculate overall average
        if overall_total_samples > 0:
            overall_avg = overall_total_loss / overall_total_samples
            self.exp_manager.log_file_only(f"Overall test loss: {overall_avg:.7f}")
            
        print("---------------------------------------\n")
        
        # Restore model state
        self.model.train()
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Test step."""
        output, gt = self.forward(batch)
        loss = self.criterion(output, gt)
        
        # --- MODIFICATION START ---
        # Instead of appending the mean batch loss, we append the total loss
        # and the number of samples for correct aggregation later.
        num_samples = gt.numel()
        self.test_total_loss.append(loss.item() * num_samples)
        self.test_total_samples.append(num_samples)
        # --- MODIFICATION END ---

    def on_test_epoch_end(self):
        """After test completes, log average test loss."""
        # --- MODIFICATION START ---
        # Calculate the true average loss: sum of all total batch losses divided
        # by the sum of all batch sample counts.
        total_loss = np.sum(self.test_total_loss)
        total_samples = np.sum(self.test_total_samples)

        if total_samples > 0:
            avg_loss = total_loss / total_samples
        else:
            avg_loss = 0.0 # Handle case with no test data

        self.log('test_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # Reset lists for the next potential test run (e.g., in a new training session)
        self.test_total_loss = []
        self.test_total_samples = []
        # --- MODIFICATION END ---


    def configure_optimizers(self):
        """Configure optimizers and LR schedulers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        
        # Custom learning rate adjustment
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(
                optimizer, 
                lr_lambda=lambda epoch: adjust_learning_rate(None, epoch, self.args, return_rate=True)
            ),
            'name': 'learning_rate',
            'interval': 'epoch',
            'frequency': 1
        }
        
        return [optimizer], [lr_scheduler]


def train_lightning_model(args, exp_manager):
    """
    Train the Lightning model and save it.
    
    Args:
        args: Arguments for the experiment
        exp_manager: ExperimentManager for experiment tracking (required)
    
    Returns:
        Trained model
    """
    # Verify exp_manager is provided
    if exp_manager is None:
        raise ValueError("exp_manager is required for training")

    from data_provider.lightning_data_module import TimeSeriesDataModule
    
    # Initialize data module
    data_module = TimeSeriesDataModule(args)
    
    # Check for resume information from exp_manager
    resume_info = exp_manager.get_resume_info()
    resume_checkpoint_path = None
    if resume_info and resume_info.get("checkpoint_path"):
        resume_checkpoint_path = resume_info["checkpoint_path"]
        exp_manager.logger.info(f"Resume detected: will resume from checkpoint {resume_checkpoint_path}")
    elif args.last_ckpt is not None:
        # Use explicit checkpoint if provided (backward compatibility)
        resume_checkpoint_path = args.last_ckpt
        exp_manager.logger.info(f"Using explicit checkpoint: {args.last_ckpt}")

    # Initialize model (Lightning will load from checkpoint during trainer.fit if ckpt_path is provided)
    model = TimeSeriesLightningModel(args, exp_manager=exp_manager)
    # Use experiment_id from exp_manager
    checkpoint_path = str(exp_manager.get_checkpoint_dir())
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    # Create callback for job history updates
    class JobHistoryCallback(pl.Callback):
        """Callback to update job history after each epoch."""
        def __init__(self, exp_manager):
            self.exp_manager = exp_manager
        
        def on_train_epoch_end(self, trainer, pl_module):
            """
            Update job history after each training epoch.
            
            This also checks for external stop signals (Hyperband pruning, etc.)
            and stops training if needed.
            """
            if trainer.sanity_checking:
                return
            
            current_epoch = trainer.current_epoch + 1  # Lightning uses 0-indexed, we use 1-indexed
            
            # Get checkpoint path (last checkpoint)
            checkpoint_path = None
            if hasattr(trainer.checkpoint_callback, 'last_model_path') and trainer.checkpoint_callback.last_model_path:
                checkpoint_path = trainer.checkpoint_callback.last_model_path
            
            # Update epoch and check for stop signals (Hyperband, etc.)
            should_stop = self.exp_manager.update_current_epoch(
                epoch=current_epoch,
                checkpoint_path=checkpoint_path
            )
            
            # Stop training if external signal detected (e.g., Hyperband pruning)
            if should_stop:
                trainer.should_stop = True
    
    # Configure callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        verbose=True,
        mode='min'
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename='checkpoint-{epoch:02d}-{val_loss:.6f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_last=True
    )
    
    job_history_callback = JobHistoryCallback(exp_manager)
    
    
    # Configure logger
    # Use experiment_id from exp_manager
    logger_name = exp_manager.get_experiment_id()
    logger_save_dir = str(exp_manager.get_experiment_dir() / "tb_logs")
    
    logger = TensorBoardLogger(
        save_dir=logger_save_dir,
        name=logger_name
    )
    
    # Advanced trainer configurations for better performance
    trainer_kwargs = {
        'max_epochs': args.train_epochs,
        'accelerator': 'gpu' if args.use_gpu else 'cpu',
        'devices': args.device_ids if args.use_multi_gpu else [args.gpu] if args.use_gpu else None,
        'strategy': 'ddp' if args.use_multi_gpu else None,
        'callbacks': [early_stopping, checkpoint_callback, job_history_callback],
        'logger': logger,
        'deterministic': True,
        'precision': getattr(args, 'precision', 32),
        'gradient_clip_val': args.gradient_clip_val if hasattr(args, 'gradient_clip_val') and args.gradient_clip_val > 0 else None,
        # Added for better performance
        'num_sanity_val_steps': 0,  # Skip sanity check for faster startup
        'enable_checkpointing': True,
        'enable_model_summary': True,
        'enable_progress_bar': True,
        'log_every_n_steps': 50,
    }
    
    # Add profiler if requested
    if hasattr(args, 'profiler') and args.profiler:
        trainer_kwargs['profiler'] = 'simple'
    
    # Configure trainer
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train model
    exp_manager.logger.info('>>>>>>>start training >>>>>>>>>>>>>>>>>>>>>>>>>>>')
        
    if not args.test:
        # Resume from checkpoint if available
        trainer.fit(model, data_module, ckpt_path=resume_checkpoint_path)
    
    # Get the path to the best model saved by the checkpoint callback
    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path or not os.path.exists(best_model_path):
        exp_manager.logger.warning("Could not find best model path. Using last model for testing.")
        # Fallback to the last saved model if best is not found
        best_model_path = checkpoint_callback.last_model_path 
    
    # Final test using the best model checkpoint
    exp_manager.logger.info(f'>>>>>>>final testing on best model: {best_model_path}>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    data_module.setup(stage='test')
    test_loaders = data_module.test_dataloader()

    from utils.tools import format_test_results
    from pathlib import Path

    # Load best checkpoint for comprehensive evaluation
    exp_manager.logger.info("Loading best checkpoint for comprehensive test evaluation...")
    best_checkpoint = torch.load(best_model_path, map_location=device)

    # Extract state dict from Lightning checkpoint and load into the Lightning model
    if isinstance(best_checkpoint, dict) and 'state_dict' in best_checkpoint:
        model.load_state_dict(best_checkpoint['state_dict'])
    else:
        # Fallback: load into underlying model
        state_dict = {key.replace("model.", "", 1): value for key, value in best_checkpoint.items()}
        model.model.load_state_dict(state_dict)

    model.eval()

    # Collect per-entity metrics
    per_entity_metrics = {}
    info_results = {}  # Legacy format for backward compatibility
    total_entities = len(test_loaders)

    with torch.inference_mode():
        for idx, (subset_id, loader) in enumerate(test_loaders.items()):
            exp_manager.logger.info(f"Final test evaluation: {subset_id} ({idx + 1}/{total_entities})")

            # Get dataset from loader for scaler access
            dataset = loader.dataset if hasattr(loader, 'dataset') else None
            scaler = getattr(dataset, 'scaler', None) if dataset else None

            # Accumulators for this entity
            total_mse_norm = 0.0
            total_mae_norm = 0.0
            total_mse_denorm = 0.0
            total_mae_denorm = 0.0
            num_samples = 0

            try:
                for batch in loader:
                    # Use the Lightning model's forward method (same as training)
                    # Batch needs to be moved to device - extract and move tensors
                    batch_on_device = tuple(
                        t.to(device) if isinstance(t, torch.Tensor) else t for t in batch
                    )
                    output, gt = model.forward(batch_on_device)
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
                            pred_denorm = torch.from_numpy(pred_denorm.reshape(pred_shape)).to(device)
                            gt_denorm = torch.from_numpy(gt_denorm.reshape(pred_shape)).to(device)

                            mse_denorm = nn.MSELoss()(pred_denorm, gt_denorm)
                            mae_denorm = nn.L1Loss()(pred_denorm, gt_denorm)

                            total_mse_denorm += mse_denorm.item() * batch_size
                            total_mae_denorm += mae_denorm.item() * batch_size
                        except Exception:
                            # Scaler failed, skip denormalized metrics
                            scaler = None

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

                    per_entity_metrics[subset_id] = entity_metrics
                    info_results[subset_id] = entity_metrics['mse_normalized']  # Legacy format

                    exp_manager.logger.info(
                        f"  {subset_id}: MSE={entity_metrics['mse_normalized']:.7f}, "
                        f"MAE={entity_metrics['mae_normalized']:.7f}"
                    )
                else:
                    exp_manager.logger.warning(f"  {subset_id}: No valid samples")

            except Exception as e:
                exp_manager.logger.error(f"  {subset_id}: Evaluation failed - {e}")
                import traceback
                traceback.print_exc()
                continue

    # Format results using helper function
    # Get best epoch from checkpoint filename (Lightning saves epoch in filename)
    best_epoch = trainer.current_epoch + 1 if hasattr(trainer, 'current_epoch') else args.train_epochs
    if best_model_path and 'epoch=' in best_model_path:
        try:
            epoch_str = best_model_path.split('epoch=')[1].split('-')[0]
            best_epoch = int(epoch_str) + 1  # Convert 0-indexed to 1-indexed
        except (IndexError, ValueError):
            pass

    final_test_metrics = format_test_results(
        per_entity_metrics=per_entity_metrics,
        best_epoch=best_epoch,
        checkpoint_path=best_model_path
    )

    if trainer.is_global_zero and final_test_metrics:  # Only the main process writes files
        # Save to metrics/test_results.json (new location)
        metrics_dir = Path(exp_manager.experiment_dir) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        results_path = metrics_dir / "test_results.json"

        with open(results_path, 'w') as f:
            json.dump(final_test_metrics, f, indent=2)
        exp_manager.logger.info(f"Test results saved to {results_path}")

        # Log overall metrics
        exp_manager.logger.info(
            f"Final test MSE (normalized): {final_test_metrics['overall']['mse_normalized']:.7f}"
        )
        exp_manager.logger.info(
            f"Final test MAE (normalized): {final_test_metrics['overall']['mae_normalized']:.7f}"
        )
        if 'mse_denormalized' in final_test_metrics['overall']:
            exp_manager.logger.info(
                f"Final test MSE (denormalized): {final_test_metrics['overall']['mse_denormalized']:.4f}"
            )
            exp_manager.logger.info(
                f"Final test MAE (denormalized): {final_test_metrics['overall']['mae_denormalized']:.4f}"
            )

        # Log to WandB summary
        if exp_manager.wandb_run is not None:
            wandb_metrics = {
                'test/mse_normalized': final_test_metrics['overall']['mse_normalized'],
                'test/mae_normalized': final_test_metrics['overall']['mae_normalized'],
                'test/num_samples': final_test_metrics['overall']['num_samples'],
                'test/num_entities': len(per_entity_metrics)
            }

            if 'mse_denormalized' in final_test_metrics['overall']:
                wandb_metrics['test/mse_denormalized'] = final_test_metrics['overall']['mse_denormalized']
                wandb_metrics['test/mae_denormalized'] = final_test_metrics['overall']['mae_denormalized']

            exp_manager.wandb_run.summary.update(wandb_metrics)
            exp_manager.logger.info("Test metrics logged to WandB summary")

        # Keep backward compatibility: also save legacy format to checkpoints/
        with open(os.path.join(checkpoint_path, 'test_results.json'), 'w') as f:
            json.dump(info_results, f)
        with open(os.path.join(checkpoint_path, 'test_results_average.json'), 'w') as f:
            json.dump({'average loss of all subsets': final_test_metrics['overall']['mse_normalized']}, f)

    if args.test:
        return
    
    # Register job end in job history
    final_epoch = trainer.current_epoch + 1 if hasattr(trainer, 'current_epoch') else args.train_epochs
    final_checkpoint = checkpoint_callback.last_model_path if hasattr(checkpoint_callback, 'last_model_path') else best_model_path
    
    # Determine job status
    if final_epoch >= args.train_epochs:
        status = "completed"
    else:
        status = "timeout"  # Job likely timed out
    
    # Get final metrics if available
    final_train_loss = None
    final_val_loss = None
    if hasattr(trainer, 'callback_metrics'):
        if 'train_loss' in trainer.callback_metrics:
            final_train_loss = trainer.callback_metrics['train_loss'].item()
        if 'val_loss' in trainer.callback_metrics:
            final_val_loss = trainer.callback_metrics['val_loss'].item()
    
    exp_manager.register_job_end(
        end_epoch=final_epoch,
        status=status,
        checkpoint_path=final_checkpoint,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss
    )
    
    return best_model_path  # Return the wrapped model for compatibility