"""
Time-LLM Model-Specific Training Module.

This module provides training infrastructure for Time-LLM, implementing
both PyTorch Lightning and standard PyTorch training paths.

Time-LLM uses a frozen LLM backbone and only trains the following components:
    - PatchEmbedding layer
    - ReprogrammingLayer
    - Word embedding mapping layer
    - Output projection (FlattenHead)

Key Features:
    - Frozen LLM backbone (only train non-LLM parameters)
    - Supports quantization (4-bit, 8-bit) for large LLMs
    - Dynamic prompts generated per-batch during forward pass

Exports:
    - train_time_llm_lightning: Lightning-based training
    - train_time_llm_pytorch: Standard PyTorch training
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional
import warnings

from models import model_init
from utils.tools import adjust_learning_rate, EarlyStopping
from cli.config.model_training import TimeLLMTrainingConfig
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

warnings.filterwarnings('ignore')


# =============================================================================
# Shared Utilities
# =============================================================================

def _select_criterion(loss_type: str) -> nn.Module:
    """
    Select the appropriate loss function.
    
    Args:
        loss_type: Loss function type ("mse" or "mae")
    
    Returns:
        PyTorch loss module
    """
    if loss_type == 'mae' or loss_type == 'l1':
        return nn.L1Loss()
    return nn.MSELoss()


def _get_time_llm_config(args) -> TimeLLMTrainingConfig:
    """
    Extract and validate Time-LLM training config from args.
    
    Args:
        args: Experiment arguments
    
    Returns:
        TimeLLMTrainingConfig instance
    """
    time_llm_config = getattr(args, 'time_llm', None)
    if time_llm_config is None:
        return TimeLLMTrainingConfig()
    elif isinstance(time_llm_config, dict):
        return TimeLLMTrainingConfig(**time_llm_config)
    return time_llm_config


def _log_trainable_params(model, logger):
    """
    Log trainable vs frozen parameter counts.
    
    Args:
        model: PyTorch model
        logger: Logger instance
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(f"[ Time-LLM ] Trainable params: {trainable:,}")
    logger.info(f"[ Time-LLM ] Frozen params: {frozen:,}")


# =============================================================================
# PyTorch Lightning Training
# =============================================================================

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping as PLEarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    pl = None


if HAS_LIGHTNING:
    class TimeLLMLightningModule(pl.LightningModule):
        """
        PyTorch Lightning module for Time-LLM with frozen LLM backbone.
        
        Handles training loop, validation, and testing with automatic
        optimization of only non-frozen parameters.
        """
        
        def __init__(
            self, 
            args, 
            exp_manager=None, 
            time_llm_config: Optional[TimeLLMTrainingConfig] = None
        ):
            """
            Initialize Time-LLM Lightning module.
            
            Args:
                args: Experiment arguments
                exp_manager: ExperimentManager for tracking
                time_llm_config: Time-LLM specific configuration
            """
            super().__init__()
            self.args = args
            self.exp_manager = exp_manager
            self.time_llm_config = time_llm_config or TimeLLMTrainingConfig()
            
            self.save_hyperparameters(ignore=['args', 'exp_manager', 'time_llm_config'])
            
            # Build the Time-LLM model
            self.model = model_init(self.args.model, self.args.model_config, self.args)
            
            # Loss function
            self.criterion = _select_criterion(self.time_llm_config.loss)
            
            # Test tracking
            self.test_total_loss = []
            self.test_total_samples = []
            self._epoch_start_time: Optional[float] = None
        
        def forward(self, batch):
            """
            Forward pass through the Time-LLM model.
            
            Args:
                batch: Tuple of batch data from DataLoader
            
            Returns:
                Tuple of (forecast, batch_y)
            """
            sample_ids, batch_x, batch_y, timestamp_x, timestamp_y, \
                batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, \
                hetero_general, hetero_channel = batch
            
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Time-LLM forward pass
            forecast = self.model(x=batch_x)
            
            return forecast, batch_y
        
        def training_step(self, batch, batch_idx):
            """Training step with loss computation."""
            forecast, batch_y = self.forward(batch)
            output = forecast[:, -self.args.output_len:, :]
            loss = self.criterion(output, batch_y)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        
        def validation_step(self, batch, batch_idx):
            """Validation step with loss computation."""
            forecast, batch_y = self.forward(batch)
            output = forecast[:, -self.args.output_len:, :]
            loss = self.criterion(output, batch_y)
            self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
            return loss
        
        def test_step(self, batch, batch_idx, dataloader_idx=0):
            """Test step with loss tracking."""
            forecast, batch_y = self.forward(batch)
            output = forecast[:, -self.args.output_len:, :]
            loss = self.criterion(output, batch_y)
            
            num_samples = batch_y.numel()
            self.test_total_loss.append(loss.item() * num_samples)
            self.test_total_samples.append(num_samples)
        
        def on_test_epoch_end(self):
            """Aggregate test metrics at epoch end."""
            total_loss = np.sum(self.test_total_loss)
            total_samples = np.sum(self.test_total_samples)
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            self.log('test_loss', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
            self.test_total_loss = []
            self.test_total_samples = []
        
        def on_train_epoch_start(self):
            """Mark epoch start for timing and GPU monitoring."""
            if self.exp_manager and hasattr(self.exp_manager, 'gpu_monitor') and self.exp_manager.gpu_monitor:
                self.exp_manager.gpu_monitor.mark_epoch_start()
            self._epoch_start_time = time.time()
        
        def on_train_epoch_end(self):
            """Log epoch completion time."""
            if self.trainer.sanity_checking:
                return
            current_epoch = self.trainer.current_epoch + 1
            if self._epoch_start_time is not None and self.exp_manager:
                epoch_time = time.time() - self._epoch_start_time
                self.exp_manager.log_file_only(f"Epoch {current_epoch} completed in {epoch_time:.2f}s")
        
        def configure_optimizers(self):
            """
            Configure optimizer for trainable parameters only.
            
            Only optimizes parameters with requires_grad=True,
            which excludes the frozen LLM backbone.
            """
            trainable_params = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=self.args.learning_rate)
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


def train_time_llm_lightning(args, exp_manager) -> Path:
    """
    Train Time-LLM model using PyTorch Lightning.
    
    Args:
        args: Experiment arguments with time_llm config
        exp_manager: ExperimentManager for tracking
    
    Returns:
        Path to best model checkpoint
    
    Raises:
        ImportError: If PyTorch Lightning is not installed
    """
    if not HAS_LIGHTNING:
        raise ImportError("PyTorch Lightning is required for Lightning training")
    
    from data_provider.lightning_data_module import TimeSeriesDataModule
    
    time_llm_config = _get_time_llm_config(args)
    
    exp_manager.logger.info("=" * 60)
    exp_manager.logger.info("Time-LLM Training (Lightning)")
    exp_manager.logger.info(f"  LLM Backbone: {time_llm_config.llm_backbone}")
    exp_manager.logger.info(f"  Quantization: {time_llm_config.quantization}")
    exp_manager.logger.info(f"  Epochs: {args.train_epochs}")
    exp_manager.logger.info("=" * 60)
    
    model = TimeLLMLightningModule(args, exp_manager, time_llm_config)
    _log_trainable_params(model.model, exp_manager.logger)
    
    data_module = TimeSeriesDataModule(args)
    checkpoint_dir = str(exp_manager.get_checkpoint_dir())
    
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='checkpoint-{epoch:02d}-{val_loss:.6f}',
        save_top_k=1, 
        monitor='val_loss', 
        mode='min', 
        save_last=True
    )
    
    class JobHistoryCallback(pl.Callback):
        """Callback to update job history with epoch progress."""
        def __init__(self, em): 
            self.em = em
        def on_train_epoch_end(self, trainer, pl_module):
            if not trainer.sanity_checking:
                self.em.update_current_epoch(trainer.current_epoch + 1)
    
    trainer = pl.Trainer(
        max_epochs=args.train_epochs,
        accelerator='gpu' if args.use_gpu else 'cpu',
        devices=[args.gpu] if args.use_gpu and not args.use_multi_gpu else (
            args.device_ids if args.use_multi_gpu else None
        ),
        strategy='ddp' if args.use_multi_gpu else 'auto',
        callbacks=[
            checkpoint_cb, 
            PLEarlyStopping(monitor='val_loss', patience=args.patience, mode='min'),
            JobHistoryCallback(exp_manager)
        ],
        logger=TensorBoardLogger(str(exp_manager.get_experiment_dir() / "tb_logs"), name="time_llm"),
        deterministic=True,
        precision=getattr(args, 'precision', 32),
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )
    
    trainer.fit(model, data_module)
    
    best_model_path = checkpoint_cb.best_model_path or checkpoint_cb.last_model_path
    
    # Run final testing
    data_module.setup(stage='test')
    test_loaders = data_module.test_dataloader()
    test_results = {}
    
    for subset_id, loader in test_loaders.items():
        trainer.test(model, dataloaders=loader, ckpt_path=best_model_path)
        test_results[subset_id] = trainer.callback_metrics['test_loss'].item()
    
    # Save test results
    if trainer.is_global_zero:
        with open(exp_manager.get_checkpoint_dir() / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
    
    exp_manager.register_job_end(
        end_epoch=trainer.current_epoch + 1, 
        status="completed", 
        checkpoint_path=best_model_path
    )
    
    return Path(best_model_path)


# =============================================================================
# Standard PyTorch Training
# =============================================================================

class TimeLLMPyTorchTrainer:
    """
    Standard PyTorch trainer for Time-LLM with frozen LLM backbone.
    
    Provides a training loop that only optimizes non-frozen parameters
    (excludes the LLM backbone which is frozen).
    """
    
    def __init__(self, args, exp_manager):
        """
        Initialize trainer.
        
        Args:
            args: Experiment arguments
            exp_manager: ExperimentManager for tracking
        """
        self.args = args
        self.exp_manager = exp_manager
        self.device = self._get_device()
        
        # Build model
        self.model = self._build_model()
        self.model.to(self.device)
        
        _log_trainable_params(self.model, exp_manager.logger)
        
        # Data provider
        from data_provider.data_factory import Data_Provider
        console = exp_manager.get_console() if exp_manager else None
        self.data_provider = Data_Provider(args, buffer=(not args.disable_buffer), console=console)
    
    def _get_device(self):
        """Get the training device."""
        if self.args.use_gpu and torch.cuda.is_available():
            return torch.device(f'cuda:{self.args.gpu}')
        return torch.device('cpu')
    
    def _build_model(self):
        """Build the Time-LLM model."""
        model = model_init(self.args.model, self.args.model_config, self.args)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _forward_step(self, batch):
        """
        Forward pass returning forecast and ground truth.
        
        Args:
            batch: Tuple of batch data from DataLoader
        
        Returns:
            Tuple of (forecast, batch_y, sample_ids)
        """
        sample_ids, batch_x, batch_y, timestamp_x, timestamp_y, \
            batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, \
            hetero_general, hetero_channel, x_time_features, y_time_features = batch
        
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        
        # Time-LLM forward pass
        forecast = self.model(x=batch_x)
        
        return forecast, batch_y, sample_ids
    
    def _train_epoch(self, train_loader, optimizer, criterion, epoch):
        """
        Train one epoch.
        
        Args:
            train_loader: Training DataLoader
            optimizer: Optimizer instance
            criterion: Loss function
            epoch: Current epoch number
        
        Returns:
            Tuple of (average_loss, epoch_time)
        """
        self.model.train()
        total_loss, total_samples = 0.0, 0
        epoch_time = time.time()
        console = self.exp_manager.console
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("• loss: {task.fields[loss]:.7f}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Epoch {epoch}", total=len(train_loader), loss=0.0)
            
            for batch in train_loader:
                optimizer.zero_grad()
                forecast, batch_y, _ = self._forward_step(batch)
                
                # Compute loss on prediction horizon
                output = forecast[:, -self.args.output_len:, :]
                loss = criterion(output, batch_y)
                
                loss.backward()
                optimizer.step()
                
                batch_size = batch_y.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                progress.update(task, advance=1, loss=loss.item())
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        epoch_time_elapsed = time.time() - epoch_time
        return avg_loss, epoch_time_elapsed
    
    def _validate(self, loader, criterion):
        """
        Run validation.
        
        Args:
            loader: Validation DataLoader
            criterion: Loss function
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss, total_samples = 0.0, 0
        
        with torch.no_grad():
            for batch in loader:
                forecast, batch_y, _ = self._forward_step(batch)
                output = forecast[:, -self.args.output_len:, :]
                loss = criterion(output, batch_y)
                
                batch_size = batch_y.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        self.model.train()
        return total_loss / total_samples if total_samples > 0 else 0.0
    
    def _test(self, loaders, criterion):
        """
        Run testing on all subsets.
        
        Args:
            loaders: Dictionary of test DataLoaders
            criterion: Loss function
        
        Returns:
            Tuple of (results_dict, overall_loss)
        """
        self.model.eval()
        results = {}
        overall_loss, overall_samples = 0.0, 0
        
        console = self.exp_manager.console
        
        with torch.no_grad():
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                entity_task = progress.add_task("Testing", total=len(loaders))
                
                for subset_id, loader in loaders.items():
                    subset_loss, subset_samples = 0.0, 0
                    
                    for batch in loader:
                        forecast, batch_y, _ = self._forward_step(batch)
                        output = forecast[:, -self.args.output_len:, :]
                        loss = criterion(output, batch_y)
                        
                        batch_size = batch_y.size(0)
                        subset_loss += loss.item() * batch_size
                        subset_samples += batch_size
                    
                    avg_loss = subset_loss / subset_samples if subset_samples > 0 else 0.0
                    results[subset_id] = avg_loss
                    overall_loss += subset_loss
                    overall_samples += subset_samples
                    
                    progress.update(entity_task, advance=1)
        
        self.model.train()
        return results, overall_loss / overall_samples if overall_samples > 0 else 0.0
    
    def train(self, time_llm_config: TimeLLMTrainingConfig) -> Path:
        """
        Run full training loop.
        
        Args:
            time_llm_config: Time-LLM specific configuration
        
        Returns:
            Path to best model checkpoint
        """
        self.exp_manager.logger.info("=" * 60)
        self.exp_manager.logger.info("Time-LLM Training (PyTorch)")
        self.exp_manager.logger.info(f"  LLM Backbone: {time_llm_config.llm_backbone}")
        self.exp_manager.logger.info(f"  Quantization: {time_llm_config.quantization}")
        self.exp_manager.logger.info(f"  Epochs: {self.args.train_epochs}")
        self.exp_manager.logger.info("=" * 60)
        
        train_loader = self.data_provider.get_train(return_type='loader')
        val_loader = self.data_provider.get_val(return_type='loader')
        test_loaders = self.data_provider.get_test(return_type='loader')
        
        # Clear data buffer
        self.data_provider.data_buffer.clear()
        
        # Only optimize non-frozen parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=self.args.learning_rate)
        criterion = _select_criterion(time_llm_config.loss)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        checkpoint_dir = self.exp_manager.get_checkpoint_dir()
        best_val_loss = float('inf')
        
        for epoch in range(1, self.args.train_epochs + 1):
            train_loss, epoch_time = self._train_epoch(train_loader, optimizer, criterion, epoch)
            val_loss = self._validate(val_loader, criterion)
            
            self.exp_manager.logger.info(
                f"Epoch {epoch}/{self.args.train_epochs} | "
                f"Train: {train_loss:.7f} | Val: {val_loss:.7f} | Time: {epoch_time:.2f}s"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), checkpoint_dir / 'checkpoint.pth')
            
            early_stopping(val_loss, self.model, str(checkpoint_dir))
            if early_stopping.early_stop:
                self.exp_manager.logger.info("Early stopping triggered")
                break
            
            adjust_learning_rate(optimizer, epoch, self.args)
            self.exp_manager.update_current_epoch(epoch)
        
        # Load best model for testing
        best_model_path = checkpoint_dir / 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        
        # Final testing
        self.exp_manager.logger.info("Running final testing...")
        test_results, overall_test_loss = self._test(test_loaders, criterion)
        
        self.exp_manager.logger.info(f"Test results: {test_results}")
        self.exp_manager.logger.info(f"Overall test loss: {overall_test_loss:.7f}")
        
        with open(checkpoint_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        self.exp_manager.register_job_end(
            end_epoch=epoch, 
            status="completed",
            checkpoint_path=str(best_model_path),
            final_train_loss=train_loss, 
            final_val_loss=val_loss
        )
        
        return best_model_path


def train_time_llm_pytorch(args, exp_manager) -> Path:
    """
    Train Time-LLM model using standard PyTorch.
    
    Args:
        args: Experiment arguments with time_llm config
        exp_manager: ExperimentManager for tracking
    
    Returns:
        Path to best model checkpoint
    """
    time_llm_config = _get_time_llm_config(args)
    exp_manager.logger.info(f"Time-LLM PyTorch Training")
    
    trainer = TimeLLMPyTorchTrainer(args, exp_manager)
    return trainer.train(time_llm_config)

