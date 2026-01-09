"""
LeRet Model-Specific Training Module.

This module provides training infrastructure for LeRet (Language-Enhanced 
Retention Network for Time Series), implementing a configurable two-stage 
curriculum learning approach for both PyTorch Lightning and standard PyTorch.

Training Stages:
    Stage 1 (Pretrain): Auto-regressive pretraining
        - Uses patch_head output for reconstruction loss
        - Trains the model to reconstruct input patches
        - Captures local temporal patterns

    Stage 2 (Finetune): Forecasting finetuning  
        - Uses sequence_head output for prediction loss
        - Trains the model for the actual forecasting task
        - Leverages pretrained representations

Both stages must occur within the same experiment for consistency.

Exports:
    - train_leret_lightning: Lightning-based training
    - train_leret_pytorch: Standard PyTorch training
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import inspect
import warnings

from models import model_init
from utils.tools import adjust_learning_rate, EarlyStopping
from cli.config.model_training import LeRetTrainingConfig

# Rich imports for progress bars
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


def _patchify_input(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """
    Convert input sequence to match auto_y output format for pretraining loss.
    
    The LeRet model's patch head outputs [B, patch_num * patch_len, C].
    This method transforms the input to the same format for loss computation.
    
    Args:
        x: Input tensor [B, seq_len, C]
        patch_len: Length of each patch
        stride: Stride between patches
    
    Returns:
        Patchified tensor [B, patch_num * patch_len, C] matching auto_y shape
    """
    # Transpose: [B, seq_len, C] -> [B, C, seq_len]
    x_t = x.permute(0, 2, 1)
    
    # Unfold into patches: [B, C, patch_num, patch_len]
    x_patches = x_t.unfold(dimension=-1, size=patch_len, step=stride)
    
    # Flatten patches: [B, C, patch_num * patch_len]
    B, C, patch_num, p_len = x_patches.shape
    x_flat = x_patches.reshape(B, C, patch_num * p_len)
    
    # Transpose back: [B, patch_num * patch_len, C]
    return x_flat.permute(0, 2, 1)


def _get_pretrain_checkpoint(exp_manager, leret_config: LeRetTrainingConfig) -> Path:
    """
    Get pretrain checkpoint path with validation.
    
    Args:
        exp_manager: ExperimentManager instance
        leret_config: LeRet training configuration
    
    Returns:
        Path to pretrain checkpoint
    """
    # Check for explicit path in config
    if leret_config.pretrain_checkpoint:
        ckpt_path = Path(leret_config.pretrain_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Specified pretrain_checkpoint not found: {ckpt_path}")
        return ckpt_path
    
    # Auto-detect from same experiment (try both .ckpt and .pth)
    for ext in ['.ckpt', '.pth']:
        expected_path = exp_manager.get_checkpoint_dir() / f'pretrain_checkpoint{ext}'
        if expected_path.exists():
            return expected_path
    
    # Check job history for pretrain completion
    if not exp_manager.job_history.get('pretrain_completed'):
        raise ValueError(
            "Cannot run finetune stage: pretrain stage not completed. "
            "Set training_stage='both' or 'pretrain' first."
        )
    raise FileNotFoundError("Pretrain checkpoint not found despite pretrain_completed=True")


def _get_leret_config(args) -> LeRetTrainingConfig:
    """
    Extract and validate LeRet training config from args.
    
    Args:
        args: Experiment arguments
    
    Returns:
        LeRetTrainingConfig instance
    """
    leret_config = getattr(args, 'leret', None)
    if leret_config is None:
        return LeRetTrainingConfig()
    elif isinstance(leret_config, dict):
        return LeRetTrainingConfig(**leret_config)
    return leret_config


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
    class LeRetLightningModule(pl.LightningModule):
        """
        PyTorch Lightning module for LeRet with two-stage training support.
        
        Handles switching between pretrain and finetune loss functions
        based on the current training stage.
        """
        
        def __init__(
            self, 
            args, 
            exp_manager=None, 
            training_stage: str = "both",
            leret_config: Optional[LeRetTrainingConfig] = None
        ):
            super().__init__()
            self.args = args
            self.exp_manager = exp_manager
            self.training_stage = training_stage
            self.leret_config = leret_config or LeRetTrainingConfig()
            
            # Determine initial stage
            self.current_stage = "pretrain" if training_stage in ["pretrain", "both"] else "finetune"
            
            self.save_hyperparameters(ignore=['args', 'exp_manager', 'leret_config'])
            
            # Build the LeRet model
            self.model = model_init(self.args.model, self.args.model_config, self.args)
            
            # Loss functions for each stage
            self.pretrain_criterion = _select_criterion(self.leret_config.pretrain_loss)
            self.finetune_criterion = _select_criterion(self.leret_config.finetune_loss)
            
            # Patching parameters
            self.patch_len = getattr(args, 'patch_len', 16)
            self.stride = getattr(args, 'stride', 8)
            
            # Test tracking
            self.test_total_loss = []
            self.test_total_samples = []
            self._epoch_start_time: Optional[float] = None
        
        def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward pass through the LeRet model."""
            sample_ids, batch_x, batch_y, timestamp_x, timestamp_y, \
                batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, \
                hetero_general, hetero_channel = batch
            
            if hasattr(self.model, 'move_to_device'):
                batch_x, batch_y, *_ = self.model.move_to_device(
                    batch_x, batch_y, timestamp_x, timestamp_y, 
                    batch_x_hetero, batch_y_hetero, hetero_x_time, 
                    hetero_y_time, hetero_general, hetero_channel, self.device
                )
            else:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
            
            # Build forward kwargs
            model_params = list(inspect.signature(self.model.forward).parameters.keys())
            forward_kwargs = {'x': batch_x} if 'x' in model_params else {}
            
            forecast, auto_y = self.model(**forward_kwargs)
            return forecast, auto_y, batch_x, batch_y
        
        def training_step(self, batch, batch_idx):
            """Training step with stage-aware loss computation."""
            forecast, auto_y, batch_x, batch_y = self.forward(batch)
            
            if self.current_stage == "pretrain":
                y_auto_target = _patchify_input(batch_x, self.patch_len, self.stride)
                loss = self.pretrain_criterion(auto_y, y_auto_target)
                self.log('pretrain_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            else:
                output = forecast[:, -self.args.output_len:, :]
                loss = self.finetune_criterion(output, batch_y)
                self.log('finetune_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        
        def validation_step(self, batch, batch_idx):
            """Validation step with stage-aware loss computation."""
            forecast, auto_y, batch_x, batch_y = self.forward(batch)
            
            if self.current_stage == "pretrain":
                y_auto_target = _patchify_input(batch_x, self.patch_len, self.stride)
                loss = self.pretrain_criterion(auto_y, y_auto_target)
            else:
                output = forecast[:, -self.args.output_len:, :]
                loss = self.finetune_criterion(output, batch_y)
            
            self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
            return loss
        
        def test_step(self, batch, batch_idx, dataloader_idx=0):
            """Test step - always uses forecasting loss."""
            forecast, auto_y, batch_x, batch_y = self.forward(batch)
            output = forecast[:, -self.args.output_len:, :]
            loss = self.finetune_criterion(output, batch_y)
            
            num_samples = batch_y.numel()
            self.test_total_loss.append(loss.item() * num_samples)
            self.test_total_samples.append(num_samples)
        
        def on_test_epoch_end(self):
            total_loss = np.sum(self.test_total_loss)
            total_samples = np.sum(self.test_total_samples)
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            self.log('test_loss', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
            self.test_total_loss = []
            self.test_total_samples = []
        
        def on_train_epoch_start(self):
            if self.exp_manager and hasattr(self.exp_manager, 'gpu_monitor') and self.exp_manager.gpu_monitor:
                self.exp_manager.gpu_monitor.mark_epoch_start()
            self._epoch_start_time = time.time()
        
        def on_train_epoch_end(self):
            if self.trainer.sanity_checking:
                return
            current_epoch = self.trainer.current_epoch + 1
            if self._epoch_start_time is not None and self.exp_manager:
                epoch_time = time.time() - self._epoch_start_time
                self.exp_manager.log_file_only(
                    f"Epoch {current_epoch} ({self.current_stage}) completed in {epoch_time:.2f}s"
                )
        
        def configure_optimizers(self):
            if self.current_stage == "pretrain" and self.leret_config.pretrain_learning_rate:
                lr = self.leret_config.pretrain_learning_rate
            else:
                lr = self.args.learning_rate
            
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
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


def _run_lightning_pretrain(args, exp_manager, data_module, leret_config):
    """Run Lightning pretrain stage."""
    exp_manager.logger.info("=" * 60)
    exp_manager.logger.info("Stage 1: Auto-Regressive Pretraining (Lightning)")
    exp_manager.logger.info(f"  Epochs: {leret_config.pretrain_epochs}")
    exp_manager.logger.info("=" * 60)
    
    model = LeRetLightningModule(args, exp_manager, "pretrain", leret_config)
    checkpoint_dir = str(exp_manager.get_checkpoint_dir())
    
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='pretrain-{epoch:02d}-{val_loss:.6f}',
        save_top_k=1, monitor='val_loss', mode='min'
    )
    
    class JobHistoryCallback(pl.Callback):
        def __init__(self, em): self.em = em
        def on_train_epoch_end(self, trainer, pl_module):
            if not trainer.sanity_checking:
                # Update epoch and check for stop signals (Hyperband pruning, etc.)
                should_stop = self.em.update_current_epoch(trainer.current_epoch + 1)
                if should_stop:
                    trainer.should_stop = True
    
    trainer = pl.Trainer(
        max_epochs=leret_config.pretrain_epochs,
        accelerator='gpu' if args.use_gpu else 'cpu',
        devices=[args.gpu] if args.use_gpu and not args.use_multi_gpu else (
            args.device_ids if args.use_multi_gpu else None
        ),
        strategy='ddp' if args.use_multi_gpu else 'auto',
        callbacks=[checkpoint_cb, JobHistoryCallback(exp_manager)],
        logger=TensorBoardLogger(str(exp_manager.get_experiment_dir() / "tb_logs"), name="pretrain"),
        deterministic=True,
        precision=getattr(args, 'precision', 32),
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )
    
    trainer.fit(model, data_module)
    
    # Save pretrain checkpoint
    pretrain_ckpt_path = exp_manager.get_checkpoint_dir() / 'pretrain_checkpoint.ckpt'
    model_state = (
        torch.load(checkpoint_cb.best_model_path, map_location='cpu').get('state_dict', model.state_dict())
        if checkpoint_cb.best_model_path else model.state_dict()
    )
    
    torch.save({
        'epoch': leret_config.pretrain_epochs,
        'state_dict': model_state,
        'stage': 'pretrain',
        'experiment_id': exp_manager.experiment_id,
        'pretrain_epochs': leret_config.pretrain_epochs,
    }, pretrain_ckpt_path)
    
    exp_manager.job_history['pretrain_completed'] = True
    exp_manager.job_history['pretrain_checkpoint'] = str(pretrain_ckpt_path)
    exp_manager._save_job_history()
    
    exp_manager.logger.info(f"Stage 1 complete. Checkpoint: {pretrain_ckpt_path}")
    return pretrain_ckpt_path


def _run_lightning_finetune(args, exp_manager, data_module, leret_config, pretrain_ckpt_path=None):
    """Run Lightning finetune stage."""
    exp_manager.logger.info("=" * 60)
    exp_manager.logger.info("Stage 2: Forecasting Finetuning (Lightning)")
    exp_manager.logger.info(f"  Epochs: {args.train_epochs}")
    exp_manager.logger.info("=" * 60)
    
    if pretrain_ckpt_path is None:
        pretrain_ckpt_path = _get_pretrain_checkpoint(exp_manager, leret_config)
    
    checkpoint = torch.load(pretrain_ckpt_path, map_location='cpu')
    
    # Validate experiment ID
    ckpt_exp_id = checkpoint.get('experiment_id')
    if ckpt_exp_id and ckpt_exp_id != exp_manager.experiment_id:
        raise ValueError(f"Experiment ID mismatch: {ckpt_exp_id} vs {exp_manager.experiment_id}")
    
    model = LeRetLightningModule(args, exp_manager, "finetune", leret_config)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    checkpoint_dir = str(exp_manager.get_checkpoint_dir())
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='checkpoint-{epoch:02d}-{val_loss:.6f}',
        save_top_k=1, monitor='val_loss', mode='min', save_last=True
    )
    
    epoch_offset = checkpoint.get('pretrain_epochs', leret_config.pretrain_epochs)
    
    class JobHistoryCallback(pl.Callback):
        def __init__(self, em, offset): self.em, self.offset = em, offset
        def on_train_epoch_end(self, trainer, pl_module):
            if not trainer.sanity_checking:
                # Update epoch and check for stop signals (Hyperband pruning, etc.)
                should_stop = self.em.update_current_epoch(trainer.current_epoch + 1 + self.offset)
                if should_stop:
                    trainer.should_stop = True
    
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
            JobHistoryCallback(exp_manager, epoch_offset)
        ],
        logger=TensorBoardLogger(str(exp_manager.get_experiment_dir() / "tb_logs"), name="finetune"),
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
    
    # Save results
    if trainer.is_global_zero:
        with open(exp_manager.get_checkpoint_dir() / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
    
    total_epochs = epoch_offset + trainer.current_epoch + 1
    exp_manager.register_job_end(end_epoch=total_epochs, status="completed", checkpoint_path=best_model_path)
    exp_manager.job_history['finetune_completed'] = True
    exp_manager.job_history['stages_completed'] = ['pretrain', 'finetune']
    exp_manager._save_job_history()
    
    return Path(best_model_path)


def train_leret_lightning(args, exp_manager) -> Path:
    """
    Train LeRet model using PyTorch Lightning.
    
    Args:
        args: Experiment arguments with leret config
        exp_manager: ExperimentManager for tracking
    
    Returns:
        Path to best model checkpoint
    """
    if not HAS_LIGHTNING:
        raise ImportError("PyTorch Lightning is required for Lightning training")
    
    from data_provider.lightning_data_module import TimeSeriesDataModule
    
    leret_config = _get_leret_config(args)
    training_stage = leret_config.training_stage
    
    exp_manager.logger.info(f"LeRet Lightning Training (stage: {training_stage})")
    
    data_module = TimeSeriesDataModule(args)
    
    if training_stage == "pretrain":
        return _run_lightning_pretrain(args, exp_manager, data_module, leret_config)
    elif training_stage == "finetune":
        return _run_lightning_finetune(args, exp_manager, data_module, leret_config)
    else:  # "both"
        pretrain_ckpt = _run_lightning_pretrain(args, exp_manager, data_module, leret_config)
        return _run_lightning_finetune(args, exp_manager, data_module, leret_config, pretrain_ckpt)


# =============================================================================
# Standard PyTorch Training
# =============================================================================

class LeRetPyTorchTrainer:
    """
    Standard PyTorch trainer for LeRet with two-stage training support.
    
    Based on exp_universal.Experiment but with LeRet-specific two-stage logic.
    """
    
    def __init__(self, args, exp_manager):
        self.args = args
        self.exp_manager = exp_manager
        self.device = self._get_device()
        
        # Build model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Patching parameters
        self.patch_len = getattr(args, 'patch_len', 16)
        self.stride = getattr(args, 'stride', 8)
        
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
        """Build the LeRet model."""
        model = model_init(self.args.model, self.args.model_config, self.args)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _forward_step(self, batch):
        """Forward pass returning (forecast, auto_y, batch_x, batch_y)."""
        sample_ids, batch_x, batch_y, timestamp_x, timestamp_y, \
            batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, \
            hetero_general, hetero_channel, x_time_features, y_time_features = batch
        
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        
        # LeRet returns (forecast, auto_y)
        forecast, auto_y = self.model(x=batch_x)
        
        return forecast, auto_y, batch_x, batch_y, sample_ids
    
    def _train_epoch_pretrain(self, train_loader, optimizer, criterion, epoch):
        """Train one epoch for pretrain stage."""
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
            task = progress.add_task(f"Pretrain Epoch {epoch}", total=len(train_loader), loss=0.0)
            
            for batch in train_loader:
                optimizer.zero_grad()
                forecast, auto_y, batch_x, batch_y, _ = self._forward_step(batch)
                
                # Pretrain loss: reconstruct patches
                y_auto_target = _patchify_input(batch_x, self.patch_len, self.stride)
                loss = criterion(auto_y, y_auto_target)
                
                loss.backward()
                optimizer.step()
                
                batch_size = batch_x.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                progress.update(task, advance=1, loss=loss.item())
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        epoch_time_elapsed = time.time() - epoch_time
        
        return avg_loss, epoch_time_elapsed
    
    def _train_epoch_finetune(self, train_loader, optimizer, criterion, epoch):
        """Train one epoch for finetune stage."""
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
            task = progress.add_task(f"Finetune Epoch {epoch}", total=len(train_loader), loss=0.0)
            
            for batch in train_loader:
                optimizer.zero_grad()
                forecast, auto_y, batch_x, batch_y, _ = self._forward_step(batch)
                
                # Finetune loss: forecast prediction
                output = forecast[:, -self.args.output_len:, :]
                loss = criterion(output, batch_y)
                
                loss.backward()
                optimizer.step()
                
                batch_size = batch_x.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                progress.update(task, advance=1, loss=loss.item())
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        epoch_time_elapsed = time.time() - epoch_time
        
        return avg_loss, epoch_time_elapsed
    
    def _validate(self, loader, criterion, stage="finetune"):
        """Run validation."""
        self.model.eval()
        total_loss, total_samples = 0.0, 0
        
        with torch.no_grad():
            for batch in loader:
                forecast, auto_y, batch_x, batch_y, _ = self._forward_step(batch)
                
                if stage == "pretrain":
                    y_auto_target = _patchify_input(batch_x, self.patch_len, self.stride)
                    loss = criterion(auto_y, y_auto_target)
                else:
                    output = forecast[:, -self.args.output_len:, :]
                    loss = criterion(output, batch_y)
                
                batch_size = batch_x.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        self.model.train()
        return total_loss / total_samples if total_samples > 0 else 0.0
    
    def _test(self, loaders, criterion):
        """Run testing on all subsets."""
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
                        forecast, auto_y, batch_x, batch_y, _ = self._forward_step(batch)
                        output = forecast[:, -self.args.output_len:, :]
                        loss = criterion(output, batch_y)
                        
                        batch_size = batch_x.size(0)
                        subset_loss += loss.item() * batch_size
                        subset_samples += batch_size
                    
                    avg_loss = subset_loss / subset_samples if subset_samples > 0 else 0.0
                    results[subset_id] = avg_loss
                    overall_loss += subset_loss
                    overall_samples += subset_samples
                    
                    progress.update(entity_task, advance=1)
        
        self.model.train()
        return results, overall_loss / overall_samples if overall_samples > 0 else 0.0
    
    def run_pretrain_stage(self, leret_config: LeRetTrainingConfig) -> Path:
        """Run Stage 1: Auto-regressive pretraining."""
        self.exp_manager.logger.info("=" * 60)
        self.exp_manager.logger.info("Stage 1: Auto-Regressive Pretraining (PyTorch)")
        self.exp_manager.logger.info(f"  Epochs: {leret_config.pretrain_epochs}")
        self.exp_manager.logger.info("=" * 60)
        
        train_loader = self.data_provider.get_train(return_type='loader')
        val_loader = self.data_provider.get_val(return_type='loader')
        
        # Clear data buffer
        self.data_provider.data_buffer.clear()
        
        # Setup
        lr = leret_config.pretrain_learning_rate or self.args.learning_rate
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = _select_criterion(leret_config.pretrain_loss)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        checkpoint_dir = self.exp_manager.get_checkpoint_dir()
        best_val_loss = float('inf')
        
        for epoch in range(1, leret_config.pretrain_epochs + 1):
            train_loss, epoch_time = self._train_epoch_pretrain(train_loader, optimizer, criterion, epoch)
            val_loss = self._validate(val_loader, criterion, stage="pretrain")
            
            self.exp_manager.logger.info(
                f"Pretrain Epoch {epoch}/{leret_config.pretrain_epochs} | "
                f"Train: {train_loss:.7f} | Val: {val_loss:.7f} | Time: {epoch_time:.2f}s"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), checkpoint_dir / 'pretrain_best.pth')
            
            early_stopping(val_loss, self.model, str(checkpoint_dir))
            if early_stopping.early_stop:
                self.exp_manager.logger.info("Early stopping triggered")
                self.exp_manager.set_completion_reason("early_stopping")
                break
            
            adjust_learning_rate(optimizer, epoch, self.args)
            
            # Update epoch and check for stop signals (Hyperband pruning, etc.)
            should_stop = self.exp_manager.update_current_epoch(epoch)
            if should_stop:
                self.exp_manager.logger.info("Training stopped: external signal detected (e.g., Hyperband pruning)")
                break
        
        # Save final pretrain checkpoint with metadata
        pretrain_ckpt_path = checkpoint_dir / 'pretrain_checkpoint.pth'
        torch.save({
            'epoch': leret_config.pretrain_epochs,
            'model_state_dict': self.model.state_dict(),
            'stage': 'pretrain',
            'experiment_id': self.exp_manager.experiment_id,
            'pretrain_epochs': leret_config.pretrain_epochs,
        }, pretrain_ckpt_path)
        
        self.exp_manager.job_history['pretrain_completed'] = True
        self.exp_manager.job_history['pretrain_checkpoint'] = str(pretrain_ckpt_path)
        self.exp_manager._save_job_history()
        
        self.exp_manager.logger.info(f"Stage 1 complete. Checkpoint: {pretrain_ckpt_path}")
        return pretrain_ckpt_path
    
    def run_finetune_stage(self, leret_config: LeRetTrainingConfig, pretrain_ckpt_path=None) -> Path:
        """Run Stage 2: Forecasting finetuning."""
        self.exp_manager.logger.info("=" * 60)
        self.exp_manager.logger.info("Stage 2: Forecasting Finetuning (PyTorch)")
        self.exp_manager.logger.info(f"  Epochs: {self.args.train_epochs}")
        self.exp_manager.logger.info("=" * 60)
        
        # Load pretrain checkpoint
        if pretrain_ckpt_path is None:
            pretrain_ckpt_path = _get_pretrain_checkpoint(self.exp_manager, leret_config)
        
        checkpoint = torch.load(pretrain_ckpt_path, map_location=self.device)
        
        # Validate experiment ID
        ckpt_exp_id = checkpoint.get('experiment_id')
        if ckpt_exp_id and ckpt_exp_id != self.exp_manager.experiment_id:
            raise ValueError(f"Experiment ID mismatch: {ckpt_exp_id} vs {self.exp_manager.experiment_id}")
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        self.exp_manager.logger.info(f"Loaded pretrain checkpoint: {pretrain_ckpt_path}")
        
        train_loader = self.data_provider.get_train(return_type='loader')
        val_loader = self.data_provider.get_val(return_type='loader')
        test_loaders = self.data_provider.get_test(return_type='loader')
        
        self.data_provider.data_buffer.clear()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = _select_criterion(leret_config.finetune_loss)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        checkpoint_dir = self.exp_manager.get_checkpoint_dir()
        epoch_offset = checkpoint.get('pretrain_epochs', leret_config.pretrain_epochs)
        best_val_loss = float('inf')
        
        for epoch in range(1, self.args.train_epochs + 1):
            train_loss, epoch_time = self._train_epoch_finetune(train_loader, optimizer, criterion, epoch)
            val_loss = self._validate(val_loader, criterion, stage="finetune")
            
            total_epoch = epoch_offset + epoch
            self.exp_manager.logger.info(
                f"Finetune Epoch {epoch}/{self.args.train_epochs} (Total: {total_epoch}) | "
                f"Train: {train_loss:.7f} | Val: {val_loss:.7f} | Time: {epoch_time:.2f}s"
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), checkpoint_dir / 'checkpoint.pth')
            
            early_stopping(val_loss, self.model, str(checkpoint_dir))
            if early_stopping.early_stop:
                self.exp_manager.logger.info("Early stopping triggered")
                self.exp_manager.set_completion_reason("early_stopping")
                break
            
            adjust_learning_rate(optimizer, epoch, self.args)
            
            # Update epoch and check for stop signals (Hyperband pruning, etc.)
            should_stop = self.exp_manager.update_current_epoch(total_epoch)
            if should_stop:
                self.exp_manager.logger.info("Training stopped: external signal detected (e.g., Hyperband pruning)")
                break
        
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
        
        total_epochs = epoch_offset + epoch
        self.exp_manager.register_job_end(
            end_epoch=total_epochs, status="completed",
            checkpoint_path=str(best_model_path),
            final_train_loss=train_loss, final_val_loss=val_loss
        )
        
        self.exp_manager.job_history['finetune_completed'] = True
        self.exp_manager.job_history['stages_completed'] = ['pretrain', 'finetune']
        self.exp_manager._save_job_history()
        
        return best_model_path


def train_leret_pytorch(args, exp_manager) -> Path:
    """
    Train LeRet model using standard PyTorch.
    
    Args:
        args: Experiment arguments with leret config
        exp_manager: ExperimentManager for tracking
    
    Returns:
        Path to best model checkpoint
    """
    leret_config = _get_leret_config(args)
    training_stage = leret_config.training_stage
    
    exp_manager.logger.info(f"LeRet PyTorch Training (stage: {training_stage})")
    
    trainer = LeRetPyTorchTrainer(args, exp_manager)
    
    if training_stage == "pretrain":
        return trainer.run_pretrain_stage(leret_config)
    elif training_stage == "finetune":
        return trainer.run_finetune_stage(leret_config)
    else:  # "both"
        pretrain_ckpt = trainer.run_pretrain_stage(leret_config)
        return trainer.run_finetune_stage(leret_config, pretrain_ckpt)

