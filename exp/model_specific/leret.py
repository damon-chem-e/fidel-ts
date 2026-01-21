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


def _determine_resume_stage(exp_manager, leret_config: LeRetTrainingConfig) -> Tuple[Optional[str], int, Optional[Path]]:
    """
    Determine which stage to resume from and the starting epoch.
    
    This function checks job_history and available checkpoints to determine:
    - Which stage (pretrain/finetune) to resume
    - What epoch to start from
    - Which checkpoint to load
    
    Args:
        exp_manager: ExperimentManager instance
        leret_config: LeRet training configuration
    
    Returns:
        Tuple of (stage, start_epoch, checkpoint_path):
        - stage: "pretrain", "finetune", or None (if no resume needed)
        - start_epoch: 1-indexed epoch to start from (within the stage)
        - checkpoint_path: Path to checkpoint to load, or None
    """
    # Get resume info from experiment manager
    resume_info = exp_manager.get_resume_info()
    if not resume_info:
        # No resume needed - start fresh
        return None, 1, None
    
    job_history = exp_manager.job_history
    pretrain_completed = job_history.get('pretrain_completed', False)
    finetune_completed = job_history.get('finetune_completed', False)
    
    # If both stages are complete, no resume needed
    if finetune_completed:
        exp_manager.logger.info("Training already complete (both pretrain and finetune finished)")
        return None, 1, None
    
    # Get the last completed epoch from resume_info
    last_epoch = resume_info.get('last_epoch', 0)
    checkpoint_path = resume_info.get('checkpoint_path')
    
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            exp_manager.logger.warning(f"Checkpoint path from resume_info does not exist: {checkpoint_path}")
            checkpoint_path = None
    
    # Determine which stage we're in based on epoch and completion flags
    pretrain_epochs = leret_config.pretrain_epochs
    
    if not pretrain_completed:
        # Still in pretrain stage
        # last_epoch is 1-indexed, so start_epoch = last_epoch + 1 (but within pretrain range)
        start_epoch = min(last_epoch + 1, pretrain_epochs)
        if start_epoch > pretrain_epochs:
            # This shouldn't happen, but handle it gracefully
            start_epoch = 1
            
        # Look for pretrain-specific checkpoint if general checkpoint not found
        if checkpoint_path is None:
            checkpoint_dir = exp_manager.get_checkpoint_dir()
            # Try pretrain_best.pth first, then any .pth file
            pretrain_best = checkpoint_dir / 'pretrain_best.pth'
            if pretrain_best.exists():
                checkpoint_path = pretrain_best
            else:
                # Try checkpoint.pth from EarlyStopping
                general_ckpt = checkpoint_dir / 'checkpoint.pth'
                if general_ckpt.exists():
                    checkpoint_path = general_ckpt
        
        exp_manager.logger.info(
            f"Resuming PRETRAIN stage from epoch {start_epoch}/{pretrain_epochs} "
            f"(last completed: {last_epoch})"
        )
        return "pretrain", start_epoch, checkpoint_path
    
    else:
        # Pretrain is done, resume finetune stage
        # Finetune epochs are counted after pretrain, so adjust
        finetune_start_epoch = last_epoch - pretrain_epochs + 1
        finetune_start_epoch = max(1, finetune_start_epoch)  # At least 1
        
        # Look for finetune checkpoint
        if checkpoint_path is None:
            checkpoint_dir = exp_manager.get_checkpoint_dir()
            # Try checkpoint.pth (finetune best) first
            finetune_ckpt = checkpoint_dir / 'checkpoint.pth'
            if finetune_ckpt.exists():
                checkpoint_path = finetune_ckpt
        
        exp_manager.logger.info(
            f"Resuming FINETUNE stage from epoch {finetune_start_epoch} "
            f"(total epoch: {last_epoch + 1}, pretrain_epochs: {pretrain_epochs})"
        )
        return "finetune", finetune_start_epoch, checkpoint_path


def _load_checkpoint_for_resume(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    exp_manager = None
) -> None:
    """
    Load checkpoint for resuming training.
    
    Handles both model state dict and optimizer state, with support for
    torch.compile checkpoints (strips '_orig_mod.' prefix).
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint to
        optimizer: Optional optimizer to load state into
        exp_manager: Optional ExperimentManager for logging
    """
    logger = exp_manager.logger if exp_manager else None
    
    if logger:
        logger.info(f"Loading checkpoint for resume: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract state dict (handle different checkpoint formats)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and not any(k in checkpoint for k in ['epoch', 'optimizer_state_dict']):
        # Checkpoint is directly the state dict
        state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Handle torch.compile checkpoints (strip '_orig_mod.' prefix)
    if isinstance(state_dict, dict) and any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        if logger:
            logger.info("Detected torch.compile checkpoint - stripping '_orig_mod.' prefix")
        state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
    
    # Load model weights
    model.load_state_dict(state_dict, strict=False)
    
    # Load optimizer state if available
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if logger:
                logger.info("Loaded optimizer state from checkpoint")
        except Exception as e:
            if logger:
                logger.warning(f"Could not load optimizer state: {e}")
    
    if logger:
        ckpt_epoch = checkpoint.get('epoch', 'unknown')
        ckpt_stage = checkpoint.get('stage', 'unknown')
        logger.info(f"Checkpoint loaded (epoch: {ckpt_epoch}, stage: {ckpt_stage})")


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
            # Note: enc_in inference for Lightning should happen in _run_lightning_* functions
            # before creating the Lightning module
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
            
            # Extract and process text embeddings if available
            language_embeddings = None
            if batch_y_hetero is not None:
                # Convert to tensor if needed
                if not isinstance(batch_y_hetero, torch.Tensor):
                    batch_y_hetero = torch.tensor(batch_y_hetero, dtype=torch.float32)
                
                batch_y_hetero = batch_y_hetero.to(self.device)
                
                # Aggregate y_hetero from [B, pred_len, num_items, text_dim] to [text_num, language_dim]
                # Strategy: Mean pooling across batch, pred_len, and num_items to get single aggregated embedding
                # batch_y_hetero: [B, pred_len, num_items, text_dim]
                if batch_y_hetero.numel() > 0:
                    # Flatten batch, pred_len, and num_items dimensions, then take mean
                    # Result: [text_dim] -> [1, text_dim] to match expected format
                    language_embeddings = batch_y_hetero.mean(dim=(0, 1, 2))  # [text_dim]
                    language_embeddings = language_embeddings.unsqueeze(0)  # [1, text_dim]
            
            # Build forward kwargs
            model_params = list(inspect.signature(self.model.forward).parameters.keys())
            forward_kwargs = {'x': batch_x} if 'x' in model_params else {}
            if 'language_embeddings' in model_params and language_embeddings is not None:
                forward_kwargs['language_embeddings'] = language_embeddings
            
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


def _run_lightning_pretrain(
    args, 
    exp_manager, 
    data_module, 
    leret_config,
    start_epoch: int = 1,
    resume_checkpoint: Optional[Path] = None
):
    """
    Run Lightning pretrain stage.
    
    Args:
        args: Experiment arguments
        exp_manager: ExperimentManager for tracking
        data_module: Lightning data module
        leret_config: LeRet training configuration
        start_epoch: Epoch to start from (1-indexed, for resume)
        resume_checkpoint: Optional checkpoint path to load for resume
    
    Returns:
        Path to pretrain checkpoint
    """
    exp_manager.logger.info("=" * 60)
    exp_manager.logger.info("Stage 1: Auto-Regressive Pretraining (Lightning)")
    exp_manager.logger.info(f"  Epochs: {leret_config.pretrain_epochs}")
    if start_epoch > 1:
        exp_manager.logger.info(f"  Resuming from epoch: {start_epoch}")
    exp_manager.logger.info("=" * 60)
    
    # Ensure enc_in is set before creating model
    _ensure_enc_in_for_lightning(args, exp_manager, data_module)
    
    model = LeRetLightningModule(args, exp_manager, "pretrain", leret_config)
    checkpoint_dir = str(exp_manager.get_checkpoint_dir())
    
    # Load checkpoint if resuming
    ckpt_path_for_resume = None
    if resume_checkpoint is not None and resume_checkpoint.exists():
        exp_manager.logger.info(f"Loading checkpoint for resume: {resume_checkpoint}")
        # Lightning can resume from checkpoint directly via ckpt_path in trainer.fit()
        ckpt_path_for_resume = str(resume_checkpoint)
    
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
    
    # Fit with optional checkpoint resume
    trainer.fit(model, data_module, ckpt_path=ckpt_path_for_resume)
    
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


def _ensure_enc_in_for_lightning(args, exp_manager, data_module):
    """Ensure enc_in is set for Lightning training by inferring from data_module."""
    # Check if enc_in is already set in model_config
    if hasattr(args, 'model_config') and hasattr(args.model_config, 'enc_in'):
        if args.model_config.enc_in is not None:
            return  # enc_in already set
    
    # Try to get from data_config.input_channel (must be not None)
    if (hasattr(args, 'data_config') and 
        hasattr(args.data_config, 'input_channel') and 
        args.data_config.input_channel is not None):
        args.model_config.enc_in = args.data_config.input_channel
        if exp_manager:
            exp_manager.logger.info(f"Inferred enc_in={args.data_config.input_channel} from data_config.input_channel")
        return
    
    # Infer from first batch of training data
    try:
        train_loader = data_module.train_dataloader()
        # Get first batch to determine channel count
        for batch in train_loader:
            sample_ids, batch_x, batch_y, *_ = batch
            enc_in = batch_x.shape[-1]  # Last dimension is number of channels
            args.model_config.enc_in = int(enc_in)
            if exp_manager:
                exp_manager.logger.info(f"Inferred enc_in={enc_in} from first training batch (shape: {batch_x.shape})")
            break
    except Exception as e:
        # If inference fails, raise informative error
        raise ValueError(
            f"enc_in must be provided in model_config_overrides. "
            f"Could not infer from data: {e}. "
            f"Example: model_config_overrides: {{enc_in: 7}}"
        )


def _run_lightning_finetune(
    args, 
    exp_manager, 
    data_module, 
    leret_config, 
    pretrain_ckpt_path: Optional[Path] = None,
    start_epoch: int = 1,
    resume_checkpoint: Optional[Path] = None
):
    """
    Run Lightning finetune stage.
    
    Args:
        args: Experiment arguments
        exp_manager: ExperimentManager for tracking
        data_module: Lightning data module
        leret_config: LeRet training configuration
        pretrain_ckpt_path: Path to pretrain checkpoint (for initial weights if not resuming)
        start_epoch: Epoch to start from within finetune stage (1-indexed, for resume)
        resume_checkpoint: Optional finetune checkpoint to load for resume
    
    Returns:
        Path to best finetune checkpoint
    """
    exp_manager.logger.info("=" * 60)
    exp_manager.logger.info("Stage 2: Forecasting Finetuning (Lightning)")
    exp_manager.logger.info(f"  Epochs: {args.train_epochs}")
    if start_epoch > 1:
        exp_manager.logger.info(f"  Resuming from finetune epoch: {start_epoch}")
    exp_manager.logger.info("=" * 60)
    
    epoch_offset = leret_config.pretrain_epochs
    
    # Determine which checkpoint to load for model initialization
    ckpt_path_for_resume = None
    
    if resume_checkpoint is not None and resume_checkpoint.exists():
        # Resuming finetune - Lightning can resume directly from checkpoint
        exp_manager.logger.info(f"Resuming from finetune checkpoint: {resume_checkpoint}")
        ckpt_path_for_resume = str(resume_checkpoint)
        # Load for epoch offset info if available
        checkpoint = torch.load(resume_checkpoint, map_location='cpu')
    else:
        # Starting finetune fresh - load pretrain checkpoint for initial weights
        if pretrain_ckpt_path is None:
            pretrain_ckpt_path = _get_pretrain_checkpoint(exp_manager, leret_config)
        
        exp_manager.logger.info(f"Loading pretrain checkpoint: {pretrain_ckpt_path}")
        checkpoint = torch.load(pretrain_ckpt_path, map_location='cpu')
    
    # Validate experiment ID
    ckpt_exp_id = checkpoint.get('experiment_id')
    if ckpt_exp_id and ckpt_exp_id != exp_manager.experiment_id:
        raise ValueError(f"Experiment ID mismatch: {ckpt_exp_id} vs {exp_manager.experiment_id}")
    
    # Ensure enc_in is set before creating model
    _ensure_enc_in_for_lightning(args, exp_manager, data_module)
    
    model = LeRetLightningModule(args, exp_manager, "finetune", leret_config)
    
    # Only manually load state dict if NOT resuming (Lightning handles resume automatically)
    if ckpt_path_for_resume is None:
        # Extract state dict and handle torch.compile checkpoints
        state_dict = checkpoint.get('state_dict', checkpoint)
        if isinstance(state_dict, dict) and any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            exp_manager.logger.info("Detected torch.compile checkpoint - stripping '_orig_mod.' prefix")
            state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
    
    checkpoint_dir = str(exp_manager.get_checkpoint_dir())
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='checkpoint-{epoch:02d}-{val_loss:.6f}',
        save_top_k=1, monitor='val_loss', mode='min', save_last=True
    )
    
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
    
    # Fit with optional checkpoint resume
    trainer.fit(model, data_module, ckpt_path=ckpt_path_for_resume)
    
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
    
    Handles resume logic for LeRet's two-stage training:
    - If resuming from pretrain stage, continues pretrain then finetune
    - If resuming from finetune stage, skips pretrain and continues finetune
    - If no resume needed, runs stages according to training_stage config
    
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
    
    exp_manager.logger.info(f"LeRet Lightning Training (configured stage: {training_stage})")
    
    # Check for resume
    resume_stage, start_epoch, resume_checkpoint = _determine_resume_stage(exp_manager, leret_config)
    
    data_module = TimeSeriesDataModule(args)
    
    # Handle resume scenarios
    if resume_stage is not None:
        exp_manager.logger.info(f"Resume detected: stage={resume_stage}, start_epoch={start_epoch}")
        
        if resume_stage == "pretrain":
            # Resume pretrain, then run finetune if training_stage is "both"
            pretrain_ckpt = _run_lightning_pretrain(
                args, exp_manager, data_module, leret_config,
                start_epoch=start_epoch,
                resume_checkpoint=resume_checkpoint
            )
            if training_stage == "both":
                return _run_lightning_finetune(args, exp_manager, data_module, leret_config, pretrain_ckpt)
            return pretrain_ckpt
        
        elif resume_stage == "finetune":
            # Resume finetune (pretrain already done)
            return _run_lightning_finetune(
                args, exp_manager, data_module, leret_config,
                pretrain_ckpt_path=None,  # Will auto-detect from job_history
                start_epoch=start_epoch,
                resume_checkpoint=resume_checkpoint
            )
    
    # No resume - run stages according to config
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
        
        # Data provider (needed to determine enc_in if not provided)
        from data_provider.data_factory import Data_Provider
        console = exp_manager.get_console() if exp_manager else None
        self.data_provider = Data_Provider(args, buffer=(not args.disable_buffer), console=console)
        
        # Determine enc_in from data if not provided in config
        self._ensure_enc_in_set()
        
        # Build model (after enc_in is determined)
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Patching parameters
        self.patch_len = getattr(args, 'patch_len', 16)
        self.stride = getattr(args, 'stride', 8)
    
    def _get_device(self):
        """Get the training device."""
        if self.args.use_gpu and torch.cuda.is_available():
            return torch.device(f'cuda:{self.args.gpu}')
        return torch.device('cpu')
    
    def _ensure_enc_in_set(self):
        """Ensure enc_in is set in model_config by inferring from data if needed."""
        # Check if enc_in is already set in model_config
        if hasattr(self.args, 'model_config') and hasattr(self.args.model_config, 'enc_in'):
            if self.args.model_config.enc_in is not None:
                return  # enc_in already set
        
        # Try to get from data_config.input_channel (must be not None)
        if (hasattr(self.args, 'data_config') and 
            hasattr(self.args.data_config, 'input_channel') and 
            self.args.data_config.input_channel is not None):
            self.args.model_config.enc_in = self.args.data_config.input_channel
            if self.exp_manager:
                self.exp_manager.logger.info(f"Inferred enc_in={self.args.data_config.input_channel} from data_config.input_channel")
            return
        
        # Infer from first batch of training data
        try:
            train_loader = self.data_provider.get_train(return_type='loader')
            # Get first batch to determine channel count
            for batch in train_loader:
                sample_ids, batch_x, batch_y, *_ = batch
                enc_in = batch_x.shape[-1]  # Last dimension is number of channels
                self.args.model_config.enc_in = int(enc_in)
                if self.exp_manager:
                    self.exp_manager.logger.info(f"Inferred enc_in={enc_in} from first training batch (shape: {batch_x.shape})")
                break
        except Exception as e:
            # If inference fails, raise informative error
            raise ValueError(
                f"enc_in must be provided in model_config_overrides. "
                f"Could not infer from data: {e}. "
                f"Example: model_config_overrides: {{enc_in: 7}}"
            )
    
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
        
        # Extract and process text embeddings if available
        language_embeddings = None
        if batch_y_hetero is not None:
            # Convert to tensor if needed
            if not isinstance(batch_y_hetero, torch.Tensor):
                batch_y_hetero = torch.tensor(batch_y_hetero, dtype=torch.float32)
            
            batch_y_hetero = batch_y_hetero.to(self.device)
            
            # Aggregate y_hetero from [B, pred_len, num_items, text_dim] to [text_num, language_dim]
            # Strategy: Mean pooling across batch, pred_len, and num_items to get single aggregated embedding
            # batch_y_hetero: [B, pred_len, num_items, text_dim]
            if batch_y_hetero.numel() > 0:
                # Flatten batch, pred_len, and num_items dimensions, then take mean
                # Result: [text_dim] -> [1, text_dim] to match expected format
                language_embeddings = batch_y_hetero.mean(dim=(0, 1, 2))  # [text_dim]
                language_embeddings = language_embeddings.unsqueeze(0)  # [1, text_dim]
        
        # LeRet returns (forecast, auto_y)
        forecast, auto_y = self.model(x=batch_x, language_embeddings=language_embeddings)
        
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
    
    def run_pretrain_stage(self, leret_config: LeRetTrainingConfig, start_epoch: int = 1, resume_checkpoint: Optional[Path] = None) -> Path:
        """
        Run Stage 1: Auto-regressive pretraining.
        
        Args:
            leret_config: LeRet training configuration
            start_epoch: Epoch to start from (1-indexed, for resume)
            resume_checkpoint: Optional checkpoint path to load for resume
        
        Returns:
            Path to pretrain checkpoint
        """
        self.exp_manager.logger.info("=" * 60)
        self.exp_manager.logger.info("Stage 1: Auto-Regressive Pretraining (PyTorch)")
        self.exp_manager.logger.info(f"  Epochs: {leret_config.pretrain_epochs}")
        if start_epoch > 1:
            self.exp_manager.logger.info(f"  Resuming from epoch: {start_epoch}")
        self.exp_manager.logger.info("=" * 60)
        
        train_loader = self.data_provider.get_train(return_type='loader')
        val_loader = self.data_provider.get_val(return_type='loader')
        
        # Clear data buffer
        self.data_provider.data_buffer.clear()
        
        # Setup optimizer and criterion
        lr = leret_config.pretrain_learning_rate or self.args.learning_rate
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = _select_criterion(leret_config.pretrain_loss)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        checkpoint_dir = self.exp_manager.get_checkpoint_dir()
        best_val_loss = float('inf')
        
        # Load checkpoint if resuming
        if resume_checkpoint is not None and resume_checkpoint.exists():
            _load_checkpoint_for_resume(
                self.model, resume_checkpoint, self.device, 
                optimizer=optimizer, exp_manager=self.exp_manager
            )
            # Adjust learning rate to match resumed epoch
            for ep in range(1, start_epoch):
                adjust_learning_rate(optimizer, ep, self.args)
        
        # Training loop - start from start_epoch instead of 1
        for epoch in range(start_epoch, leret_config.pretrain_epochs + 1):
            train_loss, epoch_time = self._train_epoch_pretrain(train_loader, optimizer, criterion, epoch)
            val_loss = self._validate(val_loader, criterion, stage="pretrain")
            
            self.exp_manager.logger.info(
                f"Pretrain Epoch {epoch}/{leret_config.pretrain_epochs} | "
                f"Train: {train_loss:.7f} | Val: {val_loss:.7f} | Time: {epoch_time:.2f}s"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'stage': 'pretrain',
                    'val_loss': val_loss,
                    'experiment_id': self.exp_manager.experiment_id,
                }, checkpoint_dir / 'pretrain_best.pth')
            
            early_stopping(val_loss, self.model, str(checkpoint_dir), epoch=epoch)
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
            'optimizer_state_dict': optimizer.state_dict(),
            'stage': 'pretrain',
            'experiment_id': self.exp_manager.experiment_id,
            'pretrain_epochs': leret_config.pretrain_epochs,
        }, pretrain_ckpt_path)
        
        self.exp_manager.job_history['pretrain_completed'] = True
        self.exp_manager.job_history['pretrain_checkpoint'] = str(pretrain_ckpt_path)
        self.exp_manager._save_job_history()
        
        self.exp_manager.logger.info(f"Stage 1 complete. Checkpoint: {pretrain_ckpt_path}")
        return pretrain_ckpt_path
    
    def run_finetune_stage(
        self, 
        leret_config: LeRetTrainingConfig, 
        pretrain_ckpt_path: Optional[Path] = None,
        start_epoch: int = 1,
        resume_checkpoint: Optional[Path] = None
    ) -> Path:
        """
        Run Stage 2: Forecasting finetuning.
        
        Args:
            leret_config: LeRet training configuration
            pretrain_ckpt_path: Path to pretrain checkpoint (for loading initial weights)
            start_epoch: Epoch to start from within finetune stage (1-indexed, for resume)
            resume_checkpoint: Optional finetune checkpoint to load for resume
        
        Returns:
            Path to best finetune checkpoint
        """
        self.exp_manager.logger.info("=" * 60)
        self.exp_manager.logger.info("Stage 2: Forecasting Finetuning (PyTorch)")
        self.exp_manager.logger.info(f"  Epochs: {self.args.train_epochs}")
        if start_epoch > 1:
            self.exp_manager.logger.info(f"  Resuming from finetune epoch: {start_epoch}")
        self.exp_manager.logger.info("=" * 60)
        
        # Get epoch offset from pretrain
        epoch_offset = leret_config.pretrain_epochs
        
        # Determine which checkpoint to load for model initialization
        if resume_checkpoint is not None and resume_checkpoint.exists():
            # Resuming finetune - load finetune checkpoint
            self.exp_manager.logger.info(f"Loading finetune checkpoint for resume: {resume_checkpoint}")
            checkpoint = torch.load(resume_checkpoint, map_location=self.device)
        else:
            # Starting finetune fresh - load pretrain checkpoint
            if pretrain_ckpt_path is None:
                pretrain_ckpt_path = _get_pretrain_checkpoint(self.exp_manager, leret_config)
            
            self.exp_manager.logger.info(f"Loading pretrain checkpoint: {pretrain_ckpt_path}")
            checkpoint = torch.load(pretrain_ckpt_path, map_location=self.device)
        
        # Validate experiment ID
        ckpt_exp_id = checkpoint.get('experiment_id')
        if ckpt_exp_id and ckpt_exp_id != self.exp_manager.experiment_id:
            raise ValueError(f"Experiment ID mismatch: {ckpt_exp_id} vs {self.exp_manager.experiment_id}")
        
        # Extract state dict and handle torch.compile checkpoints
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Checkpoint is directly the state dict
            state_dict = checkpoint
        
        if isinstance(state_dict, dict) and any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            self.exp_manager.logger.info("Detected torch.compile checkpoint - stripping '_orig_mod.' prefix")
            state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
        
        # Load weights
        self.model.load_state_dict(state_dict, strict=False)
        
        train_loader = self.data_provider.get_train(return_type='loader')
        val_loader = self.data_provider.get_val(return_type='loader')
        test_loaders = self.data_provider.get_test(return_type='loader')
        
        self.data_provider.data_buffer.clear()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = _select_criterion(leret_config.finetune_loss)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        checkpoint_dir = self.exp_manager.get_checkpoint_dir()
        best_val_loss = float('inf')
        
        # Load optimizer state if resuming from finetune checkpoint
        if resume_checkpoint is not None and resume_checkpoint.exists():
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.exp_manager.logger.info("Loaded optimizer state from finetune checkpoint")
                except Exception as e:
                    self.exp_manager.logger.warning(f"Could not load optimizer state: {e}")
            
            # Adjust learning rate to match resumed epoch
            for ep in range(1, start_epoch):
                adjust_learning_rate(optimizer, ep, self.args)
        
        # Track final values for job end
        train_loss = 0.0
        val_loss = 0.0
        final_epoch = start_epoch
        
        # Training loop - start from start_epoch
        for epoch in range(start_epoch, self.args.train_epochs + 1):
            train_loss, epoch_time = self._train_epoch_finetune(train_loader, optimizer, criterion, epoch)
            val_loss = self._validate(val_loader, criterion, stage="finetune")
            final_epoch = epoch
            
            total_epoch = epoch_offset + epoch
            self.exp_manager.logger.info(
                f"Finetune Epoch {epoch}/{self.args.train_epochs} (Total: {total_epoch}) | "
                f"Train: {train_loss:.7f} | Val: {val_loss:.7f} | Time: {epoch_time:.2f}s"
            )
            
            # Save checkpoint with full state for resume
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'total_epoch': total_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'stage': 'finetune',
                    'val_loss': val_loss,
                    'experiment_id': self.exp_manager.experiment_id,
                    'pretrain_epochs': epoch_offset,
                }, checkpoint_dir / 'checkpoint.pth')
            
            early_stopping(val_loss, self.model, str(checkpoint_dir), epoch=epoch)
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
        if best_model_path.exists():
            ckpt = torch.load(best_model_path, map_location=self.device)
            if 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            else:
                state_dict = ckpt
            
            # Handle torch.compile checkpoints
            if isinstance(state_dict, dict) and any(key.startswith('_orig_mod.') for key in state_dict.keys()):
                self.exp_manager.logger.info("Detected torch.compile checkpoint - stripping '_orig_mod.' prefix")
                state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
            
            self.model.load_state_dict(state_dict)
        
        # Final testing
        self.exp_manager.logger.info("Running final testing...")
        test_results, overall_test_loss = self._test(test_loaders, criterion)
        
        self.exp_manager.logger.info(f"Test results: {test_results}")
        self.exp_manager.logger.info(f"Overall test loss: {overall_test_loss:.7f}")
        
        with open(checkpoint_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        total_epochs = epoch_offset + final_epoch
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
    
    Handles resume logic for LeRet's two-stage training:
    - If resuming from pretrain stage, continues pretrain then finetune
    - If resuming from finetune stage, skips pretrain and continues finetune
    - If no resume needed, runs stages according to training_stage config
    
    Args:
        args: Experiment arguments with leret config
        exp_manager: ExperimentManager for tracking
    
    Returns:
        Path to best model checkpoint
    """
    leret_config = _get_leret_config(args)
    training_stage = leret_config.training_stage
    
    exp_manager.logger.info(f"LeRet PyTorch Training (configured stage: {training_stage})")
    
    # Check for resume
    resume_stage, start_epoch, resume_checkpoint = _determine_resume_stage(exp_manager, leret_config)
    
    trainer = LeRetPyTorchTrainer(args, exp_manager)
    
    # Handle resume scenarios
    if resume_stage is not None:
        exp_manager.logger.info(f"Resume detected: stage={resume_stage}, start_epoch={start_epoch}")
        
        if resume_stage == "pretrain":
            # Resume pretrain, then run finetune if training_stage is "both"
            pretrain_ckpt = trainer.run_pretrain_stage(
                leret_config, 
                start_epoch=start_epoch, 
                resume_checkpoint=resume_checkpoint
            )
            if training_stage == "both":
                return trainer.run_finetune_stage(leret_config, pretrain_ckpt)
            return pretrain_ckpt
        
        elif resume_stage == "finetune":
            # Resume finetune (pretrain already done)
            return trainer.run_finetune_stage(
                leret_config,
                pretrain_ckpt_path=None,  # Will auto-detect from job_history
                start_epoch=start_epoch,
                resume_checkpoint=resume_checkpoint
            )
    
    # No resume - run stages according to config
    if training_stage == "pretrain":
        return trainer.run_pretrain_stage(leret_config)
    elif training_stage == "finetune":
        return trainer.run_finetune_stage(leret_config)
    else:  # "both"
        pretrain_ckpt = trainer.run_pretrain_stage(leret_config)
        return trainer.run_finetune_stage(leret_config, pretrain_ckpt)

