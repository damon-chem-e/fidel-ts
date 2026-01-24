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
            TextColumn("val: {task.fields[val_loss_str]}"),
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
            val_loss_str="n/a",
            speed=0.0,
            eta=0.0,
            eta_formatted="0.0s"
        )
        
        return progress, task, logger, console

    def _filter_params_with_grad(self, params):
        """
        Filter parameters to those that have gradients available.
        
        Args:
            params: Iterable of parameters to filter.
        
        Returns:
            list: Parameters with non-None gradients.
        """
        # Step 1: guard against None inputs.
        if params is None:
            return []
        # Step 2: keep only parameters with gradients.
        return [param for param in params if param is not None and param.grad is not None]

    def _get_named_param_groups(self, model_optim):
        """
        Collect named optimizer parameter groups plus a total group.
        
        Args:
            model_optim: Optimizer providing parameter groups.
        
        Returns:
            dict: Mapping of group name -> list of parameters.
        """
        # Step 1: initialize storage for named groups.
        group_params = {}
        # Step 2: pull named groups from the optimizer, or synthesize names.
        for index, group in enumerate(getattr(model_optim, 'param_groups', [])):
            group_name = group.get('name', f'group_{index}')
            group_params[group_name] = list(group.get('params', []))
        # Step 3: add a total group for the full model.
        group_params['total'] = list(self.model.parameters())
        return group_params

    def _compute_group_norms(self, group_params):
        """
        Compute gradient norm for each parameter group without clipping.
        
        Args:
            group_params: Mapping of group name -> list of parameters.
        
        Returns:
            dict: Mapping of group name -> gradient norm (float).
        """
        # Step 1: initialize output dictionary.
        group_norms = {}
        # Step 2: compute norms for each group using max_norm=inf.
        for group_name, params in group_params.items():
            params_with_grad = self._filter_params_with_grad(params)
            if not params_with_grad:
                group_norms[group_name] = 0.0
                continue
            group_norms[group_name] = torch.nn.utils.clip_grad_norm_(
                params_with_grad,
                max_norm=float('inf')
            ).item()
        return group_norms

    def _compute_params_norm(self, params):
        """
        Compute gradient norm for a specific parameter list without clipping.
        
        Args:
            params: List of parameters to measure.
        
        Returns:
            float: L2 norm of gradients for the provided params.
        """
        # Step 1: keep only parameters with gradients.
        params_with_grad = self._filter_params_with_grad(params)
        # Step 2: return zero for empty parameter sets.
        if not params_with_grad:
            return 0.0
        # Step 3: compute L2 norm with max_norm=inf.
        return torch.nn.utils.clip_grad_norm_(
            params_with_grad,
            max_norm=float('inf')
        ).item()

    def _get_non_text_film_params(self, text_film_params):
        """
        Collect all model parameters that are not part of the text-FiLM group.
        
        Args:
            text_film_params: List of parameters belonging to the text-FiLM group.
        
        Returns:
            list: Parameters excluding text-FiLM parameters.
        """
        # Step 1: build an identity set for fast exclusion.
        text_film_ids = {id(param) for param in text_film_params}
        # Step 2: collect all parameters not in the text-FiLM group.
        return [param for param in self.model.parameters() if id(param) not in text_film_ids]

    def _get_grad_clip_mode(self):
        """
        Resolve gradient clipping mode from model configuration.
        
        Returns:
            str: Normalized clip mode ("fixed", "ema", or "none").
        """
        # Step 1: read the configured mode (default to "fixed" for legacy behavior).
        grad_clip_mode = getattr(self.args.model_config, 'grad_clip_mode', 'fixed')
        # Step 2: normalize the mode to lowercase string.
        grad_clip_mode = str(grad_clip_mode).strip().lower()
        # Step 3: validate and fallback to "fixed" when unsupported.
        if grad_clip_mode not in {"fixed", "ema", "none"}:
            return "fixed"
        return grad_clip_mode

    def _get_grad_clip_ema_settings(self, group_name):
        """
        Resolve EMA clipping settings for a specific group.
        
        Args:
            group_name: Group identifier for overrides (e.g., "text_film", "default").
        
        Returns:
            dict: EMA settings with keys: beta, mult, warmup_steps, warmup_max_norm, min, max.
        """
        # Step 1: load default EMA settings from model config.
        defaults = {
            "beta": float(getattr(self.args.model_config, 'grad_clip_ema_beta', 0.95)),
            "mult": float(getattr(self.args.model_config, 'grad_clip_ema_mult', 2.0)),
            "warmup_steps": int(getattr(self.args.model_config, 'grad_clip_ema_warmup_steps', 0)),
            "warmup_max_norm": getattr(self.args.model_config, 'grad_clip_ema_warmup_max_norm', None),
            "min": getattr(self.args.model_config, 'grad_clip_ema_min', None),
            "max": getattr(self.args.model_config, 'grad_clip_ema_max', None),
        }
        # Step 2: apply group overrides when provided.
        overrides = getattr(self.args.model_config, 'grad_clip_ema_group_overrides', None)
        if isinstance(overrides, dict) and group_name in overrides:
            defaults.update(overrides[group_name] or {})
        return defaults

    def _compute_ema_clip_max_norm(self, raw_total_norm, group_name, ema_settings):
        """
        Compute EMA-based gradient clip threshold and update internal state.
        
        Args:
            raw_total_norm: Current raw total gradient norm (float).
            group_name: Identifier for EMA state tracking.
            ema_settings: Dict of EMA hyperparameters.
        
        Returns:
            float: EMA-scaled max norm for clipping (<=0 disables).
        """
        # Step 1: read EMA hyperparameters from resolved settings.
        beta = float(ema_settings.get("beta", 0.95))
        mult = float(ema_settings.get("mult", 2.0))
        warmup_steps = int(ema_settings.get("warmup_steps", 0))
        warmup_max_norm = ema_settings.get("warmup_max_norm", None)
        min_norm = ema_settings.get("min", None)
        max_norm = ema_settings.get("max", None)
        # Step 2: ensure EMA state is initialized.
        if not hasattr(self, '_grad_clip_ema_state'):
            self._grad_clip_ema_state = {}
        # Step 3: update step counter and EMA value.
        state = self._grad_clip_ema_state.setdefault(group_name, {"value": None, "step": 0})
        state["step"] += 1
        raw_value = float(raw_total_norm)
        if state["value"] is None:
            state["value"] = raw_value
        else:
            state["value"] = beta * state["value"] + (1.0 - beta) * raw_value
        # Step 4: compute EMA-based threshold.
        clip_max_norm = state["value"] * mult
        # Step 5: optionally override during warmup.
        if warmup_steps > 0 and state["step"] <= warmup_steps:
            if warmup_max_norm is None:
                warmup_max_norm = getattr(self.args.model_config, 'grad_clip_max_norm', None)
            if warmup_max_norm is not None:
                clip_max_norm = float(warmup_max_norm)
        # Step 6: apply optional floor and ceiling.
        if min_norm is not None:
            clip_max_norm = max(float(min_norm), clip_max_norm)
        if max_norm is not None:
            clip_max_norm = min(float(max_norm), clip_max_norm)
        return float(clip_max_norm)

    def _apply_grad_clipping(self, grad_clip_mode, grad_clip_max_norm,
                             text_film_grad_clip_max_norm, group_params,
                             raw_norms):
        """
        Apply gradient clipping based on configured thresholds.
        
        Args:
            grad_clip_mode: Clipping mode ("fixed", "ema", or "none").
            grad_clip_max_norm: Global clipping threshold (None/<=0 disables).
            text_film_grad_clip_max_norm: Text-FiLM threshold (None/<=0 disables).
            group_params: Mapping of group name -> parameters.
            raw_norms: Raw gradient norms by group.
        
        Returns:
            str: Clip mode used ("global", "ema", "ema_by_group", "text_film", or "none").
        """
        # Step 1: apply EMA-based clipping when configured.
        if grad_clip_mode == "ema":
            ema_by_group = bool(getattr(self.args.model_config, 'grad_clip_ema_by_group', False))
            if ema_by_group:
                # Step 1a: get list of groups to clip with per-group EMA settings.
                group_overrides = getattr(self.args.model_config, 'grad_clip_ema_group_overrides', {}) or {}
                groups_to_clip = set(group_overrides.keys())
                # Step 1b: track which groups were actually clipped.
                any_clipped = False
                # Step 1c: clip each group with its own EMA settings.
                for group_name in groups_to_clip:
                    group_params_list = group_params.get(group_name, [])
                    if not group_params_list:
                        continue
                    # Compute norm for this group.
                    group_norm = raw_norms.get(group_name, 0.0)
                    # Get EMA settings for this group (with overrides).
                    group_settings = self._get_grad_clip_ema_settings(group_name)
                    # Compute EMA-based clip threshold.
                    group_max = self._compute_ema_clip_max_norm(
                        group_norm,
                        group_name,
                        group_settings
                    )
                    # Apply clipping if threshold is valid.
                    if group_max > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self._filter_params_with_grad(group_params_list),
                            max_norm=float(group_max)
                        )
                        any_clipped = True
                # Step 1d: clip remaining groups (not in overrides) with default EMA.
                # Collect all parameter IDs that were already clipped.
                clipped_param_ids = set()
                for group_name in groups_to_clip:
                    for param in group_params.get(group_name, []):
                        clipped_param_ids.add(id(param))
                # Find remaining parameters (excluding 'total' which is all params).
                remaining_groups = {
                    name: params for name, params in group_params.items()
                    if name != 'total' and name not in groups_to_clip
                }
                # Group remaining params by whether they've been clipped.
                unclipped_params = []
                for params_list in remaining_groups.values():
                    for param in params_list:
                        if id(param) not in clipped_param_ids:
                            unclipped_params.append(param)
                # Clip remaining params with default EMA settings.
                if unclipped_params:
                    remaining_norm = self._compute_params_norm(unclipped_params)
                    default_settings = self._get_grad_clip_ema_settings('default')
                    default_max = self._compute_ema_clip_max_norm(
                        remaining_norm,
                        'default',
                        default_settings
                    )
                    if default_max > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self._filter_params_with_grad(unclipped_params),
                            max_norm=float(default_max)
                        )
                        any_clipped = True
                # Return status based on whether any clipping occurred.
                if any_clipped:
                    return "ema_by_group"
                return "none"
            # Step 1c: apply global EMA clipping when not grouping.
            raw_total_norm = raw_norms.get("total", None)
            if raw_total_norm is None:
                return "none"
            default_settings = self._get_grad_clip_ema_settings('default')
            clip_max_norm = self._compute_ema_clip_max_norm(
                raw_total_norm,
                'total',
                default_settings
            )
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=float(clip_max_norm)
                )
                return "ema"
            return "none"
        # Step 2: apply fixed global clipping when configured.
        if grad_clip_max_norm is not None and float(grad_clip_max_norm) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=float(grad_clip_max_norm)
            )
            return "global"
        # Step 3: apply text-FiLM-only clipping when configured and present.
        text_film_params = group_params.get('text_film', [])
        if text_film_params and text_film_grad_clip_max_norm is not None and float(text_film_grad_clip_max_norm) > 0:
            torch.nn.utils.clip_grad_norm_(
                self._filter_params_with_grad(text_film_params),
                max_norm=float(text_film_grad_clip_max_norm)
            )
            return "text_film"
        # Step 4: no clipping applied.
        return "none"

    def _merge_raw_clipped_norms(self, raw_norms, clipped_norms):
        """
        Merge raw and clipped norms into a single per-group dictionary.
        
        Args:
            raw_norms: Raw (pre-clip) norms by group.
            clipped_norms: Clipped (post-clip) norms by group.
        
        Returns:
            dict: Mapping of group name -> {"raw": float, "clipped": float}.
        """
        # Step 1: initialize merged output.
        merged = {}
        # Step 2: unify keys from both dictionaries.
        all_groups = set(raw_norms.keys()) | set(clipped_norms.keys())
        # Step 3: build merged entries with defaults.
        for group_name in all_groups:
            merged[group_name] = {
                "raw": float(raw_norms.get(group_name, 0.0)),
                "clipped": float(clipped_norms.get(group_name, 0.0))
            }
        return merged

    def _compute_grad_norm(self, model_optim):
        """
        Compute gradient norms, apply optional clipping, and return group details.
        
        This supports:
        - Global clipping (all parameters) via model_config.grad_clip_max_norm
        - Text-FiLM-only clipping via model_config.text_film_grad_clip_max_norm
        
        Args:
            model_optim: Optimizer with named parameter groups.

        Returns:
            tuple: (legacy_total_norm, group_norms) where group_norms includes raw/clipped.
        """
        # Step 1: read clipping thresholds from model config.
        grad_clip_mode = self._get_grad_clip_mode()
        grad_clip_max_norm = getattr(self.args.model_config, 'grad_clip_max_norm', None)
        text_film_grad_clip_max_norm = getattr(
            self.args.model_config,
            'text_film_grad_clip_max_norm',
            None
        )
        # Step 2: collect parameter groups for norm computation.
        group_params = self._get_named_param_groups(model_optim)
        # Step 3: compute raw (pre-clip) norms.
        raw_norms = self._compute_group_norms(group_params)
        # Step 4: apply configured clipping and track the mode used.
        clip_mode = self._apply_grad_clipping(
            grad_clip_mode,
            grad_clip_max_norm,
            text_film_grad_clip_max_norm,
            group_params,
            raw_norms
        )
        # Step 5: compute clipped (post-clip) norms.
        clipped_norms = self._compute_group_norms(group_params)
        # Step 6: merge raw and clipped norms per group.
        group_norms = self._merge_raw_clipped_norms(raw_norms, clipped_norms)
        # Step 7: preserve legacy total norm behavior for existing logging.
        if clip_mode in {"global", "ema"}:
            legacy_total_norm = float(raw_norms.get("total", 0.0))
        else:
            legacy_total_norm = float(clipped_norms.get("total", raw_norms.get("total", 0.0)))
        return legacy_total_norm, group_norms

    def _train_single_batch(self, iter, model_optim, criterion, track_per_sample):
        """
        Execute a single training batch: forward pass, loss, backward, update.

        Args:
            iter: Batch data tuple from data loader
            model_optim: Optimizer for parameter updates
            criterion: Loss function
            track_per_sample: Whether to track per-sample metrics

        Returns:
            tuple: (loss_value, batch_size, sample_ids, output, gt, grad_norm, grad_norms_by_group)
        """
        # Zero gradients before forward pass
        model_optim.zero_grad()

        # Forward pass through model
        output, gt, sample_ids = self._forward_step(iter)

        # Compute loss
        loss = criterion(output, gt)

        # Backward pass
        loss.backward()


        # Clip gradients and capture raw/clipped norms by group
        grad_norm, grad_norms_by_group = self._compute_grad_norm(model_optim)

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

        return loss_value, current_batch_size, sample_ids, output, gt, grad_norm, grad_norms_by_group

    def _update_training_progress(self, progress, task, time_now, iter_count, 
                                   epoch, train_steps, current_iter, loss_value,
                                   last_val_loss=None):
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
        val_loss_str = f"{last_val_loss:.7f}" if last_val_loss is not None else "n/a"
        progress.update(
            task,
            advance=1,
            loss=loss_value,
            val_loss_str=val_loss_str,
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
        # Group grad norm logging configuration
        log_grad_norms_by_group = False
        if hasattr(self.args, 'wandb') and hasattr(self.args.wandb, 'log_grad_norms_by_group'):
            log_grad_norms_by_group = getattr(self.args.wandb, 'log_grad_norms_by_group', False)
        
        # Validation batch interval configuration (must be multiple of batch_log_interval)
        validate_every_n_batches = None
        if hasattr(self.args, 'wandb') and hasattr(self.args.wandb, 'validate_every_n_batches'):
            validate_every_n_batches = getattr(self.args.wandb, 'validate_every_n_batches', None)

        # Optional: limit validation to a small number of batches for low overhead
        validate_num_val_batches = None
        if hasattr(self.args, 'wandb') and hasattr(self.args.wandb, 'validate_num_val_batches'):
            validate_num_val_batches = getattr(self.args.wandb, 'validate_num_val_batches', None)
        
        # Track last validation loss to log on every batch step (for consistent WandB plotting)
        # Initialize to None - will use 1.0 as placeholder until first validation runs
        last_val_loss = None
        
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
                loss_value, batch_size, _, _, _, grad_norm, grad_norms_by_group = \
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
                        # Optionally log raw and clipped gradient norms per group
                        if log_grad_norms_by_group and grad_norms_by_group:
                            for group_name, norms in grad_norms_by_group.items():
                                batch_metrics[f"batch_grad_norm_raw/{group_name}"] = norms.get("raw", 0.0)
                                batch_metrics[f"batch_grad_norm_clipped/{group_name}"] = norms.get("clipped", 0.0)
                        
                        # Optionally run validation at batch level if configured
                        # Validation only runs when batch logging occurs and batch index matches validation interval
                        if (validate_every_n_batches is not None and 
                            vali_loader is not None and 
                            i % validate_every_n_batches == 0):
                            # Run validation and update last known validation loss
                            # Use 'batch_val_loss' to avoid collision with epoch-level 'val_loss'
                            if validate_num_val_batches is not None:
                                vali_loss = self._vali_partial(vali_loader, criterion, validate_num_val_batches)
                            else:
                                vali_loss = self.vali(vali_loader, criterion)
                            last_val_loss = vali_loss
                            # Set model back to training mode after validation
                            self.model.train()
                            # Avoid per-batch console logging; rely on progress bar and WandB.
                        
                        # Always log batch_val_loss if validation is enabled (for consistent WandB plotting)
                        # Use last known value, or 1.0 as placeholder until first validation runs
                        if validate_every_n_batches is not None:
                            if last_val_loss is not None:
                                batch_metrics['batch_val_loss'] = last_val_loss
                            else:
                                # Use 1.0 as placeholder until first validation runs
                                # This ensures the metric appears in WandB from the start
                                batch_metrics['batch_val_loss'] = 1.0
                        
                        self.exp_manager.log_batch_metrics(batch_metrics, batch_step=global_step)

                # Update progress bar
                time_now, iter_count = self._update_training_progress(
                    progress, task, time_now, iter_count, epoch, train_steps, i, loss_value,
                    last_val_loss=last_val_loss
                )
        
        # Calculate and log epoch statistics
        epoch_time_elapsed = time.time() - epoch_time
        train_loss = epoch_loss / total_samples if total_samples > 0 else 0.0
        logger.info(f"Epoch: {epoch + 1} cost time: {epoch_time_elapsed:.2f}s")
        
        return train_loss, epoch_time_elapsed, total_samples

    def _track_val_batch_metrics(self, iter_data, output, gt, criterion, sample_ids):
        """
        Track per-sample validation metrics for a single batch when enabled.
        """
        # Step 1: compute per-sample losses for logging.
        per_sample_loss = compute_per_sample_loss(output, gt, criterion)
        per_sample_per_channel_loss = compute_per_sample_per_channel_loss(output, gt, criterion)
        # Step 2: extract timestamps and channel names.
        timestamps = iter_data[3]
        channel_names = self._get_channel_names(per_sample_per_channel_loss.shape[1])
        # Step 3: record batch metrics for the validation split.
        self.metrics_tracker.add_batch(
            epoch=self.current_epoch, split='val', entity_id=None, sample_ids=sample_ids,
            timestamps=timestamps[:, 0] if timestamps is not None else None,
            losses=per_sample_loss, channel_ids=channel_names,
            per_channel_losses=per_sample_per_channel_loss
        )

    def _vali_partial(self, loader, criterion, num_batches):
        """
        Validate on the next N batches from the loader to reduce overhead.
        """
        # Step 1: validate inputs and handle empty loaders safely.
        if num_batches is None or num_batches <= 0:
            raise ValueError(f"num_batches must be >= 1, got {num_batches}")
        if len(loader) == 0:
            return 0.0
        # Step 2: initialize accumulators and set eval mode.
        running_loss, total_samples = 0.0, 0
        self.model.eval()
        # Step 3: respect per-sample tracking settings when enabled.
        track_per_sample = getattr(self.args, 'track_per_sample', False)
        # Step 4: maintain a persistent iterator to advance sequentially.
        if (not hasattr(self, "_vali_partial_iter") or
                not hasattr(self, "_vali_partial_loader") or
                self._vali_partial_loader is not loader):
            self._vali_partial_iter = iter(loader)
            self._vali_partial_loader = loader
        # Step 5: iterate over the next N validation batches without gradients.
        batches_done, reset_count = 0, 0
        with torch.inference_mode():
            with torch.no_grad():
                while batches_done < num_batches:
                    try:
                        iter_data = next(self._vali_partial_iter)
                    except StopIteration:
                        if reset_count >= 1:
                            break
                        self._vali_partial_iter = iter(loader)
                        self._vali_partial_loader = loader
                        reset_count += 1
                        continue
                    output, gt, sample_ids = self._forward_step(iter_data)
                    current_batch_size = gt.size(0)
                    loss = criterion(output, gt)
                    running_loss += loss.item() * current_batch_size
                    total_samples += current_batch_size
                    if track_per_sample and self.metrics_tracker:
                        self._track_val_batch_metrics(iter_data, output, gt, criterion, sample_ids)
                    batches_done += 1
        # Step 6: compute mean loss, restore train mode, and return.
        val_sample_loss = running_loss / total_samples if total_samples > 0 else 0.0
        self.model.train()
        return val_sample_loss

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
    
    def _load_best_checkpoint(self, checkpoint_dir, best_epoch=None, train_loss=None, vali_loss=None):
        """
        Load the best checkpoint, update model weights, and return metadata.
        """
        # Build the checkpoint path and load the checkpoint object.
        best_model_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
        checkpoint = torch.load(best_model_path)
        # Extract state dict and optional metadata from the checkpoint.
        has_metadata = isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
        if has_metadata:
            model_state_dict = checkpoint['model_state_dict']
            best_epoch = checkpoint.get('epoch', best_epoch or self.args.train_epochs)
            train_loss = checkpoint.get('train_loss', train_loss)
            vali_loss = checkpoint.get('val_loss', vali_loss)
        else:
            model_state_dict = checkpoint
            if best_epoch is None:
                best_epoch = self.args.train_epochs
        # Detect model wrapping and compilation status for key normalization.
        is_data_parallel = hasattr(self.model, 'module')
        model_is_compiled = hasattr(self.model, '_orig_mod') or (is_data_parallel and hasattr(self.model.module, '_orig_mod'))
        checkpoint_has_prefix = isinstance(model_state_dict, dict) and any(
            key.startswith('_orig_mod.') or key.startswith('module._orig_mod.') for key in model_state_dict.keys()
        )
        # Normalize state dict keys to match current model wrapping.
        if checkpoint_has_prefix and not model_is_compiled:
            model_state_dict = {
                key.replace('module._orig_mod.', 'module.' if is_data_parallel else '').replace('_orig_mod.', ''): value
                for key, value in model_state_dict.items()
            }
        elif not checkpoint_has_prefix and model_is_compiled:
            if is_data_parallel:
                model_state_dict = {f'module._orig_mod.{key}': value for key, value in model_state_dict.items()}
            else:
                model_state_dict = {f'_orig_mod.{key}': value for key, value in model_state_dict.items()}
        # Load the best weights into the model and return metadata.
        self.model.load_state_dict(model_state_dict)
        return {
            'checkpoint': checkpoint,
            'best_epoch': best_epoch,
            'train_loss': train_loss,
            'vali_loss': vali_loss,
            'checkpoint_path': best_model_path,
            'has_metadata': has_metadata
        }

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
        # Load best checkpoint and update model weights for consistent reporting.
        checkpoint_info = self._load_best_checkpoint(
            checkpoint_dir=path,
            best_epoch=best_epoch,
            train_loss=train_loss,
            vali_loss=vali_loss
        )
        # Unpack checkpoint metadata for downstream logging and saving.
        checkpoint = checkpoint_info['checkpoint']
        best_epoch = checkpoint_info['best_epoch']
        train_loss = checkpoint_info['train_loss']
        vali_loss = checkpoint_info['vali_loss']
        best_model_path = checkpoint_info['checkpoint_path']
        has_metadata = checkpoint_info['has_metadata']

        # Optionally update checkpoint with test metrics if available
        # This makes checkpoint.pth a complete record of all metrics from best epoch
        if test_metrics is not None and has_metadata:
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

        # Load best checkpoint before evaluation to ensure metrics reflect best epoch.
        self._load_best_checkpoint(
            checkpoint_dir=checkpoint_path,
            best_epoch=best_epoch,
            train_loss=None,
            vali_loss=None
        )

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
        Evaluate the model on the test dataset(s).

        Supports a dict of per-entity loaders or a single DataLoader when
        tensor cache is enabled.
        """
        # Normalize loaders to a dict for consistent iteration.
        loaders = self._normalize_test_loaders(loaders)
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

    def _normalize_test_loaders(self, loaders):
        """
        Normalize test loaders into a dict keyed by entity name.

        This handles both multi-entity dicts and single DataLoader cases
        (e.g., when tensor cache returns a single loader).
        """
        # Step 1: Handle missing loaders explicitly.
        if loaders is None:
            return {}
        # Step 2: If dict-like, keep the existing mapping.
        if hasattr(loaders, "items"):
            return loaders
        # Step 3: Wrap a single loader with a stable key.
        return {"test": loaders}
