import os
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from data_provider.data_factory import Data_Provider
from thop import profile
from typing import Optional
from exp.manager import ExperimentManager


class Exp_Basic(object):
    """
    Base experiment class providing core functionality for time series forecasting experiments.
    
    This abstract base class establishes the fundamental structure and common methods
    for all experiment types in the Universal Cross-Modal Time Series Forecasting Pipeline.
    It handles device management, model initialization, data provider setup, and provides
    templates for training, validation, and testing workflows.
    
    Args:
        args: Configuration object containing experiment parameters including:
            - model configuration (model type, parameters)
            - data configuration (paths, preprocessing settings)
            - training configuration (learning rate, batch size, epochs)
            - device configuration (GPU/CPU, multi-GPU settings)
            - evaluation configuration (metrics, checkpointing)
    
    Attributes:
        args: Experiment configuration object
        device: PyTorch device (CPU or CUDA)
        model: Initialized model moved to appropriate device
        data_provider: Data management object for loading datasets
        exp_manager: Optional ExperimentManager for experiment tracking and logging
    
    Example:
        ```python
        # Typically used through subclasses like Experiment
        class CustomExperiment(Exp_Basic):
            def _build_model(self):
                return YourModel(self.args.model_config)
        
        exp = CustomExperiment(args)
        exp.train()
        ```
    """
    def __init__(self, args, exp_manager: Optional[ExperimentManager] = None):
        """
        Initialize base experiment class.
        
        Args:
            args: Experiment configuration object (dotdict or argparse-like)
            exp_manager: Optional ExperimentManager for experiment tracking and wandb logging
        """
        self.args = args
        self.exp_manager = exp_manager
        self.device = self._acquire_device()
        
        # Get indicator column count before building model (needed for enc_in adjustment)
        # Create data_provider first to access required_indicator_columns
        console = exp_manager.get_console() if exp_manager else None
        self.data_provider = Data_Provider(
            args,
            buffer=(not args.disable_buffer),
            console=console,
            exp_manager=exp_manager
        )
        
        # Get number of indicator columns that will be added to the data
        num_indicator_columns = len(getattr(self.data_provider, 'required_indicator_columns', []))
        # Store in args for model_init to access
        args.num_indicator_columns = num_indicator_columns
        
        self.model = self._build_model().to(self.device)
        
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

    def _build_model(self):
        """
        Abstract method for building the forecasting model.
        
        This method must be implemented by subclasses to specify the model
        architecture and configuration for their specific use case.
        
        Returns:
            torch.nn.Module: Initialized PyTorch model
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError
        return None

    def _acquire_device(self):
        """
        Configures and returns the appropriate computing device for training.
        
        Supports both CPU and GPU training with optional multi-GPU setup.
        Sets CUDA_VISIBLE_DEVICES environment variable for GPU device selection.
        
        Returns:
            torch.device: Configured device (CPU or specific CUDA device)
        """
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    
    def _select_optimizer(self):
        """
        Creates and returns the optimizer for model training.
        
        Supports differential learning rates for projection layers (e.g., ZhangHanBest):
        - If projector_learning_rate is set, projection layers get a higher LR
        - Otherwise, all parameters use the same learning_rate (backward compatible)
        
        Per Zhang et al. (2025), the projector should train at 100x higher LR (0.01)
        than the time series model (0.0001) during joint training.
        
        Returns:
            torch.optim.Optimizer: Configured optimizer for model parameters
        """
        base_lr = self.args.learning_rate
        projector_lr = getattr(self.args, 'projector_learning_rate', None)
        
        # Get the underlying model (handle torch.compile wrapper and DataParallel)
        model = self.model
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        if hasattr(model, 'module'):
            model = model.module
        
        # Check if differential learning rates should be used
        # Only apply if projector_learning_rate is explicitly set and model has projection layers
        if projector_lr is not None and hasattr(model, 'residual_proj'):
            # Build parameter groups with different learning rates
            param_groups = self._build_differential_param_groups(base_lr, projector_lr)
            optimizer = optim.Adam(param_groups)
            
            if self.exp_manager:
                self.exp_manager.logger.info(
                    f"Using differential learning rates: base_lr={base_lr}, projector_lr={projector_lr}"
                )
        else:
            # Default: single learning rate for all parameters (backward compatible)
            optimizer = optim.Adam(self.model.parameters(), lr=base_lr)
        
        return optimizer
    
    def _build_differential_param_groups(self, base_lr: float, projector_lr: float):
        """
        Build parameter groups with different learning rates for differential training.
        
        Separates model parameters into:
        1. Projector parameters (residual_proj, prediction_head, ts_proj) - use projector_lr
        2. Base model parameters (ts_encoder, etc.) - use base_lr
        
        Args:
            base_lr: Learning rate for base model (time series encoder)
            projector_lr: Learning rate for projection layers
            
        Returns:
            List of parameter group dicts for optimizer
        """
        # Identify projection-related modules (train at higher LR)
        projector_module_names = {'residual_proj', 'prediction_head', 'ts_proj'}
        
        # Collect parameters by group
        projector_params = []
        base_params = []
        
        # Get the underlying model (handle torch.compile wrapper and DataParallel)
        model = self.model
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        if hasattr(model, 'module'):
            model = model.module
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Check if this parameter belongs to a projector module
            is_projector = any(mod_name in name for mod_name in projector_module_names)
            
            if is_projector:
                projector_params.append(param)
            else:
                base_params.append(param)
        
        # Build parameter groups
        param_groups = []
        
        if base_params:
            param_groups.append({
                'params': base_params,
                'lr': base_lr,
                'name': 'base_model'
            })
        
        if projector_params:
            param_groups.append({
                'params': projector_params,
                'lr': projector_lr,
                'name': 'projector'
            })
        
        return param_groups

    def _select_criterion(self):
        """
        Creates and returns the loss function for model training.
        
        Supports multiple loss functions based on configuration:
        - 'l1': L1Loss (Mean Absolute Error)
        - 'mse': MSELoss (Mean Squared Error)
        
        Returns:
            torch.nn.Module: Configured loss function
        """
        if self.args.loss == 'l1':
            return nn.L1Loss()
        elif self.args.loss == 'mse':
            return nn.MSELoss()
        
    
    def _get_profile(self, model):
        """
        Analyzes model computational complexity and parameter count.
        
        Uses the thop library to calculate FLOPs (Floating Point Operations)
        and parameter count for the model with sample input data.
        
        Args:
            model (torch.nn.Module): Model to profile
        
        Returns:
            tuple: (macs, params) where macs is FLOPs count and params is parameter count
        """
        _input = torch.randn(self.args.batch_size, self.args.seq_len, 1).to(self.device)
        macs, params = profile(model, inputs=(_input).to(self.device))
        print('FLOPs: ', macs)
        print('params: ', params)
        return macs, params

    def _get_data(self):
        """
        Abstract method for data retrieval.
        
        This method should be implemented by subclasses to define how
        training, validation, and test data are obtained and formatted.
        """
        pass

    def vali(self):
        """
        Abstract method for model validation.
        
        This method should be implemented by subclasses to define the
        validation procedure and metrics calculation.
        """
        pass

    def train(self):
        """
        Abstract method for model training.
        
        This method should be implemented by subclasses to define the
        complete training loop including optimization and checkpointing.
        """
        pass

    def test(self):
        """
        Abstract method for model testing.
        
        This method should be implemented by subclasses to define the
        testing procedure and final evaluation metrics.
        """
        pass
