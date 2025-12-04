"""
Unimodal Model Wrapper for Pretrained Models

This module provides a wrapper class for pretrained unimodal models to abstract
normalization behavior and provide a unified interface for loading and using
pretrained models in residual learning architectures.
"""

import os
import torch
import yaml
from utils.tools import dotdict


class UnimodalModelWrapper:
    """
    Wrapper for pretrained unimodal models to abstract normalization behavior.
    
    This class handles:
    - Loading pretrained model configs
    - Detecting normalization schemes
    - Loading checkpoints (handles different formats)
    - Initializing and freezing models
    - Providing unified interface for predictions
    
    Attributes:
        model: The frozen pretrained model
        norm_scheme: Normalization scheme ('use_norm', 'revin', 'none')
        normalizes_internally: Whether model applies normalization internally
        outputs_normalized: Whether model outputs are in normalized space
        requires_normalized_input: Whether model expects normalized input
    """
    
    def __init__(self, model, norm_scheme, normalizes_internally, 
                 outputs_normalized, requires_normalized_input):
        """
        Initialize wrapper with model and normalization behavior.
        
        Args:
            model: The pretrained model (will be frozen)
            norm_scheme: Normalization scheme ('use_norm', 'revin', 'none')
            normalizes_internally: Whether model normalizes internally
            outputs_normalized: Whether model outputs are normalized
            requires_normalized_input: Whether model requires normalized input
        """
        self.model = model
        self.norm_scheme = norm_scheme
        self.normalizes_internally = normalizes_internally
        self.outputs_normalized = outputs_normalized
        self.requires_normalized_input = requires_normalized_input
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Set to eval mode
        self.model.eval()
    
    @classmethod
    def from_config(cls, configs):
        """
        Factory method to create wrapper from configs.
        
        This method orchestrates the entire loading process:
        1. Load pretrained model config
        2. Detect normalization scheme
        3. Initialize model
        4. Load checkpoint
        5. Determine normalization behavior
        6. Return wrapper instance
        
        Args:
            configs: Configuration dictionary containing:
                - pretrained_model_path: Path to checkpoint (required)
                - pretrained_model_type: Type of model (default: 'iTransformer')
                - pretrained_model_config_path: Optional path to model config
                - seq_len, pred_len, enc_in: Model dimensions
        
        Returns:
            UnimodalModelWrapper: Configured wrapper instance
        """
        # Step 1: Load pretrained model config
        pretrained_config = cls._load_pretrained_model_config(configs)
        
        # Step 2: Detect normalization scheme
        norm_scheme = cls._detect_normalization_scheme(pretrained_config)
        
        # Step 3: Initialize model
        pretrained_type = getattr(configs, 'pretrained_model_type', 'iTransformer')
        model = cls._initialize_pretrained_model(pretrained_type, pretrained_config)
        
        # Step 4: Load checkpoint
        checkpoint_path = cls._validate_checkpoint_path(configs)
        state_dict = cls._load_checkpoint_state_dict(checkpoint_path)
        
        # Load state dict
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            # Try with strict=False for partial matches
            print(f"[Warning] Strict loading failed, trying with strict=False: {e}")
            model.load_state_dict(state_dict, strict=False)
        
        # Step 5: Determine normalization behavior based on model type
        normalizes_internally, outputs_normalized, requires_normalized_input = \
            cls._determine_normalization_behavior(pretrained_type, norm_scheme)
        
        # Step 6: Create and return wrapper
        wrapper = cls(model, norm_scheme, normalizes_internally, 
                     outputs_normalized, requires_normalized_input)
        
        print(f"[Info] Loaded and frozen pretrained {pretrained_type} model from: {checkpoint_path}")
        print(f"[Info] Normalization scheme: {norm_scheme}")
        
        return wrapper
    
    @staticmethod
    def _load_pretrained_model_config(configs):
        """
        Load and prepare pretrained model configuration.
        
        Args:
            configs: Main model configs
            
        Returns:
            dotdict: Prepared pretrained model config
        """
        # Get pretrained model type (default: iTransformer)
        pretrained_type = getattr(configs, 'pretrained_model_type', 'iTransformer')
        
        # Get pretrained model config path (optional)
        pretrained_config_path = getattr(configs, 'pretrained_model_config_path', None)
        
        # Load pretrained model config
        if pretrained_config_path is None:
            # Use default config path based on model type
            pretrained_config_path = f"model_configs/general/{pretrained_type}.yaml"
        
        if not os.path.exists(pretrained_config_path):
            raise FileNotFoundError(
                f"Pretrained model config not found at: {pretrained_config_path}"
            )
        
        # Load config
        with open(pretrained_config_path, 'r') as f:
            pretrained_config_dict = yaml.safe_load(f)
        
        pretrained_config = dotdict(pretrained_config_dict)
        
        # Set required parameters from current configs
        pretrained_config.seq_len = configs.seq_len
        pretrained_config.pred_len = configs.pred_len
        pretrained_config.enc_in = configs.enc_in
        
        return pretrained_config
    
    @staticmethod
    def _detect_normalization_scheme(pretrained_config):
        """
        Detect normalization scheme from pretrained model config.
        
        Args:
            pretrained_config: Pretrained model configuration
            
        Returns:
            str: Normalization scheme ('use_norm', 'revin', or 'none')
        """
        # Check for use_norm (iTransformer)
        if hasattr(pretrained_config, 'use_norm') and pretrained_config.use_norm:
            return 'use_norm'
        # Check for revin (TGTSF-compatible)
        elif hasattr(pretrained_config, 'revin') and pretrained_config.revin:
            return 'revin'
        else:
            return 'none'
    
    @staticmethod
    def _initialize_pretrained_model(pretrained_type, pretrained_config):
        """
        Initialize pretrained model instance.
        
        Args:
            pretrained_type: Type of model (e.g., 'iTransformer')
            pretrained_config: Model configuration
            
        Returns:
            nn.Module: Initialized (unfrozen) model
        """
        if pretrained_type == 'iTransformer':
            from models.iTransformer import Model as iTransformerModel
            return iTransformerModel(pretrained_config)
        else:
            raise ValueError(
                f"Unsupported pretrained_model_type: {pretrained_type}. "
                f"Currently only 'iTransformer' is supported."
            )
    
    @staticmethod
    def _validate_checkpoint_path(configs):
        """
        Validate that checkpoint path is provided and exists.
        
        Args:
            configs: Configuration dictionary
            
        Returns:
            str: Validated checkpoint path
            
        Raises:
            ValueError: If pretrained_model_path not provided
            FileNotFoundError: If checkpoint doesn't exist
        """
        # Require explicit pretrained_model_path
        if not hasattr(configs, 'pretrained_model_path') or not configs.pretrained_model_path:
            raise ValueError(
                "pretrained_model_path is required in config. "
                "Please provide explicit path to pretrained checkpoint."
            )
        
        pretrained_path = configs.pretrained_model_path
        
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(
                f"Pretrained model checkpoint not found at: {pretrained_path}\n"
                f"Please provide a valid pretrained_model_path in config."
            )
        
        return pretrained_path
    
    @staticmethod
    def _load_checkpoint_state_dict(checkpoint_path):
        """
        Load state dict from checkpoint, handling different formats.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            dict: Cleaned state dict ready for loading
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if checkpoint_path.endswith('.ckpt'):
            # Lightning checkpoint format
            if 'state_dict' in checkpoint:
                # Remove "model." prefix if present (Lightning wraps model)
                state_dict = {}
                for key, value in checkpoint['state_dict'].items():
                    # Handle both "model.model." (nested) and "model." prefixes
                    if key.startswith('model.model.'):
                        new_key = key.replace('model.model.', '')
                    elif key.startswith('model.'):
                        new_key = key.replace('model.', '')
                    else:
                        new_key = key
                    state_dict[new_key] = value
            else:
                state_dict = checkpoint
        else:
            # PyTorch checkpoint format (.pth)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        
        return state_dict
    
    @staticmethod
    def _determine_normalization_behavior(pretrained_type, norm_scheme):
        """
        Determine normalization behavior for specific model type.
        
        Args:
            pretrained_type: Type of model (e.g., 'iTransformer')
            norm_scheme: Detected normalization scheme
            
        Returns:
            tuple: (normalizes_internally, outputs_normalized, requires_normalized_input)
        """
        if pretrained_type == 'iTransformer' and norm_scheme == 'use_norm':
            # iTransformer with use_norm:
            # - Normalizes internally in forecast() method
            # - Returns denormalized output
            # - Requires unnormalized input
            return True, False, False
        else:
            # Other models: depends on their implementation
            # Default: assume they don't normalize internally
            return False, True, True
    
    def predict(self, x, norm_params):
        """
        Get prediction in normalized space.
        
        This method handles model-specific normalization behavior:
        - If normalizes_internally: pass unnormalized input, re-normalize output
        - If not: pass normalized input, use output as-is
        
        Args:
            x: Input tensor [B, seq_len, C] (may be normalized or not)
            norm_params: Normalization parameters from normalize_input
        
        Returns:
            torch.Tensor: Prediction in normalized space [B, pred_len, C]
        """
        if self.normalizes_internally:
            # Model normalizes internally (e.g., iTransformer)
            # Pass unnormalized input
            with torch.no_grad():
                pred_denorm = self.model(x)  # Returns denormalized
            
            # Re-normalize output to normalized space
            # (Model-specific logic - currently iTransformer)
            pred_norm = self._renormalize_output(pred_denorm, norm_params)
        else:
            # Model doesn't normalize internally
            # Pass normalized input, output is already normalized
            with torch.no_grad():
                pred_norm = self.model(x)  # x is already normalized
        
        return pred_norm
    
    def normalize_input(self, x):
        """
        Normalize input based on wrapper's normalization scheme.
        
        This method handles normalization for the pretrained unimodal model.
        The normalized input is used by both the unimodal model and the residual model.
        
        Args:
            x: Input tensor [B, L, C]
            
        Returns:
            tuple: (x_norm, norm_params)
                - x_norm: Normalized input
                - norm_params: Dictionary with normalization parameters for denormalization
        """
        if self.norm_scheme == 'use_norm':
            # Apply iTransformer normalization (Non-stationary Transformer normalization)
            # This matches the normalization in iTransformer.forecast()
            means = x.mean(1, keepdim=True).detach()
            x_norm = x - means
            stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_norm = x_norm / stdev
            
            norm_params = {
                'means': means,
                'stdev': stdev,
                'type': 'use_norm'
            }
        elif self.norm_scheme == 'revin':
            # Apply RevIN normalization
            x_mean = torch.mean(x, dim=1, keepdim=True)
            x_norm = x - x_mean
            x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x_norm = x_norm / torch.sqrt(x_var)
            
            norm_params = {
                'x_mean': x_mean,
                'x_var': x_var,
                'type': 'revin'
            }
        else:
            raise ValueError(
                f"Unsupported normalization scheme: {self.norm_scheme}. "
                f"Supported schemes are 'use_norm' and 'revin'. "
                f"If no normalization is needed, use 'revin' with revin=False in model config."
            )
        
        return x_norm, norm_params
    
    def denormalize_output(self, x_norm, norm_params):
        """
        Denormalize predictions using stored normalization parameters.
        
        This method handles denormalization based on the wrapper's normalization scheme.
        
        Args:
            x_norm: Normalized predictions [B, L, C]
            norm_params: Normalization parameters from normalize_input
            
        Returns:
            torch.Tensor: Denormalized predictions [B, L, C]
        """
        norm_type = norm_params.get('type', 'none')
        
        if norm_type == 'use_norm':
            # iTransformer denormalization
            means = norm_params['means']
            stdev = norm_params['stdev']
            # Expand stdev and means to match prediction length
            # x_norm is [B, pred_len, C], stdev/means are [B, 1, C]
            x = x_norm * (stdev[:, 0, :].unsqueeze(1).repeat(1, x_norm.shape[1], 1))
            x = x + (means[:, 0, :].unsqueeze(1).repeat(1, x_norm.shape[1], 1))
        elif norm_type == 'revin':
            # RevIN denormalization
            x_mean = norm_params['x_mean']
            x_var = norm_params['x_var']
            x = x_norm * torch.sqrt(x_var) + x_mean
        else:
            raise ValueError(
                f"Unsupported normalization type in norm_params: {norm_type}. "
                f"Supported types are 'use_norm' and 'revin'. "
                f"norm_params must come from normalize_input() method."
            )
        
        return x
    
    def _renormalize_output(self, pred_denorm, norm_params):
        """
        Re-normalize denormalized output back to normalized space.
        
        This is model-specific. Currently implemented for iTransformer.
        Would need to be extended for other models that normalize internally.
        
        Args:
            pred_denorm: Denormalized prediction [B, pred_len, C]
            norm_params: Normalization parameters
            
        Returns:
            torch.Tensor: Normalized prediction [B, pred_len, C]
        """
        # Currently iTransformer-specific
        # For use_norm: (pred - mean) / std
        means = norm_params['means']  # [B, 1, C]
        stdev = norm_params['stdev']  # [B, 1, C]
        # Expand to match prediction length
        means_pred = means[:, 0, :].unsqueeze(1).repeat(1, pred_denorm.shape[1], 1)  # [B, pred_len, C]
        stdev_pred = stdev[:, 0, :].unsqueeze(1).repeat(1, pred_denorm.shape[1], 1)  # [B, pred_len, C]
        pred_norm = (pred_denorm - means_pred) / stdev_pred
        
        return pred_norm

