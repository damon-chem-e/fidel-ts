"""
lynx Model: Residual Learning with Frozen Pretrained Unimodal Model

lynx is a modified version of TGTSF that learns the residual to a pretrained
frozen unimodal model (default: iTransformer) instead of learning the entire
prediction from scratch.

Architecture:
- Base Model: TGTSF (Text-Guided Time Series Forecasting)
- Unimodal Model: Pretrained frozen model (default: iTransformer)
- Output: final_prediction = unimodal_prediction + residual_prediction
"""

import os
import glob
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import yaml
from utils.tools import dotdict

from layers.RevIN import RevIN
from layers.TGTSF_torch import text_encoder, text_temp_cross_block, positional_encoding, TS_encoder


class Model(nn.Module):
    """
    lynx Model: Residual learning with frozen pretrained unimodal model.
    
    This model combines:
    1. A frozen pretrained unimodal model (e.g., iTransformer) for base predictions
    2. A TGTSF model that learns the residual (difference) between ground truth
       and the unimodal prediction
    
    The final output is: final_prediction = unimodal_prediction + residual_prediction
    """
    
    def __init__(self, configs):
        """
        Initialize the lynx model.
        
        Args:
            configs: Configuration dictionary containing:
                - All TGTSF parameters (enc_in, seq_len, pred_len, e_layers, etc.)
                - pretrained_model_path: Path to pretrained checkpoint (required)
                - pretrained_model_type: Type of model (default: 'iTransformer')
                - pretrained_model_config_path: Optional path to pretrained model config
                - pretrained_checkpoint_base: Optional base directory for dynamic path resolution
        """
        super().__init__()
        
        # Load TGTSF parameters (same as TGTSF)
        c_in = configs.enc_in 
        context_window = configs.seq_len
        self.pred_len = configs.pred_len
        
        n_layers = configs.e_layers 
        n_heads = configs.n_heads 
        d_model = configs.d_model 

        dropout = configs.dropout 
        
        individual = configs.individual 
    
        self.patch_len = configs.patch_len 
        self.stride = configs.stride 
        
        # Store original revin setting (may be disabled if pretrained uses different norm)
        self.original_revin = configs.revin
        
        self.out_attn_weights = False
        
        # Initialize TGTSF components
        self.TS_encoder = TS_encoder(
            embedding_dim=d_model, 
            layers=n_layers, 
            num_heads=n_heads, 
            dropout=dropout, 
            patch_len=self.patch_len, 
            stride=self.stride, 
            causal=True, 
            input_len=configs.seq_len
        )
        
        self.text_encoder = text_encoder(
            cross_layer=configs.cross_layers, 
            self_layer=configs.self_layers, 
            embedding_dim=configs.text_dim, 
            num_heads=configs.n_heads, 
            dropout=configs.dropout, 
            pred_len=configs.pred_len, 
            stride=self.stride
        )

        self.dropout = dropout

        self.mixer = text_temp_cross_block(
            text_embedding_dim=configs.text_dim, 
            temp_embedding_dim=d_model, 
            num_heads=configs.n_heads, 
            dropout=0.0, 
            self_layer=configs.mixer_self_layers
        )

        patch_num = int((context_window - self.patch_len)/self.stride + 1)
        self.patch_num = patch_num
        self.total_length = self.patch_len*2 + (int((self.pred_len - self.patch_len)/self.stride + 1) - 1) * self.stride

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.individual = individual

        self.head = nn.Linear(d_model, self.patch_len)
        
        # Load and freeze pretrained unimodal model
        self.unimodal_model, self.norm_type = self._load_pretrained_model(configs)
        
        # Store normalization type for forward pass
        # norm_type: 'use_norm' (iTransformer) or 'revin' (TGTSF-compatible)
        # If pretrained uses 'use_norm', we disable RevIN in TGTSF components
        self.use_pretrained_norm = (self.norm_type == 'use_norm')
        
        if self.use_pretrained_norm:
            # Disable RevIN in TGTSF when pretrained model uses different normalization
            self.revin = False
        else:
            # Use RevIN as normal if pretrained model also uses RevIN
            self.revin = self.original_revin
    
    def _load_pretrained_model(self, configs):
        """
        Load and freeze pretrained unimodal model.
        
        Args:
            configs: Configuration dictionary
            
        Returns:
            tuple: (unimodal_model, norm_type)
                - unimodal_model: Loaded and frozen model
                - norm_type: 'use_norm' or 'revin' indicating normalization scheme
        """
        # Get pretrained model path (support dynamic resolution)
        pretrained_path = self._resolve_checkpoint_path(configs)
        
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(
                f"Pretrained model checkpoint not found at: {pretrained_path}\n"
                f"Please provide pretrained_model_path in config or ensure pretrained_checkpoint_base "
                f"is set correctly for dynamic resolution."
            )
        
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
        
        # Detect normalization scheme
        norm_type = 'revin'  # default
        if hasattr(pretrained_config, 'use_norm') and pretrained_config.use_norm:
            norm_type = 'use_norm'
        elif hasattr(pretrained_config, 'revin') and pretrained_config.revin:
            norm_type = 'revin'
        
        # Initialize pretrained model
        if pretrained_type == 'iTransformer':
            from models.iTransformer import Model as iTransformerModel
            unimodal_model = iTransformerModel(pretrained_config)
        else:
            raise ValueError(
                f"Unsupported pretrained_model_type: {pretrained_type}. "
                f"Currently only 'iTransformer' is supported."
            )
        
        # Load checkpoint
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if pretrained_path.endswith('.ckpt'):
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
        
        # Load state dict
        try:
            unimodal_model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            # Try with strict=False for partial matches
            print(f"[Warning] Strict loading failed, trying with strict=False: {e}")
            unimodal_model.load_state_dict(state_dict, strict=False)
        
        # Freeze all parameters
        for param in unimodal_model.parameters():
            param.requires_grad = False
        
        # Set to eval mode
        unimodal_model.eval()
        
        print(f"[Info] Loaded and frozen pretrained {pretrained_type} model from: {pretrained_path}")
        print(f"[Info] Normalization scheme: {norm_type}")
        
        return unimodal_model, norm_type
    
    def _resolve_checkpoint_path(self, configs):
        """
        Resolve checkpoint path, supporting dynamic resolution based on dataset/output_len.
        
        Args:
            configs: Configuration dictionary
            
        Returns:
            str: Resolved checkpoint path
        """
        # Check if explicit path is provided
        if hasattr(configs, 'pretrained_model_path') and configs.pretrained_model_path:
            return configs.pretrained_model_path
        
        # Try dynamic resolution
        pretrained_checkpoint_base = getattr(configs, 'pretrained_checkpoint_base', None)
        if pretrained_checkpoint_base is None:
            raise ValueError(
                "Either pretrained_model_path or pretrained_checkpoint_base must be provided in config. "
                "For dynamic resolution, also provide dataset_name in configs."
            )
        
        # Get dataset name from configs (may be set by experiment suite or model_init)
        # Try multiple possible sources
        dataset_name = None
        if hasattr(configs, 'data_name'):
            dataset_name = configs.data_name
        elif hasattr(configs, 'dataset_name'):
            dataset_name = configs.dataset_name
        elif hasattr(configs, 'data_config'):
            # Try to get from data_config (could be dict or object)
            data_config = configs.data_config
            if isinstance(data_config, dict):
                dataset_name = data_config.get('name')
            elif hasattr(data_config, 'name'):
                dataset_name = data_config.name
        
        output_len = getattr(configs, 'pred_len', None)
        pretrained_type = getattr(configs, 'pretrained_model_type', 'iTransformer')
        
        if dataset_name is None or output_len is None:
            raise ValueError(
                f"Dynamic checkpoint resolution requires dataset_name and pred_len in configs. "
                f"Found dataset_name={dataset_name}, pred_len={output_len}. "
                f"Please provide pretrained_model_path explicitly or set these values."
            )
        
        # Construct path: {base_dir}/itransformer_{dataset}_{output_len}/checkpoint.ckpt
        # Normalize dataset name (replace spaces/underscores)
        dataset_normalized = dataset_name.lower().replace(' ', '_').replace('-', '_')
        experiment_name = f"{pretrained_type.lower()}_{dataset_normalized}_{output_len}"
        checkpoint_path = os.path.join(
            pretrained_checkpoint_base,
            experiment_name,
            'checkpoint.ckpt'
        )
        
        return checkpoint_path
    
    def _apply_normalization(self, x):
        """
        Apply normalization matching the pretrained model's scheme.
        
        Args:
            x: Input tensor [B, L, C]
            
        Returns:
            tuple: (x_norm, norm_params)
                - x_norm: Normalized input
                - norm_params: Dictionary with normalization parameters for denormalization
        """
        if self.use_pretrained_norm:
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
        else:
            # Use RevIN normalization (if enabled)
            if self.revin:
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
                # No normalization
                x_norm = x
                norm_params = {'type': 'none'}
        
        return x_norm, norm_params
    
    def _denormalize(self, x_norm, norm_params):
        """
        Denormalize predictions using stored normalization parameters.
        
        Args:
            x_norm: Normalized predictions [B, L, C]
            norm_params: Normalization parameters from _apply_normalization
            
        Returns:
            x: Denormalized predictions [B, L, C]
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
            # No denormalization needed
            x = x_norm
        
        return x
    
    def _tgtsf_forward(self, x_norm, news, channel_description, disable_revin=False):
        """
        TGTSF forward pass (extracted from original TGTSF forward).
        
        Args:
            x_norm: Normalized input [B, L, C]
            news: News embeddings [B, l, news_num, text_dim]
            channel_description: Channel descriptions [B, 1, C, d_model]
            disable_revin: If True, skip RevIN normalization (already applied externally)
            
        Returns:
            output: TGTSF prediction [B, pred_len, C]
        """
        # Convert description to [bs, l, nvars, d_model]
        channel_description = channel_description.unsqueeze(1)  # [bs, 1, nvars, d_model]
        description = channel_description.repeat(1, news.shape[1], 1, 1)  # [bs, l, nvars, d_model]
        
        # Apply RevIN normalization only if not disabled and RevIN is enabled
        if self.revin and not disable_revin:
            x_mean = torch.mean(x_norm, dim=1, keepdim=True)
            x = x_norm - x_mean
            x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = x / torch.sqrt(x_var)
        else:
            x = x_norm
        
        # Time series encoding
        x = self.TS_encoder(x)  # x: [bs x nvars x d_model x patch_num]
        
        # Text encoding
        t = self.text_encoder(news, description)  # t: [bs, l, nvars, d_model]
        
        # Cross-modal mixing
        x, mix_weights = self.mixer(t, x)  # x: [bs, patch_num, nvars, d_model]
        
        # Output projection
        x = self.head(x)  # x: [bs, patch_num, nvars, patch_len]
        x = x.permute(0, 2, 1, 3)
        
        # Reconstruct output from patches
        B, C, N, L = x.shape
        
        output = torch.zeros(B, C, self.total_length).to(x.device)
        count = torch.zeros(B, C, self.total_length).to(x.device)
        for i in range(N):
            start = i * self.stride
            end = start + self.patch_len
            output[:, :, start:end] += x[:, :, i, :]
            count[:, :, start:end] += 1
        
        output = output[:, :, :self.pred_len]  # [bs, nvars, pred_len]
        count = count[:, :, :self.pred_len]
        
        output = output / count
        
        x = output[:, :, :self.pred_len]  # [bs, nvars, pred_len]
        x = x.permute(0, 2, 1)  # [bs, pred_len, nvars]
        
        # Apply RevIN denormalization only if RevIN was applied in this forward
        if self.revin and not disable_revin:
            x = x * torch.sqrt(x_var) + x_mean
        
        if self.out_attn_weights:
            return x, mix_weights
        else:
            return x
    
    def forward(self, x, news, channel_description, **kwargs):
        """
        Forward pass: Combine unimodal prediction with TGTSF residual.
        
        Args:
            x: Input time series [B, seq_len, C]
            news: News embeddings [B, l, news_num, text_dim]
            channel_description: Channel descriptions [B, 1, C, d_model]
            **kwargs: Additional arguments (ignored)
            
        Returns:
            final_pred: Final prediction [B, pred_len, C]
        """
        # Step 1: Apply normalization matching the pretrained model
        x_norm, norm_params = self._apply_normalization(x)
        
        # Step 2: Get unimodal prediction
        # Note: iTransformer's forecast() applies normalization internally and returns denormalized output
        # We need to normalize it back to combine with residual in normalized space
        if self.use_pretrained_norm:
            # iTransformer applies normalization internally, so pass original (unnormalized) input
            with torch.no_grad():
                unimodal_pred_denorm = self.unimodal_model(x)  # [B, pred_len, C] (denormalized)
            
            # Normalize the unimodal prediction back to normalized space
            # Use the same normalization params as input normalization
            # For use_norm: (pred - mean) / std
            means = norm_params['means']  # [B, 1, C]
            stdev = norm_params['stdev']  # [B, 1, C]
            # Expand to match prediction length
            means_pred = means[:, 0, :].unsqueeze(1).repeat(1, unimodal_pred_denorm.shape[1], 1)  # [B, pred_len, C]
            stdev_pred = stdev[:, 0, :].unsqueeze(1).repeat(1, unimodal_pred_denorm.shape[1], 1)  # [B, pred_len, C]
            unimodal_pred_norm = (unimodal_pred_denorm - means_pred) / stdev_pred
        else:
            # Pretrained model doesn't apply normalization internally, use normalized input
            with torch.no_grad():
                unimodal_pred_norm = self.unimodal_model(x_norm)  # [B, pred_len, C]
        
        # Step 3: Get TGTSF residual prediction (operates on normalized input)
        # Disable RevIN in TGTSF if pretrained model uses different normalization
        disable_revin = self.use_pretrained_norm
        residual_pred_norm = self._tgtsf_forward(
            x_norm, news, channel_description, disable_revin=disable_revin
        )
        
        # Step 4: Combine predictions in normalized space
        final_pred_norm = unimodal_pred_norm + residual_pred_norm
        
        # Step 5: Denormalize final prediction
        final_pred = self._denormalize(final_pred_norm, norm_params)
        
        return final_pred
    
    def move_to_device(self, seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, 
                      hetero_x_time, hetero_y_time, hetero_general, hetero_channel, device):
        """
        Move data to device (same as TGTSF for compatibility).
        
        Args:
            seq_x: Input sequences
            seq_y: Target sequences
            x_time: Input timestamps
            y_time: Target timestamps
            x_hetero: Input heterogeneous features
            y_hetero: Target heterogeneous features
            hetero_x_time: Heterogeneous input timestamps
            hetero_y_time: Heterogeneous target timestamps
            hetero_general: General heterogeneous features
            hetero_channel: Channel descriptions
            device: Target device
            
        Returns:
            tuple: All inputs moved to device
        """
        # Move data to device
        seq_x = seq_x.float().to(device)
        seq_y = seq_y.float().to(device)
        hetero_channel = hetero_channel.float().to(device)
        y_hetero = y_hetero.float().to(device)
        
        # Move unimodal model to device
        self.unimodal_model = self.unimodal_model.to(device)
        
        return seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel

