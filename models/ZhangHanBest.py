"""
ZhangHanBest Model: Multimodal fusion of time series and text.

Based on Zhang, Han, et al. (2025) from Amazon team.
Architecture:
1. Time series encoder (unimodal model, returns representations)
2. Text embeddings (pre-computed via TextEmbedder)
3. Residual projection (text -> TS representation space)
4. Fusion (weighted addition)
5. Prediction head (fused representation -> predictions)
"""

import torch
import torch.nn as nn
import numpy as np
from layers.zhanghanbest_layers import ResidualProjection, PredictionHead


class Model(nn.Module):
    """
    ZhangHanBest Model: Multimodal fusion of time series and text.
    Based on Zhang, Han, et al. (2025) from Amazon team,
    "When does Multimodality Lead to Better Time Series Forecasting?"
    Best implementation of the alignment methods tested in the paper:
    average pooling of text embeddings, late fusion, addition aggregation,
    and residual projection.
    
    Architecture:
    1. Time series encoder (unimodal model, returns representations)
    2. Text embeddings (pre-computed via TextEmbedder, aggregated)
    3. Residual projection (text -> TS representation space)
    4. Fusion (weighted addition)
    5. Prediction head (fused representation -> predictions)
    """
    
    def __init__(self, configs):
        super().__init__()
        
        # Store config
        self.configs = configs
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        
        # 1. Time series encoder (unimodal model)
        self.unimodal_model_type = getattr(configs, 'unimodal_model_type', 'PatchTST')
        self.ts_encoder = self._create_unimodal_encoder(configs)
        
        # Get representation dimension from TS encoder
        # For PatchTST: d_model from config
        # For DLinear: pred_len (need to handle projection)
        # For Sundial/TimeMoE: hidden_size from model config (need to handle projection)
        self.ts_rep_dim = getattr(configs, 'd_model', 512)  # Default from config
        
        # Handle DLinear special case: representation dim is pred_len, not d_model
        if self.unimodal_model_type == 'DLinear':
            # DLinear returns [B, pred_len] after linear projections, so we need to project to d_model
            # Or set d_model=pred_len in config
            d_model_from_config = getattr(configs, 'd_model', None)
            if d_model_from_config is None or d_model_from_config != self.pred_len:
                # Add projection layer to match d_model
                self.ts_proj = nn.Linear(self.pred_len, self.ts_rep_dim)
                print(f'[ info ] ZhangHanBest: Added projection layer for DLinear: {self.pred_len} -> {self.ts_rep_dim}')
            else:
                self.ts_proj = None
                self.ts_rep_dim = self.pred_len
        elif self.unimodal_model_type in ['Sundial', 'TimeMoE']:
            # Sundial and TimeMoE return [B, hidden_size] from their internal representations
            # Get actual hidden_size from the encoder model
            encoder_hidden_size = getattr(self.ts_encoder, 'hidden_size', None)
            if encoder_hidden_size is None:
                raise ValueError(f"Encoder {self.unimodal_model_type} does not have hidden_size attribute")
            
            d_model_from_config = getattr(configs, 'd_model', None)
            if d_model_from_config is None or d_model_from_config != encoder_hidden_size:
                # Add projection layer to match d_model
                self.ts_proj = nn.Linear(encoder_hidden_size, self.ts_rep_dim)
                print(f'[ info ] ZhangHanBest: Added projection layer for {self.unimodal_model_type}: {encoder_hidden_size} -> {self.ts_rep_dim}')
            else:
                self.ts_proj = None
                self.ts_rep_dim = encoder_hidden_size
        else:
            self.ts_proj = None
        
        # 2. Text input dimension (from pre-computed embeddings)
        self.text_dim = getattr(configs, 'input_text_dim', 768)  # Embedding dimension (typically 768 for BERT)
        
        # 3. Residual projection (always uses residual connection)
        self.residual_proj = ResidualProjection(
            text_dim=self.text_dim,
            ts_rep_dim=self.ts_rep_dim,
            use_layer_norm=getattr(configs, 'residual_proj_use_layer_norm', True),
            activation=getattr(configs, 'residual_proj_activation', 'gelu'),
            dropout=getattr(configs, 'residual_proj_dropout', 0.1)
        )
        
        # 4. Fusion weight (fixed or learned)
        fusion_weight_mode = getattr(configs, 'fusion_weight_mode', 'fixed')  # 'fixed' or 'learned'
        if fusion_weight_mode == 'learned':
            # Learned parameter (sigmoid applied in forward to constrain to [0, 1])
            initial_value = getattr(configs, 'fusion_weight_initial', 0.5)
            self.fusion_weight_raw = nn.Parameter(torch.tensor(initial_value))
            self.fusion_weight_mode = 'learned'
        else:
            # Fixed hyperparameter
            fusion_weight_value = getattr(configs, 'fusion_weight', 0.5)
            self.register_buffer('fusion_weight_raw', torch.tensor(fusion_weight_value))
            self.fusion_weight_mode = 'fixed'
        
        # 5. Prediction head
        self.prediction_head = PredictionHead(
            d_model=self.ts_rep_dim,
            pred_len=self.pred_len,
            n_variates=self.enc_in,
            use_mlp=getattr(configs, 'pred_head_use_mlp', False),
            hidden_dim=getattr(configs, 'pred_head_hidden_dim', None),
            dropout=getattr(configs, 'pred_head_dropout', 0.1)
        )
        
        # 6. Normalization tracking
        # Track if TS model uses normalization internally
        self.ts_uses_revin = getattr(configs, 'revin', True)  # PatchTST default
        self.ts_uses_norm = getattr(configs, 'use_norm', False)  # iTransformer style
    
    def _create_unimodal_encoder(self, configs):
        """Create and configure unimodal time series encoder."""
        # Import and instantiate unimodal model
        if self.unimodal_model_type == 'PatchTST':
            from models.PatchTST import Model as PatchTSTModel
            return PatchTSTModel(configs)
        elif self.unimodal_model_type == 'DLinear':
            from models.DLinear import Model as DLinearModel
            return DLinearModel(configs)
        elif self.unimodal_model_type == 'Sundial':
            from models.Sundial import Model as SundialModel
            return SundialModel(configs)
        elif self.unimodal_model_type == 'TimeMoE':
            from models.TimeMoE import Model as TimeMoEModel
            return TimeMoEModel(configs)
        else:
            raise ValueError(f"Unsupported unimodal_model_type: {self.unimodal_model_type}")
    
    def forward(self, x, **kwargs):
        """
        Forward pass.
        
        Args:
            x: Time series input [B, seq_len, C]
            **kwargs: May include text inputs:
                - dataset_description: Aggregate text embeddings [B, 1, text_dim] or [B, text_dim]
                - hetero_general: Alternative name for aggregate text (legacy)
        
        Returns:
            predictions: [B, pred_len, C]
        """
        # 1. Get time series representations (aggregated)
        # Call unimodal model with return_representations=True
        ts_repr = self.ts_encoder(x, return_representations=True)  # [B, d_model] or [B, pred_len] for DLinear, [B, hidden_size] for Sundial/TimeMoE
        
        # Handle projection if needed (for DLinear, Sundial, TimeMoE when dimensions don't match)
        if self.ts_proj is not None:
            ts_repr = self.ts_proj(ts_repr)  # Project to [B, d_model]
        
        # 2. Get text representation from pre-computed embeddings
        text_repr = self._get_text_embeddings(kwargs)  # [B, text_dim]
        
        # 3. Project text to TS representation space (residual projection)
        text_repr_proj = self.residual_proj(text_repr)  # [B, d_model]
        
        # 4. Get fusion weight (fixed or learned)
        if self.fusion_weight_mode == 'learned':
            fusion_weight = torch.sigmoid(self.fusion_weight_raw)  # Constrain to [0, 1]
        else:
            fusion_weight = self.fusion_weight_raw  # Fixed value
        
        # 5. Fuse representations (both are [B, d_model])
        fused_repr = self.fuse_representations(text_repr_proj, ts_repr, fusion_weight)  # [B, d_model]
        
        # 6. Prediction (aggregated representation -> predictions for all channels)
        predictions = self.prediction_head(fused_repr)  # [B, pred_len, C]
        
        # 7. Denormalize predictions (if TS model normalized internally)
        # Note: TS model handles normalization internally when extracting representations
        # We need to track normalization params from input and apply inverse to predictions
        predictions = self._denormalize_predictions(predictions, x)
        
        return predictions
    
    def _get_text_embeddings(self, kwargs):
        """
        Extract pre-computed text embeddings from kwargs.
        
        Uses new embeddings system:
        - Time-MMD/TTC: Aggregate text from hetero_general or dataset_description
        - Fidel-TS: Dynamic aggregate text (ignore static channel descriptions)
        
        Returns:
            text_repr: [B, text_dim] - pre-computed text embeddings (aggregated per sample)
        """
        # Priority order: dataset_description (standard name) > hetero_general (legacy name)
        text_repr = None
        if 'dataset_description' in kwargs and kwargs['dataset_description'] is not None:
            text_repr = kwargs['dataset_description']
        elif 'hetero_general' in kwargs and kwargs['hetero_general'] is not None:
            text_repr = kwargs['hetero_general']
        else:
            raise ValueError(
                "Text embeddings not found in kwargs. "
                "Expected 'dataset_description' or 'hetero_general' with pre-computed embeddings. "
                "Ensure embeddings are computed by TextEmbedder in data loader."
            )
        
        # Convert to tensor if numpy array
        if isinstance(text_repr, np.ndarray):
            text_repr = torch.from_numpy(text_repr).float()
        
        # Handle different input shapes
        if len(text_repr.shape) == 3:
            # [B, 1, text_dim] -> [B, text_dim] (Time-MMD format with aggregation)
            text_repr = text_repr.squeeze(1)
        elif len(text_repr.shape) == 2:
            # Already [B, text_dim] - correct format
            pass
        elif len(text_repr.shape) == 4:
            # [B, seq_len, num_items, text_dim] - need aggregation (Fidel-TS format)
            # Aggregate over time and items: mean -> [B, text_dim]
            text_repr = text_repr.mean(dim=(1, 2))
        else:
            raise ValueError(
                f"Unexpected text embedding shape: {text_repr.shape}. "
                f"Expected [B, text_dim], [B, 1, text_dim], or [B, seq_len, num_items, text_dim]"
            )
        
        # Ensure correct device and dtype
        device = next(self.parameters()).device
        text_repr = text_repr.to(device)
        
        # Validate final shape
        if len(text_repr.shape) != 2 or text_repr.shape[1] != self.text_dim:
            raise ValueError(
                f"Text embedding shape mismatch: got {text_repr.shape}, expected [B, {self.text_dim}]"
            )
        
        return text_repr  # [B, text_dim]
    
    def _denormalize_predictions(self, predictions, x):
        """
        Denormalize predictions if TS model normalized internally.
        
        Args:
            predictions: [B, pred_len, C] - normalized predictions
            x: [B, seq_len, C] - original input (for normalization params)
        
        Returns:
            predictions: [B, pred_len, C] - denormalized predictions
        """
        # Check if TS model uses normalization
        if self.unimodal_model_type == 'PatchTST' and self.ts_uses_revin:
            # PatchTST uses RevIN internally in backbone
            # Extract normalization params from input
            x_mean = torch.mean(x, dim=1, keepdim=True)  # [B, 1, C]
            x_var = torch.var(x, dim=1, keepdim=True) + 1e-5  # [B, 1, C]
            x_std = torch.sqrt(x_var)  # [B, 1, C]
            
            # Denormalize: predictions were normalized, so reverse the normalization
            # predictions_norm = (predictions_orig - mean) / std
            # predictions_orig = predictions_norm * std + mean
            predictions = predictions * x_std + x_mean
        elif self.unimodal_model_type in ['iTransformer', 'Sundial', 'TimeMoE'] and self.ts_uses_norm:
            # iTransformer-style normalization
            means = x.mean(1, keepdim=True)  # [B, 1, C]
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [B, 1, C]
            
            # Denormalize
            predictions = predictions * stdev + means
        # For DLinear: no normalization, so no denormalization needed
        
        return predictions
    
    def fuse_representations(self, text_repr, ts_repr, fusion_weight):
        """
        Fuse text and time series representations via weighted addition.
        
        Args:
            text_repr: Text representation [B, d_model] (already projected)
            ts_repr: Time series representation [B, d_model] (already aggregated)
            fusion_weight: Weight w (scalar or tensor [1] if learned)
        
        Returns:
            fused_repr: Fused representation [B, d_model]
        """
        fused = fusion_weight * text_repr + (1 - fusion_weight) * ts_repr
        return fused

