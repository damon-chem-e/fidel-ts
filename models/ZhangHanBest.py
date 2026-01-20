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
        # For PatchTST/iTransformer/FEDformer/Informer: d_model from config
        # For DLinear: pred_len (need to handle projection)
        # For FITS: dominance_freq * 2 (real + imaginary)
        # For Sundial/TimeMoE: hidden_size from model config (need to handle projection)
        self.ts_rep_dim = getattr(configs, 'd_model', 512)  # Default from config
        
        # Handle model-specific dimension cases
        if self.unimodal_model_type == 'DLinear':
            # DLinear returns [B, pred_len] after linear projections
            d_model_from_config = getattr(configs, 'd_model', None)
            if d_model_from_config is None or d_model_from_config != self.pred_len:
                # Add projection layer to match d_model
                self.ts_proj = nn.Linear(self.pred_len, self.ts_rep_dim)
                print(f'[ info ] ZhangHanBest: Added projection layer for DLinear: {self.pred_len} -> {self.ts_rep_dim}')
            else:
                self.ts_proj = None
                self.ts_rep_dim = self.pred_len
        elif self.unimodal_model_type == 'FITS':
            # FITS returns [B, dominance_freq * 2] (real + imaginary frequency components)
            # Need to project to d_model
            H_order = getattr(configs, 'H_order', 1)
            base_T = getattr(configs, 'base_T', 24)
            dominance_freq = int(self.seq_len // base_T + 1) * H_order + 10
            fits_repr_dim = dominance_freq * 2  # Real + imaginary
            
            d_model_from_config = getattr(configs, 'd_model', None)
            if d_model_from_config is None or d_model_from_config != fits_repr_dim:
                # Add projection layer to match d_model
                self.ts_proj = nn.Linear(fits_repr_dim, self.ts_rep_dim)
                print(f'[ info ] ZhangHanBest: Added projection layer for FITS: {fits_repr_dim} -> {self.ts_rep_dim}')
            else:
                self.ts_proj = None
                self.ts_rep_dim = fits_repr_dim
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
            # PatchTST, iTransformer, FEDformer, Informer all use d_model directly
            self.ts_proj = None
        
        # Validate ts_rep_dim is an integer (handle cases where configs might pass tuples/other types)
        self.ts_rep_dim = int(self.ts_rep_dim) if self.ts_rep_dim is not None else 512
        if self.ts_rep_dim <= 0:
            raise ValueError(
                f"ZhangHanBest requires a positive integer ts_rep_dim (d_model); got {self.ts_rep_dim!r}"
            )
        
        # 2. Text input dimension (from pre-computed embeddings)
        # NOTE: `dotdict.__getattr__` returns None for missing keys, so treat None as unset.
        raw_input_text_dim = getattr(configs, 'input_text_dim', None)
        self.text_dim = 768 if raw_input_text_dim is None else raw_input_text_dim

        if not isinstance(self.text_dim, int) or self.text_dim <= 0:
            raise ValueError(
                f"ZhangHanBest requires a positive integer input_text_dim; got {self.text_dim!r}. "
                f"Set it via model_config_overrides.input_text_dim (typically 768 for BERT embeddings)."
            )
        
        # 3. Residual projection (always uses residual connection)
        # Architecture per paper: text_dim -> hidden_dim -> ts_rep_dim + residual
        # Default hidden_dim=2048 for 768 -> 2048 -> 512 projection
        raw_hidden_dim = getattr(configs, 'residual_proj_hidden_dim', 2048)
        hidden_dim = int(raw_hidden_dim) if raw_hidden_dim is not None else 2048
        if hidden_dim <= 0:
            raise ValueError(
                f"ZhangHanBest requires a positive integer residual_proj_hidden_dim; got {raw_hidden_dim!r} -> {hidden_dim!r}"
            )
        
        self.residual_proj = ResidualProjection(
            text_dim=self.text_dim,
            ts_rep_dim=self.ts_rep_dim,
            hidden_dim=hidden_dim,
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
        
        # 7. Timestamp semantics (for Fidel-TS vs Time-MMD datasets)
        # Determines which heterogeneous data source to use:
        # - t_about: Use y_hetero (news) - forecasts ABOUT prediction window (Fidel-TS)
        # - t_known: Use x_hetero (historical_events) - text from input window (Time-MMD)
        # - None: Use hetero_general (aggregate dataset description) - legacy behavior
        self.timestamp_semantics = getattr(configs, 'timestamp_semantics', None)
        
        if self.timestamp_semantics is not None:
            # If set, validate it's correct
            if self.timestamp_semantics not in ('t_about', 't_known'):
                raise ValueError(
                    f"Invalid timestamp_semantics: '{self.timestamp_semantics}'. "
                    f"Must be 't_about' or 't_known'."
                )
            print(f'[ info ] ZhangHanBest: timestamp_semantics = {self.timestamp_semantics}')
            if self.timestamp_semantics == 't_about':
                print(f'         -> Using news (y_hetero) - forecasts ABOUT prediction window')
            else:
                print(f'         -> Using historical_events (x_hetero) - avoiding lookahead bias')
        else:
            # Legacy mode: use hetero_general
            print(f'[ info ] ZhangHanBest: Using legacy mode (hetero_general aggregate text)')
            print(f'         -> Set timestamp_semantics in config to use per-timestep text')
    
    def _create_unimodal_encoder(self, configs):
        """Create and configure unimodal time series encoder."""
        # Import and instantiate unimodal model
        if self.unimodal_model_type == 'PatchTST':
            from models.PatchTST import Model as PatchTSTModel
            return PatchTSTModel(configs)
        elif self.unimodal_model_type == 'DLinear':
            from models.DLinear import Model as DLinearModel
            return DLinearModel(configs)
        elif self.unimodal_model_type == 'iTransformer':
            from models.iTransformer import Model as iTransformerModel
            return iTransformerModel(configs)
        elif self.unimodal_model_type == 'FITS':
            from models.FITS import Model as FITSModel
            return FITSModel(configs)
        elif self.unimodal_model_type == 'FEDformer':
            from models.FEDformer import Model as FEDformerModel
            return FEDformerModel(configs)
        elif self.unimodal_model_type == 'Informer':
            from models.Informer import Model as InformerModel
            return InformerModel(configs)
        elif self.unimodal_model_type == 'Sundial':
            from models.Sundial import Model as SundialModel
            return SundialModel(configs)
        elif self.unimodal_model_type == 'TimeMoE':
            from models.TimeMoE import Model as TimeMoEModel
            return TimeMoEModel(configs)
        else:
            raise ValueError(f"Unsupported unimodal_model_type: {self.unimodal_model_type}")
    
    def forward(self, x, news=None, historical_events=None, **kwargs):
        """
        Forward pass.
        
        Args:
            x: Time series input [B, seq_len, C]
            news: Text embeddings aligned to PREDICTION window (y_hetero from dataloader)
                  Shape: [B, pred_len, num_items, text_dim]
                  Used when timestamp_semantics='t_about'
            historical_events: Text embeddings aligned to INPUT window (x_hetero from dataloader)
                              Shape: [B, seq_len, num_items, text_dim]
                              Used when timestamp_semantics='t_known'
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
        # Pass news and historical_events explicitly for timestamp_semantics support
        text_repr = self._get_text_embeddings(news, historical_events, kwargs)  # [B, text_dim]
        
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
    
    def _get_text_embeddings(self, news, historical_events, kwargs):
        """
        Extract pre-computed text embeddings from appropriate source based on timestamp_semantics.
        
        Three modes:
        1. timestamp_semantics='t_about': Use news (y_hetero) - Fidel-TS datasets
        2. timestamp_semantics='t_known': Use historical_events (x_hetero) - Time-MMD datasets
        3. timestamp_semantics=None: Use hetero_general (legacy) - aggregate dataset description
        
        Args:
            news: y_hetero - text aligned to prediction window [B, pred_len, num_items, text_dim]
            historical_events: x_hetero - text aligned to input window [B, seq_len, num_items, text_dim]
            kwargs: Additional arguments including dataset_description/hetero_general
        
        Returns:
            text_repr: [B, text_dim] - pre-computed text embeddings (aggregated per sample)
        """
        text_repr = None
        
        # Select text source based on timestamp_semantics
        if self.timestamp_semantics == 't_about':
            # Use news (y_hetero) - forecasts ABOUT prediction window
            if news is None:
                raise ValueError(
                    "timestamp_semantics='t_about' requires 'news' (y_hetero) to be provided. "
                    "Ensure task='TGTSF' and y_hetero is configured in data config."
                )
            text_repr = news
        elif self.timestamp_semantics == 't_known':
            # Use historical_events (x_hetero) - text from input window
            if historical_events is None:
                raise ValueError(
                    "timestamp_semantics='t_known' requires 'historical_events' (x_hetero) to be provided. "
                    "Ensure x_hetero is configured in data config."
                )
            text_repr = historical_events
        else:
            # Legacy mode: Use hetero_general (aggregate dataset description)
            # Priority order: dataset_description (standard name) > hetero_general (legacy name)
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
            # iTransformer-style normalization (use_norm)
            means = x.mean(1, keepdim=True)  # [B, 1, C]
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [B, 1, C]
            
            # Denormalize
            predictions = predictions * stdev + means
        elif self.unimodal_model_type == 'FITS':
            # FITS uses RevIN normalization (RIN)
            # Same as PatchTST RevIN
            x_mean = torch.mean(x, dim=1, keepdim=True)  # [B, 1, C]
            x_var = torch.var(x, dim=1, keepdim=True) + 1e-5  # [B, 1, C]
            x_std = torch.sqrt(x_var)  # [B, 1, C]
            
            predictions = predictions * x_std + x_mean
        elif self.unimodal_model_type in ['FEDformer', 'Informer']:
            # These models don't use normalization by default (handled by DataEmbedding)
            # No denormalization needed
            pass
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
    
    def move_to_device(self, seq_x, seq_y, x_time, y_time, x_hetero, y_hetero,
                      hetero_x_time, hetero_y_time, hetero_general, hetero_channel, device):
        """
        Move data to device.
        
        Args:
            seq_x: Input time series [B, seq_len, C]
            seq_y: Target time series [B, pred_len, C]
            x_time: Input timestamps
            y_time: Target timestamps
            x_hetero: Historical text embeddings (maps to historical_events in forward)
                      Text aligned to INPUT window - always safe to use
            y_hetero: Prediction window text embeddings (maps to news in forward)
                      Text aligned to PREDICTION window
                      - For t_about: Forecasts ABOUT prediction window (safe with lead time assumption)
                      - For t_known: Text PUBLISHED during prediction window (LOOKAHEAD - don't use!)
            hetero_x_time: Historical text timestamps
            hetero_y_time: Prediction text timestamps
            hetero_general: General dataset description
            hetero_channel: Channel descriptions
            device: Target device
            
        Returns:
            tuple: All inputs with relevant tensors moved to device
        """
        # Move time series data
        seq_x = seq_x.float().to(device)
        seq_y = seq_y.float().to(device)
        
        # Move text based on timestamp_semantics
        if self.timestamp_semantics == 't_about':
            # Use y_hetero (news) - forecasts ABOUT prediction window
            y_hetero = y_hetero.float().to(device)
        elif self.timestamp_semantics == 't_known':
            # Use x_hetero (historical_events) - avoids lookahead bias
            x_hetero = x_hetero.float().to(device)
        # else: legacy mode, no per-timestep text to move
        
        return seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel

