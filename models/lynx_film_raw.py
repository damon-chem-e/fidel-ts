"""
lynx-film-raw Model: FiLM-Modulated Raw Signal Learning

This model learns the entire prediction structure using iTransformerFilm with FiLM modulation,
without learning a residual to a frozen unimodal model. Similar to TGTSF but using iTransformer
architecture with FiLM instead of TGTSF's patch-based architecture.

Text Input Nomenclature
=======================
This model receives text from two sources, with different names in the
dataloader vs model code:

    Dataloader Name    Model Parameter      Description
    ---------------    ---------------      -----------
    x_hetero           historical_events    Text aligned to INPUT window timestamps
    y_hetero           news                 Text aligned to PREDICTION window timestamps

The model uses `timestamp_semantics` to determine which text input to use:

    timestamp_semantics    Text Input Used    Datasets
    -------------------    ---------------    --------
    t_about                news (y_hetero)    Fidel-TS (timestamps = target time)
    t_known                historical_events  Time-MMD, TTC (timestamps = publication time)

For t_about:
    news contains forecasts/schedules ABOUT the prediction window.
    ASSUMPTION: The forecast for time t+k was known at time t.
    This is assumed safe for prediction horizons within typical forecast lead times.

For t_known:
    news would contain text PUBLISHED during the prediction window, which is
    LOOKAHEAD BIAS. Instead, we use historical_events (text about input window).
"""

from torch import nn
import torch
import copy
import os
import numpy as np
from layers.lynx_text_encoder import LynxTextEncoder
from layers.lynx_film_layers import iTransformerFilm
from utils.model_regularization import apply_norms_to_linear_layers

class Model(nn.Module):
    """
    lynx-film-raw Model:
    - Base: iTransformerFilm (modulated by text via FiLM)
    - Learning: Raw signal prediction (no residual learning)
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # Store normalization configuration
        self.revin = getattr(configs, 'revin', True)
        self.use_norm = getattr(configs, 'use_norm', False)
        
        # Timestamp semantics determines which text input to use for TGTSF task
        # - t_about: Use news (y_hetero) - text describes prediction window, assumed known beforehand
        # - t_known: Use historical_events (x_hetero) - avoid lookahead bias
        self.timestamp_semantics = getattr(configs, 'timestamp_semantics', None)
        if self.timestamp_semantics is None:
            raise ValueError(
                "lynx_film_raw requires 'timestamp_semantics' in configs. "
                "This must be set explicitly to avoid lookahead bias:\n"
                "  - 't_about': Text timestamps refer to the event/target time (Fidel-TS datasets)\n"
                "  - 't_known': Text timestamps refer to publication time (Time-MMD/TTC datasets)\n"
                "Set timestamp_semantics in data_config (hetero_info.timestamp_semantics or top-level)."
            )
        if self.timestamp_semantics not in ('t_about', 't_known'):
            raise ValueError(
                f"Invalid timestamp_semantics: '{self.timestamp_semantics}'. "
                f"Must be 't_about' or 't_known'."
            )
        print(f'[ info ] LYNX-FiLM-raw: timestamp_semantics = {self.timestamp_semantics}')
        if self.timestamp_semantics == 't_about':
            print('         -> Using news (y_hetero) - forecasts ABOUT prediction window')
        else:
            print('         -> Using historical_events (x_hetero) - avoiding lookahead bias')
        
        # Flag to log text embedding shape only once (on first batch)
        self._logged_text_shape = False
        
        # Text dimension handling:
        # - input_text_dim: Dimension of input text embeddings (e.g., 768 for BERT)
        # - text_dim: Operational dimension used internally by the model (e.g., 256)
        # If input_text_dim != text_dim, a learned projection layer is added
        #
        # NOTE: `dotdict.__getattr__` returns None for missing keys, so treat None as unset.
        self.text_dim = configs.text_dim
        raw_input_text_dim = getattr(configs, 'input_text_dim', None)
        self.input_text_dim = self.text_dim if raw_input_text_dim is None else raw_input_text_dim

        if not isinstance(self.input_text_dim, int) or self.input_text_dim <= 0:
            raise ValueError(
                f"LYNX-FiLM-raw requires a positive integer input_text_dim; got {self.input_text_dim!r}. "
                f"Set it via model_config_overrides.input_text_dim (e.g., 256 for old embeddings, 768 for BERT)."
            )

        # Learned projection layer if input dimension differs from operational dimension
        if self.input_text_dim != self.text_dim:
            self.text_projection = nn.Linear(self.input_text_dim, self.text_dim)
            print(f'[ info ] LYNX-FiLM-raw: Added learned projection layer {self.input_text_dim} -> {self.text_dim}')
        else:
            self.text_projection = None

        # 1. Text Encoder (from TGTSF)
        # Used to get text embeddings for FiLM
        self.text_encoder = LynxTextEncoder(
            cross_layer=configs.cross_layers,
            self_layer=configs.self_layers,
            embedding_dim=configs.text_dim,
            num_heads=configs.n_heads,
            dropout=configs.dropout,
            pred_len=configs.pred_len,
            stride=configs.stride,
            encoder_type=getattr(configs, 'text_encoder_type', 'cross'),
            mlp_hidden_dim=getattr(configs, 'text_encoder_mlp_hidden_dim', None),
            mlp_dropout=getattr(configs, 'text_encoder_mlp_dropout', 0.0)
        )
        # 2. Main Model (iTransformerFilm)
        # Disable internal normalization because we handle normalization externally
        # This allows us to use either RevIN or use_norm normalization scheme
        model_configs = copy.deepcopy(configs)
        model_configs.use_norm = False 
        self.model = iTransformerFilm(model_configs)
        
        # Cache text/FiLM normalization settings for post-device application
        self._store_text_film_norms_config(configs)

    def _store_text_film_norms_config(self, configs) -> None:
        """
        Store normalization settings for later application on the correct device.
        """
        # Read normalization flags from config
        self._text_film_use_weight_norm = getattr(configs, 'text_film_weight_norm', False)
        self._text_film_use_spectral_norm = getattr(configs, 'text_film_spectral_norm', False)
        # Track whether norms have been applied
        self._text_film_norms_applied = False

    def apply_text_film_norms_on_device(self) -> None:
        """
        Apply optional weight/spectral normalization after model is on device.
        """
        # Skip if already applied
        if self._text_film_norms_applied:
            return
        # Apply norms based on stored flags
        self._apply_text_film_norms(
            use_weight_norm=self._text_film_use_weight_norm,
            use_spectral_norm=self._text_film_use_spectral_norm
        )
        # Mark as applied
        self._text_film_norms_applied = True

    def _apply_text_film_norms(self, use_weight_norm: bool, use_spectral_norm: bool) -> None:
        """
        Apply optional weight or spectral normalization to text/FiLM submodules.
        
        This targets:
        - text_projection (if present)
        - text_encoder
        - FiLM generators inside iTransformerFilm
        """
        # Skip if no normalization is requested
        if not (use_weight_norm or use_spectral_norm):
            return
        # Disallow incompatible simultaneous norms
        if use_weight_norm and use_spectral_norm:
            raise ValueError("Only one of text_film_weight_norm or text_film_spectral_norm can be True.")
        # Apply normalization to text projection if it exists
        if self.text_projection is not None:
            apply_norms_to_linear_layers(
                self.text_projection,
                use_weight_norm=use_weight_norm,
                use_spectral_norm=use_spectral_norm
            )
        # Apply normalization to text encoder
        apply_norms_to_linear_layers(
            self.text_encoder,
            use_weight_norm=use_weight_norm,
            use_spectral_norm=use_spectral_norm
        )
        # Apply normalization to FiLM generators
        self.model.apply_film_param_norms(
            use_weight_norm=use_weight_norm,
            use_spectral_norm=use_spectral_norm
        )
    
    def _project_text_embeddings(self, news, channel_description):
        """
        Project text embeddings from input_text_dim to text_dim if needed.
        
        Args:
            news: Text embeddings [B, l, news_num, input_text_dim]
                  - x_hetero (historical_events): [B, seq_len, num_items, input_text_dim]
                  - y_hetero (news): [B, pred_len, num_items, input_text_dim]
                  Both come from the same dataloader format: (num_timesteps, num_items, embedding_dim) -> [B, L, N, D]
            channel_description: Channel descriptions [B, C, input_text_dim] or [B, 1, C, input_text_dim]
            
        Returns:
            news: Projected news embeddings [B, l, news_num, text_dim]
            channel_description: Projected channel descriptions [B, C, text_dim] or [B, 1, C, text_dim]
        """
        if self.text_projection is None:
            return news, channel_description
        
        # Project news embeddings: [B, l, news_num, input_text_dim] -> [B, l, news_num, text_dim]
        B, L, N, D = news.shape
        news = news.reshape(B * L * N, D)  # Flatten for projection
        news = self.text_projection(news)  # Project
        news = news.reshape(B, L, N, self.text_dim)  # Reshape back
        
        # Project channel_description: [B, C, input_text_dim] or [B, 1, C, input_text_dim] -> [B, C, text_dim] or [B, 1, C, text_dim]
        if channel_description.ndim == 3:
            # [B, C, input_text_dim]
            B_desc, C_desc, D_desc = channel_description.shape
            channel_description = channel_description.reshape(B_desc * C_desc, D_desc)
            channel_description = self.text_projection(channel_description)
            channel_description = channel_description.reshape(B_desc, C_desc, self.text_dim)
        elif channel_description.ndim == 4:
            # [B, 1, C, input_text_dim]
            B_desc, _, C_desc, D_desc = channel_description.shape
            channel_description = channel_description.reshape(B_desc * C_desc, D_desc)
            channel_description = self.text_projection(channel_description)
            channel_description = channel_description.reshape(B_desc, 1, C_desc, self.text_dim)
        
        return news, channel_description
    
    def normalize_input(self, x):
        """
        Normalize input based on configured normalization scheme.
        
        Supports both RevIN (like TGTSF) and use_norm (like iTransformer) normalization.
        
        Args:
            x: Input time series [B, seq_len, C]
            
        Returns:
            tuple: (x_norm, norm_params)
                - x_norm: Normalized input [B, seq_len, C]
                - norm_params: Dictionary with normalization parameters for denormalization
        """
        if self.revin:
            # RevIN normalization (channel-wise)
            x_mean = torch.mean(x, dim=1, keepdim=True)
            x_norm = x - x_mean
            x_var = torch.var(x_norm, dim=1, keepdim=True) + 1e-5
            x_norm = x_norm / torch.sqrt(x_var)
            norm_params = {
                'type': 'revin',
                'x_mean': x_mean,
                'x_var': x_var
            }
        elif self.use_norm:
            # iTransformer normalization (Non-stationary Transformer normalization)
            means = x.mean(1, keepdim=True).detach()
            x_norm = x - means
            stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_norm = x_norm / stdev
            norm_params = {
                'type': 'use_norm',
                'means': means,
                'stdev': stdev
            }
        else:
            # No normalization
            x_norm = x
            norm_params = {'type': 'none'}
        
        return x_norm, norm_params
    
    def denormalize_output(self, pred_norm, norm_params):
        """
        Denormalize predictions using stored normalization parameters.
        
        Args:
            pred_norm: Normalized predictions [B, pred_len, C]
            norm_params: Normalization parameters from normalize_input
            
        Returns:
            torch.Tensor: Denormalized predictions [B, pred_len, C]
        """
        if norm_params['type'] == 'revin':
            # RevIN denormalization
            x_mean = norm_params['x_mean']
            x_var = norm_params['x_var']
            final_pred = pred_norm * torch.sqrt(x_var) + x_mean
        elif norm_params['type'] == 'use_norm':
            # iTransformer denormalization
            means = norm_params['means']  # [B, 1, C]
            stdev = norm_params['stdev']  # [B, 1, C]
            # Expand to match prediction length
            means_pred = means[:, 0, :].unsqueeze(1).repeat(1, pred_norm.shape[1], 1)  # [B, pred_len, C]
            stdev_pred = stdev[:, 0, :].unsqueeze(1).repeat(1, pred_norm.shape[1], 1)  # [B, pred_len, C]
            final_pred = pred_norm * stdev_pred + means_pred
        else:
            # No denormalization needed
            final_pred = pred_norm
        
        return final_pred
        
    def forward(self, x, news=None, channel_description=None, historical_events=None, **kwargs):
        """
        Forward pass: Learn raw signal prediction using iTransformerFilm with FiLM modulation.
        
        Args:
            x: Input time series [B, seq_len, C]
            news: Text embeddings aligned to PREDICTION window (y_hetero from dataloader)
                  Shape: [B, pred_len, num_items, text_dim]
                  - For t_about: Forecasts/schedules ABOUT the prediction window (USE THIS)
                  - For t_known: Text PUBLISHED during prediction window (LOOKAHEAD - don't use!)
            channel_description: Channel descriptions [B, C, d_model] or [B, 1, C, d_model]
                Will be automatically expanded to [B, l, C, d_model] to match text time dimension
            historical_events: Text embeddings aligned to INPUT window (x_hetero from dataloader)
                              Shape: [B, seq_len, num_items, text_dim]
                              - Always safe to use (describes past, known at prediction time)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            final_pred: Raw prediction [B, pred_len, C]
        """
        # Select text input based on timestamp_semantics
        # - t_about: news (y_hetero) contains forecasts ABOUT prediction window, assumed known at t
        # - t_known: news (y_hetero) would be LOOKAHEAD; use historical_events (x_hetero) instead
        if self.timestamp_semantics == 't_about':
            # Fidel-TS datasets: timestamps = t_about (what time text describes)
            # news contains forecasts/schedules for prediction window, assumed known beforehand
            if news is None:
                raise ValueError(
                    "timestamp_semantics='t_about' requires 'news' (y_hetero) to be provided. "
                    "Ensure task='TGTSF' and y_hetero is configured in data config."
                )
            text_input = news
        elif self.timestamp_semantics == 't_known':
            # Time-MMD/TTC datasets: timestamps = t_known (when text was published)
            # Using news (y_hetero) would be lookahead bias; use historical_events instead
            if historical_events is None:
                raise ValueError(
                    "timestamp_semantics='t_known' requires 'historical_events' (x_hetero) to be provided. "
                    "Ensure x_hetero is configured in data config."
                )
            text_input = historical_events
        else:
            raise ValueError(f"Invalid timestamp_semantics: {self.timestamp_semantics}")
        
        # Step 1: Normalize input
        x_norm, norm_params = self.normalize_input(x)
        
        # Step 2: Project text embeddings if input dimension differs from operational dimension
        text_input, channel_description = self._project_text_embeddings(text_input, channel_description)
        
        # Step 3: Get Text Embeddings
        # text_encoder returns [B, L, C, text_dim] (L=time segments, C=channels)
        # Transform channel_description to match text_encoder expected input shape [B, L, C, D]
        # Handle both [B, C, D] and [B, 1, C, D] input shapes
        if len(channel_description.shape) == 3:
            # If [B, C, D], unsqueeze to [B, 1, C, D]
            channel_description = channel_description.unsqueeze(1)
        # Repeat along time dimension to match text_input shape: [B, L, C, D]
        description = channel_description.repeat(1, text_input.shape[1], 1, 1)
        text_emb = self.text_encoder(text_input, description)
        
        # Step 3: Prepare text embeddings for FiLM
        # iTransformerFilm expects [B, C, L, text_dim]
        # Text encoder output is [B, L, C, text_dim], so we permute.
        # We do not pool over L, preserving sequence info.
        # Ensure contiguity after permute for torch.compile compatibility
        # The compiled attention kernels require contiguous tensors
        text_emb = text_emb.permute(0, 2, 1, 3).contiguous()  # [B, C, L, D]
        
        # Step 4: Get Raw Prediction from iTransformerFilm
        # Pass normalized input and text embeddings
        pred_norm = self.model(x_norm, text_emb)
        
        # Step 5: Denormalize prediction
        final_pred = self.denormalize_output(pred_norm, norm_params)
        
        return final_pred
    
    def _convert_hetero_channel_to_tensor(self, hetero_channel, device):
        """
        Convert hetero_channel to a tensor and move it to the specified device.
        
        Args:
            hetero_channel: Channel descriptions as tensor, numpy array, or list of arrays/tensors
            device: Target device for the tensor
            
        Returns:
            torch.Tensor: hetero_channel as a float tensor on the specified device
            
        Raises:
            ValueError: If hetero_channel is in an invalid format (e.g., strings)
        """
        # Handle list types (from DataLoader collation)
        if isinstance(hetero_channel, list):
            if len(hetero_channel) == 0:
                raise ValueError(
                    "Empty list received for hetero_channel. This indicates a data loading issue."
                )
            elif isinstance(hetero_channel[0], str):
                raise ValueError(
                    "hetero_channel contains strings instead of embeddings. "
                    "Set 'timemmd_text_output: embedding' in data_config override. "
                    "Example in experiment suite:\n"
                    "  data_config:\n"
                    "    timemmd_text_output: embedding"
                )
            elif isinstance(hetero_channel[0], (np.ndarray, np.generic)):
                return torch.from_numpy(np.stack(hetero_channel)).float().to(device)
            elif isinstance(hetero_channel[0], torch.Tensor):
                return torch.stack(hetero_channel).float().to(device)
            else:
                raise ValueError(
                    f"Unexpected element type in hetero_channel list: {type(hetero_channel[0])}. "
                    f"Expected numpy array or tensor."
                )
        
        # Handle numpy array
        elif isinstance(hetero_channel, (np.ndarray, np.generic)):
            return torch.from_numpy(np.asarray(hetero_channel)).float().to(device)
        
        # Handle string format - clear error
        elif isinstance(hetero_channel, str):
            raise ValueError(
                "hetero_channel is a string instead of embeddings. "
                "Set 'timemmd_text_output: embedding' in data_config override. "
                "Example in experiment suite:\n"
                "  data_config:\n"
                "    timemmd_text_output: embedding"
            )
        
        # Handle tensor
        elif isinstance(hetero_channel, torch.Tensor):
            return hetero_channel.float().to(device)
        
        # Fallback - clear error
        else:
            raise ValueError(
                f"Unexpected type for hetero_channel: {type(hetero_channel)}. "
                f"Expected tensor, numpy array, or list of arrays/tensors."
            )
    
    def _convert_text_embedding_to_tensor(self, text_embedding, device, name="text_embedding"):
        """
        Convert text embedding (x_hetero or y_hetero) to a tensor and move it to the specified device.
        
        Args:
            text_embedding: Text embeddings as tensor, numpy array, or list of arrays/tensors
            device: Target device for the tensor
            name: Name of the embedding (for error messages)
            
        Returns:
            torch.Tensor: text_embedding as a float tensor on the specified device
            
        Raises:
            ValueError: If text_embedding is in an invalid format (e.g., strings)
        """
        # Handle list types (from DataLoader collation)
        if isinstance(text_embedding, list):
            if len(text_embedding) == 0:
                raise ValueError(
                    f"Empty list received for {name}. This indicates a data loading issue."
                )
            elif isinstance(text_embedding[0], str):
                raise ValueError(
                    f"{name} contains strings instead of embeddings. "
                    f"Set 'timemmd_text_output: embedding' in data_config override. "
                    f"Example in experiment suite:\n"
                    f"  data_config:\n"
                    f"    timemmd_text_output: embedding"
                )
            elif isinstance(text_embedding[0], (np.ndarray, np.generic)):
                # List of numpy arrays - stack them (adds batch dimension)
                return torch.from_numpy(np.stack(text_embedding)).float().to(device)
            elif isinstance(text_embedding[0], torch.Tensor):
                # List of tensors - stack them
                return torch.stack(text_embedding).float().to(device)
            else:
                raise ValueError(
                    f"Unexpected element type in {name} list: {type(text_embedding[0])}. "
                    f"Expected numpy array or tensor. "
                    f"If you're seeing strings, set 'timemmd_text_output: embedding' in data_config."
                )
        
        # Handle numpy array
        elif isinstance(text_embedding, (np.ndarray, np.generic)):
            return torch.from_numpy(np.asarray(text_embedding)).float().to(device)
        
        # Handle tensor
        elif isinstance(text_embedding, torch.Tensor):
            return text_embedding.float().to(device)
        
        # Fallback - clear error
        else:
            raise ValueError(
                f"Unexpected type for {name}: {type(text_embedding)}. "
                f"Expected tensor, numpy array, or list of arrays/tensors."
            )
    
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
        
        # Convert hetero_channel to tensor and move to device
        hetero_channel = self._convert_hetero_channel_to_tensor(hetero_channel, device)
        
        # Convert text embeddings to tensors and move to device
        # Convert both x_hetero and y_hetero, but only the one specified by timestamp_semantics
        # will be used in the forward pass
        # - t_about: We use y_hetero (news) - forecasts ABOUT prediction window
        # - t_known: We use x_hetero (historical_events) - avoids lookahead bias
        if x_hetero is not None:
            x_hetero = self._convert_text_embedding_to_tensor(x_hetero, device, name="x_hetero")
        if y_hetero is not None:
            y_hetero = self._convert_text_embedding_to_tensor(y_hetero, device, name="y_hetero")
        
        # Log shape of the relevant text embedding once (on first batch)
        if not self._logged_text_shape and os.environ.get('FIDEL_DEBUG', '0') == '1':
            if self.timestamp_semantics == 't_about' and y_hetero is not None:
                print(f'[ info ] LYNX-FiLM-raw: y_hetero (news) shape: {y_hetero.shape}')
                self._logged_text_shape = True
            elif self.timestamp_semantics == 't_known' and x_hetero is not None:
                print(f'[ info ] LYNX-FiLM-raw: x_hetero (historical_events) shape: {x_hetero.shape}')
                self._logged_text_shape = True
        
        return seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel

