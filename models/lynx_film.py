"""
lynx-film Model: FiLM-Modulated Residual Learning

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
"""

from torch import nn
import copy
from layers.TGTSF_torch import text_encoder
from models.unimodal_wrapper import UnimodalModelWrapper
from layers.lynx_film_layers import iTransformerFilm

class Model(nn.Module):
    """
    lynx-film Model:
    - Base: Frozen unimodal model (default: iTransformer)
    - Residual: iTransformerFilm (modulated by text via FiLM)
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # 1. Unimodal Wrapper (Frozen Baseline)
        self.unimodal_wrapper = UnimodalModelWrapper.from_config(configs)
        
        # Timestamp semantics determines which text input to use for TGTSF task
        # - t_about: Use news (y_hetero) - text describes prediction window, assumed known beforehand
        # - t_known: Use historical_events (x_hetero) - avoid lookahead bias
        self.timestamp_semantics = getattr(configs, 'timestamp_semantics', None)
        if self.timestamp_semantics is None:
            raise ValueError(
                "lynx_film requires 'timestamp_semantics' in configs. "
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
        print(f'[ info ] LYNX-FiLM: timestamp_semantics = {self.timestamp_semantics}')
        if self.timestamp_semantics == 't_about':
            print(f'         -> Using news (y_hetero) - forecasts ABOUT prediction window')
        else:
            print(f'         -> Using historical_events (x_hetero) - avoiding lookahead bias')
        
        # Text dimension handling:
        # - input_text_dim: Dimension of input text embeddings (e.g., 768 for BERT)
        # - text_dim: Operational dimension used internally by the model (e.g., 256)
        # If input_text_dim != text_dim, a learned projection layer is added
        # NOTE: `dotdict.__getattr__` returns None for missing keys, so treat None as unset.
        self.text_dim = configs.text_dim
        raw_input_text_dim = getattr(configs, 'input_text_dim', None)
        self.input_text_dim = self.text_dim if raw_input_text_dim is None else raw_input_text_dim

        if not isinstance(self.input_text_dim, int) or self.input_text_dim <= 0:
            raise ValueError(
                f"LYNX-FiLM requires a positive integer input_text_dim; got {self.input_text_dim!r}. "
                f"Set it via model_config_overrides.input_text_dim (e.g., 256 for old embeddings, 768 for BERT)."
            )
        
        # Learned projection layer if input dimension differs from operational dimension
        if self.input_text_dim != self.text_dim:
            self.text_projection = nn.Linear(self.input_text_dim, self.text_dim)
            print(f'[ info ] LYNX-FiLM: Added learned projection layer {self.input_text_dim} -> {self.text_dim}')
        else:
            self.text_projection = None
        
        # 2. Text Encoder (from TGTSF)
        # Used to get text embeddings for FiLM
        self.text_encoder = text_encoder(
            cross_layer=configs.cross_layers, 
            self_layer=configs.self_layers, 
            embedding_dim=configs.text_dim, 
            num_heads=configs.n_heads, 
            dropout=configs.dropout, 
            pred_len=configs.pred_len, 
            stride=configs.stride
        )
        
        # 3. Residual Model (iTransformerFilm)
        # Disable internal normalization because we feed it normalized input
        residual_configs = copy.deepcopy(configs)
        residual_configs.use_norm = False 
        self.residual_model = iTransformerFilm(residual_configs)
    
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
        
    def forward(self, x, news=None, channel_description=None, historical_events=None, **kwargs):
        """
        Args:
            x: Input time series [B, seq_len, C]
            news: Text embeddings aligned to PREDICTION window (y_hetero from dataloader)
                  - For t_about: Forecasts ABOUT prediction window (USE THIS)
                  - For t_known: Text PUBLISHED during prediction window (LOOKAHEAD - don't use!)
            channel_description: Channel descriptions [B, C, d_model] or [B, 1, C, d_model]
                Will be automatically expanded to [B, l, C, d_model] to match text time dimension
            historical_events: Text embeddings aligned to INPUT window (x_hetero from dataloader)
                              - Always safe to use (describes past, known at prediction time)
        """
        # Select text input based on timestamp_semantics
        if self.timestamp_semantics == 't_about':
            if news is None:
                raise ValueError(
                    "timestamp_semantics='t_about' requires 'news' (y_hetero) to be provided."
                )
            text_input = news
        elif self.timestamp_semantics == 't_known':
            if historical_events is None:
                raise ValueError(
                    "timestamp_semantics='t_known' requires 'historical_events' (x_hetero) to be provided."
                )
            text_input = historical_events
        else:
            raise ValueError(f"Invalid timestamp_semantics: {self.timestamp_semantics}")
        
        # Ensure input is on the same device as the unimodal model
        # This prevents device mismatch errors when model is on GPU but input is on CPU
        unimodal_device = next(self.unimodal_wrapper.model.parameters()).device
        x = x.to(unimodal_device)
        
        # Step 1: Normalize input using wrapper's normalization scheme
        x_norm, norm_params = self.unimodal_wrapper.normalize_input(x)
        
        # Step 2: Get unimodal prediction (wrapper handles all normalization logic)
        unimodal_pred_norm = self.unimodal_wrapper.predict(x, norm_params)
        
        # Step 3: Project text embeddings if input dimension differs from operational dimension
        text_input, channel_description = self._project_text_embeddings(text_input, channel_description)

        # ============================================================================
        # PERFORMANCE WARNING: Check for concatenated channel descriptions
        # ============================================================================
        # LYNX/FILM models can operate with C=1 (single description broadcast to all variables)
        # due to FiLM's inherent broadcasting mechanism. However, this is suboptimal when
        # per-variable descriptions are available but were concatenated during embedding generation.
        # See docs/planning/fidel_ts_embedder_channel_concat_issue.md for details.
        C_time_series = x.shape[2]  # Number of variables in time series

        if len(channel_description.shape) == 3:  # [B, C, D]
            C_desc = channel_description.shape[1]
            if C_desc == 1 and C_time_series > 1:
                print(f'[ INFO ] LYNX/FILM: Using single channel description for {C_time_series} variables. '
                      f'This works but is suboptimal if per-variable descriptions exist. '
                      f'Consider regenerating embeddings with fixed fidel_ts_embedder for improved '
                      f'semantic alignment between text and time series variables. '
                      f'See docs/planning/fidel_ts_embedder_channel_concat_issue.md')
                # Note: No broadcasting needed for FILM - it naturally handles C=1
        # ============================================================================

        # Step 4: Get Text Embeddings
        # text_encoder returns [B, L, C, text_dim] (L=time segments, C=channels)
        # Transform channel_description to match text_encoder expected input shape [B, L, C, D]
        # Handle both [B, C, D] and [B, 1, C, D] input shapes
        if len(channel_description.shape) == 3:
            # If [B, C, D], unsqueeze to [B, 1, C, D]
            channel_description = channel_description.unsqueeze(1)
        # Repeat along time dimension to match text_input shape: [B, L, C, D]
        description = channel_description.repeat(1, text_input.shape[1], 1, 1)
        text_emb = self.text_encoder(text_input, description)
        
        # Step 4: Prepare text embeddings for FiLM
        # iTransformerFilm expects [B, C, L, text_dim]
        # Text encoder output is [B, L, C, text_dim], so we permute.
        # We do not pool over L, preserving sequence info.
        # Ensure contiguity after permute for torch.compile compatibility
        # The compiled attention kernels require contiguous tensors
        text_emb = text_emb.permute(0, 2, 1, 3).contiguous()  # [B, C, L, D]
        
        # Step 5: Get Residual Prediction
        # Pass normalized input and unpooled text embeddings
        residual_pred_norm = self.residual_model(x_norm, text_emb)
        
        # Step 6: Combine in normalized space
        final_pred_norm = unimodal_pred_norm + residual_pred_norm
        
        # Step 7: Denormalize
        final_pred = self.unimodal_wrapper.denormalize_output(final_pred_norm, norm_params)
        
        return final_pred
    
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
            y_hetero: Prediction window text embeddings (maps to news in forward)
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
        hetero_channel = hetero_channel.float().to(device)
        
        # Move text based on timestamp_semantics
        # - t_about: We use y_hetero (news) - forecasts ABOUT prediction window
        # - t_known: We use x_hetero (historical_events) - avoids lookahead bias
        if self.timestamp_semantics == 't_about':
            y_hetero = y_hetero.float().to(device)
        elif self.timestamp_semantics == 't_known':
            x_hetero = x_hetero.float().to(device)
        
        # Move unimodal model to device (critical for device consistency)
        self.unimodal_wrapper.model = self.unimodal_wrapper.model.to(device)
        
        return seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel
