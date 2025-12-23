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

import torch
from torch import nn

from layers.TGTSF_torch import text_encoder, text_temp_cross_block, TS_encoder
from models.unimodal_wrapper import UnimodalModelWrapper


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
                - pretrained_model_path: Path to pretrained checkpoint (required, explicit path)
                - pretrained_model_type: Type of model (default: 'iTransformer')
                - pretrained_model_config_path: Optional path to pretrained model config
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
                f"LYNX requires a positive integer input_text_dim; got {self.input_text_dim!r}. "
                f"Set it via model_config_overrides.input_text_dim (e.g., 256 for old embeddings, 768 for BERT)."
            )
        
        # Learned projection layer if input dimension differs from operational dimension
        if self.input_text_dim != self.text_dim:
            self.text_projection = nn.Linear(self.input_text_dim, self.text_dim)
            print(f'[ info ] LYNX: Added learned projection layer {self.input_text_dim} -> {self.text_dim}')
        else:
            self.text_projection = None

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
        
        # Load and wrap pretrained unimodal model
        self.unimodal_wrapper = UnimodalModelWrapper.from_config(configs)
    
    def _project_text_embeddings(self, news, channel_description):
        """
        Project text embeddings from input_text_dim to text_dim if needed.
        
        Args:
            news: News embeddings [B, l, news_num, input_text_dim]
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
        
        # Configure RevIN based on wrapper's normalization scheme
        if self.unimodal_wrapper.norm_scheme == 'use_norm':
            # Disable RevIN in TGTSF when pretrained model uses different normalization
            self.revin = False
        else:
            # Use RevIN as normal if pretrained model also uses RevIN
            self.revin = self.original_revin
    
    
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
        # Project text embeddings if input dimension differs from operational dimension
        news, channel_description = self._project_text_embeddings(news, channel_description)
        
        # Convert description to [bs, l, nvars, d_model]
        channel_description = channel_description.unsqueeze(1)  # [bs, 1, nvars, text_dim]
        description = channel_description.repeat(1, news.shape[1], 1, 1)  # [bs, l, nvars, text_dim]
        
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
        # Ensure input is on the same device as the unimodal model
        # This prevents device mismatch errors when model is on GPU but input is on CPU
        unimodal_device = next(self.unimodal_wrapper.model.parameters()).device
        x = x.to(unimodal_device)
        
        # Step 1: Normalize input using wrapper's normalization scheme
        x_norm, norm_params = self.unimodal_wrapper.normalize_input(x)
        # norm_params contains the normalization data (mean, std, etc.)
        
        # Step 2: Get unimodal prediction (wrapper handles all normalization)
        unimodal_pred_norm = self.unimodal_wrapper.predict(x, norm_params)
        
        # Step 3: Get TGTSF residual prediction (operates on normalized input)
        # Disable RevIN in TGTSF if pretrained model uses different normalization
        # (RevIN check kept in lynx for TGTSF's internal use, wrapper handles unimodal normalization)
        disable_revin = (self.unimodal_wrapper.norm_scheme != 'revin')
        residual_pred_norm = self._tgtsf_forward(
            x_norm, news, channel_description, disable_revin=disable_revin
        )
        
        # Step 4: Combine predictions in normalized space
        final_pred_norm = unimodal_pred_norm + residual_pred_norm
        
        # Step 5: Denormalize final prediction using wrapper's denormalization
        final_pred = self.unimodal_wrapper.denormalize_output(final_pred_norm, norm_params)
        
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
        self.unimodal_wrapper.model = self.unimodal_wrapper.model.to(device)
        
        return seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel

