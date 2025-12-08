"""
lynx-film Model: FiLM-Modulated Residual Learning
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
        
    def forward(self, x, news, channel_description, **kwargs):
        """
        Args:
            x: Input time series [B, seq_len, C]
            news: News embeddings [B, l, news_num, text_dim]
            channel_description: Channel descriptions [B, 1, C, d_model]
        """
        # Ensure input is on the same device as the unimodal model
        # This prevents device mismatch errors when model is on GPU but input is on CPU
        unimodal_device = next(self.unimodal_wrapper.model.parameters()).device
        x = x.to(unimodal_device)
        
        # Step 1: Normalize input using wrapper's normalization scheme
        x_norm, norm_params = self.unimodal_wrapper.normalize_input(x)
        
        # Step 2: Get unimodal prediction (wrapper handles all normalization logic)
        unimodal_pred_norm = self.unimodal_wrapper.predict(x, norm_params)
        
        # Step 3: Get Text Embeddings
        # text_encoder returns [B, L, C, text_dim] (L=time segments, C=channels)
        # Note: TGTSF text_encoder output shape logic:
        # It reshapes news and description, passes through transformer.
        # Output is [B, L, C, D].
        text_emb = self.text_encoder(news, channel_description)
        
        # Step 4: Prepare text embeddings for FiLM
        # iTransformerFilm expects [B, C, L, text_dim]
        # Text encoder output is [B, L, C, text_dim], so we permute.
        # We do not pool over L, preserving sequence info.
        text_emb = text_emb.permute(0, 2, 1, 3) # [B, C, L, D]
        
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
        Move data to device (same as TGTSF for compatibility).
        
        This method ensures all model components and input tensors are moved to
        the specified device. It's critical for proper GPU/CPU device placement.
        
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
        # Move data tensors to device
        seq_x = seq_x.float().to(device)
        seq_y = seq_y.float().to(device)
        hetero_channel = hetero_channel.float().to(device)
        y_hetero = y_hetero.float().to(device)
        
        # Move unimodal model to device (critical for device consistency)
        self.unimodal_wrapper.model = self.unimodal_wrapper.model.to(device)
        
        return seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel
