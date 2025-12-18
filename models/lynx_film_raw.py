"""
lynx-film-raw Model: FiLM-Modulated Raw Signal Learning

This model learns the entire prediction structure using iTransformerFilm with FiLM modulation,
without learning a residual to a frozen unimodal model. Similar to TGTSF but using iTransformer
architecture with FiLM instead of TGTSF's patch-based architecture.
"""

from torch import nn
import torch
import copy
from layers.TGTSF_torch import text_encoder
from layers.lynx_film_layers import iTransformerFilm

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
        
        # Text dimension handling:
        # - input_text_dim: Dimension of input text embeddings (e.g., 768 for BERT)
        # - text_dim: Operational dimension used internally by the model (e.g., 256)
        # If input_text_dim != text_dim, a learned projection layer is added
        self.input_text_dim = getattr(configs, 'input_text_dim', configs.text_dim)
        self.text_dim = configs.text_dim
        
        # Learned projection layer if input dimension differs from operational dimension
        if self.input_text_dim != self.text_dim:
            self.text_projection = nn.Linear(self.input_text_dim, self.text_dim)
            print(f'[ info ] LYNX-FiLM-raw: Added learned projection layer {self.input_text_dim} -> {self.text_dim}')
        else:
            self.text_projection = None
        
        # 1. Text Encoder (from TGTSF)
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
        
        # Text dimension handling:
        # - input_text_dim: Dimension of input text embeddings (e.g., 768 for BERT)
        # - text_dim: Operational dimension used internally by the model (e.g., 256)
        # If input_text_dim != text_dim, a learned projection layer is added
        self.input_text_dim = getattr(configs, 'input_text_dim', configs.text_dim)
        self.text_dim = configs.text_dim
        
        # Learned projection layer if input dimension differs from operational dimension
        if self.input_text_dim != self.text_dim:
            self.text_projection = nn.Linear(self.input_text_dim, self.text_dim)
            print(f'[ info ] LYNX-FiLM-raw: Added learned projection layer {self.input_text_dim} -> {self.text_dim}')
        else:
            self.text_projection = None
        
        # 2. Main Model (iTransformerFilm)
        # Disable internal normalization because we handle normalization externally
        # This allows us to use either RevIN or use_norm normalization scheme
        model_configs = copy.deepcopy(configs)
        model_configs.use_norm = False 
        self.model = iTransformerFilm(model_configs)
    
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
        
    def forward(self, x, news, channel_description, **kwargs):
        """
        Forward pass: Learn raw signal prediction using iTransformerFilm with FiLM modulation.
        
        Args:
            x: Input time series [B, seq_len, C]
            news: News embeddings [B, l, news_num, text_dim]
            channel_description: Channel descriptions [B, C, d_model] or [B, 1, C, d_model]
                Will be automatically expanded to [B, l, C, d_model] to match news time dimension
            **kwargs: Additional arguments (ignored)
            
        Returns:
            final_pred: Raw prediction [B, pred_len, C]
        """
        # Step 1: Normalize input
        x_norm, norm_params = self.normalize_input(x)
        
        # Step 2: Project text embeddings if input dimension differs from operational dimension
        news, channel_description = self._project_text_embeddings(news, channel_description)
        
        # Step 3: Get Text Embeddings
        # text_encoder returns [B, L, C, text_dim] (L=time segments, C=channels)
        # Transform channel_description to match text_encoder expected input shape [B, L, C, D]
        # Handle both [B, C, D] and [B, 1, C, D] input shapes
        if len(channel_description.shape) == 3:
            # If [B, C, D], unsqueeze to [B, 1, C, D]
            channel_description = channel_description.unsqueeze(1)
        # Repeat along time dimension to match news shape: [B, L, C, D]
        description = channel_description.repeat(1, news.shape[1], 1, 1)
        text_emb = self.text_encoder(news, description)
        
        # Step 3: Prepare text embeddings for FiLM
        # iTransformerFilm expects [B, C, L, text_dim]
        # Text encoder output is [B, L, C, text_dim], so we permute.
        # We do not pool over L, preserving sequence info.
        text_emb = text_emb.permute(0, 2, 1, 3) # [B, C, L, D]
        
        # Step 4: Get Raw Prediction from iTransformerFilm
        # Pass normalized input and text embeddings
        pred_norm = self.model(x_norm, text_emb)
        
        # Step 5: Denormalize prediction
        final_pred = self.denormalize_output(pred_norm, norm_params)
        
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
        
        # Model components are automatically moved when model.to(device) is called
        # No need to explicitly move unimodal model (doesn't exist in this architecture)
        
        return seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel

