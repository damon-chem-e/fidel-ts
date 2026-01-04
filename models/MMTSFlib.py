"""
MM-TSFlib Model: Prediction-Level Late Fusion for Fidel-TS.

Implements the late fusion approach from MM-TSFlib where time series
predictions are ensembled with text-derived predictions at the output level.

Architecture:
    1. Time series encoder (any unimodal model, returns predictions)
    2. Text embeddings (pre-computed via LLMEmbeddingProvider)
    3. MLP projection (text_dim -> pred_len)
    4. Pooling (avg/max/min/attention)
    5. Normalization
    6. Prediction-level ensemble: (1-w)*ts_pred + w*(text_pred + prior)

Key Features:
    - Works with ANY unimodal model (PatchTST, DLinear, iTransformer, etc.)
    - Uses pre-computed embeddings (no runtime LLM)
    - Supports multiple pooling strategies
    - Optional prior history integration
    - Configurable mixing weight (prompt_weight)

Reference:
    MM-TSFlib: https://github.com/AdityaLab/MM-TSFlib
    Time-MMD: https://github.com/AdityaLab/Time-MMD
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from layers.MMTSFlib_layers import (
    TextToPredsProjection,
    TextPooling,
    normalize_embeddings
)


class Model(nn.Module):
    """
    MM-TSFlib: Prediction-Level Late Fusion Model.
    
    Ensembles time series predictions with text-derived predictions using
    a fixed mixing weight. This implements the original MM-TSFlib approach
    where both branches produce predictions that are combined at output level.
    
    Args:
        configs: Configuration object with:
            Required:
                - seq_len: Input sequence length
                - pred_len: Prediction horizon
                - enc_in: Number of input channels/variates
            
            Unimodal model:
                - unimodal_model_type: Base model type (default: 'iTransformer')
                - (other model-specific configs passed through)
            
            Text processing:
                - input_text_dim: Dimension of input text embeddings (default: 768)
                - text_mlp_reduction: Hidden dim = d_llm / reduction (default: 8)
                - text_mlp_dropout: Dropout in text MLP (default: 0.3)
                - pool_type: Pooling strategy (default: 'avg')
            
            Fusion:
                - prompt_weight: Mixing weight w (default: 0.01)
                - use_prior: Whether to add prior history (default: False)
    
    Input:
        - x: Time series [B, seq_len, enc_in]
        - **kwargs: Must include text embeddings via:
            - 'dataset_description' or 'hetero_general': Pre-computed embeddings
            - 'prior_y' (optional): Historical prior if use_prior=True
    
    Output:
        - predictions: [B, pred_len, enc_in]
    """
    
    # Supported unimodal models that can be wrapped
    SUPPORTED_MODELS = {
        'PatchTST', 'DLinear', 'iTransformer', 'FEDformer', 
        'Informer', 'FITS', 'Autoformer'
    }
    
    def __init__(self, configs):
        """
        Initialize MM-TSFlib model.
        
        Args:
            configs: Configuration object (dotdict) with model parameters
        """
        super().__init__()
        
        # =====================================================================
        # Configuration Extraction
        # =====================================================================
        
        # Required parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # Unimodal model configuration
        self.unimodal_model_type = getattr(configs, 'unimodal_model_type', 'iTransformer')
        
        # Text processing configuration
        raw_text_dim = getattr(configs, 'input_text_dim', None)
        self.text_dim = 768 if raw_text_dim is None else raw_text_dim
        self.text_mlp_reduction = getattr(configs, 'text_mlp_reduction', 8)
        self.text_mlp_dropout = getattr(configs, 'text_mlp_dropout', 0.3)
        self.pool_type = getattr(configs, 'pool_type', 'avg')
        
        # Fusion configuration
        self.use_prior = getattr(configs, 'use_prior', False)
        
        # Prompt weight: can be fixed (hyperparameter) or learned (trainable)
        self.prompt_weight_mode = getattr(configs, 'prompt_weight_mode', 'fixed')  # 'fixed' or 'learned'
        
        # Store full configs for unimodal model creation
        self.configs = configs
        
        # =====================================================================
        # Build Model Components
        # =====================================================================
        
        # 1. Unimodal time series model (returns predictions directly)
        self.ts_model = self._create_unimodal_model(configs)
        
        # 2. Text-to-predictions projection MLP
        # Maps text embeddings to prediction dimension: d_llm -> hidden -> pred_len
        self.text_projection = TextToPredsProjection(
            d_llm=self.text_dim,
            pred_len=self.pred_len,
            reduction_factor=self.text_mlp_reduction,
            dropout=self.text_mlp_dropout
        )
        
        # 3. Pooling layer to aggregate token embeddings
        self.pooling = TextPooling(pool_type=self.pool_type)
        
        # 4. Prompt weight (fixed or learned)
        if self.prompt_weight_mode == 'learned':
            # Learned parameter - sigmoid applied in forward to constrain to [0, 1]
            initial_value = getattr(configs, 'prompt_weight_initial', 0.01)
            # Use inverse sigmoid to initialize so sigmoid(raw) = initial_value
            initial_raw = torch.log(torch.tensor(initial_value / (1 - initial_value + 1e-8)))
            self.prompt_weight_raw = nn.Parameter(initial_raw)
        else:
            # Fixed hyperparameter - register as buffer (not trainable)
            prompt_weight_value = getattr(configs, 'prompt_weight', 0.01)
            self.register_buffer('prompt_weight_raw', torch.tensor(prompt_weight_value))
        
        print(f"[ info ] MMTSFlib initialized with:")
        print(f"         - Unimodal model: {self.unimodal_model_type}")
        print(f"         - Text dim: {self.text_dim}")
        print(f"         - Pool type: {self.pool_type}")
        print(f"         - Prompt weight mode: {self.prompt_weight_mode}")
        if self.prompt_weight_mode == 'learned':
            print(f"         - Prompt weight initial: {getattr(configs, 'prompt_weight_initial', 0.01)}")
        else:
            print(f"         - Prompt weight: {getattr(configs, 'prompt_weight', 0.01)}")
    
    def _create_unimodal_model(self, configs):
        """
        Create and configure unimodal time series model.
        
        The wrapped model should return predictions [B, pred_len, C] in forward().
        
        Args:
            configs: Configuration object passed to unimodal model
            
        Returns:
            Instantiated unimodal model
        """
        model_type = self.unimodal_model_type
        
        # Validate model type
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported unimodal_model_type: {model_type}. "
                f"Supported models: {self.SUPPORTED_MODELS}"
            )
        
        # Dynamic import and instantiation
        if model_type == 'PatchTST':
            from models.PatchTST import Model as UnimodalModel
        elif model_type == 'DLinear':
            from models.DLinear import Model as UnimodalModel
        elif model_type == 'iTransformer':
            from models.iTransformer import Model as UnimodalModel
        elif model_type == 'FEDformer':
            from models.FEDformer import Model as UnimodalModel
        elif model_type == 'Informer':
            from models.Informer import Model as UnimodalModel
        elif model_type == 'FITS':
            from models.FITS import Model as UnimodalModel
        elif model_type == 'Autoformer':
            from models.Autoformer import Model as UnimodalModel
        else:
            raise ValueError(f"Model {model_type} import not implemented")
        
        return UnimodalModel(configs)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass with prediction-level late fusion.
        
        Pipeline:
            1. TS model produces predictions
            2. Text embeddings projected via MLP to pred_len
            3. Text pooled to single vector
            4. Text normalized
            5. Optional: add prior history
            6. Ensemble: (1-w)*ts_pred + w*text_pred
        
        Args:
            x: Time series input [B, seq_len, enc_in]
            **kwargs: Must include text embeddings via:
                - 'dataset_description' or 'hetero_general': [B, L, text_dim]
                - 'prior_y' (optional): [B, pred_len, 1] historical prior
                - 'x_mark_enc', 'x_mark_dec' (optional): for encoder-decoder models
        
        Returns:
            predictions: [B, pred_len, enc_in]
        """
        # =====================================================================
        # 1. Time Series Predictions from Unimodal Model
        # =====================================================================
        
        ts_preds = self._forward_ts_model(x, **kwargs)  # [B, pred_len, C]
        
        # =====================================================================
        # 2. Text Processing Pipeline
        # =====================================================================
        
        # Get text embeddings from kwargs
        text_emb = self._get_text_embeddings(kwargs)  # [B, L, text_dim] or [B, text_dim]
        
        # Ensure 3D for token-level processing
        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(1)  # [B, 1, text_dim]
        
        # DEBUG: Print shape before projection
        print(f"[DEBUG] Before text_projection: shape={text_emb.shape}")
        
        # Project text to prediction dimension via MLP
        text_proj = self.text_projection(text_emb)  # [B, L, pred_len]
        
        # Pool to single vector
        if self.pool_type == 'attention':
            # Attention pooling needs TS predictions as query
            text_pooled = self.pooling(text_proj, ts_preds)  # [B, pred_len, 1]
        else:
            text_pooled = self.pooling(text_proj)  # [B, pred_len, 1]
        
        # Normalize embeddings (instance normalization)
        text_pred = normalize_embeddings(text_pooled)  # [B, pred_len, 1]
        
        # =====================================================================
        # 3. Prior Integration (Optional)
        # =====================================================================
        
        if self.use_prior and 'prior_y' in kwargs and kwargs['prior_y'] is not None:
            prior_y = kwargs['prior_y']
            
            # Convert numpy to tensor if needed
            if isinstance(prior_y, np.ndarray):
                prior_y = torch.from_numpy(prior_y).float()
            
            # Move to correct device
            prior_y = prior_y.to(text_pred.device)
            
            # Add prior to text prediction
            text_pred = text_pred + prior_y
        
        # =====================================================================
        # 4. Prediction-Level Ensemble
        # =====================================================================
        
        # Get prompt weight (fixed or learned)
        if self.prompt_weight_mode == 'learned':
            # Apply sigmoid to constrain to [0, 1]
            prompt_weight = torch.sigmoid(self.prompt_weight_raw)
        else:
            # Use fixed value from buffer
            prompt_weight = self.prompt_weight_raw
        
        # Ensemble formula: (1 - w) * ts_pred + w * text_pred
        # text_pred broadcasts from [B, pred_len, 1] to [B, pred_len, C]
        predictions = (1 - prompt_weight) * ts_preds + prompt_weight * text_pred
        
        return predictions
    
    def _forward_ts_model(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through unimodal time series model.
        
        Handles different model interfaces (simple vs encoder-decoder).
        
        Args:
            x: Time series input [B, seq_len, enc_in]
            **kwargs: Additional arguments for specific models
            
        Returns:
            ts_preds: [B, pred_len, enc_in] predictions
        """
        # Check if model needs encoder-decoder interface
        if self.unimodal_model_type in ['Informer', 'FEDformer', 'Autoformer']:
            # Encoder-decoder models need decoder input and time marks
            x_mark_enc = kwargs.get('x_mark_enc', None)
            x_mark_dec = kwargs.get('x_mark_dec', None)
            
            # Create decoder input: zeros for prediction portion
            B, T, C = x.shape
            dec_inp = torch.zeros(B, self.pred_len, C, device=x.device, dtype=x.dtype)
            
            # Forward with full interface
            try:
                preds = self.ts_model(x, x_mark_enc, dec_inp, x_mark_dec)
            except TypeError:
                # Fallback: model may not need all arguments
                preds = self.ts_model(x)
        else:
            # Simple models just need x
            preds = self.ts_model(x)
        
        # Ensure output shape is [B, pred_len, C]
        if preds.dim() == 2:
            preds = preds.unsqueeze(-1)
        
        # Take last pred_len timesteps if model returns more
        preds = preds[:, -self.pred_len:, :]
        
        return preds
    
    def _get_text_embeddings(self, kwargs: dict) -> torch.Tensor:
        """
        Extract pre-computed text embeddings from kwargs.
        
        Follows same pattern as ZhangHanBest for consistency.
        
        Priority order: dataset_description > hetero_general
        
        Args:
            kwargs: Forward pass keyword arguments
            
        Returns:
            text_emb: [B, L, text_dim] or [B, text_dim] tensor
        """
        text_emb = None
        
        # Try standard names (same priority as ZhangHanBest)
        if 'dataset_description' in kwargs and kwargs['dataset_description'] is not None:
            text_emb = kwargs['dataset_description']
        elif 'hetero_general' in kwargs and kwargs['hetero_general'] is not None:
            text_emb = kwargs['hetero_general']
        else:
            raise ValueError(
                "Text embeddings not found in kwargs. "
                "Expected 'dataset_description' or 'hetero_general' with pre-computed embeddings. "
                "Ensure your data config includes embedding settings."
            )
        
        # Convert numpy array to tensor
        if isinstance(text_emb, np.ndarray):
            text_emb = torch.from_numpy(text_emb).float()
        
        # Move to correct device
        device = next(self.parameters()).device
        text_emb = text_emb.to(device)
        
        # Handle various input shapes
        # DEBUG: Print initial shape
        print(f"[DEBUG] _get_text_embeddings: initial shape={text_emb.shape}, self.text_dim={self.text_dim}")
        
        if text_emb.dim() == 4:
            # [B, seq_len, num_items, text_dim] -> aggregate to [B, text_dim]
            text_emb = text_emb.mean(dim=(1, 2))
            print(f"[DEBUG] After 4D mean: shape={text_emb.shape}")
        elif text_emb.dim() == 3:
            # Could be [B, L, text_dim] or [B, text_dim, L] (transposed)
            # Check if last dim matches expected text_dim; if not, transpose
            print(f"[DEBUG] 3D input: shape[-1]={text_emb.shape[-1]}, shape[1]={text_emb.shape[1]}")
            if text_emb.shape[-1] != self.text_dim and text_emb.shape[1] == self.text_dim:
                # Input is [B, text_dim, L] - transpose to [B, L, text_dim]
                text_emb = text_emb.transpose(1, 2)
                print(f"[DEBUG] After transpose: shape={text_emb.shape}")
            # Now it's [B, L, text_dim] - keep for token-level processing
        elif text_emb.dim() == 2:
            # [B, text_dim] - already aggregated, fine as-is
            print(f"[DEBUG] 2D input, no change: shape={text_emb.shape}")
            pass
        else:
            raise ValueError(
                f"Unexpected text embedding shape: {text_emb.shape}. "
                f"Expected [B, text_dim], [B, L, text_dim], or [B, seq, items, text_dim]"
            )
        
        print(f"[DEBUG] _get_text_embeddings: final shape={text_emb.shape}")
        
        return text_emb
    
    def count_trainable_params(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_frozen_params(self) -> int:
        """Count number of frozen parameters."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
