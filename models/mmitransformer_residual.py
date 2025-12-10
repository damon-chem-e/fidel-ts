import torch
import torch.nn as nn
import copy
from models.unimodal_wrapper import UnimodalModelWrapper
from models.mmitransformer import Model as MMiTransformer

class Model(nn.Module):
    """
    mmitransformer_residual: Residual Multimodal Learning
    
    This model uses a frozen unimodal iTransformer as a baseline and learns
    a residual correction using the mmitransformer architecture (which integrates
    text embeddings as additional variates).
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # 1. Unimodal Wrapper (Frozen Baseline)
        self.unimodal_wrapper = UnimodalModelWrapper.from_config(configs)
        
        # 2. Residual Model (mmitransformer)
        # We disable internal normalization because we handle it via the wrapper
        residual_configs = copy.deepcopy(configs)
        residual_configs.use_norm = False 
        self.residual_model = MMiTransformer(residual_configs)
        
    def forward(self, x, news, channel_description, **kwargs):
        """
        Forward pass combining frozen baseline and residual correction.
        
        Args:
            x: Input time series [B, seq_len, N]
            news: News embeddings
            channel_description: Channel descriptions
            
        Returns:
            Prediction [B, pred_len, N]
        """
        
        # 1. Normalize Input (shared by both models)
        x_norm, norm_params = self.unimodal_wrapper.normalize_input(x)
        
        # 2. Base Prediction (Frozen Model)
        # unimodal_wrapper handles prediction in normalized space
        pred_base = self.unimodal_wrapper.predict(x_norm, norm_params)
        
        # 3. Residual Prediction (mmitransformer)
        # mmitransformer expects unnormalized input logic if use_norm=True, 
        # but here we set use_norm=False and feed it already normalized input.
        pred_residual = self.residual_model(x_norm, news, channel_description)
        
        # 4. Combine Predictions
        pred_final_norm = pred_base + pred_residual
        
        # 5. Denormalize Final Prediction
        pred_final = self.unimodal_wrapper.denormalize_output(pred_final_norm, norm_params)
        
        return pred_final


