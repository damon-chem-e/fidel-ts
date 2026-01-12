"""
LeRet: Language-Enhanced Retention Network for Time Series Forecasting.

A time series forecasting model that combines:
- RetNet backbone for O(n) complexity sequence modeling
- Patch-based input processing (like PatchTST)
- Optional language knowledge integration via fidel-ts embedder
- Two-stage training (auto-regressive pretrain + forecasting finetune)

Reference: Based on RetNet architecture from Sun et al., 
"Retentive Network: A Successor to Transformer for Large Language Models" (2023)

Usage in fidel-ts:
    configs = dotdict({
        'seq_len': 96,
        'pred_len': 24,
        'enc_in': 7,
        'd_model': 128,
        'n_heads': 8,
        'e_layers': 3,
        # ... other config parameters
    })
    model = Model(configs)
    # Returns (forecast_output, autoregressive_output)
    forecast, auto = model(x=batch_x)
"""

from typing import Callable, Optional, Tuple
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.LeRet_backbone import LeRet_backbone
from layers.LeRet_layers import series_decomp
from layers.TS_Language_Integrator import LanguageIntegrator


class Model(nn.Module):
    """
    LeRet: Language-Enhanced Retention Network for Time Series.
    
    This model implements a RetNet-based architecture adapted for time series
    forecasting, with optional language knowledge integration and support
    for two-stage training (pretraining + finetuning).
    
    Architecture:
    1. Optional series decomposition (trend + residual)
    2. RevIN normalization
    3. Patching
    4. RetNet encoder (Multi-Scale Retention)
    5. Optional language integration (cross-attention with text embeddings)
    6. Dual prediction heads:
       - Patch head: for auto-regressive pretraining (Stage 1)
       - Sequence head: for forecasting finetuning (Stage 2)
    
    Args (via configs):
        seq_len: Input sequence length
        pred_len: Prediction horizon length
        enc_in: Number of input channels
        d_model: Model dimension (default: 128)
        n_heads: Number of retention heads (default: 8)
        e_layers: Number of retention layers (default: 3)
        d_ff: FFN dimension (default: 256)
        dropout: Dropout rate (default: 0.05)
        fc_dropout: FC layer dropout (default: 0.05)
        head_dropout: Head dropout (default: 0.0)
        patch_len: Patch length (default: 16)
        stride: Patching stride (default: 8)
        padding_patch: Padding mode for patching (default: 'end')
        revin: Use RevIN normalization (default: True)
        affine: RevIN affine transform (default: False)
        subtract_last: RevIN subtract last mode (default: False)
        decomposition: Use series decomposition (default: False)
        kernel_size: Decomposition kernel size (default: 25)
        individual: Per-channel prediction heads (default: False)
        use_language: Use language integration (default: False)
        language_dim: Language embedding dimension (default: 768)
    """
    
    def __init__(
        self,
        configs,
        max_seq_len: Optional[int] = 1024,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        norm: str = 'BatchNorm',
        attn_dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: str = 'auto',
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = 'zeros',
        learn_pe: bool = True,
        pretrain_head: bool = False,
        head_type: str = 'flatten',
        verbose: bool = False,
        **kwargs
    ):
        super().__init__()
        
        # =====================================================================
        # Load core parameters from config
        # =====================================================================
        # Get enc_in from configs, with fallback to input_channel
        c_in = getattr(configs, 'enc_in', None)
        if c_in is None:
            c_in = getattr(configs, 'input_channel', None)
            if c_in is None:
                raise ValueError(
                    "enc_in must be provided in model_config_overrides. "
                    "Example: model_config_overrides: {enc_in: 7}"
                )
        
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        # Architecture parameters with defaults
        n_layers = getattr(configs, 'e_layers', 3)
        n_heads = getattr(configs, 'n_heads', 8)
        d_model = getattr(configs, 'd_model', 128)
        d_ff = getattr(configs, 'd_ff', 256)
        dropout = getattr(configs, 'dropout', 0.05)
        fc_dropout = getattr(configs, 'fc_dropout', 0.05)
        head_dropout = getattr(configs, 'head_dropout', 0.0)
        
        individual = getattr(configs, 'individual', False)
        
        # Patching parameters
        patch_len = getattr(configs, 'patch_len', 16)
        stride = getattr(configs, 'stride', 8)
        padding_patch = getattr(configs, 'padding_patch', 'end')
        
        # RevIN parameters
        revin = getattr(configs, 'revin', True)
        affine = getattr(configs, 'affine', False)
        subtract_last = getattr(configs, 'subtract_last', False)
        
        # Decomposition parameters
        decomposition = getattr(configs, 'decomposition', False)
        kernel_size = getattr(configs, 'kernel_size', 25)
        
        # Language integration parameters
        use_language = getattr(configs, 'use_language', False)
        language_dim = getattr(configs, 'language_dim', 768)
        
        # =====================================================================
        # Store config for two-stage training
        # =====================================================================
        self.use_language = use_language
        self.c_in = c_in
        self.context_window = context_window
        self.target_window = target_window
        self.d_model = d_model
        
        # Compute patch_num for language integrator
        patch_num = int((context_window - patch_len) / stride + 1)
        
        # =====================================================================
        # Optional series decomposition
        # =====================================================================
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            
            # Separate backbones for trend and residual
            self.model_trend = LeRet_backbone(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
                **kwargs
            )
            self.model_res = LeRet_backbone(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
                **kwargs
            )
        else:
            # Single backbone (no decomposition)
            self.model = LeRet_backbone(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
                **kwargs
            )
        
        # =====================================================================
        # Language Integration (optional)
        # =====================================================================
        if self.use_language:
            self.language_integrator = LanguageIntegrator(
                c_in=c_in,
                patch_num=patch_num,
                patch_len=patch_len,
                d_model=d_model,
                n_heads=n_heads,
                language_dim=language_dim,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                pre_norm=pre_norm,
                activation=act,
                res_attention=res_attention,
                n_layers=1,  # Single layer for language integration
                store_attn=store_attn,
                max_seq_len=max_seq_len,
            )
        else:
            self.language_integrator = None
    
    def forward(
        self,
        x: Optional[Tensor] = None,
        x_enc: Optional[Tensor] = None,
        language_embeddings: Optional[Tensor] = None,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through LeRet model.
        
        Args:
            x: Input time series [Batch, Seq_Len, Channels] (framework interface)
            x_enc: Alternative name for x (original interface)
            language_embeddings: Optional language embeddings [text_num, language_dim]
                                from fidel-ts embedder for language integration
            **kwargs: Additional arguments (ignored for compatibility)
        
        Returns:
            Tuple of:
                - forecast: [Batch, Pred_Len, Channels] - main forecasting output (Stage 2)
                - auto_y: [Batch, Seq_Len, Channels] - auto-regressive output (Stage 1)
        
        Note:
            Both outputs are always returned. During training:
            - Stage 1 (pretrain): Use auto_y for loss computation
            - Stage 2 (finetune): Use forecast for loss computation
        """
        # Handle both interface styles
        if x is None and x_enc is not None:
            x = x_enc
        if x is None:
            raise ValueError("Either 'x' or 'x_enc' must be provided")
        
        # x: [Batch, Input length, Channel]
        
        if self.decomposition:
            # =================================================================
            # Decomposition mode: separate trend and residual
            # =================================================================
            res_init, trend_init = self.decomp_module(x)
            
            # Transpose for backbone: [B, L, C] -> [B, C, L]
            res_init = res_init.permute(0, 2, 1)
            trend_init = trend_init.permute(0, 2, 1)
            
            # Process through separate backbones with optional language integration
            language_integrator = self.language_integrator if (self.use_language and language_embeddings is not None) else None
            res, auto_res = self.model_res(res_init, language_integrator=language_integrator, language_embeddings=language_embeddings)
            trend, auto_trend = self.model_trend(trend_init, language_integrator=language_integrator, language_embeddings=language_embeddings)
            
            # Combine outputs
            forecast = res + trend
            auto_y = auto_res + auto_trend
            
            # Transpose back: [B, C, L] -> [B, L, C]
            forecast = forecast.permute(0, 2, 1)
            auto_y = auto_y.permute(0, 2, 1)
        else:
            # =================================================================
            # Standard mode (no decomposition)
            # =================================================================
            # Transpose for backbone: [B, L, C] -> [B, C, L]
            x = x.permute(0, 2, 1)
            
            # Process through backbone with optional language integration
            language_integrator = self.language_integrator if (self.use_language and language_embeddings is not None) else None
            forecast, auto_y = self.model(x, language_integrator=language_integrator, language_embeddings=language_embeddings)
            
            # Transpose back: [B, C, L] -> [B, L, C]
            forecast = forecast.permute(0, 2, 1)
            auto_y = auto_y.permute(0, 2, 1)
        
        return forecast, auto_y


# =============================================================================
# Quick test when run directly
# =============================================================================
if __name__ == '__main__':
    """Quick test of the LeRet model."""
    
    class Configs:
        """Test configuration."""
        seq_len = 96
        pred_len = 24
        enc_in = 7
        d_model = 128
        n_heads = 8
        e_layers = 3
        d_ff = 256
        dropout = 0.05
        fc_dropout = 0.05
        head_dropout = 0.0
        patch_len = 16
        stride = 8
        padding_patch = 'end'
        revin = True
        affine = False
        subtract_last = False
        decomposition = False
        kernel_size = 25
        individual = False
        use_language = False
        language_dim = 768
    
    configs = Configs()
    model = Model(configs)
    
    print(f'LeRet parameter count: {sum(p.numel() for p in model.parameters()):,}')
    
    # Create test input
    x = torch.randn(2, 96, 7)  # [B, seq_len, enc_in]
    
    # Test forward pass
    forecast, auto_y = model(x=x)
    
    print(f'Forecast output shape: {forecast.shape}')
    print(f'Auto-reg output shape: {auto_y.shape}')
    
    assert forecast.shape == (2, 24, 7), f"Expected (2, 24, 7), got {forecast.shape}"
    
    # Expected auto_y length: patch_num * patch_len where patch_num = (96 - 16) / 8 + 1 = 11
    # So auto_y should be [2, 11 * 16, 7] = [2, 176, 7]
    # But we transpose so it's [2, 176, 7] in [B, L, C] format
    expected_auto_len = ((96 - 16) // 8 + 1) * 16  # 176
    assert auto_y.shape == (2, expected_auto_len, 7), \
        f"Expected (2, {expected_auto_len}, 7), got {auto_y.shape}"
    
    # Test with decomposition
    configs.decomposition = True
    model_decomp = Model(configs)
    forecast_d, auto_y_d = model_decomp(x=x)
    print(f'Decomposition mode - Forecast shape: {forecast_d.shape}')
    assert forecast_d.shape == (2, 24, 7)
    
    # Test with language integration
    configs.decomposition = False
    configs.use_language = True
    model_lang = Model(configs)
    
    # Language embeddings: [text_num, language_dim]
    lang_emb = torch.randn(10, 768)  # 10 text embeddings
    
    forecast_l, auto_y_l = model_lang(x=x, language_embeddings=lang_emb)
    print(f'Language mode - Forecast shape: {forecast_l.shape}')
    assert forecast_l.shape == (2, 24, 7)
    
    print('All LeRet tests passed!')

