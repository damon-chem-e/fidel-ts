"""
FEDformer: Frequency Enhanced Decomposed Transformer for Time Series Forecasting

This is a unimodal (time series only) implementation for the fidel-ts framework.
FEDformer achieves O(N) complexity through frequency-domain attention and
incorporates seasonal-trend decomposition for improved long-horizon forecasting.

Reference: Zhou et al., "FEDformer: Frequency Enhanced Decomposed Transformer
for Long-term Series Forecasting" (ICML 2022)

Key features:
- Frequency Enhanced Attention (FEA) with O(N) complexity
- Seasonal-Trend Decomposition at each layer
- Two versions: Fourier and Wavelets (this impl. uses Fourier)
- Encoder-Decoder architecture with cross-attention

Usage in fidel-ts:
    configs = dotdict({
        'seq_len': 96,
        'pred_len': 24,
        'enc_in': 7,
        'd_model': 512,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        # ... other config parameters
    })
    model = Model(configs)
    output = model(x)  # x: [B, seq_len, channels]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.FEDformer_layers import (
    series_decomp, series_decomp_multi, my_Layernorm,
    FourierBlock, FourierCrossAttention, AutoCorrelationLayer,
    FEDformerEncoderLayer, FEDformerDecoderLayer,
    FEDformerEncoder, FEDformerDecoder
)
from layers.Embed import DataEmbedding, TokenEmbedding, PositionalEmbedding, TimeFeatureEmbedding


class DataEmbedding_wo_pos(nn.Module):
    """
    Data embedding without positional encoding.
    
    FEDformer omits positional encoding because frequency-domain operations
    inherently capture sequential information through the phase of Fourier coefficients.
    """
    
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()
        
        # Value embedding via 1D convolution (mixes channels)
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        
        # Temporal embedding (time features like hour, day, etc.)
        # Only used if temporal marks are provided
        if embed_type == 'timeF':
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        else:
            self.temporal_embedding = None
            
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        # x: [B, L, C] - time series values
        # x_mark: [B, L, time_features] - optional temporal features
        
        if x_mark is not None and self.temporal_embedding is not None:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        else:
            x = self.value_embedding(x)
            
        return self.dropout(x)


class Model(nn.Module):
    """
    FEDformer: Frequency Enhanced Decomposed Transformer
    
    A unimodal time series forecasting model that operates in the frequency domain
    with O(N) complexity. Uses seasonal-trend decomposition for improved predictions.
    
    Args (via configs):
        seq_len: Input sequence length
        pred_len: Prediction horizon length
        label_len: Start token length for decoder (default: seq_len // 2)
        enc_in: Number of input channels (encoder)
        dec_in: Number of input channels (decoder), defaults to enc_in
        c_out: Number of output channels, defaults to enc_in
        d_model: Model dimension (default: 512)
        n_heads: Number of attention heads (default: 8)
        e_layers: Number of encoder layers (default: 2)
        d_layers: Number of decoder layers (default: 1)
        d_ff: Feed-forward dimension (default: 2048)
        moving_avg: Kernel size for trend extraction (default: 25)
        modes: Number of frequency modes to keep (default: 64)
        mode_select: Mode selection method: 'random' or 'low' (default: 'random')
        dropout: Dropout rate (default: 0.05)
        activation: Activation function: 'relu' or 'gelu' (default: 'gelu')
        output_attention: Whether to output attention weights (default: False)
        use_norm: Whether to use normalization (default: True)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        
        # Core parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = getattr(configs, 'label_len', configs.seq_len // 2)
        self.output_attention = getattr(configs, 'output_attention', False)
        self.use_norm = getattr(configs, 'use_norm', True)
        
        # Channel configuration
        self.enc_in = configs.enc_in
        self.dec_in = getattr(configs, 'dec_in', configs.enc_in)
        self.c_out = getattr(configs, 'c_out', configs.enc_in)
        
        # Model dimensions
        self.d_model = getattr(configs, 'd_model', 512)
        self.n_heads = getattr(configs, 'n_heads', 8)
        self.e_layers = getattr(configs, 'e_layers', 2)
        self.d_layers = getattr(configs, 'd_layers', 1)
        self.d_ff = getattr(configs, 'd_ff', 2048)
        
        # FEDformer specific
        self.modes = getattr(configs, 'modes', 64)
        self.mode_select = getattr(configs, 'mode_select', 'random')
        moving_avg = getattr(configs, 'moving_avg', 25)
        self.moving_avg = moving_avg if isinstance(moving_avg, list) else [moving_avg]
        
        # Other parameters
        self.dropout = getattr(configs, 'dropout', 0.05)
        self.activation = getattr(configs, 'activation', 'gelu')
        self.embed = getattr(configs, 'embed', 'timeF')
        self.freq = getattr(configs, 'freq', 'h')
        
        # Decomposition
        if len(self.moving_avg) > 1:
            self.decomp = series_decomp_multi(self.moving_avg)
        else:
            self.decomp = series_decomp(self.moving_avg[0])

        # Embeddings (no positional encoding - frequency domain captures position)
        self.enc_embedding = DataEmbedding_wo_pos(
            self.enc_in, self.d_model, self.embed, self.freq, self.dropout
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            self.dec_in, self.d_model, self.embed, self.freq, self.dropout
        )

        # Build attention components
        encoder_self_att = FourierBlock(
            in_channels=self.d_model,
            out_channels=self.d_model,
            seq_len=self.seq_len,
            modes=self.modes,
            mode_select_method=self.mode_select
        )
        
        decoder_self_att = FourierBlock(
            in_channels=self.d_model,
            out_channels=self.d_model,
            seq_len=self.seq_len // 2 + self.pred_len,
            modes=self.modes,
            mode_select_method=self.mode_select
        )
        
        decoder_cross_att = FourierCrossAttention(
            in_channels=self.d_model,
            out_channels=self.d_model,
            seq_len_q=self.seq_len // 2 + self.pred_len,
            seq_len_kv=self.seq_len,
            modes=self.modes,
            mode_select_method=self.mode_select
        )

        # Encoder
        self.encoder = FEDformerEncoder(
            [
                FEDformerEncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        self.d_model, self.n_heads
                    ),
                    self.d_model,
                    self.d_ff,
                    moving_avg=self.moving_avg[0] if len(self.moving_avg) == 1 else self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model)
        )
        
        # Decoder
        self.decoder = FEDformerDecoder(
            [
                FEDformerDecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        self.d_model, self.n_heads
                    ),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        self.d_model, self.n_heads
                    ),
                    self.d_model,
                    self.c_out,
                    self.d_ff,
                    moving_avg=self.moving_avg[0] if len(self.moving_avg) == 1 else self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation,
                ) for _ in range(self.d_layers)
            ],
            norm_layer=my_Layernorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, **kwargs):
        """
        Forward pass for FEDformer.
        
        Args:
            x: Input time series [B, seq_len, enc_in]
            x_mark_enc: Optional encoder temporal marks [B, seq_len, time_features]
            x_dec: Optional decoder input (if None, constructed from x)
            x_mark_dec: Optional decoder temporal marks
            **kwargs: Additional arguments (ignored for unimodal operation)
            
        Returns:
            predictions: [B, pred_len, c_out]
        """
        # Store device for creating new tensors
        device = x.device
        
        # Normalization (optional)
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x_norm = x - means
            stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_norm = x_norm / stdev
        else:
            x_norm = x
        
        # Decomposition initialization
        # Initialize decoder with trend from encoder input extended by mean
        mean = torch.mean(x_norm, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_norm)
        
        # Prepare decoder inputs
        # Trend: last label_len of trend + mean for prediction horizon
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        
        # Seasonal: last label_len of seasonal + zeros for prediction horizon
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        
        # Encoder
        enc_out = self.enc_embedding(x_norm, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # Decoder
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        
        # Final prediction: seasonal + trend
        dec_out = trend_part + seasonal_part

        # Denormalization (optional)
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.label_len + self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.label_len + self.pred_len, 1))

        # Return only prediction horizon
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :self.c_out], attns
        else:
            return dec_out[:, -self.pred_len:, :self.c_out]


if __name__ == '__main__':
    """Quick test of the model."""
    from utils.tools import dotdict
    
    configs = dotdict({
        'seq_len': 96,
        'pred_len': 24,
        'enc_in': 7,
        'd_model': 128,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'd_ff': 256,
        'modes': 32,
        'mode_select': 'random',
        'moving_avg': 25,
        'dropout': 0.05,
        'activation': 'gelu',
        'output_attention': False,
        'use_norm': True,
    })
    
    model = Model(configs)
    print(f'Parameter count: {sum(p.numel() for p in model.parameters()):,}')
    
    # Test forward pass
    x = torch.randn(2, configs.seq_len, configs.enc_in)
    output = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {output.shape}')
    assert output.shape == (2, configs.pred_len, configs.enc_in)
    print('Test passed!')

