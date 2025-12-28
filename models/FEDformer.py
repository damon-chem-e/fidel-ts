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
- Two versions: Fourier and Wavelets
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
        'version': 'Fourier',  # or 'Wavelets'
        # ... other config parameters
    })
    model = Model(configs)
    output = model(x)  # x: [B, seq_len, channels]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.FEDformer_layers import (
    series_decomp, series_decomp_multi, my_Layernorm,
    FourierBlock, FourierCrossAttention, AutoCorrelationLayer,
    FEDformerEncoderLayer, FEDformerDecoderLayer,
    FEDformerEncoder, FEDformerDecoder,
    MultiWaveletTransform, MultiWaveletCross
)
from layers.Embed import TokenEmbedding, PositionalEmbedding, TimeFeatureEmbedding, TemporalEmbedding


class DataEmbedding_wo_pos(nn.Module):
    """
    Data embedding without positional encoding.
    
    FEDformer omits positional encoding because frequency-domain operations
    inherently capture sequential information through the phase of Fourier coefficients.
    This matches the original FEDformer implementation exactly.
    """
    
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()
        
        # Value embedding via 1D convolution (mixes channels)
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        
        # Position embedding (kept but not used in forward - matches original)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        
        # Temporal embedding (always created, matches original implementation)
        # Uses TemporalEmbedding for fixed/learned, TimeFeatureEmbedding for timeF
        if embed_type != 'timeF':
            self.temporal_embedding = TemporalEmbedding(
                d_model=d_model, embed_type=embed_type, freq=freq
            )
        else:
            self.temporal_embedding = TimeFeatureEmbedding(
                d_model=d_model, embed_type=embed_type, freq=freq
            )
            
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x: [B, L, C] - time series values
        # x_mark: [B, L, time_features] - temporal features (required in original)
        # Note: Original FEDformer always uses temporal embedding
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class Model(nn.Module):
    """
    FEDformer: Frequency Enhanced Decomposed Transformer
    
    A unimodal time series forecasting model that operates in the frequency domain
    with O(N) complexity. Uses seasonal-trend decomposition for improved predictions.
    Supports both Fourier and Wavelet versions.
    
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
        version: 'Fourier' or 'Wavelets' (default: 'Fourier')
        L: Wavelet level (default: 1, only for Wavelets version)
        base: Wavelet base: 'legendre' or 'chebyshev' (default: 'legendre')
        cross_activation: Cross attention activation for Wavelets (default: 'tanh')
        dropout: Dropout rate (default: 0.05)
        activation: Activation function: 'relu' or 'gelu' (default: 'gelu')
        output_attention: Whether to output attention weights (default: False)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        
        # Version selection (Fourier or Wavelets)
        self.version = getattr(configs, 'version', 'Fourier')
        self.mode_select = getattr(configs, 'mode_select', 'random')
        self.modes = getattr(configs, 'modes', 64)
        
        # Core parameters
        self.seq_len = configs.seq_len
        self.label_len = getattr(configs, 'label_len', configs.seq_len // 2)
        self.pred_len = configs.pred_len
        self.output_attention = getattr(configs, 'output_attention', False)
        
        # Channel configuration
        # enc_in must be explicitly provided in model_config_overrides
        self.enc_in = getattr(configs, 'enc_in', None)
        if self.enc_in is None:
            raise ValueError(
                "enc_in (number of input channels) must be provided in model_config_overrides. "
                "Example: model_config_overrides: {enc_in: 4}"
            )
        # Note: dotdict.__getattr__ returns None for missing keys, so getattr() won't use defaults
        # Need explicit None check to fall back to enc_in
        self.dec_in = getattr(configs, 'dec_in', None)
        if self.dec_in is None:
            self.dec_in = self.enc_in
        self.c_out = getattr(configs, 'c_out', None)
        if self.c_out is None:
            self.c_out = self.enc_in
        
        # Model dimensions
        self.d_model = getattr(configs, 'd_model', 512)
        self.n_heads = getattr(configs, 'n_heads', 8)
        self.e_layers = getattr(configs, 'e_layers', 2)
        self.d_layers = getattr(configs, 'd_layers', 1)
        self.d_ff = getattr(configs, 'd_ff', 2048)
        
        # FEDformer specific
        moving_avg = getattr(configs, 'moving_avg', 25)
        self.moving_avg = moving_avg if isinstance(moving_avg, list) else [moving_avg]
        
        # Other parameters
        self.dropout = getattr(configs, 'dropout', 0.05)
        self.activation = getattr(configs, 'activation', 'gelu')
        self.embed = getattr(configs, 'embed', 'timeF')
        self.freq = getattr(configs, 'freq', 'h')
        
        # Wavelet-specific parameters
        self.L = getattr(configs, 'L', 1)
        self.base = getattr(configs, 'base', 'legendre')
        self.cross_activation = getattr(configs, 'cross_activation', 'tanh')
        
        # Decomposition
        kernel_size = moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embeddings (no positional encoding - frequency domain captures position)
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(
            self.enc_in, self.d_model, self.embed, self.freq, self.dropout
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            self.dec_in, self.d_model, self.embed, self.freq, self.dropout
        )

        # Build attention components based on version
        if self.version == 'Wavelets':
            # Wavelet-based attention
            encoder_self_att = MultiWaveletTransform(
                ich=self.d_model, 
                L=self.L, 
                base=self.base
            )
            decoder_self_att = MultiWaveletTransform(
                ich=self.d_model, 
                L=self.L, 
                base=self.base
            )
            decoder_cross_att = MultiWaveletCross(
                in_channels=self.d_model,
                out_channels=self.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=self.modes,
                ich=self.d_model,
                base=self.base,
                activation=self.cross_activation
            )
        else:
            # Fourier-based attention (default)
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
        
        # Print mode info (matching original)
        enc_modes = int(min(self.modes, self.seq_len // 2))
        dec_modes = int(min(self.modes, (self.seq_len // 2 + self.pred_len) // 2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

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
                    moving_avg=moving_avg,
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
                    moving_avg=moving_avg,
                    dropout=self.dropout,
                    activation=self.activation,
                ) for _ in range(self.d_layers)
            ],
            norm_layer=my_Layernorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )

    def forward(self, x=None, x_enc=None, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, **kwargs):
        """
        Forward pass for FEDformer.
        
        Supports two calling conventions:
        1. Framework interface: forward(x, **kwargs) - where x is [B, seq_len, C]
        2. Original interface: forward(x_enc, x_mark_enc, x_dec, x_mark_dec, ...)
        
        Args:
            x: Input time series [B, seq_len, C] (framework interface)
            x_enc: Input time series [B, seq_len, enc_in] (original interface)
            x_mark_enc: Encoder temporal marks [B, seq_len, time_features] (original interface)
            x_dec: Decoder input [B, label_len + pred_len, dec_in] (original interface)
            x_mark_dec: Decoder temporal marks [B, label_len + pred_len, time_features] (original interface)
            enc_self_mask: Optional encoder self-attention mask
            dec_self_mask: Optional decoder self-attention mask
            dec_enc_mask: Optional decoder cross-attention mask
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            predictions: [B, pred_len, c_out]
        """
        # Handle framework interface: convert x to FEDformer format
        if x is not None:
            # Framework passes x as [B, seq_len, C]
            x_enc = x
            # Temporal marks should be provided as direct keyword arguments (x_mark_enc, x_mark_dec)
            # If not provided, raise an error
            if x_mark_enc is None or x_mark_dec is None:
                raise ValueError(
                    "FEDformer requires temporal marks (x_mark_enc, x_mark_dec) when using framework interface. "
                    "These should be provided via the dataloader with generate_time_features=True."
                )
        
        # If using original interface, x_enc must be provided
        if x_enc is None:
            raise ValueError("Either 'x' (framework interface) or 'x_enc' (original interface) must be provided")
        
        # Temporal marks are required for FEDformer (either passed directly or via kwargs)
        if x_mark_enc is None:
            raise ValueError("x_mark_enc (encoder temporal marks) is required for FEDformer")
        if x_mark_dec is None:
            raise ValueError("x_mark_dec (decoder temporal marks) is required for FEDformer")
        
        # Decomposition initialization
        # Initialize decoder with trend from encoder input extended by mean
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)
        
        # Prepare decoder inputs
        # Trend: last label_len of trend + mean for prediction horizon
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        
        # Seasonal: last label_len of seasonal + zeros for prediction horizon
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        
        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        # Decoder
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, 
            x_mask=dec_self_mask, 
            cross_mask=dec_enc_mask,
            trend=trend_init
        )
        
        # Final prediction: seasonal + trend
        dec_out = trend_part + seasonal_part

        # Return only prediction horizon (matching original)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


if __name__ == '__main__':
    """Quick test of the model (matches original FEDformer test)."""
    
    class Configs(object):
        """Test configuration matching original FEDformer."""
        ab = 0
        modes = 32
        mode_select = 'random'
        version = 'Fourier'  # Can also test 'Wavelets'
        moving_avg = [12, 24]
        L = 1
        base = 'legendre'
        cross_activation = 'tanh'
        seq_len = 96
        label_len = 48
        pred_len = 96
        output_attention = True
        enc_in = 7
        dec_in = 7
        d_model = 16
        embed = 'timeF'
        dropout = 0.05
        freq = 'h'
        factor = 1
        n_heads = 8
        d_ff = 16
        e_layers = 2
        d_layers = 1
        c_out = 7
        activation = 'gelu'
        wavelet = 0

    configs = Configs()
    model = Model(configs)

    print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))
    
    # Create test inputs matching original
    enc = torch.randn([3, configs.seq_len, 7])
    enc_mark = torch.randn([3, configs.seq_len, 4])
    dec = torch.randn([3, configs.seq_len // 2 + configs.pred_len, 7])
    dec_mark = torch.randn([3, configs.seq_len // 2 + configs.pred_len, 4])
    
    out = model.forward(enc, enc_mark, dec, dec_mark)
    print(f'Output shape: {out[0].shape}')
    print('Fourier test passed!')
    
    # Test Wavelets version
    print('\nTesting Wavelets version...')
    configs.version = 'Wavelets'
    model_wavelet = Model(configs)
    print('Wavelet parameter number is {}'.format(sum(p.numel() for p in model_wavelet.parameters())))
    out_wavelet = model_wavelet.forward(enc, enc_mark, dec, dec_mark)
    print(f'Wavelet output shape: {out_wavelet[0].shape}')
    print('Wavelets test passed!')

