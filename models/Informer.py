"""
Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting

This is a unimodal (time series only) implementation for the fidel-ts framework.
Informer achieves O(n log n) complexity through ProbSparse attention and
uses self-attention distilling for long-horizon forecasting.

Reference: Zhou et al., "Informer: Beyond Efficient Transformer for
Long Sequence Time-Series Forecasting" (AAAI 2021)
https://arxiv.org/abs/2012.07436

Key features:
- ProbSparse Attention: O(n log n) complexity via query sparsity
- Self-attention Distilling: Halves sequence length between encoder layers
- Generative-style Decoder: Start token + zero placeholders for prediction
- Standard Encoder-Decoder architecture with cross-attention
- Temporal marks (time features) REQUIRED - matching original implementation

Usage in fidel-ts:
    # Temporal marks are automatically generated when model name is 'Informer'
    # The dataloader will provide x_mark_enc and x_mark_dec
    
    configs = dotdict({
        'seq_len': 96,
        'pred_len': 24,
        'enc_in': 7,
        'd_model': 512,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'attn': 'prob',   # ProbSparse attention
        'distil': True,   # Enable distilling
        'embed': 'timeF', # Time feature embedding (or 'fixed' for sinusoidal)
        'freq': 'h',      # Hourly frequency
        # ... other config parameters
    })
    model = Model(configs)
    # Framework provides temporal marks automatically
    output = model(x=batch_x, x_mark_enc=x_mark_enc, x_mark_dec=x_mark_dec)
"""

import torch
import torch.nn as nn

# Reuse existing fidel-ts layers (identical to original Informer)
from layers.Transformer_EncDec import (
    Encoder, EncoderLayer, ConvLayer,
    Decoder, DecoderLayer
)
from layers.SelfAttention_Family import (
    FullAttention, ProbAttention, AttentionLayer
)
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Informer: Efficient Transformer for Long Sequence Time-Series Forecasting.

    This implementation reuses the existing fidel-ts transformer layers which
    are identical to the original Informer2020 implementation.

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
        factor: ProbSparse attention factor (default: 5)
        attn: Attention type: 'prob' or 'full' (default: 'prob')
        distil: Whether to use distilling in encoder (default: True)
        dropout: Dropout rate (default: 0.05)
        activation: Activation function (default: 'gelu')
        output_attention: Whether to output attention weights (default: False)
        embed: Embedding type: 'timeF', 'fixed', 'learned' (default: 'timeF')
        freq: Time frequency for embeddings (default: 'h')
    """

    def __init__(self, configs):
        """
        Initialize Informer model.

        Args:
            configs: Configuration object/dict with model hyperparameters
        """
        super(Model, self).__init__()

        # =====================================================================
        # Core sequence parameters
        # =====================================================================
        self.seq_len = configs.seq_len
        self.label_len = getattr(configs, 'label_len', configs.seq_len // 2)
        self.pred_len = configs.pred_len
        self.output_attention = getattr(configs, 'output_attention', False)

        # =====================================================================
        # Channel configuration
        # =====================================================================
        # Get enc_in from configs, with fallback to input_channel if available
        self.enc_in = getattr(configs, 'enc_in', None)
        if self.enc_in is None:
            # Try fallback to input_channel (added by model_init from data_config)
            self.enc_in = getattr(configs, 'input_channel', None)
            if self.enc_in is None:
                raise ValueError(
                    "enc_in must be provided in model_config_overrides. "
                    "Example: model_config_overrides: {enc_in: 7}"
                )
        self.dec_in = getattr(configs, 'dec_in', self.enc_in)
        self.c_out = getattr(configs, 'c_out', self.enc_in)

        # =====================================================================
        # Model dimensions
        # =====================================================================
        self.d_model = getattr(configs, 'd_model', 512)
        self.n_heads = getattr(configs, 'n_heads', 8)
        self.e_layers = getattr(configs, 'e_layers', 2)
        self.d_layers = getattr(configs, 'd_layers', 1)
        self.d_ff = getattr(configs, 'd_ff', 2048)

        # =====================================================================
        # Informer-specific parameters
        # =====================================================================
        self.factor = getattr(configs, 'factor', 5)  # ProbSparse factor c
        self.attn = getattr(configs, 'attn', 'prob')  # 'prob' or 'full'
        self.distil = getattr(configs, 'distil', True)  # Enable distilling

        # =====================================================================
        # Other parameters
        # =====================================================================
        self.dropout = getattr(configs, 'dropout', 0.05)
        self.activation = getattr(configs, 'activation', 'gelu')
        self.embed = getattr(configs, 'embed', 'timeF')
        self.freq = getattr(configs, 'freq', 'h')

        # =====================================================================
        # Select attention mechanism
        # =====================================================================
        # ProbSparse attention for O(n log n) complexity, full for O(n^2)
        Attn = ProbAttention if self.attn == 'prob' else FullAttention

        # =====================================================================
        # Data Embeddings (Token + Positional + Temporal)
        # =====================================================================
        # Encoder embedding: transforms input time series to d_model dimension
        self.enc_embedding = DataEmbedding(
            self.enc_in, self.d_model, self.embed, self.freq, self.dropout
        )
        # Decoder embedding: transforms decoder input to d_model dimension
        self.dec_embedding = DataEmbedding(
            self.dec_in, self.d_model, self.embed, self.freq, self.dropout
        )

        # =====================================================================
        # Encoder with self-attention distilling
        # =====================================================================
        # Distilling: Conv + MaxPool between layers to halve sequence length
        # This reduces complexity and extracts dominant attention features
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, self.factor, attention_dropout=self.dropout,
                             output_attention=self.output_attention),
                        self.d_model, self.n_heads
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.e_layers)
            ],
            # Distilling conv layers between attention layers (e_layers - 1 of them)
            conv_layers=[
                ConvLayer(self.d_model) for _ in range(self.e_layers - 1)
            ] if self.distil else None,
            norm_layer=nn.LayerNorm(self.d_model)
        )

        # =====================================================================
        # Decoder with self-attention and cross-attention
        # =====================================================================
        # Decoder uses masked self-attention (causal) and full cross-attention
        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    # Masked self-attention (causal for autoregressive decoding)
                    self_attention=AttentionLayer(
                        Attn(True, self.factor, attention_dropout=self.dropout,
                             output_attention=False),
                        self.d_model, self.n_heads
                    ),
                    # Cross-attention to encoder output (always use full attention)
                    cross_attention=AttentionLayer(
                        FullAttention(False, self.factor,
                                      attention_dropout=self.dropout,
                                      output_attention=False),
                        self.d_model, self.n_heads
                    ),
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.d_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )

    def forward(self, x=None, x_enc=None, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                **kwargs):
        """
        Forward pass for Informer.

        Supports both framework interface (x as keyword) and original interface (x_enc as positional).
        
        IMPORTANT: Like the original Informer2020, temporal marks (x_mark_enc, x_mark_dec) are 
        REQUIRED for proper operation. The DataEmbedding layer uses temporal features as part of
        its embedding computation.

        Args:
            x: Input time series [B, seq_len, C] (framework interface - mapped to x_enc)
            x_enc: Encoder input [B, seq_len, enc_in] (original interface)
            x_mark_enc: Encoder temporal marks [B, seq_len, time_features] (REQUIRED)
            x_dec: Decoder input [B, label_len + pred_len, dec_in] (optional)
                   If None, auto-constructed from x_enc using generative decoding approach
            x_mark_dec: Decoder temporal marks [B, label_len + pred_len, time_features] (REQUIRED)
            enc_self_mask: Optional encoder self-attention mask
            dec_self_mask: Optional decoder self-attention mask
            dec_enc_mask: Optional decoder cross-attention mask
            **kwargs: Additional arguments (ignored for compatibility, e.g., historical_events, news)

        Returns:
            predictions: [B, pred_len, c_out]
            attns: (optional) attention weights if output_attention=True
        """
        # =====================================================================
        # Handle framework interface: convert x to x_enc
        # =====================================================================
        if x is not None:
            x_enc = x
        
        # Validate required inputs
        if x_enc is None:
            raise ValueError("Either 'x' (framework interface) or 'x_enc' (original interface) must be provided")
        
        # =====================================================================
        # Validate temporal marks (REQUIRED - matching original Informer2020)
        # =====================================================================
        # The original Informer DataEmbedding always adds temporal_embedding(x_mark)
        # without any None check - temporal marks are essential for the model
        if x_mark_enc is None:
            raise ValueError(
                "x_mark_enc (encoder temporal marks) is required for Informer. "
                "Ensure the dataloader provides time features. The model name should be 'Informer' "
                "for automatic time feature generation."
            )
        if x_mark_dec is None:
            raise ValueError(
                "x_mark_dec (decoder temporal marks) is required for Informer. "
                "Ensure the dataloader provides time features. The model name should be 'Informer' "
                "for automatic time feature generation."
            )
        
        # =====================================================================
        # Auto-construct decoder input if not provided (generative decoding)
        # =====================================================================
        # Original Informer uses: last label_len timesteps + zero-padding for pred_len
        if x_dec is None:
            x_dec = torch.zeros(
                x_enc.size(0), self.label_len + self.pred_len, x_enc.size(2),
                device=x_enc.device, dtype=x_enc.dtype
            )
            # Copy last label_len timesteps as start tokens
            x_dec[:, :self.label_len, :] = x_enc[:, -self.label_len:, :]

        # =====================================================================
        # Encoder forward pass
        # =====================================================================
        # Embed encoder input: value embedding + positional + temporal
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # Pass through encoder layers (with optional distilling)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # =====================================================================
        # Decoder forward pass
        # =====================================================================
        # Embed decoder input: value embedding + positional + temporal
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # Pass through decoder layers with cross-attention to encoder output
        dec_out = self.decoder(
            dec_out, enc_out,
            x_mask=dec_self_mask, cross_mask=dec_enc_mask
        )

        # =====================================================================
        # Return predictions (last pred_len timesteps)
        # =====================================================================
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, pred_len, c_out]


# =============================================================================
# Quick test when run directly
# =============================================================================
if __name__ == '__main__':
    """Quick test of the Informer model (matches original Informer2020 interface)."""

    class Configs:
        """Test configuration matching original Informer defaults."""
        seq_len = 96
        label_len = 48
        pred_len = 24
        enc_in = 7
        dec_in = 7
        c_out = 7
        d_model = 512
        n_heads = 8
        e_layers = 2
        d_layers = 1
        d_ff = 2048
        factor = 5
        attn = 'prob'
        distil = True
        dropout = 0.05
        activation = 'gelu'
        embed = 'timeF'  # 'timeF' uses linear projection, 'fixed' uses sinusoidal (original paper)
        freq = 'h'       # hourly frequency -> 4 time features
        output_attention = False

    configs = Configs()
    model = Model(configs)

    print(f'Informer parameter count: {sum(p.numel() for p in model.parameters()):,}')

    # Create test inputs (temporal marks REQUIRED - matching original Informer2020)
    # For freq='h', time features have 4 dimensions: [month, day, weekday, hour]
    enc = torch.randn(2, 96, 7)           # [B, seq_len, enc_in]
    enc_mark = torch.randn(2, 96, 4)      # [B, seq_len, time_features]
    dec_mark = torch.randn(2, 48 + 24, 4) # [B, label_len + pred_len, time_features]

    # Test with framework interface (x=..., auto-constructed x_dec)
    out = model(x=enc, x_mark_enc=enc_mark, x_mark_dec=dec_mark)
    print(f'Framework interface output shape: {out.shape}')
    assert out.shape == (2, 24, 7), f"Expected (2, 24, 7), got {out.shape}"

    # Test with full interface (explicit x_dec - matches original Informer2020)
    dec = torch.randn(2, 48 + 24, 7)  # [B, label_len + pred_len, dec_in]
    out_full = model(x_enc=enc, x_mark_enc=enc_mark, x_dec=dec, x_mark_dec=dec_mark)
    print(f'Original interface output shape: {out_full.shape}')
    assert out_full.shape == (2, 24, 7), f"Expected (2, 24, 7), got {out_full.shape}"

    # Test with full attention (no ProbSparse)
    configs.attn = 'full'
    configs.distil = False
    model_full = Model(configs)
    out_full_attn = model_full(x=enc, x_mark_enc=enc_mark, x_mark_dec=dec_mark)
    print(f'Full attention output shape: {out_full_attn.shape}')
    assert out_full_attn.shape == (2, 24, 7), f"Expected (2, 24, 7), got {out_full_attn.shape}"

    # Test with fixed embedding (original Informer paper default)
    configs.embed = 'fixed'
    configs.attn = 'prob'
    configs.distil = True
    model_fixed = Model(configs)
    # Fixed embedding expects integer temporal indices [month, day, weekday, hour, ...]
    enc_mark_fixed = torch.randint(0, 12, (2, 96, 4))       # Integer indices for temporal embedding
    dec_mark_fixed = torch.randint(0, 12, (2, 48 + 24, 4))
    out_fixed = model_fixed(x=enc, x_mark_enc=enc_mark_fixed, x_mark_dec=dec_mark_fixed)
    print(f'Fixed embedding output shape: {out_fixed.shape}')
    assert out_fixed.shape == (2, 24, 7), f"Expected (2, 24, 7), got {out_fixed.shape}"

    print('All Informer tests passed!')

