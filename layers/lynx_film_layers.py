import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.FiLM_layers import EncoderFilm, EncoderLayerFilm, FiLMGenerator
import numpy as np
from utils.model_regularization import apply_norms_to_linear_layers

class iTransformerFilm(nn.Module):
    """
    Modified iTransformer with FiLM modulation.
    Paper link: https://arxiv.org/abs/2310.06625
    
    FiLM Modulation and Text Input
    ==============================
    FiLM (Feature-wise Linear Modulation) uses text embeddings to generate
    channel-wise scale (gamma) and shift (beta) parameters that modulate
    the time series representation at each transformer layer.
    
    The text input is FLATTENED across all timesteps to generate GLOBAL
    modulation parameters per channel. This means:
    - ALL text timesteps contribute to a SINGLE set of (gamma, beta) per channel
    - There is NO timestamp-specific modulation in FiLM
    - All predictions are conditioned identically based on aggregate text info
    
    Text Source Based on timestamp_semantics
    ========================================
    - t_about (Fidel-TS): Uses y_hetero (news), text aligned to PREDICTION window
      -> text_seq_len = ceil(pred_len / stride)
      -> Assumption: Forecasts for t+k were available at time t
    
    - t_known (Time-MMD/TTC): Uses x_hetero (historical_events), text aligned to INPUT window
      -> text_seq_len = ceil(seq_len / hetero_stride)
      -> Avoids lookahead bias from using text published during prediction window
    
    The FiLMGenerator must be initialized with the CORRECT text_seq_len based on
    which text source will be used, since it has a fixed input dimension.
    """

    def __init__(self, configs):
        super(iTransformerFilm, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        
        # ========================================================================
        # Calculate text_seq_len based on timestamp_semantics and hetero_stride
        # ========================================================================
        # 
        # timestamp_semantics determines which text source the model receives:
        # - t_about: y_hetero with pred_len timesteps (strided by hetero_stride)
        # - t_known: x_hetero with seq_len timesteps (strided by hetero_stride)
        #
        # hetero_stride can be:
        # - Pre-computed from training.text_embedding_stride config (preferred)
        # - Computed from stride/hetero_align_stride (backward compatibility)
        #
        # FiLMGenerator FLATTENS all text timesteps into a single vector, so
        # we MUST initialize it with the exact input dimension it will receive.
        # ========================================================================
        
        timestamp_semantics = getattr(configs, 'timestamp_semantics', None)
        if timestamp_semantics is None:
            raise ValueError(
                "iTransformerFilm requires 'timestamp_semantics' in configs. "
                "This must be set explicitly to avoid lookahead bias:\n"
                "  - 't_about': Text timestamps refer to the event/target time (Fidel-TS datasets)\n"
                "  - 't_known': Text timestamps refer to publication time (Time-MMD/TTC datasets)\n"
                "Set timestamp_semantics in data_config (hetero_info.timestamp_semantics or top-level)."
            )
        if timestamp_semantics not in ('t_about', 't_known'):
            raise ValueError(
                f"Invalid timestamp_semantics: '{timestamp_semantics}'. "
                f"Must be 't_about' or 't_known'."
            )
        
        # Get hetero_stride - prefer pre-computed value, fall back to legacy computation
        # Pre-computed value comes from training.text_embedding_stride via experiment_config_builder
        if hasattr(configs, 'hetero_stride') and configs.hetero_stride is not None:
            hetero_stride = configs.hetero_stride
        else:
            # Legacy fallback: compute from stride/hetero_align_stride
            hetero_align_stride = getattr(configs, 'hetero_align_stride', True)
            stride = getattr(configs, 'stride', 1) or 1
            hetero_stride = stride if hetero_align_stride else 1
        
        if timestamp_semantics == 't_about':
            # Using y_hetero: text aligned to prediction window
            # Length = ceil(pred_len / hetero_stride)
            # Note: hetero_stride for y_hetero is applied to pred_len
            self.text_seq_len = int(np.ceil(self.pred_len / hetero_stride))
            print(f'[ info ] iTransformerFilm: timestamp_semantics=t_about')
            print(f'         -> text_seq_len = ceil({self.pred_len}/{hetero_stride}) = {self.text_seq_len} (from y_hetero/news)')
        else:  # t_known
            # Using x_hetero: text aligned to input window
            # Length = ceil(seq_len / hetero_stride)
            self.text_seq_len = int(np.ceil(self.seq_len / hetero_stride))
            print(f'[ info ] iTransformerFilm: timestamp_semantics=t_known')
            print(f'         -> text_seq_len = ceil({self.seq_len}/{hetero_stride}) = {self.text_seq_len} (from x_hetero/historical_events)')
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        
        # Encoder-only architecture with FiLM
        # Create layers
        attn_layers = [
            EncoderLayerFilm(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            ) for l in range(configs.e_layers)
        ]
        
        # Create FiLM generators (one per layer)
        # Input to generator is flattened text_emb: text_seq_len * text_dim
        # Output is 4 * d_model (gamma1, beta1, gamma2, beta2)
        # Optional configs:
        # - film_hidden_dim: Bottleneck size for FiLM MLP (forces compression)
        # - film_dropout: Dropout inside FiLM MLP for regularization
        film_hidden_dim = getattr(configs, 'film_hidden_dim', None)
        film_dropout = getattr(configs, 'film_dropout', 0.0)
        film_per_channel = getattr(configs, 'text_film_per_channel', True)
        film_generators = [
            FiLMGenerator(
                text_dim=configs.text_dim, 
                output_dim=configs.d_model, 
                seq_len=self.text_seq_len,
                hidden_dim=film_hidden_dim or configs.d_model,
                dropout=film_dropout,
                per_channel=film_per_channel
            )
            for l in range(configs.e_layers)
        ]
        
        self.encoder = EncoderFilm(
            attn_layers=attn_layers,
            film_generators=film_generators,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
    
    def forecast(self, x_enc, text_emb, x_mark_enc=None):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        # Pass text_emb to encoder for FiLM modulation
        # text_emb is expected to be [B, C, L, D]
        enc_out, attns = self.encoder(enc_out, text_emb=text_emb, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x, text_emb, **kwargs):
        dec_out = self.forecast(x, text_emb)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

    def apply_film_param_norms(self, use_weight_norm=False, use_spectral_norm=False):
        """
        Apply optional normalization to FiLM generator MLPs.
        
        Args:
            use_weight_norm: Enable weight normalization on FiLM MLP Linear layers.
            use_spectral_norm: Enable spectral normalization on FiLM MLP Linear layers.
        """
        # Skip if no normalization is requested
        if not (use_weight_norm or use_spectral_norm):
            return
        # Apply normalization to each FiLM generator
        for generator in self.encoder.film_generators:
            apply_norms_to_linear_layers(
                generator,
                use_weight_norm=use_weight_norm,
                use_spectral_norm=use_spectral_norm
            )
