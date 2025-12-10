import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.lynx_mmitransformer_text_encoder import NewsEmbedding
import numpy as np


class Model(nn.Module):
    """
    lynx_mmitransformer: Multimodal iTransformer with Channel Description Integration
    
    This is a custom architecture that integrates text embeddings into iTransformer:
    1. Channel descriptions are prepended to each variate representation
    2. News is embedded to global text variates that all time series variates attend to
    
    Architecture:
    1. Time series: [B, L, C] -> embed -> [B, C, d_model]
    2. Channel descriptions: [B, C, M] (time-agnostic) -> prepend to variates
    3. Concatenate: [B, C, d_model + M]
    4. News: [B, L, N, text_dim] -> embed -> [B, K, d_model + M] (K global text variates)
    5. Concatenate all: [B, C + K, d_model + M]
    6. Process with modified iTransformer (works with d_model + M)
    7. Project and slice to keep only time series variates
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        
        # Get dimensions
        self.d_model = configs.d_model
        self.channel_desc_dim = getattr(configs, 'channel_desc_dim', 0)  # M dimension
        self.num_text_variates = getattr(configs, 'num_text_variates', 1)  # K variates
        self.d_model_extended = self.d_model + self.channel_desc_dim  # d_model + M
        
        # 1. Time Series Embedding (iTransformer style)
        # Input: [B, L, C] -> Output: [B, C, d_model]
        self.enc_embedding = DataEmbedding_inverted(
            c_in=configs.seq_len, 
            d_model=configs.d_model, 
            dropout=configs.dropout
        )
        
        # 2. News Embedding (time-agnostic)
        # Input: [B, L, N, text_dim] -> Output: [B, K, d_model + M]
        text_seq_len = int(np.ceil(configs.pred_len / getattr(configs, 'stride', 8)))
        self.news_embedding = NewsEmbedding(
            text_dim=configs.text_dim,
            output_dim=self.d_model_extended,
            num_text_variates=self.num_text_variates,
            seq_len=text_seq_len,
            dropout=configs.dropout,
            num_heads=getattr(configs, 'news_num_heads', 4),
            num_layers=getattr(configs, 'news_num_layers', 2),
            aggregation_type=getattr(configs, 'news_aggregation_type', 'hierarchical')
        )
        
        # 3. Encoder (Modified to work with d_model + M)
        # All layers need to work with d_model_extended dimension
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), 
                        self.d_model_extended,  # Use extended dimension
                        configs.n_heads),
                    self.d_model_extended,  # d_model parameter
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model_extended)
        )
        
        # 4. Projector (works with d_model_extended)
        self.projector = nn.Linear(self.d_model_extended, configs.pred_len, bias=True)
    
    def forecast(self, x_enc, news, channel_description, x_mark_enc=None):
        """
        Args:
            x_enc: [B, L, C] - Time series input
            news: [B, L, N, text_dim] - News embeddings
            channel_description: [B, C, M] - Channel description embeddings (time-agnostic)
            x_mark_enc: Optional temporal features
        """
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N (N = number of variates/channels)

        # Step 1: Time Series Embedding
        # [B, L, N] -> [B, N, d_model]
        enc_time = self.enc_embedding(x_enc, x_mark_enc)  # [B, N, d_model]
        
        # Step 2: Channel Description Handling
        # channel_description: [B, C, M] (time-agnostic, C should match N)
        # Handle case where channel_description might be [B, 1, C, M] or [B, C, M]
        if len(channel_description.shape) == 4:
            # [B, 1, C, M] -> [B, C, M]
            channel_description = channel_description.squeeze(1)
        elif len(channel_description.shape) == 3:
            # Already [B, C, M]
            pass
        else:
            raise ValueError(f"Unexpected channel_description shape: {channel_description.shape}")
        
        # Ensure C matches N
        if channel_description.shape[1] != N:
            raise ValueError(
                f"Channel description channels ({channel_description.shape[1]}) "
                f"must match time series variates ({N})"
            )
        
        # Step 3: Concatenate Channel Description + Time Series
        # [B, N, d_model] + [B, N, M] -> [B, N, d_model + M]
        enc_variates = torch.cat([enc_time, channel_description], dim=-1)  # [B, N, d_model + M]
        
        # Step 4: News Embedding (time-agnostic)
        # [B, L, N, text_dim] -> [B, K, d_model + M]
        # Create news mask if needed (for padded news items)
        news_mask = None
        if news is not None:
            # Check if news has padding (zeros)
            news_mask = (news.sum(dim=-1) != 0).float()  # [B, L, N]
            enc_news = self.news_embedding(news, news_mask)  # [B, K, d_model + M]
        else:
            # If no news, create zero embeddings
            enc_news = torch.zeros(
                enc_variates.shape[0], 
                self.num_text_variates, 
                self.d_model_extended,
                device=enc_variates.device
            )  # [B, K, d_model + M]
        
        # Step 5: Concatenate All Variates
        # [B, N, d_model + M] + [B, K, d_model + M] -> [B, N + K, d_model + M]
        enc_in = torch.cat([enc_variates, enc_news], dim=1)  # [B, N + K, d_model + M]

        # Step 6: Transformer Encoder
        enc_out, attns = self.encoder(enc_in, attn_mask=None)  # [B, N + K, d_model + M]

        # Step 7: Projection
        # [B, N + K, d_model + M] -> [B, N + K, pred_len]
        dec_out = self.projector(enc_out)  # [B, N + K, pred_len]
        
        # Step 8: Permute and Slice
        # [B, N + K, pred_len] -> [B, pred_len, N + K]
        dec_out = dec_out.permute(0, 2, 1)  # [B, pred_len, N + K]
        
        # Slice to keep only time series variates (first N)
        # We discard the text variates which were just for attention
        dec_out = dec_out[:, :, :N]  # [B, pred_len, N]

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x, news, channel_description, **kwargs):
        dec_out = self.forecast(x, news, channel_description)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

