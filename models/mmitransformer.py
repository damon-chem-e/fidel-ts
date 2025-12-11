import torch
import torch.nn as nn
import numpy as np
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """
    mmitransformer: Multimodal iTransformer (Original Paper Implementation)
    
    This model integrates text embeddings into the iTransformer architecture.
    Text embeddings (BERT) are projected to the same dimension as time series variates
    and treated as additional tokens in the Transformer encoder.
    
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        
        # Number of recent patches to use (hyperparameter)
        # Default: use all available patches, but can be limited
        # Note: Ideally, we only use the last patch (num_recent_patches=1), making pooling
        # over the temporal dimension redundant. This is suggested based on the original
        # MMiTransformer paper which states "only use the most recent text".
        self.num_recent_patches = getattr(configs, 'num_recent_patches', None)
        
        # Calculate expected text sequence length (similar to stride-based calculation)
        # This is used to determine the input size for the projector
        stride = getattr(configs, 'stride', 8)
        expected_patches = int(np.ceil(configs.pred_len / stride))
        
        # If num_recent_patches is specified, use that; otherwise use expected_patches
        patches_to_use = self.num_recent_patches if self.num_recent_patches is not None else expected_patches
        
        # Text Projector: Projects concatenated recent text embeddings to d_model
        # Input: text_dim * patches_to_use
        # Output: d_model
        self.text_projector = nn.Linear(configs.text_dim * patches_to_use, configs.d_model)
        
        # 2. Time Series Embedding (iTransformer)
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        
        # 3. Encoder (Standard Transformer)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # 4. Projector
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
    
    def forecast(self, x_enc, news, channel_description, x_mark_enc=None):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N

        # 1. Time Series Embedding
        # B L N -> B N E
        enc_time = self.enc_embedding(x_enc, x_mark_enc)
        
        # 2. Text Processing (Original MMiTransformer approach)
        # news: [B, L, N, text_dim] - BERT embeddings (no preprocessing)
        # Only use the most recent text: pool across temporal dimension
        B, L_text, N_text, D_text = news.shape
        
        # Select most recent patches (if num_recent_patches is specified)
        if self.num_recent_patches is not None and L_text > self.num_recent_patches:
            # Use only the most recent num_recent_patches
            news = news[:, -self.num_recent_patches:, :, :]  # [B, num_recent_patches, N, text_dim]
            L_text = self.num_recent_patches
        
        # Concatenate embeddings over recent patches
        # [B, L_text, N, text_dim] -> [B, N, L_text, text_dim] -> [B, N, L_text * text_dim]
        text_emb_flat = news.permute(0, 2, 1, 3).reshape(B, N_text, L_text * D_text)
        
        # Handle case where actual patches don't match expected input size
        expected_input_size = self.text_projector.in_features
        actual_input_size = L_text * D_text
        
        if actual_input_size != expected_input_size:
            # Pad or truncate to match expected size
            if actual_input_size < expected_input_size:
                # Pad with zeros
                padding_size = expected_input_size - actual_input_size
                padding = torch.zeros(B, N_text, padding_size, device=text_emb_flat.device, dtype=text_emb_flat.dtype)
                text_emb_flat = torch.cat([text_emb_flat, padding], dim=-1)
            else:
                # Truncate (take the last expected_input_size dimensions)
                text_emb_flat = text_emb_flat[:, :, -expected_input_size:]
        
        # Project to d_model
        # [B, N, L_text * text_dim] -> [B, N, d_model]
        enc_text = self.text_projector(text_emb_flat)
        
        # 3. Concatenate Time Series and Text Embeddings
        # enc_time: [B, N, d_model]
        # enc_text: [B, N, d_model]
        # enc_in: [B, 2N, d_model]
        enc_in = torch.cat([enc_time, enc_text], dim=1)

        # 4. Transformer Encoder
        enc_out, attns = self.encoder(enc_in, attn_mask=None)

        # 5. Projection
        # [B, 2N, d_model] -> [B, 2N, pred_len]
        dec_out = self.projector(enc_out)
        
        # 6. Permute and Slice
        # [B, 2N, pred_len] -> [B, pred_len, 2N]
        dec_out = dec_out.permute(0, 2, 1)
        
        # Slice to keep only time series variates (first N)
        # We discard the text variates which were just for attention
        dec_out = dec_out[:, :, :N]

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x, news, channel_description, **kwargs):
        dec_out = self.forecast(x, news, channel_description)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

