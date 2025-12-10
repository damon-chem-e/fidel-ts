import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.TGTSF_torch import text_encoder
import numpy as np


class Model(nn.Module):
    """
    mmitransformer: Multimodal iTransformer
    
    This model integrates text embeddings into the iTransformer architecture.
    Text embeddings are projected to the same dimension as time series variates
    and treated as additional tokens in the Transformer encoder.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        
        # 1. Text Encoder (from TGTSF)
        self.text_encoder = text_encoder(
            cross_layer=configs.cross_layers, 
            self_layer=configs.self_layers, 
            embedding_dim=configs.text_dim, 
            num_heads=configs.n_heads, 
            dropout=configs.dropout, 
            pred_len=configs.pred_len, 
            stride=configs.stride
        )
        
        # Calculate text sequence length based on TGTSF text_encoder logic
        self.text_seq_len = int(np.ceil(configs.pred_len / configs.stride))
        
        # Text Projector: Projects flattened text embeddings to d_model
        # Input: text_dim * text_seq_len
        # Output: d_model
        self.text_projector = nn.Linear(configs.text_dim * self.text_seq_len, configs.d_model)
        
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
                ) for l in range(configs.e_layers)
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
        
        # 2. Text Processing
        # text_emb: [B, L_text, N, text_dim]
        text_emb = self.text_encoder(news, channel_description)
        B, L_text, N_text, D_text = text_emb.shape
        
        # Flatten text embeddings per variate
        # [B, L_text, N, text_dim] -> [B, N, L_text, text_dim] -> [B, N, L_text * text_dim]
        # Note: We assume N_text matches N (variates). 
        # TGTSF text_encoder returns [B, L_text, C, D] where C is channels (variates).
        text_emb_flat = text_emb.permute(0, 2, 1, 3).reshape(B, N_text, -1)
        
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

