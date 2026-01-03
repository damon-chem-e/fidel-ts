"""
TimeCMA Model for Fidel-TS.

Implements the dual-modality encoding with cross-modal alignment approach
from "TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting
via Cross-Modality Alignment" (AAAI 2025).

Key Features:
    - Dual-branch encoding: Time series encoder + LLM prompt encoder
    - Cross-modal attention: Aligns TS features with LLM embeddings
    - RevIN normalization: Handles non-stationary time series
    - Uses precomputed LLM embeddings (last token of GPT-2 or similar)

Architecture:
    Input: [B, L, N] + LLM embeddings [B, E, N] (E = LLM hidden dim)
    
    1. RevIN Normalization
    2. Length-to-Feature Projection: [B, L, N] → [B, N, C]
    3. Time Series Encoder (Transformer)
    4. LLM Prompt Encoder (Transformer)
    5. Cross-Modal Alignment: Q(TS) × KV(LLM)
    6. Transformer Decoder
    7. Output Projection + RevIN Denormalization
    
    Output: [B, pred_len, N]

Reference:
    TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting
    via Cross-Modality Alignment (AAAI 2025)
"""

import torch
import torch.nn as nn
from layers.StandardNorm import Normalize
from layers.TimeCMA_layers import CrossModal


class Model(nn.Module):
    """
    TimeCMA: Cross-Modality Alignment for Time Series Forecasting.
    
    This model uses dual-modality encoding with cross-modal alignment between
    time series features and LLM prompt embeddings.
    
    Args:
        configs: Configuration object with the following attributes:
            - seq_len: Input sequence length
            - pred_len: Prediction horizon length
            - enc_in: Number of input channels/variables (num_nodes)
            - channel: Hidden dimension for time series encoding (default: 32)
            - d_llm: LLM embedding dimension (default: 768 for GPT-2)
            - e_layers: Number of encoder layers (default: 1)
            - d_layers: Number of decoder layers (default: 1)
            - n_heads: Number of attention heads (default: 8)
            - d_ff: Feed-forward dimension (default: 32)
            - dropout: Dropout rate (default: 0.2)
            - device: Target device (default: 'cuda:0')
    
    Input:
        - x: Time series input [B, seq_len, num_nodes]
        - x_mark: Time features [B, seq_len, time_features] (optional, for compatibility)
        - llm_embeddings: Precomputed LLM embeddings [B, d_llm, num_nodes] or [B, d_llm, num_nodes, 1]
    
    Output:
        - predictions: [B, pred_len, num_nodes]
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # =====================================================================
        # Configuration
        # =====================================================================
        
        # Required parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_nodes = configs.enc_in  # Number of channels/variables
        
        # Optional parameters with defaults matching original TimeCMA
        self.channel = getattr(configs, 'channel', 32)
        self.d_llm = getattr(configs, 'd_llm', 768)  # GPT-2 base hidden dim
        self.e_layer = getattr(configs, 'e_layers', 1)
        self.d_layer = getattr(configs, 'd_layers', 1)
        self.n_heads = getattr(configs, 'n_heads', 8)
        self.d_ff = getattr(configs, 'd_ff', 32)
        self.dropout = getattr(configs, 'dropout', 0.2)
        self.device = getattr(configs, 'device', 'cuda:0')
        
        # =====================================================================
        # Model Components
        # =====================================================================
        
        # 1. RevIN Normalization
        self.normalize_layers = Normalize(self.num_nodes, affine=False)
        
        # 2. Length-to-Feature Projection
        # Projects sequence length dimension to hidden channel dimension
        self.length_to_feature = nn.Linear(self.seq_len, self.channel)
        
        # 3. Time Series Encoder (Transformer)
        ts_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel,
            nhead=self.n_heads,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout
        )
        self.ts_encoder = nn.TransformerEncoder(
            ts_encoder_layer,
            num_layers=self.e_layer
        )
        
        # 4. Prompt Encoder (for LLM embeddings)
        prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm,
            nhead=self.n_heads,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout
        )
        self.prompt_encoder = nn.TransformerEncoder(
            prompt_encoder_layer,
            num_layers=self.e_layer
        )
        
        # 5. Cross-Modal Alignment
        # Q: Time series features [B, C, N]
        # KV: LLM embeddings [B, E, N]
        # Output: Aligned features [B, C, N]
        self.cross = CrossModal(
            d_model=self.num_nodes,
            n_heads=1,
            d_ff=self.d_ff,
            norm='LayerNorm',
            attn_dropout=self.dropout,
            dropout=self.dropout,
            pre_norm=True,
            activation="gelu",
            res_attention=True,
            n_layers=1,
            store_attn=False
        )
        
        # 6. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.channel,
            nhead=self.n_heads,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.d_layer
        )
        
        # 7. Output Projection
        # Projects from hidden channel dimension to prediction length
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True)
    
    def param_num(self):
        """Count total number of parameters."""
        return sum([param.nelement() for param in self.parameters()])
    
    def count_trainable_params(self):
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x, channel_description, x_mark_enc=None, **kwargs):
        """
        Forward pass for TimeCMA.
        
        Adapted to fidel-ts framework interface. Maps framework parameters to
        TimeCMA's expected inputs:
            - x → input_data: Input time series [B, seq_len, num_nodes]
            - x_mark_enc → input_data_mark: Time features (optional, not used)
            - channel_description → embeddings: Precomputed LLM embeddings
                Shape: [B, d_llm, num_nodes] or [B, d_llm, num_nodes, 1]
        
        Args:
            x: Input time series [B, seq_len, num_nodes]
            channel_description: Precomputed LLM embeddings from LLMEmbeddingProvider
            x_mark_enc: Time features (optional, not used in TimeCMA)
            **kwargs: Additional arguments (ignored for compatibility)
        
        Returns:
            predictions: [B, pred_len, num_nodes]
        """
        # Map framework interface to TimeCMA's internal names
        input_data = x
        embeddings = channel_description
        
        # Ensure float tensors
        input_data = input_data.float()
        embeddings = embeddings.float()
        
        # Handle embedding shape: squeeze trailing dimension if present
        # [B, E, N, 1] → [B, E, N]
        if embeddings.dim() == 4:
            embeddings = embeddings.squeeze(-1)
        
        # Permute embeddings: [B, E, N] → [B, N, E]
        embeddings = embeddings.permute(0, 2, 1)
        
        # =====================================================================
        # 1. RevIN Normalization
        # =====================================================================
        input_data = self.normalize_layers(input_data, 'norm')
        
        # =====================================================================
        # 2. Length-to-Feature Projection
        # =====================================================================
        # [B, L, N] → [B, N, L] → [B, N, C]
        input_data = input_data.permute(0, 2, 1)
        input_data = self.length_to_feature(input_data)
        
        # =====================================================================
        # 3. Time Series Encoding
        # =====================================================================
        # [B, N, C] → [B, N, C]
        enc_out = self.ts_encoder(input_data)
        # [B, N, C] → [B, C, N] for cross-modal attention
        enc_out = enc_out.permute(0, 2, 1)
        
        # =====================================================================
        # 4. Prompt Encoding
        # =====================================================================
        # [B, N, E] → [B, N, E]
        embeddings = self.prompt_encoder(embeddings)
        # [B, N, E] → [B, E, N] for cross-modal attention
        embeddings = embeddings.permute(0, 2, 1)
        
        # =====================================================================
        # 5. Cross-Modal Alignment
        # =====================================================================
        # Q: [B, C, N], KV: [B, E, N] → [B, C, N]
        cross_out = self.cross(enc_out, embeddings, embeddings)
        # [B, C, N] → [B, N, C] for decoder
        cross_out = cross_out.permute(0, 2, 1)
        
        # =====================================================================
        # 6. Transformer Decoder
        # =====================================================================
        # [B, N, C] → [B, N, C]
        dec_out = self.decoder(cross_out, cross_out)
        
        # =====================================================================
        # 7. Output Projection
        # =====================================================================
        # [B, N, C] → [B, N, pred_len]
        dec_out = self.c_to_length(dec_out)
        # [B, N, pred_len] → [B, pred_len, N]
        dec_out = dec_out.permute(0, 2, 1)
        
        # =====================================================================
        # 8. RevIN Denormalization
        # =====================================================================
        dec_out = self.normalize_layers(dec_out, 'denorm')
        
        return dec_out
    
    def move_to_device(
        self,
        seq_x, seq_y, x_time, y_time,
        x_hetero, y_hetero,
        hetero_x_time, hetero_y_time,
        hetero_general, hetero_channel,
        device
    ):
        """
        Move data to device (compatibility with fidel-ts training loop).
        
        For TimeCMA, the `hetero_channel` is expected to contain the
        precomputed LLM embeddings (not channel descriptions like other models).
        
        Args:
            seq_x: Input time series [B, seq_len, N]
            seq_y: Target time series [B, pred_len, N]
            x_time: Input time features [B, seq_len, time_features]
            y_time: Target time features [B, pred_len, time_features]
            x_hetero: Input heterogeneous features (not used)
            y_hetero: Target heterogeneous features (not used)
            hetero_x_time: Heterogeneous input timestamps (not used)
            hetero_y_time: Heterogeneous target timestamps (not used)
            hetero_general: General heterogeneous features (not used)
            hetero_channel: LLM embeddings [B, d_llm, N] or [B, d_llm, N, 1]
            device: Target device
        
        Returns:
            tuple: All inputs (moved to device where applicable)
        """
        # Move essential tensors to device
        seq_x = seq_x.float().to(device)
        seq_y = seq_y.float().to(device)
        x_time = x_time.float().to(device) if x_time is not None else x_time
        hetero_channel = hetero_channel.float().to(device)  # LLM embeddings
        
        return (
            seq_x, seq_y, x_time, y_time,
            x_hetero, y_hetero,
            hetero_x_time, hetero_y_time,
            hetero_general, hetero_channel
        )


# =============================================================================
# Alias for Original Class Name (Compatibility)
# =============================================================================

# Original TimeCMA uses class name "Dual"
Dual = Model

