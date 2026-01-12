"""
LeRet Backbone: RetNet-based encoder for time series.

Implements the core encoding pipeline:
1. Patching: Divide time series into patches
2. Linear projection: Project patches to d_model dimension
3. RetNet encoder: Multi-scale retention layers
4. Dual prediction heads: Patch-level (pretraining) and sequence-level (forecasting)

The backbone processes each channel independently (channel-independent design).
"""

from typing import Optional, Tuple
import torch
from torch import nn
from torch import Tensor

from layers.RevIN import RevIN
from layers.RetNet import RetNetConfig, RetNetDecoder


class LeRetEncoder(nn.Module):
    """
    LeRet Encoder using RetNet backbone.
    
    Processes patched time series through a stack of retention layers.
    Uses channel-independent design where each channel is processed separately.
    
    Args:
        c_in: Number of input channels
        patch_num: Number of patches after patching
        patch_len: Length of each patch
        max_seq_len: Maximum sequence length (not used, for compatibility)
        n_layers: Number of retention layers
        d_model: Model dimension
        n_heads: Number of retention heads
        d_k: Key dimension per head (optional, defaults to d_model // n_heads)
        d_v: Value dimension per head (optional, defaults to d_model // n_heads)
        d_ff: FFN intermediate dimension
        norm: Normalization type ('BatchNorm' or 'LayerNorm')
        attn_dropout: Attention/retention dropout
        dropout: General dropout rate
        act: Activation function name
        store_attn: Whether to store attention weights (not used)
        key_padding_mask: Whether to use key padding mask (not used)
        padding_var: Padding variance (not used)
        attn_mask: Attention mask (not used)
        res_attention: Whether to use residual attention (not used)
        pre_norm: Whether to use pre-normalization
        pe: Positional encoding type (not used - RetNet uses relative positions)
        learn_pe: Whether PE is learnable (not used)
        verbose: Print debug info
    """
    
    def __init__(
        self,
        c_in: int,
        patch_num: int,
        patch_len: int,
        max_seq_len: int = 1024,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 8,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = 'BatchNorm',
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        store_attn: bool = False,
        key_padding_mask: str = 'auto',
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        pe: str = 'zeros',
        learn_pe: bool = True,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.d_model = d_model
        
        # =================================================================
        # Input projection: patch_len -> d_model
        # =================================================================
        self.W_P = nn.Linear(patch_len, d_model)
        
        # Dropout after projection
        self.dropout = nn.Dropout(dropout)
        
        # =================================================================
        # RetNet Configuration
        # =================================================================
        config = RetNetConfig(
            decoder_layers=n_layers,
            decoder_embed_dim=d_model,
            decoder_ffn_embed_dim=d_ff,
            decoder_retention_heads=n_heads,
            decoder_value_embed_dim=d_model,
            dropout=dropout,
            activation_fn=act,
            decoder_normalize_before=pre_norm,
            no_output_layer=True,  # Don't need vocab projection
        )
        
        # =================================================================
        # RetNet Encoder (uses RetNetDecoder in encoder-style)
        # =================================================================
        self.encoder = RetNetDecoder(config)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Encode patched time series.
        
        Args:
            x: Patched input [batch, n_vars, patch_len, patch_num]
            
        Returns:
            Encoded output [batch, n_vars, d_model, patch_num]
        """
        n_vars = x.shape[1]
        
        # =================================================================
        # Project patches: patch_len -> d_model
        # =================================================================
        # x: [B, C, patch_len, patch_num] -> [B, C, patch_num, patch_len]
        x = x.permute(0, 1, 3, 2)
        # x: [B, C, patch_num, patch_len] -> [B, C, patch_num, d_model]
        x = self.W_P(x)
        
        # =================================================================
        # Reshape for RetNet: combine batch and channels
        # =================================================================
        # x: [B, C, patch_num, d_model] -> [B*C, patch_num, d_model]
        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        u = self.dropout(u)
        
        # =================================================================
        # RetNet encoding
        # =================================================================
        # u: [B*C, patch_num, d_model]
        z, _ = self.encoder(u, features_only=True)
        
        # =================================================================
        # Reshape back: separate batch and channels
        # =================================================================
        # z: [B*C, patch_num, d_model] -> [B, C, patch_num, d_model]
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        # z: [B, C, patch_num, d_model] -> [B, C, d_model, patch_num]
        z = z.permute(0, 1, 3, 2)
        
        return z


class Patch_Level_Head(nn.Module):
    """
    Patch-level prediction head for auto-regressive pretraining.
    
    Predicts the original patch values from the encoded representation.
    Used in Stage 1 (pretraining) to learn patch-level temporal patterns.
    
    Args:
        d_model: Model dimension
        patch_len: Length of each patch (output dimension)
        head_dropout: Dropout rate
    """
    
    def __init__(self, d_model: int, patch_len: int, head_dropout: float = 0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(d_model, patch_len)
        self.dropout = nn.Dropout(head_dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Predict patch values.
        
        Args:
            x: Encoded representation [batch, n_vars, d_model, patch_num]
            
        Returns:
            Patch predictions [batch, n_vars, patch_num * patch_len]
        """
        # x: [B, C, d_model, patch_num] -> [B, C, patch_num, d_model]
        x = x.permute(0, 1, 3, 2)
        # x: [B, C, patch_num, d_model] -> [B, C, patch_num, patch_len]
        x = self.linear(x)
        x = self.dropout(x)
        # x: [B, C, patch_num, patch_len] -> [B, C, patch_num * patch_len]
        x = self.flatten(x)
        return x


class Flatten_Head(nn.Module):
    """
    Flattening prediction head for sequence-level forecasting.
    
    Flattens the encoded representation and projects to target length.
    Used in Stage 2 (finetuning) for the final forecasting task.
    
    Args:
        individual: If True, use separate linear layers per channel
        n_vars: Number of channels
        nf: Number of features (d_model * patch_num)
        target_window: Prediction horizon length
        head_dropout: Dropout rate
    """
    
    def __init__(
        self, 
        individual: bool, 
        n_vars: int, 
        nf: int, 
        target_window: int, 
        head_dropout: float = 0.0
    ):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            # Separate projection per channel
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            # Shared projection across channels
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Project to forecasting output.
        
        Args:
            x: Encoded representation [batch, n_vars, d_model, patch_num]
            
        Returns:
            Forecasting output [batch, n_vars, target_window]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                # z: [B, d_model, patch_num] -> [B, d_model * patch_num]
                z = self.flattens[i](x[:, i, :, :])
                # z: [B, d_model * patch_num] -> [B, target_window]
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            # x: [B, n_vars, target_window]
            x = torch.stack(x_out, dim=1)
        else:
            # x: [B, C, d_model, patch_num] -> [B, C, d_model * patch_num]
            x = self.flatten(x)
            # x: [B, C, d_model * patch_num] -> [B, C, target_window]
            x = self.linear(x)
            x = self.dropout(x)
        
        return x


class LeRet_backbone(nn.Module):
    """
    Complete LeRet backbone with RevIN, patching, encoding, and heads.
    
    Implements the full LeRet pipeline:
    1. RevIN normalization (optional)
    2. Patching
    3. RetNet encoding
    4. Dual prediction heads (patch-level and sequence-level)
    
    Args:
        c_in: Number of input channels
        context_window: Input sequence length
        target_window: Prediction horizon length
        patch_len: Length of each patch
        stride: Stride for patching
        max_seq_len: Maximum sequence length
        n_layers: Number of retention layers
        d_model: Model dimension
        n_heads: Number of retention heads
        d_k: Key dimension per head
        d_v: Value dimension per head
        d_ff: FFN intermediate dimension
        norm: Normalization type
        attn_dropout: Attention dropout
        dropout: General dropout
        act: Activation function
        key_padding_mask: Key padding mask setting
        padding_var: Padding variance
        attn_mask: Attention mask
        res_attention: Residual attention flag
        pre_norm: Pre-normalization flag
        store_attn: Store attention flag
        pe: Positional encoding type
        learn_pe: Learnable PE flag
        fc_dropout: FC layer dropout
        head_dropout: Head dropout
        padding_patch: Patch padding mode
        pretrain_head: Use pretrain head
        head_type: Head type
        individual: Individual channels flag
        revin: Use RevIN
        affine: RevIN affine flag
        subtract_last: RevIN subtract last flag
        verbose: Verbose output
    """
    
    def __init__(
        self,
        c_in: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        max_seq_len: Optional[int] = 1024,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 8,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = 'BatchNorm',
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: str = 'auto',
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = 'zeros',
        learn_pe: bool = True,
        fc_dropout: float = 0.0,
        head_dropout: float = 0.0,
        padding_patch: Optional[str] = None,
        pretrain_head: bool = False,
        head_type: str = 'flatten',
        individual: bool = False,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__()
        
        # =================================================================
        # RevIN (Reversible Instance Normalization)
        # =================================================================
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # =================================================================
        # Patching parameters
        # =================================================================
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        
        # Compute number of patches
        patch_num = int((context_window - patch_len) / stride + 1)
        
        # =================================================================
        # RetNet Backbone (Encoder)
        # =================================================================
        self.backbone = LeRetEncoder(
            c_in=c_in,
            patch_num=patch_num,
            patch_len=patch_len,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
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
            verbose=verbose,
            **kwargs
        )
        
        # =================================================================
        # Prediction Heads
        # =================================================================
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual
        
        # Sequence-level head (for forecasting - Stage 2)
        self.sequence_head = Flatten_Head(
            individual=self.individual,
            n_vars=self.n_vars,
            nf=self.head_nf,
            target_window=target_window,
            head_dropout=head_dropout
        )
        
        # Patch-level head (for pretraining - Stage 1)
        self.patch_head = Patch_Level_Head(
            d_model=d_model,
            patch_len=patch_len,
            head_dropout=head_dropout
        )
    
    def forward(
        self, 
        z: Tensor,
        language_integrator=None,
        language_embeddings=None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through LeRet backbone.
        
        Args:
            z: Input time series [batch, n_vars, seq_len]
            language_integrator: Optional language integrator module for text embeddings
            language_embeddings: Optional language embeddings [text_num, language_dim]
            
        Returns:
            Tuple of:
                - sequence_output: [batch, n_vars, target_window] (Stage 2 output)
                - patch_output: [batch, n_vars, seq_len] (Stage 1 output)
        """
        # =================================================================
        # RevIN normalization
        # =================================================================
        if self.revin:
            # z: [B, C, S] -> [B, S, C] for RevIN
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)
        
        # =================================================================
        # Patching
        # =================================================================
        # z: [B, C, S] -> [B, C, patch_num, patch_len]
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # z: [B, C, patch_num, patch_len] -> [B, C, patch_len, patch_num]
        z = z.permute(0, 1, 3, 2)
        
        # =================================================================
        # RetNet encoding
        # =================================================================
        # h: [B, C, d_model, patch_num]
        h = self.backbone(z)
        
        # =================================================================
        # Language integration (optional)
        # =================================================================
        if language_integrator is not None and language_embeddings is not None:
            # Integrate language knowledge into encoded patches
            # h: [B, C, d_model, patch_num] -> [B, C, d_model, patch_num] (enhanced)
            h = language_integrator(h, language_embeddings)
        
        # =================================================================
        # Patch-level head (Stage 1 - auto-regression)
        # =================================================================
        # auto_y: [B, C, patch_num * patch_len]
        auto_y = self.patch_head(h)
        
        # =================================================================
        # Sequence-level head (Stage 2 - forecasting)
        # =================================================================
        # z_out: [B, C, target_window]
        z_out = self.sequence_head(h)
        
        # =================================================================
        # RevIN denormalization (only for sequence output)
        # =================================================================
        if self.revin:
            # z_out: [B, C, T] -> [B, T, C] for RevIN
            z_out = z_out.permute(0, 2, 1)
            z_out = self.revin_layer(z_out, 'denorm')
            z_out = z_out.permute(0, 2, 1)
        
        return z_out, auto_y

