"""
Patch Embedding for Time Series.

This module provides patch-based embedding for time series, similar to
ViT's approach for images. Time series are divided into overlapping
patches which are then projected to an embedding space.

Key Differences from Text Tokenization:
- Operates directly on continuous values (not discrete tokens)
- Uses 1D convolution for projection (not lookup table)
- Supports overlapping patches via stride parameter

Components:
    - TokenEmbedding: Projects patches to embeddings via Conv1d
    - PatchEmbedding: Full pipeline including padding and unfolding
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Convert patches to embeddings using 1D convolution.
    
    Uses a 1x1 convolution to project patch values to embedding space.
    This is equivalent to a linear projection but implemented as conv
    for efficient batched operations.
    
    Args:
        patch_len: Length of each input patch
        d_model: Output embedding dimension
    
    Example:
        >>> embed = TokenEmbedding(patch_len=16, d_model=32)
        >>> patches = torch.randn(14, 12, 16)  # [B*C, num_patches, patch_len]
        >>> embeddings = embed(patches)  # [14, 12, 32]
    """
    
    def __init__(self, patch_len: int, d_model: int):
        """
        Initialize TokenEmbedding.
        
        Args:
            patch_len: Length of each patch (input channels for conv)
            d_model: Output embedding dimension (output channels for conv)
        """
        super().__init__()
        
        # 1x1 convolution for patch-to-embedding projection
        # in_channels = patch_len, out_channels = d_model
        self.tokenConv = nn.Conv1d(
            in_channels=patch_len,
            out_channels=d_model,
            kernel_size=1,
            padding=0,
            bias=False
        )

        # Initialize with Xavier uniform for stable training
        # Using linear nonlinearity since forward pass has no activation
        nn.init.xavier_uniform_(self.tokenConv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches to embeddings.
        
        Args:
            x: Patches tensor [B*C, num_patches, patch_len]
        
        Returns:
            Embeddings tensor [B*C, num_patches, d_model]
        """
        # Transpose for Conv1d: [B*C, patch_len, num_patches]
        x = x.permute(0, 2, 1)
        
        # Apply 1x1 convolution: [B*C, d_model, num_patches]
        x = self.tokenConv(x)
        
        # Transpose back: [B*C, num_patches, d_model]
        return x.transpose(1, 2)


class PatchEmbedding(nn.Module):
    """
    Patch embedding for time series.
    
    Converts [B, C, T] time series to [B*C, num_patches, d_model] embeddings.
    
    The patching process:
    1. Pad sequence to ensure complete patches
    2. Unfold into overlapping patches
    3. Flatten batch and channel dimensions
    4. Project patches to embeddings via TokenEmbedding
    
    Args:
        d_model: Embedding dimension for output patches
        patch_len: Length of each patch
        stride: Stride between consecutive patches (overlap if stride < patch_len)
        dropout: Dropout rate applied to embeddings
    
    Example:
        >>> patch_embed = PatchEmbedding(d_model=32, patch_len=16, stride=8)
        >>> x = torch.randn(2, 7, 96)  # [B, C, T]
        >>> embeddings, n_vars = patch_embed(x)
        >>> # embeddings: [14, 12, 32], n_vars: 7
    """
    
    def __init__(
        self, 
        d_model: int, 
        patch_len: int, 
        stride: int, 
        dropout: float = 0.1
    ):
        """
        Initialize PatchEmbedding.
        
        Args:
            d_model: Embedding dimension for patches
            patch_len: Length of each patch
            stride: Stride between patches (overlap if stride < patch_len)
            dropout: Dropout rate
        """
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        
        # Padding layer to ensure we can extract complete patches
        # Pads at the end of the sequence
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        
        # Convert patches to embeddings
        self.value_embedding = TokenEmbedding(patch_len, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Convert time series to patch embeddings.
        
        Args:
            x: Time series tensor [B, C, T]
        
        Returns:
            Tuple of:
                - embeddings: Patch embeddings [B*C, num_patches, d_model]
                - n_vars: Number of variables/channels (C)
        
        Note:
            num_patches = (T + stride - patch_len) / stride + 1
            For T=96, patch_len=16, stride=8: num_patches = (96+8-16)/8+1 = 12
        """
        # Store number of variables for reshaping later
        n_vars = x.shape[1]
        
        # Pad sequence: [B, C, T] -> [B, C, T + stride]
        x = self.padding_patch_layer(x)
        
        # Extract overlapping patches using unfold
        # [B, C, T+stride] -> [B, C, num_patches, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # Flatten batch and channel dimensions for efficient processing
        # [B, C, num_patches, patch_len] -> [B*C, num_patches, patch_len]
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        
        # Embed patches: [B*C, num_patches, patch_len] -> [B*C, num_patches, d_model]
        x = self.value_embedding(x)
        
        return self.dropout(x), n_vars

