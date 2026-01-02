"""
RetNet (Retentive Network) components for LeRet model.

A simplified implementation of RetNet for time series forecasting.
Implements parallel mode only (not chunkwise recurrent).

Reference: Sun et al., "Retentive Network: A Successor to Transformer 
for Large Language Models" (2023)

Components:
    - RetNetConfig: Configuration for RetNet architecture
    - RMSNorm: Root Mean Square Layer Normalization  
    - MultiScaleRetention: Multi-head retention mechanism (parallel mode)
    - GLU: Gated Linear Unit feedforward network
    - RetNetDecoder: Stack of retention + FFN layers
    - DecoderLayer: Single retention block
    - RetNetRelPos: Relative position encoding for retention
"""

from .config import RetNetConfig
from .rms_norm import RMSNorm
from .multiscale_retention import MultiScaleRetention
from .feedforward import GLU
from .retnet import RetNetDecoder, DecoderLayer, RetNetRelPos

__all__ = [
    'RetNetConfig',
    'RMSNorm', 
    'MultiScaleRetention',
    'GLU',
    'RetNetDecoder',
    'DecoderLayer',
    'RetNetRelPos',
]

