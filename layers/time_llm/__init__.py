"""
Time-LLM Layer Components.

This module provides the layer implementations for Time-LLM:
- ReprogrammingLayer: Cross-attention for TS-to-LLM space mapping
- PatchEmbedding: Conv1D-based patch embedding
- TokenEmbedding: Patch-to-embedding projection
- Normalize: RevIN-style instance normalization
- DynamicPromptBuilder: Per-batch prompt generation with statistics

Reference:
    Jin et al., "Time-LLM: Time Series Forecasting by Reprogramming 
    Large Language Models" (ICLR 2024)
"""

from .reprogramming import ReprogrammingLayer
from .patch_embed import PatchEmbedding, TokenEmbedding
from .normalization import Normalize
from .dynamic_prompt import DynamicPromptBuilder

__all__ = [
    'ReprogrammingLayer',
    'PatchEmbedding',
    'TokenEmbedding',
    'Normalize',
    'DynamicPromptBuilder',
]

