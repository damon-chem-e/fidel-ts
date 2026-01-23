"""
Utility functions for applying optional regularization to model submodules.
"""

from typing import Iterable

import torch.nn as nn
from torch.nn.utils import weight_norm, spectral_norm


def apply_norms_to_linear_layers(
    module: nn.Module,
    use_weight_norm: bool = False,
    use_spectral_norm: bool = False
) -> None:
    """
    Apply weight norm or spectral norm to all Linear layers in a module.
    
    Args:
        module: Module to traverse for Linear layers.
        use_weight_norm: Whether to apply weight normalization.
        use_spectral_norm: Whether to apply spectral normalization.
    """
    # Skip if no normalization is requested
    if not (use_weight_norm or use_spectral_norm):
        return
    # Disallow incompatible simultaneous norms
    if use_weight_norm and use_spectral_norm:
        raise ValueError("Only one of weight_norm or spectral_norm can be enabled at a time.")
    # Apply normalization to each Linear layer once
    for sub_module in module.modules():
        if not isinstance(sub_module, nn.Linear):
            continue
        if use_weight_norm and not hasattr(sub_module, "weight_g"):
            weight_norm(sub_module)
        if use_spectral_norm and not hasattr(sub_module, "weight_u"):
            spectral_norm(sub_module)
