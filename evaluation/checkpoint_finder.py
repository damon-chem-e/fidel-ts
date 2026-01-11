"""
Checkpoint finder for evaluation.

This module provides functions to find checkpoints in experiment directories,
using the experiment manager structure instead of pattern matching.
"""

import glob
from pathlib import Path
from typing import Optional


def find_checkpoint_from_experiment_dir(
    experiment_dir: Path,
    version: str = "best"
) -> Path:
    """
    Find checkpoint file in experiment directory.
    
    This function finds checkpoints using the experiment manager structure,
    preferring best_checkpoint files when version="best".
    
    Args:
        experiment_dir: Path to experiment directory (e.g., output/experiment_id)
        version: "best", "latest", or specific pattern (default: "best")
        
    Returns:
        Path to checkpoint file (not directory)
        
    Raises:
        FileNotFoundError: If checkpoints directory or checkpoint file not found
    """
    checkpoints_dir = experiment_dir / "checkpoints"
    
    if not checkpoints_dir.exists():
        raise FileNotFoundError(
            f"Checkpoints directory not found: {checkpoints_dir}\n"
            f"  Experiment directory: {experiment_dir}"
        )
    
    if version == "best":
        # Prefer best_checkpoint.* (saved explicitly as best)
        best_ckpt = list(checkpoints_dir.glob("best_checkpoint*"))
        if best_ckpt:
            # Filter out directories, only files
            best_ckpt = [p for p in best_ckpt if p.is_file()]
            if best_ckpt:
                return best_ckpt[0]
        
        # Fallback to checkpoint.pth (PyTorch) or last.ckpt (Lightning)
        if (checkpoints_dir / "checkpoint.pth").exists():
            return checkpoints_dir / "checkpoint.pth"
        if (checkpoints_dir / "last.ckpt").exists():
            return checkpoints_dir / "last.ckpt"
        
        # Find latest by modification time as final fallback
        all_ckpts = (
            list(checkpoints_dir.glob("checkpoint*")) + 
            list(checkpoints_dir.glob("*.ckpt")) +
            list(checkpoints_dir.glob("*.pth"))
        )
        # Filter out directories, only files
        all_ckpts = [p for p in all_ckpts if p.is_file()]
        if all_ckpts:
            return max(all_ckpts, key=lambda p: p.stat().st_mtime)
        
        raise FileNotFoundError(
            f"No checkpoint file found in {checkpoints_dir}\n"
            f"  Expected files: best_checkpoint.*, checkpoint.pth, last.ckpt, or checkpoint-*.ckpt"
        )
    
    elif version == "latest":
        # Find by modification time
        all_ckpts = (
            list(checkpoints_dir.glob("checkpoint*")) + 
            list(checkpoints_dir.glob("*.ckpt")) +
            list(checkpoints_dir.glob("*.pth"))
        )
        # Filter out directories, only files
        all_ckpts = [p for p in all_ckpts if p.is_file()]
        if not all_ckpts:
            raise FileNotFoundError(
                f"No checkpoint file found in {checkpoints_dir}\n"
                f"  Expected files: checkpoint.*, *.ckpt, or *.pth"
            )
        return max(all_ckpts, key=lambda p: p.stat().st_mtime)
    
    else:
        # Specific version pattern (backward compatibility)
        # This handles patterns like "20240101" or partial matches
        pattern = str(checkpoints_dir / f"*{version}*")
        matching_ckpts = glob.glob(pattern)
        if not matching_ckpts:
            raise FileNotFoundError(
                f"No checkpoint found matching pattern: {version}\n"
                f"  Searched in: {checkpoints_dir}\n"
                f"  Pattern: *{version}*"
            )
        # Filter out directories, only files, and sort by modification time
        matching_ckpts = [Path(p) for p in matching_ckpts if Path(p).is_file()]
        matching_ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matching_ckpts[0]


def find_checkpoint_legacy(eval_config) -> Path:
    """
    Legacy checkpoint finding using pattern matching.
    
    This is kept for backward compatibility with old evaluation configs
    that use pattern matching instead of experiment directory structure.
    
    Args:
        eval_config: Evaluation config with checkpoint_base, model, data, input_len, output_len
        
    Returns:
        Path to checkpoint directory (not file)
        
    Raises:
        FileNotFoundError: If no checkpoint found matching pattern
    """
    import os
    import glob
    
    ckpt_pattern = f'_{eval_config.model}_{eval_config.data}_{eval_config.output_len}_{eval_config.input_len}'
    
    if eval_config.version == 'latest':
        ckpt_paths = [
            os.path.join(eval_config.checkpoint_base, d) 
            for d in os.listdir(eval_config.checkpoint_base) 
            if ckpt_pattern in d and os.path.isdir(os.path.join(eval_config.checkpoint_base, d))
        ]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found with pattern: *{ckpt_pattern}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]
    elif eval_config.version == 'oldest':
        ckpt_paths = [
            os.path.join(eval_config.checkpoint_base, d) 
            for d in os.listdir(eval_config.checkpoint_base) 
            if ckpt_pattern in d and os.path.isdir(os.path.join(eval_config.checkpoint_base, d))
        ]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found with pattern: *{ckpt_pattern}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[0]
    else:
        # Specific version pattern
        pattern = os.path.join(eval_config.checkpoint_base, eval_config.version + ckpt_pattern)
        ckpt_paths = [p for p in glob.glob(pattern) if os.path.isdir(p)]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found for pattern: {pattern}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]
    
    return Path(ckpt_path)
