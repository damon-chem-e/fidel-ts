
import os
import torch
import numpy as np
import polars as pl
from typing import Dict, List, Any, Union, Optional

class PerSampleMetricsTracker:
    """
    Utility class to handle per-sample metrics collection and storage.
    """
    def __init__(self, output_dir: str):
        self.output_dir = os.path.join(output_dir, "per_sample")
        os.makedirs(self.output_dir, exist_ok=True)
        self.epoch_metrics: List[Dict[str, Any]] = []

    def add_batch(self, 
                  epoch: Optional[int], 
                  split: str, 
                  entity_id: Optional[str], 
                  timestamps: Union[np.ndarray, torch.Tensor, List],
                  losses: Union[np.ndarray, torch.Tensor, List]):
        """
        Add a batch of per-sample metrics.
        
        Args:
            epoch: Current epoch number (None for validation/test folds).
            split: Dataset split ('train', 'val', 'test').
            entity_id: Identifier for the entity (e.g. sensor ID). None for train/val.
            timestamps: Array-like of timestamps for the samples.
            losses: Array-like of per-sample loss values.
        """
        # Ensure inputs are iterable/list-like
        if isinstance(losses, torch.Tensor):
            losses = losses.detach().cpu().numpy()
        if isinstance(timestamps, torch.Tensor):
            timestamps = timestamps.detach().cpu().numpy()
            
        for t, l in zip(timestamps, losses):
            # Handle potential numpy scalars
            ts_val = t.item() if hasattr(t, 'item') else t
            loss_val = l.item() if hasattr(l, 'item') else l
            
            # For train/val, entity_id might be None. We handle this by storing null/None.
            # Polars handles None as null.
            
            self.epoch_metrics.append({
                "epoch": epoch,
                "split": split,
                "entity_id": str(entity_id) if entity_id is not None else None,
                "timestamp": str(ts_val), # Convert to string to ensure compatibility
                "loss": float(loss_val)
            })

    def save_epoch(self, epoch: int):
        """
        Save the accumulated metrics for the epoch to a parquet file and clear buffer.
        
        Args:
            epoch: Epoch number for filename (e.g., epoch_001.parquet)
        """
        if not self.epoch_metrics:
            return

        df = pl.DataFrame(self.epoch_metrics)
        
        # Enforce schema for consistency (epoch can be null for val/test)
        df = df.with_columns([
            pl.col("epoch").cast(pl.Int32, strict=False),  # Allow null values
            pl.col("split").cast(pl.Categorical),
            pl.col("entity_id").cast(pl.Utf8),
            pl.col("timestamp").cast(pl.Utf8),
            pl.col("loss").cast(pl.Float32)
        ])
        
        save_path = os.path.join(self.output_dir, f"epoch_{epoch:03d}.parquet")
        df.write_parquet(save_path)
        
        # Clear buffer
        self.epoch_metrics = []

    def save_split(self, split: str):
        """
        Save the accumulated metrics for a split (val/test) to a parquet file and clear buffer.
        Used for validation and test folds where epoch is None.
        
        Args:
            split: Dataset split name ('val' or 'test') for filename (e.g., val.parquet, test.parquet)
        """
        if not self.epoch_metrics:
            return

        df = pl.DataFrame(self.epoch_metrics)
        
        # Enforce schema for consistency (epoch can be null for val/test)
        df = df.with_columns([
            pl.col("epoch").cast(pl.Int32, strict=False),  # Allow null values
            pl.col("split").cast(pl.Categorical),
            pl.col("entity_id").cast(pl.Utf8),
            pl.col("timestamp").cast(pl.Utf8),
            pl.col("loss").cast(pl.Float32)
        ])
        
        save_path = os.path.join(self.output_dir, f"{split}.parquet")
        df.write_parquet(save_path)
        
        # Clear buffer
        self.epoch_metrics = []

    def clear(self):
        """Clear the metrics buffer without saving."""
        self.epoch_metrics = []
