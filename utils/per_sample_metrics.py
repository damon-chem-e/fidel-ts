
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

    def _convert_to_numpy(self, 
                          losses: Optional[Union[np.ndarray, torch.Tensor, List]],
                          timestamps: Optional[Union[np.ndarray, torch.Tensor, List]],
                          sample_ids: Optional[Union[np.ndarray, torch.Tensor, List]],
                          per_channel_losses: Optional[Union[np.ndarray, torch.Tensor, List]]):
        """
        Convert torch tensors to numpy arrays and ensure list format.
        
        Args:
            losses: Optional loss values to convert.
            timestamps: Optional timestamps to convert.
            sample_ids: Optional sample IDs to convert.
            per_channel_losses: Optional per-channel losses to convert.
            
        Returns:
            Tuple of (losses, timestamps, sample_ids, per_channel_losses) as numpy arrays or lists.
        """
        # Convert losses if provided
        if losses is not None and isinstance(losses, torch.Tensor):
            losses = losses.detach().cpu().numpy()
        
        # Convert timestamps if provided
        if timestamps is not None and isinstance(timestamps, torch.Tensor):
            timestamps = timestamps.detach().cpu().numpy()
        
        # Convert sample_ids if provided
        if sample_ids is not None:
            if isinstance(sample_ids, torch.Tensor):
                sample_ids = sample_ids.detach().cpu().numpy()
            sample_ids = list(sample_ids)
        
        # Convert per_channel_losses if provided
        if per_channel_losses is not None and isinstance(per_channel_losses, torch.Tensor):
            per_channel_losses = per_channel_losses.detach().cpu().numpy()
        
        return losses, timestamps, sample_ids, per_channel_losses

    def _normalize_entity_ids(self, 
                              entity_id: Optional[Union[str, List[str]]],
                              sample_ids: Optional[List[str]],
                              per_channel_losses: Optional[np.ndarray],
                              losses: Optional[np.ndarray],
                              timestamps: Optional[np.ndarray]) -> List[Optional[str]]:
        """
        Normalize entity_id into a list of entity IDs.
        
        Args:
            entity_id: Can be None, single string, or list of strings.
            sample_ids: Optional list of sample IDs to extract entity IDs from.
            per_channel_losses: Optional per-channel losses to determine batch size.
            losses: Optional losses to determine batch size.
            timestamps: Optional timestamps to determine batch size.
            
        Returns:
            List of entity IDs (or None values).
        """
        # Extract from sample_ids if entity_id is None
        if entity_id is None and sample_ids is not None:
            # Extract from sample_ids (format: "entity_id|timestamp|sequence_index")
            return [sid.split('|')[0] for sid in sample_ids]
        
        # Single entity_id for all samples
        if isinstance(entity_id, str):
            # Determine number of samples from available data
            num_samples = (len(per_channel_losses) if per_channel_losses is not None 
                          else len(losses) if losses is not None 
                          else len(timestamps) if timestamps is not None 
                          else 1)
            return [entity_id] * num_samples
        
        # Already a list or array
        if isinstance(entity_id, (list, np.ndarray)):
            return list(entity_id)
        
        # Default: create list of None values
        num_samples = (len(per_channel_losses) if per_channel_losses is not None 
                      else len(losses) if losses is not None 
                      else len(timestamps) if timestamps is not None 
                      else 1)
        return [None] * num_samples

    def _prepare_channel_ids(self, 
                             channel_ids: Optional[Union[np.ndarray, torch.Tensor, List]],
                             per_channel_losses: np.ndarray) -> List[str]:
        """
        Prepare channel IDs as a list of strings.
        
        Args:
            channel_ids: Optional channel IDs. If None, uses integer indices.
            per_channel_losses: Per-channel losses array to determine number of channels.
            
        Returns:
            List of channel ID strings.
        """
        if channel_ids is None:
            # Use integer indices based on number of channels
            num_channels = per_channel_losses.shape[1]
            return [str(i) for i in range(num_channels)]
        
        # Convert to list of strings
        if isinstance(channel_ids, (list, np.ndarray)):
            return [str(cid) for cid in channel_ids]
        
        return [str(channel_ids)]

    def _ensure_sample_ids_and_timestamps(self, 
                                          sample_ids: Optional[List[str]],
                                          timestamps: Optional[np.ndarray],
                                          entity_ids: List[Optional[str]]) -> tuple[List[str], List]:
        """
        Ensure sample_ids and timestamps are available, generating if needed.
        
        Args:
            sample_ids: Optional list of sample IDs.
            timestamps: Optional array of timestamps.
            entity_ids: List of entity IDs for generating sample IDs.
            
        Returns:
            Tuple of (sample_ids, timestamps) as lists.
        """
        # Generate sample_ids if not provided
        if sample_ids is None:
            if timestamps is None:
                raise ValueError("Either sample_ids or timestamps must be provided")
            sample_ids = [f"{eid}|{t}|0" for eid, t in zip(entity_ids, timestamps)]
        
        # Extract or generate timestamps if not provided
        if timestamps is None:
            # Extract from sample_ids if available
            timestamps = [sid.split('|')[1] if '|' in sid else str(i) 
                         for i, sid in enumerate(sample_ids)]
        
        # Convert timestamps to list if needed
        if not isinstance(timestamps, list):
            timestamps = list(timestamps)
        
        return sample_ids, timestamps

    def _add_per_channel_metrics(self, 
                                 epoch: Optional[int],
                                 split: str,
                                 sample_ids: List[str],
                                 entity_ids: List[Optional[str]],
                                 timestamps: List,
                                 losses: np.ndarray,
                                 per_channel_losses: np.ndarray,
                                 channel_ids: List[str]):
        """
        Add metrics for per-channel losses (one row per sample per channel).
        
        Args:
            epoch: Current epoch number.
            split: Dataset split name.
            sample_ids: List of sample IDs.
            entity_ids: List of entity IDs.
            timestamps: List of timestamps.
            losses: Sample-averaged losses.
            per_channel_losses: Per-channel losses array.
            channel_ids: List of channel ID strings.
        """
        # Create one row per sample per channel
        for sid, eid, t, sample_loss, channel_losses in zip(
            sample_ids, entity_ids, timestamps, losses, per_channel_losses
        ):
            # Handle potential numpy scalars
            ts_val = t.item() if hasattr(t, 'item') else t
            sample_loss_val = sample_loss.item() if hasattr(sample_loss, 'item') else sample_loss
            
            for channel_id, channel_loss in zip(channel_ids, channel_losses):
                channel_loss_val = channel_loss.item() if hasattr(channel_loss, 'item') else channel_loss
                
                self.epoch_metrics.append({
                    "sample_id": str(sid),
                    "epoch": epoch,
                    "split": split,
                    "entity_id": str(eid) if eid is not None else None,
                    "channel_id": str(channel_id),
                    "timestamp": str(ts_val),
                    "loss": float(channel_loss_val),  # Per-channel loss
                    "loss_sample_avg": float(sample_loss_val)  # Sample-averaged loss for reference
                })

    def _add_sample_level_metrics(self, 
                                  epoch: Optional[int],
                                  split: str,
                                  sample_ids: List[str],
                                  entity_ids: List[Optional[str]],
                                  timestamps: List,
                                  losses: np.ndarray):
        """
        Add metrics for sample-level losses only (one row per sample).
        
        Args:
            epoch: Current epoch number.
            split: Dataset split name.
            sample_ids: List of sample IDs.
            entity_ids: List of entity IDs.
            timestamps: List of timestamps.
            losses: Sample-level losses.
        """
        for sid, eid, t, loss in zip(sample_ids, entity_ids, timestamps, losses):
            # Handle potential numpy scalars
            ts_val = t.item() if hasattr(t, 'item') else t
            loss_val = loss.item() if hasattr(loss, 'item') else loss
            
            self.epoch_metrics.append({
                "sample_id": str(sid),
                "epoch": epoch,
                "split": split,
                "entity_id": str(eid) if eid is not None else None,
                "channel_id": None,  # No channel-specific tracking
                "timestamp": str(ts_val),
                "loss": float(loss_val),
                "loss_sample_avg": None  # Not applicable for sample-level only
            })

    def add_batch(self, 
                  epoch: Optional[int], 
                  split: str, 
                  entity_id: Optional[Union[str, List[str]]], 
                  sample_ids: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
                  timestamps: Union[np.ndarray, torch.Tensor, List] = None,
                  losses: Union[np.ndarray, torch.Tensor, List] = None,
                  channel_ids: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
                  per_channel_losses: Optional[Union[np.ndarray, torch.Tensor, List]] = None):
        """
        Add a batch of per-sample metrics, optionally with per-channel losses.
        
        Args:
            epoch: Current epoch number (None for validation/test folds).
            split: Dataset split ('train', 'val', 'test').
            entity_id: Identifier for the entity. Can be None, single string, or list of strings.
            sample_ids: Array-like of sample IDs. If None, will be generated from entity_id and timestamps.
            timestamps: Array-like of timestamps for the samples.
            losses: Array-like of per-sample loss values (averaged over channels). Required if per_channel_losses is None.
            channel_ids: Optional array-like of channel IDs/names. If None and per_channel_losses
                        is provided, uses integer indices 0, 1, 2, ...
            per_channel_losses: Optional array-like of per-channel losses [batch_size, num_channels].
                               If provided, creates one row per sample per channel.
        """
        # Convert inputs to numpy arrays/lists
        losses, timestamps, sample_ids, per_channel_losses = self._convert_to_numpy(
            losses, timestamps, sample_ids, per_channel_losses
        )
        
        # Normalize entity_id into a list
        entity_ids = self._normalize_entity_ids(
            entity_id, sample_ids, per_channel_losses, losses, timestamps
        )
        
        # Handle per-channel losses if provided
        if per_channel_losses is not None:
            # Prepare channel IDs
            channel_ids_list = self._prepare_channel_ids(channel_ids, per_channel_losses)
            
            # Ensure sample_ids and timestamps are available
            sample_ids, timestamps = self._ensure_sample_ids_and_timestamps(
                sample_ids, timestamps, entity_ids
            )
            
            # Get sample-averaged losses if provided, otherwise compute from per-channel
            if losses is None:
                losses = np.mean(per_channel_losses, axis=1)
            
            # Add per-channel metrics
            self._add_per_channel_metrics(
                epoch, split, sample_ids, entity_ids, timestamps,
                losses, per_channel_losses, channel_ids_list
            )
        else:
            # Original behavior: one row per sample (backward compatibility)
            if losses is None:
                raise ValueError("losses must be provided if per_channel_losses is None")
            if timestamps is None:
                raise ValueError("timestamps must be provided if per_channel_losses is None")
            
            # Ensure sample_ids are available
            if sample_ids is None:
                sample_ids = [f"{eid}|{t}|0" for eid, t in zip(entity_ids, timestamps)]
            
            # Convert timestamps to list if needed
            if not isinstance(timestamps, list):
                timestamps = list(timestamps)
            
            # Add sample-level metrics
            self._add_sample_level_metrics(
                epoch, split, sample_ids, entity_ids, timestamps, losses
            )

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
            pl.col("sample_id").cast(pl.Utf8),
            pl.col("epoch").cast(pl.Int32, strict=False),  # Allow null values
            pl.col("split").cast(pl.Categorical),
            pl.col("entity_id").cast(pl.Utf8),
            pl.col("channel_id").cast(pl.Utf8),  # Can be null for sample-level only
            pl.col("timestamp").cast(pl.Utf8),
            pl.col("loss").cast(pl.Float32),
            pl.col("loss_sample_avg").cast(pl.Float32, strict=False)  # Optional, can be null
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
            pl.col("sample_id").cast(pl.Utf8),
            pl.col("epoch").cast(pl.Int32, strict=False),  # Allow null values
            pl.col("split").cast(pl.Categorical),
            pl.col("entity_id").cast(pl.Utf8),
            pl.col("channel_id").cast(pl.Utf8),  # Can be null for sample-level only
            pl.col("timestamp").cast(pl.Utf8),
            pl.col("loss").cast(pl.Float32),
            pl.col("loss_sample_avg").cast(pl.Float32, strict=False)  # Optional, can be null
        ])
        
        save_path = os.path.join(self.output_dir, f"{split}.parquet")
        df.write_parquet(save_path)
        
        # Clear buffer
        self.epoch_metrics = []

    def clear(self):
        """Clear the metrics buffer without saving."""
        self.epoch_metrics = []
