"""
Prompt Builder for Time Series → LLM Prompt conversion.

This module provides ONE input source for LLMEmbedder: converting time series
values into textual prompts. This is used by TimeCMA-style models.

ARCHITECTURE NOTE:
==================

The PromptBuilder is ONE of several possible input adapters for LLMEmbedder:

1. PromptBuilder (this module):
   - Converts time series values → text prompts
   - Used by: TimeCMA and similar models
   - Template-based: configurable prompt formats

2. Raw Text Input (future):
   - Pass pre-existing text directly to LLMEmbedder.embed_texts()
   - Use case: News articles, weather reports, descriptions
   - No PromptBuilder needed

3. Custom Adapters (future extensibility):
   - Create new adapter classes for domain-specific data
   - Any data that can be converted to text strings
   - Compose with LLMEmbedder for inference

The key insight is that LLMEmbedder operates on TEXT - this module handles
the conversion from time series to text. Other input types can bypass this
module entirely and call LLMEmbedder.embed_texts() directly.

TEMPLATE SYSTEM:
================

Templates define how time series data is formatted into prompts:

- TimeCMATemplate: "From [t1] to [t2], the values were v1, v2, ... every {freq}.
                    The total trend value was {trend}"

- SimpleTemplate: "Time series values: v1, v2, v3, ..."

To add new templates:
1. Subclass PromptTemplate
2. Implement the format() method
3. Register in PromptBuilder.TEMPLATES

Example:
    builder = TSPromptBuilder(template_name='timecma_v1')
    prompts = builder.build_flat_prompts(values, timestamps, metadata)
    # Then: embeddings = llm_embedder.embed_texts(prompts)
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable

import numpy as np
import torch


class PromptTemplate:
    """
    Base class for prompt templates.
    
    Templates define how time series data is converted to text prompts
    for LLM processing.
    
    Subclasses should implement the `format` method.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize prompt template.
        
        Args:
            name: Template name
            config: Template configuration dictionary
        """
        self.name = name
        self.config = config
        self.version = config.get('version', '1.0.0')
    
    def format(self, 
               values: np.ndarray,
               timestamps: Optional[np.ndarray] = None,
               channel_idx: int = 0,
               metadata: Optional[Dict] = None) -> str:
        """
        Format time series data into a prompt string.
        
        Args:
            values: Time series values [seq_len] or [seq_len, channels]
            timestamps: Optional timestamp features [seq_len, features]
            channel_idx: Channel index for multi-channel data
            metadata: Optional metadata (dataset info, frequency, etc.)
        
        Returns:
            Formatted prompt string
        """
        raise NotImplementedError("Subclasses must implement format()")


class TimeCMATemplate(PromptTemplate):
    """
    TimeCMA-style prompt template.
    
    Format: "From [t1] to [t2], the values were v1, v2, ..., vn every {freq}.
             The total trend value was {trend}"
    
    This is the prompt format used in the original TimeCMA paper (AAAI 2025).
    """
    
    # Frequency label mapping
    FREQUENCY_MAP = {
        'h': 'hour',
        't': '15 minutes',
        'd': 'day',
        'w': 'week',
        'm': 'month',
        'b': 'business day',
        's': 'second',
        '10min': '10 minutes',
        '15min': '15 minutes',
        '30min': '30 minutes',
    }
    
    def format(self,
               values: np.ndarray,
               timestamps: Optional[np.ndarray] = None,
               channel_idx: int = 0,
               metadata: Optional[Dict] = None) -> str:
        """
        Format time series data into TimeCMA-style prompt.
        
        Args:
            values: Time series values [seq_len] or [seq_len, channels]
            timestamps: Optional timestamp features [seq_len, features]
            channel_idx: Channel index for multi-channel data
            metadata: Optional metadata with 'freq' key
        
        Returns:
            Formatted prompt string
        """
        # Extract values for this channel
        if values.ndim == 2:
            channel_values = values[:, channel_idx]
        else:
            channel_values = values
        
        # Format values based on config
        value_format = self.config.get('value_format', 'integer')
        if value_format == 'integer':
            # Handle NaN values when converting to integer
            values_str = ", ".join([str(int(v)) if not np.isnan(v) else "N/A" for v in channel_values])
        elif value_format == 'float2':
            values_str = ", ".join([f"{v:.2f}" for v in channel_values])
        else:
            values_str = ", ".join([f"{v:.4f}" for v in channel_values])
        
        # Compute trend (sum of differences)
        trend = np.sum(np.diff(channel_values))
        trend_str = f"{trend:.0f}"
        
        # Format timestamps
        if timestamps is not None and self.config.get('include_timestamps', True):
            # Extract first and last timestamps
            # For Time-MMD: timestamps is 1D array [seq_len] of int64 values
            # For standard datasets: timestamps is 2D array [seq_len, features] or 1D array [seq_len] of feature arrays
            start_ts = timestamps[0]
            end_ts = timestamps[-1]
            start_date = self._format_timestamp(start_ts, metadata)
            end_date = self._format_timestamp(end_ts, metadata)
        else:
            start_date = "[start]"
            end_date = "[end]"
        
        # Get frequency label
        freq = metadata.get('freq', 'h') if metadata else 'h'
        freq_label = self.FREQUENCY_MAP.get(freq, freq)
        
        # Build prompt
        prompt = (
            f"From {start_date} to {end_date}, "
            f"the values were {values_str} every {freq_label}. "
            f"The total trend value was {trend_str}"
        )
        
        return prompt
    
    def _format_timestamp(self, ts_input, metadata: Optional[Dict]) -> str:
        """
        Format timestamp into readable date string.
        
        Handles timestamps from all dataset types (Time-MMD, fidel-ts, TTC):
        - All datasets use int64 timestamps in YYYYMMDDHHMMSS format
        - When accessing timestamps[0] from a 1D array, we get a scalar numpy.int64
        
        Also supports (legacy/optional) timestamp feature arrays for compatibility.
        
        Args:
            ts_input: Either scalar int64 timestamp (numpy.int64) or array of timestamp features
            metadata: Optional metadata dict
        """
        # Convert to numpy array to check dimensions
        ts_array = np.asarray(ts_input)
        
        # Handle scalar or 0-d array (numpy.int64 from timestamps[0])
        # np.isscalar() returns False for numpy scalars, so check ndim instead
        if ts_array.ndim == 0:
            ts_int = int(ts_array.item())  # Extract Python int from numpy scalar
            # Parse YYYYMMDDHHMMSS format
            ts_str = str(ts_int).zfill(14)  # Ensure 14 digits
            if len(ts_str) >= 12:
                month = int(ts_str[4:6])
                day = int(ts_str[6:8])
                hour = int(ts_str[8:10]) if len(ts_str) >= 10 else 0
                minute = int(ts_str[10:12]) if len(ts_str) >= 12 else 0
                
                # Clamp values
                month = max(1, min(12, month))
                day = max(1, min(31, day))
                hour = max(0, min(23, hour))
                minute = max(0, min(59, minute))
                
                return f"{day:02d}/{month:02d} {hour:02d}:{minute:02d}"
            return "[unknown]"
        
        # Handle array of timestamp features (legacy/optional format)
        if ts_array.ndim > 0 and len(ts_array) >= 4:
            # Assuming: [month, day, weekday, hour, ...]
            month = int(ts_array[0] * 12) if ts_array[0] <= 1 else int(ts_array[0])
            day = int(ts_array[1] * 31) if ts_array[1] <= 1 else int(ts_array[1])
            hour = int(ts_array[3] * 24) if len(ts_array) > 3 and ts_array[3] <= 1 else int(ts_array[3]) if len(ts_array) > 3 else 0
            
            # Clamp values
            month = max(1, min(12, month))
            day = max(1, min(31, day))
            hour = max(0, min(23, hour))
            
            return f"{day:02d}/{month:02d} {hour:02d}:00"
        return "[unknown]"


class SimpleTemplate(PromptTemplate):
    """
    Simple prompt template with minimal formatting.
    
    Format: "Time series values: v1, v2, ..., vn"
    
    Useful for testing or when timestamps are not available.
    """
    
    def format(self,
               values: np.ndarray,
               timestamps: Optional[np.ndarray] = None,
               channel_idx: int = 0,
               metadata: Optional[Dict] = None) -> str:
        """Format time series as simple comma-separated values."""
        if values.ndim == 2:
            channel_values = values[:, channel_idx]
        else:
            channel_values = values
        
        value_format = self.config.get('value_format', 'float2')
        if value_format == 'integer':
            # Handle NaN values when converting to integer
            import numpy as np
            values_str = ", ".join([str(int(v)) if not np.isnan(v) else "N/A" for v in channel_values])
        else:
            values_str = ", ".join([f"{v:.2f}" for v in channel_values])
        
        return f"Time series values: {values_str}"


class MMTSFlibTemplate(PromptTemplate):
    """
    MM-TSFlib prompt template for time series → text conversion.
    
    Format: "<|start_prompt|Make predictions about the future based on the 
             following information: {values_description}<|<end_prompt>|>"
    
    This is the prompt format used in the original MM-TSFlib paper (Time-MMD).
    The template wraps formatted time series data with special tokens that
    help the LLM understand the prediction task context.
    
    Reference:
        MM-TSFlib: https://github.com/AdityaLab/MM-TSFlib
        Time-MMD: https://github.com/AdityaLab/Time-MMD
    """
    
    # Frequency label mapping (same as TimeCMA for consistency)
    FREQUENCY_MAP = {
        'h': 'hour',
        't': '15 minutes',
        'd': 'day',
        'w': 'week',
        'm': 'month',
        'b': 'business day',
        's': 'second',
        '10min': '10 minutes',
        '15min': '15 minutes',
        '30min': '30 minutes',
    }
    
    def format(self,
               values: np.ndarray,
               timestamps: Optional[np.ndarray] = None,
               channel_idx: int = 0,
               metadata: Optional[Dict] = None) -> str:
        """
        Format time series data into MM-TSFlib-style prompt.
        
        Args:
            values: Time series values [seq_len] or [seq_len, channels]
            timestamps: Optional timestamp features [seq_len, features]
            channel_idx: Channel index for multi-channel data
            metadata: Optional metadata with 'freq', 'dataset_name' keys
        
        Returns:
            Formatted prompt string with MM-TSFlib special tokens
        """
        # Extract values for this channel
        if values.ndim == 2:
            channel_values = values[:, channel_idx]
        else:
            channel_values = values
        
        # Format values based on config
        value_format = self.config.get('value_format', 'float2')
        if value_format == 'integer':
            # Handle NaN values when converting to integer
            values_str = ", ".join([str(int(v)) if not np.isnan(v) else "N/A" for v in channel_values])
        elif value_format == 'float2':
            values_str = ", ".join([f"{v:.2f}" for v in channel_values])
        else:
            values_str = ", ".join([f"{v:.4f}" for v in channel_values])
        
        # Get frequency label
        freq = metadata.get('freq', 'h') if metadata else 'h'
        freq_label = self.FREQUENCY_MAP.get(freq, freq)
        
        # Get dataset context if available
        dataset_name = metadata.get('dataset_name', '') if metadata else ''
        
        # Build the text info description
        if timestamps is not None and self.config.get('include_timestamps', True):
            start_ts = timestamps[0]
            end_ts = timestamps[-1]
            start_date = self._format_timestamp(start_ts, metadata)
            end_date = self._format_timestamp(end_ts, metadata)
            
            if dataset_name:
                text_info = (
                    f"Dataset: {dataset_name}. "
                    f"From {start_date} to {end_date}, "
                    f"the observed values were [{values_str}] recorded every {freq_label}."
                )
            else:
                text_info = (
                    f"From {start_date} to {end_date}, "
                    f"the observed values were [{values_str}] recorded every {freq_label}."
                )
        else:
            if dataset_name:
                text_info = (
                    f"Dataset: {dataset_name}. "
                    f"The observed values were [{values_str}] recorded every {freq_label}."
                )
            else:
                text_info = (
                    f"The observed values were [{values_str}] recorded every {freq_label}."
                )
        
        # Wrap with MM-TSFlib special tokens
        prompt = (
            f"<|start_prompt|>Make predictions about the future based on "
            f"the following information: {text_info}<|end_prompt|>"
        )
        
        return prompt
    
    def _format_timestamp(self, ts_input, metadata: Optional[Dict]) -> str:
        """
        Format timestamp into readable date string.
        
        Handles timestamps from all dataset types (Time-MMD, fidel-ts, TTC):
        - All datasets use int64 timestamps in YYYYMMDDHHMMSS format
        - When accessing timestamps[0] from a 1D array, we get a scalar numpy.int64
        
        Also supports (legacy/optional) timestamp feature arrays for compatibility.
        
        Args:
            ts_input: Either scalar int64 timestamp (numpy.int64) or array of timestamp features
            metadata: Optional metadata dict
        """
        # Convert to numpy array to check dimensions
        ts_array = np.asarray(ts_input)
        
        # Handle scalar or 0-d array (numpy.int64 from timestamps[0])
        if ts_array.ndim == 0:
            ts_int = int(ts_array.item())  # Extract Python int from numpy scalar
            # Parse YYYYMMDDHHMMSS format
            ts_str = str(ts_int).zfill(14)  # Ensure 14 digits
            if len(ts_str) >= 12:
                month = int(ts_str[4:6])
                day = int(ts_str[6:8])
                hour = int(ts_str[8:10]) if len(ts_str) >= 10 else 0
                minute = int(ts_str[10:12]) if len(ts_str) >= 12 else 0
                
                # Clamp values
                month = max(1, min(12, month))
                day = max(1, min(31, day))
                hour = max(0, min(23, hour))
                minute = max(0, min(59, minute))
                
                return f"{day:02d}/{month:02d} {hour:02d}:{minute:02d}"
            return "[unknown]"
        
        # Handle array of timestamp features (legacy/optional format)
        if ts_array.ndim > 0 and len(ts_array) >= 4:
            # Assuming: [month, day, weekday, hour, ...]
            month = int(ts_array[0] * 12) if ts_array[0] <= 1 else int(ts_array[0])
            day = int(ts_array[1] * 31) if ts_array[1] <= 1 else int(ts_array[1])
            hour = int(ts_array[3] * 24) if len(ts_array) > 3 and ts_array[3] <= 1 else int(ts_array[3]) if len(ts_array) > 3 else 0
            
            # Clamp values
            month = max(1, min(12, month))
            day = max(1, min(31, day))
            hour = max(0, min(23, hour))
            
            return f"{day:02d}/{month:02d} {hour:02d}:00"
        return "[unknown]"


class TSPromptBuilder:
    """
    Time Series Prompt Builder - converts time series data to text prompts.
    
    This is ONE of several possible input adapters for LLMEmbedder.
    It specifically handles TIME SERIES → TEXT conversion.
    
    The "TS" prefix emphasizes this is for time series input. Other prompt
    builders could be created for different input types (e.g., news, reports).
    
    Manages templates and provides batch prompt generation with
    efficient vectorized processing.
    
    Example:
        builder = TSPromptBuilder(template_name='timecma_v1')
        prompts = builder.build_flat_prompts(values, timestamps, metadata)
        # Then: embeddings = llm_embedder.embed_texts(prompts)
    """
    
    # Available templates
    TEMPLATES = {
        'timecma_v1': TimeCMATemplate,
        'simple': SimpleTemplate,
        'mmtsflib_v1': MMTSFlibTemplate,
    }
    
    def __init__(self, 
                 template_name: str = 'timecma_v1',
                 template_config: Optional[Dict] = None,
                 prompt_cache_dir: Optional[str] = None):
        """
        Initialize prompt builder.
        
        Args:
            template_name: Name of prompt template to use
            template_config: Configuration for the template
            prompt_cache_dir: Directory to cache generated prompts (optional)
        """
        self.template_name = template_name
        self.template_config = template_config or {}
        self.prompt_cache_dir = Path(prompt_cache_dir) if prompt_cache_dir else None
        
        # Initialize template
        template_class = self.TEMPLATES.get(template_name)
        if template_class is None:
            raise ValueError(f"Unknown template: {template_name}. "
                           f"Available: {list(self.TEMPLATES.keys())}")
        
        self.template = template_class(template_name, self.template_config)
    
    def build_prompt(self,
                     values: np.ndarray,
                     timestamps: Optional[np.ndarray] = None,
                     channel_idx: int = 0,
                     metadata: Optional[Dict] = None) -> str:
        """
        Build a single prompt from time series data.
        
        Args:
            values: Time series values [seq_len] or [seq_len, channels]
            timestamps: Optional timestamp features [seq_len, features]
            channel_idx: Channel index for multi-channel data
            metadata: Optional metadata dictionary
        
        Returns:
            Formatted prompt string
        """
        return self.template.format(values, timestamps, channel_idx, metadata)
    
    def build_batch_prompts(self,
                            batch_values: np.ndarray,
                            batch_timestamps: Optional[np.ndarray] = None,
                            metadata: Optional[Dict] = None) -> List[List[str]]:
        """
        Build prompts for a batch of time series.
        
        Args:
            batch_values: [B, seq_len, channels]
            batch_timestamps: [B, seq_len, features] (optional)
            metadata: Shared metadata for all samples
        
        Returns:
            List of lists: [[prompts for each channel] for each sample]
            Shape: [B][C] where B is batch size and C is number of channels
        """
        B, L, C = batch_values.shape
        all_prompts = []
        
        for b in range(B):
            sample_prompts = []
            for c in range(C):
                ts = batch_timestamps[b] if batch_timestamps is not None else None
                prompt = self.build_prompt(
                    batch_values[b], ts, c, metadata
                )
                sample_prompts.append(prompt)
            all_prompts.append(sample_prompts)
        
        return all_prompts
    
    def build_flat_prompts(self,
                           batch_values: np.ndarray,
                           batch_timestamps: Optional[np.ndarray] = None,
                           metadata: Optional[Dict] = None) -> List[str]:
        """
        Build prompts for a batch, flattened to a single list.
        
        This is optimized for batched LLM inference where we want to
        process all (sample, channel) pairs in a single batch.
        
        WARNING: For large datasets, this can cause memory explosion!
        Use iter_batch_prompts() for memory-efficient processing.
        
        Args:
            batch_values: [B, seq_len, channels]
            batch_timestamps: [B, seq_len, features] (optional)
            metadata: Shared metadata for all samples
        
        Returns:
            Flat list of prompts with length B * C
            Order: [sample0_ch0, sample0_ch1, ..., sample1_ch0, ...]
        """
        B, L, C = batch_values.shape
        all_prompts = []
        
        for b in range(B):
            for c in range(C):
                ts = batch_timestamps[b] if batch_timestamps is not None else None
                prompt = self.build_prompt(batch_values[b], ts, c, metadata)
                all_prompts.append(prompt)
        
        return all_prompts
    
    def iter_batch_prompts(
        self,
        batch_values: np.ndarray,
        batch_timestamps: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
        prompt_batch_size: int = 64,
    ):
        """
        Lazily generate prompts in batches (memory-efficient).
        
        Instead of generating ALL N*C prompts at once (which can use 20+ GB
        for large datasets), this generator yields prompts in small batches.
        
        This is the preferred method for large datasets like NYC traffic speed.
        
        Args:
            batch_values: [B, seq_len, channels] - data values
            batch_timestamps: [B, seq_len, features] (optional) - timestamps
            metadata: Shared metadata for all samples
            prompt_batch_size: Number of prompts to yield per batch
        
        Yields:
            Tuple of (prompts, sample_indices, channel_indices):
                - prompts: List[str] of length <= prompt_batch_size
                - sample_indices: List[int] sample indices for each prompt
                - channel_indices: List[int] channel indices for each prompt
        
        Example:
            for prompts, sample_idxs, channel_idxs in builder.iter_batch_prompts(
                values, timestamps, metadata, prompt_batch_size=64
            ):
                embeddings = llm.embed_texts(prompts)
                # embeddings[i] corresponds to sample_idxs[i], channel_idxs[i]
        """
        B, L, C = batch_values.shape
        
        batch_prompts = []
        batch_sample_indices = []
        batch_channel_indices = []
        
        for b in range(B):
            for c in range(C):
                # Generate prompt for this (sample, channel) pair
                ts = batch_timestamps[b] if batch_timestamps is not None else None
                prompt = self.build_prompt(batch_values[b], ts, c, metadata)
                
                batch_prompts.append(prompt)
                batch_sample_indices.append(b)
                batch_channel_indices.append(c)
                
                # Yield when batch is full
                if len(batch_prompts) >= prompt_batch_size:
                    yield batch_prompts, batch_sample_indices, batch_channel_indices
                    batch_prompts = []
                    batch_sample_indices = []
                    batch_channel_indices = []
        
        # Yield remaining prompts
        if batch_prompts:
            yield batch_prompts, batch_sample_indices, batch_channel_indices
    
    def count_prompts(self, batch_values: np.ndarray) -> int:
        """
        Count total prompts that would be generated for given data.
        
        Useful for progress bar initialization without generating prompts.
        
        Args:
            batch_values: [B, seq_len, channels]
        
        Returns:
            Total number of prompts (B * C)
        """
        B, L, C = batch_values.shape
        return B * C
    
    def get_example_prompt(self,
                           values: np.ndarray,
                           timestamps: Optional[np.ndarray] = None,
                           metadata: Optional[Dict] = None) -> str:
        """
        Generate an example prompt for documentation/metadata.
        
        Uses the first channel of the first sample.
        """
        if values.ndim == 3:
            return self.build_prompt(values[0], 
                                    timestamps[0] if timestamps is not None else None,
                                    0, metadata)
        return self.build_prompt(values, timestamps, 0, metadata)
    
    def get_template_info(self) -> Dict[str, Any]:
        """
        Get information about the current template.
        
        Returns:
            Dictionary with template name, version, and config
        """
        return {
            'template_name': self.template_name,
            'template_version': self.template.version,
            'template_config': self.template_config,
        }

