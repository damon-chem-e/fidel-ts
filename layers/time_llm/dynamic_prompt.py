"""
Dynamic Prompt Generation for Time-LLM.

This module generates prompts ON-THE-FLY during the forward pass with
per-batch statistics. This is fundamentally different from the offline
prompt generation in embedder/prompt_builder.py (used by TimeCMA).

Key Differences from TSPromptBuilder:
    - TSPromptBuilder: Precomputes prompts offline for embedding caching
    - DynamicPromptBuilder: Generates prompts per-batch during training
    
Dynamic prompts include:
    - Dataset description
    - Task description (forecasting horizon)
    - Input statistics: min, max, median, trend direction
    - Top-k autocorrelation lags (computed via FFT)
"""

from typing import List
import torch


class DynamicPromptBuilder:
    """
    Generate prompts with time series statistics during forward pass.
    
    Unlike TSPromptBuilder which is for offline precomputation, this class
    generates prompts on-the-fly during training with dynamic per-batch
    statistics.
    
    The prompts follow the Time-LLM paper format:
        "<|start_prompt|>Dataset description: {desc}
        Task description: forecast the next {pred_len} steps given the 
        previous {seq_len} steps information;
        Input statistics: min value {min}, max value {max}, 
        median value {med}, the trend of input is {upward/downward},
        top 5 lags are: {lags}<|end_prompt|>"
    
    Note:
        All methods are static since no state needs to be maintained.
    """
    
    @staticmethod
    def build_prompts(
        x_enc: torch.Tensor,
        description: str,
        pred_len: int,
        seq_len: int,
        lags: torch.Tensor,
    ) -> List[str]:
        """
        Build prompts with dynamic statistics for each sample.
        
        Computes per-sample statistics (min, max, median, trend) and
        formats them into prompt strings for the LLM.
        
        Args:
            x_enc: Flattened time series [B*N, T, 1] where B=batch, N=channels
            description: Dataset description string for context
            pred_len: Prediction/forecast length
            seq_len: Input sequence length
            lags: Top-k autocorrelation lags [B*N, top_k]
        
        Returns:
            List of prompt strings, one per sample-channel pair (length B*N)
        
        Example:
            >>> x = torch.randn(14, 96, 1)  # 2 batches * 7 channels
            >>> lags = DynamicPromptBuilder.calculate_lags(x, top_k=5)
            >>> prompts = DynamicPromptBuilder.build_prompts(
            ...     x, "ETT dataset", pred_len=96, seq_len=96, lags=lags
            ... )
            >>> len(prompts)  # 14 prompts
        """
        # Compute statistics along time dimension (dim=1)
        # Each statistic has shape [B*N, 1]
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        
        # Compute trend as sum of differences (positive = upward)
        # diff: [B*N, T-1, 1], sum: [B*N, 1]
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompts = []
        for b in range(x_enc.shape[0]):
            # Determine trend direction
            trend_dir = 'upward' if trends[b].item() > 0 else 'downward'
            
            # Format lag values as list
            lag_values = lags[b].tolist()
            
            # Build prompt with Time-LLM format
            prompt = (
                f"<|start_prompt|>Dataset description: {description} "
                f"Task description: forecast the next {pred_len} steps "
                f"given the previous {seq_len} steps information; "
                "Input statistics: "
                f"min value {min_values[b].item():.4f}, "
                f"max value {max_values[b].item():.4f}, "
                f"median value {medians[b].item():.4f}, "
                f"the trend of input is {trend_dir}, "
                f"top 5 lags are: {lag_values}<|end_prompt|>"
            )
            prompts.append(prompt)
        
        return prompts
    
    @staticmethod
    def calculate_lags(x_enc: torch.Tensor, top_k: int = 5) -> torch.Tensor:
        """
        Compute top-k autocorrelation lags via FFT.
        
        Uses FFT-based autocorrelation for efficiency:
        1. Compute FFT of signal
        2. Multiply by conjugate (power spectrum)
        3. Inverse FFT to get autocorrelation
        4. Find top-k peaks (excluding lag 0)
        
        Args:
            x_enc: Time series tensor [B*N, T, 1]
            top_k: Number of top lags to return
        
        Returns:
            Top-k lag indices [B*N, top_k] (1-indexed, lag 0 excluded)
        
        Example:
            >>> x = torch.randn(14, 96, 1)
            >>> lags = DynamicPromptBuilder.calculate_lags(x, top_k=5)
            >>> lags.shape  # [14, 5]
        """
        # Transpose for FFT: [B*N, T, 1] -> [B*N, 1, T]
        x = x_enc.permute(0, 2, 1)
        
        # Compute FFT
        q_fft = torch.fft.rfft(x, dim=-1)
        k_fft = torch.fft.rfft(x, dim=-1)
        
        # Power spectrum (autocorrelation in frequency domain)
        # This is equivalent to correlation theorem: F(corr) = F(x) * conj(F(x))
        res = q_fft * torch.conj(k_fft)
        
        # Inverse FFT to get autocorrelation in time domain
        corr = torch.fft.irfft(res, dim=-1)
        
        # Average over channel dimension (if multi-channel)
        # [B*N, 1, T] -> [B*N, T]
        mean_value = torch.mean(corr, dim=1)
        
        # Get top-k lags, excluding lag 0 (which is always highest)
        # [:, 1:] excludes lag 0
        _, lags = torch.topk(mean_value[:, 1:], top_k, dim=-1)
        
        # Adjust indices since we excluded lag 0 (add 1 to get original indices)
        return lags + 1