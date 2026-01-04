"""
Time-LLM: Time Series Forecasting by Reprogramming Large Language Models.

This module implements the Time-LLM architecture from Jin et al. (ICLR 2024).
Time-LLM reprograms frozen LLMs to process time series data by:

1. Patching time series into embeddings (PatchEmbedding)
2. Cross-attending patches with LLM word embeddings (ReprogrammingLayer)
3. Passing reprogrammed embeddings through frozen LLM
4. Projecting LLM output to predictions (FlattenHead)

Key Design Decisions:
- LLM backbone is FROZEN (requires_grad=False)
- Only train: PatchEmbedding, ReprogrammingLayer, MappingLayer, OutputProjection
- Dynamic prompts with per-batch statistics generated in forward pass
- Uses existing LLMRegistry for model loading (no duplication)

Layer components are in layers/time_llm/ for consistency with codebase structure.

Integration with fidel-ts:
- Uses LLMRegistry from embedder/ for model loading
- Uses Data_Provider for data loading
- Follows exp/model_specific/ patterns for training

Reference:
    Jin et al., "Time-LLM: Time Series Forecasting by Reprogramming 
    Large Language Models" (ICLR 2024)
"""

from typing import Optional

import torch
import torch.nn as nn

# Import layer components from layers/time_llm/
from layers.time_llm import (
    Normalize,
    PatchEmbedding,
    ReprogrammingLayer,
    DynamicPromptBuilder,
)


class FlattenHead(nn.Module):
    """
    Output projection from LLM hidden space to predictions.
    
    Flattens the patch dimension and projects to prediction length.
    
    Args:
        n_vars: Number of input variables/channels
        head_nf: Number of features in flattened representation (d_ff * num_patches)
        pred_len: Prediction length
        dropout: Dropout rate
    """
    
    def __init__(
        self, 
        n_vars: int, 
        head_nf: int, 
        pred_len: int, 
        dropout: float = 0.1
    ):
        """
        Initialize FlattenHead.
        
        Args:
            n_vars: Number of input variables/channels
            head_nf: Number of features in flattened representation
            pred_len: Prediction length
            dropout: Dropout rate
        """
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(head_nf, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project from LLM hidden space to predictions.
        
        Args:
            x: Hidden states [B, n_vars, d_ff, num_patches]
        
        Returns:
            Predictions [B, pred_len, n_vars]
        """
        # Flatten: [B, n_vars, d_ff, num_patches] -> [B, n_vars, d_ff * num_patches]
        x = self.flatten(x)
        
        # Project: [B, n_vars, d_ff * num_patches] -> [B, n_vars, pred_len]
        x = self.linear(x)
        x = self.dropout(x)
        
        # Transpose: [B, n_vars, pred_len] -> [B, pred_len, n_vars]
        return x.permute(0, 2, 1)


class TimeLLM(nn.Module):
    """
    Time-LLM: Time Series Forecasting by Reprogramming Large Language Models.
    
    This model reprograms a frozen LLM to process time series by:
    1. Patching time series into embeddings
    2. Cross-attending patches with LLM word embeddings (reprogramming)
    3. Passing reprogrammed embeddings through frozen LLM
    4. Projecting LLM output to predictions
    
    Key Design:
    - LLM is FROZEN (no gradients)
    - Only patch embedding, reprogramming, and output projection are trained
    - Dynamic prompts with per-batch statistics
    
    Args:
        configs: Configuration object with the following attributes:
            - llm_model: HuggingFace model name or alias ('LLAMA', 'GPT2', 'QWEN')
            - llm_layers: Number of LLM layers (not currently used)
            - seq_len: Input sequence length
            - pred_len: Prediction sequence length
            - patch_len: Length of each time series patch
            - stride: Stride between patches
            - d_model: Patch embedding dimension
            - d_ff: Feed-forward dimension
            - n_heads: Number of attention heads for reprogramming
            - enc_in: Number of input channels/variables
            - dropout: Dropout rate
            - prompt_domain: If True, use dataset_description in prompts
            - dataset_description: Domain-specific dataset description
            - cache_dir: Directory for LLM model weights cache
            - device: Target device
            - quantization: Quantization mode ('4bit', '8bit', or None)
    
    Example:
        >>> from utils.tools import dotdict
        >>> configs = dotdict({
        ...     'llm_model': 'gpt2',
        ...     'seq_len': 96,
        ...     'pred_len': 96,
        ...     'enc_in': 7,
        ... })
        >>> model = TimeLLM(configs)
        >>> x = torch.randn(2, 96, 7).to('cuda:0')
        >>> out = model(x)
        >>> out.shape  # [2, 96, 7]
    """
    
    # Mapping from Time-LLM aliases to HuggingFace model names
    MODEL_MAP = {
        'LLAMA': 'meta-llama/Llama-3.1-8B-Instruct',
        'LLAMA-7B': 'meta-llama/Llama-3.1-8B-Instruct',
        'LLAMA-70B': 'meta-llama/Llama-3.1-70B-Instruct',
        'GPT2': 'gpt2',
        'GPT2-MEDIUM': 'gpt2-medium',
        'GPT2-LARGE': 'gpt2-large',
        'GPT2-XL': 'gpt2-xl',
        'BERT': 'google-bert/bert-base-uncased',
        'QWEN': 'Qwen/Qwen2.5-7B-Instruct',
        'QWEN-7B': 'Qwen/Qwen2.5-7B-Instruct',
        'QWEN-14B': 'Qwen/Qwen2.5-14B-Instruct',
        'QWEN-72B': 'Qwen/Qwen2.5-72B-Instruct',
    }
    
    def __init__(self, configs):
        """
        Initialize Time-LLM model.
        
        Args:
            configs: Configuration object (dotdict) with model parameters
        """
        super().__init__()
        
        # =====================================================================
        # Extract configuration with defaults
        # =====================================================================
        
        # Required parameters (set by model_init from training config)
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        enc_in = configs.enc_in
        
        # LLM backbone configuration with defaults
        llm_model = getattr(configs, 'llm_model', 'gpt2')
        llm_layers = getattr(configs, 'llm_layers', 6)
        cache_dir = getattr(configs, 'cache_dir', './LLM_cache/')
        quantization = getattr(configs, 'quantization', None)
        
        # Patch configuration with defaults
        patch_len = getattr(configs, 'patch_len', 16)
        stride = getattr(configs, 'stride', 8)
        
        # Architecture configuration with defaults
        d_model = getattr(configs, 'd_model', 16)
        d_ff = getattr(configs, 'd_ff', 32)
        n_heads = getattr(configs, 'n_heads', 8)
        dropout = getattr(configs, 'dropout', 0.1)
        
        # Prompt configuration with defaults
        prompt_domain = getattr(configs, 'prompt_domain', False)
        dataset_description = getattr(configs, 'dataset_description', '')
        
        # Device configuration
        device = getattr(configs, 'device', 'cuda:0')
        if getattr(configs, 'gpu', None) is not None:
            device = f'cuda:{configs.gpu}'
        
        # Store configuration
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_ff = d_ff
        self.patch_len = patch_len
        self.stride = stride
        self.top_k = 5  # Number of autocorrelation lags
        self.enc_in = enc_in
        self.device_str = device
        
        # Load LLM using existing registry
        self._load_llm(llm_model, cache_dir, device, quantization)
        
        # Get LLM hidden dimension from config
        self.d_llm = self.llm_model.config.hidden_size
        
        # Freeze LLM weights - this is critical for Time-LLM
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
        # Dataset description for dynamic prompts
        self.description = dataset_description if prompt_domain else (
            "The Electricity Transformer Temperature (ETT) is a crucial "
            "indicator in the electric power long-term deployment."
        )
        
        # Patch embedding layer
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, dropout)
        
        # Word embedding mapping layer
        # Maps full vocabulary to reduced set of tokens for efficiency
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000  # Reduced vocabulary size
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        
        # Reprogramming layer for TS-to-LLM space mapping
        self.reprogramming_layer = ReprogrammingLayer(
            d_model=d_model,
            n_heads=n_heads,
            d_keys=d_ff,
            d_llm=self.d_llm,
            attention_dropout=dropout
        )
        
        # Output projection
        # Number of patches: (seq_len - patch_len) / stride + 2 (with padding)
        self.patch_nums = int((seq_len - patch_len) / stride + 2)
        self.head_nf = d_ff * self.patch_nums
        self.output_projection = FlattenHead(enc_in, self.head_nf, pred_len, dropout)
        
        # RevIN normalization layers
        self.normalize_layers = Normalize(enc_in, affine=False)
        
        self.dropout = nn.Dropout(dropout)

    def _load_llm(
        self, 
        llm_model: str, 
        cache_dir: str, 
        device: str,
        quantization: Optional[str]
    ):
        """
        Load LLM model using existing LLMRegistry.
        
        Maps Time-LLM aliases to HuggingFace model names and uses
        the shared LLMRegistry for efficient model loading.
        
        Args:
            llm_model: Model name or alias
            cache_dir: Cache directory for model weights
            device: Target device
            quantization: Quantization mode ('4bit', '8bit', or None)
        """
        # Import here to avoid circular imports
        from embedder.llm_registry import LLMRegistry
        
        # Map alias to HuggingFace model name
        model_name = self.MODEL_MAP.get(llm_model.upper(), llm_model)
        
        # Use existing LLMRegistry for model loading
        self.llm_model, _ = LLMRegistry.get_model(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            quantization=quantization,
        )
        
        # Get tokenizer from registry
        self.tokenizer = LLMRegistry.get_tokenizer(
            model_name=model_name,
            cache_dir=cache_dir,
        )
        
        # Ensure pad token is set (required for batched tokenization)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def forward(
        self, 
        x_enc: torch.Tensor = None,
        x_mark_enc: torch.Tensor = None, 
        x_dec: torch.Tensor = None, 
        x_mark_dec: torch.Tensor = None, 
        mask: torch.Tensor = None,
        x: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_enc: Input time series [B, T, C] (primary input)
            x_mark_enc: Time features for encoder (optional, not used)
            x_dec: Decoder input (not used in Time-LLM)
            x_mark_dec: Time features for decoder (not used)
            mask: Attention mask (optional, not used)
            x: Alternative input name (for compatibility with some training loops)
        
        Returns:
            Predictions [B, pred_len, C]
        """
        # Support both x_enc and x as input names
        if x_enc is None and x is not None:
            x_enc = x
        
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]

    def forecast(
        self, 
        x_enc: torch.Tensor, 
        x_mark_enc: torch.Tensor, 
        x_dec: torch.Tensor, 
        x_mark_dec: torch.Tensor
    ) -> torch.Tensor:
        """
        Forecast future values.
        
        This is the main forecasting logic implementing the Time-LLM pipeline:
        1. Normalize input using RevIN
        2. Compute dynamic prompt statistics
        3. Patch and embed time series
        4. Reprogram patches to LLM space
        5. Pass through frozen LLM
        6. Project to predictions
        7. Denormalize output
        
        Args:
            x_enc: Input time series [B, T, C]
            x_mark_enc: Time features (not used in reprogramming)
            x_dec: Decoder input (not used)
            x_mark_dec: Decoder time features (not used)
        
        Returns:
            Forecasted values [B, pred_len, C]
        """
        # Step 1: Normalize input using RevIN
        x_enc = self.normalize_layers(x_enc, 'norm')
        
        B, T, N = x_enc.size()
        
        # Step 2: Reshape for per-channel processing
        # [B, T, N] -> [B*N, T, 1]
        x_enc_flat = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        
        # Step 3: Calculate dynamic prompt statistics
        # Ensure computations are on the correct device
        x_enc_flat = x_enc_flat.to(x_enc.device)
        lags = DynamicPromptBuilder.calculate_lags(x_enc_flat, self.top_k)
        prompts = DynamicPromptBuilder.build_prompts(
            x_enc_flat, self.description, self.pred_len, self.seq_len, lags
        )
        
        # Step 4: Tokenize prompts and get prompt embeddings
        # Tokenize on CPU (faster for tokenizer) then move to device
        # Use reasonable max_length (prompts are typically ~100-150 tokens)
        with torch.no_grad():  # Tokenization doesn't need gradients
            prompt_tokens = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512  # Reduced from 2048 - prompts are much shorter
            ).input_ids.to(x_enc.device)
        
        # Get prompt embeddings from LLM's embedding layer
        # Note: Even though LLM is frozen, we need gradients for the reprogramming layer
        # that attends to these embeddings, so we don't use no_grad here
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_tokens)
        
        # Step 5: Map word embeddings to reduced vocabulary
        # word_embeddings: [vocab_size, d_llm]
        # mapping_layer: [vocab_size, num_tokens]
        # source_embeddings: [num_tokens, d_llm]
        # Note: Convert word_embeddings to float32 for mapping_layer (trainable)
        source_embeddings = self.mapping_layer(
            self.word_embeddings.permute(1, 0).float()
        ).permute(1, 0)
        
        # Step 6: Patch embedding
        # x_enc: [B, T, N] -> [B, N, T] for patching
        x_enc_perm = x_enc.permute(0, 2, 1).contiguous()
        
        # Patch embedding (in float32 for trainable layers)
        enc_out, n_vars = self.patch_embedding(x_enc_perm.float())
        
        # Step 7: Reprogramming - map TS patches to LLM space
        # enc_out: [B*N, num_patches, d_model] -> [B*N, num_patches, d_llm]
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        
        # Step 8: Concatenate prompt embeddings with reprogrammed patches
        # prompt_embeddings: [B*N, prompt_len, d_llm]
        # enc_out: [B*N, num_patches, d_llm]
        # llm_input: [B*N, prompt_len + num_patches, d_llm]
        # Convert enc_out to match LLM dtype (prompt_embeddings dtype)
        llm_dtype = prompt_embeddings.dtype
        llm_input = torch.cat([prompt_embeddings, enc_out.to(llm_dtype)], dim=1)
        
        # Step 9: Pass through frozen LLM
        # Get last hidden state from LLM
        dec_out = self.llm_model(inputs_embeds=llm_input).last_hidden_state
        
        # Step 10: Extract relevant dimensions for output projection
        # Only keep first d_ff dimensions, convert back to float32 for output projection
        dec_out = dec_out[:, :, :self.d_ff].float()
        
        # Step 11: Reshape for output projection
        # [B*N, seq_len+patches, d_ff] -> [B, N, d_ff, patches]
        dec_out = dec_out.reshape(-1, n_vars, dec_out.shape[-2], dec_out.shape[-1])
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        
        # Step 12: Output projection - get predictions
        # Only use the last patch_nums positions (from reprogrammed patches, not prompts)
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        
        # Step 13: Denormalize output back to original scale
        dec_out = self.normalize_layers(dec_out, 'denorm')
        
        return dec_out
    
    def get_trainable_params(self) -> int:
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_frozen_params(self) -> int:
        """Return count of frozen parameters (LLM backbone)."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)


# Model class expected by model_init
Model = TimeLLM

__all__ = ['Model', 'TimeLLM', 'FlattenHead']
