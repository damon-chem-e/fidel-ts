"""
RetNet Configuration for LeRet model.

Simplified configuration without MoE (Mixture of Experts) support.
Only includes parameters needed for time series forecasting.
"""


class RetNetConfig:
    """
    Configuration class for RetNet architecture.
    
    This is a simplified version without MoE support, tailored for
    time series forecasting in the fidel-ts framework.
    
    Args:
        decoder_embed_dim: Hidden dimension of the model (default: 128)
        decoder_value_embed_dim: Value dimension, typically same as embed_dim (default: 128)
        decoder_retention_heads: Number of retention heads (default: 8)
        decoder_ffn_embed_dim: FFN intermediate dimension (default: 256)
        decoder_layers: Number of retention layers (default: 3)
        decoder_normalize_before: Pre-norm vs post-norm (default: True)
        activation_fn: Activation function name (default: 'gelu')
        dropout: Dropout rate (default: 0.0)
        drop_path_rate: Drop path rate for stochastic depth (default: 0.0)
        activation_dropout: Dropout in activation (default: 0.0)
        no_scale_embedding: Disable embedding scaling (default: True)
        layernorm_embedding: Apply layernorm to embeddings (default: False)
        layernorm_eps: Epsilon for layer normalization (default: 1e-6)
        subln: Use sub-layer normalization (default: True)
        deepnorm: Use deep normalization (default: False)
        no_output_layer: Disable output projection layer (default: True)
        vocab_size: Vocabulary size, -1 for no vocabulary (default: -1)
        
    Note:
        Chunkwise recurrent mode is not implemented in this version.
        See MultiScaleRetention docstring for extension instructions.
    """
    
    def __init__(self, **kwargs):
        # =====================================================================
        # Core architecture parameters
        # =====================================================================
        self.decoder_embed_dim = kwargs.pop("decoder_embed_dim", 128)
        self.decoder_value_embed_dim = kwargs.pop("decoder_value_embed_dim", 128)
        self.decoder_retention_heads = kwargs.pop("decoder_retention_heads", 8)
        self.decoder_ffn_embed_dim = kwargs.pop("decoder_ffn_embed_dim", 256)
        self.decoder_layers = kwargs.pop("decoder_layers", 3)
        self.decoder_normalize_before = kwargs.pop("decoder_normalize_before", True)
        
        # =====================================================================
        # Activation and regularization
        # =====================================================================
        self.activation_fn = kwargs.pop("activation_fn", "gelu")
        self.dropout = kwargs.pop("dropout", 0.0)
        self.drop_path_rate = kwargs.pop("drop_path_rate", 0.0)
        self.activation_dropout = kwargs.pop("activation_dropout", 0.0)
        
        # =====================================================================
        # Embedding settings
        # =====================================================================
        self.no_scale_embedding = kwargs.pop("no_scale_embedding", True)
        self.layernorm_embedding = kwargs.pop("layernorm_embedding", False)
        self.layernorm_eps = kwargs.pop("layernorm_eps", 1e-6)
        
        # =====================================================================
        # Normalization strategy
        # =====================================================================
        self.subln = kwargs.pop("subln", True)
        self.deepnorm = kwargs.pop("deepnorm", False)
        
        # =====================================================================
        # Output settings
        # =====================================================================
        self.no_output_layer = kwargs.pop("no_output_layer", True)
        self.vocab_size = kwargs.pop("vocab_size", -1)
        
        # =====================================================================
        # Chunkwise recurrent (DISABLED - parallel mode only)
        # =====================================================================
        # These are kept for future extension but not used
        self.chunkwise_recurrent = kwargs.pop("chunkwise_recurrent", False)
        self.recurrent_chunk_size = kwargs.pop("recurrent_chunk_size", 512)
        
        # =====================================================================
        # FSDP/Distributed settings (simplified - no fairscale dependency)
        # =====================================================================
        self.checkpoint_activations = kwargs.pop("checkpoint_activations", False)
        self.fsdp = kwargs.pop("fsdp", False)
        self.ddp_rank = kwargs.pop("ddp_rank", 0)
        
        # =====================================================================
        # Apply normalization strategy constraints
        # =====================================================================
        # deepnorm and subln are mutually exclusive
        if self.deepnorm:
            self.decoder_normalize_before = False
            self.subln = False
        if self.subln:
            self.decoder_normalize_before = True
            self.deepnorm = False
    
    def override(self, args):
        """
        Override config values from another object.
        
        Args:
            args: Object with attributes matching config parameter names
        """
        for hp in self.__dict__.keys():
            if getattr(args, hp, None) is not None:
                self.__dict__[hp] = getattr(args, hp, None)
    
    def __repr__(self):
        return (
            f"RetNetConfig(\n"
            f"  decoder_embed_dim={self.decoder_embed_dim},\n"
            f"  decoder_retention_heads={self.decoder_retention_heads},\n"
            f"  decoder_ffn_embed_dim={self.decoder_ffn_embed_dim},\n"
            f"  decoder_layers={self.decoder_layers},\n"
            f"  dropout={self.dropout},\n"
            f")"
        )

