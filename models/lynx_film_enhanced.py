"""
lynx_film_enhanced Model: Unified Architecture with Configurable Text Modulation

This model provides a unified framework for text-modulated time series forecasting with:
- Configurable pathway type: "shared" (Option 9) or "parallel" (Option 10)
- Configurable gate type: "global", "channel", or "conditional"
- Configurable mixing type: "interpolative" or "additive"

Configuration Space (3 axes):
=============================
AXIS 1 - Pathway Type:
    - "shared": Shared FFN/Attention, separate LayerNorms (fewer params, good for low data)
    - "parallel": Frozen unimodal + learned text pathways (requires checkpoint, more expressive)

AXIS 2 - Gate Type:
    - "global": Single scalar gate for all samples/channels (simplest, 1 param)
    - "channel": Per-channel static gate (C params, per-variable text reliance)
    - "conditional": MLP gate conditioned on x and T (sample-adaptive, learns when text is "trash")

AXIS 3 - Mixing Type:
    - "interpolative": z = α*z_text + (1-α)*z_base (bounded, convex combination)
    - "additive": z = z_base + α*z_text (unbounded, residual style)

Text Input Nomenclature:
========================
This model follows the same text input conventions as lynx_film_raw:
    
    Dataloader Name    Model Parameter      Description
    ---------------    ---------------      -----------
    x_hetero           historical_events    Text aligned to INPUT window timestamps
    y_hetero           news                 Text aligned to PREDICTION window timestamps

The model uses `timestamp_semantics` to determine which text input to use:
    t_about: Use news (y_hetero) - Fidel-TS (timestamps = target time)
    t_known: Use historical_events (x_hetero) - Time-MMD/TTC (timestamps = publication time)

Reference:
    See docs/planning/learned_text_regularization_options.md Part III for detailed design.
"""

from torch import nn
import torch
import copy
import numpy as np
from layers.TGTSF_torch import text_encoder
from layers.enhanced_film_layers import (
    GateModule,
    PathwayMixer,
    EnhancedEncoderLayerShared,
    EnhancedEncoderLayerParallel,
    EnhancedEncoderFilm
)
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """
    lynx_film_enhanced: Unified architecture with configurable pathway, gate, and mixing types.
    
    This model extends lynx_film_raw with:
    - Dual-pathway architecture: base (learned affine) and text (FiLM modulated)
    - Configurable gating: global, channel-wise, or input-conditioned
    - Configurable mixing: interpolative (bounded) or additive (residual)
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # ═══════════════════════════════════════════════════════════════
        # ENHANCED CONFIGURATION OPTIONS
        # ═══════════════════════════════════════════════════════════════
        
        # AXIS 1: Pathway type
        self.pathway_type = getattr(configs, 'pathway_type', 'shared')
        if self.pathway_type not in ('shared', 'parallel'):
            raise ValueError(
                f"Invalid pathway_type: '{self.pathway_type}'. Must be 'shared' or 'parallel'."
            )
        
        # AXIS 2: Gate type
        self.gate_type = getattr(configs, 'gate_type', 'global')
        if self.gate_type not in ('global', 'channel', 'conditional'):
            raise ValueError(
                f"Invalid gate_type: '{self.gate_type}'. Must be 'global', 'channel', or 'conditional'."
            )
        
        # AXIS 3: Mixing type
        self.mixing_type = getattr(configs, 'mixing_type', 'interpolative')
        if self.mixing_type not in ('interpolative', 'additive'):
            raise ValueError(
                f"Invalid mixing_type: '{self.mixing_type}'. Must be 'interpolative' or 'additive'."
            )
        
        # Gate configuration
        self.gate_hidden_dim = getattr(configs, 'gate_hidden_dim', 32)
        self.gate_init_bias = getattr(configs, 'gate_init_bias', 0.0)
        
        # Parallel pathway configuration
        self.unimodal_checkpoint = getattr(configs, 'unimodal_checkpoint', None)
        if self.pathway_type == 'parallel' and self.unimodal_checkpoint is None:
            raise ValueError(
                "pathway_type='parallel' requires 'unimodal_checkpoint' to be specified. "
                "Provide path to pre-trained unimodal model checkpoint."
            )
        
        print(f'[ info ] LYNX-FiLM-enhanced configuration:')
        print(f'         - pathway_type: {self.pathway_type}')
        print(f'         - gate_type: {self.gate_type}')
        print(f'         - mixing_type: {self.mixing_type}')
        
        # ═══════════════════════════════════════════════════════════════
        # STANDARD CONFIGURATION (same as lynx_film_raw)
        # ═══════════════════════════════════════════════════════════════
        
        # Store normalization configuration
        self.revin = getattr(configs, 'revin', True)
        self.use_norm = getattr(configs, 'use_norm', False)
        
        # Store model dimensions
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.dropout = configs.dropout
        self.activation = getattr(configs, 'activation', 'gelu')
        
        # Store sequence lengths
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Number of channels (for channel gate)
        self.num_channels = getattr(configs, 'enc_in', None)
        
        # Timestamp semantics determines which text input to use
        self.timestamp_semantics = getattr(configs, 'timestamp_semantics', None)
        if self.timestamp_semantics is None:
            raise ValueError(
                "lynx_film_enhanced requires 'timestamp_semantics' in configs. "
                "This must be set explicitly to avoid lookahead bias:\n"
                "  - 't_about': Text timestamps refer to the event/target time (Fidel-TS datasets)\n"
                "  - 't_known': Text timestamps refer to publication time (Time-MMD/TTC datasets)\n"
                "Set timestamp_semantics in data_config (hetero_info.timestamp_semantics or top-level)."
            )
        if self.timestamp_semantics not in ('t_about', 't_known'):
            raise ValueError(
                f"Invalid timestamp_semantics: '{self.timestamp_semantics}'. "
                f"Must be 't_about' or 't_known'."
            )
        print(f'         - timestamp_semantics: {self.timestamp_semantics}')
        
        # Flag to log text embedding shape only once
        self._logged_text_shape = False
        
        # Text dimension handling
        self.text_dim = configs.text_dim
        raw_input_text_dim = getattr(configs, 'input_text_dim', None)
        self.input_text_dim = self.text_dim if raw_input_text_dim is None else raw_input_text_dim
        
        if not isinstance(self.input_text_dim, int) or self.input_text_dim <= 0:
            raise ValueError(
                f"LYNX-FiLM-enhanced requires a positive integer input_text_dim; got {self.input_text_dim!r}."
            )
        
        # Learned projection layer if input dimension differs from operational dimension
        if self.input_text_dim != self.text_dim:
            self.text_projection = nn.Linear(self.input_text_dim, self.text_dim)
            print(f'         - text projection: {self.input_text_dim} -> {self.text_dim}')
        else:
            self.text_projection = None
        
        # ═══════════════════════════════════════════════════════════════
        # CALCULATE TEXT SEQUENCE LENGTH
        # ═══════════════════════════════════════════════════════════════
        hetero_align_stride = getattr(configs, 'hetero_align_stride', True)
        stride = configs.stride
        hetero_stride = stride if hetero_align_stride else 1
        
        if self.timestamp_semantics == 't_about':
            self.text_seq_len = int(np.ceil(self.pred_len / hetero_stride))
            print(f'         - text_seq_len: {self.text_seq_len} (from y_hetero/news)')
        else:
            self.text_seq_len = int(np.ceil(self.seq_len / hetero_stride))
            print(f'         - text_seq_len: {self.text_seq_len} (from x_hetero/historical_events)')
        
        # ═══════════════════════════════════════════════════════════════
        # TEXT ENCODER (from TGTSF)
        # ═══════════════════════════════════════════════════════════════
        self.text_encoder = text_encoder(
            cross_layer=configs.cross_layers,
            self_layer=configs.self_layers,
            embedding_dim=configs.text_dim,
            num_heads=configs.n_heads,
            dropout=configs.dropout,
            pred_len=configs.pred_len,
            stride=configs.stride
        )
        
        # ═══════════════════════════════════════════════════════════════
        # TIME SERIES EMBEDDING
        # ═══════════════════════════════════════════════════════════════
        self.enc_embedding = DataEmbedding_inverted(
            self.seq_len, self.d_model, self.dropout
        )
        
        # ═══════════════════════════════════════════════════════════════
        # CREATE GATE AND MIXER
        # ═══════════════════════════════════════════════════════════════
        self.gate = GateModule(
            gate_type=self.gate_type,
            num_channels=self.num_channels,
            d_model=self.d_model,
            text_dim=self.text_dim,
            hidden_dim=self.gate_hidden_dim,
            init_bias=self.gate_init_bias
        )
        
        self.mixer = PathwayMixer(self.mixing_type)
        
        # ═══════════════════════════════════════════════════════════════
        # CREATE ENCODER LAYERS
        # ═══════════════════════════════════════════════════════════════
        if self.pathway_type == "shared":
            # Shared backbone architecture (Option 9)
            layers = []
            for _ in range(self.e_layers):
                attention = AttentionLayer(
                    FullAttention(
                        False,
                        attention_dropout=self.dropout,
                        output_attention=False
                    ),
                    self.d_model,
                    self.n_heads
                )
                layer = EnhancedEncoderLayerShared(
                    attention=attention,
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    text_dim=self.text_dim,
                    text_seq_len=self.text_seq_len,
                    gate_module=self.gate,
                    mixer=self.mixer,
                    dropout=self.dropout,
                    activation=self.activation
                )
                layers.append(layer)
            
            self.encoder = EnhancedEncoderFilm(
                layers=layers,
                norm_layer=nn.LayerNorm(self.d_model)
            )
        
        elif self.pathway_type == "parallel":
            # Parallel experts architecture (Option 10)
            # Load pre-trained unimodal checkpoint
            unimodal_layers = self._load_unimodal_layers(self.unimodal_checkpoint)
            
            layers = []
            for i in range(self.e_layers):
                layer = EnhancedEncoderLayerParallel(
                    frozen_layer=unimodal_layers[i],
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    text_dim=self.text_dim,
                    text_seq_len=self.text_seq_len,
                    gate_module=self.gate,
                    mixer=self.mixer,
                    n_heads=self.n_heads,
                    dropout=self.dropout,
                    activation=self.activation
                )
                layers.append(layer)
            
            self.encoder = EnhancedEncoderFilm(
                layers=layers,
                norm_layer=nn.LayerNorm(self.d_model)
            )
        
        # ═══════════════════════════════════════════════════════════════
        # OUTPUT PROJECTION
        # ═══════════════════════════════════════════════════════════════
        self.projector = nn.Linear(self.d_model, self.pred_len, bias=True)
    
    def _load_unimodal_layers(self, checkpoint_path: str) -> list:
        """
        Load encoder layers from a pre-trained unimodal checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        
        Returns:
            List of encoder layers from the checkpoint
        """
        print(f'[ info ] Loading unimodal checkpoint from: {checkpoint_path}')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state dict (handle different checkpoint formats)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Create encoder layers with matching architecture
        from layers.Transformer_EncDec import EncoderLayer
        
        layers = []
        for i in range(self.e_layers):
            # Create layer with same architecture
            attention = AttentionLayer(
                FullAttention(False, attention_dropout=self.dropout),
                self.d_model, self.n_heads
            )
            layer = EncoderLayer(
                attention,
                self.d_model,
                self.d_ff,
                dropout=self.dropout,
                activation=self.activation
            )
            
            # Extract and load layer-specific weights
            layer_prefix = f'encoder.attn_layers.{i}.'
            layer_state = {}
            for key, value in state_dict.items():
                if key.startswith(layer_prefix):
                    new_key = key[len(layer_prefix):]
                    layer_state[new_key] = value
            
            if layer_state:
                layer.load_state_dict(layer_state, strict=False)
                print(f'         - Loaded layer {i} from checkpoint')
            else:
                print(f'         - Warning: No weights found for layer {i}, using random init')
            
            layers.append(layer)
        
        return layers
    
    def _project_text_embeddings(self, news, channel_description):
        """
        Project text embeddings from input_text_dim to text_dim if needed.
        
        Args:
            news: Text embeddings [B, L, N, input_text_dim]
            channel_description: Channel descriptions [B, C, input_text_dim] or [B, 1, C, input_text_dim]
        
        Returns:
            Projected embeddings with text_dim dimension
        """
        if self.text_projection is None:
            return news, channel_description
        
        # Project news embeddings
        B, L, N, D = news.shape
        news = news.reshape(B * L * N, D)
        news = self.text_projection(news)
        news = news.reshape(B, L, N, self.text_dim)
        
        # Project channel_description
        if channel_description.ndim == 3:
            B_desc, C_desc, D_desc = channel_description.shape
            channel_description = channel_description.reshape(B_desc * C_desc, D_desc)
            channel_description = self.text_projection(channel_description)
            channel_description = channel_description.reshape(B_desc, C_desc, self.text_dim)
        elif channel_description.ndim == 4:
            B_desc, _, C_desc, D_desc = channel_description.shape
            channel_description = channel_description.reshape(B_desc * C_desc, D_desc)
            channel_description = self.text_projection(channel_description)
            channel_description = channel_description.reshape(B_desc, 1, C_desc, self.text_dim)
        
        return news, channel_description
    
    def normalize_input(self, x):
        """
        Normalize input based on configured normalization scheme.
        
        Args:
            x: Input time series [B, seq_len, C]
        
        Returns:
            tuple: (x_norm, norm_params)
        """
        if self.revin:
            # RevIN normalization
            x_mean = torch.mean(x, dim=1, keepdim=True)
            x_norm = x - x_mean
            x_var = torch.var(x_norm, dim=1, keepdim=True) + 1e-5
            x_norm = x_norm / torch.sqrt(x_var)
            norm_params = {'type': 'revin', 'x_mean': x_mean, 'x_var': x_var}
        elif self.use_norm:
            # Non-stationary Transformer normalization
            means = x.mean(1, keepdim=True).detach()
            x_norm = x - means
            stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_norm = x_norm / stdev
            norm_params = {'type': 'use_norm', 'means': means, 'stdev': stdev}
        else:
            x_norm = x
            norm_params = {'type': 'none'}
        
        return x_norm, norm_params
    
    def denormalize_output(self, pred_norm, norm_params):
        """
        Denormalize predictions using stored normalization parameters.
        
        Args:
            pred_norm: Normalized predictions [B, pred_len, C]
            norm_params: Normalization parameters from normalize_input
        
        Returns:
            Denormalized predictions
        """
        if norm_params['type'] == 'revin':
            x_mean = norm_params['x_mean']
            x_var = norm_params['x_var']
            final_pred = pred_norm * torch.sqrt(x_var) + x_mean
        elif norm_params['type'] == 'use_norm':
            means = norm_params['means']
            stdev = norm_params['stdev']
            means_pred = means[:, 0, :].unsqueeze(1).repeat(1, pred_norm.shape[1], 1)
            stdev_pred = stdev[:, 0, :].unsqueeze(1).repeat(1, pred_norm.shape[1], 1)
            final_pred = pred_norm * stdev_pred + means_pred
        else:
            final_pred = pred_norm
        
        return final_pred
    
    def forward(self, x, news=None, channel_description=None, historical_events=None, **kwargs):
        """
        Forward pass through the enhanced model.
        
        Args:
            x: Input time series [B, seq_len, C]
            news: Text embeddings aligned to PREDICTION window (y_hetero)
            channel_description: Channel descriptions [B, C, d_model] or [B, 1, C, d_model]
            historical_events: Text embeddings aligned to INPUT window (x_hetero)
            **kwargs: Additional arguments (ignored)
        
        Returns:
            final_pred: Prediction [B, pred_len, C]
        """
        # Select text input based on timestamp_semantics
        if self.timestamp_semantics == 't_about':
            if news is None:
                raise ValueError("timestamp_semantics='t_about' requires 'news' (y_hetero)")
            text_input = news
        elif self.timestamp_semantics == 't_known':
            if historical_events is None:
                raise ValueError("timestamp_semantics='t_known' requires 'historical_events' (x_hetero)")
            text_input = historical_events
        else:
            raise ValueError(f"Invalid timestamp_semantics: {self.timestamp_semantics}")
        
        # Store raw input for conditional gate
        x_raw = x.clone()
        
        # Step 1: Normalize input
        x_norm, norm_params = self.normalize_input(x)
        
        # Step 2: Project text embeddings if needed
        text_input, channel_description = self._project_text_embeddings(text_input, channel_description)
        
        # Step 3: Get Text Embeddings via text encoder
        if len(channel_description.shape) == 3:
            channel_description = channel_description.unsqueeze(1)
        description = channel_description.repeat(1, text_input.shape[1], 1, 1)
        text_emb = self.text_encoder(text_input, description)
        
        # Permute text_emb: [B, L, C, text_dim] -> [B, C, L, text_dim]
        text_emb = text_emb.permute(0, 2, 1, 3)
        
        # Step 4: Embed time series
        # x_norm: [B, seq_len, C] -> enc_out: [B, C, d_model]
        enc_out = self.enc_embedding(x_norm, None)
        
        # Step 5: Encode with enhanced encoder
        enc_out, alphas = self.encoder(enc_out, text_emb, x_raw=x_raw)
        
        # Step 6: Project to prediction length
        # enc_out: [B, C, d_model] -> [B, C, pred_len] -> [B, pred_len, C]
        dec_out = self.projector(enc_out).permute(0, 2, 1)
        
        # Step 7: Denormalize
        final_pred = self.denormalize_output(dec_out, norm_params)
        
        return final_pred
    
    def _convert_hetero_channel_to_tensor(self, hetero_channel, device):
        """Convert hetero_channel to tensor and move to device."""
        if isinstance(hetero_channel, list):
            if len(hetero_channel) == 0:
                raise ValueError("Empty list for hetero_channel")
            elif isinstance(hetero_channel[0], str):
                raise ValueError("hetero_channel contains strings - set timemmd_text_output: embedding")
            elif isinstance(hetero_channel[0], np.ndarray):
                return torch.from_numpy(np.stack(hetero_channel)).float().to(device)
            elif isinstance(hetero_channel[0], torch.Tensor):
                return torch.stack(hetero_channel).float().to(device)
            else:
                raise ValueError(f"Unexpected element type: {type(hetero_channel[0])}")
        elif isinstance(hetero_channel, np.ndarray):
            return torch.from_numpy(np.asarray(hetero_channel)).float().to(device)
        elif isinstance(hetero_channel, str):
            raise ValueError("hetero_channel is string - set timemmd_text_output: embedding")
        elif isinstance(hetero_channel, torch.Tensor):
            return hetero_channel.float().to(device)
        else:
            raise ValueError(f"Unexpected type: {type(hetero_channel)}")
    
    def _convert_text_embedding_to_tensor(self, text_embedding, device, name="text_embedding"):
        """Convert text embedding to tensor and move to device."""
        if isinstance(text_embedding, list):
            if len(text_embedding) == 0:
                raise ValueError(f"Empty list for {name}")
            elif isinstance(text_embedding[0], str):
                raise ValueError(f"{name} contains strings - set timemmd_text_output: embedding")
            elif isinstance(text_embedding[0], np.ndarray):
                return torch.from_numpy(np.stack(text_embedding)).float().to(device)
            elif isinstance(text_embedding[0], torch.Tensor):
                return torch.stack(text_embedding).float().to(device)
            else:
                raise ValueError(f"Unexpected element type in {name}: {type(text_embedding[0])}")
        elif isinstance(text_embedding, np.ndarray):
            return torch.from_numpy(np.asarray(text_embedding)).float().to(device)
        elif isinstance(text_embedding, torch.Tensor):
            return text_embedding.float().to(device)
        else:
            raise ValueError(f"Unexpected type for {name}: {type(text_embedding)}")
    
    def move_to_device(self, seq_x, seq_y, x_time, y_time, x_hetero, y_hetero,
                       hetero_x_time, hetero_y_time, hetero_general, hetero_channel, device):
        """
        Move data to device.
        
        Args:
            seq_x: Input time series [B, seq_len, C]
            seq_y: Target time series [B, pred_len, C]
            x_time, y_time: Timestamps
            x_hetero: Historical text embeddings (x_hetero from dataloader)
            y_hetero: Prediction window text embeddings (y_hetero from dataloader)
            hetero_x_time, hetero_y_time: Text timestamps
            hetero_general: General dataset description
            hetero_channel: Channel descriptions
            device: Target device
        
        Returns:
            tuple: All inputs with tensors moved to device
        """
        # Move time series data
        seq_x = seq_x.float().to(device)
        seq_y = seq_y.float().to(device)
        
        # Convert hetero_channel
        hetero_channel = self._convert_hetero_channel_to_tensor(hetero_channel, device)
        
        # Convert text embeddings
        if x_hetero is not None:
            x_hetero = self._convert_text_embedding_to_tensor(x_hetero, device, name="x_hetero")
        if y_hetero is not None:
            y_hetero = self._convert_text_embedding_to_tensor(y_hetero, device, name="y_hetero")
        
        # Log shape once
        if not self._logged_text_shape:
            if self.timestamp_semantics == 't_about' and y_hetero is not None:
                print(f'[ info ] LYNX-FiLM-enhanced: y_hetero shape: {y_hetero.shape}')
                self._logged_text_shape = True
            elif self.timestamp_semantics == 't_known' and x_hetero is not None:
                print(f'[ info ] LYNX-FiLM-enhanced: x_hetero shape: {x_hetero.shape}')
                self._logged_text_shape = True
        
        return (seq_x, seq_y, x_time, y_time, x_hetero, y_hetero,
                hetero_x_time, hetero_y_time, hetero_general, hetero_channel)
