# FEDformer Implementation Analysis

## Overview

**FEDformer** (Frequency Enhanced Decomposed Transformer) is a Transformer-based architecture for long-term time series forecasting that operates in the **frequency domain** rather than the time domain. Published at ICML 2022, it achieves O(N) computational complexity compared to O(N²) for standard attention.

**Key Paper:** Zhou et al., "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting" (ICML 2022)

---

## Table of Contents
1. [Core Architecture](#core-architecture)
2. [Key Components](#key-components)
3. [Frequency-Domain Operations](#frequency-domain-operations)
4. [Multivariate Time Series Handling](#multivariate-time-series-handling)
5. [Comparison with TGTSF](#comparison-with-tgtsf)
6. [Comparison with Lynx-FiLM-Raw](#comparison-with-lynx-film-raw)
7. [Summary Comparison Table](#summary-comparison-table)
8. [Lynx-FiLM-Raw-FEDformer Design Plan](#lynx-film-raw-fedformer-design-plan)

---

## Core Architecture

### High-Level Design

FEDformer follows an **encoder-decoder** architecture with three key innovations:

1. **Frequency Enhanced Attention (FEA)**: Replaces standard O(N²) attention with O(N) frequency-domain operations
2. **Seasonal-Trend Decomposition**: Progressive decomposition of time series into seasonal and trend components
3. **Two Versions**: Supports both Fourier-based and Wavelet-based frequency operations

```
Input Sequence
      │
      ▼
┌─────────────────┐
│  Decomposition  │  ← Moving average to extract trend
│ (Seasonal/Trend)│
└─────────────────┘
      │
      ├── Seasonal Component ──► Encoder ──► Decoder ──► Seasonal Prediction
      │                             │           │
      │                             ▼           ▼
      └── Trend Component ────────────────────────────► Trend Prediction
                                                              │
                                                              ▼
                                                     Final = Seasonal + Trend
```

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `version` | "Fourier" | "Fourier" or "Wavelets" |
| `modes` | 64 | Number of frequency modes to keep |
| `mode_select` | "random" | "random" or "low" frequency selection |
| `moving_avg` | [24] | Kernel size(s) for trend extraction |
| `d_model` | 512 | Model embedding dimension |
| `n_heads` | 8 | Number of attention heads |
| `e_layers` | 2 | Number of encoder layers |
| `d_layers` | 1 | Number of decoder layers |

---

## Key Components

### 1. Series Decomposition

The decomposition layer separates time series into **seasonal** and **trend** components using moving average:

```python
class series_decomp(nn.Module):
    """Extracts trend via moving average, seasonal as residual"""
    def __init__(self, kernel_size):
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)  # Trend
        res = x - moving_mean             # Seasonal
        return res, moving_mean
```

**Multi-scale Decomposition**: When `moving_avg` is a list (e.g., [12, 24]), it uses learnable weighted combination:
```python
class series_decomp_multi(nn.Module):
    """Combines multiple moving averages with learned weights"""
    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_mean.append(func(x).unsqueeze(-1))
        moving_mean = torch.sum(moving_mean * Softmax(self.layer(x)), dim=-1)
        return x - moving_mean, moving_mean
```

### 2. Embedding Layer

FEDformer uses **position-free embedding** because frequency-domain operations inherently capture positional information:

```python
class DataEmbedding_wo_pos(nn.Module):
    """Value + Temporal embedding (NO positional encoding)"""
    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)
```

Components:
- **TokenEmbedding**: 1D Conv with kernel_size=3, circular padding
- **TimeFeatureEmbedding**: Linear projection of time features (hour, day, week, month)

### 3. Encoder Layer

Each encoder layer performs:
1. Frequency-Enhanced Attention (FEA)
2. Decomposition (extract and discard trend)
3. Feed-Forward Network (FFN)
4. Decomposition (extract and discard trend)

```python
class EncoderLayer(nn.Module):
    def forward(self, x, attn_mask=None):
        # 1. Attention + Residual
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        
        # 2. First decomposition (discard trend)
        x, _ = self.decomp1(x)
        
        # 3. FFN
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        # 4. Second decomposition (discard trend)
        res, _ = self.decomp2(x + y)
        return res, attn
```

### 4. Decoder Layer

The decoder layer accumulates trend components across layers:

```python
class DecoderLayer(nn.Module):
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Self-attention + decomposition → trend1
        x = x + self.self_attention(x, x, x)[0]
        x, trend1 = self.decomp1(x)
        
        # Cross-attention + decomposition → trend2
        x = x + self.cross_attention(x, cross, cross)[0]
        x, trend2 = self.decomp2(x)
        
        # FFN + decomposition → trend3
        y = self.conv2(self.activation(self.conv1(x)))
        x, trend3 = self.decomp3(x + y)
        
        # Accumulate trends
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend)  # Project to c_out channels
        
        return x, residual_trend
```

---

## Frequency-Domain Operations

### Fourier Block (Self-Attention Replacement)

Performs attention in the Fourier domain with O(N) complexity:

```python
class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes, mode_select_method):
        # Select frequency modes (random or low-frequency)
        self.index = get_frequency_modes(seq_len, modes, mode_select_method)
        
        # Learnable complex weights for selected modes
        self.weights1 = nn.Parameter(
            torch.rand(8, in_channels//8, out_channels//8, len(self.index), dtype=torch.cfloat)
        )

    def forward(self, q, k, v, mask):
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)  # [B, H, E, L]
        
        # FFT to frequency domain
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Linear transform on selected modes only (sparse operation)
        out_ft = torch.zeros(B, H, E, L//2+1, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        
        # Inverse FFT back to time domain
        return torch.fft.irfft(out_ft, n=L)
```

**Mode Selection**:
- `random`: Randomly samples `modes` frequencies from available spectrum
- `low`: Takes the lowest `modes` frequencies (most energy)

### Fourier Cross Attention

Performs encoder-decoder cross-attention in frequency domain:

```python
class FourierCrossAttention(nn.Module):
    def forward(self, q, k, v, mask):
        # FFT of query and key
        xq_ft = torch.fft.rfft(xq, dim=-1)  # Query spectrum
        xk_ft = torch.fft.rfft(xk, dim=-1)  # Key spectrum
        
        # Attention in frequency domain (complex multiplication)
        xqk_ft = torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_)
        xqk_ft = xqk_ft.tanh()  # or softmax(abs(...))
        
        # Aggregate values using frequency attention
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        
        # Learnable projection + IFFT
        out = torch.fft.irfft(xqkvw / channels, n=L)
        return out
```

### Wavelet Transform (Alternative)

Multi-wavelet transform for hierarchical frequency analysis:

```python
class MultiWaveletCross(nn.Module):
    """Wavelet decomposition with Fourier cross-attention at each level"""
    def forward(self, q, k, v, mask):
        # Hierarchical decomposition
        for i in range(ns - L):
            d, q = self.wavelet_transform(q)  # Detail + Smooth
            # Apply Fourier attention at each scale
            Ud += [self.attn1(dq, dk, dv) + self.attn2(sq, sk, sv)]
        
        # Hierarchical reconstruction
        for i in reversed(range(ns - L)):
            v = v + Us[i]
            v = self.evenOdd(torch.cat([v, Ud[i]], -1))
        return v
```

---

## Multivariate Time Series Handling

### How FEDformer Processes Multiple Variables

FEDformer handles multivariate time series through a **channel-mixing embedding** strategy that fundamentally differs from channel-independent approaches (like iTransformer or PatchTST).

#### Data Flow for Multivariate Input

```
Input: x [B, L, C]  where C = number of variables (e.g., 7 for ETT datasets)
                                                    (e.g., 321 for Traffic)
                                                    (e.g., 862 for Electricity)

Step 1: TokenEmbedding (1D Conv across channels)
        Conv1d(in_channels=C, out_channels=d_model, kernel_size=3)
        
        x.permute(0, 2, 1)  →  [B, C, L]
        tokenConv(x)        →  [B, d_model, L]
        .transpose(1, 2)    →  [B, L, d_model]
        
        Result: All C channels are MIXED into a shared d_model embedding space

Step 2: Temporal Embedding (shared across all original channels)
        TimeFeatureEmbedding(d_inp=4, d_out=d_model)  # For hourly freq
        
        Result: [B, L, d_model] - same temporal embedding for all channels

Step 3: Combined Embedding
        embedded = value_embedding + temporal_embedding
        Result: [B, L, d_model]
```

#### Key Insight: Channel Mixing via Convolution

```python
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        # c_in = number of input channels (variables)
        # d_model = embedding dimension
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,      # Each channel is an input feature
            out_channels=d_model,   # All channels projected to shared space
            kernel_size=3,          # Local temporal context
            padding=1,
            padding_mode='circular'
        )

    def forward(self, x):
        # x: [B, L, C] → [B, C, L] → Conv → [B, d_model, L] → [B, L, d_model]
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
```

**This means:**
1. **Channel Correlations are Learned Early**: The conv kernel learns to weight and combine different variables based on their co-occurrence patterns
2. **No Explicit Channel Attention**: Unlike iTransformer, there's no mechanism for channels to attend to each other
3. **Shared Representation**: All subsequent processing (attention, decomposition) operates on the mixed representation

#### Comparison: Multivariate Strategies

| Model | Channel Strategy | Embedding | Attention Over |
|-------|------------------|-----------|----------------|
| **FEDformer** | Channel-Mixing | Conv1d(C→d_model) | Time (mixed channels) |
| **iTransformer** | Channel-Independent | Linear(L→d_model) per channel | Channels (each as token) |
| **TGTSF** | Channel-Independent | Patch + Linear per channel | Time within channel, then mixer |
| **Lynx-FiLM-Raw** | Channel-Independent | Linear(L→d_model) per channel | Channels with FiLM |

#### Output Projection

```python
# In Decoder
self.projection = nn.Linear(d_model, c_out, bias=True)

# Projects from shared embedding back to individual channels
# d_model → c_out (number of output channels)
```

### Implications for Multivariate Forecasting

#### Advantages of Channel-Mixing

1. **Cross-Variable Correlations**: The conv kernel can learn that "when variable A rises, variable B tends to follow"
2. **Efficient**: Single attention pass for all variables (vs. C separate passes)
3. **Robust to Missing Channels**: The learned mixing can interpolate patterns

#### Disadvantages of Channel-Mixing

1. **No Channel-Specific Conditioning**: Can't easily incorporate per-channel text descriptions
2. **Fixed Channel Order**: Conv assumes a specific ordering of channels
3. **Scaling**: Adding new channels requires retraining (changes `c_in`)

### Multivariate Results from Paper

The FEDformer paper reports significant improvements on multivariate benchmarks:

| Dataset | Channels | FEDformer MSE | Autoformer MSE | Improvement |
|---------|----------|---------------|----------------|-------------|
| ETTh1 | 7 | 0.376 | 0.435 | -13.6% |
| ETTm1 | 7 | 0.379 | 0.505 | -25.0% |
| Traffic | 862 | 0.587 | 0.613 | -4.2% |
| Electricity | 321 | 0.183 | 0.201 | -9.0% |

The channel-mixing approach combined with frequency-domain attention proves effective across different scales of multivariate data.

---

## Comparison with TGTSF

### Architecture Overview: TGTSF

**TGTSF** (Text-Guided Time Series Forecasting) is a **multimodal** model that integrates:
- Time series data
- News/event embeddings
- Channel (variable) descriptions

```
┌───────────────────────────────────────────────────────────────┐
│                          TGTSF                                │
├───────────────────────────────────────────────────────────────┤
│  Input: x [B, L, C], news [B, l, n, D], description [B, C, D] │
│                                                               │
│  1. RevIN Normalization                                       │
│  2. TS_encoder (Patch + Transformer)                          │
│  3. text_encoder (Cross-attention news ↔ descriptions)        │
│  4. Mixer (Cross-attention text ↔ temporal)                   │
│  5. Head + Patch Reconstruction                               │
│  6. RevIN Denormalization                                     │
└───────────────────────────────────────────────────────────────┘
```

### Key Differences

| Aspect | FEDformer | TGTSF |
|--------|-----------|-------|
| **Modality** | Unimodal (time series only) | Multimodal (time series + text) |
| **Attention Domain** | Frequency domain | Time domain |
| **Complexity** | O(N) | O(N²) |
| **Decomposition** | Seasonal-Trend (moving avg) | None (raw signal) |
| **Architecture** | Encoder-Decoder | Encoder-only + Head |
| **Temporal Modeling** | Full sequence attention | Patch-based attention |
| **Normalization** | None (relies on decomposition) | RevIN |

### TGTSF Components in Detail

**1. Time Series Encoder (`TS_encoder`)**:
```python
class TS_encoder(nn.Module):
    def __init__(self, patch_len=8, stride=8, ...):
        self.input_encoder = nn.Linear(patch_len, embedding_dim)
        self.attentions = nn.TransformerEncoder(...)
    
    def forward(self, x):
        x = x.unfold(-1, patch_len, stride)  # Create patches
        x = self.input_encoder(x)             # Embed patches
        x = x + self.W_pos                    # Add position
        x = self.attentions(x)                # Transformer
        return x  # [B, patch_num, C, d_model]
```

**2. Text Encoder (`text_encoder`)**:
```python
class text_encoder(nn.Module):
    def forward(self, news_emb, description_emb):
        # Cross-attention: description queries news
        text_emb = self.cross_encoder(
            tgt=description_emb,  # [B*L, C, D]
            memory=news_emb       # [B*L, N, D]
        )
        return text_emb + positional_encoding
```

**3. Text-Temporal Mixer (`text_temp_cross_block`)**:
```python
class text_temp_cross_block(nn.Module):
    def forward(self, text_emb, temp_emb):
        # Text queries temporal features
        result = self.cross_encoder(tgt=text_emb, memory=temp_emb)
        result = self.self_encoder(tgt=result, memory=result)
        return result  # [B, L, C, D]
```

**4. Patch Reconstruction (Vectorized)**:
```python
def patch_reconstruction(self, x):
    """Convert overlapping patches to continuous sequence"""
    # Uses torch.nn.functional.fold for efficient vectorized operation
    output_sum = F.fold(x_reshaped, output_size=(1, covered_length), ...)
    output_count = F.fold(ones, ...)  # Count overlaps
    return output_sum / output_count
```

### Handling of Temporal Information

| Model | Positional Encoding | Temporal Features |
|-------|---------------------|-------------------|
| FEDformer | None (freq domain captures position) | Temporal embeddings (hour, day, etc.) |
| TGTSF | Learned per-patch | Text provides temporal context |

---

## Comparison with Lynx-FiLM-Raw

### Architecture Overview: Lynx-FiLM-Raw

**Lynx-FiLM-Raw** combines iTransformer with FiLM (Feature-wise Linear Modulation) conditioning from text:

```
┌───────────────────────────────────────────────────────────────┐
│                      Lynx-FiLM-Raw                            │
├───────────────────────────────────────────────────────────────┤
│  Input: x [B, L, C], news [B, l, n, D], description [B, C, D] │
│                                                               │
│  1. RevIN/use_norm Normalization                              │
│  2. text_encoder (same as TGTSF)                              │
│  3. iTransformerFilm (inverted + FiLM)                        │
│  4. Denormalization                                           │
└───────────────────────────────────────────────────────────────┘
```

### Key Innovation: FiLM Modulation

FiLM modulates neural network layers using affine transformations derived from conditioning information (text):

```python
# FiLM: x' = γ * x + β
x = (1 + gamma) * x + beta
```

**FiLM Generator**:
```python
class FiLMGenerator(nn.Module):
    def __init__(self, text_dim, output_dim, seq_len):
        input_dim = seq_len * text_dim  # Flatten text
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim * 4)  # γ1, β1, γ2, β2
        )
    
    def forward(self, x):
        params = self.net(x.flatten(-2))  # Flatten [B, C, L, D] → [B, C, L*D]
        return torch.chunk(params, 4, dim=-1)
```

**FiLM-Modulated Encoder Layer**:
```python
class EncoderLayerFilm(nn.Module):
    def forward(self, x, gamma1, beta1, gamma2, beta2, ...):
        # Attention
        x = x + self.attention(x, x, x)[0]
        x = self.norm1(x)
        
        # FiLM Modulation 1 (post-attention)
        x = (1 + gamma1) * x + beta1
        
        # FFN
        x = self.norm2(x + self.ffn(x))
        
        # FiLM Modulation 2 (post-FFN)
        x = (1 + gamma2) * x + beta2
        
        return x
```

### iTransformer Design

The key insight of iTransformer is **inverted attention**:
- Standard Transformer: attention over time dimension
- iTransformer: attention over **variable (channel) dimension**

```python
class iTransformerFilm(nn.Module):
    def forecast(self, x_enc, text_emb):
        # x_enc: [B, L, N] where N = number of variables
        
        # Embed: [B, L, N] → [B, N, E]
        # Each variable becomes a token!
        enc_out = self.enc_embedding(x_enc)
        
        # Attention over variables (not time)
        enc_out = self.encoder(enc_out, text_emb)  # [B, N, E]
        
        # Project to predictions: [B, N, E] → [B, S, N]
        dec_out = self.projector(enc_out).permute(0, 2, 1)
        return dec_out
```

### Three-Way Comparison: Attention Mechanisms

| Model | Attention Type | Complexity | Domain |
|-------|---------------|------------|--------|
| FEDformer | Freq-Enhanced Self | O(N) | Frequency |
| TGTSF | Patch-based Self + Cross | O(P²) + O(P·N) | Time |
| Lynx-FiLM-Raw | Inverted (Variable) | O(C²) | Time |

Where: N = sequence length, P = patch count, C = channel count

---

## Summary Comparison Table

| Feature | FEDformer | TGTSF | Lynx-FiLM-Raw |
|---------|-----------|-------|---------------|
| **Paper/Year** | ICML 2022 | - | - |
| **Modality** | Unimodal | Multimodal | Multimodal |
| **Text Integration** | ❌ None | Cross-attention | FiLM modulation |
| **Base Architecture** | Transformer Enc-Dec | Transformer Enc | iTransformer Enc |
| **Attention Dimension** | Time (freq domain) | Time (patches) | Variables |
| **Computational Complexity** | O(N) | O(P²) | O(C²) |
| **Decomposition** | ✅ Seasonal-Trend | ❌ None | ❌ None |
| **Normalization** | Implicit (decomp) | RevIN | RevIN/use_norm |
| **Frequency Operations** | ✅ FFT/Wavelet | ❌ None | ❌ None |
| **Positional Encoding** | ❌ None | ✅ Learned | ✅ None (inverted) |
| **Channel Modeling** | Independent | Attention (mixer) | FiLM + Attention |

### When to Use Each Model

1. **FEDformer**: 
   - Long-horizon forecasting (O(N) scales well)
   - Strongly periodic/seasonal data
   - No external context available

2. **TGTSF**:
   - News/events impact forecasting
   - Rich channel descriptions available
   - Moderate sequence lengths

3. **Lynx-FiLM-Raw**:
   - Multivariate with correlated channels
   - Text conditioning improves all channels
   - Memory-efficient (O(C²) vs O(L²))

---

## Implementation Details

### FEDformer Forward Pass

```python
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, ...):
    # 1. Initialize decomposition
    mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, pred_len, 1)
    seasonal_init, trend_init = self.decomp(x_enc)
    
    # 2. Prepare decoder inputs
    trend_init = torch.cat([trend_init[:, -label_len:, :], mean], dim=1)
    seasonal_init = F.pad(seasonal_init[:, -label_len:, :], (0, 0, 0, pred_len))
    
    # 3. Encoder
    enc_out = self.enc_embedding(x_enc, x_mark_enc)
    enc_out, _ = self.encoder(enc_out)
    
    # 4. Decoder (with cross-attention to encoder)
    dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
    seasonal_part, trend_part = self.decoder(dec_out, enc_out, trend=trend_init)
    
    # 5. Final prediction
    return trend_part + seasonal_part
```

### TGTSF Forward Pass

```python
def forward(self, x, news, channel_description, **kwargs):
    # 1. RevIN normalization
    x = (x - x.mean(1, keepdim=True)) / sqrt(x.var(1, keepdim=True) + 1e-5)
    
    # 2. Time series encoding (patches)
    x = self.TS_encoder(x)  # [B, patch_num, C, d_model]
    
    # 3. Text encoding
    t = self.text_encoder(news, description)  # [B, l, C, text_dim]
    
    # 4. Mixing text and temporal
    x, _ = self.mixer(t, x)  # [B, patch_num, C, d_model]
    
    # 5. Prediction head + reconstruction
    x = self.head(x)  # [B, patch_num, C, patch_len]
    x = self.patch_reconstruction(x)  # [B, C, pred_len]
    
    # 6. RevIN denormalization
    return x * sqrt(x_var) + x_mean
```

### Lynx-FiLM-Raw Forward Pass

```python
def forward(self, x, news, channel_description, **kwargs):
    # 1. Normalize
    x_norm, norm_params = self.normalize_input(x)
    
    # 2. Project text if needed
    news, channel_description = self._project_text_embeddings(news, channel_description)
    
    # 3. Encode text
    text_emb = self.text_encoder(news, description)  # [B, L, C, text_dim]
    text_emb = text_emb.permute(0, 2, 1, 3)  # [B, C, L, text_dim]
    
    # 4. iTransformerFilm (FiLM-modulated inverted attention)
    pred_norm = self.model(x_norm, text_emb)
    
    # 5. Denormalize
    return self.denormalize_output(pred_norm, norm_params)
```

---

## Code References

- **FEDformer**: `benchmark_models/FEDformer/models/FEDformer.py`
- **FourierCorrelation**: `benchmark_models/FEDformer/layers/FourierCorrelation.py`
- **MultiWaveletCorrelation**: `benchmark_models/FEDformer/layers/MultiWaveletCorrelation.py`
- **TGTSF**: `models/TGTSF.py`
- **Lynx-FiLM-Raw**: `models/lynx_film_raw.py`
- **FiLM Layers**: `layers/FiLM_layers.py`
- **iTransformerFilm**: `layers/lynx_film_layers.py`

---

## Lynx-FiLM-Raw-FEDformer Design Plan

This section provides a detailed plan for creating **lynx-film-raw-fedformer**, a multimodal extension of FEDformer using FiLM conditioning. The current `lynx_film_raw.py` would be renamed to `lynx_film_raw_itransformer.py`.

### Design Goals

1. **Preserve FEDformer's Strengths**: O(N) complexity, frequency-domain attention, seasonal-trend decomposition
2. **Add Multimodal Capability**: Integrate text (news + channel descriptions) via FiLM
3. **Channel-Specific Conditioning**: Allow per-channel text to influence per-channel predictions
4. **Maintain Architecture Compatibility**: Use existing text_encoder from TGTSF

### Key Challenge: Channel-Mixing vs. FiLM

FEDformer's channel-mixing embedding complicates per-channel FiLM conditioning:

```
Current FEDformer:
x [B, L, C] → Conv1d(C, d_model) → [B, L, d_model]
                                     ↑
                                 Channels are mixed!

Desired FiLM:
text_emb [B, C, L, text_dim] → FiLMGenerator → [B, C, d_model]
                                                   ↑
                                              Per-channel γ, β
```

**Solution Options:**

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| **A** | Channel-Mixed FiLM | Keep FEDformer embedding, global FiLM | Loses per-channel conditioning |
| **B** | Channel-Independent FEDformer | Modify embedding to process channels separately | Changes core architecture |
| **C** | Hybrid: Early Channel-Mixing + Late FiLM | Mix channels in encoder, per-channel FiLM in decoder | Balanced trade-off |
| **D** | Per-Channel Frequency FiLM | Apply FiLM to frequency coefficients per channel | Novel, preserves freq-domain |

**Recommended: Option C (Hybrid Approach)**

### Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                    LYNX-FiLM-RAW-FEDformer                             │
├────────────────────────────────────────────────────────────────────────┤
│  Inputs:                                                               │
│    x [B, L, C]              - Time series                              │
│    news [B, l, n, D]        - News embeddings                          │
│    channel_desc [B, C, D]   - Channel descriptions                     │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 1. NORMALIZATION (RevIN)                                         │  │
│  │    x_norm, norm_params = normalize_input(x)                      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 2. TEXT ENCODING (from TGTSF)                                    │  │
│  │    text_emb = text_encoder(news, channel_desc)                   │  │
│  │    → [B, L, C, text_dim]                                         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 3. DECOMPOSITION (FEDformer style)                               │  │
│  │    seasonal_init, trend_init = decomp(x_norm)                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 4. ENCODER (FEDformer with Channel-Mixed FiLM)                   │  │
│  │    enc_emb = enc_embedding(x_norm)  → [B, L, d_model]            │  │
│  │    enc_text = pool_text(text_emb)   → [B, d_model] (global)      │  │
│  │    γ_enc, β_enc = film_gen_enc(enc_text)                         │  │
│  │    enc_out = encoder_film(enc_emb, γ_enc, β_enc)                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 5. DECODER (FEDformer with Per-Channel FiLM)                     │  │
│  │    dec_emb = dec_embedding(seasonal_init)  → [B, L_dec, d_model] │  │
│  │    for each decoder layer:                                        │  │
│  │      - Self-attention + decomposition                             │  │
│  │      - Cross-attention to encoder + decomposition                 │  │
│  │      - FFN + decomposition                                        │  │
│  │      - FiLM after each block (per-channel in output projection)  │  │
│  │    seasonal_part, trend_part = decoder_film(...)                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 6. OUTPUT (Per-Channel FiLM on Trend)                            │  │
│  │    γ_out, β_out = film_gen_out(text_emb)  → [B, C, 1]            │  │
│  │    prediction = (seasonal + trend) * (1 + γ_out) + β_out         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 7. DENORMALIZATION                                               │  │
│  │    final_pred = denormalize_output(prediction, norm_params)      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### New Components to Implement

#### 1. FiLMGeneratorFEDformer (Global Text → Global FiLM)

```python
class FiLMGeneratorFEDformer(nn.Module):
    """
    Generates FiLM parameters for FEDformer's channel-mixed representation.
    
    Takes global text representation (pooled across channels and time)
    and produces modulation parameters for the shared d_model space.
    """
    def __init__(self, text_dim, d_model, text_seq_len, n_channels):
        super().__init__()
        # Pool text across channels: [B, L, C, text_dim] → [B, L*text_dim]
        self.text_pool = nn.Sequential(
            nn.Linear(n_channels * text_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Generate FiLM params: 4 sets (γ1, β1, γ2, β2) for each layer position
        self.film_head = nn.Linear(d_model * text_seq_len, d_model * 4)
    
    def forward(self, text_emb):
        # text_emb: [B, L, C, text_dim]
        B, L, C, D = text_emb.shape
        
        # Pool across channels per timestep
        text_pooled = text_emb.permute(0, 1, 3, 2)  # [B, L, D, C]
        text_pooled = text_pooled.reshape(B, L, -1)  # [B, L, D*C]
        text_pooled = self.text_pool(text_pooled)    # [B, L, d_model]
        
        # Flatten and generate FiLM params
        text_flat = text_pooled.reshape(B, -1)       # [B, L*d_model]
        params = self.film_head(text_flat)           # [B, d_model*4]
        
        gamma1, beta1, gamma2, beta2 = torch.chunk(params, 4, dim=-1)
        return gamma1, beta1, gamma2, beta2  # Each [B, d_model]
```

#### 2. FiLMGeneratorPerChannel (Text → Per-Channel FiLM for Output)

```python
class FiLMGeneratorPerChannel(nn.Module):
    """
    Generates per-channel FiLM parameters for the final output modulation.
    
    Each channel gets its own γ and β based on its text description.
    """
    def __init__(self, text_dim, text_seq_len, hidden_dim=256):
        super().__init__()
        input_dim = text_seq_len * text_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)  # γ and β per channel
        )
    
    def forward(self, text_emb):
        # text_emb: [B, C, L, text_dim] (already permuted)
        B, C, L, D = text_emb.shape
        
        # Flatten temporal dimension per channel
        text_flat = text_emb.reshape(B, C, L * D)  # [B, C, L*D]
        
        # Generate γ, β per channel
        params = self.net(text_flat)  # [B, C, 2]
        gamma, beta = params[..., 0:1], params[..., 1:2]  # [B, C, 1]
        
        return gamma, beta
```

#### 3. EncoderLayerFilmFED (FEDformer Encoder with FiLM)

```python
class EncoderLayerFilmFED(nn.Module):
    """
    FEDformer encoder layer with FiLM modulation.
    
    Preserves:
    - Frequency-enhanced attention (FourierBlock or MultiWaveletTransform)
    - Progressive decomposition
    
    Adds:
    - FiLM modulation after attention and FFN
    """
    def __init__(self, attention, d_model, d_ff, moving_avg, dropout, activation):
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x, gamma1, beta1, gamma2, beta2, attn_mask=None):
        # x: [B, L, d_model]
        # gamma, beta: [B, d_model]
        
        # 1. Attention + Residual
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        
        # 2. First decomposition
        x, _ = self.decomp1(x)
        
        # 3. FiLM Modulation 1 (broadcast over sequence length)
        # gamma1, beta1: [B, d_model] → [B, 1, d_model]
        x = (1 + gamma1.unsqueeze(1)) * x + beta1.unsqueeze(1)
        
        # 4. FFN
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        # 5. Second decomposition
        x, _ = self.decomp2(x + y)
        
        # 6. FiLM Modulation 2
        x = (1 + gamma2.unsqueeze(1)) * x + beta2.unsqueeze(1)
        
        return x, attn
```

#### 4. DecoderLayerFilmFED (FEDformer Decoder with FiLM)

```python
class DecoderLayerFilmFED(nn.Module):
    """
    FEDformer decoder layer with FiLM modulation.
    
    FiLM is applied at three points:
    1. After self-attention + decomposition
    2. After cross-attention + decomposition
    3. After FFN + decomposition
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, 
                 d_ff, moving_avg, dropout, activation):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.projection = nn.Conv1d(d_model, c_out, kernel_size=3, padding=1,
                                    padding_mode='circular', bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x, cross, 
                gamma1, beta1, gamma2, beta2, gamma3, beta3,
                x_mask=None, cross_mask=None):
        # 1. Self-attention + decomposition + FiLM
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)
        x = (1 + gamma1.unsqueeze(1)) * x + beta1.unsqueeze(1)
        
        # 2. Cross-attention + decomposition + FiLM
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)
        x = (1 + gamma2.unsqueeze(1)) * x + beta2.unsqueeze(1)
        
        # 3. FFN + decomposition + FiLM
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)
        x = (1 + gamma3.unsqueeze(1)) * x + beta3.unsqueeze(1)
        
        # 4. Trend accumulation
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        
        return x, residual_trend
```

### Main Model: lynx_film_raw_fedformer.py

```python
class Model(nn.Module):
    """
    lynx-film-raw-fedformer: FEDformer with FiLM conditioning from text.
    
    Key features:
    - Frequency-enhanced attention (Fourier or Wavelet)
    - Seasonal-trend decomposition
    - RevIN normalization
    - Global FiLM in encoder (channel-mixed representation)
    - Per-channel FiLM in output (for channel-specific predictions)
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # Core FEDformer parameters
        self.version = configs.version  # 'Fourier' or 'Wavelets'
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        # Text configuration
        self.text_dim = configs.text_dim
        self.text_seq_len = int(np.ceil(configs.pred_len / configs.stride))
        
        # Normalization
        self.revin = getattr(configs, 'revin', True)
        
        # Decomposition
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)
        
        # Embeddings (same as FEDformer)
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in, configs.d_model, configs.embed, 
            configs.freq, configs.dropout
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            configs.dec_in, configs.d_model, configs.embed,
            configs.freq, configs.dropout
        )
        
        # Text Encoder (from TGTSF)
        self.text_encoder = text_encoder(
            cross_layer=configs.cross_layers,
            self_layer=configs.self_layers,
            embedding_dim=configs.text_dim,
            num_heads=configs.n_heads,
            dropout=configs.dropout,
            pred_len=configs.pred_len,
            stride=configs.stride
        )
        
        # FiLM Generators
        # Encoder: Global FiLM (one per encoder layer)
        self.film_gen_enc = nn.ModuleList([
            FiLMGeneratorFEDformer(
                text_dim=configs.text_dim,
                d_model=configs.d_model,
                text_seq_len=self.text_seq_len,
                n_channels=configs.enc_in
            ) for _ in range(configs.e_layers)
        ])
        
        # Decoder: Global FiLM (3 per decoder layer: self, cross, ffn)
        self.film_gen_dec = nn.ModuleList([
            nn.ModuleList([
                FiLMGeneratorFEDformer(...),  # After self-attn
                FiLMGeneratorFEDformer(...),  # After cross-attn
                FiLMGeneratorFEDformer(...)   # After FFN
            ]) for _ in range(configs.d_layers)
        ])
        
        # Output: Per-channel FiLM
        self.film_gen_out = FiLMGeneratorPerChannel(
            text_dim=configs.text_dim,
            text_seq_len=self.text_seq_len
        )
        
        # Encoder with FiLM
        self.encoder = EncoderFilmFED(
            attn_layers=[
                EncoderLayerFilmFED(
                    encoder_self_att,  # FourierBlock or MultiWavelet
                    configs.d_model,
                    configs.d_ff,
                    configs.moving_avg,
                    configs.dropout,
                    configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        
        # Decoder with FiLM
        self.decoder = DecoderFilmFED(
            layers=[
                DecoderLayerFilmFED(
                    decoder_self_att,
                    decoder_cross_att,
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    configs.moving_avg,
                    configs.dropout,
                    configs.activation
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
    
    def forward(self, x, news, channel_description, x_mark_enc, x_mark_dec, **kwargs):
        # 1. RevIN Normalization
        if self.revin:
            x_mean = torch.mean(x, dim=1, keepdim=True)
            x = x - x_mean
            x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = x / torch.sqrt(x_var)
        
        # 2. Text Encoding
        description = channel_description.unsqueeze(1).repeat(1, news.shape[1], 1, 1)
        text_emb = self.text_encoder(news, description)  # [B, L, C, text_dim]
        text_emb_permuted = text_emb.permute(0, 2, 1, 3)  # [B, C, L, text_dim]
        
        # 3. Decomposition for decoder initialization
        mean = torch.mean(x, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        
        # 4. Encoder with FiLM
        enc_out = self.enc_embedding(x, x_mark_enc)
        film_params_enc = [gen(text_emb) for gen in self.film_gen_enc]
        enc_out, _ = self.encoder(enc_out, film_params_enc)
        
        # 5. Decoder with FiLM
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        film_params_dec = [[gen(text_emb) for gen in layer_gens] 
                           for layer_gens in self.film_gen_dec]
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, film_params_dec, trend=trend_init
        )
        
        # 6. Combine seasonal and trend
        prediction = trend_part + seasonal_part
        prediction = prediction[:, -self.pred_len:, :]
        
        # 7. Per-channel output FiLM
        gamma_out, beta_out = self.film_gen_out(text_emb_permuted)  # [B, C, 1]
        gamma_out = gamma_out.permute(0, 2, 1)  # [B, 1, C]
        beta_out = beta_out.permute(0, 2, 1)    # [B, 1, C]
        prediction = (1 + gamma_out) * prediction + beta_out
        
        # 8. RevIN Denormalization
        if self.revin:
            prediction = prediction * torch.sqrt(x_var) + x_mean
        
        return prediction
```

### Implementation Roadmap

#### Phase 1: Core Infrastructure
1. [ ] Create `layers/FEDformer_FiLM_layers.py` with new layer classes
2. [ ] Create `models/lynx_film_raw_fedformer.py` with main model
3. [ ] Rename `models/lynx_film_raw.py` → `models/lynx_film_raw_itransformer.py`
4. [ ] Update model registry to include both variants

#### Phase 2: Frequency-Domain FiLM (Optional Enhancement)
Instead of applying FiLM in time domain after IFFT, consider modulating frequency coefficients directly:

```python
def forward_frequency_film(self, x, gamma_freq, beta_freq):
    # x: [B, L, d_model]
    x_ft = torch.fft.rfft(x, dim=1)  # [B, L//2+1, d_model]
    
    # Modulate frequency coefficients (complex-valued FiLM)
    # gamma_freq, beta_freq: [B, L//2+1, d_model] (complex)
    x_ft_mod = (1 + gamma_freq) * x_ft + beta_freq
    
    # Back to time domain
    return torch.fft.irfft(x_ft_mod, n=x.size(1))
```

This is more aligned with FEDformer's philosophy but requires complex-valued FiLM parameters.

#### Phase 3: Testing & Validation
1. [ ] Unit tests for new layer classes
2. [ ] Integration test with dummy data
3. [ ] Benchmark on ETT datasets
4. [ ] Compare with lynx-film-raw-itransformer

### Configuration Example

```yaml
model: lynx_film_raw_fedformer
model_config_overrides:
  # FEDformer specific
  version: Fourier
  mode_select: random
  modes: 64
  moving_avg: [24]
  label_len: 48
  
  # Architecture
  d_model: 512
  n_heads: 8
  e_layers: 2
  d_layers: 1
  d_ff: 2048
  
  # Text
  text_dim: 256
  cross_layers: 2
  self_layers: 3
  stride: 24
  
  # Normalization
  revin: true
```

### Expected Benefits

1. **Long-Horizon Performance**: FEDformer's O(N) scales better than iTransformer for very long sequences
2. **Periodic Data**: Frequency-domain attention naturally captures periodicities
3. **Decomposition**: Explicit seasonal-trend separation may improve interpretability
4. **Text Integration**: Per-channel FiLM at output allows channel-specific text influence

### Potential Challenges

1. **Channel Mixing**: Early mixing may limit per-channel text effectiveness
2. **Complex FiLM**: Frequency-domain FiLM requires careful implementation
3. **Hyperparameters**: More tuning needed (modes, moving_avg, FiLM placement)

---

## Conclusion

FEDformer represents a significant advance in efficient time series forecasting through frequency-domain operations and seasonal-trend decomposition. While it lacks multimodal capabilities, its O(N) complexity makes it suitable for very long sequences.

TGTSF and Lynx-FiLM-Raw extend the forecasting paradigm to incorporate textual context, with different trade-offs:
- **TGTSF** uses explicit cross-attention for deep text-temporal fusion
- **Lynx-FiLM-Raw (iTransformer)** uses lightweight FiLM conditioning with inverted attention for efficient multivariate modeling
- **Lynx-FiLM-Raw (FEDformer)** (proposed) would combine frequency-domain efficiency with text conditioning

The choice between these models depends on the specific use case: sequence length, availability of text context, number of variables, and computational constraints.

### Model Naming Convention

| Model Name | Base Architecture | Text Integration |
|------------|-------------------|------------------|
| `lynx_film_raw_itransformer` | iTransformer (inverted attention) | FiLM |
| `lynx_film_raw_fedformer` | FEDformer (frequency attention) | FiLM |
| `TGTSF` | PatchTST-style | Cross-attention mixer |

