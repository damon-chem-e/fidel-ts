# Model Architecture Guide

This document provides an intuitive description of each supported model in the fidel-ts framework, organized by their capabilities and design paradigms.

**Table of Contents:**
- [Unimodal Models](#unimodal-models)
- [Multimodal Models](#multimodal-models)
- [Foundation Models](#foundation-models)
- [LLM Socket Models](#llm-socket-models)

---

## Unimodal Models

These models work with time series data only and do not incorporate external text or multimodal information.

### DLinear (Decomposition-Linear)

**Architecture:**
- Simple linear forecasting with seasonal-trend decomposition
- Decomposes input into seasonal and trend components using a moving average
- Each component is processed by a separate linear layer
- Final prediction is the sum of seasonal and trend forecasts

**Multimodal:** No

**Future Information Usage:**
- Does not use future known information
- Could potentially be extended to use future covariates by concatenating them to the input

**LLM Usage:** No

**Key Features:**
- Extremely lightweight and fast
- Individual per-channel forecasting option
- Works well for data with clear seasonal patterns
- O(1) time complexity

---

### iTransformer

**Architecture:**
- Inverted Transformer where attention operates over variables (channels) instead of time steps
- Input: [B, seq_len, channels] → Embed to [B, channels, d_model]
- Attention mechanism treats each channel as a token
- Projects embedded channels to prediction length

**Multimodal:** No

**Future Information Usage:**
- Does not use future known information
- Could be extended to include future covariates as additional "channels"

**LLM Usage:** No

**Key Features:**
- Inverted paradigm: models relationships between variables rather than temporal patterns
- Particularly effective for multivariate forecasting
- Uses Non-stationary Transformer normalization (use_norm)
- Full self-attention O(N²) where N is number of channels

---

### PatchTST

**Architecture:**
- Patch-based Transformer that divides time series into patches (subsequences)
- Patches are embedded and processed by a Transformer encoder
- Optional series decomposition into trend and seasonal components
- Each component processed by separate PatchTST backbone

**Multimodal:** No

**Future Information Usage:**
- Does not use future known information
- Could potentially incorporate future covariates by treating them as additional input channels

**LLM Usage:** No

**Key Features:**
- Patch-based processing reduces computational complexity
- Channel-independent or channel-mixing modes
- RevIN normalization for non-stationarity
- Effective for long-range forecasting
- O(L²) where L is number of patches (much smaller than sequence length)

---

### FEDformer

**Architecture:**
- Frequency Enhanced Decomposed Transformer
- Operates in frequency domain using Fourier or Wavelet transforms
- Each encoder/decoder layer includes:
  - Frequency-domain attention (O(N) complexity)
  - Seasonal-trend decomposition
  - Feed-forward network

**Multimodal:** No

**Future Information Usage:**
- Does not use future known information
- Could incorporate future temporal marks (time features) which are already required

**LLM Usage:** No

**Key Features:**
- Two versions: Fourier-based and Wavelet-based
- O(N) complexity through frequency domain operations
- Seasonal-trend decomposition at each layer
- Encoder-decoder architecture with cross-attention
- **Requires temporal marks** (time features like hour, day, month)

---

### Informer

**Architecture:**
- Efficient Transformer with ProbSparse self-attention
- ProbSparse attention: selects top-k queries based on sparsity scores (O(N log N))
- Self-attention distilling: halves sequence length between encoder layers via convolution
- Generative decoder: uses start tokens + zero placeholders

**Multimodal:** No

**Future Information Usage:**
- Does not use future known information directly
- Uses temporal marks (time features) which are always available

**LLM Usage:** No

**Key Features:**
- ProbSparse attention for O(N log N) complexity
- Self-attention distilling reduces memory footprint
- **Requires temporal marks** (time features)
- Encoder-decoder with cross-attention
- Generative decoding style

---

### FITS (Frequency Interpolation Time Series)

**Architecture:**
- Frequency domain forecasting via interpolation
- Applies FFT to input, keeps only dominant frequencies (low-pass filter)
- Upsamples frequency spectrum using learnable complex linear layer
- Applies inverse FFT to get predictions

**Multimodal:** No

**Future Information Usage:**
- Does not use future known information
- Not naturally compatible with future covariates (operates in frequency domain)

**LLM Usage:** No

**Key Features:**
- Extremely simple and efficient
- Works entirely in frequency domain
- Individual per-channel frequency upsampling option
- RevIN normalization
- Good for periodic/seasonal data

---

## Multimodal Models

These models combine time series data with text information (channel descriptions, news, events, etc.).

### Important: Dataset-Specific Text Information

The source and nature of text information varies significantly between dataset types:

**Fidel-TS Datasets (Bear_room, NYC, Jena, etc.):**
- `channel_description` (hetero_channel): **Real per-sensor/per-channel descriptions**
  - Example: `"Room 104 temperature sensor"`, `"NYC intersection 42nd & Broadway traffic counter"`
  - These are rich, meaningful metadata stored in `static_info['channel_info']`
- `historical_events` (x_hetero) / `news` (y_hetero): **Dynamic time-aligned text**
  - Weather forecasts, event schedules, contextual information
  - Both text sources contain distinct, valuable information

**Time-MMD/TTC Datasets:**
- `channel_description` (hetero_channel): **Generic concatenated strings**
  - Single channel: Just the `channel_info` string (e.g., `"Weather variables"` or `""`)
  - Multi-channel: Generic string + column name (e.g., `"Weather variables: temperature"`)
  - ⚠️ **These are NOT rich descriptive metadata** - just basic labels
- `historical_events` (x_hetero) / `news` (y_hetero): **THE ACTUAL MEANINGFUL TEXT DATA**
  - Text from CSV columns (Final_Search_*, Final_Output, or 'text')
  - Weather descriptions, news articles, contextual information
  - **This is where the rich textual information resides for these datasets**

**Key Implication:** For Time-MMD/TTC datasets, models that use both `channel_description` and `historical_events`/`news` are primarily benefiting from the latter. The `channel_description` provides minimal semantic value compared to Fidel-TS datasets.

### TGTSF (Text-Guided Time Series Forecasting)

**Architecture:**
- Dual encoder architecture:
  - **Time Series Encoder**: Patch-based Transformer with causal attention
  - **Text Encoder**: Cross-attention between text items and channel descriptions, followed by self-attention
- **Text-Temporal Cross-Attention Mixer**: Aligns text and time series features
- Patch reconstruction to generate final predictions

**Multimodal:** Yes (time series + text)

**Future Information Usage:**
- Uses `timestamp_semantics` to control text usage:
  - **t_about**: Uses text that describes the prediction window (e.g., weather forecasts, event schedules). Assumes forecast was known in advance.
  - **t_known**: Uses text from the input window only (avoids lookahead bias for datasets where text timestamp = publication time)

**LLM Usage:** No (uses precomputed text embeddings, not an LLM directly)

**Text/Prompt Construction:**
- **Uses external text information** (does not construct prompts from time series)
- Text inputs come from the dataset's heterogeneous data:
  - **Channel descriptions**: Static text describing each variable/sensor
  - **News/Events**: Time-aligned text items (e.g., weather forecasts, news articles, event schedules)
- Text is embedded **offline** using models like BERT, GPT-2, or other encoders configured in the embedder
- Model receives precomputed embeddings: `[B, num_timesteps, num_text_items, embedding_dim]`
- Text encoder processes these embeddings via:
  1. Cross-attention between text items and channel descriptions (fuses context)
  2. Self-attention among text items (captures relationships)
  3. Output: per-channel, per-timestep text representations

**Dataset-Specific Text Sources:**
- **Fidel-TS datasets (Bear_room, NYC, etc.)**:
  - `channel_description`: Real per-sensor descriptions (e.g., "Room 104 temperature sensor")
  - `historical_events` or `news`: Dynamic time-aligned text (weather reports, events)
  - Both contain rich, meaningful information
- **Time-MMD/TTC datasets**:
  - `channel_description`: Generic string + column name (e.g., "Weather variables: temperature")
    - ⚠️ **Not actual descriptive metadata** - just concatenated strings
  - `historical_events` or `news`: **The actual meaningful text data** from CSV text columns (weather descriptions, news, events)
    - This is where the rich text information resides for these datasets

**Key Features:**
- Flexible text input via timestamp semantics
- Patch-based processing with overlapping reconstruction
- Cross-attention between text and time series
- RevIN normalization
- Learnable projection for different embedding dimensions (e.g., 768D BERT → 256D model)

---

### LYNX (Language-enhanced forecasting with residual learning)

**Architecture:**
- **Frozen Unimodal Baseline**: Pretrained model (default: iTransformer) provides base predictions
- **TGTSF Residual Branch**: Learns the difference between ground truth and baseline
- Final prediction = baseline + residual

**Multimodal:** Yes (time series + text via TGTSF branch)

**Future Information Usage:**
- Same as TGTSF: controlled by `timestamp_semantics`
- **t_about**: Uses prediction window text (forecasts, schedules)
- **t_known**: Uses input window text only

**LLM Usage:** No (uses precomputed text embeddings)

**Text/Prompt Construction:**
- **Identical to TGTSF** - uses external text information, not constructed from time series
- Text inputs: channel descriptions + time-aligned news/events
- Precomputed embeddings from offline encoders (BERT, GPT-2, etc.)
- TGTSF branch processes text via cross-attention with channel descriptions + self-attention
- Learnable projection layer if embedding dimension differs from model dimension

**Dataset-Specific Text Sources:**
- **Fidel-TS datasets (Bear_room, NYC, etc.)**:
  - `channel_description`: Real per-sensor descriptions (e.g., "Room 104 temperature sensor")
  - `historical_events` or `news`: Dynamic time-aligned text (weather reports, events)
  - Both contain rich, meaningful information
- **Time-MMD/TTC datasets**:
  - `channel_description`: Generic string + column name (e.g., "Weather variables: temperature")
    - ⚠️ **Not actual descriptive metadata** - just concatenated strings
  - `historical_events` or `news`: **The actual meaningful text data** from CSV text columns (weather descriptions, news, events)
    - This is where the rich text information resides for these datasets

**Key Features:**
- Residual learning reduces training difficulty
- Frozen baseline ensures stable predictions
- Combines strengths of unimodal and multimodal approaches
- Handles different normalization schemes (RevIN vs use_norm)

---

### LYNX-FiLM

**Architecture:**
- **Frozen Unimodal Baseline**: Pretrained model (default: iTransformer) provides base predictions
- **iTransformer-FiLM Residual Branch**: iTransformer layers modulated by text via FiLM (Feature-wise Linear Modulation)
  - Text encoder (from TGTSF) generates text embeddings
  - FiLM layers use text to compute per-channel scale and shift parameters
  - These parameters modulate iTransformer features: `output = scale * features + shift`
- Final prediction = baseline + residual

**Multimodal:** Yes (time series + text via FiLM)

**Future Information Usage:**
- Same as TGTSF/LYNX: controlled by `timestamp_semantics`
- **t_about**: Uses prediction window text
- **t_known**: Uses input window text only

**LLM Usage:** No (uses precomputed text embeddings)

**Text/Prompt Construction:**
- **Uses external text information** (same as TGTSF/LYNX)
- Text inputs: channel descriptions + time-aligned news/events
- Precomputed embeddings from offline encoders
- **Text Encoder** (TGTSF style): Cross-attention (text × channel descriptions) + self-attention
- Output text embeddings: `[B, num_timesteps, num_channels, text_dim]` (preserves temporal structure)
- **FiLM Integration**: Text embeddings → scale/shift parameters for modulating iTransformer features
  - Unlike TGTSF's cross-attention mixer, FiLM provides channel-wise multiplicative and additive modulation
  - Allows text to dynamically influence feature extraction per channel and timestep

**Dataset-Specific Text Sources:**
- **Fidel-TS datasets (Bear_room, NYC, etc.)**:
  - `channel_description`: Real per-sensor descriptions (e.g., "Room 104 temperature sensor")
  - `historical_events` or `news`: Dynamic time-aligned text (weather reports, events)
  - Both contain rich, meaningful information
- **Time-MMD/TTC datasets**:
  - `channel_description`: Generic string + column name (e.g., "Weather variables: temperature")
    - ⚠️ **Not actual descriptive metadata** - just concatenated strings
  - `historical_events` or `news`: **The actual meaningful text data** from CSV text columns (weather descriptions, news, events)
    - This is where the rich text information resides for these datasets

**Key Features:**
- More sophisticated text integration than LYNX (FiLM modulation vs cross-attention)
- Preserves temporal sequence information (no pooling over time)
- Channel-wise modulation allows per-variable text influence
- Frozen baseline + learned residual

---

### LYNX-FiLM-Raw

**Architecture:**
- Single iTransformer-FiLM model (no baseline + residual)
- Learns the entire prediction from scratch with FiLM modulation
- Text encoder generates text embeddings
- iTransformer layers modulated by text via FiLM

**Multimodal:** Yes (time series + text via FiLM)

**Future Information Usage:**
- Same as TGTSF/LYNX: controlled by `timestamp_semantics`
- **t_about**: Uses prediction window text
- **t_known**: Uses input window text only

**LLM Usage:** No (uses precomputed text embeddings)

**Text/Prompt Construction:**
- **Identical to LYNX-FiLM** - uses external text information
- Text inputs: channel descriptions + time-aligned news/events
- Precomputed embeddings from offline encoders
- Text encoder processes embeddings → FiLM modulation parameters

**Dataset-Specific Text Sources:**
- **Fidel-TS datasets (Bear_room, NYC, etc.)**:
  - `channel_description`: Real per-sensor descriptions (e.g., "Room 104 temperature sensor")
  - `historical_events` or `news`: Dynamic time-aligned text (weather reports, events)
  - Both contain rich, meaningful information
- **Time-MMD/TTC datasets**:
  - `channel_description`: Generic string + column name (e.g., "Weather variables: temperature")
    - ⚠️ **Not actual descriptive metadata** - just concatenated strings
  - `historical_events` or `news`: **The actual meaningful text data** from CSV text columns (weather descriptions, news, events)
    - This is where the rich text information resides for these datasets

**Key Features:**
- Simpler than LYNX-FiLM (no baseline model needed)
- Direct learning vs residual learning
- FiLM-based text integration
- Flexible normalization (RevIN or use_norm)

---

### TimeCMA (Cross-Modality Alignment)

**Architecture:**
- **Dual Encoder Design**:
  - **Time Series Path**: Length-to-feature projection → Transformer encoder
  - **LLM Path**: Frozen precomputed LLM embeddings → Transformer encoder
- **Cross-Modal Alignment**: Cross-attention with time series as queries, LLM embeddings as keys/values
- **Decoder**: Transformer decoder generates predictions

**Multimodal:** Yes (time series + LLM embeddings)

**Future Information Usage:**
- Designed to use channel descriptions (always available)
- Does not explicitly use future known information
- Could potentially be extended to use future text

**LLM Usage:**
- Uses **precomputed LLM embeddings** (e.g., from GPT-2)
- LLM itself is not part of the model (embeddings computed offline)
- Cross-attention aligns time series features with semantic LLM representations

**Text/Prompt Construction:**
- **Uses `historical_events` parameter** (not channel descriptions as parameter name might suggest)
- Text source depends on configuration:
  - **LLM Embedding Provider mode**: Per-sample LLM embeddings computed from dataset text/descriptions
  - **Standard mode**: Time-aligned historical text embeddings from x_hetero
- Embeddings computed **offline** (once per dataset) and cached
- Model receives: `[B, d_llm, num_channels]` embeddings via `historical_events` parameter
- **Not constructing prompts from time series** - uses precomputed text embeddings
- Cross-modal alignment: Time series features (queries) attend to LLM embeddings (keys/values)
  - Allows semantic text information to guide forecasting

**Dataset-Specific Text Sources:**
- **Fidel-TS datasets**:
  - With LLM Embedding Provider: Per-sample embeddings from channel descriptions via LLM
    - Example: "Temperature sensor located in Building A, Room 101, measuring ambient temperature in Celsius"
  - Without LLM Embedding Provider: Dynamic time-aligned text from historical events
- **Time-MMD/TTC datasets**:
  - **The text comes from `historical_events` (x_hetero)**: Embeddings of actual historical text from CSV text columns
    - Weather descriptions, news articles, contextual information
    - This is the **meaningful text data** for these datasets
  - ⚠️ **Not using channel descriptions** - those are just generic strings for these datasets

**Key Features:**
- RevIN normalization
- Cross-modal alignment via attention
- Frozen LLM embeddings (computed once)
- Suitable for scenarios with rich channel metadata

---

### GPT4MTS (GPT for Multivariate Time Series with text prompts)

**Architecture:**
- Time series → Patching → Linear projection to d_model
- **Text Prompts**: Uses precomputed text embeddings as "prompt tokens"
- **Frozen GPT-2 Backbone**: Processes concatenated [prompt_tokens, time_series_patches]
- Output projection: flattened patch representations → predictions

**Multimodal:** Yes (time series + text prompts)

**Future Information Usage:**
- Uses text prompts which could describe context
- Typically uses historical/contextual text, not future information

**LLM Usage:**
- Uses **frozen GPT-2** layers for processing
- Only training: prompt projection layer and output layer
- LLM provides semantic understanding

**Text/Prompt Construction:**
- **Uses external text information** as prompt embeddings
- Text inputs: Precomputed embeddings from historical events/context (x_hetero)
  - Example: News articles, event descriptions, weather reports aligned to input window
- **Processing pipeline**:
  1. Text embeddings (computed offline): `[B, d_llm, num_text_items]` (typically 768D from GPT-2/BERT)
  2. Patching: Embeddings divided into patches matching time series patch structure
  3. Pooling: Mean over patch dimension → `[B, num_patches, d_model]`
  4. Prompt projection: Linear layer maps LLM embeddings to GPT-2's d_model dimension
  5. Concatenation: `[prompt_embeddings, time_series_patches]` → fed to GPT-2
- **Not constructing prompts from time series** - uses external text
- GPT-2 processes the combined sequence autoregressively
- Only prompt projection and output layers are trainable

**Dataset-Specific Text Sources:**
- **Fidel-TS datasets**: Historical events/context from x_hetero (weather reports, events)
- **Time-MMD/TTC datasets**: Uses `historical_events` (x_hetero) which contains **the actual meaningful text data** from CSV text columns (weather descriptions, news, contextual information)

**Key Features:**
- Leverages pretrained LLM representations
- Minimal trainable parameters (prompt + output layers)
- RevIN normalization
- Effective for datasets with text context

---

### TimeLLM

**Architecture:**
- **Patch Embedding**: Time series → patches → learned embeddings
- **Dynamic Prompts**: Per-batch statistics (mean, std, lags, trends) → text prompts
- **Reprogramming Layer**: Cross-attention between patch embeddings and reduced LLM vocabulary
- **Frozen LLM**: Processes reprogrammed patches
- **Output Projection**: LLM hidden states → predictions

**Multimodal:** Yes (implicitly - uses LLM for semantic reasoning)

**Future Information Usage:**
- Does not use explicit future information
- Dynamic prompts describe historical statistics only

**LLM Usage:**
- Uses **frozen LLM** (GPT-2, Llama, QWEN, etc.) for processing
- Reprogramming layer maps time series patches to LLM embedding space
- LLM provides semantic reasoning without fine-tuning

**Text/Prompt Construction:**
- **Constructs prompts FROM time series statistics** (not external text)
- **Dynamic prompt generation** (computed per batch during forward pass):
  
  ```
  <|start_prompt|>Dataset description: {description}
  Task description: forecast the next {pred_len} steps given the 
  previous {seq_len} steps information;
  Input statistics: min value {min}, max value {max}, 
  median value {median}, the trend of input is {upward/downward},
  top 5 lags are: {lag1, lag2, lag3, lag4, lag5}<|end_prompt|>
  ```

- **Statistics computed from input time series**:
  - Min, max, median values
  - Trend direction (computed from first vs last value comparison)
  - Top-k autocorrelation lags (via FFT-based lag detection)
  - Dataset description (optional, from config)

- **Per-sample prompts**: Each sample-channel pair gets its own prompt with its statistics
- Prompts tokenized → LLM embedding layer → prompt embeddings
- **Reprogramming**: Time series patches cross-attend to reduced LLM vocabulary
  - Maps patches from TS space to LLM semantic space
- LLM processes: `[prompt_embeddings, reprogrammed_patches]`
- **No external text** - entirely self-contained from time series

**Key Features:**
- Dynamic prompt generation (not static)
- Frozen LLM backbone (no gradient updates)
- Reprogramming layer enables TS-to-LLM mapping
- RevIN normalization
- Efficient: only patch embedding, reprogramming, and output projection are trained

---

### LeRet (Language-Enhanced Retention Network)

**Architecture:**
- **RetNet Backbone**: Multi-scale retention mechanism (O(N) complexity)
- **Optional Language Integration**: Cross-attention with text embeddings from fidel-ts embedder
- **Dual Heads**:
  - Patch head: for auto-regressive pretraining (Stage 1)
  - Sequence head: for forecasting (Stage 2)
- Supports two-stage training

**Multimodal:** Optional (can work with or without language)

**Future Information Usage:**
- Does not use future known information
- Language integration uses channel descriptions or contextual text

**LLM Usage:**
- Can optionally use **precomputed LLM embeddings** via cross-attention
- Does not directly call an LLM

**Text/Prompt Construction:**
- **Optional - uses external text information when enabled**
- When `use_language=True`:
  - Text inputs: Channel descriptions and/or contextual text
  - Precomputed embeddings from offline encoders
  - **Language Integrator module**: Cross-attention between RetNet features (queries) and text embeddings (keys/values)
  - Allows semantic information to guide retention mechanism
- When `use_language=False`: Pure time series model (no text)
- **Not constructing prompts from time series** - uses precomputed text embeddings if available

**Dataset-Specific Text Sources:**
- **Fidel-TS datasets (Bear_room, NYC, etc.)**:
  - `channel_description`: Real per-sensor descriptions (e.g., "Room 104 temperature sensor")
  - `historical_events`: Dynamic time-aligned text (weather reports, events)
  - Both contain rich, meaningful information
- **Time-MMD/TTC datasets**:
  - `channel_description`: Generic string + column name (e.g., "Weather variables: temperature")
    - ⚠️ **Not actual descriptive metadata** - just concatenated strings
  - `historical_events`: **The actual meaningful text data** from CSV text columns (weather descriptions, news, events)
    - This is where the rich text information resides for these datasets

**Key Features:**
- RetNet for O(N) complexity (vs Transformer's O(N²))
- Two-stage training (pretrain + finetune)
- Optional language enhancement
- Patch-based processing
- RevIN normalization

---

### ZhangHanBest (Multimodal Fusion)

**Architecture:**
- **Plugin-based architecture** - wraps any supported unimodal model as the TS encoder
- Time series encoder extracts aggregated representations `[B, d_model]`
- Text embeddings (precomputed) aggregated to `[B, text_dim]`
- **Residual Projection**: Projects text embeddings to TS representation space
- **Late Fusion**: Weighted addition of text and TS representations
- **Prediction Head**: Fused representation → predictions for all channels

**Multimodal:** Yes (time series + text)

**Future Information Usage:**
- Uses aggregated text embeddings (dataset descriptions, summaries)
- Does not explicitly use future known information
- Could potentially be extended with future text embeddings

**LLM Usage:**
- Uses **precomputed LLM embeddings** (offline)
- Embeddings computed once per dataset and cached
- No runtime LLM calls

**Text/Prompt Construction:**
- **Uses external text information** - typically dataset-level descriptions
- Text embeddings aggregated from:
  - Dataset general descriptions (hetero_general)
  - Or dynamic aggregate text (dataset_description)
- Precomputed embeddings: `[B, text_dim]` (typically 768D from BERT/GPT-2)
- **Not constructing prompts from time series** - uses external metadata

**Dataset-Specific Text Sources:**
- **Fidel-TS datasets**: Dataset-level descriptions (general_info)
- **Time-MMD/TTC datasets**: Aggregated embeddings from general_info (typically generic dataset description)

**Supported Unimodal Models (Plugins):**
1. **PatchTST** - Patch-based Transformer
2. **DLinear** - Decomposition-Linear (with automatic projection layer)
3. **Sundial** - Foundation model (with automatic projection layer)
4. **TimeMoE** - Mixture-of-Experts foundation model (with automatic projection layer)

**Plugin Configuration:**
- Set `unimodal_model_type` in config (default: 'PatchTST')
- Model automatically handles dimension mismatches with projection layers
- TS encoder must support `return_representations=True` for extracting features

**Key Features:**
- Flexible plugin architecture - easy to add new unimodal models
- Representation-level fusion (not prediction-level)
- Residual projection for text-to-TS alignment
- Configurable fusion weight (fixed or learned)
- Automatic handling of different normalization schemes (RevIN, use_norm)
- Can use foundation models (Sundial, TimeMoE) as backbone

---

### MMTSFlib (Prediction-Level Late Fusion)

**Architecture:**
- **Plugin-based architecture** - wraps any supported unimodal model
- Time series model produces predictions `[B, pred_len, C]`
- Text embeddings → MLP projection → `[B, pred_len, 1]`
- Text pooling (avg/max/min/attention)
- Instance normalization
- **Late Fusion**: `(1-w)*ts_pred + w*text_pred` (prediction-level ensemble)
- Optional prior history integration

**Multimodal:** Yes (time series + text)

**Future Information Usage:**
- Uses precomputed text embeddings
- Does not explicitly use future known information
- Could incorporate future text embeddings in theory

**LLM Usage:**
- Uses **precomputed LLM embeddings** (offline)
- Embeddings computed via LLMEmbeddingProvider
- No runtime LLM calls

**Text/Prompt Construction:**
- **Uses external text information** - LLM embeddings of dataset context
- Text embeddings from `historical_events` (x_hetero in dataloader)
- Precomputed via LLMEmbeddingProvider: `[B, L, text_dim]` or `[B, text_dim]`
- Supports various pooling strategies for aggregation
- **Not constructing prompts from time series** - uses precomputed LLM embeddings

**Dataset-Specific Text Sources:**
- **Fidel-TS datasets**: Can use general descriptions or dynamic text embeddings
- **Time-MMD/TTC datasets**: Uses `historical_events` which contains **the actual meaningful text data** from CSV text columns (weather descriptions, news, events)

**Supported Unimodal Models (Plugins):**
1. **PatchTST** - Patch-based Transformer
2. **DLinear** - Decomposition-Linear
3. **iTransformer** - Inverted Transformer
4. **FEDformer** - Frequency Enhanced Decomposed Transformer
5. **Informer** - Efficient Transformer with ProbSparse attention
6. **FITS** - Frequency Interpolation Time Series
7. **Autoformer** - Decomposition Transformer (encoder-decoder)

**Plugin Configuration:**
- Set `unimodal_model_type` in config (default: 'iTransformer')
- Automatically handles simple models vs encoder-decoder models
- No special interface required - just standard forward() returning predictions

**Key Features:**
- Prediction-level fusion (different from ZhangHanBest's representation-level)
- Works with broader range of unimodal models (7 supported)
- Multiple pooling strategies (avg, max, min, attention)
- Optional prior history integration
- Configurable mixing weight (fixed or learned)
- Supports both simple and encoder-decoder architectures
- Very flexible - easy to add new unimodal models

---

## Comparison: ZhangHanBest vs MMTSFlib

| Feature | ZhangHanBest | MMTSFlib |
|---------|--------------|----------|
| **Fusion Level** | Representation-level | Prediction-level |
| **Text Integration** | Residual projection | MLP projection |
| **Fusion Method** | Weighted addition of features | Weighted addition of predictions |
| **Supported Models** | 4 (PatchTST, DLinear, Sundial, TimeMoE) | 7 (PatchTST, DLinear, iTransformer, FEDformer, Informer, FITS, Autoformer) |
| **Foundation Models** | ✅ Yes (Sundial, TimeMoE) | ❌ No |
| **Encoder-Decoder** | ❌ No | ✅ Yes (Informer, FEDformer, Autoformer) |
| **Interface Required** | `return_representations=True` | Standard `forward()` |
| **Complexity** | Higher (representation fusion) | Lower (prediction fusion) |
| **Best For** | Deep feature integration | Simple ensemble |

---

## Foundation Models

These are large-scale pretrained models that can be used zero-shot or with minimal fine-tuning.

### Chronos

**Architecture:**
- Pretrained encoder-decoder Transformer model
- Trained on large-scale time series corpus
- Quantization-based approach: discretizes time series into tokens
- Autoregressive generation in token space

**Multimodal:** No

**Future Information Usage:**
- Does not use future known information (zero-shot forecasting)
- Cannot currently incorporate future covariates (pretrained architecture)

**LLM Usage:**
- Based on T5/Transformer architecture (similar to LLMs but for time series)
- **Pretrained on time series data** (not language)

**Key Features:**
- Zero-shot forecasting (no training needed)
- Multiple model sizes available (mini, small, base, large)
- Probabilistic forecasting (generates multiple samples)
- Quantization to discrete tokens
- Ideal for quick baselines or when training data is limited

---

### TimeMoE

**Architecture:**
- Mixture-of-Experts (MoE) architecture
- Multiple expert networks specialized for different patterns
- Gating network routes inputs to appropriate experts
- Pretrained on large-scale time series data

**Multimodal:** No

**Future Information Usage:**
- Does not use future known information (zero-shot)
- Cannot incorporate future covariates (pretrained)

**LLM Usage:**
- Based on MoE Transformer architecture
- **Pretrained on time series data**
- Similar to LLM architecture but for time series

**Key Features:**
- Zero-shot or few-shot forecasting
- Mixture-of-Experts for diverse patterns
- Instance-wise normalization (mean/std)
- Autoregressive generation
- Loaded from HuggingFace with `trust_remote_code=True`

---

### Sundial (Pretrained Foundation Model)

**Architecture:**
- Large-scale pretrained foundation model for time series
- Details depend on specific model variant

**Multimodal:** Typically No (time series focused)

**Future Information Usage:**
- Does not use future known information (zero-shot)

**LLM Usage:**
- No (time series foundation model)

**Key Features:**
- Zero-shot or few-shot forecasting
- Pretrained on diverse time series datasets
- Flexible architecture

---

### ChatTime

**Architecture:**
- Uses Llama language model for time series forecasting
- **Discretization**: Converts continuous time series to discrete bins
- **Serialization**: Converts numerical sequences to text format
- **Text Generation**: LLM generates predictions as text
- **Deserialization**: Converts text back to numerical values

**Multimodal:** Yes (time series → text → LLM)

**Future Information Usage:**
- Does not use explicit future information
- Could theoretically incorporate future context in prompts

**LLM Usage:**
- Uses **frozen Llama model** for generation
- Time series treated as text sequences
- Sampling-based generation (top-k, top-p)

**Text/Prompt Construction:**
- **Hybrid: Constructs text FROM time series + uses external context**
- **Processing pipeline**:
  
  1. **Discretization**: Continuous values → discrete bins (e.g., 0-100 bins)
  2. **Serialization**: Discrete sequence → text format
     - Example: `[45, 47, 46, 48] → "45, 47, 46, 48"`
  
  3. **Prompt template** (instruction-following format):
     ```
     Below is an instruction that describes a task. 
     Write a response that appropriately completes the request.
     
     ### Instruction:
     Please predict the following sequence carefully. 
     Context knowledge you may consider: {context}
     
     ### Input:
     {serialized_time_series}
     
     ### Response:
     {model_generates_here}
     ```
  
  4. **Context includes**:
     - Dataset general description (hetero_general)
     - Channel-specific information (hetero_channel)
     - External events/text (batch_y_hetero) - e.g., weather, news
  
  5. **LLM generates** text predictions
  6. **Deserialization**: Text → discrete → continuous values

- **Both time series AND external text** used:
  - Time series converted to text for LLM input
  - External metadata/events added as context
  
- **Iterative generation** for long horizons (limited by max_pred_len per iteration)
- Multiple samples generated for each prediction → median aggregation

**Key Features:**
- Novel approach: treats time series as language
- Discretization + serialization pipeline
- Multiple samples for uncertainty estimation
- Iterative prediction for long horizons
- Computationally expensive (full LLM generation)

---

## LLM Socket Models

These models directly query LLMs via API calls for predictions.

### LLM-Socket (Time-R1, DeepSeek-R1, Qwen, etc.)

**Architecture:**
- Not a neural network - direct API interface to LLMs
- **Prompt Engineering**: Constructs prompts with:
  - System prompt (task description, dataset info)
  - Historical time series as table
  - Historical events/text as table
  - Future events/text (if t_about semantics)
  - Channel descriptions
- **LLM Response**: Parses JSON output with predictions
- **Retry Logic**: Handles format errors, timeouts, misalignments

**Multimodal:** Yes (time series + text → LLM)

**Future Information Usage:**
- **Highly flexible** - can use any future known information in prompts
- Depends on prompt template and dataset configuration
- Can use future events, schedules, forecasts, etc.

**LLM Usage:**
- **Direct LLM API calls** (OpenAI, DeepSeek, Qwen, etc.)
- Streaming or non-streaming modes
- Temperature, seed control
- Chain-of-thought reasoning (for reasoning models like R1)

**Text/Prompt Construction:**
- **Most sophisticated prompting** - uses both time series AND external text extensively
- **Prompt template structure** (loaded from `.template` files):

  **System Prompt:**
  ```
  You are a professional data analyst. You can do forecasting by 
  considering the historical time series readings along with weather 
  condition and make predictions for the following period of time. 
  This dataset is about {dataset_info}. You can consider the 
  periodicity, trend, and seasonality of the data...
  ```

  **User Prompt (Multimodal example):**
  ```
  We now predict the sensor reading for {channel_info}. 
  The historical data is as follows:
  
  [(timestamp1, value1), (timestamp2, value2), ...]
  
  The historical weather condition is as follows:
  
  [(timestamp1, weather_text1), (timestamp2, weather_text2), ...]
  
  We now have the weather forecasting for the following days:
  
  [(future_timestamp1, forecast_text1), ...]
  
  Please format your forecasting in json format in 
  [forecast_timestamp, predicted_value] pair...
  ```

- **Components included**:
  1. **Time series as table**: Historical values with timestamps
  2. **Historical events (x_hetero)**: Time-aligned text from input window
  3. **Future events (y_hetero)**: Time-aligned text for prediction window (if available)
  4. **Channel descriptions**: Metadata about what's being measured
  5. **Dataset info**: High-level dataset description
  6. **Output format specification**: JSON structure with timestamps

- **Template variants**:
  - `general_template_MM`: Multimodal (with text context)
  - `general_template_UM`: Unimodal (time series only)
  - `TimeR1_template`: Specialized for reasoning models (includes `<think>` tags)

- **Reasoning model prompts** (Time-R1, DeepSeek-R1):
  ```
  You must first conduct reasoning inside <think>...</think>.
  When you have the final answer, output inside <answer>...</answer>.
  ```

- **Response parsing**:
  - Extracts JSON from model output
  - Validates timestamp alignment
  - Retry logic if format incorrect or timestamps misaligned

- **Maximum flexibility**: Can incorporate arbitrary context, unlike fixed-architecture models

**Key Features:**
- Zero training - pure prompt engineering
- Extremely flexible (can incorporate any context)
- Supports reasoning models (DeepSeek-R1, Time-R1)
- Automatic retry with format correction
- Handles temporal alignment checking
- Can incorporate unlimited context (within token limits)
- Computationally expensive (API costs + latency)

**Variants:**
- **Time-R1**: Reasoning model specialized for time series
- **DeepSeek-R1**: General reasoning model
- **Qwen**: Strong instruction-following model (various sizes)

---

## Model Comparison Summary

| Model | Multimodal | Uses Future Info | LLM Usage | Complexity | Best For |
|-------|-----------|------------------|-----------|------------|----------|
| DLinear | No | No | No | O(1) | Fast baseline, seasonal data |
| iTransformer | No | No | No | O(N²) channels | Multivariate relationships |
| PatchTST | No | No | No | O(L²) patches | Long sequences |
| FEDformer | No | No | No | O(N) | Frequency patterns |
| Informer | No | No | No | O(N log N) | Efficient long sequences |
| FITS | No | No | No | O(N log N) | Periodic/seasonal |
| TGTSF | Yes | Configurable | Embeddings | Medium | Text-guided forecasting |
| LYNX | Yes | Configurable | Embeddings | Medium | Residual learning with text |
| LYNX-FiLM | Yes | Configurable | Embeddings | Medium | FiLM-modulated residual |
| LYNX-FiLM-Raw | Yes | Configurable | Embeddings | Medium | Direct FiLM learning |
| TimeCMA | Yes | No | Embeddings | Medium | Cross-modal alignment |
| GPT4MTS | Yes | No | Frozen GPT-2 | Medium | Text prompts + GPT-2 |
| TimeLLM | Yes | No | Frozen LLM | High | LLM reprogramming |
| LeRet | Optional | No | Optional Embeddings | O(N) | Efficient + language |
| ZhangHanBest | Yes (plugin) | No | Embeddings | Varies | Representation fusion, plugin arch |
| MMTSFlib | Yes (plugin) | No | Embeddings | Varies | Prediction fusion, plugin arch |
| Chronos | No | No | Pretrained | N/A | Zero-shot baseline |
| TimeMoE | No | No | Pretrained | N/A | Zero-shot MoE |
| ChatTime | Yes | No | Frozen Llama | Very High | Language-as-forecasting |
| LLM-Socket | Yes | Highly Flexible | Direct API | Varies | Maximum flexibility |

---

## Future Information Usage Explained

**Can use future information:**
- LLM-Socket models: Highly flexible, can include any future known information in prompts

**Configurable via `timestamp_semantics`:**
- TGTSF, LYNX, LYNX-FiLM, LYNX-FiLM-Raw:
  - `t_about`: Uses text describing prediction window (assumes forecasts known in advance)
  - `t_known`: Uses only historical text (safe, no lookahead)

**Do not use future information but could be extended:**
- Most unimodal models (DLinear, iTransformer, PatchTST, etc.)
- Could add future covariates as additional input channels or features

**Cannot use future information (pretrained):**
- Chronos, TimeMoE, Sundial: Fixed architectures, zero-shot only

---

## LLM Usage Patterns

**Direct LLM Calls (API):**
- LLM-Socket: Queries LLM via API for each prediction

**Frozen LLM Backbone:**
- TimeLLM: Frozen LLM processes reprogrammed time series patches
- GPT4MTS: Frozen GPT-2 processes prompted time series
- ChatTime: Frozen Llama generates text-formatted predictions

**Precomputed LLM Embeddings:**
- TGTSF, LYNX, LYNX-FiLM, TimeCMA, LeRet: Use embeddings computed offline
- Embeddings generated by: BERT, GPT-2, Llama, or other encoders

**No LLM:**
- DLinear, iTransformer, PatchTST, FEDformer, Informer, FITS
- Pure time series models

---

## Recommendations by Use Case

**Need fast baseline:**
→ DLinear, FITS

**Long-term forecasting:**
→ PatchTST, FEDformer, iTransformer

**Multivariate relationships:**
→ iTransformer, TimeCMA

**Have text metadata:**
→ TGTSF, LYNX-FiLM, TimeLLM, ZhangHanBest, MMTSFlib

**Have future known information:**
→ LLM-Socket (most flexible), TGTSF family (t_about mode)

**Zero-shot forecasting:**
→ Chronos, TimeMoE

**Maximum flexibility:**
→ LLM-Socket (arbitrary context, reasoning, etc.)

**Computational constraints:**
→ DLinear (fastest), LeRet (O(N) complexity)

**Research/experimentation:**
→ LYNX-FiLM-Raw (clean multimodal architecture)

**Want to combine existing unimodal model with text:**
→ **ZhangHanBest** (representation-level fusion, 4 models supported)
→ **MMTSFlib** (prediction-level fusion, 7 models supported, simpler)

**Need to use foundation models with text:**
→ **ZhangHanBest** (supports Sundial, TimeMoE as backbone)

**Ensemble unimodal predictions with text:**
→ **MMTSFlib** (late fusion at prediction level)

**Deep multimodal feature integration:**
→ **ZhangHanBest** (representation-level fusion with residual projection)

---

## Text and Prompting Strategies Summary

Models use three distinct strategies for incorporating text/LLM information:

### 1. External Text Embeddings (Precomputed Offline)

**Models:** TGTSF, LYNX, LYNX-FiLM, LYNX-FiLM-Raw, TimeCMA, GPT4MTS, LeRet (optional)

**How it works:**
- Text information (channel descriptions, news, events) embedded **offline** using encoders (BERT, GPT-2, etc.)
- Embeddings cached and loaded during training
- Model receives precomputed embeddings as input
- **No prompt construction** - uses external text metadata

**Advantages:**
- Computationally efficient (embed once, use many times)
- Can use any text encoder
- Separate embedding and forecasting stages

**Limitations:**
- Text must be pre-embedded (cannot dynamically change prompts)
- Limited to predefined text sources in dataset

---

### 2. Dynamic Prompts from Time Series Statistics

**Models:** TimeLLM

**How it works:**
- Computes statistics from input time series **during forward pass**
- Statistics: min, max, median, trend, autocorrelation lags
- Constructs natural language prompts describing these statistics
- LLM processes prompts alongside reprogrammed time series patches
- **Self-contained** - no external text needed

**Example Prompt:**
```
Dataset description: Electricity Transformer Temperature
Task description: forecast the next 96 steps given the previous 96 steps
Input statistics: min value -1.2, max value 2.4, median value 0.3,
the trend of input is upward, top 5 lags are: 24, 48, 72, 96, 168
```

**Advantages:**
- No external text required
- Adapts to each sample dynamically
- Leverages LLM's semantic understanding of statistics

**Limitations:**
- Cannot incorporate external context (news, events, etc.)
- Limited to statistical descriptions

---

### 3. Time Series Converted to Text + External Context

**Models:** ChatTime

**How it works:**
- **Discretization**: Continuous values → discrete bins
- **Serialization**: Sequence → text format (e.g., "45, 47, 46, 48...")
- Constructs instruction-following prompts with:
  - Serialized time series as input
  - External context (dataset info, channel descriptions, events)
- LLM generates text predictions
- **Deserialization**: Text → numerical values

**Advantages:**
- Treats time series as language (novel approach)
- Can incorporate external context
- Leverages full LLM capabilities

**Limitations:**
- Computationally very expensive
- Quantization errors from discretization
- Long generation times for long horizons

---

### 4. Elaborate Multi-Modal Prompts with Tables

**Models:** LLM-Socket (Time-R1, DeepSeek-R1, Qwen, etc.)

**How it works:**
- Constructs comprehensive prompts using templates
- Includes:
  - System instructions (task, dataset description)
  - Historical time series as timestamp-value tables
  - Historical events/text as timestamp-text tables
  - Future events/forecasts (if available)
  - Channel metadata
  - Output format specification (JSON)
- LLM generates structured predictions
- Parsing and validation with retry logic

**Example Components:**
```
System: You are a professional data analyst...
User: Predict sensor reading for {channel}.
Historical data: [(2024-01-01 00:00, 45.2), (2024-01-01 01:00, 46.1), ...]
Historical weather: [(2024-01-01 00:00, "Sunny, 20°C"), ...]
Weather forecast: [(2024-01-02 00:00, "Cloudy, 18°C"), ...]
Output JSON format: [[timestamp, value], ...]
```

**Advantages:**
- **Maximum flexibility** - can include any context
- Natural language reasoning (especially with R1 models)
- No training required
- Can use future information naturally

**Limitations:**
- API costs per prediction
- Slow (seconds per sample)
- Requires prompt engineering
- Output parsing can fail (needs retry logic)

---

## Prompting Strategy Comparison

| Strategy | Models | External Text | Constructs from TS | Dynamic | Flexibility | Cost |
|----------|---------|---------------|-------------------|---------|-------------|------|
| **Precomputed Embeddings** | TGTSF, LYNX, TimeCMA, GPT4MTS | ✅ Required | ❌ | ❌ Static | Low | Low (once) |
| **Dynamic Stats Prompts** | TimeLLM | ❌ None | ✅ Statistics | ✅ Per-batch | Medium | Low |
| **TS as Text** | ChatTime | ✅ Optional | ✅ Full series | ✅ Per-sample | Medium | Very High |
| **Elaborate Prompts** | LLM-Socket | ✅ Optional | ✅ Tables | ✅ Per-sample | Very High | Very High |

---

## When to Use Each Strategy

**Use Precomputed Embeddings when:**
- Have rich text metadata (channel descriptions, news)
- Want computational efficiency
- Training neural models
- Text sources are fixed

**Use Dynamic Stats Prompts when:**
- Don't have external text
- Want self-contained model
- LLM should understand data characteristics
- Need efficient inference

**Use TS-as-Text when:**
- Want to leverage LLM's full capabilities
- Exploring novel approaches
- Have computational budget
- Research/experimentation

**Use Elaborate Prompts when:**
- Need maximum flexibility
- Have complex context (multiple text sources, future info)
- Zero-shot/few-shot scenarios
- Can afford API costs
- Need reasoning capabilities (R1 models)

---

## Notes on Future Development

Several models could be extended to better support future known information:

1. **Explicit Future Covariate Channels**: Most unimodal models (iTransformer, PatchTST, etc.) could concatenate future known variables as additional input channels

2. **Future-Aware Text Embeddings**: Multimodal models could incorporate separate text encoders for "future context" vs "historical context"

3. **Conditional Generation**: Foundation models could be fine-tuned with future covariates as conditioning information

4. **Hybrid Approaches**: Combine frozen pretrained models with learnable future covariate modules

5. **Hybrid Prompting**: Combine precomputed embeddings with dynamic prompts (e.g., TimeLLM approach with external text)

The current framework's most flexible approach for incorporating arbitrary future information is through LLM-Socket models, which can be prompted with any context.
