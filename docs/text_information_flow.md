# Text Information Flow in fidel-ts

This document describes how different types of text information flow through the fidel-ts codebase, from dataloaders to models (specifically `TGTSF` and `lynx_film_raw`).

## Overview

fidel-ts distinguishes between **four types of text information**:

| Type | Temporal Nature | Description | Example |
|------|----------------|-------------|---------|
| **Channel Descriptions** | Static | Per-channel/sensor descriptions that remain constant | "Temperature sensor in room 104" |
| **General Information** | Static | Dataset-level description constant for all samples | "Bear room building sensor network data" |
| **Known Planned Future Events** | Dynamic | Text information about the prediction window (future) | Weather forecasts for next 24 hours |
| **Historical Text News** | Dynamic | Text information aligned to historical timestamps | Past weather conditions, news articles |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA CONFIGURATION                                   │
│   (data_configs/*.yaml)                                                          │
│                                                                                   │
│   ┌─────────────────────┐     ┌─────────────────────────────────────────────┐    │
│   │  Time Series Data   │     │              hetero_info:                    │    │
│   │  root_path: ...     │     │  static_path: static_info_embeddings.pkl    │    │
│   │  formatter: {i}.csv │     │  formatter: dynamic_embeddings_YYYY.pkl     │    │
│   └─────────────────────┘     └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA PROVIDER                                        │
│   (data_provider/data_factory.py)                                                │
│                                                                                   │
│   ┌────────────────────────────────────────────────────────────────────────┐     │
│   │  Heterogeneous_Dataset (for Fidel-TS)  OR  TimeMMD_HeteroGetter        │     │
│   │                                                                         │     │
│   │  Loads:                                                                 │     │
│   │  - static_data['general_info']      → General information (static)     │     │
│   │  - static_data['channel_info'][id]  → Channel descriptions (static)    │     │
│   │  - static_data['downtime_prompt']   → Downtime text (static)           │     │
│   │  - embeddings[timestamp]            → Dynamic text per timestamp       │     │
│   │                                                                         │     │
│   │  Creates: hetero_data_getter(timestamps) callable                       │     │
│   └────────────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              UNIVERSAL DATASET                                    │
│   (data_provider/data_loader.py)                                                 │
│                                                                                   │
│   __getitem__(index) returns:                                                    │
│   ┌─────────────────────────────────────────────────────────────────────────┐    │
│   │  sample_id     │ Unique identifier for the sample                       │    │
│   │  seq_x         │ Input time series [seq_len, channels]                  │    │
│   │  seq_y         │ Target time series [pred_len, channels]                │    │
│   │  x_time        │ Input timestamps                                       │    │
│   │  y_time        │ Target timestamps (prediction window)                  │    │
│   │  x_hetero      │ Historical text data [seq_len, num_items, embed_dim]   │    │
│   │  y_hetero      │ Future text data [pred_len, num_items, embed_dim]      │    │
│   │  hetero_x_time │ Timestamps for x_hetero                                │    │
│   │  hetero_y_time │ Timestamps for y_hetero                                │    │
│   │  hetero_general│ General info embedding [1, embed_dim]                  │    │
│   │  hetero_channel│ Channel descriptions [num_channels, embed_dim]         │    │
│   └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              MODEL FORWARD                                        │
│   (models/TGTSF.py, models/lynx_film_raw.py)                                     │
│                                                                                   │
│   forward(x, news, channel_description, **kwargs)                                │
│   ┌─────────────────────────────────────────────────────────────────────────┐    │
│   │  x                  ← seq_x (time series input)                         │    │
│   │  news               ← y_hetero (future text/events during pred window)  │    │
│   │  channel_description← hetero_channel (static channel descriptions)      │    │
│   └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Flow for Each Text Type

### 1. Channel Descriptions (Static)

**What it is:** Text descriptions for each channel/sensor in the dataset, describing what that channel measures.

**Source Files:**
- **Fidel-TS datasets:** Stored in `static_info.json` or `static_info_embeddings.pkl`
  - JSON format: `{"channel_info": {"104": "Room 104 temperature sensor", "107": "Room 107 CO2 sensor", ...}}`
  - Embedding format: Pre-computed embeddings of the above text

- **Time-MMD/TTC datasets:** Configured via `channel_info` in data config YAML
  - Single channel: Simple string like `"Weather variables"`
  - Multiple channels: Each channel gets a unique embedding combining `channel_info` with column name

**Flow Path:**
```
static_info_embeddings.pkl → Heterogeneous_Dataset.load_embedding()
    → self.static_data['channel_info']
    → Heterogeneous_Dataset.init_hetero_data(id)
        → get_hetero_data() returns (matched_times, general_info, channel_info, output_dynamic)
    → Universal_Dataset.__getitem__() stores as hetero_channel
    → DataLoader collates to [B, C, embed_dim]
    → Model.forward(channel_description=...) receives [B, C, embed_dim]
```

**In Model Forward:**
```python
# TGTSF.py / lynx_film_raw.py
def forward(self, x, news, channel_description, **kwargs):
    # channel_description: [B, C, embed_dim] or [B, 1, C, embed_dim]
    
    # Expand to match news time dimension
    if len(channel_description.shape) == 3:
        channel_description = channel_description.unsqueeze(1)  # [B, 1, C, D]
    description = channel_description.repeat(1, news.shape[1], 1, 1)  # [B, L, C, D]
    
    # Cross-attend channel descriptions with news
    text_emb = self.text_encoder(news, description)  # [B, L, C, D]
```

---

### 2. General Information (Static)

**What it is:** Dataset-level description that applies to all samples (e.g., data source, geographic location).

**Source Files:**
- **Fidel-TS datasets:** Stored in `static_info.json` under `"general_info"` key
  - Example: `{"general_info": "Building energy monitoring system in Bear Research Institute"}`

- **Time-MMD/TTC datasets:** Configured via `general_info` in data config YAML

**Flow Path:**
```
static_info_embeddings.pkl → Heterogeneous_Dataset.load_embedding()
    → self.static_data['general_info']
    → Heterogeneous_Dataset.init_hetero_data(id)
        → get_hetero_data() returns (matched_times, general_info, channel_info, output_dynamic)
    → Universal_Dataset.__getitem__() stores as hetero_general
    → DataLoader collates to [B, 1, embed_dim]
```

**Note:** In current TGTSF and lynx_film_raw implementations, `hetero_general` is loaded but not directly used in the forward pass. It can be incorporated into custom model architectures that need dataset-level context.

---

### 3. Known Planned Future Events (Dynamic) - `y_hetero`

**What it is:** Text information about events/conditions during the **prediction window** (future timestamps). This is the primary text input for Text-Guided Time Series Forecasting (TGTSF).

**Examples:**
- Weather forecasts for the next 24 hours
- Scheduled maintenance/downtime announcements
- Planned events that will affect the time series

**Source Files:**
- **Fidel-TS datasets:** Dynamic embeddings stored by timestamp
  - `dynamic_aggregate_text_v3.json` → embedded to `.pkl` files
  - Structure: `{timestamp_str: embedding_array}`

- **Time-MMD/TTC datasets:** Text column in CSV aligned to timestamps
  - Column names: `Final_Search_*`, `Final_Output`, or `text`

**Flow Path:**
```
dynamic_embeddings.pkl → Heterogeneous_Dataset.load_embedding()
    → self.embeddings[timestamp]
    → Universal_Dataset.__getitem__():
        # Get timestamps for prediction window
        y_time = self.timestamp[r_begin:r_end]  # r_begin = s_end, r_end = r_begin + pred_len
        
        # Fetch hetero data for these timestamps
        y_hetero = self.hetero_data_getter(y_time[::self.hetero_stride])
        # Returns embeddings for each timestamp in prediction window
    → DataLoader collates to [B, L, num_items, embed_dim]
    → Model.forward(news=y_hetero) receives [B, L, num_items, embed_dim]
```

**Task Configuration:**
```yaml
# In model_config:
task: TGTSF  # Text-Guided Time Series Forecasting

# This sets custom_input to include y_hetero:
# custom_input = ['seq_x', 'seq_y', 'x_time', 'y_time', 'hetero_y_time', 'y_hetero', 'hetero_general', 'hetero_channel']
```

**In Model Forward:**
```python
# TGTSF.py
def forward(self, x, news, channel_description, **kwargs):
    # news: [B, L, num_items, embed_dim] - text embeddings for prediction window
    # L = ceil(pred_len / stride) time segments
    # num_items = number of news items per timestamp (typically 1-2)
    
    # Cross-attend news with channel descriptions
    text_emb = self.text_encoder(news, description)
    
    # Mix with temporal embeddings
    x, mix_weights = self.mixer(text_emb, temporal_emb)
```

---

### 4. Historical Text News (Dynamic) - `x_hetero`

**What it is:** Text information about events/conditions during the **input/history window** (past timestamps).

**Examples:**
- Past weather conditions that affected the time series
- Historical news/events
- Anomaly reports from the input window

**Flow Path:**
```
dynamic_embeddings.pkl → Heterogeneous_Dataset.load_embedding()
    → self.embeddings[timestamp]
    → Universal_Dataset.__getitem__():
        # Get timestamps for input window
        x_time = self.timestamp[s_begin:s_end]
        
        # Fetch hetero data for these timestamps (if configured)
        if 'x_hetero' in self.custom_input:
            x_hetero = self.hetero_data_getter(x_time[::self.hetero_stride])
    → DataLoader collates to [B, L, num_items, embed_dim]
```

**Task Configuration:**
```yaml
# For MTSF (Multi-modal Time Series Forecasting) or Reasoning tasks:
task: MTSF  # or 'Reasoning'

# This sets custom_input to include x_hetero:
# custom_input = ['seq_x', 'seq_y', ..., 'hetero_x_time', 'x_hetero', ...]
```

**Note:** TGTSF task uses `y_hetero` (future) but not `x_hetero` (historical). To use historical text, configure `task: MTSF` or use `custom_input` override.

---

## Task Types and Input Selection

The dataloader returns different components based on the `task` configuration:

| Task | Includes x_hetero | Includes y_hetero | Description |
|------|-------------------|-------------------|-------------|
| `TSF` | ❌ | ❌ | Standard time series forecasting (no text) |
| `TGTSF` | ❌ | ✅ | Text-Guided TSF (future text only) |
| `MTSF` | ✅ | ❌ | Multi-modal TSF (historical text only) |
| `Reasoning` | ✅ | ✅ | Full multi-modal (both historical and future text) |

Custom input can be specified via `custom_input` config option:
```yaml
# Override task-defined inputs
custom_input: "seq_x,seq_y,x_time,y_time,y_hetero,hetero_channel"
```

---

## Input Format Pipeline

### Embedding Generation

Text is converted to embeddings before being passed to models:

```
Raw Text (JSON/CSV) 
    → TextEmbedder (BERT, etc.)
        → Embedding array [embed_dim] or [seq_len, embed_dim]
            → Cached to .pkl files
```

**Aggregation Methods:**
- `cls`: Use [CLS] token embedding → `[1, embed_dim]`
- `average`: Average all token embeddings → `[1, embed_dim]`
- `none`: Keep all token embeddings → `[seq_len, embed_dim]`

### Dimension Handling

Models may need to project input embeddings to their internal dimension:

```python
# Model config:
input_text_dim: 768   # BERT embedding dimension
text_dim: 256         # Model's internal text dimension

# In model:
if self.input_text_dim != self.text_dim:
    self.text_projection = nn.Linear(self.input_text_dim, self.text_dim)
    # Projects [B, L, N, 768] → [B, L, N, 256]
```

---

## Text Encoder Architecture (TGTSF)

The `text_encoder` module processes text embeddings:

```python
class text_encoder(nn.Module):
    """
    Cross-attends channel descriptions with news/events.
    
    Args:
        news_emb: [B, L, N, D] - News embeddings (N items per L time segments)
        description_emb: [B, L, C, D] - Channel descriptions (C channels)
    
    Returns:
        text_emb: [B, L, C, D] - Cross-attended text embeddings per channel
    """
    def forward(self, news_emb, description_emb):
        # Reshape for batch processing
        news_emb = news_emb.view(B*L, N, D)
        description_emb = description_emb.view(B*L, C, D)
        
        # Cross-attention: descriptions query news
        text_emb = self.cross_encoder(
            tgt=description_emb,   # Query
            memory=news_emb        # Key/Value
        )
        
        return text_emb.view(B, L, C, D)
```

---

## TGTSF vs LYNX-FiLM-raw Model Comparison

Both models use the same text inputs but process them differently:

### TGTSF

```python
def forward(self, x, news, channel_description):
    # 1. Encode time series via patches
    x = self.TS_encoder(x)  # [B, patch_num, C, d_model]
    
    # 2. Encode text via cross-attention
    text_emb = self.text_encoder(news, description)  # [B, L, C, text_dim]
    
    # 3. Mix text and temporal via cross-attention
    x = self.mixer(text_emb, x)  # Text guides temporal
    
    # 4. Project to predictions
    x = self.head(x)
    x = self.patch_reconstruction(x)
```

### LYNX-FiLM-raw

```python
def forward(self, x, news, channel_description):
    # 1. Encode text via cross-attention (same as TGTSF)
    text_emb = self.text_encoder(news, description)  # [B, L, C, text_dim]
    text_emb = text_emb.permute(0, 2, 1, 3)  # [B, C, L, D]
    
    # 2. Apply FiLM modulation to iTransformer
    # Text embeddings generate gamma/beta for feature modulation
    pred = self.model(x, text_emb)  # FiLM-modulated prediction
```

---

## Dataset-Specific Considerations

### Fidel-TS Datasets (Bear_room, Jena, NYC, etc.)

- Use `Heterogeneous_Dataset` for text handling
- Static info in `static_info_embeddings.pkl`
- Dynamic text in yearly `.pkl` files
- Support both `all_for_one` (shared text) and `each_subset` (per-entity text) modes
- Handle sensor downtime with `downtime_prompt`

### Time-MMD Datasets (Climate, Economy, etc.)

- Use `TimeMMD_Dataset` and `TimeMMD_HeteroGetter`
- Text stored in CSV columns (`Final_Search_*`, `Final_Output`)
- Auto-detection of text column
- Supports both text output (for LLMs) and embedding output (for neural models)

### TTC Datasets (climate, medical)

- Use `TimeMMD_Dataset` with `text_column: text`
- Simpler format with single `text` column
- Multi-channel support with per-channel embeddings

---

## Configuration Examples

### Full TGTSF Configuration (Fidel-TS)

```yaml
# data_configs/Bear_room/fullBear_hetero_TGTSF.yaml
root_path: ./data/Bear_room/time_series
hetero_info:
  hetero_type: all_for_one
  root_path: ./data/Bear_room/hetero/weather/report_embedding/formal_report
  formatter: wm_messages_v1.pkl
  matching: backward
  input_format: embedding
  static_path: static_info_embeddings.pkl
  embedding_config:
    model_name: bert-base-uncased
    aggregation_method: cls
```

```yaml
# model_configs/general/TGTSF.yaml
model: TGTSF
text_dim: 256
cross_layers: 3
self_layers: 3
task: TGTSF
hetero_align_stride: True
```

### Time-MMD Configuration

```yaml
# data_configs/time_mmd/Climate/config.yaml
dataset_type: time_mmd
text_column: auto
timemmd_text_output: embedding
timemmd_embed_model: bert-base-uncased
general_info: "US Precipitation monthly time series data"
channel_info: ""
```

---

## Summary Table

| Component | Source | Dataloader Output | Model Input | Shape |
|-----------|--------|-------------------|-------------|-------|
| Time series | `.csv`/`.parquet` | `seq_x`, `seq_y` | `x` | `[B, seq_len, C]` |
| Channel desc. | `static_info.pkl` | `hetero_channel` | `channel_description` | `[B, C, D]` |
| General info | `static_info.pkl` | `hetero_general` | (available) | `[B, 1, D]` |
| Future text | `dynamic_*.pkl` | `y_hetero` | `news` | `[B, L, N, D]` |
| Historical text | `dynamic_*.pkl` | `x_hetero` | (available) | `[B, L, N, D]` |

Where:
- `B` = batch size
- `C` = number of channels
- `D` = embedding dimension (e.g., 768 for BERT, 256 internal)
- `L` = number of time segments (`ceil(pred_len / stride)`)
- `N` = number of text items per timestamp (typically 1-2)
