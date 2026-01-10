# Text Information Flow in fidel-ts

This document describes how different types of text information flow through the fidel-ts codebase, from dataloaders to models (specifically `TGTSF` and `lynx_film_raw`).

> **📋 Timestamp Semantics & Nomenclature**
> 
> Before using multimodal features, read **[Timestamp Semantics](timestamp_semantics.md)** to understand the difference between `t_about` and `t_known` timestamps.
> 
> **Nomenclature mapping (dataloader → model):**
> - `x_hetero` → `historical_events` (text aligned to input window)
> - `y_hetero` → `news` (text aligned to prediction window)
> 
> **TGTSF models automatically select text source based on `timestamp_semantics`:**
> - **`t_about` (Fidel-TS):** Uses `news` (y_hetero) — forecasts about prediction window
> - **`t_known` (Time-MMD/TTC):** Uses `historical_events` (x_hetero) — avoids lookahead bias
> 
> See also: **[TGTSF Migration Plan](time_mmd_mtsf_migration_plan.md)** for implementation details.

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

#### ⚠️ Critical: Two-Timestamp Semantics for Future Events

For known planned future events, there are conceptually **TWO distinct timestamps**:

| Timestamp Type | Description | Example (Weather Forecast) |
|---------------|-------------|---------------------------|
| **Publication Time** (`t_known`) | When the information became available | Monday 8:00 AM (forecast published) |
| **Target Time** (`t_about`) | What time period the information describes | Wednesday (forecasted conditions) |

**Why this matters:** To avoid lookahead bias, the model must only use information that was **known** (published) before the prediction start time. A weather forecast for Wednesday that was published on Monday is valid to use on Tuesday, but a forecast published on Wednesday afternoon would cause lookahead bias.

#### Current Implementation: Single-Timestamp Limitation

**⚠️ Important:** The current fidel-ts implementation uses a **single timestamp** per text entry in the dynamic data files. The semantic meaning of this timestamp depends on how the data was prepared:

```
Dynamic data structure: {timestamp_str: embedding_array}
                            ↑
                    Single timestamp - meaning depends on data preparation
```

**How matching works in `Universal_Dataset.__getitem__()`:**

```python
# data_provider/data_loader.py, lines 348-353
if 'y_hetero' in self.custom_input:
    y_hetero = self.hetero_data_getter(y_time[::self.hetero_stride])
    hetero_y_time = y_hetero[0]
    # ...
```

1. `y_time` contains timestamps from the **prediction window** `[t_end_input, t_end_input + pred_len)`
2. These timestamps are passed to `hetero_data_getter()` → `time_matcher()`
3. `time_matcher()` finds text entries based on the configured `matching` strategy

**Matching Strategies and Their Implications:**

| Strategy | Behavior | Use Case | Lookahead Risk |
|----------|----------|----------|----------------|
| `backward` | Find text at or before query timestamp | When text timestamp = publication time | ✅ Safe (text was available before query time) |
| `forward` | Find text at or after query timestamp | NOT RECOMMENDED | ❌ Potential lookahead |
| `nearest` | Find closest text to query timestamp | NOT RECOMMENDED | ❌ Potential lookahead |
| `single` | Like backward, but deduplicated | Same as backward | ✅ Safe |

#### Correct Interpretation for TGTSF

For TGTSF to work correctly without lookahead bias, the data must be prepared such that:

**Option A: Timestamps represent publication time (`t_known`)**
- Dynamic data keys = when information was available
- Use `matching: backward` with `y_time` (prediction window timestamps)
- Result: Model gets text that was published before each prediction timestamp
- **Limitation:** No guarantee the text is actually *about* the prediction window

**Option B: Timestamps represent target time (`t_about`)** *(Assumed Fidel-TS approach)*
- Dynamic data keys = what time the information describes
- Weather forecast for Wednesday is stored under Wednesday's date
- Use `matching: backward` with `y_time`
- Result: Model gets text *about* the prediction window
- **Assumption:** Data curator ensured no lookahead (forecasts were available before their target time)

> **⚠️ Evidence from Fidel-TS data files:** Analysis of raw weather text in files like `merged_general_weather_report.json` reveals **consistent use of future tense** (e.g., "It's going to be a mostly cloudy day today", "Morning will be mostly cloudy", "Expect some passing clouds"). This linguistic evidence suggests timestamps represent `t_about` (what date the forecast describes), **NOT** `t_known` (when it was published). **Publication time (`t_known`) is not tracked** in the Fidel-TS dataset. Consequently, the current implementation assumes that the forecasts are all known at prediction time (assumes that forecasts for `t+k` are known at `t` where `k` is the prediction horizon). For Fidel-TS, we follow the original paper in this assumption (giving the benefit of the doubt to the original authors).

#### Data Flow Diagram for y_hetero

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Sample at index i:                                                          │
│  ┌────────────────────────┬────────────────────────────┐                     │
│  │     Input Window       │      Prediction Window     │                     │
│  │   [s_begin : s_end]    │    [r_begin : r_end]       │                     │
│  │    (seq_len steps)     │    (pred_len steps)        │                     │
│  └────────────────────────┴────────────────────────────┘                     │
│            ↓                           ↓                                     │
│         x_time                      y_time                                   │
│    (input timestamps)          (prediction timestamps)                       │
│                                        │                                     │
│                                        ▼                                     │
│                          ┌─────────────────────────────┐                     │
│                          │    hetero_data_getter()     │                     │
│                          │                             │                     │
│                          │  time_matcher(y_time)       │                     │
│                          │  matching='backward'        │                     │
│                          └──────────────┬──────────────┘                     │
│                                         │                                    │
│                                         ▼                                    │
│                          ┌─────────────────────────────┐                     │
│                          │  Dynamic Embeddings Dict    │                     │
│                          │  {                          │                     │
│                          │    "20200901120000": emb1,  │← Single timestamp   │
│                          │    "20200902120000": emb2,  │  per entry          │
│                          │    ...                      │                     │
│                          │  }                          │                     │
│                          └──────────────┬──────────────┘                     │
│                                         │                                    │
│                                         ▼                                    │
│                          y_hetero: [L, num_items, embed_dim]                 │
│                          L = ceil(pred_len / stride) time segments           │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Source Files:**
- **Fidel-TS datasets:** Dynamic embeddings stored by timestamp
  - `dynamic_aggregate_text_v3.json` → embedded to `.pkl` files
  - Structure: `{timestamp_str: embedding_array}`
  - **Timestamp meaning:** We assume it represent `t_about` (target time of forecast/event) based on linguistic analysis of raw text files (future tense: "will be", "going to be", "expect")

- **Time-MMD/TTC datasets:** Text column in CSV aligned to timestamps
  - Column names: `Final_Search_*`, `Final_Output`, or `text`
  - **Timestamp meaning:** The row's timestamp (same as time series)

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

#### Single-Timestamp Semantics for Historical Events

Unlike future events, historical text has **only ONE relevant timestamp**: when the information was available (which is also when the event occurred or was observed).

```
For historical text: t_known ≈ t_about
(Publication time equals or closely follows the event time)
```

This is simpler because:
- A news article about Monday's events was published on Monday
- Past weather observations describe the time they were recorded
- No "lead time" concept like with forecasts

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

**Matching for Historical Text:**

| Strategy | Behavior | Lookahead Risk |
|----------|----------|----------------|
| `backward` | Find text at or before input timestamp | ✅ Always safe (past is past) |
| `forward` | Find text at or after input timestamp | ⚠️ Could use future info |
| `nearest` | Find closest text | ⚠️ Could use future info |

**Recommendation:** Always use `matching: backward` for `x_hetero` to ensure text was available at each input timestamp.

**Task Configuration:**
```yaml
# For MTSF (Multi-modal Time Series Forecasting) or Reasoning tasks:
task: MTSF  # or 'Reasoning'

# This sets custom_input to include x_hetero:
# custom_input = ['seq_x', 'seq_y', ..., 'hetero_x_time', 'x_hetero', ...]
```

**Note:** TGTSF task uses `y_hetero` (future) but not `x_hetero` (historical). To use historical text, configure `task: MTSF` or use `custom_input` override.

---

## Timestamp Semantics Summary

| Text Type | # of Timestamps | Timestamp Meaning | Lookahead Concern |
|-----------|-----------------|-------------------|-------------------|
| **Channel Descriptions** | 0 (static) | N/A | None |
| **General Information** | 0 (static) | N/A | None |
| **Historical Text** (`x_hetero`) | 1 | When text was available ≈ What it describes | Low (use `backward` matching) |
| **Future Events** (`y_hetero`) | 1 (but conceptually 2) | Depends on data preparation | **High** - must ensure text was known before prediction |

### Future Work: Explicit Two-Timestamp Support

A more robust implementation would explicitly track both timestamps for future events:

```python
# Hypothetical enhanced data structure
{
    "publication_time": "20200901080000",  # When forecast was made (Monday 8am)
    "target_time": "20200903000000",       # What it forecasts (Wednesday)
    "embedding": [0.1, 0.2, ...],
    "text": "Weather forecast for Wednesday: sunny, high 75°F"
}
```

This would allow the dataloader to:
1. Filter by `publication_time <= prediction_start` (avoid lookahead)
2. Filter by `target_time` overlapping with prediction window (relevance)

Currently, this responsibility falls on the **data curator** to ensure the dynamic text files are prepared correctly

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

## Lookahead Bias Prevention

### What is Lookahead Bias?

Lookahead bias occurs when a model uses information during training or inference that would not have been available at the time of prediction in a real-world scenario.

### Sources of Lookahead Bias in Text-Guided Forecasting

| Source | Risk Level | Description |
|--------|------------|-------------|
| Using text published after prediction start | 🔴 High | Weather forecast from Wednesday used to predict Wednesday |
| Forward/nearest matching on dynamic text | 🟡 Medium | Matching may select future text entries |
| Text about future but published before | 🟢 None | This is the intended use case (e.g., forecasts) |

### Safeguards in fidel-ts

1. **Matching Strategy:** Use `matching: backward` (default for most configs)
   ```yaml
   hetero_info:
     matching: backward  # ✅ Safe - only uses text at or before query time
   ```

2. **Data Separation:** Input window and prediction window are strictly separated
   ```python
   # data_loader.py
   seq_x = self.data[s_begin:s_end]      # Input: [t, t+seq_len)
   seq_y = self.data[r_begin:r_end]      # Target: [t+seq_len, t+seq_len+pred_len)
   # r_begin = s_end, ensuring no overlap
   ```

3. **Data Curator Responsibility:** For `y_hetero` (future events), the data curator must ensure:
   - Text entries keyed by target time were actually available before that time
   - Example: A weather forecast for Wednesday stored under Wednesday's date must have been published before Wednesday

### Validation Checklist

When using TGTSF with future text information:

- [ ] Confirm `matching: backward` in data config
- [ ] Verify dynamic text file timestamps represent appropriate time reference
- [ ] For forecasts: Ensure forecasts were published before their target time
- [ ] For scheduled events: Ensure schedules were known before the event time
- [ ] Test on held-out data to detect any performance anomalies

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

---

## Appendix: Timestamp Semantics Deep Dive

### The Two-Timestamp Problem for Future Events

```
Example: Weather Forecast

Timeline:
─────────────────────────────────────────────────────────────────────►
   Mon 8am        Tue 12pm        Wed 12am        Thu 12pm
     │               │               │               │
     │               │               │               │
     ▼               ▼               ▼               ▼
  Forecast      Prediction      Forecast        End of
  Published     Start Time      Target Day      Prediction
  (t_known)     (t_pred_start)  (t_about)       Window


Question: Can we use this forecast for predictions starting Tuesday?

Answer depends on:
1. t_known < t_pred_start?  → Yes, Monday < Tuesday ✓
2. t_about overlaps prediction window? → Yes, Wednesday is in [Tue, Thu] ✓

Both conditions must be true for valid usage.
```

### Current Implementation Behavior

The current implementation stores ONE timestamp per text entry. Depending on what this timestamp represents:

**Scenario A: Timestamp = t_known (publication time)**
```
Query: y_time = [Tue 12pm, Wed 12pm, Thu 12pm]
Matching: backward
Result: Finds forecast published Mon 8am (before all query times)
✅ Correct: Uses available information
⚠️ Issue: May not be ABOUT the prediction window
```

**Scenario B: Timestamp = t_about (target time)** ← **Assumed / suggested by evidence: This is what Fidel-TS uses**
```
Query: y_time = [Tue 12pm, Wed 12pm, Thu 12pm]
Matching: backward
Result: For Wed query, finds forecast about Wed (published Mon)
✅ Correct: Gets relevant forecast
⚠️ Assumption: Data curator ensured t_known < t_about
```

> **Linguistic Evidence:** Analysis of `merged_general_weather_report.json` shows forecasts use future tense:
> - `"20170101"` → "It's **going to be** a mostly cloudy day **today**..."
> - `"20170102"` → "Early morning on January 2nd **will be** quite chilly..."
> 
> The text describes what **will happen** on the timestamp date, suggesting timestamps = `t_about`.

### Recommendation for Data Preparation

For Fidel-TS datasets using weather forecasts:

1. **Key by target time** (what the forecast is about)
2. **Ensure forecasts have lead time** (published before target)
3. **Document the lead time** in dataset metadata

Example data preparation:
```python
# When creating dynamic_aggregate_text.json
for forecast in weather_forecasts:
    # Key by target time, but verify lead time
    assert forecast.publication_time < forecast.target_time
    data[forecast.target_time.strftime('%Y%m%d%H%M%S')] = forecast.text
```

This ensures that `backward` matching with `y_time` (prediction window timestamps) retrieves forecasts that are:
1. About the prediction window (keyed by target time)
2. Known before the prediction (by data preparation invariant)
