# Timestamp Semantics in Multimodal Time Series Datasets

## Executive Summary

This document defines the critical distinction between two timestamp semantics used in multimodal time series datasets:

| Semantics | Meaning | Used By | TGTSF Text Source |
|-----------|---------|---------|-------------------|
| **`t_about`** | When the text *describes* (target time) | Fidel-TS | `news` (y_hetero) - forecasts about prediction window |
| **`t_known`** | When the text *was available* (publication time) | Time-MMD, TTC | `historical_events` (x_hetero) - avoids lookahead |

**Key Insight:** TGTSF-family models automatically select the correct text input based on `timestamp_semantics`:
- `t_about`: Model uses `news` (y_hetero) — assumed known at prediction time (with lead time assumptions)
- `t_known`: Model uses `historical_events` (x_hetero) — avoids lookahead bias by using historical text only

---

## 1. Definitions

### 1.1 `t_about` (Target Time Semantics)

The timestamp represents **what time period the text describes**.

**Example:**
```
Timestamp: 2024-01-15
Text: "Expect heavy snow and freezing temperatures throughout the day."
```
- The text is a **forecast FOR** January 15th
- The text was **published BEFORE** January 15th (e.g., on January 14th)
- The actual publication time (`t_known`) is NOT stored in the dataset

**Assumption Required:** The data curator ensured that all forecasts were published before their target time.

### 1.2 `t_known` (Publication Time Semantics)

The timestamp represents **when the text became available**.

**Example:**
```
Timestamp: 2024-01-15 14:00
Text: "Current conditions: Heavy snow, 28°F, visibility reduced."
```
- The text was **published AT** 2024-01-15 14:00
- The text describes conditions **at or before** 2024-01-15 14:00
- The target time (`t_about`) equals or precedes the publication time

---

## 2. Dataset Classification

### 2.1 Fidel-TS Datasets: `timestamp_semantics: t_about`

| Dataset | Config Location |
|---------|-----------------|
| Bear_room | `data_configs/Bear_room/` |
| California_ISO | `data_configs/California_ISO/` |
| Canada_photovoltaics_plants | `data_configs/Canada_photovoltaics_plants/` |
| Germany_Renewable_Power_Grid | `data_configs/Germany_Renewable_Power_Grid/` |
| Jena_Atmospheric_Physics | `data_configs/Jena_Atmospheric_Physics/` |
| NYC_traffic_speed | `data_configs/NYC_traffic_speed/` |

**Evidence for `t_about` classification:**

Analysis of raw weather text files (e.g., `merged_general_weather_report.json`) reveals consistent use of **future tense**:

```json
{
    "20170101": {
        "brooklyn": {
            "daily": "It's going to be a mostly cloudy day today...",
            "Morning": "Morning will see the sun coming out...",
            "Evening": "Evening will be clear..."
        }
    }
}
```

The text describes what the weather **will be** on the timestamp date, not what it **was** when published.

### 2.2 Time-MMD Datasets: `timestamp_semantics: t_known`

| Dataset | Config Location |
|---------|-----------------|
| Agriculture | `data_configs/time_mmd/Agriculture/` |
| Climate | `data_configs/time_mmd/Climate/` |
| Economy | `data_configs/time_mmd/Economy/` |
| Energy | `data_configs/time_mmd/Energy/` |
| Environment | `data_configs/time_mmd/Environment/` |
| Public_Health | `data_configs/time_mmd/Public_Health/` |
| SocialGood | `data_configs/time_mmd/SocialGood/` |
| Traffic | `data_configs/time_mmd/Traffic/` |

### 2.3 TTC Datasets: `timestamp_semantics: t_known`

| Dataset | Config Location |
|---------|-----------------|
| Climate | `data_configs/ttc/climate/` |
| Medical | `data_configs/ttc/medical/` |

---

## 3. The Lookahead Bias Problem

### 3.1 Why `y_hetero` + `t_known` = Lookahead Bias

When using `task: TGTSF`, the dataloader fetches `y_hetero` using **prediction window timestamps** (`y_time`):

```python
# data_provider/data_loader.py
r_begin = s_end  # End of input window = start of prediction
r_end = r_begin + pred_len
y_time = self.timestamp[r_begin:r_end]  # FUTURE timestamps!

if 'y_hetero' in self.custom_input:
    y_hetero = self.hetero_data_getter(y_time[::self.hetero_stride])
```

**The problem with `t_known` timestamps:**

```
Prediction starts at time T, ends at T+24

Timeline:
    T           T+12          T+24
    |------------|-------------|
    ↑            ↑             ↑
Prediction    Text at       Text at
Start         T+12 exists   T+24 exists

Query with backward matching for y_time = [T+1, T+2, ..., T+24]:
- For T+12: Returns text published ≤ T+12 (includes text from T+5, T+8...)
- For T+24: Returns text published ≤ T+24 (includes ALL text in prediction window)

RESULT: Model sees text published DURING the prediction window!
This information would NOT be available at prediction time T.
```

### 3.2 Why `y_hetero` + `t_about` is Safe (with assumptions)

For Fidel-TS datasets where timestamps = `t_about`:

```
Prediction starts at time T, ends at T+24

Text for Wed (T+12) was:
- Published on Mon (T-24)   ← t_known
- About Wed (T+12)          ← t_about (stored timestamp)

Query with backward matching for y_time = [T, T+1, ..., T+24]:
- For T+12 (Wed): Returns text about Wed
- This text was published Mon (before prediction start T)

RESULT: Model sees text that was known before prediction start.
```

**Critical Assumption:** Weather forecasts have sufficient lead time (published before target).

---

## 4. Uncertainty and Assumptions for Fidel-TS

### 4.1 What We Know

1. **Linguistic evidence:** Text uses future tense ("will be", "going to be", "expect")
2. **File naming:** Files are named `fast_general_formal_forecast_*.json`
3. **Structure:** Timestamps key forecasts about that date

### 4.2 What We Assume

We give the benefit of the doubt to the original Fidel-TS authors and assume:

1. **Forecasts have adequate lead time:** A forecast for day D was published before day D
2. **The lead time covers the prediction horizon:** For prediction horizons used in the original paper, the forecast was available at prediction start time

### 4.3 Lead Time Uncertainty

Weather forecast lead times vary:

| Forecast Type | Typical Lead Time |
|---------------|-------------------|
| Hourly forecasts | 1-6 hours |
| Daily forecasts | 1-3 days |
| Weekly outlooks | 5-7 days |

**Recommendation:** For Fidel-TS datasets, limit prediction horizons to reasonable lead times:

| Dataset | Recommended Max `pred_len` | Reasoning |
|---------|---------------------------|-----------|
| Hourly data (e.g., NYC_traffic) | 24-48 hours | Daily forecasts typically available 1-2 days ahead |
| Daily data | 7 days | Weekly forecasts typically available ~1 week ahead |

**Using longer prediction horizons risks lookahead bias** if the text for distant future timesteps wasn't actually known at prediction start.

### 4.4 Explicit Documentation

When using Fidel-TS datasets with TGTSF, document:

1. The prediction horizon used
2. The assumed forecast lead time
3. Any known limitations of the assumption

---

## 5. Configuration Specification

### 5.1 Adding `timestamp_semantics` to Dataset Configs

Each dataset config should include:

```yaml
# For Fidel-TS datasets
hetero_info:
  timestamp_semantics: t_about  # Text describes this timestamp
  # ... other hetero_info fields
```

```yaml
# For Time-MMD/TTC datasets  
hetero_info:
  timestamp_semantics: t_known  # Text was available at this timestamp
  # ... other hetero_info fields
```

### 5.2 Valid Task/Semantics Combinations

| `timestamp_semantics` | `task: TSF` | `task: TGTSF` | `task: MTSF` |
|----------------------|-------------|---------------|--------------|
| `t_about` | ✅ Safe | ✅ Safe (uses `news`, with lead time assumptions) | ✅ Safe |
| `t_known` | ✅ Safe | ✅ Safe (uses `historical_events`, no lookahead) | ✅ Safe |

**Note:** TGTSF-family models automatically select the correct text source based on `timestamp_semantics`:
- `t_about`: Uses `news` (y_hetero) - forecasts about prediction window
- `t_known`: Uses `historical_events` (x_hetero) - avoids lookahead bias

---

## 6. Correct Usage Patterns

### 6.1 Fidel-TS Datasets (`timestamp_semantics: t_about`)

```yaml
# Fidel-TS with TGTSF
# Model uses: news (y_hetero) - forecasts ABOUT the prediction window
model:
  config_path: model_configs/general/TGTSF.yaml
data:
  config_path: data_configs/NYC_traffic_speed/fullNYCTS_hetero_TGTSF_H.yaml
  # hetero_info.timestamp_semantics: t_about
training:
  # Limit pred_len to reasonable forecast lead times
  output_len: 24  # 24 hours - reasonable for daily weather forecasts
```

### 6.2 Time-MMD/TTC Datasets (`timestamp_semantics: t_known`)

```yaml
# Time-MMD/TTC with TGTSF
# Model uses: historical_events (x_hetero) - automatically avoids lookahead
model:
  config_path: model_configs/general/TGTSF.yaml
data:
  config_path: data_configs/time_mmd/Traffic/config.yaml
  # hetero_info.timestamp_semantics: t_known
training:
  output_len: 96
```

```yaml
# Time-MMD/TTC with native MTSF model (TimeCMA, MMTSFlib)
# Model uses: historical_events (x_hetero) - designed for this
model:
  config_path: model_configs/LLM/TimeCMA.yaml
data:
  config_path: data_configs/time_mmd/Traffic/config.yaml
training:
  output_len: 96
```

### 6.3 Common Mistakes to Avoid

```yaml
# ⚠️ Missing timestamp_semantics
# If not specified, model defaults to t_about
# Make sure your dataset config explicitly sets timestamp_semantics!
hetero_info:
  timestamp_semantics: t_known  # <- Add this to Time-MMD/TTC configs
```

---

## 7. Implementation Requirements

### 7.1 Nomenclature Mapping

The dataloader and model code use different names for the same data:

| Dataloader Name | Model Parameter Name | Description |
|-----------------|---------------------|-------------|
| `x_hetero` | `historical_events` | Text aligned to **input window** timestamps |
| `y_hetero` | `news` | Text aligned to **prediction window** timestamps |

The experiment runner always passes both:
```python
model.forward(
    x=batch_x,
    historical_events=batch_x_hetero,  # x_hetero → historical_events
    news=batch_y_hetero,               # y_hetero → news
    channel_description=hetero_channel
)
```

### 7.2 Model-Level Text Selection

TGTSF-family models select which text input to use based on `timestamp_semantics`:

```python
# In model __init__:
self.timestamp_semantics = getattr(configs, 'timestamp_semantics', 't_about')

# In model forward:
if self.timestamp_semantics == 't_about':
    # Fidel-TS: news contains forecasts ABOUT prediction window
    # Assumption: forecast for t+k was known at time t
    text_input = news  # y_hetero
elif self.timestamp_semantics == 't_known':
    # Time-MMD/TTC: news would be LOOKAHEAD BIAS
    # Use historical_events instead
    text_input = historical_events  # x_hetero
```

### 7.3 Model Compatibility Matrix

| Model | Task | `t_about` Text Source | `t_known` Text Source |
|-------|------|----------------------|----------------------|
| TGTSF | TGTSF | `news` (y_hetero) | `historical_events` (x_hetero) |
| lynx_film_raw | TGTSF | `news` (y_hetero) | `historical_events` (x_hetero) |
| lynx_film | TGTSF | `news` (y_hetero) | `historical_events` (x_hetero) |
| lynx | TGTSF | `news` (y_hetero) | `historical_events` (x_hetero) |
| TimeCMA | MTSF | `historical_events` (x_hetero) | `historical_events` (x_hetero) |
| MMTSFlib | MTSF | `historical_events` (x_hetero) | `historical_events` (x_hetero) |

---

## 8. Summary

1. **Know your timestamps:** Check `timestamp_semantics` in your dataset config
2. **`news` = `y_hetero`:** Text aligned to prediction window
3. **`historical_events` = `x_hetero`:** Text aligned to input window
4. **Fidel-TS (`t_about`):** TGTSF models use `news` - forecasts about future, assumed known beforehand
5. **Time-MMD/TTC (`t_known`):** TGTSF models use `historical_events` - avoids lookahead bias
6. **Document assumptions:** When using `t_about`, document the assumed forecast lead time

---

## Appendix A: Detailed Data Flow

### A.1 TGTSF Data Flow (`y_hetero`)

```
Sample at index i:
┌────────────────────────┬────────────────────────────┐
│     Input Window       │      Prediction Window     │
│   [s_begin : s_end]    │    [r_begin : r_end]       │
│    (seq_len steps)     │    (pred_len steps)        │
└────────────────────────┴────────────────────────────┘
          ↓                           ↓
       x_time                      y_time
          │                           │
          │                           ▼
          │              hetero_data_getter(y_time)
          │                           │
          │                           ▼
          │                       y_hetero
          │              [pred_len, num_items, embed_dim]
          │                           │
          ▼                           ▼
      seq_x ─────────────────────► Model
                                     │
      news = y_hetero ───────────────┘
      
Forward: model(x=seq_x, news=y_hetero, channel_description=...)
```

### A.2 MTSF Data Flow (`x_hetero`)

```
Sample at index i:
┌────────────────────────┬────────────────────────────┐
│     Input Window       │      Prediction Window     │
│   [s_begin : s_end]    │    [r_begin : r_end]       │
│    (seq_len steps)     │    (pred_len steps)        │
└────────────────────────┴────────────────────────────┘
          ↓                           
       x_time                      
          │                           
          ▼                           
 hetero_data_getter(x_time)
          │
          ▼
      x_hetero
 [seq_len, num_items, embed_dim]
          │
          ▼
      seq_x ─────────────────────► Model
                                     │
 historical_events = x_hetero ───────┘
      
Forward: model(x=seq_x, historical_events=x_hetero, ...)
```

---

## Appendix B: Lookahead Bias Visualization

```
t_known semantics + TGTSF = LOOKAHEAD BIAS
──────────────────────────────────────────

Timeline:
    T-24      T-12       T         T+12       T+24
     │         │         │          │          │
     ▼         ▼         ▼          ▼          ▼
  ┌──────────────────┐┌──────────────────────────┐
  │   Input Window   ││    Prediction Window     │
  │   (historical)   ││       (future)           │
  └──────────────────┘└──────────────────────────┘
                      ↑
                Prediction Start
                      
Text available (t_known):
  T-24: "Article about events on T-25"        ← Safe (past)
  T-12: "Report on T-12 morning conditions"   ← Safe (past)
  T:    "Breaking news at time T"             ← Borderline
  T+6:  "Midday update at T+6"               ← LOOKAHEAD!
  T+12: "Evening report at T+12"             ← LOOKAHEAD!
  T+24: "Next day summary at T+24"           ← LOOKAHEAD!

When querying y_hetero with backward matching:
  Query T+12 → Returns text from ≤T+12 → Includes T+6, T+12 texts!
  
The model sees text published AFTER prediction start (T).
This is information leakage / lookahead bias.
```

```
t_about semantics + TGTSF = SAFE (with assumptions)
───────────────────────────────────────────────────

Timeline:
    T-24      T-12       T         T+12       T+24
     │         │         │          │          │
     ▼         ▼         ▼          ▼          ▼
  ┌──────────────────┐┌──────────────────────────┐
  │   Input Window   ││    Prediction Window     │
  └──────────────────┘└──────────────────────────┘
                      ↑
                Prediction Start

Weather forecasts (t_about = target time, t_known = publication time):
  t_about=T+12: "Forecast for T+12" (published at T-24) ← Published BEFORE T
  t_about=T+24: "Forecast for T+24" (published at T-12) ← Published BEFORE T

When querying y_hetero with backward matching:
  Query T+12 → Returns forecast about T+12 (published T-24)
  
ASSUMPTION: Forecast was published before prediction start (T).
If this holds, no lookahead bias.
```
