# Migration Plan: Time-MMD/TTC Support for TGTSF-Family Models

## Overview

This document outlines the plan to enable TGTSF-family models (TGTSF, lynx_film_raw, lynx_film, lynx) to work correctly with Time-MMD and TTC datasets by respecting `timestamp_semantics`.

**Status:** Planning (NOT YET IMPLEMENTED)

---

## 1. Nomenclature Mapping

### 1.1 The Confusion: Dataloader vs Model Names

The dataloader and model code use different names for the same data:

| Dataloader Name | Model Parameter Name | Description |
|-----------------|---------------------|-------------|
| `x_hetero` | `historical_events` | Text aligned to **input window** timestamps |
| `y_hetero` | `news` | Text aligned to **prediction window** timestamps |

**The mapping in `exp_universal.py` (line ~185-189):**
```python
output = self.model(
    x=batch_x, 
    historical_events=batch_x_hetero,  # x_hetero → historical_events
    news=batch_y_hetero,               # y_hetero → news
    dataset_description=hetero_general, 
    channel_description=hetero_channel
)
```

### 1.2 Clear Definitions

**`news` (= `y_hetero`):**
- Text embeddings aligned to **prediction window** timestamps (`y_time`)
- Timestamps range from `[t, t+pred_len)` where `t` is prediction start
- **For `timestamp_semantics: t_about`:** Contains forecasts/schedules ABOUT the prediction window
- **For `timestamp_semantics: t_known`:** Contains text PUBLISHED during prediction window (LOOKAHEAD!)

**`historical_events` (= `x_hetero`):**
- Text embeddings aligned to **input window** timestamps (`x_time`)  
- Timestamps range from `[t-seq_len, t)` where `t` is prediction start
- Always safe to use (text was available before prediction start)

---

## 2. Timestamp Semantics and Text Selection

### 2.1 When `timestamp_semantics: t_about` (Fidel-TS datasets)

```
Text timestamps represent WHAT TIME the text DESCRIBES (target time).

Example: Weather forecast for Wednesday
- Timestamp in data: Wednesday (t_about)
- Actual publication: Monday (t_known, not stored)
- Assumption: t_known < t_about (forecast published before target)

For TGTSF task:
- Use `news` (y_hetero) ✅
- Text describes the prediction window
- Assumed to be known at prediction start (with lead time assumptions)
```

### 2.2 When `timestamp_semantics: t_known` (Time-MMD, TTC datasets)

```
Text timestamps represent WHEN the text WAS PUBLISHED (availability time).

Example: News article about current events
- Timestamp in data: Wednesday 2pm (when published)
- Text describes: Events on/before Wednesday 2pm

For TGTSF task:
- NEVER use `news` (y_hetero) ❌ → Would be LOOKAHEAD BIAS
- Use `historical_events` (x_hetero) ✅
- Text describes the input window, safe to use
```

### 2.3 Summary Table

| `timestamp_semantics` | Task | Text Source | Model Parameter |
|----------------------|------|-------------|-----------------|
| `t_about` | TGTSF | `y_hetero` | `news` |
| `t_about` | MTSF | `x_hetero` | `historical_events` |
| `t_known` | TGTSF | `x_hetero` | `historical_events` |
| `t_known` | MTSF | `x_hetero` | `historical_events` |

---

## 3. Implementation Plan (Option B)

### 3.1 Design Principles

1. **Explicit, not fallback:** Text selection is based on `timestamp_semantics`, NOT on which parameter is non-None
2. **Validation at model level:** TGTSF models validate and select text source in forward pass
3. **Data factory always passes both:** `news` and `historical_events` are always populated; model decides which to use
4. **In-code documentation:** Every model that uses text must document what `news` and `historical_events` mean

### 3.2 Changes to Data Factory / Experiment Runner

**No changes needed.** The current implementation already:
- Always passes `historical_events=batch_x_hetero` and `news=batch_y_hetero` to models
- Both are always populated by the dataloader (may be empty tensors if not configured)

The model decides which to use based on `timestamp_semantics`.

### 3.3 Changes to TGTSF-Family Models

Each model needs:

1. **Store `timestamp_semantics` in `__init__`:**
```python
def __init__(self, configs):
    super().__init__()
    # ...existing code...
    
    # Timestamp semantics determines which text input to use for TGTSF task
    # - t_about: Use news (y_hetero) - text describes prediction window, assumed known beforehand
    # - t_known: Use historical_events (x_hetero) - avoid lookahead bias
    self.timestamp_semantics = getattr(configs, 'timestamp_semantics', 't_about')
    if self.timestamp_semantics not in ('t_about', 't_known'):
        raise ValueError(
            f"Invalid timestamp_semantics: {self.timestamp_semantics}. "
            f"Must be 't_about' or 't_known'."
        )
```

2. **Update forward signature:**
```python
def forward(self, x, news=None, channel_description=None, historical_events=None, **kwargs):
    """
    Forward pass for TGTSF model.
    
    Args:
        x: Input time series [B, seq_len, C]
        news: Text embeddings aligned to PREDICTION window (y_hetero from dataloader)
              Shape: [B, pred_len, num_items, text_dim]
              - For t_about: Forecasts/schedules ABOUT the prediction window
              - For t_known: Text PUBLISHED during prediction window (LOOKAHEAD - do not use!)
        channel_description: Static channel descriptions [B, C, text_dim]
        historical_events: Text embeddings aligned to INPUT window (x_hetero from dataloader)
                          Shape: [B, seq_len, num_items, text_dim]
                          - Always safe to use (describes past, known at prediction time)
        **kwargs: Additional arguments (ignored)
    
    Returns:
        Predictions [B, pred_len, C]
    """
```

3. **Text selection logic in forward:**
```python
def forward(self, x, news=None, channel_description=None, historical_events=None, **kwargs):
    # Select text input based on timestamp_semantics
    # - t_about: news (y_hetero) contains forecasts ABOUT prediction window, assumed known at t
    # - t_known: news (y_hetero) would be LOOKAHEAD; use historical_events (x_hetero) instead
    if self.timestamp_semantics == 't_about':
        # Fidel-TS datasets: timestamps = t_about (what time text describes)
        # news contains forecasts/schedules for prediction window, assumed known beforehand
        if news is None:
            raise ValueError(
                "timestamp_semantics='t_about' requires 'news' (y_hetero) to be provided. "
                "Ensure task='TGTSF' and y_hetero is configured in data config."
            )
        text_input = news
    elif self.timestamp_semantics == 't_known':
        # Time-MMD/TTC datasets: timestamps = t_known (when text was published)
        # Using news (y_hetero) would be lookahead bias; use historical_events instead
        if historical_events is None:
            raise ValueError(
                "timestamp_semantics='t_known' requires 'historical_events' (x_hetero) to be provided. "
                "Ensure x_hetero is configured in data config."
            )
        text_input = historical_events
    else:
        raise ValueError(f"Invalid timestamp_semantics: {self.timestamp_semantics}")
    
    # Rest of forward pass using text_input
    # ...
```

4. **Update move_to_device:**
```python
def move_to_device(self, seq_x, seq_y, x_time, y_time, 
                   x_hetero, y_hetero, hetero_x_time, hetero_y_time, 
                   hetero_general, hetero_channel, device):
    """
    Move data to device.
    
    Args:
        x_hetero: Historical text embeddings (maps to historical_events in forward)
        y_hetero: Prediction window text embeddings (maps to news in forward)
    """
    seq_x = seq_x.float().to(device)
    seq_y = seq_y.float().to(device)
    hetero_channel = hetero_channel.float().to(device)
    
    # Move text based on timestamp_semantics
    # - t_about: We use y_hetero (news)
    # - t_known: We use x_hetero (historical_events)
    if self.timestamp_semantics == 't_about':
        y_hetero = y_hetero.float().to(device)
    elif self.timestamp_semantics == 't_known':
        x_hetero = x_hetero.float().to(device)
    
    return (seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, 
            hetero_x_time, hetero_y_time, hetero_general, hetero_channel)
```

### 3.4 Config Propagation

The `timestamp_semantics` field needs to be propagated from data config to model config:

**Option A: Add to data config, merge into model configs:**
```yaml
# data_configs/time_mmd/Traffic/config.yaml
hetero_info:
  timestamp_semantics: t_known  # Added field
  # ... existing fields
```

The experiment runner needs to ensure `timestamp_semantics` is available in the configs object passed to the model.

**Option B: Specify in experiment config overrides:**
```yaml
# configs/experiment_suites/lynx_film_raw/time_mmd_ttc.yaml
model_config_overrides:
  timestamp_semantics: t_known
```

---

## 4. Files to Modify

### 4.1 Model Files

| File | Changes |
|------|---------|
| `models/TGTSF.py` | Add `timestamp_semantics` handling, update forward signature and logic |
| `models/lynx_film_raw.py` | Add `timestamp_semantics` handling, update forward signature and logic |
| `models/lynx_film.py` | Add `timestamp_semantics` handling, update forward signature and logic |
| `models/lynx.py` | Add `timestamp_semantics` handling, update forward signature and logic |

### 4.2 Dataset Configs

Add `timestamp_semantics` field:

| Config Pattern | Value |
|---------------|-------|
| `data_configs/Bear_room/*.yaml` | `t_about` |
| `data_configs/California_ISO/*.yaml` | `t_about` |
| `data_configs/Canada_photovoltaics_plants/*.yaml` | `t_about` |
| `data_configs/Germany_Renewable_Power_Grid/*.yaml` | `t_about` |
| `data_configs/Jena_Atmospheric_Physics/*.yaml` | `t_about` |
| `data_configs/NYC_traffic_speed/*.yaml` | `t_about` |
| `data_configs/time_mmd/*/*.yaml` | `t_known` |
| `data_configs/ttc/*/*.yaml` | `t_known` |

### 4.3 Data Provider

| File | Changes |
|------|---------|
| `data_provider/data_factory.py` | Ensure `timestamp_semantics` is extracted from hetero_info and made available to model |

---

## 5. Information Flow Diagrams

### 5.1 TGTSF with `timestamp_semantics: t_about` (Fidel-TS)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         DATA LOADING                                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  x_time: [t-seq_len, ..., t-1]     y_time: [t, t+1, ..., t+pred_len-1]  │
│           ↓                                   ↓                          │
│  hetero_data_getter(x_time)        hetero_data_getter(y_time)           │
│           ↓                                   ↓                          │
│       x_hetero                            y_hetero                       │
│  (historical text)                   (forecast text)                     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                      EXPERIMENT RUNNER                                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  model.forward(                                                          │
│      x=batch_x,                                                          │
│      historical_events=batch_x_hetero,  # x_hetero                       │
│      news=batch_y_hetero,               # y_hetero                       │
│      channel_description=hetero_channel                                  │
│  )                                                                       │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                     MODEL (timestamp_semantics: t_about)                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  # t_about: news contains forecasts ABOUT prediction window              │
│  # Assumed known at prediction time (lead time assumption)               │
│  text_input = news  ← Uses y_hetero                                      │
│                                                                          │
│  # Process time series and text                                          │
│  ts_encoded = self.TS_encoder(x)                                         │
│  text_encoded = self.text_encoder(text_input, channel_description)       │
│  output = self.mixer(text_encoded, ts_encoded)                           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 5.2 TGTSF with `timestamp_semantics: t_known` (Time-MMD/TTC)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         DATA LOADING                                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  x_time: [t-seq_len, ..., t-1]     y_time: [t, t+1, ..., t+pred_len-1]  │
│           ↓                                   ↓                          │
│  hetero_data_getter(x_time)        hetero_data_getter(y_time)           │
│           ↓                                   ↓                          │
│       x_hetero                            y_hetero                       │
│  (text published                     (text published                     │
│   during input window)               during prediction window)           │
│                                      ⚠️ LOOKAHEAD IF USED!               │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                      EXPERIMENT RUNNER                                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  model.forward(                                                          │
│      x=batch_x,                                                          │
│      historical_events=batch_x_hetero,  # x_hetero                       │
│      news=batch_y_hetero,               # y_hetero (NOT USED)            │
│      channel_description=hetero_channel                                  │
│  )                                                                       │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                     MODEL (timestamp_semantics: t_known)                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  # t_known: news would contain text PUBLISHED during prediction window   │
│  # That's lookahead bias! Use historical_events instead.                 │
│  text_input = historical_events  ← Uses x_hetero                         │
│                                                                          │
│  # Process time series and text                                          │
│  ts_encoded = self.TS_encoder(x)                                         │
│  text_encoded = self.text_encoder(text_input, channel_description)       │
│  output = self.mixer(text_encoded, ts_encoded)                           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Lead Time Assumptions for `t_about`

When `timestamp_semantics: t_about`, we assume forecasts have adequate lead time:

| Dataset Type | Recommended Max `pred_len` | Reasoning |
|--------------|---------------------------|-----------|
| Hourly data | 24-48 hours | Daily weather forecasts typically available 1-2 days ahead |
| Daily data | 7 days | Weekly forecasts typically available ~1 week ahead |

**Document this assumption in any experiment using `t_about`:**

```yaml
# In experiment config comments:
# ASSUMPTION: Weather forecasts for t+k were available at time t
# for all k <= pred_len. This is reasonable for pred_len <= 24 hours
# with daily weather forecast lead times.
training:
  output_len: 24  # 24 hours - within typical forecast lead time
```

---

## 7. Testing Plan

### 7.1 Unit Tests

1. **Config validation:**
   - `timestamp_semantics='t_about'` accepted
   - `timestamp_semantics='t_known'` accepted
   - `timestamp_semantics='invalid'` raises ValueError

2. **Text selection:**
   - `t_about` → uses `news`
   - `t_known` → uses `historical_events`
   - Missing required text input raises ValueError

### 7.2 Integration Tests

1. **Fidel-TS + TGTSF + t_about:** Forward pass uses `news`
2. **Time-MMD + TGTSF + t_known:** Forward pass uses `historical_events`
3. **Regression:** Fidel-TS results match previous runs

### 7.3 Expected Behavior

| Config | Dataset | Text Used | Expected Result |
|--------|---------|-----------|-----------------|
| `t_about` | Fidel-TS | `news` (y_hetero) | Normal TGTSF behavior |
| `t_known` | Time-MMD | `historical_events` (x_hetero) | Slightly worse than TGTSF (no future info), but NO lookahead bias |

---

## Appendix A: In-Code Documentation Template

Every TGTSF-family model should include this documentation block:

```python
"""
Text Input Nomenclature
=======================

This model receives text from two sources, with different names in the
dataloader vs model code:

    Dataloader Name    Model Parameter      Description
    ---------------    ---------------      -----------
    x_hetero           historical_events    Text aligned to INPUT window timestamps
    y_hetero           news                 Text aligned to PREDICTION window timestamps

The model uses `timestamp_semantics` to determine which text input to use:

    timestamp_semantics    Text Input Used    Datasets
    -------------------    ---------------    --------
    t_about                news (y_hetero)    Fidel-TS (timestamps = target time)
    t_known                historical_events  Time-MMD, TTC (timestamps = publication time)

For t_about:
    news contains forecasts/schedules ABOUT the prediction window.
    ASSUMPTION: The forecast for time t+k was known at time t.
    This is assumed safe for prediction horizons within typical forecast lead times.

For t_known:
    news would contain text PUBLISHED during the prediction window, which is
    LOOKAHEAD BIAS. Instead, we use historical_events (text about input window).
"""
```

---

## Appendix B: Quick Reference

### Valid Configurations After Implementation

```yaml
# ✅ Fidel-TS with TGTSF (uses news/y_hetero)
data:
  config_path: data_configs/NYC_traffic_speed/config.yaml  # timestamp_semantics: t_about
model:
  config_path: model_configs/general/TGTSF.yaml
# Model uses: news (y_hetero) - forecasts ABOUT prediction window

# ✅ Time-MMD with TGTSF (uses historical_events/x_hetero)
data:
  config_path: data_configs/time_mmd/Traffic/config.yaml  # timestamp_semantics: t_known
model:
  config_path: model_configs/general/TGTSF.yaml
# Model uses: historical_events (x_hetero) - text BEFORE prediction window

# ✅ Time-MMD with TimeCMA (native MTSF model)
data:
  config_path: data_configs/time_mmd/Traffic/config.yaml
model:
  config_path: model_configs/LLM/TimeCMA.yaml
# Model uses: historical_events (x_hetero) - designed for this
```
