# Tensor Cache Embedding Shapes: Understanding N vs C

## Overview

This document explains a critical distinction in the embedding tensor shapes used by the tensor cache and text-guided time series forecasting models. It covers:

1. **N vs C dimensions** - The semantic difference between num_news_items and num_channels
2. **Downtime handling** - How the tensor cache decides to store N=1 or N=2
3. **Memory implications** - Why these choices matter for performance

## The Problems We Fixed

### Problem 1: N vs C Confusion (21x Memory Explosion)

The tensor cache was incorrectly creating embedding tensors with shape `(L, n_features, D)` where:
- `L` = number of timesteps (after hetero_stride)
- `n_features` = number of time series channels (e.g., 21 for Jena)
- `D` = embedding dimension (e.g., 1536)

With `batch_size=768` and `n_features=21`, this created **~50GB of memory allocation** per batch, causing OOM kills on 64GB RAM systems.

**Fix:** We now correctly use `num_news_items` (N), not `n_features` (C), for the embedding dimension.

### Problem 2: Downtime Indicator Handling (2x Memory Issue)

The original dataloader produces embeddings with shape `(L, 2, D)`:
- `[:, 0, :]` = actual text embedding  
- `[:, 1, :]` = downtime indicator (non-zero if sensor was down, zeros otherwise)

Early tensor cache implementations flattened this to `(L, 1536)` (where 1536 = 768 * 2), doubling the apparent embedding dimension.

**Fix:** We now intelligently handle downtime:
- **N=1** when training has no downtime: store only embeddings, 50% memory savings
- **N=2** when training has downtime: store embedding + downtime, matching original dataloader

## The Critical Distinction: N vs C

### Two Different Dimensions

The models use 4D embedding tensors with shape `[B, L, N, D]`, but there are **two different semantic meanings** for what we'll call "the middle dimension":

| Dimension | Symbol | Meaning | Source |
|-----------|--------|---------|--------|
| **num_news_items** | N | Number of news articles per timestamp | `news_emb` input |
| **num_channels** | C | Number of time series channels/features | `channel_description` input |

### Why They're Different

In the `text_encoder` (used by TGTSF, lynx_film, lynx_film_raw):

```python
def forward(self, news_emb, description_emb):
    # news_emb:        [B, L, N, D]  - N news items per timestamp
    # description_emb: [B, L, C, D]  - C channel descriptions
    
    # Cross-attention: channels attend to news
    # Query: description_emb (channels)
    # Key/Value: news_emb (news items)
```

The cross-attention mechanism works as follows:
- **Query** = channel descriptions `[B, L, C, D]` - one query per channel
- **Key/Value** = news embeddings `[B, L, N, D]` - news items to attend over

Each channel independently attends to all N news items. The number of channels (C) comes from `description_emb`, NOT from `news_emb`.

### N and C Are Independent

This means:
- **N can be 1** even when C is 21
- **N can be 5** even when C is 1
- They are completely independent dimensions

## Downtime Handling: N=1 vs N=2

### What is the Downtime Indicator?

The original Fidel-TS dataloader (`Heterogeneous_Dataset.get_hetero_data()`) produces embeddings with shape `(L, 2, D)`:

```python
# From data_loader.py, line ~937
# Concatenate dynamic embeddings and downtime indicators along num_items dimension
# Final shape: (batch, 2, embedding_dim) where 2 = num_items (dynamic + downtime)
output_dynamic = np.concatenate([output_dynamic_, downtime_data_], axis=1)
```

The two items are:
- **Item 0** (`[:, 0, :]`): The actual text embedding (e.g., weather forecast embedding)
- **Item 1** (`[:, 1, :]`): Downtime indicator
  - `zeros` if sensor was operating normally
  - `downtime_prompt_embedding` if sensor was down at that timestamp

### Why Does Downtime Matter?

The downtime indicator tells the model:
- "At this timestamp, the sensor was down - data may be unreliable"
- The model can learn to weight down predictions near downtime periods
- Useful for datasets with equipment failures, maintenance windows, etc.

### Intelligent Downtime Detection

The tensor cache now **automatically detects** whether training data has meaningful downtime:

```python
# During cache generation
has_downtime = _detect_downtime_in_training(data_provider)
num_news_items = 2 if has_downtime else 1
```

**Detection logic:**
1. Scan training samples (up to 1000 for efficiency)
2. Check if any `hetero_x[:, 1, :]` or `hetero_y[:, 1, :]` is non-zero
3. If ANY non-zero downtime found → N=2
4. If ALL zeros → N=1

### Why Only Check Training Data?

**Key insight:** If there's no downtime in training, the model won't learn to use it.

Even if validation/test sets have downtime:
- The model never saw downtime during training
- It can't learn the relationship between downtime and predictions
- Storing downtime indicators would just waste memory

### Storage Format

| Scenario | num_news_items | Storage Shape | Returned Shape |
|----------|----------------|---------------|----------------|
| No downtime in training | N=1 | `(L, D)` | `(L, 1, D)` |
| Downtime in training | N=2 | `(L, 2, D)` | `(L, 2, D)` |

### CLI Messages

When generating a tensor cache, you'll see one of these messages:

**No downtime (N=1):**
```
[ info ] No downtime in training data - storing embeddings without 
downtime indicators (N=1). Even if val/test have downtime, the model 
won't have learned to use it, so omitting saves memory.
```

**Has downtime (N=2):**
```
[ info ] Downtime detected in training data - storing embeddings with 
downtime indicators (N=2). The model will learn to use downtime information.
```

### Metadata Storage

The `num_news_items` value is stored in `metadata.json`:

```json
{
  "version": "2.1.0",
  "num_news_items": 1,  // or 2
  ...
}
```

This allows `TensorCacheDataset` to correctly reconstruct embeddings at load time.

## Which Datasets Are Affected?

### Time-MMD Datasets: Always N=1 (No Downtime Issue)

**Time-MMD datasets never had the downtime doubling problem.**

Looking at `data_provider/time_mmd_dataset.py`, the `TimeMMDTextGetter.get_hetero_data()` method returns:

```python
# Stack to shape: (num_timesteps, 1, bert_dim)
# This matches expected format: (seq_len, news_num, bert_dim)
# where news_num=1 for Time-MMD (single text per timestamp)
output_dynamic = np.stack(embedding_list, axis=0)  # (num_timesteps, 1, bert_dim)
```

**Key characteristics:**
- Always returns `(L, 1, D)` shape - already correct
- No downtime indicator - simpler format
- Single text description per timestamp
- Never concatenates additional information

**Examples:** Economy datasets, Climate datasets, all MM-TSFlib format datasets

### Fidel-TS Datasets: N=2 With Downtime (Had the Issue)

**Fidel-TS datasets use the `Heterogeneous_Dataset` class which concatenates downtime indicators.**

Looking at `data_provider/data_loader.py`, the `Heterogeneous_Dataset.get_hetero_data()` method does:

```python
# From data_loader.py, line ~937
# Concatenate dynamic embeddings and downtime indicators along num_items dimension
# Final shape: (batch, 2, embedding_dim) where 2 = num_items (dynamic + downtime)
output_dynamic = np.concatenate([output_dynamic_, downtime_data_], axis=1)
```

**Key characteristics:**
- Returns `(L, 2, D)` shape where:
  - `[:, 0, :]` = actual text embedding (e.g., weather forecast)
  - `[:, 1, :]` = downtime indicator (non-zero if sensor down, zeros otherwise)
- Designed for datasets with potential sensor failures/maintenance
- More complex format to support operational metadata

**Examples:** Jena_Atmospheric_Physics, Bear_room, California_ISO, Canada_photovoltaics_plants, Germany_Renewable_Power_Grid, NYC_traffic_speed

### Why Fidel-TS Had the 2x Dimension Issue

The old tensor cache implementation would:

1. Receive `(L, 2, D)` from `Heterogeneous_Dataset`
2. Extract per-timestep: `emb = hetero_x[i]` → shape `(2, D)`
3. Flatten: `emb.flatten()` → shape `(2*D,)` = `(1536,)` when D=768
4. Store the flattened 1536-dim vector

This **doubled the embedding dimension** from 768 to 1536, which then caused:
- Mismatch with model expectations (`input_text_dim: 768`)
- Shape errors during projection
- 2x memory usage for embeddings

### The Fix

The new implementation:
- **Detects** if training data has non-zero downtime indicators
- **Time-MMD**: Always N=1 (unchanged, already correct)
- **Fidel-TS with downtime**: N=2 (stores full `(2, D)` array)
- **Fidel-TS without downtime**: N=1 (omits unused downtime, 50% savings)

### Dataset Type Summary

| Dataset Family | Class | Original Shape | Downtime? | Tensor Cache N |
|----------------|-------|----------------|-----------|----------------|
| **Time-MMD** | `TimeMMD_Dataset` | `(L, 1, D)` | Never | Always 1 |
| **TTC** | `TimeMMD_Dataset` | `(L, 1, D)` | Never | Always 1 |
| **Fidel-TS (no downtime)** | `Heterogeneous_Dataset` | `(L, 2, D)` but `[:, 1, :] = 0` | No (all zeros) | 1 (auto-detected) |
| **Fidel-TS (with downtime)** | `Heterogeneous_Dataset` | `(L, 2, D)` | Yes | 2 (auto-detected) |

## Why N=1 Is Semantically Correct (When No Downtime)

### What N Represents

In the original Fidel-TS/Time-MMD datasets, N represents "number of news articles per timestamp." The tensor cache stores **one aggregated embedding per timestamp** because:

1. **Embedding aggregation during cache generation**: When building the cache, embeddings are stored per-timestamp, not per-news-article
2. **Semantic equivalence**: Having N=1 means "one news context per timestamp" which is exactly what we have

### Cross-Attention Still Works

With N=1:
- Each of the C channels attends to the single news embedding
- All channels see the same textual context for that timestamp
- This is semantically equivalent to "the news at this timestamp applies to all channels"

This is correct because:
- News articles describe general events (e.g., "storm approaching")
- These events affect all channels equally
- There's no channel-specific news in the dataset

## Can We Always Use N=1?

### Short Answer: Only If No Downtime in Training

The tensor cache automatically decides N based on training data:
- **N=1** when training has no downtime → 50% memory savings, semantically correct
- **N=2** when training has downtime → preserves downtime signal for model learning

### When N=1 Is Appropriate

When there's no downtime in training, N=1 is always correct because:

1. **Concatenating text before embedding**: Merge all news texts into one before computing the embedding
2. **Averaging embeddings**: Take mean of multiple news embeddings: `(L, N, D) -> mean(axis=1) -> (L, D) -> expand -> (L, 1, D)`
3. **Attention-weighted pooling**: Learn to combine multiple news items

### Reasoning for N=1 (No Downtime Case)

1. **Information preservation**: LLM embeddings are high-dimensional (768-1536 dims). Concatenating multiple short news items into one longer text before embedding preserves more information than separate embeddings that get averaged later.

2. **Computational efficiency**: N=1 means:
   - 50% less memory than N=2
   - Faster cross-attention (linear in N)
   - Smaller tensor cache files

3. **Semantic correctness**: The cross-attention mechanism already handles combining information from multiple news items. With N=1, this combination happens during embedding generation rather than during model forward pass - but the end result is equivalent.

4. **Practical equivalence**: In the datasets we use:
   - Time-MMD: One text description per timestamp
   - Fidel-TS: Forecasts are typically aggregated per timestamp
   - There's rarely meaningful distinction between multiple news items at the same timestamp

### When N=2 Is Required

N=2 is required when:
- **Training data has downtime events** (non-zero downtime indicators)
- The model should learn to use downtime information for better predictions
- Backward compatibility with models trained on original dataloader output

The tensor cache automatically detects this by scanning training samples for non-zero downtime indicators.

## Memory Impact Summary

| Configuration | Memory per Batch (768 samples) | Status |
|--------------|-------------------------------|--------|
| N=21 (old, incorrect) | ~50 GB | OOM on 64GB systems |
| N=2 (downtime in training) | ~4.8 GB | Works fine |
| N=1 (no downtime in training) | ~2.4 GB | Works fine, 50% savings |

## Affected Models

All models that use the tensor cache handle both N=1 and N=2 correctly:

| Model | How It Handles N | N=1 Safe | N=2 Safe |
|-------|------------------|----------|----------|
| `lynx_film_raw` | Cross-attention in `text_encoder` | ✓ | ✓ |
| `lynx_film` | Cross-attention in `text_encoder` | ✓ | ✓ |
| `TGTSF` | Cross-attention in `text_encoder` | ✓ | ✓ |
| `MMTSFlib` | `mean(dim=(1,2))` aggregates L and N | ✓ | ✓ |
| `ZhangHanBest` | `mean(dim=(1,2))` aggregates L and N | ✓ | ✓ |

## How Models Dynamically Handle N=1 vs N=2

**The model doesn't need to know ahead of time whether N=1 or N=2 - it dynamically adapts to whatever shape it receives.**

This is a key design feature that makes the system robust and eliminates the need for configuration changes when switching between datasets.

### The Flow

#### 1. Tensor Cache Stores Shape Information

```python
# In metadata.json
{
  "version": "2.1.0",
  "num_news_items": 1,  // or 2, determined during cache generation
  ...
}
```

#### 2. TensorCacheDataset Returns Correct Shape

```python
# In data_provider/tensor_cache.py, _getitem_indexed()
num_news_items = self.metadata.num_news_items

if num_news_items == 1:
    # Expand (L, D) -> (L, 1, D)
    hetero_x = np.expand_dims(hetero_x, axis=1)
else:
    # Already (L, 2, D), no expansion needed
    pass

# Returns: (L, N, D) where N is read from metadata
```

#### 3. Model Receives and Dynamically Adapts

The key is in `layers/TGTSF_torch.py`, the `text_encoder.forward()` method:

```python
def forward(self, news_emb, description_emb):
    # news_emb: [b, l, n, d] - N can be 1 or 2, doesn't matter!
    # description_emb: [b, l, c, d]
    
    B, L, C, D = description_emb.shape
    
    # Dynamically read N from the actual tensor shape
    news_emb = news_emb.contiguous().view(B*L, news_emb.shape[2], D)  # [b*l, n, d]
    #                                              ^^^^^^^^^^^^
    #                                              This is N - read from actual shape!
    
    news_mask = news_emb.sum(dim=-1) == 0  # (B*L, N) - works for any N
    
    # Cross-attention: each channel attends to all N news items
    text_emb = self.cross_encoder(tgt=description_emb, memory=news_emb, 
                                   memory_key_padding_mask=news_mask)
```

### Why This Works

The model uses **dynamic shapes** - it reads the N dimension from `news_emb.shape[2]` at runtime:

**N=1 case (no downtime):**
- Tensor cache returns `(L, 1, D)`
- After batching: `news_emb` is `[B, L, 1, D]`
- `news_emb.shape[2]` = 1
- Cross-attention attends to 1 news item per timestamp

**N=2 case (with downtime):**
- Tensor cache returns `(L, 2, D)`
- After batching: `news_emb` is `[B, L, 2, D]`
- `news_emb.shape[2]` = 2
- Cross-attention attends to 2 items per timestamp (embedding + downtime indicator)

### The Cross-Attention Advantage

Unlike traditional architectures where layer sizes are fixed at initialization:

```python
# Traditional: Fixed size known at init
self.linear = nn.Linear(input_size, output_size)  # Must know sizes at creation!

# Cross-attention: Dynamic over sequence dimension
self.cross_encoder = nn.TransformerDecoder(...)  # N is a sequence dimension!
#                                                 # Can be any length!
```

PyTorch's `MultiheadAttention` and `TransformerDecoder` handle variable-length sequences naturally:

```python
# Query: description_emb [B*L, C, D] - C channels
# Key/Value: news_emb [B*L, N, D] - N news items (1 or 2)

# Each of C channels independently attends to all N items
# Works for any N - it's just the sequence length!
```

### No Model Reconfiguration Needed

**Benefits of this design:**

1. ✅ **Same model weights** work with N=1 or N=2
2. ✅ **No retraining** needed when switching datasets
3. ✅ **No configuration changes** in model config files
4. ✅ **Automatic adaptation** based on tensor cache metadata
5. ✅ **Mixed datasets** - can train on N=1 data, evaluate on N=2 (though not recommended)

**The model automatically handles N=1 or N=2 because:**
- Tensor cache metadata stores `num_news_items`
- `TensorCacheDataset` reads it and returns the correct shape
- Model dynamically reads N from `news_emb.shape[2]`
- Cross-attention naturally handles variable N (it's a sequence length, not a parameter)

This is one of the key advantages of the cross-attention architecture - **the N dimension is treated as a sequence length**, not a fixed parameter, so it can vary between datasets without any model changes.

## Code Location

### Downtime Detection

Located in `data_provider/tensor_cache.py`, function `_detect_downtime_in_training()`:

```python
def _detect_downtime_in_training(data_provider, max_samples_to_check=1000):
    """Scan training samples to detect non-zero downtime indicators."""
    datasets = data_provider.get_datasets('train')
    for entity_id, dataset in datasets.items():
        for sample in dataset:
            hetero_x = sample[SAMPLE_IDX_HETERO_X]
            if hetero_x is not None and hetero_x.ndim == 3 and hetero_x.shape[1] >= 2:
                downtime_indicator = hetero_x[:, 1, :]
                if np.any(downtime_indicator != 0):
                    return True
    return False
```

### Embedding Storage

Located in `data_provider/tensor_cache.py`, function `_register_timestamp_data()`:

```python
if collector.num_news_items == 1:
    # N=1: Store only text embedding as (D,)
    emb_to_store = emb_arr[0] if emb_arr.ndim == 2 else emb_arr
else:
    # N=2: Store embedding + downtime as (2, D)
    emb_to_store = emb_arr[:2]
```

### Embedding Retrieval

Located in `data_provider/tensor_cache.py`, method `_getitem_indexed()`:

```python
num_news_items = self.metadata.num_news_items

if num_news_items == 1:
    # N=1: stored as (L, D) -> expand to (L, 1, D)
    hetero_x = np.expand_dims(hetero_x, axis=1)
# else: N=2, already stored as (L, 2, D), no expansion needed
```

## Channel Descriptions vs News: Two Types of Text Input

The models in this codebase receive **two fundamentally different types of textual input**, which have different semantics and different handling:

### Channel Descriptions (Channel-Specific, Static)

**What they are:**
- `hetero_channel` / `channel_description` in the dataloader
- Describe what each time series channel represents (e.g., "Temperature in Celsius", "Wind Speed in m/s")
- **Static per entity**: The same for all timestamps within an entity
- **Channel-specific**: Each of the C channels has its own description

**Shape:** `[B, C, D]` or `[B, 1, C, D]`
- C = number of channels
- D = embedding dimension
- The C dimension is meaningful - each channel has unique text

### News Embeddings (Global, Dynamic)

**What they are:**
- `hetero_x` / `hetero_y` / `news` / `historical_events` in the dataloader
- Describe events or context at each timestamp (e.g., "Storm approaching", "Holiday weekend")
- **Dynamic over time**: Different text at each timestamp
- **Global across channels**: The same news applies to all channels at a given timestamp

**Shape:** `[B, L, N, D]`
- L = number of timesteps
- N = number of news items per timestamp (typically 1)
- D = embedding dimension
- The N dimension is for multiple news articles per timestamp, NOT for channels

### How Models Handle This Distinction

#### TGTSF, lynx_film, and lynx_film_raw: Cross-Attention Fusion

These models use the `text_encoder` which performs **cross-attention** between channels and news:

```python
def forward(self, news_emb, description_emb):
    # news_emb:        [B, L, N, D]  - N news items (global, same for all channels)
    # description_emb: [B, L, C, D]  - C channel descriptions (channel-specific)
    
    # Cross-attention mechanism:
    # Query: description_emb (channels ask: "what news is relevant to me?")
    # Key/Value: news_emb (news provides information for channels to attend to)
```

**The fusion process:**
1. Each channel's description embedding becomes a query
2. The news embeddings become keys and values
3. Cross-attention allows each channel to "read" the news through its own lens
4. Output: Channel-specific representations that incorporate global news

**Why this works with N=1:**
- Even with a single news embedding per timestamp, each channel independently attends to it
- The cross-attention learns how relevant that global news is to each specific channel
- A "storm approaching" news might get high attention from "Wind Speed" and low attention from "Indoor Temperature"

#### MMTSFlib and ZhangHanBest: Aggregation Approach

These models don't use cross-attention; they simply aggregate:

```python
# MMTSFlib and ZhangHanBest
if text_emb.dim() == 4:
    # [B, L, N, D] -> [B, D]
    text_emb = text_emb.mean(dim=(1, 2))  # Average across time AND news items
```

This approach:
- Loses temporal alignment (all timesteps averaged)
- Loses any distinction between multiple news items
- Produces a single global text representation per sample
- Is inherently compatible with N=1 (nothing to average)

### Why the Original Bug Caused Memory Explosion

The original tensor cache code incorrectly assumed:

> "News embeddings should have shape `[B, L, C, D]` where C = n_features"

This led to `np.repeat(news_emb, n_features, axis=1)`, creating 21 copies of each news embedding.

**Why this was wrong:**
- News is **global** (same for all channels) - it doesn't need a C dimension
- The C dimension in the model comes from `channel_description`, not from `news_emb`
- The cross-attention handles the channel-to-news relationship dynamically

**Memory impact:**
- With n_features=21, this created 21x memory overhead
- The repeated data was semantically identical (just copies)
- No model actually used the repeated dimension meaningfully

### Implications for Data Loading

| Text Type | Source | Scope | Tensor Cache Shape | Repeated? |
|-----------|--------|-------|-------------------|-----------|
| Channel Description | `hetero_channel` | Per-channel, static | `[C, D]` per entity | No (each is unique) |
| News Embeddings | `hetero_x`/`hetero_y` | Global, per-timestamp | `[L, 1, D]` | No (N=1 is correct) |

The fix correctly recognizes that:
- **Channel descriptions** need a C dimension because each channel's description is different
- **News embeddings** don't need a C dimension because the same news applies to all channels

## Future Architecture: Global vs Local News

The current models treat all news as **global** (affecting all channels equally). However, real-world data often has **channel-specific news**:

- Stock market: News specifically about Apple vs general macro news
- Energy grid: News about a specific power plant vs weather affecting the whole region
- Products: Reviews for a specific product vs general consumer sentiment

### The Panel Data Architecture (See `future/panel-a.md`)

The document `future/panel-a.md` outlines a future architecture that handles both **global** and **local** (channel-specific) news:

#### The "Hub-and-Spoke" FiLM Architecture

1. **Local Modulation (Hard-wired):**
   - Entity-specific text directly modulates only that entity's time series
   - Apple's 10-K filing immediately affects Apple's stock prediction
   - No learning required for this direct connection

2. **Global Context (Learned):**
   - K learned "Global Factor" tokens aggregate market-wide information
   - Each entity attends to these global factors
   - Cross-entity effects (Apple news affecting Microsoft) are learned through this hub

3. **Macro News (Broadcast):**
   - Generic news (interest rates, holidays) affects all entities
   - Injected into every entity's representation

#### How This Relates to N

In the future architecture:
- **N could vary per entity**: Apple might have 3 news items while Microsoft has 1
- **Some N items are local**: Apple's earnings call only for Apple
- **Some N items are global**: Macro news shared across all entities
- The architecture routes information correctly without repeating embeddings

### Can We Always Use N=1? Revisited

For **current models** (TGTSF, lynx_film, lynx_film_raw): **Yes, N=1 is always correct.**

These models treat all news as global anyway. Even if we had multiple news items per timestamp, the optimal approach is:
1. **Concatenate texts before embedding**: "Storm approaching. Holiday weekend." → single embedding
2. **This preserves more information** than separate embeddings averaged later
3. **Memory efficient**: No redundant copies

For **future panel-aware models**: N might need structure:
- Local news: `N_local` items specific to each entity
- Global news: `N_global` items shared across entities
- But even then, **we wouldn't repeat** - each news item exists once and is routed appropriately

## Conclusion

The distinction between `num_news_items` (N) and `num_channels` (C) is subtle but critical:
- **N** = how many news articles per timestamp (from the dataset, typically 1)
- **C** = how many time series channels (from the data config, e.g., 21)

These are independent dimensions serving different semantic purposes:
- **Channel descriptions** are **local** (channel-specific) and need dimension C
- **News embeddings** are **global** (shared across channels) and only need dimension N (typically 1)

The original code incorrectly assumed N should equal C, leading to massive memory waste. The fix correctly uses N=1, which is both:
1. **Semantically correct**: One aggregated news embedding per timestamp, shared across channels
2. **Memory efficient**: 21x reduction for datasets with many channels
3. **Forward compatible**: Future panel-aware architectures will handle local vs global news through routing, not repetition
