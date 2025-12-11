# Embeddings Upgrades: Supporting Token Sequence Dimension

## Overview

This document outlines the plan to upgrade the data provider and model architecture to support preserving the token sequence dimension in text embeddings, enabling text self-attention within articles.

## Current State

### Current Embedding Process

**Location:** `data_provider/data_loader.py`

1. **Text Tokenization** (lines 430-434, 460-464):
   - Input: Raw text strings
   - Process: Tokenization with padding/truncation (max_length=512)
   - Output: `input_ids`, `attention_mask`
   - Shape: `[batch, seq_len]` where `seq_len` ≤ 512

2. **Model Forward Pass** (lines 439-440, 469-470):
   - Input: `input_ids`, `attention_mask`
   - Process: BERT/transformer model forward pass
   - Output: `outputs.last_hidden_state`
   - Shape: `[batch, seq_len, hidden_dim]` where:
     - `seq_len` = token sequence length (up to 512)
     - `hidden_dim` = model embedding dimension (e.g., 768 for BERT-base)

3. **Sequence Dimension Removal** (lines 442, 472):
   - **CRITICAL**: Only [CLS] token is extracted
   - Code: `outputs.last_hidden_state[:, 0, :]`
   - Shape: `[batch, hidden_dim]` (sequence dimension lost!)

4. **Storage** (lines 475, 754):
   - Embeddings stored in DataFrame/stacked
   - Final shape: `[postemb_max_len, postemb_d]` where:
     - `postemb_max_len` = N (number of news items per time step)
     - `postemb_d` = `hidden_dim` (embedding dimension)
   - **No sequence dimension preserved**

### Current Data Flow

```
Raw Text → Tokenization → Model → [batch, seq_len, hidden_dim]
                                    ↓ (extract [CLS] only)
                              [batch, hidden_dim]
                                    ↓ (stack/pad)
                          [postemb_max_len, hidden_dim]
                                    ↓ (to model)
                          [B, L, N, text_dim]
```

**Final shape to models:** `[B, L, N, text_dim]` where:
- B = batch size
- L = temporal sequence length
- N = number of news items per time step
- text_dim = embedding dimension (no token sequence!)

## Desired State

### Desired Embedding Process

**Goal:** Preserve token sequence dimension to enable text self-attention

**Desired Data Flow:**
```
Raw Text → Tokenization → Model → [batch, seq_len, hidden_dim]
                                    ↓ (keep full sequence)
                              [batch, seq_len, hidden_dim]
                                    ↓ (stack/pad)
                    [postemb_max_len, seq_len, hidden_dim]
                                    ↓ (to model)
                    [B, L, N, seq_len, text_dim]
```

**Final shape to models:** `[B, L, N, seq_len, text_dim]` where:
- B = batch size
- L = temporal sequence length
- N = number of news items per time step
- **seq_len = token sequence length (NEW!)**
- text_dim = embedding dimension

## Required Changes

### 1. Data Provider Changes (`data_provider/data_loader.py`)

#### 1.1 Add Configuration Parameter

**Location:** `Heterogeneous_Dataset.__init__()`

**Add parameter:**
```python
postemb_pooling: str = 'cls'  # Options: 'cls', 'mean', 'none'
```

- `'cls'`: Current behavior - extract [CLS] token only (default for backward compatibility)
- `'mean'`: Mean pool over sequence dimension
- `'none'`: Keep full sequence dimension

#### 1.2 Modify `convert_plain_text_to_embeddings()`

**Current (line 442):**
```python
text_embedding = outputs.last_hidden_state[:, 0, :].to('cpu')
return text_embedding[0]  # [hidden_dim]
```

**New:**
```python
if self.postemb_pooling == 'cls':
    text_embedding = outputs.last_hidden_state[:, 0, :].to('cpu')
    return text_embedding[0]  # [hidden_dim]
elif self.postemb_pooling == 'mean':
    # Mean pool over sequence, accounting for padding
    attention_mask_expanded = attention_mask.unsqueeze(-1).float()
    masked_embeddings = outputs.last_hidden_state * attention_mask_expanded
    text_embedding = masked_embeddings.sum(dim=1) / attention_mask_expanded.sum(dim=1).clamp(min=1e-8)
    return text_embedding[0].to('cpu')  # [hidden_dim]
elif self.postemb_pooling == 'none':
    # Keep full sequence
    return outputs.last_hidden_state[0].to('cpu')  # [seq_len, hidden_dim]
```

#### 1.3 Modify `convert_df_text_to_embeddings()`

**Current (line 472):**
```python
batch_embeddings = outputs.last_hidden_state[:, 0, :].to('cpu')
ls_embeddings.extend(batch_embeddings)
```

**New:**
```python
if self.postemb_pooling == 'cls':
    batch_embeddings = outputs.last_hidden_state[:, 0, :].to('cpu')
elif self.postemb_pooling == 'mean':
    attention_mask_expanded = attention_mask.unsqueeze(-1).float()
    masked_embeddings = outputs.last_hidden_state * attention_mask_expanded
    batch_embeddings = masked_embeddings.sum(dim=1) / attention_mask_expanded.sum(dim=1).clamp(min=1e-8)
    batch_embeddings = batch_embeddings.to('cpu')
elif self.postemb_pooling == 'none':
    batch_embeddings = outputs.last_hidden_state.to('cpu')  # [batch, seq_len, hidden_dim]
ls_embeddings.extend(batch_embeddings)
```

**Note:** When `postemb_pooling='none'`, `ls_embeddings` will contain tensors of shape `[seq_len, hidden_dim]` instead of `[hidden_dim]`.

#### 1.4 Modify Storage and Padding Logic

**Location:** Lines 754-761, 818-825

**Current:**
```python
output_dynamic = torch.stack([matched_df['embeddings'] for matched_df in matched_embed], dim=0)
# Shape: [num_matched_times, hidden_dim]

if output_dynamic.size(0) < self.postemb_max_len:
    padding_output = torch.zeros(self.postemb_max_len, self.postemb_d)
    padding_output[:output_dynamic.size(0)] = output_dynamic
    output_dynamic = padding_output
```

**New (handle both cases):**
```python
output_dynamic = torch.stack([matched_df['embeddings'] for matched_df in matched_embed], dim=0)

if self.postemb_pooling == 'none':
    # Shape: [num_matched_times, seq_len, hidden_dim]
    if output_dynamic.size(0) < self.postemb_max_len:
        # Pad along news items dimension
        padding_shape = (self.postemb_max_len - output_dynamic.size(0), 
                        output_dynamic.size(1), 
                        output_dynamic.size(2))
        padding_output = torch.zeros(padding_shape, dtype=output_dynamic.dtype)
        output_dynamic = torch.cat([output_dynamic, padding_output], dim=0)
    elif output_dynamic.size(0) > self.postemb_max_len:
        output_dynamic = output_dynamic[:self.postemb_max_len]
    # Final shape: [postemb_max_len, seq_len, hidden_dim]
else:
    # Shape: [num_matched_times, hidden_dim]
    if output_dynamic.size(0) < self.postemb_max_len:
        padding_output = torch.zeros(self.postemb_max_len, self.postemb_d)
        padding_output[:output_dynamic.size(0)] = output_dynamic
        output_dynamic = padding_output
    elif output_dynamic.size(0) > self.postemb_max_len:
        output_dynamic = output_dynamic[:self.postemb_max_len]
    # Final shape: [postemb_max_len, hidden_dim]
```

#### 1.5 Handle Pre-computed Embeddings (`load_embedding()`)

**Location:** Lines 563-644

**Issue:** Pre-computed embeddings (from `.pkl` files) may have been created with old pooling method.

**Solution:**
- Add validation to check embedding shape
- If shape is `[hidden_dim]` but `postemb_pooling='none'`, raise warning/error
- If shape is `[seq_len, hidden_dim]` but `postemb_pooling='cls'`, handle appropriately
- Document that pre-computed embeddings must match `postemb_pooling` setting

### 2. Data Factory Changes (`data_provider/data_factory.py`)

#### 2.1 Pass `postemb_pooling` Parameter

**Location:** Wherever `Heterogeneous_Dataset` is instantiated

**Add parameter:**
```python
hetero_dataset = Heterogeneous_Dataset(
    ...,
    postemb_pooling=configs.postemb_pooling if hasattr(configs, 'postemb_pooling') else 'cls'
)
```

### 3. Configuration Changes

#### 3.1 Add to Config Files

**Location:** Model config YAML files (e.g., `model_configs/general/lynx_mmitransformer.yaml`)

**Add:**
```yaml
# Text Embedding Parameters
postemb_pooling: 'none'  # 'cls' (default), 'mean', or 'none' (keep sequence)
```

### 4. Model Changes

#### 4.1 Update Model Input Handling

**Models that need updates:**
- `lynx_mmitransformer` (primary target)
- `mmitransformer` (if needed)
- `lynx_film` (if needed)
- `TGTSF` (if needed)

**Changes needed:**
1. Detect input shape (4D vs 5D)
2. Handle both cases gracefully
3. Add text self-attention when sequence dimension available

#### 4.2 Shape Detection Logic

```python
if len(news.shape) == 4:
    # Current: [B, L, N, text_dim]
    has_sequence_dim = False
elif len(news.shape) == 5:
    # New: [B, L, N, seq_len, text_dim]
    has_sequence_dim = True
    seq_len = news.shape[3]
else:
    raise ValueError(f"Unexpected news shape: {news.shape}")
```

### 5. Lightning Module Changes (`exp/exp_lightning.py`)

#### 5.1 Pass Configuration

**Location:** Where data loaders are created

**Ensure `postemb_pooling` is passed from config to data factory**

### 6. Testing and Validation

#### 6.1 Unit Tests

**Create tests for:**
1. `postemb_pooling='cls'` (backward compatibility)
2. `postemb_pooling='mean'` (mean pooling)
3. `postemb_pooling='none'` (full sequence)
4. Shape validation at each step
5. Padding/truncation logic for both cases

#### 6.2 Integration Tests

**Test:**
1. End-to-end data flow with sequence dimension
2. Model forward pass with 5D input
3. Backward compatibility with 4D input

### 7. Documentation Updates

#### 7.1 Update Docstrings

**Update:**
- `Heterogeneous_Dataset` class docstring
- Method docstrings for embedding functions
- Configuration documentation

#### 7.2 Migration Guide

**Create guide for:**
- Upgrading existing pre-computed embeddings
- Configuring new models to use sequence dimension
- Backward compatibility considerations

## Implementation Order

1. **Phase 1: Data Provider** (Backward Compatible)
   - Add `postemb_pooling` parameter (default 'cls')
   - Modify embedding functions to support all pooling options
   - Update storage/padding logic
   - Add validation and error handling

2. **Phase 2: Configuration**
   - Add `postemb_pooling` to config files
   - Update data factory to pass parameter

3. **Phase 3: Model Updates**
   - Update `lynx_mmitransformer` to handle 5D input
   - Add text self-attention when sequence available
   - Update other models as needed

4. **Phase 4: Testing**
   - Unit tests for data provider
   - Integration tests
   - Backward compatibility tests

5. **Phase 5: Documentation**
   - Update docstrings
   - Create migration guide
   - Update README

## Backward Compatibility

### Critical Considerations

1. **Default Behavior**: `postemb_pooling='cls'` maintains current behavior
2. **Pre-computed Embeddings**: Must match pooling method used
3. **Model Compatibility**: Models must handle both 4D and 5D inputs gracefully
4. **Config Migration**: Old configs without `postemb_pooling` default to 'cls'

### Breaking Changes

**None** - All changes are backward compatible with proper defaults.

## Performance Considerations

### Memory Impact

**With `postemb_pooling='none'`:**
- Memory increase: `seq_len` × `hidden_dim` per news item
- Example: 512 tokens × 768 dim = 393,216 floats per news item (vs 768 with 'cls')
- For batch of 32, L=12, N=10: ~1.5GB additional memory

### Computation Impact

**Text Self-Attention:**
- Complexity: O(seq_len²) per news item
- For seq_len=512: ~262K operations per news item
- Can be significant for large batches

### Optimization Strategies

1. **Truncation**: Reduce `max_length` if sequence too long
2. **Gradient Checkpointing**: For memory efficiency
3. **Mixed Precision**: Use FP16/BF16
4. **Selective Usage**: Only use sequence dimension when needed

## Future Enhancements

1. **Adaptive Pooling**: Learn pooling weights instead of fixed methods
2. **Hierarchical Attention**: Attention over tokens, then over news items
3. **Sparse Attention**: Reduce computation for long sequences
4. **Caching**: Cache token-level embeddings for reuse



---

some other useful information. This should be included in in-code comments in the data provider and in configs where we can choose between cls token or including the entire sequence. The difference for us is that if we extract the CLS token, we get generic representation of the entire sequence (untrained / unsupervised by our task), but if we take the entire sequence, the attention across the sequence can be trained / supervised by our task, which for longer passages of text could be incredibly useful. This should also be documented in code.

BERT (Bidirectional Encoder Representations from Transformers) produces **embeddings for every token in the input passage**, and it also provides a special embedding—the **CLS embedding**—which is commonly used as a single representation for the **entire passage**.

Here is a breakdown of what BERT produces and the role of the CLS embedding:

---

## 📝 BERT's Output Embeddings

BERT takes an input text passage and transforms it into a sequence of **contextual embeddings**.

* **Embeddings Per Token (Word Embeddings):** BERT generates a unique, high-dimensional vector (an embedding) for **every token** in the input sequence.
    * **Contextual:** Unlike older models (like Word2Vec or GloVe) that assigned a single, fixed vector to a word regardless of its usage, BERT's embeddings are **contextual**. This means the embedding for the word "bank" will be different in "river bank" versus "money bank."
    * **Application:** These token-level embeddings are crucial for **sequence labeling tasks** like Named Entity Recognition (NER), Part-of-Speech (POS) tagging, and Question Answering, where you need to classify or predict an output for each individual word/token.
    * **Final Output:** The final output of the BERT encoder is a sequence of vectors, one for each token in the input.

---

## 🏷️ What is the CLS Embedding?

The CLS embedding is the final hidden state vector corresponding to a special token, **`[CLS]`**, which is strategically added to the very beginning of the input passage.

### 1. The `[CLS]` Token

* **Classification Token:** `[CLS]` stands for **Classification**.
* **Input Modification:** Before any text is passed to BERT, the tokenizer prepends the `[CLS]` token to the input sequence (e.g., `[CLS] The movie was great [SEP]`). A `[SEP]` (Separator) token is also added at the end of a sentence or between two sentences.
* **Bidirectional Attention:** Since BERT's Transformer layers are bidirectional (via the self-attention mechanism), the `[CLS]` token is allowed to "attend" (pay attention) to every other token in the input sequence, and vice versa.

### 2. The CLS Embedding's Role

* **Aggregate Representation:** By attending to all other tokens, the final hidden state vector of the `[CLS]` token (the CLS embedding) is designed to capture a **summary or aggregate representation of the entire input passage**.
* **Classification Tasks:** This is its primary intended use. For tasks like **sentiment analysis** or **topic classification**—where you need one label for the entire text—the CLS embedding is typically extracted and fed into a simple classification layer (e.g., a fully connected layer) for the final prediction.

In summary, BERT produces **both**: a sequence of **per-token embeddings** and a single **CLS embedding** that represents the entire passage. 
