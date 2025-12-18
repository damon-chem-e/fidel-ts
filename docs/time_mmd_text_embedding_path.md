# Where Text Gets Embedded: Time-MMD → Model Forward Pass

## Current State: Time-MMD Returns JSON Strings

When using `configs/experiments/time_mmd_test.yaml`:

1. **Data Provider Output**: `TimeMMD_HeteroGetter` returns **list of JSON strings**
   - Format: `['{"text": "..."}', '{"text": "..."}', ...]`
   - Location: `data_provider/time_mmd_dataset.py` line 123-125

2. **DataLoader Output**: JSON strings are passed through unchanged
   - `batch_x_hetero`: List of lists of JSON strings
   - `batch_y_hetero`: List of lists of JSON strings

3. **Experiment Class**: Passes JSON strings directly to model
   - Location: `exp/exp_universal.py` line 177
   - `self.model(x=batch_x, historical_events=batch_x_hetero, news=batch_y_hetero, ...)`

## The Problem: Models Expect Embeddings

**TGTSF Model** expects:
- `news`: `[Batch, l, news_num, text_dim]` - **Already embeddings**, not JSON strings
- Location: `models/TGTSF.py` line 189, 207

**Current Mismatch**:
- Time-MMD provides: JSON strings (list of strings)
- TGTSF expects: Embeddings (torch.Tensor with shape `[B, l, n, d]`)

## Where Embedding Should Happen (But Currently Doesn't)

### Option 1: In the Model's Forward Pass (Not Implemented)

**Location**: `models/TGTSF.py` → `forward()` method

**What should happen** (but doesn't currently):
```python
def forward(self, x, news, channel_description, **kwargs):
    # news is currently expected to be embeddings [B, l, n, d]
    # But with Time-MMD, news is JSON strings
    
    # SHOULD DO (but doesn't):
    if isinstance(news, list):  # JSON strings from Time-MMD
        # Convert JSON strings to embeddings
        news_embeddings = self._embed_text(news)  # [B, l, n, d]
    else:
        news_embeddings = news  # Already embeddings
    
    t = self.text_encoder(news_embeddings, description)
    # ... rest of forward pass
```

**Current Reality**: TGTSF doesn't have this conversion logic, so it would fail if given JSON strings.

### Option 2: In move_to_device() Method (Not Implemented)

**Location**: `models/TGTSF.py` → `move_to_device()` method (line 68-75)

**What should happen** (but doesn't currently):
```python
def move_to_device(self, seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, ...):
    # x_hetero and y_hetero are JSON strings from Time-MMD
    
    # SHOULD DO (but doesn't):
    if isinstance(y_hetero, list):  # JSON strings
        # Convert to embeddings
        y_hetero = self._convert_json_to_embeddings(y_hetero)  # [B, l, n, d]
    
    y_hetero = y_hetero.float().to(device)
    # ... rest
```

**Current Reality**: `move_to_device()` just moves tensors to device, doesn't convert text.

### Option 3: In TimeMMD_HeteroGetter (Not Implemented)

**Location**: `data_provider/time_mmd_dataset.py` → `TimeMMD_HeteroGetter.__call__()`

**What should happen** (but doesn't currently):
```python
def __call__(self, timestamps):
    matched_times, matched_texts = self._match_timestamps(timestamps)
    
    # Currently returns JSON strings
    if self.output_format == 'json':
        output_dynamic = [json.dumps({'text': text}) for text in matched_texts]
    
    # SHOULD ALSO SUPPORT (but doesn't):
    elif self.output_format == 'embedding':
        # Convert text to embeddings on-the-fly
        embeddings = self._get_embeddings(matched_texts)  # Uses tokenizer/model
        output_dynamic = embeddings  # torch.Tensor [len, embedding_dim]
```

**Current Reality**: `output_format='embedding'` returns zeros (line 132-145).

## How Standard fidel-ts Handles This

### Standard Flow (with Heterogeneous_Dataset):

1. **Heterogeneous_Dataset** loads text from JSON files
2. **If `postemb` is set**: Converts text to embeddings during initialization
   - Location: `data_provider/data_loader.py` line 508-510
   - Uses: `convert_df_text_to_embeddings()` method
   - Uses: HuggingFace tokenizer/model (BERT, etc.)
3. **get_hetero_data()** returns embeddings (torch.Tensor)
4. **Model receives**: Embeddings directly

### Time-MMD Flow (Current):

1. **TimeMMD_Dataset** loads text from CSV
2. **TimeMMD_HeteroGetter** returns JSON strings (no embedding)
3. **Model receives**: JSON strings (mismatch!)

## Where Embedding ACTUALLY Happens (For Models That Work)

### ChatTime Model

**Location**: `models/ChatTime.py` → `forward()` method (line 172)

**How it works**:
```python
def forward(self, x, batch_x_hetero, batch_y_hetero, hetero_general, hetero_channel, ...):
    # ChatTime receives JSON strings and uses them directly in prompts
    context = hetero_general + hetero_channel + batch_y_hetero  # Line 172
    # context is concatenated strings, not embeddings
    
    # Text is embedded by the LLM tokenizer during generation
    pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, ...)
    samples = pipe(serialized_series)  # Line 193
    # Tokenizer embeds text here, inside the LLM pipeline
```

**Key Point**: ChatTime doesn't need pre-embedded text because it uses an LLM that tokenizes/embeds internally.

### LYNX Models

**Location**: `models/lynx_film_raw.py`, `models/lynx_film.py`, `models/lynx.py`

**How it works**:
- These models expect `news` to already be embeddings `[B, l, n, d]`
- They use `TextEmbedder` or similar components
- **Assumption**: Text is pre-embedded (from Heterogeneous_Dataset with `postemb`)

## The Missing Piece: Time-MMD → Embeddings

**Current Gap**: Time-MMD datasets return JSON strings, but models like TGTSF expect embeddings.

**Solutions**:

### Solution 1: Add Embedding to TimeMMD_HeteroGetter

**Location**: `data_provider/time_mmd_dataset.py`

**Implementation**:
```python
def __call__(self, timestamps):
    matched_times, matched_texts = self._match_timestamps(timestamps)
    
    if self.output_format == 'embedding':
        # Get embedding model from Heterogeneous_Dataset
        embeddings = self._get_embeddings(matched_texts)
        return matched_times, self.general_info, self.channel_info, embeddings
```

**Requires**: 
- Access to tokenizer/model (from Heterogeneous_Dataset or config)
- Similar to `Heterogeneous_Dataset.convert_plain_text_to_embeddings()`

### Solution 2: Add Embedding in Model's move_to_device()

**Location**: `models/TGTSF.py` → `move_to_device()` method

**Implementation**:
```python
def move_to_device(self, ..., x_hetero, y_hetero, ...):
    # Check if hetero data is JSON strings
    if isinstance(y_hetero, list) and len(y_hetero) > 0:
        if isinstance(y_hetero[0], str):  # JSON strings
            # Convert to embeddings
            y_hetero = self._embed_text_batch(y_hetero)
    
    y_hetero = y_hetero.float().to(device)
```

**Requires**: Model to have embedding capability (SentenceTransformer, etc.)

### Solution 3: Use Heterogeneous_Dataset for Embedding

**Location**: `data_provider/data_factory.py`

**Implementation**:
- Create a `Heterogeneous_Dataset` instance alongside `TimeMMD_Dataset`
- Use it to convert Time-MMD text to embeddings
- Pass embeddings to model instead of JSON strings

## Summary: Where Embedding Happens

| Model | Text Input Format | Embedding Location | Works with Time-MMD? |
|-------|------------------|-------------------|---------------------|
| **DLinear** | Not used | N/A | ✅ (ignores text) |
| **ChatTime** | JSON strings | Inside LLM tokenizer | ✅ (uses strings directly) |
| **TGTSF** | Embeddings `[B,l,n,d]` | **NOWHERE** (expects pre-embedded) | ❌ (mismatch) |
| **LYNX** | Embeddings `[B,l,n,d]` | **NOWHERE** (expects pre-embedded) | ❌ (mismatch) |

## Answer to Your Question

**Q: Where does raw text become embedded if JSON is passed by the data loader?**

**A: Currently, it doesn't!**

- **Time-MMD** returns JSON strings
- **TGTSF/LYNX** expect embeddings
- **No conversion happens** - this is a gap in the current implementation

**For models that work**:
- **ChatTime**: Embedding happens inside the LLM's tokenizer during text generation (line 193 in ChatTime.py)
- **Standard fidel-ts**: Embedding happens in `Heterogeneous_Dataset` during initialization if `postemb` is set (line 508-510 in data_loader.py)

**To make TGTSF work with Time-MMD**, you need to add embedding conversion either:
1. In `TimeMMD_HeteroGetter` when `output_format='embedding'`
2. In the model's `move_to_device()` or `forward()` method
3. By using `Heterogeneous_Dataset` to convert text to embeddings

