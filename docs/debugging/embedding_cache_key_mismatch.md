# Debugging Embedding Cache Key Mismatch

## Problem

You're seeing this error:
```
Cache found at data/time_mmd/Traffic/embeddings_3af7b92ef045b153 but 531/531 keys missing.
First missing: ['19800101000000', '19800201000000', '19800301000000']
```

This means:
- ✅ Cache directory exists
- ✅ Metadata matches
- ✅ Embeddings file exists
- ❌ **ALL keys are missing** (0% match)

## Possible Causes

### 1. Key Type Mismatch (Most Likely)
The cache might have been saved with **integer keys** but you're searching with **string keys** (or vice versa).

**Check:**
```python
# Run the diagnostic script
python scripts/inspect_embedding_cache.py data/time_mmd/Traffic/embeddings_3af7b92ef045b153
```

Look for:
- `Cached key types: ['int']` vs `Requested key types: ['str']` → **TYPE MISMATCH**

**Fix:**
- If cache has int keys but you need str keys: Convert when loading
- If cache has str keys but you need int keys: Convert when loading
- Regenerate cache with correct key format

### 2. Corrupted Pickle File
The pickle file might be corrupted or in an incompatible format.

**Symptoms:**
- `_pickle.UnpicklingError: invalid load key`
- File loads but returns unexpected structure

**Check:**
```python
import joblib
emb = joblib.load('data/time_mmd/Traffic/embeddings_3af7b92ef045b153/embeddings.pkl')
print(type(emb))  # Should be dict
print(len(emb))   # Should be > 0
print(list(emb.keys())[:5])  # Show first 5 keys
```

**Fix:**
- Regenerate embeddings on GPU node
- Check if file was created with different Python/joblib version

### 3. Different Timestamp Format
The cache might use a different timestamp format than what's being requested.

**Check:**
```python
# What format are the cached keys?
import joblib
emb = joblib.load('data/time_mmd/Traffic/embeddings_3af7b92ef045b153/embeddings.pkl')
sample_keys = list(emb.keys())[:10]
print("Cached key format:", sample_keys)

# What format are you requesting?
# Should be: ['19800101000000', '19800201000000', ...]
```

**Common mismatches:**
- `'19800101000000'` (string) vs `19800101000000` (int)
- `'1980-01-01 00:00:00'` (ISO format) vs `'19800101000000'` (compact)
- `19800101` (date only) vs `19800101000000` (date + time)

## Diagnostic Steps

### Step 1: Inspect the Cache File
```bash
python scripts/inspect_embedding_cache.py data/time_mmd/Traffic/embeddings_3af7b92ef045b153
```

This will show:
- Metadata configuration
- Number of keys in cache
- Sample keys and their types
- Whether expected keys exist

### Step 2: Check Key Types
```python
import joblib
from pathlib import Path

cache_dir = Path('data/time_mmd/Traffic/embeddings_3af7b92ef045b153')
emb = joblib.load(cache_dir / 'embeddings.pkl')

# Check key types
key_types = set(type(k).__name__ for k in emb.keys())
print(f"Key types in cache: {key_types}")

# Check sample keys
sample_keys = list(emb.keys())[:10]
print(f"Sample keys: {sample_keys}")

# Check if your expected keys exist (as string)
expected = ['19800101000000', '19800201000000', '19800301000000']
for exp_key in expected:
    found_str = exp_key in emb
    found_int = int(exp_key) in emb if exp_key.isdigit() else False
    print(f"'{exp_key}': {found_str} (as str), {found_int} (as int)")
```

### Step 3: Check Metadata
```python
import json
from pathlib import Path

cache_dir = Path('data/time_mmd/Traffic/embeddings_3af7b92ef045b153')
with open(cache_dir / 'metadata.json', 'r') as f:
    metadata = json.load(f)

print(json.dumps(metadata, indent=2))
```

Compare with what your current config expects:
- `model_name`: Should match
- `aggregation_method`: Should match
- `embedding_dim`: Should match
- `max_length`: Should match

## Solutions

### Solution 1: Regenerate Embeddings (Recommended)
If the cache format is incompatible, regenerate on a GPU node:

```bash
# On GPU node
python -m cli.tensor_cache generate configs/experiment_suites/tensor_cache_quick_test.yaml
```

This will:
1. Compute embeddings with correct key format
2. Save to cache with matching metadata
3. Work on CPU-only nodes afterward

### Solution 2: Fix Key Type Mismatch
If it's just a type mismatch, you could modify the loading code to convert keys:

```python
# In embedder.py _load_from_cache, after loading:
if isinstance(cached_embeddings, dict):
    # Convert keys to strings if needed
    if text_keys and isinstance(text_keys[0], str):
        # Requested keys are strings, convert cached keys
        cached_embeddings = {str(k): v for k, v in cached_embeddings.items()}
    elif text_keys and isinstance(text_keys[0], int):
        # Requested keys are ints, convert cached keys
        cached_embeddings = {int(k): v for k, v in cached_embeddings.items()}
```

**But this is a workaround - better to regenerate with correct format.**

### Solution 3: Use Old PKL Format
If you have old .pkl files that work:

```yaml
# In data config
timemmd_force_reembed: false
# And in hetero_info or dataset config
use_old_pkl: true
```

## Prevention

To avoid this in the future:

1. **Consistent Key Format**: Always use string timestamps: `str(timestamp)`
2. **Version Compatibility**: Use same Python/joblib version for saving and loading
3. **Metadata Validation**: Ensure metadata matches before using cache
4. **Test Cache Loading**: After generating cache, verify it loads correctly

## Quick Check Script

Save this as `check_cache.py`:

```python
#!/usr/bin/env python3
import joblib
import json
from pathlib import Path
import sys

cache_dir = Path(sys.argv[1])
emb_path = cache_dir / 'embeddings.pkl'
meta_path = cache_dir / 'metadata.json'

print(f"Cache: {cache_dir}")
print(f"Embeddings file exists: {emb_path.exists()}")
print(f"Metadata file exists: {meta_path.exists()}")

if meta_path.exists():
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"\nMetadata: {json.dumps(meta, indent=2)}")

if emb_path.exists():
    try:
        emb = joblib.load(emb_path)
        print(f"\nEmbeddings type: {type(emb)}")
        if isinstance(emb, dict):
            print(f"Number of keys: {len(emb)}")
            sample = list(emb.keys())[:5]
            print(f"Sample keys: {sample}")
            print(f"Key types: {set(type(k).__name__ for k in emb.keys())}")
    except Exception as e:
        print(f"\nERROR loading: {e}")
```

Run: `python check_cache.py data/time_mmd/Traffic/embeddings_3af7b92ef045b153`
