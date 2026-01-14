# Tensor Cache DataLoader Initialization

## What Happens After "Using tensor cache from:" Message

When you see the message `[ info ] Using tensor cache from: ./data/...`, the following sequence occurs:

### 1. DataLoader Creation Phase (10-30 seconds typical)

The code creates three DataLoaders sequentially:

```python
train_loader = self._get_data(flag='train')   # Can take 5-15s
vali_loader = self._get_data(flag='val')      # Can take 3-8s  
test_loader = self._get_data(flag='test')     # Can take 2-5s
```

For each DataLoader:
- **TensorCacheDataset initialization**
  - Loads `metadata.json` (fast)
  - Memory-maps `.npy` arrays using `np.load(..., mmap_mode='r')` (fast)
  - Counts samples from first array (fast)

- **PyTorch DataLoader initialization** (SLOW - this is where the stall happens)
  - Spawns `num_workers` processes (typically 4-8)
  - Each worker process:
    - Imports all Python modules (torch, numpy, etc.)
    - Loads the dataset class
    - Opens file handles to memory-mapped arrays
    - Initializes internal state
  - Sets up inter-process communication queues
  - Prefetches first `prefetch_factor * num_workers` batches

### 2. First Batch Fetch (additional 1-5 seconds)

When training starts, the first call to `next(iter(train_loader))`:
- Triggers actual disk I/O for memory-mapped arrays
- OS page cache warming
- First data transfer from workers to main process

## Why It Takes Time

### Worker Process Initialization (Main Culprit)

Each worker process needs to:
1. Fork/spawn from main process (~0.5-2s per worker)
2. Import modules (torch, numpy, custom code) (~1-3s per worker)
3. Reconstruct dataset object (~0.1-0.5s per worker)
4. Open file handles to `.npy` files (~0.1-0.5s per worker)

With `num_workers=8`, this can take **8-15 seconds** even though the work happens in parallel.

### Memory-Mapped File Initialization

- First access to memory-mapped arrays triggers OS page faults
- OS needs to read file metadata and set up virtual memory mappings
- For large files (>1GB per split), this can add 2-5 seconds

### Prefetching

- DataLoader prefetches `prefetch_factor * num_workers` batches
- Each prefetch requires array slicing and tensor conversion
- With `prefetch_factor=4` and `num_workers=8`, this means 32 batches are prepared upfront

## Optimization Strategies

### 1. Reduce Number of Workers (Fastest Improvement)

```yaml
training:
  num_workers: 2  # Instead of 8
  prefetch_factor: 2  # Instead of 4
```

**Trade-off**: Slower batch loading during training, but 3-5x faster initialization

### 2. Use Persistent Workers (PyTorch 1.7+)

```python
DataLoader(..., persistent_workers=True)
```

**Benefit**: Workers stay alive between epochs, avoiding re-initialization overhead

### 3. Disable Prefetching Initially

```yaml
training:
  prefetch_factor: None  # Disables prefetching
```

**Trade-off**: First batches arrive slower, but initialization is instant

### 4. Use Single-Process DataLoader for Small Datasets

```yaml
training:
  num_workers: 0  # Use main process only
```

**Benefit**: No worker spawn overhead (~10s saved)
**When to use**: Dataset < 10K samples, or when debugging

### 5. Warm Up File System Cache

Before running experiments:
```bash
# Linux/macOS
cat data/time_mmd/Traffic/tensor_cache/*/seq_x.npy > /dev/null

# Windows (PowerShell)
Get-Content data/time_mmd/Traffic/tensor_cache/*/seq_x.npy | Out-Null
```

**Benefit**: OS page cache pre-warmed, faster memory-mapping

## Expected Timings (Reference)

With default settings (`num_workers=4`, `prefetch_factor=4`):

| Phase | Time Range |
|-------|-----------|
| Metadata loading | < 0.1s |
| Memory-mapping arrays | 0.1-0.5s |
| Worker process spawn | 2-8s |
| Worker initialization | 3-10s |
| Prefetch batches | 1-5s |
| **Total initialization** | **10-30s** |

With optimized settings (`num_workers=2`, `prefetch_factor=2`):

| Phase | Time Range |
|-------|-----------|
| Worker process spawn | 1-4s |
| Worker initialization | 2-5s |
| Prefetch batches | 0.5-2s |
| **Total initialization** | **5-15s** |

## Monitoring DataLoader Performance

New logging messages added (as of this update):

```
[ info ] Using tensor cache from: ./data/time_mmd/Traffic/tensor_cache/a2f9f7f4
[ info ] Initializing data loaders (this may take a moment)...
[ info ] Creating train DataLoader from tensor cache (num_workers=4)...
[ info ] train DataLoader created successfully
[ info ] Creating val DataLoader from tensor cache (num_workers=4)...
[ info ] val DataLoader created successfully
[ info ] Creating test DataLoader from tensor cache (num_workers=4)...
[ info ] test DataLoader created successfully
[ info ] All data loaders initialized successfully
[ info ] Starting training: 1234 steps/epoch, epochs 1-10
```

## Implementation Details

### Code Flow

1. `runs/pytorch.py:258` → `exp.train()`
2. `exp/exp_universal.py:883` → `self._setup_training()`
3. `exp/exp_universal.py:281-283` → Creates 3 DataLoaders
4. `data_provider/data_factory.py:812` → `_get_tensor_cache_dataloader('train', ...)`
5. `data_provider/tensor_cache.py:599` → Creates `TensorCacheDataset`
6. `data_provider/tensor_cache.py:447` → Loads metadata
7. `data_provider/tensor_cache.py:450` → Memory-maps arrays
8. `data_provider/tensor_cache.py:603` → Creates PyTorch `DataLoader` ← **Stall occurs here**

### Key Files Modified

- `data_provider/data_factory.py`: Added logging around DataLoader creation
- `exp/exp_universal.py`: Added logging around training setup
- `data_provider/tensor_cache.py`: Added debug logging for initialization steps

## Debugging Slow Initialization

If initialization takes >60 seconds, check:

1. **Disk I/O bottleneck**
   ```bash
   iostat -x 1  # Linux
   ```

2. **Memory pressure**
   ```bash
   free -h  # Linux
   Get-Counter '\Memory\Available MBytes'  # Windows
   ```

3. **Network file system latency** (if data is on NFS/CIFS)
   - Copy cache to local disk for faster access

4. **Python import overhead**
   - Profile with: `python -X importtime -m cli.suite run ...`

5. **Too many workers for available CPU cores**
   - Set `num_workers <= CPU_cores - 2`

## Future Optimizations

Potential improvements (not yet implemented):

1. **Lazy worker spawn**: Only spawn workers when first batch is requested
2. **Shared memory mode**: Use `torch.multiprocessing` shared tensors instead of memory-mapped files
3. **Pre-spawn worker pool**: Keep a warm pool of workers across experiments
4. **Progressive prefetch**: Start with small prefetch, increase as training proceeds
5. **DataLoader caching**: Serialize initialized DataLoader state for instant reuse
