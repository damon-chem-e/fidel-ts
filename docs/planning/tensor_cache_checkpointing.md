# Tensor Cache Checkpointing & Resumability

**Status**: Planning (not yet implemented)  
**Created**: 2026-01-14  
**Priority**: Low (nice-to-have for robustness)

## Problem Statement

Currently, tensor cache generation has no checkpointing or resumability:

1. **No incremental flushing** - Memory-mapped arrays only flushed at the very end
2. **No progress tracking** - If killed, we don't know where we stopped
3. **No resumability** - Must restart from scratch after interruption
4. **All-or-nothing** - Partial progress is completely lost

For large datasets (e.g., 87 entities, 2.2M samples, ~2+ hours), losing progress
due to job timeout or cluster preemption is painful.

## Current Implementation

```python
# In _process_with_rich_progress() / _process_with_tqdm()

# Arrays created at start with FULL size
arrays = self._create_mmap_arrays(split_dir, array_shapes)

# Data written in chunks
for entity_id, dataset in datasets.items():
    for chunk in chunks:
        self._write_chunk(arrays, chunk_samples, ...)
    # NO flush here!

# Flush ONLY at the very end
for arr in arrays.values():
    arr.flush()

# metadata.json written only on completion
```

**Key files**: `data_provider/tensor_cache.py`

## Proposed Solution: Entity-Level Checkpointing

### Design Principles

1. **Flush after each entity** - Minimize data loss window to ~1 entity
2. **Track progress in checkpoint file** - Enable resumption from any point
3. **Atomic operations** - Use temp file + rename for crash safety
4. **Backward compatible** - Old caches without progress file still work

### New File: `progress.json` (per split)

Located at `{cache_dir}/{split}/progress.json` during generation:

```json
{
  "version": "1.0.0",
  "split": "train",
  "config_hash": "be7d47cc37efc24a",
  "started_at": "2026-01-14T12:30:00",
  "last_updated": "2026-01-14T13:45:00",
  "total_entities": 87,
  "total_samples": 2225982,
  "completed_entities": ["Bear_room", "Jena_Atmospheric_Physics", ...],
  "current_write_position": 156000,
  "status": "in_progress"
}
```

**Lifecycle**:
- Created when generation starts
- Updated atomically after each entity completes
- **Deleted** when generation completes successfully
- Presence indicates incomplete cache

### Modified Generation Flow

```python
def _generate_split(self, flag: str) -> Tuple[dict, dict]:
    progress_file = split_dir / 'progress.json'
    
    # =========================================================================
    # RESUMABILITY CHECK
    # =========================================================================
    if progress_file.exists():
        progress = load_progress(progress_file)
        
        # Validate config hash matches (detect config changes)
        if progress.get('config_hash') != self.config_hash:
            logger.warning("Config changed since last run, starting fresh")
            completed_entities = set()
            current_idx = 0
        elif progress['status'] == 'in_progress':
            logger.info(f"Resuming: {len(progress['completed_entities'])}/{progress['total_entities']} entities done")
            completed_entities = set(progress['completed_entities'])
            current_idx = progress['current_write_position']
            # Arrays already exist - open in read/write mode
            arrays = self._open_existing_arrays(split_dir, array_shapes)
        else:
            # Previous run completed but progress file remains (shouldn't happen)
            completed_entities = set()
            current_idx = 0
    else:
        # Fresh start
        completed_entities = set()
        current_idx = 0
        arrays = self._create_mmap_arrays(split_dir, array_shapes)
    
    # Initialize progress file
    save_progress(progress_file, {
        'version': '1.0.0',
        'split': flag,
        'config_hash': self.config_hash,
        'started_at': datetime.now().isoformat(),
        'total_entities': len(datasets),
        'total_samples': total_samples,
        'completed_entities': list(completed_entities),
        'current_write_position': current_idx,
        'status': 'in_progress'
    })
    
    # =========================================================================
    # PROCESS WITH CHECKPOINTING
    # =========================================================================
    for entity_id, dataset in datasets.items():
        # Skip already-completed entities
        if entity_id in completed_entities:
            logger.debug(f"Skipping completed entity: {entity_id}")
            continue
        
        entity_samples = len(dataset)
        
        # Process entity chunks...
        for chunk_start in range(0, entity_samples, self.chunk_size):
            # ... existing chunk processing logic ...
            self._write_chunk(arrays, chunk_samples, write_start, write_end)
        
        # =====================================================================
        # CHECKPOINT AFTER EACH ENTITY
        # =====================================================================
        # 1. Flush all arrays to disk
        for arr in arrays.values():
            arr.flush()
        
        # 2. Update tracking
        current_idx += entity_samples
        completed_entities.add(entity_id)
        
        # 3. Save progress atomically
        save_progress(progress_file, {
            'completed_entities': list(completed_entities),
            'current_write_position': current_idx,
            'last_updated': datetime.now().isoformat(),
            'status': 'in_progress'
        })
    
    # =========================================================================
    # COMPLETION
    # =========================================================================
    # Remove progress file to indicate success
    # (Its absence = cache is complete and valid)
    progress_file.unlink()
    
    return shapes, entity_info
```

### Helper Functions

```python
def save_progress(path: Path, data: dict):
    """
    Atomically save progress file.
    
    Uses write-to-temp + rename pattern for crash safety.
    If we crash during write, the old file remains intact.
    """
    temp_path = path.with_suffix('.progress.tmp')
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)
    temp_path.rename(path)  # Atomic on POSIX, mostly atomic on Windows


def load_progress(path: Path) -> dict:
    """Load progress file with error handling."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Corrupted progress file, starting fresh: {e}")
        return {'status': 'corrupted'}


def _open_existing_arrays(self, split_dir: Path, shapes: dict) -> Dict[str, np.memmap]:
    """
    Open existing memory-mapped arrays for resumption.
    
    Unlike _create_mmap_arrays which creates new files,
    this opens existing files in read/write mode.
    """
    arrays = {}
    for name, shape in shapes.items():
        dtype = self.ARRAY_SPECS[name]['dtype']
        filepath = split_dir / f"{name}.npy"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Expected array file not found: {filepath}")
        
        # Open existing file in read/write mode
        arrays[name] = np.lib.format.open_memmap(
            str(filepath),
            dtype=dtype,
            mode='r+',  # Read/write, file must exist
            shape=shape
        )
    
    return arrays
```

### Modified Cache Validation

```python
def validate_cache(cache_dir: Path, config: dict) -> Tuple[bool, str]:
    """Validate cache, detecting incomplete generation."""
    cache_dir = Path(cache_dir)
    
    # Check for incomplete cache (progress file exists)
    for flag in ['train', 'val', 'test']:
        progress_file = cache_dir / flag / 'progress.json'
        if progress_file.exists():
            try:
                progress = load_progress(progress_file)
                if progress.get('status') == 'in_progress':
                    completed = len(progress.get('completed_entities', []))
                    total = progress.get('total_entities', '?')
                    return False, (
                        f"Cache generation incomplete for {flag} split "
                        f"({completed}/{total} entities). "
                        f"Run 'generate' to resume or use --force to restart."
                    )
            except Exception:
                return False, f"Corrupted progress file in {flag} split"
    
    # ... existing validation logic ...
```

### CLI Integration

```python
# In generate command, before creating Data_Provider:

# Check for resumable cache
for split in split_list:
    progress_file = cache_dir / split / 'progress.json'
    if progress_file.exists() and not force:
        progress = load_progress(progress_file)
        if progress.get('status') == 'in_progress':
            completed = len(progress.get('completed_entities', []))
            total = progress.get('total_entities', '?')
            console.print(f"  [yellow]⚠ Found incomplete {split} cache ({completed}/{total} entities)[/yellow]")
            console.print(f"  [yellow]  Resuming from checkpoint...[/yellow]")
```

## Flush Frequency Analysis

| Strategy | Data Loss Window | I/O Overhead | Complexity |
|----------|------------------|--------------|------------|
| **Per entity** | ~30k samples (~30s) | Low | Simple |
| Per N chunks | ~10k samples (~10s) | Medium | Medium |
| Time-based (5 min) | Variable | Low | Medium |
| Per chunk | ~10k samples | High | Simple |

**Recommendation**: Per-entity is the sweet spot. Most entities have 20-50k samples,
so data loss window is typically <1 minute. Implementation is straightforward.

For very large entities (>100k samples), could add optional intra-entity checkpointing,
but this adds complexity and isn't necessary for current datasets.

## Edge Cases

### 1. Config Changed Between Runs

```python
if progress.get('config_hash') != self.config_hash:
    logger.warning("Config changed, cannot resume. Starting fresh.")
    # Delete partial cache and start over
    shutil.rmtree(split_dir)
    split_dir.mkdir()
```

### 2. Entity Order Changed

If entity ordering changes (e.g., different file system, dict ordering),
we need to handle this gracefully:

```python
# Track entity -> write_position mapping, not just completed list
"entity_positions": {
    "Bear_room": {"start": 0, "end": 25600},
    "Jena_Atmospheric_Physics": {"start": 25600, "end": 51200},
    ...
}
```

However, for simplicity v1, we can assume entity order is stable and just skip
completed entities. If order changes, the resumed cache may have gaps but will
still work (zeros in unwritten regions).

### 3. Corrupted Progress File

```python
try:
    progress = load_progress(progress_file)
except:
    logger.warning("Corrupted progress file, starting fresh")
    progress_file.unlink()
    # Fall through to fresh start
```

### 4. Killed During Flush

Memory-mapped flush is generally atomic at the OS level. The progress file
uses atomic write (temp + rename), so the worst case is:
- Arrays flushed but progress not updated → Entity re-processed (safe, just duplicate work)
- Progress updated but arrays not flushed → May have zeros, but next run will detect via progress

## Implementation Checklist

When implementing, modify these files:

- [ ] `data_provider/tensor_cache.py`:
  - [ ] Add `save_progress()` and `load_progress()` helper functions
  - [ ] Add `_open_existing_arrays()` method
  - [ ] Modify `_generate_split()` to check for and handle resumption
  - [ ] Add entity-level flushing and progress updates
  - [ ] Remove progress file on successful completion

- [ ] `data_provider/tensor_cache.py` - `validate_cache()`:
  - [ ] Check for progress.json to detect incomplete caches
  - [ ] Return appropriate message for resumable caches

- [ ] `cli/tensor_cache.py`:
  - [ ] Add resume detection messaging in generate command
  - [ ] Consider `--no-resume` flag to force fresh start

## Testing Plan

1. **Basic resumption**: Generate, kill mid-way, resume, verify completion
2. **Config change detection**: Change config, verify fresh start
3. **Corrupted progress file**: Corrupt file, verify graceful handling
4. **Full completion**: Verify progress file deleted on success
5. **Validation of incomplete**: Verify `validate` command detects incomplete caches

## Future Enhancements

1. **Parallel entity processing**: With per-entity tracking, could process entities in parallel
2. **Intra-entity checkpointing**: For very large entities (>500k samples)
3. **Incremental updates**: When new entities added to dataset, only generate new ones
4. **Compression**: Optionally compress checkpoint files for large entity lists
