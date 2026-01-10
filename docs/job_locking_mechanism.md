# Job Locking Mechanism for Concurrent Sweep Agents

## Overview

This document describes the implemented job locking mechanism that enables multiple local sweep agents to run concurrently on the same machine/location without race conditions. The mechanism prevents multiple agents from attempting to resume the same `NEEDS_RESUME` job simultaneously.

## Problem Solved

Previously, when multiple local sweep agents ran concurrently on the same machine/location, they could both see the same `NEEDS_RESUME` jobs in the registry and attempt to resume them simultaneously. This caused:
- Race conditions where both agents tried to resume the same job
- Potential conflicts in checkpoint access
- Wasted compute resources

## Solution

The implemented solution adds atomic job locking to the sweep registry, ensuring that only one agent can acquire a job for resumption at a time.

## Implementation Details

### 1. Lock Fields in Registry

Each run entry in the registry now supports the following lock fields:

```python
{
    "locked_by": Optional[str],      # machine_id of agent that locked this job
    "locked_at": Optional[str],       # ISO timestamp when lock was acquired
    "lock_timeout": Optional[int],    # Lock timeout in seconds (optional, for future use)
}
```

These fields are automatically managed by the locking system and do not need to be set manually.

### 2. Atomic Lock Acquisition

The `SweepRegistry.try_lock_and_get_run()` method provides atomic lock acquisition:

- Finds `NEEDS_RESUME` runs from this location (excluding already-locked runs)
- Checks for stale locks and releases them automatically
- Atomically locks the highest-priority available run
- Changes status from `NEEDS_RESUME` to `RUNNING`
- Returns the locked run info, or `None` if no runs are available

The locking is atomic thanks to the existing `_with_lock()` mechanism using `fcntl.flock()`, which ensures only one agent can modify the registry at a time.

### 3. Stale Lock Detection

The system automatically detects and releases stale locks (locks older than the threshold, default 48 hours). This handles cases where:
- An agent crashed while holding a lock
- A process was killed before it could release the lock
- Network interruptions prevented proper lock release

Stale lock detection is performed:
- During lock acquisition (`try_lock_and_get_run()`)
- During registry reconciliation (`_reconcile_registry()`)

### 4. Lock Release

Locks are automatically released when:
- A run completes successfully (`mark_complete()`)
- A run is marked as needing resumption (`mark_needs_resume()`)
- A run fails (`mark_failed()`)
- An error occurs during resumption (lock released in exception handler)

### 5. Registry Reconciliation with Locks

The reconciliation process (Option B approach) handles locked runs intelligently:

- **Runs locked by this agent**: Reconciled (check actual state from `job_history.json`)
- **Runs locked by another agent**: Skipped (respect the lock)
- **Legacy runs (RUNNING but not locked)**: Reconciled (backward compatibility)
- **Stale locks**: Released automatically, then reconciled

## Usage

### Running Concurrent Agents

To run multiple sweep agents concurrently on the same location:

```bash
# Agent 1 (with reconciliation - recommended for the first agent)
python scripts/local_sweep_agent.py SWEEP_ID --project my-project --count 10

# Agent 2 (no reconciliation, concurrent)
python scripts/local_sweep_agent.py SWEEP_ID --project my-project --count 10 --no-reconcile-registry

# Agent 3 (no reconciliation, concurrent)
python scripts/local_sweep_agent.py SWEEP_ID --project my-project --count 10 --no-reconcile-registry
```

### Command-Line Options

#### `--no-reconcile-registry`

Skip registry reconciliation on startup. Use this flag when running concurrent agents to avoid conflicts where multiple agents try to reconcile the same registry simultaneously.

**When to use:**
- Running multiple agents concurrently on the same location
- The first agent should run with reconciliation enabled (default) to heal any stale entries
- Subsequent concurrent agents should use this flag

**When NOT to use:**
- Running a single agent (reconciliation is beneficial and safe)
- First agent in a batch (let it handle reconciliation)

#### `--lock-staleness-hours` (default: 48.0)

Hours threshold for detecting stale locks. Locks older than this threshold are considered stale and automatically released.

**Example:**
```bash
# Use 24-hour threshold instead of default 48 hours
python scripts/local_sweep_agent.py SWEEP_ID --project my-project --lock-staleness-hours 24.0
```

### Execution Flow for Concurrent Agents

1. **Agent 1 (with reconciliation):**
   - Runs `_reconcile_registry()` on startup
   - Heals any stale `RUNNING` entries
   - Releases stale locks
   - Proceeds to normal loop using atomic lock acquisition

2. **Agents 2 & 3 (no reconciliation):**
   - Skip `_reconcile_registry()` on startup (avoid conflicts)
   - Go directly to checking for incomplete runs
   - Use `try_lock_and_get_run()` to atomically acquire jobs
   - If a job is already locked by another agent, it's automatically skipped
   - Agent continues to next iteration, may pick up new jobs or start new trials

3. **After all agents complete:**
   - Run one final reconciliation pass to clean up any remaining issues
   - Can be done by running: `python scripts/local_sweep_agent.py SWEEP_ID --project my-project --resume-only`

## Machine ID and Location

### Location

The `location` field identifies the machine/location where checkpoints exist. It is derived from:
- `SWEEP_AGENT_LOCATION` environment variable (if set)
- Hostname (fallback)

**Purpose:** Only resume runs from the same location where checkpoints exist.

### Machine ID

The `machine_id` field identifies the specific agent instance. It is derived from:
- `SWEEP_AGENT_MACHINE_ID` environment variable (if set)
- Hostname (fallback)

**Purpose:** Used for locking to identify which agent instance locked a job.

**Key Insight:** Multiple agents can run on the same location (same machine), but they have different machine IDs. This allows concurrent execution while maintaining location awareness.

## API Reference

### SweepRegistry Methods

#### `try_lock_and_get_run(machine_id: str, stale_threshold_seconds: int = 172800) -> Optional[Dict[str, Any]]`

Atomically lock and return a run that needs resumption.

**Args:**
- `machine_id`: Machine ID of this agent (from `get_machine_id()`)
- `stale_threshold_seconds`: Lock age threshold in seconds (default: 48 hours)

**Returns:**
- Run info dictionary if successfully locked, `None` if no available runs

#### `get_runs_needing_resume(exclude_locked: bool = True) -> List[Dict[str, Any]]`

Get all runs that need resumption on this location.

**Args:**
- `exclude_locked`: If `True`, exclude runs locked by another agent (default: `True`)

**Returns:**
- List of run info dictionaries, sorted by priority

#### `release_lock(wandb_run_id: str, machine_id: Optional[str] = None) -> bool`

Manually release a lock on a run (usually not needed, locks are auto-released).

**Args:**
- `wandb_run_id`: W&B run ID
- `machine_id`: Optional machine ID - only releases if locked by this machine

**Returns:**
- `True` if lock was released, `False` otherwise

#### `is_lock_stale(run_info: Dict[str, Any], stale_threshold_seconds: int = 172800) -> bool`

Check if a lock is stale (locked too long without progress).

**Args:**
- `run_info`: Run info dictionary
- `stale_threshold_seconds`: Lock age threshold in seconds

**Returns:**
- `True` if lock is stale, `False` otherwise

### SweepManager Methods

#### `run_loop(count: Optional[int] = None, resume_only: bool = False, new_only: bool = False, reconcile_registry: bool = True, lock_staleness_hours: float = 48.0) -> int`

Main loop for running sweeps with job locking support.

**Args:**
- `count`: Maximum number of runs to execute (None = unlimited)
- `resume_only`: If `True`, only resume incomplete runs
- `new_only`: If `True`, skip checking for incomplete runs
- `reconcile_registry`: If `True`, reconcile registry on startup (default: `True`)
- `lock_staleness_hours`: Hours threshold for stale lock detection (default: 48.0)

**Returns:**
- Number of runs completed

## Edge Cases and Error Handling

### Agent Crash During Execution

**Problem:** Agent locks a job, then crashes before releasing lock.

**Solution:**
- Stale lock detection during reconciliation and lock acquisition
- Locks older than the threshold (default 48 hours) are automatically released
- Next agent to attempt lock acquisition will release stale lock and proceed

### Lock Acquisition Failure

**Problem:** Agent tries to lock a job but another agent already locked it.

**Solution:**
- `try_lock_and_get_run()` returns `None` if no available jobs
- Agent continues to next iteration, may pick up new jobs or start new trials
- No error is raised - this is expected behavior in concurrent scenarios

### Registry File Lock Contention

**Problem:** Multiple agents trying to access registry simultaneously.

**Solution:**
- Existing `_with_lock()` mechanism using `fcntl.flock()` handles this
- Operations are serialized automatically
- Agents wait for the lock to be released (blocking, but brief)

### Partial Lock Release

**Problem:** Exception occurs after lock acquired but before status update.

**Solution:**
- Try/finally blocks in `_resume_run()` ensure lock release on exceptions
- Lock is released in exception handler if resumption fails
- Registry operations are atomic thanks to `_with_lock()`

## Backward Compatibility

### Legacy Runs

Runs without lock fields are treated as unlocked:
- `try_lock_and_get_run()` handles missing `locked_by` field gracefully
- Legacy `RUNNING` runs are reconciled normally (marked as "legacy" during reconciliation)

### Existing Behavior

Single agent behavior remains unchanged:
- `get_runs_needing_resume()` without `exclude_locked` works as before (defaults to `True`)
- Reconciliation still runs by default (can be disabled with `--no-reconcile-registry`)
- All existing functionality continues to work

## Future Enhancements

Potential improvements for future versions:

1. **Lock Timeout**: Configurable lock timeout that automatically releases locks after a period
2. **Lock Status Monitoring**: CLI command to view locked jobs and which agent has them
3. **Manual Lock Release**: CLI command to manually release a specific lock
4. **Distributed Locking**: Consider distributed lock system (Redis, etcd) for multi-machine scenarios

## Summary

The job locking mechanism enables safe concurrent execution of multiple sweep agents on the same location. Key features:

1. **Atomic lock acquisition** prevents race conditions
2. **Stale lock detection** handles agent crashes gracefully
3. **Automatic lock release** on completion, failure, or error
4. **Backward compatible** with existing single-agent workflows
5. **Configurable staleness threshold** via `--lock-staleness-hours` flag

The solution maintains the existing architecture while adding the necessary coordination for concurrent execution.
