# Job Locking Mechanism Plan for Concurrent Sweep Agents

## Problem Statement

When multiple local sweep agents run concurrently on the same machine/location, they can both see the same `NEEDS_RESUME` jobs in the registry and attempt to resume them simultaneously. This causes:
- Race conditions where both agents try to resume the same job
- Potential conflicts in checkpoint access
- Wasted compute resources

## Current System Overview

### Registry Status Flow
1. **REGISTERED** → Run registered but not yet training
2. **RUNNING** → Currently training (set by `mark_running()`)
3. **NEEDS_RESUME** → Interrupted, needs resumption (set by `mark_needs_resume()`)
4. **COMPLETED** → Successfully completed (set by `mark_complete()`)
5. **FAILED** → Failed, not resumable (set by `mark_failed()`)

### Current Resumption Logic
- `get_runs_needing_resume()` returns all runs with status `NEEDS_RESUME` from the same location
- No locking mechanism prevents multiple agents from picking up the same job
- Multiple agents can call this method simultaneously and get the same results

### Reconcile Registry Mechanism
- Runs automatically at startup (unless `new_only=True`)
- Finds all `RUNNING` entries for this location
- Checks `job_history.json` to determine actual state
- Updates registry: `COMPLETED` if done, `NEEDS_RESUME` if interrupted

## Proposed Solution

### 1. Job Locking in Registry

#### 1.1 Add Lock Fields to Run Info
Add the following fields to each run entry in the registry:
```python
{
    "locked_by": Optional[str],      # machine_id of agent that locked this job
    "locked_at": Optional[str],       # ISO timestamp when lock was acquired
    "lock_timeout": Optional[int],    # Lock timeout in seconds (optional, for stale lock detection)
}
```

#### 1.2 Atomic Lock Acquisition Method
Add a new method to `SweepRegistry`:
```python
def try_lock_and_get_run(self, machine_id: str) -> Optional[Dict[str, Any]]:
    """
    Atomically lock and return a run that needs resumption.
    
    This method:
    1. Finds a NEEDS_RESUME run from this location
    2. Checks if it's already locked by another agent
    3. If not locked, atomically locks it and changes status to RUNNING
    4. Returns the run info if successfully locked, None otherwise
    
    This prevents race conditions where multiple agents try to grab the same job.
    
    Args:
        machine_id: Machine ID of this agent (from get_machine_id())
        
    Returns:
        Run info dictionary if successfully locked, None if no available runs
    """
```

**Implementation Details:**
- Use `_with_lock()` to ensure atomicity
- Filter out runs that are already locked (check `locked_by` field)
- When locking, set:
  - `locked_by = machine_id`
  - `locked_at = datetime.now().isoformat()`
  - `status = RUNNING`
- Sort by priority (highest progress first) before attempting lock
- Return the first successfully locked run

#### 1.3 Update `get_runs_needing_resume()` 
Modify to exclude locked runs:
```python
def get_runs_needing_resume(self, exclude_locked: bool = True) -> List[Dict[str, Any]]:
    """
    Get all runs that need resumption on this location.
    
    Args:
        exclude_locked: If True, exclude runs that are currently locked by another agent
        
    Returns:
        List of run info dictionaries, sorted by priority
    """
```

**Changes:**
- Add filter to exclude runs where `locked_by` is set and not equal to current `machine_id`
- This allows agents to see their own locked runs but not others'

#### 1.4 Lock Release on Completion/Interruption
Update existing methods to release locks:
- `mark_complete()`: Clear `locked_by`, `locked_at`, `lock_timeout`
- `mark_needs_resume()`: Clear lock fields (job will be available for locking again)
- `mark_failed()`: Clear lock fields

### 2. SweepManager Integration

#### 2.1 Update `get_incomplete_runs()` Method
Change from:
```python
def get_incomplete_runs(self) -> List[Dict[str, Any]]:
    registry = self._find_registry_for_sweep()
    if registry is None:
        return []
    self.registry = registry
    return registry.get_runs_needing_resume()
```

To:
```python
def get_incomplete_runs(self) -> List[Dict[str, Any]]:
    registry = self._find_registry_for_sweep()
    if registry is None:
        return []
    self.registry = registry
    # Use atomic lock acquisition instead of just listing
    locked_run = registry.try_lock_and_get_run(self.config.machine_id)
    if locked_run:
        return [locked_run]  # Return as list for compatibility
    return []
```

**Alternative Approach (if we want to keep list-based):**
- Keep `get_runs_needing_resume()` but filter out locked runs
- In `run_loop()`, when picking a run to resume, use `try_lock_and_get_run()` instead
- This allows the manager to see all available runs but only lock one at a time

#### 2.2 Update `_resume_run()` Method
Ensure lock is released if resumption fails:
- If `_resume_run()` returns `False`, the lock should be released
- Add cleanup in `finally` block to release lock on exceptions

### 3. Reconcile Registry Interaction

#### 3.1 Add `--no-reconcile-registry` Flag
**CLI Changes (`scripts/local_sweep_agent.py`):**
```python
parser.add_argument(
    '--no-reconcile-registry',
    action='store_true',
    help='Skip registry reconciliation on startup. Use when running concurrent agents.'
)
```

**SweepManager Changes:**
- Add `reconcile_registry: bool = True` parameter to `run_loop()`
- Only call `_reconcile_registry()` if `reconcile_registry=True`

#### 3.2 Update `_reconcile_registry()` Logic
Modify to handle locked runs:

**Option A: Only reconcile runs locked by this agent**
```python
def _reconcile_registry(self) -> None:
    # Find all RUNNING entries for this location
    # Filter to only those locked by this machine_id
    running_runs = []
    for run_id, run_info in data["runs"].items():
        if run_info.get("status") == RunStatus.RUNNING.value:
            if run_info.get("location") == self.config.location:
                # Only reconcile runs locked by this agent
                if run_info.get("locked_by") == self.config.machine_id:
                    running_runs.append((run_id, run_info))
```

**Option B: Reconcile all RUNNING runs, but respect locks**
- If a run is locked by another agent, skip it (don't reconcile)
- If a run is locked by this agent, reconcile it
- If a run is RUNNING but not locked (legacy), reconcile it

**Recommendation: Option B** - More robust, handles legacy entries and concurrent scenarios.

#### 3.3 Stale Lock Detection (Optional Enhancement)
Add mechanism to detect and release stale locks:
- If `locked_at` is older than a threshold (e.g., 24 hours), consider lock stale
- During reconciliation, check if locking agent is still alive (optional: check process)
- If stale, release lock and mark as `NEEDS_RESUME`

**Implementation:**
```python
def _is_lock_stale(self, run_info: Dict[str, Any], stale_threshold_seconds: int = 86400) -> bool:
    """
    Check if a lock is stale (locked too long without progress).
    
    Args:
        run_info: Run info dictionary
        stale_threshold_seconds: Lock age threshold in seconds (default: 24 hours)
        
    Returns:
        True if lock is stale, False otherwise
    """
    locked_at = run_info.get("locked_at")
    if not locked_at:
        return False
    
    from datetime import datetime
    locked_time = datetime.fromisoformat(locked_at)
    age_seconds = (datetime.now() - locked_time).total_seconds()
    return age_seconds > stale_threshold_seconds
```

### 4. Machine ID and Location

#### 4.1 Current System
- `location`: Identifies the machine/location (from `SWEEP_AGENT_LOCATION` or hostname)
- `machine_id`: Already exists in registry (from `get_machine_id()`)

#### 4.2 Usage
- **Location**: Used to filter runs (only resume runs from same location where checkpoints exist)
- **Machine ID**: Used for locking (identifies which specific agent instance locked a job)

**Key Insight:** Multiple agents can run on the same location (same machine), but they have different machine IDs (or we can use process ID + hostname for uniqueness).

### 5. Workflow for Concurrent Agents

#### 5.1 Concurrent Agent Setup
```bash
# Agent 1 (with reconciliation)
python scripts/local_sweep_agent.py SWEEP_ID --project my-project --count 10

# Agent 2 (no reconciliation, concurrent)
python scripts/local_sweep_agent.py SWEEP_ID --project my-project --count 10 --no-reconcile-registry

# Agent 3 (no reconciliation, concurrent)
python scripts/local_sweep_agent.py SWEEP_ID --project my-project --count 10 --no-reconcile-registry
```

#### 5.2 Execution Flow
1. **Agent 1 (with reconciliation):**
   - Runs `_reconcile_registry()` on startup
   - Heals any stale `RUNNING` entries
   - Then proceeds to normal loop

2. **Agents 2 & 3 (no reconciliation):**
   - Skip `_reconcile_registry()` on startup
   - Go directly to checking for incomplete runs
   - Use `try_lock_and_get_run()` to atomically acquire jobs
   - If a job is already locked by another agent, skip it and try next

3. **After all agents complete:**
   - Run one final reconciliation pass to clean up any remaining issues
   - Can be done by running agent with `--resume-only --no-reconcile-registry` or manually

### 6. Edge Cases and Error Handling

#### 6.1 Agent Crash During Execution
- **Problem:** Agent locks a job, then crashes before releasing lock
- **Solution:** 
  - Stale lock detection during reconciliation
  - Lock timeout mechanism (optional)
  - Manual lock release via CLI tool (future enhancement)

#### 6.2 Lock Acquisition Failure
- **Problem:** Agent tries to lock a job but another agent already locked it
- **Solution:** 
  - `try_lock_and_get_run()` returns `None` if no available jobs
  - Agent continues to next iteration, may pick up new jobs or start new trials

#### 6.3 Registry File Lock Contention
- **Problem:** Multiple agents trying to access registry simultaneously
- **Solution:** 
  - Existing `_with_lock()` mechanism using `fcntl.flock()` handles this
  - Operations are serialized automatically

#### 6.4 Partial Lock Release
- **Problem:** Exception occurs after lock acquired but before status update
- **Solution:** 
  - Use try/finally blocks in `_resume_run()` to ensure lock release
  - Consider transaction-like semantics for lock + status update

### 7. Implementation Steps

#### Phase 1: Registry Changes
1. Add lock fields to run info structure
2. Implement `try_lock_and_get_run()` method
3. Update `get_runs_needing_resume()` to exclude locked runs
4. Update `mark_complete()`, `mark_needs_resume()`, `mark_failed()` to release locks
5. Add `_is_lock_stale()` helper method

#### Phase 2: SweepManager Changes
1. Update `get_incomplete_runs()` to use atomic lock acquisition
2. Add lock release in `_resume_run()` error handling
3. Add `reconcile_registry` parameter to `run_loop()`
4. Update `_reconcile_registry()` to handle locked runs

#### Phase 3: CLI Changes
1. Add `--no-reconcile-registry` flag to `local_sweep_agent.py`
2. Pass flag to `SweepManager.run_loop()`

#### Phase 4: Testing
1. Test concurrent agents picking up different jobs
2. Test lock acquisition failure (both agents see same job)
3. Test stale lock detection
4. Test reconciliation with locked runs
5. Test agent crash scenario (lock not released)

### 8. Backward Compatibility

#### 8.1 Legacy Runs
- Runs without lock fields should be treated as unlocked
- `try_lock_and_get_run()` should handle missing `locked_by` field gracefully

#### 8.2 Existing Behavior
- Single agent behavior should remain unchanged
- `get_runs_needing_resume()` without `exclude_locked` should work as before (for compatibility)

### 9. Future Enhancements

#### 9.1 Lock Timeout
- Add configurable lock timeout
- Automatically release locks after timeout period

#### 9.2 Lock Status Monitoring
- Add CLI command to view locked jobs: `python scripts/view_locked_jobs.py SWEEP_ID`
- Show which agent has which job locked

#### 9.3 Manual Lock Release
- Add CLI command to manually release a lock: `python scripts/release_lock.py SWEEP_ID RUN_ID`

#### 9.4 Distributed Locking
- If needed in future, consider distributed lock system (Redis, etcd) for multi-machine scenarios

## Summary

This plan adds a job locking mechanism to prevent concurrent sweep agents from resuming the same job. Key components:

1. **Atomic lock acquisition** via `try_lock_and_get_run()` prevents race conditions
2. **Lock fields** in registry track which agent has which job
3. **Reconciliation flag** (`--no-reconcile-registry`) allows concurrent agents to skip reconciliation
4. **Stale lock detection** handles agent crashes gracefully
5. **Backward compatible** with existing single-agent workflows

The solution maintains the existing architecture while adding the necessary coordination for concurrent execution.
