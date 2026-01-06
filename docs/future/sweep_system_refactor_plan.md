# Sweep System Refactor Plan

## Overview

This document outlines a comprehensive refactoring plan for the sweep resumption system. The goals are:

1. **Single ownership of status**: One component is responsible for registry/wandb updates
2. **Clear signal flow**: All signals flow to a central manager
3. **No globals**: Class-based state management
4. **No redundancy**: Clear separation of concerns
5. **Correct completion detection**: Handle early stopping, hyperband, convergence

---

## Current Architecture Problems

### Problem 1: Multiple Places Update Registry

Currently, the registry can be updated from:
- `wandb_sweep_wrapper.signal_handler()` - On SIGTERM
- `wandb_sweep_wrapper.run_suite_sweep()` - On completion/failure
- `exp/manager.end_experiment()` - On experiment end

**Conflict scenario:**
```
Training completes early stopping at epoch 50/100
    │
    ▼
end_experiment() called with sweep_completed=None
    │
    ▼
_detect_training_completed() returns False (50 < 100)
    │
    ▼
Registry marked as NEEDS_RESUME ← WRONG! Should be COMPLETED
```

### Problem 2: Global Variables

```python
# smart_sweep_agent.py
_shutdown_requested = False
_current_run = None

# wandb_sweep_wrapper.py
_shutdown_requested = False
_sweep_registry = None
```

These globals create hidden dependencies and are not thread-safe.

### Problem 3: Redundant Signal Handlers

Both `smart_sweep_agent.py` and `wandb_sweep_wrapper.py` have their own signal handlers doing similar things.

### Problem 4: Unclear Component Boundaries

- `smart_sweep_agent.py` contains execution logic that belongs in wrapper
- `wandb_sweep_wrapper.py` contains agent logic (registry management)
- Significant overlap and confusion

---

## Proposed Architecture

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LOCAL SWEEP AGENT (CLI)                              │
│                     scripts/local_sweep_agent.py                            │
│                                                                             │
│  - CLI argument parsing                                                     │
│  - Entry point for users                                                    │
│  - Creates SweepManager and calls run_loop()                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SWEEP MANAGER                                      │
│                        exp/sweep_manager.py                                  │
│                                                                             │
│  OWNS:                                                                      │
│  - SweepRegistry instance                                                   │
│  - Shutdown state (replaces globals)                                        │
│  - Signal handlers                                                          │
│  - All status updates (single point of truth)                               │
│                                                                             │
│  RESPONSIBILITIES:                                                          │
│  - Check local registry for runs needing resumption                        │
│  - Request new hyperparameters from wandb controller                       │
│  - Delegate execution to SweepExecutor                                     │
│  - Receive completion signals and update registry                          │
│  - Coordinate with wandb for visibility updates                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SWEEP EXECUTOR                                      │
│                       exp/sweep_executor.py                                  │
│                                                                             │
│  OWNS:                                                                      │
│  - Nothing stateful (pure executor)                                         │
│                                                                             │
│  RESPONSIBILITIES:                                                          │
│  - Execute a single sweep trial (suite or single experiment)               │
│  - Apply hyperparameters to config                                         │
│  - Run training via SuiteExecutor or run functions                         │
│  - Detect completion signals (early stopping, hyperband, epochs)           │
│  - Return SweepTrialResult to SweepManager                                 │
│                                                                             │
│  DOES NOT:                                                                  │
│  - Update registry                                                          │
│  - Handle signals directly                                                  │
│  - Manage state                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       EXPERIMENT MANAGER                                     │
│                         exp/manager.py                                       │
│                                                                             │
│  OWNS:                                                                      │
│  - Experiment directory and metadata                                        │
│  - Job history and checkpoint tracking                                      │
│  - Wandb run (if enabled)                                                   │
│                                                                             │
│  RESPONSIBILITIES:                                                          │
│  - Track epochs and checkpoints                                             │
│  - Provide completion reason signals via callback/return                   │
│  - Manage experiment lifecycle                                              │
│                                                                             │
│  DOES NOT (in sweep context):                                               │
│  - Update sweep registry directly                                           │
│  - Make completion decisions (reports to SweepExecutor)                    │
│                                                                             │
│  NOTE: Not every experiment is part of a sweep. Sweep-related              │
│  functionality is optional and only activated when in sweep context.       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Specifications

### SweepManager

```python
class SweepManager:
    """
    Central orchestrator for sweep execution with local-first resumption.
    
    This class is the single owner of:
    - Registry status updates
    - Signal handling
    - Shutdown state
    
    All completion signals flow TO this class, which makes the final
    decision about how to record the run's status.
    """
    
    def __init__(
        self,
        entity: str,
        project: str,
        sweep_id: str,
        output_dir: Path,
        location: str = None  # From env or hostname
    ):
        # State (no globals)
        self.shutdown_requested = False
        self.current_run_id: Optional[str] = None
        
        # Components (owned)
        self.registry: Optional[SweepRegistry] = None
        self.executor: SweepExecutor = SweepExecutor()
        
        # Signal registration
        self._register_signal_handlers()
    
    def run_loop(self, count: Optional[int] = None, resume_only: bool = False):
        """Main loop: check for incomplete, resume or start new."""
        pass
    
    def _handle_signal(self, signum, frame):
        """
        Signal handler - ONLY sets shutdown flag.
        Does NOT update registry (that's done in finalization).
        """
        self.shutdown_requested = True
        self._interrupt_reason = InterruptReason.from_signal(signum)
    
    def _finalize_run(self, result: SweepTrialResult):
        """
        SINGLE POINT OF TRUTH for status updates.
        Called after every run, whether completed, interrupted, or failed.
        """
        if result.is_complete:
            self.registry.mark_complete(
                wandb_run_id=self.current_run_id,
                completion_reason=result.completion_reason,
                final_epoch=result.final_epoch,
                total_epochs=result.total_epochs
            )
        elif result.should_resume:
            self.registry.mark_needs_resume(
                wandb_run_id=self.current_run_id,
                interrupt_reason=result.interrupt_reason,
                final_epoch=result.final_epoch,
                total_epochs=result.total_epochs
            )
        else:
            self.registry.mark_failed(
                wandb_run_id=self.current_run_id,
                error_message=result.error
            )
        
        # Also update wandb for visibility (not authoritative)
        self._update_wandb_summary(result)
```

### SweepExecutor

```python
class SweepExecutor:
    """
    Executes a single sweep trial.
    
    This is a pure executor - it runs training and reports results.
    It does NOT:
    - Manage registry
    - Handle signals
    - Make decisions about resumption
    """
    
    def execute_trial(
        self,
        config_path: str,
        sweep_params: Dict[str, Any],
        experiment_type: str = "pytorch",
        shutdown_check: Callable[[], bool] = None  # Callback to check shutdown
    ) -> SweepTrialResult:
        """
        Execute a single sweep trial.
        
        Args:
            config_path: Path to config file
            sweep_params: Hyperparameters from sweep controller
            experiment_type: Type of experiment
            shutdown_check: Callback that returns True if shutdown requested
            
        Returns:
            SweepTrialResult with completion status and metadata
        """
        try:
            # Run training
            # (Pass shutdown_check to training loop for graceful exit)
            
            # Detect completion reason
            completion_reason = self._detect_completion_reason()
            
            return SweepTrialResult(
                success=True,
                interrupted=False,
                final_epoch=...,
                total_epochs=...,
                completion_reason=completion_reason,
                error=None
            )
        except Exception as e:
            return SweepTrialResult(
                success=False,
                interrupted=shutdown_check() if shutdown_check else False,
                final_epoch=...,
                total_epochs=...,
                completion_reason=None,
                error=str(e)
            )
    
    def _detect_completion_reason(self) -> str:
        """
        Detect why training completed.
        
        Checks (in order):
        1. Explicit completion_reason set by training code
        2. wandb.run.stopped (Hyperband pruning)
        3. current_epoch >= total_epochs (all epochs)
        """
        pass
```

### SweepTrialResult

```python
@dataclass
class SweepTrialResult:
    """
    Result from executing a sweep trial.
    
    This is the communication contract between SweepExecutor and SweepManager.
    The executor populates this, the manager uses it to update registry.
    """
    success: bool                          # Did training run without exception?
    interrupted: bool                       # Was interrupted by signal?
    final_epoch: int                        # Last completed epoch
    total_epochs: int                       # Total epochs planned
    completion_reason: Optional[str]        # all_epochs, early_stopping, hyperband, converged
    interrupt_reason: Optional[str]         # slurm_timeout, sigint, etc.
    error: Optional[str]                    # Error message if failed
    experiment_id: Optional[str]            # For registry tracking
    suite_id: Optional[str]                 # For registry tracking
    
    @property
    def is_complete(self) -> bool:
        """
        Is this run complete (should NOT be resumed)?
        
        A run is complete if:
        1. It has a valid completion reason (early_stopping, hyperband, etc.)
        2. OR it finished all epochs
        """
        if self.completion_reason in [
            'all_epochs', 'early_stopping', 'hyperband', 'converged', 'manual_stop'
        ]:
            return True
        return False
    
    @property
    def should_resume(self) -> bool:
        """
        Should this run be resumed later?
        
        A run should be resumed if:
        1. It was interrupted (not failed)
        2. It made progress (final_epoch > 0)
        3. It's not already complete
        """
        if self.is_complete:
            return False
        if self.interrupted and self.final_epoch > 0:
            return True
        return False
```

---

## Signal Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SIGNAL SOURCES                                     │
└─────────────────────────────────────────────────────────────────────────────┘
        │                    │                    │                    │
        ▼                    ▼                    ▼                    ▼
   ┌─────────┐          ┌─────────┐          ┌─────────┐          ┌─────────┐
   │ SIGTERM │          │ Training│          │ Hyperband│         │Exception│
   │ SIGINT  │          │ Callback│          │ Pruning  │          │  Error  │
   └────┬────┘          └────┬────┘          └────┬────┘          └────┬────┘
        │                    │                    │                    │
        │ Sets flag          │ Sets reason        │ Sets reason        │ Returns
        │                    │                    │                    │ error
        ▼                    ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SWEEP MANAGER                                       │
│                                                                             │
│  shutdown_requested ─────────┐                                              │
│                              │                                              │
│  After training exits:       ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              _finalize_run(SweepTrialResult)                        │   │
│  │                                                                      │   │
│  │  1. Examine result.completion_reason                                │   │
│  │  2. Check result.interrupted                                         │   │
│  │  3. Decide: COMPLETED vs NEEDS_RESUME vs FAILED                     │   │
│  │  4. Update LOCAL REGISTRY (authoritative)                           │   │
│  │  5. Update wandb summary (for visibility)                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  SINGLE POINT OF TRUTH: Only _finalize_run() updates registry             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Completion Reason Detection

### Signal Sources and Detection Points

| Signal | Source | Detection Point | Detection Method |
|--------|--------|-----------------|------------------|
| All epochs | Training loop | SweepExecutor | `current_epoch >= total_epochs` |
| Early stopping | Training callback | ExperimentManager | `exp_manager.set_completion_reason("early_stopping")` |
| Hyperband | Sweep controller | SweepExecutor | `wandb.run.stopped == True` |
| Convergence | Training code | ExperimentManager | `exp_manager.set_completion_reason("converged")` |
| SIGTERM | OS | SweepManager | Signal handler sets flag |
| Exception | Training code | SweepExecutor | try/except catches |

### Detection Flow in SweepExecutor

```python
def _detect_completion_reason(self) -> Optional[str]:
    """
    Detect completion reason in priority order.
    
    Priority:
    1. Explicit reason set by training code (most authoritative)
    2. Hyperband pruning (sweep controller decision)
    3. All epochs completed (default)
    """
    
    # 1. Check explicit reason from ExperimentManager
    if self.exp_manager and self.exp_manager._completion_reason:
        return self.exp_manager._completion_reason
    
    # 2. Check Hyperband pruning
    if wandb.run and getattr(wandb.run, 'stopped', False):
        return 'hyperband'
    
    # 3. Check epochs
    current = self.exp_manager.job_history.get('current_epoch', 0)
    total = self.exp_manager.job_history.get('total_epochs', 0)
    if current >= total > 0:
        return 'all_epochs'
    
    # No completion reason found (training didn't complete)
    return None
```

---

## File Structure After Refactor

```
exp/
├── manager.py              # ExperimentManager (unchanged, but sweep registry calls removed)
├── sweep_registry.py       # SweepRegistry (unchanged)
├── sweep_manager.py        # NEW: SweepManager class
└── sweep_executor.py       # Refactored from wandb_sweep_wrapper.py

scripts/
├── local_sweep_agent.py    # Refactored from smart_sweep_agent.py (CLI only)
├── submit_sweep_jobs.py    # SLURM submission helper (unchanged)
└── slurm_templates/
    └── smart_sweep_job.sh  # Rename to local_sweep_job.sh
```

---

## Changes to ExperimentManager

### What Changes

1. **Remove direct registry updates** - No longer calls `registry.mark_complete()` etc.
2. **Keep completion reason tracking** - `set_completion_reason()` stays
3. **Return completion info** - `end_experiment()` returns info instead of updating registry

### What Stays the Same

1. **Job history tracking** - Still manages job_history.json
2. **Epoch tracking** - Still calls `update_current_epoch()`
3. **Checkpoint management** - Still handles checkpoints
4. **Wandb integration** - Still manages wandb run (for non-sweep cases)

### New Docstring

```python
class ExperimentManager:
    """
    Manages experiment tracking, metadata capture, and output organization.
    
    Note on Sweep Integration:
        Not every experiment runs as part of a sweep. When running in sweep
        context, the ExperimentManager provides completion signals to the
        SweepExecutor via:
        - set_completion_reason(): For early stopping, convergence, etc.
        - _completion_reason attribute: Read by executor
        
        The ExperimentManager does NOT directly update the sweep registry.
        That is the responsibility of the SweepManager, which is the single
        point of truth for sweep status.
    """
```

---

## Changes to Current Files

### smart_sweep_agent.py → local_sweep_agent.py

**Keep:**
- CLI argument parsing
- Entry point

**Remove:**
- Signal handlers (move to SweepManager)
- `get_incomplete_runs()` (dead code, was W&B API based)
- `is_run_completed()` (dead code)
- `resume_incomplete_run()` (move to SweepManager)
- Global variables

**Add:**
- Instantiate SweepManager
- Call `manager.run_loop()`

### wandb_sweep_wrapper.py → sweep_executor.py

**Keep:**
- `apply_sweep_parameters_to_suite()`
- `apply_sweep_parameters_to_experiment()`
- `is_suite_config()`
- Core execution logic

**Remove:**
- Signal handlers (move to SweepManager)
- Registry management (move to SweepManager)
- Global variables
- `main()` entry point (not needed, called programmatically)

**Add:**
- `SweepExecutor` class
- `execute_trial()` method that returns `SweepTrialResult`
- `_detect_completion_reason()` method

---

## Implementation Order

### Phase 1: Create Core Classes (No Breaking Changes)

1. Create `exp/sweep_manager.py` with `SweepManager` class
2. Create `SweepTrialResult` dataclass
3. Create `exp/sweep_executor.py` with `SweepExecutor` class (wrapper around existing functions)

### Phase 2: Migrate Signal Handling

1. Move signal handlers from wrapper to SweepManager
2. Remove globals, use instance attributes
3. Implement `_finalize_run()` as single point of truth

### Phase 3: Migrate CLI

1. Rename `smart_sweep_agent.py` to `local_sweep_agent.py`
2. Simplify to just CLI parsing + SweepManager instantiation
3. Remove dead code (`get_incomplete_runs`, etc.)

### Phase 4: Clean Up Old Files

1. Rename `wandb_sweep_wrapper.py` to `sweep_executor.py`
2. Remove redundant code
3. Update imports throughout codebase

### Phase 5: Update ExperimentManager

1. Remove direct registry update calls
2. Add docstring noting sweep context is optional
3. Ensure completion reason is properly exposed

---

## Sleep Logic Fix

Current code:
```python
if success:
    runs_completed += 1
else:
    # Check if sweep might be exhausted
    time.sleep(10)
```

Fixed comment:
```python
if success:
    runs_completed += 1
else:
    # Brief pause before retrying after failure.
    # This prevents rapid-fire retries that could:
    # 1. Overwhelm the system if there's a persistent issue
    # 2. Spam logs with repeated failure messages
    # 3. Interfere with cleanup processes after a crash
    time.sleep(10)
```

---

## find_local_registries Explanation and Fix

### Current Behavior

```python
def find_local_registries(output_dir: Path, sweep_id: str) -> List[SweepRegistry]:
    for registry_file in output_dir.rglob(SweepRegistry.REGISTRY_FILENAME):
        # ...
```

This recursively searches ALL subdirectories for `.sweep_registry.json` files.

### Why It's Confusing

1. **Non-deterministic**: If multiple suites exist, order depends on filesystem
2. **Expensive**: Scans entire output directory tree
3. **Unclear**: Why do we need to search? Shouldn't we know where it is?

### Better Approach

The SweepManager should track which registry it created/uses:

```python
class SweepManager:
    def __init__(self, ...):
        # Registry path is deterministic based on sweep config
        self.registry_dir = self._determine_registry_dir()
        self.registry = SweepRegistry(
            suite_dir=str(self.registry_dir),
            sweep_id=self.sweep_id,
            location=self.location
        )
    
    def _determine_registry_dir(self) -> Path:
        """
        Determine registry directory from sweep configuration.
        
        For suites: output_dir / suite_name
        For single experiments: output_dir / sweep_{sweep_id}
        """
        # Get from sweep config or use default
        pass
```

The recursive search becomes a **fallback for recovery** only:

```python
def _recover_registry(self) -> Optional[SweepRegistry]:
    """
    Fallback: Search for existing registry if normal path fails.
    
    This is used when resuming on a new machine or after config changes.
    Should not be the normal path.
    """
    for registry_file in self.output_dir.rglob(SweepRegistry.REGISTRY_FILENAME):
        # ...
```

---

## Testing Plan

### Unit Tests

1. **SweepTrialResult**
   - Test `is_complete` for all completion reasons
   - Test `should_resume` for various scenarios

2. **SweepManager**
   - Test signal handler sets flag correctly
   - Test `_finalize_run` updates registry correctly for each scenario
   - Test early stopping → COMPLETED
   - Test SIGTERM → NEEDS_RESUME
   - Test exception with progress → NEEDS_RESUME
   - Test exception without progress → FAILED

3. **SweepExecutor**
   - Test `_detect_completion_reason` priority order
   - Test hyperband detection
   - Test all epochs detection

### Integration Tests

1. Run full sweep trial, verify registry state
2. Simulate SIGTERM mid-training, verify resumption works
3. Test early stopping marks as complete
4. Test multi-location scenario (registries stay separate)

---

## Migration Checklist

- [ ] Create `exp/sweep_manager.py`
- [ ] Create `SweepTrialResult` in `exp/sweep_executor.py`
- [ ] Create `SweepExecutor` class in `exp/sweep_executor.py`
- [ ] Move signal handling to SweepManager
- [ ] Implement `_finalize_run()` as single point of truth
- [ ] Create `scripts/local_sweep_agent.py` (simplified CLI)
- [ ] Remove dead code from old files
- [ ] Update ExperimentManager docstrings
- [ ] Remove ExperimentManager direct registry updates
- [ ] Update SLURM templates
- [ ] Update submit_sweep_jobs.py
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Update user documentation

---

## Summary

### Key Architectural Decisions

1. **Single Owner**: SweepManager is the only component that updates registry
2. **Signal Flow**: All signals flow TO SweepManager, not processed in-place
3. **No Globals**: All state is instance attributes of SweepManager
4. **Clear Boundaries**:
   - LocalSweepAgent: CLI entry point only
   - SweepManager: Orchestration and state ownership
   - SweepExecutor: Pure execution, returns results
   - ExperimentManager: Experiment lifecycle (sweep-agnostic)

### Naming Convention

| Old Name | New Name | Role |
|----------|----------|------|
| `smart_sweep_agent.py` | `local_sweep_agent.py` | CLI entry point |
| `wandb_sweep_wrapper.py` | `sweep_executor.py` | Single trial execution |
| (new) | `sweep_manager.py` | Orchestrator, state owner |

### Completion Detection Fix

```
Before: Only checked epochs → early stopping marked as incomplete
After:  Check completion_reason first → early stopping correctly marked complete
```

This refactor eliminates the bugs, removes redundancy, and creates a clean architecture with clear ownership.
