"""
Sweep Manager - Central orchestrator for sweep execution with local-first resumption.

This module provides the SweepManager class which is the SINGLE OWNER of:
    - Sweep registry updates (no other component updates registry)
    - Signal handling (SIGTERM, SIGINT)
    - Shutdown state
    - Wandb run management for sweeps

All completion signals flow TO this class, which makes the final decision
about how to record the run's status in the registry.

Design Principles:
    - Single ownership: Only SweepManager updates the registry
    - No globals: All state is instance attributes
    - Clear signal flow: Signals → SweepManager → Registry
    - Local-first: Registry is authoritative, wandb is for visibility

Architecture:
    LocalSweepAgent (CLI)
           │
           ▼
    SweepManager (this class)
           │
           ├── Owns: SweepRegistry, signal handlers, shutdown state
           │
           ├── Delegates: SweepExecutor for actual training
           │
           └── Updates: Registry and wandb based on SweepTrialResult

Usage:
    manager = SweepManager(
        entity="my-team",
        project="my-project",
        sweep_id="abc123",
        output_dir="./output"
    )
    
    # Run the main loop (checks for incomplete runs, then gets new hparams)
    manager.run_loop(count=5)
"""

import sys
import signal
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Add project root to path if needed
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from exp.sweep_registry import (
    SweepRegistry,
    get_location,
    RunStatus,
    CompletionReason as RegistryCompletionReason,
    InterruptReason as RegistryInterruptReason
)
from exp.sweep_executor import SweepExecutor, SweepTrialResult, CompletionReason, InterruptReason


@dataclass
class SweepConfig:
    """Configuration for the sweep manager."""
    entity: str
    project: str
    sweep_id: str
    output_dir: Path
    location: str
    config_path: Optional[str] = None
    experiment_type: str = "pytorch"


class SweepManager:
    """
    Central orchestrator for sweep execution with local-first resumption.
    
    This class is the SINGLE OWNER of:
        - Registry status updates (mark_complete, mark_needs_resume, etc.)
        - Signal handling (registers handlers, sets shutdown flag)
        - Shutdown state (no globals)
        - Wandb summary updates (for visibility, not authoritative)
    
    The manager delegates actual training execution to SweepExecutor, which
    returns a SweepTrialResult. The manager then uses this result to update
    the registry.
    
    Signal Flow:
        SIGTERM/SIGINT → _signal_handler() → sets shutdown_requested
        Training detects shutdown via callback → exits gracefully
        SweepExecutor returns SweepTrialResult with interrupted=True
        SweepManager._finalize_run() updates registry
    
    Attributes:
        config: SweepConfig with sweep parameters
        shutdown_requested: Flag set by signal handlers
        current_run_id: W&B run ID of current trial (for tracking)
        registry: SweepRegistry instance (owned by manager)
        executor: SweepExecutor instance (stateless)
        _interrupt_reason: Reason for interruption (set by signal handler)
    
    Example:
        manager = SweepManager(
            entity="my-team",
            project="my-project", 
            sweep_id="abc123",
            output_dir="./output"
        )
        manager.run_loop(count=5, resume_only=False)
    """
    
    def __init__(
        self,
        entity: str,
        project: str,
        sweep_id: str,
        output_dir: str = "./output",
        location: Optional[str] = None,
        config_path: Optional[str] = None,
        experiment_type: str = "pytorch"
    ):
        """
        Initialize the sweep manager.
        
        Args:
            entity: W&B entity/username
            project: W&B project name
            sweep_id: W&B sweep ID
            output_dir: Base output directory for experiments
            location: Location identifier (default: from env or hostname)
            config_path: Path to base config file (can also be set per-trial)
            experiment_type: Default experiment type (pytorch, lightning, etc.)
        """
        # Configuration
        self.config = SweepConfig(
            entity=entity,
            project=project,
            sweep_id=sweep_id,
            output_dir=Path(output_dir).resolve(),
            location=location or get_location(),
            config_path=config_path,
            experiment_type=experiment_type
        )
        
        # State (no globals)
        self.shutdown_requested: bool = False
        self.current_run_id: Optional[str] = None
        self._interrupt_reason: Optional[str] = None
        self._current_wandb_run = None
        
        # Components
        self.registry: Optional[SweepRegistry] = None
        self.executor: SweepExecutor = SweepExecutor()
        
        # Register signal handlers
        self._register_signal_handlers()
        
        print("[SweepManager] Initialized")
        print(f"[SweepManager] Entity: {entity}")
        print(f"[SweepManager] Project: {project}")
        print(f"[SweepManager] Sweep ID: {sweep_id}")
        print(f"[SweepManager] Location: {self.config.location}")
        print(f"[SweepManager] Output dir: {self.config.output_dir}")
    
    def _register_signal_handlers(self) -> None:
        """
        Register signal handlers for graceful shutdown.
        
        The signal handler ONLY sets the shutdown flag. It does NOT:
            - Update the registry (that's done in _finalize_run)
            - Call wandb (that's done in _finalize_run)
            - Raise exceptions
        
        This ensures clean signal handling without race conditions.
        """
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            print(f"\n[SweepManager] Received {signal_name} signal")
            
            # Set shutdown flag (single source of truth)
            self.shutdown_requested = True
            
            # Record interrupt reason
            if signum == signal.SIGTERM:
                self._interrupt_reason = InterruptReason.SLURM_TIMEOUT.value
            elif signum == signal.SIGINT:
                self._interrupt_reason = InterruptReason.SIGINT.value
            else:
                self._interrupt_reason = InterruptReason.SIGTERM.value
            
            print(f"[SweepManager] Shutdown requested (reason: {self._interrupt_reason})")
        
        # Register handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # USR1 for preemption warnings (if available)
        try:
            signal.signal(signal.SIGUSR1, signal_handler)
        except (AttributeError, ValueError):
            pass  # Not available on Windows
    
    def _get_sweep_root(self) -> Path:
        """
        Get the sweep root directory for this sweep.
        
        The sweep root directory follows a deterministic structure:
        output_dir / sweep_{sweep_id} /
            .sweep_registry.json          # Registry file
            {suite_id}/                   # Suite directories (if part of suite)
                {experiment_id}/
            {experiment_id}/              # Single experiment directories (if not part of suite)
        
        This ensures:
        - Registry is always at a predictable location
        - Experiments are organized directly under sweep root
        - Single sweep root per sweep_id eliminates ambiguity
        
        Returns:
            Path to sweep root directory
        """
        sweep_root = self.config.output_dir / f"sweep_{self.config.sweep_id}"
        sweep_root.mkdir(parents=True, exist_ok=True)
        return sweep_root
    
    def _get_or_create_registry(self) -> SweepRegistry:
        """
        Get or create the sweep registry for this sweep.
        
        The registry is always stored at:
        sweep_root / .sweep_registry.json
        
        where sweep_root = output_dir / sweep_{sweep_id}
        
        Returns:
            SweepRegistry instance
        """
        sweep_root = self._get_sweep_root()
        
        if self.registry is None or self.registry.sweep_root_dir != str(sweep_root):
            self.registry = SweepRegistry(
                sweep_root_dir=str(sweep_root),
                sweep_id=self.config.sweep_id,
                location=self.config.location
            )
        return self.registry
    
    def _find_registry_for_sweep(self) -> Optional[SweepRegistry]:
        """
        Find an existing registry for this sweep.
        
        Uses deterministic path: output_dir / sweep_{sweep_id} / .sweep_registry.json
        
        This method implements the single sweep root directory structure:
        - Sweep root: output_dir / sweep_{sweep_id}/
        - Registry: sweep_root / .sweep_registry.json
        - Experiments: sweep_root / {suite_id} / {experiment_id}/ (or sweep_root / {experiment_id}/ for single experiments)
        
        Also checks legacy locations for migration compatibility.
        
        Returns:
            SweepRegistry if found, None otherwise
        """
        if not self.config.output_dir.exists():
            return None
        
        # Primary location: output_dir / sweep_{sweep_id} / .sweep_registry.json
        sweep_root = self.config.output_dir / f"sweep_{self.config.sweep_id}"
        registry_path = sweep_root / SweepRegistry.REGISTRY_FILENAME
        
        if registry_path.exists():
            try:
                registry = SweepRegistry(
                    sweep_root_dir=str(sweep_root),
                    sweep_id=self.config.sweep_id,
                    location=self.config.location,
                    create_if_missing=False
                )
                # Verify it's the right sweep
                data = registry._load()
                if data.get('sweep_id') == self.config.sweep_id:
                    return registry
            except Exception:
                pass
        
        
        return None
    
    def _reconcile_registry(self) -> None:
        """
        Reconcile registry state by checking RUNNING entries against job_history.json.
        
        This method implements self-healing for runs that were interrupted before
        _finalize_run() could update the registry. It:
        
        1. Finds all runs with status RUNNING on this location
        2. Reads job_history.json for each run to get actual progress
        3. Determines if run completed (current_epoch >= total_epochs or completion_reason set)
        4. Updates registry: COMPLETED if done, NEEDS_RESUME if interrupted
        
        This ensures the registry accurately reflects run state even after:
        - SIGTERM that kills the process before _finalize_run()
        - Hard crashes or OOM kills
        - Network interruptions
        
        Called automatically at the start of run_loop() to ensure consistency.
        """
        registry = self._find_registry_for_sweep()
        if registry is None:
            return
        
        self.registry = registry
        data = registry._load()
        
        # Find all RUNNING entries for this location
        running_runs = []
        for run_id, run_info in data["runs"].items():
            if run_info.get("status") == RunStatus.RUNNING.value:
                if run_info.get("location") == self.config.location:
                    running_runs.append((run_id, run_info))
        
        if not running_runs:
            return
        
        print(f"[SweepManager] Reconciling {len(running_runs)} RUNNING entry(ies)...")
        
        for wandb_run_id, run_info in running_runs:
            experiment_id = run_info.get("experiment_id")
            suite_id = run_info.get("suite_id")
            
            if not experiment_id:
                print(f"[SweepManager] Warning: Run {wandb_run_id} has no experiment_id, skipping reconciliation")
                continue
            
            # Determine experiment directory path
            sweep_root = self._get_sweep_root()
            if suite_id:
                # New structure: sweep_root / suite_id / experiment_id
                exp_dir = sweep_root / suite_id / experiment_id
            else:
                # Single experiment: sweep_root / experiment_id
                exp_dir = sweep_root / experiment_id
                if not exp_dir.exists() and suite_id:
                    # Try legacy suites structure
                    exp_dir = self.config.output_dir / suite_id / experiment_id
            
            job_history_path = exp_dir / "job_history.json"
            
            if not job_history_path.exists():
                print(f"[SweepManager] Warning: job_history.json not found for {wandb_run_id} at {job_history_path}")
                # Mark as needs_resume with epoch 0 (will start fresh)
                registry.mark_needs_resume(
                    wandb_run_id=wandb_run_id,
                    interrupt_reason=InterruptReason.UNKNOWN.value,
                    final_epoch=0,
                    total_epochs=run_info.get("total_epochs", 100)
                )
                continue
            
            # Read job_history.json (authoritative source for epoch progress)
            try:
                import json
                with open(job_history_path, 'r', encoding='utf-8') as f:
                    job_history = json.load(f)
            except Exception as e:
                print(f"[SweepManager] Error reading job_history.json for {wandb_run_id}: {e}")
                continue
            
            current_epoch = job_history.get("current_epoch", 0)
            total_epochs = job_history.get("total_epochs", run_info.get("total_epochs", 100))
            
            # Check for completion reason in job_history (set by ExperimentManager)
            completion_reason = None
            if "completion_reason" in job_history:
                completion_reason = job_history["completion_reason"]
            
            # Check if training completed
            is_complete = False
            if completion_reason:
                # Explicit completion reason set (early stopping, hyperband, etc.)
                is_complete = True
            elif current_epoch >= total_epochs > 0:
                # All epochs completed
                is_complete = True
                completion_reason = CompletionReason.ALL_EPOCHS.value
            
            if is_complete:
                # Run completed successfully
                registry.mark_complete(
                    wandb_run_id=wandb_run_id,
                    completion_reason=completion_reason or RegistryCompletionReason.ALL_EPOCHS.value,
                    final_epoch=current_epoch,
                    total_epochs=total_epochs
                )
                print(f"[SweepManager] Reconciled {wandb_run_id}: COMPLETED (epoch {current_epoch}/{total_epochs}, reason: {completion_reason})")
            else:
                # Run was interrupted, needs resumption
                registry.mark_needs_resume(
                    wandb_run_id=wandb_run_id,
                    interrupt_reason=InterruptReason.UNKNOWN.value,
                    final_epoch=current_epoch,
                    total_epochs=total_epochs
                )
                print(f"[SweepManager] Reconciled {wandb_run_id}: NEEDS_RESUME (epoch {current_epoch}/{total_epochs})")
    
    def get_incomplete_runs(self) -> List[Dict[str, Any]]:
        """
        Get all incomplete runs from local registry for this location.
        
        This is the AUTHORITATIVE source for resumption decisions.
        Only returns runs that:
            1. Have status NEEDS_RESUME
            2. Were started on this location (checkpoints exist locally)
        
        Returns:
            List of run info dicts, sorted by priority (highest first)
        """
        # Find registry
        registry = self._find_registry_for_sweep()
        if registry is None:
            return []
        
        self.registry = registry
        return registry.get_runs_needing_resume()
    
    def run_loop(
        self,
        count: Optional[int] = None,
        resume_only: bool = False,
        new_only: bool = False
    ) -> int:
        """
        Main loop: check for incomplete runs, resume or start new.
        
        This is the primary entry point for running sweeps. It:
            1. Checks local registry for runs needing resumption
            2. Resumes incomplete runs (from this location)
            3. Requests new hyperparameters from wandb (if not resume_only)
            4. Executes trials and records results
        
        Args:
            count: Maximum number of runs to execute (None = unlimited)
            resume_only: If True, only resume incomplete runs
            new_only: If True, skip checking for incomplete runs
            
        Returns:
            Number of runs completed
        """
        import wandb
        
        runs_completed = 0
        api = wandb.Api()
        
        mode = 'resume_only' if resume_only else ('new_only' if new_only else 'smart (resume first)')
        print("[SweepManager] Starting run loop")
        print(f"[SweepManager] Mode: {mode}")
        print(f"[SweepManager] Max runs: {count if count else 'unlimited'}")
        
        # Reconcile registry on startup to heal any RUNNING entries
        # This handles cases where SIGTERM killed the process before _finalize_run()
        if not new_only:
            self._reconcile_registry()
        
        while True:
            # Check run limit
            if count is not None and runs_completed >= count:
                print(f"[SweepManager] Completed {runs_completed} runs. Exiting.")
                break
            
            # Check shutdown
            if self.shutdown_requested:
                print(f"[SweepManager] Shutdown requested. Exiting after {runs_completed} runs.")
                break
            
            # Step 1: Check for incomplete runs (unless new_only)
            if not new_only:
                incomplete_runs = self.get_incomplete_runs()
                
                if incomplete_runs:
                    print(f"[SweepManager] Found {len(incomplete_runs)} incomplete run(s)")
                    
                    # Resume highest priority run
                    run_info = incomplete_runs[0]
                    print(f"[SweepManager] Resuming: {run_info['wandb_run_id']}")
                    print(f"[SweepManager]   Progress: {run_info.get('final_epoch', 0)}/{run_info.get('total_epochs', '?')}")
                    
                    success = self._resume_run(run_info)
                    if success:
                        runs_completed += 1
                        # Check count limit immediately after incrementing
                        if count is not None and runs_completed >= count:
                            print(f"[SweepManager] Completed {runs_completed} runs. Exiting.")
                            break
                    else:
                        # Brief pause before trying next
                        time.sleep(5)
                    
                    continue
            
            # Step 2: Check if we should start new runs
            if resume_only:
                print("[SweepManager] No incomplete runs and resume_only=True. Exiting.")
                break
            
            # Step 3: Check sweep state
            try:
                sweep = api.sweep(f"{self.config.entity}/{self.config.project}/{self.config.sweep_id}")
                if sweep.state == 'FINISHED':
                    print("[SweepManager] Sweep is finished. Exiting.")
                    break
                elif sweep.state == 'PAUSED':
                    print("[SweepManager] Sweep is paused. Waiting...")
                    time.sleep(60)
                    continue
            except Exception as e:
                print(f"[SweepManager] Warning: Could not check sweep state: {e}")
            
            # Step 4: Run new trial
            print("[SweepManager] Starting new trial...")
            success = self._run_new_trial()
            
            if success:
                runs_completed += 1
                # Check count limit immediately after incrementing
                if count is not None and runs_completed >= count:
                    print(f"[SweepManager] Completed {runs_completed} runs. Exiting.")
                    break
            else:
                # Brief pause before retrying to prevent rapid-fire failures
                # This avoids overwhelming the system if there's a persistent issue
                time.sleep(10)
        
        print(f"[SweepManager] Run loop finished. Total completed: {runs_completed}")
        return runs_completed
    
    def _find_config_path(
        self,
        wandb_config: Dict[str, Any],
        run_info: Dict[str, Any],
        experiment_id: Optional[str] = None,
        suite_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Find config_path from multiple sources in priority order.
        
        This method checks multiple locations to find the config_path needed for resumption:
        1. Wandb run config (_config_path)
        2. Registry run info (config_path)
        3. Sweep config via API (_config_path)
        4. Experiment metadata.json (config_path or config.config_path)
        5. Suite metadata.json if part of suite (config_path)
        
        Args:
            wandb_config: Config dictionary from wandb.run.config
            run_info: Run info dictionary from registry
            experiment_id: Optional experiment ID for checking metadata
            suite_id: Optional suite ID for checking suite metadata
            
        Returns:
            Config path string if found, None otherwise
        """
        # Priority 1: Check wandb config
        config_path = wandb_config.get('_config_path', self.config.config_path)
        if config_path:
            return config_path
        
        # Priority 2: Check registry run info
        config_path = run_info.get('config_path')
        if config_path:
            return config_path
        
        # Priority 3: Check sweep config via API
        try:
            import wandb
            api = wandb.Api()
            sweep = api.sweep(f"{self.config.entity}/{self.config.project}/{self.config.sweep_id}")
            config_path = sweep.config.get('_config_path')
            if config_path:
                return config_path
        except Exception:
            pass
        
        # Priority 4: Check experiment metadata if experiment_id available
        if experiment_id:
            sweep_root = self._get_sweep_root()
            if suite_id:
                exp_dir = sweep_root / suite_id / experiment_id
            else:
                exp_dir = sweep_root / experiment_id
            
            # Check metadata.json for config_path
            metadata_path = exp_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    import json
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    config_path = metadata.get('config', {}).get('config_path') or metadata.get('config_path')
                    if config_path:
                        return config_path
                except Exception:
                    pass
            
            # Priority 5: Check suite_metadata.json if part of suite
            if suite_id:
                suite_metadata_path = sweep_root / suite_id / "suite_metadata.json"
                if suite_metadata_path.exists():
                    try:
                        import json
                        with open(suite_metadata_path, 'r', encoding='utf-8') as f:
                            suite_metadata = json.load(f)
                        config_path = suite_metadata.get('config_path')
                        if config_path:
                            return config_path
                    except Exception:
                        pass
        
        return None
    
    def _find_experiment_ids_for_resume(
        self,
        wandb_config: Dict[str, Any],
        run_info: Dict[str, Any],
        suite_config: Optional[Dict[str, Any]] = None,
        suite_id: Optional[str] = None,
        experiment_id: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Find experiment_ids mapping for suite resumption using multiple sources.
        
        This is a modular method that tries multiple sources in priority order:
        1. wandb config (_experiment_names or _experiment_ids)
        2. Registry run info (experiment_ids field)
        3. Suite metadata files (if available)
        4. Fallback: Use single experiment_id if only one experiment in suite
        
        Args:
            wandb_config: Config dictionary from wandb.run.config
            run_info: Run info dictionary from registry
            suite_config: Optional suite configuration dictionary
            suite_id: Optional suite ID for checking metadata
            experiment_id: Optional single experiment ID for fallback
        
        Returns:
            Dictionary mapping experiment names to experiment IDs
        """
        experiment_ids = {}
        
        # Priority 1: Check wandb config for _experiment_names or _experiment_ids
        # _experiment_names maps original config names -> IDs (more robust)
        if '_experiment_names' in wandb_config:
            # Use the more robust mapping (original names -> IDs)
            experiment_ids = wandb_config['_experiment_names']
        elif '_experiment_ids' in wandb_config:
            # experiment_ids maps executor names (with _output suffix) -> IDs
            # We'll use it but may need to match carefully
            exp_ids_from_wandb = wandb_config['_experiment_ids']
            if isinstance(exp_ids_from_wandb, dict):
                experiment_ids = exp_ids_from_wandb
        
        # Priority 2: Check registry run info
        if not experiment_ids and 'experiment_ids' in run_info:
            experiment_ids = run_info['experiment_ids']
        
        # Priority 3: Try reading from suite metadata (if available)
        if not experiment_ids and suite_id:
            try:
                import json
                sweep_root = self._get_sweep_root()
                suite_metadata_path = sweep_root / suite_id / "suite_metadata.json"
                if suite_metadata_path.exists():
                    with open(suite_metadata_path, 'r', encoding='utf-8') as f:
                        suite_metadata = json.load(f)
                    if 'experiment_ids' in suite_metadata:
                        experiment_ids = suite_metadata['experiment_ids']
            except Exception:
                pass
        
        # Priority 4: Fallback - if only one experiment in suite and we have experiment_id
        if not experiment_ids and experiment_id and suite_config:
            experiments = suite_config.get('suite', {}).get('experiments', [])
            enabled_experiments = [e for e in experiments if e.get('enabled', True)]
            if len(enabled_experiments) == 1:
                # Single experiment suite - use the provided experiment_id
                exp_name = enabled_experiments[0].get('name', '')
                if exp_name:
                    experiment_ids = {exp_name: experiment_id}
        
        return experiment_ids
    
    def _resume_run(self, run_info: Dict[str, Any]) -> bool:
        """
        Resume an incomplete run.
        
        Args:
            run_info: Run info from registry
            
        Returns:
            True if run completed (success or handled failure)
        """
        import wandb
        
        wandb_run_id = run_info['wandb_run_id']
        experiment_id = run_info.get('experiment_id')
        suite_id = run_info.get('suite_id')
        
        try:
            # Initialize wandb run (resume)
            self._current_wandb_run = wandb.init(
                entity=self.config.entity,
                project=self.config.project,
                id=wandb_run_id,
                resume='must',
                reinit=True
            )
            self.current_run_id = wandb_run_id
            
            # Get config from wandb
            config = dict(self._current_wandb_run.config)
            
            # Find config_path using modular method
            config_path = self._find_config_path(
                wandb_config=config,
                run_info=run_info,
                experiment_id=experiment_id,
                suite_id=suite_id
            )
            
            if not config_path:
                print("[SweepManager] Error: No config path found")
                print(f"[SweepManager]   Checked: wandb config, registry, sweep config, experiment metadata")
                return False
            
            # Get hyperparams (exclude internal metadata)
            sweep_params = {k: v for k, v in config.items() if not k.startswith('_')}
            
            # Prepare resume IDs using modular method
            # Load suite config if this is a suite run (needed for finding experiment_ids)
            suite_config = None
            if suite_id or config.get('_suite_id'):
                try:
                    from runs.suite_executor import load_suite_config
                    suite_config = load_suite_config(config_path)
                except Exception:
                    pass  # Not a suite or can't load, will use fallback
            
            resume_ids = {
                'suite_id': suite_id or config.get('_suite_id'),
                'experiment_id': experiment_id or config.get('_experiment_id'),
                'experiment_ids': self._find_experiment_ids_for_resume(
                    wandb_config=config,
                    run_info=run_info,
                    suite_config=suite_config,
                    suite_id=suite_id or config.get('_suite_id'),
                    experiment_id=experiment_id or config.get('_experiment_id')
                )
            }
            
            # Ensure we have registry (using new sweep root structure)
            self._get_or_create_registry()
            
            # Mark as running
            self.registry.mark_running(wandb_run_id, run_info.get('total_epochs', 100))
            
            # Execute trial with sweep root directory
            sweep_root = self._get_sweep_root()
            result = self.executor.execute_trial(
                config_path=config_path,
                sweep_params=sweep_params,
                experiment_type=config.get('_experiment_type', self.config.experiment_type),
                resume_ids=resume_ids,
                shutdown_check=lambda: self.shutdown_requested,
                wandb_run_id=wandb_run_id,
                sweep_root=sweep_root
            )
            
            # Add run info to result
            result.wandb_run_id = wandb_run_id
            result.experiment_id = experiment_id
            result.suite_id = suite_id
            
            # Finalize (update registry and wandb)
            self._finalize_run(result)
            
            return result.is_complete or result.should_resume
            
        except Exception as e:
            print(f"[SweepManager] Error resuming run: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.current_run_id = None
            self._current_wandb_run = None
            if wandb.run is not None:
                try:
                    wandb.finish()
                except Exception:
                    pass
    
    def _run_new_trial(self) -> bool:
        """
        Run a new sweep trial with hyperparameters from wandb.
        
        Uses wandb.agent to get new hyperparameters from the sweep controller.
        
        Returns:
            True if trial completed (success or handled failure)
        """
        import wandb
        
        success = False
        
        def run_trial():
            nonlocal success
            
            try:
                # wandb.init is called by wandb.agent
                wandb.init()
                self._current_wandb_run = wandb.run
                # Capture wandb_run_id immediately before it might become None
                # (ExperimentManager.end_experiment() finishes the run, setting wandb.run to None)
                wandb_run_id = wandb.run.id if wandb.run else None
                self.current_run_id = wandb_run_id
                
                if wandb_run_id is None:
                    print("[SweepManager] Warning: wandb.run is None after init, cannot proceed")
                    return
                
                # Get config
                config = dict(wandb.run.config)
                config_path = config.get('_config_path', self.config.config_path)
                
                if not config_path:
                    # Try to get from sweep config via API
                    try:
                        api = wandb.Api()
                        sweep = api.sweep(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.sweep_id}")
                        config_path = sweep.config.get('_config_path')
                    except Exception:
                        pass
                
                if not config_path:
                    print("[SweepManager] Error: No config path found")
                    return
                
                # Store metadata in wandb config for resumption (config_path, location, etc.)
                wandb_config_updates = {
                    '_location': self.config.location,
                    '_config_path': config_path,  # Store config path for resumption
                    '_experiment_type': config.get('_experiment_type', self.config.experiment_type)
                }
                # Only update if not already set (wandb may have locked these during sweep init)
                try:
                    wandb.run.config.update(wandb_config_updates, allow_val_change=True)
                except Exception as e:
                    # If update fails (e.g., sweep controller locked params), log warning but continue
                    print(f"[SweepManager] Warning: Could not update all wandb config values: {e}")
                
                # Get hyperparams
                sweep_params = {k: v for k, v in config.items() if not k.startswith('_')}
                
                # Execute trial with sweep root directory
                sweep_root = self._get_sweep_root()
                result = self.executor.execute_trial(
                    config_path=config_path,
                    sweep_params=sweep_params,
                    experiment_type=config.get('_experiment_type', self.config.experiment_type),
                    shutdown_check=lambda: self.shutdown_requested,
                    sweep_root=sweep_root
                )
                
                # Add wandb info (use captured value, not wandb.run.id which may be None)
                result.wandb_run_id = wandb_run_id
                
                # Ensure registry exists
                self._get_or_create_registry()
                
                # Register run in registry if not already
                if not self.registry.run_exists(result.wandb_run_id):
                    # Get experiment_ids if this is a suite run
                    experiment_ids = None
                    if result.suite_id:
                        try:
                            import wandb
                            if wandb.run and hasattr(wandb.run, 'config'):
                                experiment_ids = wandb.run.config.get('_experiment_names') or wandb.run.config.get('_experiment_ids')
                        except Exception:
                            pass
                    
                    self.registry.register_run(
                        wandb_run_id=result.wandb_run_id,
                        experiment_id=result.experiment_id or "unknown",
                        hyperparams=sweep_params,
                        suite_id=result.suite_id,
                        config_path=config_path,
                        experiment_ids=experiment_ids
                    )
                
                # Finalize
                self._finalize_run(result)
                
                success = result.is_complete or result.should_resume
                
            except Exception as e:
                print(f"[SweepManager] Error in trial: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.current_run_id = None
                self._current_wandb_run = None
        
        # Run exactly one trial via wandb.agent
        try:
            wandb.agent(
                self.config.sweep_id,
                function=run_trial,
                entity=self.config.entity,
                project=self.config.project,
                count=1
            )
        except Exception as e:
            error_msg = str(e)
            print(f"[SweepManager] wandb.agent error: {error_msg}")
            
            # Check if sweep is decommissioned/not running
            if "not running" in error_msg.lower() or "decommissioned" in error_msg.lower():
                print("[SweepManager] Sweep is no longer running. Exiting gracefully.")
                # Don't treat this as an error - sweep was intentionally stopped
                # Return True so the loop exits gracefully
                return True
        
        return success
    
    def _finalize_run(self, result: SweepTrialResult) -> None:
        """
        Finalize a run - SINGLE POINT OF TRUTH for status updates.
        
        This is the ONLY place where the registry is updated after training.
        It examines the result and decides:
            - COMPLETED: Training finished successfully
            - NEEDS_RESUME: Training was interrupted, can be resumed
            - FAILED: Training failed, should not be resumed
        
        Args:
            result: SweepTrialResult from executor
        """
        if self.registry is None:
            print("[SweepManager] Warning: No registry to update")
            return
        
        if not result.wandb_run_id:
            print("[SweepManager] Warning: No wandb_run_id in result")
            return
        
        # Get current epoch info from result
        # If final_epoch is not set (default 0), read from job_history.json
        # The ExperimentManager updates job_history after each epoch, so this
        # gives us the authoritative final epoch even if training was interrupted.
        final_epoch = result.final_epoch
        total_epochs = result.total_epochs
        
        # Read from job_history.json if needed (authoritative source)
        if result.experiment_id:
            sweep_root = self._get_sweep_root()
            
            # Determine experiment directory path
            if result.suite_id:
                exp_dir = sweep_root / result.suite_id / result.experiment_id
            else:
                exp_dir = sweep_root / result.experiment_id
            
            job_history_path = exp_dir / "job_history.json"
            
            if job_history_path.exists():
                try:
                    import json
                    with open(job_history_path, 'r', encoding='utf-8') as f:
                        job_history = json.load(f)
                    
                    print("[SweepManager]   Read job_history.json successfully")
                    print(f"[SweepManager]   job_history keys: {list(job_history.keys())}")
                    
                    # Read final_epoch from job_history if not already set
                    if final_epoch == 0 and "current_epoch" in job_history:
                        final_epoch = job_history["current_epoch"]
                        result.final_epoch = final_epoch
                        print(f"[SweepManager]   Read final_epoch from job_history: {final_epoch}")
                    elif final_epoch == 0:
                        print("[SweepManager]   WARNING: final_epoch=0 and 'current_epoch' not in job_history")
                    
                    # Read total_epochs from job_history if not already set
                    if total_epochs == 0 and "total_epochs" in job_history:
                        total_epochs = job_history["total_epochs"]
                        result.total_epochs = total_epochs
                        print(f"[SweepManager]   Read total_epochs from job_history: {total_epochs}")
                    elif total_epochs == 0:
                        print("[SweepManager]   WARNING: total_epochs=0 and 'total_epochs' not in job_history")
                    
                    # Read completion_reason from job_history if available
                    if not result.completion_reason and "completion_reason" in job_history:
                        result.completion_reason = job_history["completion_reason"]
                        print(f"[SweepManager]   Read completion_reason from job_history: {result.completion_reason}")
                except Exception as e:
                    print(f"[SweepManager]   ERROR: Could not read job_history.json: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[SweepManager]   WARNING: job_history.json does not exist at {job_history_path}")
        
        # If completion_reason is not set but training completed all epochs, set it
        # This is a fallback: if we successfully read epochs from job_history and they indicate
        # completion, we infer the completion_reason even if it wasn't explicitly set
        if not result.completion_reason and final_epoch > 0 and total_epochs > 0:
            if final_epoch >= total_epochs:
                result.completion_reason = CompletionReason.ALL_EPOCHS.value
                print(f"[SweepManager]   Inferred completion_reason='all_epochs' from epochs ({final_epoch}/{total_epochs})")
        
        # Debug: Log result state before determining status
        # EXPLANATION: SweepExecutor intentionally doesn't set final_epoch/completion_reason
        # (see comment in _execute_suite_trial line 460-464). It expects us to read from
        # job_history.json. If that reading fails or values are missing, we get:
        # - final_epoch=0, total_epochs=0, completion_reason=None
        # - is_complete=False (because 0 >= 0 > 0 is False)
        # - should_resume=False (because success=True, interrupted=False, error=None)
        # - Falls through to "failed" branch even though training succeeded
        print(f"[SweepManager] Finalizing run: {result.wandb_run_id}")
        print(f"[SweepManager]   Result from executor: success={result.success}, interrupted={result.interrupted}, error={result.error}")
        print(f"[SweepManager]   Initial values: final_epoch={result.final_epoch}, total_epochs={result.total_epochs}, completion_reason={result.completion_reason}")
        print(f"[SweepManager]   After reading job_history: final_epoch={final_epoch}, total_epochs={total_epochs}, completion_reason={result.completion_reason}")
        print(f"[SweepManager]   Status checks: is_complete={result.is_complete}, should_resume={result.should_resume}")
        if result.experiment_id:
            sweep_root = self._get_sweep_root()
            if result.suite_id:
                exp_dir = sweep_root / result.suite_id / result.experiment_id
            else:
                exp_dir = sweep_root / result.experiment_id
            job_history_path = exp_dir / "job_history.json"
            print(f"[SweepManager]   job_history.json path: {job_history_path}")
            print(f"[SweepManager]   job_history.json exists: {job_history_path.exists()}")
        
        # Determine final status
        if result.is_complete:
            # Training completed successfully
            completion_reason = result.completion_reason or CompletionReason.ALL_EPOCHS.value
            
            self.registry.mark_complete(
                wandb_run_id=result.wandb_run_id,
                completion_reason=completion_reason,
                final_epoch=final_epoch,
                total_epochs=total_epochs
            )
            
            print(f"[SweepManager] Run completed: {result.wandb_run_id}")
            print(f"[SweepManager]   Reason: {completion_reason}")
            print(f"[SweepManager]   Epochs: {final_epoch}/{total_epochs}")
            
        elif result.should_resume:
            # Training was interrupted, should resume later
            interrupt_reason = result.interrupt_reason or self._interrupt_reason or InterruptReason.UNKNOWN.value
            
            self.registry.mark_needs_resume(
                wandb_run_id=result.wandb_run_id,
                interrupt_reason=interrupt_reason,
                final_epoch=final_epoch,
                total_epochs=total_epochs
            )
            
            print(f"[SweepManager] Run needs resume: {result.wandb_run_id}")
            print(f"[SweepManager]   Reason: {interrupt_reason}")
            print(f"[SweepManager]   Progress: {final_epoch}/{total_epochs}")
            
        else:
            # Neither is_complete nor should_resume is True
            # This happens when:
            # 1. completion_reason is None/not recognized
            # 2. final_epoch < total_epochs (or total_epochs=0)
            # 3. success=True, interrupted=False, error=None (so should_resume=False)
            # 
            # If success=True but we couldn't determine completion, try to infer from epochs
            # This is a fallback for cases where job_history reading failed or values are missing
            if result.success and not result.error and final_epoch > 0 and total_epochs > 0:
                if final_epoch >= total_epochs:
                    # Actually completed all epochs, mark as complete
                    completion_reason = CompletionReason.ALL_EPOCHS.value
                    self.registry.mark_complete(
                        wandb_run_id=result.wandb_run_id,
                        completion_reason=completion_reason,
                        final_epoch=final_epoch,
                        total_epochs=total_epochs
                    )
                    print(f"[SweepManager] Run completed (inferred from epochs): {result.wandb_run_id}")
                    print(f"[SweepManager]   Reason: {completion_reason}")
                    print(f"[SweepManager]   Epochs: {final_epoch}/{total_epochs}")
                else:
                    # Partial progress, mark as needs resume
                    self.registry.mark_needs_resume(
                        wandb_run_id=result.wandb_run_id,
                        interrupt_reason=InterruptReason.UNKNOWN.value,
                        final_epoch=final_epoch,
                        total_epochs=total_epochs
                    )
                    print(f"[SweepManager] Run needs resume (inferred from epochs): {result.wandb_run_id}")
                    print(f"[SweepManager]   Progress: {final_epoch}/{total_epochs}")
            else:
                # Training failed or we can't determine status
                # This branch is reached when:
                # - success=False (exception occurred), OR
                # - success=True but final_epoch=0 or total_epochs=0 (couldn't read from job_history)
                self.registry.mark_failed(
                    wandb_run_id=result.wandb_run_id,
                    error_message=result.error or "Could not determine completion status (final_epoch=0 or total_epochs=0)"
                )
                
                print(f"[SweepManager] Run failed: {result.wandb_run_id}")
                print(f"[SweepManager]   Error: {result.error or 'Could not determine completion status'}")
                if result.success and (final_epoch == 0 or total_epochs == 0):
                    print("[SweepManager]   DIAGNOSIS: Training succeeded but couldn't read epoch info from job_history.json")
                    print("[SweepManager]   This suggests job_history.json is missing, unreadable, or doesn't contain epoch info")
        
        # Also update wandb summary (for visibility, not authoritative)
        self._update_wandb_summary(result)
        
        # Clear interrupt reason for next run
        self._interrupt_reason = None
    
    def _update_wandb_summary(self, result: SweepTrialResult) -> None:
        """
        Update wandb summary with completion status.
        
        This is for VISIBILITY only - the local registry is authoritative
        for resumption decisions.
        
        Args:
            result: SweepTrialResult from executor
        """
        import wandb
        
        if wandb.run is None:
            return
        
        try:
            summary_update = {
                '_sweep_completed': result.is_complete,
                '_final_epoch': result.final_epoch,
                '_total_epochs': result.total_epochs,
                '_location': self.config.location,
            }
            
            if result.completion_reason:
                summary_update['_completion_reason'] = result.completion_reason
            if result.interrupt_reason:
                summary_update['_interrupt_reason'] = result.interrupt_reason
            if result.error:
                summary_update['_error'] = result.error
            
            wandb.run.summary.update(summary_update)
            
        except Exception as e:
            print(f"[SweepManager] Warning: Could not update wandb summary: {e}")
