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

from exp.sweep_registry import SweepRegistry, get_location
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
    
    def _get_or_create_registry(self, suite_dir: Path) -> SweepRegistry:
        """
        Get or create the sweep registry for this sweep.
        
        Args:
            suite_dir: Directory for the suite/sweep
            
        Returns:
            SweepRegistry instance
        """
        if self.registry is None or self.registry.suite_dir != suite_dir:
            self.registry = SweepRegistry(
                suite_dir=str(suite_dir),
                sweep_id=self.config.sweep_id,
                location=self.config.location
            )
        return self.registry
    
    def _find_registry_for_sweep(self) -> Optional[SweepRegistry]:
        """
        Find an existing registry for this sweep.
        
        Uses deterministic path: output_dir / sweep_{sweep_id} / .sweep_registry.json
        
        Returns:
            SweepRegistry if found, None otherwise
        """
        if not self.config.output_dir.exists():
            return None
        
        # Deterministic path: output_dir / sweep_{sweep_id} / .sweep_registry.json
        sweep_dir = self.config.output_dir / f"sweep_{self.config.sweep_id}"
        registry_path = sweep_dir / SweepRegistry.REGISTRY_FILENAME
        
        if registry_path.exists():
            try:
                registry = SweepRegistry(
                    suite_dir=str(sweep_dir),
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
        
        # Also check suites directory (for legacy/migration cases)
        suites_dir = self.config.output_dir / "suites"
        if suites_dir.exists():
            for suite_subdir in suites_dir.iterdir():
                if not suite_subdir.is_dir():
                    continue
                registry_path = suite_subdir / SweepRegistry.REGISTRY_FILENAME
                if registry_path.exists():
                    try:
                        registry = SweepRegistry(
                            suite_dir=str(suite_subdir),
                            sweep_id=self.config.sweep_id,
                            location=self.config.location,
                            create_if_missing=False
                        )
                        data = registry._load()
                        if data.get('sweep_id') == self.config.sweep_id:
                            return registry
                    except Exception:
                        continue
        
        return None
    
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
            else:
                # Brief pause before retrying to prevent rapid-fire failures
                # This avoids overwhelming the system if there's a persistent issue
                time.sleep(10)
        
        print(f"[SweepManager] Run loop finished. Total completed: {runs_completed}")
        return runs_completed
    
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
            config_path = config.get('_config_path', self.config.config_path)
            
            if not config_path:
                print("[SweepManager] Error: No config path found")
                return False
            
            # Get hyperparams (exclude internal metadata)
            sweep_params = {k: v for k, v in config.items() if not k.startswith('_')}
            
            # Prepare resume IDs
            resume_ids = {
                'suite_id': suite_id or config.get('_suite_id'),
                'experiment_id': experiment_id or config.get('_experiment_id'),
                'experiment_ids': config.get('_experiment_ids', {})
            }
            
            # Ensure we have registry
            if suite_id:
                suite_dir = self.config.output_dir / suite_id
            else:
                suite_dir = self.config.output_dir / f"sweep_{self.config.sweep_id}"
            self._get_or_create_registry(suite_dir)
            
            # Mark as running
            self.registry.mark_running(wandb_run_id, run_info.get('total_epochs', 100))
            
            # Execute trial
            result = self.executor.execute_trial(
                config_path=config_path,
                sweep_params=sweep_params,
                experiment_type=config.get('_experiment_type', self.config.experiment_type),
                resume_ids=resume_ids,
                shutdown_check=lambda: self.shutdown_requested,
                wandb_run_id=wandb_run_id
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
                self.current_run_id = wandb.run.id
                
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
                
                # Store location in wandb
                wandb.run.config.update({
                    '_location': self.config.location
                }, allow_val_change=True)
                
                # Get hyperparams
                sweep_params = {k: v for k, v in config.items() if not k.startswith('_')}
                
                # Execute trial
                result = self.executor.execute_trial(
                    config_path=config_path,
                    sweep_params=sweep_params,
                    experiment_type=config.get('_experiment_type', self.config.experiment_type),
                    shutdown_check=lambda: self.shutdown_requested
                )
                
                # Add wandb info
                result.wandb_run_id = wandb.run.id
                
                # Ensure registry exists
                suite_dir = self.config.output_dir / (result.suite_id or f"sweep_{self.config.sweep_id}")
                self._get_or_create_registry(suite_dir)
                
                # Register run in registry if not already
                if not self.registry.run_exists(result.wandb_run_id):
                    self.registry.register_run(
                        wandb_run_id=result.wandb_run_id,
                        experiment_id=result.experiment_id or "unknown",
                        hyperparams=sweep_params,
                        suite_id=result.suite_id
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
            print(f"[SweepManager] wandb.agent error: {e}")
        
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
        if final_epoch == 0 and result.experiment_id:
            # Try to read from job_history
            final_epoch = self.executor._read_final_epoch_from_job_history(
                experiment_id=result.experiment_id,
                suite_id=result.suite_id,
                output_dir=self.config.output_dir
            ) or 0
        
        total_epochs = result.total_epochs
        
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
            # Training failed
            self.registry.mark_failed(
                wandb_run_id=result.wandb_run_id,
                error_message=result.error
            )
            
            print(f"[SweepManager] Run failed: {result.wandb_run_id}")
            print(f"[SweepManager]   Error: {result.error}")
        
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
