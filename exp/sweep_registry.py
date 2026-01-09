"""
Local Sweep Registry for tracking run completion status.

This module provides a file-based registry for tracking which runs in a sweep
have completed vs need resumption. The registry is stored locally (per-sweep)
to avoid race conditions with wandb server updates.

Key benefits:
- No race condition: Local file I/O is atomic with locking
- Location-aware: Runs resume on the machine where checkpoints exist
- Fast: No API calls needed to determine resumption status

Usage:
    registry = SweepRegistry(sweep_root_dir="/path/to/sweep_root", sweep_id="abc123")
    
    # Register a new run
    registry.register_run(
        wandb_run_id="xyz789",
        experiment_id="20240106_abc123",
        hyperparams={"lr": 0.001}
    )
    
    # Mark run as complete
    registry.mark_complete(
        wandb_run_id="xyz789",
        completion_reason="all_epochs",
        final_epoch=100,
        total_epochs=100
    )
    
    # Get runs needing resumption
    incomplete = registry.get_runs_needing_resume()
"""

import os
import json
import fcntl
import socket
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum


class RunStatus(str, Enum):
    """Status of a run in the registry."""
    REGISTERED = "registered"      # Run started but not yet training
    RUNNING = "running"            # Currently training
    NEEDS_RESUME = "needs_resume"  # Interrupted, needs resumption
    COMPLETED = "completed"        # Successfully completed
    FAILED = "failed"              # Failed (not resumable)


class CompletionReason(str, Enum):
    """Reason why a run completed."""
    ALL_EPOCHS = "all_epochs"           # Finished all planned epochs
    EARLY_STOPPING = "early_stopping"   # Patience-based early stopping
    HYPERBAND = "hyperband"             # Sweep controller pruned (Hyperband)
    CONVERGED = "converged"             # Reached convergence threshold
    MANUAL_STOP = "manual_stop"         # Manually stopped via UI


class InterruptReason(str, Enum):
    """Reason why a run was interrupted."""
    SLURM_TIMEOUT = "slurm_timeout"     # SLURM job time limit
    SIGTERM = "sigterm"                 # Generic SIGTERM
    SIGINT = "sigint"                   # Ctrl+C / SIGINT
    OOM = "oom"                         # Out of memory
    UNKNOWN = "unknown"                 # Unknown interruption


def get_location() -> str:
    """
    Get the location identifier for this machine.
    
    Uses SWEEP_AGENT_LOCATION environment variable if set,
    otherwise falls back to hostname.
    
    Returns:
        Location string identifying where this agent is running
    """
    return os.environ.get('SWEEP_AGENT_LOCATION', socket.gethostname())


def get_machine_id() -> str:
    """
    Get a unique machine identifier.
    
    Uses SWEEP_AGENT_MACHINE_ID environment variable if set,
    otherwise falls back to hostname.
    
    Returns:
        Machine ID string
    """
    return os.environ.get('SWEEP_AGENT_MACHINE_ID', socket.gethostname())


class SweepRegistry:
    """
    Local file-based registry for tracking sweep run status.
    
    Each sweep has its own registry file stored in the sweep root directory.
    The registry uses file locking to handle concurrent access from
    multiple SLURM jobs or processes.
    
    Attributes:
        sweep_root_dir: Path to the sweep root directory (output_dir/sweep_{sweep_id})
        sweep_id: W&B sweep ID
        location: Location identifier for this machine
        registry_path: Path to the registry JSON file
    """
    
    REGISTRY_FILENAME = ".sweep_registry.json"
    
    def __init__(
        self,
        sweep_root_dir: str,
        sweep_id: str,
        location: Optional[str] = None,
        create_if_missing: bool = True
    ):
        """
        Initialize the sweep registry.
        
        Args:
            sweep_root_dir: Path to the sweep root directory (output_dir/sweep_{sweep_id})
            sweep_id: W&B sweep ID
            location: Location identifier (default: from env or hostname)
            create_if_missing: Create registry file if it doesn't exist
        """
        self.sweep_root_dir = Path(sweep_root_dir).resolve()
        self.sweep_id = sweep_id
        self.location = location or get_location()
        self.machine_id = get_machine_id()
        
        # Registry file is in the sweep root directory
        self.registry_path = self.sweep_root_dir / self.REGISTRY_FILENAME
        self.lock_path = self.sweep_root_dir / f"{self.REGISTRY_FILENAME}.lock"
        
        # Create directory and initialize registry if needed
        if create_if_missing:
            self.sweep_root_dir.mkdir(parents=True, exist_ok=True)
            if not self.registry_path.exists():
                self._initialize_registry()
    
    def _initialize_registry(self) -> None:
        """Create an empty registry file."""
        initial_data = {
            "sweep_id": self.sweep_id,
            "created_at": datetime.now().isoformat(),
            "location": self.location,
            "runs": {}
        }
        self._save(initial_data)
    
    def _load(self) -> Dict[str, Any]:
        """
        Load registry data from file.
        
        Returns:
            Registry data dictionary
        """
        if not self.registry_path.exists():
            return {
                "sweep_id": self.sweep_id,
                "created_at": datetime.now().isoformat(),
                "location": self.location,
                "runs": {}
            }
        
        with open(self.registry_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save(self, data: Dict[str, Any]) -> None:
        """
        Save registry data to file.
        
        Args:
            data: Registry data dictionary
        """
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _with_lock(self, func):
        """
        Execute a function with exclusive file lock.
        
        This ensures atomic read-modify-write operations when
        multiple processes might access the registry.
        
        Args:
            func: Function to execute while holding the lock
            
        Returns:
            Return value of func
        """
        # Ensure lock file exists
        self.lock_path.touch(exist_ok=True)
        
        with open(self.lock_path, 'r+') as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                return func()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    
    def register_run(
        self,
        wandb_run_id: str,
        experiment_id: str,
        hyperparams: Optional[Dict[str, Any]] = None,
        suite_id: Optional[str] = None,
        config_path: Optional[str] = None,
        experiment_ids: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Register a new run in the registry.
        
        Called when a sweep trial starts, before training begins.
        
        Args:
            wandb_run_id: W&B run ID
            experiment_id: Local experiment ID
            hyperparams: Hyperparameters for this run
            suite_id: Suite ID if part of a suite
            config_path: Path to config file used for this run (for resumption)
            experiment_ids: Optional dict mapping experiment names to IDs (for suite runs)
        """
        def update():
            data = self._load()
            data["runs"][wandb_run_id] = {
                "wandb_run_id": wandb_run_id,
                "experiment_id": experiment_id,
                "suite_id": suite_id,
                "status": RunStatus.REGISTERED.value,
                "location": self.location,
                "machine_id": self.machine_id,
                "hyperparams": hyperparams or {},
                "config_path": config_path,  # Store config path for resumption
                "experiment_ids": experiment_ids or {},  # Store experiment_ids dict for suite runs
                "registered_at": datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None,
                "final_epoch": None,
                "total_epochs": None,
                "completion_reason": None,
                "interrupt_reason": None,
            }
            self._save(data)
        
        self._with_lock(update)
    
    def mark_running(
        self,
        wandb_run_id: str,
        total_epochs: int
    ) -> None:
        """
        Mark a run as currently running.
        
        Called when training actually begins.
        
        Args:
            wandb_run_id: W&B run ID
            total_epochs: Total epochs planned for training
        """
        def update():
            data = self._load()
            if wandb_run_id in data["runs"]:
                data["runs"][wandb_run_id]["status"] = RunStatus.RUNNING.value
                data["runs"][wandb_run_id]["started_at"] = datetime.now().isoformat()
                data["runs"][wandb_run_id]["total_epochs"] = total_epochs
                self._save(data)
        
        self._with_lock(update)
    
    def update_epoch(
        self,
        wandb_run_id: str,
        current_epoch: int
    ) -> None:
        """
        Update the current epoch for a running run.
        
        Called after each epoch completes.
        
        Args:
            wandb_run_id: W&B run ID
            current_epoch: Current epoch number (1-indexed)
        """
        def update():
            data = self._load()
            if wandb_run_id in data["runs"]:
                data["runs"][wandb_run_id]["final_epoch"] = current_epoch
                self._save(data)
        
        self._with_lock(update)
    
    def mark_complete(
        self,
        wandb_run_id: str,
        completion_reason: str,
        final_epoch: int,
        total_epochs: int
    ) -> None:
        """
        Mark a run as successfully completed.
        
        This method also releases any lock on the run since the run is finished
        and no longer needs to be resumed.
        
        Args:
            wandb_run_id: W&B run ID
            completion_reason: Why the run completed (see CompletionReason)
            final_epoch: Final epoch reached
            total_epochs: Total epochs that were planned
        """
        def update():
            data = self._load()
            if wandb_run_id in data["runs"]:
                run_info = data["runs"][wandb_run_id]
                
                # Release lock fields (run is complete, no longer needs locking)
                self._release_lock_fields(run_info)
                
                # Update status and completion info
                run_info.update({
                    "status": RunStatus.COMPLETED.value,
                    "completed_at": datetime.now().isoformat(),
                    "final_epoch": final_epoch,
                    "total_epochs": total_epochs,
                    "completion_reason": completion_reason,
                })
                self._save(data)
        
        self._with_lock(update)
    
    def mark_needs_resume(
        self,
        wandb_run_id: str,
        interrupt_reason: str,
        final_epoch: int,
        total_epochs: int
    ) -> None:
        """
        Mark a run as needing resumption.
        
        Called when training is interrupted (e.g., SLURM timeout).
        This method releases any existing lock so the run becomes available
        for locking by any agent when resumption is attempted.
        
        Args:
            wandb_run_id: W&B run ID
            interrupt_reason: Why the run was interrupted (see InterruptReason)
            final_epoch: Last completed epoch
            total_epochs: Total epochs planned
        """
        def update():
            data = self._load()
            if wandb_run_id in data["runs"]:
                run_info = data["runs"][wandb_run_id]
                
                # Release lock fields (run will be available for locking again)
                self._release_lock_fields(run_info)
                
                # Update status and interrupt info
                run_info.update({
                    "status": RunStatus.NEEDS_RESUME.value,
                    "final_epoch": final_epoch,
                    "total_epochs": total_epochs,
                    "interrupt_reason": interrupt_reason,
                })
                self._save(data)
        
        self._with_lock(update)
    
    def mark_failed(
        self,
        wandb_run_id: str,
        error_message: Optional[str] = None
    ) -> None:
        """
        Mark a run as failed (not resumable).
        
        This method also releases any lock on the run since failed runs
        cannot be resumed and don't need locking.
        
        Args:
            wandb_run_id: W&B run ID
            error_message: Optional error message
        """
        def update():
            data = self._load()
            if wandb_run_id in data["runs"]:
                run_info = data["runs"][wandb_run_id]
                
                # Release lock fields (run failed, no longer needs locking)
                self._release_lock_fields(run_info)
                
                # Update status and error info
                run_info.update({
                    "status": RunStatus.FAILED.value,
                    "completed_at": datetime.now().isoformat(),
                    "error_message": error_message,
                })
                self._save(data)
        
        self._with_lock(update)
    
    def get_run(self, wandb_run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific run.
        
        Args:
            wandb_run_id: W&B run ID
            
        Returns:
            Run info dictionary or None if not found
        """
        data = self._load()
        return data["runs"].get(wandb_run_id)
    
    def is_lock_stale(
        self, 
        run_info: Dict[str, Any], 
        stale_threshold_seconds: int = 172800
    ) -> bool:
        """
        Check if a lock is stale (locked too long without progress).
        
        A lock is considered stale if:
        - The run has a locked_at timestamp
        - The time since locked_at exceeds the threshold (default: 48 hours)
        
        This helps recover from cases where an agent crashed while holding a lock.
        
        Args:
            run_info: Run info dictionary
            stale_threshold_seconds: Lock age threshold in seconds (default: 48 hours = 172800)
            
        Returns:
            True if lock is stale, False otherwise
        """
        locked_at = run_info.get("locked_at")
        if not locked_at:
            # No lock timestamp means not locked
            return False
        
        try:
            locked_time = datetime.fromisoformat(locked_at)
            age_seconds = (datetime.now() - locked_time).total_seconds()
            return age_seconds > stale_threshold_seconds
        except (ValueError, TypeError):
            # Invalid timestamp format - treat as not stale (will be handled by other checks)
            return False
    
    def _release_lock_fields(self, run_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove lock-related fields from a run info dictionary.
        
        This is a helper method to clear lock fields when releasing a lock.
        Used internally by lock release operations.
        
        Args:
            run_info: Run info dictionary to modify
            
        Returns:
            Modified run info dictionary with lock fields removed
        """
        # Remove lock fields if they exist
        run_info.pop("locked_by", None)
        run_info.pop("locked_at", None)
        run_info.pop("lock_timeout", None)
        return run_info
    
    def release_lock(self, wandb_run_id: str, machine_id: Optional[str] = None) -> bool:
        """
        Release a lock on a run.
        
        This method can be used to manually release a lock if needed.
        If machine_id is provided, only releases the lock if it's held by that machine.
        If machine_id is None, releases the lock regardless of who holds it.
        
        Args:
            wandb_run_id: W&B run ID
            machine_id: Optional machine ID - if provided, only releases if locked by this machine
            
        Returns:
            True if lock was released, False otherwise
        """
        def update():
            data = self._load()
            if wandb_run_id not in data["runs"]:
                return False
            
            run_info = data["runs"][wandb_run_id]
            
            # Check if locked
            locked_by = run_info.get("locked_by")
            if not locked_by:
                # Not locked, nothing to release
                return False
            
            # If machine_id provided, only release if locked by that machine
            if machine_id is not None and locked_by != machine_id:
                # Locked by different machine, don't release
                return False
            
            # Release the lock
            self._release_lock_fields(run_info)
            self._save(data)
            return True
        
        return self._with_lock(update)
    
    def get_runs_needing_resume(self, exclude_locked: bool = True) -> List[Dict[str, Any]]:
        """
        Get all runs that need resumption on this location.
        
        Only returns runs that:
        1. Have status NEEDS_RESUME
        2. Were started on this location (checkpoints are here)
        3. Are not locked by another agent (if exclude_locked=True)
        
        Args:
            exclude_locked: If True, exclude runs that are currently locked by another agent.
                           Runs locked by this agent (same machine_id) are always included.
        
        Returns:
            List of run info dictionaries, sorted by priority
        """
        data = self._load()
        
        incomplete = []
        for run_id, run_info in data["runs"].items():
            # Check status
            if run_info.get("status") != RunStatus.NEEDS_RESUME.value:
                continue
            
            # Check location - only resume runs started on this machine
            if run_info.get("location") != self.location:
                continue
            
            # Check if locked by another agent (if exclude_locked is True)
            if exclude_locked:
                locked_by = run_info.get("locked_by")
                if locked_by and locked_by != self.machine_id:
                    # Locked by another agent, skip it
                    continue
            
            # Calculate priority based on progress
            final_epoch = run_info.get("final_epoch", 0) or 0
            total_epochs = run_info.get("total_epochs", 1) or 1
            progress = final_epoch / total_epochs if total_epochs > 0 else 0
            
            incomplete.append({
                **run_info,
                "progress": progress,
                "priority": progress * 100,  # Higher progress = higher priority
            })
        
        # Sort by priority (highest first)
        incomplete.sort(key=lambda x: -x["priority"])
        
        return incomplete
    
    def try_lock_and_get_run(
        self, 
        machine_id: str,
        stale_threshold_seconds: int = 172800
    ) -> Optional[Dict[str, Any]]:
        """
        Atomically lock and return a run that needs resumption.
        
        This method prevents race conditions where multiple agents try to grab
        the same job. It:
        1. Finds NEEDS_RESUME runs from this location (excluding already-locked runs)
        2. Checks for stale locks and releases them
        3. Atomically locks the highest-priority available run
        4. Changes status from NEEDS_RESUME to RUNNING
        5. Returns the locked run info
        
        The locking is atomic thanks to _with_lock(), which ensures only one
        agent can modify the registry at a time.
        
        Args:
            machine_id: Machine ID of this agent (from get_machine_id())
            stale_threshold_seconds: Lock age threshold for stale lock detection (default: 48 hours)
            
        Returns:
            Run info dictionary if successfully locked, None if no available runs
        """
        def update():
            data = self._load()
            
            # Get all runs needing resume, excluding ones locked by other agents
            # But we'll check stale locks manually here
            candidate_runs = []
            for run_id, run_info in data["runs"].items():
                # Check status
                if run_info.get("status") != RunStatus.NEEDS_RESUME.value:
                    continue
                
                # Check location
                if run_info.get("location") != self.location:
                    continue
                
                # Check lock status
                locked_by = run_info.get("locked_by")
                
                # If locked by another agent, check if stale
                if locked_by and locked_by != machine_id:
                    if self.is_lock_stale(run_info, stale_threshold_seconds):
                        # Stale lock - release it and make this run available
                        print(f"[SweepRegistry] Releasing stale lock on {run_id} (locked by {locked_by})")
                        self._release_lock_fields(run_info)
                        # Continue to add this run as a candidate
                    else:
                        # Locked by another agent and not stale - skip
                        continue
                
                # Calculate priority
                final_epoch = run_info.get("final_epoch", 0) or 0
                total_epochs = run_info.get("total_epochs", 1) or 1
                progress = final_epoch / total_epochs if total_epochs > 0 else 0
                
                candidate_runs.append({
                    "run_id": run_id,
                    "run_info": run_info,
                    "progress": progress,
                    "priority": progress * 100,
                })
            
            if not candidate_runs:
                return None
            
            # Sort by priority (highest first)
            candidate_runs.sort(key=lambda x: -x["priority"])
            
            # Lock the highest priority run
            target = candidate_runs[0]
            run_id = target["run_id"]
            run_info = target["run_info"]
            
            # Atomically lock the run
            run_info["locked_by"] = machine_id
            run_info["locked_at"] = datetime.now().isoformat()
            run_info["status"] = RunStatus.RUNNING.value
            
            # Save the updated registry
            self._save(data)
            
            # Return the locked run with priority info
            return {
                **run_info,
                "progress": target["progress"],
                "priority": target["priority"],
            }
        
        return self._with_lock(update)
    
    def get_all_runs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all runs in the registry.
        
        Returns:
            Dictionary mapping run IDs to run info
        """
        data = self._load()
        return data.get("runs", {})
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about runs in the registry.
        
        Returns:
            Dictionary with counts by status
        """
        data = self._load()
        stats = {
            "total": 0,
            "registered": 0,
            "running": 0,
            "needs_resume": 0,
            "completed": 0,
            "failed": 0,
        }
        
        for run_info in data["runs"].values():
            stats["total"] += 1
            status = run_info.get("status", "unknown")
            if status in stats:
                stats[status] += 1
        
        return stats
    
    def run_exists(self, wandb_run_id: str) -> bool:
        """
        Check if a run exists in the registry.
        
        Args:
            wandb_run_id: W&B run ID
            
        Returns:
            True if run exists, False otherwise
        """
        data = self._load()
        return wandb_run_id in data["runs"]
    
    def import_run(
        self,
        wandb_run_id: str,
        experiment_id: str,
        status: str = RunStatus.NEEDS_RESUME.value,
        **kwargs
    ) -> None:
        """
        Import a run from another location.
        
        Used when manually copying checkpoints between machines.
        
        Args:
            wandb_run_id: W&B run ID
            experiment_id: Local experiment ID
            status: Status to set (default: needs_resume)
            **kwargs: Additional run info fields
        """
        def update():
            data = self._load()
            data["runs"][wandb_run_id] = {
                "wandb_run_id": wandb_run_id,
                "experiment_id": experiment_id,
                "status": status,
                "location": self.location,
                "machine_id": self.machine_id,
                "imported_at": datetime.now().isoformat(),
                "imported_from": kwargs.get("imported_from", "unknown"),
                **kwargs
            }
            self._save(data)
        
        self._with_lock(update)
