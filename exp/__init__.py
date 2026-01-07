"""
Experiment management and tracking module.

This module provides classes for:
- ExperimentManager: Core experiment tracking and metadata management
- SweepManager: Central orchestrator for W&B hyperparameter sweeps
- SweepExecutor: Execution of individual sweep trials
- SweepRegistry: Local file-based registry for sweep run status

For sweep-related functionality, see docs/SWEEP_COMPREHENSIVE_GUIDE.md.
"""

from exp.manager import ExperimentManager
from exp.sweep_registry import (
    SweepRegistry,
    RunStatus,
    CompletionReason,
    InterruptReason,
    get_location,
    get_machine_id
)
from exp.sweep_executor import (
    SweepExecutor,
    SweepTrialResult,
    CompletionReason as ExecutorCompletionReason,
    InterruptReason as ExecutorInterruptReason
)
from exp.sweep_manager import SweepManager, SweepConfig

__all__ = [
    # Core experiment management
    'ExperimentManager',
    
    # Sweep management
    'SweepManager',
    'SweepConfig',
    'SweepExecutor',
    'SweepTrialResult',
    
    # Registry
    'SweepRegistry',
    'RunStatus',
    'CompletionReason',
    'InterruptReason',
    
    # Utilities
    'get_location',
    'get_machine_id',
]
