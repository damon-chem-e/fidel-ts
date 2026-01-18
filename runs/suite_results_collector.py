"""
Collect and aggregate results from all experiments in a suite.

This module provides functionality to gather metrics and status information
from completed and in-progress experiments within a suite.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json


class ExperimentResult:
    """Container for a single experiment's results."""

    def __init__(
        self,
        name: str,
        experiment_id: str,
        status: str,
        epochs_completed: int,
        total_epochs: int,
        completion_reason: Optional[str],
        train_mse: Optional[float],
        val_mse: Optional[float],
        test_mse: Optional[float],
        experiment_dir: Optional[Path]
    ):
        """
        Initialize experiment result.

        Args:
            name: Experiment name from suite config
            experiment_id: Generated experiment ID
            status: Status string (completed/failed/skipped/not_run/partial)
            epochs_completed: Number of epochs completed
            total_epochs: Total epochs configured
            completion_reason: Reason for completion (all_epochs/early_stopping/etc)
            train_mse: Training MSE (normalized)
            val_mse: Validation MSE (normalized)
            test_mse: Test MSE (normalized)
            experiment_dir: Path to experiment directory
        """
        self.name = name
        self.experiment_id = experiment_id
        self.status = status
        self.epochs_completed = epochs_completed
        self.total_epochs = total_epochs
        self.completion_reason = completion_reason
        self.train_mse = train_mse
        self.val_mse = val_mse
        self.test_mse = test_mse
        self.experiment_dir = experiment_dir


class SuiteResultsCollector:
    """Collect results from all experiments in a suite."""

    def __init__(
        self,
        suite_dir: Path,
        suite_config: Dict[str, Any],
        experiment_id_registry: Dict[str, str]
    ):
        """
        Initialize results collector.

        Args:
            suite_dir: Path to suite directory
            suite_config: Suite configuration dictionary
            experiment_id_registry: Dictionary mapping experiment names to experiment IDs
                                   (populated by SuiteExecutor as experiments are created)
        """
        self.suite_dir = suite_dir
        self.suite_config = suite_config
        self.experiment_id_registry = experiment_id_registry
        self.results: List[ExperimentResult] = []

    def collect_results(self) -> List[ExperimentResult]:
        """
        Collect results from all experiments defined in suite.

        Returns:
            List of ExperimentResult objects (one per experiment)
        """
        experiments = self.suite_config.get('suite', {}).get('experiments', [])

        for exp_config in experiments:
            if not exp_config.get('enabled', True):
                # Skip disabled experiments (don't include in results)
                continue

            exp_name = exp_config.get('name', 'unknown')
            result = self._collect_experiment_result(exp_name, exp_config)
            self.results.append(result)

        return self.results

    def _collect_experiment_result(
        self,
        exp_name: str,
        exp_config: Dict[str, Any]
    ) -> ExperimentResult:
        """
        Collect results for a single experiment.

        Handles multiple cases:
        - Experiment completed successfully
        - Experiment failed during execution
        - Experiment skipped (already complete)
        - Experiment not run (suite stopped before reaching it)

        Args:
            exp_name: Experiment name from suite config
            exp_config: Experiment configuration dictionary

        Returns:
            ExperimentResult object
        """
        # Get experiment ID from registry (registered by ExperimentManager)
        experiment_id = self.experiment_id_registry.get(exp_name)

        if experiment_id is None:
            # Experiment was never started (suite stopped before it)
            return ExperimentResult(
                name=exp_name,
                experiment_id="N/A",
                status="not_run",
                epochs_completed=0,
                total_epochs=0,
                completion_reason=None,
                train_mse=None,
                val_mse=None,
                test_mse=None,
                experiment_dir=None
            )

        experiment_dir = self.suite_dir / experiment_id

        # Load job history
        job_history = self._load_job_history(experiment_dir)

        # Load metrics
        metrics = self._load_metrics(experiment_dir)

        # Determine status
        status = self._determine_status(job_history, metrics)

        # Extract metrics
        train_mse = self._extract_train_mse(job_history, metrics)
        val_mse = self._extract_val_mse(job_history, metrics)
        test_mse = self._extract_test_mse(metrics)

        return ExperimentResult(
            name=exp_name,
            experiment_id=experiment_id,
            status=status,
            epochs_completed=job_history.get('current_epoch', 0) if job_history else 0,
            total_epochs=job_history.get('total_epochs', 0) if job_history else 0,
            completion_reason=job_history.get('completion_reason') if job_history else None,
            train_mse=train_mse,
            val_mse=val_mse,
            test_mse=test_mse,
            experiment_dir=experiment_dir
        )

    def _load_job_history(self, experiment_dir: Path) -> Optional[Dict[str, Any]]:
        """Load job_history.json if it exists."""
        job_history_path = experiment_dir / "job_history.json"
        if not job_history_path.exists():
            return None

        try:
            with open(job_history_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _load_metrics(self, experiment_dir: Path) -> Optional[Dict[str, Any]]:
        """Load metrics/metrics.json if it exists."""
        metrics_path = experiment_dir / "metrics" / "metrics.json"
        if not metrics_path.exists():
            return None

        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _determine_status(
        self,
        job_history: Optional[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]]
    ) -> str:
        """
        Determine experiment status.

        Returns one of: completed, failed, skipped, not_run, partial

        Args:
            job_history: Job history dictionary from job_history.json
            metrics: Metrics dictionary from metrics/metrics.json

        Returns:
            Status string
        """
        if job_history is None:
            return "not_run"

        current_epoch = job_history.get('current_epoch', 0)
        total_epochs = job_history.get('total_epochs', 0)
        completion_reason = job_history.get('completion_reason')

        # Check if completed
        if current_epoch >= total_epochs:
            return "completed"

        # Check if early stopped
        if completion_reason == "early_stopping":
            return "completed"

        # Check if failed
        jobs = job_history.get('jobs', [])
        if jobs:
            last_job = jobs[-1]
            if last_job.get('status') == 'failed':
                return "failed"

        # Check if skipped (experiment already complete, not rerun)
        # This would be indicated by completion_reason existing but no recent jobs
        if completion_reason and not jobs:
            return "skipped"

        # Partial completion
        if current_epoch > 0 and current_epoch < total_epochs:
            return "partial"

        return "not_run"

    def _extract_train_mse(
        self,
        job_history: Optional[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]]
    ) -> Optional[float]:
        """Extract final training MSE (normalized)."""
        # Try metrics.json first
        if metrics:
            if 'final_train_loss' in metrics:
                return metrics['final_train_loss']
            if 'train_loss' in metrics:
                return metrics['train_loss']

        # Try job_history.json
        if job_history:
            jobs = job_history.get('jobs', [])
            if jobs:
                last_job = jobs[-1]
                if 'final_train_loss' in last_job:
                    return last_job['final_train_loss']

        return None

    def _extract_val_mse(
        self,
        job_history: Optional[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]]
    ) -> Optional[float]:
        """Extract final validation MSE (normalized)."""
        # Try metrics.json first
        if metrics:
            if 'final_val_loss' in metrics:
                return metrics['final_val_loss']
            if 'val_loss' in metrics:
                return metrics['val_loss']

        # Try job_history.json
        if job_history:
            jobs = job_history.get('jobs', [])
            if jobs:
                last_job = jobs[-1]
                if 'final_val_loss' in last_job:
                    return last_job['final_val_loss']

        return None

    def _extract_test_mse(self, metrics: Optional[Dict[str, Any]]) -> Optional[float]:
        """Extract test MSE (normalized)."""
        if metrics is None:
            return None

        # Try various keys for test metrics
        test_keys = ['test/mse_normalized', 'final_test_loss', 'test_loss']
        for key in test_keys:
            if key in metrics:
                return metrics[key]

        return None
