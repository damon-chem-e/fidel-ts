# Suite Results Summary Tables - Implementation Plan

## Overview

Add rich-formatted summary tables at the end of suite execution to display:
1. **Metrics Table**: Train/val/test normalized MSE for each experiment
2. **Status Table**: Experiment metadata including completion status, epochs, and completion reasons

These tables provide a quick overview of suite results without needing to check individual experiment directories.

## Requirements

### Table 1: Metrics Summary
- **Columns**: Experiment Name, Experiment ID, Train MSE (norm), Val MSE (norm), Test MSE (norm)
- **Data Source**: `{experiment_dir}/metrics/metrics.json`
- **Behavior**:
  - Only include experiments that have metrics available
  - Display "N/A" or "-" for missing metrics (e.g., if test wasn't run)
  - Sort by experiment name or execution order
- **Footer**: Display suite ID after the table

### Table 2: Experiment Status
- **Columns**:
  - Experiment Name
  - Experiment ID (abbreviated)
  - Status (completed/failed/not_run/skipped)
  - Epochs Completed
  - Total Epochs
  - Completion Reason
- **Data Source**: `{experiment_dir}/job_history.json`
- **Behavior**:
  - Include ALL experiments in the suite (even those not run)
  - If `continue_on_error=False` and suite stopped early, mark unrun experiments with status "not_run_error"
  - Use abbreviations for status and completion reasons
  - Include legend below table mapping abbreviations to full descriptions
- **Statuses**:
  - `COMP` - Completed (all epochs finished)
  - `FAIL` - Failed (error occurred)
  - `SKIP` - Skipped (already complete, not rerun)
  - `NORUN` - Not run (suite stopped before this experiment)
  - `PART` - Partial (some epochs completed, but not all)
- **Completion Reasons**:
  - `ALL` - All epochs completed
  - `EARLY` - Early stopping triggered
  - `TIME` - Timeout (SLURM or manual)
  - `ERR` - Error during training
  - `UNK` - Unknown (reason not recorded)

## Current State Analysis

### SuiteExecutor Structure
Located in `runs/suite_executor.py`:
- `SuiteExecutor.__init__()`: Initializes suite, creates suite directory
- `SuiteExecutor.execute()`: Main execution loop, iterates through experiments
- `SuiteExecutor._execute_experiment()`: Runs individual experiment
- `SuiteExecutor._run_single_experiment()`: Calls appropriate run function (pytorch/lightning/llm/fm)

Current execution flow:
```python
def execute(self):
    experiments = [...]
    success_count = 0
    error_count = 0

    for exp_config in experiments:
        try:
            self._execute_experiment(exp_config)
            success_count += 1
        except Exception as e:
            error_count += 1
            if not continue_on_error:
                raise

    logger.info(f"Suite execution completed: {success_count} successful, {error_count} errors")
    # <-- ADD TABLE GENERATION HERE
    return {...}
```

### Experiment Data Structure

Each experiment has:
- **Directory**: `{output_dir}/{suite_name}/{experiment_id}/`
- **Metrics file**: `{experiment_dir}/metrics/metrics.json`
  ```json
  {
    "train_loss": 0.123,
    "val_loss": 0.456,
    "test/mse_normalized": 0.789,
    "test/mae_normalized": 0.234,
    "final_train_loss": 0.100,
    "final_val_loss": 0.400,
    "final_test_loss": 0.750,
    ...
  }
  ```
- **Job history**: `{experiment_dir}/job_history.json`
  ```json
  {
    "experiment_id": "model_data_config_12abc",
    "experiment_name": "dlinear_solar",
    "total_epochs": 20,
    "current_epoch": 20,
    "jobs": [
      {
        "job_number": 1,
        "start_time": "2024-01-17T10:00:00",
        "end_time": "2024-01-17T12:00:00",
        "status": "completed",
        "start_epoch": 1,
        "end_epoch": 20,
        "final_train_loss": 0.100,
        "final_val_loss": 0.400
      }
    ],
    "completion_reason": "all_epochs"
  }
  ```

### Metrics Key Mapping
From analysis of `exp/exp_universal.py` and `exp/exp_lightning.py`:
- **Train**: `final_train_loss` (from job_history or metrics.json)
- **Val**: `final_val_loss` (from job_history or metrics.json)
- **Test**: `test/mse_normalized` (from metrics.json after test evaluation)

Alternative keys to check (for robustness):
- Train: `train_loss` (last logged), `final_train_loss`
- Val: `val_loss` (last logged), `final_val_loss`
- Test: `test/mse_normalized`, `final_test_loss`

## Implementation Plan

### Phase 1: Data Collection Module

Create a new module `runs/suite_results_collector.py`:

```python
"""
Collect and aggregate results from all experiments in a suite.
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
        experiment_dir: Path
    ):
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
            with open(job_history_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _load_metrics(self, experiment_dir: Path) -> Optional[Dict[str, Any]]:
        """Load metrics/metrics.json if it exists."""
        metrics_path = experiment_dir / "metrics" / "metrics.json"
        if not metrics_path.exists():
            return None

        try:
            with open(metrics_path, 'r') as f:
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
```

### Phase 2: Table Rendering Module

Create `runs/suite_results_display.py`:

```python
"""
Display suite results using rich tables.
"""

from pathlib import Path
from typing import List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from runs.suite_results_collector import ExperimentResult


# Status abbreviation mapping
STATUS_ABBREV = {
    'completed': 'COMP',
    'failed': 'FAIL',
    'skipped': 'SKIP',
    'not_run': 'NORUN',
    'partial': 'PART'
}

STATUS_LEGEND = {
    'COMP': 'Completed (all epochs or early stopping)',
    'FAIL': 'Failed (error during execution)',
    'SKIP': 'Skipped (already complete)',
    'NORUN': 'Not run (suite stopped before this experiment)',
    'PART': 'Partial (some epochs completed)'
}

# Completion reason abbreviation mapping
REASON_ABBREV = {
    'all_epochs': 'ALL',
    'early_stopping': 'EARLY',
    'timeout': 'TIME',
    'error': 'ERR',
    None: 'UNK'
}

REASON_LEGEND = {
    'ALL': 'All epochs completed',
    'EARLY': 'Early stopping triggered',
    'TIME': 'Timeout (SLURM or manual)',
    'ERR': 'Error during training',
    'UNK': 'Unknown (reason not recorded)'
}


class SuiteResultsDisplay:
    """Display suite results as rich-formatted tables."""

    def __init__(self, console: Console = None):
        self.console = console or Console()

    def display_all_tables(
        self,
        results: List[ExperimentResult],
        suite_name: str,
        suite_id: str
    ):
        """
        Display both metrics and status tables.

        Args:
            results: List of ExperimentResult objects
            suite_name: Base suite name (without timestamp)
            suite_id: Full suite ID (with timestamp)
        """
        self.console.print("\n")
        self.console.print(Panel.fit(
            f"[bold cyan]Suite Results Summary[/bold cyan]\n"
            f"Suite: [yellow]{suite_name}[/yellow]",
            border_style="cyan"
        ))
        self.console.print("\n")

        # Display metrics table
        self._display_metrics_table(results)
        self.console.print(f"\n[dim]Suite ID: {suite_id}[/dim]\n")

        # Display status table
        self._display_status_table(results)

        # Display legends
        self._display_legends()

    def _display_metrics_table(self, results: List[ExperimentResult]):
        """Display table with train/val/test MSE metrics."""
        table = Table(
            title="Experiment Metrics (Normalized MSE)",
            show_header=True,
            header_style="bold magenta",
            title_style="bold white"
        )

        table.add_column("Experiment Name", style="cyan", no_wrap=False)
        table.add_column("Experiment ID", style="dim", no_wrap=True)
        table.add_column("Train MSE", justify="right", style="green")
        table.add_column("Val MSE", justify="right", style="yellow")
        table.add_column("Test MSE", justify="right", style="blue")

        for result in results:
            # Only include experiments that have at least some metrics
            # (skip experiments that were never run)
            if result.status == "not_run":
                continue

            # Format experiment ID (show last 8 chars for readability)
            exp_id_short = result.experiment_id[-12:] if len(result.experiment_id) > 12 else result.experiment_id

            # Format metrics (N/A if not available)
            train_mse_str = f"{result.train_mse:.7f}" if result.train_mse is not None else "N/A"
            val_mse_str = f"{result.val_mse:.7f}" if result.val_mse is not None else "N/A"
            test_mse_str = f"{result.test_mse:.7f}" if result.test_mse is not None else "N/A"

            table.add_row(
                result.name,
                exp_id_short,
                train_mse_str,
                val_mse_str,
                test_mse_str
            )

        self.console.print(table)

    def _display_status_table(self, results: List[ExperimentResult]):
        """Display table with experiment status and completion info."""
        table = Table(
            title="Experiment Status",
            show_header=True,
            header_style="bold magenta",
            title_style="bold white"
        )

        table.add_column("Experiment Name", style="cyan", no_wrap=False)
        table.add_column("Exp ID", style="dim", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Epochs", justify="center", style="yellow")
        table.add_column("Reason", justify="center", style="blue")

        for result in results:
            # Format experiment ID (last 8 chars)
            exp_id_short = result.experiment_id[-8:] if len(result.experiment_id) > 8 else result.experiment_id

            # Get status abbreviation
            status_abbrev = STATUS_ABBREV.get(result.status, result.status.upper()[:5])

            # Color status based on completion
            if result.status == 'completed':
                status_str = f"[green]{status_abbrev}[/green]"
            elif result.status == 'failed':
                status_str = f"[red]{status_abbrev}[/red]"
            elif result.status == 'partial':
                status_str = f"[yellow]{status_abbrev}[/yellow]"
            else:
                status_str = f"[dim]{status_abbrev}[/dim]"

            # Format epochs
            epochs_str = f"{result.epochs_completed}/{result.total_epochs}"
            if result.total_epochs == 0:
                epochs_str = "N/A"

            # Get reason abbreviation
            reason_abbrev = REASON_ABBREV.get(result.completion_reason, 'UNK')

            table.add_row(
                result.name,
                exp_id_short,
                status_str,
                epochs_str,
                reason_abbrev
            )

        self.console.print(table)

    def _display_legends(self):
        """Display legends for abbreviations used in tables."""
        self.console.print("\n[bold]Legends:[/bold]")

        # Status legend
        self.console.print("  [underline]Status:[/underline]")
        for abbrev, desc in STATUS_LEGEND.items():
            self.console.print(f"    • {abbrev} = {desc}")

        # Reason legend
        self.console.print("\n  [underline]Completion Reason:[/underline]")
        for abbrev, desc in REASON_LEGEND.items():
            self.console.print(f"    • {abbrev} = {desc}")

        self.console.print("\n")
```

### Phase 3: Integration into SuiteExecutor

Modify `runs/suite_executor.py`:

```python
# Add imports at top
from runs.suite_results_collector import SuiteResultsCollector
from runs.suite_results_display import SuiteResultsDisplay
from rich.console import Console

# In SuiteExecutor.__init__(), add:
self.console = Console()
# Registry to track experiment names -> experiment IDs
# Populated as experiments are initialized by ExperimentManager
self.experiment_id_registry: Dict[str, str] = {}

# In SuiteExecutor.execute(), after the execution loop:
def execute(self) -> Optional[Dict[str, Any]]:
    experiments = [...]

    # ... existing execution loop ...

    logger.info(f"Suite execution completed: {success_count} successful, {error_count} errors")

    # NEW: Collect and display results
    try:
        self._display_suite_results()
    except Exception as e:
        # Don't fail suite if results display fails
        logger.warning(f"Failed to display suite results: {e}")

    # Return suite and experiment IDs if requested
    if self.return_ids:
        return {
            'suite_id': self.suite_name,
            'experiment_ids': self.experiment_ids.copy()
        }
    return None

# Add new method to SuiteExecutor:
def register_experiment_id(self, experiment_name: str, experiment_id: str):
    """
    Register an experiment ID with the suite executor.

    This method is called by ExperimentManager when an experiment ID is generated,
    allowing the suite to track which experiments were started and their IDs.

    Args:
        experiment_name: Name of the experiment (from suite config)
        experiment_id: Generated experiment ID
    """
    self.experiment_id_registry[experiment_name] = experiment_id
    logger.debug(f"Registered experiment '{experiment_name}' with ID '{experiment_id}'")

# Add new method to SuiteExecutor:
def _display_suite_results(self):
    """Collect and display suite results tables."""
    collector = SuiteResultsCollector(
        self.suite_dir,
        self.suite_config,
        self.experiment_id_registry  # Pass the registry
    )
    results = collector.collect_results()

    display = SuiteResultsDisplay(console=self.console)
    display.display_all_tables(
        results=results,
        suite_name=self.suite_name_base,
        suite_id=self.suite_name
    )
```

Modify `runs/pytorch.py`, `runs/lightning.py`, `runs/llm.py`, and `runs/fm.py`:

```python
# In the run() function, after creating ExperimentManager, register the experiment ID
def run(experiment_config, suite_name=None, suite_info=None, output_dir=None,
        init_only=False, return_ids=False, sweep=False, suite_executor=None):
    """
    Run PyTorch experiment.

    Args:
        ...
        suite_executor: Optional SuiteExecutor instance for registering experiment IDs
    """
    # Create ExperimentManager
    exp_manager = ExperimentManager(
        config=experiment_config,
        output_dir=output_dir or "./output",
        experiment_name=experiment_config.experiment_name,
        suite_name=suite_name,
        suite_info=suite_info,
        init_only=init_only,
        sweep=sweep
    )

    # Register experiment ID with suite executor if in suite context
    if suite_executor is not None:
        suite_executor.register_experiment_id(
            experiment_config.experiment_name,
            exp_manager.experiment_id
        )

    # ... rest of run logic ...
```

Modify `SuiteExecutor._run_single_experiment()` to pass suite executor reference:

```python
def _run_single_experiment(self, config: Dict[str, Any], experiment_name: str) -> Optional[str]:
    """
    Run a single experiment configuration.

    Args:
        config: Complete experiment configuration dictionary
        experiment_name: Name of the experiment

    Returns:
        experiment_id if return_ids is enabled, None otherwise
    """
    exp_type = config.get('experiment', {}).get('type', 'pytorch')

    logger.info(f"Running experiment '{experiment_name}' with type '{exp_type}'")

    # Prepare suite information to pass to experiments
    suite_info = {
        "name": self.suite_name_base,
        "name_with_timestamp": self.suite_name,
        "timestamp": self.suite_timestamp,
        "description": self.suite_info.get('description', ''),
        "tags": self.suite_info.get('tags', [])
    }

    # Execute based on experiment type, passing suite_executor=self
    result = None
    if exp_type == 'pytorch':
        result = run_pytorch(
            experiment_config,
            suite_name=self.suite_name,
            suite_info=suite_info,
            output_dir=str(self.output_dir),
            init_only=self.init_only,
            return_ids=self.return_ids,
            sweep=self.sweep,
            suite_executor=self  # Pass suite executor reference
        )
    elif exp_type == 'lightning':
        result = run_lightning(
            experiment_config,
            suite_name=self.suite_name,
            suite_info=suite_info,
            output_dir=str(self.output_dir),
            init_only=self.init_only,
            return_ids=self.return_ids,
            sweep=self.sweep,
            suite_executor=self  # Pass suite executor reference
        )
    # ... similar for llm and fm ...

    # Extract experiment_id from result if available
    if self.return_ids and result:
        return result.get('experiment_id')
    return None
```

## Edge Cases and Considerations

### 1. Experiments with Multiple output_lens
If an experiment has multiple output_lens (see line 409 in suite_executor.py), it creates multiple sub-experiments:
- Each gets a unique name: `{exp_name}_output{output_len}`
- Each gets its own experiment_id
- Solution: Treat as separate experiments in the table

### 2. Missing Metrics
- Train/Val MSE: Should exist if at least one epoch ran
- Test MSE: Only exists if test evaluation ran (may be disabled)
- Display "N/A" for missing metrics

### 3. Skipped Experiments (already complete)
- Status: "SKIP"
- Metrics: Should be available from previous run
- Implementation: Read metrics from existing experiment directory

### 4. Suite Resumption
- When resuming a suite, some experiments may be complete, others partial
- Collector should handle both new and resumed experiments
- Read from existing job_history.json to determine state

### 5. Continue on Error = False
- If suite stops due to error, remaining experiments won't have directories
- Status: "NORUN"
- Implementation: Check if experiment_id is in registry (will be None if experiment never started)

### 6. Evaluation-Only Experiments
- Suite configs can have `type: evaluation` experiments
- These don't have train/val, only test metrics
- Handle gracefully (show N/A for train/val)

### 7. Very Long Experiment Names
- Wrap or truncate in table display
- Use `no_wrap=False` for experiment name column

## Testing Strategy

### Unit Tests
Create `tests/test_suite_results.py`:

1. Test `SuiteResultsCollector`:
   - Test with complete experiments
   - Test with failed experiments
   - Test with missing metrics
   - Test with missing job_history
   - Test with skipped experiments

2. Test `SuiteResultsDisplay`:
   - Test table rendering (smoke test)
   - Test with empty results
   - Test with mixed statuses

### Integration Tests
Create test suite configs:
1. Suite with all successful experiments
2. Suite with mixed success/failure
3. Suite with some skipped experiments
4. Suite with continue_on_error=False and early failure

Run on RunPod with small experiments (1-2 epochs) to verify end-to-end behavior.

## Dependencies

### New Dependencies
- `rich` - Already used in the codebase (see `exp/manager.py`)
- No new external dependencies required

### Existing Dependencies
- `pathlib` (stdlib)
- `json` (stdlib)
- `typing` (stdlib)

## Implementation Checklist

- [ ] Create `runs/suite_results_collector.py`
  - [ ] Implement `ExperimentResult` dataclass
  - [ ] Implement `SuiteResultsCollector` class
  - [ ] Add `experiment_id_registry` parameter to constructor
  - [ ] Implement `collect_results()` method
  - [ ] Update `_collect_experiment_result()` to use registry instead of inferring
  - [ ] Implement helper methods for loading and parsing files
  - [ ] Implement status determination logic
  - [ ] Implement metrics extraction logic

- [ ] Create `runs/suite_results_display.py`
  - [ ] Define status/reason abbreviations and legends
  - [ ] Implement `SuiteResultsDisplay` class
  - [ ] Implement metrics table rendering
  - [ ] Implement status table rendering
  - [ ] Implement legend rendering with bullet points
  - [ ] Test rich formatting and colors

- [ ] Modify `runs/suite_executor.py`
  - [ ] Add imports for new modules
  - [ ] Add console initialization
  - [ ] Add `experiment_id_registry` dictionary
  - [ ] Add `register_experiment_id()` method
  - [ ] Add `_display_suite_results()` method
  - [ ] Call display method at end of `execute()`
  - [ ] Add error handling (display failures shouldn't break suite)
  - [ ] Pass `suite_executor=self` to run functions

- [ ] Modify `runs/pytorch.py`, `runs/lightning.py`, `runs/llm.py`, `runs/fm.py`
  - [ ] Add `suite_executor` parameter to run() functions
  - [ ] Call `suite_executor.register_experiment_id()` after creating ExperimentManager
  - [ ] Ensure registration happens even if experiment fails later

- [ ] Create tests
  - [ ] Unit tests for `SuiteResultsCollector`
  - [ ] Unit tests for `SuiteResultsDisplay`
  - [ ] Integration test with small test suite

- [ ] Documentation
  - [ ] Update `docs/CLAUDE.md` to mention new feature
  - [ ] Add examples of output to documentation

## Example Output

```
╭─────────────────────────────────────────────────────────────╮
│         Suite Results Summary                                │
│ Suite: linear_models                                         │
╰─────────────────────────────────────────────────────────────╯


         Experiment Metrics (Normalized MSE)
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ Experiment Name  ┃ Experiment ID┃ Train MSE ┃ Val MSE  ┃ Test MSE ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ dlinear_solar    │ ...abc123def │  0.0012345│ 0.0014567│ 0.0015678│
│ nlinear_solar    │ ...def456ghi │  0.0013456│ 0.0015678│ 0.0016789│
│ patchtst_solar   │ ...ghi789jkl │  0.0011234│ 0.0013456│      N/A │
└──────────────────┴──────────────┴───────────┴──────────┴──────────┘

Suite ID: linear_models_20240117_103045


                  Experiment Status
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Experiment Name  ┃ Exp ID   ┃ Status ┃ Epochs ┃ Reason ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ dlinear_solar    │ abc123de │  COMP  │ 20/20  │  ALL   │
│ nlinear_solar    │ def456gh │  COMP  │ 15/20  │ EARLY  │
│ patchtst_solar   │ ghi789jk │  PART  │ 10/20  │  TIME  │
│ fits_solar       │      N/A │ NORUN  │   N/A  │  UNK   │
└──────────────────┴──────────┴────────┴────────┴────────┘

Legends:
  Status:
    • COMP = Completed (all epochs or early stopping)
    • FAIL = Failed (error during execution)
    • SKIP = Skipped (already complete)
    • NORUN = Not run (suite stopped before this experiment)
    • PART = Partial (some epochs completed)

  Completion Reason:
    • ALL = All epochs completed
    • EARLY = Early stopping triggered
    • TIME = Timeout (SLURM or manual)
    • ERR = Error during training
    • UNK = Unknown (reason not recorded)
```

## Alternative Approaches Considered

### 1. Store Results in Suite Metadata
**Approach**: Store aggregated results in `suite_metadata.json` as suite executes
- **Pros**: Centralized results, easier to parse later
- **Cons**: Need to update metadata during execution, adds complexity
- **Decision**: Rejected - prefer stateless collection at the end

### 2. Use pandas DataFrame for Display
**Approach**: Collect results in DataFrame, use tabulate/prettytable
- **Pros**: Easier data manipulation
- **Cons**: Adds pandas dependency, `rich` is already used in codebase
- **Decision**: Rejected - `rich` is sufficient and already available

### 3. Separate CLI Command for Results Display
**Approach**: Add `python -m cli.suite results <suite_path>`
- **Pros**: Can view results without re-running suite
- **Cons**: Doesn't meet requirement of displaying at end of execution
- **Decision**: Could add as future enhancement, but implement inline display first

## Future Enhancements

1. **Export to CSV/JSON**: Add option to save tables as files
2. **Comparison Mode**: Compare results across multiple suite runs
3. **Detailed Metrics**: Add columns for MAE, RMSE, etc.
4. **Filtering**: Add options to filter by status, model type, etc.
5. **Sorting**: Allow sorting by metric values
6. **Plot Generation**: Generate comparison plots alongside tables
7. **HTML Report**: Generate rich HTML report with tables and plots

## Summary

This plan provides a comprehensive approach to adding suite results summary tables:
- **Phase 1**: Data collection from experiment directories
- **Phase 2**: Rich-formatted table rendering
- **Phase 3**: Integration into suite execution

The implementation is:
- **Non-invasive**: Doesn't modify existing experiment logic significantly
- **Robust**: Handles edge cases (missing data, errors, resumption)
- **User-friendly**: Clear visual display with abbreviations and legends (bullet-point format)
- **Maintainable**: Modular design with separate collection and display logic
- **Reliable**: Experiment IDs are registered by ExperimentManager when generated, not inferred from directories

**Key Design Decision - Experiment ID Registration:**
Instead of inferring experiment IDs from directory names (brittle, error-prone), we use a registration pattern where:
1. SuiteExecutor maintains an `experiment_id_registry` dictionary
2. When ExperimentManager creates an experiment, it calls `suite_executor.register_experiment_id()`
3. The registry is passed to SuiteResultsCollector for reliable ID lookup
4. This ensures accurate tracking even if directories have naming inconsistencies or experiments fail early

All code lives in the suite management layer (`runs/` directory) as requested.
