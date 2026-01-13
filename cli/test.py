"""
Testing/Evaluation command-line interface for Fidel-TS.

This module provides CLI commands for evaluating trained models:
- Standard evaluation (for PyTorch-trained models)
- Lightning evaluation (for Lightning-trained models)
- LLM evaluation (for LLM-based forecasting results)
- Suite evaluation (for all experiments within a suite)
"""

import typer
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from cli.config.loader import load_config_with_nested, load_config, load_yaml_config
from cli.config.models import ExperimentConfig
from evaluation.config_builder import build_evaluation_config_from_experiment_config
from utils.tools import dotdict

app = typer.Typer(
    name="test",
    help="Evaluate trained time series forecasting models",
    add_completion=False
)


def _validate_experiment_checkpoint(
    output_dir: Path,
    suite_id: str,
    experiment_id: str
) -> Tuple[bool, str, Optional[Path]]:
    """
    Validate that an experiment has a valid checkpoint for evaluation.
    
    Checks:
    1. Experiment directory exists
    2. Configs directory exists with experiment_config.yaml
    3. Checkpoints directory exists with at least one checkpoint file
    
    Args:
        output_dir: Base output directory
        suite_id: Suite ID (with timestamp)
        experiment_id: Experiment resume ID
        
    Returns:
        Tuple of (is_valid, reason, experiment_dir):
            - is_valid: True if checkpoint exists and is valid
            - reason: Human-readable reason if invalid, empty string if valid
            - experiment_dir: Path to experiment directory if valid, None otherwise
    """
    # Build experiment directory path
    experiment_dir = output_dir / suite_id / experiment_id
    
    # Check 1: Experiment directory exists
    if not experiment_dir.exists():
        return False, f"Experiment directory not found: {experiment_dir}", None
    
    # Check 2: Configs directory and experiment_config.yaml exist
    config_path = experiment_dir / "configs" / "experiment_config.yaml"
    if not config_path.exists():
        return False, f"Experiment config not found: {config_path}", None
    
    # Check 3: Checkpoints directory exists with at least one checkpoint
    checkpoints_dir = experiment_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return False, f"Checkpoints directory not found: {checkpoints_dir}", None
    
    # Look for checkpoint files (pth, ckpt)
    checkpoint_files = (
        list(checkpoints_dir.glob("*.pth")) +
        list(checkpoints_dir.glob("*.ckpt")) +
        list(checkpoints_dir.glob("best_checkpoint*"))
    )
    # Filter to only files (not directories)
    checkpoint_files = [f for f in checkpoint_files if f.is_file()]
    
    if not checkpoint_files:
        return False, f"No checkpoint files found in: {checkpoints_dir}", None
    
    return True, "", experiment_dir


def _extract_experiments_from_suite(
    suite_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Extract enabled experiments from suite configuration.
    
    Args:
        suite_config: Parsed suite configuration dictionary
        
    Returns:
        List of enabled experiment configurations
    """
    suite_info = suite_config.get('suite', {})
    experiments = suite_info.get('experiments', [])
    
    # Filter to enabled experiments only
    return [
        exp for exp in experiments
        if exp.get('enabled', True)
    ]


def _get_experiment_resume_id(exp_config: Dict[str, Any]) -> Optional[str]:
    """
    Extract resume_experiment_id from experiment configuration.
    
    Checks both top-level overrides and nested experiment block.
    
    Args:
        exp_config: Single experiment configuration from suite
        
    Returns:
        resume_experiment_id if found, None otherwise
    """
    overrides = exp_config.get('overrides', {})
    
    # Check top-level overrides first
    resume_id = overrides.get('resume_experiment_id')
    if resume_id:
        return resume_id
    
    # Check nested experiment block
    experiment_block = overrides.get('experiment', {})
    return experiment_block.get('resume_experiment_id')


def _load_experiment_config_from_directory(experiment_dir: Path) -> ExperimentConfig:
    """
    Load experiment config from saved config in experiment directory.
    
    This is the preferred method for suite experiments since the config
    is already saved in the experiment directory during training.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        ExperimentConfig instance loaded from saved config
        
    Raises:
        FileNotFoundError: If experiment config not found
    """
    config_path = experiment_dir / "configs" / "experiment_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Experiment config not found: {config_path}\n"
            f"  Experiment directory: {experiment_dir}\n"
            f"  This experiment may not have been saved with the new format."
        )
    
    return ExperimentConfig.from_yaml(config_path)


def _load_and_resolve_config(
    config_path: str,
    resume_experiment_id: Optional[str],
    resume_suite_id: Optional[str],
    output_dir: str
) -> tuple[ExperimentConfig, str, Optional[str]]:
    """
    Load and resolve experiment configuration.
    
    Handles both suite configs and regular experiment configs, including
    backward compatibility for old evaluation config format.
    
    Args:
        config_path: Path to config file (suite or experiment config)
        resume_experiment_id: Experiment ID from CLI (optional)
        resume_suite_id: Suite ID from CLI (optional)
        output_dir: Base output directory
        
    Returns:
        tuple: (config, exp_id, suite_id)
            - config: ExperimentConfig instance
            - exp_id: Experiment ID to use
            - suite_id: Suite ID (None if standalone)
            
    Raises:
        FileNotFoundError: If experiment directory or config not found
        typer.Exit: If resume_experiment_id is required but not provided
    """
    # First, check if this is a suite config or experiment config
    config_dict = load_yaml_config(config_path)
    
    # If it's a suite config, we need resume_experiment_id to load config from experiment directory
    if 'suite' in config_dict:
        # This is a suite config - need resume_experiment_id to find the experiment directory
        # Get resume_suite_id from CLI or config file
        suite_info = config_dict.get('suite', {})
        suite_id = resume_suite_id or suite_info.get('resume_suite_id') or suite_info.get('name', 'unknown')
        
        # For resume_experiment_id, user must specify which experiment to test
        # (suite configs have multiple experiments, so we need to know which one)
        if not resume_experiment_id:
            typer.echo(
                "Error: resume_experiment_id is required when using a suite config.\n"
                "  Provide it via CLI (--resume-id).\n"
                "  Note: Each experiment in the suite has its own resume_experiment_id in the config file.",
                err=True
            )
            raise typer.Exit(code=1)
        
        exp_id = resume_experiment_id
        
        # Determine experiment directory and load config from there
        output_path = Path(output_dir).resolve()
        if suite_id:
            experiment_dir = output_path / suite_id / exp_id
        else:
            experiment_dir = output_path / exp_id
        
        if not experiment_dir.exists():
            raise FileNotFoundError(
                f"Experiment directory not found: {experiment_dir}\n"
                f"  Experiment ID: {exp_id}\n"
                f"  Suite ID: {suite_id if suite_id else 'N/A (standalone)'}"
            )
        
        # Load config from experiment directory (preferred - has actual training config)
        config = _load_experiment_config_from_directory(experiment_dir)
        return config, exp_id, suite_id
    
    # This is a regular experiment config - check for backward compatibility first
    config_hierarchy = load_config_with_nested(config_path)
    nested_configs = config_hierarchy.get('nested', {})
    
    # If evaluation section exists as nested config (old format), use backward compatibility
    if 'evaluation' in nested_configs or (hasattr(config_hierarchy['primary'], 'evaluation') and 
                                          isinstance(config_hierarchy['primary'].evaluation, str)):
        # Old format: use nested evaluation config (return config with None IDs for backward compat)
        config = config_hierarchy['primary']
        return config, None, None
    
    # New format: use experiment config with resume_experiment_id
    config = load_config(config_path)
    
    # Get resume_experiment_id from CLI override or config
    exp_id = resume_experiment_id or config.resume_experiment_id
    suite_id = resume_suite_id or config.resume_suite_id
    
    return config, exp_id, suite_id


@app.command()
def standard(
    config_path: str = typer.Argument(..., help="Path to experiment configuration file (same as training)"),
    resume_experiment_id: Optional[str] = typer.Option(None, "--resume-id", "-r", help="Experiment ID to evaluate (overrides config.resume_experiment_id)"),
    resume_suite_id: Optional[str] = typer.Option(None, "--resume-suite-id", help="Suite ID if experiment is part of a suite (overrides config.resume_suite_id)"),
    output_dir: str = typer.Option("./output", "--output-dir", "-o", help="Base output directory"),
    version: str = typer.Option("best", "--version", "-v", help="Checkpoint version: 'best', 'latest', or specific pattern"),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="GPU device ID (overrides config.device.gpu)"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size (overrides config.training.batch_size)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without running evaluation")
):
    """
    Evaluate standard PyTorch-trained models.
    
    This command evaluates models trained with the standard PyTorch pipeline.
    Uses the same experiment config file as training, requiring resume_experiment_id
    to identify which experiment to evaluate.
    
    Examples:
        # Using config with resume_experiment_id
        python -m cli.test standard configs/experiments/dlinear_solar.yaml
        
        # Override resume ID via CLI
        python -m cli.test standard configs/experiments/dlinear_solar.yaml --resume-id 20240101-abc123
        
        # Suite experiment
        python -m cli.test standard configs/experiments/dlinear_solar.yaml --resume-id 20240101-abc123 --resume-suite-id my_suite_20240101_120000
        
        # Override version, device, and batch size
        python -m cli.test standard configs/experiments/dlinear_solar.yaml --version latest --device 1 --batch-size 64
    """
    try:
        # Load and resolve config (handles suite configs, regular configs, and backward compatibility)
        config, exp_id, suite_id = _load_and_resolve_config(
            config_path, resume_experiment_id, resume_suite_id, output_dir
        )
        
        # Handle backward compatibility: old evaluation config format
        if exp_id is None:
            # Old format: use nested evaluation config
            typer.echo("Warning: Using legacy evaluation config format. Consider migrating to resume_experiment_id.", err=True)
            
            if dry_run:
                typer.echo(f"✓ Config validated: {config_path}")
                typer.echo(f"  Model: {config.model.name}")
                typer.echo(f"  Data: {config.data.name}")
                return
            
            from evaluation.standard import evaluate
            typer.echo(f"Starting standard evaluation with config: {config_path}")
            evaluate(config)
            return
        
        # Validate that resume_experiment_id is provided
        if not exp_id:
            typer.echo(
                "Error: resume_experiment_id is required for evaluation.\n"
                "  Provide it either in the config file (resume_experiment_id: '...') or via CLI (--resume-id).",
                err=True
            )
            raise typer.Exit(code=1)
        
        # Build evaluation config from experiment config
        eval_config = build_evaluation_config_from_experiment_config(
            config=config,
            resume_experiment_id=exp_id,
            resume_suite_id=suite_id,
            output_dir=output_dir,
            version=version,
            device_override=device,
            batch_size_override=batch_size
        )
        
        if dry_run:
            typer.echo(f"✓ Config validated: {config_path}")
            typer.echo(f"  Experiment ID: {exp_id}")
            if suite_id:
                typer.echo(f"  Suite ID: {suite_id}")
            typer.echo(f"  Model: {eval_config.model}")
            typer.echo(f"  Data: {eval_config.data}")
            typer.echo(f"  Task: {eval_config.task}")
            typer.echo(f"  Input/Output length: {eval_config.input_len}/{eval_config.output_len}")
            typer.echo(f"  Batch size: {eval_config.batch_size}")
            typer.echo(f"  Device: {eval_config.device}")
            typer.echo(f"  Checkpoint version: {eval_config.version}")
            typer.echo(f"  Experiment directory: {eval_config.experiment_dir}")
            return
        
        # Import here to avoid circular imports
        from evaluation.standard import evaluate
        
        # Create config structure expected by evaluate function
        config_with_eval = dotdict({
            'evaluation': eval_config,
            'data_config': None  # Will use checkpoint's data config
        })
        
        typer.echo(f"Starting standard evaluation for experiment: {exp_id}")
        typer.echo(f"  Config: {config_path}")
        typer.echo(f"  Experiment directory: {eval_config.experiment_dir}")
        evaluate(config_with_eval)
        
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error during evaluation: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def lightning(
    config_path: str = typer.Argument(..., help="Path to experiment configuration file (same as training)"),
    resume_experiment_id: Optional[str] = typer.Option(None, "--resume-id", "-r", help="Experiment ID to evaluate (overrides config.resume_experiment_id)"),
    resume_suite_id: Optional[str] = typer.Option(None, "--resume-suite-id", help="Suite ID if experiment is part of a suite (overrides config.resume_suite_id)"),
    output_dir: str = typer.Option("./output", "--output-dir", "-o", help="Base output directory"),
    version: str = typer.Option("best", "--version", "-v", help="Checkpoint version: 'best', 'latest', or specific pattern"),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="GPU device ID (overrides config.device.gpu)"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size (overrides config.training.batch_size)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without running evaluation")
):
    """
    Evaluate PyTorch Lightning-trained models.
    
    This command evaluates models trained with PyTorch Lightning.
    Uses the same experiment config file as training, requiring resume_experiment_id
    to identify which experiment to evaluate.
    
    Examples:
        # Using config with resume_experiment_id
        python -m cli.test lightning configs/experiments/dlinear_solar.yaml
        
        # Override resume ID via CLI
        python -m cli.test lightning configs/experiments/dlinear_solar.yaml --resume-id 20240101-abc123
    """
    try:
        # First, check if this is a suite config or experiment config
        config_dict = load_yaml_config(config_path)
        
        # If it's a suite config, we need resume_experiment_id to load config from experiment directory
        if 'suite' in config_dict:
            # This is a suite config - need resume_experiment_id to find the experiment directory
            # Get resume_suite_id from CLI or config file
            suite_info = config_dict.get('suite', {})
            suite_id = resume_suite_id or suite_info.get('resume_suite_id') or suite_info.get('name', 'unknown')
            
            # For resume_experiment_id, user must specify which experiment to test
            # (suite configs have multiple experiments, so we need to know which one)
            if not resume_experiment_id:
                typer.echo(
                    "Error: resume_experiment_id is required when using a suite config.\n"
                    "  Provide it via CLI (--resume-id).\n"
                    "  Note: Each experiment in the suite has its own resume_experiment_id in the config file.",
                    err=True
                )
                raise typer.Exit(code=1)
            
            exp_id = resume_experiment_id
            
            # Determine experiment directory and load config from there
            output_path = Path(output_dir).resolve()
            if suite_id:
                experiment_dir = output_path / suite_id / exp_id
            else:
                experiment_dir = output_path / exp_id
            
            if not experiment_dir.exists():
                raise FileNotFoundError(
                    f"Experiment directory not found: {experiment_dir}\n"
                    f"  Experiment ID: {exp_id}\n"
                    f"  Suite ID: {suite_id if suite_id else 'N/A (standalone)'}"
                )
            
            # Load config from experiment directory (preferred - has actual training config)
            config = _load_experiment_config_from_directory(experiment_dir)
        else:
            # This is a regular experiment config - check for backward compatibility first
            config_hierarchy = load_config_with_nested(config_path)
            nested_configs = config_hierarchy.get('nested', {})
            
            # If evaluation section exists as nested config (old format), use backward compatibility
            if 'evaluation' in nested_configs or (hasattr(config_hierarchy['primary'], 'evaluation') and 
                                                  isinstance(config_hierarchy['primary'].evaluation, str)):
                # Old format: use nested evaluation config
                typer.echo("Warning: Using legacy evaluation config format. Consider migrating to resume_experiment_id.", err=True)
                config = config_hierarchy['primary']
                
                if dry_run:
                    typer.echo(f"✓ Config validated: {config_path}")
                    typer.echo(f"  Model: {config.model.name}")
                    typer.echo(f"  Data: {config.data.name}")
                    return
                
                from evaluation.standard import evaluate
                typer.echo(f"Starting standard evaluation with config: {config_path}")
                evaluate(config)
                return
            
            # New format: use experiment config with resume_experiment_id
            config = load_config(config_path)
            
            # Get resume_experiment_id from CLI override or config
            exp_id = resume_experiment_id or config.resume_experiment_id
            suite_id = resume_suite_id or config.resume_suite_id
        
        # Validate that resume_experiment_id is provided
        if not exp_id:
            typer.echo(
                "Error: resume_experiment_id is required for evaluation.\n"
                "  Provide it either in the config file (resume_experiment_id: '...') or via CLI (--resume-id).",
                err=True
            )
            raise typer.Exit(code=1)
        
        # Build evaluation config from experiment config
        eval_config = build_evaluation_config_from_experiment_config(
            config=config,
            resume_experiment_id=exp_id,
            resume_suite_id=suite_id,
            output_dir=output_dir,
            version=version,
            device_override=device,
            batch_size_override=batch_size
        )
        
        if dry_run:
            typer.echo(f"✓ Config validated: {config_path}")
            typer.echo(f"  Experiment ID: {exp_id}")
            if suite_id:
                typer.echo(f"  Suite ID: {suite_id}")
            typer.echo(f"  Model: {eval_config.model}")
            typer.echo(f"  Data: {eval_config.data}")
            typer.echo(f"  Task: {eval_config.task}")
            typer.echo(f"  Input/Output length: {eval_config.input_len}/{eval_config.output_len}")
            typer.echo(f"  Batch size: {eval_config.batch_size}")
            typer.echo(f"  Device: {eval_config.device}")
            typer.echo(f"  Checkpoint version: {eval_config.version}")
            return
        
        # Import here to avoid circular imports
        from evaluation.lightning import evaluate
        
        # Create config structure expected by evaluate function
        config_with_eval = dotdict({
            'evaluation': eval_config,
            'data_config': None  # Will use checkpoint's data config
        })
        
        typer.echo(f"Starting Lightning evaluation for experiment: {exp_id}")
        typer.echo(f"  Config: {config_path}")
        typer.echo(f"  Experiment directory: {eval_config.experiment_dir}")
        evaluate(config_with_eval)
        
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error during evaluation: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def llm(
    config_path: str = typer.Argument(..., help="Path to evaluation configuration file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without running evaluation")
):
    """
    Evaluate LLM-based forecasting results.
    
    This command evaluates predictions from LLM-based forecasting experiments.
    Reads JSON prediction files and calculates standardized metrics.
    
    Example:
        python -m cli.test llm configs/experiments/llm_solar.yaml
    """
    try:
        config_hierarchy = load_config_with_nested(config_path)
        config = config_hierarchy['primary']
        
        if dry_run:
            typer.echo(f"✓ Config validated: {config_path}")
            typer.echo(f"  Checkpoint: {config.evaluation.ckpt_id}")
            return
        
        # Import here to avoid circular imports
        from evaluation.llm import evaluate
        
        typer.echo(f"Starting LLM evaluation with config: {config_path}")
        evaluate(config)
        
    except FileNotFoundError as e:
        typer.echo(f"Error: Config file not found: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error during evaluation: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def suite(
    suite_config_path: str = typer.Argument(..., help="Path to suite configuration file"),
    resume_suite_id: Optional[str] = typer.Option(None, "--resume-suite-id", "-s", help="Suite ID to evaluate (overrides config.suite.resume_suite_id)"),
    output_dir: str = typer.Option("./output", "--output-dir", "-o", help="Base output directory"),
    eval_type: str = typer.Option("standard", "--eval-type", "-t", help="Evaluation type: 'standard' or 'lightning'"),
    version: str = typer.Option("best", "--version", "-v", help="Checkpoint version: 'best', 'latest', or specific pattern"),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="GPU device ID (overrides config.device.gpu)"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size (overrides config.training.batch_size)"),
    filter_experiments: Optional[str] = typer.Option(None, "--filter", "-f", help="Filter experiments by name pattern"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate and show which experiments would be evaluated without running"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error", help="Continue evaluating remaining experiments after an error")
):
    """
    Evaluate all experiments within a suite.
    
    This command evaluates all trained experiments in a suite that have valid
    resume_experiment_id values and checkpoints. Experiments without valid
    checkpoints are logged and skipped.
    
    The resume_suite_id can be provided via CLI (--resume-suite-id) or read from
    the suite config file (suite.resume_suite_id).
    
    Examples:
        # Evaluate suite using resume_suite_id from config
        python -m cli.test suite configs/experiment_suites/lynx_film_raw/time_mmd_ttc.yaml
        
        # Override resume_suite_id via CLI
        python -m cli.test suite configs/experiment_suites/lynx_film_raw/time_mmd_ttc.yaml --resume-suite-id lynx_film_raw_time_mmd_ttc_20260112_165034
        
        # Filter to specific experiments
        python -m cli.test suite configs/experiment_suites/lynx_film_raw/time_mmd_ttc.yaml --filter "climate"
        
        # Dry run to see which experiments would be evaluated
        python -m cli.test suite configs/experiment_suites/lynx_film_raw/time_mmd_ttc.yaml --dry-run
        
        # Use lightning evaluation
        python -m cli.test suite configs/experiment_suites/lynx_film_raw/time_mmd_ttc.yaml --eval-type lightning
    """
    try:
        # Validate eval_type
        if eval_type not in ['standard', 'lightning']:
            typer.echo(f"Error: Invalid eval_type '{eval_type}'. Must be 'standard' or 'lightning'.", err=True)
            raise typer.Exit(code=1)
        
        # Load suite config
        suite_config = load_yaml_config(suite_config_path)
        
        # Verify this is a suite config
        if 'suite' not in suite_config:
            typer.echo(
                f"Error: Not a suite config file: {suite_config_path}\n"
                f"  Expected 'suite' key in config. Use 'standard' or 'lightning' command for single experiments.",
                err=True
            )
            raise typer.Exit(code=1)
        
        suite_info = suite_config.get('suite', {})
        suite_name = suite_info.get('name', 'unknown')
        
        # Resolve resume_suite_id (CLI overrides config)
        resolved_suite_id = resume_suite_id or suite_info.get('resume_suite_id')
        
        if not resolved_suite_id:
            typer.echo(
                f"Error: resume_suite_id is required for suite evaluation.\n"
                f"  Provide it via CLI (--resume-suite-id) or in config file (suite.resume_suite_id).",
                err=True
            )
            raise typer.Exit(code=1)
        
        # Resolve output_dir (CLI overrides config execution.output_dir)
        # This matches how suite_executor resolves output_dir
        execution_config = suite_info.get('execution', {})
        if output_dir == "./output":  # Default value, check config
            output_dir = execution_config.get('output_dir', './output')
        
        # Extract experiments from suite
        experiments = _extract_experiments_from_suite(suite_config)
        
        # Apply filter if specified
        if filter_experiments:
            original_count = len(experiments)
            experiments = [
                exp for exp in experiments
                if filter_experiments.lower() in exp.get('name', '').lower()
            ]
            typer.echo(f"Filtered to {len(experiments)}/{original_count} experiments matching '{filter_experiments}'")
        
        if not experiments:
            typer.echo(f"No enabled experiments found in suite '{suite_name}'")
            return
        
        # Resolve output directory
        output_path = Path(output_dir).resolve()
        
        # Validate suite directory exists
        suite_dir = output_path / resolved_suite_id
        if not suite_dir.exists():
            typer.echo(
                f"Error: Suite directory not found: {suite_dir}\n"
                f"  Suite ID: {resolved_suite_id}\n"
                f"  Output dir: {output_path}",
                err=True
            )
            raise typer.Exit(code=1)
        
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Suite Evaluation: {suite_name}")
        typer.echo(f"Suite ID: {resolved_suite_id}")
        typer.echo(f"Suite Directory: {suite_dir}")
        typer.echo(f"Evaluation Type: {eval_type}")
        typer.echo(f"Total Experiments: {len(experiments)}")
        typer.echo(f"{'='*60}\n")
        
        # Categorize experiments by validity
        valid_experiments: List[Tuple[Dict[str, Any], str, Path]] = []  # (exp_config, resume_id, exp_dir)
        invalid_experiments: List[Tuple[Dict[str, Any], str]] = []  # (exp_config, reason)
        
        for exp in experiments:
            exp_name = exp.get('name', 'unknown')
            resume_id = _get_experiment_resume_id(exp)
            
            # Check 1: Has resume_experiment_id
            if not resume_id:
                invalid_experiments.append((exp, "Missing resume_experiment_id in config"))
                continue
            
            # Check 2: Validate checkpoint exists
            is_valid, reason, exp_dir = _validate_experiment_checkpoint(
                output_path, resolved_suite_id, resume_id
            )
            
            if is_valid:
                valid_experiments.append((exp, resume_id, exp_dir))
            else:
                invalid_experiments.append((exp, reason))
        
        # Report invalid experiments
        if invalid_experiments:
            typer.echo(f"⚠ {len(invalid_experiments)} experiment(s) cannot be evaluated:\n")
            for exp, reason in invalid_experiments:
                exp_name = exp.get('name', 'unknown')
                resume_id = _get_experiment_resume_id(exp)
                typer.echo(f"  ✗ {exp_name}")
                typer.echo(f"    Resume ID: {resume_id or 'N/A'}")
                typer.echo(f"    Reason: {reason}")
                typer.echo()
        
        # Report valid experiments
        if valid_experiments:
            typer.echo(f"✓ {len(valid_experiments)} experiment(s) ready for evaluation:\n")
            for exp, resume_id, exp_dir in valid_experiments:
                exp_name = exp.get('name', 'unknown')
                typer.echo(f"  ✓ {exp_name}")
                typer.echo(f"    Resume ID: {resume_id}")
                typer.echo(f"    Directory: {exp_dir}")
                typer.echo()
        else:
            typer.echo("No experiments with valid checkpoints found. Nothing to evaluate.")
            return
        
        # Summary before evaluation
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Summary: {len(valid_experiments)} valid, {len(invalid_experiments)} invalid")
        typer.echo(f"{'='*60}\n")
        
        # Dry run: stop here
        if dry_run:
            typer.echo("Dry run complete. No evaluations performed.")
            return
        
        # Execute evaluations
        success_count = 0
        error_count = 0
        
        for i, (exp, resume_id, exp_dir) in enumerate(valid_experiments, 1):
            exp_name = exp.get('name', 'unknown')
            
            typer.echo(f"\n[{i}/{len(valid_experiments)}] Evaluating: {exp_name}")
            typer.echo(f"  Resume ID: {resume_id}")
            typer.echo(f"  Directory: {exp_dir}")
            
            try:
                # Load experiment config from saved directory
                config = _load_experiment_config_from_directory(exp_dir)
                
                # Extract evaluation options from suite config if present
                # This allows adding evaluation options to the suite config for already-trained experiments
                exp_overrides = exp.get('overrides', {})
                evaluation_overrides = None
                if 'evaluation' in exp_overrides and exp_overrides['evaluation'] is not None:
                    evaluation_overrides = exp_overrides['evaluation']
                
                # Build evaluation config
                eval_config = build_evaluation_config_from_experiment_config(
                    config=config,
                    resume_experiment_id=resume_id,
                    resume_suite_id=resolved_suite_id,
                    output_dir=output_dir,
                    version=version,
                    device_override=device,
                    batch_size_override=batch_size,
                    evaluation_overrides=evaluation_overrides
                )
                
                # Create config structure expected by evaluate function
                config_with_eval = dotdict({
                    'evaluation': eval_config,
                    'data_config': None  # Will use checkpoint's data config
                })
                
                # Import and run appropriate evaluation
                if eval_type == 'lightning':
                    from evaluation.lightning import evaluate
                else:
                    from evaluation.standard import evaluate
                
                evaluate(config_with_eval)
                
                success_count += 1
                typer.echo(f"  ✓ Evaluation completed successfully")
                
            except Exception as e:
                error_count += 1
                typer.echo(f"  ✗ Evaluation failed: {e}", err=True)
                
                if not continue_on_error:
                    typer.echo("\nStopping suite evaluation due to error.", err=True)
                    raise typer.Exit(code=1)
        
        # Final summary
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Suite Evaluation Complete")
        typer.echo(f"  Successful: {success_count}")
        typer.echo(f"  Failed: {error_count}")
        typer.echo(f"  Skipped (invalid): {len(invalid_experiments)}")
        typer.echo(f"{'='*60}\n")
        
        if error_count > 0:
            raise typer.Exit(code=1)
        
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error during suite evaluation: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

