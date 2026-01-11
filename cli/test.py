"""
Testing/Evaluation command-line interface for Fidel-TS.

This module provides CLI commands for evaluating trained models:
- Standard evaluation (for PyTorch-trained models)
- Lightning evaluation (for Lightning-trained models)
- LLM evaluation (for LLM-based forecasting results)
"""

import typer
from pathlib import Path
from typing import Optional
from cli.config.loader import load_config_with_nested, load_config
from cli.config.models import ExperimentConfig
from evaluation.config_builder import build_evaluation_config_from_experiment_config
from utils.tools import dotdict

app = typer.Typer(
    name="test",
    help="Evaluate trained time series forecasting models",
    add_completion=False
)


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
        # Check for backward compatibility (old evaluation config format)
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
        # Check for backward compatibility (old evaluation config format)
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
            
            from evaluation.lightning import evaluate
            typer.echo(f"Starting Lightning evaluation with config: {config_path}")
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


if __name__ == "__main__":
    app()

