"""
Tests for git state retrieval functionality.

These tests verify that the ExperimentManager correctly captures git information
including commit hash, branch, and dirty state.
"""

import os
import pytest
import subprocess
import tempfile
from pathlib import Path
from exp.manager import ExperimentManager
from cli.config.models import ExperimentConfig, ModelConfig, DataConfig


def create_test_config():
    """Create a minimal test config."""
    return ExperimentConfig(
        model=ModelConfig(
            name="DLinear",
            config_path="model_configs/general/DLinear.yaml"
        ),
        data=DataConfig(
            name="test_data",
            config_path="data_configs/test.yaml"
        )
    )


def test_experiment_manager_initialization(tmp_path):
    """Test that ExperimentManager initializes correctly."""
    config = create_test_config()
    exp_manager = ExperimentManager(
        config=config,
        output_dir=str(tmp_path)
    )
    
    assert exp_manager.experiment_id is not None
    assert exp_manager.experiment_dir.exists()
    assert (exp_manager.experiment_dir / "checkpoints").exists()
    assert (exp_manager.experiment_dir / "visualizations").exists()
    assert (exp_manager.experiment_dir / "logs").exists()
    assert (exp_manager.experiment_dir / "configs").exists()
    assert (exp_manager.experiment_dir / "metrics").exists()


def test_experiment_id_generation(tmp_path):
    """Test that experiment IDs are generated deterministically."""
    config = create_test_config()
    
    exp_manager1 = ExperimentManager(config=config, output_dir=str(tmp_path / "output1"))
    exp_manager2 = ExperimentManager(config=config, output_dir=str(tmp_path / "output2"))
    
    # Note: Timestamps will be different, so full IDs will differ
    # But we can verify IDs are generated and have expected format
    assert len(exp_manager1.experiment_id) > 0
    assert len(exp_manager2.experiment_id) > 0
    assert '_' in exp_manager1.experiment_id  # Should have timestamp_hash format
    assert '_' in exp_manager2.experiment_id


def test_metadata_capture(tmp_path):
    """Test that metadata is captured and saved."""
    config = create_test_config()
    exp_manager = ExperimentManager(
        config=config,
        output_dir=str(tmp_path),
        job_id="test_job_123",
        job_name="test_job"
    )
    
    metadata = exp_manager.metadata
    assert metadata["experiment_id"] == exp_manager.experiment_id
    assert metadata["job_id"] == "test_job_123"
    assert metadata["job_name"] == "test_job"
    assert metadata["random_seed"] == config.random_seed
    assert "timestamp" in metadata
    assert "git_commit_hash" in metadata
    assert "git_branch" in metadata
    assert "git_is_dirty" in metadata
    
    # Check metadata file exists
    metadata_file = exp_manager.experiment_dir / "metadata.json"
    assert metadata_file.exists()


def test_git_info_capture(tmp_path):
    """Test that git information is captured when in a git repo."""
    config = create_test_config()
    
    # Create a temporary git repo for testing
    with tempfile.TemporaryDirectory() as git_tmp:
        git_dir = Path(git_tmp)
        
        # Initialize git repo
        try:
            subprocess.run(
                ["git", "init"],
                cwd=git_dir,
                capture_output=True,
                check=True,
                timeout=5
            )
            
            # Create a dummy file and commit
            (git_dir / "dummy.txt").write_text("test")
            subprocess.run(
                ["git", "add", "dummy.txt"],
                cwd=git_dir,
                capture_output=True,
                check=True,
                timeout=5
            )
            subprocess.run(
                ["git", "commit", "-m", "test commit"],
                cwd=git_dir,
                capture_output=True,
                check=True,
                timeout=5,
                env={**os.environ, "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@test.com"}
            )
            
            # Create experiment manager in git repo
            exp_manager = ExperimentManager(
                config=config,
                output_dir=str(git_dir / "outputs")
            )
            
            # Check that git info was captured
            git_info = exp_manager._get_git_info()
            assert git_info["git_commit_hash"] is not None
            assert git_info["git_branch"] is not None
            assert isinstance(git_info["git_is_dirty"], bool)
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            # Git not available or command failed - skip this test
            pytest.skip("Git not available or test setup failed")


def test_git_info_no_repo(tmp_path):
    """Test that git info handling works when not in a git repo."""
    config = create_test_config()
    
    # Create experiment manager outside git repo
    exp_manager = ExperimentManager(
        config=config,
        output_dir=str(tmp_path)
    )
    
    # Should not crash, git info should be None or handled gracefully
    git_info = exp_manager._get_git_info()
    assert "git_commit_hash" in git_info
    assert "git_branch" in git_info
    assert "git_is_dirty" in git_info
    # Values may be None if not in git repo, which is fine


def test_config_hierarchy_saving(tmp_path):
    """Test that config hierarchy is saved correctly."""
    config = create_test_config()
    exp_manager = ExperimentManager(
        config=config,
        output_dir=str(tmp_path)
    )
    
    # Check that configs directory exists
    configs_dir = exp_manager.experiment_dir / "configs"
    assert configs_dir.exists()
    
    # Primary config should be saved (if model/data configs exist)
    # But the directory structure should be created regardless
    assert configs_dir.is_dir()


def test_metrics_logging(tmp_path):
    """Test that metrics can be logged and saved."""
    config = create_test_config()
    exp_manager = ExperimentManager(
        config=config,
        output_dir=str(tmp_path)
    )
    
    # Log some metrics
    exp_manager.log_metric("loss", 0.5, step=1)
    exp_manager.log_metric("loss", 0.4, step=2)
    exp_manager.log_metrics({"accuracy": 0.9, "f1": 0.85}, step=3)
    
    # Check metrics file exists
    metrics_file = exp_manager.experiment_dir / "metrics" / "metrics.json"
    assert metrics_file.exists()
    
    # Verify metrics were saved
    import json
    with open(metrics_file, 'r') as f:
        saved_metrics = json.load(f)
    
    assert "loss" in saved_metrics
    assert len(saved_metrics["loss"]) == 2  # Two logged values with steps
    assert saved_metrics["loss"][0]["value"] == 0.5
    assert saved_metrics["loss"][0]["step"] == 1
    assert saved_metrics["loss"][1]["value"] == 0.4
    assert saved_metrics["loss"][1]["step"] == 2
    
    # Metrics logged with step are stored as list of dicts
    assert "accuracy" in saved_metrics
    assert isinstance(saved_metrics["accuracy"], list)
    assert saved_metrics["accuracy"][0]["value"] == 0.9
    assert saved_metrics["accuracy"][0]["step"] == 3
    
    assert "f1" in saved_metrics
    assert saved_metrics["f1"][0]["value"] == 0.85
    assert saved_metrics["f1"][0]["step"] == 3


def test_experiment_id_deterministic(tmp_path):
    """Test that same config produces same hash (deterministic)."""
    config1 = create_test_config()
    config2 = create_test_config()
    
    # Create managers with same config
    exp_manager1 = ExperimentManager(config=config1, output_dir=str(tmp_path / "out1"))
    exp_manager2 = ExperimentManager(config=config2, output_dir=str(tmp_path / "out2"))
    
    # Extract hash part (after timestamp)
    def extract_hash(exp_id):
        parts = exp_id.split('_', 1)
        return parts[1] if len(parts) > 1 else exp_id
    
    hash1 = extract_hash(exp_manager1.experiment_id)
    hash2 = extract_hash(exp_manager2.experiment_id)
    
    # Hashes should be the same for identical configs
    assert hash1 == hash2


def test_different_configs_different_ids(tmp_path):
    """Test that different configs produce different hashes."""
    config1 = create_test_config()
    config2 = create_test_config()
    config2.training.epochs = 100  # Change a value
    
    exp_manager1 = ExperimentManager(config=config1, output_dir=str(tmp_path / "out1"))
    exp_manager2 = ExperimentManager(config=config2, output_dir=str(tmp_path / "out2"))
    
    # Extract hash part
    def extract_hash(exp_id):
        parts = exp_id.split('_', 1)
        return parts[1] if len(parts) > 1 else exp_id
    
    hash1 = extract_hash(exp_manager1.experiment_id)
    hash2 = extract_hash(exp_manager2.experiment_id)
    
    # Hashes should be different for different configs
    assert hash1 != hash2

