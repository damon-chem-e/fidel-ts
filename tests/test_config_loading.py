"""
Tests for configuration loading functionality.

These tests verify that the Pydantic-based config loading system works correctly,
including validation, nested configs, error handling, and config saving/copying.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from cli.config.loader import load_config, load_config_with_nested, resolve_config_path
from cli.config.models import ExperimentConfig, ModelConfig, DataConfig
from exp.manager import ExperimentManager
from exp.manager import ExperimentManager


def test_resolve_config_path_absolute():
    """Test resolving absolute paths."""
    abs_path = Path("/absolute/path/config.yaml")
    resolved = resolve_config_path(str(abs_path))
    assert resolved == abs_path


def test_resolve_config_path_relative():
    """Test resolving relative paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        config_path = "subdir/config.yaml"
        resolved = resolve_config_path(config_path, base_dir)
        assert resolved == base_dir / config_path


def test_experiment_config_creation():
    """Test creating ExperimentConfig from dict."""
    config_dict = {
        "model": {
            "name": "DLinear",
            "config_path": "model_configs/general/DLinear.yaml"
        },
        "data": {
            "name": "solar",
            "config_path": "data_configs/fullsolar.yaml"
        },
        "training": {
            "epochs": 20,
            "batch_size": 96,
            "learning_rate": 5e-4
        },
        "device": {
            "use_gpu": True,
            "gpu": 0
        }
    }
    
    config = ExperimentConfig(**config_dict)
    assert config.model.name == "DLinear"
    assert config.data.name == "solar"
    assert config.training.epochs == 20
    assert config.device.use_gpu is True


def test_experiment_config_defaults():
    """Test that default values are applied correctly."""
    config_dict = {
        "model": {
            "name": "DLinear",
            "config_path": "model_configs/general/DLinear.yaml"
        },
        "data": {
            "name": "solar",
            "config_path": "data_configs/fullsolar.yaml"
        }
    }
    
    config = ExperimentConfig(**config_dict)
    # Check defaults
    assert config.training.epochs == 20
    assert config.training.batch_size == 96
    assert config.training.learning_rate == 5e-4
    assert config.device.use_gpu is True
    assert config.random_seed == 2021


def test_experiment_config_validation():
    """Test that validation catches invalid values."""
    config_dict = {
        "model": {
            "name": "DLinear",
            "config_path": "model_configs/general/DLinear.yaml"
        },
        "data": {
            "name": "solar",
            "config_path": "data_configs/fullsolar.yaml"
        },
        "training": {
            "epochs": -1,  # Invalid: should be >= 1
            "batch_size": 96
        }
    }
    
    with pytest.raises(Exception):  # Pydantic validation error
        ExperimentConfig(**config_dict)


def test_load_config_from_yaml(tmp_path):
    """Test loading config from YAML file."""
    config_file = tmp_path / "test_config.yaml"
    config_dict = {
        "model": {
            "name": "DLinear",
            "config_path": "model_configs/general/DLinear.yaml"
        },
        "data": {
            "name": "solar",
            "config_path": "data_configs/fullsolar.yaml"
        },
        "training": {
            "epochs": 10,
            "batch_size": 32
        }
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(config_dict, f)
    
    config = load_config(str(config_file))
    assert isinstance(config, ExperimentConfig)
    assert config.model.name == "DLinear"
    assert config.training.epochs == 10
    assert config.training.batch_size == 32


def test_load_config_with_nested(tmp_path):
    """Test loading config with nested subconfigs."""
    # Create primary config
    primary_config = tmp_path / "primary.yaml"
    primary_dict = {
        "model": {
            "name": "DLinear",
            "config_path": "model_configs/general/DLinear.yaml"
        },
        "data": {
            "name": "solar",
            "config_path": "data_configs/fullsolar.yaml"
        },
        "plotting": "plotting_config.yaml"
    }
    
    with open(primary_config, 'w') as f:
        yaml.dump(primary_dict, f)
    
    # Create nested plotting config
    plotting_config = tmp_path / "plotting_config.yaml"
    plotting_dict = {
        "figure": {
            "figsize": [15, 7],
            "dpi": 300
        }
    }
    
    with open(plotting_config, 'w') as f:
        yaml.dump(plotting_dict, f)
    
    # Load with nested configs
    result = load_config_with_nested(str(primary_config), base_dir=tmp_path)
    
    assert 'primary' in result
    assert 'nested' in result
    assert 'config_paths' in result
    assert isinstance(result['primary'], ExperimentConfig)
    assert 'plotting' in result['nested']
    assert result['nested']['plotting']['figure']['figsize'] == [15, 7]


def test_config_to_yaml_roundtrip(tmp_path):
    """Test saving and loading config maintains values."""
    config_dict = {
        "model": {
            "name": "TGTSF",
            "config_path": "model_configs/general/TGTSF.yaml"
        },
        "data": {
            "name": "solar",
            "config_path": "data_configs/fullsolar.yaml"
        },
        "training": {
            "epochs": 15,
            "batch_size": 64,
            "ahead": "day"
        },
        "random_seed": 42
    }
    
    config = ExperimentConfig(**config_dict)
    output_file = tmp_path / "output.yaml"
    config.to_yaml(output_file)
    
    # Reload and verify
    reloaded = ExperimentConfig.from_yaml(output_file)
    assert reloaded.model.name == config.model.name
    assert reloaded.training.epochs == config.training.epochs
    assert reloaded.training.batch_size == config.training.batch_size
    assert reloaded.random_seed == config.random_seed


def test_config_missing_required_field():
    """Test that missing required fields raise errors."""
    config_dict = {
        "model": {
            "name": "DLinear"
            # Missing config_path
        },
        "data": {
            "name": "solar",
            "config_path": "data_configs/fullsolar.yaml"
        }
    }
    
    with pytest.raises(Exception):  # Pydantic validation error
        ExperimentConfig(**config_dict)

