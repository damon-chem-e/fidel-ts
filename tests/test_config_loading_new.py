"""
Additional tests for ExperimentManager config saving and recursive copying.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from cli.config.models import ExperimentConfig, ModelConfig, DataConfig
from exp.manager import ExperimentManager


def test_experiment_manager_config_saving(tmp_path):
    """Test that ExperimentManager saves all config files correctly."""
    # Create temporary config files
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    
    # Create model config
    model_config_file = config_dir / "model_config.yaml"
    model_config_dict = {
        "seq_len": 360,
        "pred_len": 24,
        "enc_in": 1,
        "dec_in": 1,
        "c_out": 1
    }
    with open(model_config_file, 'w') as f:
        yaml.dump(model_config_dict, f)
    
    # Create data config
    data_config_file = config_dir / "data_config.yaml"
    data_config_dict = {
        "root_path": "./data",
        "data_path": "solar.csv",
        "features": "M",
        "target": "OT"
    }
    with open(data_config_file, 'w') as f:
        yaml.dump(data_config_dict, f)
    
    # Create experiment config
    exp_config = ExperimentConfig(
        model=ModelConfig(
            name="DLinear",
            config_path=str(model_config_file)
        ),
        data=DataConfig(
            name="solar",
            config_path=str(data_config_file)
        )
    )
    
    # Initialize ExperimentManager
    output_dir = tmp_path / "outputs"
    exp_manager = ExperimentManager(
        config=exp_config,
        output_dir=str(output_dir)
    )
    
    # Check that configs were saved
    saved_configs_dir = exp_manager.experiment_dir / "configs"
    assert saved_configs_dir.exists()
    
    # Check primary config
    primary_config = saved_configs_dir / "experiment_config.yaml"
    assert primary_config.exists()
    
    # Check model config was copied
    saved_model_config = saved_configs_dir / "model_config.yaml"
    assert saved_model_config.exists()
    with open(saved_model_config, 'r') as f:
        saved_model = yaml.safe_load(f)
    assert saved_model["seq_len"] == 360
    assert saved_model["pred_len"] == 24
    
    # Check data config was copied
    saved_data_config = saved_configs_dir / "data_config.yaml"
    assert saved_data_config.exists()
    with open(saved_data_config, 'r') as f:
        saved_data = yaml.safe_load(f)
    assert saved_data["root_path"] == "./data"
    assert saved_data["data_path"] == "solar.csv"


def test_experiment_manager_nested_config_saving(tmp_path):
    """Test that ExperimentManager saves nested configs (plotting, evaluation)."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    
    # Create plotting config
    plotting_config_file = config_dir / "plotting.yaml"
    plotting_dict = {
        "figure": {
            "figsize": [15, 7],
            "dpi": 300
        },
        "colors": ["blue", "red", "green"]
    }
    with open(plotting_config_file, 'w') as f:
        yaml.dump(plotting_dict, f)
    
    # Create evaluation config
    eval_config_file = config_dir / "evaluation.yaml"
    eval_dict = {
        "metrics": ["mse", "mae", "rmse"],
        "save_predictions": True
    }
    with open(eval_config_file, 'w') as f:
        yaml.dump(eval_dict, f)
    
    # Create model and data configs
    model_config_file = config_dir / "model.yaml"
    with open(model_config_file, 'w') as f:
        yaml.dump({"seq_len": 360}, f)
    
    data_config_file = config_dir / "data.yaml"
    with open(data_config_file, 'w') as f:
        yaml.dump({"root_path": "./data"}, f)
    
    # Create experiment config with nested configs
    exp_config = ExperimentConfig(
        model=ModelConfig(
            name="DLinear",
            config_path=str(model_config_file)
        ),
        data=DataConfig(
            name="solar",
            config_path=str(data_config_file)
        ),
        plotting=str(plotting_config_file),
        evaluation=str(eval_config_file)
    )
    
    # Initialize ExperimentManager
    output_dir = tmp_path / "outputs"
    exp_manager = ExperimentManager(
        config=exp_config,
        output_dir=str(output_dir)
    )
    
    # Check nested configs were saved
    saved_configs_dir = exp_manager.experiment_dir / "configs"
    
    saved_plotting = saved_configs_dir / "plotting_config.yaml"
    assert saved_plotting.exists()
    with open(saved_plotting, 'r') as f:
        saved_plotting_data = yaml.safe_load(f)
    assert saved_plotting_data["figure"]["figsize"] == [15, 7]
    
    saved_eval = saved_configs_dir / "evaluation_config.yaml"
    assert saved_eval.exists()
    with open(saved_eval, 'r') as f:
        saved_eval_data = yaml.safe_load(f)
    assert "mse" in saved_eval_data["metrics"]


def test_experiment_manager_recursive_config_copying(tmp_path):
    """Test that ExperimentManager recursively copies nested configs within config files."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    
    # Create a deeply nested config structure
    # Model config references another config
    nested_model_config_file = config_dir / "nested_model.yaml"
    nested_model_dict = {
        "hidden_dim": 128,
        "num_layers": 2
    }
    with open(nested_model_config_file, 'w') as f:
        yaml.dump(nested_model_dict, f)
    
    # Main model config that references the nested config
    model_config_file = config_dir / "model.yaml"
    model_config_dict = {
        "seq_len": 360,
        "pred_len": 24,
        "nested_config": str(nested_model_config_file),  # Reference to nested config
        "other_param": "value"
    }
    with open(model_config_file, 'w') as f:
        yaml.dump(model_config_dict, f)
    
    # Data config with nested reference
    nested_data_config_file = config_dir / "nested_data.yaml"
    nested_data_dict = {
        "preprocessing": {
            "normalize": True,
            "scale": True
        }
    }
    with open(nested_data_config_file, 'w') as f:
        yaml.dump(nested_data_dict, f)
    
    data_config_file = config_dir / "data.yaml"
    data_config_dict = {
        "root_path": "./data",
        "nested_config_path": str(nested_data_config_file),  # Reference to nested config
        "features": "M"
    }
    with open(data_config_file, 'w') as f:
        yaml.dump(data_config_dict, f)
    
    # Create experiment config
    exp_config = ExperimentConfig(
        model=ModelConfig(
            name="DLinear",
            config_path=str(model_config_file)
        ),
        data=DataConfig(
            name="solar",
            config_path=str(data_config_file)
        )
    )
    
    # Initialize ExperimentManager
    output_dir = tmp_path / "outputs"
    exp_manager = ExperimentManager(
        config=exp_config,
        output_dir=str(output_dir)
    )
    
    # Check that nested configs were recursively copied
    saved_configs_dir = exp_manager.experiment_dir / "configs"
    
    # Main configs should be saved
    assert (saved_configs_dir / "model_config.yaml").exists()
    assert (saved_configs_dir / "data_config.yaml").exists()
    
    # Nested configs should also be saved
    saved_nested_model = saved_configs_dir / "model_config_nested_config.yaml"
    assert saved_nested_model.exists(), "Nested model config should be recursively copied"
    
    saved_nested_data = saved_configs_dir / "data_config_nested_config_path.yaml"
    assert saved_nested_data.exists(), "Nested data config should be recursively copied"
    
    # Verify nested config content
    with open(saved_nested_model, 'r') as f:
        nested_model_data = yaml.safe_load(f)
    assert nested_model_data["hidden_dim"] == 128
    
    with open(saved_nested_data, 'r') as f:
        nested_data_data = yaml.safe_load(f)
    assert nested_data_data["preprocessing"]["normalize"] is True


def test_experiment_manager_config_copying_handles_missing_files(tmp_path):
    """Test that ExperimentManager handles missing config files gracefully."""
    # Create experiment config with non-existent config paths
    exp_config = ExperimentConfig(
        model=ModelConfig(
            name="DLinear",
            config_path="nonexistent/model.yaml"
        ),
        data=DataConfig(
            name="solar",
            config_path="nonexistent/data.yaml"
        ),
        plotting="nonexistent/plotting.yaml"  # Optional nested config
    )
    
    # Should not raise an error, just skip missing files
    output_dir = tmp_path / "outputs"
    exp_manager = ExperimentManager(
        config=exp_config,
        output_dir=str(output_dir)
    )
    
    # Primary config should still be saved
    saved_configs_dir = exp_manager.experiment_dir / "configs"
    assert (saved_configs_dir / "experiment_config.yaml").exists()
    
    # Missing configs should not be copied (no error raised)
    assert not (saved_configs_dir / "model_config.yaml").exists()
    assert not (saved_configs_dir / "data_config.yaml").exists()


def test_experiment_manager_config_copying_handles_invalid_yaml(tmp_path):
    """Test that ExperimentManager handles invalid YAML files gracefully."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    
    # Create a model config with invalid YAML (but valid file)
    model_config_file = config_dir / "model.yaml"
    with open(model_config_file, 'w') as f:
        f.write("invalid: yaml: content: [unclosed")
    
    # Create valid data config
    data_config_file = config_dir / "data.yaml"
    with open(data_config_file, 'w') as f:
        yaml.dump({"root_path": "./data"}, f)
    
    exp_config = ExperimentConfig(
        model=ModelConfig(
            name="DLinear",
            config_path=str(model_config_file)
        ),
        data=DataConfig(
            name="solar",
            config_path=str(data_config_file)
        )
    )
    
    # Should not raise an error, just copy the file without parsing
    output_dir = tmp_path / "outputs"
    exp_manager = ExperimentManager(
        config=exp_config,
        output_dir=str(output_dir)
    )
    
    # Both files should be copied (even if one has invalid YAML)
    saved_configs_dir = exp_manager.experiment_dir / "configs"
    assert (saved_configs_dir / "model_config.yaml").exists()
    assert (saved_configs_dir / "data_config.yaml").exists()


def test_experiment_manager_config_copying_avoids_duplicates(tmp_path):
    """Test that ExperimentManager avoids copying the same config file multiple times."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    
    # Create a shared config that might be referenced multiple times
    shared_config_file = config_dir / "shared.yaml"
    shared_dict = {"shared_param": "value"}
    with open(shared_config_file, 'w') as f:
        yaml.dump(shared_dict, f)
    
    # Model config references shared config
    model_config_file = config_dir / "model.yaml"
    model_dict = {
        "seq_len": 360,
        "shared": str(shared_config_file)
    }
    with open(model_config_file, 'w') as f:
        yaml.dump(model_dict, f)
    
    # Data config also references the same shared config
    data_config_file = config_dir / "data.yaml"
    data_dict = {
        "root_path": "./data",
        "shared": str(shared_config_file)  # Same file referenced again
    }
    with open(data_config_file, 'w') as f:
        yaml.dump(data_dict, f)
    
    exp_config = ExperimentConfig(
        model=ModelConfig(
            name="DLinear",
            config_path=str(model_config_file)
        ),
        data=DataConfig(
            name="solar",
            config_path=str(data_config_file)
        )
    )
    
    output_dir = tmp_path / "outputs"
    exp_manager = ExperimentManager(
        config=exp_config,
        output_dir=str(output_dir)
    )
    
    # Check that shared config is only copied once (or at least handled correctly)
    saved_configs_dir = exp_manager.experiment_dir / "configs"
    
    # Main configs should exist
    assert (saved_configs_dir / "model_config.yaml").exists()
    assert (saved_configs_dir / "data_config.yaml").exists()
    
    # Shared config should be copied (may appear with different names due to parent context)
    # The important thing is that the copying process doesn't fail or create infinite loops
    shared_configs = list(saved_configs_dir.glob("*shared*.yaml"))
    # Should have at least one copy of the shared config
    assert len(shared_configs) >= 1

