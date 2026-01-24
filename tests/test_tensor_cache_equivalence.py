"""
Tests to ensure tensor cache outputs match the baseline dataloader outputs.

These tests act as TDD specifications for the upcoming tensor cache module.
They exercise the suite CLI entrypoint and compare cached samples against
the existing Data_Provider output for identical configs and samples.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from cli import suite as suite_cli
from data_provider.data_factory import Data_Provider
from runs import suite_executor
from runs.pytorch import config_to_args

# NOTE: These imports define the expected public API for the new module.
from tensor_cache import TensorCacheDataset, TensorCacheGenerator, build_cache_config


def _write_csv_series(path: Path, rows: int, start_time: datetime) -> None:
    """Write a minimal time series CSV with a timestamp and a single value column."""
    # Step 1: Build row data with deterministic timestamps and values.
    data_rows = []
    for i in range(rows):
        timestamp = start_time + timedelta(hours=i)
        data_rows.append({"date": timestamp.isoformat(), "value": float(i)})
    # Step 2: Write the CSV header and rows to disk.
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "value"])
        writer.writeheader()
        writer.writerows(data_rows)


def _write_id_info(path: Path, entity_id: str) -> None:
    """Write a minimal id_info.json file mapping a single entity id."""
    # Step 1: Create a minimal mapping with a single entity id.
    payload = {entity_id: {}}
    # Step 2: Persist the mapping to disk as JSON.
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_yaml(path: Path, payload: dict) -> None:
    """Write a YAML file with the provided payload."""
    # Step 1: Open the target file for writing.
    with path.open("w", encoding="utf-8") as handle:
        # Step 2: Dump the payload in YAML format.
        yaml.safe_dump(payload, handle, sort_keys=False)


def _create_data_config(tmp_path: Path, entity_id: str) -> Path:
    """Create a minimal data config and dataset files for testing."""
    # Step 1: Create the dataset directory and CSV file.
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    csv_path = data_root / f"{entity_id}.csv"
    _write_csv_series(csv_path, rows=50, start_time=datetime(2024, 1, 1))
    # Step 2: Write id_info.json required by Data_Provider.
    _write_id_info(data_root / "id_info.json", entity_id)
    # Step 3: Build and persist the data config YAML.
    data_config_path = tmp_path / "data_config.yaml"
    data_config = {
        "root_path": str(data_root),
        "spliter": "ratio",
        "split_info": "7:1:2",
        "timestamp_col": "date",
        "target": "all",
        "id_info": "id_info.json",
        "id": [entity_id],
        "formatter": "{i}.csv",
        "sampling_rate": "1h",
        "base_T": 24,
        "time_zone": None,
        "downsample": None,
        "hetero_info": None,
    }
    _write_yaml(data_config_path, data_config)
    return data_config_path


def _create_template_config(
    tmp_path: Path, model_config_path: Path, data_config_path: Path, output_dir: Path
) -> Path:
    """Create a minimal experiment template config for suite execution."""
    # Step 1: Define a minimal training config for fast execution.
    template_config = {
        "model": {"name": "DLinear", "config_path": str(model_config_path)},
        "data": {"name": "synthetic", "config_path": str(data_config_path)},
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "input_len": 4,
            "output_len": 2,
            "num_workers": 0,
            "prefetch_factor": 1,
            "scale": True,
            "disable_buffer": True,
        },
        "device": {"use_gpu": False},
        "wandb": {"enabled": False, "mode": "disabled"},
        "experiment": {"type": "pytorch", "output_dir": str(output_dir)},
    }
    # Step 2: Persist the template to disk.
    template_path = tmp_path / "template.yaml"
    _write_yaml(template_path, template_config)
    return template_path


def _create_suite_config(tmp_path: Path, template_path: Path) -> Path:
    """Create a suite config pointing to the provided template."""
    # Step 1: Build suite config referencing the template.
    suite_config = {
        "suite": {
            "name": "tensor_cache_test_suite",
            "experiments": [
                {
                    "name": "tensor_cache_equivalence",
                    "enabled": True,
                    "template": str(template_path),
                    "overrides": {},
                }
            ],
        }
    }
    # Step 2: Persist the suite config to disk.
    suite_path = tmp_path / "suite.yaml"
    _write_yaml(suite_path, suite_config)
    return suite_path


class _DummyExpManager:
    """Lightweight ExperimentManager stub for config_to_args."""

    def __init__(self, checkpoint_dir: Path) -> None:
        # Step 1: Store the checkpoint directory path.
        self._checkpoint_dir = checkpoint_dir

    def get_checkpoint_dir(self) -> Path:
        """Return a guaranteed-existing checkpoint directory."""
        # Step 1: Ensure the directory exists.
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Step 2: Return the path for config_to_args.
        return self._checkpoint_dir


def _build_args(config, tmp_path: Path):
    """Convert an ExperimentConfig into args using a dummy manager."""
    # Step 1: Create a stub experiment manager for config_to_args.
    manager = _DummyExpManager(tmp_path / "checkpoints")
    # Step 2: Build args from the config.
    return config_to_args(config, manager)


def _build_val_loader(data_provider: Data_Provider, batch_size: int) -> DataLoader:
    """Build a deterministic validation dataloader with no shuffling."""
    # Step 1: Fetch validation datasets without a loader.
    datasets = data_provider.get_val(return_type="set")
    # Step 2: Concatenate datasets to mirror training behavior.
    concat_dataset = ConcatDataset([datasets[key] for key in datasets.keys()])
    # Step 3: Create a deterministic DataLoader.
    return DataLoader(
        concat_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


def _assert_tensor_equal(left: torch.Tensor, right: torch.Tensor) -> None:
    """Assert tensors match exactly in shape, dtype, and values."""
    # Step 1: Validate shapes and dtypes first for clear errors.
    assert left.shape == right.shape
    assert left.dtype == right.dtype
    # Step 2: Enforce exact equality within machine precision.
    torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def _assert_batch_equivalent(left_batch, right_batch) -> None:
    """Assert that two batches match element-by-element."""
    # Step 1: Ensure the tuple lengths match.
    assert len(left_batch) == len(right_batch)
    # Step 2: Compare each element with type-aware logic.
    for left_item, right_item in zip(left_batch, right_batch):
        if isinstance(left_item, torch.Tensor):
            _assert_tensor_equal(left_item, right_item)
        elif isinstance(left_item, np.ndarray):
            np.testing.assert_allclose(left_item, right_item, rtol=0.0, atol=0.0)
        else:
            assert left_item == right_item


def _install_baseline_stub(capture: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch run_pytorch to capture the baseline dataloader batch."""
    # Step 1: Define a stub that captures the first validation batch.
    def _run_pytorch_stub(config, suite_name=None, suite_info=None, output_dir=None, init_only=False):
        # Step 1: Convert config to args without invoking full ExperimentManager.
        args = _build_args(config, tmp_path)
        # Step 2: Build the baseline Data_Provider.
        data_provider = Data_Provider(args, buffer=(not args.disable_buffer))
        # Step 3: Capture the first deterministic validation batch.
        loader = _build_val_loader(data_provider, batch_size=args.batch_size)
        capture["baseline_batch"] = next(iter(loader))
        capture["config"] = config
    # Step 2: Monkeypatch the suite executor to use the stub.
    monkeypatch.setattr(suite_executor, "run_pytorch", _run_pytorch_stub)


def _build_cache_batch(config, tmp_path: Path):
    """Generate tensor cache for val split and return the first batch."""
    # Step 1: Build args and Data_Provider identical to baseline.
    args = _build_args(config, tmp_path)
    data_provider = Data_Provider(args, buffer=(not args.disable_buffer))
    # Step 2: Generate tensor cache using the new module API.
    cache_dir = tmp_path / "tensor_cache"
    cache_config = build_cache_config(args)
    TensorCacheGenerator(data_provider, cache_dir, cache_config, verbose=False).generate(flags=["val"])
    # Step 3: Load the cached dataset and create a deterministic loader.
    cache_dataset = TensorCacheDataset(cache_dir, flag="val")
    cache_loader = DataLoader(
        cache_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    # Step 4: Return the first batch for comparison.
    return next(iter(cache_loader))


def test_tensor_cache_matches_dataloader_output(tmp_path: Path, monkeypatch) -> None:
    """Ensure cached outputs match baseline dataloader outputs via suite CLI."""
    # Step 1: Seed RNGs for deterministic batching.
    torch.manual_seed(0)
    np.random.seed(0)
    # Step 2: Create minimal data/config assets in a temp directory.
    entity_id = "series_0"
    data_config_path = _create_data_config(tmp_path, entity_id)
    repo_root = Path(__file__).resolve().parents[1]
    model_config_path = repo_root / "model_configs" / "general" / "DLinear.yaml"
    template_path = _create_template_config(tmp_path, model_config_path, data_config_path, tmp_path / "outputs")
    suite_path = _create_suite_config(tmp_path, template_path)
    # Step 3: Capture baseline batch by running the suite entrypoint.
    capture = {}
    _install_baseline_stub(capture, tmp_path, monkeypatch)
    suite_cli.run(suite_config_path=str(suite_path), filter_experiments=None, dry_run=False, init_only=False)
    # Step 4: Build cache batch from the same config and compare.
    cache_batch = _build_cache_batch(capture["config"], tmp_path)
    _assert_batch_equivalent(capture["baseline_batch"], cache_batch)
