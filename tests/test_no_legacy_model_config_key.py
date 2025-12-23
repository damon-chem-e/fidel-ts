"""
Regression tests to prevent reintroducing the legacy 'model_config' key.

Why:
- In Pydantic v2, 'model_config' is a reserved attribute used for model configuration.
- This repo previously used a user-facing YAML key named 'model_config' for model-YAML overrides.
- That legacy behavior is now removed. Users must use 'model_config_overrides'.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cli.config.models import ExperimentConfig


def _iter_yaml_files(repo_root: Path) -> list[Path]:
    """
    Collect YAML files that represent user-facing configs.

    We intentionally scan:
    - configs/ (templates, experiments, suites)
    - model_configs/ (model YAMLs)
    - data_configs/ (data YAMLs)
    """
    roots = ["configs", "model_configs", "data_configs"]
    files: list[Path] = []
    for rel in roots:
        base = repo_root / rel
        if not base.exists():
            continue
        files.extend(sorted(base.rglob("*.yaml")))
        files.extend(sorted(base.rglob("*.yml")))
    return files


def _contains_key_recursively(obj, key: str) -> bool:
    """Return True if `key` appears anywhere in a nested dict/list structure."""
    if isinstance(obj, dict):
        if key in obj:
            return True
        return any(_contains_key_recursively(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_key_recursively(v, key) for v in obj)
    return False


def test_repo_yamls_do_not_contain_model_config_key():
    """
    Ensure repo YAMLs do not contain a 'model_config' key anywhere.

    This is intentionally strict: we want to keep 'model_config' reserved for Pydantic internals.
    """
    repo_root = Path(__file__).resolve().parents[1]
    yaml_files = _iter_yaml_files(repo_root)
    assert yaml_files, "Expected to find YAML configs under configs/, model_configs/, or data_configs/"

    offenders: list[Path] = []
    for path in yaml_files:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if _contains_key_recursively(data, "model_config"):
            offenders.append(path)

    assert offenders == [], f"Found forbidden key 'model_config' in: {offenders}"


def test_experiment_config_rejects_legacy_model_config_key():
    """ExperimentConfig should hard-fail on top-level 'model_config'."""
    bad = {
        "model": {"name": "TGTSF", "config_path": "model_configs/general/TGTSF.yaml"},
        "data": {"name": "weather", "config_path": "data_configs/weather/weather.yaml"},
        "model_config": {"input_text_dim": 768},
    }
    with pytest.raises(Exception):
        ExperimentConfig(**bad)


