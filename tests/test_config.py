"""Tests for config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from epistemic_tribunal.config import TribunalSettings, load_config


def test_default_config_loads() -> None:
    """load_config() should succeed with the bundled default.yaml."""
    cfg = load_config()
    assert isinstance(cfg, TribunalSettings)


def test_default_weights_positive() -> None:
    cfg = load_config()
    w = cfg.tribunal.weights
    assert w.uncertainty > 0
    assert w.critic > 0
    assert w.memory > 0
    assert w.invariant > 0


def test_default_thresholds() -> None:
    cfg = load_config()
    assert 0.0 < cfg.tribunal.selection_threshold <= 1.0
    assert 0.0 < cfg.tribunal.resample_threshold <= 1.0
    assert cfg.tribunal.resample_threshold <= cfg.tribunal.selection_threshold


def test_generator_list_nonempty() -> None:
    cfg = load_config()
    assert len(cfg.generators.enabled) > 0


def test_load_custom_yaml(tmp_path: Path) -> None:
    """Custom YAML file should override individual values."""
    yaml_content = """
tribunal:
  selection_threshold: 0.77
  resample_threshold: 0.33
generators:
  enabled: [greedy]
  seed: 99
"""
    yaml_file = tmp_path / "custom.yaml"
    yaml_file.write_text(yaml_content)
    cfg = load_config(yaml_file)
    assert cfg.tribunal.selection_threshold == pytest.approx(0.77)
    assert cfg.tribunal.resample_threshold == pytest.approx(0.33)
    assert cfg.generators.enabled == ["greedy"]
    assert cfg.generators.seed == 99


def test_load_missing_file_uses_defaults(tmp_path: Path) -> None:
    """A non-existent config path should fall back to defaults."""
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert isinstance(cfg, TribunalSettings)


def test_env_override_ledger_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRIBUNAL_LEDGER_PATH", "/tmp/test_ledger.db")
    cfg = load_config()
    assert cfg.ledger.path == "/tmp/test_ledger.db"
