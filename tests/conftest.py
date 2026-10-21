"""Shared fixtures: one synthetic snapshot set per test session, and the real data dir."""

from __future__ import annotations

from pathlib import Path

import pytest

from ga4app import config
from ga4app.synthetic import make_synthetic_snapshots

SYNTHETIC_SEED = 7


@pytest.fixture(scope="session")
def _synthetic_build(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("synthetic")
    truth = make_synthetic_snapshots(out, seed=SYNTHETIC_SEED)
    return out, truth


@pytest.fixture(scope="session")
def synthetic_dir(_synthetic_build: tuple[Path, dict]) -> Path:
    """A complete synthetic snapshot set (seed 7, 240 products). Treat it as read-only."""
    return _synthetic_build[0]


@pytest.fixture(scope="session")
def synthetic_truth(_synthetic_build: tuple[Path, dict]) -> dict:
    """The effects planted in ``synthetic_dir`` (see ``ga4app.synthetic``)."""
    return _synthetic_build[1]


@pytest.fixture(scope="session")
def real_data_dir() -> Path:
    """The real snapshot dir; skips the test until the pipeline has written its manifest."""
    if not (config.DEFAULT_DATA_DIR / "manifest.json").exists():
        pytest.skip("real snapshots not built (data/manifest.json missing)")
    return config.DEFAULT_DATA_DIR
