"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

import garmin_mcp.server as server_module
from garmin_mcp.cache import NightStore


@pytest.fixture(autouse=True)
def _isolated_hrv_store(tmp_path: Path) -> None:
    """Every test gets its own on-disk HRV night cache, never the real one."""
    server_module.set_hrv_store_for_testing(NightStore(tmp_path / "hrv_nights.json"))
