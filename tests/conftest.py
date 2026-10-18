"""Shared test setup. Every test runs offline: raw downloads and the cache go to a temp dir, and the
per-host throttle is off so the suite stays fast."""
from __future__ import annotations

from pathlib import Path

import pytest

from ipo_radar import config
from ipo_radar.live import _http

FIXTURES = Path(__file__).parent / "fixtures" / "live"


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@pytest.fixture
def live_isolation(tmp_path, monkeypatch):
    """Opt-in (live test modules use pytestmark): keeps other test areas' config untouched."""
    monkeypatch.setattr(config, "RAW", tmp_path / "raw")
    monkeypatch.setattr(config, "LIVE_CACHE", tmp_path / "cache")
    monkeypatch.setattr(config, "STATE_DB", tmp_path / "state.sqlite")
    monkeypatch.setattr(_http, "MIN_INTERVAL_S", 0.0)
    _http._last_request.clear()
    yield
