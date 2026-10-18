"""Paths and user config. Shared by every module; do not duplicate path logic elsewhere."""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

ROOT = Path(os.environ.get("IPO_RADAR_ROOT", Path(__file__).resolve().parents[2]))
RAW = ROOT / "data" / "raw"            # untouched downloads, one sub-folder per source
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
CONFIG_DIR = Path(os.environ.get("IPO_RADAR_CONFIG", Path.home() / ".config" / "ipo-radar"))
CONFIG_FILE = CONFIG_DIR / "config.toml"
STATE_DB = CONFIG_DIR / "state.sqlite"  # alert dedupe + GMP/subscription history
LIVE_CACHE = CONFIG_DIR / "live-cache"
USER_AGENT = "ipo-radar/0.1 (personal research tool; contact: local user)"

IPOS_PARQUET = PROCESSED / "ipos.parquet"
INDICES_PARQUET = PROCESSED / "indices.parquet"   # columns: date, symbol, close


def load_config() -> dict:
    """User config (notifier topic, thresholds). Missing file -> {}. Secrets live here, never in the repo."""
    if not CONFIG_FILE.exists():
        return {}
    with CONFIG_FILE.open("rb") as f:
        return tomllib.load(f)
