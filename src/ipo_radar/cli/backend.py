"""The only module that reads the network, the live cache, the history file, the alert state or the optional
model package. Commands reach it through `service`; tests replace these functions with fakes.

A failure becomes a value (an error string, a status, a ModelUnavailable message) rather than a crash, so the
terminal always has something honest to show.
"""
from __future__ import annotations

import importlib
import re
import sqlite3
import subprocess
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ipo_radar import config
from ipo_radar.alerts import daemon, notify
from ipo_radar.alerts.notify import Alert, Notifier
from ipo_radar.cli.analysis import SourceStatus
from ipo_radar.cli.model_rules import unused_stages
from ipo_radar.errors import SourceError
from ipo_radar.live import health as live_health
from ipo_radar.live import merge
from ipo_radar.schema import LiveIPO, Prediction

LIVE_MAX_AGE_S = 300   # `ipo live` and `ipo show` reuse the merged cache up to this age, then fetch again
NOT_TRAINED = "model not trained yet: run `ipo train`"
SOURCES = ("nse", "chittorgarh", "gmp")
TIMER_NAME = f"{daemon.SERVICE_NAME}.timer"
_UNAVAILABLE = re.compile(r"^(nse|chittorgarh|GMP) unavailable: (.*)$")


class ModelUnavailable(Exception):
    """The model cannot predict: not trained yet, or it failed to load. The message is shown as is."""


class HistoryMissing(Exception):
    """data/processed/ipos.parquet has not been built yet."""


@dataclass
class LiveResult:
    ipos: list[LiveIPO]
    health: list[SourceStatus]
    error: str | None = None   # set only when no IPO data could be fetched at all


@dataclass
class Predictor:
    predict: Callable[[LiveIPO], Prediction]
    version: str | None
    trained_at: str | None = None   # ISO timestamp from the bundle
    n_rows: int | None = None       # IPOs the bundle was trained on
    not_used: dict[str, str] = field(default_factory=dict)  # stage -> 'worse than the median' caveat reason


def health_from(ipos: list[LiveIPO]) -> list[SourceStatus]:
    """Source status from the warnings this fetch attached. A source is down when any IPO says so.

    With an empty list there is nothing to check against, so the status is 'unknown', not 'ok'.
    """
    down: dict[str, str] = {}
    for ipo in ipos:
        for warning in ipo.warnings:
            match = _UNAVAILABLE.match(warning)
            if match:
                name = "gmp" if match.group(1) == "GMP" else match.group(1)
                down.setdefault(name, match.group(2))
    out = []
    for name in SOURCES:
        if name in down:
            out.append(SourceStatus(name, "down", down[name]))
        elif ipos:
            out.append(SourceStatus(name, "ok"))
        else:
            out.append(SourceStatus(name, "unknown", "no IPOs listed to check against"))
    return out


def load_live(refresh: bool) -> LiveResult:
    """refresh=True goes to the network. refresh=False reuses the merged cache while it is a few minutes old."""
    try:
        ipos = merge.fetch_all(use_cache=not refresh, ttl_seconds=LIVE_MAX_AGE_S)
    except SourceError as exc:
        return LiveResult([], [], str(exc))
    return LiveResult(ipos, health_from(ipos))


def predictor() -> Predictor:
    """The model's predict function, or ModelUnavailable. The model package is written separately, so its
    absence is normal and reported as 'not trained yet'; any other import or load error is reported as such."""
    try:  # import_module, not `from ... import`: the latter reports a missing submodule with the parent's name
        model_predict = importlib.import_module("ipo_radar.model.predict")
    except ImportError as exc:
        missing = isinstance(exc, ModuleNotFoundError) and exc.name == "ipo_radar.model.predict"
        message = NOT_TRAINED if missing else f"model failed to load: {type(exc).__name__}"
        raise ModelUnavailable(message) from exc
    try:
        loaded = model_predict.load_model()
    except FileNotFoundError as exc:
        raise ModelUnavailable(NOT_TRAINED) from exc
    except Exception as exc:
        raise ModelUnavailable(f"model failed to load: {type(exc).__name__}: {exc}") from exc
    if loaded is None:
        raise ModelUnavailable(NOT_TRAINED)
    version = getattr(loaded, "version", None)
    trained_at = getattr(loaded, "trained_at", None)
    n_rows = getattr(loaded, "n_rows", None)
    return Predictor(model_predict.predict, version if isinstance(version, str) else None,
                     trained_at if isinstance(trained_at, str) else None,
                     n_rows if isinstance(n_rows, int) else None, unused_stages(loaded))


def model_files() -> list[Path]:
    return sorted(config.MODELS.glob("*.joblib")) if config.MODELS.exists() else []


def history_mtime() -> float | None:
    path = config.IPOS_PARQUET
    return path.stat().st_mtime if path.exists() else None


def load_history() -> pd.DataFrame:
    path = config.IPOS_PARQUET
    if not path.exists():
        raise HistoryMissing(str(path))
    return pd.read_parquet(path)


def notifiers() -> list[Notifier]:
    return notify.build_notifiers()


def alert_cycle(ipos: list[LiveIPO], predict_fn: Callable[[LiveIPO], Prediction | None],
                channels: list[Notifier]) -> daemon.RunResult:
    """One alert cycle over data already fetched: evaluate, send, record. Same rules as `ipo alerts run-once`."""
    return daemon.run_once(lambda: ipos, predict_fn, notifiers=channels)  # type: ignore[arg-type]


def send_test(channels: list[Notifier], alert: Alert) -> list[notify.DispatchResult]:
    return notify.dispatch([alert], channels)


def save_alerts(*, topic: str | None, ntfy_server: str | None, telegram_bot_token: str | None,
                telegram_chat_id: str | int | None) -> str | None:
    """Write the alert settings to the user config (mode 600). Returns the ntfy topic in effect."""
    return notify.save_alert_config(topic, ntfy_server=ntfy_server, telegram_bot_token=telegram_bot_token,
                                    telegram_chat_id=telegram_chat_id)


def recent_alerts(limit: int = 10) -> list[tuple[str, str]] | None:
    """Last alerts marked sent, newest first. None when the state file cannot be read."""
    if not config.STATE_DB.exists():
        return []
    try:
        with closing(sqlite3.connect(f"file:{config.STATE_DB}?mode=ro", uri=True)) as conn:
            rows = conn.execute(
                "SELECT sent_at, title FROM sent_alerts ORDER BY sent_at DESC LIMIT ?", (limit,)
            ).fetchall()
    except sqlite3.OperationalError as exc:
        # The alert tables are created on the first alert run; the file may exist earlier (live snapshots).
        return [] if "no such table" in str(exc) else None
    except sqlite3.Error:
        return None
    return [(str(sent), str(title)) for sent, title in rows]


def check_sources() -> list[live_health.SourceHealth]:
    return live_health.check_sources()


def systemd_dir() -> Path:
    return Path.home() / ".config" / "systemd" / "user"


def systemctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, timeout=60, check=False)
