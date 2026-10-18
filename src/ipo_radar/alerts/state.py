"""SQLite state for alerts: which events were already sent, and the last observed numbers per IPO.

WAL mode lets the 15-minute timer and an interactive `ipo fetch` read and write at the same time. Dedupe must
survive process restarts, which is why it lives here and not in memory.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Any

from ipo_radar import config
from ipo_radar.alerts.notify import Alert

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sent_alerts (
    key TEXT PRIMARY KEY,
    sent_at TEXT NOT NULL,
    title TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ipo_id TEXT NOT NULL,
    taken_at TEXT NOT NULL,
    json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS snapshots_by_ipo ON snapshots (ipo_id, id);
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class State:
    """Connection to the alert state database. Use as a context manager or call close()."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or config.STATE_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> State:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def already_sent(self, key: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM sent_alerts WHERE key = ?", (key,)).fetchone()
        return row is not None

    def mark_sent(self, alert: Alert) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO sent_alerts (key, sent_at, title) VALUES (?, ?, ?)",
                (alert.key, _now_iso(), alert.title),
            )

    def last_snapshot(self, ipo_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT json FROM snapshots WHERE ipo_id = ? ORDER BY id DESC LIMIT 1", (ipo_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def save_snapshot(self, ipo_id: str, data: dict[str, Any]) -> None:
        # allow_nan=False: a NaN here would be stored as invalid JSON and break the next comparison.
        payload = json.dumps(data, allow_nan=False, sort_keys=True)
        with self._conn:
            self._conn.execute(
                "INSERT INTO snapshots (ipo_id, taken_at, json) VALUES (?, ?, ?)", (ipo_id, _now_iso(), payload)
            )

    def prune(self, days: int) -> int:
        """Drop sent-alert marks and snapshots older than `days`. The newest snapshot per IPO is always kept,
        because the next run compares against it. Returns the number of rows removed."""
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")
        with self._conn:
            sent = self._conn.execute("DELETE FROM sent_alerts WHERE sent_at < ?", (cutoff,)).rowcount
            snaps = self._conn.execute(
                "DELETE FROM snapshots WHERE taken_at < ? AND id NOT IN "
                "(SELECT MAX(id) FROM snapshots GROUP BY ipo_id)",
                (cutoff,),
            ).rowcount
        return sent + snaps
