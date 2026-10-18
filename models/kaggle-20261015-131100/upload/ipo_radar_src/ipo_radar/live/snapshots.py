"""Our own subscription snapshots, one row per category per fetch, kept in the state DB.

Why: Chittorgarh publishes no day-wise table, and NSE gives only the current figures. The day-wise figures
for an open issue therefore exist only if we record them while it is open. merge fills sub_by_day from these
rows when the sites give nothing, marked sub_by_day_source 'own snapshot'.

Rules: a row is written only for a known value (unknown stays unknown, never 0). For each (IPO, day,
category) the latest reading of that day is the day's figure. Rows are not training data: nothing in the
historical model reads this table. Only the table below is created here; alerts/state.py is not edited.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path

from ipo_radar import config
from ipo_radar.live._util import issue_day_index
from ipo_radar.schema import LiveIPO

TABLE = "live_subscription_snapshots"
OWN_SOURCE = "own snapshot"
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    ipo_id TEXT NOT NULL,
    day TEXT NOT NULL,           -- ISO date the reading was taken on
    taken_at TEXT NOT NULL,      -- UTC ISO 8601
    category TEXT NOT NULL,      -- a SUB_COLS key, e.g. sub_retail
    value REAL NOT NULL,         -- times subscribed; unknown values are never stored
    source TEXT NOT NULL         -- where the value came from: nse | chittorgarh
);
CREATE INDEX IF NOT EXISTS {TABLE}_by_ipo ON {TABLE} (ipo_id, day, category, taken_at);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def record(ipos: list[LiveIPO], *, day: date, taken_at: datetime, db_path: Path | None = None) -> int:
    """Append one row per known subscription value of each IPO. Returns the number of rows written."""
    if taken_at.tzinfo is None:
        raise ValueError("taken_at must be timezone-aware")
    stamp = taken_at.astimezone(timezone.utc).isoformat()
    rows = [(ipo.ipo_id, day.isoformat(), stamp, key, reading.value, reading.source)
            for ipo in ipos for key, reading in ipo.sub.items() if reading.value is not None]
    if not rows:
        return 0
    with closing(_connect(db_path or config.STATE_DB)) as conn, conn:
        conn.executemany(f"INSERT INTO {TABLE} (ipo_id, day, taken_at, category, value, source) "
                         "VALUES (?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


def day_figures(ipos: list[LiveIPO], *, db_path: Path | None = None) -> dict[str, dict[int, dict[str, float]]]:
    """ipo_id -> {business day number: {category: value}}, the last reading of each day per category.

    Only days from the opening day on are kept. An IPO with no recorded rows is absent from the result.
    """
    out: dict[str, dict[int, dict[str, float]]] = {}
    path = db_path or config.STATE_DB
    if not Path(path).exists():
        return out
    with closing(_connect(path)) as conn:
        for ipo in ipos:
            if ipo.open_date is None:
                continue
            rows = conn.execute(
                f"SELECT day, category, value FROM {TABLE} WHERE ipo_id = ? ORDER BY taken_at, rowid",
                (ipo.ipo_id,)).fetchall()
            per_day: dict[int, dict[str, float]] = {}
            for day_text, category, value in rows:  # later rows overwrite earlier ones: last reading wins
                number = issue_day_index(ipo.open_date, date.fromisoformat(day_text))
                if number is not None:
                    per_day.setdefault(number, {})[category] = value
            if per_day:
                out[ipo.ipo_id] = per_day
    return out
