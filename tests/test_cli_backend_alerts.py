"""recent_alerts must tell "no alerts yet" apart from "state file unreadable"."""
import sqlite3

from ipo_radar import config
from ipo_radar.cli import backend


def test_missing_file_means_no_alerts(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DB", tmp_path / "state.sqlite")
    assert backend.recent_alerts() == []


def test_file_without_alert_tables_means_no_alerts(tmp_path, monkeypatch):
    db = tmp_path / "state.sqlite"
    sqlite3.connect(db).execute("CREATE TABLE live_subscription_snapshots (x)").connection.commit()
    monkeypatch.setattr(config, "STATE_DB", db)
    assert backend.recent_alerts() == []


def test_garbage_file_is_unreadable(tmp_path, monkeypatch):
    db = tmp_path / "state.sqlite"
    db.write_bytes(b"not a database" * 100)
    monkeypatch.setattr(config, "STATE_DB", db)
    assert backend.recent_alerts() is None


def test_returns_newest_first(tmp_path, monkeypatch):
    db = tmp_path / "state.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE sent_alerts (key TEXT PRIMARY KEY, sent_at TEXT, title TEXT)")
    conn.executemany("INSERT INTO sent_alerts VALUES (?,?,?)", [("a", "2026-10-14T10:00", "old"), ("b", "2026-10-15T10:00", "new")])
    conn.commit()
    conn.close()
    monkeypatch.setattr(config, "STATE_DB", db)
    assert backend.recent_alerts() == [("2026-10-15T10:00", "new"), ("2026-10-14T10:00", "old")]
