"""State tests: dedupe and snapshots must survive a restart, and prune must keep the newest snapshot."""
from __future__ import annotations

import subprocess
import sys

import pytest

from ipo_radar.alerts.notify import Alert
from ipo_radar.alerts.state import State


def _alert(key: str = "ipo1:opens-today:2026-10-15") -> Alert:
    return Alert(key=key, title="Acme opens today", body="x")


def test_mark_sent_then_already_sent(tmp_path):
    with State(tmp_path / "s.sqlite") as st:
        assert st.already_sent("ipo1:opens-today:2026-10-15") is False
        st.mark_sent(_alert())
        st.mark_sent(_alert())  # second mark is a no-op, not an error
        assert st.already_sent("ipo1:opens-today:2026-10-15") is True


def test_dedupe_persists_across_processes(tmp_path):
    db = tmp_path / "s.sqlite"
    with State(db) as st:
        st.mark_sent(_alert("ipo9:closes-today:2026-10-15"))
    code = (
        "import sys; from pathlib import Path; from ipo_radar.alerts.state import State;"
        "st = State(Path(sys.argv[1])); print(st.already_sent('ipo9:closes-today:2026-10-15'));"
        "print(st.already_sent('ipo9:other'))"
    )
    out = subprocess.run([sys.executable, "-c", code, str(db)], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["True", "False"]


def test_wal_mode_is_enabled(tmp_path):
    with State(tmp_path / "s.sqlite") as st:
        assert st._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_snapshots_return_latest_per_ipo(tmp_path):
    with State(tmp_path / "s.sqlite") as st:
        assert st.last_snapshot("ipo1") is None
        st.save_snapshot("ipo1", {"gmp": 10.0, "sub": {"sub_total": 0.5}})
        st.save_snapshot("ipo1", {"gmp": 25.0, "sub": {"sub_total": 1.2}})
        st.save_snapshot("ipo2", {"gmp": None, "sub": {}})
        assert st.last_snapshot("ipo1") == {"gmp": 25.0, "sub": {"sub_total": 1.2}}
        assert st.last_snapshot("ipo2") == {"gmp": None, "sub": {}}


def test_save_snapshot_rejects_nan_instead_of_storing_invalid_json(tmp_path):
    with State(tmp_path / "s.sqlite") as st:
        with pytest.raises(ValueError):
            st.save_snapshot("ipo1", {"gmp": float("nan")})
        assert st.last_snapshot("ipo1") is None


def test_prune_drops_old_rows_but_keeps_newest_snapshot_per_ipo(tmp_path):
    with State(tmp_path / "s.sqlite") as st:
        st.mark_sent(_alert("old"))
        st.mark_sent(_alert("new"))
        st.save_snapshot("ipo1", {"gmp": 1.0})
        st.save_snapshot("ipo1", {"gmp": 2.0})
        st.save_snapshot("ipo2", {"gmp": 3.0})
        ancient = "2000-01-01T00:00:00+00:00"
        with st._conn:
            st._conn.execute("UPDATE sent_alerts SET sent_at = ? WHERE key = 'old'", (ancient,))
            st._conn.execute("UPDATE snapshots SET taken_at = ?", (ancient,))

        removed = st.prune(days=30)

        assert removed == 2  # one sent mark and ipo1's older snapshot; ipo2's only snapshot is kept
        assert st.already_sent("old") is False
        assert st.already_sent("new") is True
        assert st.last_snapshot("ipo1") == {"gmp": 2.0}
        assert st.last_snapshot("ipo2") == {"gmp": 3.0}
