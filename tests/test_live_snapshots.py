"""Own subscription snapshots: the store, the day mapping, the merge fill, and a two-day fetch_all run."""
import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

from ipo_radar import config
from ipo_radar.live import merge, snapshots
from ipo_radar.schema import LiveIPO, Sourced

pytestmark = pytest.mark.usefixtures("live_isolation")
OPEN = date(2026, 10, 13)  # Tuesday: day 1 = 13 Oct, day 2 = 14 Oct, day 3 = 15 Oct
NOW = datetime(2026, 10, 14, 17, 30, tzinfo=timezone.utc)


def ipo(sub=None, **kw) -> LiveIPO:
    base = dict(ipo_id="hd fire protect", market="IN_MAIN", company="HD Fire Protect", open_date=OPEN,
                close_date=date(2026, 10, 15), sources=["nse"])
    base.update(kw)
    rec = LiveIPO(**base)
    rec.sub = sub or {}
    return rec


def val(v, source="nse", at=NOW):
    return Sourced(v, source, at)


def test_day_figures_keep_the_last_reading_of_each_day_and_skip_unknowns(tmp_path):
    db = tmp_path / "s.sqlite"
    snapshots.record([ipo({"sub_total": val(3.0), "sub_retail": val(1.0)})], day=date(2026, 10, 13),
                     taken_at=NOW, db_path=db)
    snapshots.record([ipo({"sub_total": val(5.0)})], day=date(2026, 10, 13),
                     taken_at=NOW + timedelta(hours=2), db_path=db)  # later reading of day 1 wins
    snapshots.record([ipo({"sub_total": val(None)})], day=date(2026, 10, 14), taken_at=NOW, db_path=db)  # unknown: not stored
    figures = snapshots.day_figures([ipo()], db_path=db)["hd fire protect"]
    assert figures == {1: {"sub_total": 5.0, "sub_retail": 1.0}}


def test_days_are_business_days_from_the_opening_day(tmp_path):
    db = tmp_path / "s.sqlite"
    snapshots.record([ipo({"sub_total": val(2.0)})], day=date(2026, 10, 16), taken_at=NOW, db_path=db)  # Friday
    snapshots.record([ipo({"sub_total": val(4.0)})], day=date(2026, 10, 19), taken_at=NOW, db_path=db)  # Monday
    figures = snapshots.day_figures([ipo(open_date=date(2026, 10, 15))], db_path=db)["hd fire protect"]
    assert figures == {2: {"sub_total": 2.0}, 3: {"sub_total": 4.0}}  # Thu 15 Oct, Fri 16 Oct, Mon 19 Oct


def test_unknown_ipo_or_missing_database_gives_nothing(tmp_path):
    assert snapshots.day_figures([ipo()], db_path=tmp_path / "none.sqlite") == {}
    db = tmp_path / "s.sqlite"
    snapshots.record([ipo({"sub_total": val(1.0)})], day=OPEN, taken_at=NOW, db_path=db)
    assert snapshots.day_figures([ipo(ipo_id="other")], db_path=db) == {}


def test_naive_timestamp_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        snapshots.record([ipo({"sub_total": val(1.0)})], day=OPEN, taken_at=datetime(2026, 10, 13, 12),
                         db_path=tmp_path / "s.sqlite")


def test_store_uses_its_own_table_and_leaves_alert_state_alone(tmp_path):
    db = tmp_path / "s.sqlite"
    snapshots.record([ipo({"sub_total": val(1.0)})], day=OPEN, taken_at=NOW, db_path=db)
    with sqlite3.connect(db) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert snapshots.TABLE in names and "snapshots" not in names


def test_default_database_is_the_state_db(tmp_path, monkeypatch):
    target = tmp_path / "state-here.sqlite"
    monkeypatch.setattr(config, "STATE_DB", target)
    assert snapshots.record([ipo({"sub_total": val(1.0)})], day=OPEN, taken_at=NOW) == 1
    assert target.exists()


# ---- merge: the fill never overwrites a site's day and is marked ---------------------------------------
def test_own_snapshots_fill_only_days_the_sites_left_empty_and_are_marked():
    rec = ipo({"sub_total": val(3.0)})
    rec.sub_by_day = {3: {"sub_total": 9.0}}
    rec.sub_by_day_source = "chittorgarh"
    merge.apply_own_snapshots([rec], {"hd fire protect": {1: {"sub_total": 1.4}, 3: {"sub_total": 99.0}}})
    assert rec.sub_by_day == {1: {"sub_total": 1.4}, 3: {"sub_total": 9.0}}  # day 3 kept from the site
    assert rec.sub_by_day_source == "chittorgarh; own snapshot"


def test_no_own_rows_leaves_sub_by_day_empty_and_unmarked():
    rec = ipo({"sub_total": val(3.0)})
    merge.apply_own_snapshots([rec], {})
    assert rec.sub_by_day == {} and rec.sub_by_day_source == ""


def test_own_snapshots_are_marked_as_own_source_only():
    rec = ipo()
    merge.apply_own_snapshots([rec], {"hd fire protect": {2: {"sub_total": 4.0}}})
    assert rec.sub_by_day_source == "own snapshot"


# ---- fetch_all over two real days (sources patched, store real) ----------------------------------------
def _nse_row(total):
    rec = ipo({"sub_total": val(total, "nse", NOW)}, sources=["nse"])
    return [rec]


def test_fetch_all_on_two_days_fills_day_one_from_the_first_run(monkeypatch):
    state = {"total": 3.0, "day": date(2026, 10, 13)}
    monkeypatch.setattr(merge.nse, "fetch", lambda: _nse_row(state["total"]))
    monkeypatch.setattr(merge.chittorgarh, "fetch", lambda: [])
    monkeypatch.setattr(merge, "_run_gmp", lambda: (None, "GMP module not available (test)"))
    monkeypatch.setattr(merge, "_today", lambda: state["day"])
    first = merge.fetch_all(use_cache=False)
    assert first[0].sub_by_day == {1: {"sub_total": 3.0}}  # run on day 1: day 1 is its own snapshot
    assert first[0].sub_by_day_source == "own snapshot"
    state.update(total=6.0, day=date(2026, 10, 14))
    second = merge.fetch_all(use_cache=False)
    assert second[0].sub_by_day == {1: {"sub_total": 3.0}, 2: {"sub_total": 6.0}}


def test_fetch_all_day_one_figure_is_the_last_run_of_that_day(monkeypatch):
    state = {"total": 1.0}
    monkeypatch.setattr(merge.nse, "fetch", lambda: _nse_row(state["total"]))
    monkeypatch.setattr(merge.chittorgarh, "fetch", lambda: [])
    monkeypatch.setattr(merge, "_run_gmp", lambda: (None, "GMP module not available (test)"))
    monkeypatch.setattr(merge, "_today", lambda: date(2026, 10, 13))
    merge.fetch_all(use_cache=False)
    state["total"] = 2.5
    out = merge.fetch_all(use_cache=False)
    assert out[0].sub_by_day == {1: {"sub_total": 2.5}}
