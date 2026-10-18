"""Offline tests for data/indices.py: file readers, the splice rules, the coverage report and failure paths."""
from __future__ import annotations

import pandas as pd
import pytest

from ipo_radar import config
from ipo_radar.data import indices
from ipo_radar.data.indices import Layout, Rung, splice
from ipo_radar.errors import SourceError


def _frame(dates, closes) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates), "close": list(map(float, closes))})


def _bdays(start: str, n: int) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=n)]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Kaggle folder, parquet output and ladder all point at tmp_path; nothing real is read or written."""
    monkeypatch.setattr(indices, "KAGGLE", tmp_path / "kaggle")
    monkeypatch.setattr(config, "INDICES_PARQUET", tmp_path / "out" / "indices.parquet")
    (tmp_path / "kaggle").mkdir()


def _write(tmp_path_rel: str, text: str) -> None:
    path = indices.KAGGLE / tmp_path_rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---- readers ---------------------------------------------------------------------------------------

def test_long_file_filters_symbol_and_drops_blank_closes(tmp_path):
    _write("long/data.csv",
           "date,symbol,close\n2010-01-04,^X,100\n2010-01-05,^X,\n2010-01-06,^Y,999\n2010-01-07,^X,101\n")
    rung = Rung("long X", "long", "data.csv", Layout("date", "close", symbol_col="symbol"), key="^X")
    out = indices._read_one(indices.KAGGLE / "long" / "data.csv", rung)
    assert out["close"].tolist() == [100.0, 101.0]          # blank close is missing, not zero; ^Y row excluded
    assert out["date"].is_monotonic_increasing


def test_custom_date_format_and_descending_file_are_sorted(tmp_path):
    _write("dmy/vix.csv", "Date,Close\n19 Mar 2026,22.8\n18 Mar 2026,18.7\n17 Mar 2026,19.8\n")
    rung = Rung("dmy", "dmy", "vix.csv", Layout("Date", "Close", date_format="%d %b %Y"))
    out = indices._read_one(indices.KAGGLE / "dmy" / "vix.csv", rung)
    assert out["date"].iloc[0] == pd.Timestamp("2026-03-17")
    assert out["close"].iloc[-1] == pytest.approx(22.8)


def test_missing_column_raises_naming_the_rung():
    _write("layout/x.csv", "DATE,PRICE\n2010-01-04,1\n")
    rung = Rung("layout X", "layout", "x.csv", Layout("date", "close"))
    with pytest.raises(SourceError, match="layout X.*layout changed"):
        indices._read_one(indices.KAGGLE / "layout" / "x.csv", rung)


def test_too_many_unreadable_rows_reject_the_file():
    rows = "\n".join(f"2010-01-{d:02d},oops" for d in range(1, 10))
    _write("bad/x.csv", "date,close\n" + rows + "\n")
    rung = Rung("bad", "bad", "x.csv", Layout("date", "close"))
    with pytest.raises(SourceError, match="unreadable"):
        indices._read_one(indices.KAGGLE / "bad" / "x.csv", rung)


def test_empty_file_is_a_source_error():
    _write("empty/x.csv", "")
    rung = Rung("empty", "empty", "x.csv", Layout("date", "close"))
    with pytest.raises(SourceError):
        indices._read_one(indices.KAGGLE / "empty" / "x.csv", rung)


def test_missing_folder_is_a_source_error_naming_the_rung():
    with pytest.raises(SourceError, match="ghost.*no file matching"):
        indices._load_rung(Rung("ghost", "nope", "x.csv", Layout("date", "close")))


def test_glob_pattern_concatenates_files_in_order():
    _write("many/a_2008.csv", "Date,Close\n2008-01-02,1\n2008-01-03,2\n")
    _write("many/b_2009.csv", "Date,Close\n2009-01-02,3\n")
    rung = Rung("many", "many", "*_20*.csv", Layout("Date", "Close"))
    out = indices._load_rung(rung)
    assert out["close"].tolist() == [1.0, 2.0, 3.0]


# ---- splice rules ----------------------------------------------------------------------------------

def _series(n: int = 40, start: str = "2010-01-04", base: float = 100.0) -> pd.DataFrame:
    dates = _bdays(start, n)
    return _frame(dates, [base + i for i in range(n)])


CAL = _bdays("2010-01-04", 90)


def _value(day: str) -> float:
    return 100.0 + CAL.index(day)           # one shared price history for every test series


def test_agreeing_rung_adds_only_missing_dates():
    primary = _frame(CAL[:40], [_value(d) for d in CAL[:40]])
    cand = _frame(CAL[20:70], [_value(d) for d in CAL[20:70]])        # 20 shared days, then 30 new
    out, used, notes = splice("T", [("primary", primary), ("cand", cand)])
    assert used == ["primary", "cand"]
    assert out["date"].is_unique
    assert set(out["date"]) == set(pd.to_datetime(CAL[:70]))
    assert any("agrees on 20 shared dates" in n for n in notes)


def test_gap_is_never_forward_filled():
    # Primary misses CAL[25:30]. The candidate has no rows there either, so the gap must stay a gap.
    primary = _frame(CAL[:25] + CAL[30:60], [_value(d) for d in CAL[:25] + CAL[30:60]])
    cand = _frame(CAL[:25] + CAL[60:70], [_value(d) for d in CAL[:25] + CAL[60:70]])
    out, _, _ = splice("T", [("primary", primary), ("cand", cand)])
    gap = set(pd.to_datetime(CAL[25:30]))
    assert gap.isdisjoint(set(out["date"]))
    assert len(out) == 25 + 30 + 10


def test_disagreement_above_limit_stops_the_splice():
    primary = _series(40)
    bad = primary.copy()
    bad.loc[10, "close"] = bad.loc[10, "close"] * 1.02           # 2% on one shared day
    later = _frame(_bdays("2010-03-12", 30), [500.0] * 30)        # would extend, but must not be reached
    out, used, notes = splice("T", [("primary", primary), ("bad", bad), ("later", later)])
    assert used == ["primary"]
    assert len(out) == len(primary)
    assert any("splicing stopped" in n for n in notes)


def test_difference_just_under_the_limit_is_accepted():
    primary = _series(40)
    close_ok = primary.copy()
    close_ok.loc[5, "close"] = close_ok.loc[5, "close"] * 1.004   # 0.4 %
    out, used, _ = splice("T", [("primary", primary), ("ok", close_ok)])
    assert used == ["primary", "ok"]


def test_difference_just_over_the_limit_stops():
    primary = _series(40)
    close_bad = primary.copy()
    close_bad.loc[5, "close"] = close_bad.loc[5, "close"] * 1.006  # 0.6 %
    _, used, _ = splice("T", [("primary", primary), ("bad", close_bad)])
    assert used == ["primary"]


def test_too_little_overlap_is_not_used_but_the_ladder_continues():
    primary = _frame(CAL[:40], [_value(d) for d in CAL[:40]])
    short = _frame(CAL[35:45], [999.0] * 10)                       # 5 shared days, below MIN_OVERLAP
    later = _frame(CAL[10:60], [_value(d) for d in CAL[10:60]])     # 30 shared days, agrees
    out, used, notes = splice("T", [("primary", primary), ("short", short), ("later", later)])
    assert "short" not in used
    assert any("not used" in n for n in notes)
    assert "later" in used
    assert 999.0 not in set(out["close"])


def test_single_rung_is_primary_only():
    primary = _series(30)
    out, used, notes = splice("T", [("primary", primary)])
    assert used == ["primary"] and len(out) == 30


def test_no_rungs_gives_empty_series():
    out, used, notes = splice("T", [])
    assert out.empty and used == []


# ---- build report ----------------------------------------------------------------------------------

def _two_symbol_ladder(tmp_path, monkeypatch):
    dates = _bdays("2004-12-20", 200)
    _write("aaa/a.csv", "date,close\n" + "\n".join(f"{d},{100 + i}" for i, d in enumerate(dates)) + "\n")
    monkeypatch.setattr(indices, "SYMBOLS", ("AAA", "BBB"))
    monkeypatch.setattr(indices, "LADDER", {
        "AAA": [Rung("aaa", "aaa", "a.csv", Layout("date", "close"))],
        "BBB": [Rung("bbb missing", "bbb", "b.csv", Layout("date", "close"))],
    })


def test_build_reports_every_symbol_and_writes_only_the_built_ones(tmp_path, monkeypatch):
    _two_symbol_ladder(tmp_path, monkeypatch)
    report = indices.build_report()
    by = {c.symbol: c for c in report.coverage}
    assert by["AAA"].status == "ok" and by["AAA"].sources == ("aaa",)
    assert by["BBB"].status == "no source" and any("no file matching" in n for n in by["BBB"].notes)
    stored = pd.read_parquet(config.INDICES_PARQUET)
    assert set(stored["symbol"]) == {"AAA"}
    assert list(stored.columns) == ["date", "symbol", "close"]


def test_rows_before_start_are_dropped_and_coverage_starts_at_start(tmp_path, monkeypatch):
    _two_symbol_ladder(tmp_path, monkeypatch)
    cov = {c.symbol: c for c in indices.build_report().coverage}["AAA"]
    assert cov.first >= indices.START
    assert cov.rows == len(pd.read_parquet(config.INDICES_PARQUET))


def test_raises_only_when_no_symbol_can_be_built(tmp_path, monkeypatch):
    monkeypatch.setattr(indices, "SYMBOLS", ("BBB",))
    monkeypatch.setattr(indices, "LADDER", {"BBB": [Rung("bbb missing", "bbb", "b.csv", Layout("date", "close"))]})
    with pytest.raises(SourceError, match="no index symbol could be built.*BBB"):
        indices.build_report()
    assert not config.INDICES_PARQUET.exists()


def test_build_indices_returns_the_frame_contract(tmp_path, monkeypatch):
    _two_symbol_ladder(tmp_path, monkeypatch)
    df = indices.build_indices()
    assert list(df.columns) == ["date", "symbol", "close"]
    assert set(df["symbol"]) == {"AAA"}
