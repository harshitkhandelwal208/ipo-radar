"""Offline tests for data/context.py using synthetic index calendars with known closes."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ipo_radar.data.context import add_context
from ipo_radar.schema import CONTEXT_COLS

PRE_OPEN = ["idx_ret_5d", "idx_ret_20d", "idx_vol_20d", "vix_level"]

NIFTY_HOLIDAY = pd.Timestamp("2024-01-15")   # Monday, absent from the NIFTY calendar
HSI_HOLIDAY = pd.Timestamp("2024-02-10")     # Saturday in this calendar; keeps the HSI calendar distinct


def _days(holidays=()) -> list[pd.Timestamp]:
    skip = set(pd.to_datetime(list(holidays)))
    return [d for d in pd.bdate_range("2023-11-01", "2024-03-29") if d not in skip]


NIFTY_DAYS = _days([NIFTY_HOLIDAY])
VIX_DAYS = _days([NIFTY_HOLIDAY])
US_DAYS = _days([NIFTY_HOLIDAY])
HSI_DAYS = _days([HSI_HOLIDAY])


def _closes(n: int, base: float, seed: int) -> np.ndarray:
    """Positive, non-monotonic closes so every return window is distinct."""
    rng = np.random.default_rng(seed)
    return base * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))


def _series(symbol: str, days, closes) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(days), "symbol": symbol, "close": closes})


def _indices(**overrides) -> pd.DataFrame:
    frames = {
        "NIFTY50": _series("NIFTY50", NIFTY_DAYS, _closes(len(NIFTY_DAYS), 22000, 1)),
        "INDIAVIX": _series("INDIAVIX", VIX_DAYS, _closes(len(VIX_DAYS), 14, 2)),
        "SP500": _series("SP500", US_DAYS, _closes(len(US_DAYS), 4500, 3)),
        "VIX": _series("VIX", US_DAYS, _closes(len(US_DAYS), 18, 4)),
        "HSI": _series("HSI", HSI_DAYS, _closes(len(HSI_DAYS), 17000, 5)),
    }
    frames.update(overrides)
    return pd.concat(frames.values(), ignore_index=True)


def _ipos(rows: list[tuple[str, object, object]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"ipo_id": f"id{i}", "market": m, "open_date": o, "close_date": c}
        for i, (m, o, c) in enumerate(rows)
    ])


def _ts(s: str) -> date:
    return pd.Timestamp(s).date()


def _pos_before(days, when: str) -> int:
    t = pd.Timestamp(when)
    return max(i for i, d in enumerate(days) if d < t)


def _closes_of(indices: pd.DataFrame, symbol: str) -> np.ndarray:
    return indices.loc[indices["symbol"] == symbol, "close"].to_numpy(float)


# ---- definitions, checked against values computed independently -----------------------------------

def test_pre_open_features_use_previous_trading_day():
    idx = _indices()
    out = add_context(_ipos([("IN_MAIN", _ts("2024-02-07"), _ts("2024-02-09"))]), idx)
    row = out.iloc[0]
    c = _closes_of(idx, "NIFTY50")
    i0 = _pos_before(NIFTY_DAYS, "2024-02-07")
    assert NIFTY_DAYS[i0] == pd.Timestamp("2024-02-06")
    assert row["idx_ret_5d"] == pytest.approx((c[i0] / c[i0 - 5] - 1) * 100)
    assert row["idx_ret_20d"] == pytest.approx((c[i0] / c[i0 - 20] - 1) * 100)
    logs = np.diff(np.log(c[i0 - 20 : i0 + 1]))
    assert row["idx_vol_20d"] == pytest.approx(np.std(logs, ddof=1) * np.sqrt(252) * 100)
    vix = idx.loc[idx.symbol == "INDIAVIX", "close"].to_numpy(float)
    j0 = _pos_before(VIX_DAYS, "2024-02-07")
    assert row["vix_level"] == pytest.approx(vix[j0])


def test_all_context_columns_are_filled_and_float():
    out = add_context(_ipos([("IN_MAIN", _ts("2024-02-07"), _ts("2024-02-09"))]), _indices())
    assert set(CONTEXT_COLS) <= set(out.columns)
    assert all(pd.api.types.is_float_dtype(out[c]) for c in CONTEXT_COLS)
    assert out[CONTEXT_COLS].notna().all(axis=None)


def test_weekend_open_uses_friday_close():
    idx = _indices()
    out = add_context(_ipos([
        ("IN_MAIN", _ts("2024-02-10"), _ts("2024-02-12")),   # Saturday
        ("IN_MAIN", _ts("2024-02-12"), _ts("2024-02-14")),   # Monday, last trading day before is Friday too
        ("IN_MAIN", _ts("2024-02-09"), _ts("2024-02-12")),   # Friday, last trading day before is Thursday
    ]), idx)
    sat, mon, fri = out.iloc[0], out.iloc[1], out.iloc[2]
    for col in PRE_OPEN:
        assert sat[col] == pytest.approx(mon[col]), col
    assert sat["idx_ret_5d"] != pytest.approx(fri["idx_ret_5d"])
    c = _closes_of(idx, "NIFTY50")
    f = _pos_before(NIFTY_DAYS, "2024-02-10")
    assert NIFTY_DAYS[f] == pd.Timestamp("2024-02-09")
    assert sat["idx_ret_5d"] == pytest.approx((c[f] / c[f - 5] - 1) * 100)


def test_holiday_open_uses_previous_trading_day():
    out = add_context(_ipos([
        ("IN_MAIN", _ts("2024-01-15"), _ts("2024-01-17")),   # exchange holiday (Monday)
        ("IN_MAIN", _ts("2024-01-16"), _ts("2024-01-18")),   # next day; last trading day before is also Friday 12th
    ]), _indices())
    holiday, next_day = out.iloc[0], out.iloc[1]
    for col in PRE_OPEN:
        assert holiday[col] == pytest.approx(next_day[col]), col


def test_us_uses_sp500_and_vix_hk_has_no_vix():
    idx = _indices()
    out = add_context(_ipos([
        ("US", _ts("2024-02-07"), _ts("2024-02-09")),
        ("HK", _ts("2024-02-07"), _ts("2024-02-09")),
    ]), idx)
    us, hk = out.iloc[0], out.iloc[1]
    sp = _closes_of(idx, "SP500")
    i0 = _pos_before(US_DAYS, "2024-02-07")
    assert us["idx_ret_5d"] == pytest.approx((sp[i0] / sp[i0 - 5] - 1) * 100)
    vix = _closes_of(idx, "VIX")
    assert us["vix_level"] == pytest.approx(vix[i0])
    hs = _closes_of(idx, "HSI")
    h0 = _pos_before(HSI_DAYS, "2024-02-07")
    assert hk["idx_ret_5d"] == pytest.approx((hs[h0] / hs[h0 - 5] - 1) * 100)
    assert np.isnan(hk["vix_level"])


def test_other_market_is_entirely_nan():
    out = add_context(_ipos([("OTHER", _ts("2024-02-07"), _ts("2024-02-09"))]), _indices())
    assert out[CONTEXT_COLS].isna().all(axis=None)


# ---- no look-ahead ---------------------------------------------------------------------------------

def _shift_from(indices: pd.DataFrame, symbol: str, start: str, factor: float) -> pd.DataFrame:
    idx = indices.copy()
    mask = (idx["symbol"] == symbol) & (idx["date"] >= pd.Timestamp(start))
    idx.loc[mask, "close"] = idx.loc[mask, "close"] * factor
    return idx


def test_changing_closes_on_or_after_open_does_not_change_pre_open_features():
    base = _indices()
    ipos = _ipos([("IN_MAIN", _ts("2024-02-07"), _ts("2024-02-09")),
                  ("US", _ts("2024-02-07"), _ts("2024-02-09"))])
    before = add_context(ipos, base)
    changed = base
    for sym in ("NIFTY50", "INDIAVIX", "SP500", "VIX"):
        changed = _shift_from(changed, sym, "2024-02-07", 1000.0)   # on open_date and every later day
    after = add_context(ipos, changed)
    for col in PRE_OPEN:
        pd.testing.assert_series_equal(before[col], after[col], check_names=False)


def test_issue_window_does_depend_on_the_issue_closes():
    # Documents that idx_ret_issue_window is the one feature that reads inside the issue window.
    base = _indices()
    ipos = _ipos([("IN_MAIN", _ts("2024-02-07"), _ts("2024-02-09"))])
    before = add_context(ipos, base)["idx_ret_issue_window"].iloc[0]
    after = add_context(ipos, _shift_from(base, "NIFTY50", "2024-02-09", 1.5))["idx_ret_issue_window"].iloc[0]
    assert before != pytest.approx(after)


def test_changing_the_day_before_open_does_change_features():
    # Sanity check that the look-ahead test above is not vacuous.
    base = _indices()
    ipos = _ipos([("IN_MAIN", _ts("2024-02-07"), _ts("2024-02-09"))])
    before = add_context(ipos, base)["idx_ret_5d"].iloc[0]
    tweaked = _shift_from(base, "NIFTY50", "2024-02-06", 1.2)
    after = add_context(ipos, tweaked)["idx_ret_5d"].iloc[0]
    assert before != pytest.approx(after)


# ---- issue window ----------------------------------------------------------------------------------

def test_issue_window_runs_from_pre_open_close_to_close_date():
    idx = _indices()
    out = add_context(_ipos([("IN_MAIN", _ts("2024-02-07"), _ts("2024-02-16"))]), idx)
    c = _closes_of(idx, "NIFTY50")
    i0 = _pos_before(NIFTY_DAYS, "2024-02-07")
    end = NIFTY_DAYS.index(pd.Timestamp("2024-02-16"))
    assert out["idx_ret_issue_window"].iloc[0] == pytest.approx((c[end] / c[i0] - 1) * 100)


def test_issue_window_close_on_holiday_uses_last_trading_day_before_it():
    idx = _indices()
    out = add_context(_ipos([("IN_MAIN", _ts("2024-01-10"), _ts("2024-01-15"))]), idx)
    c = _closes_of(idx, "NIFTY50")
    i0 = _pos_before(NIFTY_DAYS, "2024-01-10")          # Tue 9 Jan
    end = NIFTY_DAYS.index(pd.Timestamp("2024-01-12"))  # Fri 12 Jan, last trading day on or before the holiday
    assert out["idx_ret_issue_window"].iloc[0] == pytest.approx((c[end] / c[i0] - 1) * 100)


def test_issue_window_nan_when_no_trading_day_inside_it():
    out = add_context(_ipos([("IN_MAIN", _ts("2024-02-07"), _ts("2024-02-06"))]), _indices())
    assert np.isnan(out["idx_ret_issue_window"].iloc[0])


# ---- NaN handling ----------------------------------------------------------------------------------

def test_open_before_history_start_is_nan():
    out = add_context(_ipos([("IN_MAIN", _ts("2023-10-02"), _ts("2023-10-04"))]), _indices())
    assert out[CONTEXT_COLS].isna().all(axis=None)


def test_short_history_gives_nan_only_for_longer_windows():
    # Only 7 trading days of NIFTY history precede this open: 5-day return exists, 20-day and vol do not.
    out = add_context(_ipos([("IN_MAIN", _ts("2023-11-10"), _ts("2023-11-13"))]), _indices())
    row = out.iloc[0]
    assert np.isfinite(row["idx_ret_5d"])
    assert np.isnan(row["idx_ret_20d"])
    assert np.isnan(row["idx_vol_20d"])


def test_vix_before_its_start_is_nan_but_nifty_is_not():
    idx = _indices()
    idx = idx[~((idx["symbol"] == "INDIAVIX") & (idx["date"] < pd.Timestamp("2024-02-01")))]
    out = add_context(_ipos([("IN_MAIN", _ts("2024-01-20"), _ts("2024-01-22"))]), idx)
    assert np.isnan(out["vix_level"].iloc[0])
    assert np.isfinite(out["idx_ret_5d"].iloc[0])


def test_missing_and_unparseable_dates_are_nan():
    out = add_context(_ipos([
        ("IN_MAIN", None, _ts("2024-02-09")),
        ("IN_MAIN", _ts("2024-02-07"), None),
        ("IN_MAIN", "not-a-date", "2024-02-09"),
    ]), _indices())
    assert out.iloc[0][PRE_OPEN].isna().all()
    assert out.iloc[1][PRE_OPEN].notna().all()          # close date missing only affects the issue window
    assert np.isnan(out.iloc[1]["idx_ret_issue_window"])
    assert out.iloc[2][CONTEXT_COLS].isna().all()


def test_missing_benchmark_symbol_raises_instead_of_silent_nan():
    idx = _indices()
    idx = idx[idx["symbol"] != "HSI"]
    with pytest.raises(ValueError, match="HSI"):
        add_context(_ipos([("HK", _ts("2024-02-07"), _ts("2024-02-09"))]), idx)


def test_input_frame_is_not_modified():
    ipos = _ipos([("IN_MAIN", _ts("2024-02-07"), _ts("2024-02-09"))])
    snapshot = ipos.copy()
    add_context(ipos, _indices())
    pd.testing.assert_frame_equal(ipos, snapshot)


# ---- live_context ----------------------------------------------------------------------------------

from datetime import date as _date  # noqa: E402

from ipo_radar.data import context as ctx  # noqa: E402

LIVE_DAY = _date(2024, 2, 7)                  # Wednesday; last close before it is Tue 2024-02-06


@pytest.fixture
def live_env(monkeypatch):
    """Synthetic local series, NSE stubbed out. Tests change the stub or the series as needed."""
    idx = _indices()
    monkeypatch.setattr(ctx.index_data, "load_indices", lambda: idx)
    monkeypatch.setattr(ctx, "_nse_previous_closes", lambda: ({}, []))
    return idx


def _truncate(idx: pd.DataFrame, symbol: str, last: str) -> pd.DataFrame:
    keep = ~((idx["symbol"] == symbol) & (idx["date"] > pd.Timestamp(last)))
    return idx[keep].reset_index(drop=True)


def test_live_matches_add_context_for_an_ipo_opening_today(live_env):
    idx = live_env
    live = ctx.live_context("IN_MAIN", today=LIVE_DAY)
    hist = add_context(_ipos([("IN_MAIN", LIVE_DAY, _date(2024, 2, 9))]), idx).iloc[0]
    for col in ("idx_ret_5d", "idx_ret_20d", "idx_vol_20d", "vix_level"):
        assert live[col] == pytest.approx(hist[col]), col
    assert np.isnan(live["idx_ret_issue_window"])          # never known before the issue closes
    assert live["context_as_of"] == _date(2024, 2, 6)


def test_us_and_hk_live_match_add_context(live_env):
    idx = live_env
    for market in ("US", "HK"):
        live = ctx.live_context(market, today=LIVE_DAY)
        hist = add_context(_ipos([(market, LIVE_DAY, _date(2024, 2, 9))]), idx).iloc[0]
        for col in ("idx_ret_5d", "idx_ret_20d", "idx_vol_20d", "vix_level"):
            assert (np.isnan(live[col]) and np.isnan(hist[col])) or live[col] == pytest.approx(hist[col]), (market, col)


def test_todays_own_close_does_not_change_live_features(live_env, monkeypatch):
    base = live_env
    before = ctx.live_context("IN_MAIN", today=LIVE_DAY)
    changed = _shift_from(base, "NIFTY50", "2024-02-07", 10.0)
    monkeypatch.setattr(ctx.index_data, "load_indices", lambda: changed)
    after = ctx.live_context("IN_MAIN", today=LIVE_DAY)
    for col in ("idx_ret_5d", "idx_ret_20d", "idx_vol_20d", "vix_level"):
        assert before[col] == pytest.approx(after[col]), col


def test_nse_previous_close_is_appended_when_newer(live_env, monkeypatch):
    # Local NIFTY stops before 2024-03-22; the NSE dated close for 2024-03-28 must extend the series.
    idx = _truncate(live_env, "NIFTY50", "2024-03-21")
    monkeypatch.setattr(ctx.index_data, "load_indices", lambda: idx)
    local = idx.loc[idx["symbol"] == "NIFTY50", "close"].to_numpy(float)
    value = 23456.0
    monkeypatch.setattr(ctx, "_nse_previous_closes", lambda: ({"NIFTY50": (pd.Timestamp("2024-03-28"), value)}, []))
    live = ctx.live_context("IN_MAIN", today=_date(2024, 4, 1))
    assert live["context_as_of"] == _date(2024, 3, 28)
    closes = np.append(local, value)
    assert live["idx_ret_5d"] == pytest.approx((closes[-1] / closes[-6] - 1) * 100)
    assert any("NSE previous close 2024-03-28 appended" in n for n in live["context_notes"])


def test_nse_mismatch_on_a_shared_date_is_reported_not_overwritten(live_env, monkeypatch):
    local = live_env.loc[(live_env["symbol"] == "NIFTY50") & (live_env["date"] == pd.Timestamp("2024-03-29"))]
    v = float(local["close"].iloc[0])
    monkeypatch.setattr(ctx, "_nse_previous_closes", lambda: ({"NIFTY50": (pd.Timestamp("2024-03-29"), v * 1.02)}, []))
    live = ctx.live_context("IN_MAIN", today=_date(2024, 4, 1))
    assert any("differs" in n for n in live["context_notes"])
    assert live["idx_ret_5d"] == pytest.approx(
        ((live_env.loc[live_env["symbol"] == "NIFTY50", "close"].iloc[-1]) /
         live_env.loc[live_env["symbol"] == "NIFTY50", "close"].iloc[-6] - 1) * 100)


def test_nse_unavailable_falls_back_to_local_with_note(live_env, monkeypatch):
    monkeypatch.setattr(ctx, "_nse_previous_closes",
                        lambda: ({}, ["NSE allIndices unavailable, local series used: timeout"]))
    live = ctx.live_context("IN_MAIN", today=LIVE_DAY)
    assert np.isfinite(live["idx_ret_5d"]) and np.isfinite(live["vix_level"])
    assert any("NSE allIndices unavailable" in n for n in live["context_notes"])


def test_stale_benchmark_is_nan_with_note(live_env, monkeypatch):
    monkeypatch.setattr(ctx.index_data, "load_indices", lambda: _truncate(live_env, "NIFTY50", "2024-01-31"))
    live = ctx.live_context("IN_MAIN", today=_date(2024, 2, 12))
    assert np.isnan(live["idx_ret_5d"]) and np.isnan(live["idx_vol_20d"])
    assert any("NIFTY50 last close 2024-01-31 is older" in n for n in live["context_notes"])
    assert np.isfinite(live["vix_level"])                    # India VIX is fresh and still reported


@pytest.mark.parametrize("last, fresh", [("2024-02-05", True), ("2024-02-02", False)])
def test_staleness_boundary_is_five_weekdays(live_env, monkeypatch, last, fresh):
    # Today is Mon 2024-02-12. Last close Mon 02-05 is 5 weekdays back (fresh); Fri 02-02 is 6 (stale).
    monkeypatch.setattr(ctx.index_data, "load_indices", lambda: _truncate(live_env, "NIFTY50", last))
    live = ctx.live_context("IN_MAIN", today=_date(2024, 2, 12))
    assert np.isfinite(live["idx_ret_5d"]) == fresh


def test_us_stale_vix_is_nan_while_sp500_is_used(live_env, monkeypatch):
    monkeypatch.setattr(ctx.index_data, "load_indices", lambda: _truncate(live_env, "VIX", "2024-01-05"))
    live = ctx.live_context("US", today=LIVE_DAY)
    assert np.isnan(live["vix_level"])
    assert np.isfinite(live["idx_ret_5d"])
    assert any("VIX last close" in n for n in live["context_notes"])


def test_hk_has_no_vix_and_says_nothing_about_it(live_env):
    live = ctx.live_context("HK", today=LIVE_DAY)
    assert np.isnan(live["vix_level"])
    assert not any("VIX" in n for n in live["context_notes"])


def test_other_market_is_all_nan_with_note(live_env):
    live = ctx.live_context("OTHER", today=LIVE_DAY)
    assert all(np.isnan(live[c]) for c in CONTEXT_COLS)
    assert live["context_as_of"] is None and live["context_notes"]


def test_missing_parquet_is_all_nan_with_note(monkeypatch):
    monkeypatch.setattr(ctx.index_data, "load_indices", lambda: None)
    live = ctx.live_context("US", today=LIVE_DAY)
    assert all(np.isnan(live[c]) for c in CONTEXT_COLS)
    assert any("not built" in n for n in live["context_notes"])


def test_unknown_market_raises():
    with pytest.raises(ValueError, match="unknown market"):
        ctx.live_context("MARS", today=LIVE_DAY)


class _FakeClient:
    def close(self) -> None:
        pass


@pytest.fixture
def nse_stub(monkeypatch):
    """Replace the NSE session and JSON calls; tests set `payload` or `error`."""
    state: dict = {"payload": None, "error": None}
    monkeypatch.setattr(ctx._http, "make_client", lambda: _FakeClient())
    monkeypatch.setattr(ctx.nse, "_warm", lambda client: None)

    def fake_api(client, path):
        if state["error"] is not None:
            raise ctx.SourceError(state["error"])
        return state["payload"]

    monkeypatch.setattr(ctx.nse, "_api_json", fake_api)
    return state


def test_nse_parse_reads_dated_previous_close(nse_stub):
    nse_stub["payload"] = {"data": [
        {"index": "NIFTY 50", "previousDay": "14-Oct-2026", "previousDayVal": 22622.7},
        {"index": "INDIA VIX", "previousDay": "14-Oct-2026", "previousDayVal": 14.63},
    ]}
    points, notes = ctx._nse_previous_closes()
    assert notes == []
    assert points["NIFTY50"] == (pd.Timestamp("2026-10-14"), 22622.7)
    assert points["INDIAVIX"] == (pd.Timestamp("2026-10-14"), 14.63)


def test_nse_missing_row_and_bad_values_are_noted_not_guessed(nse_stub):
    nse_stub["payload"] = {"data": [
        {"index": "NIFTY 50", "previousDay": "not a date", "previousDayVal": 22622.7},
        {"index": "INDIA VIX", "previousDay": "14-Oct-2026", "previousDayVal": -1},
    ]}
    points, notes = ctx._nse_previous_closes()
    assert points == {}
    assert any("NIFTY 50 previous close unreadable" in n for n in notes)
    assert any("INDIA VIX previous close not positive" in n for n in notes)


def test_nse_absent_row_is_noted(nse_stub):
    nse_stub["payload"] = {"data": [{"index": "NIFTY BANK", "previousDay": "14-Oct-2026", "previousDayVal": 1.0}]}
    points, notes = ctx._nse_previous_closes()
    assert points == {} and len(notes) == 2 and all("has no" in n for n in notes)


def test_nse_error_returns_no_points_and_a_note(nse_stub):
    nse_stub["error"] = "nse: HTTP 503 for allIndices"
    points, notes = ctx._nse_previous_closes()
    assert points == {}
    assert notes and "NSE allIndices unavailable" in notes[0]


def test_nse_layout_change_is_noted(nse_stub):
    nse_stub["payload"] = {"rows": []}
    points, notes = ctx._nse_previous_closes()
    assert points == {} and "layout changed" in notes[0]
