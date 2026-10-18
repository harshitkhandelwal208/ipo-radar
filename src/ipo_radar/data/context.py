"""Market-regime features, from the last trading day STRICTLY BEFORE the date of interest (no look-ahead).

Benchmarks (SPEC): IN_MAIN and IN_SME -> NIFTY50 with INDIAVIX, US -> SP500 with VIX, HK -> HSI with no
volatility series (vix_level NaN), OTHER -> none. Each lookup uses the benchmark's own trading calendar, so a
weekend or holiday date maps to the previous trading day.

Definitions:
- idx_ret_5d / idx_ret_20d: percent change of the close over the trailing 5 / 20 trading days.
- idx_vol_20d: sample stdev of the last 20 daily log returns, annualised with sqrt(252), in percent.
- vix_level: the volatility index close on its last trading day before the date.
- idx_ret_issue_window: percent change from the last close before open_date to the last close on or before
  close_date. It uses data from inside the issue and is known only at close_date, so it is EXCLUDED from any
  pre-open prediction. live_context never fills it.

NaN is written (documented, never guessed) when: open_date or close_date is missing or unparseable; open_date is
before the index history starts; the trailing window is longer than the history before open_date; the market
has no benchmark or no volatility series. A benchmark symbol missing from the indices frame raises ValueError,
because that is a broken input and not a missing value.

NYSE_COMP and NASDAQ_COMP are not in schema.CONTEXT_COLS, so they are not filled here.

Live (live_context): the same features as of today. The benchmark's last close must be at most
LIVE_MAX_STALE_WEEKDAYS weekdays old, or its features are NaN with a note. For NIFTY 50 and INDIA VIX the
previous close from NSE allIndices (a dated close, not the intraday quote) is appended to the local series when it
is newer. NSE is optional: when it cannot be reached, the local series is used alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ipo_radar.data import indices as index_data
from ipo_radar.errors import SourceError
from ipo_radar.live import _http, nse
from ipo_radar.schema import CONTEXT_COLS, MARKETS

BENCHMARKS: dict[str, tuple[str | None, str | None]] = {
    "IN_MAIN": ("NIFTY50", "INDIAVIX"),
    "IN_SME": ("NIFTY50", "INDIAVIX"),
    "US": ("SP500", "VIX"),
    "HK": ("HSI", None),
    "OTHER": (None, None),
}
assert set(BENCHMARKS) == set(MARKETS)

TRADING_DAYS = 252
LIVE_MAX_STALE_WEEKDAYS = 5          # weekdays, not exchange holidays: a holiday counts as a day, so this is stricter
NSE_NAMES = {"NIFTY50": "NIFTY 50", "INDIAVIX": "INDIA VIX"}   # NSE allIndices row names
NSE_MISMATCH = 0.005                 # relative gap that is reported when NSE and the local close disagree on a date


@dataclass(frozen=True)
class _Calendar:
    dates: np.ndarray   # datetime64[ns], strictly increasing
    close: np.ndarray   # float, all > 0


def _calendar_from_frame(frame: pd.DataFrame) -> _Calendar:
    rows = frame[["date", "close"]].dropna()
    rows = rows[rows["close"] > 0].sort_values("date").drop_duplicates("date", keep="last")
    if rows.empty:
        raise ValueError("no usable rows for the benchmark")
    dates = pd.to_datetime(rows["date"]).to_numpy("datetime64[ns]")
    return _Calendar(dates, rows["close"].to_numpy(float))


def _calendar(indices: pd.DataFrame, symbol: str) -> _Calendar:
    rows = indices.loc[indices["symbol"] == symbol]
    if rows.empty:
        raise ValueError(f"indices frame has no rows for {symbol}")
    return _calendar_from_frame(rows)


def _last_before(cal: _Calendar, when: np.ndarray) -> np.ndarray:
    """Position of the last trading day strictly before each date; -1 if none or the date is NaT."""
    pos = np.searchsorted(cal.dates, when, side="left") - 1
    return np.where(np.isnat(when), -1, pos)


def _last_on_or_before(cal: _Calendar, when: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(cal.dates, when, side="right") - 1
    return np.where(np.isnat(when), -1, pos)


def _trailing(cal: _Calendar, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ret_5d, ret_20d (percent) and vol_20d (annualised percent) at each position; NaN where history is short."""
    close = cal.close
    rets = []
    for k in (5, 20):
        ret = np.full(len(pos), np.nan)
        ok = pos >= k
        ret[ok] = (close[pos[ok]] / close[pos[ok] - k] - 1.0) * 100.0
        rets.append(ret)
    log_ret = np.full(len(close), np.nan)
    log_ret[1:] = np.diff(np.log(close))
    vol = np.full(len(pos), np.nan)
    ok = pos >= 20
    if ok.any():
        # Window j holds log_ret[j:j+20]; the 20 returns ending at position p start at p-19 (always >= 1 here).
        windows = sliding_window_view(log_ret, 20)
        vol[ok] = windows[pos[ok] - 19].std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS) * 100.0
    return rets[0], rets[1], vol


def _issue_window(cal: _Calendar, pos_pre: np.ndarray, close_when: np.ndarray) -> np.ndarray:
    """Percent change from the pre-open close to the close on or before close_date. NaN unless a trading day lies in it."""
    pos_close = _last_on_or_before(cal, close_when)
    ok = (pos_pre >= 0) & (pos_close > pos_pre)
    out = np.full(len(pos_pre), np.nan)
    out[ok] = (cal.close[pos_close[ok]] / cal.close[pos_pre[ok]] - 1.0) * 100.0
    return out


def add_context(ipos: pd.DataFrame, indices: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ipos with CONTEXT_COLS filled from the index closes. The input is not modified."""
    out = ipos.copy()
    for col in CONTEXT_COLS:
        out[col] = np.nan
    open_when = pd.to_datetime(out["open_date"], errors="coerce").to_numpy("datetime64[ns]")
    close_when = pd.to_datetime(out["close_date"], errors="coerce").to_numpy("datetime64[ns]")
    for market, (bench, vol) in BENCHMARKS.items():
        mask = (out["market"] == market).to_numpy()
        if bench is None or not mask.any():
            continue
        cal = _calendar(indices, bench)
        pos = _last_before(cal, open_when[mask])
        ret5, ret20, vol20 = _trailing(cal, pos)
        out.loc[mask, "idx_ret_5d"] = ret5
        out.loc[mask, "idx_ret_20d"] = ret20
        out.loc[mask, "idx_vol_20d"] = vol20
        out.loc[mask, "idx_ret_issue_window"] = _issue_window(cal, pos, close_when[mask])
        if vol is not None:
            vcal = _calendar(indices, vol)
            vpos = _last_before(vcal, open_when[mask])
            vix = np.full(len(vpos), np.nan)
            ok = vpos >= 0
            vix[ok] = vcal.close[vpos[ok]]
            out.loc[mask, "vix_level"] = vix
    return out


# ---- live ------------------------------------------------------------------------------------------

def _nse_previous_closes() -> tuple[dict[str, tuple[pd.Timestamp, float]], list[str]]:
    """Dated previous closes for NIFTY 50 and INDIA VIX from NSE allIndices. Empty dict plus a note on failure."""
    client = _http.make_client()
    try:
        nse._warm(client)
        payload = nse._api_json(client, "/api/allIndices")
    except SourceError as exc:
        return {}, [f"NSE allIndices unavailable, local series used: {exc}"]
    finally:
        client.close()
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}, ["NSE allIndices has no data list (layout changed?), local series used"]
    out: dict[str, tuple[pd.Timestamp, float]] = {}
    notes: list[str] = []
    for symbol, name in NSE_NAMES.items():
        row = next((r for r in rows if isinstance(r, dict) and r.get("index") == name), None)
        if row is None:
            notes.append(f"NSE allIndices has no {name} row")
            continue
        try:
            when = pd.to_datetime(row["previousDay"], format="%d-%b-%Y")
            close = float(row["previousDayVal"])
        except (KeyError, TypeError, ValueError) as exc:
            notes.append(f"NSE {name} previous close unreadable ({type(exc).__name__})")
            continue
        if not math.isfinite(close) or close <= 0:
            notes.append(f"NSE {name} previous close not positive: {close}")
            continue
        out[symbol] = (when, close)
    return out, notes


def _live_calendar(local: pd.DataFrame | None, symbol: str, nse_point: tuple[pd.Timestamp, float] | None
                   ) -> tuple[_Calendar | None, list[str]]:
    """Local closes plus, when newer, the dated NSE previous close. Never overwrites a local value."""
    notes: list[str] = []
    frame = (local.loc[local["symbol"] == symbol, ["date", "close"]] if local is not None
             else pd.DataFrame(columns=["date", "close"]))
    if nse_point is not None:
        when, close = nse_point
        last = frame["date"].max() if len(frame) else None
        if last is None or when > last:
            frame = pd.concat([frame, pd.DataFrame({"date": [when], "close": [close]})], ignore_index=True)
            notes.append(f"{symbol}: NSE previous close {when.date()} appended")
        elif when == last:
            local_close = float(frame.loc[frame["date"] == last, "close"].iloc[-1])
            if abs(close / local_close - 1.0) > NSE_MISMATCH:
                notes.append(f"{symbol}: NSE close {close} differs from local {local_close} on {when.date()}")
    if frame.empty or frame["close"].dropna().empty:
        return None, notes + [f"no local data for {symbol}"]
    return _calendar_from_frame(frame), notes


def _is_fresh(cal: _Calendar, today: pd.Timestamp) -> bool:
    last = pd.Timestamp(cal.dates[-1]).to_datetime64().astype("datetime64[D]")
    return int(np.busday_count(last, today.to_datetime64().astype("datetime64[D]"))) <= LIVE_MAX_STALE_WEEKDAYS


def _live_benchmark(cal: _Calendar | None, symbol: str, today: pd.Timestamp, notes: list[str]
                    ) -> tuple[np.ndarray | None, pd.Timestamp | None]:
    """Position of the last close before today, as a one-element array, with that date. (None, None) when stale or missing."""
    if cal is None:
        return None, None
    if not _is_fresh(cal, today):
        last = pd.Timestamp(cal.dates[-1]).date()
        notes.append(f"{symbol} last close {last} is older than {LIVE_MAX_STALE_WEEKDAYS} trading days; NaN")
        return None, None
    pos = int(_last_before(cal, np.array([today.to_datetime64()]))[0])
    if pos < 0:
        notes.append(f"{symbol} has no close before {today.date()}; NaN")
        return None, None
    return np.array([pos]), pd.Timestamp(cal.dates[pos])


def live_context(market: str, today: date | None = None) -> dict:
    """CONTEXT_COLS as of today for a live IPO in `market`, plus 'context_as_of' and 'context_notes'.

    idx_ret_issue_window is always NaN here: it needs the close of the issue, which does not exist yet.
    """
    if market not in BENCHMARKS:
        raise ValueError(f"unknown market {market!r}")
    today_ts = pd.Timestamp(today or date.today())
    bench, vol = BENCHMARKS[market]
    out: dict = {col: math.nan for col in CONTEXT_COLS}
    notes: list[str] = []
    as_of = None
    if bench is None:
        notes.append(f"{market} has no benchmark index; context NaN")
        return {**out, "context_as_of": None, "context_notes": notes}
    local = index_data.load_indices()
    if local is None:
        notes.append("indices.parquet not built; context NaN")
        return {**out, "context_as_of": None, "context_notes": notes}

    nse_points: dict[str, tuple[pd.Timestamp, float]] = {}
    if market in ("IN_MAIN", "IN_SME"):
        nse_points, nse_notes = _nse_previous_closes()
        notes.extend(nse_notes)

    bench_cal, bench_notes = _live_calendar(local, bench, nse_points.get(bench))
    notes.extend(bench_notes)
    pos, as_of_ts = _live_benchmark(bench_cal, bench, today_ts, notes)
    if pos is not None:
        ret5, ret20, vol20 = _trailing(bench_cal, pos)
        out["idx_ret_5d"], out["idx_ret_20d"], out["idx_vol_20d"] = float(ret5[0]), float(ret20[0]), float(vol20[0])
        as_of = as_of_ts.date()
    notes.append("idx_ret_issue_window not computed live: it needs the issue close")

    if vol is not None:
        vol_cal, vol_notes = _live_calendar(local, vol, nse_points.get(vol))
        notes.extend(vol_notes)
        vpos, _ = _live_benchmark(vol_cal, vol, today_ts, notes)
        if vpos is not None:
            out["vix_level"] = float(vol_cal.close[vpos[0]])
    return {**out, "context_as_of": as_of, "context_notes": notes}
