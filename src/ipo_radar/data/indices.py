"""Daily closes for the index symbols used as market context -> data/processed/indices.parquet.

Offline by design: every rung reads a file already downloaded under data/raw/kaggle/<owner>__<slug>/
(catalogue and warnings in docs/data_sources/kaggle.md). No network is used here; live values come from
context.live_context.

Source ladder per symbol: the first rung is the primary series. Later rungs are spliced in only where
they overlap the series so far and agree on the overlap: the largest close difference must be at most
MAX_DIFF, measured over at least MIN_OVERLAP shared dates. A rung that disagrees stops the splice for that
symbol and is reported. Missing days are never forward-filled, so a gap stays a gap (context becomes NaN).

Known exclusions (see docs/data_sources/kaggle.md):
- debashis74017 VIX_History.csv is US VIX, not India VIX.
- saurabhbadole NIFTY and SENSEX files hold about one month (rolling); only its India VIX file is used.
- ruanqian Shanghai file is forward-adjusted (前复权), so it differs from raw closes; not used.
- mukhazarahmad INDICES_DATA.csv has a two-row header and is not needed for any symbol; not used.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ipo_radar import config
from ipo_radar.errors import SourceError

SYMBOLS = ("NIFTY50", "SENSEX", "INDIAVIX", "SP500", "NYSE_COMP", "NASDAQ_COMP", "VIX", "HSI", "SHCOMP")
START = pd.Timestamp("2005-01-01")      # India VIX simply starts later (2009/2010)
KAGGLE = config.RAW / "kaggle"
MAX_DIFF = 0.005                        # 0.5 %: largest close difference allowed on an overlap
MIN_OVERLAP = 20                        # shared dates needed before two sources can be compared
MAX_BAD_ROWS = 0.001                    # share of rows with an unreadable date or close before a file is rejected


@dataclass(frozen=True)
class Layout:
    date_col: str
    close_col: str
    symbol_col: str | None = None       # long files: keep the rows whose symbol_col equals the rung key
    date_format: str | None = None
    skiprows: tuple[int, ...] | None = None
    excel: bool = False


@dataclass(frozen=True)
class Rung:
    label: str                          # shown in coverage notes and docs
    folder: str                         # folder name under data/raw/kaggle/
    pattern: str                        # file name or glob inside that folder
    layout: Layout
    key: str | None = None              # symbol inside a long file (e.g. "^GSPC")


BELBINO = "belbino__global-financial-markets-dataset-2010-present"
BELBINO_FILE = "global_financial_markets_2000_Now.csv"
MATTIUZC = "mattiuzc__stock-exchange-data"
MATTIUZC_FILE = "indexProcessed.csv"
_LONG_BELBINO = Layout("date", "close", symbol_col="symbol")
_LONG_MATTIUZC = Layout("Date", "Close", symbol_col="Index")

# Ordered per symbol: primary first. Only files that were checked on disk are listed.
LADDER: dict[str, list[Rung]] = {
    "SP500": [
        Rung("belbino ^GSPC", BELBINO, BELBINO_FILE, _LONG_BELBINO, key="^GSPC"),
        Rung("henryhan117 SPX.csv", "henryhan117__sp-500-historical-data", "SPX.csv", Layout("Date", "Close")),
    ],
    "NASDAQ_COMP": [
        Rung("belbino ^IXIC", BELBINO, BELBINO_FILE, _LONG_BELBINO, key="^IXIC"),
        Rung("mattiuzc IXIC", MATTIUZC, MATTIUZC_FILE, _LONG_MATTIUZC, key="IXIC"),
    ],
    "HSI": [
        Rung("belbino ^HSI", BELBINO, BELBINO_FILE, _LONG_BELBINO, key="^HSI"),
        Rung("rushil999 HSI.csv", "rushil999__hang-seng-historical-prices", "HSI.csv", Layout("Date", "Close")),
        Rung("mattiuzc HSI", MATTIUZC, MATTIUZC_FILE, _LONG_MATTIUZC, key="HSI"),
    ],
    "SHCOMP": [
        Rung("belbino 000001.SS", BELBINO, BELBINO_FILE, _LONG_BELBINO, key="000001.SS"),
        Rung("mattiuzc 000001.SS", MATTIUZC, MATTIUZC_FILE, _LONG_MATTIUZC, key="000001.SS"),
        Rung("code1110 YF000001-SS", "code1110__yfinance-world-indices-price-data", "YF000001-SS.csv",
             Layout("date", "close")),
    ],
    "SENSEX": [
        Rung("belbino ^BSESN", BELBINO, BELBINO_FILE, _LONG_BELBINO, key="^BSESN"),
        Rung("avtnshm CSVForDate (1).xlsx", "avtnshm__bombay-stock-exchange-bse-sensex-index-2005-2024",
             "CSVForDate (1).xlsx", Layout("Date", "Close", excel=True)),
    ],
    "NIFTY50": [
        Rung("belbino ^NSEI", BELBINO, BELBINO_FILE, _LONG_BELBINO, key="^NSEI"),
        Rung("sudalairajkumar NIFTY 50.csv", "sudalairajkumar__nifty-indices-dataset", "NIFTY 50.csv",
             Layout("Date", "Close")),
        Rung("debashis74017 NIFTY 50_Data.csv", "debashis74017__stock-market-index-data-india-1990-2022",
             "NIFTY 50_Data.csv", Layout("Date", "Close", date_format="%d %b %Y")),
    ],
    # A rung that conflicts stops the splice, so the agreeing extension (saurabhbadole, to today) comes
    # before the conflicting ones (alangeorgev, sudalairajkumar; see docs/data_sources/indices.md).
    "INDIAVIX": [
        Rung("ahaanverma00 INDIA VIX", "ahaanverma00__nifty-500-and-india-vix-daily-data-2010-2026",
             "INDIA VIX*.csv", Layout("Date", "Close", date_format="%d %b %Y")),
        Rung("saurabhbadole India VIX", "saurabhbadole__indian-stock-market-indices-daily-updates",
             "India VIX.csv", Layout("Price", "Close", skiprows=(1, 2))),
        Rung("alangeorgev India VIX", "alangeorgev__india-vix-historical-2025",
             "India VIX Historical Data 101.csv", Layout("Date", "Price", date_format="%d-%m-%Y")),
        Rung("sudalairajkumar INDIAVIX", "sudalairajkumar__nifty-indices-dataset", "INDIAVIX.csv",
             Layout("Date", "Close")),
    ],
    "VIX": [
        Rung("guillemservera vix_daily", "guillemservera__vix-cboe-volatility-index-daily-updated",
             "vix_daily.csv", Layout("date", "close")),
        Rung("debashis74017 VIX_History (US CBOE)", "debashis74017__stock-market-index-data-india-1990-2022",
             "VIX_History.csv", Layout("DATE", "CLOSE", date_format="%m/%d/%Y")),
        Rung("code1110 YF-VIX", "code1110__yfinance-world-indices-price-data", "YF-VIX.csv",
             Layout("date", "close")),
    ],
    "NYSE_COMP": [
        Rung("mattiuzc NYA", MATTIUZC, MATTIUZC_FILE, _LONG_MATTIUZC, key="NYA"),
        Rung("pavankrishnanarne ^NYA", "pavankrishnanarne__global-stock-market-2008-present",
             "*_Glob*_Data.csv", Layout("Date", "Close", symbol_col="Ticker"), key="^NYA"),
        Rung("code1110 YF-NYA", "code1110__yfinance-world-indices-price-data", "YF-NYA.csv",
             Layout("date", "close")),
    ],
}


@dataclass(frozen=True)
class SymbolCoverage:
    symbol: str
    status: str                          # "ok" or "no source"
    sources: tuple[str, ...]             # rungs whose rows are in the series, primary first
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    rows: int
    notes: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class IndicesBuild:
    frame: pd.DataFrame                  # columns date, symbol, close
    coverage: tuple[SymbolCoverage, ...]


def _read_one(path: Path, rung: Rung) -> pd.DataFrame:
    """One file -> frame[date, close]. Unreadable rows beyond MAX_BAD_ROWS reject the file (layout changed)."""
    lay = rung.layout
    try:
        raw = pd.read_excel(path) if lay.excel else pd.read_csv(path, skiprows=lay.skiprows)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise SourceError(f"{rung.label}: unreadable file {path.name} ({type(exc).__name__})") from exc
    if lay.symbol_col is not None:
        if lay.symbol_col not in raw.columns:
            raise SourceError(f"{rung.label}: column {lay.symbol_col!r} missing (layout changed?)")
        raw = raw[raw[lay.symbol_col] == rung.key]
    missing = [c for c in (lay.date_col, lay.close_col) if c not in raw.columns]
    if missing:
        raise SourceError(f"{rung.label}: columns {missing} missing in {path.name} (layout changed?)")

    dates = pd.to_datetime(raw[lay.date_col], format=lay.date_format, errors="coerce")
    closes = pd.to_numeric(raw[lay.close_col], errors="coerce")
    bad_dates = dates.isna() & raw[lay.date_col].notna()
    bad_closes = closes.isna() & raw[lay.close_col].notna()
    n = max(len(raw), 1)
    if (bad_dates.sum() + bad_closes.sum()) / n > MAX_BAD_ROWS:
        raise SourceError(f"{rung.label}: {int(bad_dates.sum())} unreadable dates and "
                          f"{int(bad_closes.sum())} unreadable closes in {path.name} (layout changed?)")
    out = pd.DataFrame({"date": dates, "close": closes}).dropna()     # blanks stay missing, never 0
    out = out[out["close"] > 0].sort_values("date").drop_duplicates("date", keep="last")
    return out.reset_index(drop=True)


def _load_rung(rung: Rung) -> pd.DataFrame:
    files = sorted(p for p in (KAGGLE / rung.folder).glob(rung.pattern) if p.is_file())
    if not files:
        raise SourceError(f"{rung.label}: no file matching {rung.pattern!r} in {rung.folder}")
    frames = [_read_one(p, rung) for p in files]
    return pd.concat(frames, ignore_index=True).sort_values("date").drop_duplicates("date", keep="last")


def splice(symbol: str, loaded: list[tuple[str, pd.DataFrame]]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Merge rungs in ladder order. Returns (frame, labels used, notes).

    A later rung adds only the dates the series does not have yet. Dates are never filled from a
    neighbour, so a gap stays a gap. A rung whose overlap disagrees by more than MAX_DIFF stops the splice.
    """
    notes: list[str] = []
    if not loaded:
        return pd.DataFrame(columns=["date", "close"]), [], notes
    label0, acc = loaded[0]
    used = [label0]
    notes.append(f"{label0}: primary, {len(acc)} rows")
    for label, cand in loaded[1:]:
        both = acc.merge(cand, on="date", suffixes=("_acc", "_cand"))
        if len(both) < MIN_OVERLAP:
            notes.append(f"{label}: {len(both)} shared dates (need {MIN_OVERLAP}), not used")
            continue
        worst = float(((both["close_cand"] / both["close_acc"]) - 1.0).abs().max())
        if worst > MAX_DIFF:
            notes.append(f"{label}: closes differ by up to {worst:.2%} on {len(both)} shared dates "
                         f"(limit {MAX_DIFF:.1%}); splicing stopped for {symbol}")
            break
        new = cand[~cand["date"].isin(acc["date"])]
        acc = pd.concat([acc, new], ignore_index=True).sort_values("date").reset_index(drop=True)
        used.append(label)
        notes.append(f"{label}: agrees on {len(both)} shared dates (max diff {worst:.3%}), added {len(new)} rows")
    return acc, used, notes


def _build_symbol(symbol: str) -> tuple[pd.DataFrame, SymbolCoverage]:
    loaded: list[tuple[str, pd.DataFrame]] = []
    notes: list[str] = []
    for rung in LADDER.get(symbol, []):
        try:
            frame = _load_rung(rung)
        except SourceError as exc:
            notes.append(str(exc))
            continue
        frame = frame[frame["date"] >= START].reset_index(drop=True)
        if frame.empty:
            notes.append(f"{rung.label}: no rows on or after {START.date()}")
            continue
        loaded.append((rung.label, frame))
    series, used, splice_notes = splice(symbol, loaded)
    notes.extend(splice_notes)
    series = series[series["date"] >= START].reset_index(drop=True)
    if series.empty:
        cov = SymbolCoverage(symbol, "no source", (), None, None, 0, tuple(notes))
        return pd.DataFrame(columns=["date", "symbol", "close"]), cov
    out = series.assign(symbol=symbol)[["date", "symbol", "close"]]
    cov = SymbolCoverage(symbol, "ok", tuple(used), out["date"].min(), out["date"].max(), len(out), tuple(notes))
    return out, cov


def build_report() -> IndicesBuild:
    """Build every symbol that has a source. Raises SourceError only if no symbol could be built."""
    frames: list[pd.DataFrame] = []
    coverage: list[SymbolCoverage] = []
    for symbol in SYMBOLS:
        frame, cov = _build_symbol(symbol)
        coverage.append(cov)
        if cov.status == "ok":
            frames.append(frame)
    if not frames:
        raise SourceError("no index symbol could be built: " + " ; ".join(
            f"{c.symbol}: {'; '.join(c.notes) or 'no rung'}" for c in coverage))
    out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    config.INDICES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.INDICES_PARQUET, index=False)
    return IndicesBuild(out, tuple(coverage))


def build_indices() -> pd.DataFrame:
    """Build and write data/processed/indices.parquet (columns date, symbol, close)."""
    return build_report().frame


def load_indices() -> pd.DataFrame | None:
    """The written series, or None when indices.parquet has not been built yet."""
    if not config.INDICES_PARQUET.exists():
        return None
    return pd.read_parquet(config.INDICES_PARQUET)


def _print(report: IndicesBuild) -> None:
    print(f"{'symbol':<12}{'status':<11}{'first':<12}{'last':<12}{'rows':>7}  sources")
    for c in report.coverage:
        first = c.first.date() if c.first is not None else "-"
        last = c.last.date() if c.last is not None else "-"
        print(f"{c.symbol:<12}{c.status:<11}{str(first):<12}{str(last):<12}{c.rows:>7}  {' + '.join(c.sources) or '-'}")
    for c in report.coverage:
        for note in c.notes:
            print(f"  [{c.symbol}] {note}")
    print(f"wrote {config.INDICES_PARQUET} ({len(report.frame)} rows)")


def main() -> None:
    _print(build_report())


if __name__ == "__main__":
    main()
