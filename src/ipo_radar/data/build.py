"""Merge the per-market tables into data/processed/ipos.parquet and write data/processed/COVERAGE.md.

Inputs are india.parquet and global.parquet, both required: a silently missing market would bias the model, so
a missing file raises FileNotFoundError. Context columns are recomputed from indices.parquet when it exists
(data.context.add_context); otherwise they stay NaN and COVERAGE.md says so. When two rows share an ipo_id the
row with more filled fields is kept. Impossible values (a non-positive price, a gain below -100%, a negative
subscription) are blanked, never clipped, and every blanking is counted in COVERAGE.md.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ipo_radar import config
from ipo_radar.data.context import add_context
from ipo_radar.schema import (ALL_COLS, CONTEXT_COLS, DATE_COLS, GMP_COLS, ID_COLS, MARKETS, SUB_COLS, SUB_DAY_COLS)

INPUTS = ("india", "global")
STR_COLS = ID_COLS + ["currency", "source", "source_url"]
POSITIVE_COLS = ["issue_price", "price_low", "price_high", "lot_size", "issue_size_m", "issue_size_usd_m",
                 "mcap_issue_usd_m"]
PERCENT_COLS = ["fresh_issue_pct", "ofs_pct"]                       # 0 to 100 by definition
GAIN_COLS = ["list_open_gain_pct", "list_high_gain_pct", "list_low_gain_pct", "list_close_gain_pct"]
NON_NEGATIVE_COLS = SUB_COLS + SUB_DAY_COLS + ["n_underwriters"]
# One definition for every market (the per-market builders each counted different fields).
CORE_FOR_COUNT = ["issue_price", "issue_size_m", "listing_date", "list_open_gain_pct", "list_close_gain_pct"]


@dataclass
class BuildReport:
    input_rows: dict[str, int] = field(default_factory=dict)
    coerce_failures: dict[str, int] = field(default_factory=dict)  # text that could not be read as number/date
    dropped: dict[str, int] = field(default_factory=dict)          # rows removed, with the reason
    blanked: dict[str, int] = field(default_factory=dict)          # cells set to missing, with the rule
    flagged: dict[str, int] = field(default_factory=dict)          # rows kept but suspicious
    notes: list[str] = field(default_factory=list)
    context_filled: bool = False


def _read_input(root: Path, name: str) -> pd.DataFrame:
    path = root / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: build the {name} table first")
    return pd.read_parquet(path)


def _conform(df: pd.DataFrame, label: str, report: BuildReport) -> pd.DataFrame:
    """Exactly schema.ALL_COLS in order, with the dtype each column type needs. Missing columns become NaN."""
    extra = [c for c in df.columns if c not in ALL_COLS]
    if extra:
        report.notes.append(f"{label}: columns not in the schema dropped: {', '.join(map(str, extra))}")
    absent = [c for c in ALL_COLS if c not in df.columns]
    if absent:
        report.notes.append(f"{label}: columns absent, written as missing: {', '.join(absent)}")
    out = pd.DataFrame(index=df.index)
    for col in ALL_COLS:
        src = df[col] if col in df.columns else pd.Series(np.nan, index=df.index, dtype="object")
        if col in STR_COLS:
            out[col] = src.astype("string")
        elif col in DATE_COLS:
            parsed = pd.to_datetime(src, errors="coerce")
            _count_failures(report, f"{label}.{col}", src, parsed)
            out[col] = parsed.astype("datetime64[ns]")
        elif col == "n_missing_core":
            out[col] = 0.0                                      # recomputed in _finish
        else:
            num = pd.to_numeric(src, errors="coerce")
            _count_failures(report, f"{label}.{col}", src, num)
            out[col] = num.astype("float64")
    return out


def _count_failures(report: BuildReport, key: str, src: pd.Series, out: pd.Series) -> None:
    failed = int((src.notna() & out.isna()).sum())
    if failed:
        report.coerce_failures[key] = failed


def _blank(df: pd.DataFrame, mask: pd.Series, cols: list[str], rule: str, report: BuildReport) -> None:
    hits = 0
    for col in cols:
        hit = mask & df[col].notna()
        hits += int(hit.sum())
        df.loc[hit, col] = np.nan
    if hits:
        report.blanked[rule] = report.blanked.get(rule, 0) + hits


def _sanity(df: pd.DataFrame, report: BuildReport) -> None:
    for col in POSITIVE_COLS:
        _blank(df, df[col] <= 0, [col], f"{col} not positive", report)
    _blank(df, df["price_low"] > df["price_high"], ["price_low", "price_high"], "price band inverted", report)
    for col in PERCENT_COLS:
        _blank(df, (df[col] < 0) | (df[col] > 100), [col], f"{col} outside 0-100", report)
    for col in GAIN_COLS:
        _blank(df, df[col] < -100, [col], f"{col} below -100%", report)
    # An exact 0 is almost never a real listing gain, GMP or subscription multiple: sources write 0 where no value
    # exists. Open gains are exactly 0 in 8-17% of rows from several sources, against 0-3% for close gains from the same
    # sources. Subscription multiples (final and day-wise) are blanked the same way, as the research report does.
    for col in GAIN_COLS + GMP_COLS + SUB_COLS + SUB_DAY_COLS:
        _blank(df, df[col] == 0, [col], f"{col} exactly 0 (placeholder)", report)
    for col in NON_NEGATIVE_COLS:
        _blank(df, df[col] < 0, [col], f"{col} negative", report)
    _blank(df, df["is_loss_making"].notna() & ~df["is_loss_making"].isin([0, 1]), ["is_loss_making"],
           "is_loss_making not 0/1", report)
    _blank(df, df["n_underwriters"].notna() & (df["n_underwriters"] != df["n_underwriters"].round()),
           ["n_underwriters"], "n_underwriters not whole", report)
    # Dates out of order are kept and flagged: which date is wrong cannot be known from the row.
    for name, late, early in (("close before open", "open_date", "close_date"),
                              ("listing before close", "close_date", "listing_date")):
        bad = int((df[early] < df[late]).sum())
        if bad:
            report.flagged[name] = bad


def build_frame(frames: Mapping[str, pd.DataFrame], indices: pd.DataFrame | None = None
                ) -> tuple[pd.DataFrame, BuildReport]:
    """Pure merge: conform, drop, sanity-check, add context. The caller decides where the frames come from."""
    report = BuildReport()
    parts = []
    for label, raw in frames.items():
        report.input_rows[label] = len(raw)
        part = _conform(raw, label, report)
        unknown = sorted(set(part["market"].dropna()) - set(MARKETS))
        if unknown:
            raise ValueError(f"{label}: unknown market values {unknown}; expected one of {MARKETS}")
        no_market = part["market"].isna()
        if no_market.any():
            report.dropped[f"{label}: no market"] = int(no_market.sum())
            part = part[~no_market]
        no_id = part["ipo_id"].isna()
        if no_id.any():
            report.dropped[f"{label}: no ipo_id"] = int(no_id.sum())
            part = part[~no_id]
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)

    filled = df[ALL_COLS].notna().sum(axis=1)
    keep = df.assign(_filled=filled).sort_values("_filled", ascending=False, kind="stable")
    before = len(keep)
    keep = keep.drop_duplicates("ipo_id", keep="first").drop(columns="_filled")
    if before > len(keep):
        report.dropped["duplicate ipo_id (less filled row)"] = before - len(keep)
    df = keep.sort_values(["listing_date", "ipo_id"], na_position="last").reset_index(drop=True)

    _sanity(df, report)
    if indices is not None:
        df = add_context(df, indices)
        report.context_filled = True
    else:
        for col in CONTEXT_COLS:
            df[col] = np.nan
        report.notes.append("indices.parquet not found: context columns are NaN")
    df["n_missing_core"] = df[CORE_FOR_COUNT].isna().sum(axis=1).astype("int64")
    return df[ALL_COLS].reset_index(drop=True), report


def _md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def coverage_markdown(df: pd.DataFrame, report: BuildReport, built_at: datetime) -> str:
    """Per market: row count, listing-year histogram and non-null share of every column."""
    markets = [m for m in MARKETS if m in set(df["market"].dropna())]
    lines = [
        "# Coverage of data/processed/ipos.parquet", "",
        f"Built {built_at.strftime('%Y-%m-%d %H:%M UTC')}. Rows: {len(df)}. Each market is listed separately; "
        "non-null shares are over the rows of that market. Missing values are NaN, never 0.", "",
        "## Rows per market", "",
    ]
    rows = []
    for m in markets:
        sub = df[df["market"] == m]
        years = sub["listing_date"].dt.year.dropna()
        span = f"{int(years.min())}-{int(years.max())}" if len(years) else "no listing dates"
        rows.append([m, len(sub), int(sub["list_open_gain_pct"].notna().sum()),
                     int(sub["list_close_gain_pct"].notna().sum()), span])
    lines += [_md_table(["market", "rows", "open-gain target", "close-gain target", "listing years"], rows), ""]

    lines += ["## Listing-year histogram", ""]
    years = sorted(df["listing_date"].dt.year.dropna().astype(int).unique())
    hist = []
    for y in years:
        cells = [int(((df["market"] == m) & (df["listing_date"].dt.year == y)).sum()) for m in markets]
        hist.append([y, *cells])
    lines += [_md_table(["listing year", *markets], hist) if hist else "No listing dates.", ""]

    lines += ["## Non-null share by column (%)", ""]
    cols = ["all", *markets]
    table = []
    for col in ALL_COLS:
        row = [col, f"{df[col].notna().mean() * 100:.1f}" if len(df) else "-"]
        for m in markets:
            sub = df.loc[df["market"] == m, col]
            row.append(f"{sub.notna().mean() * 100:.1f}" if len(sub) else "-")
        table.append(row)
    lines += [_md_table(["column", *cols], table), ""]

    lines += ["## Build notes", ""]
    lines.append("- Input rows: " + (", ".join(f"{k} {v}" for k, v in report.input_rows.items()) or "none"))
    lines.append("- Context columns: " + ("filled from indices.parquet (data.context.add_context)"
                                          if report.context_filled else "NOT filled (NaN)"))
    for title, data in (("Dropped", report.dropped), ("Blanked cells (kept row, value set to missing)", report.blanked),
                        ("Flagged, kept as they are", report.flagged),
                        ("Text that could not be read (set to missing)", report.coerce_failures)):
        if data:
            lines.append(f"- {title}: " + "; ".join(f"{k}: {v}" for k, v in data.items()))
        else:
            lines.append(f"- {title}: none")
    lines += [f"- {n}" for n in report.notes]
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def build_all(processed: Path | None = None) -> Path:
    """Read india/global (and indices when present), write ipos.parquet and COVERAGE.md. Returns the parquet path."""
    root = processed or config.PROCESSED
    frames = {name: _read_input(root, name) for name in INPUTS}
    idx_path = root / "indices.parquet"
    indices = pd.read_parquet(idx_path) if idx_path.exists() else None
    df, report = build_frame(frames, indices)
    out = root / "ipos.parquet"
    _atomic_write(out, lambda p: df.to_parquet(p, index=False))
    text = coverage_markdown(df, report, datetime.now(timezone.utc))
    _atomic_write(root / "COVERAGE.md", lambda p: p.write_text(text, encoding="utf-8"))
    return out


def main() -> None:
    path = build_all()
    df = pd.read_parquet(path)
    counts = df["market"].value_counts().to_dict()
    print(f"wrote {path} ({len(df)} rows: {counts})")
    print(f"wrote {path.with_name('COVERAGE.md')}")


if __name__ == "__main__":
    main()
