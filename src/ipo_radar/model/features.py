"""Feature engineering shared by training, evaluation and serving. There is one design function, so the columns a
live IPO gets are the columns the model was trained on by construction.

Stages are cumulative: each stage sees what the earlier stages saw and nothing from later ones. The historical GMP
is one final reading taken shortly before listing, so GMP features enter only at 'pre_listing'. Missing values stay
NaN (LightGBM handles them); nothing is filled with 0. `log_` features are log1p of the raw value.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ipo_radar.schema import (CONTEXT_COLS, GMP_COLS, ISSUE_COLS, MARKETS, SUB_COLS, SUB_DAY_COLS, LiveIPO)

STAGES = ("pre_open", "day1", "day2", "close", "pre_listing")
TARGET_OPEN = "list_open_gain_pct"
TARGET_CLOSE = "list_close_gain_pct"
SECTOR_TOP_N = 25
OTHER = "other"
CATEGORICAL = ("market", "exchange", "sector")
PRE_OPEN_CONTEXT = ("idx_ret_5d", "idx_ret_20d", "idx_vol_20d", "vix_level")

# Raw schema columns each stage adds. asof_expand blanks the columns of every later stage.
STAGE_RAW: dict[str, tuple[str, ...]] = {
    "pre_open": tuple(ISSUE_COLS) + PRE_OPEN_CONTEXT,
    "day1": tuple(f"{c}_d1" for c in ("sub_qib", "sub_nii", "sub_retail", "sub_total")),
    "day2": tuple(f"{c}_d2" for c in ("sub_qib", "sub_nii", "sub_retail", "sub_total")),
    "close": tuple(SUB_COLS) + ("idx_ret_issue_window",) + tuple(c for c in SUB_DAY_COLS if c.endswith("_d3")),
    "pre_listing": tuple(GMP_COLS),
}

# Features each stage adds (cumulative in stage_features). Categoricals are listed in pre_open.
_NEW_FEATURES: dict[str, tuple[str, ...]] = {
    "pre_open": (
        "log_issue_size_usd_m", "log_issue_size_m", "log_price_ref", "lot_size", "fresh_issue_pct", "ofs_pct",
        "log_mcap_issue_usd_m", "pe_issue", "roe", "roce", "debt_equity", "revenue_growth_pct", "pat_margin_pct",
        "is_loss_making", "anchor_pct", "n_underwriters", "listing_month", "listing_weekday",
        "days_close_to_listing", *PRE_OPEN_CONTEXT, *CATEGORICAL),
    "day1": ("log_sub_qib_d1", "log_sub_nii_d1", "log_sub_retail_d1", "log_sub_total_d1",
             "ratio_retail_qib_d1", "ratio_nii_qib_d1"),
    "day2": ("log_sub_qib_d2", "log_sub_nii_d2", "log_sub_retail_d2", "log_sub_total_d2",
             "ratio_retail_qib_d2", "ratio_nii_qib_d2"),
    "close": ("log_sub_qib", "log_sub_nii", "log_sub_snii", "log_sub_bnii", "log_sub_retail",
              "log_sub_employee", "log_sub_shareholder", "log_sub_total", "ratio_retail_qib", "ratio_nii_qib",
              "has_category_sub", "idx_ret_issue_window"),
    "pre_listing": ("gmp_pct_last", "has_gmp"),
}

# Groups for importance reporting. Every feature must belong to exactly one group (tested).
GROUPS: dict[str, tuple[str, ...]] = {
    "issue": ("log_issue_size_usd_m", "log_issue_size_m", "log_price_ref", "lot_size", "fresh_issue_pct",
              "ofs_pct", "anchor_pct", "n_underwriters"),
    "financials": ("log_mcap_issue_usd_m", "pe_issue", "roe", "roce", "debt_equity", "revenue_growth_pct",
                   "pat_margin_pct", "is_loss_making"),
    "subscription": tuple(f for f in [*_NEW_FEATURES["day1"], *_NEW_FEATURES["day2"], *_NEW_FEATURES["close"]]
                          if f not in ("idx_ret_issue_window",)),
    "gmp": ("gmp_pct_last", "has_gmp"),
    "market_context": (*PRE_OPEN_CONTEXT, "idx_ret_issue_window"),
    "calendar": ("listing_month", "listing_weekday", "days_close_to_listing"),
    "identity": CATEGORICAL,
}

# Inputs that must be present for a stage to be trusted (only counted when the market normally has them).
CORE_NEW: dict[str, tuple[str, ...]] = {
    "pre_open": ("log_price_ref", "log_issue_size_m", "lot_size", "fresh_issue_pct", "idx_ret_20d"),
    "day1": ("log_sub_total_d1", "log_sub_qib_d1"),
    "day2": ("log_sub_total_d2", "log_sub_qib_d2"),
    "close": ("log_sub_total", "log_sub_qib", "log_sub_retail"),
    "pre_listing": ("gmp_pct_last",),
}
CORE_LABELS = {
    "log_price_ref": "price band or issue price", "log_issue_size_m": "issue size",
    "lot_size": "lot size", "fresh_issue_pct": "fresh-issue share", "idx_ret_20d": "index 20-day return",
    "log_sub_total_d1": "day-1 total subscription", "log_sub_qib_d1": "day-1 QIB subscription",
    "log_sub_total_d2": "day-2 total subscription", "log_sub_qib_d2": "day-2 QIB subscription",
    "log_sub_total": "total subscription", "log_sub_qib": "QIB subscription",
    "log_sub_retail": "retail subscription", "gmp_pct_last": "GMP",
}
RANGE_PREFIXES = ("log_sub_", "log_issue_size_m")   # features checked against the training 1st-99th percentile


@dataclass(frozen=True)
class Vocab:
    """Category levels learned from training rows. Sectors outside the top list are 'other'; a missing sector stays
    missing."""
    sectors: tuple[str, ...]
    exchanges: tuple[str, ...]


def stage_index(stage: str) -> int:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    return STAGES.index(stage)


def stage_features(stage: str) -> list[str]:
    """Feature names for a stage, in a fixed order: the stage's own features plus those of every earlier stage."""
    out: list[str] = []
    for s in STAGES[: stage_index(stage) + 1]:
        out += list(_NEW_FEATURES[s])
    return out


def core_features(stage: str) -> tuple[str, ...]:
    out: tuple[str, ...] = ()
    for s in STAGES[: stage_index(stage) + 1]:
        out += CORE_NEW[s]
    return out


def group_of(feature: str) -> str:
    for group, names in GROUPS.items():
        if feature in names:
            return group
    raise KeyError(f"feature {feature!r} has no importance group")


def fit_vocab(df: pd.DataFrame) -> Vocab:
    counts = df["sector"].dropna().astype(str).value_counts()
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))  # ties by name: the list is reproducible
    sectors = tuple(name for name, _ in ranked[:SECTOR_TOP_N])
    exchanges = tuple(sorted(df["exchange"].dropna().astype(str).unique()))
    return Vocab(sectors, exchanges)


def to_model_scale(pct: np.ndarray | pd.Series | float) -> np.ndarray:
    """Monotone transform for the regression target: sign(y) * log1p(|y| / 100). Quantiles map back exactly."""
    arr = np.asarray(pct, dtype=float)
    return np.sign(arr) * np.log1p(np.abs(arr) / 100.0)


def from_model_scale(z: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(z, dtype=float)
    return np.sign(arr) * 100.0 * np.expm1(np.abs(arr))


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _log(s: pd.Series) -> pd.Series:
    return np.log1p(_num(s).where(_num(s) > -1))


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    d = _num(den)
    return _num(num) / d.where(d > 0)


def _categorical(values: list, categories: list) -> pd.Categorical:
    return pd.Categorical(values, categories=categories)


def _all_features(rows: pd.DataFrame, vocab: Vocab) -> pd.DataFrame:
    """Every feature any stage can use, computed from raw schema columns. design() selects a stage's subset."""
    r = rows
    out = pd.DataFrame(index=rows.index)
    lo, hi = _num(r["price_low"]), _num(r["price_high"])
    # Band midpoint only: the final issue price is set at close, so using it before the issue would leak it.
    out["log_price_ref"] = np.log1p(((lo + hi) / 2).where(lambda x: x > -1))
    out["log_issue_size_usd_m"] = _log(r["issue_size_usd_m"])
    out["log_issue_size_m"] = _log(r["issue_size_m"])
    out["log_mcap_issue_usd_m"] = _log(r["mcap_issue_usd_m"])
    for col in ("lot_size", "fresh_issue_pct", "ofs_pct", "pe_issue", "roe", "roce", "debt_equity",
                "revenue_growth_pct", "pat_margin_pct", "is_loss_making", "anchor_pct", "n_underwriters",
                "idx_ret_5d", "idx_ret_20d", "idx_vol_20d", "vix_level", "idx_ret_issue_window",
                "gmp_pct_last"):
        out[col] = _num(r[col])
    listing = pd.to_datetime(r["listing_date"], errors="coerce")
    out["listing_month"] = listing.dt.month.astype("float64")
    out["listing_weekday"] = listing.dt.weekday.astype("float64")
    close = pd.to_datetime(r["close_date"], errors="coerce")
    out["days_close_to_listing"] = (listing - close).dt.days.astype("float64")
    for suffix in ("", "_d1", "_d2"):
        cols = ["sub_qib", "sub_nii", "sub_retail", "sub_total"]
        if suffix == "":
            cols += ["sub_snii", "sub_bnii", "sub_employee", "sub_shareholder"]
        for col in cols:
            out[f"log_{col}{suffix}"] = _log(r[f"{col}{suffix}"])
        out[f"ratio_retail_qib{suffix}"] = _ratio(r[f"sub_retail{suffix}"], r[f"sub_qib{suffix}"])
        out[f"ratio_nii_qib{suffix}"] = _ratio(r[f"sub_nii{suffix}"], r[f"sub_qib{suffix}"])
    out["has_category_sub"] = r[["sub_qib", "sub_nii", "sub_retail"]].notna().any(axis=1).astype(float)
    out["has_gmp"] = _num(r["gmp_pct_last"]).notna().astype(float)
    sector = r["sector"]
    top = set(vocab.sectors)
    out_sector = [None if pd.isna(x) else (str(x) if str(x) in top else OTHER) for x in sector]
    out["sector"] = _categorical(out_sector, [*vocab.sectors, OTHER])
    out["market"] = _categorical([x if x in MARKETS else None for x in r["market"]], list(MARKETS))
    out["exchange"] = _categorical([None if pd.isna(x) else (str(x) if str(x) in vocab.exchanges else None)
                                    for x in r["exchange"]], list(vocab.exchanges))
    return out


def design(rows: pd.DataFrame, stage: str, vocab: Vocab) -> pd.DataFrame:
    """The model's input matrix for one stage: columns exactly stage_features(stage), in that order."""
    feats = stage_features(stage)
    full = _all_features(rows, vocab)
    return full[feats]


def asof_expand(df: pd.DataFrame) -> pd.DataFrame:
    """One row per IPO per stage. Columns of later stages are set to NaN, so a stage cannot see the future.
    Targets are kept (they are labels, not features)."""
    parts = []
    for i, stage in enumerate(STAGES):
        part = df.copy()
        later = [c for s in STAGES[i + 1:] for c in STAGE_RAW[s] if c in part.columns]
        if later:
            part[later] = np.nan
        part["as_of"] = pd.Categorical([stage] * len(part), categories=list(STAGES))
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def make_xy(df: pd.DataFrame, target: str, vocab: Vocab | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Design matrix and target for the rows of one stage (df has a single as_of), dropping rows without the target."""
    stages = sorted(set(df["as_of"].astype(str)))
    if len(stages) != 1:
        raise ValueError(f"make_xy needs rows of one stage, got {stages}")
    vocab = vocab or fit_vocab(df)
    keep = df[target].notna()
    rows = df.loc[keep]
    return design(rows, stages[0], vocab), rows[target].astype(float)


def live_gmp_pct(live: LiveIPO) -> float | None:
    """GMP as % of the issue price: the reading's own percent, else INR over the issue price (or top of band)."""
    pct = live.gmp.get("gmp_pct")
    if pct is not None and _finite(pct.value):
        return float(pct.value)
    inr = live.gmp.get("gmp")
    price = live.issue_price or live.price_high
    if inr is not None and _finite(inr.value) and price:
        return float(inr.value) / float(price) * 100.0
    return None


def _finite(value: object) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(float(value))


def _opt(value: object) -> float:
    return float(value) if _finite(value) else math.nan


def _day(value: date | None) -> pd.Timestamp:
    return pd.NaT if value is None else pd.Timestamp(value)


def live_raw_row(live: LiveIPO, context: Mapping[str, float] | None = None) -> dict:
    """Schema-shaped raw values for one live IPO. Fields the live feed does not carry are missing, not guessed."""
    ctx = dict(context) if context else {}
    row: dict = {c: math.nan for c in (*ISSUE_COLS, *SUB_COLS, *SUB_DAY_COLS, *GMP_COLS, *CONTEXT_COLS)}
    row.update({
        "ipo_id": live.ipo_id, "market": live.market, "exchange": None, "sector": None,
        "open_date": _day(live.open_date), "close_date": _day(live.close_date),
        "listing_date": _day(live.listing_date), "price_low": _opt(live.price_low),
        "price_high": _opt(live.price_high), "issue_price": _opt(live.issue_price),
        "lot_size": _opt(live.lot_size), "issue_size_m": _opt(live.issue_size_m),
        "fresh_issue_pct": _opt(live.fresh_issue_pct),
        "ofs_pct": _opt(100 - live.fresh_issue_pct) if _finite(live.fresh_issue_pct) else math.nan,
        "gmp_pct_last": _opt(live_gmp_pct(live)),
    })
    for key in SUB_COLS:
        sourced = live.sub.get(key)
        row[key] = _opt(sourced.value) if sourced is not None else math.nan
    for day, values in live.sub_by_day.items():
        for key, value in values.items():
            if f"{key}_d{day}" in row:
                row[f"{key}_d{day}"] = _opt(value)
    for key in CONTEXT_COLS:
        row[key] = _opt(ctx.get(key))
    return row


def live_to_row(live: LiveIPO, as_of: str, vocab: Vocab, context: Mapping[str, float] | None = None) -> pd.DataFrame:
    """One-row design matrix for a live IPO at a stage: the same columns as training, in the same order."""
    raw = pd.DataFrame([live_raw_row(live, context)])
    return design(raw, as_of, vocab)
