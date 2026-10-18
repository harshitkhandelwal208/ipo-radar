"""Single source of truth for the data contract. Every module imports column names from here.

Rule for all code: a value that is not known is None / NaN. Never fill a guess. Never convert a missing
subscription or GMP figure to 0. Each live value carries the source it came from and when it was fetched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

# ---- markets -------------------------------------------------------------------------------------
# Market codes. `segment` distinguishes India main board from SME (they behave very differently).
MARKETS = ("IN_MAIN", "IN_SME", "US", "HK", "OTHER")

# ---- historical table: one row per IPO (data/processed/ipos.parquet) --------------------------------
ID_COLS = ["ipo_id", "market", "country", "exchange", "company", "symbol", "sector"]
DATE_COLS = ["open_date", "close_date", "listing_date"]
ISSUE_COLS = [
    "currency", "price_low", "price_high", "issue_price", "lot_size",
    "issue_size_m",        # issue size in millions of local currency (INR: crore*10)
    "issue_size_usd_m",    # same, converted at listing-date FX (blank if FX unknown)
    "fresh_issue_pct",     # share of the issue that is a fresh issue (0-100); rest is offer for sale
    "ofs_pct",
    "mcap_issue_usd_m",    # market cap at issue price, USD millions
    "pe_issue", "roe", "roce", "debt_equity", "revenue_growth_pct", "pat_margin_pct", "is_loss_making",
    "anchor_pct", "n_underwriters",
]
# Subscription multiples (times subscribed) at the close of the issue. Indian category names are used for
# all markets; HK public-offer -> sub_retail, HK international tranche -> sub_qib (documented approximations;
# models always receive `market`). NaN when the market has no such tranche.
SUB_COLS = ["sub_qib", "sub_nii", "sub_snii", "sub_bnii", "sub_retail", "sub_employee", "sub_shareholder", "sub_total"]
# Day-wise subscription snapshots (end of issue day N; index starts at 1). Only the total and the big three.
SUB_DAY_COLS = [f"{c}_d{d}" for d in (1, 2, 3) for c in ("sub_qib", "sub_nii", "sub_retail", "sub_total")]
GMP_COLS = ["gmp_open", "gmp_last", "gmp_pct_open", "gmp_pct_last"]  # INR amount and % of issue price. India only.
# Targets (percent vs issue price). Listing day = first trading day.
TARGET_COLS = ["list_open_gain_pct", "list_high_gain_pct", "list_low_gain_pct", "list_close_gain_pct"]
CONTEXT_COLS = [  # market regime, filled by data.context (from index series; as of the day BEFORE open_date)
    "idx_ret_5d", "idx_ret_20d", "idx_vol_20d", "vix_level", "idx_ret_issue_window",
]
META_COLS = ["source", "source_url", "n_missing_core"]
ALL_COLS = (ID_COLS + DATE_COLS + ISSUE_COLS + SUB_COLS + SUB_DAY_COLS + GMP_COLS + TARGET_COLS
            + CONTEXT_COLS + META_COLS)

# ---- live objects ---------------------------------------------------------------------------------
@dataclass
class Sourced:
    """A live number with provenance. `value=None` means unknown (source down or field absent)."""
    value: float | None
    source: str = ""
    fetched_at: datetime | None = None
    note: str = ""


@dataclass
class LiveIPO:
    ipo_id: str
    market: str                      # one of MARKETS
    company: str
    symbol: str = ""
    status: str = "unknown"          # upcoming | open | closed | allotment | listing_today | listed
    open_date: date | None = None
    close_date: date | None = None
    allotment_date: date | None = None
    listing_date: date | None = None
    price_low: float | None = None
    price_high: float | None = None
    issue_price: float | None = None  # final price once discovered (after close); None while it is only a band
    lot_size: int | None = None
    issue_size_m: float | None = None
    fresh_issue_pct: float | None = None
    # live subscription, by category, each with provenance
    sub: dict[str, Sourced] = field(default_factory=dict)       # keys from SUB_COLS
    sub_by_day: dict[int, dict[str, float]] = field(default_factory=dict)
    sub_by_day_source: str = ""  # where sub_by_day came from: 'chittorgarh', 'own snapshot', or both joined
    gmp: dict[str, Sourced] = field(default_factory=dict)       # keys: 'gmp' (INR), 'gmp_pct', plus one per source
    gmp_history: list[tuple[datetime, float, str]] = field(default_factory=list)  # (when, gmp INR, source)
    listing: dict[str, float] = field(default_factory=dict)     # once listed: open/high/low/close price
    warnings: list[str] = field(default_factory=list)           # e.g. "GMP sources disagree: 40 vs 70"
    sources: list[str] = field(default_factory=list)


# ---- model I/O ------------------------------------------------------------------------------------
@dataclass
class Prediction:
    ipo_id: str
    as_of: str                       # 'pre_open' | 'day1' | 'day2' | 'close'
    open_gain_pct_p10: float
    open_gain_pct_p50: float
    open_gain_pct_p90: float
    p_positive_open: float           # calibrated probability that listing open > issue price
    close_gain_pct_p50: float | None
    drivers: list[tuple[str, float]] # (feature, contribution in pct points), top 6
    baseline_gmp_pct: float | None   # what "just trust the GMP" would say
    model_version: str
    reliable: bool                   # False when the inputs fall outside what the model saw in training
    caveats: list[str]
