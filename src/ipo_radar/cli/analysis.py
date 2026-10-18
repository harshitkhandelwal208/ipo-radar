"""Per-IPO analysis for `ipo show` and the dashboard: the numbers, their history context, the model view,
risk flags and a short summary that is labelled as an estimate.

Pure: it takes one LiveIPO, an optional Prediction and an optional history DataFrame (past IPOs, schema
columns). No network and no disk, so every rule is tested with small hand-built inputs.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable

import pandas as pd

from ipo_radar.cli import fmt
from ipo_radar.cli.model_rules import NO_MODEL_CAVEAT, NO_MODEL_REASON, own_range
from ipo_radar.schema import LiveIPO, Prediction

STALE_AFTER = timedelta(minutes=60)   # a GMP reading older than this is flagged as possibly out of date
HUGE_ISSUE_CR = 5000.0                # crore; issues this large need very deep demand
OFS_HIGH_PCT = 70.0                   # offer-for-sale share above which most money goes to existing holders
MIN_BUCKET_N = 5                      # fewer past IPOs than this in a GMP bucket is too few to describe
POST_CLOSE = ("closed", "allotment", "listing_today", "listed")
NO_SKILL_TEXT = "no demonstrated skill"  # the model's own caveat wording; see model/predict.py

SUB_ROWS: tuple[tuple[str, str], ...] = (
    ("sub_qib", "QIB"), ("sub_nii", "NII"), ("sub_snii", "sNII"), ("sub_bnii", "bNII"),
    ("sub_retail", "Retail"), ("sub_employee", "Employee"), ("sub_shareholder", "Shareholder"),
    ("sub_total", "Total"),
)
DAY_SUB_KEYS = frozenset({"sub_qib", "sub_nii", "sub_retail", "sub_total"})  # the only keys with day-wise data
GMP_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (-math.inf, 0.0, "below 0%"), (0.0, 10.0, "0 to 10%"), (10.0, 20.0, "10 to 20%"),
    (20.0, 35.0, "20 to 35%"), (35.0, 50.0, "35 to 50%"), (50.0, 75.0, "50 to 75%"),
    (75.0, 100.0, "75 to 100%"), (100.0, math.inf, "over 100%"),
)
SEGMENT_NAMES = {
    "IN_MAIN": "India main board", "IN_SME": "India SME", "US": "US", "HK": "Hong Kong",
    "OTHER": "other market", "": "segment unknown",
}
STATUS_TEXT = {
    "upcoming": "upcoming", "open": "open for bids", "closed": "bidding closed",
    "allotment": "allotment stage", "listing_today": "lists today", "listed": "listed", "unknown": "status unknown",
}


@dataclass
class SourceStatus:
    name: str
    state: str          # ok | down | unknown
    detail: str = ""


@dataclass
class Timeline:
    opens: date | None
    closes: date | None
    allotment: date | None
    listing: date | None
    today: date
    phase: str          # e.g. 'issue day 2 of 3', 'opens in 2 days'


@dataclass
class Issue:
    price_low: float | None
    price_high: float | None
    lot_size: int | None
    min_retail: float | None   # lot size x top of band: what one retail lot costs at the cut-off
    size_cr: float | None
    fresh_pct: float | None
    ofs_pct: float | None
    ofs_cr: float | None


@dataclass
class SubRow:
    key: str
    label: str
    now: float | None
    days: list[float | None]   # end-of-day 1..3, None where the source does not report it
    pctile: int | None         # share of past same-segment IPOs with a figure at or below `now`
    pctile_n: int
    source: str
    fetched_at: datetime | None


@dataclass
class GmpReading:
    source: str
    value: float | None
    fetched_at: datetime | None


@dataclass
class GmpHistory:
    bucket: str
    n: int
    median_gain: float
    p10: float
    p90: float
    share_positive: float      # percent of those IPOs that opened above the issue price


@dataclass
class Gmp:
    readings: list[GmpReading]
    median: float | None
    pct_of_top: float | None
    conflicts: list[str]
    history: GmpHistory | None
    history_note: str | None
    miss_median: float | None  # GMP% vs actual open gain, median absolute gap across past IPOs
    miss_n: int


@dataclass
class ModelView:
    as_of: str
    p10: float
    p50: float
    p90: float
    used: bool                 # False: the model is worse than the historical median here, so it is not the headline
    headline: float | None     # the estimate shown: the model's P50 when used, else the segment's training median
    median: float | None       # the segment's training median of the open gain (the model's own baseline)
    reason: str | None         # why the model is not used at this stage
    no_skill: bool             # True: the headline has no demonstrated skill over the median, so it gets a '~'
    p_positive: float          # probability in 0..1
    close_p50: float | None
    baseline_gmp_pct: float | None
    reliable: bool
    drivers: list[tuple[str, float]]
    caveats: list[str]
    version: str


@dataclass
class Flag:
    level: str                 # 'warn' (amber) or 'info' (dim)
    text: str
    label: str = ""            # short phrase for the summary; the full text is shown in the flags list


@dataclass
class Analysis:
    ipo_id: str
    company: str
    symbol: str
    market: str
    segment: str
    status: str
    as_of: datetime
    timeline: Timeline
    issue: Issue
    subs: list[SubRow]
    gmp: Gmp
    model: ModelView | None
    flags: list[Flag]
    warnings: list[str]
    sources: list[str]
    summary: str

    def sub(self, key: str) -> SubRow | None:
        return next((s for s in self.subs if s.key == key), None)


def _clean(value: object) -> float | None:
    """A finite float, or None. Missing and NaN stay missing."""
    if not fmt.known(value):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def newest_reading(ipos: list[LiveIPO]) -> datetime | None:
    """The newest fetch time among all subscription and GMP readings: how fresh the data is."""
    stamps = [s.fetched_at for i in ipos for s in [*i.sub.values(), *i.gmp.values()] if s.fetched_at is not None]
    return max(stamps, default=None)


def _segment(history: pd.DataFrame | None, market: str) -> pd.DataFrame | None:
    if history is None or not market or "market" not in history.columns:
        return None
    return history[history["market"] == market]


def _rank(value: float | None, series: pd.Series | None) -> tuple[int | None, int]:
    """Percentile of `value` among the non-missing past figures: the share at or below it."""
    if value is None or series is None:
        return None, 0
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return None, 0
    return round(float((vals <= value).mean() * 100)), int(vals.size)


def _timeline(ipo: LiveIPO, today: date) -> Timeline:
    if ipo.status == "upcoming" and ipo.open_date is not None:
        days = (ipo.open_date - today).days
        phase = "opens today" if days <= 0 else f"opens in {days} day{'s' if days != 1 else ''}"
    elif ipo.status == "open" and ipo.open_date is not None and ipo.close_date is not None:
        total = (ipo.close_date - ipo.open_date).days + 1
        phase = f"issue day {(today - ipo.open_date).days + 1} of {total}"
    else:
        phase = STATUS_TEXT.get(ipo.status, "status unknown")
    return Timeline(ipo.open_date, ipo.close_date, ipo.allotment_date, ipo.listing_date, today, phase)


def _issue(ipo: LiveIPO) -> Issue:
    size_cr = _clean(ipo.issue_size_m)
    size_cr = size_cr / 10 if size_cr is not None else None
    fresh = _clean(ipo.fresh_issue_pct)
    ofs = 100 - fresh if fresh is not None else None
    min_retail = ipo.lot_size * ipo.price_high if ipo.lot_size and ipo.price_high else None
    ofs_cr = size_cr * ofs / 100 if size_cr is not None and ofs is not None else None
    return Issue(_clean(ipo.price_low), _clean(ipo.price_high), ipo.lot_size, _clean(min_retail),
                 size_cr, fresh, ofs, ofs_cr)


def _subs(ipo: LiveIPO, seg: pd.DataFrame | None) -> list[SubRow]:
    rows = []
    for key, label in SUB_ROWS:
        s = ipo.sub.get(key)
        days = [
            _clean(ipo.sub_by_day.get(d, {}).get(key)) if key in DAY_SUB_KEYS else None
            for d in (1, 2, 3)
        ]
        now = _clean(s.value) if s is not None else None
        series = seg[key] if seg is not None and key in seg.columns else None
        pctile, n = _rank(now, series)
        rows.append(SubRow(key, label, now, days, pctile, n,
                           (s.source or "") if s is not None else "", s.fetched_at if s is not None else None))
    return rows


def _gmp_history(pct: float | None, seg: pd.DataFrame | None) -> tuple[GmpHistory | None, str | None, float | None, int]:
    """Bucket history by GMP% and describe what those IPOs actually opened at. Too few -> a note, no numbers."""
    needed = {"gmp_pct_last", "list_open_gain_pct"}
    if seg is None or not needed <= set(seg.columns):
        return None, "no past IPO history with GMP and listing gains", None, 0
    pairs = seg[["gmp_pct_last", "list_open_gain_pct"]].apply(pd.to_numeric, errors="coerce").dropna()
    if pairs.empty:
        return None, "no past IPOs with both a GMP and a listing gain", None, 0
    miss = float((pairs["gmp_pct_last"] - pairs["list_open_gain_pct"]).abs().median())
    miss_n = int(len(pairs))
    if pct is None:
        return None, "no GMP reading, so there is no bucket to compare", miss, miss_n
    lo, hi, label = next(b for b in GMP_BUCKETS if b[0] <= pct < b[1])
    gains = pairs.loc[(pairs["gmp_pct_last"] >= lo) & (pairs["gmp_pct_last"] < hi), "list_open_gain_pct"]
    if len(gains) < MIN_BUCKET_N:
        return None, (f"only {len(gains)} past IPO(s) had GMP {label}; too few to describe"), miss, miss_n
    hist = GmpHistory(label, int(len(gains)), float(gains.median()), float(gains.quantile(0.1)),
                      float(gains.quantile(0.9)), float((gains > 0).mean() * 100))
    return hist, None, miss, miss_n


def _gmp(ipo: LiveIPO, seg: pd.DataFrame | None) -> Gmp:
    readings = [GmpReading(s.source or key, _clean(s.value), s.fetched_at)
                for key, s in ipo.gmp.items() if key not in ("gmp", "gmp_pct")]
    median = _clean(ipo.gmp["gmp"].value) if "gmp" in ipo.gmp else None
    pct = _clean(ipo.gmp["gmp_pct"].value) if "gmp_pct" in ipo.gmp else None
    if pct is None and median is not None and ipo.price_high:
        pct = round(median / ipo.price_high * 100, 2)
    conflicts = [w for w in ipo.warnings if "GMP sources disagree" in w]
    hist, note, miss, miss_n = _gmp_history(pct, seg)
    return Gmp(readings, median, pct, conflicts, hist, note, miss, miss_n)


def _model(pred: Prediction, reason: str | None) -> ModelView:
    """When the model is not used, predict() has already put the segment's training median in the P50."""
    used = NO_MODEL_CAVEAT not in pred.caveats
    if not used and reason is None:
        reason = NO_MODEL_REASON
    median = None if used else _clean(pred.open_gain_pct_p50)
    no_skill = (not pred.reliable) or any(NO_SKILL_TEXT in c for c in pred.caveats) or not used
    # When the model is not used, the interval in the Prediction is shifted to the median; the model's own
    # range is what its caveat reports, and that is what the reference line shows.
    p10, p50, p90 = (pred.open_gain_pct_p10, pred.open_gain_pct_p50, pred.open_gain_pct_p90)
    own = None if used else own_range(pred.caveats)
    if own is not None:
        p10, p50, p90 = own
    return ModelView(
        as_of=pred.as_of, p10=p10, p50=p50, p90=p90,
        used=used, headline=pred.open_gain_pct_p50 if used else median, median=median,
        reason=reason, no_skill=no_skill,
        p_positive=pred.p_positive_open, close_p50=_clean(pred.close_gain_pct_p50),
        baseline_gmp_pct=_clean(pred.baseline_gmp_pct), reliable=pred.reliable,
        drivers=list(pred.drivers), caveats=list(pred.caveats), version=pred.model_version,
    )


def _flags(ipo: LiveIPO, issue: Issue, subs: list[SubRow], gmp: Gmp, model: ModelView | None,
           today: date, now: datetime) -> list[Flag]:
    out: list[Flag] = []
    by = {s.key: s.now for s in subs}
    qib, nii, retail = by.get("sub_qib"), by.get("sub_nii"), by.get("sub_retail")
    if ipo.market == "IN_SME":
        out.append(Flag("info", "SME platform: smaller, thinner-traded issues; treat estimates with more caution",
                        "SME platform"))
    if issue.ofs_pct is not None and issue.ofs_pct > OFS_HIGH_PCT:
        out.append(Flag("warn", f"offer for sale is {issue.ofs_pct:.0f}% of the issue: most of the money goes "
                                "to existing holders, not the company", f"OFS {issue.ofs_pct:.0f}%"))
    if issue.size_cr is not None and issue.size_cr >= HUGE_ISSUE_CR:
        out.append(Flag("info", f"very large issue ({fmt.rupees(issue.size_cr, decimals=0)} cr): it needs "
                                "deep demand to fill", "very large issue"))
    closing_soon = (ipo.status == "open" and ipo.close_date is not None
                    and (ipo.close_date - today).days <= 1)
    if qib is not None and qib < 1 and (closing_soon or ipo.status in POST_CLOSE):
        when = "with the issue closing soon" if closing_soon else "at close"
        out.append(Flag("warn", f"QIB at {qib:.2f}x {when}: institutional demand is thin", f"QIB {qib:.2f}x"))
    if None not in (retail, qib, nii) and retail >= 1 and qib < 1 and nii < 1:
        out.append(Flag("warn", f"retail-only demand: retail {retail:.1f}x, but QIB {qib:.2f}x "
                                f"and NII {nii:.2f}x", "retail-only demand"))
    for r in gmp.readings:
        age = fmt.age_seconds(r.fetched_at, now)
        if r.value is not None and age is not None and age > STALE_AFTER.total_seconds():
            out.append(Flag("warn", f"GMP from {r.source} is {fmt.age_text(r.fetched_at, now)}: "
                                    "it may be out of date", "stale GMP"))
    # Subscription differences are not flagged here: the merge's own single-line warning is shown once, under
    # Data notes in `ipo show`, as written.
    out += [Flag("warn", w, "GMP sources disagree") for w in gmp.conflicts]
    if model is not None and not model.used:
        out.append(Flag("warn", f"model not used for the headline: {model.reason}; the headline is the "
                                "segment's training median", "model not used"))
    if model is not None and not model.reliable:
        out.append(Flag("warn", "model inputs fall outside its training data: the estimate is less reliable",
                        "model outside training data"))
    out.append(Flag("info", "loss-making status unknown: financials are not in the live feed",
                    "loss-making unknown"))
    return out


def _summary(model: ModelView | None, gmp: Gmp, subs: list[SubRow], flags: list[Flag]) -> str:
    parts: list[str] = []
    if model is not None and not model.used:
        if model.median is not None:
            parts.append(f"Estimate: the model is not used for the headline here, because it is {model.reason}. "
                         f"The segment's training median listing-open gain is {model.median:+.0f}%; the model's "
                         f"range ({model.p10:+.0f}% to {model.p90:+.0f}%) is shown for reference only.")
        else:
            parts.append(f"Estimate: the model is not used here ({model.reason}), and no training median is "
                         "available for this segment.")
    elif model is not None:
        parts.append(f"Estimate: the model puts the listing-open gain at {model.p50:+.0f}% (range "
                     f"{model.p10:+.0f}% to {model.p90:+.0f}%), with a {fmt.probability_text(model.p_positive)} estimated "
                     "chance of opening above the issue price.")
    else:
        parts.append("Estimate: no model view is available yet.")
    if gmp.median is not None:
        text = f"The grey-market premium is {fmt.rupees(gmp.median, sign=True)}"
        if gmp.pct_of_top is not None:
            text += (f" ({gmp.pct_of_top:+.0f}% of the top price), which on its own would suggest about "
                     f"{gmp.pct_of_top:+.0f}%")
        parts.append(text + ". Grey-market figures are unofficial.")
    else:
        parts.append("No grey-market reading is available.")
    total = next((s.now for s in subs if s.key == "sub_total"), None)
    qib = next((s.now for s in subs if s.key == "sub_qib"), None)
    if total is not None:
        extra = f", QIB {fmt.multiple(qib)}" if qib is not None else ""
        parts.append(f"Subscription stands at {fmt.multiple(total)} overall{extra}.")
    concerns = [f.label or f.text for f in flags if f.level == "warn"]
    if concerns:
        parts.append("Main concerns: " + "; ".join(concerns[:2]) + ".")
    parts.append("These figures come from past IPOs and unofficial sources, and can be wrong.")
    return " ".join(parts)


def build_analysis(ipo: LiveIPO, pred: Prediction | None, *, now: datetime,
                   history: pd.DataFrame | None = None, unused: dict[str, str] | None = None) -> Analysis:
    """`now` must be timezone-aware in IST; the day-count logic uses its date. `unused` gives the reason, per stage,
    that the model is worse than the historical median (see cli.model_rules)."""
    today = now.date()
    seg = _segment(history, ipo.market)
    issue = _issue(ipo)
    subs = _subs(ipo, seg)
    gmp = _gmp(ipo, seg)
    model = None
    if pred is not None:
        model = _model(pred, (unused or {}).get(pred.as_of))
    flags = _flags(ipo, issue, subs, gmp, model, today, now)
    return Analysis(
        ipo_id=ipo.ipo_id, company=ipo.company, symbol=ipo.symbol, market=ipo.market,
        segment=SEGMENT_NAMES.get(ipo.market, ipo.market), status=ipo.status, as_of=now,
        timeline=_timeline(ipo, today), issue=issue, subs=subs, gmp=gmp, model=model, flags=flags,
        warnings=list(ipo.warnings), sources=list(ipo.sources),
        summary=_summary(model, gmp, subs, flags),
    )


def predict_all(predict_fn: Callable[[LiveIPO], Prediction | None], ipos: list[LiveIPO]
                ) -> tuple[dict[str, Prediction], list[str]]:
    """Predictions for every IPO not yet listed. A failing IPO is reported, the others still predict."""
    preds: dict[str, Prediction] = {}
    errors: list[str] = []
    for ipo in ipos:
        if ipo.status == "listed":
            continue
        try:
            pred = predict_fn(ipo)
        except Exception as exc:  # one bad row must not hide the rest of the dashboard
            errors.append(f"{ipo.company}: {type(exc).__name__}")
            continue
        if pred is not None:
            preds[ipo.ipo_id] = pred
    return preds, errors


def to_jsonable(obj: Any) -> Any:
    """Dataclasses, dates and lists to JSON-ready values. Non-finite floats and missing values become null."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return obj
