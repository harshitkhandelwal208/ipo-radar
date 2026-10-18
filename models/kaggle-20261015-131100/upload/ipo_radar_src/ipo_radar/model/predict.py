"""Serving: one live IPO in, one schema.Prediction out.

The stage comes from the live state: before the issue opens -> pre_open; while it is open -> day1 or day2 by the
latest day with data; after close -> close; after close with a live GMP reading -> pre_listing. The segment (IN_MAIN,
IN_SME, GLOBAL) decides which trained model set answers, as the bundle's routing says. Market context comes from the
public data.context.live_context (live NSE values for India, cached per hour). `drivers` are LightGBM pred_contrib
values rescaled to the p50 move in percentage points: an approximation, because the model works on the transformed
scale.
"""
from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ipo_radar import config
from ipo_radar.data import context as ctx_mod
from ipo_radar.model import ModelNotTrained
from ipo_radar.model import features as F
from ipo_radar.model.bundle import Bundle, StageModels
from ipo_radar.model.design import segment_of
from ipo_radar.model.fitting import adjust_intervals, calibrate_prob, raw_components
from ipo_radar.schema import CONTEXT_COLS, LiveIPO, Prediction

MIN_MARKET_ROWS = 100    # fewer training IPOs in the market than this and the estimate is flagged
SME_MIN_ROWS = 200       # an SME issue is flagged when the training set has fewer SME IPOs than this
TOP_DRIVERS = 6
CLOSED_STATUS = ("closed", "allotment", "listing_today", "listed")
SUB_KEYS = ("sub_qib", "sub_nii", "sub_snii", "sub_bnii", "sub_retail", "sub_employee", "sub_shareholder",
            "sub_total")
GMP_CAVEAT = "live GMP is earlier than the final pre-listing reading the model was trained on"
VALUATION_CAVEAT = "valuation and financial ratios are not in the live feed; the model treats them as unknown"
SKILL_P50_CAVEAT = "no demonstrated skill over the historical median at this stage"
SKILL_PROB_CAVEAT = "the chance of opening above the issue price does not beat the base rate at this stage"
NO_SKILL_EVIDENCE = "no demonstrated skill: no pre-2025 walk-forward evidence for this stage and segment"


def _model_file(path: Path) -> Path:
    return path if path.suffix == ".joblib" else path / "bundle.joblib"


def load_model(path: Path | None = None) -> Bundle:
    """The saved bundle (default config.MODELS), reloaded when the file changes. Raises ModelNotTrained when none."""
    file = _model_file(Path(config.MODELS if path is None else path))
    if not file.exists():
        raise ModelNotTrained(f"no trained model at {file}: run `ipo train`")
    return _load_cached(str(file), file.stat().st_mtime_ns)


@functools.lru_cache(maxsize=4)
def _load_cached(file: str, mtime_ns: int) -> Bundle:
    obj = joblib.load(file)
    if not isinstance(obj, Bundle):
        raise ModelNotTrained(f"{file} is not a model bundle: run `ipo train`")
    for models in [obj.stages, *obj.alt_sets.values()]:
        for stage, sm in models.items():
            if sm.features != F.stage_features(stage):
                raise ModelNotTrained("the saved model was built with another feature set: run `ipo train`")
    return obj


def _has_value(live: LiveIPO, key: str) -> bool:
    sourced = live.sub.get(key)
    return sourced is not None and isinstance(sourced.value, (int, float)) and math.isfinite(float(sourced.value))


LAST_DAY_CAVEAT = ("issue still open: final subscription is usually higher (institutional bids arrive late), "
                   "so this understates demand")


def choose_stage(live: LiveIPO, today: date | None = None) -> tuple[str, list[str]]:
    """The stage the live state supports, with caveats about what is not used. On the last issue day with live
    category subscription but no day-wise snapshot, the 'close' stage is used on the live 'Now' values."""
    today = today or date.today()
    days = sorted(d for d, values in live.sub_by_day.items() if values)
    if live.status in CLOSED_STATUS:
        if F.live_gmp_pct(live) is not None:
            return "pre_listing", [GMP_CAVEAT]
        caveats = [] if any(_has_value(live, k) for k in SUB_KEYS) else [
            "issue closed but the final subscription is not published yet"]
        return "close", caveats
    if live.status == "open":
        notes = ["day-3 subscription is not used by the model"] if any(d >= 3 for d in days) else []
        usable = [d for d in days if d <= 2]
        if usable and usable[-1] == 2:
            return "day2", notes
        if usable and usable[-1] == 1:
            return "day1", notes
        last_day = live.close_date is not None and today >= live.close_date
        if last_day and any(_has_value(live, k) for k in SUB_KEYS):
            return "close", notes + [LAST_DAY_CAVEAT]
        return "pre_open", notes + ["issue is open but no day-1 or day-2 subscription yet: pre-open model applied"]
    return "pre_open", []


def _resolve(models: dict[str, StageModels], stage: str) -> tuple[str, str | None]:
    """The stage whose model will be used. A stage without its own model falls back to the latest earlier one."""
    if stage in models:
        return stage, None
    earlier = [s for s in models if F.stage_index(s) < F.stage_index(stage)]
    if not earlier:
        raise ModelNotTrained(f"no model covers stage {stage}: run `ipo train`")
    used = max(earlier, key=F.stage_index)
    return used, f"no model for stage {stage}; the {used} model was applied"


@functools.lru_cache(maxsize=64)
def _live_context_cached(market: str, as_of_iso: str, hour: str) -> tuple[tuple[tuple[str, float], ...], tuple[str, ...]]:
    """Public live_context, cached per market, day and hour (NSE is polled at most once an hour per market)."""
    try:
        raw = ctx_mod.live_context(market, today=date.fromisoformat(as_of_iso))
    except Exception as exc:  # unknown market or a source failure: missing context, never a crash
        return tuple((c, math.nan) for c in CONTEXT_COLS), (f"live context unavailable: {type(exc).__name__}",)
    values = tuple((c, float(raw[c]) if raw.get(c) is not None else math.nan) for c in CONTEXT_COLS)
    notes = [str(n) for n in raw.get("context_notes", [])]
    as_of = raw.get("context_as_of")
    notes.append(f"market context as of {as_of}" if as_of else "market context has no dated value")
    return values, tuple(notes)


def live_context_for(market: str, open_date: date | None = None) -> tuple[dict[str, float], list[str]]:
    """Index features for an IPO in `market`, as of the last close before the issue opens (or today, if the issue
    has not yet opened or its date is unknown). Uses the public data.context.live_context."""
    today = date.today()
    as_of = today if open_date is None else min(open_date, today)
    values, notes = _live_context_cached(market, as_of.isoformat(), datetime.now().strftime("%Y%m%d%H"))
    return dict(values), list(notes)


def _reliability(sm: StageModels, market: str, row: pd.DataFrame) -> list[str]:
    """Reasons the estimate should not be trusted as much as a backtest suggests. Empty means reliable."""
    reasons: list[str] = []
    n = sm.n_by_market.get(market, 0)
    if n == 0:
        reasons.append(f"market {market or 'unknown'} not seen in training for this stage")
    elif n < MIN_MARKET_ROWS:
        reasons.append(f"only {n} training IPOs for {market}")
    if market == "IN_SME" and sm.n_by_market.get("IN_SME", 0) < SME_MIN_ROWS:
        reasons.append(f"few SME IPOs in training ({sm.n_by_market.get('IN_SME', 0)})")
    missing = [F.CORE_LABELS.get(f, f) for f in sm.core.get(market, ()) if pd.isna(row[f].iloc[0])]
    if missing:
        reasons.append("missing core inputs: " + ", ".join(missing))
    outside = []
    for f, (lo, hi) in sm.ranges.get(market, {}).items():
        value = row[f].iloc[0]
        if not pd.isna(value) and (value < lo or value > hi):
            outside.append(f)
    if outside:
        reasons.append("outside the 1st-99th percentile of training: " + ", ".join(outside))
    return reasons


def _drivers(model, X: pd.DataFrame, z_pred: float, p50_pct: float) -> list[tuple[str, float]]:
    """Top contributions in percentage points. Approximation: pred_contrib on the transformed scale, rescaled so the
    drivers add up to the p50 move (p50 minus the bias term) in percentage points."""
    contrib = np.asarray(model.predict(X, pred_contrib=True))[0]
    bias = float(contrib[-1])
    base_pct = float(F.from_model_scale(bias))
    denom = z_pred - bias
    scale = (p50_pct - base_pct) / denom if abs(denom) > 1e-12 else 0.0
    pairs = [(name, float(c) * scale) for name, c in zip(X.columns, contrib[:-1])]
    pairs.sort(key=lambda kv: -abs(kv[1]))
    return [(name, round(value, 2)) for name, value in pairs[:TOP_DRIVERS] if abs(value) > 1e-9]


def predict(live: LiveIPO, as_of: str | None = None, bundle: Bundle | None = None,
            context: Mapping[str, float] | None = None, today: date | None = None) -> Prediction:
    """Estimate the listing gain for one live IPO. `as_of` overrides the stage chosen from the live state; `context`
    overrides the market context (default: live_context_for)."""
    b = bundle if bundle is not None else load_model()
    caveats: list[str] = []
    if as_of is None:
        stage, caveats = choose_stage(live, today)
    else:
        F.stage_index(as_of)  # raises ValueError for an unknown stage
        stage = as_of
    segment = segment_of(live.market)
    set_name = b.routing.get(segment) if segment else None
    if set_name not in (None, "pooled") and set_name not in b.alt_sets:
        caveats.append(f"model set {set_name} is not in the bundle; the pooled model was used")
    models = b.set_models(set_name)
    stage, fallback = _resolve(models, stage)
    if fallback:
        caveats.append(fallback)
    sm = models[stage]
    if context is None:
        context, notes = live_context_for(live.market, live.open_date)
        caveats += notes
    row = F.live_to_row(live, stage, b.vocab, context)
    if list(row.columns) != sm.features:
        raise ModelNotTrained("the input columns do not match the saved model: run `ipo train`")

    gmp_mode = b.design.get("gmp_model", "gbm") if stage == "pre_listing" else "gbm"
    comp = raw_components(sm, row, gmp_mode)
    lo, hi = adjust_intervals(sm.conf, comp["p10"], comp["p50"], comp["p90"], [segment])
    p10, p50, p90 = float(lo[0]), float(comp["p50"][0]), float(hi[0])
    prob = float(calibrate_prob(sm, comp["prob_raw"])[0])
    if sm.iso is None:
        caveats.append("probability not calibrated: too few out-of-fold points in training")
    if gmp_mode != "gbm" and stage == "pre_listing" and not comp["used_linear"][0]:
        caveats.append("linear GMP model needs GMP and total subscription; GBM used")
    if stage == "pre_listing" and live.issue_price is None and F.live_gmp_pct(live) is not None:
        caveats.append("GMP percent is measured against the top of the price band (issue price not final)")
    close = float(comp["close"][0])
    close = close if math.isfinite(close) else None

    skill = b.skill.get(stage, {}).get(segment) if segment else None
    skill_ok = skill is not None and skill["p50_beats_median"] and skill["brier_beats_base"]
    if skill is None:
        caveats.append(NO_SKILL_EVIDENCE)
    else:
        if not skill["p50_beats_median"]:
            caveats.append(SKILL_P50_CAVEAT)
        if not skill["brier_beats_base"]:
            caveats.append(SKILL_PROB_CAVEAT)
    reasons = _reliability(sm, live.market, row)
    caveats += reasons
    caveats.append(VALUATION_CAVEAT)
    z50 = float(sm.q50.predict(row)[0])
    return Prediction(
        ipo_id=live.ipo_id, as_of=stage,
        open_gain_pct_p10=p10, open_gain_pct_p50=p50, open_gain_pct_p90=p90,
        p_positive_open=prob, close_gain_pct_p50=close,
        drivers=_drivers(sm.q50, row, z50, float(comp["gbm"][0, 1])),
        baseline_gmp_pct=F.live_gmp_pct(live), model_version=b.version,
        reliable=not reasons and skill_ok, caveats=caveats,
    )
