"""Fitting one stage, and the prediction components shared by serving and the walk-forward evaluation.

Per stage: quantile regressors (p10, p50, p90) of the open gain on the transformed scale, a median regressor for the
close gain, and a classifier for 'opens above the issue price'. Calibration uses inner walk-forward predictions made
only from data listed before each inner block, so no calibrator sees the rows it is later judged on:
  - isotonic regression for the classifier probability;
  - conformal (CQR) adjustment of p10 and p90, per segment when a segment has enough residuals, else pooled.
pre_listing also gets three linear quantile regressions on GMP%, log total subscription and an IN_SME dummy; the
serving choice between GBM, linear and their average is set by the design.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import QuantileRegressor

from ipo_radar.model import features as F
from ipo_radar.model.bundle import Ensemble, StageModels
from ipo_radar.model.design import SEGMENTS, Design, segment_of

MIN_STAGE_ROWS = 30          # fewer open-gain rows than this and the stage gets no model
MIN_BLOCK_TRAIN = 60         # an inner calibration block needs at least this many earlier rows
MIN_CAL_POINTS = 30          # fewer out-of-fold points and the classifier stays uncalibrated
MIN_LINEAR_ROWS = 30         # GMP rows needed for the linear GMP model
MIN_RANGE_ROWS = 20          # fewer values and no 1st-99th percentile range is stored
MIN_SEG_RESID = 60           # out-of-fold residuals needed for a segment's own conformal adjustment
COVERAGE_MIN = 0.5           # a core input counts only if the market has it for at least this share of rows
CAL_BLOCKS = 3               # inner walk-forward blocks used for calibration
QUICK_CAL_BLOCKS = 2
NOMINAL_ALPHA = 0.2          # 80% intervals
MONOTONE_FEATURES = ("gmp_pct_last",)
LINEAR_COLS = ("gmp_pct_last", "log_sub_total", "is_in_sme")
QUANTILES = (("q10", 0.1), ("q50", 0.5), ("q90", 0.9))


def lgb_params(n_rows: int, seed: int, quick: bool, monotone: list[int] | None = None,
               params: dict | None = None) -> dict:
    """Conservative boosting by default: shallow trees, strong regularisation, leaves sized to the data. `params`
    (from the Optuna search) overrides these."""
    out = dict(
        n_estimators=40 if quick else 300, learning_rate=0.03, num_leaves=7, max_depth=3,
        min_child_samples=int(np.clip(n_rows // 40, 5, 40)), subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, deterministic=True, force_row_wise=True,
        n_jobs=1, verbose=-1,
    )
    if params:
        out.update(params)
        if "force_col_wise" in params:
            out.pop("force_row_wise", None)
    if monotone is not None:
        out["monotone_constraints"] = monotone
    return out


def monotone_vector(features: Sequence[str], enabled: bool) -> list[int] | None:
    """+1 (increasing) for every log subscription feature and GMP; free (0) for everything else."""
    if not enabled:
        return None
    return [1 if (f.startswith("log_sub_") or f in MONOTONE_FEATURES) else 0 for f in features]


def _xy(rows: pd.DataFrame, stage: str, vocab: F.Vocab, target: str) -> tuple[pd.DataFrame, pd.Series]:
    sub = rows[rows[target].notna()]
    return F.design(sub, stage, vocab), sub[target].astype(float)


def _quantile_level(scores: np.ndarray) -> float:
    """Conformal quantile at (1-alpha)(1+1/n), the finite-sample correction of CQR (Romano et al. 2019)."""
    n = len(scores)
    level = min(1.0, (1.0 - NOMINAL_ALPHA) * (1.0 + 1.0 / n))
    return float(np.quantile(scores, level, method="higher"))


def _fit_members(X: pd.DataFrame, z: np.ndarray, yb: np.ndarray, cX: pd.DataFrame | None, cz: np.ndarray | None,
                 mono: list[int] | None, seed: int, quick: bool, params: dict | None) -> dict:
    # LightGBM refuses monotone constraints with the quantile and L1 objectives, so they apply to the classifier only.
    kw = lgb_params(len(X), seed, quick, None, params)
    out: dict = {}
    for name, alpha in QUANTILES:
        out[name] = LGBMRegressor(objective="quantile", alpha=alpha, **kw).fit(X, z)
    out["clf"] = LGBMClassifier(**lgb_params(len(X), seed, quick, mono, params)).fit(X, yb)
    out["close"] = (LGBMRegressor(objective="quantile", alpha=0.5, **kw).fit(cX, cz)
                    if cX is not None else None)
    return out


def _segments(rows: pd.DataFrame) -> np.ndarray:
    return np.array([segment_of(m) for m in rows["market"].astype(str)], dtype=object)


def inner_oof(rows: pd.DataFrame, stage: str, vocab: F.Vocab, spec: Design, seeds: Sequence[int], quick: bool,
              params: dict | None) -> pd.DataFrame:
    """Out-of-fold predictions for the last inner walk-forward blocks of `rows` (one stage, open-gain rows).
    Columns: segment, year, y, p10, p50, p90 (percent, sorted, averaged over seeds), p_raw (classifier)."""
    o = rows[rows[F.TARGET_OPEN].notna()]
    years = o["listing_date"].dt.year
    n_blocks = QUICK_CAL_BLOCKS if quick else CAL_BLOCKS
    mono = monotone_vector(F.stage_features(stage), spec.monotone)
    parts = []
    for block in sorted(years.unique())[-n_blocks:]:
        tr, te = o[years < block], o[years == block]
        if len(tr) < MIN_BLOCK_TRAIN or te.empty:
            continue
        Xtr, ytr = F.design(tr, stage, vocab), tr[F.TARGET_OPEN].to_numpy(float)
        Xte, yte = F.design(te, stage, vocab), te[F.TARGET_OPEN].to_numpy(float)
        acc = np.zeros((len(te), 4))
        for seed in seeds:
            m = _fit_members(Xtr, F.to_model_scale(ytr), (ytr > 0).astype(int), None, None, mono, seed, quick, params)
            pct = np.sort(F.from_model_scale(np.column_stack([m["q10"].predict(Xte), m["q50"].predict(Xte),
                                                               m["q90"].predict(Xte)])), axis=1)
            acc[:, :3] += pct
            acc[:, 3] += m["clf"].predict_proba(Xte)[:, 1]
        acc /= len(seeds)
        parts.append(pd.DataFrame({"segment": _segments(te), "year": int(block), "y": yte,
                                   "p10": acc[:, 0], "p50": acc[:, 1], "p90": acc[:, 2], "p_raw": acc[:, 3]}))
    if not parts:
        return pd.DataFrame(columns=["segment", "year", "y", "p10", "p50", "p90", "p_raw"])
    return pd.concat(parts, ignore_index=True)


def conformal_from_oof(oof: pd.DataFrame) -> tuple[dict[str, float], dict[str, int]]:
    """Adjustment per segment (and ALL) from out-of-fold CQR scores: score = max(p10 - y, y - p90), positive when the
    outcome falls outside the raw interval. A segment gets its own adjustment only with MIN_SEG_RESID residuals."""
    if oof.empty:
        return {}, {}
    scores = np.maximum(oof["p10"].to_numpy(float) - oof["y"].to_numpy(float),
                        oof["y"].to_numpy(float) - oof["p90"].to_numpy(float))
    conf = {"ALL": _quantile_level(scores)}
    counts = {"ALL": len(scores)}
    segs = oof["segment"].to_numpy(object)
    for seg in SEGMENTS:
        s = scores[segs == seg]
        if len(s) >= MIN_SEG_RESID:
            conf[seg] = _quantile_level(s)
            counts[seg] = len(s)
    return conf, counts


def adjust_intervals(conf: dict[str, float], p10: np.ndarray, p50: np.ndarray, p90: np.ndarray,
                     segments: Sequence[str | None]) -> tuple[np.ndarray, np.ndarray]:
    """Widen (or narrow) the interval by the conformal adjustment; the median stays inside it."""
    if not conf:
        return p10, p90
    q = np.array([conf.get(s, conf["ALL"]) if s else conf["ALL"] for s in segments], dtype=float)
    return np.minimum(p10 - q, p50), np.maximum(p90 + q, p50)


def iso_from_oof(oof: pd.DataFrame) -> IsotonicRegression | None:
    if len(oof) < MIN_CAL_POINTS:
        return None
    t = (oof["y"].to_numpy(float) > 0).astype(float)
    if not 0 < t.sum() < len(t):
        return None
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(oof["p_raw"].to_numpy(float), t)


def _linear_matrix(X: pd.DataFrame, markets: np.ndarray) -> np.ndarray:
    return np.column_stack([X["gmp_pct_last"].to_numpy(float), X["log_sub_total"].to_numpy(float),
                            (markets == "IN_SME").astype(float)])


def fit_gmp_linear(rows: pd.DataFrame, stage: str, vocab: F.Vocab) -> list | None:
    """Linear quantile regressions (p10, p50, p90) of the open gain in percent on GMP%, log total subscription and an
    IN_SME dummy. GMP exists only for India, so there is no GLOBAL dummy. pre_listing only."""
    if stage != "pre_listing":
        return None
    X, y = _xy(rows, stage, vocab, F.TARGET_OPEN)
    ok = (X["gmp_pct_last"].notna() & X["log_sub_total"].notna()).to_numpy()
    if ok.sum() < MIN_LINEAR_ROWS:
        return None
    markets = rows.loc[X.index, "market"].astype(str).to_numpy()[ok]
    A = _linear_matrix(X[ok], markets)
    return [QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(A, y.to_numpy()[ok])
            for q in (0.1, 0.5, 0.9)]


def raw_components(sm: StageModels, X: pd.DataFrame, gmp_mode: str = "gbm") -> dict[str, np.ndarray]:
    """Open-gain quantiles (percent, sorted), calibrated-later probability and close median for design rows X.
    gmp_mode other than 'gbm' uses the linear GMP model where GMP and subscription are present."""
    z = np.column_stack([sm.q10.predict(X), sm.q50.predict(X), sm.q90.predict(X)])
    gbm = np.sort(F.from_model_scale(z), axis=1)
    pct = gbm.copy()
    used = np.zeros(len(X), dtype=bool)
    if gmp_mode in ("linear", "blend") and sm.gmp_linear is not None:
        ok = (X["gmp_pct_last"].notna() & X["log_sub_total"].notna()).to_numpy()
        if ok.any():
            markets = X["market"].astype(object).to_numpy()[ok]
            A = _linear_matrix(X[ok], markets)
            lin = np.sort(np.column_stack([m.predict(A) for m in sm.gmp_linear]), axis=1)
            pct[ok] = lin if gmp_mode == "linear" else 0.5 * (gbm[ok] + lin)
            used = ok
    close = (F.from_model_scale(sm.close_q50.predict(X)) if sm.close_q50 is not None
             else np.full(len(X), np.nan))
    return {"p10": pct[:, 0], "p50": pct[:, 1], "p90": pct[:, 2], "gbm": gbm,
            "prob_raw": sm.clf.predict_proba(X)[:, 1], "close": close, "used_linear": used}


def calibrate_prob(sm: StageModels, prob_raw: np.ndarray) -> np.ndarray:
    return sm.iso.predict(prob_raw) if sm.iso is not None else prob_raw


def gmp_counts(rows: pd.DataFrame) -> dict[str, dict[int, int]]:
    """GMP-bearing training IPOs (open-gain target present) per segment and listing year."""
    g = rows[rows[F.TARGET_OPEN].notna() & rows["gmp_pct_last"].notna()]
    out: dict[str, dict[int, int]] = {s: {} for s in SEGMENTS}
    for seg, year in zip(_segments(g), g["listing_date"].dt.year.astype(int)):
        out[seg][year] = out[seg].get(year, 0) + 1
    return out


def _core_and_ranges(X: pd.DataFrame, markets: np.ndarray, stage: str) -> tuple[dict, dict]:
    core: dict[str, tuple[str, ...]] = {}
    ranges: dict[str, dict[str, tuple[float, float]]] = {}
    wanted = F.core_features(stage)
    for m in sorted(set(markets)):
        mask = markets == m
        core[m] = tuple(f for f in wanted if f in X.columns and X.loc[mask, f].notna().mean() >= COVERAGE_MIN)
        ranges[m] = {}
        for f in X.columns:
            if not f.startswith(F.RANGE_PREFIXES):
                continue
            vals = X.loc[mask, f].dropna()
            if len(vals) >= MIN_RANGE_ROWS:
                ranges[m][f] = (float(vals.quantile(0.01)), float(vals.quantile(0.99)))
    return core, ranges


def fit_stage(rows: pd.DataFrame, stage: str, vocab: F.Vocab, spec: Design, seeds: Sequence[int], quick: bool,
              params: dict | None = None) -> StageModels | None:
    """All models for one stage. `rows` is asof_expand output restricted to that stage (and to the training set).
    One member per seed; the members are averaged at prediction time. None when too few open-gain rows."""
    open_rows = rows[rows[F.TARGET_OPEN].notna()]
    if len(open_rows) < MIN_STAGE_ROWS:
        return None
    X, y = _xy(rows, stage, vocab, F.TARGET_OPEN)
    z, yb = F.to_model_scale(y.to_numpy()), (y.to_numpy() > 0).astype(int)
    cX, cy = _xy(rows, stage, vocab, F.TARGET_CLOSE)
    close_ok = len(cX) >= MIN_STAGE_ROWS
    mono = monotone_vector(list(X.columns), spec.monotone)
    members = [_fit_members(X, z, yb, cX if close_ok else None,
                            F.to_model_scale(cy.to_numpy()) if close_ok else None, mono, seed, quick, params)
               for seed in seeds]

    def ens(key: str) -> Ensemble:
        return Ensemble([m[key] for m in members])

    oof = inner_oof(rows, stage, vocab, spec, seeds, quick, params)
    conf, conf_n = conformal_from_oof(oof) if spec.conformal else ({}, {})
    markets = open_rows["market"].astype(str).to_numpy()
    core, ranges = _core_and_ranges(X, markets, stage)
    return StageModels(
        stage=stage, features=list(X.columns), q10=ens("q10"), q50=ens("q50"), q90=ens("q90"),
        close_q50=ens("close") if close_ok else None, clf=ens("clf"), iso=iso_from_oof(oof),
        n_open=len(open_rows), n_close=len(cX), n_by_market={m: int((markets == m).sum()) for m in sorted(set(markets))},
        core=core, ranges=ranges, conf=conf, conf_n=conf_n,
        gmp_linear=fit_gmp_linear(rows, stage, vocab),
        gmp_counts=gmp_counts(rows) if stage == "pre_listing" else {},
    )
