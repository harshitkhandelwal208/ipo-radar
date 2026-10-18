"""Hyper-parameter search for the listing-gain model: Optuna over LightGBM, scored walk-forward.

Why a search at all: the table is a few thousand IPOs, so one LightGBM fit takes well under a second and the
search is cheap locally. A large search is still worth running (on Kaggle, see kaggle.py) because the
walk-forward folds and many trials give a far less lucky choice than a handful of hand-picked settings.

Scoring is the product's own yardstick: the pinball loss of the open-gain median (p50), plus a penalty for the
p10-p90 band missing its nominal 80% coverage. Folds are expanding windows in listing-date order, so a model is
only ever scored on IPOs that listed after the rows it trained on (never a random split).

Contracts used: features.asof_expand(df) (one row per IPO per stage, column as_of) and
features.make_xy(rows, target) -> (X, y) for the rows of one stage, one output row per input row, in order. The
search scores the pre_open stage only; the listing date and market are read by position from those rows.
"""
from __future__ import annotations

import importlib
import logging
import math
import time
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TARGET = "list_open_gain_pct"
SEARCH_STAGE = "pre_open"     # the stage shown before the issue opens; features.make_xy takes one stage at a time
DATE_COL = "listing_date"
QUANTILES = (0.1, 0.5, 0.9)
NOMINAL_COVERAGE = 0.8        # p10-p90 should contain 80% of listings
COVERAGE_WEIGHT = 20.0        # score points per unit of coverage gap: 0.05 gap costs 1 gain point
MIN_TRAIN_ROWS = 40
MIN_VALID_ROWS = 10

# Bounded search space. Every trial is drawn inside these ranges; the table test checks it.
SPACE: dict[str, tuple[Any, ...]] = {
    "num_leaves": ("int", 4, 48),
    "max_depth": ("int", 2, 8),
    "learning_rate": ("float_log", 0.01, 0.2),
    "n_estimators": ("int_step", 50, 400, 50),
    "min_child_samples": ("int", 5, 60),
    "subsample": ("float", 0.6, 1.0),
    "colsample_bytree": ("float", 0.5, 1.0),
    "reg_lambda": ("float_log", 1e-3, 10.0),
}
N_FOLDS = 4
SURVIVOR_TAG = "+px_join"     # rows whose `source` carries this were priced from a survivor-biased feed
FIXED_LGBM: dict[str, Any] = {"subsample_freq": 1, "n_jobs": 1, "deterministic": True, "force_col_wise": True,
                              "verbose": -1}


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss. Lower is better; same units as the target (gain percentage points)."""
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Share of outcomes inside [lo, hi]."""
    return float(np.mean((y >= lo) & (y <= hi)))


def lgbm_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    """Searched parameters plus the fixed ones. train() uses this too, so the final model is built like the trials."""
    return {**params, **FIXED_LGBM}


def build_regressor(params: dict[str, Any], alpha: float, seed: int):
    """One LightGBM quantile regressor. Imported lazily so the module imports without LightGBM installed."""
    import lightgbm as lgb

    return lgb.LGBMRegressor(objective="quantile", alpha=alpha, random_state=seed, **lgbm_kwargs(params))


def walk_forward_folds(dates: np.ndarray, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds. Fold k trains on blocks before block k and validates on block k.

    Blocks are cut over unique listing dates, so IPOs sharing a date never straddle a split and every training
    date is strictly before every validation date. Folds that are too small are dropped (logged).
    """
    uniq = np.unique(dates)
    if len(uniq) < n_folds + 1:
        raise ValueError(f"need at least {n_folds + 1} distinct listing dates for {n_folds} folds, got {len(uniq)}")
    rank = np.searchsorted(uniq, dates)
    block = np.minimum((rank * (n_folds + 1)) // len(uniq), n_folds)
    folds = []
    for k in range(1, n_folds + 1):
        train = np.flatnonzero(block < k)
        valid = np.flatnonzero(block == k)
        if len(train) < MIN_TRAIN_ROWS or len(valid) < MIN_VALID_ROWS:
            log.info("walk-forward fold %d skipped: %d train rows, %d validation rows", k, len(train), len(valid))
            continue
        folds.append((train, valid))
    if not folds:
        raise ValueError("no walk-forward fold has enough rows; the search needs more listed IPOs")
    return folds


def _suggest(trial, name: str, spec: tuple[Any, ...]):
    kind = spec[0]
    if kind == "int":
        return trial.suggest_int(name, spec[1], spec[2])
    if kind == "int_step":
        return trial.suggest_int(name, spec[1], spec[2], step=spec[3])
    if kind == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    if kind == "float_log":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    raise ValueError(f"unknown search kind {kind!r} for {name}")


def _features():
    return importlib.import_module("ipo_radar.model.features")


def _prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Features, target, listing dates and (market, survivor flag) for rows with a known target and listing date.

    features.make_xy drops rows without a target, so rows are matched back by their index label (asof_expand gives
    a unique index). A label that is not in the input is an error: it would mean the wrong row's date is used.
    """
    for col in (TARGET, DATE_COL):
        if col not in df.columns:
            raise ValueError(f"search needs column {col!r}")
    feats = _features()
    expanded = feats.asof_expand(df)
    rows = expanded[expanded["as_of"].astype(str) == SEARCH_STAGE]
    if not rows.index.is_unique:
        raise ValueError("features.asof_expand must return a unique index")
    X, y = feats.make_xy(rows, TARGET)
    position = rows.index.get_indexer(X.index)
    if (position < 0).any():
        raise ValueError("features.make_xy returned rows that were not in its input")
    picked = rows.iloc[position]
    dates = pd.to_datetime(picked[DATE_COL]).to_numpy()
    y_arr = np.asarray(y, dtype=float)
    keep = np.isfinite(y_arr) & ~pd.isna(dates)
    if keep.sum() < MIN_TRAIN_ROWS + MIN_VALID_ROWS:
        raise ValueError(f"only {int(keep.sum())} rows have a target and a listing date; need more")
    if (~keep).any():
        log.info("search skips %d rows without a target or listing date", int((~keep).sum()))
    meta = pd.DataFrame({
        "market": picked["market"].astype(str).to_numpy()[keep],
        "survivor": picked["source"].astype(str).str.contains(SURVIVOR_TAG, regex=False).to_numpy()[keep],
    })
    return X.iloc[np.flatnonzero(keep)], y_arr[keep], dates[keep], meta


def _evaluate(X, y: np.ndarray, folds, params: dict[str, Any], seed: int) -> dict[str, float]:
    """Pooled out-of-sample pinball (p50) and coverage over all validation rows of all folds."""
    ys, preds = [], {q: [] for q in QUANTILES}
    for train, valid in folds:
        Xtr, ytr = X.iloc[train], y[train]
        Xva = X.iloc[valid]
        for q in QUANTILES:
            model = build_regressor(params, q, seed)
            model.fit(Xtr, ytr)
            preds[q].append(np.asarray(model.predict(Xva), dtype=float))
        ys.append(y[valid])
    y_all = np.concatenate(ys)
    p = {q: np.concatenate(preds[q]) for q in QUANTILES}
    pin50 = pinball(y_all, p[0.5], 0.5)
    cov = coverage(y_all, p[0.1], p[0.9])
    score = pin50 + COVERAGE_WEIGHT * abs(cov - NOMINAL_COVERAGE)
    if not math.isfinite(score):
        raise RuntimeError("search produced a non-finite score; check the features for infinite values")
    return {"score": score, "pinball_p50": pin50, "coverage_p10_p90": cov, "n_valid": float(len(y_all))}


def search(df: pd.DataFrame, n_trials: int, seed: int, time_budget_s: float | None = None,
           n_folds: int = N_FOLDS) -> tuple[dict[str, Any], pd.DataFrame]:
    """Optuna (TPE, seeded) over SPACE. Returns (best searched params, one row per trial).

    The time budget stops the study between trials, so a run that hits it has fewer trials than n_trials; the
    returned table says how many completed. With no budget the result depends only on the seed and the data.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    import optuna

    X, y, dates, _ = _prepare(df)
    folds = walk_forward_folds(dates, n_folds)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))

    def objective(trial) -> float:
        params = {name: _suggest(trial, name, spec) for name, spec in SPACE.items()}
        metrics = _evaluate(X, y, folds, params, seed)
        for key, value in metrics.items():
            if key != "score":
                trial.set_user_attr(key, value)
        return metrics["score"]

    started = time.monotonic()
    study.optimize(objective, n_trials=n_trials, timeout=time_budget_s, show_progress_bar=False)
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not done:
        raise RuntimeError("no trial completed; raise the time budget or check the data")
    best = dict(study.best_trial.params)
    log.info("search: %d of %d trials completed in %.0f s, best score %.4f",
             len(done), len(study.trials), time.monotonic() - started, study.best_value)
    return best, trial_table(study)


def fold_counts(df: pd.DataFrame, n_folds: int = N_FOLDS) -> pd.DataFrame:
    """Rows per walk-forward fold, split (train or valid), market, and how many came from survivor-biased sources.

    Shown so a thin block is visible rather than hidden: a market with few validation rows in a fold says little
    about that fold. Folds are numbered in the order they are used (folds too small to score are not numbered).
    """
    _, _, dates, meta = _prepare(df)
    rows = []
    for number, (train, valid) in enumerate(walk_forward_folds(dates, n_folds), start=1):
        for split, idx in (("train", train), ("valid", valid)):
            part = meta.iloc[idx]
            for market, group in part.groupby("market"):
                rows.append({"fold": number, "split": split, "market": market, "rows": len(group),
                             "survivor_rows": int(group["survivor"].sum())})
    return pd.DataFrame(rows)


def trial_table(study) -> pd.DataFrame:
    """One row per trial: state, score, the fold metrics and the searched parameters (prefixed p_)."""
    rows = []
    for t in study.trials:
        row: dict[str, Any] = {"trial": t.number, "state": t.state.name,
                               "score": t.value if t.value is not None else math.nan}
        row.update(t.user_attrs)
        row.update({f"p_{k}": v for k, v in t.params.items()})
        rows.append(row)
    return pd.DataFrame(rows)
