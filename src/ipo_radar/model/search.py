"""Release choices made on the 2022-2024 walk-forward blocks only: shrinkage weights per stage, and a bounded Optuna
search over LightGBM settings. Nothing here reads 2025-2026.

Shrinkage: p50' = w * p50 + (1 - w) * segment training median, w in SHRINK_GRID, chosen by the pinball loss of the
p50 on the selection records (ties go to the larger w, so the model is kept unless shrinking clearly helps).

Search: TPE over learning_rate, num_leaves, min_child_samples, feature_fraction, bagging_fraction and reg_lambda, with
the default settings enqueued as trial 0. The objective is the mean pinball loss of p10, p50 and p90 over the 2022-2024
walk-forward blocks (pooled model, one seed, no calibration). Budget: at most SEARCH_TRIALS trials and SEARCH_BUDGET_S
seconds on CPU.
"""
from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor

from ipo_radar.model import features as F
from ipo_radar.model.design import SELECTION_YEARS
from ipo_radar.model.fitting import MIN_STAGE_ROWS, lgb_params
from ipo_radar.model.metrics import pinball

SHRINK_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
SEARCH_TRIALS = 40
SEARCH_BUDGET_S = 1800
DEFAULT_TUNING = {"learning_rate": 0.03, "num_leaves": 7, "min_child_samples": 40, "feature_fraction": 0.8,
                  "bagging_fraction": 0.8, "reg_lambda": 5.0}


def lgbm_overrides(tuning: dict) -> dict:
    """Search names to LightGBM names (the sklearn wrapper's colsample_bytree and subsample)."""
    out = dict(tuning)
    out["colsample_bytree"] = out.pop("feature_fraction")
    out["subsample"] = out.pop("bagging_fraction")
    return out


def choose_shrink(raw: pd.DataFrame) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Weight per stage (and the pinball loss of every grid weight) from raw selection records."""
    weights: dict[str, float] = {}
    table: dict[str, dict[str, float]] = {}
    rows = raw[raw["open_y"].notna()]
    for stage, rs in rows.groupby("stage", sort=False):
        y = rs["open_y"].to_numpy(float)
        p50 = rs["p50"].to_numpy(float)
        med = rs["med_open"].to_numpy(float)
        ok = np.isfinite(med) & np.isfinite(p50)
        if not ok.any():
            continue
        losses = {w: pinball(y[ok], w * p50[ok] + (1.0 - w) * med[ok], 0.5) for w in SHRINK_GRID}
        weights[stage] = min(SHRINK_GRID, key=lambda w: (round(losses[w], 9), -w))
        table[stage] = {f"{w:.2f}": losses[w] for w in SHRINK_GRID}
    return weights, table


def _cache(base: pd.DataFrame, years) -> list[tuple]:
    """Design matrices per (test year, stage), built once so every trial only refits."""
    expanded = F.asof_expand(base)
    items = []
    for year in years:
        vocab = F.fit_vocab(base[base["listing_date"].dt.year < year])
        for stage in F.STAGES:
            s = expanded[expanded["as_of"] == stage]
            yrs = s["listing_date"].dt.year
            tr = s[(yrs < year) & s[F.TARGET_OPEN].notna()]
            te = s[(yrs == year) & s[F.TARGET_OPEN].notna()]
            if len(tr) < MIN_STAGE_ROWS or te.empty:
                continue
            items.append((F.design(tr, stage, vocab), F.to_model_scale(tr[F.TARGET_OPEN].to_numpy(float)),
                          F.design(te, stage, vocab), te[F.TARGET_OPEN].to_numpy(float)))
    return items


def tuning_loss(tuning: dict, cache: list[tuple], seed: int) -> float:
    """Mean pinball loss of p10, p50 and p90 over the cached walk-forward blocks."""
    overrides = lgbm_overrides(tuning)
    total = 0.0
    weight = 0
    for Xtr, ztr, Xte, yte in cache:
        for alpha in (0.1, 0.5, 0.9):
            model = LGBMRegressor(objective="quantile", alpha=alpha,
                                  **lgb_params(len(Xtr), seed, False, None, overrides)).fit(Xtr, ztr)
            total += pinball(yte, F.from_model_scale(model.predict(Xte)), alpha) * len(yte)
        weight += 3 * len(yte)
    return total / weight


def search(base: pd.DataFrame, n_trials: int = SEARCH_TRIALS, timeout_s: float = SEARCH_BUDGET_S, seed: int = 7,
           years=SELECTION_YEARS) -> dict:
    """Best tuning on the selection blocks, with the default's loss for comparison and the full trial table."""
    cache = _cache(base, years)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(dict(DEFAULT_TUNING))

    def objective(trial: optuna.Trial) -> float:
        tuning = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        }
        return tuning_loss(tuning, cache, seed)

    study.optimize(objective, n_trials=n_trials, timeout=timeout_s, show_progress_bar=False)
    best = {k: study.best_params[k] for k in DEFAULT_TUNING}
    return {"best": best, "best_loss": float(study.best_value), "default_loss": float(study.trials[0].value),
            "n_trials": len(study.trials), "trials": [
                {"number": t.number, "loss": t.value, **t.params} for t in study.trials if t.value is not None]}
