"""Scoring of walk-forward records: pinball loss, bootstrap intervals on paired differences, block metrics, calibration
bins, and the per-stage skill flags stored in the bundle.

Records have one row per (test IPO, stage) with the model's p10/p50/p90 (percent), p_pos (calibrated probability),
the segment-specific training median and base rate, and the GMP and ridge benchmarks where they exist.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

BOOT_RESAMPLES = 2000
BOOT_SEED = 11
MIN_VERDICT_ROWS = 30
QUANTILE_COLS = ((0.1, "p10"), (0.5, "p50"), (0.9, "p90"))


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1.0) * d)))


def ci(diff: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float] | None:
    """Mean of paired differences with a 95% bootstrap interval over IPOs. None when there are too few rows."""
    if len(diff) < MIN_VERDICT_ROWS:
        return None
    idx = rng.integers(0, len(diff), size=(BOOT_RESAMPLES, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(diff.mean()), float(lo), float(hi)


def verdict(interval: tuple[float, float, float] | None, n: int) -> str:
    if interval is None or n < MIN_VERDICT_ROWS:
        return "too few rows to judge"
    _, lo, hi = interval
    if hi < 0:
        return "beats it (CI entirely below zero)"
    if lo > 0:
        return "worse than it (CI entirely above zero)"
    return "no clear difference (CI spans zero)"


def open_rows(r: pd.DataFrame) -> pd.DataFrame:
    return r[r["open_y"].notna()]


def summary(r: pd.DataFrame) -> dict:
    """Plain numbers for experiments: MAE, coverage, mean pinball, classifier scores (open-gain rows only)."""
    o = open_rows(r)
    if o.empty:
        return {"n": 0}
    y = o["open_y"].to_numpy(float)
    yb = (y > 0).astype(int)
    pins = [pinball(y, o[c].to_numpy(float), a) for a, c in QUANTILE_COLS]
    out = {
        "n": int(len(o)), "mae": float(np.abs(y - o["p50"].to_numpy(float)).mean()),
        "mae_median": float(np.abs(y - o["med_open"].to_numpy(float)).mean()),
        "coverage": float(((y >= o["p10"].to_numpy(float)) & (y <= o["p90"].to_numpy(float))).mean() * 100),
        "pinball_mean": float(np.mean(pins)),
        "brier": float(brier_score_loss(yb, o["p_pos"].to_numpy(float))),
        "brier_base": float(np.mean((yb - o["base_pos"].to_numpy(float)) ** 2)),
    }
    out["auc"] = float(roc_auc_score(yb, o["p_pos"].to_numpy(float))) if 0 < yb.sum() < len(yb) else math.nan
    return out


def block(r: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Full metric block for the report: summary plus bootstrap intervals against the baselines."""
    o = open_rows(r)
    out: dict = {"n": len(o)}
    if o.empty:
        return out
    base = summary(r)
    out.update(base)
    y = o["open_y"].to_numpy(float)
    err = np.abs(y - o["p50"].to_numpy(float))
    out["pinball"] = {str(a): pinball(y, o[c].to_numpy(float), a) for a, c in QUANTILE_COLS}
    out["mae_zero"] = float(np.abs(y).mean())
    out["diff_median"] = ci(err - np.abs(y - o["med_open"].to_numpy(float)), rng)
    out["diff_zero"] = ci(err - np.abs(y), rng)
    yb = (y > 0).astype(float)
    p = o["p_pos"].to_numpy(float)
    out["diff_brier"] = ci((yb - p) ** 2 - (yb - o["base_pos"].to_numpy(float)) ** 2, rng)
    g = o[o["gmp_pct"].notna()]
    out["n_gmp"] = len(g)
    if len(g):
        gy = g["open_y"].to_numpy(float)
        gerr = np.abs(gy - g["p50"].to_numpy(float))
        ggmp = np.abs(gy - g["gmp_pct"].to_numpy(float))
        out["mae_gmp_rows"], out["mae_gmp"] = float(gerr.mean()), float(ggmp.mean())
        out["diff_gmp"] = ci(gerr - ggmp, rng)
    lin = o[o["lin_open"].notna()]
    out["n_lin"] = len(lin)
    if len(lin):
        ly = lin["open_y"].to_numpy(float)
        lerr = np.abs(ly - lin["p50"].to_numpy(float))
        llin = np.abs(ly - lin["lin_open"].to_numpy(float))
        out["mae_lin_rows"], out["mae_lin"] = float(lerr.mean()), float(llin.mean())
        out["diff_lin"] = ci(lerr - llin, rng)
    c = r[r["close_y"].notna() & r["close_p50"].notna()]
    out["n_close"] = len(c)
    if len(c):
        cy = c["close_y"].to_numpy(float)
        cerr = np.abs(cy - c["close_p50"].to_numpy(float))
        cmed = np.abs(cy - c["med_close"].to_numpy(float))
        out["mae_close"], out["mae_close_median"] = float(cerr.mean()), float(cmed.mean())
        out["diff_close_median"] = ci(cerr - cmed, rng)
    return out


def calibration(r: pd.DataFrame) -> list[tuple[str, int, float, float]]:
    """Ten equal-width bins of the predicted probability: (label, n, mean predicted %, observed %)."""
    o = open_rows(r)
    edges = np.linspace(0, 1, 11)
    bins = np.clip(np.digitize(o["p_pos"].to_numpy(float), edges[1:-1]), 0, 9)
    yb = (o["open_y"].to_numpy(float) > 0).astype(float)
    p = o["p_pos"].to_numpy(float)
    rows = []
    for b in range(10):
        m = bins == b
        n = int(m.sum())
        rows.append((f"{edges[b] * 100:.0f}-{edges[b + 1] * 100:.0f}%", n,
                     float(p[m].mean()) * 100 if n else math.nan, float(yb[m].mean()) * 100 if n else math.nan))
    return rows


def skill(r: pd.DataFrame, seed: int = BOOT_SEED) -> dict[str, dict[str, dict]]:
    """Per stage and segment: does the p50 beat the segment's training median, and does the classifier beat the
    segment's training base rate (Brier)? A flag is True only when the 95% interval of the difference lies below zero.
    Segments with too few rows get no entry."""
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, dict]] = {}
    for (stage, segment), rs in r.groupby(["stage", "segment"], sort=False):
        o = open_rows(rs)
        if len(o) < MIN_VERDICT_ROWS:
            continue
        y = o["open_y"].to_numpy(float)
        err = np.abs(y - o["p50"].to_numpy(float))
        d_med = ci(err - np.abs(y - o["med_open"].to_numpy(float)), rng)
        yb = (y > 0).astype(float)
        p = o["p_pos"].to_numpy(float)
        d_brier = ci((yb - p) ** 2 - (yb - o["base_pos"].to_numpy(float)) ** 2, rng)
        s = summary(rs)
        out.setdefault(stage, {})[segment] = {
            "n": int(len(o)),
            "p50_beats_median": bool(d_med is not None and d_med[2] < 0),
            "mae_diff": list(d_med) if d_med else None,
            "brier_beats_base": bool(d_brier is not None and d_brier[2] < 0),
            "brier_diff": list(d_brier) if d_brier else None,
            "auc": s.get("auc"),
        }
    return out
