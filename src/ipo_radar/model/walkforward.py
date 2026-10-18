"""Walk-forward engine: for each test year, models trained only on IPOs listed before that year predict the IPOs listed
in it. Used by the experiments, by the skill flags saved in the bundle, and by the evaluation report.

Baselines are per segment and do not depend on the routing, so every design is scored against the same yardsticks:
the segment's training median (open and close), its training base rate, a ridge on log total subscription, and the
GMP reading itself (where one exists).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ipo_radar.model import features as F
from ipo_radar.model.design import SEGMENTS, SET_MARKETS, Design, segment_of
from ipo_radar.model.fitting import adjust_intervals, calibrate_prob, fit_stage, raw_components, shrink_quantiles

LIN_FEATURE = {"day1": "log_sub_total_d1", "day2": "log_sub_total_d2", "close": "log_sub_total",
               "pre_listing": "log_sub_total"}
RIDGE_ALPHA = 10.0
MIN_RIDGE_ROWS = 30
RECORD_COLS = ["stage", "year", "market", "segment", "set", "ipo_id", "open_y", "close_y", "p10", "p50", "p90",
               "p_pos", "close_p50", "med_open", "med_close", "base_pos", "lin_open", "gmp_pct", "used_linear"]


def expanded_frame(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The base table with a row id and segment, and its one-row-per-stage expansion."""
    base = base.reset_index(drop=True).copy()
    base["_row"] = np.arange(len(base))
    base["segment"] = [segment_of(m) for m in base["market"].astype(str)]
    return base, F.asof_expand(base)


def _ridge(tr: pd.DataFrame, te: pd.DataFrame, stage: str, vocab: F.Vocab) -> np.ndarray:
    """Ridge on log total subscription (the stage's own total snapshot). NaN where the stage has no such input."""
    out = np.full(len(te), np.nan)
    feat = LIN_FEATURE.get(stage)
    if feat is None:
        return out
    tr_open = tr[tr[F.TARGET_OPEN].notna()]
    x_tr = F.design(tr_open, stage, vocab)[[feat]]
    ok = x_tr[feat].notna().to_numpy()
    if ok.sum() < MIN_RIDGE_ROWS:
        return out
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(x_tr[ok], F.to_model_scale(tr_open[F.TARGET_OPEN].to_numpy(float)[ok]))
    x_te = F.design(te, stage, vocab)[[feat]]
    has = x_te[feat].notna().to_numpy()
    if has.any():
        out[has] = F.from_model_scale(model.predict(x_te[has]))
    return out


def baselines(tr_all: pd.DataFrame, te: pd.DataFrame, stage: str, vocab: F.Vocab) -> pd.DataFrame:
    """Per test row: segment training median (open, close), base rate of opening above the issue price, and the
    ridge prediction. Only rows of the same segment as the training rows are used."""
    out = pd.DataFrame(index=te.index, columns=["med_open", "med_close", "base_pos", "lin_open"], dtype=float)
    for seg in SEGMENTS:
        t = te[te["segment"] == seg]
        tr = tr_all[tr_all["segment"] == seg]
        if t.empty or tr.empty:
            continue
        o = tr[tr[F.TARGET_OPEN].notna()][F.TARGET_OPEN].to_numpy(float)
        c = tr[tr[F.TARGET_CLOSE].notna()][F.TARGET_CLOSE].to_numpy(float)
        out.loc[t.index, "med_open"] = np.median(o) if len(o) else np.nan
        out.loc[t.index, "med_close"] = np.median(c) if len(c) else np.nan
        out.loc[t.index, "base_pos"] = float((o > 0).mean()) if len(o) else np.nan
        out.loc[t.index, "lin_open"] = _ridge(tr, t, stage, vocab)
    return out


def run_walk(base: pd.DataFrame, spec: Design, years: Sequence[int], seeds: Sequence[int], quick: bool,
             params: dict | None = None, stages: Sequence[str] = F.STAGES, keep_models: bool = False
             ) -> tuple[pd.DataFrame, dict]:
    """Out-of-sample records for each test year, stage and routed model set. Returns (records, models) where models
    maps (set, stage, year) to the fitted StageModels when keep_models is True."""
    base, expanded = expanded_frame(base)
    recs: list[pd.DataFrame] = []
    models: dict = {}
    for year in years:
        vocab = F.fit_vocab(base[base["listing_date"].dt.year < year])
        for stage in stages:
            rows = expanded[expanded["as_of"] == stage]
            listed = rows["listing_date"].dt.year
            tr_all, te_all = rows[listed < year], rows[listed == year]
            if te_all.empty:
                continue
            for set_name in spec.sets_used():
                segs = [g for g, name in spec.routing.items() if name == set_name]
                te = te_all[te_all["segment"].isin(segs)]
                if not segs or te.empty:
                    continue
                tr = tr_all[tr_all["market"].astype(str).isin(SET_MARKETS[set_name])]
                sm = fit_stage(tr, stage, vocab, spec, seeds, quick, params)
                if sm is None:
                    continue
                if keep_models:
                    models[(set_name, stage, year)] = sm
                comp = raw_components(sm, F.design(te, stage, vocab),
                                      spec.gmp_model if stage == "pre_listing" else "gbm")
                lo, hi = adjust_intervals(sm.conf, comp["p10"], comp["p50"], comp["p90"],
                                          te["segment"].tolist())
                bl = baselines(tr_all, te, stage, vocab)
                gmp = base["gmp_pct_last"].to_numpy(float)[te["_row"].to_numpy()]
                recs.append(pd.DataFrame({
                    "stage": stage, "year": year, "market": te["market"].astype(str).to_numpy(),
                    "segment": te["segment"].to_numpy(object), "set": set_name,
                    "ipo_id": te["ipo_id"].astype(str).to_numpy(), "open_y": te[F.TARGET_OPEN].to_numpy(float),
                    "close_y": te[F.TARGET_CLOSE].to_numpy(float), "p10": lo, "p50": comp["p50"], "p90": hi,
                    "p_pos": calibrate_prob(sm, comp["prob_raw"]), "close_p50": comp["close"],
                    "med_open": bl["med_open"].to_numpy(float), "med_close": bl["med_close"].to_numpy(float),
                    "base_pos": bl["base_pos"].to_numpy(float), "lin_open": bl["lin_open"].to_numpy(float),
                    "gmp_pct": gmp, "used_linear": comp["used_linear"],
                }))
    records = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame(columns=RECORD_COLS)
    return records, models


def apply_shrink(records: pd.DataFrame, shrink: dict[str, float]) -> pd.DataFrame:
    """Records with the open-gain median blended toward each segment's training median (fitting.shrink_quantiles).
    Stages without a weight keep w = 1. Close-gain predictions are unchanged."""
    if records.empty or not shrink:
        return records.copy()
    out = records.copy()
    w = out["stage"].map(lambda s: shrink.get(s, 1.0)).to_numpy(float)
    med = out["med_open"].to_numpy(float)
    p10, p50, p90 = (out[c].to_numpy(float) for c in ("p10", "p50", "p90"))
    ok = np.isfinite(med) & np.isfinite(p50)
    n10, n50, n90 = shrink_quantiles(p10, p50, p90, w, np.where(ok, med, p50))
    keep = ~ok | (w == 1.0)
    out["p10"] = np.where(keep, p10, n10)
    out["p50"] = np.where(keep, p50, n50)
    out["p90"] = np.where(keep, p90, n90)
    return out
