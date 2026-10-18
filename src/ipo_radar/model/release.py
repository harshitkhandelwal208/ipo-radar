"""Release run, local CPU only: every choice is made on the 2022-2024 walk-forward blocks.

Steps: (1) the previous configuration on the previous and current data, (2) the bounded Optuna search, (3) the
previous configuration against the tuned one on 2022-2024 (single seed), which decides the parameters, (4) the final
bundle (5 seeds, shrinkage chosen inside train on 2022-2024), (5) a comparison file that the report reads.
2025-2026 is only scored once, for the record; it does not enter any choice.

`python -m ipo_radar.model.release` writes models/release_config.json, models/release_comparison.json and the bundle.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ipo_radar import config
from ipo_radar.model import features as F
from ipo_radar.model import metrics
from ipo_radar.model.bundle import Bundle
from ipo_radar.model.design import DEFAULT_DESIGN, LOCKED_YEARS, SELECTION_YEARS
from ipo_radar.model.search import SEARCH_BUDGET_S, SEARCH_TRIALS, lgbm_overrides, search
from ipo_radar.model.train import RELEASE_CONFIG, train
from ipo_radar.model.walkforward import apply_shrink, run_walk

RELEASE_SEEDS = [7, 8, 9, 10, 11]
PREVIOUS_DATA = config.MODELS / "local-archive" / "superseded-sub-zero" / "ipos.before-sub-zero.parquet"
COMPARISON = config.MODELS / "release_comparison.json"


def _clean(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _stage_blocks(records: pd.DataFrame) -> dict[str, dict]:
    """Metric block per stage and overall. The bootstrap is seeded so the numbers are reproducible."""
    rng = np.random.default_rng(metrics.BOOT_SEED)
    out = {"ALL": metrics.block(records, rng)}
    for stage in F.STAGES:
        sub = records[records["stage"] == stage]
        if len(sub):
            out[stage] = metrics.block(sub, rng)
    return out


def _selection(records: pd.DataFrame) -> pd.DataFrame:
    return records[records["year"].isin(SELECTION_YEARS)]


def run_release(new_df: pd.DataFrame, previous_df: pd.DataFrame, out_dir: Path = config.MODELS,
                n_trials: int | None = None, timeout_s: float | None = None) -> dict:
    n_trials = SEARCH_TRIALS if n_trials is None else n_trials
    timeout_s = SEARCH_BUDGET_S if timeout_s is None else timeout_s
    new = new_df[new_df["listing_date"].notna()].reset_index(drop=True)
    old = previous_df[previous_df["listing_date"].notna()].reset_index(drop=True)

    previous_old = _selection(run_walk(old, DEFAULT_DESIGN, SELECTION_YEARS, [7], False)[0])
    previous_new = _selection(run_walk(new, DEFAULT_DESIGN, SELECTION_YEARS, [7], False)[0])
    found = search(new, n_trials=n_trials, timeout_s=timeout_s, seed=7)
    tuned_overrides = lgbm_overrides(found["best"])
    tuned_single = _selection(run_walk(new, DEFAULT_DESIGN, SELECTION_YEARS, [7], False, tuned_overrides)[0])
    prev_loss = metrics.summary(previous_new)["pinball_mean"]
    tuned_loss = metrics.summary(tuned_single)["pinball_mean"]
    use_tuned = tuned_loss < prev_loss
    params = tuned_overrides if use_tuned else {}
    config_file = {"params": params, "seeds": RELEASE_SEEDS, "params_decision": "tuned" if use_tuned else "default",
                   "search_best": found["best"], "search_default_loss": found["default_loss"],
                   "search_best_loss": found["best_loss"], "search_trials": found["n_trials"],
                   "single_seed_pinball": {"default": prev_loss, "tuned": tuned_loss}}
    RELEASE_CONFIG.write_text(json.dumps(config_file, indent=2), encoding="utf-8")

    report = train(new, out_dir=out_dir, params=params, seeds=RELEASE_SEEDS)
    selection = pd.read_parquet(out_dir / "selection_records.parquet")
    latest = pd.read_parquet(out_dir / "latest_records.parquet")
    bundle = joblib.load(out_dir / "bundle.joblib")
    assert isinstance(bundle, Bundle)
    after = apply_shrink(selection, bundle.shrink)
    result = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bundle_version": report.version,
        "selection_years": list(SELECTION_YEARS),
        "previous_config_previous_data": {"single_seed": True, "stages": _stage_blocks(previous_old)},
        "previous_config_current_data": {"single_seed": True, "stages": _stage_blocks(previous_new)},
        "tuned_config_current_data": {"single_seed": True, "stages": _stage_blocks(tuned_single)},
        "after_release": {"seeds": RELEASE_SEEDS, "shrink": bundle.shrink, "stages": _stage_blocks(after)},
        "parameters": {"decision": config_file["params_decision"], "chosen": params,
                       "search_best": found["best"], "search_default_loss": found["default_loss"],
                       "search_best_loss": found["best_loss"], "search_trials": found["n_trials"],
                       "trial_table": found["trials"]},
        "shrink": {"weights": bundle.shrink, "pinball_by_weight": report.shrink_table},
        "latest_blocks": {"years": list(LOCKED_YEARS), "note": "scored once for the record, not used in any choice",
                          "stages": _stage_blocks(apply_shrink(latest, bundle.shrink))},
    }
    COMPARISON.write_text(json.dumps(_clean(result), indent=2, default=str), encoding="utf-8")
    return result


def main() -> None:
    previous = pd.read_parquet(PREVIOUS_DATA)
    new = pd.read_parquet(config.IPOS_PARQUET)
    result = run_release(new, previous)
    print(f"bundle {result['bundle_version']}; parameters {result['parameters']['decision']}; "
          f"shrink {result['shrink']['weights']}")


if __name__ == "__main__":
    main()
