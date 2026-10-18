"""The pre-2025 ablation that chose the design. Walk-forward blocks 2022-2024 only; blocks 2025-2026 are never touched
here. Items run in order and each kept choice carries forward:

  1. interval calibration (conformal), 2. monotone constraints, 3. GMP model at pre_listing (gbm / linear / blend),
  4. segment routing (pooled, India-only, one model per segment), 6. band-only price (leak fix; see record_leak_fix).

Every option's numbers are written to models/experiments.json, including the dropped ones. A choice is kept only if
its criterion improves on 2022-2024, and the criterion is stated next to it.

`python -m ipo_radar.model.experiments` runs items 1-4 and records the pre-fix baseline for item 6.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from ipo_radar import config
from ipo_radar.model import features as F
from ipo_radar.model import metrics
from ipo_radar.model.design import SEGMENTS, SELECTION_YEARS, Design
from ipo_radar.model.walkforward import run_walk

SEED = 7
ROUTING_OPTIONS = {
    "pooled": {"IN_MAIN": "pooled", "IN_SME": "pooled", "GLOBAL": "pooled"},
    "india": {"IN_MAIN": "india", "IN_SME": "india", "GLOBAL": None},
    "separate": {"IN_MAIN": "IN_MAIN", "IN_SME": "IN_SME", "GLOBAL": "GLOBAL"},
}


def _clean(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _by_stage(records: pd.DataFrame) -> dict:
    r = records
    out = {"ALL": metrics.summary(r)}
    for stage in F.STAGES:
        out[stage] = metrics.summary(r[r["stage"] == stage])
    return out


def _walk(base, spec: Design, seeds, quick, stages=F.STAGES) -> pd.DataFrame:
    records, _ = run_walk(base, spec, SELECTION_YEARS, seeds, quick, stages=stages)
    return records


def _coverage_gap(summary: dict) -> float:
    return abs(summary["coverage"] - 80.0)


def run(base: pd.DataFrame, seeds=(SEED,), quick: bool = False, out: Path | None = None) -> dict:
    out = out or config.MODELS / "experiments.json"
    steps: list[dict] = []
    spec = Design()

    # 1. interval calibration
    off, on = _walk(base, Design(conformal=False), seeds, quick), _walk(base, Design(conformal=True), seeds, quick)
    s_off, s_on = _by_stage(off), _by_stage(on)
    keep = _coverage_gap(s_on["ALL"]) < _coverage_gap(s_off["ALL"]) and s_on["ALL"]["pinball_mean"] <= s_off["ALL"]["pinball_mean"]
    spec.conformal = keep
    steps.append({"item": "1. interval calibration (conformal, CQR)", "criterion":
                  "keep if |coverage - 80| shrinks and mean pinball does not rise (2022-2024, all stages)",
                  "options": {"off": s_off, "on": s_on}, "kept": "on" if keep else "off"})

    # 2. monotone constraints. LightGBM allows them only on the classifier (not the quantile models), so the criterion
    # is the classifier's Brier score.
    off, on = _walk(base, spec, seeds, quick), _walk(base, Design(conformal=spec.conformal, monotone=True), seeds, quick)
    s_off, s_on = _by_stage(off), _by_stage(on)
    keep = s_on["ALL"]["brier"] < s_off["ALL"]["brier"]
    spec.monotone = keep
    steps.append({"item": "2. monotone constraints (increasing in log subscription and GMP%; free elsewhere)",
                  "criterion": "keep if the classifier's Brier score falls (2022-2024). Applies to the classifier only: "
                               "LightGBM rejects monotone constraints with the quantile objective",
                  "options": {"unconstrained": s_off, "monotone": s_on}, "kept": "monotone" if keep else "unconstrained"})

    # 3. GMP model at pre_listing, scored on the GMP-bearing rows only
    gmp_rows = lambda r: r["gmp_pct"].notna().to_numpy()  # noqa: E731
    options = {}
    for mode in ("gbm", "linear", "blend"):
        rec = _walk(base, Design(conformal=spec.conformal, monotone=spec.monotone, gmp_model=mode), seeds, quick,
                    stages=("pre_listing",))
        options[mode] = {"pre_listing_gmp_rows": metrics.summary(rec[gmp_rows(rec)]),
                         "pre_listing_all": metrics.summary(rec[rec["stage"] == "pre_listing"])}
    best = min(options, key=lambda m: (options[m]["pre_listing_gmp_rows"]["pinball_mean"], m != "gbm"))
    spec.gmp_model = best
    steps.append({"item": "3. GMP-aware pre_listing model (GBM, linear quantile regression, or their average)",
                  "criterion": "keep the option with the lowest mean pinball on GMP-bearing rows (2022-2024)",
                  "options": options, "kept": best})

    # 4. segment routing: the same design, routed three ways. Each segment keeps the option that scores best on it.
    per_option, routed = {}, {}
    for name, routing in ROUTING_OPTIONS.items():
        rec = _walk(base, Design(conformal=spec.conformal, monotone=spec.monotone, gmp_model=spec.gmp_model,
                                 routing=dict(routing)), seeds, quick)
        per_option[name] = rec
    choices: dict[str, str] = {}
    seg_options: dict[str, dict] = {}
    for seg in SEGMENTS:
        cands = [n for n, r in ROUTING_OPTIONS.items() if r.get(seg) is not None]
        scores = {}
        for n in cands:
            rec = per_option[n]
            sub = rec[rec["segment"] == seg]
            scores[n] = metrics.summary(sub)
        seg_options[seg] = scores
        choices[seg] = min(cands, key=lambda n: (scores[n].get("pinball_mean", math.inf), n != "pooled"))
        routed[seg] = ROUTING_OPTIONS[choices[seg]][seg]
    steps.append({"item": "4. segment ablation (pooled / India-only / one model per segment)",
                  "criterion": "per segment, keep the option with the lowest mean pinball (2022-2024, all stages)",
                  "options": {s: seg_options[s] for s in SEGMENTS}, "kept": choices})
    spec.routing = routed

    # 6 (pre-fix baseline): the chosen design with the features as they stand before the band-only price fix
    final = _walk(base, spec, seeds, quick)
    steps.append({"item": "6. band-only price for pre_open (leak fix)", "criterion":
                  "always kept: pre_open may use the price band only; recorded before and after",
                  "options": {"before": _by_stage(final)}, "kept": "pending"})

    result = {"selection_years": list(SELECTION_YEARS), "seeds": list(seeds), "quick": quick,
              "final_design": spec.as_dict(), "steps": steps}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_clean(result), indent=2, default=str), encoding="utf-8")
    return result


def record_leak_fix(base: pd.DataFrame, seeds=(SEED,), quick: bool = False, out: Path | None = None) -> dict:
    """Run the chosen design on the fixed features and fill in item 6's 'after' numbers."""
    out = out or config.MODELS / "experiments.json"
    result = json.loads(out.read_text(encoding="utf-8"))
    spec = Design(conformal=result["final_design"]["conformal"], monotone=result["final_design"]["monotone"],
                  gmp_model=result["final_design"]["gmp_model"], routing=dict(result["final_design"]["routing"]))
    after = _by_stage(_walk(base, spec, seeds, quick))
    for step in result["steps"]:
        if step["item"].startswith("6."):
            step["options"]["after"] = after
            step["kept"] = "always (fix)"
    out.write_text(json.dumps(_clean(result), indent=2, default=str), encoding="utf-8")
    return result


def main() -> None:
    base = pd.read_parquet(config.IPOS_PARQUET)
    base = base[base["listing_date"].notna()].reset_index(drop=True)
    result = run(base)
    for step in result["steps"]:
        print(step["item"], "->", step["kept"])


if __name__ == "__main__":
    main()
