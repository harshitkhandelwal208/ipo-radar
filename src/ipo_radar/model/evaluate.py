"""Walk-forward evaluation of the chosen design -> docs/MODEL_REPORT.md.

The protocol is at the top of the report. Blocks 2022-2024 made every choice (model/experiments.py) and set the skill
flags. Blocks 2025-2026 are locked: the chosen design is evaluated on them once and shown as the headline next to
2023-2024. Each block is predicted by models trained only on IPOs listed strictly before it.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ipo_radar import config
from ipo_radar.model import features as F
from ipo_radar.model import metrics
from ipo_radar.model.design import LOCKED_YEARS, PROTOCOL, SELECTION_YEARS, segment_of
from ipo_radar.model.train import data_hash
from ipo_radar.model.bundle import Bundle
from ipo_radar.model.predict import load_model
from ipo_radar.model.search import SHRINK_GRID
from ipo_radar.model.walkforward import apply_shrink

REPORT_YEARS = tuple(SELECTION_YEARS) + tuple(LOCKED_YEARS)
SHOW_YEARS = (2023, 2024)
REPORT_MARKETS = ("IN_MAIN", "IN_SME", "US", "HK", "OTHER")
LOCKED_NOTE = (
    "2025-2026 is no longer a holdout: it has been used for serving caveats and reviewed repeatedly. It is reported "
    "once, for the record, and no choice was made from it. No clean holdout remains until new IPOs are tracked after "
    "release.")
PREVIOUS_HEADLINE = config.MODELS / "superseded-zero-fix" / "locked_headline.json"
BUILD_COVERAGE = config.PROCESSED / "COVERAGE.md"


@dataclass
class EvalResult:
    report_path: str
    test_years: list[int]
    n_test_rows: int
    data_hash: str
    verdicts: dict[str, str] = field(default_factory=dict)
    skill: dict = field(default_factory=dict)


def _table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _f(x: object, nd: int = 1, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{float(x):+.{nd}f}" if sign else f"{float(x):.{nd}f}"


def _ci(interval) -> str:
    if interval is None:
        return "n/a"
    return f"{interval[0]:+.1f} [{interval[1]:+.1f}, {interval[2]:+.1f}]"


def _headline(rec: pd.DataFrame, rng: np.random.Generator) -> dict[str, dict]:
    out = {"ALL": metrics.block(rec, rng)}
    for stage in F.STAGES:
        sub = rec[rec["stage"] == stage]
        if len(sub):
            out[stage] = metrics.block(sub, rng)
    return out


def _headline_table(blocks: dict[str, dict], skill: dict) -> str:
    rows = []
    for stage in [*F.STAGES, "ALL"]:
        b = blocks.get(stage)
        if not b or not b.get("n"):
            continue
        s = skill.get(stage, {})
        p50_ok = [g for g, v in s.items() if v["p50_beats_median"]]
        prob_ok = [g for g, v in s.items() if v["brier_beats_base"]]
        rows.append([
            stage, b["n"], _f(b.get("mae")), _f(b.get("mae_median")), _ci(b.get("diff_median")),
            _f(b.get("mae_zero")), b.get("n_gmp", 0), _f(b.get("mae_gmp_rows")) if b.get("n_gmp") else "n/a",
            _f(b.get("mae_gmp")) if b.get("n_gmp") else "n/a",
            _ci(b.get("diff_gmp")) if b.get("n_gmp") else "n/a",
            _f(b.get("coverage"), 0), _f(b.get("auc"), 3), _f(b.get("brier"), 3), _f(b.get("brier_base"), 3),
            ", ".join(p50_ok) or "none" if s else "n/a", ", ".join(prob_ok) or "none" if s else "n/a",
        ])
    return _table(["stage", "n", "MAE p50", "MAE median", "p50 - median (95% CI)", "MAE zero", "n with GMP",
                   "MAE model (GMP rows)", "MAE GMP", "model - GMP (CI)", "coverage p10-p90 %", "AUC", "Brier",
                   "Brier base rate", "2022-24 skill: p50 beats median (segments)",
                   "2022-24 skill: prob beats base (segments)"], rows)


def _close_table(blocks: dict[str, dict]) -> str:
    rows = []
    for stage in [*F.STAGES, "ALL"]:
        b = blocks.get(stage, {})
        if b.get("n_close"):
            rows.append([stage, b["n_close"], _f(b.get("mae_close")), _f(b.get("mae_close_median")),
                         _ci(b.get("diff_close_median")), metrics.verdict(b.get("diff_close_median"), b["n_close"])])
    if not rows:
        return "No close-gain rows."
    return _table(["stage", "n (close)", "MAE close model", "MAE close median", "model - median (CI)", "verdict"], rows)


def _segment_skill_table(skill: dict) -> str:
    rows = []
    for stage in F.STAGES:
        for segment, s in skill.get(stage, {}).items():
            rows.append([stage, segment, s["n"], _ci(s["mae_diff"]), "yes" if s["p50_beats_median"] else "no",
                         _ci(s["brier_diff"]), "yes" if s["brier_beats_base"] else "no", _f(s["auc"], 3)])
    if not rows:
        return "No segment has enough 2022-2024 rows for a skill flag."
    return _table(["stage", "segment", "n", "model - median MAE (CI)", "p50 beats median", "Brier - base rate (CI)",
                   "prob beats base rate", "AUC"], rows)


def _calibration_table(rec: pd.DataFrame) -> str:
    parts = []
    for stage in F.STAGES:
        sub = rec[rec["stage"] == stage]
        if sub.empty:
            continue
        rows = [[b, n, _f(p, 1), _f(o, 1)] for b, n, p, o in metrics.calibration(sub)]
        parts += [f"### {stage}", "", _table(["predicted bin", "n", "mean predicted %", "observed opened above %"], rows),
                  ""]
    return "\n".join(parts)


def _block_table(rec: pd.DataFrame) -> str:
    rows = []
    for year in sorted(rec["year"].unique()):
        for stage in F.STAGES:
            sub = rec[(rec["year"] == year) & (rec["stage"] == stage)]
            s = metrics.summary(sub)
            if s.get("n"):
                rows.append([int(year), stage, s["n"], _f(s["mae"]), _f(s["mae_median"]), _f(s["coverage"], 0),
                             _f(s["auc"], 3), _f(s["brier"], 3), _f(s["brier_base"], 3)])
    return _table(["listing year", "stage", "n", "MAE p50", "MAE median", "coverage %", "AUC", "Brier",
                   "Brier base rate"], rows)


def _markets_table(rec: pd.DataFrame, rng: np.random.Generator) -> str:
    rows = []
    for stage in F.STAGES:
        for m in REPORT_MARKETS:
            sub = rec[(rec["stage"] == stage) & (rec["market"] == m)]
            if sub.empty:
                continue
            b = metrics.block(sub, rng)
            if b.get("n"):
                rows.append([stage, m, b["n"], _f(b["mae"]), _f(b["mae_median"]), _ci(b["diff_median"]),
                             _f(b["pinball"]["0.5"]), _f(b["coverage"], 0), _f(b["auc"], 3)])
    return _table(["stage", "market", "n", "MAE p50", "MAE median", "p50 - median (CI)", "pinball p50",
                   "coverage %", "AUC"], rows)


def _experiments_section(path: Path) -> list[str]:
    if not path.exists():
        return ["The experiment file is missing: run `python -m ipo_radar.model.experiments`."]
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = [f"Selection blocks {', '.join(map(str, data['selection_years']))}. Every option is listed, kept or "
             "dropped. Numbers are pooled over the selection blocks unless stated.", ""]
    for step in data["steps"]:
        lines += [f"### {step['item']}", "", f"Criterion: {step['criterion']}. **Kept: {step['kept']}**.", ""]
        opts = step["options"]
        if step["item"].startswith("4."):
            rows = []
            for seg, by_opt in opts.items():
                for name, s in by_opt.items():
                    rows.append([seg, name, s.get("n", 0), _f(s.get("mae")), _f(s.get("coverage"), 0),
                                 _f(s.get("pinball_mean"), 3)])
            lines += [_table(["segment", "routing option", "n", "MAE p50", "coverage %", "mean pinball"], rows), ""]
        elif step["item"].startswith("3."):
            rows = [[name, o["pre_listing_gmp_rows"].get("n", 0), _f(o["pre_listing_gmp_rows"].get("mae")),
                     _f(o["pre_listing_gmp_rows"].get("coverage"), 0),
                     _f(o["pre_listing_gmp_rows"].get("pinball_mean"), 3)] for name, o in opts.items()]
            lines += [_table(["option", "GMP rows", "MAE p50", "coverage %", "mean pinball"], rows), ""]
        else:
            rows = []
            for name, o in opts.items():
                s = o.get("ALL", o)
                rows.append([name, s.get("n", 0), _f(s.get("mae")), _f(s.get("coverage"), 0),
                             _f(s.get("pinball_mean"), 3), _f(s.get("brier"), 3), _f(s.get("auc"), 3)])
            lines += [_table(["option", "n", "MAE p50", "coverage %", "mean pinball", "Brier", "AUC"], rows), ""]
    return lines


def _gmp_counts(base: pd.DataFrame) -> str:
    g = base[base[F.TARGET_OPEN].notna() & base["gmp_pct_last"].notna()].copy()
    g["segment"] = [segment_of(m) for m in g["market"].astype(str)]
    g["year"] = g["listing_date"].dt.year.astype(int)
    rows = []
    for seg in ("IN_MAIN", "IN_SME", "GLOBAL"):
        sub = g[g["segment"] == seg]
        by_year = sub["year"].value_counts().sort_index()
        rows.append([seg, len(sub), len(sub[sub["year"] < 2025]),
                     ", ".join(f"{y}: {n}" for y, n in by_year.items()) or "none"])
    return _table(["segment", "GMP-bearing rows (all years)", "before 2025", "rows per listing year"], rows)


def _gain_table(bundle: Bundle) -> str:
    """Gain share per feature group of the final pooled median models. In-sample: the release bundle is trained on all
    IPOs, so this shows what the model uses, not how well it generalises (permutation needs a holdout fit)."""
    per_stage = {}
    for stage, sm in bundle.stages.items():
        raw = sm.q50.booster_.feature_importance(importance_type="gain")
        total = float(raw.sum()) or 1.0
        gains = {g: 0.0 for g in F.GROUPS}
        for name, value in zip(sm.features, raw):
            gains[F.group_of(name)] += float(value) / total * 100
        per_stage[stage] = gains
    stages = list(per_stage)
    return _table(["group (gain %)", *stages], [[g, *[_f(per_stage[s][g]) for s in stages]] for g in F.GROUPS])


RELEASE_COLUMNS = (("previous_config_previous_data", "previous config, previous data (1 seed)"),
                   ("previous_config_current_data", "previous config, current data (1 seed)"),
                   ("tuned_config_current_data", "tuned params, current data (1 seed)"),
                   ("after_release", "release: tuned if chosen, 5 seeds, shrink"))


def _release_table(comparison: dict) -> str:
    rows = []
    for stage in [*F.STAGES, "ALL"]:
        row = [stage]
        for key, _ in RELEASE_COLUMNS:
            b = comparison.get(key, {}).get("stages", {}).get(stage, {})
            if not b.get("n"):
                row.append("n/a")
                continue
            pinball = float(np.mean(list(b["pinball"].values())))
            row.append(f"MAE {_f(b['mae'])}; pinball {_f(pinball, 2)}; coverage {_f(b['coverage'], 0)}%; "
                       f"vs median {_ci(b.get('diff_median'))}")
        rows.append(row)
    return _table(["stage", *[label for _, label in RELEASE_COLUMNS]], rows)


def _shrink_table(comparison: dict) -> str:
    weights = comparison.get("shrink", {}).get("weights", {})
    losses = comparison.get("shrink", {}).get("pinball_by_weight", {})
    rows = [[s, weights[s], *[_f(losses.get(s, {}).get(f"{w:.2f}"), 3) for w in SHRINK_GRID]]
            for s in F.STAGES if s in weights]
    return _table(["stage", "chosen w", *[f"pinball at w={w:.2f}" for w in SHRINK_GRID]], rows)


def _parameters_text(comparison: dict) -> list[str]:
    p = comparison.get("parameters", {})
    if not p:
        return ["No release comparison file was found: run `python -m ipo_radar.model.release`."]
    return [
        f"Decision: **{p.get('decision')}** parameters (the tuned set is used only if its 2022-2024 single-seed pinball "
        f"loss is lower than the default's). Chosen overrides: `{json.dumps(p.get('chosen', {}))}`.",
        f"Optuna search: {p.get('search_trials')} trials on 2022-2024. Default settings loss "
        f"{_f(p.get('search_default_loss'), 4)}, best {_f(p.get('search_best_loss'), 4)}. Best settings: "
        f"`{json.dumps({k: round(float(v), 4) for k, v in p.get('search_best', {}).items()})}`.",
    ]


def _rows_table(base: pd.DataFrame) -> str:
    rows = []
    for m in REPORT_MARKETS:
        sub = base[base["market"] == m]
        if sub.empty:
            continue
        yrs = sub["listing_date"].dt.year
        rows.append([m, len(sub), int(sub[F.TARGET_OPEN].notna().sum()), int(sub[F.TARGET_CLOSE].notna().sum()),
                     int(sub["gmp_pct_last"].notna().sum()), int(sub["sub_qib"].notna().sum()),
                     int(sub["open_date"].notna().sum()), f"{int(yrs.min())}-{int(yrs.max())}"])
    return _table(["market", "IPOs", "open-gain target", "close-gain target", "GMP reading", "QIB subscription",
                   "open date known", "listing years"], rows)


def _verdict_line(stage: str, b: dict) -> str:
    if not b.get("n"):
        return "no test rows"
    text = (f"model MAE {b['mae']:.1f} pp vs training median {b['mae_median']:.1f} pp "
            f"({_ci(b.get('diff_median'))}, n={b['n']}): {metrics.verdict(b.get('diff_median'), b['n'])}")
    if b.get("n_gmp"):
        text += (f"; GMP rows n={b['n_gmp']}: model {b['mae_gmp_rows']:.1f} pp vs GMP {b['mae_gmp']:.1f} pp "
                 f"({_ci(b.get('diff_gmp'))})")
    return text + "."


def _placeholder_counts() -> dict[str, int]:
    """Cells blanked as placeholder zeros, as the last build recorded them in COVERAGE.md."""
    if not BUILD_COVERAGE.exists():
        return {}
    text = BUILD_COVERAGE.read_text(encoding="utf-8")
    return {m[1]: int(m[2]) for m in re.finditer(r"(\w+) exactly 0 \(placeholder\): (\d+)", text)}


def _data_change_section(h_locked: dict, previous: Path) -> list[str]:
    """Old (before the blanking) and new locked headline numbers, generated from the saved snapshot."""
    lines = ["## Data change: placeholder zeros blanked", ""]
    counts = _placeholder_counts()
    blanked = ", ".join(f"{k.replace('_gain_pct', '').replace('list_', '').replace('gmp_pct_last', 'GMP %')} {v}"
                        for k, v in counts.items() if v) or "not recorded"
    lines += [
        "After the locked blocks 2025-2026 were first evaluated, the build started blanking exact-zero listing gains and "
        "GMP readings as placeholders, because sources write 0 where no value exists. Cells blanked: "
        f"{blanked}. The locked blocks were evaluated again on the changed table, so they are no longer blind to it. "
        "The earlier report is kept at models/superseded-zero-fix/MODEL_REPORT.before-zero-fix.md.", ""]
    if not previous.exists():
        return lines + ["No earlier headline snapshot was found, so the before/after comparison is not available.", ""]
    old = json.loads(previous.read_text(encoding="utf-8"))["stages"]
    rows = []
    for stage in F.STAGES:
        now = h_locked.get(stage, {})
        before = old.get(stage)
        if not before or not now.get("n"):
            continue
        rows.append([stage, f"{before['n']} to {now['n']}", f"{before['mae']:.1f} to {_f(now.get('mae'))}",
                     f"{before['mae_median']:.1f} to {_f(now.get('mae_median'))}",
                     f"{_ci(before['diff_median'])} to {_ci(now.get('diff_median'))}",
                     f"{before['coverage']:.0f} to {_f(now.get('coverage'), 0)}",
                     f"{before['auc']:.3f} to {_f(now.get('auc'), 3)}"])
    return lines + [_table(["stage", "open rows", "MAE p50", "MAE median", "p50 - median (95% CI)", "coverage %",
                            "AUC"], rows), ""]


def _names(stages: list[str]) -> str:
    return ", ".join(stages) if stages else "none"


def _selection_table(status: dict) -> str:
    rows = []
    for stage in F.STAGES:
        s = status.get(stage)
        if s:
            rows.append([stage, s["n"], _ci(s["p50_minus_median"]), "yes" if s["worse"] else "no",
                         "yes" if s["beats"] else "no"])
    if not rows:
        return "No selection rows."
    return _table(["stage", "n", "p50 - median (95% CI), pooled over segments", "worse than median",
                   "beats median"], rows)


def _names_or_none(stages: list[str]) -> str:
    return ", ".join(stages) if stages else "none"


def _what_it_tells(locked: dict, skill: dict, shown: dict, selection: dict) -> list[str]:
    """Plain statements computed from the numbers above; nothing here is written in advance of them."""
    def in_order(names) -> str:
        names = [s for s in F.STAGES if s in set(names)]
        return ", ".join(names) if names else "none"

    stages = [s for s in F.STAGES if locked.get(s, {}).get("n")]
    if not stages:
        return ["- No locked block has rows for this table, so nothing can be said about 2025-2026 yet."]
    worse, better, same = [], [], []
    for s in stages:
        d = locked[s].get("diff_median")
        if d is None:
            continue
        (worse if d[1] > 0 else better if d[2] < 0 else same).append(s)
    flagged_p50 = {s for s in F.STAGES for v in skill.get(s, {}).values() if v["p50_beats_median"]}
    flagged_text = "; ".join(
        f"{s} ({', '.join(g for g, v in skill[s].items() if v['p50_beats_median'])})"
        for s in F.STAGES if any(v["p50_beats_median"] for v in skill.get(s, {}).values())) or "none"
    worse_sel = [s for s in F.STAGES if selection.get(s, {}).get("worse")]
    flagged_prob = {s for s in F.STAGES for v in skill.get(s, {}).values() if v["brier_beats_base"]}
    prob_beats = [s for s in stages if (locked[s].get("brier") is not None
                                        and locked[s]["brier"] < locked[s]["brier_base"])]
    auc = {s: locked[s]["auc"] for s in stages if locked[s].get("auc") is not None}
    cov = [locked[s]["coverage"] for s in stages if locked[s].get("coverage") is not None]
    cov_shown = [shown[s]["coverage"] for s in F.STAGES if shown.get(s, {}).get("coverage") is not None]
    pre = shown.get("pre_listing", {})
    gmp_line = (f"On 2023-2024 the pre-listing p50 was {pre['mae_gmp_rows']:.1f} pp off on the "
                f"{pre['n_gmp']} GMP-bearing IPOs, against {pre['mae_gmp']:.1f} pp for the GMP reading itself"
                if pre.get("n_gmp") else "There were no GMP-bearing IPOs in 2023-2024")
    return [
        f"- **Against the historical median, the model did not beat it on the locked blocks.** Per stage and segment, "
        f"the 2022-2024 flags (95% CI of the p50 minus median difference entirely below zero) passed at: {flagged_text}. "
        f"Pooled over 2022-2024, the p50 is worse than the median at {_names_or_none(worse_sel)} (see the selection table). "
        f"On 2025-2026 it is worse at {in_order(worse)}, better at {in_order(better)}, and not clearly different at "
        f"{in_order(same)}.",
        f"- **The p10-p90 band is too wide on the locked blocks.** It held {min(cov):.0f}-{max(cov):.0f}% of locked "
        f"outcomes (target 80%), against {min(cov_shown):.0f}-{max(cov_shown):.0f}% on 2023-2024. The conformal "
        "widening was fitted on 2022-2024 and does not transfer to the newer blocks; treat the band as a loose guide.",
        f"- **The probability of opening above the issue price is only weakly informative.** Its Brier score beats the "
        f"base rate at {in_order(prob_beats)} on the locked blocks; the 2022-2024 flags said it did at "
        f"{in_order(flagged_prob)}. AUC on the locked blocks runs from {min(auc.values()):.2f} to "
        f"{max(auc.values()):.2f}. Stages without a passing flag are caveated and reported as not reliable.",
        f"- **GMP cannot be checked on the locked blocks.** GMP readings stop in 2024 and there are none in 2025-2026. "
        f"{gmp_line}, so GMP remains the stronger signal where it exists.",
        "- **GLOBAL has no open-gain outcome on the locked blocks.** US IPOs from 2024 carry only close-gain targets and "
        "OTHER carries none, so the open-gain numbers for US, HK and OTHER come from 2022-2023 only. The close-gain "
        "table above includes the 2024-2026 US IPOs.",
        "- **Use it for the size of the typical move at most, not for direction.** The missing context (no open dates "
        "for US rows, about half of SME rows) and the missing category subscription (mostly pre-2023 mainboard) limit "
        "what it can see at serving time.",
    ]


def evaluate(df: pd.DataFrame | None = None, out_path: Path | None = None) -> EvalResult:
    """Write the report from the saved release: the bundle, its selection and latest-block records, and the
    comparison file. Run `python -m ipo_radar.model.release` first."""
    base = pd.read_parquet(config.IPOS_PARQUET) if df is None else df
    base = base[base["listing_date"].notna()].reset_index(drop=True)
    sel_path, lat_path = config.MODELS / "selection_records.parquet", config.MODELS / "latest_records.parquet"
    if not (sel_path.exists() and lat_path.exists()):
        raise ValueError("no release records found: run `python -m ipo_radar.model.release` first")
    bundle = load_model()
    shrink = getattr(bundle, "shrink", None) or {}
    selection = apply_shrink(pd.read_parquet(sel_path), shrink)
    latest = apply_shrink(pd.read_parquet(lat_path), shrink)
    rng = np.random.default_rng(metrics.BOOT_SEED)
    h_locked = _headline(latest, rng)
    h_shown = _headline(selection[selection["year"].isin(SHOW_YEARS)], rng)
    skill = metrics.skill(selection)
    sel_status = metrics.stage_status(selection)
    comparison_file = config.MODELS / "release_comparison.json"
    comparison = json.loads(comparison_file.read_text(encoding="utf-8")) if comparison_file.exists() else {}

    dhash = data_hash(base)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Model report: release evaluation", "",
        f"Generated {stamp}. Bundle {bundle.version}, data hash {dhash}, seeds {bundle.seeds}. Design: "
        f"`{json.dumps(bundle.design)}`. Shrink weights: `{json.dumps(getattr(bundle, "shrink", None) or {})}`. Parameters: "
        f"`{json.dumps(bundle.params)}`.", "",
        *_data_change_section(h_locked, PREVIOUS_HEADLINE), "",
        "## Protocol", "",
        f"- {PROTOCOL}",
        f"- Selection blocks: listing years {', '.join(map(str, SELECTION_YEARS))}. Every choice in this release (shrink "
        "weights, the Optuna search, the parameter decision) was made on those blocks only. Every block is predicted by "
        "models trained only on IPOs listed before it.",
        f"- {LOCKED_NOTE}", "",
        "## Release: 2022-2024 before and after", "",
        "Pinball is the mean of the p10, p50 and p90 losses. 'vs median' is the model's p50 MAE minus the segment's "
        "training median MAE on the same IPOs, with a 95% bootstrap interval (negative is better).", "",
        _release_table(comparison), "",
        *_parameters_text(comparison), "",
        "Shrink weights were chosen per stage by the pinball loss of the blended p50 on 2022-2024:", "",
        _shrink_table(comparison), "",
        "## Headline: 2025-2026 (recorded once, not used for any choice)", "",
        "Units: percentage points (pp) of the issue price. Intervals are 95% bootstrap intervals over IPOs. The per-stage "
        "skill flags come from 2022-2024 only.", "",
        _headline_table(h_locked, skill), "",
        "## Same measures, 2023-2024", "",
        _headline_table(h_shown, skill), "",
        "## What this model can and cannot tell you", "",
        *_what_it_tells(h_locked, skill, h_shown, sel_status), "",
        "## Per-stage skill flags (2022-2024, per segment, saved in the bundle)", "",
        _segment_skill_table(skill), "",
        "## Selection blocks 2022-2024, pooled over segments", "", _selection_table(sel_status), "",
        "## Experiments: every option, kept or dropped (earlier design search, 2022-2024)", "",
        *_experiments_section(config.MODELS / "experiments.json"),
        "## Close-gain median, locked blocks", "", _close_table(h_locked), "",
        "## Locked blocks by market", "", _markets_table(latest, rng), "",
        "## Calibration of P(opens above issue price), locked blocks", "", _calibration_table(latest), "",
        "## Every block (2022-2026)", "", _block_table(pd.concat([selection, latest], ignore_index=True)), "",
        "## GMP-bearing training rows", "", _gmp_counts(base), "",
        "## Feature-group importance (final bundle, in-sample gain)", "", _gain_table(bundle), "",
        "## Rows per market", "", _rows_table(base), "",
        "## Data warnings", "",
        "- Survivorship: the table holds only IPOs that listed; every estimate is optimistic for the average issue.",
        "- Sources joined by date: India and US tables come from different sites; date or price mismatches show up as "
        "noise.",
        "- GMP is unofficial grey-market data, taken shortly before listing, and exists only for India from 2020.",
        "- Exact-zero subscription multiples and listing gains are treated as missing placeholders (see the build notes).",
        "- The live feed carries no valuation or financial ratios, so those inputs are missing at serving time.",
        "- 2026 is a partial year.",
        "- Monotone constraints apply to the classifier only: LightGBM rejects them for quantile objectives.",
        "- The release bundle is trained on all IPOs, so its in-sample numbers are not evidence of accuracy.",
    ]
    out = out_path or config.ROOT / "docs" / "MODEL_REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    verdicts = {s: _verdict_line(s, h_locked[s]) for s in F.STAGES if s in h_locked}
    years = sorted({int(y) for y in pd.concat([selection, latest])["year"].unique()})
    return EvalResult(report_path=str(out), test_years=years,
                      n_test_rows=int(pd.concat([selection, latest])["ipo_id"].nunique()), data_hash=dhash,
                      verdicts=verdicts, skill=skill)


def main() -> None:
    result = evaluate()
    print(f"wrote {result.report_path}")
    for stage, line in result.verdicts.items():
        print(f"{stage}: {line}")


if __name__ == "__main__":
    main()
