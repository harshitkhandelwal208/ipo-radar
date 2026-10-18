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
from ipo_radar.model.experiments import SEED
from ipo_radar.model import features as F
from ipo_radar.model import metrics
from ipo_radar.model.design import DEFAULT_DESIGN, LOCKED_YEARS, PROTOCOL, SELECTION_YEARS, Design, segment_of
from ipo_radar.model.fitting import raw_components
from ipo_radar.model.train import data_hash
from ipo_radar.model.walkforward import expanded_frame, run_walk

REPORT_YEARS = tuple(SELECTION_YEARS) + tuple(LOCKED_YEARS)
SHOW_YEARS = (2023, 2024)
REPORT_MARKETS = ("IN_MAIN", "IN_SME", "US", "HK", "OTHER")
LOCKED_NOTE = (
    "2025-2026 were seen once in the earlier report (before the protocol), and again after the placeholder blanking "
    "(see Data change). Nothing was chosen from them, but they are no longer blind.")
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


def _importance(models: dict, base: pd.DataFrame, expanded: pd.DataFrame, year: int) -> tuple[str, str]:
    """Gain share per group (the pooled median models of `year`) and permutation increase on that year's rows."""
    rng = np.random.default_rng(SEED)
    vocab = F.fit_vocab(base[base["listing_date"].dt.year < year])
    gain_rows, perm_rows = [], []
    groups = list(F.GROUPS)
    per_stage: dict[str, dict] = {}
    for stage in F.STAGES:
        sm = models.get(("pooled", stage, year))
        if sm is None:
            continue
        raw = sm.q50.booster_.feature_importance(importance_type="gain")
        total = float(raw.sum()) or 1.0
        gains = {g: 0.0 for g in groups}
        for name, value in zip(sm.features, raw):
            gains[F.group_of(name)] += float(value) / total * 100
        rows = expanded[(expanded["as_of"] == stage) & (expanded["listing_date"].dt.year == year)]
        Xte = F.design(rows, stage, vocab)
        mask = rows[F.TARGET_OPEN].notna().to_numpy()
        Xo, y = Xte[mask], rows[F.TARGET_OPEN].to_numpy(float)[mask]
        base_mae = float(np.abs(y - F.from_model_scale(sm.q50.predict(Xo))).mean())
        inc = {}
        for g in groups:
            cols = [f for f in sm.features if F.group_of(f) == g]
            if not cols or len(Xo) < metrics.MIN_VERDICT_ROWS:
                continue
            deltas = []
            for _ in range(5):
                order = rng.permutation(len(Xo))
                Xp = Xo.copy()
                for c in cols:
                    Xp[c] = Xo[c].iloc[order].set_axis(Xo.index)
                deltas.append(float(np.abs(y - F.from_model_scale(sm.q50.predict(Xp))).mean()) - base_mae)
            inc[g] = float(np.mean(deltas))
        per_stage[stage] = {"gain": gains, "perm": inc}
    if not per_stage:
        return "n/a", "n/a"
    stages = list(per_stage)
    gain = _table(["group (gain %)", *stages],
                  [[g, *[_f(per_stage[s]["gain"][g]) for s in stages]] for g in groups])
    perm = _table(["group (permutation, pp)", *stages],
                  [[g, *[_f(per_stage[s]["perm"].get(g), 2, sign=True) if g in per_stage[s]["perm"] else "n/a"
                         for s in stages]] for g in groups])
    return gain, perm


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


def evaluate(df: pd.DataFrame | None = None, out_path: Path | None = None, seed: int = 7, quick: bool = False,
             spec: Design | None = None) -> EvalResult:
    """Evaluate the chosen design on 2022-2026 walk-forward blocks and write the report."""
    base = pd.read_parquet(config.IPOS_PARQUET) if df is None else df
    base = base[base["listing_date"].notna()].reset_index(drop=True)
    if base.empty:
        raise ValueError("no IPO has a listing date: nothing to evaluate")
    spec = spec or DEFAULT_DESIGN
    records, models = run_walk(base, spec, REPORT_YEARS, [seed], quick, keep_models=True)
    if records.empty:
        raise ValueError("no walk-forward block had enough rows to evaluate")
    rng = np.random.default_rng(metrics.BOOT_SEED)
    sel = records[records["year"].isin(SELECTION_YEARS)]
    locked = records[records["year"].isin(LOCKED_YEARS)]
    shown = records[records["year"].isin(SHOW_YEARS)]
    skill = metrics.skill(sel)
    sel_status = metrics.stage_status(sel) if len(sel) else {}
    h_locked = _headline(locked, rng)
    h_shown = _headline(shown, rng)

    base_x, expanded = expanded_frame(base)
    gain_t, perm_t = _importance(models, base_x, expanded, max(LOCKED_YEARS))

    dhash = data_hash(base)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    design_dict = spec.as_dict()
    lines = [
        "# Model report: walk-forward evaluation", "",
        f"Generated {stamp}. Data hash {dhash}, seed {seed}{', quick mode' if quick else ''}. Design: "
        f"`{json.dumps(design_dict)}`.", "",
        *_data_change_section(h_locked, PREVIOUS_HEADLINE), "",
        "## Protocol", "",
        f"- {PROTOCOL}",
        f"- Selection blocks: listing years {', '.join(map(str, SELECTION_YEARS))}. Locked blocks: "
        f"{', '.join(map(str, LOCKED_YEARS))}. Every block is predicted by models trained only on IPOs listed before it.",
        f"- {LOCKED_NOTE}",
        "- The design above is fixed in code (model/design.py) before this report is generated; the locked numbers "
        "below are the one evaluation of that design.", "",
        "## Headline: locked blocks 2025-2026", "",
        "Units: percentage points (pp) of the issue price. 'p50 - median' is the model's MAE minus the MAE of the "
        "segment's training median, on the same IPOs; a negative value is better. Intervals are 95% bootstrap intervals "
        "over IPOs. Skill flags come from 2022-2024 only.", "",
        _headline_table(h_locked, skill), "",
        "## Same measures, 2023-2024 (selection era, shown for comparison)", "",
        _headline_table(h_shown, skill), "",
        "## What this model can and cannot tell you", "",
        *_what_it_tells(h_locked, skill, h_shown, sel_status), "",
        "## Selection blocks 2022-2024, pooled over segments", "", _selection_table(sel_status), "",
        "## Per-stage skill flags (2022-2024, saved in the bundle)", "",
        _segment_skill_table(skill), "",
        "## Experiments: every option, kept or dropped (2022-2024)", "",
        *_experiments_section(config.MODELS / "experiments.json"),
        "## Close-gain median, locked blocks", "", _close_table(h_locked), "",
        "## Locked blocks by market", "", _markets_table(locked, rng), "",
        "## Calibration of P(opens above issue price), locked blocks", "", _calibration_table(locked),
        "## Every block (2022-2026)", "", _block_table(records), "",
        "## GMP-bearing training rows", "", _gmp_counts(base), "",
        "## Feature-group importance (last locked block, pooled model)", "",
        "Gain share of the median model; permutation is the increase in open-gain MAE (pp) when the group's inputs "
        "are shuffled on the block.", "", gain_t, "", perm_t, "",
        "## Rows per market", "", _rows_table(base), "",
        "## Data warnings", "",
        "- Survivorship: the table holds only IPOs that listed; every estimate is optimistic for the average issue.",
        "- Sources joined by date: India and US tables come from different sites; date or price mismatches show up as "
        "noise.",
        "- GMP is unofficial grey-market data, taken shortly before listing, and exists only for India from 2020.",
        "- The live feed carries no valuation or financial ratios, so those inputs are missing at serving time.",
        "- The walk-forward blocks are years of listing; 2026 is a partial year.",
        "- Monotone constraints apply to the classifier only: LightGBM rejects them for quantile objectives.",
    ]
    out = out_path or config.ROOT / "docs" / "MODEL_REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    verdicts = {s: _verdict_line(s, h_locked[s]) for s in F.STAGES if s in h_locked}
    return EvalResult(report_path=str(out), test_years=sorted(int(y) for y in records["year"].unique()),
                      n_test_rows=int(records["ipo_id"].nunique()), data_hash=dhash, verdicts=verdicts, skill=skill)


def main() -> None:
    result = evaluate()
    print(f"wrote {result.report_path}")
    for stage, line in result.verdicts.items():
        print(f"{stage}: {line}")


if __name__ == "__main__":
    main()
