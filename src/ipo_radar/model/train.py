"""Train the per-stage listing-gain models for every routed model set and save one bundle.

The pooled set (all markets) is always trained: the Kaggle job and older callers read it as bundle.stages. Other sets
(India-only, one per segment) are trained when the design routes a segment to them. Skill flags are computed from the
walk-forward blocks 2022-2024 and stored in the bundle, so serving can say when a stage has no demonstrated skill.

`python -m ipo_radar.model.train` trains on data/processed/ipos.parquet and writes models/bundle.joblib and
models/train_report.json.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from ipo_radar import config
from ipo_radar.model import metrics
from ipo_radar.model import features as F
from ipo_radar.model.bundle import Bundle
from ipo_radar.model.design import DEFAULT_DESIGN, LOCKED_YEARS, PROTOCOL, SELECTION_YEARS, SET_MARKETS, Design, segment_of
from ipo_radar.model.fitting import fit_stage
from ipo_radar.model.search import choose_shrink
from ipo_radar.model.walkforward import apply_shrink, run_walk


@dataclass
class TrainReport:
    version: str
    trained_at: str
    n_rows: int
    quick: bool
    params: dict
    seeds: list[int]
    stages: dict[str, dict]
    design: dict
    gmp_rows: dict[str, dict[str, int]]
    skill: dict
    recent: dict
    shrink: dict
    shrink_table: dict
    data_hash: str
    model_path: str
    metrics_path: str
    warnings: list[str] = field(default_factory=list)


def data_hash(df: pd.DataFrame) -> str:
    """Eight hex characters identifying the training table, so a bundle can be matched to its data."""
    digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return digest.hexdigest()[:8]


def _atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _stage_summary(sm) -> dict:
    if sm is None:
        return {"trained": False}
    return {"trained": True, "n_open": sm.n_open, "n_close": sm.n_close, "n_by_market": sm.n_by_market,
            "calibrated": sm.iso is not None, "conformal": dict(sm.conf), "conformal_residuals": dict(sm.conf_n),
            "gmp_linear": sm.gmp_linear is not None, "n_features": len(sm.features),
            "core_by_market": {m: list(v) for m, v in sm.core.items()}}


def train(df: pd.DataFrame | None = None, out_dir: Path = config.MODELS, seed: int = 7, quick: bool = False,
          params: dict | None = None, seeds: Sequence[int] | None = None, spec: Design = DEFAULT_DESIGN
          ) -> TrainReport:
    """Train every stage of every routed set and save models/bundle.joblib.

    params: LightGBM overrides (the Optuna search's best settings). seeds: one model per seed, averaged at prediction
    time; default is [seed]. Rows without a listing date are dropped (reported)."""
    seed_list = list(seeds) if seeds else [seed]
    base = pd.read_parquet(config.IPOS_PARQUET) if df is None else df
    warnings: list[str] = []
    usable = base[base["listing_date"].notna()].reset_index(drop=True)
    if len(usable) < len(base):
        warnings.append(f"{len(base) - len(usable)} rows without a listing date are not used")
    if usable[[F.TARGET_OPEN, F.TARGET_CLOSE]].notna().to_numpy().sum() == 0:
        raise ValueError("no rows with a listing-gain target: nothing to train")
    vocab = F.fit_vocab(usable)
    expanded = F.asof_expand(usable)
    expanded["segment"] = [segment_of(m) for m in expanded["market"].astype(str)]

    sets: dict[str, dict] = {}
    for name in spec.sets_used():
        in_set = expanded[expanded["market"].astype(str).isin(SET_MARKETS[name])]
        models = {}
        for stage in F.STAGES:
            sm = fit_stage(in_set[in_set["as_of"] == stage], stage, vocab, spec, seed_list, quick, params)
            if sm is None:
                warnings.append(f"{name}/{stage}: fewer than 30 rows with an open-gain target; no model")
            else:
                models[stage] = sm
        sets[name] = models
    pooled = sets.pop("pooled")
    if not pooled:
        raise ValueError("no stage had enough rows to train")

    selection = [y for y in SELECTION_YEARS if (usable["listing_date"].dt.year == y).any()]
    raw, _ = run_walk(usable, spec, selection, seed_list, quick, params)
    shrink, shrink_table = choose_shrink(raw) if len(raw) else ({}, {})
    selected = apply_shrink(raw, shrink)
    skill = metrics.skill(selected) if len(selected) else {}
    if not skill:
        warnings.append("no walk-forward block in 2022-2024 had enough rows: skill flags are empty")
    latest = [y for y in LOCKED_YEARS if (usable["listing_date"].dt.year == y).any()]
    latest_raw, _ = run_walk(usable, spec, latest, seed_list, quick, params)
    latest_shrunk = apply_shrink(latest_raw, shrink)
    recent = {stage: {**st, "years": latest} for stage, st in metrics.stage_status(latest_shrunk).items()} \
        if len(latest_shrunk) else {}

    dhash = data_hash(usable)
    now = datetime.now(timezone.utc)
    version = f"{now:%Y%m%d}-{dhash}"
    bundle = Bundle(version=version, trained_at=now.isoformat(timespec="seconds"), n_rows=len(usable),
                    stages=pooled, vocab=vocab, seed=seed, seeds=seed_list, design=spec.as_dict(),
                    routing=dict(spec.routing), alt_sets=sets, skill=skill, recent=recent, shrink=shrink,
                    params=dict(params or {}), protocol=PROTOCOL)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(out_dir / "selection_records.parquet", index=False)
    latest_raw.to_parquet(out_dir / "latest_records.parquet", index=False)
    model_path = out_dir / "bundle.joblib"
    metrics_path = out_dir / "train_report.json"
    _atomic_write(model_path, lambda p: joblib.dump(bundle, p))
    gmp_rows: dict[str, dict[str, int]] = {}
    if "pre_listing" in pooled:
        gmp_rows = {seg: {str(y): n for y, n in years.items()}
                    for seg, years in pooled["pre_listing"].gmp_counts.items()}
    report = TrainReport(
        version=version, trained_at=bundle.trained_at, n_rows=len(usable), quick=quick,
        params=dict(params or {}), seeds=seed_list,
        stages={s: _stage_summary(pooled.get(s)) for s in F.STAGES}, design=spec.as_dict(), gmp_rows=gmp_rows,
        skill=skill, recent=recent, shrink=shrink, shrink_table=shrink_table, data_hash=dhash,
        model_path=str(model_path), metrics_path=str(metrics_path),
        warnings=warnings,
    )
    _atomic_write(metrics_path, lambda p: p.write_text(json.dumps(asdict(report), indent=2, default=str),
                                                       encoding="utf-8"))
    return report


RELEASE_CONFIG = config.MODELS / "release_config.json"


def main() -> None:
    """Train with the release settings (params and seeds) when models/release_config.json exists."""
    if RELEASE_CONFIG.exists():
        cfg = json.loads(RELEASE_CONFIG.read_text(encoding="utf-8"))
        report = train(params=cfg["params"], seeds=cfg["seeds"])
    else:
        report = train()
    print(f"trained bundle {report.version} on {report.n_rows} IPOs -> {report.model_path}")
    for stage, info in report.stages.items():
        print(f"  {stage}: trained={info.get('trained')}")
    for w in report.warnings:
        print(f"  warning: {w}")


if __name__ == "__main__":
    main()
