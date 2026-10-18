"""Training tests on the synthetic table: the model learns the planted signal, quantiles are ordered, the calibrator is
monotone, artifacts are written and reloadable, and the failure paths raise clearly."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from ipo_radar.model import features as F
from ipo_radar.model import predict as P
from ipo_radar.model.bundle import Bundle
from ipo_radar.model.train import train
from model_synth import make_ipos

VERSION = re.compile(r"^\d{8}-[0-9a-f]{8}$")


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return make_ipos(n_per_year=120, years=range(2015, 2025), seed=11)


@pytest.fixture(scope="module")
def trained(data, tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    train_part = data[data["listing_date"].dt.year < 2022]
    report = train(train_part, out_dir=out, seed=7, quick=False)
    return report, out, train_part


def test_train_writes_bundle_and_metrics_and_bundle_reloads(trained):
    report, out, _ = trained
    assert VERSION.match(report.version)
    assert (out / "bundle.joblib").exists()
    summary = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    assert summary["version"] == report.version
    assert set(summary["stages"]) == set(F.STAGES)
    bundle = P.load_model(out)
    assert isinstance(bundle, Bundle)
    assert bundle.version == report.version and bundle.n_rows == report.n_rows
    assert set(bundle.stages) == set(F.STAGES)
    for stage, sm in bundle.stages.items():
        assert sm.features == F.stage_features(stage)


def test_model_learns_the_planted_signal_on_held_out_years(trained, data):
    _, out, _ = trained
    bundle = P.load_model(out)
    test = data[data["listing_date"].dt.year >= 2022]
    stage = "close"
    sm = bundle.stages[stage]
    rows = F.asof_expand(test)
    rows = rows[(rows["as_of"] == stage) & rows[F.TARGET_CLOSE].notna()]
    X = F.design(rows, stage, bundle.vocab)
    y = rows[F.TARGET_CLOSE].to_numpy(float)
    p50 = F.from_model_scale(sm.close_q50.predict(X))
    train_close = data.loc[data["listing_date"].dt.year < 2022, F.TARGET_CLOSE].dropna()
    model_mae = np.abs(y - p50).mean()
    median_mae = np.abs(y - train_close.median()).mean()
    assert model_mae < 0.8 * median_mae, (model_mae, median_mae)


def test_separately_fitted_quantiles_rarely_cross_and_are_always_finite(trained, data):
    """Crossing is possible for independent quantile models; serving repairs it (see test_model_predict)."""
    _, out, _ = trained
    bundle = P.load_model(out)
    expanded = F.asof_expand(data[data["listing_date"].dt.year >= 2022])
    for stage, sm in bundle.stages.items():
        rows = expanded[expanded["as_of"] == stage]
        X = F.design(rows, stage, bundle.vocab)
        q = np.column_stack([sm.q10.predict(X), sm.q50.predict(X), sm.q90.predict(X)])
        assert np.isfinite(q).all()
        crossed = np.mean((q[:, 0] > q[:, 1]) | (q[:, 1] > q[:, 2]))
        assert crossed < 0.05, (stage, crossed)


def test_calibrator_is_monotone_and_maps_into_probabilities(trained):
    _, out, _ = trained
    bundle = P.load_model(out)
    for stage, sm in bundle.stages.items():
        if sm.iso is None:
            continue
        grid = np.linspace(0, 1, 201)
        mapped = sm.iso.predict(grid)
        assert np.all(np.diff(mapped) >= -1e-12), stage
        assert mapped.min() >= 0 and mapped.max() <= 1


def test_seeded_training_is_reproducible(data, tmp_path):
    small = data[data["listing_date"].dt.year < 2020]
    a = train(small, out_dir=tmp_path / "a", seed=3, quick=True)
    b = train(small, out_dir=tmp_path / "b", seed=3, quick=True)
    ma, mb = P.load_model(tmp_path / "a"), P.load_model(tmp_path / "b")
    X = F.design(F.asof_expand(small).query("as_of == 'close'").head(50), "close", ma.vocab)
    np.testing.assert_allclose(ma.stages["close"].q50.predict(X), mb.stages["close"].q50.predict(X))
    assert a.data_hash == b.data_hash


def test_too_few_rows_for_every_stage_fails_clearly_and_writes_nothing(data, tmp_path):
    with pytest.raises(ValueError, match="no stage had enough rows"):
        train(data.head(12), out_dir=tmp_path, quick=True)
    assert not (tmp_path / "bundle.joblib").exists()


def test_training_without_any_target_fails_clearly(data, tmp_path):
    empty = data.copy()
    empty["list_open_gain_pct"] = np.nan
    empty["list_close_gain_pct"] = np.nan
    with pytest.raises(ValueError, match="no rows with a listing-gain target"):
        train(empty, out_dir=tmp_path, quick=True)


def test_rows_without_listing_date_are_dropped_and_reported(data, tmp_path):
    frame = data[data["listing_date"].dt.year < 2021].copy()
    frame.loc[frame.index[:5], "listing_date"] = pd.NaT
    report = train(frame, out_dir=tmp_path, quick=True)
    assert report.n_rows == len(frame) - 5
    assert any("without a listing date" in w for w in report.warnings)


def test_seeds_build_one_member_per_seed_and_params_reach_lightgbm(data, tmp_path):
    from ipo_radar.model.bundle import Ensemble
    report = train(data[data["listing_date"].dt.year < 2021], out_dir=tmp_path, quick=True, seeds=[1, 2],
                   params={"num_leaves": 5})
    bundle = P.load_model(tmp_path)
    sm = bundle.stages["pre_open"]
    assert isinstance(sm.q50, Ensemble) and len(sm.q50.members) == 2
    assert sm.q50.members[0].get_params()["num_leaves"] == 5
    assert report.seeds == [1, 2] and report.params == {"num_leaves": 5}


def test_routed_india_set_is_trained_and_saved(data, tmp_path):
    from ipo_radar.model.design import Design
    spec = Design(routing={"IN_MAIN": "india", "IN_SME": "india", "GLOBAL": "pooled"})
    train(data, out_dir=tmp_path, quick=True, spec=spec)
    bundle = P.load_model(tmp_path)
    assert "india" in bundle.alt_sets and bundle.routing["IN_MAIN"] == "india"
    assert set(bundle.alt_sets["india"]) == set(F.STAGES)
    assert set(bundle.alt_sets["india"]["pre_open"].n_by_market) <= {"IN_MAIN", "IN_SME"}


def test_skill_flags_come_from_the_2022_2024_blocks(data, tmp_path):
    train(data, out_dir=tmp_path, quick=True)
    bundle = P.load_model(tmp_path)
    assert bundle.skill, "the synthetic table has 2022-2024 blocks, so skill flags must be present"
    flags = next(iter(next(iter(bundle.skill.values())).values()))
    assert {"n", "p50_beats_median", "brier_beats_base", "mae_diff", "brier_diff"} <= set(flags)
