"""Release choices: shrinkage weights, the interval after shrinking, search parameter mapping and a short search."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ipo_radar.model import features as F
from ipo_radar.model import predict as P
from ipo_radar.model.search import DEFAULT_TUNING, SHRINK_GRID, choose_shrink, lgbm_overrides, search
from ipo_radar.model.walkforward import apply_shrink
from model_synth import make_ipos


def _records(p50: np.ndarray, y: np.ndarray, med: float, stage: str = "close") -> pd.DataFrame:
    n = len(y)
    return pd.DataFrame({"stage": stage, "open_y": y, "p50": p50, "p10": p50 - 10, "p90": p50 + 10,
                         "med_open": np.full(n, med)})


def test_shrink_is_picked_toward_the_median_when_the_model_is_noise():
    rng = np.random.default_rng(0)
    y = rng.normal(5, 10, 400)
    noise_p50 = rng.normal(0, 40, 400)               # a useless model far from the outcomes
    weights, table = choose_shrink(_records(noise_p50, y, med=5.0))
    assert weights["close"] < 1.0
    assert set(table["close"]) == {f"{w:.2f}" for w in SHRINK_GRID}


def test_shrink_keeps_the_model_when_it_is_the_best_predictor():
    rng = np.random.default_rng(1)
    y = rng.normal(5, 10, 400)
    good_p50 = y + rng.normal(0, 1, 400)              # nearly the outcome itself
    weights, _ = choose_shrink(_records(good_p50, y, med=5.0))
    assert weights["close"] == 1.0


def test_ties_go_to_the_larger_weight_so_the_model_is_kept():
    y = np.full(50, 3.0)
    weights, _ = choose_shrink(_records(np.full(50, 3.0), y, med=3.0))   # every weight gives the same loss
    assert weights["close"] == 1.0


def test_apply_shrink_blends_the_p50_and_moves_the_whole_interval_with_it():
    rec = _records(np.array([40.0, -30.0]), np.array([0.0, 0.0]), med=10.0)
    shrunk = apply_shrink(rec, {"close": 0.5})
    assert shrunk["p50"].tolist() == pytest.approx([25.0, -10.0])
    assert (shrunk["p10"] <= shrunk["p50"]).all() and (shrunk["p90"] >= shrunk["p50"]).all()
    assert (shrunk["p90"] - shrunk["p10"]).tolist() == pytest.approx([20.0, 20.0])   # width unchanged
    assert apply_shrink(rec, {}).equals(rec)
    zero = apply_shrink(rec, {"close": 0.0})
    assert zero["p50"].tolist() == pytest.approx([10.0, 10.0])


def test_search_names_map_to_lightgbm_names():
    mapped = lgbm_overrides(dict(DEFAULT_TUNING))
    assert mapped["colsample_bytree"] == DEFAULT_TUNING["feature_fraction"]
    assert mapped["subsample"] == DEFAULT_TUNING["bagging_fraction"]
    assert "feature_fraction" not in mapped and "bagging_fraction" not in mapped


def test_short_search_runs_inside_its_bounds_and_reports_the_default():
    base = make_ipos(n_per_year=40, years=range(2019, 2025), seed=9)
    base = base[base["listing_date"].notna()].reset_index(drop=True)
    result = search(base, n_trials=3, timeout_s=120, seed=7, years=(2022, 2023, 2024))
    assert result["n_trials"] == 3
    assert result["trials"][0]["loss"] == pytest.approx(result["default_loss"])
    best = result["best"]
    assert 0.01 <= best["learning_rate"] <= 0.1 and 4 <= best["num_leaves"] <= 31
    assert 5 <= best["min_child_samples"] <= 60 and 0.5 <= best["feature_fraction"] <= 1.0
    assert result["best_loss"] <= result["default_loss"] + 1e-12


def test_predict_says_when_the_estimate_is_shrunk_or_not_used():
    import dataclasses
    from test_model_predict import CONTEXT, _ipo, _stub_bundle
    bundle = dataclasses.replace(_stub_bundle(q10=-0.2, q50=0.4, q90=0.8), shrink={"pre_open": 0.5})
    bundle.stages["pre_open"].open_median = {"ALL": 10.0, "IN_MAIN": 10.0}
    p = P.predict(_ipo(), bundle=bundle, context=CONTEXT)
    assert P.SHRINK_CAVEAT in p.caveats
    assert p.open_gain_pct_p50 == pytest.approx(0.5 * F.from_model_scale(0.4) + 0.5 * 10.0, abs=1e-6)
    zero = dataclasses.replace(bundle, shrink={"pre_open": 0.0})
    q = P.predict(_ipo(), bundle=zero, context=CONTEXT)
    assert q.open_gain_pct_p50 == pytest.approx(10.0)
    assert P.NO_MODEL_CAVEAT in q.caveats
    assert q.open_gain_pct_p90 - q.open_gain_pct_p10 > 0          # the interval is kept, not collapsed
    assert any(c.startswith("model's own range, not used") for c in q.caveats)


def test_bundle_without_the_shrink_attribute_serves_unblended_and_never_raises():
    import dataclasses
    from test_model_predict import CONTEXT, _ipo, _stub_bundle
    old = _stub_bundle(q10=-0.2, q50=0.4, q90=0.8)
    del old.__dict__["shrink"]                      # a bundle pickled before the field existed
    plain = P.predict(_ipo(), bundle=old, context=CONTEXT)
    assert plain.open_gain_pct_p50 == pytest.approx(F.from_model_scale(0.4))
    assert not any(c == P.SHRINK_CAVEAT or c == P.NO_MODEL_CAVEAT for c in plain.caveats)
    bad = dataclasses.replace(_stub_bundle(q10=-0.2, q50=0.4, q90=0.8), shrink={"pre_open": "not a number"})
    assert P.predict(_ipo(), bundle=bad, context=CONTEXT).open_gain_pct_p50 == pytest.approx(F.from_model_scale(0.4))
