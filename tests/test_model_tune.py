"""Optuna search over LightGBM: bounds, determinism, walk-forward folds without leakage, scoring, refusals.

Uses a stand-in features module (the Model agent's real one may differ); the search code is the real one."""
from __future__ import annotations

import types
import sys

import numpy as np
import pandas as pd
import pytest

from ipo_radar.model import tune
from model_helpers import make_ipos


@pytest.fixture(autouse=True)
def stub_features(monkeypatch):
    from model_helpers import FEATURES_STUB

    module = types.ModuleType("ipo_radar.model.features")
    exec(compile(FEATURES_STUB, "ipo_radar.model.features", "exec"), module.__dict__)
    monkeypatch.setitem(sys.modules, "ipo_radar.model.features", module)


def test_pinball_and_coverage_on_hand_values():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([0.0, 2.0, 2.0])
    assert tune.pinball(y, pred, 0.5) == pytest.approx((0.5 + 0.0 + 0.5) / 3)
    assert tune.coverage(y, np.zeros(3), np.array([5.0, 5.0, 2.5])) == pytest.approx(2 / 3)


def test_walk_forward_folds_never_train_on_the_future():
    dates = pd.to_datetime(make_ipos(300, seed=3)["listing_date"]).to_numpy()
    folds = tune.walk_forward_folds(dates, 4)
    assert folds
    for train, valid in folds:
        assert dates[train].max() < dates[valid].min()
        assert len(train) >= tune.MIN_TRAIN_ROWS and len(valid) >= tune.MIN_VALID_ROWS


def test_folds_keep_same_day_listings_together():
    dates = np.repeat(np.array(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"],
                               dtype="datetime64[ns]"), 40)
    for train, valid in tune.walk_forward_folds(dates, 4):
        assert set(dates[train]).isdisjoint(set(dates[valid]))


def test_too_few_dates_or_rows_for_any_fold_is_refused():
    dates = np.repeat(np.array(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"],
                               dtype="datetime64[ns]"), 3)
    with pytest.raises(ValueError, match="no walk-forward fold"):
        tune.walk_forward_folds(dates, 4)
    with pytest.raises(ValueError, match="distinct listing dates"):
        tune.walk_forward_folds(dates[:6], 9)


def test_search_respects_the_bounded_space_and_returns_a_table():
    best, table = tune.search(make_ipos(200, seed=1), n_trials=6, seed=5, n_folds=3)
    assert set(best) == set(tune.SPACE)
    assert len(table) == 6 and set(table["state"]) == {"COMPLETE"}
    for name, spec in tune.SPACE.items():
        values = table[f"p_{name}"]
        assert values.min() >= spec[1] - 1e-12 and values.max() <= spec[2] + 1e-12, name
        if spec[0] == "int_step":
            assert ((values - spec[1]) % spec[3] == 0).all(), name
    for col in ("score", "pinball_p50", "coverage_p10_p90", "n_valid"):
        assert col in table.columns and np.isfinite(table[col]).all()
    assert (table["coverage_p10_p90"].between(0, 1)).all()


def test_same_seed_gives_the_same_best_params_and_table():
    df = make_ipos(180, seed=2)
    best_a, table_a = tune.search(df, n_trials=5, seed=11, n_folds=3)
    best_b, table_b = tune.search(df, n_trials=5, seed=11, n_folds=3)
    assert best_a == best_b
    pd.testing.assert_frame_equal(table_a, table_b)


def test_best_params_are_the_lowest_scoring_trial():
    best, table = tune.search(make_ipos(160, seed=4), n_trials=5, seed=3, n_folds=3)
    best_row = table.loc[table["score"].idxmin()]
    for name in tune.SPACE:
        assert best[name] == pytest.approx(best_row[f"p_{name}"])


def test_time_budget_stops_between_trials():
    best, table = tune.search(make_ipos(200, seed=6), n_trials=2000, seed=1, time_budget_s=1.5, n_folds=3)
    assert 1 <= len(table) < 2000
    assert set(best) == set(tune.SPACE)


@pytest.mark.parametrize("n_trials", [0, -1])
def test_zero_trials_is_refused(n_trials):
    with pytest.raises(ValueError, match="n_trials"):
        tune.search(make_ipos(100), n_trials=n_trials, seed=1)


def test_missing_target_column_is_refused():
    df = make_ipos(100).drop(columns=["list_open_gain_pct"])
    with pytest.raises(ValueError, match="list_open_gain_pct"):
        tune.search(df, n_trials=2, seed=1)


def test_too_few_labelled_rows_is_refused():
    with pytest.raises(ValueError, match="need more"):
        tune.search(make_ipos(30), n_trials=2, seed=1)


def test_make_xy_rows_must_come_from_its_input(monkeypatch):
    shifted = types.ModuleType("ipo_radar.model.features")
    shifted.asof_expand = lambda df: df.assign(as_of="pre_open")
    shifted.make_xy = lambda df, target: (df[["sub_total"]].set_axis(df.index + 10_000), df[target])
    monkeypatch.setitem(sys.modules, "ipo_radar.model.features", shifted)
    with pytest.raises(ValueError, match="not in its input"):
        tune.search(make_ipos(100), n_trials=2, seed=1)


def test_fold_counts_report_every_market_and_survivor_rows():
    df = make_ipos(240, seed=8)
    counts = tune.fold_counts(df, n_folds=3)
    assert set(counts.columns) == {"fold", "split", "market", "rows", "survivor_rows"}
    assert set(counts["market"]) == {"IN_MAIN", "IN_SME", "US", "HK"}
    valid = counts[counts["split"] == "valid"]
    assert valid["rows"].sum() <= 240 and (valid["survivor_rows"] <= valid["rows"]).all()
    assert counts["survivor_rows"].sum() > 0  # make_ipos tags every fifth row with +px_join


def test_lgbm_kwargs_adds_the_fixed_settings_without_changing_the_searched_ones():
    kwargs = tune.lgbm_kwargs({"num_leaves": 9})
    assert kwargs["num_leaves"] == 9
    assert kwargs["subsample_freq"] == 1 and kwargs["deterministic"] is True and kwargs["n_jobs"] == 1
