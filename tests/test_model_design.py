"""Design rules, the fitting primitives (monotone scope, conformal coverage, interval adjustment, seed ensemble) and the
live-context helper."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ipo_radar.data import context as ctx_mod
from ipo_radar.model import features as F
from ipo_radar.model import predict as P
from ipo_radar.model.bundle import Ensemble
from ipo_radar.model.design import DEFAULT_DESIGN, SET_MARKETS, Design, segment_of
from ipo_radar.model.fitting import (MIN_SEG_RESID, adjust_intervals, conformal_from_oof, lgb_params,
                                     monotone_vector)
from ipo_radar.schema import LiveIPO


# ---- design -------------------------------------------------------------------------------------------
def test_segments_group_us_hk_and_other_as_global():
    assert segment_of("IN_MAIN") == "IN_MAIN" and segment_of("IN_SME") == "IN_SME"
    assert {segment_of(m) for m in ("US", "HK", "OTHER")} == {"GLOBAL"}
    assert segment_of("MARS") is None and segment_of(None) is None


def test_design_rejects_unknown_routing_and_gmp_model():
    with pytest.raises(ValueError, match="gmp_model"):
        Design(gmp_model="forest")
    with pytest.raises(ValueError, match="cannot be routed"):
        Design(routing={"IN_MAIN": "GLOBAL", "IN_SME": "pooled", "GLOBAL": "pooled"})
    with pytest.raises(ValueError, match="unknown segment"):
        Design(routing={"EU": "pooled"})


def test_sets_used_always_includes_pooled_and_skips_uncovered_segments():
    spec = Design(routing={"IN_MAIN": "india", "IN_SME": "india", "GLOBAL": None})
    assert spec.sets_used() == ["india", "pooled"]
    assert Design().sets_used() == ["pooled"]


def test_default_design_uses_only_defined_sets():
    for name in DEFAULT_DESIGN.sets_used():
        assert name in SET_MARKETS


# ---- fitting ------------------------------------------------------------------------------------------
def test_monotone_vector_marks_subscription_and_gmp_only():
    feats = ["log_sub_total", "log_sub_qib_d1", "gmp_pct_last", "idx_ret_20d", "market", "lot_size"]
    assert monotone_vector(feats, True) == [1, 1, 1, 0, 0, 0]
    assert monotone_vector(feats, False) is None


def test_lgb_params_override_and_quick_mode():
    base = lgb_params(1000, 3, quick=True)
    assert base["n_estimators"] == 40
    tuned = lgb_params(1000, 3, quick=False, params={"num_leaves": 12, "learning_rate": 0.05})
    assert tuned["num_leaves"] == 12 and tuned["learning_rate"] == 0.05 and tuned["n_estimators"] == 300


def test_conformal_adjustment_reaches_nominal_coverage_out_of_fold():
    """Raw intervals that are too narrow get widened until ~80% of held-out outcomes fall inside."""
    rng = np.random.default_rng(0)
    n = 800
    y = rng.standard_t(df=3, size=n) * 10
    oof = pd.DataFrame({"segment": ["IN_MAIN"] * n, "year": 2020, "y": y, "p10": np.full(n, -3.0),
                        "p50": np.zeros(n), "p90": np.full(n, 3.0), "p_raw": np.full(n, 0.5)})
    conf, counts = conformal_from_oof(oof)
    assert counts["ALL"] == n and conf["ALL"] > 0
    lo, hi = adjust_intervals(conf, oof["p10"].to_numpy(), oof["p50"].to_numpy(), oof["p90"].to_numpy(),
                              oof["segment"].tolist())
    coverage = np.mean((y >= lo) & (y <= hi))
    assert 0.78 <= coverage <= 0.92, coverage


def test_segment_adjustment_needs_enough_residuals():
    n = MIN_SEG_RESID - 10
    oof = pd.DataFrame({"segment": ["IN_SME"] * n + ["IN_MAIN"] * (MIN_SEG_RESID + 5),
                        "year": 2020, "y": np.zeros(2 * MIN_SEG_RESID - 5), "p10": -1.0, "p50": 0.0, "p90": 1.0,
                        "p_raw": 0.5})
    conf, counts = conformal_from_oof(oof)
    assert "IN_SME" not in conf and "IN_MAIN" in conf
    lo, hi = adjust_intervals(conf, np.array([-1.0, -1.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0]),
                              ["IN_SME", "IN_MAIN"])
    assert lo[0] == pytest.approx(-1.0 - conf["ALL"]) and lo[1] == pytest.approx(-1.0 - conf["IN_MAIN"])


def test_adjusted_interval_always_contains_the_median():
    lo, hi = adjust_intervals({"ALL": -50.0}, np.array([-2.0]), np.array([5.0]), np.array([9.0]), [None])
    assert lo[0] <= 5.0 <= hi[0]


def test_ensemble_averages_members_and_matches_a_single_member():
    class M:
        def __init__(self, v):
            self.v = v

        def predict(self, X, pred_contrib=False):
            return np.full(len(X), self.v)

    assert np.allclose(Ensemble([M(2.0)]).predict(np.zeros((3, 1))), 2.0)
    assert np.allclose(Ensemble([M(2.0), M(4.0)]).predict(np.zeros((3, 1))), 3.0)
    with pytest.raises(ValueError):
        Ensemble([])


# ---- live context ------------------------------------------------------------------------------------
def test_live_context_for_uses_the_public_helper_and_never_a_future_date(monkeypatch):
    seen = []

    def fake(market, today=None):
        seen.append((market, today))
        return {"idx_ret_5d": 1.0, "idx_ret_20d": 2.0, "idx_vol_20d": 3.0, "vix_level": 4.0,
                "idx_ret_issue_window": float("nan"), "context_as_of": today, "context_notes": ["ok"]}

    monkeypatch.setattr(ctx_mod, "live_context", fake)
    P._live_context_cached.cache_clear()
    values, notes = P.live_context_for("IN_MAIN", open_date=date(2999, 1, 1))
    assert seen and seen[-1][0] == "IN_MAIN"
    assert seen[-1][1] <= date.today()     # a future open uses today's closes, not later ones
    assert values["idx_ret_20d"] == 2.0 and "ok" in notes


def test_live_context_failure_is_missing_context_not_a_crash(monkeypatch):
    def boom(market, today=None):
        raise RuntimeError("nse down")

    monkeypatch.setattr(ctx_mod, "live_context", boom)
    P._live_context_cached.cache_clear()
    values, notes = P.live_context_for("IN_SME")
    assert np.isnan(values["idx_ret_20d"])
    assert any("unavailable" in n for n in notes)


def test_live_row_context_is_missing_when_no_context_is_given():
    live = LiveIPO(ipo_id="z", market="IN_MAIN", company="Z", status="upcoming", price_low=10.0, price_high=11.0)
    row = F.live_raw_row(live)
    assert np.isnan(row["idx_ret_20d"])
