"""Serving tests: stage choice from the live state, routing to model sets, the skill gate, the reliable flag rules, the
conformal adjustment, and the missing-model path. A stub bundle covers the pure logic; a real quick bundle covers the
reliability rules."""
from __future__ import annotations

import dataclasses
from datetime import date

import numpy as np
import pytest

from ipo_radar import config
from ipo_radar.model import ModelNotTrained
from ipo_radar.model import features as F
from ipo_radar.model import predict as P
from ipo_radar.model.bundle import Bundle, StageModels
from ipo_radar.model.design import SEGMENTS
from ipo_radar.model.train import train
from ipo_radar.schema import LiveIPO, Prediction, Sourced
from model_synth import make_ipos

CONTEXT = {"idx_ret_5d": 1.0, "idx_ret_20d": 2.0, "idx_vol_20d": 15.0, "vix_level": 18.0}


def _ipo(**kw) -> LiveIPO:
    base = dict(ipo_id="P1", market="IN_MAIN", company="Pred Ltd", status="upcoming",
                open_date=date(2026, 10, 20), close_date=date(2026, 10, 22), listing_date=date(2026, 10, 27),
                price_low=95.0, price_high=100.0, lot_size=150, issue_size_m=120.0, fresh_issue_pct=80.0)
    base.update(kw)
    return LiveIPO(**base)


def _skill(p50: bool = True, prob: bool = True) -> dict:
    flag = {"n": 100, "p50_beats_median": p50, "p50_worse_than_median": False, "mae_diff": [-1.0, -2.0, 0.0],
            "brier_beats_base": prob,
            "brier_diff": [-0.01, -0.02, 0.0], "auc": 0.6}
    return {s: {g: dict(flag) for g in SEGMENTS} for s in F.STAGES}


# ---- stage choice (pure) -----------------------------------------------------------------------------
def test_upcoming_issue_uses_pre_open():
    assert P.choose_stage(_ipo(status="upcoming")) == ("pre_open", [])


def test_open_issue_uses_latest_day_with_data():
    assert P.choose_stage(_ipo(status="open", sub_by_day={1: {"sub_total": 2.0}}))[0] == "day1"
    assert P.choose_stage(_ipo(status="open", sub_by_day={1: {"sub_total": 2.0}, 2: {"sub_total": 5.0}}))[0] == "day2"
    stage, notes = P.choose_stage(_ipo(status="open", sub_by_day={1: {}, 2: {}, 3: {"sub_total": 9.0}}))
    assert stage == "pre_open" and any("day-3" in n for n in notes)
    stage, notes = P.choose_stage(_ipo(status="open", sub_by_day={1: {"sub_total": 1.0}, 2: {"sub_total": 2.0},
                                                                 3: {"sub_total": 3.0}}))
    assert stage == "day2" and any("day-3" in n for n in notes)


def test_last_issue_day_with_live_subscription_uses_close_with_the_caveat():
    sub = {"sub_total": Sourced(9.0, "nse"), "sub_qib": Sourced(1.5, "nse"), "sub_retail": Sourced(4.0, "nse")}
    live = _ipo(status="open", close_date=date(2026, 10, 15), sub=sub)
    stage, notes = P.choose_stage(live, today=date(2026, 10, 15))
    assert stage == "close"
    assert P.LAST_DAY_CAVEAT in notes


def test_open_before_the_last_day_with_no_snapshot_keeps_pre_open():
    sub = {"sub_total": Sourced(9.0, "nse")}
    live = _ipo(status="open", close_date=date(2026, 10, 15), sub=sub)
    stage, notes = P.choose_stage(live, today=date(2026, 10, 14))
    assert stage == "pre_open" and P.LAST_DAY_CAVEAT not in notes


def test_last_issue_day_without_any_subscription_keeps_pre_open():
    live = _ipo(status="open", close_date=date(2026, 10, 15))
    stage, _ = P.choose_stage(live, today=date(2026, 10, 15))
    assert stage == "pre_open"


def test_day1_snapshot_beats_the_last_day_rule():
    sub = {"sub_total": Sourced(9.0, "nse")}
    live = _ipo(status="open", close_date=date(2026, 10, 15), sub=sub, sub_by_day={1: {"sub_total": 2.0}})
    stage, notes = P.choose_stage(live, today=date(2026, 10, 15))
    assert stage == "day1" and P.LAST_DAY_CAVEAT not in notes


def test_closed_issue_uses_close_or_pre_listing_with_a_gmp_caveat():
    stage, notes = P.choose_stage(_ipo(status="closed", sub={"sub_total": Sourced(10.0, "nse")}))
    assert stage == "close" and notes == []
    stage, notes = P.choose_stage(_ipo(status="closed", sub={"sub_total": Sourced(10.0, "nse")},
                                       gmp={"gmp_pct": Sourced(12.0, "gmp")}))
    assert stage == "pre_listing"
    assert any("earlier than the final pre-listing reading" in n for n in notes)


def test_closed_without_final_subscription_is_noted():
    stage, notes = P.choose_stage(_ipo(status="closed"))
    assert stage == "close" and any("not published" in n for n in notes)


# ---- missing model -------------------------------------------------------------------------------------
def test_missing_model_raises_model_not_trained_with_the_fix(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS", tmp_path)
    with pytest.raises(ModelNotTrained, match="ipo train") as info:
        P.predict(_ipo())
    assert isinstance(info.value, FileNotFoundError)   # cli.backend maps this to its "run ipo train" message
    with pytest.raises(ModelNotTrained):
        P.load_model(tmp_path / "missing.joblib")


def test_model_built_with_another_feature_set_is_refused(tmp_path):
    import joblib
    stub = _stub_bundle()
    stub.stages["pre_open"].features = ["lot_size"]
    joblib.dump(stub, tmp_path / "bundle.joblib")
    with pytest.raises(ModelNotTrained, match="another feature set"):
        P.load_model(tmp_path)


# ---- stub bundle --------------------------------------------------------------------------------------
class _Const:
    def __init__(self, value: float):
        self.value = value

    def predict(self, X, pred_contrib: bool = False):
        if pred_contrib:
            out = np.zeros((len(X), X.shape[1] + 1))
            out[:, -1] = self.value
            return out
        return np.full(len(X), self.value)


class _Prob:
    def predict_proba(self, X):
        p = np.full(len(X), 0.6)
        return np.column_stack([1 - p, p])


def _stub_stage(q10: float, q50: float, q90: float, conf: dict | None = None) -> StageModels:
    return StageModels(
        stage="pre_open", features=F.stage_features("pre_open"), q10=_Const(q10), q50=_Const(q50), q90=_Const(q90),
        close_q50=None, clf=_Prob(), iso=None, n_open=500, n_close=0, n_by_market={"IN_MAIN": 500, "IN_SME": 300},
        core={}, ranges={}, conf=conf or {})


def _stub_bundle(q10: float = 5.0, q50: float = 0.0, q90: float = -5.0, **kw) -> Bundle:
    return Bundle(version="stub-1", trained_at="now", n_rows=500, stages={"pre_open": _stub_stage(q10, q50, q90)},
                  vocab=F.Vocab(sectors=("A",), exchanges=()), seed=7, **kw)


def test_crossed_quantiles_are_put_in_order_and_the_median_stays_inside():
    p = P.predict(_ipo(), bundle=_stub_bundle(), context=CONTEXT)
    assert p.open_gain_pct_p10 <= p.open_gain_pct_p50 <= p.open_gain_pct_p90
    assert p.open_gain_pct_p10 == pytest.approx(-p.open_gain_pct_p90)


def test_conformal_adjustment_widens_the_interval_around_the_median():
    bundle = Bundle(version="stub-2", trained_at="now", n_rows=1,
                    stages={"pre_open": _stub_stage(-10.0, 0.0, 10.0, conf={"ALL": 3.0})},
                    vocab=F.Vocab(sectors=(), exchanges=()), seed=7)
    p = P.predict(_ipo(), bundle=bundle, context=CONTEXT)
    base = P.predict(_ipo(), bundle=dataclasses.replace(bundle, stages={"pre_open": _stub_stage(-10.0, 0.0, 10.0)}),
                     context=CONTEXT)
    assert p.open_gain_pct_p10 == pytest.approx(base.open_gain_pct_p10 - 3.0, abs=1e-6)
    assert p.open_gain_pct_p90 == pytest.approx(base.open_gain_pct_p90 + 3.0, abs=1e-6)


def test_prediction_fields_have_the_schema_types():
    p = P.predict(_ipo(), bundle=_stub_bundle(q10=-1, q50=0.5, q90=2), context=CONTEXT)
    assert isinstance(p, Prediction)
    assert p.as_of == "pre_open" and p.model_version == "stub-1"
    assert 0.0 <= p.p_positive_open <= 1.0
    assert p.close_gain_pct_p50 is None and p.baseline_gmp_pct is None
    assert len(p.drivers) <= P.TOP_DRIVERS


def test_missing_stage_falls_back_to_the_earlier_model_and_says_so():
    p = P.predict(_ipo(), as_of="day2", bundle=_stub_bundle(), context=CONTEXT)
    assert p.as_of == "pre_open"
    assert any("no model for stage day2" in c for c in p.caveats)


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match="unknown stage"):
        P.predict(_ipo(), as_of="after_listing", bundle=_stub_bundle())


def test_no_stage_at_or_before_the_requested_one_is_a_missing_model():
    bundle = _stub_bundle()
    bundle = dataclasses.replace(bundle, stages={"close": bundle.stages["pre_open"]})
    with pytest.raises(ModelNotTrained, match="no model covers"):
        P.predict(_ipo(), as_of="pre_open", bundle=bundle, context=CONTEXT)


def test_segment_routing_sends_an_india_sme_issue_to_its_own_model_set():
    pooled = _stub_stage(0.0, 0.0, 0.0)
    sme = _stub_stage(1.0, 1.0, 1.0)
    bundle = Bundle(version="routed", trained_at="now", n_rows=1, stages={"pre_open": pooled},
                    vocab=F.Vocab(sectors=(), exchanges=()), seed=7,
                    routing={"IN_MAIN": "pooled", "IN_SME": "IN_SME", "GLOBAL": "pooled"},
                    alt_sets={"IN_SME": {"pre_open": sme}})
    p_sme = P.predict(_ipo(market="IN_SME"), bundle=bundle, context=CONTEXT)
    p_main = P.predict(_ipo(market="IN_MAIN"), bundle=bundle, context=CONTEXT)
    assert p_sme.open_gain_pct_p50 == pytest.approx(F.from_model_scale(1.0))
    assert p_main.open_gain_pct_p50 == pytest.approx(0.0, abs=1e-9)


# ---- real quick bundle: skill gate and reliability ------------------------------------------------------
@pytest.fixture(scope="module")
def real_bundle(tmp_path_factory):
    df = make_ipos(n_per_year=120, years=range(2015, 2025), seed=21)
    out = tmp_path_factory.mktemp("pred_models")
    train(df, out_dir=out, seed=7, quick=True)
    return P.load_model(out)


def test_normal_inputs_in_a_seen_market_are_reliable_when_skill_is_shown(real_bundle):
    b = dataclasses.replace(real_bundle, skill=_skill(True, True))
    p = P.predict(_ipo(), bundle=b, context=CONTEXT)
    assert p.reliable, p.caveats
    assert p.model_version == real_bundle.version


def test_no_skill_evidence_is_unreliable_with_the_caveat(real_bundle):
    b = dataclasses.replace(real_bundle, skill={})
    p = P.predict(_ipo(), bundle=b, context=CONTEXT)
    assert not p.reliable
    assert P.NO_SKILL_EVIDENCE in p.caveats


def test_no_clear_difference_gets_its_own_caveat(real_bundle):
    b = dataclasses.replace(real_bundle, skill=_skill(p50=False, prob=True))
    p = P.predict(_ipo(), bundle=b, context=CONTEXT)
    assert not p.reliable
    assert P.NO_CLEAR_CAVEAT in p.caveats
    assert not any("worse than" in c for c in p.caveats)


def _recent(worse: bool) -> dict:
    return {"worse": worse, "n": 500, "years": [2025, 2026], "p50_minus_median": [1.5, 0.9, 2.0] if worse
            else [-0.3, -1.0, 0.4]}


def test_latest_blocks_worse_than_median_gets_the_worse_caveat(real_bundle):
    b = dataclasses.replace(real_bundle, skill=_skill(True, True), recent={"pre_open": _recent(True)})
    p = P.predict(_ipo(), bundle=b, context=CONTEXT)
    assert not p.reliable
    assert any(c.startswith(P.WORSE_CAVEAT) for c in p.caveats)
    assert P.NO_CLEAR_CAVEAT not in p.caveats


def test_latest_blocks_no_clear_difference_is_not_called_worse(real_bundle):
    b = dataclasses.replace(real_bundle, skill=_skill(True, True), recent={"pre_open": _recent(False)})
    p = P.predict(_ipo(), bundle=b, context=CONTEXT)
    assert p.reliable, p.caveats
    assert not any(c.startswith(P.WORSE_CAVEAT) for c in p.caveats)


def test_failing_base_rate_test_adds_the_probability_caveat(real_bundle):
    b = dataclasses.replace(real_bundle, skill=_skill(p50=True, prob=False))
    p = P.predict(_ipo(), bundle=b, context=CONTEXT)
    assert not p.reliable and P.SKILL_PROB_CAVEAT in p.caveats


def test_unseen_market_is_not_reliable(real_bundle):
    p = P.predict(_ipo(market="HK"), bundle=real_bundle, context=CONTEXT)
    assert not p.reliable
    assert any("not seen in training" in c for c in p.caveats)


def test_missing_core_inputs_are_listed_and_unreliable(real_bundle):
    b = dataclasses.replace(real_bundle, skill=_skill(True, True))
    p = P.predict(_ipo(price_low=None, price_high=None, issue_price=None), bundle=b, context=CONTEXT)
    assert not p.reliable
    assert any(c.startswith("missing core inputs") and "price band" in c for c in p.caveats)


def test_subscription_outside_training_range_is_flagged(real_bundle):
    b = dataclasses.replace(real_bundle, skill=_skill(True, True))
    live = _ipo(status="closed", sub={"sub_total": Sourced(1e6, "nse"), "sub_qib": Sourced(5e5, "nse"),
                                      "sub_retail": Sourced(4e5, "nse")})
    p = P.predict(live, bundle=b, context=CONTEXT)
    assert not p.reliable
    assert any("outside the 1st-99th percentile" in c for c in p.caveats)


def test_thin_market_is_flagged(real_bundle, monkeypatch):
    b = dataclasses.replace(real_bundle, skill=_skill(True, True))
    monkeypatch.setattr(P, "MIN_MARKET_ROWS", 10**6)
    p = P.predict(_ipo(), bundle=b, context=CONTEXT)
    assert not p.reliable and any("training IPOs" in c for c in p.caveats)


def test_sme_issue_is_flagged_when_sme_rows_are_few(real_bundle, monkeypatch):
    b = dataclasses.replace(real_bundle, skill=_skill(True, True))
    monkeypatch.setattr(P, "SME_MIN_ROWS", 10**6)
    p = P.predict(_ipo(market="IN_SME"), bundle=b, context=CONTEXT)
    assert not p.reliable and any("few SME IPOs" in c for c in p.caveats)


def test_drivers_are_ordered_by_size_and_limited(real_bundle):
    p = P.predict(_ipo(), bundle=real_bundle, context=CONTEXT)
    sizes = [abs(v) for _, v in p.drivers]
    assert sizes == sorted(sizes, reverse=True)
    assert len(p.drivers) <= P.TOP_DRIVERS


def test_live_gmp_and_pre_listing_prediction_round_trip(real_bundle):
    live = _ipo(status="closed", listing_date=date(2026, 10, 27),
                sub={"sub_total": Sourced(30.0, "nse"), "sub_qib": Sourced(4.0, "nse"),
                     "sub_retail": Sourced(20.0, "nse")},
                gmp={"gmp": Sourced(20.0, "gmp"), "gmp_pct": Sourced(20.0, "gmp")})
    p = P.predict(live, bundle=real_bundle, context=CONTEXT)
    assert p.as_of == "pre_listing" and p.baseline_gmp_pct == 20.0
    assert p.open_gain_pct_p10 <= p.open_gain_pct_p50 <= p.open_gain_pct_p90
    assert 0.0 <= p.p_positive_open <= 1.0
    assert p.close_gain_pct_p50 is not None
