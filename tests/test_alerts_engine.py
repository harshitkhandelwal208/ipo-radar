"""Rule table tests for alerts/engine.py with synthetic LiveIPO and Prediction objects. All offline."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pytest

from ipo_radar.alerts.engine import IST, evaluate, snapshot_of
from ipo_radar.alerts.notify import Alert
from ipo_radar.schema import LiveIPO, Prediction, Sourced

NOW = datetime(2026, 10, 15, 11, 0, tzinfo=IST)     # daytime, outside quiet hours
TODAY = date(2026, 10, 15)
FETCHED = NOW - timedelta(minutes=4)


class MemState:
    """Stands in for State: an in-memory sent set and snapshot map."""

    def __init__(self, snapshots: dict[str, dict[str, Any]] | None = None) -> None:
        self.sent: set[str] = set()
        self.snapshots = snapshots or {}

    def already_sent(self, key: str) -> bool:
        return key in self.sent

    def last_snapshot(self, ipo_id: str) -> dict[str, Any] | None:
        return self.snapshots.get(ipo_id)

    def mark(self, alerts: list[Alert]) -> None:
        self.sent.update(a.key for a in alerts)


def sourced(value: float | None, source: str = "nse") -> Sourced:
    return Sourced(value=value, source=source, fetched_at=FETCHED)


def make_ipo(**kw: Any) -> LiveIPO:
    base: dict[str, Any] = dict(ipo_id="ipo1", market="IN_MAIN", company="Acme Ltd", status="open",
                                price_low=95.0, price_high=100.0, lot_size=150, issue_size_m=2500.0,
                                sources=["nse", "chittorgarh"])
    base.update(kw)
    return LiveIPO(**base)


def make_pred(**kw: Any) -> Prediction:
    base: dict[str, Any] = dict(ipo_id="ipo1", as_of="pre_open", open_gain_pct_p10=-5.0, open_gain_pct_p50=12.0,
                                open_gain_pct_p90=30.0, p_positive_open=0.7, close_gain_pct_p50=None,
                                drivers=[], baseline_gmp_pct=None, model_version="t", reliable=True, caveats=[])
    base.update(kw)
    return Prediction(**base)


def run(live, state=None, preds=None, now=NOW, cfg=None) -> list[Alert]:
    return evaluate(live, preds or {}, state or MemState(), now=now, cfg=cfg or {})


def keys(alerts: list[Alert]) -> list[str]:
    return [a.key for a in alerts]


# ---- once-only behaviour ----------------------------------------------------------------------------

def test_opens_today_fires_once_and_second_evaluation_is_empty():
    ipo = make_ipo(open_date=TODAY, close_date=TODAY + timedelta(days=3))
    state = MemState()
    first = run([ipo], state, {"ipo1": make_pred()})
    assert keys(first) == ["ipo1:opens-today:2026-10-15"]
    state.mark(first)
    assert run([ipo], state, {"ipo1": make_pred()}) == []


def test_each_rule_fires_once_across_repeated_evaluations():
    ipo = make_ipo(
        open_date=TODAY, close_date=TODAY, allotment_date=TODAY,
        listing_date=TODAY, price_low=100.0, price_high=100.0, status="listing_today",
        sub={"sub_total": sourced(12.0), "sub_qib": sourced(30.0), "sub_nii": sourced(6.0),
             "sub_retail": sourced(2.0)},
        gmp={"gmp": sourced(25.0)},
        listing={"open": 130.0},
        warnings=["nse: unavailable"],
    )
    state = MemState()
    first = run([ipo], state, {"ipo1": make_pred()})
    assert set(keys(first)) >= {
        "ipo1:opens-today:2026-10-15",
        "ipo1:closes-today:2026-10-15",
        "ipo1:allotment:2026-10-15",
        "ipo1:listing-today:2026-10-15",
        "ipo1:listed-open:2026-10-15",
        "source-down:nse:2026-10-15",
    }
    state.mark(first)
    assert run([ipo], state, {"ipo1": make_pred()}) == []


def test_opens_soon_within_two_days_but_not_three():
    state = MemState()
    soon = run([make_ipo(open_date=TODAY + timedelta(days=2))], state)
    assert keys(soon) == ["ipo1:opens-soon:2026-10-17"]
    assert soon[0].priority == 2
    assert run([make_ipo(open_date=TODAY + timedelta(days=3))], MemState()) == []


def test_opens_today_priority_and_text_with_band_lot_size_and_model_view():
    ipo = make_ipo(open_date=TODAY, sub={"sub_total": sourced(1.0)})
    (a,) = run([ipo], MemState(), {"ipo1": make_pred()})
    assert a.title == "Acme Ltd opens today"
    assert a.priority == 3
    assert a.body.splitlines()[0] == "band ₹95-₹100 · lot 150 · size ₹250 cr"
    assert "model +12% open (range -5% to +30%)" in a.body
    assert a.body.splitlines()[-1] == "nse · 4 min old"


def test_band_line_falls_back_to_single_price_and_raw_size_outside_india():
    ipo = make_ipo(market="US", price_low=None, price_high=20.0, issue_size_m=300.0, open_date=TODAY,
                   sub={"sub_total": sourced(1.0)})
    (a,) = run([ipo], MemState())
    assert a.body.splitlines()[0] == "price $20 · lot 150 · size 300m"


def test_fixed_price_issue_shows_one_price_not_a_band():
    (a,) = run([make_ipo(open_date=TODAY, price_low=40.0, price_high=40.0, sub={"sub_total": sourced(1.0)})],
               MemState())
    assert a.body.splitlines()[0] == "price ₹40 · lot 150 · size ₹250 cr"


def test_closes_today_lists_known_subscriptions_and_gmp_only():
    ipo = make_ipo(close_date=TODAY, sub={"sub_qib": sourced(3.2), "sub_retail": sourced(None)},
                   gmp={"gmp": sourced(-15.0)})
    (a,) = run([ipo], MemState())
    assert a.key == "ipo1:closes-today:2026-10-15"
    assert a.priority == 4
    assert a.body.splitlines()[0] == "QIB 3.2x · GMP -₹15"


def test_allotment_day_fires_with_reminder():
    (a,) = run([make_ipo(allotment_date=TODAY)], MemState())
    assert a.key == "ipo1:allotment:2026-10-15"
    assert a.title == "Allotment today: Acme Ltd"
    assert "registrar" in a.body


def test_listing_tomorrow_uses_model_range():
    (a,) = run([make_ipo(listing_date=TODAY + timedelta(days=1))], MemState(), {"ipo1": make_pred()})
    assert a.key == "ipo1:listing-tomorrow:2026-10-16"
    assert a.priority == 4
    assert a.body.splitlines()[0] == "model +12% open (range -5% to +30%)"


def test_listing_tomorrow_without_model_says_so():
    (a,) = run([make_ipo(listing_date=TODAY + timedelta(days=1))], MemState())
    assert a.body.splitlines()[0] == "no model view"


def test_model_view_flags_unreliable_inputs():
    (a,) = run([make_ipo(open_date=TODAY)], MemState(), {"ipo1": make_pred(reliable=False)})
    assert "[inputs outside training data]" in a.body


def _listed_open(alerts: list[Alert]) -> list[Alert]:
    return [a for a in alerts if ":listed-open:" in a.key]


def test_listed_open_compares_realised_gain_with_prediction_inside_range():
    ipo = make_ipo(listing_date=TODAY, price_low=100.0, price_high=100.0, listing={"open": 115.0})
    (a,) = _listed_open(run([ipo], MemState(), {"ipo1": make_pred()}))
    assert a.key == "ipo1:listed-open:2026-10-15"
    assert a.title == "Acme Ltd opened +15.0%"
    assert "model said +12% (range -5% to +30%), inside the range" in a.body


def test_listed_open_outside_range_and_no_gain_without_fixed_price():
    ipo = make_ipo(listing_date=TODAY, price_low=100.0, price_high=100.0, listing={"open": 150.0})
    (a,) = _listed_open(run([ipo], MemState(), {"ipo1": make_pred()}))
    assert "outside the range" in a.body

    book_built = make_ipo(listing_date=TODAY, price_low=95.0, price_high=100.0, listing={"open": 150.0})
    assert _listed_open(run([book_built], MemState())) == []


# ---- milestones -------------------------------------------------------------------------------------

def _total(value: float | None) -> LiveIPO:
    return make_ipo(sub={"sub_total": sourced(value)})


def test_milestone_fires_on_upward_crossing_only_for_highest_threshold():
    state = MemState({"ipo1": {"status": "open", "gmp": None, "sub": {"sub_total": 0.9}}})
    (a,) = run([_total(12.0)], state)
    assert a.key == "ipo1:milestone:sub_total:10x"
    assert a.title == "Acme Ltd: total crosses 10x"
    assert a.priority == 3


def test_milestone_does_not_fire_when_already_above_threshold():
    state = MemState({"ipo1": {"status": "open", "gmp": None, "sub": {"sub_total": 5.2}}})
    assert run([_total(5.5)], state) == []


def test_milestone_does_not_fire_on_downward_move():
    state = MemState({"ipo1": {"status": "open", "gmp": None, "sub": {"sub_total": 12.0}}})
    assert run([_total(4.0)], state) == []


def test_milestone_fires_exactly_on_threshold_value():
    state = MemState({"ipo1": {"status": "open", "gmp": None, "sub": {"sub_total": 4.9}}})
    (a,) = run([_total(5.0)], state)
    assert a.key == "ipo1:milestone:sub_total:5x"


def test_milestone_needs_a_previous_value():
    assert run([_total(12.0)], MemState()) == []
    state = MemState({"ipo1": {"status": "open", "gmp": None, "sub": {}}})
    assert run([_total(12.0)], state) == []


def test_milestone_ignores_none_current_value():
    state = MemState({"ipo1": {"status": "open", "gmp": None, "sub": {"sub_total": 0.5}}})
    assert run([_total(None)], state) == []


def test_milestone_for_each_category_is_independent():
    ipo = make_ipo(sub={"sub_qib": sourced(6.0), "sub_nii": sourced(1.4), "sub_retail": sourced(0.5)})
    state = MemState({"ipo1": {"status": "open", "gmp": None,
                               "sub": {"sub_qib": 2.0, "sub_nii": 0.2, "sub_retail": 0.9}}})
    assert keys(run([ipo], state)) == ["ipo1:milestone:sub_qib:5x", "ipo1:milestone:sub_nii:1x"]


def test_milestone_thresholds_come_from_config():
    state = MemState({"ipo1": {"status": "open", "gmp": None, "sub": {"sub_total": 0.5}}})
    (a,) = run([_total(3.0)], state, cfg={"thresholds": {"milestones": [2, 3]}})
    assert a.key == "ipo1:milestone:sub_total:3x"


# ---- GMP --------------------------------------------------------------------------------------------

def _gmp(value: float | None) -> LiveIPO:
    return make_ipo(gmp={"gmp": sourced(value, "ipomonitor")})


def _prev_gmp(value: float | None) -> MemState:
    return MemState({"ipo1": {"status": "open", "gmp": value, "sub": {}}})


@pytest.mark.parametrize(
    ("prev", "cur", "fires"),
    [
        (100.0, 115.0, False),   # move 15 < max(10, 20% of 100 = 20)
        (100.0, 120.0, True),    # move 20 == threshold, fires
        (30.0, 39.0, False),     # move 9 < max(10, 6)
        (30.0, 40.0, True),      # move 10 == max(10, 6)
        (-30.0, -41.0, True),    # threshold uses the magnitude of the previous value
        (-30.0, -39.0, False),
    ],
)
def test_gmp_move_threshold(prev, cur, fires):
    alerts = run([_gmp(cur)], _prev_gmp(prev))
    assert bool(alerts) is fires
    if fires:
        (a,) = alerts
        assert a.key == f"ipo1:gmp-move:{prev:.0f}>{cur:.0f}"
        assert a.priority == 3


def test_gmp_sign_flip_fires_even_when_the_move_is_small():
    (a,) = run([_gmp(-2.0)], _prev_gmp(5.0))
    assert a.title == "Acme Ltd GMP turned negative"
    assert a.priority == 4
    (b,) = run([_gmp(3.0)], _prev_gmp(-4.0))
    assert b.title == "Acme Ltd GMP turned positive"


def test_gmp_no_alert_when_previous_or_current_is_none_or_missing():
    assert run([_gmp(None)], _prev_gmp(100.0)) == []
    assert run([_gmp(500.0)], _prev_gmp(None)) == []
    assert run([make_ipo()], _prev_gmp(100.0)) == []


def test_gmp_thresholds_come_from_config():
    # 6 INR move: below the default 10 INR floor, above the configured 5 INR floor
    assert run([_gmp(16.0)], _prev_gmp(10.0)) == []
    (a,) = run([_gmp(16.0)], _prev_gmp(10.0), cfg={"thresholds": {"gmp_move_min_inr": 5}})
    assert a.key == "ipo1:gmp-move:10>16"


# ---- first observation, missing data, text ----------------------------------------------------------

def test_first_observation_stores_without_change_alerts_but_schedule_still_fires():
    ipo = make_ipo(open_date=TODAY, gmp={"gmp": sourced(50.0)}, sub={"sub_total": sourced(40.0)})
    assert keys(run([ipo], MemState())) == ["ipo1:opens-today:2026-10-15"]


def test_none_values_never_produce_alerts_or_crash():
    ipo = make_ipo(open_date=None, close_date=None, listing_date=None,
                   sub={"sub_qib": sourced(None), "sub_total": sourced(None)},
                   gmp={"gmp": sourced(None)}, listing={})
    state = MemState({"ipo1": {"status": "open", "gmp": 100.0, "sub": {"sub_qib": 0.2, "sub_total": 0.2}}})
    assert run([ipo], state) == []


def test_snapshot_of_keeps_numbers_and_status_only():
    snap = snapshot_of(make_ipo(sub={"sub_total": sourced(3.0), "sub_qib": sourced(None)},
                                gmp={"gmp": sourced(20.0)}))
    assert snap == {"status": "open", "gmp": 20.0, "sub": {"sub_total": 3.0, "sub_qib": None}}


def test_evaluation_is_deterministic_for_the_same_inputs():
    ipo = make_ipo(open_date=TODAY, close_date=TODAY)
    state = MemState()
    assert run([ipo], state) == run([ipo], state)


# ---- quiet hours ------------------------------------------------------------------------------------

def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, 15, hour, minute, tzinfo=IST)


@pytest.mark.parametrize(("when", "quiet"), [
    (_at(23, 30), True),
    (_at(0, 30), True),
    (_at(6, 59), True),
    (_at(7, 0), False),      # end is exclusive
    (_at(22, 59), False),
    (_at(12, 0), False),
])
def test_quiet_window_crosses_midnight(when, quiet):
    (a,) = run([make_ipo(open_date=TODAY)], MemState(), now=when)
    assert a.priority == (1 if quiet else 3)


def test_quiet_hours_keep_closes_today_and_listing_day_loud():
    ipo = make_ipo(close_date=TODAY, listing_date=TODAY, listing={})
    alerts = {a.key: a.priority for a in run([ipo], MemState(), now=_at(23, 30))}
    assert alerts == {"ipo1:closes-today:2026-10-15": 4, "ipo1:listing-today:2026-10-15": 4}


def test_quiet_hours_silence_listing_tomorrow_and_gmp_flip():
    ipo = make_ipo(listing_date=TODAY + timedelta(days=1), gmp={"gmp": sourced(-3.0)})
    alerts = run([ipo], _prev_gmp(10.0), now=_at(23, 30))
    assert sorted(a.priority for a in alerts) == [1, 1]


def test_quiet_hours_can_be_disabled_or_moved():
    ipo = make_ipo(open_date=TODAY)
    (a,) = run([ipo], MemState(), now=_at(23, 30), cfg={"thresholds": {"quiet_hours": ""}})
    assert a.priority == 3
    (b,) = run([ipo], MemState(), now=_at(12, 0), cfg={"thresholds": {"quiet_hours": "11:00-13:00"}})
    assert b.priority == 1


def test_bad_quiet_hours_string_is_rejected():
    with pytest.raises(ValueError, match="quiet_hours"):
        run([make_ipo(open_date=TODAY)], MemState(), cfg={"thresholds": {"quiet_hours": "late"}})


# ---- data source down --------------------------------------------------------------------------------

def test_data_source_down_alerts_once_per_source_per_day():
    live = [
        make_ipo(ipo_id="a", company="Alpha", warnings=["nse: unavailable (HTTP 503)"]),
        make_ipo(ipo_id="b", company="Beta", warnings=["nse unavailable", "chittorgarh stale (2 h old)"]),
    ]
    state = MemState()
    alerts = run(live, state)
    assert keys(alerts) == ["source-down:nse:2026-10-15", "source-down:chittorgarh:2026-10-15"]
    nse = alerts[0]
    assert nse.title == "Data source down: nse"
    assert nse.body.splitlines()[0] == "2 IPO(s) affected: Alpha, Beta"
    assert nse.priority == 2
    state.mark(alerts)
    assert run(live, state) == []


def test_data_source_down_ignores_unrelated_warnings():
    live = [make_ipo(warnings=["GMP sources disagree: 40 vs 70"])]
    assert run(live, MemState()) == []
