"""Pure CLI pieces: formatting, name matching, the per-IPO analysis rules and the source-health derivation."""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import pandas as pd

from cli_helpers import (fake_ipos, fake_predict, history_frame, hd_fire_sme_conflict, no_skill_predict, now,
                         today, tonbo)
from ipo_radar.alerts.engine import IST
from ipo_radar.cli import fmt, match
from ipo_radar.cli.analysis import build_analysis, predict_all, to_jsonable
from ipo_radar.cli.backend import health_from
from ipo_radar.schema import LiveIPO, Sourced


# ---- fmt -------------------------------------------------------------------------------------------
def test_indian_grouping_and_rupees():
    assert fmt.group_inr(1234567) == "12,34,567"
    assert fmt.group_inr(999) == "999"
    assert fmt.group_inr(-100000) == "-1,00,000"
    assert fmt.rupees(14700) == "₹14,700"
    assert fmt.rupees(1234.5) == "₹1,234.50"
    assert fmt.rupees(42.5, sign=True) == "+₹42.50"
    assert fmt.rupees(-10, sign=True) == "-₹10"
    assert fmt.rupees(0, sign=True) == "₹0"


def test_unknown_values_are_dashes_never_zero():
    for value in (None, float("nan"), pd.NA):
        assert fmt.rupees(value) == fmt.DASH
        assert fmt.multiple(value) == fmt.DASH
        assert fmt.percent(value, sign=True) == fmt.DASH
        assert fmt.crore_text(value) == fmt.DASH
        assert fmt.log_share(value) is None
    assert fmt.multiple(0.0) == "0.00x"  # a real zero is a value, not an unknown


def test_multiples_percent_and_crore():
    assert fmt.multiple(3.456) == "3.46x"
    assert fmt.multiple(123.45) == "123.5x"
    assert fmt.percent(12.34, sign=True) == "+12%"
    assert fmt.percent(12.34, sign=True, digits=1) == "+12.3%"
    assert fmt.crore_text(1234.0) == "₹1,234 cr"
    assert fmt.crore_text(12.5) == "₹12.5 cr"


def test_log_share_clips_at_100x_and_is_monotonic():
    assert fmt.log_share(0.0) == 0.0
    assert abs(fmt.log_share(1.0) - 0.150) < 0.001
    assert abs(fmt.log_share(10.0) - 0.520) < 0.001
    assert fmt.log_share(100.0) == 1.0
    assert fmt.log_share(5000.0) == 1.0
    assert fmt.log_share(2.0) < fmt.log_share(3.0)


def test_age_text_relative_and_unknown():
    n = datetime(2026, 10, 15, 12, 0, tzinfo=IST)
    assert fmt.age_text(n - timedelta(seconds=10), n) == "just now"
    assert fmt.age_text(n - timedelta(minutes=3), n) == "3 min ago"
    assert fmt.age_text(n - timedelta(hours=2), n) == "2 h ago"
    assert fmt.age_text(n - timedelta(days=4), n) == "4 d ago"
    assert fmt.age_text(None, n) == "age unknown"


def test_day_shows_year_only_when_it_differs():
    d = datetime(2026, 10, 15).date()
    assert fmt.day(d, year_of=d) == "15 Oct"
    assert fmt.day(datetime(2027, 1, 2).date(), year_of=d) == "2 Jan 2027"
    assert fmt.day(None) == fmt.DASH


# ---- match -----------------------------------------------------------------------------------------
def test_exact_symbol_and_name_win():
    ipos = fake_ipos()
    assert match.resolve("TONBO", ipos).match.ipo_id == "tonbo imaging"
    assert match.resolve("tonbo imaging india limited", ipos).match.ipo_id == "tonbo imaging"


def test_unique_substring_matches_and_ambiguity_is_reported():
    ipos = fake_ipos()
    assert match.resolve("zet", ipos).match.company.startswith("Zetwerk")
    two = [tonbo(), LiveIPO(ipo_id="tonbo labs", market="IN_MAIN", company="Tonbo Labs", symbol="TLABS")]
    assert match.resolve("tonbo", two).match.symbol == "TONBO"  # an exact symbol beats a name prefix
    res = match.resolve("tonb", two)
    assert res.match is None and res.ambiguous and len(res.choices) == 2


def test_no_match_offers_nearest_names():
    res = match.resolve("zetwrk", fake_ipos())
    assert res.match is None and not res.ambiguous
    assert any("Zetwerk" in i.company for i in res.choices)
    assert match.resolve("   ", fake_ipos()).match is None


# ---- analysis --------------------------------------------------------------------------------------
def test_timeline_phase_and_min_retail():
    a = build_analysis(tonbo(), None, now=now())
    assert a.timeline.phase == "issue day 1 of 5"
    assert a.issue.min_retail == 21 * 700.0
    assert a.issue.ofs_pct == 100.0
    assert a.issue.size_cr == 852.0


def test_subscription_percentiles_use_same_segment_only():
    a = build_analysis(tonbo(), None, now=now(), history=history_frame())
    total = a.sub("sub_total")
    # six main-board past totals (SME rows excluded): none is at or below 0.8x, so p0
    assert total.pctile == 0 and total.pctile_n == 6
    # NII 1.2x: only the past 1.1x is at or below it, so 1 of 6 is p17 (SME rows would have changed this)
    nii = a.sub("sub_nii")
    assert nii.pctile == 17 and nii.pctile_n == 6


def test_gmp_history_bucket_needs_enough_past_ipos():
    # 71% sits in the '50 to 75%' bucket: two history rows only, too few to describe
    a = build_analysis(tonbo(), None, now=now(), history=history_frame())
    assert a.gmp.history is None
    assert "too few" in (a.gmp.history_note or "")
    assert a.gmp.miss_n == 6 and a.gmp.miss_median is not None


def test_gmp_history_describes_a_populated_bucket():
    frame = history_frame()
    extra = frame.iloc[[3]].copy()
    for i in range(4):  # make the 50-75% bucket hold five IPOs
        row = extra.copy()
        row["gmp_pct_last"] = 60.0 + i
        row["list_open_gain_pct"] = 40.0 + i
        frame = pd.concat([frame, row], ignore_index=True)
    a = build_analysis(tonbo(), None, now=now(), history=frame)
    assert a.gmp.history is not None and a.gmp.history.n >= 5
    assert a.gmp.history.bucket == "50 to 75%"
    assert 0 <= a.gmp.history.share_positive <= 100


def test_no_skill_flag_from_reliability_or_caveat():
    ipo = tonbo()
    assert build_analysis(ipo, fake_predict(ipo), now=now()).model.no_skill is False
    assert build_analysis(ipo, no_skill_predict(ipo), now=now()).model.no_skill is True
    reliable_but_caveated = fake_predict(ipo)
    reliable_but_caveated.caveats = ["no demonstrated skill over the historical median at this stage"]
    assert build_analysis(ipo, reliable_but_caveated, now=now()).model.no_skill is True


def test_subscription_difference_is_not_turned_into_a_flag():
    ipo = tonbo()
    line = "subscription sources differ: QIB 0.4x (nse) vs 0.5x (chittorgarh) and 3 more categories"
    ipo.warnings = [line]
    a = build_analysis(ipo, None, now=now())
    assert all(line not in f.text for f in a.flags)   # shown once, under Data notes in `ipo show`
    assert a.warnings == [line]                        # kept as written by live.merge


def test_flags_cover_ofs_conflicts_stale_and_sme():
    a = build_analysis(hd_fire_sme_conflict(), fake_predict(hd_fire_sme_conflict()), now=now())
    labels = {f.label for f in a.flags}
    assert "GMP sources disagree" in labels
    assert "stale GMP" in labels  # ipowatch reading is 3 hours old
    assert "QIB 0.60x" in labels  # closing today, QIB under 1x
    assert "SME platform" in labels
    assert "retail-only demand" in labels  # retail 3.0x while QIB 0.6x and NII 0.8x
    assert a.summary.startswith("Estimate:")


def test_retail_only_and_loss_making_unknown_flags():
    ipo = tonbo()
    ipo.sub["sub_qib"] = Sourced(0.2, "nse", now())
    ipo.sub["sub_nii"] = Sourced(0.5, "nse", now())
    ipo.sub["sub_retail"] = Sourced(2.5, "nse", now())
    a = build_analysis(ipo, None, now=now())
    assert any(f.label == "retail-only demand" for f in a.flags)
    assert any(f.label == "loss-making unknown" for f in a.flags)


def test_huge_issue_and_unreliable_model_are_flagged():
    ipo = tonbo()
    ipo.issue_size_m = 60000.0  # 6,000 crore
    pred = fake_predict(ipo)
    pred.reliable = False
    a = build_analysis(ipo, pred, now=now())
    assert any(f.label == "very large issue" for f in a.flags)
    assert any(f.label == "model outside training data" for f in a.flags)


def test_missing_values_stay_none_through_the_analysis():
    bare = LiveIPO(ipo_id="bare", market="", company="Bare Co", status="upcoming")
    a = build_analysis(bare, None, now=now())
    assert a.issue.price_low is None and a.issue.min_retail is None and a.issue.size_cr is None
    assert all(s.now is None for s in a.subs)
    assert a.gmp.median is None and a.gmp.readings == []
    assert a.model is None
    assert "No grey-market reading" in a.summary
    as_json = to_jsonable(a)
    assert as_json["issue"]["min_retail"] is None and as_json["gmp"]["median"] is None


def test_summary_is_explicitly_an_estimate():
    a = build_analysis(tonbo(), fake_predict(tonbo()), now=now())
    assert a.summary.startswith("Estimate: the model puts the listing-open gain at +12%")
    assert "can be wrong" in a.summary


def test_predict_all_skips_listed_and_reports_failures():
    def flaky(ipo: LiveIPO):
        if ipo.ipo_id == "zetwerk":
            raise RuntimeError("bad row")
        return fake_predict(ipo)

    preds, errors = predict_all(flaky, fake_ipos())
    assert "old co" not in preds          # listed IPOs are not predicted
    assert "zetwerk" not in preds and errors == ["Zetwerk Manufacturing Businesses Limited: RuntimeError"]
    assert "tonbo imaging" in preds


def test_to_jsonable_maps_nan_dates_and_tuples():
    assert to_jsonable({"a": float("nan"), "b": (1, 2), "c": datetime(2026, 10, 15).date()}) == {
        "a": None, "b": [1, 2], "c": "2026-10-15"}
    assert math.isfinite(1.0)


def test_health_from_warnings_marks_down_ok_and_unknown():
    ipos = fake_ipos()
    ipos[0].warnings = ["nse unavailable: HTTP 503", "GMP unavailable: GMP module not available"]
    status = {s.name: s for s in health_from(ipos)}
    assert status["nse"].state == "down" and status["nse"].detail == "HTTP 503"
    assert status["chittorgarh"].state == "ok"
    assert status["gmp"].state == "down"
    assert {s.state for s in health_from([])} == {"unknown"}
