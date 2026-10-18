"""Merge rules with synthetic LiveIPOs: disagreement, missing sources, GMP, status, cache and JSON round trip."""
import sys
import types
from datetime import date, datetime, timedelta, timezone

import pytest

import ipo_radar.live as live_pkg
from ipo_radar import config
from ipo_radar.errors import SourceError
from ipo_radar.live import merge
from ipo_radar.live.gmp import GmpReading
from ipo_radar.live.keys import company_key
from ipo_radar.schema import LiveIPO, Sourced

pytestmark = pytest.mark.usefixtures("live_isolation")
NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
TODAY = date(2026, 10, 15)
D15 = date(2026, 10, 15)


def ipo(name, source="nse", **kw) -> LiveIPO:
    base = dict(ipo_id=company_key(name), market="IN_MAIN", company=name, sources=[source])
    base.update(kw)
    return LiveIPO(**base)


def sub(value, source="nse", at=NOW, note=""):
    return Sourced(value, source, at, note)


# ---- status ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("today, expected", [
    (date(2026, 10, 12), "upcoming"),
    (date(2026, 10, 13), "open"),          # open day
    (date(2026, 10, 15), "open"),          # close day is still open
    (date(2026, 10, 16), "closed"),
    (date(2026, 10, 19), "allotment"),     # allotment date reached, listing not yet
    (date(2026, 10, 21), "listing_today"),
    (date(2026, 10, 22), "listed"),
])
def test_status_is_derived_from_dates_with_patched_today(monkeypatch, today, expected):
    monkeypatch.setattr(merge, "_today", lambda: today)
    issue = ipo("Demo Ltd", open_date=date(2026, 10, 13), close_date=date(2026, 10, 15),
                allotment_date=date(2026, 10, 19), listing_date=date(2026, 10, 21))
    assert merge.derive_status(issue, merge._today()) == expected


def test_status_without_allotment_date_stays_closed_until_listing():
    issue = ipo("Demo Ltd", open_date=date(2026, 10, 13), close_date=date(2026, 10, 15),
                listing_date=date(2026, 10, 21))
    assert merge.derive_status(issue, date(2026, 10, 19)) == "closed"


def test_status_unknown_without_open_date():
    assert merge.derive_status(ipo("Demo Ltd"), TODAY) == "unknown"


# ---- matching -------------------------------------------------------------------------------------
def test_same_company_across_sources_is_one_record():
    nse_rows = [ipo("Zetwerk Manufacturing Businesses Limited", symbol="ZETWERK",
                    open_date=date(2026, 10, 15), sub={"sub_total": sub(0.01)})]
    cg_rows = [ipo("Zetwerk Manufacturing Business", source="chittorgarh", open_date=date(2026, 10, 15),
                   sub={"sub_total": sub(0.01, "chittorgarh")})]
    out = merge.merge_sources(nse_rows, cg_rows, None, failures={}, today=TODAY, now=NOW)
    assert len(out) == 1 and out[0].sources == ["nse", "chittorgarh"]


def test_different_companies_with_a_shared_prefix_are_not_merged():
    out = merge.merge_sources([ipo("Tata Motors Limited")],
                              [ipo("Tata Motors Finance Limited", source="chittorgarh")],
                              None, failures={}, today=TODAY, now=NOW)
    assert len(out) == 2


def test_symbol_match_when_names_differ_entirely():
    nse_rows = [ipo("Alpha Widgets Limited", symbol="ALPW")]
    cg_rows = [ipo("Completely Different Name", source="chittorgarh", symbol="ALPW")]
    out = merge.merge_sources(nse_rows, cg_rows, None, failures={}, today=TODAY, now=NOW)
    assert len(out) == 1


# ---- subscription and fields ----------------------------------------------------------------------
def test_exchange_value_wins_and_chittorgarh_fills_gaps():
    nse_rows = [ipo("Demo Ltd", sub={"sub_qib": sub(1.0), "sub_total": sub(10.0)})]
    cg_rows = [ipo("Demo Ltd", source="chittorgarh",
                   sub={"sub_qib": sub(1.02, "chittorgarh"), "sub_nii": sub(3.3, "chittorgarh"),
                        "sub_total": sub(10.2, "chittorgarh")})]
    rec = merge.merge_sources(nse_rows, cg_rows, None, failures={}, today=TODAY, now=NOW)[0]
    assert rec.sub["sub_qib"].source == "nse" and rec.sub["sub_qib"].value == 1.0
    assert rec.sub["sub_nii"].source == "chittorgarh" and rec.sub["sub_nii"].value == 3.3
    assert rec.warnings == []  # 1.0 vs 1.02 and 10 vs 10.2 are within 5%


def test_disagreement_over_five_percent_is_a_warning_and_keeps_exchange_value():
    nse_rows = [ipo("Demo Ltd", sub={"sub_total": sub(10.0)})]
    cg_rows = [ipo("Demo Ltd", source="chittorgarh", sub={"sub_total": sub(12.0, "chittorgarh", note="snapshot X")})]
    rec = merge.merge_sources(nse_rows, cg_rows, None, failures={}, today=TODAY, now=NOW)[0]
    assert rec.sub["sub_total"].value == 10.0
    assert rec.warnings == ["subscription sources differ: total 10x (nse) vs 12x (chittorgarh)"]


def test_zero_and_zero_do_not_warn():
    nse_rows = [ipo("Demo Ltd", sub={"sub_qib": sub(0.0)})]
    cg_rows = [ipo("Demo Ltd", source="chittorgarh", sub={"sub_qib": sub(0.0, "chittorgarh")})]
    assert merge.merge_sources(nse_rows, cg_rows, None, failures={}, today=TODAY, now=NOW)[0].warnings == []


def test_date_and_price_disagreements_are_warnings_not_overwrites():
    nse_rows = [ipo("Demo Ltd", open_date=date(2026, 10, 13), price_high=271.0)]
    cg_rows = [ipo("Demo Ltd", source="chittorgarh", open_date=date(2026, 10, 14), price_high=270.0)]
    rec = merge.merge_sources(nse_rows, cg_rows, None, failures={}, today=TODAY, now=NOW)[0]
    assert rec.open_date == date(2026, 10, 13) and rec.price_high == 271.0
    assert any("open_date disagree" in w for w in rec.warnings)
    assert any("price_high disagree" in w for w in rec.warnings)


def test_chittorgarh_only_issue_keeps_its_own_subscription_without_warning():
    cg_rows = [ipo("Solo Ltd", source="chittorgarh", sub={"sub_nii": sub(2.0, "chittorgarh")})]
    rec = merge.merge_sources([], cg_rows, None, failures={}, today=TODAY, now=NOW)[0]
    assert rec.sub["sub_nii"].source == "chittorgarh" and rec.warnings == []


# ---- source failures ------------------------------------------------------------------------------
def test_failed_nse_warns_on_every_ipo_and_its_fields_stay_missing():
    cg_rows = [ipo("A One Ltd", source="chittorgarh"), ipo("B Two Ltd", source="chittorgarh")]
    out = merge.merge_sources([], cg_rows, None, failures={"nse": "HTTP 403 for x"}, today=TODAY, now=NOW)
    assert len(out) == 2
    for rec in out:
        assert "nse unavailable: HTTP 403 for x" in rec.warnings
        assert not any(k.startswith("sub_") and rec.sub[k].source == "nse" for k in rec.sub)


def test_failed_chittorgarh_warns_on_every_ipo():
    nse_rows = [ipo("A One Ltd", sub={"sub_total": sub(3.0)})]
    out = merge.merge_sources(nse_rows, [], None, failures={"chittorgarh": "timeout"}, today=TODAY, now=NOW)
    assert "chittorgarh unavailable: timeout" in out[0].warnings
    assert out[0].sub["sub_total"].value == 3.0


# ---- GMP ------------------------------------------------------------------------------------------
def _gmp_case(readings, price_high=100.0, failures=None):
    nse_rows = [ipo("Demo Ltd", price_high=price_high)]
    gmp_map = None if readings is None else {company_key("Demo Ltd"): readings}
    return merge.merge_sources(nse_rows, [], gmp_map, failures=failures or {}, today=TODAY, now=NOW)[0]


def test_gmp_absent_is_silent():
    rec = _gmp_case(None)
    assert rec.gmp == {} and rec.warnings == []


def test_gmp_median_per_source_and_percent_of_price_high():
    rec = _gmp_case([sub(40, "a", NOW), sub(60, "b", NOW), sub(50, "c", NOW)])
    assert rec.gmp["gmp"].value == 50.0
    assert rec.gmp["gmp_pct"].value == 50.0  # 50 / 100 * 100
    assert {"a", "b", "c"} <= set(rec.gmp)
    assert rec.gmp_history and rec.gmp_history[0][2] == "a"


def test_gmp_disagreement_warns_only_beyond_max_ten_rupees_or_quarter():
    wide = _gmp_case([sub(40, "a", NOW), sub(60, "b", NOW)])
    assert wide.warnings == ["GMP sources disagree: 40 (a) vs 60 (b)"]
    narrow = _gmp_case([sub(40, "a", NOW), sub(45, "b", NOW)])
    assert narrow.warnings == []  # 5 rupees is within max(10, 25%)


def test_stale_and_empty_gmp_readings_do_not_make_a_median():
    stale = NOW - timedelta(hours=7)
    rec = _gmp_case([sub(40, "a", stale), sub(None, "b", NOW)])
    assert "gmp" not in rec.gmp and "gmp_pct" not in rec.gmp
    assert rec.gmp["a"].value == 40  # the per-source reading is still kept for display


def test_gmp_failure_warns_on_every_ipo():
    rec = _gmp_case(None, failures={"gmp": "timeout"})
    assert rec.warnings == ["GMP unavailable: timeout"] and rec.gmp == {}


def test_gmp_for_an_unknown_company_is_dropped_without_error():
    nse_rows = [ipo("Demo Ltd")]
    out = merge.merge_sources(nse_rows, [], {"someone else": [sub(10, "a", NOW)]}, failures={}, today=TODAY, now=NOW)
    assert out[0].gmp == {}


# ---- fetch_all: caching, failures, module handling --------------------------------------------------
def _patch_sources(monkeypatch, nse_fn, cg_fn, gmp_fn=None):
    calls = {"nse": 0, "cg": 0}

    def counted(name, fn):
        def wrapper(*a, **k):
            calls[name] += 1
            return fn()
        return wrapper

    monkeypatch.setattr(merge.nse, "fetch", counted("nse", nse_fn))
    monkeypatch.setattr(merge.chittorgarh, "fetch", counted("cg", cg_fn))
    monkeypatch.setattr(merge, "_run_gmp", gmp_fn or (lambda: (None, "GMP module not available (test)")))
    monkeypatch.setattr(merge, "_today", lambda: TODAY)
    return calls


def test_fetch_all_caches_and_serves_from_cache_within_ttl(monkeypatch):
    calls = _patch_sources(monkeypatch, lambda: [ipo("Demo Ltd", symbol="DEMO")], lambda: [])
    first = merge.fetch_all(use_cache=True, ttl_seconds=120)
    second = merge.fetch_all(use_cache=True, ttl_seconds=120)
    assert calls == {"nse": 1, "cg": 1} and first == second
    assert (config.LIVE_CACHE / "merged.json").exists()


def test_use_cache_false_forces_refresh(monkeypatch):
    calls = _patch_sources(monkeypatch, lambda: [ipo("Demo Ltd")], lambda: [])
    merge.fetch_all(use_cache=True)
    merge.fetch_all(use_cache=False)
    assert calls == {"nse": 2, "cg": 2}


def test_expired_cache_is_refetched(monkeypatch):
    calls = _patch_sources(monkeypatch, lambda: [ipo("Demo Ltd")], lambda: [])
    merge.fetch_all(use_cache=True)
    merge.fetch_all(use_cache=True, ttl_seconds=0)
    assert calls["nse"] == 2


def test_corrupt_cache_is_ignored(monkeypatch):
    calls = _patch_sources(monkeypatch, lambda: [ipo("Demo Ltd")], lambda: [])
    config.LIVE_CACHE.mkdir(parents=True)
    (config.LIVE_CACHE / "merged.json").write_text("{not json")
    assert merge.fetch_all(use_cache=True)
    assert calls["nse"] == 1


def test_both_exchange_and_chittorgarh_failing_raises(monkeypatch):
    def fail():
        raise SourceError("boom")
    _patch_sources(monkeypatch, fail, fail)
    with pytest.raises(SourceError, match="no IPO data"):
        merge.fetch_all(use_cache=False)


def test_one_source_failing_still_returns_and_is_not_cached(monkeypatch):
    def fail():
        raise SourceError("HTTP 500")
    _patch_sources(monkeypatch, fail, lambda: [ipo("Demo Ltd", source="chittorgarh")])
    rows = merge.fetch_all(use_cache=True)
    assert rows and "nse unavailable: HTTP 500" in rows[0].warnings
    assert not (config.LIVE_CACHE / "merged.json").exists()


def test_missing_gmp_module_is_a_warning_not_a_crash(monkeypatch):
    # `from ipo_radar.live import gmp` reads the package attribute first, so drop it as well as the sys.modules entry
    monkeypatch.delattr(live_pkg, "gmp", raising=False)
    monkeypatch.setitem(sys.modules, "ipo_radar.live.gmp", None)  # import now raises ImportError
    _patch_sources(monkeypatch, lambda: [ipo("Demo Ltd")], lambda: [], gmp_fn=merge._run_gmp)
    rows = merge.fetch_all(use_cache=False)
    assert rows[0].warnings[0].startswith("GMP unavailable: GMP module not available")


def test_gmp_crash_is_contained(monkeypatch):
    fake = types.ModuleType("ipo_radar.live.gmp")

    def boom():
        raise RuntimeError("bug in gmp")
    fake.fetch = boom
    monkeypatch.setattr(live_pkg, "gmp", fake, raising=False)
    monkeypatch.setitem(sys.modules, "ipo_radar.live.gmp", fake)
    _patch_sources(monkeypatch, lambda: [ipo("Demo Ltd")], lambda: [], gmp_fn=merge._run_gmp)
    rows = merge.fetch_all(use_cache=False)
    assert rows and rows[0].warnings[0].startswith("GMP unavailable: GMP module error (RuntimeError: bug in gmp)")


# ---- JSON cache round trip ------------------------------------------------------------------------
def test_json_round_trip_is_lossless_for_a_fully_populated_record():
    issue = LiveIPO(
        ipo_id="demo", market="IN_SME", company="Demo Ltd", symbol="DEMO", status="open",
        open_date=date(2026, 10, 13), close_date=date(2026, 10, 15), allotment_date=None,
        listing_date=date(2026, 10, 21), price_low=100.0, price_high=110.0, lot_size=1200,
        issue_size_m=42.5, fresh_issue_pct=100.0,
        sub={"sub_total": sub(2.5, "nse", NOW, "updateTime x"), "sub_qib": sub(0.0, "chittorgarh", None)},
        sub_by_day={1: {"sub_total": 0.4}, 3: {"sub_total": 2.5, "sub_qib": 1.1}},
        gmp={"a": sub(44.0, "a", NOW), "gmp": sub(44.0, "gmp", NOW, "median")},
        gmp_history=[(NOW, 44.0, "a")], listing={"open": 120.0}, warnings=["w1"], sources=["nse", "chittorgarh"],
    )
    text = merge.to_json([issue])
    back = merge.from_json(text)
    assert back == [issue]
    assert isinstance(back[0].sub_by_day[1], dict) and 1 in back[0].sub_by_day  # int keys restored
    assert isinstance(back[0].gmp_history[0], tuple)


def test_market_and_day_wise_snapshot_come_from_chittorgarh_when_nse_lacks_them():
    nse_rows = [ipo("Demo Ltd", market="")]
    cg_rows = [ipo("Demo Ltd", source="chittorgarh", market="IN_SME", sub_by_day={1: {"sub_total": 0.5}})]
    rec = merge.merge_sources(nse_rows, cg_rows, None, failures={}, today=TODAY, now=NOW)[0]
    assert rec.market == "IN_SME" and rec.sub_by_day == {1: {"sub_total": 0.5}}
    assert "market unknown" not in " ".join(rec.warnings)


def test_missing_subscription_values_are_skipped_not_compared():
    nse_rows = [ipo("Demo Ltd", sub={"sub_nii": sub(None)})]
    cg_rows = [ipo("Demo Ltd", source="chittorgarh", sub={"sub_nii": sub(3.0, "chittorgarh")})]
    rec = merge.merge_sources(nse_rows, cg_rows, None, failures={}, today=TODAY, now=NOW)[0]
    assert rec.sub["sub_nii"].value is None and rec.warnings == []


def test_same_company_twice_in_the_exchange_feed_is_one_record():
    rows = [ipo("Demo Ltd", symbol="DEMO", sub={"sub_total": sub(2.0)}),
            ipo("Demo Ltd", symbol="DEMO", sub={"sub_total": sub(2.0)})]
    assert len(merge.merge_sources(rows, [], None, failures={}, today=TODAY, now=NOW)) == 1


def test_gmp_module_raising_source_error_is_a_warning(monkeypatch):
    fake = types.ModuleType("ipo_radar.live.gmp")

    def timeout():
        raise SourceError("timeout")
    fake.fetch = timeout
    monkeypatch.setattr(live_pkg, "gmp", fake, raising=False)
    monkeypatch.setitem(sys.modules, "ipo_radar.live.gmp", fake)
    assert merge._run_gmp() == (None, "timeout")


# ---- subscription warnings: one per IPO, named categories, sampling rule ------------------------------
def _rec_pair(nse_subs, cg_subs, *, status_open=False, nse_at=None, cg_at=None):
    """An NSE record with NSE subs and a Chittorgarh row with its own subs, sampled at the given times."""
    nse_rows = [ipo("Demo Ltd", open_date=date(2026, 10, 13), close_date=date(2026, 10, 19),
                    sub={k: sub(v, "nse", nse_at or NOW) for k, v in nse_subs.items()})]
    cg_rows = [ipo("Demo Ltd", source="chittorgarh", open_date=date(2026, 10, 13), close_date=date(2026, 10, 19),
                   sub={k: sub(v, "chittorgarh", cg_at or NOW) for k, v in cg_subs.items()})]
    today = date(2026, 10, 15) if status_open else date(2026, 10, 25)
    return merge.merge_sources(nse_rows, cg_rows, None, failures={}, today=today, now=NOW)[0]


def test_one_warning_per_ipo_names_at_most_three_categories_and_counts_the_rest():
    nse_subs = {"sub_qib": 1.0, "sub_nii": 2.0, "sub_bnii": 3.0, "sub_snii": 4.0, "sub_retail": 1.2, "sub_total": 9.0}
    cg_subs = {"sub_qib": 2.0, "sub_nii": 4.0, "sub_bnii": 6.0, "sub_snii": 8.0, "sub_retail": 1.9, "sub_total": 18.0}
    rec = _rec_pair(nse_subs, cg_subs)
    subs = [w for w in rec.warnings if w.startswith("subscription")]
    assert len(subs) == 1
    # named in schema order (SUB_COLS): QIB, NII, sNII; the other three are counted
    assert subs[0] == ("subscription sources differ: QIB 1x (nse) vs 2x (chittorgarh), "
                       "NII 2x (nse) vs 4x (chittorgarh), sNII 4x (nse) vs 8x (chittorgarh) and 3 more categories")


def test_small_absolute_differences_are_not_reported_even_when_relatively_large():
    rec = _rec_pair({"sub_qib": 0.01}, {"sub_qib": 0.04})  # 300% relative, but only 0.03x
    assert not any(w.startswith("subscription") for w in rec.warnings)


def test_mismatch_sampled_apart_while_the_issue_is_open_is_suppressed():
    rec = _rec_pair({"sub_retail": 1.2}, {"sub_retail": 1.9},
                    status_open=True, nse_at=NOW, cg_at=NOW + timedelta(minutes=30))
    assert not any(w.startswith("subscription") for w in rec.warnings)


def test_same_moment_mismatch_while_open_is_kept():
    rec = _rec_pair({"sub_retail": 1.2}, {"sub_retail": 1.9},
                    status_open=True, nse_at=NOW, cg_at=NOW + timedelta(minutes=4))
    assert any(w == "subscription sources differ: retail 1.2x (nse) vs 1.9x (chittorgarh)" for w in rec.warnings)


def test_mismatch_after_close_is_kept_even_if_sampled_apart():
    rec = _rec_pair({"sub_retail": 1.2}, {"sub_retail": 1.9},
                    status_open=False, nse_at=NOW, cg_at=NOW + timedelta(hours=3))
    assert any(w.startswith("subscription sources differ: retail") for w in rec.warnings)


# ---- GMP after listing, issue price, GMP matched by alias ---------------------------------------------
def test_listed_ipo_gets_no_gmp_block_but_keeps_its_listing_date():
    listed = ipo("Demo Ltd", open_date=date(2026, 10, 1), close_date=date(2026, 10, 5), listing_date=date(2026, 10, 8))
    gmp_map = {company_key("Demo Ltd"): [sub(40.0, "a", NOW)]}
    rec = merge.merge_sources([listed], [], gmp_map, failures={}, today=date(2026, 10, 15), now=NOW)[0]
    assert rec.status == "listed" and rec.gmp == {} and rec.gmp_history == []
    assert rec.listing_date == date(2026, 10, 8)


def test_closed_but_not_listed_ipo_keeps_gmp():
    closed = ipo("Demo Ltd", open_date=date(2026, 10, 12), close_date=date(2026, 10, 14))
    gmp_map = {company_key("Demo Ltd"): [sub(40.0, "a", NOW)]}
    rec = merge.merge_sources([closed], [], gmp_map, failures={}, today=date(2026, 10, 15), now=NOW)[0]
    assert rec.status == "closed" and rec.gmp["gmp"].value == 40.0


def test_gmp_gets_no_warning_for_listed_ipos_when_gmp_fails():
    listed = ipo("Demo Ltd", open_date=date(2026, 10, 1), close_date=date(2026, 10, 5), listing_date=date(2026, 10, 8))
    rec = merge.merge_sources([listed], [], None, failures={"gmp": "timeout"}, today=date(2026, 10, 15), now=NOW)[0]
    assert not any(w.startswith("GMP unavailable") for w in rec.warnings)


def test_issue_price_comes_from_chittorgarh_when_nse_has_none_and_warns_on_disagreement():
    nse_rows = [ipo("Demo Ltd")]
    cg_rows = [ipo("Demo Ltd", source="chittorgarh", issue_price=271.0, price_high=271.0)]
    rec = merge.merge_sources(nse_rows, cg_rows, None, failures={}, today=TODAY, now=NOW)[0]
    assert rec.issue_price == 271.0 and rec.warnings == []
    nse_rows = [ipo("Demo Ltd", issue_price=270.0)]
    rec = merge.merge_sources(nse_rows, cg_rows, None, failures={}, today=TODAY, now=NOW)[0]
    assert rec.issue_price == 270.0 and any("issue_price disagree" in w for w in rec.warnings)


def test_gmp_site_name_matched_by_alias_with_its_open_day_and_market(monkeypatch):
    from ipo_radar.live.gmp import GmpReading
    rec = ipo("Zetwerk Manufacturing Businesses Limited", market="IN_MAIN", open_date=D15, close_date=D15 + timedelta(days=4))
    reading = GmpReading(45.0, "investorgain", NOW, "x", open_date=D15, market="IN_MAIN")
    out = merge.merge_sources([rec], [], {"zetwerk": [reading]}, failures={}, today=date(2026, 10, 16), now=NOW)
    assert out[0].gmp["gmp"].value == 45.0
    wrong_day = GmpReading(45.0, "investorgain", NOW, "x", open_date=date(2026, 10, 20), market="IN_MAIN")
    fresh = ipo("Zetwerk Manufacturing Businesses Limited", market="IN_MAIN", open_date=D15, close_date=D15 + timedelta(days=4))
    out = merge.merge_sources([fresh], [], {"zetwerk": [wrong_day]}, failures={}, today=date(2026, 10, 16), now=NOW)
    assert out[0].gmp == {}  # a different day is not the same IPO


def test_two_gmp_names_for_one_ipo_are_pooled_into_one_median():
    rec = ipo("Sahajanand Medical Technologies Limited", open_date=D15, close_date=D15 + timedelta(days=4),
              price_high=344.0)
    gmp_map = {
        "sahajanand medical technologies": [GmpReading(44.0, "investorgain", NOW, "", open_date=D15, market="IN_MAIN"),
                                            GmpReading(47.0, "ipopremium", NOW, "", open_date=D15, market="IN_MAIN")],
        "sahajanand medical": [GmpReading(42.0, "ipowatch", NOW, "", open_date=D15),
                               GmpReading(37.0, "ipocentral", NOW, "", open_date=D15, market="IN_MAIN")],
    }
    out = merge.merge_sources([rec], [], gmp_map, failures={}, today=D15, now=NOW)[0]
    assert out.gmp["gmp"].value == pytest.approx(43.0)  # median of 44, 47, 42, 37
    assert {k for k in out.gmp if k not in ("gmp", "gmp_pct")} == {"investorgain", "ipopremium", "ipowatch", "ipocentral"}


def test_a_site_listing_one_ipo_twice_counts_once():
    rec = ipo("Demo Widgets Limited", open_date=D15, close_date=D15 + timedelta(days=4))
    gmp_map = {"demo widgets": [GmpReading(10.0, "ipowatch", NOW, "", open_date=D15)],
               "demo widgets industries": [GmpReading(90.0, "ipowatch", NOW, "", open_date=D15)]}
    out = merge.merge_sources([rec], [], gmp_map, failures={}, today=D15, now=NOW)[0]
    assert out.gmp["gmp"].value == 10.0  # the first reading from that site wins; the second is dropped
