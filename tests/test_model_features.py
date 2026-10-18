"""Feature tests: leakage across stages, transform inversion, group coverage, live-row parity, category levels."""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from ipo_radar.model import features as F
from ipo_radar.schema import LiveIPO, Sourced
from model_synth import make_ipos


@pytest.fixture(scope="module")
def ipos() -> pd.DataFrame:
    return make_ipos(n_per_year=60, years=range(2018, 2024), seed=5)


@pytest.fixture(scope="module")
def vocab(ipos) -> F.Vocab:
    return F.fit_vocab(ipos)


def _stage_rows(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    expanded = F.asof_expand(df)
    return expanded[expanded["as_of"] == stage]


# ---- leakage ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("stage, later_cols", [
    ("pre_open", ["sub_qib", "sub_total", "gmp_pct_last", "idx_ret_issue_window", "sub_qib_d1"]),
    ("day1", ["sub_qib", "sub_total", "gmp_pct_last", "idx_ret_issue_window", "sub_qib_d2"]),
    ("close", ["gmp_pct_last", "gmp_last"]),
])
def test_changing_a_later_stage_column_does_not_change_earlier_features(ipos, vocab, stage, later_cols):
    changed = ipos.copy()
    for col in later_cols:
        changed[col] = changed[col].fillna(0) + 999.0
    before = F.design(_stage_rows(ipos, stage), stage, vocab)
    after = F.design(_stage_rows(changed, stage), stage, vocab)
    pd.testing.assert_frame_equal(before, after)


def test_future_target_never_enters_the_features(ipos, vocab):
    changed = ipos.copy()
    changed["list_open_gain_pct"] = changed["list_open_gain_pct"] * -7 + 1000
    changed["list_close_gain_pct"] = changed["list_close_gain_pct"] + 500
    for stage in F.STAGES:
        before = F.design(_stage_rows(ipos, stage), stage, vocab)
        after = F.design(_stage_rows(changed, stage), stage, vocab)
        pd.testing.assert_frame_equal(before, after)


def test_stage_columns_are_blank_in_every_earlier_stage(ipos):
    expanded = F.asof_expand(ipos)
    pre = expanded[expanded["as_of"] == "pre_open"]
    assert pre["gmp_pct_last"].isna().all()
    assert pre["sub_qib"].isna().all()
    close = expanded[expanded["as_of"] == "close"]
    assert close["gmp_pct_last"].isna().all()
    assert close["sub_total"].notna().any()
    assert set(expanded["as_of"].astype(str)) == set(F.STAGES)
    assert len(expanded) == len(ipos) * len(F.STAGES)


def test_no_calendar_year_or_time_trend_feature():
    names = F.stage_features("pre_listing")
    assert not any(n in names for n in ("year", "listing_year", "open_year"))
    assert "listing_month" in names and "days_close_to_listing" in names


# ---- structure --------------------------------------------------------------------------------------
def test_stage_features_are_cumulative_and_gmp_only_at_pre_listing():
    assert {"market", "exchange", "sector"} <= set(F.stage_features("pre_open"))
    assert set(F.stage_features("pre_open")) < set(F.stage_features("day1")) < set(F.stage_features("day2"))
    assert set(F.stage_features("day2")) < set(F.stage_features("close")) < set(F.stage_features("pre_listing"))
    assert "gmp_pct_last" not in F.stage_features("close")
    assert "gmp_pct_last" in F.stage_features("pre_listing")
    assert "log_sub_qib_d2" not in F.stage_features("day1")


def test_every_feature_has_an_importance_group():
    for stage in F.STAGES:
        for name in F.stage_features(stage):
            assert F.group_of(name) in F.GROUPS


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match="unknown stage"):
        F.stage_features("after_listing")


# ---- transform ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("y", [-100.0, -72.5, -0.5, 0.0, 3.0, 47.0, 333.0])
def test_transform_inverts_exactly(y):
    back = F.from_model_scale(F.to_model_scale(np.array([y])))
    assert back[0] == pytest.approx(y, abs=1e-9)


def test_transform_is_monotone_and_odd():
    grid = np.linspace(-95, 400, 500)
    z = F.to_model_scale(grid)
    assert np.all(np.diff(z) > 0)
    assert np.allclose(F.to_model_scale(-grid), -z)


# ---- categories and missingness ---------------------------------------------------------------------
def test_sector_levels_keep_top_n_and_map_the_rest_to_other(ipos):
    vocab = F.fit_vocab(ipos)
    assert len(vocab.sectors) == F.SECTOR_TOP_N
    rows = ipos.head(3).copy()
    rows.iloc[0, rows.columns.get_loc("sector")] = "Never Seen Before"
    rows.iloc[1, rows.columns.get_loc("sector")] = None
    rows.iloc[2, rows.columns.get_loc("sector")] = vocab.sectors[0]
    X = F.design(rows, "pre_open", vocab)
    assert X["sector"].iloc[0] == F.OTHER
    assert pd.isna(X["sector"].iloc[1])
    assert X["sector"].iloc[2] == vocab.sectors[0]


def test_missing_gmp_is_flagged_zero_but_the_percent_stays_nan(ipos, vocab):
    rows = _stage_rows(ipos, "pre_listing")
    X = F.design(rows, "pre_listing", vocab)
    no_gmp = rows["gmp_pct_last"].isna()
    assert (X.loc[no_gmp, "has_gmp"] == 0).all()
    assert X.loc[no_gmp, "gmp_pct_last"].isna().all()
    assert (X.loc[~no_gmp, "has_gmp"] == 1).all()


def test_ratios_do_not_divide_by_zero(ipos, vocab):
    rows = _stage_rows(ipos, "close").head(5).copy()
    rows["sub_qib"] = 0.0
    X = F.design(rows, "close", vocab)
    assert X["ratio_retail_qib"].isna().all()


# ---- live rows --------------------------------------------------------------------------------------
def _live(**kw) -> LiveIPO:
    base = dict(ipo_id="L1", market="IN_MAIN", company="Live Ltd", status="open",
                open_date=date(2026, 10, 20), close_date=date(2026, 10, 22), listing_date=date(2026, 10, 27),
                price_low=95.0, price_high=100.0, lot_size=150, issue_size_m=2000.0, fresh_issue_pct=80.0)
    base.update(kw)
    return LiveIPO(**base)


@pytest.mark.parametrize("stage", F.STAGES)
def test_live_row_has_exactly_the_training_feature_columns(ipos, vocab, stage):
    trained = F.design(_stage_rows(ipos, stage).head(10), stage, vocab)
    live = F.live_to_row(_live(), stage, vocab, context={"idx_ret_20d": 1.0})
    assert list(live.columns) == list(trained.columns) == F.stage_features(stage)
    for col in F.CATEGORICAL:
        assert list(live[col].cat.categories) == list(trained[col].cat.categories)


def test_live_gmp_percent_falls_back_to_inr_over_price_and_is_not_invented():
    with_pct = _live(gmp={"gmp_pct": Sourced(12.5, "a"), "gmp": Sourced(40.0, "a")})
    assert F.live_gmp_pct(with_pct) == 12.5
    inr_only = _live(gmp={"gmp": Sourced(10.0, "a")})
    assert F.live_gmp_pct(inr_only) == pytest.approx(10.0)   # 10 INR over the top of the 100 band
    assert F.live_gmp_pct(_live()) is None


def test_live_sub_by_day_and_totals_map_to_the_right_columns():
    live = _live(sub={"sub_total": Sourced(12.0, "nse", datetime(2026, 10, 21)),
                      "sub_qib": Sourced(None, "nse")},
                 sub_by_day={1: {"sub_total": 3.0, "sub_qib": 0.5}, 2: {"sub_total": 7.0}})
    row = F.live_raw_row(live)
    assert row["sub_total_d1"] == 3.0 and row["sub_qib_d1"] == 0.5
    assert row["sub_total_d2"] == 7.0 and np.isnan(row["sub_qib_d2"])
    assert row["sub_total"] == 12.0
    assert np.isnan(row["sub_qib"])            # a None value stays missing, not 0


def test_live_fields_the_feed_does_not_carry_are_missing_not_guessed():
    row = F.live_raw_row(_live())
    for col in ("pe_issue", "roe", "debt_equity", "mcap_issue_usd_m", "anchor_pct", "n_underwriters"):
        assert np.isnan(row[col])
    assert row["exchange"] is None and row["sector"] is None
    assert row["ofs_pct"] == pytest.approx(20.0)


def test_model_package_never_imports_live_or_alert_state():
    """Training and serving read only the processed tables. The live cache, the sqlite GMP history and the scraped
    feeds (investorgain asks not to be used for AI training) must stay out of the model package."""
    from pathlib import Path
    root = Path(F.__file__).parent
    banned = ("ipo_radar.live", "ipo_radar.alerts", "sqlite3", "LIVE_CACHE", "STATE_DB", "gmp_history")
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{path.name} mentions {token}"


def test_pre_open_price_feature_uses_the_band_only(ipos, vocab):
    """The final issue price is set at close: it must never reach a pre_open feature, and a missing band is NaN."""
    rows = _stage_rows(ipos, "pre_open").head(20).copy()
    changed = rows.copy()
    changed["issue_price"] = changed["issue_price"] * 7 + 3
    a = F.design(rows, "pre_open", vocab)
    b = F.design(changed, "pre_open", vocab)
    pd.testing.assert_series_equal(a["log_price_ref"], b["log_price_ref"])
    no_band = rows.copy()
    no_band["price_low"] = np.nan
    no_band["price_high"] = np.nan
    assert F.design(no_band, "pre_open", vocab)["log_price_ref"].isna().all()
