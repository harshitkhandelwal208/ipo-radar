"""Tests for data.build on synthetic frames: schema order and dtypes, duplicates, blanking, missing inputs, context."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ipo_radar.data import build as B
from ipo_radar.schema import ALL_COLS, CONTEXT_COLS, DATE_COLS, ID_COLS, MARKETS
from model_synth import make_indices, make_ipos


def _frames(n: int = 40) -> dict[str, pd.DataFrame]:
    df = make_ipos(n_per_year=n, years=range(2019, 2022), seed=3)
    return {"india": df[df["market"].isin(["IN_MAIN", "IN_SME"])].reset_index(drop=True),
            "global": df[df["market"] == "US"].reset_index(drop=True)}


# ---- schema -----------------------------------------------------------------------------------------
def test_output_has_exact_schema_order_and_dtypes():
    df, report = B.build_frame(_frames())
    assert list(df.columns) == ALL_COLS
    for col in ID_COLS:
        assert str(df[col].dtype) == "string", col
    for col in DATE_COLS:
        assert df[col].dtype == np.dtype("datetime64[ns]"), col
    assert df["n_missing_core"].dtype == np.dtype("int64")
    assert df["issue_price"].dtype == np.dtype("float64")
    assert report.input_rows == {"india": len(_frames()["india"]), "global": len(_frames()["global"])}


def test_missing_schema_columns_are_added_as_nan_and_noted():
    frames = _frames()
    frames["global"] = frames["global"].drop(columns=["gmp_last", "pe_issue"])
    df, report = B.build_frame(frames)
    assert df["gmp_last"].isna().all() or df.loc[df["market"] == "US", "gmp_last"].isna().all()
    assert any("pe_issue" in n for n in report.notes)


def test_unknown_market_raises_clearly():
    frames = _frames()
    frames["india"].loc[0, "market"] = "MARS"
    with pytest.raises(ValueError, match="unknown market"):
        B.build_frame(frames)


def test_rows_without_market_or_id_are_dropped_and_counted():
    frames = _frames()
    frames["india"].loc[0, "market"] = None
    frames["india"].loc[1, "ipo_id"] = None
    df, report = B.build_frame(frames)
    assert report.dropped["india: no market"] == 1
    assert report.dropped["india: no ipo_id"] == 1
    assert len(df) == len(frames["india"]) + len(frames["global"]) - 2


# ---- duplicates --------------------------------------------------------------------------------------
def test_duplicate_ipo_id_keeps_the_row_with_more_filled_fields():
    frames = _frames()
    sparse = frames["india"].iloc[[0]].copy()
    sparse["ipo_id"] = frames["global"]["ipo_id"].iloc[0]   # collide with a US row id
    full = frames["global"].iloc[[0]].copy()
    full["ipo_id"] = frames["india"]["ipo_id"].iloc[1]
    frames["india"] = pd.concat([frames["india"], sparse], ignore_index=True)
    frames["global"] = pd.concat([frames["global"], full], ignore_index=True)
    frames["india"].loc[frames["india"].index[-1], "sub_total"] = np.nan
    df, report = B.build_frame(frames)
    assert df["ipo_id"].is_unique
    assert report.dropped["duplicate ipo_id (less filled row)"] == 2
    assert len(df) == len(_frames()["india"]) + len(_frames()["global"])


# ---- sanity ------------------------------------------------------------------------------------------
def test_impossible_values_are_blanked_and_counted_not_clipped():
    frames = _frames()
    india = frames["india"]
    india.loc[0, "issue_price"] = 0.0
    india.loc[1, "list_open_gain_pct"] = -150.0
    india.loc[2, "sub_qib"] = -3.0
    india.loc[3, "fresh_issue_pct"] = 120.0
    india.loc[4, "price_low"], india.loc[4, "price_high"] = 200.0, 100.0
    df, report = B.build_frame(frames)
    row = df.set_index("ipo_id")
    assert np.isnan(row.loc[india.loc[0, "ipo_id"], "issue_price"])
    assert np.isnan(row.loc[india.loc[1, "ipo_id"], "list_open_gain_pct"])
    assert np.isnan(row.loc[india.loc[2, "ipo_id"], "sub_qib"])
    assert np.isnan(row.loc[india.loc[3, "ipo_id"], "fresh_issue_pct"])
    assert np.isnan(row.loc[india.loc[4, "ipo_id"], "price_low"])
    assert np.isnan(row.loc[india.loc[4, "ipo_id"], "price_high"])
    assert report.blanked["issue_price not positive"] >= 1
    assert report.blanked["list_open_gain_pct below -100%"] >= 1


def test_unreadable_text_in_numeric_column_is_counted_not_guessed():
    frames = _frames()
    frames["india"]["lot_size"] = frames["india"]["lot_size"].astype("object")
    frames["india"].loc[0, "lot_size"] = "about 50"
    df, report = B.build_frame(frames)
    assert any("lot_size" in k for k in report.coerce_failures)
    assert np.isnan(df.loc[df["ipo_id"] == frames["india"].loc[0, "ipo_id"], "lot_size"].iloc[0])


# ---- context -----------------------------------------------------------------------------------------
def test_without_indices_context_is_nan_and_the_build_says_so():
    df, report = B.build_frame(_frames(), indices=None)
    assert not report.context_filled
    assert df[CONTEXT_COLS].isna().all().all()
    assert any("indices.parquet not found" in n for n in report.notes)


def test_with_indices_context_is_filled_where_open_dates_exist():
    df, report = B.build_frame(_frames(), indices=make_indices())
    assert report.context_filled
    india = df[df["market"] == "IN_MAIN"]
    assert india["idx_ret_20d"].notna().mean() > 0.9
    us = df[df["market"] == "US"]
    assert us["idx_ret_20d"].isna().all()    # US rows have no open date in this synthetic table


# ---- files -------------------------------------------------------------------------------------------
def test_build_all_writes_parquet_and_coverage_and_fails_on_missing_input(tmp_path):
    for name, frame in _frames().items():
        frame.to_parquet(tmp_path / f"{name}.parquet", index=False)
    out = B.build_all(processed=tmp_path)
    assert out == tmp_path / "ipos.parquet"
    written = pd.read_parquet(out)
    assert list(written.columns) == ALL_COLS
    text = (tmp_path / "COVERAGE.md").read_text(encoding="utf-8")
    assert "## Rows per market" in text and "## Listing-year histogram" in text
    assert "| IN_MAIN |" in text or "| IN_SME |" in text
    assert "## Non-null share by column (%)" in text
    assert "indices.parquet not found" in text
    (tmp_path / "global.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="global"):
        B.build_all(processed=tmp_path)


def test_markets_used_are_the_schema_markets():
    df, _ = B.build_frame(_frames())
    assert set(df["market"].dropna().unique()) <= set(MARKETS)


# ---- placeholder zeros ------------------------------------------------------------------------------
def test_exact_zero_listing_gain_and_gmp_are_blanked_and_counted_not_trained_as_flat():
    frames = _frames()
    india = frames["india"].copy()
    india.loc[0, "list_open_gain_pct"] = 0.0      # placeholder: a missing code written as 0
    india.loc[1, "gmp_last"] = 0.0
    india.loc[2, "list_close_gain_pct"] = 4.5     # a real non-zero value must survive
    frames["india"] = india
    df, report = B.build_frame(frames)
    by_id = df.set_index("ipo_id")
    assert np.isnan(by_id.loc[india.loc[0, "ipo_id"], "list_open_gain_pct"])
    assert np.isnan(by_id.loc[india.loc[1, "ipo_id"], "gmp_last"])
    assert by_id.loc[india.loc[2, "ipo_id"], "list_close_gain_pct"] == 4.5
    assert report.blanked["list_open_gain_pct exactly 0 (placeholder)"] >= 1
    assert report.blanked["gmp_last exactly 0 (placeholder)"] >= 1


def test_exact_zero_subscription_multiples_are_blanked_including_day_columns():
    frames = _frames()
    india = frames["india"].copy()
    india.loc[0, "sub_total"] = 0.0               # placeholder multiple
    india.loc[1, "sub_qib_d1"] = 0.0              # day-wise placeholder
    india.loc[2, "sub_retail"] = 0.0
    india.loc[2, "sub_total_d2"] = 2.5            # a real non-zero day value must survive
    frames["india"] = india
    df, report = B.build_frame(frames)
    by_id = df.set_index("ipo_id")
    assert np.isnan(by_id.loc[india.loc[0, "ipo_id"], "sub_total"])
    assert np.isnan(by_id.loc[india.loc[1, "ipo_id"], "sub_qib_d1"])
    assert np.isnan(by_id.loc[india.loc[2, "ipo_id"], "sub_retail"])
    assert by_id.loc[india.loc[2, "ipo_id"], "sub_total_d2"] == 2.5
    assert report.blanked["sub_total exactly 0 (placeholder)"] >= 1
    assert report.blanked["sub_qib_d1 exactly 0 (placeholder)"] >= 1
    assert report.blanked["sub_retail exactly 0 (placeholder)"] >= 1
