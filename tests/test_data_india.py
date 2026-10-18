"""Offline tests for ipo_radar.data.india and india_sources. Fixtures are trimmed copies of the real sheets:
tests/fixtures/india/ (HF) and tests/fixtures/india/kaggle/ (Kaggle, same layout as data/raw/kaggle)."""
from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import huggingface_hub
import numpy as np
import pandas as pd
import pytest

from ipo_radar.data import india as I
from ipo_radar.data import india_sources as KS
from ipo_radar.errors import SourceError
from ipo_radar.schema import ALL_COLS, CONTEXT_COLS

FIX = Path(__file__).parent / "fixtures" / "india"
FIXK = FIX / "kaggle"
AS_OF = date(2026, 10, 15)
ROHAN = FIXK / KS.KAGGLE_SLUGS["rohan"]


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return I.build_india(FIX, FIXK, as_of=AS_OF)


def _row(df: pd.DataFrame, ipo_id: str) -> pd.Series:
    rows = df[df.ipo_id == ipo_id]
    assert len(rows) == 1, f"{ipo_id} not unique"
    return rows.iloc[0]


def _copy(tmp_path: Path) -> tuple[Path, Path]:
    shutil.copytree(FIX, tmp_path / "hf", ignore=shutil.ignore_patterns("kaggle"))
    shutil.copytree(FIXK, tmp_path / "kaggle")
    return tmp_path / "hf", tmp_path / "kaggle"


# ---- contract -------------------------------------------------------------------------------------
def test_columns_are_exactly_the_schema(df):
    assert list(df.columns) == ALL_COLS


def test_markets_counts_and_unique_ids(df):
    assert set(df.market) == {"IN_MAIN", "IN_SME"}
    assert df.market.value_counts().to_dict() == {"IN_MAIN": 11, "IN_SME": 7}
    assert df.ipo_id.is_unique


def test_context_columns_left_for_context_module(df):
    assert df[CONTEXT_COLS].isna().all().all()


def test_issue_price_inside_band_when_band_known(df):
    known = df.price_low.notna()
    assert ((df.issue_price[known] >= df.price_low[known]) & (df.issue_price[known] <= df.price_high[known])).all()


def test_issue_size_converted_from_crore_to_millions(df):
    # Mahindra: 277.96 crore in the source, so 2779.6 million INR.
    assert _row(df, "IN_MAIN:607").issue_size_m == pytest.approx(2779.6)


def test_sub_total_is_final_total_not_last_day(df):
    assert _row(df, "IN_MAIN:607").sub_total == pytest.approx(9.80)


def test_listing_open_gain_formula(df):
    raw = pd.read_excel(FIX / I.MAIN_FILE, sheet_name="data_ordered")
    r = raw[raw.mapping_key == 607].iloc[0]
    expected = (float(r["NSE_Open"]) / float(r["Final_Issue_Price"]) - 1) * 100
    assert _row(df, "IN_MAIN:607").list_open_gain_pct == pytest.approx(expected)


def test_inconsistent_listing_bar_blanks_high_and_low_only(df):
    # Open above the day's high in the source: high/low targets are not trustworthy, open/close stay.
    r = _row(df, "IN_MAIN:138")
    assert np.isnan(r.list_high_gain_pct) and np.isnan(r.list_low_gain_pct)
    assert not np.isnan(r.list_open_gain_pct) and not np.isnan(r.list_close_gain_pct)


def test_share_count_cell_is_not_read_as_multiple(df):
    # SME row 504 has day_3_qib = "406,400.00", an absolute share count, not a multiple.
    assert np.isnan(_row(df, "IN_SME:504").sub_qib_d3)


def test_sme_targets_use_overall_prices(df):
    raw = pd.read_excel(FIX / I.SME_FILE, sheet_name="data_ordered")
    r = raw[raw.mapping_key == 620].iloc[0]
    expected = (float(r["Open"]) / float(r["Final_Issue_Price"]) - 1) * 100
    assert _row(df, "IN_SME:620").list_open_gain_pct == pytest.approx(expected)


def test_sme_lot_size_comes_from_kaggle(df):
    # The SME sheet has no lot size; warcoder supplies it for linked IPOs.
    assert _row(df, "IN_SME:620").lot_size == pytest.approx(6000)


def test_gmp_only_from_gmp_sources_and_open_never_set(df):
    assert df.gmp_open.isna().all() and df.gmp_pct_open.isna().all()
    assert df.loc[df.gmp_last.notna(), "ipo_id"].tolist() == ["IN_MAIN:613", "IN_MAIN:356", "IN_SME:620", "IN_SME:382"]


def test_gmp_value_and_percent_of_issue_price(df):
    r = _row(df, "IN_MAIN:613")          # Gland Pharma, issue 1500, GMP 118
    assert r.gmp_last == pytest.approx(118.0)
    assert r.gmp_pct_last == pytest.approx(118.0 / 1500 * 100)


def test_n_missing_core_counts_the_four_core_fields(df):
    r = _row(df, "IN_SME:906")
    expected = sum(pd.isna(getattr(r, c)) for c in ["issue_price", "sub_total", "sub_retail", "list_open_gain_pct"])
    assert r.n_missing_core == expected


def test_source_fields_set(df):
    assert df.source.str.len().gt(0).all()
    assert df.source_url.str.startswith("https://").all()


def test_build_stats_report_counts():
    stats: dict[str, int] = {}
    I.build_india(FIX, FIXK, as_of=AS_OF, stats=stats)
    assert stats["hf_share_count_cells_blanked"] >= 1
    assert stats["bar_inconsistent_high_low_blanked"] >= 1
    assert stats["hf_gmp_matched"] == 2
    assert stats["nikhil_gmp_used"] == 1


def test_coverage_table_lists_each_market(df):
    table = I.coverage_table(df)
    assert "IN_MAIN (n=11)" in table and "IN_SME (n=7)" in table


# ---- Kaggle mappings -----------------------------------------------------------------------------
def test_rohan_category_subscriptions_map_to_schema(df):
    subs = pd.read_csv(ROHAN / "cg_subs.csv")
    issue = pd.read_csv(ROHAN / "cg_issue.csv")
    cg_id = issue.loc[issue.company.str.startswith("Apeejay"), "cg_ipo_id"].iloc[0]
    expected = subs.loc[subs.cg_ipo_id == cg_id].iloc[0]
    row = df[df.company.str.startswith("Apeejay")].iloc[0]
    assert row.sub_snii == pytest.approx(expected["sNII (x)"])
    assert row.sub_bnii == pytest.approx(expected["bNII (x)"])
    assert row.sub_total == pytest.approx(expected["Total (x)"])


def test_karan_hni_maps_to_sub_nii(df):
    karan = pd.read_excel(FIXK / KS.KAGGLE_SLUGS["karan"] / "Initial Public Offering (Updated).xlsx")
    hni = karan.loc[karan.IPO_Name.str.startswith("Power Mech"), "HNI"].iloc[0]
    assert _row(df, "IN_MAIN:138").sub_nii == pytest.approx(hni)


def test_karan_list_price_is_not_loaded():
    k = KS.load_karan(FIXK)
    assert k.px_open.isna().all() and k.px_close.isna().all()


def test_warcoder_market_follows_exchange_label():
    w = KS.load_warcoder(FIXK)
    raw = pd.read_csv(FIXK / KS.KAGGLE_SLUGS["warcoder"] / "IPO details 2007-2024(FEB).csv")
    expected = np.where(raw["Exchange"].astype(str).str.contains("SME"), "IN_SME", "IN_MAIN")
    assert w.market.tolist() == expected.tolist()
    assert (raw["Exchange"].astype(str).str.contains("SME")).any()   # the fixture covers an SME label


def test_exact_duplicates_within_a_source_are_dropped():
    stats: dict[str, int] = {}
    I._drop_exact_duplicates(KS.load_all(FIXK), stats)
    assert stats["kaggle_exact_duplicate_rows_dropped"] >= 1


def test_new_ipos_from_2024_are_included(df):
    new = df[df.listing_date >= "2024-01-01"]
    assert set(new.company.str.slice(0, 7)) >= {"Apeejay", "KRN Hea"}
    assert new.market.eq("IN_MAIN").all()
    assert (df.listing_date <= pd.Timestamp(AS_OF)).all()


def test_exchange_labels_normalise():
    s = pd.Series(["BSE, NSE", "BSE SME", "NSE SME", "NSE", None, "BSE, NSE, MCX-SX"])
    out = KS.exchange_norm(s).tolist()
    assert out[:4] == ["BSE,NSE", "BSE", "NSE", "NSE"] and out[5] == "BSE,MCX-SX,NSE"
    assert pd.isna(out[4])


def test_zero_price_is_missing_not_zero():
    assert KS.price(pd.Series(["0", "0.0", "163.8", "-", None])).tolist()[2] == pytest.approx(163.8)
    out = KS.price(pd.Series(["0", "0.0", "-"]))
    assert out.isna().all()


# ---- agreement rules ------------------------------------------------------------------------------
@pytest.mark.parametrize("field, a, b, expected", [
    ("sub_total", 100.0, 100.9, True),       # within 1%
    ("sub_total", 100.0, 101.2, False),      # beyond 1%
    ("px_open", 100.0, 100.4, True),         # within 0.5%
    ("px_open", 100.0, 100.6, False),
    ("issue_cr", 1000.0, 1004.0, True),
    ("fresh_pct", 10.0, 10.4, True),         # within 0.5 percentage points
    ("fresh_pct", 10.0, 10.6, False),
    ("lot", 20.0, 20.0, True),
    ("lot", 20.0, 25.0, False),
    ("sub_total", 0.0, 0.0, True),
])
def test_agreement_tolerances(field, a, b, expected):
    assert I._agree(field, a, b) is expected


def test_single_source_value_kept_and_counted():
    stats: dict[str, dict[str, int]] = {}
    assert I._resolve_value("sub_total", [("hf", 5.0), ("karan", None)], stats) == 5.0
    assert stats["sub_total"] == {"single": 1, "overlap": 0, "disagree": 0}


def test_disagreeing_pair_resolves_to_nan_and_is_counted():
    stats: dict[str, dict[str, int]] = {}
    assert I._resolve_value("sub_total", [("hf", 5.0), ("karan", 6.0)], stats) is None
    assert stats["sub_total"] == {"single": 0, "overlap": 1, "disagree": 1}


def test_agreeing_pair_resolves_to_highest_priority_source():
    stats: dict[str, dict[str, int]] = {}
    assert I._resolve_value("sub_total", [("hf", 5.0), ("karan", 5.02)], stats) == 5.0
    assert stats["sub_total"]["disagree"] == 0


def test_subscription_disagreement_from_source_blanks_the_cell(tmp_path):
    hf, kg = _copy(tmp_path)
    p = kg / KS.KAGGLE_SLUGS["rohan"] / "cg_subs.csv"
    subs = pd.read_csv(p)
    mask = subs.company.str.contains("Mahindra", case=False)
    subs.loc[mask, "Total (x)"] = subs.loc[mask, "Total (x)"] * 1.05
    subs.to_csv(p, index=False)
    stats: dict[str, int] = {}
    out = I.build_india(hf, kg, as_of=AS_OF, stats=stats)
    assert np.isnan(_row(out, "IN_MAIN:607").sub_total)
    assert stats["field_sub_total_disagree"] >= 1


@pytest.mark.parametrize("factor, kept", [(1.002, True), (1.01, False)])
def test_listing_price_disagreement_threshold(tmp_path, factor, kept):
    hf, kg = _copy(tmp_path)
    p = kg / KS.KAGGLE_SLUGS["rohan"] / "cg_listing.csv"
    lst = pd.read_csv(p)
    names = pd.read_csv(kg / KS.KAGGLE_SLUGS["rohan"] / "cg_issue.csv")
    cg = names.loc[names.company.str.contains("Mahindra", case=False), "cg_ipo_id"].iloc[0]
    lst.loc[lst.cg_ipo_id == cg, "Open Price on Listing (Rs.)"] *= factor
    lst.to_csv(p, index=False)
    out = I.build_india(hf, kg, as_of=AS_OF)
    gain = _row(out, "IN_MAIN:607").list_open_gain_pct
    assert (not np.isnan(gain)) is kept


def test_nikhil_gmp_rejected_when_it_disagrees_with_hf(tmp_path):
    hf, kg = _copy(tmp_path)
    p = kg / KS.KAGGLE_SLUGS["nikhil"] / "Mainline IPO GMP Performance.csv"
    g = pd.read_csv(p)
    g["GMP"] = g["GMP"] * 2
    g.to_csv(p, index=False)
    stats: dict[str, int] = {}
    out = I.build_india(hf, kg, as_of=AS_OF, stats=stats)
    assert stats["nikhil_gmp_used"] == 0
    assert _row(out, "IN_MAIN:613").gmp_last == pytest.approx(118.0)   # HF value still stands


def test_nikhil_gmp_validated_when_it_agrees(df):
    assert _row(df, "IN_MAIN:613").gmp_last == pytest.approx(118.0)
    stats: dict[str, int] = {}
    I.build_india(FIX, FIXK, as_of=AS_OF, stats=stats)
    assert stats["nikhil_gmp_agreement_pct"] == 100.0


# ---- date and universe boundaries ----------------------------------------------------------------
def _rec(src: str, name: str, listing: str, sym: str = "") -> dict:
    return {"src": src, "nkey": KS.name_key(name), "sym": sym, "listing": pd.Timestamp(listing)}


def test_link_accepts_one_day_and_rejects_two_days():
    one = pd.DataFrame([_rec("hf", "Alpha Foods Ltd", "2015-05-10"),
                        _rec("karan", "Alpha Foods Limited", "2015-05-11")])
    two = pd.DataFrame([_rec("hf", "Delta Metals", "2015-05-10"),
                        _rec("karan", "Delta Metals", "2015-05-12")])
    assert I._link(one)[0] == I._link(one)[1]        # one day apart: linked
    assert I._link(two)[0] != I._link(two)[1]        # two days apart: not linked


def test_chained_links_cannot_span_more_than_one_day():
    rec = pd.DataFrame([_rec("hf", "Alpha Foods", "2015-05-10"),
                        _rec("karan", "Alpha Foods", "2015-05-11"),
                        _rec("aman", "Alpha Foods", "2015-05-12")])
    ent = I._link(rec)
    assert ent[0] == ent[1] and ent[0] != ent[2]


def test_link_keeps_only_one_to_one_pairs():
    rec = pd.DataFrame([_rec("hf", "Beta Corp", "2016-06-01"),
                        _rec("karan", "Beta Corp", "2016-06-01"),
                        _rec("karan", "Beta Corp", "2016-06-01")])
    ent = I._link(rec)
    assert ent[0] != ent[1] and ent[0] != ent[2]   # two candidates in one source: no link


def test_link_never_merges_two_records_of_one_source():
    rec = pd.DataFrame([_rec("karan", "Gamma Ltd", "2017-01-01"), _rec("karan", "Gamma Ltd", "2017-01-01")])
    ent = I._link(rec)
    assert ent[0] != ent[1]


@pytest.mark.parametrize("a, b, related", [
    ("l t finance", "l t finance holdings", True),         # leading tokens, two or more
    ("finance", "finance holdings", False),                # one token is too short
    ("edserv soft systems", "edserv softsystems", False),  # spelling change: not linked by name
    ("", "", False),                                       # blank names are never a link
])
def test_name_relation(a, b, related):
    assert I._name_related(a, b) is related


def test_universe_window_boundaries():
    frame = pd.DataFrame({
        "market": ["IN_MAIN"] * 4, "ipo_id": ["IN_MAIN:1", "IN_MAIN:2", "IN_MAIN:3", "IN_MAIN:4"],
        "listing": pd.to_datetime(["2008-12-31", "2009-01-01", "2026-10-15", "2026-10-16"]),
        "sub_total": [1.0] * 4, "px_open": [None] * 4, "px_close": [None] * 4,
        **{c: [None] * 4 for c in I.SUB_FIELDS if c != "sub_total"},
    })
    stats: dict[str, int] = {}
    kept = I._universe(frame, AS_OF, stats)
    assert kept.ipo_id.tolist() == ["IN_MAIN:2", "IN_MAIN:3"]
    assert stats["excluded_outside_window"] == 2


def test_kaggle_only_row_needs_an_outcome():
    cols = {c: [None, None] for c in I.SUB_FIELDS + ["px_open", "px_close"]}
    frame = pd.DataFrame({"market": ["IN_SME", "IN_SME"], "ipo_id": [None, None],
                          "listing": pd.to_datetime(["2024-03-01", "2024-03-02"]), **cols})
    frame.loc[1, "sub_total"] = 2.0
    stats: dict[str, int] = {}
    kept = I._universe(frame, AS_OF, stats)
    assert len(kept) == 1 and kept.listing.iloc[0] == pd.Timestamp("2024-03-02")
    assert stats["excluded_kaggle_only_without_outcome"] == 1


# ---- failure paths ---------------------------------------------------------------------------------
def test_hf_file_missing_raises(tmp_path):
    with pytest.raises(SourceError, match="missing"):
        I.build_india(tmp_path, FIXK)


def test_kaggle_file_missing_raises(tmp_path):
    hf, kg = _copy(tmp_path)
    (kg / KS.KAGGLE_SLUGS["karan"] / "Initial Public Offering (Updated).xlsx").unlink()
    with pytest.raises(SourceError, match="missing"):
        I.build_india(hf, kg, as_of=AS_OF)


def test_corrupt_kaggle_workbook_raises(tmp_path):
    hf, kg = _copy(tmp_path)
    (kg / KS.KAGGLE_SLUGS["karan"] / "Initial Public Offering (Updated).xlsx").write_bytes(b"not a workbook")
    with pytest.raises(SourceError, match="could not read"):
        I.build_india(hf, kg, as_of=AS_OF)


def test_zero_byte_kaggle_file_raises(tmp_path):
    hf, kg = _copy(tmp_path)
    (kg / KS.KAGGLE_SLUGS["warcoder"] / "IPO details 2007-2024(FEB).csv").write_bytes(b"")
    with pytest.raises(SourceError, match="could not read|no rows"):
        I.build_india(hf, kg, as_of=AS_OF)


def test_changed_column_layout_raises(tmp_path):
    hf, kg = _copy(tmp_path)
    p = kg / KS.KAGGLE_SLUGS["rohan"] / "cg_issue.csv"
    pd.read_csv(p).drop(columns=["Issue Amount (Rs.cr.)"]).to_csv(p, index=False)
    with pytest.raises(SourceError, match="layout changed.*Issue Amount"):
        I.build_india(hf, kg, as_of=AS_OF)


def test_changed_date_format_raises(tmp_path):
    hf, kg = _copy(tmp_path)
    p = kg / KS.KAGGLE_SLUGS["aman"] / "ipo_historical_data_all_pages.csv"
    d = pd.read_csv(p)
    d["Issue Details List Date"] = "2026/05/14"
    d.to_csv(p, index=False)
    with pytest.raises(SourceError, match="do not match format"):
        I.build_india(hf, kg, as_of=AS_OF)


def test_thabresh_with_too_few_columns_raises(tmp_path):
    hf, kg = _copy(tmp_path)
    (kg / KS.KAGGLE_SLUGS["thabresh"] / "ipo.csv").write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
    with pytest.raises(SourceError, match="layout changed"):
        I.build_india(hf, kg, as_of=AS_OF)


def test_empty_kaggle_table_raises(tmp_path):
    hf, kg = _copy(tmp_path)
    p = kg / KS.KAGGLE_SLUGS["warcoder"] / "IPO details 2007-2024(FEB).csv"
    header = p.read_text(encoding="utf-8").splitlines()[0]
    p.write_text(header + "\n", encoding="utf-8")
    with pytest.raises(SourceError, match="no rows"):
        I.build_india(hf, kg, as_of=AS_OF)


def test_download_failures_become_source_error(monkeypatch, tmp_path):
    def boom(**_kwargs):
        raise ConnectionError("DNS lookup failed")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    with pytest.raises(SourceError, match="download"):
        I.download(tmp_path)
