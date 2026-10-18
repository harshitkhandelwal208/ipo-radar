"""Offline tests for data/global_ipos.py. Fixtures are trimmed copies of the real source files (see tests/fixtures/global)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import httpx
import pandas as pd
import pytest

from ipo_radar import schema
from ipo_radar.data import global_ipos as g
from ipo_radar.errors import SourceError

FIX = Path(__file__).parent / "fixtures" / "global"
KFIX = FIX / "kaggle"


@pytest.fixture(autouse=True)
def _no_disk_no_wait(monkeypatch, tmp_path):
    """Keep tests off the real data/raw folder and skip the 1 s politeness delay."""
    monkeypatch.setattr(g, "RAW_DIR", tmp_path / "iposcoop")
    monkeypatch.setattr(g, "MIN_DELAY_S", 0.0)


def _html(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def _table(rows: str) -> str:
    """Wrap raw <tr> rows in the real table header so the parser sees the expected layout."""
    head = _html("iposcoop_pricings_trim.html")
    start = head.index("<thead>")
    end = head.index("</thead>") + len("</thead>")
    return f"<html><body><table class=\"standard-table ipolist\">{head[start:end]}<tbody>{rows}</tbody></table></body></html>"


def _row(company="Test Co", symbol="TSTC", industry="Health Care", date="01/15/2025",
         shares="10.0", price="$10.00", close="$12.00") -> str:
    return (f"<tr><td>{company}</td><td>{symbol}</td><td>{industry}</td><td>{date}</td>"
            f"<td>{shares}</td><td>{price}</td><td>{close}</td><td>$0.00</td><td>0.00%</td><td>S/O</td></tr>")


def _handler(robots: str = "User-agent: *\nDisallow: /wp-admin/\n", body: str | None = None,
             status: int = 200, raise_exc: Exception | None = None):
    def handle(request: httpx.Request) -> httpx.Response:
        if raise_exc is not None:
            raise raise_exc
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=robots)
        return httpx.Response(status, text=_html("iposcoop_pricings_trim.html") if body is None else body)
    return handle


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), headers={"User-Agent": "test"})


@pytest.fixture
def price_sets(monkeypatch):
    """Point the US price-set lookup at the fixture folders (jacksoncrow first, then benjaminpo)."""
    jack, bj = KFIX / "jacksoncrow", KFIX / "benjaminpo"
    monkeypatch.setattr(g, "PRICE_SETS", (("jacksoncrow", jack), ("benjaminpo", bj)))
    return jack, bj


# ---- IPO Scoop: parsing and the happy path -------------------------------------------------------

def test_parse_real_fixture_rows():
    rows = g.parse_pricings(_html("iposcoop_pricings_trim.html"), "https://example/x")
    assert [r["symbol"] for r in rows] == ["VHCPU", "ANDG", "MDLN", "ONEG", "AMRO"]
    assert rows[1]["close_1d"] == "$23.50"


def test_build_from_real_fixture_counts_and_gains():
    frame, report = g.build_from_html({"pricings-2025.html": _html("iposcoop_pricings_trim.html")})
    assert report["raw_rows"] == 5
    assert report["excluded_spac"] == 1          # Vine Hill, industry "Blank Check"
    assert report["excluded_not_traded_yet"] == 1  # Amaero, first-day close shown as $0.00
    assert report["kept"] == 3
    by_sym = frame.set_index("symbol")
    # Gain is measured on the 1st-day close, not the source's "Return" column (which uses the current price).
    assert by_sym.loc["ANDG", "list_close_gain_pct"] == pytest.approx((23.50 / 16.0 - 1) * 100)
    assert by_sym.loc["MDLN", "list_close_gain_pct"] == pytest.approx((41.0 / 29.0 - 1) * 100)
    assert by_sym.loc["ONEG", "list_close_gain_pct"] == pytest.approx(-25.0)
    assert by_sym.loc["ANDG", "issue_size_usd_m"] == pytest.approx(11.0 * 16.0)
    assert by_sym.loc["ANDG", "ipo_id"] == "US:andg:2025-12-17"
    assert by_sym.loc["ANDG", "currency"] == "USD"
    assert by_sym.loc["ANDG", "source"] == "iposcoop"


def test_build_global_output_has_exact_schema_and_typed_dates():
    frame, _ = g.build_from_html({"pricings-2025.html": _html("iposcoop_pricings_trim.html")})
    df = g._finish(frame)
    assert list(df.columns) == schema.ALL_COLS
    assert pd.api.types.is_datetime64_any_dtype(df["listing_date"])
    assert set(df["market"]) == {"US"}
    assert df["open_date"].isna().all()           # the source has no open date
    assert df["list_open_gain_pct"].isna().all()  # nor an open price
    assert (df["n_missing_core"] == 0).all()


def test_dedupe_keeps_one_row_per_ipo_id():
    html = _html("iposcoop_pricings_trim.html")
    frame, report = g.build_from_html({"pricings-2025.html": html, "current-year-pricings.html": html})
    assert report["duplicates_removed"] == 3
    assert frame["ipo_id"].is_unique


def test_unit_warrant_and_right_symbols_are_excluded():
    html = _table(_row(symbol="ABCDU") + _row(symbol="ABCDW", company="W Co") + _row(symbol="ABCDR", company="R Co")
                  + _row(symbol="GOOD", company="Good Co"))
    frame, report = g.build_from_html({"pricings-2025.html": html})
    assert report["excluded_unit_warrant_right_by_symbol"] == 3
    assert list(frame["symbol"]) == ["GOOD"]


def test_sanity_blanks_out_of_band_gain_and_drops_bad_price(capsys):
    html = _table(_row(symbol="ZERO", price="$0.00", close="$1.00")          # issue price not positive: row dropped
                  + _row(symbol="HUGE", price="$1.00", close="$30.00")       # +2900% out of band: gain blanked
                  + _row(symbol="OKAY", price="$10.00", close="$12.00"))
    frame, report = g.build_from_html({"pricings-2025.html": html})
    assert report["sanity_issue_price_not_positive"] == 1
    assert report["sanity_blanked_list_close_gain_pct"] == 1
    assert report["sanity_dropped_total"] == 1
    assert set(frame["symbol"]) == {"HUGE", "OKAY"}
    assert pd.isna(frame.set_index("symbol").loc["HUGE", "list_close_gain_pct"])
    out = capsys.readouterr().out
    assert "SANITY VIOLATION issue_price_not_positive: US:zero:" in out
    assert "SANITY VIOLATION list_close_gain_pct_out_of_band: US:huge:" in out


def test_sanity_keeps_valid_close_when_open_is_out_of_band():
    frame = pd.DataFrame([{"ipo_id": "US:x:2010-01-01", "company": "X", "issue_price": 7.0,
                           "list_open_gain_pct": 6300.0, "list_high_gain_pct": None, "list_low_gain_pct": None,
                           "list_close_gain_pct": 5.0}])
    report: Counter = Counter()
    out = g._sanity(frame, report)
    assert len(out) == 1
    assert pd.isna(out.iloc[0]["list_open_gain_pct"])
    assert out.iloc[0]["list_close_gain_pct"] == pytest.approx(5.0)
    assert report["sanity_blanked_list_open_gain_pct"] == 1


def test_missing_offer_price_is_excluded_not_zeroed():
    html = _table(_row(symbol="NOPX", price="N/A") + _row(symbol="GOOD", company="Good Co"))
    frame, report = g.build_from_html({"pricings-2025.html": html})
    assert report["excluded_no_offer_price"] == 1
    assert list(frame["symbol"]) == ["GOOD"]
    # When every row is excluded the build fails loudly instead of returning an empty table silently.
    with pytest.raises(SourceError, match="no usable"):
        g.build_from_html({"pricings-2025.html": _table(_row(symbol="NOPX", price="N/A"))})


def test_coverage_table_lists_every_market():
    frame, _ = g.build_from_html({"pricings-2025.html": _html("iposcoop_pricings_trim.html")})
    text = g.coverage(g._finish(frame))
    assert "US" in text and "HK" in text and "OTHER" in text
    assert "ipo_id" in text and "list_close_gain_pct" in text


# ---- IPO Scoop failure paths ---------------------------------------------------------------------

def test_changed_layout_raises():
    with pytest.raises(SourceError, match="layout changed"):
        g.parse_pricings(_html("iposcoop_layout_changed.html"), "https://example/x")


def test_table_with_no_rows_raises():
    with pytest.raises(SourceError, match="no rows"):
        g.parse_pricings(_html("iposcoop_no_rows.html"), "https://example/x")


def test_changed_date_format_raises():
    with pytest.raises(SourceError, match="date format"):
        g.build_from_html({"pricings-2025.html": _table(_row(date="2025-01-15"))})


def test_http_error_raises():
    with _client(_handler(status=500)) as client, pytest.raises(SourceError, match="HTTP 500"):
        g.load_pages(refresh=True, client=client)


def test_timeout_raises():
    with _client(_handler(raise_exc=httpx.ReadTimeout("slow"))) as client, pytest.raises(SourceError, match="timeout"):
        g.load_pages(refresh=True, client=client)


def test_empty_body_raises():
    with _client(_handler(body="   ")) as client, pytest.raises(SourceError, match="empty body"):
        g.load_pages(refresh=True, client=client)


def test_robots_disallow_refuses_to_fetch():
    robots = "User-agent: *\nDisallow: /2025-pricings/\n"
    with _client(_handler(robots=robots)) as client, pytest.raises(SourceError, match="robots.txt disallows"):
        g.load_pages(refresh=True, client=client)


def test_cache_is_used_without_network():
    cache = g.RAW_DIR
    cache.mkdir(parents=True)
    html = _html("iposcoop_pricings_trim.html")
    for name in g.PAGES:
        (cache / name).write_text(html, encoding="utf-8")

    def no_network(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.url}")

    with _client(no_network) as client:
        pages = g.load_pages(refresh=False, client=client)
    assert set(pages) == set(g.PAGES)


def test_download_writes_cache_when_missing():
    html = _html("iposcoop_pricings_trim.html")
    with _client(_handler(body=html)) as client:
        pages = g.load_pages(refresh=False, client=client)
    assert (g.RAW_DIR / "pricings-2025.html").read_text(encoding="utf-8") == html
    assert set(pages) == set(g.PAGES)


# ---- shared rules --------------------------------------------------------------------------------

def test_split_like_ratio_detects_split_factors_and_not_normal_moves():
    assert g._split_like(0.5)            # 2-for-1 signature
    assert g._split_like(0.1)            # 10-for-1
    assert g._split_like(20.0)           # reverse 1-for-20 signature
    assert not g._split_like(1.2)        # normal first-day pop
    assert not g._split_like(0.8)


def test_suspect_split_blanks_all_targets_and_counts():
    report: Counter = Counter()
    tg = g._targets_from_prices(10.0, 5.1, 5.2, 4.9, 5.0, report)  # close/offer = 0.5
    assert all(v is None for v in tg.values())
    assert report["suspect_split_factor_blanked"] == 1


# ---- Kaggle loaders ------------------------------------------------------------------------------

def test_thedevastator_loader_excludes_spac_and_parses_fields():
    df, report = g.load_thedevastator(KFIX / "thedevastator_trim.csv")
    assert report["td_excluded_spac"] == 1
    by = df.set_index("symbol")
    assert set(by.index) == {"FB", "GM", "GNRC"}
    assert by.loc["FB", "exchange"] == "NASDAQ"
    assert by.loc["GM", "exchange"] == "NYSE"
    assert by.loc["FB", "list_open_gain_pct"] == pytest.approx((42.05 / 38.0 - 1) * 100, rel=1e-3)
    assert by.loc["FB", "list_close_gain_pct"] == pytest.approx((38.23 / 38.0 - 1) * 100, rel=1e-3)
    assert by.loc["FB", "listing_date"] == pd.Timestamp("2012-05-18").date()
    assert by.loc["FB", "source"] == "kaggle:thedevastator"


def test_thedevastator_layout_change_raises(tmp_path):
    bad = tmp_path / "td.csv"
    bad.write_text("Company Name,Symbol,Market,Price,Offer Amount,Date Priced,sector,firstday_open\n"
                   "X,X,NASDAQ,10,1,2012-01-01,Tech,11\n", encoding="utf-8")
    with pytest.raises(SourceError, match="expected layout"):
        g.load_thedevastator(bad)


def test_proselotis_closes_use_closeday0_only():
    closes = g.load_proselotis_closes(KFIX / "proselotis_trim.csv")
    by = {s: (d, c) for s, d, c in zip(closes.symbol, closes.listing_date, closes.close)}
    assert by["FB"] == (pd.Timestamp("2012-05-18").date(), pytest.approx(38.23, rel=1e-4))
    assert by["GM"][0] == pd.Timestamp("2010-11-18").date()


def test_proselotis_layout_change_raises(tmp_path):
    bad = tmp_path / "pro.csv"
    bad.write_text("Symbol,Year,Month,Day\nFB,2012,5,18\n", encoding="latin-1")
    with pytest.raises(SourceError, match="expected layout"):
        g.load_proselotis_closes(bad)


def test_shivamb_loader_excludes_spac_and_missing_price():
    deals, report = g.load_shivamb(KFIX / "shivamb_trim.csv")
    assert report["shivamb_excluded_spac"] == 1
    assert set(deals.symbol) == {"CRWD", "LYFT", "ZM", "QQQQ"}
    assert deals.set_index("symbol").loc["ZM", "listing_date"] == pd.Timestamp("2019-04-18").date()


def test_shivamb_layout_change_raises(tmp_path):
    bad = tmp_path / "shiv.csv"
    bad.write_text("ID,IPO Date,Symbol,Company Name\n1,\"Jun 12, 2019\",CRWD,CrowdStrike\n", encoding="utf-8")
    with pytest.raises(SourceError, match="expected layout"):
        g.load_shivamb(bad)


def test_indonesia_loader_reads_offer_and_day1_close(tmp_path):
    df, report = g.load_indonesia(KFIX / "indonesia_trim.csv")
    by = df.set_index("symbol")
    assert by.loc["RMKO", "issue_price"] == 450.0          # Starting_Price, read as the offer price
    assert by.loc["RMKO", "list_close_gain_pct"] == pytest.approx((560.0 / 450.0 - 1) * 100)
    assert by.loc["RMKO", "listing_date"] == pd.Timestamp("2023-07-31").date()
    assert set(df.market) == {"OTHER"} and set(df.country) == {"ID"}
    bad = tmp_path / "id.csv"
    bad.write_text("Code,Name,Listing_Date,Sector,Starting_Price,Day-1\nZZZZ,Z Tbk,01/02/2023,Tech,0,100\n", encoding="utf-8")
    df2, rep2 = g.load_indonesia(bad)
    assert df2.empty and rep2["id_excluded_no_offer_price"] == 1


# ---- exact-date join rule and source suffix -------------------------------------------------------

def test_shivamb_join_is_exact_and_tagged(price_sets):
    deals, report = g.load_shivamb(KFIX / "shivamb_trim.csv")
    joined = g.load_shivamb_joined(deals, report)
    by = joined.set_index("symbol")
    assert set(by.index) == {"CRWD", "LYFT", "ZM"}  # QQQQ has no price file: not joined
    # Expected values come from the first row of the fixture files, not from memory.
    zm = pd.read_csv(KFIX / "jacksoncrow" / "ZM.csv").iloc[0]
    assert by.loc["ZM", "list_close_gain_pct"] == pytest.approx((zm["Close"] / 36.0 - 1) * 100, rel=1e-9)
    assert by.loc["ZM", "list_open_gain_pct"] == pytest.approx((zm["Open"] / 36.0 - 1) * 100, rel=1e-9)
    assert (joined["source"] == "kaggle:shivamb+px_join").all()
    assert report["shivamb_no_price_file"] == 1
    assert report["shivamb_joined_via_jacksoncrow"] == 3
    lyft = pd.read_csv(KFIX / "jacksoncrow" / "LYFT.csv").iloc[0]
    assert by.loc["LYFT", "list_close_gain_pct"] == pytest.approx((lyft["Close"] / 72.0 - 1) * 100, rel=1e-9)


def test_join_rejects_listing_date_off_by_one(price_sets, tmp_path):
    bad = tmp_path / "shiv.csv"
    bad.write_text("ID,IPO Date,Symbol,Company Name,IPO Price,Current,Return\n"
                   "1,\"Apr 19, 2019\",ZM,Zoom Video Communications,$36.00,$1,0%\n", encoding="utf-8")
    deals, report = g.load_shivamb(bad)
    joined = g.load_shivamb_joined(deals, report)
    assert joined.empty
    assert report["shivamb_no_exact_first_row"] == 1


def test_join_falls_back_to_second_price_set(monkeypatch, tmp_path):
    """A ticker with no exact match in the first set is tried in the second set before giving up."""
    first = tmp_path / "first"
    first.mkdir()
    (first / "GM.csv").write_text("Date,Open,High,Low,Close\n2011-01-03,1,1,1,1\n", encoding="utf-8")
    second = KFIX / "benjaminpo"
    monkeypatch.setattr(g, "PRICE_SETS", (("first", first), ("second", second)))
    fr, why = g.us_exact_first_row("GM", pd.Timestamp("2010-11-18").date(), {})
    assert fr is not None and fr["set"] == "second" and fr["date"] == pd.Timestamp("2010-11-18").date()
    assert why == "second"


def test_no_price_file_and_mismatch_are_distinct_reasons(price_sets):
    fr, why = g.us_exact_first_row("NOTATICKER", pd.Timestamp("2019-01-01").date(), {})
    assert fr is None and why == "no_price_file"
    fr2, why2 = g.us_exact_first_row("ZM", pd.Timestamp("2019-01-01").date(), {})
    assert fr2 is None and why2 == "no_exact_first_row"


# ---- cross-source disagreement -----------------------------------------------------------------

def test_disagreeing_second_source_blanks_close_and_keeps_open(price_sets):
    """thedevastator says GM closed at 27.32; benjaminpo first row is 34.19 (more than 0.5% apart): close blanked."""
    td, _ = g.load_thedevastator(KFIX / "thedevastator_trim.csv")
    gm = td[td.symbol == "GM"].copy()
    alt = pd.DataFrame([{"ipo_id": gm.iloc[0]["ipo_id"], "close": 34.19, "source": "benjaminpo"}])
    report: Counter = Counter()
    out = g.resolve(gm, alt, report)
    assert report["close_disagree_blanked"] == 1
    assert pd.isna(out.iloc[0]["list_close_gain_pct"])
    assert out.iloc[0]["list_open_gain_pct"] == pytest.approx((35.0 / 33.0 - 1) * 100, rel=1e-3)


def test_agreeing_second_source_is_confirmed(price_sets):
    td, _ = g.load_thedevastator(KFIX / "thedevastator_trim.csv")
    fb = td[td.symbol == "FB"].copy()
    alt = pd.DataFrame([{"ipo_id": fb.iloc[0]["ipo_id"], "close": 38.23 * 1.001, "source": "proselotis"}])
    report: Counter = Counter()
    out = g.resolve(fb, alt, report)
    assert report["close_confirmed_by_second_source"] == 1
    assert out.iloc[0]["list_close_gain_pct"] == pytest.approx((38.23 / 38.0 - 1) * 100, rel=1e-3)


def test_row_with_no_target_left_after_disagreement_is_dropped():
    primary = pd.DataFrame([{"ipo_id": "US:x:2015-01-01", "market": "US", "company": "X", "symbol": "X",
                             "listing_date": pd.Timestamp("2015-01-01").date(), "issue_price": 10.0,
                             "list_open_gain_pct": None, "list_high_gain_pct": None, "list_low_gain_pct": None,
                             "list_close_gain_pct": 20.0, "source": "kaggle:thedevastator"}])
    alt = pd.DataFrame([{"ipo_id": "US:x:2015-01-01", "close": 10.0, "source": "benjaminpo"}])
    report: Counter = Counter()
    out = g.resolve(primary, alt, report)
    assert out.empty and report["no_target_dropped"] == 1


# ---- Hong Kong ------------------------------------------------------------------------------------

def test_hk_join_exact_rule_and_public_offer_multiple():
    deals = pd.DataFrame([
        {"stock_id": 9992, "company_name": "POP MART", "listing_date": "2020-12-11",
         "pricing_final_offer_price_source": 38.5, "hk_public_offering_subscription_level_times_source": 356.74,
         "offer_size_gross_proceeds_hkd_m": 5225.04, "offer_size_board_lot_size": 100},
        {"stock_id": 9992, "company_name": "POP MART wrong date", "listing_date": "2020-12-10",
         "pricing_final_offer_price_source": 38.5, "hk_public_offering_subscription_level_times_source": 1.0,
         "offer_size_gross_proceeds_hkd_m": 1.0, "offer_size_board_lot_size": 100},
        {"stock_id": 700, "company_name": "No history", "listing_date": "2020-01-01",
         "pricing_final_offer_price_source": 10.0, "hk_public_offering_subscription_level_times_source": 2.0,
         "offer_size_gross_proceeds_hkd_m": 1.0, "offer_size_board_lot_size": 100},
    ])
    first = g.load_hk_first_rows({"9992.HK", "0700.HK"}, path=KFIX / "hk_px_trim.csv")
    report: Counter = Counter()
    out = g.build_hk(deals, first, report)
    assert len(out) == 1                                        # no-history and wrong-date rows are dropped
    row = g._finish(out).iloc[0]
    assert row["ipo_id"] == "HK:9992:2020-12-11"
    assert row["sub_retail"] == pytest.approx(356.74)           # public-offer multiple -> sub_retail
    assert pd.isna(row["sub_qib"])                              # international multiple not in the source
    assert row["list_close_gain_pct"] == pytest.approx((69.0 / 38.5 - 1) * 100, rel=1e-3)
    assert row["source"] == "kaggle:ng3412+px_join"
    assert row["currency"] == "HKD" and pd.isna(row["issue_size_usd_m"])
    assert report["hk_no_exact_first_row"] == 1
    assert report["hk_no_price_history"] == 1


# ---- end-to-end wiring ----------------------------------------------------------------------------

def test_build_global_fails_loudly_when_raw_files_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(g, "TD_CSV", tmp_path / "absent.csv")
    with pytest.raises(SourceError, match="raw Kaggle files missing"):
        g.build_global()
