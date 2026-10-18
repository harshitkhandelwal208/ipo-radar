"""Chittorgarh detail parsing against saved pages, discovery, and failure paths via httpx.MockTransport."""
import re
from datetime import date, datetime

import httpx
import pytest

from conftest import fixture_text
from ipo_radar.errors import SourceError
from ipo_radar.live import chittorgarh
from ipo_radar.live._http import make_client
from ipo_radar.schema import LiveIPO

pytestmark = pytest.mark.usefixtures("live_isolation")
NOW = datetime(2026, 10, 15, 12, 0).astimezone()


def test_hdfire_ofs_issue_and_latest_snapshot():
    ipo = chittorgarh.parse_detail(fixture_text("chittorgarh_hdfire_ofs_snapshot.html"))
    assert (ipo.company, ipo.market, ipo.ipo_id) == ("HD Fire Protect", "IN_MAIN", "hd fire protect")
    assert (ipo.open_date, ipo.close_date, ipo.listing_date) == (date(2026, 10, 13), date(2026, 10, 15), date(2026, 10, 21))
    assert (ipo.price_low, ipo.price_high, ipo.lot_size) == (258.0, 271.0, 55)
    assert ipo.issue_size_m == 7120.0  # Rs 712 crore = 7120 million
    assert ipo.fresh_issue_pct == 0.0  # "OFS only"
    assert ipo.sub["sub_qib"].value == pytest.approx(1.49)
    assert ipo.sub["sub_bnii"].value == pytest.approx(50.53)
    assert "sub_employee" not in ipo.sub  # the page's employee field is empty: absent, not 0
    assert ipo.sub["sub_total"].note.startswith("snapshot")


def test_intraday_snapshot_of_an_open_issue_is_not_filed_as_a_day():
    # 09:54 on day 3 is not an end-of-day figure; Chittorgarh has no day-wise table, so nothing is invented
    ipo = chittorgarh.parse_detail(fixture_text("chittorgarh_hdfire_ofs_snapshot.html"))
    assert ipo.sub_by_day == {} and ipo.sub_by_day_source == ""
    assert ipo.sub["sub_total"].value == 16.76  # the live snapshot itself is still there


def test_closed_issue_final_snapshot_is_the_end_of_its_last_business_day():
    # Acme: 30 Sep to 6 Oct; 6 Oct at 17:54 is business day 5 (weekends are not counted)
    ipo = chittorgarh.parse_detail(fixture_text("chittorgarh_acme_closed_end_of_day.html"))
    assert set(ipo.sub_by_day) == {5}
    assert ipo.sub_by_day[5]["sub_total"] == pytest.approx(87.0168)
    assert ipo.sub_by_day_source == "chittorgarh"


def test_business_day_index_skips_weekends():
    from ipo_radar.live._util import issue_day_index
    fri = date(2026, 10, 9)  # Friday
    assert issue_day_index(fri, fri) == 1
    assert issue_day_index(fri, date(2026, 10, 12)) == 2  # Monday is the next business day
    assert issue_day_index(fri, date(2026, 10, 8)) is None


def test_fresh_only_sme_is_market_in_sme():
    ipo = chittorgarh.parse_detail(fixture_text("chittorgarh_ravita_sme_fresh.html"))
    assert ipo.market == "IN_SME" and ipo.fresh_issue_pct == 100.0 and ipo.lot_size == 1200
    assert (ipo.price_low, ipo.price_high) == (105.0, 112.0)
    assert ipo.sub_by_day == {}


def test_upcoming_issue_has_no_subscription_and_no_guess():
    ipo = chittorgarh.parse_detail(fixture_text("chittorgarh_jio_upcoming_no_snapshot.html"))
    assert ipo.sub == {} and ipo.sub_by_day == {}
    assert ipo.open_date == date(2026, 10, 21) and ipo.issue_size_m == 302130.0


def test_open_issue_intraday_snapshot_gives_no_day_figure():
    ipo = chittorgarh.parse_detail(fixture_text("chittorgarh_tonbo_open.html"))
    assert ipo.sub_by_day == {}
    assert ipo.sub["sub_qib"].value == 0.0  # a reported zero is kept
    assert ipo.sub["sub_qib"].value == 0.0  # a reported zero is kept


def test_missing_ipo_date_row_is_a_layout_change():
    html = fixture_text("chittorgarh_hdfire_ofs_snapshot.html").replace("IPO Date", "Issue Dates")
    with pytest.raises(SourceError, match="layout changed"):
        chittorgarh.parse_detail(html)


def test_missing_heading_is_a_layout_change():
    html = re.sub(r"<h1[^>]*>.*?</h1>", "", fixture_text("chittorgarh_hdfire_ofs_snapshot.html"), flags=re.S)
    with pytest.raises(SourceError, match="no company heading"):
        chittorgarh.parse_detail(html)


def test_snapshot_with_non_numeric_value_raises():
    html = fixture_text("chittorgarh_hdfire_ofs_snapshot.html")
    assert '\\"qib\\":1.49' in html  # the escaped payload form this test edits
    with pytest.raises(SourceError, match="non-numeric qib"):
        chittorgarh.parse_detail(html.replace('\\"qib\\":1.49', '\\"qib\\":\\"abc\\"'))


def test_snapshot_json_that_does_not_decode_raises():
    html = fixture_text("chittorgarh_hdfire_ofs_snapshot.html").replace('\\"nii_offered\\"', '\\"nii_offered')
    with pytest.raises(SourceError, match="unreadable"):
        chittorgarh.parse_detail(html)


def test_discovery_reads_real_and_escaped_links_and_dedupes():
    body = fixture_text("chittorgarh_list_links.html")
    client = make_client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=body)))
    paths = chittorgarh._discover(client)
    assert paths == ["/ipo/tonbo-imaging-india-ipo/3215/", "/ipo/hd-fire-protect-ipo/2628/",
                     "/ipo/sahajanand-medical-technologies-ipo/2550/", "/ipo/green-asia-impex-ipo/2963/"]


def test_discovery_with_no_links_raises():
    client = make_client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text="<html></html>")))
    with pytest.raises(SourceError, match="no IPO links"):
        chittorgarh._discover(client)


def _site(detail_status=None, list_status=200, empty_detail=False, timeout_home=False):
    """Serves the links fixture for list pages and the real detail fixtures for IPO pages."""
    detail_files = {"/ipo/tonbo-imaging-india-ipo/3215/": "chittorgarh_tonbo_open.html",
                    "/ipo/hd-fire-protect-ipo/2628/": "chittorgarh_hdfire_ofs_snapshot.html"}
    links = fixture_text("chittorgarh_list_links.html")

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/" and timeout_home:
            raise httpx.ConnectTimeout("no route", request=request)
        if path.startswith("/report/") or path == "/":
            return httpx.Response(list_status, text=links)
        if detail_status and path in detail_status:
            return httpx.Response(detail_status[path], text="denied")
        if empty_detail and path == "/ipo/hd-fire-protect-ipo/2628/":
            return httpx.Response(200, content=b"")
        if path in detail_files:
            return httpx.Response(200, text=fixture_text(detail_files[path]))
        return httpx.Response(200, text="<html>no table</html>")  # the other links fail the layout check
    return handler


def test_fetch_returns_good_pages_and_skips_failed_ones():
    client = make_client(transport=httpx.MockTransport(_site(detail_status={"/ipo/tonbo-imaging-india-ipo/3215/": 403})))
    rows = chittorgarh.fetch(client)
    assert {r.company for r in rows} == {"HD Fire Protect"}  # the rest fail the layout check in this fixture


def test_fetch_all_pages_failing_raises():
    client = make_client(transport=httpx.MockTransport(_site(list_status=429)))
    with pytest.raises(SourceError, match="no IPO links"):
        chittorgarh.fetch(client)


def test_empty_detail_body_is_skipped_not_guessed():
    client = make_client(transport=httpx.MockTransport(_site(empty_detail=True)))
    rows = chittorgarh.fetch(client)
    assert {r.company for r in rows} == {"Tonbo Imaging India"}


def test_timeout_on_home_page_still_uses_list_pages():
    client = make_client(transport=httpx.MockTransport(_site(timeout_home=True)))
    rows = chittorgarh.fetch(client)
    assert rows


@pytest.mark.parametrize("status", [403, 429, 500])
def test_every_page_http_error_raises_source_error(status):
    client = make_client(transport=httpx.MockTransport(lambda r: httpx.Response(status, text="no")))
    with pytest.raises(SourceError, match="no IPO links"):
        chittorgarh.fetch(client)


def test_timeout_on_every_page_raises_source_error():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)
    with pytest.raises(SourceError, match="no IPO links"):
        chittorgarh.fetch(make_client(transport=httpx.MockTransport(handler)))


def test_fresh_split_from_shares_when_the_label_is_not_recognised():
    html = fixture_text("chittorgarh_hdfire_ofs_snapshot.html")
    # Half of the 2,62,84,500 shares are an OFS (the value is split by <!-- --> comments in the real page).
    head, label, tail = html.partition("Offer for Sale</a>")
    edited = (head + label + tail.replace("2,62,84,500", "1,31,42,250", 1)).replace("OFS only", "Mixed")
    ipo = chittorgarh.parse_detail(edited)
    assert ipo.fresh_issue_pct == pytest.approx(50.0)


def test_fresh_pct_is_unknown_without_label_or_shares():
    assert chittorgarh._fresh_pct("", 100.0, None) is None


def test_single_day_issue_window():
    assert chittorgarh._issue_window("15 Oct, 2026") == (date(2026, 10, 15), date(2026, 10, 15))
    assert chittorgarh._issue_window("31 Feb, 2026") is None
    assert chittorgarh._issue_window("soon") is None


def test_unreadable_issue_date_raises():
    html = fixture_text("chittorgarh_hdfire_ofs_snapshot.html").replace("13 to 15 Oct, 2026", "13 to 15 Smarch, 2026")
    with pytest.raises(SourceError, match="unreadable IPO Date"):
        chittorgarh.parse_detail(html)


def test_snapshot_marker_without_a_list_raises():
    with pytest.raises(SourceError, match="without a list"):
        chittorgarh._snapshots("ipoBiddingDetails only")


def test_snapshot_list_of_non_objects_raises():
    with pytest.raises(SourceError, match="not a list of objects"):
        chittorgarh._snapshots('ipoBiddingDetails\\":[1, 2]')


def test_snapshot_without_or_with_bad_bid_date_raises():
    ipo = LiveIPO(ipo_id="x", market="IN_MAIN", company="X")
    with pytest.raises(SourceError, match="without bid_date"):
        chittorgarh._apply_snapshots(ipo, [{"qib": 1}])
    with pytest.raises(SourceError, match="unreadable bid_date"):
        chittorgarh._apply_snapshots(ipo, [{"bid_date": "2026-10-15"}])


def test_every_detail_page_failing_after_discovery_raises():
    links = fixture_text("chittorgarh_list_links.html")

    def handler(request):
        if request.url.path.startswith("/report/") or request.url.path == "/":
            return httpx.Response(200, text=links)
        return httpx.Response(500, text="broken")
    with pytest.raises(SourceError, match="all 4 IPO pages failed"):
        chittorgarh.fetch(make_client(transport=httpx.MockTransport(handler)))


def test_list_page_network_error_is_skipped_when_others_answer():
    links = fixture_text("chittorgarh_list_links.html")

    def handler(request):
        if request.url.path == "/":
            raise httpx.ConnectError("no route", request=request)
        if request.url.path.startswith("/report/"):
            return httpx.Response(200, text=links)
        return httpx.Response(200, text=fixture_text("chittorgarh_hdfire_ofs_snapshot.html"))
    rows = chittorgarh.fetch(make_client(transport=httpx.MockTransport(handler)))
    assert rows and all(r.company for r in rows)


def test_issue_price_comes_from_the_issue_price_row_never_from_the_band():
    hd = chittorgarh.parse_detail(fixture_text("chittorgarh_hdfire_ofs_snapshot.html"))
    assert hd.issue_price == 271.0  # "Issue Price ₹271 per share" row
    assert hd.price_high == 271.0 and hd.price_low == 258.0
    tonbo = chittorgarh.parse_detail(fixture_text("chittorgarh_tonbo_open.html"))
    assert tonbo.issue_price is None and tonbo.price_high == 700.0  # band only: no final price, no copy of the top
    jio = chittorgarh.parse_detail(fixture_text("chittorgarh_jio_upcoming_no_snapshot.html"))
    assert jio.issue_price is None


def test_snapshot_is_sampled_at_its_own_time_in_ist():
    hd = chittorgarh.parse_detail(fixture_text("chittorgarh_hdfire_ofs_snapshot.html"))
    got = hd.sub["sub_total"].fetched_at
    assert got.replace(tzinfo=None) == datetime(2026, 10, 15, 9, 54)
    assert got.utcoffset().total_seconds() == 19800
