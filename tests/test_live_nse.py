"""NSE parsers against real saved responses, and every failure path through httpx.MockTransport."""
import json
from datetime import date, datetime

import httpx
import pytest

from conftest import fixture_bytes, fixture_text
from ipo_radar.errors import SourceError
from ipo_radar.live import nse
from ipo_radar.live._http import make_client

pytestmark = pytest.mark.usefixtures("live_isolation")
NOW = datetime(2026, 10, 15, 12, 0).astimezone()
TODAY = date(2026, 10, 15)


def test_bid_details_maps_reported_categories_and_skips_sub_rows():
    got = nse.parse_bid_details(json.loads(fixture_text("nse_bid_details_hdfire.json")), "HDFIRE", NOW)
    assert set(got) == {"sub_qib", "sub_nii", "sub_bnii", "sub_snii", "sub_retail", "sub_employee", "sub_total"}
    assert got["sub_qib"].value == pytest.approx(1.447860938086533)
    assert got["sub_bnii"].value == pytest.approx(55.972415984667364)
    assert got["sub_snii"].value == pytest.approx(43.41383935519903)
    assert got["sub_employee"].value == pytest.approx(7.593058562231156)
    assert got["sub_total"].value == pytest.approx(16.742714309693792)
    # sampled at NSE's own updateTime (IST), not the fetch time
    assert all(v.source == "nse" and v.fetched_at.utcoffset().total_seconds() == 19800 for v in got.values())
    assert all(v.fetched_at.date() == date(2026, 10, 15) for v in got.values())
    assert "sub_shareholder" not in got  # no shareholder quota on this issue: absent, not 0


def test_sme_bid_details_without_multiples_report_nothing():
    # SME bid tables give bids and applications but no times-subscribed column, so no key is set.
    assert nse.parse_bid_details(json.loads(fixture_text("nse_bid_details_ravita_sme.json")), "RAVITA", NOW) == {}


def test_blank_category_value_is_absent_not_zero():
    payload = {"data": [{"srNo": "1", "category": "Qualified Institutional Buyers(QIBs)", "noOfTime": ""},
                        {"srNo": None, "category": "Total", "noOfTime": "0.0"}]}
    got = nse.parse_bid_details(payload, "X", NOW)
    assert "sub_qib" not in got
    assert got["sub_total"].value == 0.0  # a real reported zero is kept


def test_bid_details_layout_change_raises():
    with pytest.raises(SourceError, match="no 'data' list"):
        nse.parse_bid_details({"rows": []}, "X", NOW)
    with pytest.raises(SourceError, match="non-numeric"):
        nse.parse_bid_details({"data": [{"srNo": "1", "category": "Qualified Institutional Buyers(QIBs)",
                                         "noOfTime": "n/a"}]}, "X", NOW)


def test_current_issue_fixture_groups_by_symbol_and_parses_dates():
    grouped = nse._parse_current(json.loads(fixture_text("nse_ipo_current_issue.json")))
    assert "TONBO" in grouped and "HDFIRE" in grouped and "RAVITA" in grouped
    ipo = nse._issue_row(grouped["TONBO"][0])
    assert (ipo.company, ipo.market, ipo.symbol) == ("Tonbo Imaging India Limited", "IN_MAIN", "TONBO")
    assert ipo.open_date == date(2026, 10, 15) and ipo.close_date == date(2026, 10, 19)
    assert (ipo.price_low, ipo.price_high) == (665.0, 700.0)
    sme = nse._issue_row(grouped["RAVITA"][0])
    assert sme.market == "IN_SME" and sme.price_low is None  # SME rows carry no issuePrice


def test_past_issue_band_comes_from_price_range():
    rows = json.loads(fixture_text("nse_public_past_issues_trim.json"))
    rk = next(r for r in rows if "R.K. Fashion" in (r.get("companyName") or ""))
    ipo = nse._issue_row(rk)
    assert (ipo.price_low, ipo.price_high) == (77.0, 82.0)
    assert ipo.market == "IN_SME"


def _route(current=None, past=None, upcoming=None, bids=None, status=200, body=None):
    """MockTransport handler serving the real fixtures. Any override replaces that endpoint's answer."""
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/market-data/all-upcoming-issues-ipo":
            return httpx.Response(200, text="<html>ok</html>")
        if status != 200 and path == "/api/ipo-current-issue":
            return httpx.Response(status, text="denied")
        if path == "/api/ipo-current-issue":
            return httpx.Response(200, content=current if current is not None else fixture_bytes("nse_ipo_current_issue.json"))
        if path == "/api/all-upcoming-issues":
            return httpx.Response(200, content=upcoming if upcoming is not None else fixture_bytes("nse_all_upcoming_ipo.json"))
        if path == "/api/public-past-issues":
            return httpx.Response(200, content=past if past is not None else fixture_bytes("nse_public_past_issues_trim.json"))
        if path == "/api/ipo-bid-details":
            symbol = request.url.params["symbol"]
            files = {"HDFIRE": "nse_bid_details_hdfire.json", "TONBO": "nse_bid_details_tonbo.json",
                     "RAVITA": "nse_bid_details_ravita_sme.json"}
            if bids is not None:
                return httpx.Response(bids.get(symbol, 200), content=b'{"data": []}')
            return httpx.Response(200, content=fixture_bytes(files[symbol]) if symbol in files else b'{"data": []}')
        return httpx.Response(404, text="not found")
    return handler


def test_fetch_full_flow_with_real_fixtures_and_cookie_warmup():
    client = make_client(transport=httpx.MockTransport(_route()))
    rows = nse.fetch(client, today=TODAY)
    by_symbol = {r.symbol: r for r in rows}
    tonbo = by_symbol["TONBO"]
    assert tonbo.status == "unknown" and tonbo.open_date == date(2026, 10, 15)  # merge derives the status
    assert "sub_employee" in tonbo.sub and tonbo.sub["sub_employee"].source == "nse"
    assert "sub_total" in by_symbol["HDFIRE"].sub
    assert by_symbol["RAVITA"].sub.keys() == {"sub_total"}  # SME: only the summary total
    assert any(r.company == "Eventions Limited" for r in rows)
    assert not any(r.company.startswith("Sri Priyanka") for r in rows)  # not in the 14-day window
    assert not any(r.company == "SMC Global Securities Limited" for r in rows)  # DEBT is not an equity issue


def test_past_window_excludes_old_listings():
    old = json.dumps([{"companyName": "Old Co Limited", "symbol": "OLD", "securityType": "SME",
                       "ipoStartDate": "01-JAN-2026", "ipoEndDate": "05-JAN-2026", "listingDate": "10-JAN-2026",
                       "priceRange": "Rs.10 to Rs.12"}]).encode()
    rows = nse.fetch(make_client(transport=httpx.MockTransport(_route(past=old))), today=TODAY)
    assert not any(r.symbol == "OLD" for r in rows)


def test_upcoming_and_past_do_not_duplicate_current_symbols():
    rows = nse.fetch(make_client(transport=httpx.MockTransport(_route())), today=TODAY)
    symbols = [r.symbol for r in rows]
    assert len(symbols) == len(set(symbols))


def test_one_bid_detail_failure_costs_only_that_issue_its_split():
    client = make_client(transport=httpx.MockTransport(_route(bids={"HDFIRE": 500})))
    rows = {r.symbol: r for r in nse.fetch(client, today=TODAY)}
    assert "sub_total" in rows["TONBO"].sub
    assert set(rows["HDFIRE"].sub) == {"sub_total"}  # the summary row still comes from the current feed
    assert any("category split unavailable" in w for w in rows["HDFIRE"].warnings)


@pytest.mark.parametrize("status, message", [(403, "HTTP 403"), (429, "HTTP 429"), (500, "HTTP 500")])
def test_http_errors_raise_source_error(status, message):
    client = make_client(transport=httpx.MockTransport(_route(status=status)))
    with pytest.raises(SourceError, match=message):
        nse.fetch(client, today=TODAY)


def test_429_is_not_retried_but_403_is_retried_once_after_warmup():
    calls = {"current": 0}

    def handler(request):
        if request.url.path == "/api/ipo-current-issue":
            calls["current"] += 1
            if calls["current"] == 1:
                return httpx.Response(403, text="denied")
            return httpx.Response(200, content=fixture_bytes("nse_ipo_current_issue.json"))
        return _route()(request)

    rows = nse.fetch(make_client(transport=httpx.MockTransport(handler)), today=TODAY)
    assert calls["current"] == 2 and rows


def test_timeout_raises_source_error():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)
    with pytest.raises(SourceError, match="timeout"):
        nse.fetch(make_client(transport=httpx.MockTransport(handler)), today=TODAY)


def test_empty_body_raises_source_error():
    client = make_client(transport=httpx.MockTransport(_route(current=b"   \n")))
    with pytest.raises(SourceError, match="empty body"):
        nse.fetch(client, today=TODAY)


def test_html_instead_of_json_raises_source_error():
    client = make_client(transport=httpx.MockTransport(_route(current=b"<html>Access Denied</html>")))
    with pytest.raises(SourceError, match="did not return JSON"):
        nse.fetch(client, today=TODAY)


def test_changed_layout_raises_source_error():
    client = make_client(transport=httpx.MockTransport(_route(current=b'{"status": "ok", "items": []}')))
    with pytest.raises(SourceError, match="not a list"):
        nse.fetch(client, today=TODAY)
    client = make_client(transport=httpx.MockTransport(_route(current=b'[{"symbol": "X"}]')))
    with pytest.raises(SourceError, match="without a company name"):
        nse.fetch(client, today=TODAY)


def test_classification_of_shareholder_and_sub_rows():
    assert nse._classify("5", "Shareholder(s)") == "sub_shareholder"
    assert nse._classify("1(a)", "Foreign Institutional Investors(FIIs)") is None
    assert nse._classify("2.1(a)", "Corporates") is None


def test_negative_or_non_object_bid_rows_raise():
    with pytest.raises(SourceError, match="implausible"):
        nse.parse_bid_details({"data": [{"srNo": "1", "category": "Qualified Institutional Buyers(QIBs)",
                                         "noOfTime": "-1"}]}, "X", NOW)
    with pytest.raises(SourceError, match="is not an object"):
        nse.parse_bid_details({"data": ["row"]}, "X", NOW)


def test_non_object_current_row_raises():
    with pytest.raises(SourceError, match="not an object"):
        nse._parse_current([1])


def test_upcoming_and_past_not_lists_raise():
    with pytest.raises(SourceError, match="all-upcoming-issues is not a list"):
        nse.fetch(make_client(transport=httpx.MockTransport(_route(upcoming=b'{"x": 1}'))), today=TODAY)
    with pytest.raises(SourceError, match="public-past-issues is not a list"):
        nse.fetch(make_client(transport=httpx.MockTransport(_route(past=b"{}"))), today=TODAY)


def test_past_issue_already_in_current_feed_is_not_duplicated():
    past = json.dumps([{"companyName": "Tonbo Imaging India Limited", "symbol": "TONBO", "securityType": "EQ",
                        "ipoStartDate": "15-OCT-2026", "ipoEndDate": "19-OCT-2026", "listingDate": "-",
                        "priceRange": "Rs.665 to Rs.700"}]).encode()
    rows = nse.fetch(make_client(transport=httpx.MockTransport(_route(past=past))), today=TODAY)
    assert [r.symbol for r in rows].count("TONBO") == 1


def test_connection_error_is_a_source_error():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)
    with pytest.raises(SourceError, match="network error"):
        nse.fetch(make_client(transport=httpx.MockTransport(handler)), today=TODAY)


def test_sampled_at_reads_nse_update_time_or_falls_back():
    fallback = datetime(2026, 10, 15, 9, 0).astimezone()
    got = nse._sampled_at("Updated as on 15-Oct-2026 10:48:40 hrs", fallback)
    assert got == datetime(2026, 10, 15, 10, 48, 40, tzinfo=nse.IST)
    assert nse._sampled_at("no time here", fallback) is fallback
    assert nse._sampled_at(None, fallback) is fallback


def test_final_issue_price_from_past_feed_only_when_single_number():
    rows = json.loads(fixture_text("nse_public_past_issues_trim.json"))
    rk = next(r for r in rows if "R.K. Fashion" in (r.get("companyName") or ""))
    assert nse._issue_row(rk).issue_price == 82.0
    smc = next(r for r in rows if r.get("symbol") == "SMCG04")
    assert nse._issue_row(smc).issue_price is None  # '-' is not a price
    grouped = nse._parse_current(json.loads(fixture_text("nse_ipo_current_issue.json")))
    assert nse._issue_row(grouped["TONBO"][0]).issue_price is None  # 'Rs.665 to Rs.700' is a band, not a price
    assert nse._final_price("  82 ") == 82.0 and nse._final_price("Rs.82") is None
