"""GMP parsers against saved real responses, every failure path, and the history table. Offline only."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from ipo_radar.errors import SourceError
from ipo_radar.live import _http, gmp
from ipo_radar.schema import Sourced

FIX = Path(__file__).parent / "fixtures" / "gmp"
NOW = datetime(2026, 10, 15, 5, 0, tzinfo=timezone.utc)
HOSTS = {
    "investorgain": "www.investorgain.com",
    "ipowatch": "ipowatch.in",
    "ipopremium": "www.ipopremium.in",
    "ipocentral": "ipocentral.in",
}
PARSERS = {
    "investorgain": gmp._fetch_investorgain,
    "ipowatch": gmp._fetch_ipowatch,
    "ipopremium": gmp._fetch_ipopremium,
    "ipocentral": gmp._fetch_ipocentral,
}
FIXTURES = {
    "investorgain": "investorgain_live.html",
    "ipowatch": "ipowatch_home.html",
    "ipopremium": "ipopremium_home.html",
    "ipocentral": "ipocentral_gmp.json",
}


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(_http, "MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(gmp, "RETRY_PAUSE_S", 0.0)


def fixture_bytes(site: str) -> bytes:
    return (FIX / FIXTURES[site]).read_bytes()


def client_for(responder) -> httpx.Client:
    """responder(host) -> httpx.Response, or raises the exception to simulate a network failure."""
    return httpx.Client(transport=httpx.MockTransport(lambda req: responder(req.url.host)))


def serve(site: str, body: bytes | None = None, status: int = 200) -> httpx.Client:
    """A client that answers only this site's host with the fixture (or given body and status)."""
    payload = fixture_bytes(site) if body is None else body

    def responder(host):
        if host == HOSTS[site]:
            return httpx.Response(status, content=payload)
        return httpx.Response(500, content=b"unexpected host")

    return client_for(responder)


def run_site(site: str, client: httpx.Client, tmp_path: Path) -> dict[str, Sourced]:
    return PARSERS[site](gmp._Session(client, tmp_path, NOW))


def run_fixture(site: str, tmp_path: Path, mutate=None) -> dict[str, Sourced]:
    body = fixture_bytes(site)
    if mutate is not None:
        body = mutate(body.decode("utf-8")).encode("utf-8")
    return run_site(site, serve(site, body), tmp_path)


# ---- value parsing -----------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("--", None), ("-", None), ("NA", None), ("", None), ("₹-", None), ("₹--(0.00%)", None),
    ("0", 0.0), ("₹0", 0.0), ("-45", -45.0), ("−45", -45.0), ("₹1,250", 1250.0),
    ("120 (17.32%)", 120.0), ("₹120(17.32%)", 120.0), ("1.5", 1.5), ("-0.5", -0.5),
])
def test_parse_gmp_values(text, expected):
    assert gmp.parse_gmp(text) == expected


@pytest.mark.parametrize("text", ["83 ↓ / 120 ↑", "abc", "12%", "12x", "1.2.3"])
def test_parse_gmp_rejects_anything_else(text):
    with pytest.raises(ValueError):
        gmp.parse_gmp(text)


def test_clean_name_drops_venue_tags_only():
    assert gmp._clean_name("Jio Platforms Ltd. (Mainboard)") == "Jio Platforms Ltd."
    assert gmp._clean_name("Shellz India Ltd (BSE SME)") == "Shellz India Ltd"
    assert gmp._clean_name("Paras Healthcare") == "Paras Healthcare"


# ---- real fixtures ------------------------------------------------------------------------------

def test_investorgain_fixture(tmp_path):
    res = run_fixture("investorgain", tmp_path)
    key = gmp.company_key
    assert len(res) == 7  # the repeated header row is not a company
    paras = res[key("Paras Healthcare")]
    assert paras.value == 120.0 and paras.source == "investorgain" and paras.fetched_at == NOW
    assert "updated 15-Oct 10:37" in paras.note
    assert res[key("Tonbo Imaging")].value == 500.0
    assert res[key("Shivchem Agro")].value == 1.5
    assert res[key("Acme India Industries")].value == 43.0  # name cell carries an email-obfuscation tag
    assert res[key("Bench Mark Infotech Services")].value == 15.0  # listed IPO still has a GMP cell
    # the site shows '--' for Shellz: unknown, not zero
    assert res[key("Shellz India")].value is None


def test_ipowatch_fixture(tmp_path):
    res = run_fixture("ipowatch", tmp_path)
    key = gmp.company_key
    assert len(res) == 6  # the IPO-date table without a GMP column is skipped
    assert res[key("Jio Platforms")].value == 220.0
    assert res[key("Paras Healthcare")].value == 85.0
    assert res[key("Bonfiglioli Transmissions")].value == 180.0
    assert res[key("Ravita Engineering")].value == 7.0
    # ipowatch shows a real 0 where investorgain shows '--'
    assert res[key("Shellz India")].value == 0.0
    assert "Upcoming" in res[key("Jio Platforms")].note


def test_ipopremium_fixture(tmp_path):
    res = run_fixture("ipopremium", tmp_path)
    key = gmp.company_key
    assert len(res) == 6  # the empty 'GMP Rumors' header table is skipped
    assert res[key("Paras Healthcare")].value == 100.0  # "Paras Healthcare Limited (MAINBOARD)" on the site
    assert res[key("Jio Platforms")].value == 216.0
    assert res[key("Zetwerk Manufacturing")].value == 22.0
    assert res[key("Shellz India")].value == 0.0
    assert res[key("Tonbo Imaging")].value == 500.0
    assert "band 1065–1119" in res[key("Jio Platforms")].note


def test_ipocentral_fixture(tmp_path):
    res = run_fixture("ipocentral", tmp_path)
    key = gmp.company_key
    assert len(res) == 4
    assert res[key("Jio Platforms")].value == 205.0
    assert res[key("Tonbo Imaging")].value == 510.0
    assert res[key("Zetwerk")].value == 19.0
    assert res[key("Ravita Engineering Services")].value == 5.0
    assert "site updated 2026-10-15 01:13 UTC" in res[key("Jio Platforms")].note


# ---- failure paths: each must raise SourceError, never return garbage -------------------------

@pytest.mark.parametrize("site", list(HOSTS))
@pytest.mark.parametrize("status", [403, 429, 500])
def test_http_error_status_is_source_error(site, status, tmp_path):
    with pytest.raises(SourceError, match=f"HTTP {status}"):
        run_site(site, serve(site, status=status), tmp_path)


@pytest.mark.parametrize("site", list(HOSTS))
def test_timeout_is_source_error(site, tmp_path):
    def responder(host):
        raise httpx.ReadTimeout("slow", request=httpx.Request("GET", f"https://{host}/"))
    with pytest.raises(SourceError, match="timeout"):
        run_site(site, client_for(responder), tmp_path)


@pytest.mark.parametrize("site", list(HOSTS))
def test_connection_failure_is_source_error(site, tmp_path):
    def responder(host):
        raise httpx.ConnectError("dns", request=httpx.Request("GET", f"https://{host}/"))
    with pytest.raises(SourceError, match="network error"):
        run_site(site, client_for(responder), tmp_path)


def test_dropped_connections_are_retried_then_succeed(tmp_path):
    calls = {"n": 0}

    def responder(host):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise httpx.ConnectError("reset", request=httpx.Request("GET", f"https://{host}/"))
        return httpx.Response(200, content=fixture_bytes("ipowatch"))

    res = run_site("ipowatch", client_for(responder), tmp_path)
    assert calls["n"] == 3 and len(res) == 6


def test_persistent_connection_failure_gives_up_after_retries(tmp_path):
    calls = {"n": 0}

    def responder(host):
        calls["n"] += 1
        raise httpx.ConnectError("reset", request=httpx.Request("GET", f"https://{host}/"))

    with pytest.raises(SourceError, match="network error"):
        run_site("ipowatch", client_for(responder), tmp_path)
    assert calls["n"] == gmp.RETRIES + 1


@pytest.mark.parametrize("status", [403, 429])
def test_blocked_or_rate_limited_is_not_retried(status, tmp_path):
    calls = {"n": 0}

    def responder(host):
        calls["n"] += 1
        return httpx.Response(status, content=b"no")

    with pytest.raises(SourceError, match=f"HTTP {status}"):
        run_site("ipowatch", client_for(responder), tmp_path)
    assert calls["n"] == 1


@pytest.mark.parametrize("site", list(HOSTS))
def test_empty_body_is_source_error(site, tmp_path):
    with pytest.raises(SourceError, match="empty"):
        run_site(site, serve(site, body=b"  \n"), tmp_path)


@pytest.mark.parametrize("site", ["investorgain", "ipowatch", "ipopremium"])
def test_page_without_gmp_table_is_source_error(site, tmp_path):
    with pytest.raises(SourceError, match="layout changed"):
        run_site(site, serve(site, body=b"<html><body><p>maintenance</p></body></html>"), tmp_path)


def test_investorgain_renamed_gmp_column_is_source_error(tmp_path):
    with pytest.raises(SourceError, match="layout changed"):
        run_fixture("investorgain", tmp_path, mutate=lambda s: s.replace(">GMP", ">Premium", 1))


def test_investorgain_missing_bold_value_is_source_error(tmp_path):
    with pytest.raises(SourceError, match="layout changed"):
        run_fixture("investorgain", tmp_path, mutate=lambda s: s.replace("<b>120</b>", "120", 1))


def test_investorgain_garbage_value_is_source_error(tmp_path):
    with pytest.raises(SourceError, match="not a GMP value"):
        run_fixture("investorgain", tmp_path, mutate=lambda s: s.replace("<b>120</b>", "<b>12x</b>", 1))


def test_ipowatch_renamed_gmp_column_is_source_error(tmp_path):
    with pytest.raises(SourceError, match="layout changed"):
        run_fixture("ipowatch", tmp_path, mutate=lambda s: s.replace(">GMP*<", ">Premium<"))


def test_ipocentral_not_json_is_source_error(tmp_path):
    with pytest.raises(SourceError, match="not JSON"):
        run_site("ipocentral", serve("ipocentral", body=b"<html>blocked</html>"), tmp_path)


def test_ipocentral_wrong_shape_is_source_error(tmp_path):
    with pytest.raises(SourceError, match="unexpected JSON shape"):
        run_site("ipocentral", serve("ipocentral", body=b'[{"name": "x"}]'), tmp_path)


def test_ipocentral_item_without_num_is_source_error(tmp_path):
    body = b'{"MB": [{"name": "Foo", "gmpText": "5"}], "SME": []}'
    with pytest.raises(SourceError, match="layout changed"):
        run_site("ipocentral", serve("ipocentral", body=body), tmp_path)


def test_ipocentral_no_ipos_is_source_error(tmp_path):
    with pytest.raises(SourceError, match="no IPOs"):
        run_site("ipocentral", serve("ipocentral", body=b'{"MB": [], "SME": []}'), tmp_path)


def test_duplicate_company_is_source_error_not_a_guess(tmp_path):
    body = b'{"MB": [{"name": "Foo Ltd", "num": "5"}, {"name": "Foo Limited", "num": "9"}], "SME": []}'
    with pytest.raises(SourceError, match="duplicate"):
        run_site("ipocentral", serve("ipocentral", body=body), tmp_path)


# ---- fetch(): collects successes, raises only when everything fails ----------------------------

def test_fetch_keeps_working_sources_when_one_fails(tmp_path):
    def responder(host):
        if host == HOSTS["ipocentral"]:
            return httpx.Response(429, content=b"slow down")
        for site, h in HOSTS.items():
            if host == h:
                return httpx.Response(200, content=fixture_bytes(site))
        return httpx.Response(500)

    out = gmp.fetch(client=client_for(responder), raw_dir=tmp_path, now=NOW)
    jio = out[gmp.company_key("Jio Platforms")]
    # the investorgain fixture has no Jio row; ipopremium (216) and ipowatch (220) both answer
    assert sorted(r.source for r in jio) == ["ipopremium", "ipowatch"]
    assert {r.value for r in jio} == {216.0, 220.0}
    assert all(r.source != "ipocentral" for rs in out.values() for r in rs)


def test_fetch_raises_listing_every_failure_when_all_sources_fail(tmp_path):
    client = client_for(lambda host: httpx.Response(500, content=b"down"))
    with pytest.raises(SourceError, match="GMP unavailable") as exc:
        gmp.fetch(client=client, raw_dir=tmp_path, now=NOW)
    for site in HOSTS:
        assert site in str(exc.value)


def test_fetch_saves_raw_responses_with_timestamp(tmp_path):
    client = client_for(lambda host: httpx.Response(200, content=b"<html>x</html>"))
    with pytest.raises(SourceError):
        gmp.fetch(client=client, raw_dir=tmp_path, now=NOW)
    saved = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert "investorgain/20261015T050000Z.html" in saved
    assert "ipocentral/20261015T050000Z.json" in saved


def test_raw_copy_is_written_with_the_run_timestamp(tmp_path):
    session = gmp._Session(client_for(lambda host: httpx.Response(200, content=b"<p>ok</p>")), tmp_path, NOW)
    assert session.get("ipopremium", "https://www.ipopremium.in/", "html") == "<p>ok</p>"
    assert (tmp_path / "ipopremium" / "20261015T050000Z.html").read_bytes() == b"<p>ok</p>"


# ---- history table ------------------------------------------------------------------------------

def test_history_round_trip_oldest_first_and_keeps_zero_and_negative(tmp_path):
    db = tmp_path / "state.sqlite"
    key = gmp.company_key("Paras Healthcare")
    later = NOW + timedelta(hours=1)
    readings = {key: [Sourced(120.0, "investorgain", later, "")]}
    assert gmp.record_readings(readings, db_path=db) == 1
    readings = {key: [
        Sourced(120.0, "investorgain", NOW, ""),
        Sourced(None, "ipowatch", NOW, ""),          # unknown: not stored
        Sourced(0.0, "ipopremium", NOW + timedelta(hours=2), ""),
        Sourced(-10.0, "ipocentral", NOW + timedelta(hours=3), ""),
    ]}
    assert gmp.record_readings(readings, db_path=db) == 3
    got = gmp.history(key, db_path=db)
    assert got == [
        (NOW, 120.0, "investorgain"),
        (later, 120.0, "investorgain"),
        (NOW + timedelta(hours=2), 0.0, "ipopremium"),
        (NOW + timedelta(hours=3), -10.0, "ipocentral"),
    ]
    assert gmp.history("no such ipo", db_path=db) == []


def test_history_uses_wal_and_creates_table(tmp_path):
    db = tmp_path / "sub" / "state.sqlite"  # parent folder is created on demand
    gmp.record_readings({"k": [Sourced(1.0, "s", NOW, "")]}, db_path=db)
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='gmp_history'").fetchone()
    finally:
        conn.close()


def test_history_rejects_naive_timestamp(tmp_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        gmp.record_readings({"k": [Sourced(1.0, "s", datetime(2026, 10, 15))]}, db_path=tmp_path / "s.sqlite")
