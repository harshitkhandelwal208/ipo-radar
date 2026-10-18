"""GMP zero versus dash: the real saved markup for four sites, and the policy that decides what a 0 means.

Evidence (see the gmp.py docstring): investorgain prints '--' with '0 ↓ / 0 ↑' for no quote; ipopremium and
ipowatch print 0 for the same IPOs at the same times. A 0 contradicted by another site's dash is not a reading.
"""
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import pytest

from ipo_radar.live import _http, gmp
from ipo_radar.live.gmp import GmpReading, _open_day, _resolve_zeros
from ipo_radar.schema import Sourced

pytestmark = pytest.mark.usefixtures("live_isolation")
FIX = Path(__file__).parent / "fixtures" / "gmp"
NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)


class FakeSession:
    """Stands in for gmp._Session: serves one fixture per site and never touches the network."""

    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.now = NOW

    def get(self, site: str, url: str, ext: str) -> str:
        return self.pages[site]


def _pages() -> dict[str, str]:
    return {
        "investorgain": (FIX / "investorgain_zero_vs_dash.html").read_text(encoding="utf8"),
        "ipopremium": (FIX / "ipopremium_zero_vs_dash.html").read_text(encoding="utf8"),
        "ipowatch": (FIX / "ipowatch_zero_vs_dash.html").read_text(encoding="utf8"),
        "ipocentral": (FIX / "ipocentral_zero_vs_dash.json").read_text(encoding="utf8"),
    }


def _site_readings() -> dict[str, dict[str, GmpReading]]:
    session = FakeSession(_pages())
    return {
        "investorgain": gmp._fetch_investorgain(session),
        "ipopremium": gmp._fetch_ipopremium(session),
        "ipowatch": gmp._fetch_ipowatch(session),
        "ipocentral": gmp._fetch_ipocentral(session),
    }


def _combined() -> dict[str, list[Sourced]]:
    combined: dict[str, list[Sourced]] = {}
    for readings in _site_readings().values():
        for key, reading in readings.items():
            combined.setdefault(key, []).append(reading)
    return combined


# ---- what each site's markup says, before any policy ------------------------------------------------
def test_investorgain_dash_is_unknown_and_its_range_says_no_quote():
    shellz = _site_readings()["investorgain"]["shellz"]
    assert shellz.value is None
    assert "0 ↓ / 0 ↑" in shellz.note
    assert (shellz.open_date, shellz.market) == (date(2026, 10, 15), "IN_SME")
    tonbo = _site_readings()["investorgain"]["tonbo imaging"]
    assert tonbo.value == 500.0 and tonbo.market == "IN_MAIN"


def test_ipopremium_zero_is_read_as_zero_and_its_type_column_gives_the_market():
    shellz = _site_readings()["ipopremium"]["shellz"]
    assert shellz.value == 0.0
    assert (shellz.open_date, shellz.market) == (date(2026, 10, 15), "IN_SME")
    assert _site_readings()["ipopremium"]["tonbo imaging"].market == "IN_MAIN"


def test_ipowatch_zero_and_its_open_day_but_no_market():
    shellz = _site_readings()["ipowatch"]["shellz"]
    assert shellz.value == 0.0 and shellz.open_date == date(2026, 10, 15) and shellz.market == ""


def test_ipocentral_sections_give_the_market():
    tonbo = _site_readings()["ipocentral"]["tonbo imaging"]
    assert tonbo.value == 510.0 and tonbo.market == "IN_MAIN" and tonbo.open_date == date(2026, 10, 15)


@pytest.mark.parametrize("text, expected", [
    ("15-Oct", date(2026, 10, 15)),
    ("15-Oct GMP: 500", date(2026, 10, 15)),
    ("15 - 19 Oct", date(2026, 10, 15)),
    ("21-23 Oct", date(2026, 10, 21)),
    ("Oct 15, 2026", date(2026, 10, 15)),
    ("not a date", None),
    ("", None),
])
def test_open_day_forms(text, expected):
    assert _open_day(text, 2026) == expected


# ---- the policy ---------------------------------------------------------------------------------------
def test_a_zero_that_another_site_shows_as_a_dash_is_not_a_reading():
    resolved = _combined()
    _resolve_zeros(resolved)
    by_site = {r.source: r for r in resolved["shellz"]}
    assert by_site["investorgain"].value is None
    assert by_site["ipopremium"].value is None
    assert by_site["ipowatch"].value is None
    assert "site shows 0 where another shows no data (investorgain)" in by_site["ipopremium"].note


def test_real_values_are_not_touched_by_the_policy():
    resolved = _combined()
    _resolve_zeros(resolved)
    assert {r.value for r in resolved["tonbo imaging"]} == {500.0, 510.0}


def test_a_zero_no_other_site_contradicts_stays_zero():
    lone = [GmpReading(0.0, "ipopremium", NOW, "x")]
    _resolve_zeros({"sjp": lone})
    assert lone[0].value == 0.0 and lone[0].note == "x"


def test_zeros_confirmed_by_other_zeros_with_no_dash_stay_zero():
    items = [GmpReading(0.0, "ipopremium", NOW), GmpReading(0.0, "ipowatch", NOW), GmpReading(40.0, "a", NOW)]
    _resolve_zeros({"k": items})
    assert [r.value for r in items] == [0.0, 0.0, 40.0]


def test_one_dash_drops_zeros_from_every_other_site():
    items = [GmpReading(None, "investorgain", NOW), GmpReading(0.0, "ipowatch", NOW),
             GmpReading(0.0, "ipopremium", NOW)]
    _resolve_zeros({"k": items})
    assert [r.value for r in items] == [None, None, None]


# ---- end to end through fetch(), with the four pages behind a mock transport ----------------------------
def test_fetch_applies_the_policy_and_keeps_raw_copies(tmp_path):
    pages = _pages()
    routes = {"investorgain.com": pages["investorgain"], "ipowatch.in": pages["ipowatch"],
              "ipopremium.in": pages["ipopremium"], "ipocentral.in": pages["ipocentral"]}

    def handler(request: httpx.Request) -> httpx.Response:
        for host, body in routes.items():
            if host in request.url.host:
                return httpx.Response(200, text=body)
        return httpx.Response(404)

    client = _http.make_client(transport=httpx.MockTransport(handler))
    out = gmp.fetch(client=client, raw_dir=tmp_path, now=NOW)
    shellz = {r.source: r.value for r in out["shellz"]}
    assert shellz["investorgain"] is None and shellz["ipopremium"] is None and shellz["ipowatch"] is None
    assert {r.value for r in out["tonbo imaging"]} == {500.0, 510.0}
    assert all(isinstance(r, Sourced) for items in out.values() for r in items)
    assert list((tmp_path / "ipopremium").iterdir())  # the raw page was kept
