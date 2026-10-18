"""check_sources with mocked transports: each source reports ok or a clear reason, and nothing raises."""
import sys
import types

import httpx
import pytest

from conftest import fixture_bytes
from ipo_radar.live import _http, health

pytestmark = pytest.mark.usefixtures("live_isolation")


def _world(monkeypatch, overrides=None):
    """Routes each host to a canned answer. overrides maps a host to a status code or 'timeout'."""
    overrides = overrides or {}

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if overrides.get(host) == "timeout":
            raise httpx.ReadTimeout("slow", request=request)
        if host in overrides:
            return httpx.Response(overrides[host], text="denied")
        if host == "www.nseindia.com":
            if request.url.path.startswith("/market-data/"):
                return httpx.Response(200, text="<html></html>")
            return httpx.Response(200, content=fixture_bytes("nse_ipo_current_issue.json"))
        if host == "www.chittorgarh.com":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if host == "ntfy.sh":
            return httpx.Response(200, text='{"healthy":true}')
        return httpx.Response(404)

    real = _http.make_client
    monkeypatch.setattr(_http, "make_client",
                        lambda transport=None, user_agent=_http.BROWSER_UA: real(
                            transport=httpx.MockTransport(handler), user_agent=user_agent))
    fake_gmp = types.ModuleType("ipo_radar.live.gmp")
    fake_gmp.fetch = lambda: {"demo": []}
    monkeypatch.setitem(sys.modules, "ipo_radar.live.gmp", fake_gmp)


def _by_name(rows):
    return {r.name: r for r in rows}


def test_all_sources_ok(monkeypatch):
    _world(monkeypatch)
    rows = _by_name(health.check_sources())
    assert set(rows) == {"nse", "chittorgarh", "gmp", "ntfy"}
    assert all(r.ok for r in rows.values())
    assert all(r.last_success is not None and r.latency_ms is not None for r in rows.values())
    assert rows["ntfy"].detail.startswith("ntfy.sh health")


def test_blocked_nse_is_reported_not_raised(monkeypatch):
    _world(monkeypatch, {"www.nseindia.com": 403})
    nse = _by_name(health.check_sources())["nse"]
    assert nse.ok is False and "HTTP 403" in nse.detail and nse.last_success is None


def test_chittorgarh_timeout_is_reported(monkeypatch):
    _world(monkeypatch, {"www.chittorgarh.com": "timeout"})
    row = _by_name(health.check_sources())["chittorgarh"]
    assert row.ok is False and "timeout" in row.detail


def test_ntfy_server_error_is_reported(monkeypatch):
    _world(monkeypatch, {"ntfy.sh": 500})
    row = _by_name(health.check_sources())["ntfy"]
    assert row.ok is False and "HTTP 500" in row.detail


def test_missing_gmp_module_is_reported(monkeypatch):
    _world(monkeypatch)
    monkeypatch.setitem(sys.modules, "ipo_radar.live.gmp", None)
    row = _by_name(health.check_sources())["gmp"]
    assert row.ok is False and "GMP module not available" in row.detail


def test_unexpected_check_error_never_escapes(monkeypatch):
    _world(monkeypatch)

    def broken():
        raise RuntimeError("bug")
    row = health._probe("x", broken)
    assert row.ok is False and row.detail == "RuntimeError: bug"
