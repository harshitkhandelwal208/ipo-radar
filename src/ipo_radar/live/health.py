"""One cheap request per source, so `ipo doctor` can say which feed is down without a full fetch.

The cheapest real endpoints are used: NSE's current-issue JSON (after the cookie warm-up), Chittorgarh's
robots.txt, the GMP module's own fetch when it exists, and ntfy.sh's health route.
"""
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from datetime import datetime

from ipo_radar import config
from ipo_radar.errors import SourceError
from ipo_radar.live import _http, chittorgarh, nse

NTFY_HEALTH = "https://ntfy.sh/v1/health"


@dataclass
class SourceHealth:
    name: str
    ok: bool
    latency_ms: float | None
    detail: str
    last_success: datetime | None


def _probe(name: str, check) -> SourceHealth:
    start = time.monotonic()
    try:
        detail = check()
    except Exception as exc:  # a diagnostic must report a broken check, not die on it
        return SourceHealth(name, False, None, f"{type(exc).__name__}: {exc}", None)
    latency = round((time.monotonic() - start) * 1000, 1)
    return SourceHealth(name, True, latency, detail, datetime.now().astimezone())


def _check_nse() -> str:
    with _http.make_client() as client:
        nse._warm(client)
        body = _http.get(client, nse.BASE + nse.CURRENT, nse.SOURCE, headers=nse._api_headers())
    return f"ipo-current-issue returned {len(body)} bytes"


def _check_chittorgarh() -> str:
    with _http.make_client() as client:
        body = _http.get(client, chittorgarh.BASE + "/robots.txt", chittorgarh.SOURCE)
    return f"robots.txt returned {len(body)} bytes"


def _check_gmp() -> str:
    try:
        gmp = importlib.import_module("ipo_radar.live.gmp")
    except ImportError as exc:
        raise SourceError(f"GMP module not available ({exc})") from exc
    readings = gmp.fetch()  # the module's own fetch is the cheapest real call it offers
    return f"{len(readings)} IPO(s) with GMP readings"


def _check_ntfy() -> str:
    with _http.make_client(user_agent=config.USER_AGENT) as client:  # the project's own UA: an API, not a page
        body = _http.get(client, NTFY_HEALTH, "ntfy")
    return f"ntfy.sh health: {body.decode(errors='replace').strip()[:80]}"


def check_sources() -> list[SourceHealth]:
    """Probe every source once and report. Never raises: a failing source is a row with ok=False."""
    return [
        _probe("nse", _check_nse),
        _probe("chittorgarh", _check_chittorgarh),
        _probe("gmp", _check_gmp),
        _probe("ntfy", _check_ntfy),
    ]
