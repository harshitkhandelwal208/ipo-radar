"""Polite HTTP for the live sources: one request per second per host, raw copies on disk, and every
failure (timeout, non-200, empty body) raised as SourceError so callers never see a half answer.

Sites block the default python-httpx User-Agent, so each source passes a browser-like UA. The project
UA (config.USER_AGENT) is still used by health checks against the ntfy.sh API.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import httpx

from ipo_radar import config
from ipo_radar.errors import SourceError

BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
MIN_INTERVAL_S = 1.0  # per host; tests set this to 0
TIMEOUT_S = 20.0

class HTTPStatusError(SourceError):
    """A non-200 answer. Carries the status so callers can decide to retry (403 may clear after a cookie)."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


_lock = threading.Lock()
_last_request: dict[str, float] = {}


def make_client(transport: httpx.BaseTransport | None = None, user_agent: str = BROWSER_UA) -> httpx.Client:
    """A client with a cookie jar (NSE needs the home-page cookie) and redirects followed."""
    return httpx.Client(
        headers={"User-Agent": user_agent, "Accept-Language": "en-US,en;q=0.9"},
        timeout=TIMEOUT_S, follow_redirects=True, transport=transport,
    )


def _throttle(host: str) -> None:
    with _lock:
        now = time.monotonic()
        wait = _last_request.get(host, 0.0) + MIN_INTERVAL_S - now
        _last_request[host] = now + max(wait, 0.0)
    if wait > 0:
        time.sleep(wait)


def get(client: httpx.Client, url: str, source: str, *, headers: dict[str, str] | None = None) -> bytes:
    """GET with throttling. Returns the body; raises SourceError on any failure."""
    _throttle(httpx.URL(url).host)
    try:
        resp = client.get(url, headers=headers)
    except httpx.TimeoutException as exc:
        raise SourceError(f"{source}: timeout fetching {url}") from exc
    except httpx.HTTPError as exc:
        raise SourceError(f"{source}: network error fetching {url}: {exc}") from exc
    if resp.status_code != 200:
        hint = " (rate limited)" if resp.status_code == 429 else " (blocked?)" if resp.status_code == 403 else ""
        raise HTTPStatusError(f"{source}: HTTP {resp.status_code} for {url}{hint}", resp.status_code)
    if not resp.content.strip():
        raise SourceError(f"{source}: empty body for {url}")
    return resp.content


def save_raw(source: str, name: str, body: bytes) -> Path:
    """Keep the latest raw copy per name under data/raw/<source>/ (overwritten, so the folder stays bounded)."""
    folder = config.RAW / source
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(body)
    return path
