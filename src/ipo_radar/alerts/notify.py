"""Phone notifications: ntfy.sh (free app, no account) and Telegram bot.

ntfy: the topic name is the only secret, so it is random (`new_ntfy_topic`). Headers are ASCII-only on the
wire, so a non-ASCII title is RFC 2047 encoded (ntfy decodes it); the body is UTF-8 and carries the rupee sign.
Telegram: the bot token sits in the request URL, so error reasons never include the URL or the exception text.
"""
from __future__ import annotations

import base64
import logging
import os
import re
import secrets
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

import httpx
import tomli_w

from ipo_radar import config
from ipo_radar.errors import SourceError

log = logging.getLogger(__name__)

NTFY_SERVER = "https://ntfy.sh"
TELEGRAM_API = "https://api.telegram.org"
ATTEMPTS = 3          # total tries per send, including the first
BACKOFF_S = 1.0       # waits 1 s, then 2 s between tries
TIMEOUT_S = 10.0
_TOPIC_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass
class Alert:
    key: str                 # stable event id; dedupe uses it, so the same event never alerts twice
    title: str               # line 1 on the phone: the news
    body: str                # lines 2-3: numbers, then source and age
    priority: int = 3        # ntfy scale 1 (silent) to 5 (urgent)
    tags: list[str] = field(default_factory=list)
    click_url: str | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.priority <= 5:
            raise ValueError(f"priority must be 1-5, got {self.priority}")


@dataclass(frozen=True)
class DispatchResult:
    alert_key: str
    channel: str
    status: str              # "sent" | "failed"
    reason: str = ""


class Notifier(Protocol):
    name: str

    def send(self, alert: Alert) -> None:
        """Deliver one alert. Raises SourceError when the channel fails for good."""


def new_ntfy_topic() -> str:
    """Unguessable topic. Anyone who knows the topic can read or post to it, so it is treated as a secret."""
    return "ipo-radar-" + secrets.token_urlsafe(16)


def _header_value(text: str) -> str:
    """ASCII passes through; anything else becomes an RFC 2047 encoded word that ntfy decodes."""
    text = " ".join(text.split())
    try:
        text.encode("ascii")
        return text
    except UnicodeEncodeError:
        return "=?UTF-8?B?" + base64.b64encode(text.encode("utf-8")).decode("ascii") + "?="


def _post_with_retry(send_once: Callable[[], httpx.Response], channel: str, sleep: Callable[[float], None]) -> None:
    """Retry timeouts, transport errors, 429 and 5xx with exponential back-off. Other 4xx fail at once."""
    last = ""
    for attempt in range(ATTEMPTS):
        try:
            resp = send_once()
        except httpx.TransportError as exc:  # includes timeouts; the class name is safe to show
            last = type(exc).__name__
        else:
            if resp.status_code < 400:
                return
            if resp.status_code < 500 and resp.status_code != 429:
                raise SourceError(f"{channel} rejected the alert: HTTP {resp.status_code}")
            last = f"HTTP {resp.status_code}"
        if attempt < ATTEMPTS - 1:
            sleep(BACKOFF_S * 2**attempt)
    raise SourceError(f"{channel} failed after {ATTEMPTS} tries: {last}")


class NtfyNotifier:
    name = "ntfy"

    def __init__(
        self,
        topic: str,
        server: str = NTFY_SERVER,
        token: str | None = None,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not _TOPIC_RE.match(topic):
            raise ValueError("ntfy topic may only contain letters, digits, '-' and '_'")
        self._url = f"{server.rstrip('/')}/{topic}"
        self._token = token
        self._client = client or httpx.Client(timeout=TIMEOUT_S, headers={"User-Agent": config.USER_AGENT})
        self._sleep = sleep

    def send(self, alert: Alert) -> None:
        headers = {"Title": _header_value(alert.title), "Priority": str(alert.priority)}
        if alert.tags:
            headers["Tags"] = ",".join(alert.tags)
        if alert.click_url:
            headers["Click"] = quote(alert.click_url, safe=":/?&=%#+@,;~!$*()[]'")
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        body = alert.body.encode("utf-8")
        _post_with_retry(
            lambda: self._client.post(self._url, content=body, headers=headers), self.name, self._sleep
        )


class TelegramNotifier:
    name = "telegram"

    def __init__(
        self,
        bot_token: str,
        chat_id: str | int,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._url = f"{TELEGRAM_API}/bot{bot_token}/sendMessage"
        self._chat_id = chat_id
        self._client = client or httpx.Client(timeout=TIMEOUT_S, headers={"User-Agent": config.USER_AGENT})
        self._sleep = sleep

    def send(self, alert: Alert) -> None:
        text = f"{alert.title}\n{alert.body}"
        if alert.click_url:
            text += f"\n{alert.click_url}"
        payload = {
            "chat_id": self._chat_id,
            "text": text[:4000],  # Telegram rejects messages over 4096 characters
            "disable_notification": alert.priority <= 1,
            "disable_web_page_preview": True,
        }
        _post_with_retry(lambda: self._client.post(self._url, json=payload), self.name, self._sleep)


def build_notifiers(config_dict: dict | None = None, *, client: httpx.Client | None = None) -> list[Notifier]:
    """Notifiers for every channel that has settings. Reads config.load_config()['alerts'] by default."""
    if config_dict is None:
        config_dict = config.load_config()
    alerts = config_dict.get("alerts", {})
    notifiers: list[Notifier] = []
    if alerts.get("ntfy_topic"):
        notifiers.append(
            NtfyNotifier(alerts["ntfy_topic"], alerts.get("ntfy_server") or NTFY_SERVER,
                         alerts.get("ntfy_token"), client=client)
        )
    if alerts.get("telegram_bot_token") and alerts.get("telegram_chat_id") is not None:
        notifiers.append(
            TelegramNotifier(alerts["telegram_bot_token"], alerts["telegram_chat_id"], client=client)
        )
    return notifiers


def dispatch(alerts: list[Alert], notifiers: list[Notifier]) -> list[DispatchResult]:
    """Send every alert to every channel. One failing channel is recorded and the rest still run."""
    results: list[DispatchResult] = []
    for alert in alerts:
        for notifier in notifiers:
            try:
                notifier.send(alert)
            except SourceError as exc:
                results.append(DispatchResult(alert.key, notifier.name, "failed", str(exc)))
                log.warning("alert %s not sent via %s: %s", alert.key, notifier.name, exc)
            except Exception as exc:  # a bug in one channel must not stop the others
                reason = f"unexpected {type(exc).__name__}"
                results.append(DispatchResult(alert.key, notifier.name, "failed", reason))
                log.warning("alert %s not sent via %s: %s", alert.key, notifier.name, reason)
            else:
                results.append(DispatchResult(alert.key, notifier.name, "sent"))
    return results


def save_alert_config(
    topic: str | None = None,
    *,
    ntfy_server: str | None = None,
    telegram_bot_token: str | None = None,
    telegram_chat_id: str | int | None = None,
    path: Path | None = None,
) -> str | None:
    """Merge alert settings into the user config file (mode 600). Returns the ntfy topic in effect.

    With no topic given, a new one is generated only if none is configured yet, so re-running setup
    never silently breaks the phone subscription. Tokens are written but never printed.
    """
    path = path or config.CONFIG_FILE
    data: dict = {}
    if path.exists():
        with path.open("rb") as f:
            data = tomllib.load(f)
    alerts = dict(data.get("alerts", {}))
    if topic is None and "ntfy_topic" not in alerts:
        topic = new_ntfy_topic()
    if topic is not None:
        if not _TOPIC_RE.match(topic):
            raise ValueError("ntfy topic may only contain letters, digits, '-' and '_'")
        alerts["ntfy_topic"] = topic
    if ntfy_server:
        alerts["ntfy_server"] = ntfy_server
    if telegram_bot_token:
        alerts["telegram_bot_token"] = telegram_bot_token
    if telegram_chat_id is not None:
        alerts["telegram_chat_id"] = telegram_chat_id
    data["alerts"] = alerts

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(tomli_w.dumps(data).encode("utf-8"))
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return alerts.get("ntfy_topic")
