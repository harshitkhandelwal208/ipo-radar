"""Notifier tests. Every request goes through httpx.MockTransport, so nothing reaches ntfy.sh or Telegram."""
from __future__ import annotations

import base64
import json
import os
import stat
import tomllib

import httpx
import pytest

from ipo_radar.alerts.notify import (
    Alert,
    NtfyNotifier,
    TelegramNotifier,
    build_notifiers,
    dispatch,
    new_ntfy_topic,
    save_alert_config,
)
from ipo_radar.errors import SourceError

NO_SLEEP = lambda _s: None  # noqa: E731  retry back-off is not under test here


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _alert(**kw) -> Alert:
    base = dict(key="ipo1:opens-today:2026-10-15", title="Acme opens today", body="band ₹95-100 · lot 150",
                priority=3, tags=["calendar", "rocket"], click_url="https://example.com/ipo1")
    base.update(kw)
    return Alert(**base)


class Recorder:
    name = "rec"

    def __init__(self) -> None:
        self.sent: list[Alert] = []

    def send(self, alert: Alert) -> None:
        self.sent.append(alert)


# ---- ntfy -----------------------------------------------------------------------------------------

def test_ntfy_request_headers_and_body_are_exact():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    notifier = NtfyNotifier("ipo-radar-abc_DEF-123", client=_client(handler), sleep=NO_SLEEP)
    notifier.send(_alert())

    req = seen[0]
    assert req.method == "POST"
    assert str(req.url) == "https://ntfy.sh/ipo-radar-abc_DEF-123"
    assert req.headers["Title"] == "Acme opens today"
    assert req.headers["Priority"] == "3"
    assert req.headers["Tags"] == "calendar,rocket"
    assert req.headers["Click"] == "https://example.com/ipo1"
    assert "Authorization" not in req.headers
    assert req.content == "band ₹95-100 · lot 150".encode("utf-8")


def test_ntfy_token_is_sent_as_bearer_and_optional_headers_are_omitted():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    notifier = NtfyNotifier("ipo-radar-x", token="tk_123", client=_client(handler), sleep=NO_SLEEP)
    notifier.send(_alert(tags=[], click_url=None))
    assert seen[0].headers["Authorization"] == "Bearer tk_123"
    assert "Tags" not in seen[0].headers
    assert "Click" not in seen[0].headers


def test_ntfy_non_ascii_title_is_rfc2047_encoded_and_does_not_crash():
    seen: list[httpx.Request] = []
    notifier = NtfyNotifier("ipo-radar-x", client=_client(lambda r: (seen.append(r), httpx.Response(200))[1]),
                            sleep=NO_SLEEP)
    title = "Ünïcode Ltd ₹ closes today"
    notifier.send(_alert(title=title, body="QIB 3.2x · GMP ₹25"))

    header = seen[0].headers["Title"]
    assert header.startswith("=?UTF-8?B?") and header.endswith("?=")
    encoded = header[len("=?UTF-8?B?"):-2]
    assert base64.b64decode(encoded).decode("utf-8") == title
    assert seen[0].content.decode("utf-8") == "QIB 3.2x · GMP ₹25"


def test_ntfy_ascii_title_is_sent_unchanged_and_whitespace_collapsed():
    seen: list[httpx.Request] = []
    notifier = NtfyNotifier("ipo-radar-x", client=_client(lambda r: (seen.append(r), httpx.Response(200))[1]),
                            sleep=NO_SLEEP)
    notifier.send(_alert(title="Acme\n  lists   tomorrow"))
    assert seen[0].headers["Title"] == "Acme lists tomorrow"


def test_ntfy_rejects_topic_with_unsafe_characters():
    with pytest.raises(ValueError):
        NtfyNotifier("bad/topic", client=_client(lambda r: httpx.Response(200)))


def test_ntfy_retries_5xx_with_backoff_then_succeeds():
    waits: list[float] = []
    responses = iter([503, 500, 200])
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(next(responses))

    NtfyNotifier("ipo-radar-x", client=_client(handler), sleep=waits.append).send(_alert())
    assert len(calls) == 3
    assert waits == [1.0, 2.0]


def test_ntfy_retries_429_and_gives_up_as_source_error():
    calls = []
    notifier = NtfyNotifier("ipo-radar-x",
                            client=_client(lambda r: (calls.append(r), httpx.Response(429))[1]), sleep=NO_SLEEP)
    with pytest.raises(SourceError, match="failed after 3 tries: HTTP 429"):
        notifier.send(_alert())
    assert len(calls) == 3


def test_ntfy_timeout_on_every_try_raises_source_error():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(SourceError, match="ReadTimeout"):
        NtfyNotifier("ipo-radar-x", client=_client(handler), sleep=NO_SLEEP).send(_alert())


def test_ntfy_client_error_is_not_retried():
    calls = []
    notifier = NtfyNotifier("ipo-radar-x",
                            client=_client(lambda r: (calls.append(r), httpx.Response(404))[1]), sleep=NO_SLEEP)
    with pytest.raises(SourceError, match="HTTP 404"):
        notifier.send(_alert())
    assert len(calls) == 1


def test_ntfy_connection_error_is_retried_then_raised():
    def handler(request):
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(SourceError, match="ConnectError"):
        NtfyNotifier("ipo-radar-x", client=_client(handler), sleep=NO_SLEEP).send(_alert())


def test_new_ntfy_topic_is_prefixed_and_unique():
    a, b = new_ntfy_topic(), new_ntfy_topic()
    assert a.startswith("ipo-radar-") and len(a) > len("ipo-radar-") + 16
    assert a != b


# ---- Telegram -------------------------------------------------------------------------------------

def test_telegram_sends_sendmessage_payload_and_silences_priority_1():
    seen: list[httpx.Request] = []
    tg = TelegramNotifier("123:SECRET", 42, client=_client(lambda r: (seen.append(r), httpx.Response(200))[1]),
                          sleep=NO_SLEEP)
    tg.send(_alert(priority=1, click_url=None))

    req = seen[0]
    assert str(req.url) == "https://api.telegram.org/bot123:SECRET/sendMessage"
    payload = json.loads(req.content)
    assert payload["chat_id"] == 42
    assert payload["text"] == "Acme opens today\nband ₹95-100 · lot 150"
    assert payload["disable_notification"] is True


def test_telegram_failure_reason_never_contains_the_token():
    def handler(request):
        raise httpx.ConnectError("could not reach https://api.telegram.org/bot123:SECRET/sendMessage",
                                 request=request)

    with pytest.raises(SourceError) as info:
        TelegramNotifier("123:SECRET", 42, client=_client(handler), sleep=NO_SLEEP).send(_alert())
    assert "SECRET" not in str(info.value)
    assert "ConnectError" in str(info.value)


def test_telegram_http_error_is_reported_without_url():
    tg = TelegramNotifier("123:SECRET", 42, client=_client(lambda r: httpx.Response(401)), sleep=NO_SLEEP)
    with pytest.raises(SourceError) as info:
        tg.send(_alert())
    assert "HTTP 401" in str(info.value)
    assert "SECRET" not in str(info.value)


# ---- build and dispatch ---------------------------------------------------------------------------

def test_build_notifiers_from_config_dict():
    client = _client(lambda r: httpx.Response(200))
    cfg = {"alerts": {"ntfy_topic": "ipo-radar-t", "telegram_bot_token": "1:x", "telegram_chat_id": "9"}}
    names = [n.name for n in build_notifiers(cfg, client=client)]
    assert names == ["ntfy", "telegram"]


def test_build_notifiers_reads_the_user_config_by_default(monkeypatch):
    from ipo_radar import config

    monkeypatch.setattr(config, "load_config", lambda: {"alerts": {"ntfy_topic": "ipo-radar-t"}})
    assert [n.name for n in build_notifiers(client=_client(lambda r: httpx.Response(200)))] == ["ntfy"]


def test_alert_priority_outside_one_to_five_is_rejected():
    with pytest.raises(ValueError, match="priority"):
        _alert(priority=6)


def test_build_notifiers_with_no_settings_returns_nothing():
    assert build_notifiers({}, client=_client(lambda r: httpx.Response(200))) == []
    assert build_notifiers({"alerts": {"telegram_bot_token": "1:x"}}, client=_client(lambda r: httpx.Response(200))) == []


def test_one_failing_channel_does_not_block_the_other():
    calls = []

    def ntfy_down(request):
        calls.append(request.url.host)
        if request.url.host == "ntfy.sh":
            return httpx.Response(500)
        return httpx.Response(200)

    client = _client(ntfy_down)
    notifiers = [
        NtfyNotifier("ipo-radar-t", client=client, sleep=NO_SLEEP),
        TelegramNotifier("1:x", 9, client=client, sleep=NO_SLEEP),
    ]
    results = dispatch([_alert()], notifiers)

    by_channel = {r.channel: r for r in results}
    assert by_channel["ntfy"].status == "failed"
    assert by_channel["ntfy"].reason == "ntfy failed after 3 tries: HTTP 500"
    assert by_channel["telegram"].status == "sent"
    assert by_channel["telegram"].reason == ""
    assert calls.count("api.telegram.org") == 1


def test_dispatch_records_unexpected_errors_and_continues():
    rec = Recorder()

    class Bug:
        name = "bug"

        def send(self, alert):
            raise RuntimeError("secret detail that must not leak")

    results = dispatch([_alert()], [Bug(), rec])
    assert [(r.channel, r.status, r.reason) for r in results] == [
        ("bug", "failed", "unexpected RuntimeError"),
        ("rec", "sent", ""),
    ]
    assert len(rec.sent) == 1


def test_dispatch_with_no_notifiers_returns_empty():
    assert dispatch([_alert()], []) == []


def test_dispatch_sends_every_alert_to_every_channel():
    a, b = Recorder(), Recorder()
    b.name = "other"
    results = dispatch([_alert(key="k1"), _alert(key="k2")], [a, b])
    assert len(a.sent) == 2 and len(b.sent) == 2
    assert [r.status for r in results] == ["sent"] * 4


# ---- config writing -------------------------------------------------------------------------------

def test_save_alert_config_creates_secure_file_and_generates_topic(tmp_path):
    path = tmp_path / "config.toml"
    topic = save_alert_config(path=path)
    assert topic and topic.startswith("ipo-radar-")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert tomllib.loads(path.read_text())["alerts"]["ntfy_topic"] == topic


def test_save_alert_config_merges_and_keeps_topic_and_other_sections(tmp_path, capsys):
    path = tmp_path / "config.toml"
    path.write_text('[thresholds]\nquiet_hours = "22:00-06:00"\n\n[alerts]\nntfy_topic = "ipo-radar-keep"\n')

    topic = save_alert_config(path=path, telegram_bot_token="111:TOKEN", telegram_chat_id="77")

    assert topic == "ipo-radar-keep"
    data = tomllib.loads(path.read_text())
    assert data["thresholds"]["quiet_hours"] == "22:00-06:00"
    assert data["alerts"] == {"ntfy_topic": "ipo-radar-keep", "telegram_bot_token": "111:TOKEN",
                              "telegram_chat_id": "77"}
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert "TOKEN" not in capsys.readouterr().out


def test_save_alert_config_stores_a_custom_ntfy_server(tmp_path):
    path = tmp_path / "config.toml"
    save_alert_config("ipo-radar-s", ntfy_server="https://ntfy.example.org", path=path)
    assert tomllib.loads(path.read_text())["alerts"]["ntfy_server"] == "https://ntfy.example.org"


def test_save_alert_config_rejects_bad_topic(tmp_path):
    with pytest.raises(ValueError):
        save_alert_config("has space", path=tmp_path / "c.toml")
    assert not (tmp_path / "c.toml").exists()
