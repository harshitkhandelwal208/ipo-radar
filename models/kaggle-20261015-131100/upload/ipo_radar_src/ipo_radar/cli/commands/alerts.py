"""`ipo alerts ...`: phone setup, the one real test send, status, run-once, the watch loop and the systemd timer.

Only `ipo alerts test` sends a notification on purpose. The other commands send only what the alert rules
produce for real IPO events.
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.text import Text

from ipo_radar import config
from ipo_radar.alerts import daemon, notify
from ipo_radar.alerts.engine import DEFAULT_THRESHOLDS, IST
from ipo_radar.alerts.notify import Alert
from ipo_radar.cli import backend, render, service
from ipo_radar.cli.analysis import to_jsonable
from ipo_radar.cli.render import DIM, WARN
from ipo_radar.cli.ui import PlainOption, Ui, fail, ui_from
from ipo_radar.errors import SourceError
from ipo_radar.schema import LiveIPO

alerts_app = typer.Typer(help="Phone alerts: setup, test, status, run, watch and the systemd timer.",
                         no_args_is_help=True)


def _mask(value: str) -> str:
    """Last four characters of a chat id, which is an identifier rather than a credential. Tokens and
    topics are never printed, not even partly."""
    return f"…{value[-4:]}" if len(value) > 4 else "set"


def _chat_id(raw: str | None) -> str | int | None:
    if raw is None:
        return None
    return int(raw) if raw.lstrip("-").isdigit() else raw


def _alerts_cfg() -> dict:
    return config.load_config().get("alerts", {})


def _no_model(ipo: LiveIPO) -> None:
    """Stands in for the model when it is not trained: the alert rules then run without model lines."""
    return None


@alerts_app.command("setup")
def setup(
    ctx: typer.Context,
    plain: PlainOption = False,
    topic: str | None = typer.Option(None, "--topic", help="Use this ntfy topic instead of a generated one."),
    ntfy_server: str | None = typer.Option(None, "--ntfy-server", help="A self-hosted ntfy server (default ntfy.sh)."),
    telegram_token: str | None = typer.Option(None, "--telegram-token", envvar="IPO_TELEGRAM_TOKEN",
                                              help="Telegram bot token (optional channel)."),
    telegram_chat_id: str | None = typer.Option(None, "--telegram-chat-id",
                                                help="Telegram chat id for that bot (needed with the token)."),
) -> None:
    """Create the ntfy topic and print how to subscribe on the phone. Keeps an existing topic."""
    ui = ui_from(ctx, plain)
    if bool(telegram_token) != bool(telegram_chat_id):
        fail(ui, "Telegram needs both --telegram-token and --telegram-chat-id")
    if ntfy_server is not None and not ntfy_server.startswith(("https://", "http://")):
        fail(ui, "--ntfy-server must start with https:// (or http:// for a local server)")
    had_topic = bool(_alerts_cfg().get("ntfy_topic"))
    try:
        saved = backend.save_alerts(topic=topic, ntfy_server=ntfy_server, telegram_bot_token=telegram_token,
                                    telegram_chat_id=_chat_id(telegram_chat_id))
    except ValueError as exc:
        fail(ui, str(exc))
    cfg = _alerts_cfg()
    server = (cfg.get("ntfy_server") or notify.NTFY_SERVER).rstrip("/")
    url = f"{server}/{saved}"
    console = ui.console()
    kept = "kept the existing topic: your phone subscription still works" if had_topic and topic is None else "new topic"
    console.print(Text(f"Phone alerts are set up ({kept}).", style="" if ui.plain else "bold"))
    console.print(Text(f"  topic   {saved}"))
    console.print(Text(f"  config  {config.CONFIG_FILE} (readable by you only)", style=DIM), soft_wrap=True)
    if cfg.get("telegram_bot_token"):
        console.print(Text(f"  telegram saved (token set, chat {_mask(str(cfg.get('telegram_chat_id', '')))})",
                           style=DIM))
    console.print()
    console.print(Text("On your phone", style="bold"))
    console.print("  1. Install the free 'ntfy' app (Play Store, F-Droid or App Store). No account needed.")
    console.print("  2. Tap + and choose 'Subscribe to topic'.")
    console.print(f"  3. Server: {server} (leave the default). Topic: {saved}")
    console.print("  4. Back here, run: ipo alerts test")
    console.print()
    console.print(Text("Subscribe URL", style="bold"))
    console.print(render.secret_panel(url, ui))


@alerts_app.command("test")
def test(ctx: typer.Context, plain: PlainOption = False) -> None:
    """Send one clearly labelled test notification to every configured channel. The only deliberate send."""
    ui = ui_from(ctx, plain)
    channels = backend.notifiers()
    if not channels:
        fail(ui, "no notifier configured: run `ipo alerts setup` first")
    alert = Alert(
        key=f"test:{datetime.now(IST):%Y%m%d%H%M%S}",
        title="TEST from ipo-radar (not an IPO alert)",
        body="If you can read this, phone alerts work. No IPO data is involved.",
        priority=3, tags=["test_tube"],
    )
    results = backend.send_test(channels, alert)
    console = ui.console()
    for r in results:
        if r.status == "sent":
            console.print(Text(f"  {r.channel}: sent", style="" if ui.plain else "green"))
        else:
            console.print(Text(f"  {r.channel}: failed ({r.reason})", style="" if ui.plain else "red"))
    if not any(r.status == "sent" for r in results):
        fail(ui, "no channel delivered the test alert")


@alerts_app.command("status")
def status(
    ctx: typer.Context,
    plain: PlainOption = False,
    as_json: bool = typer.Option(False, "--json", help="Print the status as JSON (secrets masked)."),
) -> None:
    """Configured channels (secrets masked), quiet hours, the timer, and the last alerts sent."""
    ui = ui_from(ctx, plain)
    cfg = _alerts_cfg()
    channels: list[tuple[str, str]] = []
    if cfg.get("ntfy_topic"):
        channels.append(("ntfy", f"topic set, server {cfg.get('ntfy_server') or notify.NTFY_SERVER}"))
    if cfg.get("telegram_bot_token") and cfg.get("telegram_chat_id") is not None:
        channels.append(("telegram", f"bot token set, chat {_mask(str(cfg['telegram_chat_id']))}"))
    if not channels:
        channels.append(("none", "not configured: run `ipo alerts setup`"))
    thresholds = {**DEFAULT_THRESHOLDS, **(config.load_config().get("thresholds") or {})}
    quiet = f"{thresholds['quiet_hours']} IST (alerts go silent; closes and listings still ring)" \
        if thresholds["quiet_hours"] else "off"
    timer_file = backend.systemd_dir() / backend.TIMER_NAME
    timer = "installed" if timer_file.exists() else "not installed (ipo alerts install-timer)"
    sent = backend.recent_alerts(10)
    console = ui.console()
    if as_json:
        typer.echo(json.dumps(to_jsonable({"channels": channels, "quiet_hours": quiet, "timer": timer,
                                           "last_sent": sent}), ensure_ascii=False, indent=2))
        return
    console.print(render.alerts_status(channels, quiet, timer, sent, ui))


@alerts_app.command("run-once")
def run_once(ctx: typer.Context, plain: PlainOption = False) -> None:
    """One alert cycle: fetch, predict, evaluate, send what is new. What the systemd timer runs."""
    ui = ui_from(ctx, plain)
    view = service.build_view(refresh=True)
    if view.live.error and not view.live.ipos:
        fail(ui, f"no live IPO data: {view.live.error}")
    outcome = service.send_alerts(view)
    if outcome.status == "no_notifier":
        fail(ui, outcome.note)
    items = [] if outcome.run is None else [
        (a.title, [(r.channel, r.status, r.reason) for r in outcome.run.results if r.alert_key == a.key])
        for a in outcome.run.alerts]
    ui.console().print(render.alert_lines(outcome.status, outcome.note, items, ui))


@alerts_app.command("watch")
def watch(ctx: typer.Context, plain: PlainOption = False) -> None:
    """Run the alert loop in the foreground: every 5 min while an IPO is open, else every 30 min. Ctrl-C stops."""
    ui = ui_from(ctx, plain)
    if not backend.notifiers():
        fail(ui, "no notifier configured: run `ipo alerts setup` first")
    logging.getLogger().setLevel(logging.INFO)
    console = ui.console()

    def fetch_ipos():  # type: ignore[no-untyped-def]
        result = backend.load_live(refresh=True)
        if result.error and not result.ipos:
            raise SourceError(result.error)
        return result.ipos

    try:
        predict_fn = backend.predictor().predict
    except backend.ModelUnavailable as exc:
        console.print(Text(f"{exc}; alerts go out without model lines", style=WARN))
        predict_fn = _no_model
    console.print(Text("watching for IPO events. Ctrl-C to stop.", style=DIM))
    daemon.run_loop(fetch_ipos, predict_fn)


@alerts_app.command("install-timer")
def install_timer(
    ctx: typer.Context,
    plain: PlainOption = False,
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the unit files and change nothing."),
) -> None:
    """Write the systemd user units, reload systemd and enable the timer (every 15 minutes)."""
    ui = ui_from(ctx, plain)
    exe = Path(sys.prefix) / "bin" / "ipo"
    units = daemon.render_systemd_units(str(exe))
    dest = backend.systemd_dir()
    console = ui.console()
    if dry_run:
        for name, text in units.items():  # soft_wrap: paths and unit lines must stay copyable
            console.print(Text(f"# {dest / name}", style=DIM), soft_wrap=True)
            console.print(Text(text), soft_wrap=True)
        return
    if not exe.exists():
        fail(ui, f"the ipo script is not at {exe}; run this from the project venv")
    if not yes and not typer.confirm(f"Write {len(units)} unit files to {dest} and enable the timer?",
                                     default=False):
        fail(ui, "cancelled; nothing was changed")
    dest.mkdir(parents=True, exist_ok=True)
    for name, text in units.items():
        (dest / name).write_text(text, encoding="utf-8")
    console.print(Text(f"wrote {len(units)} unit files to {dest}"))
    if shutil.which("systemctl") is None:
        fail(ui, f"systemctl not found; units were written, enable them yourself with systemctl --user "
                 f"enable --now {backend.TIMER_NAME}")
    for args in (("daemon-reload",), ("enable", "--now", backend.TIMER_NAME)):
        proc = backend.systemctl(*args)
        if proc.returncode != 0:
            fail(ui, f"systemctl --user {' '.join(args)} failed: {proc.stderr.strip() or proc.returncode}")
    console.print(Text(f"timer enabled: {backend.TIMER_NAME} runs every 15 minutes. "
                       "Check with: systemctl --user list-timers"))


@alerts_app.command("uninstall-timer")
def uninstall_timer(
    ctx: typer.Context,
    plain: PlainOption = False,
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation."),
) -> None:
    """Disable the timer and remove its unit files."""
    ui = ui_from(ctx, plain)
    dest = backend.systemd_dir()
    present = [dest / name for name in daemon.render_systemd_units() if (dest / name).exists()]
    console = ui.console()
    if not present:
        console.print(Text("no timer units installed: nothing to remove"))
        return
    if not yes and not typer.confirm(f"Disable {backend.TIMER_NAME} and remove {len(present)} unit file(s)?",
                                     default=False):
        fail(ui, "cancelled; nothing was changed")
    if shutil.which("systemctl") is not None:
        proc = backend.systemctl("disable", "--now", backend.TIMER_NAME)
        if proc.returncode != 0:
            console.print(Text(f"systemctl disable reported: {proc.stderr.strip() or proc.returncode}", style=WARN))
    for path in present:
        path.unlink()
    if shutil.which("systemctl") is not None:
        backend.systemctl("daemon-reload")
    console.print(Text(f"removed {len(present)} unit file(s) from {dest}"))
