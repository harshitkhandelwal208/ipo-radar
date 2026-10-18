"""One alert cycle (fetch, predict, evaluate, dispatch, record) and a long-running loop around it.

An alert counts as delivered when at least one channel sent it. Only delivered alerts are marked sent, and an
IPO's snapshot is advanced only when all of its alerts were delivered. A failed alert is therefore retried on
the next run, because its condition still holds against the old snapshot.
"""
from __future__ import annotations

import logging
import signal
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from ipo_radar import config
from ipo_radar.alerts.engine import IST, evaluate, snapshot_of
from ipo_radar.alerts.notify import Alert, DispatchResult, Notifier, build_notifiers, dispatch
from ipo_radar.alerts.state import State
from ipo_radar.schema import LiveIPO, Prediction

log = logging.getLogger(__name__)

FAST_S = 300     # while any IPO is open
SLOW_S = 1800    # otherwise
PRUNE_DAYS = 60
SERVICE_NAME = "ipo-radar-alerts"


@dataclass
class RunResult:
    live: list[LiveIPO]
    alerts: list[Alert]
    results: list[DispatchResult] = field(default_factory=list)
    prediction_errors: list[str] = field(default_factory=list)
    dry_run: bool = False


def _predict_all(live: list[LiveIPO], predict_fn: Callable[[LiveIPO], Prediction]) -> tuple[dict[str, Prediction], list[str]]:
    """A failed prediction costs only the model line in the alert; the rest of the run continues."""
    preds: dict[str, Prediction] = {}
    errors: list[str] = []
    for ipo in live:
        try:
            preds[ipo.ipo_id] = predict_fn(ipo)
        except Exception as exc:
            errors.append(f"{ipo.ipo_id}: {type(exc).__name__}")
            log.warning("no model view for %s: %s", ipo.ipo_id, type(exc).__name__)
    return preds, errors


def run_once(
    fetch_fn: Callable[[], list[LiveIPO]],
    predict_fn: Callable[[LiveIPO], Prediction],
    dry_run: bool = False,
    *,
    notifiers: list[Notifier] | None = None,
    state: State | None = None,
    cfg: dict | None = None,
    now: datetime | None = None,
) -> RunResult:
    """Run one cycle. A dry run evaluates and returns the alerts but sends, marks and stores nothing."""
    cfg = config.load_config() if cfg is None else cfg
    live = fetch_fn()
    preds, pred_errors = _predict_all(live, predict_fn)

    own_state = state is None
    st = state if state is not None else State()
    try:
        alerts = evaluate(live, preds, st, now=now, cfg=cfg)
        if dry_run:
            return RunResult(live, alerts, [], pred_errors, dry_run=True)

        if notifiers is None:
            notifiers = build_notifiers(cfg)
        results = dispatch(alerts, notifiers)
        delivered = {r.alert_key for r in results if r.status == "sent"}
        undelivered = {a.key for a in alerts} - delivered

        for alert in alerts:
            if alert.key in delivered:
                st.mark_sent(alert)
        for ipo in live:
            if any(k.startswith(f"{ipo.ipo_id}:") for k in undelivered):
                continue  # keep the old snapshot so the change is detected again next run
            st.save_snapshot(ipo.ipo_id, snapshot_of(ipo))
        st.prune(PRUNE_DAYS)
        return RunResult(live, alerts, results, pred_errors)
    finally:
        if own_state:
            st.close()


def _is_open(ipo: LiveIPO, today: date) -> bool:
    if ipo.status == "open":
        return True
    return ipo.open_date is not None and ipo.close_date is not None and ipo.open_date <= today <= ipo.close_date


class _SignalStop:
    """Sets `stop` on SIGINT/SIGTERM and restores the previous handlers on exit. No-op off the main thread."""

    def __init__(self, stop: threading.Event) -> None:
        self._stop = stop
        self._previous: dict[int, object] = {}

    def __enter__(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._previous[sig] = signal.signal(sig, lambda *_: self._stop.set())

    def __exit__(self, *_: object) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)


def run_loop(
    fetch_fn: Callable[[], list[LiveIPO]],
    predict_fn: Callable[[LiveIPO], Prediction],
    *,
    fast_s: float = FAST_S,
    slow_s: float = SLOW_S,
    max_iterations: int | None = None,
    stop: threading.Event | None = None,
) -> int:
    """Poll forever: every `fast_s` while an IPO is open, otherwise every `slow_s`. Returns the iteration count.

    An exception in one iteration is logged and the loop continues after `fast_s`. SIGINT or SIGTERM ends
    the loop after the current iteration.
    """
    stop = stop or threading.Event()
    iterations = 0
    with _SignalStop(stop):
        while not stop.is_set():
            interval = fast_s
            try:
                result = run_once(fetch_fn, predict_fn)
                today = datetime.now(IST).date()
                interval = fast_s if any(_is_open(i, today) for i in result.live) else slow_s
                log.info("alert cycle: %d IPOs, %d alerts, %d deliveries", len(result.live),
                         len(result.alerts), len(result.results))
            except Exception:
                log.exception("alert cycle failed; retrying in %s s", fast_s)
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break
            stop.wait(interval)
    return iterations


def render_systemd_units(ipo_bin: str | None = None) -> dict[str, str]:
    """Text of the user service and timer, keyed by file name, for ~/.config/systemd/user/.

    `ipo_bin` must be an absolute path because systemd does not search PATH for ExecStart. It defaults to the
    `ipo` script of the interpreter running this code (the project venv).
    """
    exe = ipo_bin or str(Path(sys.prefix) / "bin" / "ipo")
    service = f"""[Unit]
Description=ipo-radar: check live IPO data and send phone alerts

[Service]
Type=oneshot
ExecStart={exe} alerts run-once
"""
    timer = f"""[Unit]
Description=Run ipo-radar alerts every 15 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=15min
Persistent=true
Unit={SERVICE_NAME}.service

[Install]
WantedBy=timers.target
"""
    return {f"{SERVICE_NAME}.service": service, f"{SERVICE_NAME}.timer": timer}
