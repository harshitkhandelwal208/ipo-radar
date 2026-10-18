"""Daemon tests: one cycle (dry run and real), failure handling, the loop and the systemd unit text. Offline."""
from __future__ import annotations

import os
import signal
import threading
from datetime import date, datetime, timedelta

import pytest

from ipo_radar.alerts import daemon
from ipo_radar.alerts.daemon import render_systemd_units, run_loop, run_once
from ipo_radar.alerts.engine import IST
from ipo_radar.alerts.notify import Alert
from ipo_radar.alerts.state import State
from ipo_radar.errors import SourceError
from ipo_radar.schema import LiveIPO, Prediction, Sourced

NOW = datetime(2026, 10, 15, 11, 0, tzinfo=IST)
TODAY = date(2026, 10, 15)


class Recorder:
    name = "rec"

    def __init__(self) -> None:
        self.sent: list[Alert] = []

    def send(self, alert: Alert) -> None:
        self.sent.append(alert)


class AlwaysDown:
    name = "down"

    def __init__(self) -> None:
        self.calls = 0

    def send(self, alert: Alert) -> None:
        self.calls += 1
        raise SourceError("down: HTTP 503")


def _ipo(**kw) -> LiveIPO:
    base = dict(ipo_id="ipo1", market="IN_MAIN", company="Acme Ltd", status="open", open_date=TODAY,
                sub={"sub_total": Sourced(3.0, "nse", NOW - timedelta(minutes=2))})
    base.update(kw)
    return LiveIPO(**base)


def _pred(ipo: LiveIPO) -> Prediction:
    return Prediction(ipo_id=ipo.ipo_id, as_of="pre_open", open_gain_pct_p10=0.0, open_gain_pct_p50=5.0,
                      open_gain_pct_p90=10.0, p_positive_open=0.6, close_gain_pct_p50=None, drivers=[],
                      baseline_gmp_pct=None, model_version="t", reliable=True, caveats=[])


def _fetch(live: list[LiveIPO]):
    return lambda: live


@pytest.fixture(autouse=True)
def no_real_side_effects(monkeypatch, tmp_path):
    """run_loop calls run_once with defaults. Stub the channels and the state DB so no test can reach
    ntfy.sh, Telegram or the user's real state file, whatever is configured on this machine."""
    monkeypatch.setattr(daemon, "build_notifiers", lambda *a, **k: [])
    monkeypatch.setattr(daemon, "State", lambda *a, **k: State(tmp_path / "loop-state.sqlite"))


@pytest.fixture
def state(tmp_path):
    with State(tmp_path / "state.sqlite") as st:
        yield st


def test_dry_run_returns_alerts_and_touches_nothing(state):
    rec = Recorder()
    result = run_once(_fetch([_ipo()]), _pred, dry_run=True, notifiers=[rec], state=state, cfg={}, now=NOW)

    assert [a.key for a in result.alerts] == ["ipo1:opens-today:2026-10-15"]
    assert result.dry_run is True
    assert rec.sent == []
    assert state.already_sent("ipo1:opens-today:2026-10-15") is False
    assert state.last_snapshot("ipo1") is None


def test_real_run_delivers_marks_and_saves_snapshot_then_second_run_is_quiet(state):
    rec = Recorder()
    first = run_once(_fetch([_ipo()]), _pred, notifiers=[rec], state=state, cfg={}, now=NOW)
    assert [a.key for a in rec.sent] == ["ipo1:opens-today:2026-10-15"]
    assert [r.status for r in first.results] == ["sent"]
    assert state.already_sent("ipo1:opens-today:2026-10-15") is True
    assert state.last_snapshot("ipo1") == {"status": "open", "gmp": None, "sub": {"sub_total": 3.0}}

    second = run_once(_fetch([_ipo()]), _pred, notifiers=[rec], state=state, cfg={}, now=NOW)
    assert second.alerts == []
    assert len(rec.sent) == 1


def test_undelivered_alert_is_retried_and_keeps_the_old_snapshot(state):
    down = AlwaysDown()
    ipo = _ipo(open_date=None, sub={"sub_total": Sourced(12.0, "nse", NOW)})
    state.save_snapshot("ipo1", {"status": "open", "gmp": None, "sub": {"sub_total": 3.0}})

    first = run_once(_fetch([ipo]), _pred, notifiers=[down], state=state, cfg={}, now=NOW)
    assert [a.key for a in first.alerts] == ["ipo1:milestone:sub_total:10x"]
    assert first.results[0].status == "failed"
    assert state.already_sent("ipo1:milestone:sub_total:10x") is False
    assert state.last_snapshot("ipo1")["sub"]["sub_total"] == 3.0  # not advanced, so the crossing repeats

    rec = Recorder()
    retry = run_once(_fetch([ipo]), _pred, notifiers=[rec], state=state, cfg={}, now=NOW)
    assert [a.key for a in rec.sent] == ["ipo1:milestone:sub_total:10x"]
    assert retry.results[0].status == "sent"
    assert state.last_snapshot("ipo1")["sub"]["sub_total"] == 12.0


def test_one_dead_channel_does_not_block_the_live_one(state):
    down, rec = AlwaysDown(), Recorder()
    result = run_once(_fetch([_ipo()]), _pred, notifiers=[down, rec], state=state, cfg={}, now=NOW)
    status = {r.channel: r.status for r in result.results}
    assert status == {"down": "failed", "rec": "sent"}
    assert state.already_sent("ipo1:opens-today:2026-10-15") is True


def test_prediction_failure_drops_only_the_model_line(state):
    def broken(ipo):
        raise RuntimeError("model file missing")

    rec = Recorder()
    result = run_once(_fetch([_ipo()]), broken, notifiers=[rec], state=state, cfg={}, now=NOW)
    assert result.prediction_errors == ["ipo1: RuntimeError"]
    assert "model" not in rec.sent[0].body
    assert len(rec.sent) == 1


def test_fetch_failure_propagates_to_the_caller(state):
    def fetch():
        raise SourceError("nse unavailable")

    with pytest.raises(SourceError):
        run_once(fetch, _pred, notifiers=[], state=state, cfg={}, now=NOW)


def test_no_channels_means_alerts_stay_pending_and_snapshot_is_not_advanced(state):
    result = run_once(_fetch([_ipo()]), _pred, notifiers=[], state=state, cfg={}, now=NOW)
    assert len(result.alerts) == 1
    assert result.results == []
    assert state.already_sent("ipo1:opens-today:2026-10-15") is False
    assert state.last_snapshot("ipo1") is None


def test_run_loop_polls_fast_while_open_and_slow_otherwise():
    waits: list[float] = []

    class Waiter(threading.Event):
        def wait(self, timeout=None):
            waits.append(timeout)
            return False

    today = datetime.now(IST).date()
    open_now = [_ipo(open_date=today - timedelta(days=1), close_date=today + timedelta(days=1))]
    closed = [_ipo(open_date=today - timedelta(days=30), close_date=today - timedelta(days=28), status="listed")]
    stop_fast = Waiter()
    # the loop does not wait after its last iteration, so three iterations show two waits
    run_loop(_fetch(open_now), _pred, max_iterations=3, stop=stop_fast)
    assert waits == [300, 300]

    waits.clear()
    run_loop(_fetch(closed), _pred, max_iterations=3, stop=Waiter())
    assert waits == [1800, 1800]


def test_run_loop_survives_an_exception_in_one_iteration(caplog):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise SourceError("nse down")
        return []

    waits: list[float] = []

    class Waiter(threading.Event):
        def wait(self, timeout=None):
            waits.append(timeout)
            return False

    iterations = run_loop(flaky, _pred, fast_s=7, slow_s=9, max_iterations=3, stop=Waiter())
    assert iterations == 3
    assert calls["n"] == 3
    assert waits == [7, 9]  # after the failure it waits the fast interval; with no IPOs it waits slow
    assert "alert cycle failed" in caplog.text


def test_run_loop_stops_cleanly_on_sigint_and_restores_the_handler():
    previous = signal.getsignal(signal.SIGINT)
    seen = {"n": 0}

    def fetch():
        seen["n"] += 1
        os.kill(os.getpid(), signal.SIGINT)  # the loop must stop, not raise KeyboardInterrupt
        return []

    iterations = run_loop(fetch, _pred, fast_s=0, slow_s=0, max_iterations=10)
    assert iterations == 1
    assert seen["n"] == 1
    assert signal.getsignal(signal.SIGINT) is previous


def test_run_loop_works_off_the_main_thread_without_touching_signals():
    """Signal handlers can only be installed on the main thread, so the loop must run there without them."""
    done: dict[str, int] = {}

    def target():
        done["n"] = run_loop(_fetch([]), _pred, max_iterations=1, stop=threading.Event())

    worker = threading.Thread(target=target)
    worker.start()
    worker.join(timeout=10)
    assert done == {"n": 1}


def test_render_systemd_units_use_an_absolute_exec_and_fifteen_minute_timer():
    units = render_systemd_units("/opt/ipo/bin/ipo")
    service = units["ipo-radar-alerts.service"]
    timer = units["ipo-radar-alerts.timer"]
    assert "Type=oneshot" in service
    assert "ExecStart=/opt/ipo/bin/ipo alerts run-once" in service
    assert "OnUnitActiveSec=15min" in timer
    assert "Unit=ipo-radar-alerts.service" in timer
    assert "WantedBy=timers.target" in timer


def test_render_systemd_units_defaults_to_the_venv_script():
    exec_line = [ln for ln in render_systemd_units()["ipo-radar-alerts.service"].splitlines()
                 if ln.startswith("ExecStart=")][0]
    path = exec_line[len("ExecStart="):].split(" alerts")[0]
    assert os.path.isabs(path) and path.endswith("/bin/ipo")


def test_module_exposes_the_interval_defaults():
    assert (daemon.FAST_S, daemon.SLOW_S) == (300, 1800)
