"""Every `ipo` command through typer's CliRunner, with the backend replaced by fakes. No network, no real state:
config, state, cache, history and systemd paths all point into tmp_path, and notifiers are recorders."""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import types
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from cli_helpers import (Broken, Recorder, fake_ipos, fake_predict, history_frame, no_model_predict,
                         no_skill_predict, now, tonbo)
from ipo_radar.cli.model_rules import WORSE_REASON
from ipo_radar import config
from ipo_radar.alerts import daemon
from ipo_radar.cli import backend
from ipo_radar.cli.app import app
from ipo_radar.live import health as live_health
from ipo_radar.schema import LiveIPO

runner = CliRunner()


class Env:
    """What the fake backend reports. Tests change these attributes before invoking a command."""

    def __init__(self) -> None:
        self.ipos: list[LiveIPO] = fake_ipos()
        self.error: str | None = None
        self.model_error: str | None = None
        self.predict = fake_predict
        self.history: pd.DataFrame | None = None
        self.channels: list = []
        self.sources: list[live_health.SourceHealth] = [
            live_health.SourceHealth(n, True, 12.0, f"{n} ok", None) for n in ("nse", "chittorgarh", "gmp", "ntfy")]
        self.refresh_calls: list[bool] = []
        self.unused: dict[str, str] = {}   # stage -> why the model is not the headline there


@pytest.fixture
def env(tmp_path, monkeypatch) -> Env:
    e = Env()
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "cfg" / "config.toml")
    monkeypatch.setattr(config, "STATE_DB", tmp_path / "cfg" / "state.sqlite")
    monkeypatch.setattr(config, "LIVE_CACHE", tmp_path / "cfg" / "cache")
    monkeypatch.setattr(config, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(config, "IPOS_PARQUET", tmp_path / "processed" / "ipos.parquet")
    monkeypatch.setattr(config, "INDICES_PARQUET", tmp_path / "processed" / "indices.parquet")
    monkeypatch.setattr(config, "MODELS", tmp_path / "models")
    monkeypatch.setattr(backend, "systemd_dir", lambda: tmp_path / "units")

    def load_live(refresh: bool) -> backend.LiveResult:
        e.refresh_calls.append(refresh)
        if e.error:
            return backend.LiveResult([], [], e.error)
        return backend.LiveResult(list(e.ipos), backend.health_from(e.ipos))

    def predictor() -> backend.Predictor:
        if e.model_error:
            raise backend.ModelUnavailable(e.model_error)
        return backend.Predictor(e.predict, "test-1", "2026-10-15T07:00:31+00:00", 4933, dict(e.unused))

    def load_history() -> pd.DataFrame:
        if e.history is None:
            raise backend.HistoryMissing("no file")
        return e.history

    monkeypatch.setattr(backend, "load_live", load_live)
    monkeypatch.setattr(backend, "predictor", predictor)
    monkeypatch.setattr(backend, "load_history", load_history)
    monkeypatch.setattr(backend, "notifiers", lambda: list(e.channels))
    monkeypatch.setattr(backend, "check_sources", lambda: list(e.sources))
    monkeypatch.setattr(backend, "model_files", lambda: [])
    return e


def run(args: list[str], *, columns: int = 80, env_vars: dict[str, str] | None = None):
    vars_ = {"COLUMNS": str(columns), "NO_COLOR": "", **(env_vars or {})}
    result = runner.invoke(app, args, env=vars_, catch_exceptions=True)
    assert result.exception is None or isinstance(result.exception, SystemExit), repr(result.exception)
    return result


# ---- fetch -------------------------------------------------------------------------------------------
def test_fetch_happy_path_shows_dashboard_footer_and_no_notifier_note(env):
    env.model_error = backend.NOT_TRAINED
    r = run(["--plain", "fetch"])
    assert r.exit_code == 0, r.output
    assert "Tonbo Imaging" in r.output
    assert "model not trained yet: run `ipo train`" in r.output
    assert "sources  nse ok · chittorgarh ok · gmp ok" in r.output
    assert "newest reading" in r.output
    assert "no notifier configured: run `ipo alerts setup`" in r.output


def test_fetch_refreshes_from_the_network(env):
    run(["fetch", "--no-alerts"])
    assert env.refresh_calls == [True]
    run(["fetch", "--cached", "--no-alerts"])
    assert env.refresh_calls == [True, False]


def test_fetch_sends_new_alerts_to_configured_channel(env):
    rec = Recorder()
    env.channels = [rec]
    r = run(["--plain", "fetch"])
    assert r.exit_code == 0, r.output
    titles = [a.title for a in rec.sent]
    assert "Zetwerk Manufacturing Businesses Limited opens in 2 days" in titles
    assert "Last chance: HD Fire Protect Limited closes today" in titles
    assert "recorder sent" in r.output


def test_fetch_no_alerts_never_touches_notifiers(env):
    rec = Recorder()
    env.channels = [rec]
    r = run(["--plain", "fetch", "--no-alerts"])
    assert r.exit_code == 0
    assert rec.sent == []
    assert "skipped (--no-alerts)" in r.output


def test_fetch_reports_a_failing_channel_without_crashing(env):
    env.channels = [Broken()]
    r = run(["--plain", "fetch"])
    assert r.exit_code == 0, r.output
    assert "broken failed: broken: HTTP 503" in r.output


def test_fetch_all_sources_down_exits_1_with_the_reason(env):
    env.error = "no IPO data: nse: HTTP 503; chittorgarh: timed out"
    r = run(["fetch"])
    assert r.exit_code == 1
    assert "no live IPO data: no IPO data: nse: HTTP 503; chittorgarh: timed out" in r.output


def test_fetch_json_all_sources_down_is_still_valid_json(env):
    env.error = "no IPO data: nse: down; chittorgarh: down"
    r = run(["fetch", "--json"])
    assert r.exit_code == 1
    assert json.loads(r.output)["error"].startswith("no live IPO data")


def test_fetch_shows_a_down_source_and_dashes_its_numbers(env):
    env.ipos[0].warnings = ["nse unavailable: HTTP 503"]
    env.ipos[0].sub.clear()
    r = run(["--plain", "fetch"])
    assert r.exit_code == 0
    assert "nse down (HTTP 503)" in r.output


def test_fetch_json_parses_and_keeps_missing_as_null(env):
    env.model_error = None
    r = run(["fetch", "--json"])
    data = json.loads(r.output)
    assert data["model"] == {"message": None, "version": "test-1"}
    zetwerk = next(i for i in data["ipos"] if i["company"].startswith("Zetwerk"))
    assert zetwerk["issue"]["min_retail"] is not None
    assert zetwerk["gmp"]["median"] is None          # no reading: null, never 0
    assert all(s["now"] is None for s in zetwerk["subs"])
    assert data["alerts"]["status"] == "no_notifier"


def test_fetch_with_model_shows_estimates(env):
    r = run(["--plain", "fetch", "--no-alerts"])
    assert "+12%" in r.output
    assert "model not trained yet" not in r.output


def test_fetch_no_predict_skips_the_model(env):
    r = run(["--plain", "fetch", "--no-alerts", "--no-predict"])
    assert r.exit_code == 0
    assert "predictions skipped (--no-predict)" in r.output


def test_a_model_that_crashes_on_one_ipo_is_reported_not_fatal(env):
    def predict(ipo):
        if ipo.ipo_id == "zetwerk":
            raise RuntimeError("bad feature")
        return fake_predict(ipo)

    env.predict = predict
    r = run(["--plain", "fetch", "--no-alerts"])
    assert r.exit_code == 0
    assert "could not predict 1 IPO(s)" in r.output  # the line itself wraps at 80 columns; the count does not


# ---- live -------------------------------------------------------------------------------------------
def test_live_reads_the_cache_not_the_network(env):
    r = run(["--plain", "live"])
    assert r.exit_code == 0
    assert env.refresh_calls == [False]
    assert "Tonbo Imaging" in r.output


def test_live_json(env):
    data = json.loads(run(["live", "--json"]).output)
    assert {i["status"] for i in data["ipos"]} >= {"open", "upcoming"}


# ---- show -------------------------------------------------------------------------------------------
def test_show_has_every_section_and_the_advice_line(env):
    r = run(["--plain", "show", "tonbo"])
    assert r.exit_code == 0, r.output
    for section in ("Issue", "Subscription", "GMP (unofficial grey market)", "Model estimate",
                    "Risk flags", "Summary (estimate)"):
        assert section in r.output
    assert "min retail ₹14,700" in r.output
    assert "not investment advice" in r.output
    assert "OFS 100%" in r.output and "issue day 1 of 5" in r.output


def test_show_accepts_symbol_and_partial_name(env):
    assert "Tonbo Imaging" in run(["--plain", "show", "TONBO"]).output
    assert "Zetwerk" in run(["--plain", "show", "zet"]).output


def test_show_ambiguous_name_asks_which_one(env):
    env.ipos.append(LiveIPO(ipo_id="tonbo labs", market="IN_MAIN", company="Tonbo Labs", symbol="TLABS",
                            status="open"))
    r = run(["show", "tonb"])
    assert r.exit_code == 1
    assert "matches 2 IPOs. Did you mean" in r.output
    assert '"Tonbo Labs" (TLABS)' in r.output


def test_show_unknown_name_suggests_the_nearest(env):
    r = run(["show", "zetwrk"])
    assert r.exit_code == 1
    assert "no live IPO matches 'zetwrk'" in r.output
    assert "Closest:" in r.output and "Zetwerk" in r.output


def test_show_missing_history_is_a_note_not_a_failure(env):
    r = run(["--plain", "show", "tonbo"])
    assert r.exit_code == 0
    assert "no history yet: percentiles and the GMP comparison need `ipo data build`" in r.output


def test_show_with_history_shows_percentiles_and_gmp_context(env):
    env.history = history_frame()
    r = run(["--plain", "show", "tonbo"])
    assert r.exit_code == 0
    assert "vs past: p73 means" in r.output
    assert "p0" in r.output  # the Total figure is below every past main-board total


def test_show_json_is_valid_and_has_no_secret_fields(env):
    data = json.loads(run(["show", "tonbo", "--json"]).output)
    assert data["company"] == "Tonbo Imaging India Limited"
    assert data["model"]["p50"] == 12.0
    assert "token" not in json.dumps(data).lower()


# ---- history ----------------------------------------------------------------------------------------
def test_history_missing_is_not_available_exit_2(env):
    r = run(["history"])
    assert r.exit_code == 2
    assert "not available yet" in r.output and "ipo data build" in r.output


def test_history_filters_sorts_and_limits(env):
    env.history = history_frame()
    sme = json.loads(run(["history", "--market", "in_sme", "--json"]).output)
    assert len(sme) == 2 and {r["market"] for r in sme} == {"IN_SME"}
    top = json.loads(run(["history", "--sort", "gain", "--limit", "2", "--json"]).output)
    assert len(top) == 2 and top[0]["list_open_gain_pct"] >= top[1]["list_open_gain_pct"]
    found = run(["--plain", "history", "--search", "past main 3"])
    assert found.exit_code == 0 and "Past Main 3" in found.output


def test_history_bad_since_is_an_error(env):
    env.history = history_frame()
    r = run(["history", "--since", "last-week"])
    assert r.exit_code == 1 and "YYYY-MM-DD" in r.output


def test_history_table_is_readable_at_80_and_160(env):
    env.history = history_frame()
    for width in (80, 160):
        r = run(["--plain", "history", "--limit", "5"], columns=width)
        assert r.exit_code == 0
        assert max(len(line) for line in r.output.splitlines()) <= width


# ---- train / eval / research / data build -----------------------------------------------------------
@pytest.mark.parametrize("module", ["ipo_radar.model.train", "ipo_radar.model.evaluate",
                                    "ipo_radar.research", "ipo_radar.data.build"])
def test_unbuilt_modules_are_not_available_exit_2(env, monkeypatch, module):
    monkeypatch.setitem(sys.modules, module, None)  # simulates the module not existing yet
    args = {"ipo_radar.model.train": ["train"], "ipo_radar.model.evaluate": ["eval"],
            "ipo_radar.research": ["research"], "ipo_radar.data.build": ["data", "build"]}[module]
    env.history = history_frame()
    r = run(args)
    assert r.exit_code == 2
    assert "not available yet" in r.output


def _fake_train_module(monkeypatch, calls: list) -> None:
    """A stand-in for ipo_radar.model.train.train that records its keyword arguments."""
    def train(df, **kwargs):
        calls.append((len(df), kwargs))
        return types.SimpleNamespace(
            version="v9", trained_at="2026-10-15T07:00:00+00:00", n_rows=len(df), quick=kwargs.get("quick", False),
            model_path="/scratch/bundle.joblib", metrics_path="/scratch/train_report.json",
            stages={"pre_open": {"trained": True}, "close": {"trained": False}}, warnings=["close: 3 rows"])

    fake = types.ModuleType("ipo_radar.model.train")
    fake.train = train  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ipo_radar.model.train", fake)


def test_train_runs_the_model_and_summarises_it(env, monkeypatch):
    calls: list = []
    _fake_train_module(monkeypatch, calls)
    env.history = history_frame()
    r = run(["--plain", "train"])
    assert r.exit_code == 0, r.output
    assert "trained bundle v9 on 8 IPOs, at 2026-10-15" in r.output
    assert "pre_open     trained" in r.output and "close        not trained" in r.output
    assert "! close: 3 rows" in r.output
    assert calls == [(8, {"quick": False})]


def test_train_quick_and_out_dir_are_passed_through(env, monkeypatch, tmp_path):
    calls: list = []
    _fake_train_module(monkeypatch, calls)
    env.history = history_frame()
    r = run(["train", "--quick", "--out-dir", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "(quick mode)" in r.output
    assert calls == [(8, {"quick": True, "out_dir": tmp_path})]


def test_train_kaggle_passes_quick_to_the_kaggle_run(env, monkeypatch):
    seen = {}
    fake = types.ModuleType("ipo_radar.model.kaggle")

    def run_train(quick=False, **kwargs):
        seen["quick"] = quick
        return Path("/scratch/kaggle-run")

    fake.run_train = run_train  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ipo_radar.model.kaggle", fake)
    r = run(["train", "--where", "kaggle", "--quick"])
    assert r.exit_code == 0, r.output
    assert seen == {"quick": True} and "kaggle run folder: /scratch/kaggle-run" in r.output


def test_eval_prints_the_report_path_and_verdicts(env, monkeypatch):
    seen = {}
    fake = types.ModuleType("ipo_radar.model.evaluate")

    def evaluate(df=None, quick=False, **kwargs):
        seen["quick"] = quick
        return types.SimpleNamespace(report_path="docs/MODEL_REPORT.md", test_years=[2025, 2026], n_test_rows=521,
                                     data_hash="07cf78bd", verdicts={"pre_open": "no better than the median"})

    fake.evaluate = evaluate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ipo_radar.model.evaluate", fake)
    env.history = history_frame()
    r = run(["--plain", "eval", "--quick"])
    assert r.exit_code == 0, r.output
    assert "wrote docs/MODEL_REPORT.md" in r.output and "2025, 2026" in r.output
    assert "pre_open     no better than the median" in r.output
    assert seen == {"quick": True}


def test_train_without_history_is_not_available(env, monkeypatch):
    fake = types.ModuleType("ipo_radar.model.train")
    fake.train = lambda df: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ipo_radar.model.train", fake)
    r = run(["train"])
    assert r.exit_code == 2 and "ipo data build" in r.output


def test_train_kaggle_exits_2_while_the_kaggle_module_is_missing(env, monkeypatch):
    monkeypatch.setitem(sys.modules, "ipo_radar.model.kaggle", None)  # simulates the module not existing
    r = run(["train", "--where", "kaggle"])
    assert r.exit_code == 2 and "use --where local" in r.output


def test_data_build_reports_the_written_path(env, monkeypatch):
    fake = types.ModuleType("ipo_radar.data.build")
    fake.build_all = lambda: Path("/x/ipos.parquet")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ipo_radar.data.build", fake)
    r = run(["data", "build"])
    assert r.exit_code == 0 and "built /x/ipos.parquet" in r.output


# ---- alerts -----------------------------------------------------------------------------------------
def test_alerts_setup_creates_then_keeps_the_topic(env):
    r1 = run(["--plain", "alerts", "setup"])
    assert r1.exit_code == 0, r1.output
    assert "Subscribe URL" in r1.output and "https://ntfy.sh/ipo-radar-" in r1.output
    topic = config.load_config()["alerts"]["ntfy_topic"]
    r2 = run(["--plain", "alerts", "setup"])
    assert config.load_config()["alerts"]["ntfy_topic"] == topic
    assert "kept the existing topic" in r2.output
    assert oct(config.CONFIG_FILE.stat().st_mode & 0o777) == "0o600"


def test_alerts_setup_telegram_needs_both_values(env):
    r = run(["alerts", "setup", "--telegram-token", "123:abc"])
    assert r.exit_code == 1 and "needs both" in r.output
    assert not config.CONFIG_FILE.exists()


def test_alerts_setup_rejects_a_bad_server_url(env):
    r = run(["alerts", "setup", "--ntfy-server", "ntfy.example"])
    assert r.exit_code == 1 and "must start with" in r.output


def test_alerts_test_sends_one_labelled_alert(env):
    rec = Recorder()
    env.channels = [rec]
    r = run(["--plain", "alerts", "test"])
    assert r.exit_code == 0, r.output
    assert len(rec.sent) == 1 and rec.sent[0].title.startswith("TEST from ipo-radar")
    assert "recorder: sent" in r.output


def test_alerts_test_without_a_channel_sends_nothing(env):
    r = run(["alerts", "test"])
    assert r.exit_code == 1 and "run `ipo alerts setup` first" in r.output


def test_alerts_test_when_every_channel_fails_exits_1(env):
    env.channels = [Broken()]
    r = run(["alerts", "test"])
    assert r.exit_code == 1 and "no channel delivered" in r.output


def test_alerts_status_never_prints_secrets(env):
    config.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.CONFIG_FILE.write_text('[alerts]\nntfy_topic = "ipo-radar-TOPICSECRET9"\n'
                                  'telegram_bot_token = "123456:BOTSECRETVALUE"\ntelegram_chat_id = 98765\n')
    for args in (["alerts", "status"], ["alerts", "status", "--json"]):
        r = run(args)
        assert r.exit_code == 0, r.output
        assert "TOPICSECRET" not in r.output and "BOTSECRET" not in r.output
    assert "telegram" in run(["--plain", "alerts", "status"]).output


def test_alerts_status_without_channels_says_how_to_fix(env):
    r = run(["--plain", "alerts", "status"])
    assert "not configured: run `ipo alerts setup`" in r.output
    assert "Last alerts sent" in r.output


def test_alerts_run_once_needs_a_channel(env):
    assert run(["alerts", "run-once"]).exit_code == 1
    rec = Recorder()
    env.channels = [rec]
    r = run(["--plain", "alerts", "run-once"])
    assert r.exit_code == 0, r.output
    assert rec.sent  # the opens-soon and closes-today alerts for the fake data


def test_alerts_watch_needs_a_channel_then_runs_the_loop(env, monkeypatch):
    assert run(["alerts", "watch"]).exit_code == 1
    seen = {}

    def fake_loop(fetch_fn, predict_fn, **kwargs):
        seen["ipos"] = fetch_fn()
        seen["pred"] = predict_fn(seen["ipos"][0])
        return 1

    monkeypatch.setattr(daemon, "run_loop", fake_loop)
    env.channels = [Recorder()]
    r = run(["--plain", "alerts", "watch"])
    assert r.exit_code == 0, r.output
    assert len(seen["ipos"]) == len(fake_ipos())


def test_install_timer_dry_run_writes_nothing(env, tmp_path):
    r = run(["--plain", "alerts", "install-timer", "--dry-run"])
    assert r.exit_code == 0
    assert "ExecStart=" in r.output and "ipo-radar-alerts.timer" in r.output
    assert not (tmp_path / "units").exists()


def test_install_timer_writes_units_and_enables_timer(env, tmp_path, monkeypatch):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "ipo").write_text("#!/bin/sh\n")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(backend, "systemctl",
                        lambda *a: calls.append(a) or types.SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemctl")
    r = run(["--plain", "alerts", "install-timer", "--yes"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "units" / "ipo-radar-alerts.timer").exists()
    assert calls == [("daemon-reload",), ("enable", "--now", "ipo-radar-alerts.timer")]


def test_install_timer_without_systemctl_says_how_to_finish(env, tmp_path, monkeypatch):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "ipo").write_text("#!/bin/sh\n")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda name: None)
    r = run(["alerts", "install-timer", "--yes"])
    assert r.exit_code == 1 and "systemctl not found" in r.output
    assert (tmp_path / "units" / "ipo-radar-alerts.service").exists()


def test_install_timer_confirmation_declined_changes_nothing(env, tmp_path, monkeypatch):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "ipo").write_text("#!/bin/sh\n")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    r = runner.invoke(app, ["alerts", "install-timer"], input="n\n", env={"COLUMNS": "80"})
    assert r.exit_code == 1 and "nothing was changed" in r.output
    assert not (tmp_path / "units").exists()


def test_uninstall_timer_removes_units(env, tmp_path, monkeypatch):
    units = tmp_path / "units"
    units.mkdir()
    (units / "ipo-radar-alerts.timer").write_text("x")
    (units / "ipo-radar-alerts.service").write_text("x")
    monkeypatch.setattr(backend, "systemctl", lambda *a: types.SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemctl")
    r = run(["alerts", "uninstall-timer", "--yes"])
    assert r.exit_code == 0 and "removed 2 unit file(s)" in r.output
    assert not (units / "ipo-radar-alerts.timer").exists()


# ---- doctor -----------------------------------------------------------------------------------------
def _write_data_files(root: Path, newest_open: date | None = None) -> None:
    """Write the three data files. The history rows are dated 2025, unless `newest_open` moves them."""
    root.mkdir(parents=True, exist_ok=True)
    frame = history_frame()
    if newest_open is not None:
        frame["open_date"] = pd.Timestamp(newest_open)
    frame.to_parquet(root / "ipos.parquet")
    pd.DataFrame({"date": [], "symbol": [], "close": []}).to_parquet(root / "indices.parquet")
    (root / "COVERAGE.md").write_text("# coverage\n")


def test_doctor_all_green_exits_0(env, monkeypatch):
    _write_data_files(config.PROCESSED)
    env.channels = [Recorder()]
    r = run(["--plain", "doctor"])
    assert r.exit_code == 0, r.output
    assert "all required checks pass" in r.output
    assert "history table" in r.output and "rows" in r.output


def test_doctor_shows_model_version_training_date_and_rows(env):
    _write_data_files(config.PROCESSED, newest_open=now().date() - timedelta(days=3))
    env.channels = [Recorder()]
    r = run(["--plain", "doctor"], columns=160)
    assert "version test-1; trained 2026-10-15; 4933 IPOs in training" in r.output
    assert "opened" in r.output and "3 days ago" in r.output


def test_doctor_says_how_old_the_newest_ipo_row_is_and_warns_when_stale(env):
    _write_data_files(config.PROCESSED)  # the fake history is dated 2025
    env.channels = [Recorder()]
    r = run(["--plain", "doctor"], columns=160)
    assert r.exit_code == 0  # stale data is a warning, not a failure
    assert "newest IPO row" in r.output and "warn" in r.output
    assert "refresh the raw downloads" in r.output


def test_doctor_reports_each_failure_with_a_fix(env):
    env.model_error = "model not trained yet: run `ipo train`"
    env.sources = [live_health.SourceHealth("nse", False, None, "HTTP 503", None)]
    r = run(["--plain", "doctor"])
    assert r.exit_code == 1
    assert "fix model: run `ipo train`" in r.output
    assert "fix history table: run `ipo data build`" in r.output
    assert "fix phone alerts: run `ipo alerts setup`" in r.output
    assert "HTTP 503" in r.output


def test_doctor_json_is_parseable(env):
    env.model_error = "model not trained yet: run `ipo train`"
    data = json.loads(run(["doctor", "--json"]).output)
    assert data["ok"] is False
    assert any(c["name"] == "model" and c["result"] == "fail" for c in data["checks"])


def test_doctor_a_crashing_check_is_reported(env, monkeypatch):
    def boom():
        raise RuntimeError("probe broke")

    monkeypatch.setattr(backend, "check_sources", boom)
    r = run(["doctor"])
    assert r.exit_code == 1 and "check crashed: RuntimeError: probe broke" in r.output


# ---- output modes and widths --------------------------------------------------------------------------
def test_plain_output_has_no_escape_codes_and_colour_mode_does(env):
    plain = run(["--plain", "show", "tonbo"], env_vars={"FORCE_COLOR": "1"})
    assert "\x1b[" not in plain.output
    coloured = run(["show", "tonbo"], env_vars={"FORCE_COLOR": "1"})
    assert "\x1b[" in coloured.output


@pytest.mark.parametrize("columns", [80, 160])
def test_views_fit_the_width_and_never_raise(env, columns):
    env.history = history_frame()
    for args in (["--plain", "fetch", "--no-alerts"], ["--plain", "show", "tonbo"], ["--plain", "live"]):
        r = run(args, columns=columns)
        assert r.exit_code == 0, (args, r.output)
        widest = max(len(line) for line in r.output.splitlines())
        assert widest <= columns, (args, widest)


def test_help_lists_every_command(env):
    r = run(["--help"])
    assert r.exit_code == 0
    for name in ("fetch", "live", "show", "history", "train", "eval", "research", "doctor", "alerts", "data"):
        assert name in r.output
    assert "--plain" in r.output


def test_is_the_console_script_target(env):
    from ipo_radar.cli import app as app_module

    assert app_module.app is app
    assert dataclasses.is_dataclass(backend.LiveResult)
    assert os.path.exists(config.ROOT / "pyproject.toml")


# ---- model skill marker, warnings shown once, global flag after the command ----------------------------
NO_SKILL_FOOTER = ("~ no demonstrated skill over the historical median at this stage (see docs/MODEL_REPORT.md); "
                   "the grey-market premium is often the better guide")


def test_dashboard_marks_a_p50_without_demonstrated_skill(env):
    env.predict = no_skill_predict
    r = run(["--plain", "fetch", "--no-alerts"], columns=200)  # the footer is one long line: wraps below 200
    assert r.exit_code == 0, r.output
    assert "~ +12%" in r.output
    assert NO_SKILL_FOOTER in r.output


def test_dashboard_marks_the_same_way_in_colour_mode(env):
    env.predict = no_skill_predict
    r = run(["live"], columns=200, env_vars={"FORCE_COLOR": "1"})
    assert "~ " in r.output and NO_SKILL_FOOTER in r.output


def test_dashboard_has_no_marker_when_the_model_has_skill(env):
    r = run(["--plain", "live"], columns=200)
    assert "+12%" in r.output
    assert "~" not in r.output
    assert NO_SKILL_FOOTER not in r.output


def test_skill_marker_comes_from_a_caveat_even_when_reliable(env):
    def caveated(ipo):
        p = fake_predict(ipo)
        p.caveats = ["no demonstrated skill over the historical median at this stage"]
        return p

    env.predict = caveated
    r = run(["--plain", "live"], columns=200)
    assert "~ +12%" in r.output and NO_SKILL_FOOTER in r.output


def test_subscription_warning_appears_once_in_show(env):
    line = "subscription sources differ: QIB 0.4x (nse) vs 0.5x (chittorgarh) and 3 more categories"
    env.ipos[0].warnings = [line]
    r = run(["--plain", "show", "tonbo"], columns=160)
    assert r.exit_code == 0
    assert r.output.count(line) == 1


def test_gmp_conflict_appears_once_in_show(env):
    r = run(["--plain", "show", "HD Fire"], columns=160)
    assert r.exit_code == 0
    assert r.output.count("GMP sources disagree: 90 (investorgain) vs 150 (ipowatch)") == 1


def test_plain_is_accepted_after_the_command(env):
    coloured = run(["show", "tonbo"], env_vars={"FORCE_COLOR": "1"})
    assert "\x1b[" in coloured.output
    after = run(["show", "tonbo", "--plain"], env_vars={"FORCE_COLOR": "1"})
    assert after.exit_code == 0
    assert "\x1b[" not in after.output
    for args in (["fetch", "--plain", "--no-alerts"], ["live", "--plain"], ["history", "--plain"],
                 ["alerts", "status", "--plain"], ["doctor", "--plain"]):
        assert "\x1b[" not in run(args, env_vars={"FORCE_COLOR": "1"}).output, args


# ---- median in place of the model where the model is worse than the historical median ----------------
MEDIAN_FOOTER = ("median (model not used here) = the segment's training median of the open gain, "
                 "shown where the model is worse than the historical median at that stage")


def test_dashboard_shows_the_median_where_the_model_is_not_used(env):
    env.predict = no_model_predict   # predict() put the segment median in the P50 and added the NO_MODEL caveat
    r = run(["--plain", "live"], columns=200)
    assert r.exit_code == 0, r.output
    assert "~ +32% median" in r.output
    assert "+12%" not in r.output
    assert MEDIAN_FOOTER in r.output


def test_show_labels_the_median_and_keeps_the_model_range_as_reference(env):
    env.predict = no_model_predict
    env.unused = {"pre_open": WORSE_REASON}
    r = run(["--plain", "show", "tonbo"], columns=160)
    assert r.exit_code == 0, r.output
    assert "~ +32%  median (model not used here)" in r.output
    assert "the model is worse than the historical median at this stage on the latest blocks" in r.output
    assert "model, reference only: P10 -4%   P50 +12%   P90 +38%" in r.output
    assert "not used for the headline" in r.output


def test_show_uses_the_model_p50_when_the_model_is_used(env):
    r = run(["--plain", "show", "tonbo"], columns=160)
    assert "P10 -4%   P50 +12%   P90 +38%" in r.output
    assert "median (model not used here)" not in r.output


def test_backend_reads_the_bundle_and_reports_unused_stages(monkeypatch):
    """The real backend.predictor() with a fake model package: its bundle's recent flags decide the stages."""
    bundle = types.SimpleNamespace(
        version="v-test", trained_at="2026-10-18T07:00:00+00:00", n_rows=5000,
        recent={"pre_open": {"worse": True}, "day1": {"worse": False}}, shrink={"day1": 0.0})
    fake = types.ModuleType("ipo_radar.model.predict")
    fake.load_model = lambda: bundle  # type: ignore[attr-defined]
    fake.predict = lambda ipo, as_of=None: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ipo_radar.model.predict", fake)
    pred = backend.predictor()
    assert pred.version == "v-test" and pred.n_rows == 5000
    assert set(pred.not_used) == {"pre_open"}   # the blend weight is the model's; only the 'worse' flag is read


def test_show_bounds_a_saturated_probability_instead_of_reading_certainty(env):
    def certain(ipo):
        p = fake_predict(ipo)
        p.p_positive_open = 0.97
        return p

    env.predict = certain
    r = run(["--plain", "show", "tonbo"], columns=160)
    assert "P(opens above issue) >=95%" in r.output
    assert "P(opens above issue) 100%" not in r.output
    assert "chance of opening above the issue price" in r.output
