"""Operations: train, eval, research and data build wrap modules that other parts of the project write, and
doctor checks the whole installation.

Those modules are imported lazily. If one is missing the command says 'not available yet' and exits 2, so the
CLI still works before the model or the data builder exists.
"""
from __future__ import annotations

import importlib
import importlib.metadata as metadata
import json
import stat
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import typer
from rich.text import Text

from ipo_radar import config
from ipo_radar.alerts.engine import IST
from ipo_radar.cli import backend, fmt, render
from ipo_radar.cli.analysis import to_jsonable
from ipo_radar.cli.render import DIM, WARN, Check
from ipo_radar.cli.ui import EXIT_OK, PlainOption, Ui, fail, not_available, ui_from

data_app = typer.Typer(help="Build and inspect the data files.", no_args_is_help=True)

_DEPENDENCIES = {  # import-free check: distribution name -> what it is for
    "typer": "CLI", "rich": "terminal output", "httpx": "live sources", "selectolax": "HTML parsing",
    "pandas": "history table", "pyarrow": "parquet files", "numpy": "numbers", "openpyxl": "xlsx sources",
    "platformdirs": "config paths", "tomli-w": "config writing",
}


class Where(str, Enum):
    local = "local"
    kaggle = "kaggle"


def _function_or_exit(ui: Ui, module: str, name: str, what: str, hint: str = "") -> Callable[..., Any]:
    """The named function from a module another part of the project writes. Missing -> exit 2 with a clear
    message; a module that exists but needs an uninstalled package -> exit 1 naming the package."""
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name is not None and module.startswith(exc.name):
            not_available(ui, f"{what} is not written yet" + (f"; {hint}" if hint else ""))
        fail(ui, f"{what} needs a package that is not installed: {exc.name}")
    except ImportError as exc:
        fail(ui, f"{what} failed to import: {exc}")
    fn = getattr(mod, name, None)
    if fn is None:
        not_available(ui, f"{what} has no {name}() yet")
    return fn


def _history_frame(ui: Ui):  # type: ignore[no-untyped-def]
    try:
        return backend.load_history()
    except backend.HistoryMissing:
        not_available(ui, "the history table is not built yet: run `ipo data build` first")


@data_app.command("build")
def data_build(ctx: typer.Context, plain: PlainOption = False) -> None:
    """Build data/processed/ipos.parquet, indices.parquet and COVERAGE.md from the raw downloads."""
    ui = ui_from(ctx, plain)
    build_all = _function_or_exit(ui, "ipo_radar.data.build", "build_all", "the data builder")
    try:
        path = build_all()
    except Exception as exc:
        fail(ui, f"build failed: {type(exc).__name__}: {exc}")
    ui.console().print(Text(f"built {path}"))


def train(
    ctx: typer.Context,
    plain: PlainOption = False,
    where: Where = typer.Option(
        Where.local, "--where", case_sensitive=False,
        help="local (default, this machine) or kaggle: a free Kaggle CPU kernel runs the large hyper-parameter "
             "search and multi-seed fit. The table is small, so this buys search breadth, not speed. Needs "
             "KAGGLE_API_TOKEN or `kaggle auth login`."),
    quick: bool = typer.Option(False, "--quick", help="Fewer trees and one seed: a fast sanity check."),
    out_dir: Path | None = typer.Option(None, "--out-dir",
                                        help="Save the local bundle here (default models/). Use a scratch folder "
                                             "for experiments, so the real bundle is not replaced."),
) -> None:
    """Train the listing-gain model on the history table. Local is the default and needs nothing else.

    --where kaggle is optional: it uploads the table as a private Kaggle dataset, runs the search on a Kaggle
    CPU kernel, and replaces models/bundle.joblib only after the downloaded bundle passes its checks."""
    ui = ui_from(ctx, plain)
    if where is Where.kaggle:
        run_kaggle = _function_or_exit(ui, "ipo_radar.model.kaggle", "run_train", "Kaggle training",
                                       "use --where local for now")
        try:
            run_dir = run_kaggle(quick=quick)
        except Exception as exc:
            fail(ui, f"Kaggle training failed: {exc}")
        ui.console().print(Text(f"kaggle run folder: {run_dir}"))
        return
    run_train = _function_or_exit(ui, "ipo_radar.model.train", "train", "model training")
    df = _history_frame(ui)
    kwargs: dict[str, Any] = {"quick": quick}
    if out_dir is not None:
        kwargs["out_dir"] = out_dir
    try:
        report = run_train(df, **kwargs)
    except Exception as exc:
        fail(ui, f"training failed: {type(exc).__name__}: {exc}")
    _train_summary(ui, report)


def _train_summary(ui: Ui, report: Any) -> None:
    """What was trained, where it was saved, and which stages had enough rows."""
    console = ui.console()
    console.print(Text(f"trained bundle {report.version} on {report.n_rows} IPOs"
                       f"{' (quick mode)' if report.quick else ''}, at {report.trained_at}"))
    console.print(Text(f"  saved   {report.model_path}", style=DIM), soft_wrap=True)  # paths must stay whole
    console.print(Text(f"  report  {report.metrics_path}", style=DIM), soft_wrap=True)
    for stage, info in report.stages.items():
        state = "trained" if info.get("trained") else "not trained (too few rows)"
        console.print(Text(f"  {stage:<12} {state}"))
    for warning in report.warnings:
        console.print(Text(f"  ! {warning}", style=WARN))


def evaluate(
    ctx: typer.Context,
    plain: PlainOption = False,
    quick: bool = typer.Option(False, "--quick", help="Fewer trees and one seed: a fast pass, same report."),
) -> None:
    """Walk-forward evaluation against the GMP and training-median baselines. Writes docs/MODEL_REPORT.md."""
    ui = ui_from(ctx, plain)
    run_eval = _function_or_exit(ui, "ipo_radar.model.evaluate", "evaluate", "model evaluation")
    df = _history_frame(ui)
    try:
        result = run_eval(df, quick=quick)
    except Exception as exc:
        fail(ui, f"evaluation failed: {type(exc).__name__}: {exc}")
    console = ui.console()
    console.print(Text(f"wrote {result.report_path}"))
    console.print(Text(f"walk-forward test years {', '.join(map(str, result.test_years))}; "
                       f"{result.n_test_rows} test rows; data {result.data_hash}", style=DIM))
    for stage, line in result.verdicts.items():
        console.print(Text(f"  {stage:<12} {line}"))


def research(ctx: typer.Context, plain: PlainOption = False) -> None:
    """Run the research analysis (ipo_radar.research, once it is written)."""
    ui = ui_from(ctx, plain)
    _function_or_exit(ui, "ipo_radar.research", "run", "the research module")()


# ---- doctor ------------------------------------------------------------------------------------------
def _safe(name: str, fn: Callable[[], list[Check]]) -> list[Check]:
    """A check that crashes is reported as a failing check, so the rest of the report still prints."""
    try:
        return fn()
    except Exception as exc:
        return [Check(name, "fail", f"check crashed: {type(exc).__name__}: {exc}")]


def _python_check() -> list[Check]:
    version = ".".join(str(v) for v in sys.version_info[:3])
    if sys.version_info >= (3, 12):
        return [Check("python", "ok", f"{version} at {sys.executable}")]
    return [Check("python", "fail", f"{version}; needs 3.12 or newer", "use the project venv (.venv)")]


def _dependency_checks() -> list[Check]:
    missing = []
    for dist in _DEPENDENCIES:
        try:
            metadata.version(dist)
        except metadata.PackageNotFoundError:
            missing.append(dist)
    if missing:
        return [Check("python packages", "fail", f"missing: {', '.join(missing)}",
                      "run `.venv/bin/pip install -e .` in the project")]
    return [Check("python packages", "ok", f"all {len(_DEPENDENCIES)} present")]


STALE_DATA_DAYS = 60   # the newest IPO row older than this means the raw downloads need refreshing


def _file_check(label: str, path: Path, fix: str, *, rows: bool = False) -> Check:
    if not path.exists():
        return Check(label, "fail", f"missing: {path}", fix)
    now = datetime.now(IST)
    written = datetime.fromtimestamp(path.stat().st_mtime, IST)
    detail = f"written {fmt.day(written.date(), year_of=now.date())} ({fmt.age_text(written, now)})"
    if rows:
        import pyarrow.parquet as pq

        detail = f"{pq.ParquetFile(path).metadata.num_rows} rows, {detail}"
    return Check(label, "ok", detail)


def _freshness_check() -> Check:
    """How many days old the newest IPO row is, from the history table's open dates."""
    path = config.IPOS_PARQUET
    if not path.exists():
        return Check("newest IPO row", "fail", "no history table to check", "run `ipo data build`")
    dates = pd.to_datetime(pd.read_parquet(path, columns=["open_date"])["open_date"], errors="coerce")
    if dates.dropna().empty:
        return Check("newest IPO row", "fail", "no open dates in the history table", "run `ipo data build`")
    newest = dates.max().date()
    days = (datetime.now(IST).date() - newest).days
    detail = f"opened {newest.isoformat()}, {days} day{'s' if days != 1 else ''} ago"
    if days > STALE_DATA_DAYS:
        return Check("newest IPO row", "warn", detail + f" (over {STALE_DATA_DAYS})",
                     "refresh the raw downloads in data/raw/, then run `ipo data build`")
    return Check("newest IPO row", "ok", detail)


def _data_checks() -> list[Check]:
    return [
        _file_check("history table", config.IPOS_PARQUET, "run `ipo data build`", rows=True),
        _file_check("indices", config.INDICES_PARQUET, "run `ipo data build`", rows=True),
        _file_check("coverage report", config.PROCESSED / "COVERAGE.md", "run `ipo data build`"),
        _freshness_check(),
    ]


def _model_check() -> list[Check]:
    try:
        predictor = backend.predictor()
    except backend.ModelUnavailable as exc:
        fix = "run `ipo train`" if str(exc) == backend.NOT_TRAINED else "check models/ and retrain with `ipo train`"
        return [Check("model", "fail", str(exc), fix)]
    parts = [f"version {predictor.version or 'not reported'}"]
    if predictor.trained_at:
        parts.append(f"trained {predictor.trained_at[:10]}")
    if predictor.n_rows is not None:
        parts.append(f"{predictor.n_rows} IPOs in training")
    return [Check("model", "ok", "; ".join(parts))]


def _source_checks() -> list[Check]:
    out = []
    for h in backend.check_sources():
        if h.ok:
            out.append(Check(f"source {h.name}", "ok", f"{h.detail} ({h.latency_ms:.0f} ms)"))
        else:
            out.append(Check(f"source {h.name}", "fail", h.detail[:100],
                             "try again later; check your connection"))
    return out


def _notifier_check() -> list[Check]:
    channels = backend.notifiers()
    if channels:
        return [Check("phone alerts", "ok", "configured: " + ", ".join(c.name for c in channels))]
    return [Check("phone alerts", "warn", "no notifier configured", "run `ipo alerts setup`")]


def _config_check() -> list[Check]:
    path = config.CONFIG_FILE
    if not path.exists():
        return [Check("config file", "optional", f"none yet ({path})")]
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        return [Check("config file", "warn", f"{path} is readable by other users (it may hold tokens)",
                      f"chmod 600 {path}")]
    return [Check("config file", "ok", f"{path}, private to you")]


def _timer_check() -> list[Check]:
    if (backend.systemd_dir() / backend.TIMER_NAME).exists():
        return [Check("alerts timer", "ok", "installed")]
    return [Check("alerts timer", "optional", "not installed (optional)", "run `ipo alerts install-timer`")]


def doctor(
    ctx: typer.Context,
    plain: PlainOption = False,
    as_json: bool = typer.Option(False, "--json", help="Print the checks as JSON."),
) -> None:
    """Check Python packages, data files, the model, each live source, phone alerts and the config."""
    ui = ui_from(ctx, plain)
    checks: list[Check] = []
    for name, fn in (("python", _python_check), ("packages", _dependency_checks), ("data", _data_checks),
                     ("model", _model_check), ("sources", _source_checks), ("alerts", _notifier_check),
                     ("config", _config_check), ("timer", _timer_check)):
        checks += _safe(name, fn)
    failed = [c for c in checks if c.result == "fail"]
    if as_json:
        typer.echo(json.dumps({"ok": not failed, "checks": to_jsonable(checks)}, ensure_ascii=False, indent=2))
    else:
        ui.console().print(render.checks_table(checks, ui))
        ui.console().print(Text(f"\n{len(failed)} failing check(s)" if failed else "\nall required checks pass",
                                style=DIM))
    raise typer.Exit(1 if failed else EXIT_OK)
