"""Use-cases that several commands share: build the dashboard view, and run one alert cycle.

No terminal output happens here. Commands decide how to show what these return.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from ipo_radar.alerts import daemon
from ipo_radar.alerts.engine import IST
from ipo_radar.cli import backend
from ipo_radar.cli.analysis import Analysis, build_analysis, newest_reading, predict_all
from ipo_radar.schema import Prediction


STATUS_ORDER = {"open": 0, "upcoming": 1, "allotment": 2, "closed": 2, "listing_today": 3, "listed": 4}


@dataclass
class View:
    now: datetime
    live: backend.LiveResult
    analyses: list[Analysis]
    preds: dict[str, Prediction]
    model_message: str | None         # why there are no predictions; None when predictions exist
    model_version: str | None
    prediction_errors: list[str] = field(default_factory=list)
    data_as_of: datetime | None = None
    history_note: str | None = None


@dataclass
class AlertOutcome:
    status: str                       # ran | no_notifier | skipped
    note: str = ""
    run: daemon.RunResult | None = None


def load_history_or_note() -> tuple[pd.DataFrame | None, str | None]:
    """The past-IPO table, or None plus the line that says why it is missing. Never raises."""
    try:
        return backend.load_history(), None
    except backend.HistoryMissing:
        return None, "no history yet: percentiles and the GMP comparison need `ipo data build`"
    except Exception as exc:  # a corrupt file must not stop the live dashboard
        return None, f"history unreadable ({type(exc).__name__}): rebuild it with `ipo data build`"


def build_view(*, refresh: bool, predict: bool = True) -> View:
    """Live data, predictions and analyses for every IPO, in one place so every command agrees."""
    now = datetime.now(IST)
    live = backend.load_live(refresh=refresh)
    history, history_note = load_history_or_note()
    preds: dict[str, Prediction] = {}
    errors: list[str] = []
    model_message: str | None = None
    version: str | None = None
    if not predict:
        model_message = "predictions skipped (--no-predict)"
    else:
        try:
            predictor = backend.predictor()
        except backend.ModelUnavailable as exc:
            model_message = str(exc)
        else:
            version = predictor.version
            preds, errors = predict_all(predictor.predict, live.ipos)
    analyses = [build_analysis(ipo, preds.get(ipo.ipo_id), now=now, history=history) for ipo in live.ipos]
    analyses.sort(key=lambda a: (STATUS_ORDER.get(a.status, 9), a.company))
    return View(
        now=now, live=live, analyses=analyses, preds=preds, model_message=model_message,
        model_version=version, prediction_errors=errors, data_as_of=newest_reading(live.ipos),
        history_note=history_note,
    )


def send_alerts(view: View) -> AlertOutcome:
    """Evaluate and deliver alerts for the view. With no notifier configured nothing is evaluated or recorded."""
    channels = backend.notifiers()
    if not channels:
        return AlertOutcome("no_notifier", "no notifier configured: run `ipo alerts setup` to get phone alerts")
    run = backend.alert_cycle(view.live.ipos, lambda ipo: view.preds.get(ipo.ipo_id), channels)
    return AlertOutcome("ran", run=run)
