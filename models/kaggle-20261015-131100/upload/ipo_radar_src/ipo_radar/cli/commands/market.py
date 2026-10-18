"""Market views: fetch (refresh, predict, dashboard, alerts), live (dashboard from cache), show (one IPO),
history (the past-IPO table)."""
from __future__ import annotations

import json
from datetime import date, datetime
from enum import Enum

import pandas as pd
import typer
from rich.text import Text

from ipo_radar.alerts import daemon
from ipo_radar.alerts.engine import IST
from ipo_radar.cli import backend, fmt, match, render, service
from ipo_radar.cli.analysis import to_jsonable
from ipo_radar.cli.render import DIM, WARN, HistoryRow
from ipo_radar.cli.ui import EXIT_ERROR, PlainOption, Ui, fail, not_available, ui_from
from ipo_radar.schema import ALL_COLS


class Market(str, Enum):
    IN_MAIN = "IN_MAIN"
    IN_SME = "IN_SME"
    US = "US"
    HK = "HK"
    OTHER = "OTHER"


class SortKey(str, Enum):
    date = "date"
    gain = "gain"
    subscription = "subscription"
    size = "size"
    name = "name"


def _emit_json(payload: object) -> None:
    typer.echo(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2))


def _no_live_data(ui: Ui, view: service.View, as_json: bool) -> None:
    """Exit 1. In JSON mode the reason goes into the JSON only, so stdout stays parseable."""
    message = f"no live IPO data: {view.live.error}"
    if as_json:
        _emit_json({"error": message})
        raise typer.Exit(EXIT_ERROR)
    fail(ui, message)


def _alert_items(run: daemon.RunResult | None) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """Each alert with the delivery results recorded for it."""
    if run is None:
        return []
    by_key: dict[str, list[tuple[str, str, str]]] = {}
    for r in run.results:
        by_key.setdefault(r.alert_key, []).append((r.channel, r.status, r.reason))
    return [(a.title, by_key.get(a.key, [])) for a in run.alerts]


def _print_dashboard(ui: Ui, view: service.View, alerts: service.AlertOutcome | None) -> None:
    console = ui.console()
    if not view.analyses:
        console.print(Text("No live or upcoming IPOs in the feeds right now.", style=DIM))
    else:
        console.print(render.dashboard(view.analyses, ui=ui, width=console.width,
                                       model_message=view.model_message))
    console.print()
    console.print(render.health_block(view.live.health, view.data_as_of, view.now, view.now))
    if view.prediction_errors:
        console.print(Text(f"model could not predict {len(view.prediction_errors)} IPO(s): "
                           + "; ".join(view.prediction_errors[:3]), style=WARN))
    if view.history_note:
        console.print(Text(f"history: {view.history_note}", style=DIM))
    if alerts is not None:
        console.print(render.alert_lines(alerts.status, alerts.note, _alert_items(alerts.run), ui))


def _dashboard_json(view: service.View, alerts: service.AlertOutcome | None) -> dict[str, object]:
    return {
        "fetched_at": view.now,
        "sources": view.live.health,
        "model": {"message": view.model_message, "version": view.model_version},
        "history_note": view.history_note,
        "ipos": view.analyses,
        "alerts": None if alerts is None else {
            "status": alerts.status, "note": alerts.note,
            "items": [{"title": t, "deliveries": [{"channel": c, "status": s, "reason": r} for c, s, r in d]}
                      for t, d in _alert_items(alerts.run)],
        },
    }


def fetch(
    ctx: typer.Context,
    plain: PlainOption = False,
    refresh: bool = typer.Option(True, "--refresh/--cached",
                                 help="Fetch from every source (default), or reuse the cache of the last 5 min."),
    no_alerts: bool = typer.Option(False, "--no-alerts", help="Do not evaluate or send alerts."),
    no_predict: bool = typer.Option(False, "--no-predict", help="Skip the model."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON instead of the dashboard."),
) -> None:
    """Refresh every live source, predict, show the dashboard and send alerts."""
    ui = ui_from(ctx, plain)
    view = service.build_view(refresh=refresh, predict=not no_predict)
    if view.live.error and not view.live.ipos:
        _no_live_data(ui, view, as_json)
    alerts = (service.AlertOutcome("skipped", "skipped (--no-alerts)") if no_alerts
              else service.send_alerts(view))
    if as_json:
        _emit_json(_dashboard_json(view, alerts))
        return
    _print_dashboard(ui, view, alerts)


def live(
    ctx: typer.Context,
    plain: PlainOption = False,
    as_json: bool = typer.Option(False, "--json", help="Print JSON instead of the dashboard."),
) -> None:
    """The dashboard from the cache. Goes to the network only when the cache is older than 5 minutes."""
    ui = ui_from(ctx, plain)
    view = service.build_view(refresh=False)
    if view.live.error and not view.live.ipos:
        _no_live_data(ui, view, as_json)
    if as_json:
        _emit_json(_dashboard_json(view, None))
        return
    _print_dashboard(ui, view, None)


def show(
    ctx: typer.Context,
    plain: PlainOption = False,
    query: str = typer.Argument(..., help="Company name or exchange symbol, e.g. 'tonbo' or 'TONBOIMP'."),
    as_json: bool = typer.Option(False, "--json", help="Print the analysis as JSON."),
) -> None:
    """The detailed analysis of one IPO."""
    ui = ui_from(ctx, plain)
    view = service.build_view(refresh=False)
    if view.live.error and not view.live.ipos:
        _no_live_data(ui, view, as_json)
    res = match.resolve(query, view.live.ipos)
    if res.ambiguous:
        names = ", ".join(f'"{i.company}"' + (f" ({i.symbol})" if i.symbol else "") for i in res.choices)
        fail(ui, f"'{query}' matches {len(res.choices)} IPOs. Did you mean {names}? Use the full name.")
    if res.match is None:
        hint = (" Closest: " + ", ".join(f'"{i.company}"' for i in res.choices)) if res.choices else ""
        fail(ui, f"no live IPO matches '{query}'.{hint} Run `ipo live` to see what is listed.")
    analysis = next(a for a in view.analyses if a.ipo_id == res.match.ipo_id)
    if as_json:
        _emit_json(analysis)
        return
    console = ui.console()
    console.print(render.show_view(analysis, ui=ui, width=console.width, model_message=view.model_message,
                                   history_note=view.history_note))


def _as_date(value: object) -> date | None:
    if not fmt.known(value):
        return None
    return pd.Timestamp(value).date()  # type: ignore[arg-type]


def _select(df: pd.DataFrame, market: Market | None, since: date | None, sort: SortKey,
            search: str | None, limit: int) -> pd.DataFrame:
    out = df.copy()
    out["_opened"] = pd.to_datetime(out["open_date"], errors="coerce")
    if market is not None:
        out = out[out["market"] == market.value]
    if since is not None:
        out = out[out["_opened"] >= pd.Timestamp(since)]
    if search:
        needle = search.lower()
        out = out[out["company"].astype(str).str.lower().str.contains(needle, regex=False)
                  | out["symbol"].astype(str).str.lower().str.contains(needle, regex=False)]
    column, ascending = {
        SortKey.date: ("_opened", False), SortKey.gain: ("list_open_gain_pct", False),
        SortKey.subscription: ("sub_total", False), SortKey.size: ("issue_size_m", False),
        SortKey.name: ("company", True),
    }[sort]
    return out.sort_values(column, ascending=ascending, na_position="last", kind="stable").head(limit)


def _history_row(r: pd.Series) -> HistoryRow:
    def num(name: str) -> float | None:
        value = r.get(name)
        return float(value) if fmt.known(value) else None  # type: ignore[arg-type]

    opened = _as_date(r.get("open_date"))
    size_m = num("issue_size_m")
    return HistoryRow(
        opened=opened.isoformat() if opened else fmt.DASH,
        company=str(r.get("company", "")),
        market=str(r.get("market", "") or fmt.DASH),
        size_cr=size_m / 10 if size_m is not None else None,
        sub_total=num("sub_total"),
        gmp_pct=num("gmp_pct_last"),
        open_gain=num("list_open_gain_pct"),
        close_gain=num("list_close_gain_pct"),
    )


def history(
    ctx: typer.Context,
    plain: PlainOption = False,
    market: Market | None = typer.Option(None, "--market", case_sensitive=False,
                                         help="IN_MAIN, IN_SME, US, HK or OTHER."),
    since: str | None = typer.Option(None, "--since", help="Opened on or after YYYY-MM-DD."),
    sort: SortKey = typer.Option(SortKey.date, "--sort", case_sensitive=False,
                                 help="date (newest first), gain, subscription, size or name."),
    limit: int = typer.Option(25, "--limit", min=1, max=1000, help="Rows to show."),
    search: str | None = typer.Option(None, "--search", help="Match company name or symbol."),
    as_json: bool = typer.Option(False, "--json", help="Print the rows as JSON."),
) -> None:
    """Browse the past-IPO table (data/processed/ipos.parquet)."""
    ui = ui_from(ctx, plain)
    try:
        df = backend.load_history()
    except backend.HistoryMissing:
        not_available(ui, "the history table is not built yet: run `ipo data build`")
    except Exception as exc:
        fail(ui, f"history unreadable ({type(exc).__name__}): rebuild it with `ipo data build`")
    since_date: date | None = None
    if since is not None:
        try:
            since_date = date.fromisoformat(since)
        except ValueError:
            fail(ui, f"--since must be YYYY-MM-DD, got {since!r}")
    picked = _select(df, market, since_date, sort, search, limit)
    if as_json:
        typer.echo(picked[[c for c in ALL_COLS if c in picked.columns]].to_json(orient="records", date_format="iso",
                                                                            indent=2))
        return
    console = ui.console()
    view_now = datetime.now(IST)
    if picked.empty:
        console.print(Text("No IPOs match those filters.", style=DIM))
        return
    console.print(render.history_table([_history_row(r) for _, r in picked.iterrows()], ui, console.width))
    mtime = backend.history_mtime()
    built = f", history built {fmt.age_text(datetime.fromtimestamp(mtime, IST), view_now)}" if mtime else ""
    console.print(Text(f"{len(picked)} of {len(df)} past IPOs shown{built}.", style=DIM))
