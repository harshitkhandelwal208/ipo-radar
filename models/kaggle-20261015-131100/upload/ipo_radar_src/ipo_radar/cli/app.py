"""`ipo`: the command-line front end. Commands live in cli/commands/; this module only wires them together."""
from __future__ import annotations

import logging

import typer

from ipo_radar.cli.commands import alerts as alerts_cmd
from ipo_radar.cli.commands import market, ops
from ipo_radar.cli.ui import Ui

app = typer.Typer(
    name="ipo",
    help="IPO radar: live subscription, GMP, model estimates and phone alerts. Estimates, not advice.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main(
    ctx: typer.Context,
    plain: bool = typer.Option(False, "--plain", help="No colour or boxes, for piping. Also accepted after any command, e.g. `ipo fetch --plain`."),
) -> None:
    logging.basicConfig(level=logging.ERROR, format="%(name)s: %(message)s")  # warnings are shown as results
    ctx.obj = Ui(plain=plain)


app.command("fetch")(market.fetch)
app.command("live")(market.live)
app.command("show")(market.show)
app.command("history")(market.history)
app.command("train")(ops.train)
app.command("eval")(ops.evaluate)
app.command("research")(ops.research)
app.command("doctor")(ops.doctor)
app.add_typer(alerts_cmd.alerts_app, name="alerts")
app.add_typer(ops.data_app, name="data")
