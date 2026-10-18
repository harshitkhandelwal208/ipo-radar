"""Output mode, console factory and exit codes. The root callback builds one Ui; every command reads it."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, NoReturn

import typer
from rich import box as rich_box
from rich.console import Console
from rich.text import Text

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_AVAILABLE = 2  # the feature or data is not built yet; not a failure of the user's input


@dataclass(frozen=True)
class Ui:
    """plain: no colour, no boxes, ASCII glyphs where a fancy one exists. Piping output uses plain."""

    plain: bool = False

    def console(self, *, stderr: bool = False) -> Console:
        return Console(stderr=stderr, color_system=None if self.plain else "auto", highlight=False, emoji=False)

    def glyph(self, fancy: str, plain: str) -> str:
        return plain if self.plain else fancy

    def table_box(self) -> rich_box.Box | None:
        return None if self.plain else rich_box.SIMPLE_HEAD

    def panel_box(self) -> rich_box.Box:
        return rich_box.ASCII if self.plain else rich_box.ROUNDED


# Every command accepts --plain as well as the root flag, so `ipo fetch --plain` works (Typer only parses
# root options before the command name).
PlainOption = Annotated[bool, typer.Option("--plain", help="No colour or boxes, for piping.")]


def ui_from(ctx: typer.Context, plain: bool) -> Ui:
    """The effective output mode: plain if either the root flag or this command's flag asked for it."""
    root = ctx.obj if isinstance(ctx.obj, Ui) else Ui()
    return Ui(plain=plain or root.plain)


def fail(ui: Ui, message: str, code: int = EXIT_ERROR) -> NoReturn:
    """Print an error to stderr and exit. `message` is not markup, so brackets in names are safe."""
    ui.console(stderr=True).print(Text(f"error: {message}", style="" if ui.plain else "bold red"))
    raise typer.Exit(code)


def not_available(ui: Ui, message: str) -> NoReturn:
    """Exit 2: the feature exists in the plan but is not built or its data is missing."""
    ui.console(stderr=True).print(Text(f"not available yet: {message}", style="" if ui.plain else "yellow"))
    raise typer.Exit(EXIT_NOT_AVAILABLE)
