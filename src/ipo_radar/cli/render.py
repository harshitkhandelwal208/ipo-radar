"""Rich renderables for each view. They take Analysis objects and plain values and never touch the network.

Text that came from a source (company names, warnings) is always a Text object, never markup, so a bracket
in a name cannot break the layout. One accent colour; green and red only where they mean positive or
negative, amber for stale or conflicting data, dim for unknown.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ipo_radar.cli import fmt
from rich.padding import Padding

from ipo_radar.cli.analysis import STATUS_TEXT, Analysis, ModelView, SourceStatus, SubRow
from ipo_radar.cli.ui import Ui

ACCENT = "cyan"
GOOD = "green"
BAD = "red"
WARN = "yellow"
DIM = "dim"
WIDE = 110                  # at and above this width, extra columns appear
SHORT_STATUS = {
    "upcoming": "upcoming", "open": "open", "closed": "closed", "allotment": "allotment",
    "listing_today": "lists today", "listed": "listed", "unknown": "unknown",
}


@dataclass(frozen=True)
class Check:
    """One `ipo doctor` line. result: ok | fail | warn | optional."""
    name: str
    result: str
    detail: str
    fix: str = ""


@dataclass(frozen=True)
class HistoryRow:
    opened: str
    company: str
    market: str
    size_cr: float | None
    sub_total: float | None
    gmp_pct: float | None
    open_gain: float | None
    close_gain: float | None


def _table(ui: Ui, **kwargs: object) -> Table:
    return Table(box=ui.table_box(), show_edge=False, pad_edge=False,
                 header_style="bold" if ui.plain else f"bold {ACCENT}", **kwargs)  # type: ignore[arg-type]


def heading(title: str, ui: Ui) -> Rule:
    return Rule(Text(title, style=f"bold {ACCENT}"), characters=ui.glyph("─", "-"), style=DIM, align="left")


def _sub_text(value: float | None) -> Text:
    if value is None:
        return Text(fmt.DASH, style=DIM)
    return Text(fmt.multiple(value), style=GOOD if value >= 1 else BAD)


def _bar(fraction: float | None, width: int, ui: Ui, style: str) -> Text:
    """Filled cells for the share, empty cells dim. Unknown draws nothing, so the dash alone says 'no data'."""
    if fraction is None:
        return Text(" " * width)
    filled = round(fraction * width)
    return Text.assemble((ui.glyph("█", "#") * filled, style), (ui.glyph("░", "-") * (width - filled), DIM))


def _sub_cell(row: SubRow | None, ui: Ui, wide: bool) -> Text:
    if row is None:
        return Text(fmt.DASH, style=DIM)
    cell = Text()
    if wide:
        style = GOOD if (row.now or 0) >= 1 else BAD
        cell.append_text(_bar(fmt.log_share(row.now), 10, ui, style))
        cell.append(" ")
    cell.append_text(_sub_text(row.now))
    return cell


def _price_band(low: float | None, high: float | None) -> str:
    if low is None and high is None:
        return fmt.DASH
    if low is not None and high is not None and low != high:
        return f"{fmt.rupees(low)} to {fmt.rupees(high)}"
    return fmt.rupees(low if low is not None else high)


def _gmp_cell(a: Analysis) -> Text:
    if a.gmp.median is None:
        return Text(fmt.DASH, style=DIM)
    style = GOOD if a.gmp.median > 0 else BAD if a.gmp.median < 0 else ""
    text = fmt.rupees(a.gmp.median, sign=True)
    if a.gmp.pct_of_top is not None:
        text += f" ({a.gmp.pct_of_top:+.0f}%)"
    return Text(text, style=style)


def _flag_cell(a: Analysis) -> Text:
    warn = sum(1 for f in a.flags if f.level == "warn")
    return Text(f"{warn} !", style=WARN) if warn else Text("")


def _dates_text(a: Analysis) -> str:
    """'15–19 Oct' within one month, '30 Sep–6 Oct' across two; one date alone when the close is unknown."""
    opens, closes = a.timeline.opens, a.timeline.closes
    if opens is None or closes is None:
        return fmt.day(opens or closes)
    if (opens.year, opens.month) == (closes.year, closes.month):
        return f"{opens.day}–{closes.day} {closes:%b}"
    return f"{fmt.day(opens)}–{fmt.day(closes)}"


def dashboard(rows: list[Analysis], *, ui: Ui, width: int, model_message: str | None) -> RenderableType:
    """Three tiers by width: compact (<100) drops the bar and dates; medium (100-139) adds both; wide (140+)
    adds the band and the model's P10-P90 range. Every tier keeps the same cells, so nothing is cut off."""
    show_bar = width >= 100
    dates = width >= 100
    wide = width >= 140
    show_model = model_message is None
    t = _table(ui)
    t.add_column("Company", no_wrap=True, overflow="ellipsis", max_width=28 if wide else 22)
    t.add_column("Status", no_wrap=True)
    if dates:
        t.add_column("Dates", no_wrap=True)
    if wide:
        t.add_column("Band", no_wrap=True, justify="right")
    t.add_column("Subscribed", no_wrap=True, justify="right")
    t.add_column("GMP", no_wrap=True, justify="right")
    if show_model:
        t.add_column("Model P50 (P10 to P90)" if wide else "Model P50", no_wrap=True, justify="right")
    t.add_column("Flags", no_wrap=True, justify="right")
    for a in rows:
        status_style = f"bold {ACCENT}" if a.status == "open" else ""
        cells: list[RenderableType] = [Text(a.company, style="bold"),
                                       Text(SHORT_STATUS.get(a.status, a.status), style=status_style)]
        if dates:
            cells.append(Text(_dates_text(a)))
        if wide:
            cells.append(Text(_price_band(a.issue.price_low, a.issue.price_high)))
        cells.append(_sub_cell(a.sub("sub_total"), ui, show_bar))
        cells.append(_gmp_cell(a))
        if show_model:
            cells.append(_model_cell(a.model, wide))
        cells.append(_flag_cell(a))
        t.add_row(*cells)
    parts: list[RenderableType] = [t]
    if show_model:
        parts.append(Text("Model P50 is an estimate of the listing-open gain. Not investment advice.", style=DIM))
        if any(a.model is not None and a.model.no_skill for a in rows):
            parts.append(Text(NO_SKILL_FOOTER, style=DIM))
        if any(a.model is not None and not a.model.used for a in rows):
            parts.append(Text("median (model not used here) = the segment's training median of the open gain, "
                              "shown where the model is worse than the historical median at that stage", style=DIM))
    else:
        parts.append(Text(str(model_message), style=WARN))
    return Group(*parts)


NO_SKILL_FOOTER = ("~ no demonstrated skill over the historical median at this stage (see docs/MODEL_REPORT.md); "
                   "the grey-market premium is often the better guide")


MEDIAN_LABEL = "median (model not used here)"


def _model_cell(m: ModelView | None, wide: bool) -> Text:
    if m is None:
        return Text(fmt.DASH, style=DIM)
    if not m.used:  # the model is worse than the median here: show the median, marked as such
        return Text.assemble(("~ ", DIM), (f"{m.median:+.0f}% median" if m.median is not None else "— median",
                                           DIM if m.median is None else ""))
    text = f"{m.p50:+.0f}%"
    if wide:
        text += f" ({m.p10:+.0f} to {m.p90:+.0f})"
    if m.no_skill:
        return Text.assemble(("~ ", DIM), text)
    return Text(text)


def _grid(rows: list[tuple[str, str]]) -> RenderableType:
    """Label and value columns; a wrapped value keeps its indent instead of falling back to column 0."""
    g = Table.grid(padding=(0, 2))
    g.add_column(style=DIM, no_wrap=True)
    g.add_column(ratio=1)
    for label, value in rows:
        g.add_row(label, value)
    return Padding(g, (0, 0, 0, 2))


def _bullets(items: list[tuple[str, str, str]]) -> RenderableType:
    """(marker, text, style) rows with a hanging indent."""
    g = Table.grid(padding=(0, 1))
    g.add_column(no_wrap=True)
    g.add_column(ratio=1)
    for marker, text, style in items:
        g.add_row(Text(marker, style=style), Text(text, style=style))
    return Padding(g, (0, 0, 0, 2))


def _header(a: Analysis) -> Text:
    t = Text()
    t.append(a.company, style="bold")
    if a.symbol:
        t.append(f"  {a.symbol}", style=DIM)
    t.append(f"\n{a.segment} · ", style=DIM)
    t.append(STATUS_TEXT.get(a.status, a.status), style=f"bold {ACCENT}")
    return t


def _timeline(a: Analysis, ui: Ui) -> Text:
    tl = a.timeline
    stations = [("Open", tl.opens), ("Close", tl.closes), ("Allot", tl.allotment), ("List", tl.listing)]
    line = Text("  ")
    for i, (label, when) in enumerate(stations):
        if i:
            line.append(ui.glyph(" ── ", " -- "), style=DIM)
        if when is None:
            line.append(f"{label} {fmt.DASH}", style=DIM)
        elif when == tl.today:
            line.append(f"{ui.glyph('◉', '@')} {label} {fmt.day(when, year_of=tl.today)}", style=f"bold {ACCENT}")
        elif when < tl.today:
            line.append(f"{ui.glyph('●', '#')} {label} {fmt.day(when, year_of=tl.today)}")
        else:
            line.append(f"{ui.glyph('○', 'o')} {label} {fmt.day(when, year_of=tl.today)}", style=DIM)
    line.append("\n  ")
    line.append(f"today {fmt.day(tl.today, year_of=tl.today)}: {tl.phase}", style=DIM)
    return line


def _issue(a: Analysis) -> RenderableType:
    i = a.issue
    lot = f"{i.lot_size} shares" if i.lot_size else fmt.DASH
    minimum = (f"min retail {fmt.rupees(i.min_retail)} (1 lot, top price)" if i.min_retail is not None
               else "min retail " + fmt.DASH)
    ofs = fmt.percent(i.ofs_pct)
    if i.ofs_cr:  # no OFS amount line when the issue is all fresh money
        ofs += f" ({fmt.crore_text(i.ofs_cr)})"
    return _grid([
        ("Band", f"{_price_band(i.price_low, i.price_high)} · lot {lot} · {minimum}"),
        ("Size", f"{fmt.crore_text(i.size_cr)} · fresh {fmt.percent(i.fresh_pct)} · OFS {ofs}"),
    ])


def _subs(a: Analysis, ui: Ui, width: int, history_note: str | None) -> list[RenderableType]:
    wide = width >= WIDE
    t = _table(ui)
    t.add_column("Category", no_wrap=True)
    t.add_column("Now", no_wrap=True, justify="right")
    for d in (1, 2, 3):
        t.add_column(f"Day {d}", no_wrap=True, justify="right")
    t.add_column("vs past", no_wrap=True, justify="right")
    t.add_column("Source", no_wrap=True)
    for row in a.subs:
        days = [_sub_text(v) if row.key in ("sub_qib", "sub_nii", "sub_retail", "sub_total") else
                Text(fmt.DASH, style=DIM) for v in row.days]
        source = Text(f"{row.source}, {fmt.age_text(row.fetched_at, a.as_of)}" if row.source else fmt.DASH,
                      style="" if row.source else DIM)
        pct = Text(f"p{row.pctile}") if row.pctile is not None else Text(fmt.DASH, style=DIM)
        t.add_row(Text(row.label, style="bold" if row.key == "sub_total" else ""),
                  _sub_cell(row, ui, wide), *days, pct, source)
    notes: list[RenderableType] = [t]
    past = max((r.pctile_n for r in a.subs), default=0)
    if past:
        notes.append(Text(f"vs past: p73 means the figure is higher than 73% of the past {a.segment} IPOs "
                          f"that reported it (up to {past}).", style=DIM))
    elif history_note is None:
        notes.append(Text("vs past: no past IPOs in the history to compare with.", style=DIM))
    return notes


def _gmp(a: Analysis, ui: Ui, history_note: str | None) -> list[RenderableType]:
    g = a.gmp
    lines: list[RenderableType] = []
    if not g.readings:
        lines.append(Text("  no reading from any aggregator: unavailable, or not listed for this IPO", style=DIM))
    for r in g.readings:
        pct = fmt.percent(r.value / a.issue.price_high * 100, sign=True) if (
            r.value is not None and a.issue.price_high) else fmt.DASH
        line = Text(f"  {r.source:<13}")
        line.append(f"{fmt.rupees(r.value, sign=True):>9}", style=_sign_style(r.value))
        line.append(f"  {pct:>6}  {fmt.age_text(r.fetched_at, a.as_of)}", style=DIM)
        lines.append(line)
    if g.median is not None:
        lines.append(_grid([("Median", f"{fmt.rupees(g.median, sign=True)} · "
                                         f"{fmt.percent(g.pct_of_top, sign=True)} of the top price")]))
    if g.history is not None:
        h = g.history
        lines.append(Text(
            f"  Past {a.segment} IPOs with GMP {h.bucket} (n={h.n}): listing-open gain median "
            f"{h.median_gain:+.0f}%, middle 80% between {h.p10:+.0f}% and {h.p90:+.0f}%, "
            f"{h.share_positive:.0f}% opened above issue."))
    elif g.history_note and history_note is None:
        lines.append(Text(f"  {g.history_note}", style=DIM))
    if g.miss_median is not None:
        lines.append(Text(f"  Across {g.miss_n} past IPOs, GMP% missed the actual open gain by "
                          f"{g.miss_median:.0f} points at the median."))
    return lines


def _range_bar(m: ModelView, ui: Ui, width: int) -> Text:
    """P10..P90 band on one axis, P50 as the marker, the GMP baseline as a second marker, zero as a rule."""
    low = min(m.p10, m.baseline_gmp_pct if m.baseline_gmp_pct is not None else m.p10, 0.0)
    high = max(m.p90, m.baseline_gmp_pct if m.baseline_gmp_pct is not None else m.p90, 0.0)
    pad = (high - low) * 0.05 or 1.0
    low, high = low - pad, high + pad

    def col(value: float) -> int:
        return max(0, min(width - 1, round((value - low) / (high - low) * (width - 1))))

    cells = [(ui.glyph("─", "-"), DIM) for _ in range(width)]
    for c in range(col(m.p10), col(m.p90) + 1):
        cells[c] = (ui.glyph("━", "="), ACCENT)
    cells[col(0.0)] = (ui.glyph("┆", ":"), DIM)
    if m.baseline_gmp_pct is not None:
        cells[col(m.baseline_gmp_pct)] = (ui.glyph("◆", "x"), WARN)
    cells[col(m.p50)] = (ui.glyph("●", "O"), f"bold {ACCENT}")
    line = Text("  ")
    for ch, style in cells:
        line.append(ch, style=style)
    return line


def _model(a: Analysis, ui: Ui, width: int, model_message: str | None) -> list[RenderableType]:
    m = a.model
    if m is None:
        return [Text(model_message or "no model view for this IPO", style=WARN if model_message else DIM)]
    bar_width = min(44, max(16, width - 34))
    baseline = fmt.percent(m.baseline_gmp_pct, sign=True) if m.baseline_gmp_pct is not None else fmt.DASH
    lines: list[RenderableType] = []
    if m.used:
        lines += [Text("  listing-open gain (estimate)", style=DIM)]
    else:
        headline = (f"{m.headline:+.0f}%" if m.headline is not None else fmt.DASH)
        lines += [
            Text.assemble(("  ~ ", DIM), (headline, "bold"), ("  " + MEDIAN_LABEL, "bold")),
            Text(f"  the model is {m.reason}; its range below is for reference only", style=DIM),
        ]
    lines += [
        _range_bar(m, ui, bar_width),
        Text(f"  {'' if m.used else 'model, reference only: '}P10 {m.p10:+.0f}%   P50 {m.p50:+.0f}%   "
             f"P90 {m.p90:+.0f}%   P(opens above issue) {fmt.probability_text(m.p_positive)}",
             style="" if m.used else DIM),
        Text.assemble("  ", ("just trust GMP: ", DIM), (baseline, WARN),
                      ("   reliable: ", DIM), ("yes" if m.reliable else "no", GOOD if m.reliable else WARN)),
    ]
    if m.drivers:
        drivers = ", ".join(f"{name} {value:+.1f}" for name, value in m.drivers[:6])
        lines.append(Text(f"  drivers (pts): {drivers}", style=DIM))
    if m.close_p50 is not None:
        lines.append(Text(f"  close-day gain, median estimate {m.close_p50:+.0f}%", style=DIM))
    for caveat in m.caveats:
        lines.append(Text(f"  caveat: {caveat}", style=DIM))
    return lines


def _flags(a: Analysis, ui: Ui) -> list[RenderableType]:
    if not a.flags:
        return [Text("  none raised", style=DIM)]
    items = [(ui.glyph("▲", "!"), f.text, WARN if f.level == "warn" else DIM) for f in a.flags]
    return [_bullets(items)]


def show_view(a: Analysis, *, ui: Ui, width: int, model_message: str | None,
              history_note: str | None) -> RenderableType:
    parts: list[RenderableType] = [_header(a), _timeline(a, ui), heading("Issue", ui), _issue(a)]
    parts += [heading("Subscription", ui), *_subs(a, ui, width, history_note)]
    if history_note:
        parts.append(Text(f"  {history_note}", style=DIM))
    parts += [heading("GMP (unofficial grey market)", ui), *_gmp(a, ui, history_note)]
    parts += [heading("Model estimate", ui), *_model(a, ui, width, model_message)]
    parts += [heading("Risk flags", ui), *_flags(a, ui)]
    notes = [w for w in a.warnings if "GMP sources disagree" not in w]  # GMP conflicts are flagged, not repeated
    if notes:
        parts += [heading("Data notes", ui), _bullets([("!", w, WARN) for w in notes])]
    parts += [heading("Summary (estimate)", ui), Text(a.summary)]
    parts.append(Text("\nEstimates only, not investment advice. Grey-market figures are unofficial.", style=DIM))
    return Group(*parts)


def health_block(health: list[SourceStatus], data_as_of: datetime | None, fetched_at: datetime,
                 now: datetime) -> Text:
    """The footer: each source's state, and how fresh the newest number is."""
    line = Text()
    if not health:
        line.append("sources  no per-source status (no IPOs to check against)", style=DIM)
    else:
        line.append("sources  ", style=DIM)
        for i, s in enumerate(health):
            if i:
                line.append(" · ", style=DIM)
            style = {"ok": GOOD, "down": BAD}.get(s.state, DIM)
            line.append(f"{s.name} {s.state}", style=style)
            if s.state == "down" and s.detail:
                line.append(f" ({s.detail[:60]})", style=DIM)
    line.append("\n")
    data = fmt.age_text(data_as_of, now) if data_as_of else "age unknown"
    line.append(f"data     newest reading {data} · fetched {fetched_at:%H:%M} IST", style=DIM)
    return line


def alert_lines(outcome_status: str, note: str, sent: list[tuple[str, list[tuple[str, str, str]]]],
                ui: Ui) -> RenderableType:
    """sent: (alert title, [(channel, status, reason)]). Empty when nothing was sent."""
    if outcome_status != "ran":
        return Text(f"alerts   {note}", style=WARN if outcome_status == "no_notifier" else DIM)
    if not sent:
        return Text("alerts   none new", style=DIM)
    rows: list[RenderableType] = [Text(f"alerts   {len(sent)} new", style="")]
    for title, deliveries in sent:
        line = Text(f"  {ui.glyph('→', '>')} ", style=DIM)
        line.append(title)
        for channel, status, reason in deliveries:
            style = GOOD if status == "sent" else BAD
            line.append(f"  [{channel} {status}" + (f": {reason}]" if reason else "]"), style=style)
        rows.append(line)
    return Group(*rows)


def checks_table(checks: list[Check], ui: Ui) -> RenderableType:
    t = _table(ui)
    t.add_column("Check", no_wrap=True)
    t.add_column("Result", no_wrap=True)
    t.add_column("Detail", overflow="fold")
    styles = {"ok": GOOD, "fail": BAD, "warn": WARN, "optional": DIM}
    for c in checks:
        t.add_row(Text(c.name), Text(c.result, style=styles.get(c.result, "")), Text(c.detail))
    fixes = [c for c in checks if c.fix and c.result in ("fail", "warn")]
    parts: list[RenderableType] = [t]
    if fixes:
        parts.append(Text(""))
        parts += [Text.assemble((f"fix {c.name}: ", "bold"), (c.fix, "")) for c in fixes]
    return Group(*parts)


def history_table(rows: list[HistoryRow], ui: Ui, width: int) -> RenderableType:
    wide = width >= WIDE
    t = _table(ui)
    t.add_column("Opened", no_wrap=True)
    t.add_column("Company", no_wrap=True, overflow="ellipsis", max_width=34 if wide else 22)
    if wide:
        t.add_column("Market", no_wrap=True)
    t.add_column("Size", no_wrap=True, justify="right")
    t.add_column("Sub", no_wrap=True, justify="right")
    t.add_column("GMP%", no_wrap=True, justify="right")
    t.add_column("Open", no_wrap=True, justify="right")
    t.add_column("Close", no_wrap=True, justify="right")
    for r in rows:
        cells: list[RenderableType] = [Text(r.opened), Text(r.company)]
        if wide:
            cells.append(Text(r.market, style=DIM))
        cells += [
            Text(fmt.crore_text(r.size_cr)),
            _sub_text(r.sub_total),
            Text(fmt.percent(r.gmp_pct)),
            Text(fmt.percent(r.open_gain, sign=True), style=_sign_style(r.open_gain)),
            Text(fmt.percent(r.close_gain, sign=True), style=_sign_style(r.close_gain)),
        ]
        t.add_row(*cells)
    return t


def _sign_style(value: float | None) -> str:
    if value is None:
        return DIM
    return GOOD if value > 0 else BAD if value < 0 else ""


def alerts_status(channels: list[tuple[str, str]], quiet: str, timer: str, sent: list[tuple[str, str]] | None,
                  ui: Ui) -> RenderableType:
    """Channel table, quiet hours, timer state and the last alerts sent (sent=None means unreadable)."""
    t = _table(ui)
    t.add_column("Channel", no_wrap=True)
    t.add_column("Setting", overflow="fold")
    for name, setting in channels:
        t.add_row(Text(name, style="bold"), Text(setting))
    parts: list[RenderableType] = [t, Text(f"quiet hours  {quiet}", style=DIM), Text(f"timer        {timer}", style=DIM),
                                   heading("Last alerts sent", ui)]
    if sent is None:
        parts.append(Text("  the alert state file could not be read", style=WARN))
    elif not sent:
        parts.append(Text("  none yet", style=DIM))
    else:
        parts += [Text(f"  {when[:16].replace('T', ' ')}  {title}") for when, title in sent]
    return Group(*parts)


def secret_panel(text: str, ui: Ui) -> RenderableType:
    return Panel.fit(Text(text), box=ui.panel_box(), padding=(0, 1))
