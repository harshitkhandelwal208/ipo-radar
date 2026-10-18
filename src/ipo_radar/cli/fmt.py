"""Text formatting for the terminal: Indian digit grouping, multiples, percentages, ages and dates.

Unknown values render as a dash, never as 0 (SPEC rule 1). Pure functions, so they are tested without a
terminal.
"""
from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd

DASH = "—"


def known(value: object) -> bool:
    """False for None, NaN and pandas' missing markers. Every other value is a real reading."""
    if value is None or value is pd.NA or value is pd.NaT:
        return False
    return not (isinstance(value, float) and math.isnan(value))


def group_inr(number: int) -> str:
    """Indian digit grouping: 1234567 -> '12,34,567'. The last three digits stay together."""
    sign = "-" if number < 0 else ""
    digits = str(abs(number))
    if len(digits) <= 3:
        return sign + digits
    head, tail = digits[:-3], digits[-3:]
    groups: list[str] = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return sign + ",".join(groups + [tail])


def rupees(value: float | None, *, sign: bool = False, decimals: int | None = None) -> str:
    """₹1,23,456. Whole rupees by default; two decimals only when the amount has paise."""
    if not known(value):
        return DASH
    number = float(value)  # type: ignore[arg-type]
    if decimals is None:
        decimals = 0 if number.is_integer() else 2
    rounded = round(number, decimals)
    whole, _, frac = f"{abs(rounded):.{decimals}f}".partition(".")
    text = "₹" + group_inr(int(whole)) + (f".{frac}" if frac else "")
    if rounded < 0:
        return "-" + text
    return ("+" + text) if sign and rounded > 0 else text


def crore_text(cr: float | None) -> str:
    """Rupees in crore, e.g. '₹1,234 cr'. One crore is ten million rupees."""
    if not known(cr):
        return DASH
    return rupees(float(cr), decimals=0 if cr >= 100 else 1) + " cr"  # type: ignore[operator]


def multiple(value: float | None) -> str:
    """Times subscribed: 3.45x below 10, 12.3x above."""
    if not known(value):
        return DASH
    return f"{value:,.2f}x" if value < 10 else f"{value:,.1f}x"  # type: ignore[operator]


def percent(value: float | None, *, sign: bool = False, digits: int = 0) -> str:
    if not known(value):
        return DASH
    return f"{value:+.{digits}f}%" if sign else f"{value:.{digits}f}%"  # type: ignore[str-format]


def day(value: date | None, *, year_of: date | None = None) -> str:
    """'15 Oct', or '15 Oct 2027' when the year differs from `year_of`."""
    if value is None:
        return DASH
    if year_of is None or value.year == year_of.year:
        return f"{value.day} {value:%b}"
    return f"{value.day} {value:%b %Y}"


def _aware(stamp: datetime) -> datetime:
    return stamp if stamp.tzinfo else stamp.astimezone()


def age_seconds(stamp: datetime | None, now: datetime) -> float | None:
    if stamp is None:
        return None
    return (_aware(now) - _aware(stamp)).total_seconds()


def age_text(stamp: datetime | None, now: datetime) -> str:
    """'just now', '3 min ago', '2 h ago', '4 d ago'. Unknown age says so."""
    seconds = age_seconds(stamp, now)
    if seconds is None:
        return "age unknown"
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    return f"{int(seconds // 86400)} d ago"


def probability_text(p: float | None) -> str:
    """A probability in 0..1 as a percentage. A saturated value reads as a bound, not as certainty: 0.97 is
    '>=95%', 0.02 is '<=5%'. Anything else is a whole percentage."""
    if not known(p):
        return DASH
    if p >= 0.95:
        return ">=95%"
    if p <= 0.05:
        return "<=5%"
    return f"{p * 100:.0f}%"


def log_share(value: float | None, cap: float = 100.0) -> float | None:
    """Bar position 0..1 on a log scale that reaches 1 at `cap` times: 1x is 0.15, 10x is 0.52."""
    if not known(value):
        return None
    clipped = min(max(float(value), 0.0), cap)  # type: ignore[arg-type]
    return math.log10(1 + clipped) / math.log10(1 + cap)
