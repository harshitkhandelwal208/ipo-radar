"""Resolve a typed name ('tonbo', 'TONBO', 'Tonbo Imaging') to one live IPO, or say why it cannot.

Exact matches on name or symbol win. Otherwise a substring of the company key is used, but only when it
leaves exactly one candidate: guessing between two IPOs would show the user the wrong analysis.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass

from ipo_radar.live.keys import company_key
from ipo_radar.schema import LiveIPO


@dataclass(frozen=True)
class Resolution:
    match: LiveIPO | None = None
    choices: tuple[LiveIPO, ...] = ()   # candidates when ambiguous, nearest names when nothing matched
    ambiguous: bool = False


def resolve(query: str, ipos: list[LiveIPO]) -> Resolution:
    text = query.strip()
    if not text:
        return Resolution()
    key = company_key(text)
    upper = text.upper()
    exact = [i for i in ipos
             if i.ipo_id == key or i.company.strip().lower() == text.lower()
             or (i.symbol and i.symbol.upper() == upper)]
    if len(exact) == 1:
        return Resolution(match=exact[0])
    if exact:
        return Resolution(choices=tuple(exact), ambiguous=True)

    loose = [i for i in ipos if key and (key in i.ipo_id or (i.symbol and i.symbol.upper().startswith(upper)))]
    if len(loose) == 1:
        return Resolution(match=loose[0])
    if loose:
        return Resolution(choices=tuple(loose), ambiguous=True)

    by_key = {i.ipo_id: i for i in ipos}
    close = difflib.get_close_matches(key, list(by_key), n=3, cutoff=0.6) if key else []
    return Resolution(choices=tuple(by_key[k] for k in close))
