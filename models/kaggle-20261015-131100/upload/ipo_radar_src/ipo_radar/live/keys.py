"""Company matching key shared by every live source.

Sources spell the same company differently ("Tonbo Imaging India Limited", "Tonbo Imaging India IPO").
Stripping legal suffixes, the word "IPO" and punctuation makes those collide on one key. It is a
heuristic: two different companies with the same words collide too, so merge also matches by symbol.
"""
from __future__ import annotations

import re
from datetime import date

# Words that name the legal form or the offering, not the company. "india" is dropped as well, per the
# project spec, because every Indian issuer carries it and it is the most common false-match word.
_DROP = frozenset({
    "limited", "ltd", "pvt", "private", "india", "indian", "co", "company", "corp", "corporation",
    "inc", "llp", "ipo", "sme", "bse", "nse",
})
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def company_key(name: str) -> str:
    """Lower-case key with legal suffixes, 'IPO' and punctuation removed. Empty string if nothing is left."""
    text = name.lower().replace("&", " and ")
    tokens = [t for t in _NON_ALNUM.split(text) if t and t not in _DROP]
    return " ".join(tokens)


def same_company(key_a: str, key_b: str, *, market_a: str = "", market_b: str = "",
                 open_a: date | None = None, open_b: date | None = None) -> bool:
    """Conservative alias test for two company keys that may spell one company differently.

    Joins only when all of these hold:
    - the words of the shorter key are the leading words of the longer one ("sahajanand medical" joins
      "sahajanand medical technologies"). The shorter key's last word may be the start of the longer key's
      word when it has at least 5 letters ("businesses" joins "business");
    - both open dates are known and equal. This is the main guard: names alone never join, and two different
      IPOs with a shared name prefix ("Tata Motors" and "Tata Motors Finance") open on different days;
    - the markets agree wherever both are known (IN_MAIN vs IN_SME never join);
    - a one-word key ("zetwerk") also needs at least 5 letters.
    Known limit: two different companies with the same leading words that open on the same day in the same
    market would join. That is rare, not impossible; the merge warnings are the review trail.
    """
    if not key_a or not key_b:
        return False
    ta, tb = key_a.split(), key_b.split()
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if not short or len(short) > len(long_):
        return False
    for i, word in enumerate(short):
        other = long_[i]
        last = i == len(short) - 1
        if word == other:
            continue
        if last and len(word) >= 5 and other.startswith(word):
            continue
        return False
    if open_a is None or open_b is None or open_a != open_b:
        return False
    if market_a and market_b and market_a != market_b:
        return False
    if len(short) == 1 and len(short[0]) < 5:
        return False
    return True
