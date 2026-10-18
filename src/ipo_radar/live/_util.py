"""Small parsers shared by the live sources. Each returns None for text it cannot read, never a guess."""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def parse_number(text: str | None) -> float | None:
    """'2,62,84,500shares' -> 26284500.0; '' or None -> None."""
    if not text:
        return None
    m = _NUM.search(text.replace("₹", ""))
    if not m:
        return None
    return float(m.group(0).replace(",", ""))


def parse_price_band(text: str | None) -> tuple[float | None, float | None]:
    """'Rs.665 to Rs.700' or '₹271 per share' -> (low, high). A single price gives (price, price)."""
    if not text:
        return None, None
    nums = [float(n.replace(",", "")) for n in _NUM.findall(text.replace("₹", ""))]
    if not nums:
        return None, None
    if len(nums) == 1:
        return nums[0], nums[0]
    return min(nums[0], nums[1]), max(nums[0], nums[1])


def issue_day_index(open_date: date, day: date) -> int | None:
    """Business-day number of `day` in an issue that opened on `open_date` (1 = the opening day).

    Issues run over trading days, so weekends are not counted. Exchange holidays are not known here, so a
    holiday inside an issue shifts the later days by one. None before the opening day.
    """
    if day < open_date:
        return None
    count, current = 0, open_date
    while current <= day:
        if current.weekday() < 5:
            count += 1
        current += timedelta(days=1)
    return count


def parse_date(text: str | None) -> date | None:
    """Accepts '15-Oct-2026', '15-OCT-2026' and 'Oct 15, 2026'. Anything else is None."""
    if not text:
        return None
    s = text.strip()
    for fmt in ("%d-%b-%Y", "%b %d, %Y", "%d %b %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None
