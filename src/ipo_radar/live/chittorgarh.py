"""Chittorgarh IPO pages: dates, price band, lot size, issue size, fresh/OFS split, market and the
subscription snapshots (per day and latest) that the site shows on each IPO page.

robots.txt (checked 2026-10-15): generic `User-agent: *` is Allow: /, except /ipo/ipo_discussions.asp.
Several named AI and ad bots are disallowed outright; this module does not identify as any of them. The
disclaimer (article 238) says no user may "distribute, modify, transmit, or use the contents ... for public
or commercial purposes without prior written permission". Nothing there forbids automated reading for
personal use, so the verdict is: scrape the public IPO pages only, personal use, one request per second,
raw copies kept on disk, no redistribution. The data-usage article linked from that disclaimer is gone
(it redirects to the article index), so that policy could not be read. The live subscription page
(ipo_subscription_status_live.asp) now redirects to the mainboard list, so it is not used.

The list table on the report pages is rendered client-side and is not in the HTML. The universe is
therefore the IPO links on the home page and the list pages: current and recent issues, not the full
history. Pages that fail are logged and skipped; if every page fails, SourceError is raised.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, time, timedelta, timezone

import httpx
from selectolax.lexbor import LexborHTMLParser

from ipo_radar.errors import SourceError
from ipo_radar.live import _http
from ipo_radar.live._util import issue_day_index, parse_date, parse_number, parse_price_band
from ipo_radar.live.keys import company_key
from ipo_radar.schema import LiveIPO, Sourced

SOURCE = "chittorgarh"
BASE = "https://www.chittorgarh.com"
LIST_PAGES = (
    "/",
    "/report/ipo-in-india-list-main-board-sme/82/all/?year=2026",
    "/report/ipo-in-india-list-main-board-sme/82/mainboard/",
)
# Links appear as href attributes and also inside escaped JSON in the page payload, so match the path itself.
_IPO_LINK = re.compile(r"/ipo/([a-z0-9-]+-ipo)/(\d+)/")
_DATE_RANGE = re.compile(r"(\d{1,2})\s*(?:([A-Za-z]{3})\s*)?to\s+(\d{1,2})\s+([A-Za-z]{3}),?\s*(\d{4})")
_DATE_SINGLE = re.compile(r"(\d{1,2})\s+([A-Za-z]{3}),?\s*(\d{4})")
_CRORE = re.compile(r"₹\s*([\d,]+(?:\.\d+)?)\s*Cr")
_SNAPSHOT_FIELDS = {  # snapshot field -> schema key
    "qib": "sub_qib", "nii": "sub_nii", "nii_big": "sub_bnii", "nii_small": "sub_snii",
    "rii": "sub_retail", "emp": "sub_employee", "total": "sub_total",
}
_log = logging.getLogger(__name__)
IST = timezone(timedelta(hours=5, minutes=30))  # the site's snapshot times are Indian local time, no zone given
END_OF_DAY = time(17, 0)


def _rows(tree: LexborHTMLParser) -> dict[str, str]:
    """Two-cell label/value rows across all tables. The first occurrence of a label wins."""
    out: dict[str, str] = {}
    for tr in tree.css("tr"):
        cells = tr.css("th, td")
        if len(cells) == 2:
            label = cells[0].text(strip=True)
            out.setdefault(label, cells[1].text(strip=True))
    return out


def _issue_window(text: str, year_hint: int | None = None) -> tuple[date, date] | None:
    """'13 to 15 Oct, 2026' -> (13 Oct, 15 Oct). Cross-month '28 Oct to 2 Nov, 2026' is handled too."""
    m = _DATE_RANGE.search(text)
    if m:
        d1, mon1, d2, mon2, year = m.groups()
        mon1 = mon1 or mon2
        try:
            return (datetime.strptime(f"{d1} {mon1} {year}", "%d %b %Y").date(),
                    datetime.strptime(f"{d2} {mon2} {year}", "%d %b %Y").date())
        except ValueError:
            return None
    m = _DATE_SINGLE.search(text)
    if m:
        try:
            d = datetime.strptime(" ".join(m.groups()), "%d %b %Y").date()
            return d, d
        except ValueError:
            return None
    return None


def _name(tree: LexborHTMLParser) -> str:
    h1 = tree.css_first("h1")
    raw = h1.text(separator=" ", strip=True) if h1 else ""
    name = re.sub(r"\s+", " ", re.sub(r"\s*IPO Details$", "", raw)).strip()
    if not name:
        raise SourceError("chittorgarh: IPO page has no company heading (layout changed?)")
    return name


def _market(listing_at: str) -> str:
    """'NSE SME' -> IN_SME; 'BSE, NSE' -> IN_MAIN. Anything else is unknown, so it stays ''."""
    if "SME" in listing_at:
        return "IN_SME"
    if "BSE" in listing_at or "NSE" in listing_at:
        return "IN_MAIN"
    return ""


def _fresh_pct(sale_type: str, total: float | None, ofs: float | None) -> float | None:
    """Fresh share of the issue. The sale-type label is trusted first; shares are used only without it."""
    low = sale_type.lower()
    if "ofs only" in low:
        return 0.0
    if "fresh" in low and "ofs" not in low:
        return 100.0
    if total and ofs is not None:
        return round(100.0 * (total - ofs) / total, 4)
    return None


def _snapshots(html: str) -> list[dict]:
    """The page's embedded bidding snapshots. The JSON sits inside an escaped Next.js payload, so the
    escaped quotes are undone before decoding. Returns [] when the page has none (upcoming issues)."""
    marker = html.find("ipoBiddingDetails")
    if marker < 0:
        return []
    start = html.find("[", marker)
    if start < 0:
        raise SourceError("chittorgarh: ipoBiddingDetails without a list (layout changed?)")
    text = html[start:].replace('\\"', '"')
    try:
        rows, _ = json.JSONDecoder().raw_decode(text)
    except ValueError as exc:
        raise SourceError("chittorgarh: subscription snapshot JSON unreadable (layout changed?)") from exc
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise SourceError("chittorgarh: subscription snapshot is not a list of objects")
    return rows


def _snapshot_value(raw: object, field: str) -> float | None:
    """Empty string or null means not reported (None). A non-empty non-number is a layout change."""
    if raw in (None, ""):
        return None
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise SourceError(f"chittorgarh: non-numeric {field} in snapshot: {raw!r}") from exc


def _apply_snapshots(ipo: LiveIPO, rows: list[dict]) -> None:
    """Latest snapshot becomes the live subscription, stamped with the time it was sampled (IST).

    sub_by_day takes only end-of-day figures: a snapshot from 17:00 IST on day N (bidding closes at 17:00 on the
    last day) is day N's figure. Chittorgarh publishes no day-wise table, so intraday snapshots are not filed
    under a day; an open issue's day figures come only from our own snapshots (live/snapshots.py)."""
    parsed = []
    for row in rows:
        when = row.get("bid_date")
        if not isinstance(when, str):
            raise SourceError("chittorgarh: snapshot without bid_date (layout changed?)")
        try:
            stamp = datetime.strptime(when, "%b %d, %Y %H:%M")
        except ValueError as exc:
            raise SourceError(f"chittorgarh: unreadable bid_date {when!r}") from exc
        values = {key: _snapshot_value(row.get(field), field) for field, key in _SNAPSHOT_FIELDS.items()}
        parsed.append((stamp, values))
    if not parsed:
        return
    parsed.sort(key=lambda p: p[0])
    latest_stamp, latest = parsed[-1]
    sampled = latest_stamp.replace(tzinfo=IST)
    for key, value in latest.items():
        if value is not None:
            ipo.sub[key] = Sourced(value, SOURCE, sampled, note=f"snapshot {latest_stamp:%b %d %H:%M}")
    if ipo.open_date is None:
        return
    for stamp, values in parsed:  # later end-of-day snapshots overwrite earlier ones on the same day
        if stamp.time() < END_OF_DAY:
            continue
        day = issue_day_index(ipo.open_date, stamp.date())
        if day is not None:
            ipo.sub_by_day[day] = {k: v for k, v in values.items()
                                   if v is not None and k in ("sub_qib", "sub_nii", "sub_retail", "sub_total")}
    if ipo.sub_by_day:
        ipo.sub_by_day_source = SOURCE


def parse_detail(html: str) -> LiveIPO:
    """One IPO page. Raises SourceError when the required rows (IPO Date, Price Band) are missing."""
    tree = LexborHTMLParser(html)
    rows = _rows(tree)
    if "IPO Date" not in rows or ("Price Band" not in rows and "Issue Price" not in rows):
        raise SourceError("chittorgarh: IPO page lacks IPO Date / Price Band rows (layout changed?)")
    name = _name(tree)
    window = _issue_window(rows["IPO Date"])
    if window is None:
        raise SourceError(f"chittorgarh: unreadable IPO Date {rows['IPO Date']!r} for {name}")
    listing = _first_date(rows.get("Listing Date", ""))
    low, high = parse_price_band(rows.get("Price Band") or rows.get("Issue Price"))
    # the 'Issue Price' row appears once the price is fixed; a band-only page has no final price
    issue_price = parse_price_band(rows.get("Issue Price"))[0] if "Issue Price" in rows else None
    lot = parse_number(rows.get("Lot Size"))
    total_shares = parse_number(rows.get("Total Issue Size"))
    ofs_shares = parse_number(rows.get("Offer for Sale"))
    crore = _CRORE.search(rows.get("Total Issue Size", ""))
    ipo = LiveIPO(
        ipo_id=company_key(name), market=_market(rows.get("Listing At", "")), company=name,
        open_date=window[0], close_date=window[1], listing_date=listing,
        price_low=low, price_high=high, issue_price=issue_price,
        lot_size=int(lot) if lot else None,
        issue_size_m=round(float(crore.group(1).replace(",", "")) * 10, 4) if crore else None,
        fresh_issue_pct=_fresh_pct(rows.get("Sale Type", ""), total_shares, ofs_shares),
        sources=[SOURCE],
    )
    _apply_snapshots(ipo, _snapshots(html))
    return ipo


def _first_date(text: str) -> date | None:
    m = re.search(r"([A-Za-z]{3}\s+\d{1,2},\s*\d{4})", text)
    return parse_date(m.group(1)) if m else None


def _discover(client: httpx.Client) -> list[str]:
    """Unique IPO page paths linked from the home and list pages, in page order."""
    found: list[str] = []
    for number, page in enumerate(LIST_PAGES):
        try:
            body = _http.get(client, BASE + page, SOURCE)
        except SourceError as exc:
            _log.warning("chittorgarh list page failed: %s", exc)
            continue
        _http.save_raw(SOURCE, f"list-{number}.html", body)
        for slug, ident in _IPO_LINK.findall(body.decode("utf-8", "replace")):
            path = f"/ipo/{slug}/{ident}/"
            if path not in found:
                found.append(path)
    if not found:
        raise SourceError("chittorgarh: no IPO links on any list page (layout changed or blocked?)")
    return found


def fetch(client: httpx.Client | None = None) -> list[LiveIPO]:
    """Every IPO page reachable from the list pages. Failed pages are logged and skipped."""
    owns = client is None
    client = client or _http.make_client()
    try:
        paths = _discover(client)
        out: list[LiveIPO] = []
        errors: list[str] = []
        for path in paths:
            try:
                body = _http.get(client, BASE + path, SOURCE)
                html = body.decode("utf-8", "replace")
                out.append(parse_detail(html))
                _http.save_raw(SOURCE, path.strip("/").replace("/", "_") + ".html", body)
            except SourceError as exc:
                _log.warning("chittorgarh page skipped: %s", exc)
                errors.append(str(exc))
        if not out:
            raise SourceError(f"chittorgarh: all {len(paths)} IPO pages failed; first: {errors[0]}")
        return out
    finally:
        if owns:
            client.close()
