"""Live grey market premium (GMP) readings for current and upcoming Indian IPOs.

GMP is unofficial grey-market sentiment, not exchange data (SPEC rule 2). Each reading is one aggregator's
number for one IPO, stamped with the fetch time. A failing source raises SourceError internally and is left
out; fetch() raises only when every source fails, so the caller can show "GMP unavailable".

HTTP goes through live/_http.py (one request per second per host, browser User-Agent, SourceError on every
failure). ipowatch drops roughly a third of its connections, so a dropped connection, empty body or 5xx is
retried twice more, 2 s apart; 403 and 429 are never retried.

Sources, with the robots.txt verdict (checked 2026-10-15):
- investorgain.com, /report/live-ipo-gmp/331/ (308 to /report/ipo-gmp-live/331/). robots.txt: Allow for all
  agents (Content-Signal search=yes, ai-train=no). USED. Server-rendered table with mainboard and SME rows.
  ai-train=no means these readings are for display and alerts; do not feed them to model training without a
  decision on that.
- ipocentral.in, /wp-json/ipo-gmp/v1/data (the JSON behind its GMP ticker). robots.txt disallows only
  /wp-admin/. USED. The ticker covers only a handful of IPOs and the endpoint is undocumented.
- ipopremium.in, home page GMP table. robots.txt allows / and disallows /api/, /v1/, /otpless/, /file/; none
  of those are requested. USED. The site labels the column "GMP Rumors" and says it is indicative only.
- ipowatch.in, home page "Current" GMP tables (mainboard and SME). robots.txt allows the home page. USED.
  Intermittent connection resets were seen; a reset is reported as SourceError.
- chittorgarh.com: REJECTED. It has no GMP data of its own; its GMP links go to investorgain, so counting it
  would double-count investorgain.
- ipoplatform.com: REJECTED. robots.txt allows the site, but neither the home page nor the sitemap has a GMP
  page or feed.
- ipotoday: REJECTED. The host does not resolve.

Zero versus dash (evidence: the raw pages under data/raw/gmp/, 15 Oct 2026, 7 snapshots that all four sites
were saved in, plus 15 investorgain snapshots):
- investorgain prints "--" when it has no quote. Every dash in all 15 snapshots (255 cells) is written as
  "--" with the range "0 ↓ / 0 ↑" beside it. It never printed a bold 0 as the headline value.
- ipopremium prints 0 for 63 company readings. Each time, investorgain showed the dash for the same IPO.
  It never printed 0 where investorgain showed a number.
- ipowatch prints ₹0 with an estimated listing price equal to the issue price, "(0.00%)", for 14 readings.
  Each time investorgain showed the dash.
- So the 0 on ipopremium and ipowatch is the same no-quote state as investorgain's dash, written as a number.
  Two sites printing 0 is therefore one placeholder seen twice, not two confirmations.
- No real flat premium of exactly 0 appears in the saved pages. A real quote comes with a non-zero range.
Policy (_resolve_zeros): a reading of 0 becomes None, with a note, when another site lists the same IPO as
unknown. A 0 that no other site contradicts stays 0. In the saved data only SJP Ultrasonics (ipopremium
alone) has an uncontradicted 0.

Name matching: this module returns company keys (company_key). Aggregators spell names differently, so
keys.same_company() joins them when the rules there hold. Each GmpReading carries the site's open day and
market where the site gives them, for those rules.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
from selectolax.lexbor import LexborHTMLParser, LexborNode

from ipo_radar import config
from ipo_radar.errors import SourceError
from ipo_radar.live import _http
from ipo_radar.live._util import parse_date
from ipo_radar.live.keys import company_key
from ipo_radar.schema import Sourced

RAW_DIR = config.RAW / "gmp"
RETRIES = 2
RETRY_PAUSE_S = 2.0

INVESTORGAIN_URL = "https://www.investorgain.com/report/live-ipo-gmp/331/"
IPOWATCH_URL = "https://ipowatch.in/"
IPOPREMIUM_URL = "https://www.ipopremium.in/"
IPOCENTRAL_URL = "https://ipocentral.in/wp-json/ipo-gmp/v1/data"


_UNKNOWN_TOKENS = {"", "--", "-", "NA", "N/A", "NAN", "NONE"}
_PERCENT_IN_PARENS = re.compile(r"\(\s*[-+]?\d+(?:\.\d+)?\s*%\s*\)")
_NUMBER = re.compile(r"[+-]?\d+(?:\.\d+)?")


def parse_gmp(text: str) -> float | None:
    """Parse one GMP value in INR. Unknown markers ('--', '-', 'NA', blank) give None, never 0.

    A real 0 is 0.0. Percentage text such as '(17.32%)' is dropped first, so '₹120 (17.32%)' gives 120.0.
    Raises ValueError for anything else, so a changed layout is caught rather than read as a number.
    """
    s = text.replace("₹", "").replace(",", "").replace("−", "-").strip()
    s = _PERCENT_IN_PARENS.sub("", s).strip()
    if s.upper() in _UNKNOWN_TOKENS:
        return None
    if not _NUMBER.fullmatch(s):
        raise ValueError(f"not a GMP value: {text!r}")
    return float(s)


@dataclass
class GmpReading(Sourced):
    """A Sourced reading plus what the site said about the issue: its open day and market (may be unknown).

    merge uses these to tell apart two IPOs whose names are spelled differently on different sites.
    """
    open_date: date | None = None
    market: str = ""  # IN_MAIN | IN_SME | "" when the site does not say


_DAY_MON = re.compile(r"^\s*(\d{1,2})\s*(?:-\s*\d{1,2}\s*)?-?\s*([A-Za-z]{3})")


def _open_day(text: str, year: int) -> date | None:
    """'15-Oct', '15 - 19 Oct', '21-23 Oct GMP: 500' -> the first day. Sites omit the year, so the run's year is used."""
    m = _DAY_MON.match(text)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)} {m.group(2)} {year}", "%d %b %Y").date()
        except ValueError:
            return None
    return parse_date(text.strip())  # 'Oct 15, 2026' form


def _clean_name(name: str) -> str:
    """Drop the venue tags some sites append: 'Foo Ltd (BSE SME)' -> 'Foo Ltd'."""
    return re.sub(r"\s*\((?:mainboard|bse sme|nse sme|sme|bse|nse)\)\s*$", "", name.strip(), flags=re.I)


def _get_with_retry(client: httpx.Client, site: str, url: str) -> bytes:
    """Up to RETRIES more attempts after a dropped connection, empty body or 5xx.

    403 and 429 are never retried: the site has blocked us or asked us to slow down.
    """
    last: SourceError | None = None
    for attempt in range(RETRIES + 1):
        if attempt:
            time.sleep(RETRY_PAUSE_S)
        try:
            return _http.get(client, url, site)
        except _http.HTTPStatusError as e:
            if e.status in (403, 429):
                raise
            last = e
        except SourceError as e:
            last = e
    assert last is not None
    raise last


class _Session:
    """One fetch run: the HTTP client, the run timestamp and the folder for raw copies."""

    def __init__(self, client: httpx.Client, raw_dir: Path, now: datetime) -> None:
        self.client = client
        self.raw_dir = raw_dir
        self.now = now
        self.stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def get(self, site: str, url: str, ext: str) -> str:
        """Fetch one page or JSON document, keep the raw bytes as data/raw/gmp/<site>/<stamp>.<ext>."""
        body = _get_with_retry(self.client, site, url)
        folder = self.raw_dir / site
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{self.stamp}.{ext}").write_bytes(body)
        return body.decode("utf-8", errors="replace")


def _add(out: dict[str, GmpReading], site: str, name: str, value: float | None, note: str, now: datetime, *,
         open_date: date | None = None, market: str = "") -> None:
    key = company_key(_clean_name(name))
    if not key:
        raise SourceError(f"{site}: row with no usable name")
    if key in out:
        raise SourceError(f"{site}: duplicate company {key!r}; refusing to guess which row is right")
    out[key] = GmpReading(value=value, source=site, fetched_at=now, note=note, open_date=open_date, market=market)


def _tables_with(tree: LexborHTMLParser, site: str, *headers: str) -> list[tuple[LexborNode, list[str]]]:
    """Every table whose header row starts with each wanted column name and that has at least one data row."""
    found = []
    for tb in tree.css("table"):
        rows = tb.css("tr")
        if len(rows) < 2:
            continue
        heads = [c.text(strip=True) for c in rows[0].css("th,td")]
        if all(any(h.startswith(w) for h in heads) for w in headers):
            found.append((tb, heads))
    if not found:
        raise SourceError(f"{site}: no table with columns {headers} (layout changed?)")
    return found


def _col(heads: list[str], prefix: str) -> int | None:
    return next((i for i, h in enumerate(heads) if h.startswith(prefix)), None)


def _fetch_investorgain(session: _Session) -> dict[str, GmpReading]:
    site = "investorgain"
    html = session.get(site, INVESTORGAIN_URL, "html")
    out: dict[str, GmpReading] = {}
    for tb, heads in _tables_with(LexborHTMLParser(html), site, "Name", "GMP"):
        ni, gi, ui, oi = _col(heads, "Name"), _col(heads, "GMP"), _col(heads, "Updated"), _col(heads, "Open")
        for row in tb.css("tr")[1:]:
            cells = row.css("td")
            if not cells:  # repeated header rows and spacer rows
                continue
            if len(cells) != len(heads):
                raise SourceError(f"{site}: row has {len(cells)} cells, header has {len(heads)}")
            anchor = cells[ni].css_first("a")
            # the GMP value is the bold text outside the small range line ("83 ↓ / 120 ↑")
            bold = next((b for b in cells[gi].css("b") if b.parent is not None and b.parent.tag != "small"), None)
            if anchor is None or bold is None:
                raise SourceError(f"{site}: name or GMP cell missing (layout changed?)")
            try:
                value = parse_gmp(bold.text(strip=True))
            except ValueError as e:
                raise SourceError(f"{site}: {e}") from e
            pct = _PERCENT_IN_PARENS.search(cells[gi].text(separator=" ", strip=True))
            small = cells[gi].css_first("small")
            tags = [b.text(strip=True) for b in cells[ni].css("span.badge")]
            market = "IN_SME" if any("SME" in t for t in tags) else ("IN_MAIN" if "IPO" in tags else "")
            opened = _open_day(cells[oi].text(strip=True), session.now.year) if oi is not None else None
            parts = [" ".join(tags), f"gmp {pct.group(0)[1:-1]}" if pct else "",
                     f"range {small.text(strip=True)}" if small else "",
                     f"updated {cells[ui].text(strip=True)}" if ui is not None else ""]
            _add(out, site, anchor.text(strip=True), value, "; ".join(p for p in parts if p), session.now,
                 open_date=opened, market=market)
    return out


def _fetch_ipowatch(session: _Session) -> dict[str, GmpReading]:
    site = "ipowatch"
    html = session.get(site, IPOWATCH_URL, "html")
    out: dict[str, GmpReading] = {}
    for tb, heads in _tables_with(LexborHTMLParser(html), site, "Company", "GMP"):
        gi, ei, di = _col(heads, "GMP"), _col(heads, "Est. Gain"), _col(heads, "Date")
        for row in tb.css("tr")[1:]:
            cells = row.css("td")
            if not cells:
                continue
            if len(cells) != len(heads):
                raise SourceError(f"{site}: row has {len(cells)} cells, header has {len(heads)}")
            anchor = cells[0].css_first("a")
            if anchor is None:
                raise SourceError(f"{site}: company name missing (layout changed?)")
            try:
                value = parse_gmp(cells[gi].text(strip=True))
            except ValueError as e:
                raise SourceError(f"{site}: {e}") from e
            status = cells[0].css_first("span")
            parts = [status.text(strip=True) if status else "",
                     f"est. gain {cells[ei].text(strip=True)}" if ei is not None else "",
                     f"date {cells[di].text(strip=True)}" if di is not None else ""]
            # this site's two tables do not say which is SME, so the market stays unknown
            opened = _open_day(cells[di].text(strip=True), session.now.year) if di is not None else None
            _add(out, site, anchor.text(strip=True), value, "; ".join(p for p in parts if p), session.now,
                 open_date=opened)
    return out


def _fetch_ipopremium(session: _Session) -> dict[str, GmpReading]:
    site = "ipopremium"
    html = session.get(site, IPOPREMIUM_URL, "html")
    out: dict[str, GmpReading] = {}
    for tb, heads in _tables_with(LexborHTMLParser(html), site, "Company", "GMP"):
        ni, gi = _col(heads, "Company"), _col(heads, "GMP")
        ti, bi, li, oi = _col(heads, "Type"), _col(heads, "Price Band"), _col(heads, "Listing"), _col(heads, "Open")
        for row in tb.css("tr")[1:]:
            cells = row.css("td")
            if not cells:
                continue
            if len(cells) != len(heads):
                raise SourceError(f"{site}: row has {len(cells)} cells, header has {len(heads)}")
            try:
                value = parse_gmp(cells[gi].text(strip=True))
            except ValueError as e:
                raise SourceError(f"{site}: {e}") from e
            kind = cells[ti].text(strip=True).upper() if ti is not None else ""
            market = {"EQ": "IN_MAIN", "SME": "IN_SME"}.get(kind, "")
            opened = _open_day(cells[oi].text(strip=True), session.now.year) if oi is not None else None
            parts = [f"type {kind}" if ti is not None else "",
                     f"band {cells[bi].text(strip=True)}" if bi is not None else "",
                     f"listing {cells[li].text(strip=True)}" if li is not None else ""]
            _add(out, site, cells[ni].text(strip=True), value, "; ".join(p for p in parts if p), session.now,
                 open_date=opened, market=market)
    return out


def _fetch_ipocentral(session: _Session) -> dict[str, GmpReading]:
    site = "ipocentral"
    raw = session.get(site, IPOCENTRAL_URL, "json")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SourceError(f"{site}: response is not JSON") from e
    if not isinstance(data, dict) or not isinstance(data.get("MB"), list) or not isinstance(data.get("SME"), list):
        raise SourceError(f"{site}: unexpected JSON shape (layout changed?)")
    site_stamp = data.get("updated")
    out: dict[str, GmpReading] = {}
    for section in ("MB", "SME"):
        for item in data[section]:
            if not isinstance(item, dict) or "name" not in item or "num" not in item:
                raise SourceError(f"{site}: item without name/num (layout changed?)")
            try:
                value = parse_gmp(str(item["num"]))
            except ValueError as e:
                raise SourceError(f"{site}: {e}") from e
            parts = [section, f"gmp {item.get('pct', '')}".strip(), f"date {item.get('date', '')}".strip()]
            if isinstance(site_stamp, int):
                parts.append(f"site updated {datetime.fromtimestamp(site_stamp, timezone.utc):%Y-%m-%d %H:%M} UTC")
            market = {"MB": "IN_MAIN", "SME": "IN_SME"}[section]
            opened = _open_day(str(item.get("date", "")), session.now.year)
            _add(out, site, str(item["name"]), value, "; ".join(p for p in parts if p), session.now,
                 open_date=opened, market=market)
    if not out:
        raise SourceError(f"{site}: no IPOs in response")
    return out


_SOURCES: tuple[tuple[str, Callable[[_Session], dict[str, GmpReading]]], ...] = (
    ("investorgain", _fetch_investorgain),
    ("ipowatch", _fetch_ipowatch),
    ("ipopremium", _fetch_ipopremium),
    ("ipocentral", _fetch_ipocentral),
)


def _resolve_zeros(readings: dict[str, list[GmpReading]]) -> None:
    """Zero policy (evidence in the module docstring). A 0 is not a reading when another site shows a dash for
    the same IPO: it becomes None with a note. A 0 that no other site contradicts is kept as 0."""
    for items in readings.values():
        for r in items:
            if r.value != 0:
                continue
            dashes = [o.source for o in items if o is not r and o.value is None]
            if dashes:
                r.value = None
                note = f"site shows 0 where another shows no data ({', '.join(dashes)})"
                r.note = f"{r.note}; {note}" if r.note else note


def fetch(client: httpx.Client | None = None, raw_dir: Path = RAW_DIR,
          now: datetime | None = None) -> dict[str, list[Sourced]]:
    """Company key -> one reading per aggregator that answered (value None = site shows it unknown).

    Readings are GmpReading (a Sourced). Raises SourceError only when every source fails; the message lists each failure.
    """
    now = now or datetime.now(timezone.utc)
    own_client = client is None
    if client is None:
        client = _http.make_client()
    session = _Session(client, raw_dir, now)
    out: dict[str, list[GmpReading]] = {}
    failures: list[str] = []
    try:
        for _, fn in _SOURCES:
            try:
                readings = fn(session)
            except SourceError as e:
                failures.append(str(e))
                continue
            for key, reading in readings.items():
                out.setdefault(key, []).append(reading)
    finally:
        if own_client:
            client.close()
    if len(failures) == len(_SOURCES):
        raise SourceError("GMP unavailable: " + "; ".join(failures))
    _resolve_zeros(out)
    return out


# ---- history: lets the alert engine detect GMP jumps ----------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS gmp_history (
    ts TEXT NOT NULL,            -- UTC ISO 8601, fetch time
    company_key TEXT NOT NULL,
    source TEXT NOT NULL,
    gmp REAL NOT NULL            -- INR; unknown readings are never stored
);
CREATE INDEX IF NOT EXISTS gmp_history_key ON gmp_history (company_key, ts);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def record_readings(readings: dict[str, list[Sourced]], db_path: Path = config.STATE_DB) -> int:
    """Append known readings to gmp_history. Unknown (None) readings are skipped, not stored as 0."""
    rows = []
    for key, items in readings.items():
        for r in items:
            if r.value is None:
                continue
            if r.fetched_at is None or r.fetched_at.tzinfo is None:
                raise ValueError(f"reading for {key!r} from {r.source!r} has no timezone-aware fetched_at")
            rows.append((r.fetched_at.astimezone(timezone.utc).isoformat(), key, r.source, r.value))
    if rows:
        with closing(_connect(db_path)) as conn, conn:
            conn.executemany("INSERT INTO gmp_history (ts, company_key, source, gmp) VALUES (?, ?, ?, ?)", rows)
    return len(rows)


def history(company_key: str, db_path: Path = config.STATE_DB) -> list[tuple[datetime, float, str]]:
    """(fetched_at, gmp INR, source) for one IPO, oldest first. Empty if nothing was recorded."""
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT ts, gmp, source FROM gmp_history WHERE company_key = ? ORDER BY ts, rowid", (company_key,)
        ).fetchall()
    return [(datetime.fromisoformat(ts), gmp, source) for ts, gmp, source in rows]
