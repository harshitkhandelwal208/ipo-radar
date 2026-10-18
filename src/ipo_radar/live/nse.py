"""NSE India public JSON: active and upcoming issues, per-category bids, recent listings.

Why: NSE is the exchange itself, so its subscription figures are the reference when sources disagree.

robots.txt (checked 2026-10-15): `User-agent: *`, Allow: /, Disallow: /market-data-test only. The
/api/* JSON used here is not disallowed. NSE's website terms were not reviewed. Use is personal and
rate-limited to one request per second. NSE sets session cookies on its IPO page; the API needs them, so
that page is loaded first.

Endpoints found in the NSE IPO pages' JS: ipo-current-issue (active issues, one 'Total' row),
all-upcoming-issues?category=ipo (dates, price band), ipo-bid-details?symbol=&series= (per category
times subscribed), public-past-issues (listing dates, price range). The 'forthcoming' categories were
empty when checked.
"""
from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timedelta, timezone

import httpx

from ipo_radar.errors import SourceError
from ipo_radar.live import _http
from ipo_radar.live._http import HTTPStatusError
from ipo_radar.live._util import parse_date, parse_price_band
from ipo_radar.live.keys import company_key
from ipo_radar.schema import LiveIPO, Sourced

SOURCE = "nse"
BASE = "https://www.nseindia.com"
# The root page answers 403 to scripted clients, the IPO page does not and sets the same session cookies.
REFERER = f"{BASE}/market-data/all-upcoming-issues-ipo"
CURRENT = "/api/ipo-current-issue"
UPCOMING = "/api/all-upcoming-issues?category=ipo"
PAST = "/api/public-past-issues"
BIDS = "/api/ipo-bid-details?symbol={symbol}&series={series}"
PAST_WINDOW_DAYS = 14             # only issues that closed recently matter for live status
PAST_SERIES = frozenset({"EQ", "SME", "BE"})  # other securityType values are bonds or other instruments
_TOP_LEVEL_SR = re.compile(r"^\d+(?:\.\d+)?$")  # '1', '2', '2.1' are reported rows; '1(a)' are sub-rows


def _classify(sr: str | None, category: str) -> str | None:
    """Map one bid-details row to a schema key. Sub-rows (FIIs, Cut Off, ...) and unknown rows map to None."""
    if category == "Total" and sr is None:
        return "sub_total"
    if not sr or not _TOP_LEVEL_SR.match(sr):
        return None
    if sr == "1" and "Qualified" in category:
        return "sub_qib"
    if sr == "2" and "Non Institutional" in category:
        return "sub_nii"
    if sr == "2.1" and "Non Institutional" in category:
        return "sub_bnii"
    if sr == "2.2" and "Non Institutional" in category:
        return "sub_snii"
    if sr == "3" and "Retail" in category:
        return "sub_retail"
    if sr == "4" and "Employee" in category:
        return "sub_employee"
    if "Shareholder" in category:
        return "sub_shareholder"
    return None


IST = timezone(timedelta(hours=5, minutes=30))  # NSE's own clock; its updateTime carries no zone
_UPDATED = re.compile(r"(\d{1,2}-[A-Za-z]{3}-\d{4})\s+(\d{1,2}:\d{2}:\d{2})")


def _sampled_at(update: object, fallback: datetime) -> datetime:
    """When NSE sampled the bids: its updateTime ('Updated as on 15-Oct-2026 10:48:40 hrs'), else the fetch time."""
    m = _UPDATED.search(update) if isinstance(update, str) else None
    if m:
        try:
            return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%d-%b-%Y %H:%M:%S").replace(tzinfo=IST)
        except ValueError:
            pass
    return fallback


def _times(raw: object, where: str) -> float | None:
    """noOfTime as a float. Absent or blank means not reported (None), never 0."""
    if raw in (None, ""):
        return None
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise SourceError(f"nse: non-numeric subscription in {where}: {raw!r} (layout changed?)") from exc
    if not math.isfinite(value) or value < 0:
        raise SourceError(f"nse: implausible subscription in {where}: {raw!r}")
    return value


def parse_bid_details(payload: object, symbol: str, fetched_at: datetime) -> dict[str, Sourced]:
    """Per-category subscription from ipo-bid-details. Missing categories are absent from the dict."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise SourceError(f"nse: bid-details for {symbol} has no 'data' list (layout changed?)")
    update = payload.get("updateTime")
    sampled = _sampled_at(update, fetched_at)
    out: dict[str, Sourced] = {}
    for row in payload["data"]:
        if not isinstance(row, dict):
            raise SourceError(f"nse: bid-details row for {symbol} is not an object")
        category = str(row.get("category") or "")
        if category == "Category":  # the header row repeats the field names
            continue
        key = _classify(row.get("srNo"), category)
        if key is None or key in out:
            continue
        value = _times(row.get("noOfTime"), f"bid-details {symbol}/{category}")
        if value is not None:
            out[key] = Sourced(value, SOURCE, sampled, note=f"{category}; {update}")
    return out


def _series_market(series: str) -> str:
    """EQ is India main board, SME is SME. BE (trade-for-trade) does not say which, so it is left unknown."""
    return {"EQ": "IN_MAIN", "SME": "IN_SME"}.get(series, "")


def _name(row: dict) -> str:
    name = row.get("companyName") or row.get("company")
    if not isinstance(name, str) or not name.strip():
        raise SourceError("nse: issue row without a company name (layout changed?)")
    return name.strip()


def _final_price(raw: object) -> float | None:
    """A single price such as '82'. Anything else (blank, '-', a range) is not a final price."""
    if not isinstance(raw, str) or not re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", raw):
        return None
    return float(raw)


def _issue_row(row: dict) -> LiveIPO:
    name = _name(row)
    series = str(row.get("series") or row.get("securityType") or "")  # past issues use securityType
    market = _series_market(series)
    # past issues carry the band in priceRange and a single price in issuePrice; current ones only issuePrice
    low, high = parse_price_band(row.get("priceRange") or row.get("issuePrice"))
    # past issues state the final price in issuePrice ('82'); a current band like 'Rs.665 to Rs.700' is not one
    final = _final_price(row.get("issuePrice")) if "priceRange" in row else None
    ipo = LiveIPO(
        ipo_id=company_key(name), market=market, company=name, symbol=str(row.get("symbol") or ""), issue_price=final,
        open_date=parse_date(row.get("issueStartDate") or row.get("ipoStartDate")),
        close_date=parse_date(row.get("issueEndDate") or row.get("ipoEndDate")),
        price_low=low, price_high=high, sources=[SOURCE],
    )
    return ipo


def _fetch_bids(client: httpx.Client, ipo: LiveIPO, fetched_at: datetime) -> None:
    """Category detail for one active issue. A failure here only costs that IPO its category split."""
    series = "SME" if ipo.market == "IN_SME" else "EQ"
    url = BASE + BIDS.format(symbol=ipo.symbol, series=series)
    try:
        body = _http.get(client, url, SOURCE, headers=_api_headers())
    except SourceError as exc:
        ipo.warnings.append(f"nse category split unavailable: {exc}")
        return
    try:
        payload = json.loads(body)
        ipo.sub.update(parse_bid_details(payload, ipo.symbol, fetched_at))
    except (ValueError, SourceError) as exc:
        ipo.warnings.append(f"nse category split unreadable: {exc}")
    _http.save_raw(SOURCE, f"bid-{_safe(ipo.symbol)}.json", body)


def _safe(symbol: str) -> str:
    """File-name-safe symbol; NSE symbols are alphanumeric, so anything else is replaced."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", symbol) or "unnamed"


def _api_headers() -> dict[str, str]:
    return {"Accept": "application/json, text/plain, */*", "Referer": REFERER}


def _warm(client: httpx.Client) -> None:
    """Load the IPO page so NSE sets its session cookies. The API returns 403 without them."""
    _http.get(client, REFERER, SOURCE, headers={"Accept": "text/html"})


def _api_json(client: httpx.Client, path: str, *, retry_cookies: bool = True) -> object:
    try:
        body = _http.get(client, BASE + path, SOURCE, headers=_api_headers())
    except HTTPStatusError as exc:
        if retry_cookies and exc.status in (401, 403):
            _warm(client)  # the session cookie expired or was never set
            return _api_json(client, path, retry_cookies=False)
        raise
    try:
        return json.loads(body)
    except ValueError as exc:
        raise SourceError(f"nse: {path} did not return JSON (blocked or changed?)") from exc


def _parse_current(payload: object) -> dict[str, list[dict]]:
    """Group current-issue rows by symbol. The 'Total' row (or the only row) is the issue's summary."""
    if not isinstance(payload, list):
        raise SourceError("nse: ipo-current-issue is not a list (layout changed?)")
    grouped: dict[str, list[dict]] = {}
    for row in payload:
        if not isinstance(row, dict):
            raise SourceError("nse: ipo-current-issue row is not an object")
        grouped.setdefault(str(row.get("symbol") or _name(row)), []).append(row)
    return grouped


def fetch(client: httpx.Client | None = None, today: date | None = None) -> list[LiveIPO]:
    """Current, upcoming and recently listed NSE issues. Raises SourceError only if the feed itself fails."""
    owns = client is None
    client = client or _http.make_client()
    try:
        return _fetch(client, today or date.today())
    finally:
        if owns:
            client.close()


def _fetch(client: httpx.Client, today: date) -> list[LiveIPO]:
    fetched_at = datetime.now().astimezone()
    _warm(client)
    current_raw = _api_json(client, CURRENT)
    _http.save_raw(SOURCE, "ipo-current-issue.json", json.dumps(current_raw).encode())
    upcoming_raw = _api_json(client, UPCOMING)
    past_raw = _api_json(client, PAST)
    _http.save_raw(SOURCE, "public-past-issues.json", json.dumps(past_raw).encode())

    out: dict[str, LiveIPO] = {}
    for symbol, rows in _parse_current(current_raw).items():
        summary = next((r for r in rows if r.get("category") in (None, "Total")), rows[0])
        ipo = _issue_row(summary)
        total = _times(summary.get("noOfTime"), symbol)
        if total is not None:
            ipo.sub["sub_total"] = Sourced(total, SOURCE, fetched_at)
        out[symbol] = ipo
    for symbol in list(out):
        _fetch_bids(client, out[symbol], fetched_at)

    if not isinstance(upcoming_raw, list):
        raise SourceError("nse: all-upcoming-issues is not a list (layout changed?)")
    for row in upcoming_raw:
        if isinstance(row, dict):
            key = str(row.get("symbol") or _name(row))
            out.setdefault(key, _issue_row(row))

    if not isinstance(past_raw, list):
        raise SourceError("nse: public-past-issues is not a list (layout changed?)")
    cutoff = today - timedelta(days=PAST_WINDOW_DAYS)
    for row in past_raw:
        if not isinstance(row, dict) or str(row.get("securityType")) not in PAST_SERIES:
            continue
        end = parse_date(row.get("ipoEndDate"))
        if end is None or end < cutoff:
            continue
        key = str(row.get("symbol") or _name(row))
        if key in out:
            continue
        ipo = _issue_row(row)
        ipo.listing_date = parse_date(row.get("listingDate"))
        out[key] = ipo
    return list(out.values())
