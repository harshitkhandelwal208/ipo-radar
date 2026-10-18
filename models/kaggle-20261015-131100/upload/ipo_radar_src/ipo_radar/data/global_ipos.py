"""Non-India IPO deals (US, HK, OTHER). Writes data/processed/global.parquet.

Sources and their verdicts are in docs/data_sources/global.md. Summary:
- US 2024-2026: IPO Scoop priced-IPO tables (https://www.iposcoop.com/...-pricings/). robots.txt allows the pages
  (only /wp-admin/ disallowed). No terms page found, so the licence is unstated. Used for local modelling only.
  Its "Return" column is measured against the current price, so the target is recomputed from the 1st-day close.
- US 2010-2018: Kaggle thedevastator/dataset-on-ipo-from-2010-2018 (CC0): offer price, first-day open and close.
  Kaggle proselotis/financial-ipo-data (CC0): first-day close only (openDay0 is wrong), used as a cross-check.
- US 2019-2021: Kaggle shivamb/company-ipos-2019-2021 (CC0): symbol, IPO date and offer price, no price history.
  The first-day OHLC comes from Kaggle benjaminpo/sp500-spx-components-stock-price (CC0, per-ticker daily prices to
  2022-02). Joined on symbol with an EXACT first-row date match only. Survivorship: this price set includes
  delisted tickers, but a deal whose ticker has no file is not joined, so joined rows skew to names that
  later had price files.
- HK 2020-2024: Kaggle ng3412/hkex-ipo-dataset (MIT): offer price, public-offer multiple, proceeds. First-day
  OHLC from Kaggle eren2222/hongkong-daily-historical-stock-market-2000-2024 (Apache-2.0), exact-date join only.
  Only HK deals with a join are kept (a row without a target is useless). Survivorship as above.
- OTHER (Indonesia, 2022-2023): Kaggle leanderangelo/indonesia-ipo-jan2022-july2023 (CC-BY-NC-SA-4.0, personal use):
  Starting_Price read as the IPO offer price (unverified), Day-1 as the first-day close.
- Rejected here (see docs): Nasdaq API (robots disallows all), stockanalysis.com (terms ban bulk use and ML),
  stooq (robots disallows all), Ritter IPOALL/IPO-age (no per-deal price), fahmirk e-IPO (return basis unknown).

Price data caveat: benjaminpo, eren2222 and the proselotis/thedevastator adjusted closes are split-adjusted, and some
are dividend-adjusted. A first-day close is only the raw price when no split followed the listing; the dataset
gives no split list, so a close whose ratio to the offer price is about 1/k or k (k in 2,3,4,5,10,20) is blanked and
counted. Any other disagreement between two sources on the first-day close (more than 0.5%) blanks the close gain.
"""
from __future__ import annotations

import argparse
import re
import time
import urllib.robotparser
from collections import Counter
from datetime import date, datetime

import httpx
import pandas as pd
from selectolax.lexbor import LexborHTMLParser as HTMLParser

from ipo_radar import config, schema
from ipo_radar.errors import SourceError

BASE = "https://www.iposcoop.com"
# cache file name -> page path. The "current year" page is the only one that holds the year-to-date deals.
PAGES = {
    "pricings-2024.html": "/2024-pricings/",
    "pricings-2025.html": "/2025-pricings/",
    "current-year-pricings.html": "/current-year-pricings/",
}
RAW_DIR = config.RAW / "iposcoop"
EXPECTED_HEADER = ["Company", "Symbol", "Industry", "Offer Date", "Shares (millions)", "Offer Price",
                   "1st Day Close", "Current Price", "Return", "SCOOP Rating"]
KAGGLE = config.RAW / "kaggle"
TD_CSV = KAGGLE / "thedevastator__dataset-on-ipo-from-2010-2018" / "ipo_stock_2010_2018.csv"
PRO_CSV = KAGGLE / "proselotis__financial-ipo-data" / "IPODataFull.csv"
SHIV_CSV = KAGGLE / "shivamb__company-ipos-2019-2021" / "company_ipo.csv"
BJ_DIR = KAGGLE / "benjaminpo__sp500-spx-components-stock-price" / "data" / "1d"
JACKSON_DIR = KAGGLE / "jacksoncrow__stock-market-dataset" / "stocks"   # Nasdaq tickers, daily to 2020-04
HK_IPO_XLSX = KAGGLE / "ng3412__hkex-ipo-dataset" / "hkex_ipo_ds.xlsx"
HK_PX_CSV = KAGGLE / "eren2222__hongkong-daily-historical-stock-market-2000-2024" / "HK Stock Market 2000-2024.csv"
ID_CSV = KAGGLE / "leanderangelo__indonesia-ipo-jan2022-july2023" / "IPO 2023 Analysis - Cleaned.csv"

GAIN_MIN, GAIN_MAX = -95.0, 2000.0           # first-day gain sanity band, percent
UNIT_SYMBOL = re.compile(r"^[A-Z]{4}[URW]$")  # Nasdaq 5-letter convention: U units, W warrants, R rights
SPAC_NAME = re.compile(r"acquisition|blank check|\bspac\b|merger corp|capital investment|opportunity corp", re.I)
SPLIT_FACTORS = (2, 3, 4, 5, 10, 20)
SPLIT_TOL = 0.01                              # relative tolerance around 1/k and k
DISAGREE_TOL = 0.005                          # two sources differ by more than 0.5% on the first-day close
CORE_COLS = ["company", "symbol", "listing_date", "issue_price", "list_close_gain_pct"]
TARGETS = ["list_open_gain_pct", "list_high_gain_pct", "list_low_gain_pct", "list_close_gain_pct"]
MIN_DELAY_S = 1.0
TIMEOUT_S = 30.0
_last_request = 0.0


# ---- polite fetching of the IPO Scoop pages ------------------------------------------------------

def _get(url: str, client: httpx.Client) -> str:
    """One polite GET (>= 1 s apart). Every failure becomes a SourceError; no partial bodies pass through."""
    global _last_request
    wait = MIN_DELAY_S - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    try:
        resp = client.get(url)
    except httpx.TimeoutException as exc:
        raise SourceError(f"timeout fetching {url}: {exc}") from exc
    except httpx.HTTPError as exc:
        raise SourceError(f"network error fetching {url}: {exc}") from exc
    finally:
        _last_request = time.monotonic()
    if resp.status_code != 200:
        raise SourceError(f"HTTP {resp.status_code} for {url}")
    if not resp.text.strip():
        raise SourceError(f"empty body for {url}")
    return resp.text


def _check_robots(client: httpx.Client, url: str) -> None:
    """Refuse to fetch a page that robots.txt disallows for our user agent."""
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(_get(BASE + "/robots.txt", client).splitlines())
    if not rp.can_fetch(config.USER_AGENT, url):
        raise SourceError(f"robots.txt disallows {url}; not fetching")


def load_pages(refresh: bool = False, client: httpx.Client | None = None) -> dict[str, str]:
    """Return {cache name: html}. Uses the on-disk cache unless `refresh` is set; only cache misses hit the network."""
    pages: dict[str, str] = {}
    own_client = client is None
    if own_client:
        client = httpx.Client(headers={"User-Agent": config.USER_AGENT}, timeout=TIMEOUT_S, follow_redirects=True)
    try:
        for name, path in PAGES.items():
            cached = RAW_DIR / name
            if cached.exists() and cached.stat().st_size > 0 and not refresh:
                pages[name] = cached.read_text(encoding="utf-8")
                continue
            _check_robots(client, BASE + path)
            html = _get(BASE + path, client)
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            cached.write_text(html, encoding="utf-8")
            pages[name] = html
    finally:
        if own_client:
            client.close()
    return pages


def parse_pricings(html: str, source_url: str) -> list[dict]:
    """Return raw string rows from the priced-IPO table. A changed layout raises SourceError."""
    table = None
    for t in HTMLParser(html).css("table"):
        if [th.text(strip=True) for th in t.css("thead th")] == EXPECTED_HEADER:
            table = t
            break
    if table is None:
        raise SourceError(f"IPO Scoop layout changed: no table with header {EXPECTED_HEADER} in {source_url}")
    rows = []
    for tr in table.css("tbody tr"):
        cells = [td.text(strip=True) for td in tr.css("td")]
        if len(cells) != len(EXPECTED_HEADER):
            raise SourceError(f"IPO Scoop row has {len(cells)} cells, expected {len(EXPECTED_HEADER)}: {source_url}")
        rows.append({"company": cells[0], "symbol": cells[1], "industry": cells[2], "offer_date": cells[3],
                     "shares_m": cells[4], "offer_price": cells[5], "close_1d": cells[6], "source_url": source_url})
    if not rows:
        raise SourceError(f"IPO Scoop table has no rows: {source_url}")
    return rows


def _money(text: str) -> float | None:
    """'$16.00' -> 16.0. Blank or non-numeric is None (missing), never 0."""
    text = text.replace("$", "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _date(text: str, source_url: str) -> date:
    try:
        return datetime.strptime(text, "%m/%d/%Y").date()
    except ValueError as exc:
        raise SourceError(f"IPO Scoop date format changed ({text!r}) in {source_url}") from exc


# ---- shared rules ----------------------------------------------------------------------------------

def _split_like(ratio: float) -> bool:
    """True when close/offer is about 1/k or k: the signature of a later split in a split-adjusted series."""
    for k in SPLIT_FACTORS:
        if abs(ratio - 1 / k) <= SPLIT_TOL / k or abs(ratio - k) <= SPLIT_TOL * k:
            return True
    return False


def _gain(price: float | None, issue: float) -> float | None:
    if price is None or issue <= 0:
        return None
    return (price / issue - 1.0) * 100.0


def _us_spac_or_unit(name: str, symbol: str) -> str | None:
    """Reason to exclude a US record as a SPAC or unit/warrant/right, or None."""
    if SPAC_NAME.search(name or ""):
        return "spac"
    if UNIT_SYMBOL.match(symbol.upper()):
        return "unit_warrant_right"
    return None


def _targets_from_prices(issue: float, o: float | None, h: float | None, lo: float | None, c: float | None,
                         report: Counter) -> dict:
    """Gains from first-day prices. A close whose ratio looks like a split blanks all four targets."""
    if c is not None and issue > 0 and _split_like(c / issue):
        report["suspect_split_factor_blanked"] += 1
        return {t: None for t in TARGETS}
    return {"list_open_gain_pct": _gain(o, issue), "list_high_gain_pct": _gain(h, issue),
            "list_low_gain_pct": _gain(lo, issue), "list_close_gain_pct": _gain(c, issue)}


# ---- IPO Scoop (US 2024-2026) ----------------------------------------------------------------------

def build_from_html(pages: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """Turn the saved pricing pages into schema rows. Returns (frame, report with exclusion counts)."""
    report: Counter = Counter()
    recs: list[dict] = []
    for name, html in pages.items():
        url = BASE + PAGES[name]
        for raw in parse_pricings(html, url):
            report["raw_rows"] += 1
            symbol = raw["symbol"].upper()
            if raw["industry"] == "Blank Check":
                report["excluded_spac"] += 1
                continue
            if UNIT_SYMBOL.match(symbol):
                report["excluded_unit_warrant_right_by_symbol"] += 1
                continue
            price = _money(raw["offer_price"])
            close = _money(raw["close_1d"])
            if price is None:
                report["excluded_no_offer_price"] += 1
                continue
            if close is None or close == 0:
                report["excluded_not_traded_yet"] += 1  # the source shows $0.00 before the first trade
                continue
            listing = _date(raw["offer_date"], url)
            shares = _money(raw["shares_m"])
            tg = _targets_from_prices(price, None, None, None, close if price > 0 else None, report)
            recs.append({
                "ipo_id": f"US:{symbol.lower()}:{listing.isoformat()}",
                "market": "US", "country": "US", "exchange": None,
                "company": raw["company"], "symbol": symbol, "sector": raw["industry"],
                "listing_date": listing,
                "currency": "USD", "issue_price": price,
                "issue_size_m": shares * price if shares is not None else None,
                "issue_size_usd_m": shares * price if shares is not None else None,
                **tg,
                "source": "iposcoop", "source_url": raw["source_url"],
            })
    frame = pd.DataFrame(recs)
    if frame.empty:
        raise SourceError("no usable IPO rows after exclusions")
    frame = _sanity(frame, report)
    frame = _dedupe(frame, report)
    report["kept"] = len(frame)
    return frame, dict(report)


# ---- Kaggle loaders -------------------------------------------------------------------------------

def _read_csv(path, usecols=None, **kwargs) -> pd.DataFrame:
    """read_csv that turns a missing column (a changed export) into SourceError instead of a pandas ValueError."""
    try:
        return pd.read_csv(path, usecols=usecols, **kwargs)
    except ValueError as exc:
        raise SourceError(f"{path.name}: columns do not match the expected layout ({exc})") from exc


def _require(df: pd.DataFrame, cols: list[str], path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SourceError(f"{path.name}: expected columns missing {missing}; layout changed")


def load_thedevastator(path=TD_CSV) -> tuple[pd.DataFrame, Counter]:
    """US 2010-2018 deals with first-day open and adjusted close. Listing date = 'Date Priced' (checked: it equals the
    first trading row in benjaminpo for 439 of 505 matched symbols)."""
    cols = ["Company Name", "Symbol", "Market", "Price", "Offer Amount", "Date Priced", "sector",
            "firstday_open", "firstday_adjclose"]
    df = _read_csv(path, usecols=cols)
    _require(df, cols, path)
    report: Counter = Counter({"td_raw_rows": len(df)})
    recs = []
    for _, r in df.iterrows():
        symbol = str(r["Symbol"]).strip().upper()
        if (reason := _us_spac_or_unit(str(r["Company Name"]), symbol)):
            report[f"td_excluded_{reason}"] += 1
            continue
        price = r["Price"]
        if pd.isna(price) or price <= 0:
            report["td_excluded_no_offer_price"] += 1
            continue
        try:
            listing = datetime.strptime(str(r["Date Priced"])[:10], "%Y-%m-%d").date()
        except ValueError as exc:
            raise SourceError(f"thedevastator date format changed ({r['Date Priced']!r})") from exc
        exch = str(r["Market"])
        if "NASDAQ" in exch.upper():
            exchange = "NASDAQ"
        elif "AMERICAN" in exch.upper() or "MKT" in exch.upper():
            exchange = "NYSE American"
        else:
            exchange = "NYSE"
        tg = _targets_from_prices(float(price), _f(r["firstday_open"]), None, None, _f(r["firstday_adjclose"]), report)
        recs.append({
            "ipo_id": f"US:{symbol.lower()}:{listing.isoformat()}", "market": "US", "country": "US",
            "exchange": exchange, "company": r["Company Name"], "symbol": symbol, "sector": r["sector"],
            "listing_date": listing, "currency": "USD", "issue_price": float(price),
            "issue_size_m": _f(r["Offer Amount"]) / 1e6 if _f(r["Offer Amount"]) is not None else None,
            "issue_size_usd_m": _f(r["Offer Amount"]) / 1e6 if _f(r["Offer Amount"]) is not None else None,
            **tg, "source": "kaggle:thedevastator", "source_url": "https://www.kaggle.com/datasets/thedevastator/dataset-on-ipo-from-2010-2018",
        })
    return pd.DataFrame(recs), report


def _f(v) -> float | None:
    """A float or None. NaN is missing, never 0."""
    if v is None or pd.isna(v):
        return None
    return float(v)


def load_proselotis_closes(path=PRO_CSV) -> pd.DataFrame:
    """Cross-check observations only: (symbol, listing_date, close). openDay0 and the high/low columns are not used
    because the catalog found openDay0 wrong (FB, TWTR)."""
    df = _read_csv(path, usecols=["Symbol", "Year", "Month", "Day", "closeDay0"], encoding="latin-1")
    _require(df, ["Symbol", "Year", "Month", "Day", "closeDay0"], path)
    d = pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day), errors="coerce")
    out = pd.DataFrame({"symbol": df["Symbol"].astype(str).str.upper(), "listing_date": d.dt.date,
                        "close": pd.to_numeric(df["closeDay0"], errors="coerce")})
    return out.dropna(subset=["listing_date", "close"])


def load_shivamb(path=SHIV_CSV) -> tuple[pd.DataFrame, Counter]:
    """US 2019-2021 IPOs with offer price. SPAC and units are excluded (name and symbol rules)."""
    df = _read_csv(path, usecols=["IPO Date", "Symbol", "Company Name", "IPO Price"])
    _require(df, ["IPO Date", "Symbol", "Company Name", "IPO Price"], path)
    report: Counter = Counter({"shivamb_raw_rows": len(df)})
    recs = []
    for _, r in df.iterrows():
        symbol = str(r["Symbol"]).strip().upper()
        if (reason := _us_spac_or_unit(str(r["Company Name"]), symbol)):
            report[f"shivamb_excluded_{reason}"] += 1
            continue
        price = _money(str(r["IPO Price"]))
        if price is None or price <= 0:
            report["shivamb_excluded_no_offer_price"] += 1
            continue
        try:
            listing = datetime.strptime(str(r["IPO Date"]).strip(), "%b %d, %Y").date()
        except ValueError as exc:
            raise SourceError(f"shivamb date format changed ({r['IPO Date']!r})") from exc
        recs.append({"symbol": symbol, "company": r["Company Name"], "listing_date": listing, "issue_price": price})
    return pd.DataFrame(recs), report


PRICE_SETS = (("jacksoncrow", JACKSON_DIR), ("benjaminpo", BJ_DIR))  # tried in this order for US joins


def _first_row(directory, symbol: str, cache: dict) -> dict | None:
    """First trading row (first non-blank Close) of a per-ticker daily file, or None when the ticker has no file."""
    key = (str(directory), symbol.replace(".", "-"))
    if key in cache:
        return cache[key]
    path = directory / f"{key[1]}.csv"
    row = None
    if path.exists():
        px = _read_csv(path, usecols=["Date", "Open", "High", "Low", "Close"]).dropna(subset=["Close"])
        if not px.empty:
            r = px.iloc[0]
            row = {"date": datetime.strptime(str(r["Date"])[:10], "%Y-%m-%d").date(), "open": _f(r["Open"]),
                   "high": _f(r["High"]), "low": _f(r["Low"]), "close": _f(r["Close"])}
    cache[key] = row
    return row


def us_exact_first_row(symbol: str, listing: date, cache: dict) -> tuple[dict | None, str]:
    """(first row, price set name) for the first price set whose first trading row is exactly `listing`.

    Returns (None, reason) when no set has a file for the ticker, or every file starts on another date.
    """
    seen_file = False
    for name, directory in PRICE_SETS:
        fr = _first_row(directory, symbol, cache)
        if fr is None:
            continue
        seen_file = True
        if fr["date"] == listing:
            return {**fr, "set": name}, name
    return None, ("no_exact_first_row" if seen_file else "no_price_file")


def load_shivamb_joined(deals: pd.DataFrame, report: Counter) -> pd.DataFrame:
    """Join IPO deals to first trading rows. Exact-date rule: the first trading row must be the listed date."""
    cache: dict = {}
    recs = []
    for _, d in deals.iterrows():
        fr, why = us_exact_first_row(d["symbol"], d["listing_date"], cache)
        if fr is None:
            report[f"shivamb_{why}"] += 1
            continue
        report[f"shivamb_joined_via_{fr['set']}"] += 1
        tg = _targets_from_prices(float(d["issue_price"]), fr["open"], fr["high"], fr["low"], fr["close"], report)
        if tg["list_close_gain_pct"] is None and tg["list_open_gain_pct"] is None:
            report["shivamb_joined_no_target"] += 1
            continue
        recs.append({
            "ipo_id": f"US:{d['symbol'].lower()}:{d['listing_date'].isoformat()}", "market": "US", "country": "US",
            "exchange": None, "company": d["company"], "symbol": d["symbol"], "sector": None,
            "listing_date": d["listing_date"], "currency": "USD", "issue_price": float(d["issue_price"]),
            **tg, "source": "kaggle:shivamb+px_join",
            "source_url": "https://www.kaggle.com/datasets/shivamb/company-ipos-2019-2021 + "
                          "https://www.kaggle.com/datasets/benjaminpo/sp500-spx-components-stock-price",
        })
        report["shivamb_joined"] += 1
    return pd.DataFrame(recs)


# ---- Hong Kong (ng3412 deals joined to eren2222 prices) -------------------------------------------

def load_hk_deals(path=HK_IPO_XLSX) -> pd.DataFrame:
    """HK IPOs with offer price, public-offer multiple (sub_retail), gross proceeds (HKD m) and lot size."""
    cols = ["stock_id", "company_name", "listing_date", "pricing_final_offer_price_source",
            "hk_public_offering_subscription_level_times_source", "offer_size_gross_proceeds_hkd_m",
            "offer_size_board_lot_size"]
    df = pd.read_excel(path)
    _require(df, cols, path)
    return df[cols]


def load_hk_first_rows(codes: set[str], path=HK_PX_CSV) -> dict[str, dict]:
    """First valid trading row per Yahoo-style ticker (e.g. '0700.HK') from the eren2222 history CSV."""
    parts = []
    for chunk in _read_csv(path, usecols=["Date", "Ticker", "Open", "High", "Low", "Close"],
                           chunksize=2_000_000):
        parts.append(chunk[chunk.Ticker.isin(codes)].dropna(subset=["Close"]))
    if not parts:
        return {}
    px = pd.concat(parts, ignore_index=True).sort_values(["Ticker", "Date"]).drop_duplicates("Ticker", keep="first")
    return {r.Ticker: {"date": datetime.strptime(str(r.Date)[:10], "%Y-%m-%d").date(), "open": _f(r.Open),
                       "high": _f(r.High), "low": _f(r.Low), "close": _f(r.Close)} for r in px.itertuples()}


def build_hk(deals: pd.DataFrame, first_rows: dict[str, dict], report: Counter) -> pd.DataFrame:
    recs = []
    for _, d in deals.iterrows():
        report["hk_raw_rows"] += 1
        code = f"{int(d['stock_id']):04d}"
        listing = datetime.strptime(str(d["listing_date"])[:10], "%Y-%m-%d").date()
        price = _f(d["pricing_final_offer_price_source"])
        if price is None or price <= 0:
            report["hk_excluded_no_offer_price"] += 1
            continue
        fr = first_rows.get(f"{code}.HK")
        if fr is None:
            report["hk_no_price_history"] += 1
            continue
        if fr["date"] != listing:
            report["hk_no_exact_first_row"] += 1
            continue
        tg = _targets_from_prices(price, fr["open"], fr["high"], fr["low"], fr["close"], report)
        if tg["list_close_gain_pct"] is None and tg["list_open_gain_pct"] is None:
            report["hk_joined_no_target"] += 1
            continue
        proceeds = _f(d["offer_size_gross_proceeds_hkd_m"])
        recs.append({
            "ipo_id": f"HK:{code}:{listing.isoformat()}", "market": "HK", "country": "HK", "exchange": "HKEX",
            "company": d["company_name"], "symbol": code, "sector": None, "listing_date": listing,
            "currency": "HKD", "issue_price": price, "lot_size": _f(d["offer_size_board_lot_size"]),
            "issue_size_m": proceeds, "issue_size_usd_m": None,  # no FX source: USD left blank
            "sub_retail": _f(d["hk_public_offering_subscription_level_times_source"]),  # public-offer multiple
            **tg, "source": "kaggle:ng3412+px_join",
            "source_url": "https://www.kaggle.com/datasets/ng3412/hkex-ipo-dataset + "
                          "https://www.kaggle.com/datasets/eren2222/hongkong-daily-historical-stock-market-2000-2024",
        })
        report["hk_joined"] += 1
    return pd.DataFrame(recs)


# ---- Indonesia (leanderangelo) -------------------------------------------------------------------

def load_indonesia(path=ID_CSV) -> tuple[pd.DataFrame, Counter]:
    """Starting_Price read as the offer price (unverified basis); Day-1 close as the first-day close. Open/high/low
    not in the source. Issue size is left blank: the 'IPO' column's meaning (shares or IDR) is not documented."""
    cols = ["Code", "Name", "Listing_Date", "Sector", "Starting_Price", "Day-1"]
    df = _read_csv(path)
    _require(df, cols, path)
    recs = []
    report: Counter = Counter({"id_raw_rows": len(df)})
    for _, r in df.iterrows():
        price = _f(r["Starting_Price"])
        close = _f(r["Day-1"])
        if price is None or price <= 0:
            report["id_excluded_no_offer_price"] += 1
            continue
        if close is None or close <= 0:
            report["id_excluded_no_first_day_close"] += 1
            continue
        listing = datetime.strptime(str(r["Listing_Date"]), "%d/%m/%Y").date()
        code = str(r["Code"]).strip().upper()
        recs.append({
            "ipo_id": f"OTHER:{code.lower()}:{listing.isoformat()}", "market": "OTHER", "country": "ID",
            "exchange": "IDX", "company": r["Name"], "symbol": code, "sector": r["Sector"],
            "listing_date": listing, "currency": "IDR", "issue_price": price,
            **_targets_from_prices(price, None, None, None, close, report),
            "source": "kaggle:leanderangelo", "source_url": "https://www.kaggle.com/datasets/leanderangelo/indonesia-ipo-jan2022-july2023",
        })
    return pd.DataFrame(recs), report


# ---- resolution across sources ---------------------------------------------------------------------

def _sanity(frame: pd.DataFrame, report: Counter) -> pd.DataFrame:
    """Check the contract. A row with an invalid issue price is dropped. An out-of-band first-day gain is blanked
    alone (the other gains of the same row may still be valid). Every violation is printed and counted."""
    frame = frame.copy()
    bad_price = ~(frame["issue_price"] > 0)
    for _, row in frame[bad_price].iterrows():
        print(f"SANITY VIOLATION issue_price_not_positive: {row['ipo_id']} {row['company']}")
    report["sanity_issue_price_not_positive"] += int(bad_price.sum())
    for col in TARGETS:
        vals = frame[col]
        out = vals.notna() & ~vals.between(GAIN_MIN, GAIN_MAX) & ~bad_price
        for _, row in frame[out].iterrows():
            print(f"SANITY VIOLATION {col}_out_of_band: {row['ipo_id']} {row['company']} value={row[col]:.1f}")
        report[f"sanity_blanked_{col}"] += int(out.sum())
        frame.loc[out, col] = None
    report["sanity_dropped_total"] = int(bad_price.sum())
    return frame[~bad_price].copy()


def _dedupe(frame: pd.DataFrame, report: Counter) -> pd.DataFrame:
    """One row per ipo_id, keeping the row with the most filled schema fields."""
    filled = frame.notna().sum(axis=1)
    frame = frame.assign(_filled=filled).sort_values("_filled", ascending=False, kind="stable")
    before = len(frame)
    frame = frame.drop_duplicates("ipo_id", keep="first").drop(columns="_filled")
    report["duplicates_removed"] += before - len(frame)
    return frame


def resolve(primary: pd.DataFrame, alt_closes: pd.DataFrame, report: Counter) -> pd.DataFrame:
    """Pick one primary row per ipo_id, then cross-check its first-day close against other sources.

    alt_closes: columns ipo_id, close (price), source. A close that differs from the primary by more than 0.5% blanks
    list_close_gain_pct (counted). Agreeing observations are counted as confirmations. Rows left with no target
    are dropped (a row without a target is useless for training).
    """
    primary = _dedupe(primary, report)
    alt_map = alt_closes.groupby("ipo_id")["close"].apply(list).to_dict() if len(alt_closes) else {}
    out = []
    for _, row in primary.iterrows():
        row = row.copy()
        others = alt_map.get(row["ipo_id"], [])
        close_gain = row["list_close_gain_pct"]
        if others and close_gain is not None and not pd.isna(close_gain) and row["issue_price"] > 0:
            close_px = row["issue_price"] * (1 + close_gain / 100)
            if any(abs(o / close_px - 1) > DISAGREE_TOL for o in others):
                row["list_close_gain_pct"] = None
                report["close_disagree_blanked"] += 1
            else:
                report["close_confirmed_by_second_source"] += 1
        out.append(row)
    frame = pd.DataFrame(out) if out else primary.iloc[0:0]
    has_target = frame[TARGETS].notna().any(axis=1)
    report["no_target_dropped"] += int((~has_target).sum())
    return frame[has_target].copy()


def _finish(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the remaining schema columns (NaN where the source has no value) in schema order."""
    for col in schema.ALL_COLS:
        if col not in frame.columns:
            frame[col] = None
    frame["n_missing_core"] = frame[CORE_COLS].isna().sum(axis=1)
    for col in schema.DATE_COLS:
        frame[col] = pd.to_datetime(frame[col])
    frame = frame.sort_values(["market", "listing_date", "ipo_id"]).reset_index(drop=True)
    return frame[schema.ALL_COLS]


def _require_raw(*paths) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SourceError(f"raw Kaggle files missing (run the Kaggle download first): {missing}")


def build_global(refresh: bool = False) -> pd.DataFrame:
    """Build the non-India IPO table. Missing values are NaN; only rows with at least one first-day target are kept."""
    report: Counter = Counter()
    _require_raw(TD_CSV, PRO_CSV, SHIV_CSV, BJ_DIR, HK_IPO_XLSX, HK_PX_CSV, ID_CSV)
    scoop, scoop_rep = build_from_html(load_pages(refresh=refresh))
    report.update({f"iposcoop_{k}": v for k, v in scoop_rep.items()})

    td, td_rep = load_thedevastator()
    report.update(td_rep)
    # Cross-check closes for thedevastator rows: benjaminpo first row (exact date) and proselotis closeDay0.
    cache: dict = {}
    alt = []
    for _, r in td.iterrows():
        fr, _why = us_exact_first_row(r["symbol"], r["listing_date"], cache)
        if fr is not None and fr["close"] is not None:
            if _split_like(fr["close"] / r["issue_price"]):
                report["alt_price_set_suspect_split_skipped"] += 1
            else:
                alt.append({"ipo_id": r["ipo_id"], "close": fr["close"], "source": fr["set"]})
                report[f"alt_close_from_{fr['set']}"] += 1
    pro = load_proselotis_closes()
    pro = pro.assign(symbol=pro["symbol"].str.upper())
    key = td.assign(_d=td["listing_date"])[["ipo_id", "symbol", "_d", "issue_price"]]
    pm = pro.merge(key, left_on=["symbol", "listing_date"], right_on=["symbol", "_d"])
    for r in pm.itertuples():
        if _split_like(r.close / r.issue_price):
            report["alt_proselotis_suspect_split_skipped"] += 1
        else:
            alt.append({"ipo_id": r.ipo_id, "close": r.close, "source": "proselotis"})
    report["proselotis_matched_to_td"] = len(pm)

    shiv, shiv_rep = load_shivamb()
    report.update(shiv_rep)
    shiv_joined = load_shivamb_joined(shiv, report)

    deals = load_hk_deals()
    codes = {f"{int(c):04d}.HK" for c in deals["stock_id"]}
    hk = build_hk(deals, load_hk_first_rows(codes), report)

    idn, id_rep = load_indonesia()
    report.update(id_rep)

    primary = pd.concat([scoop, td, shiv_joined, hk, idn], ignore_index=True)
    primary = _sanity(primary, report)
    alt_df = pd.DataFrame(alt, columns=["ipo_id", "close", "source"])
    frame = resolve(primary, alt_df, report)
    _print_report(report, frame)
    return _finish(frame)


def _print_report(report: Counter, frame: pd.DataFrame) -> None:
    print("exclusions, joins and checks:")
    for key in sorted(report):
        print(f"  {key}: {report[key]}")
    print(f"  kept_rows: {len(frame)}")


def coverage(df: pd.DataFrame) -> str:
    """Rows per market and year, then non-null % of every schema column per market."""
    lines = ["rows per market and listing year:"]
    years = df.assign(year=df["listing_date"].dt.year).groupby(["market", "year"]).size()
    lines.append(years.to_string() if len(years) else "  (no rows)")
    lines.append("")
    lines.append("non-null % per column (blank column = no rows for that market):")
    pct = {}
    for m in schema.MARKETS:
        sub = df[df["market"] == m]
        pct[m] = (sub[schema.ALL_COLS].notna().mean() * 100).round(1) if len(sub) else None
    table = pd.DataFrame({m: (v if v is not None else pd.Series(dtype=float)) for m, v in pct.items()})
    lines.append(table.reindex(schema.ALL_COLS).to_string())
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download the IPO Scoop pricing pages")
    args = parser.parse_args()
    df = build_global(refresh=args.refresh)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    out = config.PROCESSED / "global.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")
    print(coverage(df))


if __name__ == "__main__":
    main()
