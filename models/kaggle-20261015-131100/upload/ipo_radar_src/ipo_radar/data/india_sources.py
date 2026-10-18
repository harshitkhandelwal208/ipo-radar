"""Kaggle loaders for India IPOs, reduced to one canonical record table per source.

Every loader reads a file that was downloaded once into data/raw/kaggle/<owner>__<slug>/ and returns a
DataFrame with CANON columns (NaN where the source has no such field). Layouts are checked; a changed
layout or an unreadable file raises SourceError instead of returning guessed values. Which Kaggle columns
are used, and why the others are not, is in docs/data_sources/india.md.

Values of 0 in price columns are treated as missing (a 0 listing open next to a real close was seen in
amansingh2116's SME file). Subscription and amount cells that cannot be parsed become NaN.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ipo_radar.errors import SourceError

KAGGLE_SLUGS = {
    "rohan": "rohandeogaonkar__india-mainboard-ipos-20yr",
    "karan": "karanammithul__ipo-data-india-2010-2025",
    "aman": "amansingh2116__indian-ipo-2000-may-2026-data",
    "warcoder": "warcoder__ipo-mainboard-and-sme-basic-details-india",
    "thabresh": "thabresh__ipo-performance",
    "balabaskar": "balabaskar__indian-ipo-dataset-2010-2021",
    "nikhil": "nikhilraj7700__ipo-watch-debuts-gmps-trajectories",
    "rimarusama": "rimarusama__ipo-pastissue-india-nse",
}
CANON = ["src", "name", "nkey", "sym", "market", "listing", "open_dt", "close_dt", "issue_price", "issue_cr",
         "lot", "exchange", "fresh_pct", "sub_qib", "sub_nii", "sub_snii", "sub_bnii", "sub_retail",
         "sub_employee", "sub_shareholder", "sub_total", "px_open", "px_high", "px_low", "px_close", "gmp", "url"]
STOP_WORDS = {"ipo", "limited", "ltd", "the", "sme", "and", "co", "company", "india", "private", "pvt", "in", "of"}
MAX_BAD_DATE_SHARE = 0.05   # more unparsed dates than this means the layout changed


def name_key(text: object) -> str:
    """Normalised company name for matching: lower-case tokens, no brackets, no stop words."""
    if pd.isna(text):
        return ""
    s = re.sub(r"\(.*?\)", " ", str(text).lower())
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(t for t in s.split() if t not in STOP_WORDS)


def gmp_symbol(text: object) -> str:
    """NSE symbol from the investorgain name form 'Name IPO (SYMBOL,BSECODE)'."""
    m = re.search(r"\(([^)]*)\)", str(text))
    return m.group(1).split(",")[0].strip().upper() if m else ""


def num(s: pd.Series) -> pd.Series:
    """Numbers with thousands separators; '-', blanks and text become NaN."""
    cleaned = s.astype("string").str.replace(",", "", regex=False).str.strip()
    cleaned = cleaned.replace({"-": pd.NA, "": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def price(s: pd.Series) -> pd.Series:
    """Prices where 0 and negative values mean missing."""
    v = num(s)
    return v.where(v > 0)


def parse_dates(s: pd.Series, fmt: str, label: str) -> pd.Series:
    """Explicit-format dates. Too many unparsed non-empty cells means the layout changed."""
    raw = s.astype("string").str.strip()
    raw = raw.mask(raw.isin(["-", ""]))
    out = pd.to_datetime(raw, format=fmt, errors="coerce")
    bad = int((raw.notna() & out.isna()).sum())
    if raw.notna().sum() and bad / raw.notna().sum() > MAX_BAD_DATE_SHARE:
        raise SourceError(f"{label}: {bad} dates do not match format {fmt!r}; layout changed")
    return out


def exchange_norm(s: pd.Series) -> pd.Series:
    """'BSE, NSE' / 'BSE SME' / 'NSE' -> 'BSE,NSE' / 'BSE' / 'NSE' (SME marker dropped, tokens sorted)."""
    def one(text: object) -> str | None:
        if pd.isna(text):
            return None
        toks = {t.strip().replace("SME", "").strip() for t in re.split(r"[,/]", str(text))}
        toks = sorted(t for t in toks if t in {"NSE", "BSE", "MCX-SX"})
        return ",".join(toks) or None
    return s.map(one)


def read_csv(path: Path, need: list[str], label: str, **kw) -> pd.DataFrame:
    df = _read(path, lambda p: pd.read_csv(p, encoding_errors="replace", low_memory=False, **kw), label)
    _need(df, need, label)
    return df


def read_xlsx(path: Path, need: list[str], label: str) -> pd.DataFrame:
    df = _read(path, lambda p: pd.read_excel(p), label)
    _need(df, need, label)
    return df


def _read(path: Path, reader, label: str) -> pd.DataFrame:
    if not path.exists():
        raise SourceError(f"Kaggle file missing: {path} ({label})")
    try:
        df = reader(path)
    except Exception as exc:  # zero-byte, corrupt or undecodable file
        raise SourceError(f"could not read {path} ({label}): {exc}") from exc
    if df.empty:
        raise SourceError(f"{label}: no rows in {path.name}")
    return df


def _need(df: pd.DataFrame, need: list[str], label: str) -> None:
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SourceError(f"{label}: layout changed, missing columns {missing}")


def _canon(n: int, src: str, url: str) -> pd.DataFrame:
    out = pd.DataFrame(index=range(n), columns=CANON, dtype="object")
    out["src"] = src
    out["url"] = url
    return out


def _kaggle_url(key: str) -> str:
    return f"https://www.kaggle.com/datasets/{KAGGLE_SLUGS[key]}"


def load_rohan(root: Path) -> list[pd.DataFrame]:
    """Chittorgarh-derived mainboard table (cg_issue, cg_subs, cg_listing joined on cg_ipo_id) and
    finalipodata.csv (lot size and offer price only). The first is 'rohan', the second 'rohan_main'."""
    d = root / KAGGLE_SLUGS["rohan"]
    iss = read_csv(d / "cg_issue.csv", ["cg_ipo_id", "company", "Issue Category", "Listing Date", "ListingDate",
                                        "Issue Amount (Rs.cr.)", "Fresh Capital (Rs.cr.)", "Listing at",
                                        "nse_symbol", "Opening Date", "Closing Date"], "rohan cg_issue")
    lst = read_csv(d / "cg_listing.csv", ["cg_ipo_id", "Listing Date", "Issue Price (Rs.)",
                                          "Open Price on Listing (Rs.)", "Close Price on Listing (Rs.)"],
                   "rohan cg_listing")
    sub = read_csv(d / "cg_subs.csv", ["cg_ipo_id", "company", "QIB (x)", "sNII (x)", "bNII (x)", "NII (x)",
                                       "Retail (x)", "Employee (x)", "Shareholder (x)", "Total (x)"],
                   "rohan cg_subs")
    iss = iss.rename(columns={"company": "name_i"})[["cg_ipo_id", "name_i", "Issue Category", "Listing Date",
                                                     "ListingDate", "Issue Amount (Rs.cr.)",
                                                     "Fresh Capital (Rs.cr.)", "Listing at", "nse_symbol",
                                                     "Opening Date", "Closing Date"]]
    lst = lst[["cg_ipo_id", "Listing Date", "Issue Price (Rs.)", "Open Price on Listing (Rs.)",
               "Close Price on Listing (Rs.)"]].rename(columns={"Listing Date": "Listing Date_l"})
    sub = sub.rename(columns={"company": "name_s"})[["cg_ipo_id", "name_s", "QIB (x)", "sNII (x)", "bNII (x)",
                                                     "NII (x)", "Retail (x)", "Employee (x)", "Shareholder (x)",
                                                     "Total (x)"]]
    base = iss.merge(lst, on="cg_ipo_id", how="outer").merge(sub, on="cg_ipo_id", how="outer")
    n = len(base)
    out = _canon(n, "rohan", _kaggle_url("rohan"))
    out["name"] = base["name_i"].fillna(base["name_s"])
    out["nkey"] = out["name"].map(name_key)
    out["sym"] = base["nse_symbol"].fillna("").astype(str).str.upper().str.strip()
    out["market"] = np.where(base["Issue Category"] == "Mainboard", "IN_MAIN", None)
    ld1 = parse_dates(base["Listing Date_l"].fillna(base["Listing Date"]), "%d-%b-%Y", "rohan listing")
    ld2 = parse_dates(base["ListingDate"].astype("string").str[:10], "%Y-%m-%d", "rohan ListingDate")
    out["listing"] = ld1.fillna(ld2)
    out["open_dt"] = parse_dates(base["Opening Date"], "%d-%b-%Y", "rohan open")
    out["close_dt"] = parse_dates(base["Closing Date"], "%d-%b-%Y", "rohan close")
    out["issue_price"] = price(base["Issue Price (Rs.)"])
    out["issue_cr"] = num(base["Issue Amount (Rs.cr.)"])
    out["exchange"] = exchange_norm(base["Listing at"])
    fresh_cr = num(base["Fresh Capital (Rs.cr.)"])
    out["fresh_pct"] = (fresh_cr / out["issue_cr"] * 100).where(fresh_cr.notna() & (out["issue_cr"] > 0))
    out["sub_qib"] = num(base["QIB (x)"])
    out["sub_nii"] = num(base["NII (x)"])
    out["sub_snii"] = num(base["sNII (x)"])
    out["sub_bnii"] = num(base["bNII (x)"])
    out["sub_retail"] = num(base["Retail (x)"])
    out["sub_employee"] = num(base["Employee (x)"])
    out["sub_shareholder"] = num(base["Shareholder (x)"])
    out["sub_total"] = num(base["Total (x)"])
    out["px_open"] = price(base["Open Price on Listing (Rs.)"])
    out["px_close"] = price(base["Close Price on Listing (Rs.)"])
    rohan = out

    fin = read_csv(d / "finalipodata.csv", ["Name", "List Date", "Offer Price", "Lot size"], "rohan finalipodata")
    m = _canon(len(fin), "rohan_main", _kaggle_url("rohan"))
    m["name"] = fin["Name"]
    m["nkey"] = m["name"].map(name_key)
    m["market"] = "IN_MAIN"
    m["listing"] = parse_dates(fin["List Date"], "%Y-%m-%d", "rohan_main list date")
    m["issue_price"] = price(fin["Offer Price"])
    m["lot"] = price(fin["Lot size"])
    return [rohan, m]


def load_karan(root: Path) -> pd.DataFrame:
    """karanammithul: mainboard and SME mixed, no market flag. 'List Price' is deliberately not loaded:
    its definition could not be matched to any listing-day field (see docs/data_sources/india.md)."""
    df = read_xlsx(root / KAGGLE_SLUGS["karan"] / "Initial Public Offering (Updated).xlsx",
                   ["Date", "IPO_Name", "Issue_Size(crores)", "Offer Price", "QIB", "HNI", "RII", "Total"],
                   "karan")
    out = _canon(len(df), "karan", _kaggle_url("karan"))
    out["name"] = df["IPO_Name"]
    out["nkey"] = out["name"].map(name_key)
    out["listing"] = pd.to_datetime(df["Date"], errors="coerce")
    out["issue_price"] = price(df["Offer Price"])
    out["issue_cr"] = num(df["Issue_Size(crores)"])
    out["sub_qib"] = num(df["QIB"])
    out["sub_nii"] = num(df["HNI"])
    out["sub_retail"] = num(df["RII"])
    out["sub_total"] = num(df["Total"])
    return out


def load_aman(root: Path) -> list[pd.DataFrame]:
    """amansingh2116 mainboard (with subscriptions) and SME (with lot size and listing open/close)."""
    d = root / KAGGLE_SLUGS["aman"]
    main = read_csv(d / "ipo_historical_data_all_pages.csv",
                    ["Issue Details List Date", "Issue Details IPO Name",
                     "Subscription (No of times) Issue Size (₹.Cr)", "Subscription (No of times) QIB",
                     "Subscription (No of times) NII", "Subscription (No of times) RII",
                     "Subscription (No of times) Total", "Price(₹) Offer Price(₹)"], "aman mainboard")
    m = _canon(len(main), "aman", _kaggle_url("aman"))
    m["name"] = main["Issue Details IPO Name"]
    m["nkey"] = m["name"].map(name_key)
    m["market"] = "IN_MAIN"
    m["listing"] = parse_dates(main["Issue Details List Date"], "%d %b %Y", "aman mainboard date")
    m["issue_cr"] = num(main["Subscription (No of times) Issue Size (₹.Cr)"])
    m["issue_price"] = price(main["Price(₹) Offer Price(₹)"])
    m["sub_qib"] = num(main["Subscription (No of times) QIB"])
    m["sub_nii"] = num(main["Subscription (No of times) NII"])
    m["sub_retail"] = num(main["Subscription (No of times) RII"])
    m["sub_total"] = num(main["Subscription (No of times) Total"])

    sme = read_csv(d / "sme_ipo_historical_data_all_pages.csv",
                   ["Issue Details Listing Date", "Issue Details IPO Name", "Issue Details Lot Size",
                    "Issue Details Issue Size (₹.Cr)", "Issue Details Offer Price (₹)", "Price Listing Open",
                    "Price Listing Close"], "aman SME")
    s = _canon(len(sme), "aman_sme", _kaggle_url("aman"))
    s["name"] = sme["Issue Details IPO Name"]
    s["nkey"] = s["name"].map(name_key)
    s["market"] = "IN_SME"
    s["listing"] = parse_dates(sme["Issue Details Listing Date"], "%d-%b-%Y", "aman SME date")
    lot = num(sme["Issue Details Lot Size"])
    s["lot"] = lot.where(lot > 0)
    s["issue_cr"] = num(sme["Issue Details Issue Size (₹.Cr)"])
    s["issue_price"] = price(sme["Issue Details Offer Price (₹)"])
    s["px_open"] = price(sme["Price Listing Open"])
    s["px_close"] = price(sme["Price Listing Close"])
    return [m, s]


def load_warcoder(root: Path) -> pd.DataFrame:
    """Mainboard and SME basic details. Exchange text 'BSE SME' / 'NSE SME' marks SME; other rows are
    mainboard. Only rows with a listing date can be linked; 1009 rows have none."""
    df = read_csv(root / KAGGLE_SLUGS["warcoder"] / "IPO details 2007-2024(FEB).csv",
                  ["Issuer Company", "Open Date", "Close Date", "Listing Date", "Issue Price (Rs)",
                   "Issue Size (Rs Cr.)", "Lot Size", "Exchange"], "warcoder")
    out = _canon(len(df), "warcoder", _kaggle_url("warcoder"))
    out["name"] = df["Issuer Company"].astype("string").str.replace(" IPO", "", regex=False)
    out["nkey"] = out["name"].map(name_key)
    is_sme = df["Exchange"].astype(str).str.contains("SME", regex=False)
    out["market"] = np.where(is_sme, "IN_SME", "IN_MAIN")
    out["exchange"] = exchange_norm(df["Exchange"])
    out["open_dt"] = parse_dates(df["Open Date"], "%b %d, %Y", "warcoder open")
    out["close_dt"] = parse_dates(df["Close Date"], "%b %d, %Y", "warcoder close")
    out["listing"] = parse_dates(df["Listing Date"], "%b %d, %Y", "warcoder listing")
    out["issue_price"] = price(df["Issue Price (Rs)"])
    out["issue_cr"] = num(df["Issue Size (Rs Cr.)"])
    lot = num(df["Lot Size"])
    out["lot"] = lot.where(lot > 0)
    return out


def load_thabresh(root: Path) -> pd.DataFrame:
    """Two header lines, then positional columns: date, name, issue size, QIB, HNI, RII, Total, issue
    price, listing open, listing close. Date is taken as the listing date (matches HF on the overlap)."""
    df = _read(root / KAGGLE_SLUGS["thabresh"] / "ipo.csv",
               lambda p: pd.read_csv(p, header=None, skiprows=2, encoding_errors="replace"), "thabresh")
    if df.shape[1] < 11:
        raise SourceError(f"thabresh: expected at least 11 columns, found {df.shape[1]}; layout changed")
    out = _canon(len(df), "thabresh", _kaggle_url("thabresh"))
    out["name"] = df[2]
    out["nkey"] = out["name"].map(name_key)
    out["listing"] = parse_dates(df[1], "%d-%m-%Y", "thabresh date")
    out["issue_cr"] = num(df[3])
    out["sub_qib"] = num(df[4])
    out["sub_nii"] = num(df[5])
    out["sub_retail"] = num(df[6])
    out["sub_total"] = num(df[7])
    out["issue_price"] = price(df[8])
    out["px_open"] = price(df[9])
    out["px_close"] = price(df[10])
    return out


def load_balabaskar(root: Path) -> pd.DataFrame:
    df = read_csv(root / KAGGLE_SLUGS["balabaskar"] / "ipo_2010_2021.csv",
                  ["Date", "IPO Name", "Issue Size (in crores)", "QIB", "HNI", "RII", "Total", "Issue",
                   "Listing Open", "Listing Close"], "balabaskar")
    out = _canon(len(df), "balabaskar", _kaggle_url("balabaskar"))
    out["name"] = df["IPO Name"]
    out["nkey"] = out["name"].map(name_key)
    out["listing"] = parse_dates(df["Date"], "%d-%m-%y", "balabaskar date")
    out["issue_cr"] = num(df["Issue Size (in crores)"])
    out["issue_price"] = price(df["Issue"])
    out["sub_qib"] = num(df["QIB"])
    out["sub_nii"] = num(df["HNI"])
    out["sub_retail"] = num(df["RII"])
    out["sub_total"] = num(df["Total"])
    out["px_open"] = price(df["Listing Open"])
    out["px_close"] = price(df["Listing Close"])
    return out


def load_nikhil(root: Path) -> pd.DataFrame:
    """Investorgain-style GMP tracker for mainboard IPOs 2020-2023. Only the GMP and the total
    subscription are used; the GMP is validated against HF before it is used (see india.py)."""
    df = read_csv(root / KAGGLE_SLUGS["nikhil"] / "Mainline IPO GMP Performance.csv",
                  ["IPO", "Listing Date", "IPO_Size", "Subscription", "GMP", "IPO Price"], "nikhil")
    out = _canon(len(df), "nikhil", _kaggle_url("nikhil"))
    out["name"] = df["IPO"].map(lambda s: re.sub(r"\s*\(.*?\)\s*", " ", str(s)).strip())
    out["nkey"] = out["name"].map(name_key)
    out["sym"] = df["IPO"].map(gmp_symbol)
    out["market"] = "IN_MAIN"
    out["listing"] = parse_dates(df["Listing Date"], "%d-%m-%Y", "nikhil listing")
    out["issue_cr"] = num(df["IPO_Size"])
    out["issue_price"] = price(df["IPO Price"])
    out["sub_total"] = num(df["Subscription"])
    out["gmp"] = num(df["GMP"])
    return out


def load_rimarusama(root: Path) -> pd.DataFrame:
    """NSE past-issue list. Its only use here is the market (EQ = mainboard, SME = SME) and dates."""
    df = read_csv(root / KAGGLE_SLUGS["rimarusama"] / "IPO-PastIssue-05-May-2025.csv",
                  ["COMPANY NAME", "SECURITY TYPE", "ISSUE PRICE", "Symbol", "ISSUE START DATE",
                   "ISSUE END DATE", "DATE OF LISTING"], "rimarusama")
    out = _canon(len(df), "rimarusama", _kaggle_url("rimarusama"))
    out["name"] = df["COMPANY NAME"]
    out["nkey"] = out["name"].map(name_key)
    out["sym"] = df["Symbol"].fillna("").astype(str).str.upper().str.strip()
    seg = df["SECURITY TYPE"].astype(str).str.strip()
    out["market"] = np.where(seg == "EQ", "IN_MAIN", np.where(seg == "SME", "IN_SME", None))
    out["open_dt"] = parse_dates(df["ISSUE START DATE"], "%d-%b-%y", "rimarusama open")
    out["close_dt"] = parse_dates(df["ISSUE END DATE"], "%d-%b-%y", "rimarusama close")
    out["listing"] = parse_dates(df["DATE OF LISTING"], "%d-%b-%y", "rimarusama listing")
    out["issue_price"] = price(df["ISSUE PRICE"])
    return out


def load_all(root: Path) -> pd.DataFrame:
    """Every Kaggle source as one canonical record table (one row per source row)."""
    parts = [*load_rohan(root), load_karan(root), *load_aman(root), load_warcoder(root), load_thabresh(root),
             load_balabaskar(root), load_nikhil(root), load_rimarusama(root)]
    return pd.concat(parts, ignore_index=True)[CANON]
