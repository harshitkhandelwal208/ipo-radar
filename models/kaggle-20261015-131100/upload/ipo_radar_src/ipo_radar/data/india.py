"""India IPOs (main board + SME): HuggingFace Indian_IPO_datasets as the anchor, Kaggle sources to fill gaps.

Sources
- HF `sohomghosh/Indian_IPO_datasets` (licence CC BY-NC-SA 4.0, personal non-commercial use only). Its IPO
  sheets come from chittorgarh.com and its GMP files from investorgain.com. Files were downloaded once into
  data/raw/hf_indian_ipo/. Nothing is scraped here.
- Kaggle datasets listed in docs/data_sources/kaggle.md, read by india_sources.py from data/raw/kaggle/.

Linking records (one record per source row) into IPOs
  1. Two records link only if their listing dates differ by at most 1 day AND (normalised name equal OR
     NSE symbol equal, both non-empty).
  2. A link is kept only if each record has exactly one candidate in the other source (1:1).
  3. Links are merged by union-find, and an IPO may hold at most one record per source label.
  Unlinked records stay separate and are only used if they can stand alone (see universe rules).

Resolving fields of one IPO
  - Values from all linked records are compared pairwise. Subscriptions agree within 1% (relative),
    prices and issue amounts and GMP within 0.5% (relative), fresh-issue share within 0.5 percentage
    points, and dates, lot size, exchange and market must be equal.
  - If every pair agrees, the value from the highest-priority source is used (HF first).
  - If any pair disagrees, the field is NaN. Nothing is picked silently. Counts go to the stats.
  - A field with a single source is kept as it is. The stats count those too.
  - HF-only fields (price band, offer-for-sale share, day-wise subscriptions, financial ratios) are taken from HF.
  Targets (list_*_gain_pct) are computed from resolved listing prices and the resolved issue price.
  The karanammithul/amansingh2116 "List Price" column is not used: it matches no listing-day field.

GMP
  The HF GMP (investorgain tracker) is always used. nikhilraj7700 GMP is used only if it agrees with HF on
  at least GMP_MIN_AGREEMENT of their overlap (see _validate_nikhil). Otherwise it is dropped. The tracker
  gives one reading per IPO with no timestamp: it is gmp_last, and gmp_open stays NaN.

Universe
  Listing date within [START, as_of]. Rows with no listing date, an unknown market, or a listing date after
  as_of are excluded and counted. Upcoming IPOs are not history, so they are excluded.

Other rules
  - HF subscription, GMP and target rules are unchanged from the first build. SME category values at close
    are the last day with a total. Mainline category values at close come only from Kaggle.
  - Subscription cells above MAX_MULTIPLE are share counts and become NaN.
  - A listing-day bar whose open or close lies outside its high-low range has its high and low targets blanked.
  - Gains below -100% are impossible and are blanked.

Run `python -m ipo_radar.data.india` to write data/processed/india.parquet and print coverage.
"""
from __future__ import annotations

import re
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ipo_radar import config
from ipo_radar.data import india_sources as KS
from ipo_radar.errors import SourceError
from ipo_radar.schema import ALL_COLS, CONTEXT_COLS, DATE_COLS, ID_COLS, SUB_DAY_COLS

HF_REPO = "sohomghosh/Indian_IPO_datasets"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}"
RAW_DIR = config.RAW / "hf_indian_ipo"
KAGGLE_DIR = config.RAW / "kaggle"
OUT_PARQUET = config.PROCESSED / "india.parquet"
MAIN_FILE = "ipo_mainline_final_data_v18.xlsx"
SME_FILE = "SME_data_final_v11.xlsx"
GMP_MAIN_FILE = "GMP_data/GMP_Mainline_2019_2024YTD_Investorgrain_CLEAN.xlsx"
GMP_SME_FILE = "GMP_data/GMP_SME_2019_2024YTD_Investorgrain_CLEAN_v2.xlsx"
MAX_MULTIPLE = 10_000.0          # above this a subscription cell is a share count, not a multiple
N_DAYS = 11                      # SME sheet has up to 11 subscription days
START = pd.Timestamp("2009-01-01")   # HF history starts 2009; earlier Kaggle rows are not used
GMP_MIN_AGREEMENT = 0.95         # nikhilraj7700 GMP must agree with HF on this share of the overlap

REL_TOL = {"issue_price": 0.005, "issue_cr": 0.005, "px_open": 0.005, "px_high": 0.005, "px_low": 0.005,
           "px_close": 0.005, "gmp": 0.005}
SUB_FIELDS = ["sub_qib", "sub_nii", "sub_snii", "sub_bnii", "sub_retail", "sub_employee", "sub_shareholder",
              "sub_total"]
REL_TOL.update({f: 0.01 for f in SUB_FIELDS})
ABS_TOL = {"fresh_pct": 0.5}
EXACT_FIELDS = ["listing", "open_dt", "close_dt", "exchange", "lot", "market"]
RESOLVED = list(REL_TOL) + list(ABS_TOL) + EXACT_FIELDS
HF_ONLY = ["company", "sector", "price_low", "price_high", "ofs_pct", "pat_margin_pct", "is_loss_making",
           "ipo_id"] + SUB_DAY_COLS
EXTRA = HF_ONLY + ["gmp_hf"]
RANK = {"hf": 0, "rohan": 1, "rohan_main": 2, "aman": 3, "aman_sme": 4, "karan": 5, "warcoder": 6,
        "thabresh": 7, "balabaskar": 8, "rimarusama": 9, "nikhil": 10}
REQUIRED_MAIN = ["mapping_key", "Company Name", "Issuer Company", "NSE_symbol", "Open Date", "Close Date",
                 "Listing Date", "Final_Issue_Price", "Price Band", "Issue Size (Rs Cr.)", "Lot Size",
                 "NSE_Open", "NSE_High", "NSE_Low", "NSE_Last_Trade", "Total_subscriptions",
                 "Fresh Issue", "Offer for Sale", "Total Issue Size", "Sector", "Exchange"]
REQUIRED_SME = ["mapping_key", "Company Name", "Issuer Company", "NSE_symbol", "Listing Date", "Close Date",
                "Final_Issue_Price", "Price Band", "Total Issue Size", "Open", "High", "Low", "Last_Trade",
                "Fresh Issue", "Offer for Sale", "Sector", "day_1_date"]
REQUIRED_GMP = ["IPO", "Listing Date", "Subscription", "GMP"]


# ---- HF parsing helpers ---------------------------------------------------------------------------
def _crore(text: object) -> float | None:
    """'... aggregating up to 1,202.34 Cr)' -> 1202.34. '[.]' or no figure means unknown."""
    m = re.search(r"aggregating up to ([\d,]+(?:\.\d+)?) Cr", str(text))
    return float(m.group(1).replace(",", "")) if m else None


def _shares(text: object) -> float | None:
    """'18,326,227 shares (...)' -> 18326227. '[.] shares' is unknown."""
    m = re.match(r"\s*([\d,]+) shares", str(text))
    return float(m.group(1).replace(",", "")) if m else None


def _band(text: object) -> tuple[float | None, float | None]:
    m = re.search(r"([\d,]+(?:\.\d+)?) to ([\d,]+(?:\.\d+)?)", str(text))
    if not m:
        return None, None
    return float(m.group(1).replace(",", "")), float(m.group(2).replace(",", ""))


def _date(s: pd.Series) -> pd.Series:
    """HF date strings ('Dec 29, 2022', 'Thursday, July 7, 2022'). Unparseable -> NaT."""
    cleaned = s.astype("string").str.replace(r"^[A-Za-z]+, (?=[A-Za-z]+ \d)", "", regex=True)
    return pd.to_datetime(cleaned, format="mixed", errors="coerce")


def _read(path: Path, sheet: str | int = 0) -> pd.DataFrame:
    if not path.exists():
        raise SourceError(f"India IPO file missing: {path}. Download the HuggingFace dataset first (docs/data_sources/india.md).")
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except Exception as exc:  # corrupt or non-xlsx download
        raise SourceError(f"could not read {path}: {exc}") from exc


def _check(df: pd.DataFrame, required: list[str], label: str) -> None:
    if df.empty:
        raise SourceError(f"India IPO file {label}: no rows")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SourceError(f"India IPO file {label}: layout changed, missing columns {missing}")


def _clean_multiple(s: pd.Series, stats: dict[str, int]) -> pd.Series:
    """HF subscription multiples. Values above MAX_MULTIPLE are share counts and become NaN (counted)."""
    v = KS.num(s)
    bad = v > MAX_MULTIPLE
    stats["hf_share_count_cells_blanked"] = stats.get("hf_share_count_cells_blanked", 0) + int(bad.sum())
    return v.mask(bad | (v < 0))


def _day_frame(df: pd.DataFrame, stats: dict[str, int]) -> dict[tuple[int, str], pd.Series]:
    out: dict[tuple[int, str], pd.Series] = {}
    for d in range(1, N_DAYS + 1):
        for cat in ("qib", "nii", "retail", "total", "emp"):
            col = f"day_{d}_{cat}"
            if col in df.columns:
                out[(d, cat)] = _clean_multiple(df[col], stats)
    return out


def _last_day_index(day: dict[tuple[int, str], pd.Series], index: pd.Index) -> pd.Series:
    """Highest day number with a total, per row (NaN when no day has a total)."""
    no = pd.Series(np.nan, index=index, dtype="float64")
    for d in range(1, N_DAYS + 1):
        s = day.get((d, "total"))
        if s is not None:
            no = no.where(~s.notna(), float(d))
    return no


def _value_on_day(day: dict[tuple[int, str], pd.Series], cat: str, day_no: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=day_no.index, dtype="float64")
    for d in range(1, N_DAYS + 1):
        s = day.get((d, cat))
        if s is not None:
            out = out.where(day_no != d, s)
    return out


def _gmp_symbol(text: object) -> str:
    return KS.gmp_symbol(text)


def _join_gmp(hf_names: list[set[str]], hf_symbol: list[str], hf_listing: pd.Series,
              gmp: pd.DataFrame) -> dict[int, int]:
    """{gmp_row_position: hf_row_position} for 1:1 matches (listing date within 1 day, name or symbol)."""
    gmp_names = gmp["IPO"].map(lambda s: KS.name_key(re.sub(r"\s*\(.*?\)\s*", " ", str(s))))
    gmp_sym = gmp["IPO"].map(_gmp_symbol)
    gmp_listing = pd.to_datetime(gmp["Listing Date"], format="%d-%b-%Y", errors="coerce")
    by_date: dict[pd.Timestamp, list[int]] = {}
    for hi, d in enumerate(hf_listing):
        if pd.notna(d):
            by_date.setdefault(d, []).append(hi)
    cand: list[tuple[int, int]] = []
    for gi in range(len(gmp)):
        gd = gmp_listing.iloc[gi]
        if pd.isna(gd):
            continue
        for offset in (-1, 0, 1):
            for hi in by_date.get(gd + pd.Timedelta(days=offset), []):
                name_ok = gmp_names.iloc[gi] != "" and gmp_names.iloc[gi] in hf_names[hi]
                sym_ok = gmp_sym.iloc[gi] != "" and gmp_sym.iloc[gi] == hf_symbol[hi]
                if name_ok or sym_ok:
                    cand.append((gi, hi))
    gi_count: dict[int, int] = {}
    hi_count: dict[int, int] = {}
    for gi, hi in cand:
        gi_count[gi] = gi_count.get(gi, 0) + 1
        hi_count[hi] = hi_count.get(hi, 0) + 1
    return {gi: hi for gi, hi in cand if gi_count[gi] == 1 and hi_count[hi] == 1}


def _hf_gmp(df: pd.DataFrame, gmp: pd.DataFrame, stats: dict[str, int]) -> pd.Series:
    """HF GMP (INR) per HF row via the join rules in the module docstring."""
    hf_names = []
    for i in range(len(df)):
        names = {KS.name_key(df["Company Name"].iloc[i]),
                 KS.name_key(str(df["Issuer Company"].iloc[i]).replace(" IPO", ""))}
        hf_names.append(names - {""})
    hf_sym = df["NSE_symbol"].fillna("").astype("string").str.upper().str.strip().tolist()
    matched = _join_gmp(hf_names, hf_sym, _date(df["Listing Date"]), gmp)
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    value = KS.num(gmp["GMP"])
    for gi, hi in matched.items():
        out.iloc[hi] = value.iloc[gi]
    stats["hf_gmp_rows"] = len(gmp)
    stats["hf_gmp_matched"] = len(matched)
    return out


def _hf_records(raw: Path, market: str, stats: dict[str, int]) -> pd.DataFrame:
    """HF rows of one market as canonical records plus the HF-only extras."""
    if market == "IN_MAIN":
        df = _read(raw / MAIN_FILE, "data_ordered")
        _check(df, REQUIRED_MAIN, MAIN_FILE)
        gmp_file = GMP_MAIN_FILE
    else:
        df = _read(raw / SME_FILE, "data_ordered")
        _check(df, REQUIRED_SME, SME_FILE)
        gmp_file = GMP_SME_FILE
    gmp = _read(raw / gmp_file)
    _check(gmp, REQUIRED_GMP, gmp_file)
    fname = MAIN_FILE if market == "IN_MAIN" else SME_FILE

    rec = pd.DataFrame(index=df.index, columns=KS.CANON + EXTRA, dtype="object")
    rec["src"] = "hf"
    rec["url"] = f"{HF_URL}/blob/main/{fname}"
    company = df["Company Name"].astype("string")
    issuer = df["Issuer Company"].astype("string").str.replace(" IPO", "", regex=False)
    rec["name"] = company.where(company.notna(), issuer)
    rec["nkey"] = rec["name"].map(KS.name_key)
    rec["sym"] = df["NSE_symbol"].fillna("").astype(str).str.upper().str.strip()
    rec["market"] = market
    rec["company"] = rec["name"]
    rec["sector"] = df["Sector"]
    rec["listing"] = _date(df["Listing Date"])
    rec["close_dt"] = _date(df["Close Date"])
    rec["issue_price"] = KS.price(df["Final_Issue_Price"])
    band = df["Price Band"].map(_band)
    low = pd.Series([b[0] for b in band], index=df.index, dtype="float64")
    high = pd.Series([b[1] for b in band], index=df.index, dtype="float64")
    rec["price_low"], rec["price_high"] = low, high
    if market == "IN_MAIN":
        rec["open_dt"] = _date(df["Open Date"])
        rec["issue_cr"] = KS.num(df["Issue Size (Rs Cr.)"])
        lot = KS.num(df["Lot Size"])
        rec["lot"] = lot.where(lot > 0)
        rec["exchange"] = KS.exchange_norm(df["Exchange"])
    else:
        rec["open_dt"] = _date(df["day_1_date"])
        rec["issue_cr"] = df["Total Issue Size"].map(_crore).astype("float64")
        rec["exchange"] = None
    fresh = df["Fresh Issue"].map(_shares).astype("float64")
    ofs = df["Offer for Sale"].map(_shares).astype("float64")
    total = df["Total Issue Size"].map(_shares).astype("float64")
    rec["fresh_pct"] = (fresh / total * 100).where(fresh.notna() & (total > 0))
    rec["ofs_pct"] = (ofs / total * 100).where(ofs.notna() & (total > 0))

    if market == "IN_MAIN":
        rec["sub_total"] = _clean_multiple(df["Total_subscriptions"], stats)
        day = _day_frame(df, stats)
        for d in (1, 2, 3):
            for cat in ("qib", "nii", "retail", "total"):
                if (d, cat) in day:
                    rec[f"sub_{cat}_d{d}"] = day[(d, cat)]
        px = {"px_open": "NSE_Open", "px_high": "NSE_High", "px_low": "NSE_Low", "px_close": "NSE_Last_Trade"}
        for k, col in px.items():
            rec[k] = KS.price(df[col])
    else:
        day = _day_frame(df, stats)
        day_no = _last_day_index(day, df.index)
        rec["sub_total"] = _value_on_day(day, "total", day_no)
        rec["sub_qib"] = _value_on_day(day, "qib", day_no)
        rec["sub_nii"] = _value_on_day(day, "nii", day_no)
        rec["sub_retail"] = _value_on_day(day, "retail", day_no)
        rec["sub_employee"] = _value_on_day(day, "emp", day_no)
        for d in (1, 2, 3):
            for cat in ("qib", "nii", "retail", "total"):
                if (d, cat) in day:
                    rec[f"sub_{cat}_d{d}"] = day[(d, cat)]
        px = {"px_open": "Open", "px_high": "High", "px_low": "Low", "px_close": "Last_Trade"}
        for k, col in px.items():
            rec[k] = KS.price(df[col])
    rec["gmp_hf"] = _hf_gmp(df, gmp, stats)
    rec["gmp"] = rec["gmp_hf"]
    rec["ipo_id"] = market + ":" + df["mapping_key"].astype("Int64").astype("string")
    return rec


# ---- record linking ----------------------------------------------------------------------------------
def _name_related(a: str, b: str) -> bool:
    """Exact normalised name, or one name's tokens are the leading tokens of the other ('l t finance' and
    'l t finance holdings'). The prefix form needs at least two tokens and is only used on the same date."""
    if a == b:
        return a != ""
    ta, tb = a.split(), b.split()
    if len(ta) > len(tb):
        ta, tb = tb, ta
    return len(ta) >= 2 and tb[:len(ta)] == ta


def _link(rec: pd.DataFrame) -> np.ndarray:
    """Entity id per record: union of 1:1 links (date within 1 day, name or symbol, distinct sources)."""
    n = len(rec)
    src = rec["src"].to_numpy(object)
    nkey = rec["nkey"].to_numpy(object)
    sym = rec["sym"].to_numpy(object)
    listing = rec["listing"].tolist()
    by_day: dict[pd.Timestamp, list[int]] = {}
    for i, d in enumerate(listing):
        if pd.notna(d):
            by_day.setdefault(pd.Timestamp(d).normalize(), []).append(i)
    cand: list[list[tuple[int, int, bool]]] = [[] for _ in range(n)]
    for i, d in enumerate(listing):
        if pd.isna(d):
            continue
        base = pd.Timestamp(d).normalize()
        for off in (-1, 0, 1):
            for j in by_day.get(base + pd.Timedelta(days=off), []):
                if j == i or src[j] == src[i]:
                    continue
                name_ok = _name_related(nkey[i], nkey[j]) and (off == 0 or nkey[i] == nkey[j])
                sym_ok = sym[i] != "" and sym[i] == sym[j]
                if name_ok or sym_ok:
                    cand[i].append((j, abs(off), name_ok))
    count: list[dict[str, int]] = [{} for _ in range(n)]
    for i in range(n):
        for j, _, _ in cand[i]:
            count[i][src[j]] = count[i].get(src[j], 0) + 1
    links = []
    for i in range(n):
        for j, delta, name_ok in cand[i]:
            if i < j and count[i][src[j]] == 1 and count[j][src[i]] == 1:
                links.append((delta, 0 if name_ok else 1, i, j))
    links.sort()
    parent = list(range(n))
    members: list[set[str]] = [{src[i]} for i in range(n)]
    lo = [pd.Timestamp(d) if pd.notna(d) else None for d in listing]
    hi = list(lo)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, _, i, j in links:
        ri, rj = find(i), find(j)
        if ri == rj or members[ri] & members[rj]:
            continue
        # Links chain, so an IPO's records must all fall within one day of each other.
        if max(hi[ri], hi[rj]) - min(lo[ri], lo[rj]) > pd.Timedelta(days=1):
            continue
        parent[rj] = ri
        members[ri] |= members[rj]
        lo[ri] = min(lo[ri], lo[rj])
        hi[ri] = max(hi[ri], hi[rj])
    return np.array([find(i) for i in range(n)])


# ---- resolution ----------------------------------------------------------------------------------------
def _isnull(v: object) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT or v is pd.NA


def _agree(field: str, a: object, b: object) -> bool:
    if field in EXACT_FIELDS:
        return a == b
    if field in ABS_TOL:
        return abs(float(a) - float(b)) <= ABS_TOL[field]
    a, b = float(a), float(b)
    scale = max(abs(a), abs(b))
    return scale == 0 or abs(a - b) <= REL_TOL[field] * scale


def _resolve_value(field: str, vals: list[tuple[str, object]], stats: dict[str, dict[str, int]]) -> object:
    vals = [(s, v) for s, v in vals if not _isnull(v)]
    if not vals:
        return None
    st = stats.setdefault(field, {"single": 0, "overlap": 0, "disagree": 0})
    if len(vals) == 1:
        st["single"] += 1
        return vals[0][1]
    st["overlap"] += 1
    if all(_agree(field, a[1], b[1]) for a, b in combinations(vals, 2)):
        return vals[0][1]
    st["disagree"] += 1
    return None


def _validate_nikhil(rec: pd.DataFrame, ent: np.ndarray, stats: dict[str, int]) -> bool:
    """True when nikhilraj7700 GMP agrees with HF GMP on enough of their overlap to be used."""
    src = rec["src"].to_numpy(object)
    gmp = rec["gmp"].to_numpy(object)
    pairs = 0
    agree = 0
    by_ent: dict[int, list[int]] = {}
    for i, e in enumerate(ent):
        by_ent.setdefault(int(e), []).append(i)
    for members in by_ent.values():
        hf = [i for i in members if src[i] == "hf" and not _isnull(gmp[i])]
        nk = [i for i in members if src[i] == "nikhil" and not _isnull(gmp[i])]
        for a in hf:
            for b in nk:
                pairs += 1
                agree += _agree("gmp", gmp[a], gmp[b])
    rate = agree / pairs if pairs else 0.0
    stats["nikhil_gmp_overlap"] = pairs
    stats["nikhil_gmp_agreement_pct"] = round(rate * 100, 1)
    return pairs > 0 and rate >= GMP_MIN_AGREEMENT


def _drop_exact_duplicates(kg: pd.DataFrame, stats: dict[str, int]) -> pd.DataFrame:
    """Repeated rows within one Kaggle source (amansingh2116 repeats IPOs across scraped pages) are dropped.
    Only rows identical in every canonical value count as repeats; same-name rows that differ are kept."""
    keys = [c for c in KS.CANON if c != "url"]
    before = len(kg)
    kg = kg.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)
    stats["kaggle_exact_duplicate_rows_dropped"] = before - len(kg)
    return kg


def _entity_frame(rec: pd.DataFrame, ent: np.ndarray, as_of: date, stats: dict[str, int]) -> pd.DataFrame:
    """One row per IPO with resolved fields, before the target and range checks."""
    src = rec["src"].tolist()
    rank = [RANK[s] for s in src]
    col = {f: rec[f].tolist() for f in set(RESOLVED + HF_ONLY + ["gmp_hf", "name", "sym", "ipo_id", "url"])
           if f in rec.columns}
    groups: dict[int, list[int]] = {}
    for i, e in enumerate(ent):
        groups.setdefault(int(e), []).append(i)
    fstats: dict[str, dict[str, int]] = {}
    rows = []
    for members in groups.values():
        members = sorted(members, key=lambda i: rank[i])
        hf = [i for i in members if src[i] == "hf"]
        row: dict[str, object] = {}
        for f in RESOLVED:
            if f in col and f != "market":
                row[f] = _resolve_value(f, [(src[i], col[f][i]) for i in members], fstats)
        row["market"] = _resolve_value("market", [(src[i], col["market"][i]) for i in members], fstats)
        for f in HF_ONLY + ["gmp_hf"]:
            row[f] = col[f][hf[0]] if hf and f in col else None
        row["name"] = col["name"][members[0]]
        row["sym"] = next((col["sym"][i] for i in members if col["sym"][i]), "")
        row["ipo_id"] = col["ipo_id"][hf[0]] if hf else None
        row["source"] = "+".join(sorted({src[i] for i in members}))
        row["source_url"] = col["url"][hf[0]] if hf else col["url"][members[0]]
        rows.append(row)
    stats.update({f"field_{k}_{s}": v for k, d in fstats.items() for s, v in d.items()})
    return pd.DataFrame(rows)


def _universe(frame: pd.DataFrame, as_of: date, stats: dict[str, int]) -> pd.DataFrame:
    """Keep IPOs with a known market and a listing date in [START, as_of]. A Kaggle-only IPO must also carry an
    outcome (any subscription or listing price), otherwise it is only a basic-details row and is excluded.
    HF rows always stay. Everything else is counted."""
    has_outcome = frame[SUB_FIELDS + ["px_open", "px_close"]].notna().any(axis=1)
    from_hf = frame["ipo_id"].notna()
    no_market = frame["market"].isna()
    no_list = frame["listing"].isna()
    outside = ~no_list & ((frame["listing"] < START) | (frame["listing"] > pd.Timestamp(as_of)))
    no_outcome = ~from_hf & ~has_outcome
    stats["excluded_unknown_market"] = int((no_market & ~no_list).sum())
    stats["excluded_no_listing_date"] = int(no_list.sum())
    stats["excluded_outside_window"] = int((outside & ~no_market).sum())
    stats["excluded_kaggle_only_without_outcome"] = int((no_outcome & ~no_market & ~no_list & ~outside).sum())
    keep = ~no_market & ~no_list & ~outside & (from_hf | has_outcome)
    return frame[keep].reset_index(drop=True)


def _targets(frame: pd.DataFrame, stats: dict[str, int]) -> None:
    """Listing-day gains from resolved prices, plus the impossible-value and bar-consistency rules."""
    issue = frame["issue_price"].astype("float64")
    px = {k: frame[k].astype("float64").where(lambda s: s > 0) for k in ("px_open", "px_high", "px_low", "px_close")}
    bad_bar = ((px["px_open"] > px["px_high"]) | (px["px_open"] < px["px_low"])
               | (px["px_close"] > px["px_high"]) | (px["px_close"] < px["px_low"]))
    stats["bar_inconsistent_high_low_blanked"] = int(bad_bar.sum())
    for k, name in (("px_open", "open"), ("px_high", "high"), ("px_low", "low"), ("px_close", "close")):
        gain = (px[k] / issue - 1) * 100
        impossible = gain < -100
        stats[f"impossible_gain_{name}_blanked"] = int(impossible.sum())
        gain = gain.mask(impossible)
        if name in ("high", "low"):
            gain = gain.mask(bad_bar)
        stats[f"gain_{name}_outside_-90_500"] = int(((gain < -90) | (gain > 500)).sum())
        frame[f"list_{name}_gain_pct"] = gain


def _finalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index, columns=ALL_COLS, dtype="object")
    ids = [i if isinstance(i, str) else f"{m}:{d:%Y%m%d}:{n[:30] or 'x'}"
           for i, m, d, n in zip(frame["ipo_id"], frame["market"], frame["listing"], frame["name"])]
    seen: dict[str, int] = {}
    unique = []
    for i in ids:
        seen[i] = seen.get(i, 0) + 1
        unique.append(i if seen[i] == 1 else f"{i}#{seen[i]}")
    out["ipo_id"] = unique
    out["market"] = frame["market"]
    out["country"] = "IN"
    out["exchange"] = frame["exchange"]
    out["company"] = frame["company"].where(frame["company"].notna(), frame["name"])
    out["symbol"] = frame["sym"]
    out["sector"] = frame["sector"]
    out["open_date"] = frame["open_dt"]
    out["close_date"] = frame["close_dt"]
    out["listing_date"] = frame["listing"]
    out["currency"] = "INR"
    out["price_low"] = frame["price_low"]
    out["price_high"] = frame["price_high"]
    out["issue_price"] = frame["issue_price"]
    out["lot_size"] = frame["lot"]
    out["issue_size_m"] = frame["issue_cr"] * 10.0
    out["fresh_issue_pct"] = frame["fresh_pct"]
    out["ofs_pct"] = frame["ofs_pct"]
    out["pat_margin_pct"] = frame["pat_margin_pct"]
    out["is_loss_making"] = frame["is_loss_making"]
    for c in SUB_FIELDS + SUB_DAY_COLS:
        out[c] = frame[c]
    out["gmp_last"] = frame["gmp"]
    out["gmp_pct_last"] = (frame["gmp"] / frame["issue_price"] * 100).where(frame["gmp"].notna())
    for c in ("list_open_gain_pct", "list_high_gain_pct", "list_low_gain_pct", "list_close_gain_pct"):
        out[c] = frame[c]
    for c in CONTEXT_COLS:
        out[c] = np.nan
    out["source"] = frame["source"]
    out["source_url"] = frame["source_url"]
    out["n_missing_core"] = out[["issue_price", "sub_total", "sub_retail", "list_open_gain_pct"]].isna().sum(axis=1)
    for c in ID_COLS + ["currency", "source", "source_url"]:
        out[c] = out[c].astype("string")
    for c in DATE_COLS:
        out[c] = pd.to_datetime(out[c]).astype("datetime64[ns]")
    for c in ALL_COLS:
        if c not in ID_COLS + DATE_COLS + ["currency", "source", "source_url"] and c != "n_missing_core":
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out["n_missing_core"] = out["n_missing_core"].astype("int64")
    return out[ALL_COLS].reset_index(drop=True)


def build_india(raw_dir: Path = RAW_DIR, kaggle_dir: Path | None = KAGGLE_DIR, as_of: date | None = None,
                stats: dict[str, int] | None = None) -> pd.DataFrame:
    """One row per India IPO (IN_MAIN, IN_SME), columns exactly schema.ALL_COLS.

    kaggle_dir=None builds from HF alone (used by tests). Missing or changed files raise SourceError.
    `stats` (optional) receives counts for the caller to print.
    """
    stats = {} if stats is None else stats
    as_of = as_of or date.today()
    recs = [_hf_records(raw_dir, "IN_MAIN", stats), _hf_records(raw_dir, "IN_SME", stats)]
    if kaggle_dir is not None:
        kg = _drop_exact_duplicates(KS.load_all(kaggle_dir), stats)
        for c in EXTRA:
            kg[c] = np.nan
        recs.append(kg)
    rec = pd.concat(recs, ignore_index=True)
    nik = rec["src"] == "nikhil"
    ent_all = _link(rec)
    if kaggle_dir is not None and nik.any():
        if not _validate_nikhil(rec, ent_all, stats):
            rec.loc[nik, "gmp"] = np.nan
            stats["nikhil_gmp_used"] = 0
        else:
            stats["nikhil_gmp_used"] = 1
    frame = _entity_frame(rec, ent_all, as_of, stats)
    frame = _universe(frame, as_of, stats)
    _targets(frame, stats)
    return _finalize(frame)


def coverage_table(df: pd.DataFrame) -> str:
    """Per market: row count and non-null % for every schema column."""
    rows = []
    for market, sub in df.groupby("market"):
        pct = (sub.notna().mean() * 100).round(0).astype(int)
        rows.append(pd.Series(pct, name=f"{market} (n={len(sub)})"))
    return pd.concat(rows, axis=1).T.to_string()


def category_coverage_by_year(df: pd.DataFrame) -> str:
    """Non-null % of the category subscription columns by listing year, per market."""
    parts = []
    for market, sub in df.groupby("market"):
        year = sub["listing_date"].dt.year
        t = sub.groupby(year)[SUB_FIELDS].agg(lambda s: s.notna().mean() * 100).round(0).astype(int)
        t["n"] = sub.groupby(year).size()
        parts.append(f"{market}\n{t.to_string()}")
    return "\n\n".join(parts)


def download(dest: Path = RAW_DIR) -> Path:
    """Fetch the HF workbooks into dest. Any failure becomes a SourceError."""
    try:
        from huggingface_hub import snapshot_download  # dev extra; imported here so tests never need it
        snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=str(dest),
                          allow_patterns=[MAIN_FILE, SME_FILE, GMP_MAIN_FILE, GMP_SME_FILE, "README.md"])
    except Exception as exc:  # network, DNS, HTTP status, timeout, or a missing package
        raise SourceError(f"download from huggingface.co/datasets/{HF_REPO} failed: {exc}") from exc
    return dest


def main() -> None:
    stats: dict[str, int] = {}
    if not (RAW_DIR / MAIN_FILE).exists():
        print(f"downloading {HF_REPO} into {RAW_DIR}")
        download(RAW_DIR)
    df = build_india(stats=stats)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET} rows={len(df)}")
    print("counts by market:", df["market"].value_counts().to_dict())
    print("listing date range:", df["listing_date"].min().date(), "to", df["listing_date"].max().date())
    print("non-null % per column (market in rows):")
    print(coverage_table(df))
    print("category subscription coverage by listing year:")
    print(category_coverage_by_year(df))
    print("rows with list_open_gain_pct NaN:", int(df["list_open_gain_pct"].isna().sum()))
    print("stats:", stats)


if __name__ == "__main__":
    main()
