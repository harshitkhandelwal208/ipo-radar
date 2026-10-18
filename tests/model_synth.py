"""Synthetic IPO table and index series for the model and build tests.

The open gain depends strongly on log total subscription (a known signal the tests can assert), the close gain on
the open gain, and GMP is a noisy copy of the open gain from 2020 on. Missingness follows the real table: QIB
subscription only from 2017, GMP only from 2020, no open dates for US rows (so no index context there).
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from ipo_radar.schema import ALL_COLS, DATE_COLS, ID_COLS, SUB_COLS

STR_COLS = ID_COLS + ["currency", "source", "source_url"]
SECTORS = [f"Sector {i:02d}" for i in range(30)]   # more than SECTOR_TOP_N so the 'other' bucket is exercised


def make_ipos(n_per_year: int = 100, years: range = range(2015, 2025), seed: int = 0) -> pd.DataFrame:
    """Columns exactly schema.ALL_COLS with the dtypes data.build produces. Markets: IN_MAIN, IN_SME, US (no HK, so
    tests can check an unseen market)."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        for k in range(n_per_year):
            u = rng.random()
            market = "IN_MAIN" if u < 0.3 else "IN_SME" if u < 0.7 else "US"
            india = market != "US"
            listing = date(year, 1, 1) + timedelta(days=int(rng.integers(0, 360)))
            close = listing - timedelta(days=int(rng.integers(3, 9)))
            opened = close - timedelta(days=int(rng.integers(2, 4)))
            sub_total = float(np.exp(rng.normal(1.0, 1.0)))
            open_gain = float(np.clip(25.0 * np.log(sub_total) - 5.0 + rng.normal(0, 5.0), -90, None))
            close_gain = float(np.clip(0.8 * open_gain + rng.normal(0, 3.0), -90, None))
            has_cat = india and year >= 2017 and rng.random() < 0.8
            gmp = float(open_gain + rng.normal(0, 5.0)) if india and year >= 2020 and rng.random() < 0.7 else np.nan
            low = float(rng.uniform(20, 500))
            high = low * 1.1
            lot = float(rng.choice([25, 50, 100, 150])) if india else np.nan
            size_m = float(rng.lognormal(4.0, 1.0))
            fresh = float(rng.uniform(40, 100)) if india else np.nan
            rec = {
                "ipo_id": f"{market}-{year}-{k:04d}", "market": market, "country": "IN" if india else "US",
                "exchange": (("NSE" if rng.random() < 0.6 else "BSE") if india else "NASDAQ"),
                "company": f"Co {year}-{k}", "symbol": f"S{year % 100}{k:03d}",
                "sector": None if rng.random() < 0.05 else str(rng.choice(SECTORS)),
                "open_date": pd.NaT if market == "US" else pd.Timestamp(opened),
                "close_date": pd.NaT if market == "US" else pd.Timestamp(close),
                "listing_date": pd.Timestamp(listing),
                "currency": "INR" if india else "USD", "price_low": low if india else np.nan,
                "price_high": high if india else np.nan, "issue_price": high if india else round(low / 50, 2),
                "lot_size": lot, "issue_size_m": size_m if india else np.nan,
                "issue_size_usd_m": size_m * 0.012 if not india else np.nan,
                "fresh_issue_pct": fresh, "ofs_pct": 100.0 - fresh if india else np.nan,
                "mcap_issue_usd_m": np.nan, "pe_issue": float(rng.uniform(5, 60)) if rng.random() < 0.5 else np.nan,
                "roe": np.nan, "roce": np.nan, "debt_equity": np.nan, "revenue_growth_pct": np.nan,
                "pat_margin_pct": np.nan, "is_loss_making": np.nan, "anchor_pct": np.nan, "n_underwriters": np.nan,
                "sub_qib": sub_total * float(np.exp(rng.normal(0, 0.3))) if has_cat else np.nan,
                "sub_nii": sub_total * 0.9 if has_cat else np.nan,
                "sub_snii": np.nan, "sub_bnii": np.nan,
                "sub_retail": sub_total * 0.7 if has_cat else np.nan,
                "sub_employee": np.nan, "sub_shareholder": np.nan, "sub_total": sub_total if india else np.nan,
                "gmp_open": np.nan, "gmp_last": gmp * high / 100 if not np.isnan(gmp) else np.nan,
                "gmp_pct_open": np.nan, "gmp_pct_last": gmp,
                "list_open_gain_pct": open_gain, "list_high_gain_pct": open_gain + 3,
                "list_low_gain_pct": open_gain - 3, "list_close_gain_pct": close_gain,
                "source": "synthetic", "source_url": "", "n_missing_core": 0,
            }
            for col in ("idx_ret_5d", "idx_ret_20d", "idx_vol_20d", "vix_level", "idx_ret_issue_window"):
                rec[col] = float(rng.normal(0, 3)) if india else np.nan
            for d in (1, 2):
                rec[f"sub_qib_d{d}"] = sub_total * (0.5 if d == 1 else 0.8) if has_cat else np.nan
                rec[f"sub_nii_d{d}"] = np.nan
                rec[f"sub_retail_d{d}"] = np.nan
                rec[f"sub_total_d{d}"] = sub_total * (0.4 if d == 1 else 0.8) if india else np.nan
            for d in (3,):
                for c in ("sub_qib", "sub_nii", "sub_retail", "sub_total"):
                    rec[f"{c}_d{d}"] = np.nan
            rows.append(rec)
    df = pd.DataFrame(rows)
    for col in ALL_COLS:
        if col in STR_COLS:
            df[col] = df[col].astype("string")
        elif col in DATE_COLS:
            df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
        elif col == "n_missing_core":
            df[col] = df[col].astype("int64")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    assert set(SUB_COLS) <= set(df.columns)
    return df[ALL_COLS].reset_index(drop=True)


def make_indices(start: str = "2014-01-01", end: str = "2026-10-14", seed: int = 1) -> pd.DataFrame:
    """Daily closes (random walks) for the nine benchmark symbols, columns date, symbol, close."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    frames = []
    for symbol in ("NIFTY50", "SENSEX", "INDIAVIX", "SP500", "NYSE_COMP", "NASDAQ_COMP", "VIX", "HSI", "SHCOMP"):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
        if symbol.endswith("VIX") or symbol == "INDIAVIX":
            close = 15 + np.abs(rng.normal(0, 3, len(dates)))
        frames.append(pd.DataFrame({"date": dates, "symbol": symbol, "close": close}))
    out = pd.concat(frames, ignore_index=True)
    out["date"] = out["date"].astype("datetime64[us]")
    out["symbol"] = out["symbol"].astype("string")
    return out
