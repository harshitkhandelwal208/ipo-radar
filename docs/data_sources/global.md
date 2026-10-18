# Non-India IPO data sources (global.parquet)

Owned by `src/ipo_radar/data/global_ipos.py`. Built by `python -m ipo_radar.data.global_ipos`, which reads only local
files (no network except the IPO Scoop pages, which are cached). Verified 2026-10-15.

Raw downloads: `data/raw/iposcoop/` and `data/raw/kaggle/<owner>__<slug>/` (git-ignored). Kaggle access was
read-only (`datasets list / files / metadata / download`), through the scratchpad wrapper. No token is recorded here.

## Output summary

- `data/processed/global.parquet`: 2146 rows, columns exactly `schema.ALL_COLS`, `ipo_id` unique, every
  first-day gain inside -95% to +2000%, `issue_price` > 0.
- Markets: US 1721, HK 323, OTHER 102. HK and OTHER are now present (previously absent).
- Gains: `list_close_gain_pct` filled on 1754 rows, `list_open_gain_pct` on 1554, high/low on about 480.
- Survivorship: 482 rows (US 2019-2021 joins 159, HK joins 323) were joined from price datasets and carry the
  `+px_join` source suffix. Use `source.str.contains("px_join")` to split them in the model report.

## Exact-date join rule (all price joins)

A deal takes first-day OHLC from a per-ticker price file only if the file's first non-blank Close row is dated
exactly the listed date. Otherwise its targets stay NaN. The check is deliberate: it stops a wrong-ticker or
wrong-date join from silently producing a target. Measured outcomes:
- shivamb IPO dates vs jacksoncrow first rows: 154 exact out of 223 non-SPAC deals with a file (191 of them in 2019).
  Mismatches with a file are mostly other companies that reused the ticker. Among benjaminpo mismatches no one-day
  offset pattern appeared, which supports reading IPO Date as the first trading day (not verified against a regulator).
- thedevastator "Date Priced" vs benjaminpo first rows: 439 exact out of 505 matched symbols (same conclusion).
- HK ng3412 listing_date vs eren2222 first rows: 325 exact, 7 mismatched (for example 2155: listing 2021-06-28,
  first price row 2021-06-29; left NaN).

## Corporate actions (split and dividend adjustment)

Price sets here are split-adjusted Yahoo-style series (verified on NVIDIA's 4-for-1 split in 2021: benjaminpo's 2019
Close is about one quarter of the 2021 post-split level). A first-day close is therefore raw only if no split
followed the listing, and the sets carry no split list. Guard used:
- A close whose ratio to the offer price is about 1/k or k (k in 2, 3, 4, 5, 10, 20, tolerance 1%) blanks all four
  targets for that deal. Counted as `suspect_split_factor_blanked` (15 in this run) and `alt_*_suspect_split_skipped`.
- Residual risk: a split whose ratio is not in that set, or a split whose effect is hidden by the tolerance, is not
  caught. Joined closes should be treated as unverified until checked against an exchange archive.

## Cross-source agreement (0.5% rule)

When a second source has a first-day close for the same ipo_id (benjaminpo or jacksoncrow for thedevastator rows,
proselotis closeDay0 for thedevastator rows), and the two differ by more than 0.5%, the close gain is blanked and
counted as `close_disagree_blanked` (391). Agreements are counted as `close_confirmed_by_second_source` (603).
The disagreements are mostly thedevastator's dividend-adjusted `firstday_adjclose` (for example GM: thedevastator 27.32
against benjaminpo 34.19; the benjaminpo value matches the first-day close I recall, GNRC 8.46 against 12.84 likewise).
Rows left with no target after blanking are dropped (`no_target_dropped`, 64). Open-only rows remain (392 rows have a
gain but no close gain).

## Survivorship bias (read before using joined rows)

- jacksoncrow/stock-market-dataset: Nasdaq-traded tickers as of 2020-04 only. NYSE names and anything delisted before
  2020-04 are absent. Joined 2019 US rows therefore skew to Nasdaq names that survived to 2020.
- benjaminpo/sp500-spx-components-stock-price: 3761 per-ticker files, including historical tickers, but data ends
  2022-02-16. 835 of the 1065 non-SPAC priced shivamb deals had no file in either set, so the joined set is a
  minority of the 2019-2021 IPOs.
- eren2222 (HK): 1580 companies to 2024-05-23. 268 of 600 ng3412 deals had no price history at all, and HK rows
  exist only for deals that did join. The HK sample is therefore a selected subset (later years and
  delisted names are under-represented).
- Model implication: the joined rows are not a random sample of IPOs. Treat `+px_join` rows as a separate
  evaluation stratum, and never compare their win rates with the unjoined (thedevastator, IPO Scoop) rows without
  this caveat.

## Sources used

| Source | URL | Licence | Robots / terms | Date range | Rows used | Columns it fills |
|---|---|---|---|---|---|---|
| IPO Scoop priced-IPO tables | https://www.iposcoop.com/2024-pricings/, /2025-pricings/, /current-year-pricings/ | Not stated; "Copyright 2006-2026 IPOScoop.com". No terms page found (common paths 404). | robots.txt: only /wp-admin/ disallowed; pages allowed. Checked at run time. | Listings 2024-01-09 to 2026-10-09 | 490 (541 raw; 26 SPAC, 23 not yet traded, 2 suspect split, rest kept) | company, symbol, sector (Industry), listing_date (Offer Date), currency, issue_price, issue_size_m and issue_size_usd_m (shares x price, approx.), list_close_gain_pct (from 1st Day Close; the source's Return column is vs current price and is ignored) |
| Kaggle thedevastator/dataset-on-ipo-from-2010-2018 (`ipo_stock_2010_2018.csv`) | https://www.kaggle.com/datasets/thedevastator/dataset-on-ipo-from-2010-2018 | CC0-1.0 | Kaggle dataset; no robots applies. | 2010-01 to 2018-12 (Date Priced = first trading row, verified) | 1072 | company, symbol, exchange (NASDAQ, NYSE, NYSE American), sector, listing_date, issue_price (Price), issue_size_m and issue_size_usd_m (Offer Amount / 1e6, USD), list_open_gain_pct (firstday_open), list_close_gain_pct (firstday_adjclose, dividend-adjusted, cross-checked) |
| Kaggle proselotis/financial-ipo-data (`IPODataFull.csv`) | https://www.kaggle.com/datasets/proselotis/financial-ipo-data | CC0-1.0 | as above | 1996 to 2018 | 0 new rows (cross-check only) | closeDay0 used only as a second observation for thedevastator rows (946 matched by symbol and listing date). openDay0 and highDay0/lowDay0 not used (catalog: openDay0 wrong for FB, TWTR). No offer price in this file, so 1996-2009 rows cannot get a target. |
| Kaggle shivamb/company-ipos-2019-2021 (`company_ipo.csv`) | https://www.kaggle.com/datasets/shivamb/company-ipos-2019-2021 | CC0-1.0 | as above | IPO dates 2019-01-04 to 2021-12-21 | 159 (of 1765 raw; 670 SPAC/units by name or unit symbol, 30 no price) | symbol, company, listing_date (IPO Date), issue_price (IPO Price). First-day OHLC comes from the join below. |
| Kaggle jacksoncrow/stock-market-dataset (`stocks/<SYM>.csv`) | https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset | CC0-1.0 | as above | Nasdaq tickers, daily to 2020-04 (survivors) | 154 joined (2019 IPOs) | Open, High, Low, Close (Yahoo, split-adjusted; first row only) |
| Kaggle benjaminpo/sp500-spx-components-stock-price (`data/1d/<SYM>.csv`) | https://www.kaggle.com/datasets/benjaminpo/sp500-spx-components-stock-price | CC0-1.0 (per metadata) | as above | 1962 to 2022-02-16; 3761 tickers (includes historical) | 6 joined (2019-2021), 39 cross-checks for thedevastator | Open, High, Low, Close (split-adjusted) |
| Kaggle ng3412/hkex-ipo-dataset (`hkex_ipo_ds.xlsx`) | https://www.kaggle.com/datasets/ng3412/hkex-ipo-dataset | MIT | as above | Listings 2020-01-08 to 2026-06-15 | 323 joined (of 600) | company, symbol (4-digit code), exchange HKEX, listing_date, issue_price, issue_size_m (gross proceeds HKD m), lot_size, sub_retail (public-offer multiple; see mapping), currency HKD |
| Kaggle eren2222/hongkong-daily-historical-stock-market-2000-2024 (`HK Stock Market 2000-2024.csv`) | https://www.kaggle.com/datasets/eren2222/hongkong-daily-historical-stock-market-2000-2024 | Apache-2.0 | as above. The dataset description contains text copied from a Japan dataset: treat its descriptive text as unreliable, use only the numbers. | 2000-01-04 to 2024-05-23; 1580 companies | used for HK joins | Open, High, Low, Close (first row only) |
| Kaggle leanderangelo/indonesia-ipo-jan2022-july2023 (`IPO 2023 Analysis - Cleaned.csv`) | https://www.kaggle.com/datasets/leanderangelo/indonesia-ipo-jan2022-july2023 | CC-BY-NC-SA-4.0 (non-commercial, share-alike) | as above | Listings 2022-01-03 to 2023-12-06 | 102 | company, symbol (Code), sector, exchange IDX, listing_date (dd/mm/yyyy), currency IDR, issue_price (Starting_Price, read as the offer price, basis unverified), list_close_gain_pct (Day-1 close). Issue size blank: the 'IPO' column's unit is undocumented. |

Kaggle downloads checked and not used: tanavbajaj/yahoo-finance-all-stocks-dataset-daily-update (ODbL; 503 tickers, an
S&P 500-style snapshot ending 2023-09; one exact join in 2019-2021 probe; deleted from raw). Price sets not downloaded
because the catalog already rejected them or because the join probe showed no gain: svaningelgem/nyse-daily-stock-prices
(499 NYSE names, 2026, not a fit for the 2019-2021 IPO list), raymondsunartio/6000-nasdaq-stocks (overlaps jacksoncrow,
ends 2020-02), olegshpagin/usa-stocks-prices-ohlcv (2.9 GB), davnnis2003 HK 2018-2020 (stock-level, catalog rejected).

## Hong Kong: deal fields and the sub_retail mapping

- sub_retail = HK public-offer oversubscription multiple (ng3412 `hk_public_offering_subscription_level_times_source`).
- sub_qib = NaN for HK. The international placing multiple is not in ng3412 (catalog confirmed). Do not infer it.
- issue_size_usd_m = NaN for HK: converting HKD proceeds needs listing-date FX, and no FX series is in this build.
- Only HK deals with an exact first-row join are kept. A deal with no price history or a date mismatch is absent from
  the output (not kept with blank targets). The counts are in the run report (`hk_no_price_history` 268,
  `hk_no_exact_first_row` 7).

## FX rule

- US: already USD, no conversion. issue_size_usd_m = issue_size_m.
- HK (HKD) and OTHER (IDR): issue_size_usd_m is blank because no FX source is included. issue_size_m is HKD million for
  HK. It is blank for Indonesia.

## Exclusions (run of 2026-10-15)

| Rule | Count | Source |
|---|---|---|
| SPAC / blank check (source label or name pattern: acquisition, blank check, SPAC, merger corp, capital investment, opportunity corp) | 26 (IPO Scoop), 670 (shivamb), 0 (thedevastator, proselotis) | label or name |
| Units, warrants, rights (5-letter symbol ending U, W or R) | 0 (IPO Scoop), 1 (thedevastator) | symbol |
| Not yet traded (first-day close shown as $0.00) | 23 | IPO Scoop |
| Offer price missing or not positive | 30 (shivamb), 1 (thedevastator), 0 (IPO Scoop) | source |
| No exact first-row join | 70 (shivamb), 7 (HK) | price-set rule |
| No price file at all | 835 (shivamb), 268 (HK) | price-set rule |
| Suspect split factor (blanks all targets) | 15 | split guard |
| Sanity: issue price not positive | 0 | contract |
| Sanity: first-day gain out of -95..+2000 (blanks that gain only) | close 51, open 50, high 1 | contract |
| No first-day target left (dropped) | 64 | build rule |
| Closed-end funds, ADR-only listings, sub-$1 deals | not excluded (no source flag) | |

## Coverage: non-null % per column and rows per market and listing year

Rows: US 2010-2021 from thedevastator, shivamb joins and IPO Scoop 2024-2026. HK 2020-2024. OTHER 2022-2023.

| column | US | HK | OTHER |
|---|---|---|---|
| ipo_id | 100.0 | 100.0 | 100.0 |
| market | 100.0 | 100.0 | 100.0 |
| country | 100.0 | 100.0 | 100.0 |
| exchange | 62.3 | 100.0 | 100.0 |
| company | 100.0 | 100.0 | 100.0 |
| symbol | 100.0 | 100.0 | 100.0 |
| sector | 89.6 | 0.0 | 100.0 |
| open_date | 0.0 | 0.0 | 0.0 |
| close_date | 0.0 | 0.0 | 0.0 |
| listing_date | 100.0 | 100.0 | 100.0 |
| currency | 100.0 | 100.0 | 100.0 |
| price_low | 0.0 | 0.0 | 0.0 |
| price_high | 0.0 | 0.0 | 0.0 |
| issue_price | 100.0 | 100.0 | 100.0 |
| lot_size | 0.0 | 100.0 | 0.0 |
| issue_size_m | 90.8 | 99.1 | 0.0 |
| issue_size_usd_m | 90.8 | 0.0 | 0.0 |
| fresh_issue_pct | 0.0 | 0.0 | 0.0 |
| ofs_pct | 0.0 | 0.0 | 0.0 |
| mcap_issue_usd_m | 0.0 | 0.0 | 0.0 |
| pe_issue | 0.0 | 0.0 | 0.0 |
| roe | 0.0 | 0.0 | 0.0 |
| roce | 0.0 | 0.0 | 0.0 |
| debt_equity | 0.0 | 0.0 | 0.0 |
| revenue_growth_pct | 0.0 | 0.0 | 0.0 |
| pat_margin_pct | 0.0 | 0.0 | 0.0 |
| is_loss_making | 0.0 | 0.0 | 0.0 |
| anchor_pct | 0.0 | 0.0 | 0.0 |
| n_underwriters | 0.0 | 0.0 | 0.0 |
| sub_qib | 0.0 | 0.0 | 0.0 |
| sub_nii | 0.0 | 0.0 | 0.0 |
| sub_snii | 0.0 | 0.0 | 0.0 |
| sub_bnii | 0.0 | 0.0 | 0.0 |
| sub_retail | 0.0 | 100.0 | 0.0 |
| sub_employee | 0.0 | 0.0 | 0.0 |
| sub_shareholder | 0.0 | 0.0 | 0.0 |
| sub_total | 0.0 | 0.0 | 0.0 |
| sub_qib_d1 | 0.0 | 0.0 | 0.0 |
| sub_nii_d1 | 0.0 | 0.0 | 0.0 |
| sub_retail_d1 | 0.0 | 0.0 | 0.0 |
| sub_total_d1 | 0.0 | 0.0 | 0.0 |
| sub_qib_d2 | 0.0 | 0.0 | 0.0 |
| sub_nii_d2 | 0.0 | 0.0 | 0.0 |
| sub_retail_d2 | 0.0 | 0.0 | 0.0 |
| sub_total_d2 | 0.0 | 0.0 | 0.0 |
| sub_qib_d3 | 0.0 | 0.0 | 0.0 |
| sub_nii_d3 | 0.0 | 0.0 | 0.0 |
| sub_retail_d3 | 0.0 | 0.0 | 0.0 |
| sub_total_d3 | 0.0 | 0.0 | 0.0 |
| gmp_open | 0.0 | 0.0 | 0.0 |
| gmp_last | 0.0 | 0.0 | 0.0 |
| gmp_pct_open | 0.0 | 0.0 | 0.0 |
| gmp_pct_last | 0.0 | 0.0 | 0.0 |
| list_open_gain_pct | 71.5 | 100.0 | 0.0 |
| list_high_gain_pct | 9.2 | 99.7 | 0.0 |
| list_low_gain_pct | 9.2 | 100.0 | 0.0 |
| list_close_gain_pct | 77.2 | 100.0 | 100.0 |
| idx_ret_5d | 0.0 | 0.0 | 0.0 |
| idx_ret_20d | 0.0 | 0.0 | 0.0 |
| idx_vol_20d | 0.0 | 0.0 | 0.0 |
| vix_level | 0.0 | 0.0 | 0.0 |
| idx_ret_issue_window | 0.0 | 0.0 | 0.0 |
| source | 100.0 | 100.0 | 100.0 |
| source_url | 100.0 | 100.0 | 100.0 |
| n_missing_core | 100.0 | 100.0 | 100.0 |


**Gaps:** US 2022-2023 has no rows. No Kaggle IPO list with offer prices covers those years (searched: ipo, us ipo, nasdaq ipo, nyse ipo, ipo listings, initial public offering, ipo performance, ipo first day, spac ipo, and more). US 2019-2021 is only partly covered: 2019 through the Nasdaq-only jacksoncrow set, 2020-2021 through a few benjaminpo matches. HK has nothing before 2020 (ng3412 starts 2020-01) and nothing after 2024-05 (eren2222 ends there). OTHER is Indonesia only.

Rows per market and listing year (from global.parquet):

| market | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| US | 68 | 60 | 75 | 137 | 183 | 114 | 87 | 157 | 191 | 135 | 19 | 5 | - | - | 168 | 207 | 115 | 1721 |
| HK | - | - | - | - | - | - | - | - | - | - | 105 | 76 | 62 | 63 | 17 | - | - | 323 |
| OTHER | - | - | - | - | - | - | - | - | - | - | - | - | 50 | 52 | - | - | - | 102 |

## Rejected sources (kept from the earlier audit, plus Kaggle items)

| Source | Verdict and reason |
|---|---|
| Nasdaq IPO calendar JSON (`api.nasdaq.com/api/ipo/calendar`) | robots.txt disallows every crawler. Not fetched. |
| stockanalysis.com IPO pages | robots allows `/ipos/`, but the terms of use forbid bots and bulk downloads and forbid ML training. Excluded. |
| stooq.com historical prices | robots.txt: `User-agent: *  Disallow: /`. Not used. |
| Jay Ritter IPOALL.xlsx / IPO-age.xlsx | IPOALL is monthly statistics. IPO-age has 16,030 US IPOs with no offer price or first-day return. No targets. Raw files kept. |
| SEC EDGAR | Needs 424B4 prospectus parsing for offer price, still no first-day close. Not built. |
| aastocks.com IPO pages (HK) | No IPO table in the static HTML (data is loaded by script). |
| HKEX new-listings page | `/eng/newlisting/index.htm` returned 404. Per-deal HKEX allotment PDFs would need a PDF extractor (not installed). |
| Kaggle: fahmirk/e-ipo-realtime-data (Indonesia e-IPO) | Return definition (vs final price or vs listing) unverified; excluded. |
| Kaggle: chetanjadhav2324 (India GMP) and other India sets | India, outside this file. |
| Kaggle: davnnis2003 HK 2018-2020 | Stock-level, short; catalog rejected. |
| Japan, Korea, China A-share, Singapore, UK, Australia IPO lists with first-day price | Searched again (queries: japan ipo, korea ipo, china ipo, singapore ipo, uk ipo london, australia ipo asx): nothing relevant returned. Not found. |

## Kaggle candidates not downloaded

Not downloaded (listed for completeness):
  `raymondsunartio/6000-nasdaq-stocks-historical-daily-prices` (licence "unknown" in metadata; Nasdaq names as of 2020-02;
  overlaps jacksoncrow), `svaningelgem/nyse-daily-stock-prices` (copyright-authors; 499 NYSE
  names, 1962-2026; not checked against the IPO list), `qks1lver/nasdaq-and-nyse-stocks-histories` (882 MB, ends
  2018), `olegshpagin/usa-stocks-prices-ohlcv` (2.9 GB, not downloaded).

## Reproduce

```
.venv/bin/python -m ipo_radar.data.global_ipos                 # cached IPO Scoop pages, local Kaggle files
.venv/bin/python -m ipo_radar.data.global_ipos --refresh       # re-download IPO Scoop pages (robots checked, 1 req/s)
.venv/bin/pytest tests/test_data_global.py -q
```

## Caveats, consolidated

1. IPO Scoop licence is unstated; use is for local modelling only, not redistribution.
2. Kaggle licences are the dataset metadata values. eren2222 descriptive text is copied from another dataset; numbers
   were spot-checked only (Pop Mart 9992.HK, 2020-12-11 opening at 77.10 and closing at 69.00).
3. All price sets are split-adjusted; the split guard is heuristic (see above).
4. The IPO Date / Offer Date semantics are supported by exact-match counts but not checked against a regulator.
5. Survivorship: `+px_join` rows are a selected subset (see above).
6. Indonesia offer-price basis is unverified; issue size unknown.
7. Spot checks passed for: FB, TWTR, BABA (thedevastator open and close match the catalog's known values), Lyft, Zoom,
   CrowdStrike and Beyond Meat (shivamb joins, first-day close consistent with values recalled), Medline (IPO Scoop).
   Generac and GM closes blanked by the disagreement rule, where benjaminpo matched the value recalled.
