# Kaggle data sources (IPO history and index history)

Catalog written 2026-10-15 by the Kaggle search agent. Owned by that agent. Other agents own the other files in
this folder.

## How this was collected

- Access: read-only Kaggle wrapper (`datasets list / files / metadata / download` only). No uploads, kernels or
  account changes. No credentials appear in this file or in the raw folders.
- Raw files: `data/raw/kaggle/<owner>__<slug>/` (git-ignored). One folder per dataset. Nothing was executed from
  the downloads; they were only read with pandas (inside the venv) to check structure and values.
- Searches run: ipo, ipo india, ipo gmp, ipo subscription, ipo listing gain, hong kong ipo, hkex, nasdaq ipo,
  nyse ipo, us ipo, ipo dataset, ipo performance, initial public offering (pages 1-3 where available),
  nifty 50, sensex, india vix, hang seng, nikkei, shanghai composite, nasdaq composite, nyse composite,
  s&p 500 daily, vix daily, stock exchange data, ipo japan / korea / china / singapore / australia / uk
  (no IPO-level dataset returned for those markets).
- 62 dataset folders are on disk. One download (maheshmani13 Sensex) returned HTTP 404 even though the
  wrapper exited 0; its folder was removed. Every remaining folder was checked for error logs.
- Quality checks: row counts, date ranges, null counts, and a handful of known deal or index values, compared
  against other datasets and, where I could, against public listing figures. Items marked "unverified" were
  not matched to a primary source.

Licences are the Kaggle `licenses` field in dataset metadata. "other", "unknown" and "copyright-authors" are
kept as-is and flagged. None of the kept datasets forbids personal analysis. Redistribution rules differ
(see notes).

## Kept: India IPOs

| Slug | Licence | Rows | Date range | Schema columns it can fill | Quality notes |
|---|---|---|---|---|---|
| karanammithul/ipo-data-india-2010-2025 (file `Initial Public Offering (Updated).xlsx`) | CC0-1.0 | 652 | 2010-01-04 to 2026-05-14 (`Date` = listing date) | `company`, `listing_date`, `issue_price` (Offer Price), `issue_size_m` (crores x 10), `sub_qib`, `sub_nii` (HNI, not split), `sub_retail` (RII), `sub_total`, `list_open_gain_pct` (tentative, see note), mainboard and SME mixed with no flag (`market` must be inferred) | Best broad India base. Offer prices agree with the rohan dataset for 99.5% of 565 matched names. Spot checks vs public listing prices: LIC 949 to 867.20, One97/Paytm 2150 to 1955, Nykaa 1125 to 2001, Tata Technologies 500 to 1199.95, Zomato 76 to 115 (from memory, not re-checked on an exchange archive). **"List Price" definition is undocumented**: it equals the Chittorgarh listing open within 1% for only 59% of 562 matched IPOs. Mainboard and SME are not separated. No close/high/low, no GMP, no sNII/bNII, no lot size. The older `Initial Public Offering.xlsx` in the same folder (561 rows, to 2025-08) is superseded. |
| rohandeogaonkar/india-mainboard-ipos-20yr | "other" (dataset says research and education only; upstream terms apply) | 999 IPOs in `finalipodata.csv`; `cg_subs` 869; `cg_listing` 992; `cg_gmp` 44 | 2005-02-18 to 2026-09-24 listing dates (some issue dates to 2026-10) | `issue_price`, `issue_size_m` (Total Shares Offered x price, check), `lot_size`, `sub_qib`, `sub_nii` (bNII+sNII), `sub_retail`, `sub_total`, `list_open_gain_pct`, `list_close_gain_pct`, `gmp_open` / `gmp_last` (GMP column, 470 of 999 non-null), `sector`, `context` via NIFTY 50 at listing (`Nifty 50` columns), `source`=Chittorgarh | Only dataset found with sNII and bNII split, listing open AND close, lot size and 20 years. Listing price in `finalipodata` equals the Chittorgarh listing open (98.8% agree). Post-listing outcomes (`ipo_outcomes.csv`, `6m/12m/24m` returns) are **future information**: never use them as model features. Yahoo-based long-horizon returns skew to survivors (README says so). Licence: built from Chittorgarh, NSE, SEBI and ipowatch pages. Usable for personal analysis; redistribution is not cleared. |
| amansingh2116/indian-ipo-2000-may-2026-data (`ipo_historical_data_all_pages.csv`, `sme_ipo_historical_data_all_pages.csv`) | CC-BY-SA-4.0 | Mainboard 1192; SME 1473 | Mainboard 1999-12-27 to 2026-05-14; SME 2012-03-13 to 2026-05-19 | `sub_qib`, `sub_nii` (NII column), `sub_retail` (RII), `sub_total`, `issue_size_m`, `issue_price`, `lot_size` (SME), `list_close_gain_pct` (listing close column), `market`=IN_MAIN / IN_SME | Only SME source with listing outcomes to 2026. Scraped web tables (MoneyControl layout). 402 subscription cells null in mainboard file, 126 CMP cells null. Column headers are long and need mapping. **Not spot-checked** beyond row structure. |
| warcoder/ipo-mainboard-and-sme-basic-details-india (`IPO details 2007-2024(FEB).csv`) | "other" | 1764 | 2007-01-08 to 2024-02-28 (open date) | `open_date`, `close_date`, `listing_date` (1009 rows null), `issue_price`, `issue_size_m`, `lot_size` (38 null), `exchange` (BSE, NSE, or both). Covers mainboard and SME. | Best source for **lot size** and dates pre-2010 to 2024. No subscription and no listing prices. Overlaps karanammithul on only 25% of names (format differs), so it is useful as a second universe check. Licence unclear: personal use only. |
| thabresh/ipo-performance (`ipo.csv`) | DbCL-1.0 (open database licence, share-alike) | 320 (multi-row header) | 2010 to 2022 | `issue_price`, `issue_size_m`, `sub_*`, `list_open_gain_pct`, `list_close_gain_pct`, `cmp` | Carries listing **open and close** for names that karanammithul lacks. Needs header rows skipped (two header rows). Overlaps karanammithul on about 69% of names. |
| balabaskar/indian-ipo-dataset-2010-2021 (`ipo_2010_2021.csv`, `NIFTY_2010_2021.csv`) | CC0-1.0 | 264 IPOs; 2873 Nifty days | IPOs 2010-02-03 to 2021-07-29; Nifty 2010-01-04 to 2021-07-30 | IPOs: `sub_*`, `issue_price`, `list_open_gain_pct`, `list_close_gain_pct`, `issue_size_m`. Nifty: `NIFTY50` OHLC (`indices.parquet`). | Spot check: Zomato offer 76, listing open 115. Nifty file has full OHLC for 2010-2021 (no nulls). |
| shuvendupritam/ipo-details-2014-2023-nse-bse (10 CSVs, files "(1)" to "(10)") | MIT | About 320 across files | 2014-02-28 to 2023-12-29 | `issue_price`, `list_close_gain_pct` (listing-day close), `company`, `listing_date`, `current price` | Each file covers a different year slice (not duplicates). Listing-day close only, no open, no subscription. Good for close-price targets 2014-2023. |
| rimarusama/ipo-pastissue-india-nse (`IPO-PastIssue-05-May-2025.csv`) | MIT | 1122 (mainboard and SME) | 2012-09-04 to 2025-04-29 (issue start) | `open_date`, `close_date`, `issue_price` (19 null), `price_band` (text "Rs.55 TO RS. 58", needs parse), `listing_date` (some "-"), `segment` (EQ / SME), `symbol` | Only India source with **price band text** for most IPOs and dates through April 2025. No subscription. Use for dates and price bands. |
| ishans474/indian-ipo-underpricing-2010-2025 | CC0-1.0 | 509 | 2010-01-04 to 2025-08-06 | `issue_price`, `sub_*`, `list_close_gain_pct` (target "Underpricing"), NIFTY 20d/60d return and volatility, India VIX at issue (context) | Carries **issue-time market context** (NIFTY 20/60-day returns, India VIX). Target matches listing gain for the JSW Energy example (2.00%). The derived columns were built by the publisher; check their timing before use as features. Names match karanammithul 100%. |
| nikhilraj7700/ipo-watch-debuts-gmps-trajectories (`Mainline IPO GMP Performance.csv`) | MIT | 152 | 2020-09-21 to 2023-11-15 (listing date) | `gmp_last` (GMP, unofficial), `sub_total` (Subscription), `issue_size_m`, `issue_price`, `list_price` (listing price), `gmp_pct` derived | GMP source is "ipowatch" per the README (not in this file). Timing of GMP reading is not documented. Useful for a **GMP baseline test** on 2020-2023 deals. Only total subscription, no categories. |
| nilayyyka/india-ipo-gmp-vs-actual-listing (`gmp_vs_listing.csv`) | CC-BY-4.0 | 35 | 2026-01-10 to 2026-10-08 | `gmp_last`, `issue_price`, `list_close_gain_pct` (actual listing gain column), `gmp_pct` | Tiny and recent (2026). GMP source not stated. Direction matched actual in 26 of 35. Use as a small recent validation set, not for training. |
| roshan0kumar0/ipo-dataset (`IPO In Year.csv`) | CC0-1.0 | 190 | Listed 2025-12-31 to 2026-07-01 | `issue_price`, `list_close` (Listing Day Close, 13 null), `ltp` (current price), `list_*_gain` | Recent NSE listings. Not verified against any primary source (I have no reliable 2026 reference). Listing-day close '-' for some rows. Use as a freshness check, not a training base. |
| soumyadipghorai/all-ipo-stocks-of-moneycontrol | DbCL-1.0 | 302 | 2010-01-07 to 2022-08-02 | `sub_*`, `issue_price`, `list_open_gain_pct`, `list_close_gain_pct` | Same MoneyControl family as thabresh; kept as a cross-check only. |
| leanderangelo/indonesia-ipo-jan2022-july2023 (`IPO 2023 Analysis - Cleaned.csv`) | CC-BY-NC-SA-4.0 (non-commercial, share-alike) | 102 | 2022-01-25 to 2023-12-06 | OTHER: `issue_price` (Starting_Price, basis unverified), `list_close_gain_pct` via Day-1 close, `day_1..day_7` close path | Indonesia (IDX). Price columns are IDR; the `IPO` column holds IDR values with dot thousands separators (text). Non-commercial licence: fine for this personal project, flag if it is ever shared. |
| fahmirk/e-ipo-realtime-data (`e-IPO Data.csv`) | ODbL-1.0 (attribution, share-alike) | 252 (243 closed, 8 canceled, 1 offering) | Recent Indonesian e-IPO deals (dates not parsed) | OTHER: `issue_price` (Final Price, Rp), `list_*_gain` via `Return D1..D7`, `listing_date`, `status` | Indonesia, recent. Return definition (vs final price or vs listing price) is **unverified**. Filter canceled rows. The warrant file in the same folder is out of scope. |
| ng3412/hkex-ipo-dataset (`hkex_ipo_ds.xlsx`, with `hkex_ipo_data_dictionary.md`) | MIT | 600 | 2020-01-08 to 2026-06-15 (listing date) | HK: `issue_price` (offer_price_hkd), `sub_retail` (HK public-offer multiple, times), `issue_size_m` (gross proceeds HKD m), `global_offer_shares`, `intl_placing_shares`, applications counts, `company`, `listing_date`, `stock_code` | Only HK per-deal source found. Spot checks against offer prices I recall (not re-checked on an exchange archive): Pop Mart 38.50, CATL 263.00, Mixue 202.50. **Missing: first-day price (no listing gain) and the international tranche multiple (`sub_qib` stays NaN).** One stock code appears twice on the same listing date (6600, documented). Eight rows lack successful-application counts. |
| proselotis/financial-ipo-data (`IPODataFull.csv`, latin-1) | CC0-1.0 | 3762 (3699 symbols) | 1996 to 2018 (`Year`, `Month`, `Day`) | US: `list_close_gain_pct` (via `closeDay0` vs offer, offer price not identified), `list_high/low` (day 0), `symbol`, `listing_date`, very wide fundamentals (about 1650 columns, mostly pre-IPO financials) | **openDay0 disagrees with known first-day opens**: FB 38.00 (known 42.05), TWTR 44.00 (known 45.10). closeDay0 matches: FB 38.23, TWTR 44.90, BABA 93.89. Treat open values as unreliable. Column semantics are undocumented. Use for US first-day close only, and only after cross-checking. |
| thedevastator/dataset-on-ipo-from-2010-2018 (`ipo_stock_2010_2018.csv`, `ipo_clean_2010_2018.csv`, `ipo_stock_2010_2018_v2.csv`) | CC0-1.0 | 1136 (stock file, first-day data) / 1600 (clean file, no first-day data) | 2010-01-22 to 2018-12-27 (date priced) | US: `issue_price` (Price), `issue_size_m` (Offer Amount), `shares`, `list_open_gain_pct` (firstday_open vs Price), `list_close_gain_pct` (firstday_adjclose), `exchange` (NASDAQ / NYSE / AMEX), `company`, `symbol`, `sector`, `industry` | **Best US source found.** First-day open and close match known values I could recall: FB 42.05 / 38.23, TWTR 45.10 / 44.90, BABA 92.70 / 93.89, GM 35.00 open. `firstday_adjclose` is adjusted (equals raw for these). No high/low. Stops at 2018 (no 2019-2026 US first-day data found). `spy.csv` in the folder is an ETF, not an index, and is ignored here. |
| shivamb/company-ipos-2019-2021 (`company_ipo.csv`) | CC0-1.0 | 1765 | 2019-01-04 to 2021-12-21 | US listing calendar only: `symbol`, `listing_date`, `issue_price` ($10 for SPAC units), `company` | **No first-day trading data.** Mostly SPAC units. Lowest priority; only useful as a 2019-2021 US symbol and date list. |

### Notes on India IPO merging (for the integrator)

- Duplicate families: the karanammithul / manitmehta / chetanjadhav / skworld / ishans474 set covers the same
  2010-2025 IPO universe (name overlap 97-100%). The MoneyControl family (balabaskar, thabresh, soumyadip, aimack,
  paradoxlover, mananpatell, deepaksaw, rakesh) overlaps about 67-70% with karanammithul. Do not add them up.
- GMP: `chetanjadhav2324` has a GMP column that is non-null for **all 652** rows, including 2010-2015 (when no
  public GMP series is known to exist) and 40 zero values. Its provenance is undocumented, so it is rejected for
  GMP use (see rejected list). GMP from rohandeogaonkar (470 non-null) and nikhilraj7700 (152) is the more
  credible source. Both are unofficial grey-market figures and must stay labelled as such (SPEC rule 2).
- Subscription categories: sNII and bNII only in rohandeogaonkar (`cg_subs`). Employee and shareholder categories
  are not in any dataset found.

## Kept: indices and global price history

| Slug | Licence | Rows | Date range | Schema / index symbols it can fill | Quality notes |
|---|---|---|---|---|---|
| belbino/global-financial-markets-dataset-2010-present (`global_financial_markets_2000_Now.csv`) | CC0-1.0 (underlying Yahoo Finance data) | 191,170 (34 symbols) | 2000-01-03 to 2026-10-13 | `SP500` (^GSPC), `NASDAQ_COMP` (^IXIC), `HSI` (^HSI), `SHCOMP` (000001.SS), `NIKKEI` (^N225), `SENSEX` (^BSESN), `NIFTY50` (^NSEI, from 2007), `KOSPI`, `DAX`, `FTSE` | **Best current daily series found.** Only dataset with S&P 500, Nasdaq, Hang Seng, Shanghai, Nikkei, Sensex and Nifty all running to today. Null closes: 0; duplicate date-symbol pairs: 0. Nifty 2020-03-23 close 7610.25 agrees with two other datasets. S&P 2020-03-16 close 2386.13. No NYSE Composite, no VIX. Yahoo data: check Yahoo terms before redistribution. |
| mattiuzc/stock-exchange-data (use `indexProcessed.csv`; `indexData.csv` has 2204 empty rows) | "copyright-authors" (no explicit reuse grant; personal use only) | 104,224 (14 indices) | 1965-01-05 to 2021-06-03 | `NYSE_COMP` (NYA), `NASDAQ_COMP` (IXIC), `HSI`, `SHCOMP` (000001.SS), `NIKKEI` (N225), `NIFTY50` (NSEI), No S&P 500 in this file. Also GSPTSE, GDAXI, SSMI, KS11, TWII, 399001.SZ, N100, J203.JO | **The "Stock Exchange Data" style dataset** the user asked for. Long history (NYA from 1965). Spot checks: IXIC 2000-03-10 close 5048.62, N225 1989-12-29 close 38915.87, NSEI 2020-03-23 close 7610.25, all matching known values. Stops in mid-2021. The first trading day of some series has O=H=L=C (e.g. HSI 1986-12-31): check first rows. `indexInfo.csv` maps symbols to exchanges. |
| henryhan117/sp-500-historical-data (`SPX.csv`) | DbCL-1.0 | 23,323 | 1927-12-30 to 2020-11-04 | `SP500` long history (pre-2000 part) | Long S&P 500 history for training context before 2000. Stops at 2020-11 (use belbino for later). |
| mukhazarahmad/worldwide-stock-market-indices-data (`INDICES_DATA.csv`) | "copyright-authors" | 6510 rows, 33 indices | 1996-12-31 to 2022-11-08 | `NYSE_COMP` (^NYA), `SP500` (^GSPC), `NASDAQ_COMP` (^IXIC), Dow (^DJI), others | **Two-row header** (ticker row, then field row): needs parsing. NYA 1996-2022. Good second source for NYA. |
| pavankrishnanarne/global-stock-market-2008-present (yearly CSVs 2008-2023) | CC0-1.0 | About 40,000 across 16 yearly files | 2008-08-01 to 2023-07-28 | `NYSE_COMP` (^NYA) from 2008; also ^GSPC, ^IXIC, ^DJI, 000001.SS, ^N225, ^NSEI, ^BSESN | Ticker column identifies symbol. Only NYA source covering 2023 besides code1110 (which stops 2022). |
| code1110/yfinance-world-indices-price-data (`YF-*.csv`) | "other" (Yahoo data) | About 4800 per index | 2002-12-02 to 2022-02-02 | `NYSE_COMP` (YF-NYA), `VIX` (YF-VIX), `N225`, `DJI`, `DAX`, `SHCOMP` (YF000001-SS to 2022-01-28), `AXJO`, `FCHI`, `GSPTSE`, `MERV`, `XAX` | Yahoo-sourced, redistribution unclear. Gives VIX 2002-2022 and NYA to 2022. |
| guillemservera/vix-cboe-volatility-index-daily-updated (`vix_daily.csv`) | CC0-1.0 | 8685 | 1990-01-02 to 2024-06-24 | `VIX` (US, CBOE) | Daily US VIX from inception. **Ends 2024-06-24**; no source found for 2024-06 to 2026. |
| debashis74017/stock-market-index-data-india-1990-2022 (`NIFTY 50_Data.csv`, `NIFTY BANK_Data.csv`, sector files, `VIX_History.csv`, `INDIA VIX_minute.csv`) | CC0-1.0 | NIFTY 50: 5645; VIX_History: 8216; minute VIX: 938,695 (42 MB) | NIFTY 2000-01-03 to 2022-09-09; VIX_History 1990-01-02 to 2022-08-11; minute VIX 2015-01-09 to 2025-03-05 | `NIFTY50` (daily OHLC to 2022-09), `VIX` (**this VIX_History is US CBOE VIX**: 1990-01-02 close 17.24 matches its inception), `INDIAVIX` (only via minute file, daily close not derived) | NIFTY 2020-03-16 9197.40 and 2020-03-23 7610.25 match sudalairajkumar. Minute India VIX file would need resampling to a daily close (not done, and daily close definition needs checking against the NSE file). |
| sudalairajkumar/nifty-indices-dataset (`NIFTY 50.csv`, `INDIAVIX.csv`, sector and cap indices) | CC-BY-SA-4.0 | NIFTY 50: 5970; INDIAVIX: 3061 | NIFTY 2000-01-03 to 2024-01-02; INDIAVIX 2009-03-02 to 2021-07-07 | `NIFTY50` (to 2024-01), `INDIAVIX` (2009-2021), NIFTY 500 | Matches debashis on the 2020 check dates. Adds P/E and P/B columns. Share-alike licence: keep derived outputs under the same terms if shared. |
| jatinsh/nifty-50-index-2000-2024 (`NIFTY50_2000_to_2024.csv`) | CC0-1.0 | 5972 | 2000-07-03 to 2024-07-05 | `NIFTY50` (extends the sudalairajkumar file to 2024-07) | Secondary. Its `vix_data.csv` (2023-06 to 2024-07) is not verified as India VIX; not used. |
| ahaanverma00/nifty-500-and-india-vix-daily-data-2010-2026 (`INDIA VIX 01 Jan 2010 - 20 Mar 2026.csv`) | CC BY 4.0 | 4021 | 2010-01-04 to 2026-03-19 | `INDIAVIX` (2010 to 2026-03) | Main India VIX source to 2026-03. The `NIFTY 500` file's close column did not parse (all null): not used. |
| alangeorgev/india-vix-historical-2025 (`India VIX Historical Data 101.csv`) | Apache-2.0 | 3822 | 2010-01-02 to 2025-12-05 | `INDIAVIX` (2010 to 2025-12) | Investing.com-style export (`Price` column is the close). Fills India VIX gap between 2026-03 and today only with saurabhbadole. |
| saurabhbadole/indian-stock-market-indices-daily-updates (`India VIX.csv` only) | "other" | 1664 | 2020-01-02 to 2026-10-14 | `INDIAVIX` (to today) | Only file kept from this dataset. Its NIFTY and SENSEX files cover only about one month (rolling window), so they are rejected as history. Licence unclear: personal use. |
| avtnshm/bombay-stock-exchange-bse-sensex-index-2005-2024 (`CSVForDate (1).xlsx`) | Apache-2.0 | 6989 | 1995-11-03 to 2024-01-12 | `SENSEX` (1995 to 2024-01) | Long Sensex history. Overlaps belbino (2000 onward) for the same dates; use belbino for recent, this for pre-2000. |
| rushil999/hang-seng-historical-prices (`HSI.csv`) | Apache-2.0 | 9339 (252 close nulls) | 1986-12-31 to 2023-10-20 | `HSI` (long history to 2023-10) | Long HSI history. Belbino covers 2000-2026 for recent. |
| rushil999/nikkei-225-historical-prices (`N225 (1).csv`) | Apache-2.0 | 15085 (625 close nulls) | 1965-01-05 to 2023-10-20 | `NIKKEI` (to 2023-10) | Secondary to belbino/mattiuzc for the same years. |
| ruanqian/shanghai-stock-composite-index (`上证综合指数前复权日线数据.CSV`) | CC0-1.0 | 6231 | 1990-12-19 to 2016-06-08 | `SHCOMP` (1990 to 2016) | GBK-encoded file with Chinese column names (read with `encoding="gbk"`). Only source for SHCOMP before 1997. Adjusted (前复权) series. |

## Rejected datasets (and why)

| Slug | Reason |
|---|---|
| debashis74017/nifty-50-minute-data | 874 MB minute data (size). Daily counterpart taken from debashis74017 index file. |
| debashis74017/algo-trading-data-nifty-100-data-with-indicators | 4.1 GB intraday stock data. Out of scope and too large. |
| debashis74017/stock-market-data-nifty-100-stocks-5-min-data; ...nifty-50-stocks-1-min-data | Intraday stock-level data; not index history. Not downloaded. |
| hk7797/stock-market-india | 1.6 GB stock-level data. Not downloaded. |
| leowang7/hkex-stock | 969 MB HKEX equity file (`hkex_equity_data_20221022_all.csv`) with no IPO data. Not downloaded. |
| chetanjadhav2324/indian-ipo-dataset-with-grey-market-premium-gmp | Duplicate of karanammithul (100% name overlap). Its GMP column is non-null for all 652 rows, including 2010-2015, and has 40 zeros; provenance undocumented. Not used for GMP. |
| skworld/indian-ipo-dataset | Duplicate of karanammithul (every name in the train file appears there). |
| manitmehta/comprehensive-indian-ipo-dataset-2010-2025 | Duplicate of karanammithul (97% overlap). Adds no fields. |
| aimack/ipo-data-india2021 | Duplicate of the MoneyControl family (67% overlap with karanammithul). Covered by thabresh. |
| paradoxlover/ipo-history-data | Licence "unknown". Duplicate of MoneyControl family (to 2021). |
| mananpatell/ipo-v4 (`IPO2.csv`) | Licence "unknown". Duplicate of the balabaskar/karanammithul universe. |
| deepaksaw/ipo-dataset | Licence "unknown". Duplicate of thabresh `ipo.csv` (same layout and row count). |
| rakeshhansrajani/ipo2010-2023 (`IPOs.xlsx`) and rakeshhansrajani/ipo-sme-2010-2023 (`IPO_sme.xlsx`) | Licence "unknown". Duplicates of MoneyControl family (2010-2023). |
| divyanshbhatia007/ipo-performance-dataset | Licence "unknown". No header row (first record read as a header); only 2022-2025, 181 rows. |
| jonathanbesomi/cboe-volatility-index-vix (`VIX.csv`) | Licence "unknown". Duplicate of guillemservera VIX over 1990-2021. |
| tunguz/vix-cboe-volatility-index (`vix-daily_csv.csv`) | CC BY-NC 4.0 (non-commercial). Duplicate range (2004-2020) of guillemservera. |
| kornilovag94/vix-index-cboe-volatility-index (`VIX_data.csv`) | CC BY-NC 4.0. Semicolon-separated with decimal commas and a trailing bad row. Duplicate of guillemservera. |
| pdquant/sp500-daily-19862018 (`spx.csv`) | Subset of henryhan117 (1986-2018) and belbino. Duplicate. |
| pakkanmeric/hang-seng-prices; pakkanmeric/nasdaq-composite | Licence "unknown". Duplicates of rushil999 HSI and mattiuzc IXIC to 2022. |
| davnnis2003/hong-kong-stock-prices-23-may-2018-22-may-2020 | Short (2018-2020). Mostly individual HK stocks, not index history. HSI duplicates belbino. |
| ylmzasel/nikkei-225-daily-index-20202025 | Duplicate of belbino ^N225 (2020-2025 covered). |
| arslanr369/nasdaq-historical-prices-2014-2024 | Duplicate of belbino ^IXIC for 2014-2024. |
| ayushchandramaurya/nifty50-index-data; maheshmani13/nifty-50-nsei-index-price-data-2015-2023 | Duplicates of sudalairajkumar and belbino NSEI over the same years. |
| manish4348/indiavix-jan2016-apr2023 | Duplicate of ahaanverma00 India VIX for 2016-2023. |
| abhilashanil/india-vix (`INDIA_VIX.csv`) | Licence "unknown". Covers 2009-2020; the same period is in sudalairajkumar (CC BY-SA). Redundant. |
| aliiihussain/daily-global-stock-market-indicators | Only 2020-2024, duplicates belbino over those years. |
| gelasiusgalvindy/stock-indices-around-the-world | Licence "other". 2017-2022 only. Several series have all-null close (`Close*` headers). Redundant. |
| maheshmani13/bse-sensex-10-year-stock-price | **Download failed** (HTTP 404 on the file; the wrapper exited 0). Folder removed. Sensex covered by avtnshm and belbino. Re-try later if a 2015-2025 Sensex file is needed. |
| saurabhbadole NIFTY 50 / SENSEX / other index files | Only about one month of data (rolling window). |
| andrewmvd/india-stock-market `stocks_df.csv` | 223 MB stock-level file. Not downloaded. Its `nse_indexes.csv` (29 NSE indices incl. NIFTY 50 and INDIA VIX, 1995-11-06 to 2022-01-31, 87,536 rows) **is kept** (extracted from the zip into the same folder). Not described in the table above because of time; integrator should check it before use. |
| yash16jr/s-and-p500-daily-update-dataset | 0 KB file in the search listing. Not downloaded. |
| shiveshprakash/34-year-daily-stock-data | Stock-level, not index history. Not downloaded. |
| Japan, Korea, China A-share, Singapore, UK, Australia IPO datasets | None found in searches on Kaggle (queries returned unrelated stock-price files). |

## Honest gaps

- **India IPO**: no Kaggle source has sNII/bNII split for more than one dataset; no employee or shareholder
  categories; no day-wise subscription; GMP is incomplete (470 of 999 in the best file). Listing price definitions
  disagree between karanammithul and Chittorgarh (59% agreement within 1%). 2026 rows are not verified.
- **US IPO**: first-day open and close only for 2010-2018 (thedevastator). Close-only for 1996-2018 (proselotis,
  with unreliable open). **No US first-day data found for 2019-2026.** No high or low for any US IPO (proselotis
  has day-0 high/low but unreliable open).
- **HK IPO**: offer price and public-offer multiple for 2020-2026 (ng3412). **No first-day price, no
  international tranche multiple**, so there is no HK listing-gain target from Kaggle.
- **OTHER**: Indonesia only (2022-2023 and recent e-IPO). No China A-share, Japan, Korea, Singapore, UK or
  Australia IPO datasets found.
- **Indices**: US VIX stops at 2024-06-24 (no source found for 2024-06 to 2026). NYSE Composite has no source
  after 2023-07 (pavankrishnanarne). S&P 500 has a source for every year from 1927 to 2020 (henry) and from 2000 to
  today (belbino). India VIX runs 2010 to 2026-03 (ahaanverma), 2010 to 2025-12 (alangeorgev), and 2020 to
  2026-10 only from saurabhbadole (licence "other"). The 2026-03 to 2026-10 India VIX therefore rests on one
  dataset with an unclear licence.
- Yahoo-derived series (belbino, code1110, pavankrishnanarne, guillemservera) may be subject to Yahoo's terms
  on redistribution. Personal use only, no republishing.

## Priority (what should fill which gap)

- **India IPO outcomes and subscription**: `rohandeogaonkar/india-mainboard-ipos-20yr` (main table, 2005-2026),
  then `karanammithul/ipo-data-india-2010-2025` (issue price, sub_retail/qib/nii, listing price, 2010-2026).
- **India IPO listing close and SME**: `amansingh2116/indian-ipo-2000-may-2026-data` (SME and mainboard to 2026),
  `thabresh/ipo-performance` (listing open and close, 2010-2022).
- **India GMP**: `rohandeogaonkar` (`gmp`), then `nikhilraj7700/ipo-watch-debuts-gmps-trajectories` (2020-2023).
- **India lot size and dates pre-2010**: `warcoder/ipo-mainboard-and-sme-basic-details-india`.
- **US first-day open and close (offer price too)**: `thedevastator/dataset-on-ipo-from-2010-2018`.
- **US first-day close only, longer history**: `proselotis/financial-ipo-data` (close only; open not used).
- **HK offer price and public-offer multiple**: `ng3412/hkex-ipo-dataset`.
- **OTHER (Indonesia)**: `fahmirk/e-ipo-realtime-data` (recent), `leanderangelo/indonesia-ipo-jan2022-july2023`.
- **Indices**: `belbino/global-financial-markets-dataset-2010-present` (SP500, NASDAQ, HSI, SHCOMP, Nikkei, Sensex,
  Nifty to today), `mattiuzc/stock-exchange-data` (NYA, IXIC, long history to 2021), `henryhan117/sp-500-historical-data`
  (S&P 1927-2020), `debashis74017/stock-market-index-data-india-1990-2022` and `sudalairajkumar/nifty-indices-dataset`
  (NIFTY to 2022-2024), `ahaanverma00/nifty-500-and-india-vix-daily-data-2010-2026` (India VIX), `guillemservera/vix-cboe-volatility-index-daily-updated` (US VIX to 2024-06).
