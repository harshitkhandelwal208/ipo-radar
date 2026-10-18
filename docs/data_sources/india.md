# India IPO data: sources, rules and coverage

Built by `src/ipo_radar/data/india.py` (+ `india_sources.py` for the Kaggle loaders). Output:
`data/processed/india.parquet`, columns exactly `schema.ALL_COLS`. Run `python -m ipo_radar.data.india`.
Tests: `tests/test_data_india.py` (offline; fixtures in `tests/fixtures/india/` and `tests/fixtures/india/kaggle/`).

## Licences and personal use

- **HuggingFace `sohomghosh/Indian_IPO_datasets`**: CC BY-NC-SA 4.0 (dataset README front matter). Personal,
  non-commercial use only. The output parquet is derived from it and carries the same restriction.
  Its IPO sheets come from chittorgarh.com and its GMP files from investorgain.com.
- **Kaggle sources used** (licence as shown in the catalog, `docs/data_sources/kaggle.md`):
  - `rohandeogaonkar/india-mainboard-ipos-20yr`: "other" (research and education; upstream terms apply). Chittorgarh-derived.
  - `karanammithul/ipo-data-india-2010-2025`: CC0-1.0.
  - `amansingh2116/indian-ipo-2000-may-2026-data`: CC-BY-SA-4.0.
  - `warcoder/ipo-mainboard-and-sme-basic-details-india`: "other" (personal use only).
  - `thabresh/ipo-performance`: DbCL-1.0.
  - `balabaskar/indian-ipo-dataset-2010-2021`: CC0-1.0.
  - `nikhilraj7700/ipo-watch-debuts-gmps-trajectories`: MIT (GMP source is ipowatch, per its README).
  - `rimarusama/ipo-pastissue-india-nse`: MIT.
  Personal analysis only. Do not redistribute the derived parquet.

## Files used

| Source | File | Used for |
|---|---|---|
| HF | `ipo_mainline_final_data_v18.xlsx` (sheet data_ordered) | IN_MAIN anchor rows, NSE listing prices, subscriptions (final total), HF-only fields |
| HF | `SME_data_final_v11.xlsx` (sheet data_ordered) | IN_SME anchor rows, overall listing prices, last-day subscriptions |
| HF | `GMP_data/GMP_Mainline_2019_2024YTD_Investorgrain_CLEAN.xlsx`, `GMP_data/GMP_SME_2019_2024YTD_Investorgrain_CLEAN_v2.xlsx` | gmp_last (linked to HF rows) |
| Kaggle rohan | `cg_issue.csv`, `cg_subs.csv`, `cg_listing.csv` (joined on cg_ipo_id) | Chittorgarh mainboard: issue amount, fresh capital, exchange, dates, categories (QIB, NII, sNII, bNII, retail, employee, shareholder, total), listing open/close |
| Kaggle rohan | `finalipodata.csv` | lot size, offer price (linked by name and date, as source `rohan_main`) |
| Kaggle karan | `Initial Public Offering (Updated).xlsx` | QIB, HNI (= NII), RII (= retail), total, issue size, offer price, listing date |
| Kaggle aman | `ipo_historical_data_all_pages.csv` (mainboard) | QIB, NII, RII, total, issue size, offer price, listing date. Exact repeats across scraped pages dropped |
| Kaggle aman | `sme_ipo_historical_data_all_pages.csv` | SME lot size, issue size, offer price, listing open and close (0 = missing) |
| Kaggle warcoder | `IPO details 2007-2024(FEB).csv` | exchange, lot size, issue price and size. Market from the exchange label ("BSE SME" / "NSE SME" = IN_SME, else IN_MAIN). Only rows with a listing date can link |
| Kaggle thabresh | `ipo.csv` (two header lines, positional) | QIB, HNI, RII, total, issue price, listing open and close |
| Kaggle balabaskar | `ipo_2010_2021.csv` | QIB, HNI, RII, total, issue price, listing open and close |
| Kaggle nikhil | `Mainline IPO GMP Performance.csv` | GMP (only if it validates against HF, see below), total subscription, issue price |
| Kaggle rimarusama | `IPO-PastIssue-05-May-2025.csv` | market (EQ = IN_MAIN, SME = IN_SME), dates, issue price |

## Not used, and why

- `chetanjadhav2324` GMP: rejected in the catalog (non-null for all rows, zeros, undocumented). Not loaded.
- `rohandeogaonkar` GMP (`cg_gmp.csv`, GMP column): not used, per the instruction that GMP comes from HF and nikhilraj7700 only.
- `rohandeogaonkar` `ipo_outcomes.csv` and the 6/12/24-month returns: future information, never a feature.
- `karanammithul` and `amansingh2116` "List Price" (and karan "Listing Gain", which equals List Price / Offer - 1 exactly): not used.
  Tested against the listing-day fields, the agreement is: karan List Price vs HF NSE open 57% (262 pairs),
  vs rohan Chittorgarh listing open 43% (484), vs rohan listing close 5% (485); amansingh List Price vs rohan
  open 40% (292). No definition reaches a reliable match, so no listing target uses it.
- `amansingh2116` "Listing Gains(%)" and "CMP" columns: not used (derived or current prices).
- `thabresh` / `balabaskar` / `karan` / `amansingh` mainboard are the MoneyControl family. Not loaded as separate
  "truths": they are compared with the Chittorgarh family below.
- `ishans474/indian-ipo-underpricing-2010-2025`: a derived copy of the karan universe with publisher-built
  targets and features. Not loaded.
- `soumyadipghorai`, `shuvendupritam`, `roshan0kumar0`, `ng3412` (HK), `leanderangelo`, `fahmirk`, index files:
  out of scope for this step (no new fields needed, or not India IPO history).
- Kaggle rows for IPOs listed before 2009-01-01 (the HF window starts 2009) and after the build date are excluded.

## Linking records to IPOs

1. Two records link only if their listing dates differ by at most 1 day AND (normalised name equal, OR name
   tokens where one name is the leading tokens of the other on the same day, with at least two tokens, OR
   NSE symbol equal and non-empty). Names are lower-cased, brackets and punctuation removed, stop words
   (ipo, limited, ltd, sme, ...) dropped.
2. A link is kept only if each record has exactly one candidate in the other source (1:1).
3. Links are merged by union-find. An IPO holds at most one record per source, and all its records must fall
   within one day of each other (no chaining across a longer span).
4. Records that cannot link stay separate. A Kaggle-only IPO is kept only if it has at least one outcome
   (any subscription value or a listing price). Basic-details-only rows (warcoder, rimarusama) are excluded.

## Resolving fields

- Values from all records of an IPO are compared pairwise.
  Subscriptions: 1% relative. Issue price, issue amount, listing prices and GMP: 0.5% relative.
  Fresh-issue share: 0.5 percentage points. Dates, lot size, exchange and market: must be equal.
- All pairs agree: the value of the highest-priority source is used (HF, rohan, rohan_main, aman, aman_sme,
  karan, warcoder, thabresh, balabaskar, rimarusama, nikhil).
- Any pair disagrees: the cell is NaN and the disagreement is counted. Nothing is picked silently.
- One source only: the value is kept and counted as single-source.
- HF-only fields (price band, offer-for-sale share, day-wise subscriptions, PAT margin, loss flag) come from HF.
- Targets: listing open, high, low and close gains are computed from the resolved listing prices and the
  resolved issue price. A listing-day bar whose open or close lies outside its high-low range has its high and
  low gains blanked. A gain below -100% is impossible and blanked. Gains outside -90% to +500% are kept and
  counted (close: 58 cases).

### Disagreement counts (overlap = IPOs with two or more values for the field)

| field | overlap | disagree | rate |
|---|---:|---:|---:|
| sub_total | 841 | 186 | 22% |
| sub_qib | 802 | 160 | 20% |
| sub_nii | 802 | 180 | 22% |
| sub_retail | 802 | 214 | 27% |
| px_open | 948 | 92 | 10% |
| px_close | 943 | 232 | 25% |
| issue_price | 1950 | 6 | 0.3% |
| issue_cr | 1516 | 123 | 8% |
| fresh_pct | 91 | 3 | 3% |
| listing / open_dt / close_dt | 1961 / 1041 / 1041 | 3 / 2 / 7 | <1% |
| lot / exchange / market | 708 / 435 / 1755 | 1 / 0 / 0 | ~0% |
| gmp | 141 | 0 | 0% |

Pairwise, by source (sub_total, same IPO): the Chittorgarh family agrees with itself. HF vs rohan: 0 of 399.
The MoneyControl family agrees with itself: aman vs karan 0 of 457, balabaskar vs thabresh 0 of 261.
Across the two families they differ: karan vs rohan 29% (150 of 513), aman vs rohan 28% (114 of 414),
HF vs karan 16% (55 of 355) and HF vs aman 15% (40 of 273). Retail disagrees most (karan vs rohan 35%).
The investorgain final total in the GMP files agrees with HF's total on 97% of matched rows (checked in the first
build), so the Chittorgarh family is the better-supported side. The rule still blanks the disagreeing cells,
it does not choose a side.

Listing prices: HF NSE listing prices agree with rohan (Chittorgarh) on 99.8% of IPOs. Against the
MoneyControl family they differ a lot: thabresh and balabaskar vs rohan about 50% (thabresh = Price.1 / Price.2
open and close, balabaskar Listing Open / Listing Close). amansingh SME listing close disagrees with HF's SME
Last_Trade on 45% of overlaps (open agrees on 98%). Those cells are NaN.

### GMP validation

nikhilraj7700 GMP was compared with HF GMP on 141 IPOs (both values present). Agreement 100%. The rule
is that it is used only if agreement is at least 95% (GMP_MIN_AGREEMENT). It is used. Its rows without an HF
match fill gmp_last. gmp_open stays NaN: the tracker has one reading per IPO and no timestamp.

## Universe and exclusions (counts for this build)

- Kept: 2787 IPOs (IN_MAIN 1029, IN_SME 1758),
  listing dates 2009-03-02 to 2026-10-08.
- Excluded: unknown market 272; no listing date 954 (includes warcoder rows without one); outside the window
  661 (before 2009-01-01, or after the build date 2026-10-15, including upcoming rohan rows); Kaggle-only without
  an outcome 236.
- Exact duplicate rows within a Kaggle source dropped: 174 (amansingh repeats IPOs across scraped pages).
- HF share-count cells blanked (values above 10,000 in subscription columns): 6.
- Listing bars inconsistent in the source (open or close outside high-low): high and low gains blanked for 10 IPOs.
- Impossible gains (below -100%): 0.

## Chittorgarh backfill: not done

The mainboard 2015+ retail coverage is 74% after resolution. The gap is mostly conflicts between the two
families, not missing sources. For 2015+ mainboard IPOs (entity level, before the universe filter), 208 have a
retail value in some source that the rule blanks as a conflict, and 23 have no retail value in any source.
A Chittorgarh page would be a third source that agrees with rohan and disagrees with karan and aman, so it would
be blanked by the same rule. The Chittorgarh history list is client-side rendered, so no bounded list of
per-IPO URLs is available for the 23. No pages were fetched. `data/raw/chittorgarh/` holds 27 IPO pages from
the live work; no history cache was created.

## Unmapped schema columns

- `issue_size_usd_m`: no FX series.
- `mcap_issue_usd_m`: needs post-issue share count. Not in any source used.
- `pe_issue`, `roe`, `roce`, `debt_equity`, `anchor_pct`, `n_underwriters`: not mapped. rohan `finalipodata.csv`
  has ROE, ROCE, D/E, PAT margin, pre-issue P/E, anchor % columns. They are not mapped in this step because their
  definitions (pre- vs post-issue P/E, net-worth basis) were not checked against a second source.
- `revenue_growth_pct`: not mapped (year ordering of revenue blocks did not check out).
- `sub_snii` / `sub_bnii`: rohan only (from 2022 onwards in the coverage table).
- `sub_shareholder`: one IPO only.
- `gmp_open`, `gmp_pct_open`: no open-day reading.
- `ofs_pct`: HF only.
- `price_low` / `price_high`: HF only (SME band text in rimarusama not parsed).
- Context columns (`idx_*`, `vix_level`): filled by `data.context`.

## Coverage (non-null %) by market

| column | IN_MAIN | IN_SME |
|---|---:|---:|
| ipo_id | 100 | 100 |
| market | 100 | 100 |
| country | 100 | 100 |
| exchange | 72 | 26 |
| company | 100 | 100 |
| symbol | 70 | 25 |
| sector | 40 | 36 |
| open_date | 72 | 47 |
| close_date | 72 | 47 |
| listing_date | 100 | 100 |
| currency | 100 | 100 |
| price_low | 40 | 8 |
| price_high | 40 | 8 |
| issue_price | 100 | 100 |
| lot_size | 72 | 94 |
| issue_size_m | 91 | 100 |
| issue_size_usd_m | 0 | 0 |
| fresh_issue_pct | 58 | 19 |
| ofs_pct | 13 | 4 |
| mcap_issue_usd_m | 0 | 0 |
| pe_issue | 0 | 0 |
| roe | 0 | 0 |
| roce | 0 | 0 |
| debt_equity | 0 | 0 |
| revenue_growth_pct | 0 | 0 |
| pat_margin_pct | 0 | 0 |
| is_loss_making | 0 | 0 |
| anchor_pct | 0 | 0 |
| n_underwriters | 0 | 0 |
| sub_qib | 84 | 6 |
| sub_nii | 82 | 4 |
| sub_snii | 35 | 0 |
| sub_bnii | 35 | 0 |
| sub_retail | 79 | 32 |
| sub_employee | 26 | 0 |
| sub_shareholder | 1 | 0 |
| sub_total | 82 | 32 |
| sub_qib_d1 | 40 | 5 |
| sub_nii_d1 | 40 | 4 |
| sub_retail_d1 | 40 | 32 |
| sub_total_d1 | 40 | 32 |
| sub_qib_d2 | 40 | 5 |
| sub_nii_d2 | 40 | 4 |
| sub_retail_d2 | 40 | 32 |
| sub_total_d2 | 40 | 32 |
| sub_qib_d3 | 8 | 6 |
| sub_nii_d3 | 8 | 4 |
| sub_retail_d3 | 8 | 32 |
| sub_total_d3 | 8 | 32 |
| gmp_open | 0 | 0 |
| gmp_last | 16 | 14 |
| gmp_pct_open | 0 | 0 |
| gmp_pct_last | 16 | 14 |
| list_open_gain_pct | 64 | 99 |
| list_high_gain_pct | 40 | 38 |
| list_low_gain_pct | 40 | 38 |
| list_close_gain_pct | 67 | 90 |
| idx_ret_5d | 0 | 0 |
| idx_ret_20d | 0 | 0 |
| idx_vol_20d | 0 | 0 |
| vix_level | 0 | 0 |
| idx_ret_issue_window | 0 | 0 |
| source | 100 | 100 |
| source_url | 100 | 100 |
| n_missing_core | 100 | 100 |

## Category subscription coverage by listing year

Non-null % of each category at close; `n` is the number of IPOs in that listing year.

### IN_MAIN

| year | n | sub_qib | sub_nii | sub_snii | sub_bnii | sub_retail | sub_employee | sub_shareholder | sub_total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2009 | 21 | 100 | 100 | 0 | 0 | 100 | 52 | 0 | 100 |
| 2010 | 105 | 97 | 96 | 0 | 0 | 96 | 25 | 0 | 97 |
| 2011 | 58 | 95 | 95 | 0 | 0 | 93 | 2 | 0 | 93 |
| 2012 | 16 | 100 | 94 | 0 | 0 | 94 | 25 | 0 | 94 |
| 2013 | 6 | 100 | 100 | 0 | 0 | 100 | 17 | 0 | 100 |
| 2014 | 9 | 100 | 100 | 0 | 0 | 100 | 0 | 0 | 100 |
| 2015 | 27 | 96 | 96 | 0 | 0 | 96 | 22 | 0 | 93 |
| 2016 | 36 | 94 | 94 | 0 | 0 | 86 | 28 | 0 | 94 |
| 2017 | 47 | 96 | 94 | 0 | 0 | 83 | 32 | 0 | 96 |
| 2018 | 38 | 100 | 100 | 0 | 0 | 66 | 26 | 0 | 89 |
| 2019 | 24 | 88 | 88 | 0 | 0 | 71 | 29 | 0 | 88 |
| 2020 | 24 | 92 | 92 | 0 | 0 | 92 | 25 | 0 | 96 |
| 2021 | 91 | 77 | 77 | 0 | 0 | 77 | 31 | 0 | 78 |
| 2022 | 46 | 85 | 70 | 48 | 48 | 70 | 24 | 0 | 70 |
| 2023 | 85 | 73 | 71 | 62 | 62 | 71 | 16 | 0 | 68 |
| 2024 | 134 | 72 | 66 | 61 | 61 | 62 | 25 | 0 | 69 |
| 2025 | 154 | 66 | 66 | 68 | 68 | 64 | 32 | 3 | 67 |
| 2026 | 108 | 93 | 92 | 89 | 89 | 93 | 32 | 3 | 92 |
### IN_SME

| year | n | sub_qib | sub_nii | sub_snii | sub_bnii | sub_retail | sub_employee | sub_shareholder | sub_total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2012 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2013 | 34 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2014 | 41 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2015 | 42 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2016 | 66 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2017 | 183 | 1 | 3 | 0 | 0 | 4 | 0 | 0 | 5 |
| 2018 | 209 | 3 | 10 | 0 | 0 | 66 | 1 | 0 | 66 |
| 2019 | 73 | 0 | 7 | 0 | 0 | 67 | 1 | 0 | 67 |
| 2020 | 45 | 0 | 0 | 0 | 0 | 69 | 0 | 0 | 69 |
| 2021 | 77 | 5 | 4 | 0 | 0 | 69 | 1 | 0 | 69 |
| 2022 | 154 | 7 | 6 | 0 | 0 | 69 | 0 | 0 | 69 |
| 2023 | 249 | 29 | 10 | 0 | 0 | 72 | 0 | 0 | 72 |
| 2024 | 250 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 2 |
| 2025 | 270 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2026 | 52 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
