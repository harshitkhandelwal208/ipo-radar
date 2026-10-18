# Research: what goes with listing-day gains and losses

Source: `data/processed/ipos.parquet`, 4933 IPOs (IN_MAIN 1029, IN_SME 1758, US 1721, HK 323, OTHER 102). The numbers were computed on the post-build table (`data/processed/COVERAGE.md`, built 2026-10-18). Table digest `14663db9`, seed 7. Written by `ipo research` (`src/ipo_radar/research.py`); a rerun on the same table gives the same file (checked: two runs compared byte for byte).

## Summary

- Listing day, median open gain: IN_MAIN +9.1 (n 608), IN_SME +6.2 (n 1585), US +9.5 (n 1093), HK +1.0 (n 283). Above issue at the open: IN_MAIN 76%, IN_SME 81%, US 76%, HK 55%. Descriptive.
- Total subscription against the close gain: IN_MAIN rho +0.66 [+0.60, +0.71] (n 513, strong); IN_SME rho +0.62 [+0.55, +0.68] (n 424, strong). Both segments point the same way: higher goes with higher gains.
- Categories once total is held fixed (IN_MAIN, close): only total has a CI clear of zero; the categories alone reach R2 0.28 against 0.29 for total alone, and adding them to total gives 0.29 (n 463).
- Retail-QIB flag (retail >= 3x, QIB < 1x): on 19 IN_MAIN IPOs (0 IN_SME), too few to judge. The other IN_MAIN IPOs with both multiples known have a close median of -1.1 (n 169); the 'both at least 3x' group has +23.3 (n 246). Descriptive, not tested.
- Employee quota (IN_MAIN, close): rho +0.37 [+0.25, +0.47] (n 244), strong. Holding total fixed, the employee coefficient is -0.40 [-3.35, +2.56] (n 178), so the raw correlation mostly tracks total subscription. Shareholder quota: n<20 for 8 IPOs, not testable.
- GMP (India pooled, open, n 310): MAE 15.6 pp against 35.0 pp for predicting the segment median; mean error (GMP minus open) -3.2 pp; rho +0.84 [+0.79, +0.88], strong.
- GMP against subscription (India pooled, open, n 274): R2 0.68 from GMP, 0.40 from total, 0.68 from both. Adding GMP to total adds +0.28 [+0.21, +0.37]; adding total to GMP adds +0.00 [+0.00, +0.01].
- Market conditions (close): NIFTY 20-day return rho +0.09 [+0.02, +0.16] in IN_MAIN (weak) and +0.08 [-0.00, +0.16] in IN_SME (not distinguishable from noise); India VIX rho -0.05 [-0.12, +0.03] in IN_MAIN (not distinguishable from noise) and -0.06 [-0.14, +0.02] in IN_SME (not distinguishable from noise). US, HK and OTHER have no as-of context in the table.
- Issue size (close): rho +0.04 [-0.04, +0.13] in IN_MAIN (not distinguishable from noise), +0.22 [+0.18, +0.27] in IN_SME (weak), +0.24 [+0.18, +0.29] in US (weak).
- Cross-market: HK public-offer multiple rho +0.20 [+0.10, +0.31] (n 293, weak), the same sign as India retail. The US has no subscription field. Biggest caveats: survivorship (joined rows 159 US, 323 HK), placeholders blanked by the build (open 389, close 128, GMP 50) and 0 subscription zeros set to missing here, SME category data mostly pre-2024, GMP untimed and ending 2024, and 94 intervals with no multiplicity correction.

## Question and method

Question: which of GMP, subscription (QIB, HNI with sNII and bNII, retail, employee, shareholder, total), issue size, offer-for-sale share, market conditions, segment and market relate to the listing-day gain and loss, and how reliably?

- Gains are percent of the issue price on the first trading day (open and close).
- NaN is missing and is never filled. Exact zeros are missing codes: the build blanks gain and GMP placeholders, and this module sets subscription zeros to missing (section 1).
- Intervals: 95% percentile bootstraps over IPOs, 2000 resamples, seed 7. Rank correlations are Spearman. Regressions are OLS with HC3 standard errors and normal 95% intervals.
- Strength words: 'not distinguishable from noise' when the interval spans zero; 'strong' when |rho| >= 0.3 and the interval excludes zero; otherwise 'weak'. The 0.3 threshold is a convention, not a test.
- Multiplicity: this report computes 94 intervals on associations, coefficients and R-squared differences, and 58 of them exclude zero. If nothing were there, roughly 5 would exclude zero by chance (a rough count: the tests are not independent). No correction is applied. The bucket and year tables are descriptive: they are read, not tested, and no cell was chosen after looking at its outcome.


## 1. Data used

Rows per market. Counts are non-null values; NaN is missing. `+px_join` marks IPOs whose prices came from a joined price file, a selected subset (docs/data_sources/global.md).

| market | IPOs | listing years | open gain | close gain | GMP | QIB | retail | total sub | VIX context | +px_join |
|---|---|---|---|---|---|---|---|---|---|---|
| IN_MAIN | 1029 | 2009-2026 | 608 | 682 | 156 | 848 | 795 | 844 | 717 | 0 |
| IN_SME | 1758 | 2012-2026 | 1585 | 1560 | 211 | 97 | 570 | 571 | 825 | 0 |
| US | 1721 | 2010-2026 | 1093 | 1257 | 0 | 0 | 0 | 0 | 0 | 159 |
| HK | 323 | 2020-2024 | 283 | 293 | 0 | 0 | 323 | 0 | 0 | 323 |
| OTHER | 102 | 2022-2023 | 0 | 101 | 0 | 0 | 0 | 0 | 0 | 0 |

Rows per listing year and market:

| listing year | IN_MAIN | IN_SME | US | HK | OTHER |
|---|---|---|---|---|---|
| 2009 | 21 | 0 | 0 | 0 | 0 |
| 2010 | 105 | 0 | 68 | 0 | 0 |
| 2011 | 58 | 0 | 60 | 0 | 0 |
| 2012 | 16 | 13 | 75 | 0 | 0 |
| 2013 | 6 | 34 | 137 | 0 | 0 |
| 2014 | 9 | 41 | 183 | 0 | 0 |
| 2015 | 27 | 42 | 114 | 0 | 0 |
| 2016 | 36 | 66 | 87 | 0 | 0 |
| 2017 | 47 | 183 | 157 | 0 | 0 |
| 2018 | 38 | 209 | 191 | 0 | 0 |
| 2019 | 24 | 73 | 135 | 0 | 0 |
| 2020 | 24 | 45 | 19 | 105 | 0 |
| 2021 | 91 | 77 | 5 | 76 | 0 |
| 2022 | 46 | 154 | 0 | 62 | 50 |
| 2023 | 85 | 249 | 0 | 63 | 52 |
| 2024 | 134 | 250 | 168 | 17 | 0 |
| 2025 | 154 | 270 | 207 | 0 | 0 |
| 2026 | 108 | 52 | 115 | 0 | 0 |

Non-null share of each field by market (denominator: that market's rows):

| field | IN_MAIN | IN_SME | US | HK | OTHER |
|---|---|---|---|---|---|
| list_open_gain_pct | 59% | 90% | 64% | 88% | 0% |
| list_close_gain_pct | 66% | 89% | 73% | 91% | 99% |
| sub_qib | 82% | 6% | 0% | 0% | 0% |
| sub_nii | 82% | 4% | 0% | 0% | 0% |
| sub_snii | 35% | 0% | 0% | 0% | 0% |
| sub_bnii | 35% | 0% | 0% | 0% | 0% |
| sub_retail | 77% | 32% | 0% | 100% | 0% |
| sub_employee | 26% | 0% | 0% | 0% | 0% |
| sub_shareholder | 1% | 0% | 0% | 0% | 0% |
| sub_total | 82% | 32% | 0% | 0% | 0% |
| gmp_pct_last | 15% | 12% | 0% | 0% | 0% |
| idx_ret_20d | 72% | 47% | 0% | 0% | 0% |
| vix_level | 70% | 47% | 0% | 0% | 0% |
| issue_size_m | 91% | 100% | 91% | 99% | 0% |
| fresh_issue_pct | 58% | 19% | 0% | 0% | 0% |
| ofs_pct | 13% | 4% | 0% | 0% | 0% |
| price_high | 40% | 8% | 0% | 0% | 0% |

Category coverage by listing year, IN_MAIN (share of that year's rows):

| year | n | QIB | NII | sNII | retail | total | GMP |
|---|---|---|---|---|---|---|---|
| 2009 | 21 | 100% | 100% | 0% | 100% | 100% | 0% |
| 2010 | 105 | 96% | 96% | 0% | 96% | 97% | 0% |
| 2011 | 58 | 76% | 95% | 0% | 93% | 93% | 0% |
| 2012 | 16 | 94% | 94% | 0% | 94% | 94% | 0% |
| 2013 | 6 | 100% | 100% | 0% | 100% | 100% | 0% |
| 2014 | 9 | 100% | 100% | 0% | 100% | 100% | 0% |
| 2015 | 27 | 96% | 96% | 0% | 96% | 93% | 0% |
| 2016 | 36 | 94% | 94% | 0% | 86% | 94% | 0% |
| 2017 | 47 | 96% | 94% | 0% | 77% | 96% | 0% |
| 2018 | 38 | 100% | 100% | 0% | 66% | 89% | 0% |
| 2019 | 24 | 88% | 88% | 0% | 71% | 88% | 0% |
| 2020 | 24 | 92% | 92% | 0% | 92% | 96% | 42% |
| 2021 | 91 | 77% | 77% | 0% | 73% | 78% | 67% |
| 2022 | 46 | 85% | 70% | 48% | 70% | 70% | 70% |
| 2023 | 85 | 71% | 69% | 62% | 69% | 68% | 62% |
| 2024 | 134 | 72% | 66% | 61% | 61% | 69% | 0% |
| 2025 | 154 | 66% | 66% | 68% | 62% | 67% | 0% |
| 2026 | 108 | 93% | 92% | 89% | 90% | 92% | 0% |

Category coverage by listing year, IN_SME (share of that year's rows):

| year | n | QIB | NII | sNII | retail | total | GMP |
|---|---|---|---|---|---|---|---|
| 2012 | 13 | 0% | 0% | 0% | 0% | 0% | 0% |
| 2013 | 34 | 0% | 0% | 0% | 0% | 0% | 0% |
| 2014 | 41 | 0% | 0% | 0% | 0% | 0% | 0% |
| 2015 | 42 | 0% | 0% | 0% | 0% | 0% | 0% |
| 2016 | 66 | 0% | 0% | 0% | 0% | 0% | 0% |
| 2017 | 183 | 1% | 3% | 0% | 4% | 5% | 0% |
| 2018 | 209 | 3% | 10% | 0% | 66% | 66% | 0% |
| 2019 | 73 | 0% | 7% | 0% | 67% | 67% | 0% |
| 2020 | 45 | 0% | 0% | 0% | 69% | 69% | 0% |
| 2021 | 77 | 5% | 4% | 0% | 69% | 69% | 3% |
| 2022 | 154 | 7% | 6% | 0% | 69% | 69% | 31% |
| 2023 | 249 | 29% | 10% | 0% | 72% | 72% | 64% |
| 2024 | 250 | 1% | 0% | 0% | 2% | 2% | 1% |
| 2025 | 270 | 0% | 0% | 0% | 0% | 0% | 0% |
| 2026 | 52 | 0% | 0% | 0% | 0% | 0% | 0% |

**Field problems the reader needs to know about**

- Schema columns missing from this table: none. A missing column is read as all-NaN, so the sections that need it print 'n<20' or 'no rows'.
- Exact-zero placeholders. The build blanks a 0 in the gain and GMP columns (rule `<col> exactly 0 (placeholder)` in data/build.py). Blanked by the build (data/processed/COVERAGE.md, built 2026-10-18): open 389, high 45, low 75, close 128, gmp_last 50, gmp_pct_last 50. The spike evidence was taken on the pre-build table (US open gains: 138 at 0.00%, against 34 in (0, 1] and 10 in [-1, 0)). Subscription zeros are not in that rule, so this module sets them to missing. Counted from this run's input, by market:

None were found.

- Survivorship: 159 of 1721 US rows and 323 of 323 HK rows are joined price-file IPOs, a selected subset. Every estimate describes IPOs that listed, so it is optimistic for the average issue.
- SME category subscription: QIB is present for 29% of 2023 SME listings and 0% of 2024-2026 listings. SME category results rest on the pre-2024 listings, mostly 2023.
- GMP: 367 India rows with a reading, listed 2020-2024. One untimed grey-market reading per IPO (no open-day timestamp), taken before listing. None for US, HK or OTHER.
- Market context (`idx_*`, `vix_level`) exists only for IN_MAIN and IN_SME. US, HK and OTHER have no open date, so as-of context cannot be computed. VIX context ends with the 2026 listings.
- HK `sub_retail` is the public-offer multiple. HK has no QIB or NII field, and US has no subscription field at all.
- US open gains are missing for the 2024-2026 listings (close targets only), so the US open-gain rows stop in 2023.
- Cross-check with docs/MODEL_REPORT.md, generated before the build blanked the placeholders (data hash 07cf78bd). Its open-gain counts (IN_MAIN 658, IN_SME 1746, US 1231, HK 323) equal the counts here plus 50, 161, 138 and 40 placeholders. Its IN_MAIN QIB count (863) equals the count here plus 15 subscription zeros the build does not blank, and its GMP counts (165, 252) equal the counts here plus 9 and 41 placeholders. Its row counts therefore include the placeholders. The model report needs re-running on the post-build table (not done here).
- Intervals treat IPOs as independent. IPOs listing on the same day share market conditions, so the intervals are somewhat too narrow.


## 2. Listing outcomes

Gain against the issue price, in percent. 'Above issue' means a gain above 0. 'Loss > 10%' means a fall of more than 10%. Medians and percentiles; nothing is winsorized.

**Listing-day open gain by segment**

| segment | n | above issue | median | p10 | p90 | loss > 10% |
|---|---|---|---|---|---|---|
| IN_MAIN | 608 | 76% ████████░░ | +9.1 | -7.0 | +55.3 | 5% |
| IN_SME | 1585 | 81% ████████░░ | +6.2 | -6.1 | +90.0 | 8% |
| US | 1093 | 76% ████████░░ | +9.5 | -10.5 | +59.9 | 11% |
| HK | 283 | 55% ██████░░░░ | +1.0 | -19.0 | +51.3 | 21% |
| OTHER | 0 | n<20 | n<20 | n<20 | n<20 | n<20 |

**Listing-day close gain by segment**

| segment | n | above issue | median | p10 | p90 | loss > 10% |
|---|---|---|---|---|---|---|
| IN_MAIN | 682 | 69% ███████░░░ | +9.4 | -11.1 | +58.7 | 11% |
| IN_SME | 1560 | 65% ██████░░░░ | +4.4 | -54.6 | +80.5 | 23% |
| US | 1257 | 70% ███████░░░ | +8.4 | -17.4 | +59.2 | 15% |
| HK | 293 | 57% ██████░░░░ | +1.6 | -26.9 | +55.8 | 27% |
| OTHER | 101 | 79% ████████░░ | +17.3 | -9.0 | +35.0 | 2% |

**IN_MAIN by listing year**

| year | n open | open above issue | open median | n close | close above issue | close median | close loss > 10% |
|---|---|---|---|---|---|---|---|
| 2009 | 12 | n<20 | n<20 | 16 | n<20 | n<20 | n<20 |
| 2010 | 51 | +7.2 | 84% ████████░░ | 60 | +7.6 | 63% ██████░░░░ | 23% |
| 2011 | 30 | +3.6 | 80% ████████░░ | 33 | +0.5 | 52% █████░░░░░ | 48% |
| 2012 | 12 | n<20 | n<20 | 11 | n<20 | n<20 | n<20 |
| 2013 | 3 | n<20 | n<20 | 2 | n<20 | n<20 | n<20 |
| 2014 | 5 | n<20 | n<20 | 5 | n<20 | n<20 | n<20 |
| 2015 | 16 | n<20 | n<20 | 17 | n<20 | n<20 | n<20 |
| 2016 | 25 | +9.6 | 72% ███████░░░ | 22 | +14.9 | 73% ███████░░░ | 14% |
| 2017 | 23 | +17.4 | 74% ███████░░░ | 30 | +2.1 | 67% ███████░░░ | 3% |
| 2018 | 19 | n<20 | n<20 | 23 | +0.0 | 52% █████░░░░░ | 17% |
| 2019 | 16 | n<20 | n<20 | 17 | n<20 | n<20 | n<20 |
| 2020 | 12 | n<20 | n<20 | 13 | n<20 | n<20 | n<20 |
| 2021 | 43 | +16.0 | 74% ███████░░░ | 59 | +17.6 | 71% ███████░░░ | 8% |
| 2022 | 24 | +3.7 | 67% ███████░░░ | 30 | +3.9 | 60% ██████░░░░ | 10% |
| 2023 | 53 | +20.2 | 87% █████████░ | 56 | +21.1 | 86% █████████░ | 0% |
| 2024 | 86 | +18.1 | 81% ████████░░ | 88 | +21.1 | 80% ████████░░ | 5% |
| 2025 | 92 | +8.3 | 77% ████████░░ | 104 | +5.1 | 67% ███████░░░ | 6% |
| 2026 | 86 | +10.0 | 73% ███████░░░ | 96 | +7.5 | 69% ███████░░░ | 12% |

**IN_SME by listing year**

| year | n open | open above issue | open median | n close | close above issue | close median | close loss > 10% |
|---|---|---|---|---|---|---|---|
| 2012 | 13 | n<20 | n<20 | 11 | n<20 | n<20 | n<20 |
| 2013 | 31 | +4.8 | 77% ████████░░ | 33 | -21.0 | 33% ███░░░░░░░ | 58% |
| 2014 | 39 | +4.0 | 74% ███████░░░ | 40 | -4.5 | 42% ████░░░░░░ | 40% |
| 2015 | 41 | +3.3 | 85% █████████░ | 42 | -32.9 | 26% ███░░░░░░░ | 64% |
| 2016 | 60 | +5.3 | 87% █████████░ | 63 | -37.8 | 25% ███░░░░░░░ | 63% |
| 2017 | 162 | +6.7 | 77% ████████░░ | 147 | +0.4 | 56% ██████░░░░ | 31% |
| 2018 | 185 | +2.0 | 78% ████████░░ | 147 | +1.0 | 60% ██████░░░░ | 25% |
| 2019 | 60 | +1.7 | 77% ████████░░ | 53 | +0.1 | 51% █████░░░░░ | 30% |
| 2020 | 31 | +1.8 | 84% ████████░░ | 29 | +0.2 | 55% ██████░░░░ | 34% |
| 2021 | 73 | +3.9 | 81% ████████░░ | 59 | +1.0 | 58% ██████░░░░ | 29% |
| 2022 | 139 | +14.8 | 88% █████████░ | 132 | +14.7 | 78% ████████░░ | 16% |
| 2023 | 219 | +19.6 | 91% █████████░ | 235 | +12.8 | 80% ████████░░ | 9% |
| 2024 | 240 | +39.1 | 91% █████████░ | 248 | +35.7 | 88% █████████░ | 6% |
| 2025 | 248 | +4.2 | 65% ███████░░░ | 269 | +3.9 | 63% ██████░░░░ | 23% |
| 2026 | 44 | +1.9 | 59% ██████░░░░ | 52 | +0.6 | 56% ██████░░░░ | 21% |

**US by listing year**

| year | n open | open above issue | open median | n close | close above issue | close median | close loss > 10% |
|---|---|---|---|---|---|---|---|
| 2010 | 55 | +9.5 | 78% ████████░░ | 36 | +15.0 | 81% ████████░░ | 6% |
| 2011 | 48 | +6.5 | 79% ████████░░ | 31 | +3.9 | 74% ███████░░░ | 6% |
| 2012 | 68 | +10.0 | 82% ████████░░ | 41 | +15.8 | 88% █████████░ | 5% |
| 2013 | 126 | +11.5 | 79% ████████░░ | 78 | +17.6 | 81% ████████░░ | 8% |
| 2014 | 155 | +10.6 | 77% ████████░░ | 116 | +9.6 | 72% ███████░░░ | 9% |
| 2015 | 84 | +9.9 | 77% ████████░░ | 57 | +12.8 | 79% ████████░░ | 12% |
| 2016 | 78 | +10.0 | 81% ████████░░ | 54 | +13.9 | 78% ████████░░ | 7% |
| 2017 | 151 | +8.3 | 70% ███████░░░ | 80 | +14.4 | 82% ████████░░ | 12% |
| 2018 | 179 | +8.7 | 69% ███████░░░ | 156 | +7.4 | 69% ███████░░░ | 17% |
| 2019 | 127 | +8.6 | 80% ████████░░ | 129 | +13.3 | 69% ███████░░░ | 14% |
| 2020 | 17 | n<20 | n<20 | 18 | n<20 | n<20 | n<20 |
| 2021 | 5 | n<20 | n<20 | 5 | n<20 | n<20 | n<20 |
| 2024 | 0 | n<20 | n<20 | 158 | +1.6 | 54% █████░░░░░ | 28% |
| 2025 | 0 | n<20 | n<20 | 194 | +6.2 | 65% ███████░░░ | 19% |
| 2026 | 0 | n<20 | n<20 | 104 | +3.4 | 62% ██████░░░░ | 20% |

**HK by listing year**

| year | n open | open above issue | open median | n close | close above issue | close median | close loss > 10% |
|---|---|---|---|---|---|---|---|
| 2020 | 96 | +6.8 | 58% ██████░░░░ | 100 | +3.3 | 62% ██████░░░░ | 28% |
| 2021 | 66 | +0.6 | 53% █████░░░░░ | 70 | +0.1 | 50% █████░░░░░ | 26% |
| 2022 | 50 | +0.6 | 56% ██████░░░░ | 50 | -0.0 | 50% █████░░░░░ | 20% |
| 2023 | 55 | +0.0 | 51% █████░░░░░ | 57 | +1.1 | 61% ██████░░░░ | 32% |
| 2024 | 16 | n<20 | n<20 | 16 | n<20 | n<20 | n<20 |

**OTHER by listing year**

| year | n open | open above issue | open median | n close | close above issue | close median | close loss > 10% |
|---|---|---|---|---|---|---|---|
| 2022 | 0 | n<20 | n<20 | 49 | +24.6 | 84% ████████░░ | 0% |
| 2023 | 0 | n<20 | n<20 | 52 | +13.2 | 75% ████████░░ | 4% |


## 3. Subscription

Spearman rank correlation of each subscription multiple with the listing-day gain, with a 95% bootstrap interval over IPOs. Ranks are used because multiples are very skewed. The reading is on the close: 'not distinguishable from noise' when the interval spans zero; 'strong' when |rho| >= 0.3 and the interval excludes zero; otherwise 'weak'.

| segment | multiple | n open | rho open [95% CI] | n close | rho close [95% CI] | reading (close) |
|---|---|---|---|---|---|---|
| IN_MAIN | QIB | 471 | +0.69 [+0.64, +0.73] | 526 | +0.60 [+0.53, +0.67] | strong |
| IN_MAIN | NII (HNI) | 461 | +0.73 [+0.68, +0.77] | 515 | +0.63 [+0.57, +0.68] | strong |
| IN_MAIN | sNII | 322 | +0.63 [+0.56, +0.69] | 350 | +0.57 [+0.50, +0.64] | strong |
| IN_MAIN | bNII | 322 | +0.64 [+0.57, +0.70] | 350 | +0.58 [+0.51, +0.64] | strong |
| IN_MAIN | retail | 435 | +0.64 [+0.57, +0.69] | 485 | +0.53 [+0.47, +0.59] | strong |
| IN_MAIN | employee | 217 | +0.41 [+0.30, +0.52] | 244 | +0.37 [+0.25, +0.47] | strong |
| IN_MAIN | shareholder | 6 | n<20 | 8 | n<20 | not tested (n<20) |
| IN_MAIN | total | 460 | +0.75 [+0.71, +0.79] | 513 | +0.66 [+0.60, +0.71] | strong |
| IN_SME | QIB | 89 | +0.59 [+0.43, +0.71] | 88 | +0.54 [+0.35, +0.68] | strong |
| IN_SME | NII (HNI) | 64 | +0.75 [+0.65, +0.83] | 62 | +0.68 [+0.54, +0.78] | strong |
| IN_SME | sNII | 0 | n<20 | 0 | n<20 | no data |
| IN_SME | bNII | 0 | n<20 | 0 | n<20 | no data |
| IN_SME | retail | 501 | +0.63 [+0.56, +0.68] | 423 | +0.59 [+0.51, +0.65] | strong |
| IN_SME | employee | 5 | n<20 | 6 | n<20 | not tested (n<20) |
| IN_SME | shareholder | 0 | n<20 | 0 | n<20 | no data |
| IN_SME | total | 502 | +0.65 [+0.58, +0.70] | 424 | +0.62 [+0.55, +0.68] | strong |
| HK | public-offer (retail) | 283 | +0.22 [+0.11, +0.33] | 293 | +0.20 [+0.10, +0.31] | weak |


### 3.1 Which category carries the information once total is held fixed

Regressions of the gain on log multiples (natural log: ln 10 = 2.30 is a tenfold rise). Model A: total only. Model B: QIB, NII and retail, no total. Model C: total plus QIB, NII and retail. All three use one complete-case sample, so their R-squared values compare. Standard errors are HC3. The fit is OLS on raw gains, so a few very large gains pull the coefficients; the rank correlations above are the robust check.

| segment | outcome | n (complete cases) | R2 model A (total) | R2 model B (categories) | R2 model C (both) |
|---|---|---|---|---|---|
| IN_MAIN | close | 463 | 0.290 | 0.276 | 0.292 |
| IN_MAIN | open | 415 | 0.411 | 0.397 | 0.412 |
| IN_SME | close | 39 | 0.369 | 0.394 | 0.411 |
| IN_SME | open | 36 | 0.381 | 0.384 | 0.417 |

Model C coefficients (pp of gain per unit of ln multiple):

| segment | outcome | variable | coef | 95% CI (HC3) | reading |
|---|---|---|---|---|---|
| IN_MAIN | close | ln_total | +16.89 | [+3.87, +29.91] | CI excludes 0 |
| IN_MAIN | close | ln_qib | -2.54 | [-9.19, +4.12] | CI spans 0 |
| IN_MAIN | close | ln_nii | -1.66 | [-5.97, +2.64] | CI spans 0 |
| IN_MAIN | close | ln_retail | -1.25 | [-5.55, +3.05] | CI spans 0 |
| IN_MAIN | open | ln_total | +13.43 | [+8.31, +18.55] | CI excludes 0 |
| IN_MAIN | open | ln_qib | -1.16 | [-3.60, +1.29] | CI spans 0 |
| IN_MAIN | open | ln_nii | -0.47 | [-2.36, +1.42] | CI spans 0 |
| IN_MAIN | open | ln_retail | -0.40 | [-3.74, +2.94] | CI spans 0 |
| IN_SME | close | ln_total | +55.60 | [-66.53, +177.74] | CI spans 0 |
| IN_SME | close | ln_qib | -22.28 | [-63.96, +19.39] | CI spans 0 |
| IN_SME | close | ln_nii | -9.55 | [-59.59, +40.50] | CI spans 0 |
| IN_SME | close | ln_retail | -8.95 | [-59.25, +41.35] | CI spans 0 |
| IN_SME | open | ln_total | +67.46 | [-51.18, +186.10] | CI spans 0 |
| IN_SME | open | ln_qib | -20.89 | [-60.14, +18.36] | CI spans 0 |
| IN_SME | open | ln_nii | -14.12 | [-62.69, +34.45] | CI spans 0 |
| IN_SME | open | ln_retail | -16.56 | [-64.39, +31.27] | CI spans 0 |

A coefficient is what that category adds once total is held fixed. Categories are parts of total, so the coefficients describe the mix of demand at a given total, not a separate cause. The data are observational.


### 3.2 Employee and shareholder quotas

| category | n open | rho open [95% CI] | n close | rho close [95% CI] | reading (close) |
|---|---|---|---|---|---|
| IN_MAIN employee | 217 | +0.41 [+0.30, +0.52] | 244 | +0.37 [+0.25, +0.47] | strong |
| IN_MAIN employee, listed 2021 or later |  |  | 167 | +0.43 [+0.29, +0.55] | strong |
| IN_MAIN shareholder | 8 | n<20 | n<20 | n<20 | not tested |
| IN_SME employee | 6 | n<20 | n<20 | n<20 | not tested |

| IN_MAIN close gain | n | close median | close above issue |
|---|---|---|---|
| employee recorded | 244 | +11.3 | 71% ███████░░░ |
| employee not recorded | 438 | +7.7 | 68% ███████░░░ |

Holding total fixed (IN_MAIN, close, n 178): employee coefficient -0.40 per unit of ln multiple, 95% CI [-3.35, +2.56] (CI spans 0).

The groups differ by era: employee multiples are recorded for 23% of IN_MAIN listings before 2021 and 28% from 2021, and the employee multiple moves with total subscription, so the raw correlation mixes quota with demand. Shareholder quotas are recorded for 8 IPOs, too few to test.


### 3.3 Retail against QIB: divergence flag

Each IPO falls in one group. The cut-offs (3x and 1x) were fixed before the outcomes were looked at. Rows without both a QIB and a retail multiple are left out. These are descriptive medians, not tests.

**IN_MAIN** (rows with both QIB and retail: 775)

| group | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| A: retail >= 3x and QIB < 1x (flag) | 17 | n<20 | n<20 | 19 | n<20 | n<20 |
| B: QIB >= 3x and retail < 1x | 35 | +1.6 | 57% ██████░░░░ | 37 | +4.2 | 57% ██████░░░░ |
| C: both >= 3x | 229 | +22.0 | 91% █████████░ | 246 | +23.3 | 87% █████████░ |
| D: everything else with both known | 141 | +0.9 | 57% ██████░░░░ | 169 | -1.1 | 46% █████░░░░░ |

**IN_SME** (rows with both QIB and retail: 97)

| group | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| A: retail >= 3x and QIB < 1x (flag) | 1 | n<20 | n<20 | 0 | n<20 | n<20 |
| B: QIB >= 3x and retail < 1x | 1 | n<20 | n<20 | 1 | n<20 | n<20 |
| C: both >= 3x | 76 | +62.5 | 99% ██████████ | 75 | +61.0 | 96% ██████████ |
| D: everything else with both known | 11 | n<20 | n<20 | 12 | n<20 | n<20 |


## 4. GMP

GMP is an unofficial grey-market percentage of the issue price, one untimed reading per IPO. The error is GMP minus the open gain, so a positive bias means GMP was too high on average. 'MAE constant' is the error of predicting the segment's median open gain (in-sample, all the segment's rows): the bar GMP has to clear.

| segment | n | MAE GMP (pp) | MAE constant (pp) | bias: mean error (pp) | rho GMP vs open [95% CI] | reading |
|---|---|---|---|---|---|---|
| IN_MAIN | 118 | 10.4 | 27.5 | -0.8 | +0.89 [+0.81, +0.93] | strong |
| IN_SME | 192 | 18.8 | 39.6 | -4.7 | +0.82 [+0.76, +0.87] | strong |
| India pooled | 310 | 15.6 | 35.0 | -3.2 | +0.84 [+0.79, +0.88] | strong |

**Bucket table, India pooled.** Descriptive. Cells with fewer than 20 rows print 'n<20'.

| GMP % | n open | open median | open above issue | n | median error (pp) |
|---|---|---|---|---|---|
| below 0 | 34 | -0.4 | 50% █████░░░░░ | 34 | -2.3 |
| 0 to 10 | 71 | +5.1 | 79% ████████░░ | 71 | -0.6 |
| 10 to 30 | 72 | +13.9 | 89% █████████░ | 72 | +1.9 |
| 30 to 60 | 64 | +43.7 | 100% ██████████ | 64 | -1.4 |
| 60 to 100 | 51 | +80.7 | 98% ██████████ | 51 | -4.0 |
| 100 and above | 18 | n<20 | n<20 | 18 | n<20 |

**By listing year.** GMP readings exist only for 2020-2024 listings.

| year | IN_MAIN n | MAE (pp) | rho [95% CI] | IN_SME n | MAE (pp) | rho [95% CI] |
|---|---|---|---|---|---|---|
| 2019 | 0 | n<20 | n<20 | 0 | n<20 | n<20 |
| 2020 | 6 | n<20 | n<20 | 0 | n<20 | n<20 |
| 2021 | 39 | 13.9 | +0.91 [+0.79, +0.96] | 2 | n<20 | n<20 |
| 2022 | 23 | 5.3 | +0.76 [+0.38, +0.94] | 41 | 17.0 | +0.88 [+0.78, +0.93] |
| 2023 | 50 | 10.1 | +0.87 [+0.71, +0.94] | 147 | 19.1 | +0.82 [+0.74, +0.87] |
| 2024 | 0 | n<20 | n<20 | 2 | n<20 | n<20 |

**Largest misses, India.** The ten IPOs with the biggest absolute open-gain error.

| company | listed | segment | GMP % | open gain % | error (pp) |
|---|---|---|---|---|---|
| Sungarner Energies Limited IPO | 2023-08-31 | IN_SME | +54.2 | +201.2 | -147.0 |
| Goyal Salt Limited IPO | 2023-10-11 | IN_SME | +102.6 | +242.1 | -139.5 |
| Phantom Digital Effects Limited IP | 2022-10-21 | IN_SME | +78.9 | +215.8 | -136.8 |
| sigachi industries limited | 2021-11-15 | IN_MAIN | +138.0 | +249.7 | -111.7 |
| Anlon Technology Solutions Limited | 2023-01-10 | IN_SME | +60.0 | +151.1 | -91.1 |
| Srivari Spices and Foods Limited I | 2023-08-18 | IN_SME | +64.3 | +141.7 | -77.4 |
| Navoday Enterprises Limited IPO | 2021-06-25 | IN_SME | +70.0 | -2.5 | +72.5 |
| C P S Shapers Limited IPO | 2023-09-07 | IN_SME | +73.0 | +143.2 | -70.3 |
| Trident Techlabs Limited IPO | 2023-12-29 | IN_SME | +114.3 | +180.4 | -66.1 |
| Shoora Designs Limited IPO | 2023-08-29 | IN_SME | +25.0 | +90.0 | -65.0 |


### 4.1 GMP against subscription (nested)

The same rows for every model in a row: a GMP reading and the total subscription multiple. R-squared is the share of the variance in the gain that a model explains. 'Adding' is the change in R-squared from adding one variable to the other, with a 95% bootstrap interval over IPOs.

| segment | outcome | n | R2 GMP only | R2 total only | R2 both | adding GMP to total | adding total to GMP |
|---|---|---|---|---|---|---|---|
| IN_MAIN | open | 82 | 0.842 | 0.379 | 0.860 | +0.482 [+0.291, +0.604] | +0.019 [+0.000, +0.039] |
| IN_MAIN | close | 94 | 0.748 | 0.371 | 0.757 | +0.386 [+0.198, +0.523] | +0.009 [+0.000, +0.036] |
| IN_SME | open | 192 | 0.637 | 0.397 | 0.641 | +0.245 [+0.170, +0.326] | +0.004 [+0.000, +0.022] |
| IN_SME | close | 191 | 0.624 | 0.382 | 0.629 | +0.247 [+0.172, +0.333] | +0.005 [+0.000, +0.024] |
| India pooled | open | 274 | 0.684 | 0.400 | 0.685 | +0.284 [+0.212, +0.368] | +0.001 [+0.000, +0.011] |
| India pooled | close | 285 | 0.658 | 0.387 | 0.659 | +0.273 [+0.200, +0.359] | +0.001 [+0.000, +0.013] |


## 5. Market conditions

Index context is measured as of the day before the open: the benchmark's trailing 5- and 20-day returns in percent, and the India VIX close. The table has it for India only (NIFTY 50 and India VIX). US, HK and OTHER have no open date, so they have no context.

| segment | condition | n open | rho open [95% CI] | n close | rho close [95% CI] | reading (close) |
|---|---|---|---|---|---|---|
| IN_MAIN | NIFTY 50, 20-day return | 596 | +0.13 [+0.05, +0.20] | 670 | +0.09 [+0.02, +0.16] | weak |
| IN_MAIN | NIFTY 50, 5-day return | 596 | +0.10 [+0.02, +0.18] | 670 | +0.09 [+0.01, +0.17] | weak |
| IN_MAIN | India VIX level | 579 | -0.10 [-0.18, -0.02] | 649 | -0.05 [-0.12, +0.03] | not distinguishable from noise |
| IN_SME | NIFTY 50, 20-day return | 735 | +0.06 [-0.01, +0.13] | 641 | +0.08 [-0.00, +0.16] | not distinguishable from noise |
| IN_SME | NIFTY 50, 5-day return | 735 | -0.02 [-0.09, +0.05] | 641 | -0.01 [-0.09, +0.07] | not distinguishable from noise |
| IN_SME | India VIX level | 735 | -0.12 [-0.20, -0.05] | 641 | -0.06 [-0.14, +0.02] | not distinguishable from noise |
| US | all context fields | 0 | n<20 | 0 | n<20 | no context in this table |
| HK | all context fields | 0 | n<20 | 0 | n<20 | no context in this table |
| OTHER | all context fields | 0 | n<20 | 0 | n<20 | no context in this table |

**IN_MAIN by NIFTY 50 20-day return**

| bucket | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| below -5% | 34 | -0.9 | 47% █████░░░░░ | 42 | +0.8 | 52% █████░░░░░ |
| -5% to 0% | 184 | +9.9 | 77% ████████░░ | 207 | +9.4 | 68% ███████░░░ |
| 0% to +5% | 299 | +11.1 | 78% ████████░░ | 329 | +11.1 | 71% ███████░░░ |
| +5% and above | 79 | +8.4 | 80% ████████░░ | 92 | +10.0 | 74% ███████░░░ |

**IN_MAIN by India VIX**

| bucket | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| below 15 | 347 | +11.4 | 78% ████████░░ | 383 | +10.5 | 71% ███████░░░ |
| 15 to 20 | 135 | +10.4 | 76% ████████░░ | 149 | +10.5 | 72% ███████░░░ |
| 20 to 25 | 67 | +4.0 | 67% ███████░░░ | 85 | +6.4 | 64% ██████░░░░ |
| 25 and above | 30 | +2.5 | 73% ███████░░░ | 32 | +1.1 | 50% █████░░░░░ |

**IN_SME by NIFTY 50 20-day return**

| bucket | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| below -5% | 35 | +1.8 | 77% ████████░░ | 23 | +1.4 | 65% ███████░░░ |
| -5% to 0% | 244 | +6.7 | 88% █████████░ | 223 | +7.3 | 80% ████████░░ |
| 0% to +5% | 369 | +8.5 | 81% ████████░░ | 325 | +11.1 | 79% ████████░░ |
| +5% and above | 87 | +12.4 | 85% █████████░ | 70 | +18.1 | 86% █████████░ |

**IN_SME by India VIX**

| bucket | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| below 15 | 453 | +9.7 | 83% ████████░░ | 411 | +10.5 | 80% ████████░░ |
| 15 to 20 | 191 | +5.8 | 87% █████████░ | 152 | +8.8 | 82% ████████░░ |
| 20 to 25 | 79 | +3.8 | 86% █████████░ | 68 | +7.2 | 76% ████████░░ |
| 25 and above | 12 | n<20 | n<20 | 10 | n<20 | n<20 |


## 6. Issue characteristics

Issue size is in millions of the listing currency (INR m for India, USD m for US, HKD m for HK). Bucket edges are quartiles of each market's own distribution, so the edges are printed.

| characteristic | segment | n open | rho open [95% CI] | n close | rho close [95% CI] | reading (close) |
|---|---|---|---|---|---|---|
| issue size | IN_MAIN | 550 | +0.01 [-0.07, +0.09] | 609 | +0.04 [-0.04, +0.13] | not distinguishable from noise |
| issue size | IN_SME | 1580 | +0.10 [+0.05, +0.15] | 1559 | +0.22 [+0.18, +0.27] | weak |
| issue size | US | 944 | +0.05 [-0.01, +0.12] | 1102 | +0.24 [+0.18, +0.29] | weak |
| issue size | HK | 280 | -0.02 [-0.15, +0.11] | 290 | +0.05 [-0.08, +0.17] | not distinguishable from noise |
| price band top (INR) | IN_MAIN | 307 | +0.11 [+0.00, +0.23] | 357 | +0.09 [-0.02, +0.20] | not distinguishable from noise |
| price band top (INR) | IN_SME | 137 | +0.05 [-0.13, +0.22] | 125 | +0.11 [-0.07, +0.28] | not distinguishable from noise |
| offer-for-sale share (%) | IN_MAIN | 106 | +0.09 [-0.08, +0.27] | 119 | +0.05 [-0.12, +0.21] | not distinguishable from noise |
| offer-for-sale share (%) | IN_SME | 66 | -0.21 [-0.43, +0.04] | 52 | -0.22 [-0.46, +0.06] | not distinguishable from noise |
| fresh-issue share (%) | IN_MAIN | 487 | -0.09 [-0.18, +0.00] | 545 | -0.08 [-0.17, +0.01] | not distinguishable from noise |
| fresh-issue share (%) | IN_SME | 298 | +0.19 [+0.09, +0.29] | 281 | +0.11 [-0.00, +0.21] | not distinguishable from noise |

**IN_MAIN by issue size quartile**

| size quartile | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| below 2,353 | 138 | +7.6 | 86% █████████░ | 153 | +10.5 | 71% ███████░░░ |
| 2,353 to 5,720 | 135 | +12.3 | 75% ███████░░░ | 154 | +9.2 | 68% ███████░░░ |
| 5,720 to 12,622 | 147 | +10.0 | 74% ███████░░░ | 160 | +10.8 | 72% ███████░░░ |
| 12,622 and above | 130 | +8.4 | 78% ████████░░ | 142 | +12.5 | 76% ████████░░ |

**IN_SME by issue size quartile**

| size quartile | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| below 72 | 381 | +5.5 | 83% ████████░░ | 362 | +0.3 | 54% █████░░░░░ |
| 72 to 162 | 394 | +5.2 | 80% ████████░░ | 371 | +2.5 | 62% ██████░░░░ |
| 162 to 337 | 400 | +6.6 | 81% ████████░░ | 406 | +6.2 | 71% ███████░░░ |
| 337 and above | 405 | +10.2 | 79% ████████░░ | 420 | +10.3 | 71% ███████░░░ |

**US by issue size quartile**

| size quartile | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| below 40 | 130 | +6.6 | 70% ███████░░░ | 321 | +1.2 | 54% █████░░░░░ |
| 40 to 100 | 312 | +8.1 | 76% ████████░░ | 278 | +9.6 | 76% ████████░░ |
| 100 to 240 | 289 | +13.3 | 76% ████████░░ | 248 | +19.4 | 77% ████████░░ |
| 240 and above | 213 | +9.4 | 77% ████████░░ | 255 | +12.6 | 77% ████████░░ |

**HK by issue size quartile**

| size quartile | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| below 156 | 78 | +7.0 | 59% ██████░░░░ | 78 | +2.5 | 56% ██████░░░░ |
| 156 to 454 | 72 | +1.3 | 56% ██████░░░░ | 70 | +1.1 | 53% █████░░░░░ |
| 454 to 1,499 | 64 | -0.0 | 47% █████░░░░░ | 70 | +0.7 | 59% ██████░░░░ |
| 1,499 and above | 66 | +1.5 | 58% ██████░░░░ | 72 | +2.8 | 58% ██████░░░░ |

**IN_MAIN by price band top, quartiles (INR)**

| price band top | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| below 125 | 75 | +4.8 | 76% ████████░░ | 89 | +2.4 | 60% ██████░░░░ |
| 125 to 263 | 73 | +4.1 | 71% ███████░░░ | 85 | +7.6 | 73% ███████░░░ |
| 263 to 523 | 79 | +14.5 | 73% ███████░░░ | 86 | +15.9 | 70% ███████░░░ |
| 523 and above | 80 | +10.3 | 79% ████████░░ | 97 | +9.6 | 70% ███████░░░ |

**IN_MAIN by offer-for-sale share**

| OFS share | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| below 25% | 12 | n<20 | n<20 | 14 | n<20 | n<20 |
| 25% to 50% | 20 | +10.1 | 80% ████████░░ | 21 | +10.5 | 81% ████████░░ |
| 50% to 75% | 21 | +8.0 | 81% ████████░░ | 27 | +10.3 | 74% ███████░░░ |
| 75% and above | 53 | +14.0 | 75% ████████░░ | 57 | +11.5 | 72% ███████░░░ |

**IN_MAIN by fresh-issue share**

| fresh share | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| below 50% | 138 | +13.2 | 78% ████████░░ | 156 | +15.3 | 74% ███████░░░ |
| 50% to 99.9% | 183 | +10.0 | 72% ███████░░░ | 201 | +7.3 | 68% ███████░░░ |
| 100% | 166 | +6.2 | 80% ████████░░ | 188 | +7.6 | 67% ███████░░░ |

**IN_SME by price band top, quartiles (INR)**

| price band top | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| below 52 | 34 | +17.9 | 91% █████████░ | 32 | +17.8 | 91% █████████░ |
| 52 to 74 | 35 | +20.0 | 97% ██████████ | 36 | +19.2 | 92% █████████░ |
| 74 to 100 | 29 | +20.0 | 93% █████████░ | 24 | +52.3 | 83% ████████░░ |
| 100 and above | 39 | +37.3 | 87% █████████░ | 33 | +44.1 | 82% ████████░░ |

**IN_SME by offer-for-sale share**

| OFS share | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| below 25% | 25 | +18.0 | 84% ████████░░ | 21 | +22.2 | 81% ████████░░ |
| 25% to 50% | 21 | +3.1 | 86% █████████░ | 17 | n<20 | n<20 |
| 50% to 75% | 13 | n<20 | n<20 | 10 | n<20 | n<20 |
| 75% and above | 7 | n<20 | n<20 | 4 | n<20 | n<20 |

**IN_SME by fresh-issue share**

| fresh share | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| below 50% | 12 | n<20 | n<20 | 9 | n<20 | n<20 |
| 50% to 99.9% | 53 | +6.5 | 87% █████████░ | 43 | +10.0 | 81% ████████░░ |
| 100% | 233 | +15.4 | 89% █████████░ | 229 | +15.4 | 82% ████████░░ |


## 7. Cross-market comparison

Does 'higher subscription, higher gain' hold outside India? India is compared on its total and its retail multiple. HK has only the public-offer multiple, compared here with India's retail multiple. The medians are for the lowest multiple (under 3x) and the highest (10x and above).

| segment | multiple | n close | rho close [95% CI] | close median, under 3x | close median, 10x and above | reading |
|---|---|---|---|---|---|---|
| IN_MAIN | total | 513 | +0.66 [+0.60, +0.71] | -4.5 (n 162) | +25.6 (n 266) | strong |
| IN_SME | total | 424 | +0.62 [+0.55, +0.68] | +1.3 (n 167) | +33.5 (n 187) | strong |
| IN_MAIN | retail | 485 | +0.53 [+0.47, +0.59] | -0.8 (n 190) | +29.7 (n 171) | strong |
| IN_SME | retail | 423 | +0.59 [+0.51, +0.65] | +1.7 (n 153) | +31.9 (n 200) | strong |
| HK | public-offer (retail) | 293 | +0.20 [+0.10, +0.31] | -0.1 (n 55) | +3.7 (n 177) | weak |
| US | none: no subscription field | 0 | n/a | n/a | n/a | cannot be tested |
| OTHER | none: no subscription field | 0 | n/a | n/a | n/a | cannot be tested |

HK is a selected subset: every HK row is a joined price-file IPO, and the HK sample lacks later years and delisted names (docs/data_sources/global.md). The US has no subscription data, so the pattern cannot be tested there; section 2 gives its gains only.


## 8. Practical reading for the live tool

Each row is one signal read at the close of the issue, with what followed historically. Rules are descriptive and one signal at a time: nothing combines signals or forecasts a specific IPO. The interval is a 95% bootstrap interval for the share above issue.

| if at close you see | segment | n close | close above issue [95% CI] | close median (p25 to p75) | open median | reading |
|---|---|---|---|---|---|---|
| total subscription below 3x | IN_MAIN | 162 | 37% [30, 44] | -4.5 (-11.0 to +2.5) | -0.5 | descriptive |
| total subscription 3x to 10x | IN_MAIN | 85 | 61% [51, 71] | +2.7 (-5.2 to +15.4) | +5.0 | descriptive |
| total subscription 10x and above | IN_MAIN | 266 | 90% [86, 94] | +25.6 (+10.0 to +45.8) | +22.4 | descriptive |
| retail >= 3x and QIB < 1x (flag) | IN_MAIN | 19 | n<20 | n<20 | n<20 | n<20 |
| India VIX 20 or higher | IN_MAIN | 117 | 60% [50, 68] | +6.0 (-8.1 to +29.8) | +3.8 | descriptive |
| total subscription below 3x | IN_SME | 167 | 69% [62, 76] | +1.3 (-2.3 to +7.5) | +1.6 | descriptive |
| total subscription 3x to 10x | IN_SME | 70 | 79% [69, 89] | +4.9 (+0.2 to +11.8) | +5.0 | descriptive |
| total subscription 10x and above | IN_SME | 187 | 90% [86, 95] | +33.5 (+11.4 to +78.2) | +36.3 | descriptive |
| retail >= 3x and QIB < 1x (flag) | IN_SME | 0 | n<20 | n<20 | n<20 | n<20 |
| India VIX 20 or higher | IN_SME | 78 | 74% [64, 83] | +6.5 (-0.7 to +39.3) | +3.0 | descriptive |
| GMP below 0% | India | 42 | 38% [24, 52] | -3.2 (-9.8 to +3.3) | -0.4 | descriptive |
| GMP 0% to 30% | India | 159 | 76% [69, 82] | +9.8 (+0.3 to +20.1) | +7.5 | descriptive |
| GMP 30% and above | India | 130 | 99% [98, 100] | +68.1 (+42.5 to +99.5) | +64.8 | descriptive |


### What the data cannot support

- GMP: no reading after 2024. Nothing can be checked for 2025-2026 listings, and GMP is untimed, so it cannot be matched to the open at any time of day.
- US: no listings in 2022, 2023 in this table, so the US pattern is unknown for those years.
- No live US or HK source exists in the tool (`src/ipo_radar/live` covers India only). None of the rules above can be applied to a US or HK IPO at serving time.
- No as-of index context for US, HK or OTHER: their rows have no open date.
- HK subscription is the public-offer multiple only. US has no subscription field, and OTHER (Indonesia) has close gains only, with no issue size.
- sNII and bNII exist from 2022 on, and the SME category fields are mostly pre-2024 (section 1).
- Survivorship: the joined rows (159 US, 323 HK) are a selected subset, so every rate here is optimistic for the average issue.
- First-day outcomes only. Withdrawn issues and IPOs that never listed are not in the table, so the rates describe listed IPOs.
- Observational. A pattern that held historically can change with allotment rules, anchor books or market conditions the table does not cover.
- Many estimates are reported with no correction for multiple testing (see Method). A 'CI excludes zero' result is one of many and should be read that way.


## Appendix A. Subscription bucket tables (descriptive)

Each table bins one multiple into the buckets shown. Cells with fewer than 20 rows print 'n<20'. Rates are shares of IPOs with the gain above issue; the bar is the same share out of ten.

**IN_MAIN: QIB**

| multiple | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| <1x | 32 | +3.1 | 78% ████████░░ | 37 | -0.6 | 49% █████░░░░░ |
| 1-3x | 80 | -1.3 | 45% ████░░░░░░ | 99 | -4.5 | 38% ████░░░░░░ |
| 3-10x | 91 | +1.1 | 55% █████░░░░░ | 102 | -0.4 | 47% █████░░░░░ |
| 10-30x | 60 | +9.3 | 85% ████████░░ | 72 | +8.8 | 76% ████████░░ |
| 30-100x | 93 | +20.4 | 90% █████████░ | 98 | +20.8 | 90% █████████░ |
| >100x | 115 | +37.4 | 98% ██████████ | 118 | +39.2 | 97% ██████████ |

**IN_MAIN: NII (HNI)**

| multiple | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| <1x | 57 | -3.4 | 30% ███░░░░░░░ | 71 | -4.1 | 35% ████░░░░░░ |
| 1-3x | 69 | +1.2 | 59% ██████░░░░ | 81 | -3.2 | 42% ████░░░░░░ |
| 3-10x | 74 | +3.8 | 73% ███████░░░ | 85 | +0.4 | 52% █████░░░░░ |
| 10-30x | 50 | +8.4 | 82% ████████░░ | 57 | +8.7 | 79% ████████░░ |
| 30-100x | 83 | +16.7 | 88% █████████░ | 93 | +18.0 | 84% ████████░░ |
| >100x | 128 | +35.7 | 99% ██████████ | 128 | +33.8 | 98% ██████████ |

**IN_MAIN: sNII**

| multiple | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| <1x | 36 | -0.4 | 50% █████░░░░░ | 42 | -1.2 | 40% ████░░░░░░ |
| 1-3x | 37 | +0.5 | 54% █████░░░░░ | 39 | -2.3 | 44% ████░░░░░░ |
| 3-10x | 33 | +5.1 | 58% ██████░░░░ | 38 | -0.2 | 50% █████░░░░░ |
| 10-30x | 52 | +10.3 | 81% ████████░░ | 58 | +8.8 | 79% ████████░░ |
| 30-100x | 92 | +22.3 | 90% █████████░ | 99 | +23.6 | 84% ████████░░ |
| >100x | 72 | +32.3 | 99% ██████████ | 74 | +29.5 | 97% ██████████ |

**IN_MAIN: bNII**

| multiple | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| <1x | 33 | -1.9 | 42% ████░░░░░░ | 39 | -2.5 | 36% ████░░░░░░ |
| 1-3x | 32 | -0.2 | 50% █████░░░░░ | 35 | -2.3 | 40% ████░░░░░░ |
| 3-10x | 34 | +2.5 | 65% ██████░░░░ | 37 | +1.8 | 62% ██████░░░░ |
| 10-30x | 45 | +10.6 | 78% ████████░░ | 50 | +8.1 | 70% ███████░░░ |
| 30-100x | 95 | +17.6 | 88% █████████░ | 103 | +21.6 | 83% ████████░░ |
| >100x | 83 | +29.0 | 99% ██████████ | 86 | +29.2 | 97% ██████████ |

**IN_MAIN: retail**

| multiple | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| <1x | 60 | -0.6 | 48% █████░░░░░ | 69 | -3.2 | 42% ████░░░░░░ |
| 1-3x | 103 | +0.8 | 57% ██████░░░░ | 121 | -0.0 | 50% █████░░░░░ |
| 3-10x | 108 | +10.0 | 84% ████████░░ | 124 | +6.0 | 69% ███████░░░ |
| 10-30x | 93 | +21.4 | 91% █████████░ | 97 | +23.6 | 88% █████████░ |
| 30-100x | 58 | +35.5 | 97% ██████████ | 58 | +33.8 | 97% ██████████ |
| >100x | 13 | n<20 | n<20 | 16 | n<20 | n<20 |

**IN_MAIN: employee**

| multiple | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| <1x | 59 | +3.2 | 69% ███████░░░ | 69 | +1.2 | 55% ██████░░░░ |
| 1-3x | 60 | +7.3 | 78% ████████░░ | 67 | +9.3 | 66% ███████░░░ |
| 3-10x | 53 | +13.2 | 83% ████████░░ | 59 | +13.5 | 78% ████████░░ |
| 10-30x | 28 | +21.5 | 100% ██████████ | 31 | +25.6 | 90% █████████░ |
| 30-100x | 12 | n<20 | n<20 | 13 | n<20 | n<20 |
| >100x | 5 | n<20 | n<20 | 5 | n<20 | n<20 |

**IN_MAIN: shareholder**

| multiple | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| <1x | 1 | n<20 | n<20 | 2 | n<20 | n<20 |
| 1-3x | 0 | n<20 | n<20 | 0 | n<20 | n<20 |
| 3-10x | 4 | n<20 | n<20 | 4 | n<20 | n<20 |
| 10-30x | 0 | n<20 | n<20 | 0 | n<20 | n<20 |
| 30-100x | 1 | n<20 | n<20 | 2 | n<20 | n<20 |
| >100x | 0 | n<20 | n<20 | 0 | n<20 | n<20 |

**IN_MAIN: total**

| multiple | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| <1x | 7 | n<20 | n<20 | 9 | n<20 | n<20 |
| 1-3x | 123 | -0.1 | 49% █████░░░░░ | 153 | -4.4 | 37% ████░░░░░░ |
| 3-10x | 79 | +5.0 | 76% ████████░░ | 85 | +2.7 | 61% ██████░░░░ |
| 10-30x | 52 | +11.7 | 79% ████████░░ | 62 | +9.4 | 76% ████████░░ |
| 30-100x | 133 | +21.9 | 95% █████████░ | 136 | +24.8 | 93% █████████░ |
| >100x | 66 | +43.3 | 100% ██████████ | 68 | +45.9 | 99% ██████████ |

**IN_SME: QIB**

| multiple | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| <1x | 1 | n<20 | n<20 | 0 | n<20 | n<20 |
| 1-3x | 9 | n<20 | n<20 | 12 | n<20 | n<20 |
| 3-10x | 15 | n<20 | n<20 | 14 | n<20 | n<20 |
| 10-30x | 22 | +39.1 | 95% ██████████ | 21 | +45.8 | 95% ██████████ |
| 30-100x | 37 | +90.0 | 100% ██████████ | 36 | +93.3 | 97% ██████████ |
| >100x | 5 | n<20 | n<20 | 5 | n<20 | n<20 |

**IN_SME: NII (HNI)**

| multiple | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| <1x | 0 | n<20 | n<20 | 0 | n<20 | n<20 |
| 1-3x | 20 | +3.9 | 80% ████████░░ | 17 | n<20 | n<20 |
| 3-10x | 4 | n<20 | n<20 | 4 | n<20 | n<20 |
| 10-30x | 8 | n<20 | n<20 | 7 | n<20 | n<20 |
| 30-100x | 8 | n<20 | n<20 | 9 | n<20 | n<20 |
| >100x | 24 | +61.5 | 100% ██████████ | 25 | +61.0 | 96% ██████████ |

**IN_SME: retail**

| multiple | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| <1x | 97 | +1.6 | 84% ████████░░ | 64 | +1.0 | 83% ████████░░ |
| 1-3x | 112 | +1.7 | 75% ████████░░ | 89 | +2.3 | 67% ███████░░░ |
| 3-10x | 80 | +4.2 | 78% ████████░░ | 70 | +4.0 | 69% ███████░░░ |
| 10-30x | 67 | +10.0 | 84% ████████░░ | 64 | +9.7 | 84% ████████░░ |
| 30-100x | 65 | +23.3 | 91% █████████░ | 64 | +22.4 | 84% ████████░░ |
| >100x | 80 | +78.6 | 98% ██████████ | 72 | +77.2 | 97% ██████████ |

**IN_SME: employee**

| multiple | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| <1x | 0 | n<20 | n<20 | 1 | n<20 | n<20 |
| 1-3x | 4 | n<20 | n<20 | 4 | n<20 | n<20 |
| 3-10x | 1 | n<20 | n<20 | 1 | n<20 | n<20 |
| 10-30x | 0 | n<20 | n<20 | 0 | n<20 | n<20 |
| 30-100x | 0 | n<20 | n<20 | 0 | n<20 | n<20 |
| >100x | 0 | n<20 | n<20 | 0 | n<20 | n<20 |

**IN_SME: total**

| multiple | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| <1x | 6 | n<20 | n<20 | 1 | n<20 | n<20 |
| 1-3x | 222 | +1.7 | 77% ████████░░ | 166 | +1.2 | 69% ███████░░░ |
| 3-10x | 77 | +5.0 | 83% ████████░░ | 70 | +4.9 | 79% ████████░░ |
| 10-30x | 50 | +9.3 | 82% ████████░░ | 51 | +8.9 | 82% ████████░░ |
| 30-100x | 65 | +26.7 | 94% █████████░ | 64 | +28.1 | 89% █████████░ |
| >100x | 82 | +78.0 | 98% ██████████ | 72 | +77.2 | 97% ██████████ |

**HK: public-offer (retail)**

| multiple | n open | open median | open above issue | n close | close median | close above issue |
|---|---|---|---|---|---|---|
| <1x | 12 | n<20 | n<20 | 14 | n<20 | n<20 |
| 1-3x | 38 | -0.2 | 42% ████░░░░░░ | 41 | +0.0 | 51% █████░░░░░ |
| 3-10x | 62 | +2.2 | 58% ██████░░░░ | 61 | +0.9 | 57% ██████░░░░ |
| 10-30x | 68 | +4.5 | 59% ██████░░░░ | 71 | +3.7 | 55% █████░░░░░ |
| 30-100x | 36 | -1.0 | 44% ████░░░░░░ | 37 | +0.6 | 54% █████░░░░░ |
| >100x | 67 | +14.1 | 67% ███████░░░ | 69 | +6.1 | 71% ███████░░░ |

