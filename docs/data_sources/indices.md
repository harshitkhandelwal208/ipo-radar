# Index series

Built by `ipo_radar/data/indices.py` into `data/processed/indices.parquet` (columns `date, symbol, close`).
Run: `.venv/bin/python -m ipo_radar.data.indices`. The build is offline: it reads the Kaggle files under
`data/raw/kaggle/<owner>__<slug>/` (catalogue and warnings in [kaggle.md](kaggle.md)). Live values come from
`ipo_radar/data/context.live_context`, not from this file.

Window: 2005-01-01 to the last date each source has. Dates are never forward-filled: a day a source does not
have stays absent, and context becomes NaN for IPOs that need it.

## Result of the last build (2026-10-15)

| Symbol | Status | First | Last | Rows | Sources (in splice order) |
|---|---|---|---|---|---|
| NIFTY50 | ok | 2005-01-03 | 2026-10-13 | 5,398 | belbino `^NSEI` + sudalairajkumar `NIFTY 50.csv` + debashis74017 `NIFTY 50_Data.csv` |
| SENSEX | ok | 2005-01-03 | 2026-10-13 | 5,397 | belbino `^BSESN` + avtnshm `CSVForDate (1).xlsx` |
| INDIAVIX | ok | 2010-01-04 | 2026-10-14 | 4,164 | ahaanverma00 `INDIA VIX` + saurabhbadole `India VIX.csv` |
| SP500 | ok | 2005-01-03 | 2026-10-13 | 5,479 | belbino `^GSPC` + henryhan117 `SPX.csv` |
| NYSE_COMP | ok | 2005-01-03 | 2023-07-28 | 4,674 | mattiuzc `NYA` + pavankrishnanarne `^NYA` |
| NASDAQ_COMP | ok | 2005-01-03 | 2026-10-13 | 5,479 | belbino `^IXIC` + mattiuzc `IXIC` |
| VIX | ok | 2005-01-03 | 2024-06-24 | 4,901 | guillemservera `vix_daily.csv` |
| HSI | ok | 2005-01-03 | 2026-10-13 | 5,364 | belbino `^HSI` + rushil999 `HSI.csv` + mattiuzc `HSI` |
| SHCOMP | ok | 2005-01-04 | 2026-10-13 | 5,282 | belbino `000001.SS` + mattiuzc `000001.SS` + code1110 `YF000001-SS` |

Total 46,138 rows. Every symbol was built, so `indices.parquet` was written.

Gaps that matter for the model:
- **VIX (US) ends 2024-06-24.** No file found for 2024-06 to today. US IPOs after that get `vix_level` NaN.
- **NYSE_COMP ends 2023-07-28.** No later source. Context for NYSE-era IPOs after that is NaN.
- **INDIAVIX starts 2010-01-04.** The 2009 rows in sudalairajkumar are not used (see the stopped splice below).
- **SP500, NIFTY50, SENSEX, HSI, NASDAQ_COMP, SHCOMP** run to 2026-10-13 (the belbino end date).
- The 2026-10-14 closes for NIFTY 50 and INDIA VIX come from NSE at live time (see below), not from this file.

## Splice rules

Each symbol has an ordered ladder of rungs (`LADDER` in `indices.py`). The first rung is the primary series.
A later rung is spliced in only if:

1. at least `MIN_OVERLAP = 20` dates are shared with the series so far, and
2. the largest close difference on the shared dates is at most `MAX_DIFF = 0.5%`.

If a rung with enough overlap disagrees by more than 0.5% on any shared date, splicing **stops for that symbol**
and the later rungs are ignored. A rung with too little overlap is skipped, and the ladder continues. Dates a
later rung adds are only dates the series does not have yet, so a gap is never filled from a neighbour.

Splice log from the last build (shared dates, largest difference):

| Symbol | Rung | Result |
|---|---|---|
| NIFTY50 | sudalairajkumar | agrees, 3,994 shared, max 0.009%, adds 719 rows (2005-2007) |
| NIFTY50 | debashis74017 | agrees, 4,388 shared, max 0.009%, adds 0 |
| SENSEX | avtnshm | agrees, 4,678 shared, max 0.352%, adds 43 rows (2005-2007) |
| INDIAVIX | saurabhbadole | agrees, 1,519 shared, max 0.060%, adds 143 rows (2026-03 to 2026-10) |
| INDIAVIX | alangeorgev | **stopped**: max 4.99% (2013-08-22) on 3,822 shared dates |
| SP500 | henryhan117 | agrees, 3,989 shared, max 0.000%, adds 0 |
| NYSE_COMP | pavankrishnanarne | agrees, 3,229 shared, max 0.000%, adds 544 rows (2021-06 to 2023-07) |
| NYSE_COMP | code1110 | **stopped**: max 1.62% on 4,302 shared dates (2021-2022 days after pavankrishnanarne) |
| NASDAQ_COMP | mattiuzc | agrees, 4,130 shared, max 0.000%, adds 0 |
| VIX | debashis74017 VIX_History | **stopped**: max 4.20% (2012-12-24) on 4,432 shared dates |
| HSI | rushil999, mattiuzc | agree, max 0.000%, add 0 |
| SHCOMP | mattiuzc, code1110 | agree, max 0.000% and 0.036%, add 0 |

### Conflicts, measured (for deciding whether the 0.5% rule should change)

The rule uses the largest single-day difference, so one bad print stops a splice. Measured on all shared dates:

- **INDIAVIX ahaanverma00 vs alangeorgev**: 3,819 shared dates, median |difference| 0.011%, p95 0.036%, 8 dates
  above 0.5%, largest 4.99% on 2013-08-22. Shifting one business day makes the median 2.6%, so the dates line up.
  This looks like a few bad prints, not a different series.
- **VIX guillemservera vs debashis74017 VIX_History**: 4,432 shared dates, median 0.000%, about 9 dates above 0.5%,
  largest 4.20% on 2012-12-24. debashis adds no dates in range anyway (it ends 2022-08), so the outcome does not change.
- **NYSE_COMP pavankrishnanarne vs code1110**: exact agreement up to 2020, then up to 1.62% on some 2021-2022
  days. The 2021-06 to 2023-07 NYA extension therefore rests on pavankrishnanarne alone for those dates.
- **SENSEX avtnshm vs belbino**: 0.352% largest, accepted. Some Sensex values differ by up to about a third of a percent.

If the rule is relaxed to allow isolated bad days (for example, stop only when more than 1% of shared dates
differ), alangeorgev would be accepted for INDIAVIX, but it adds no new dates (ahaanverma00 and saurabhbadole
already cover 2010 to 2026-10). It would only add a second check. VIX would stay as it is.

## Sources checked and not used

- **saurabhbadole NIFTY / SENSEX files**: one month of rolling data, not history. Its India VIX file is used
  (above).
- **ruanqian Shanghai (1990-2016)**: forward-adjusted (前复权), so it does not match the raw closes used elsewhere.
  Not needed: belbino and mattiuzc already cover SHCOMP from 2000.
- **mukhazarahmad INDICES_DATA.csv** (NYA 1996-2022): two-row header; only a cross-check for NYSE. Not used.
- **andrewmvd nse_indexes.csv** (NIFTY 50 and INDIA VIX, 1995-2022): a possible cross-check for NIFTY and India VIX
  before 2010. Not used yet; the integrator should check it first (kaggle.md, rejected-datasets table, andrewmvd row).
- **jatinsh NIFTY50 2000-2024**: duplicates sudalairajkumar.
- **debashis74017 INDIA VIX_minute.csv**: intraday; a daily close is not defined in this data.
- **FRED** (VIXCLS, SP500, NASDAQCOM): not used. FRED over plain HTTP/1.1 timed out from this network, and the
  project was not to use it unless it worked that way.
- **NSE history endpoint** (`/api/historical/indicesHistory`): returned 503 from this network. Not used.

## Live context (`live_context`)

- NIFTY 50 and INDIA VIX: NSE `allIndices` (through the existing `live/nse.py` session helper). Its
  `previousDay` / `previousDayVal` pair is a dated close, not the intraday quote. It is appended to the local
  series only when it is newer than the local last date. On 2026-10-15 it added NIFTY 50 for 2026-10-14
  (22,622.70). The local INDIA VIX close for 2026-10-14 (14.625) agrees with NSE (14.63).
- SP500, VIX, HSI: local series only, with no network call.
- A benchmark whose last close is more than 5 weekdays before today gives NaN, with a note. Weekdays are used
  because there is no exchange holiday calendar, which makes the rule slightly stricter.
- NSE unreachable: the local series is used alone, with a note.
- `idx_ret_issue_window` is always NaN in live use, because it needs the issue close.

## Licences (from the Kaggle metadata; see kaggle.md)

- CC0-1.0: belbino (Yahoo-derived, check Yahoo's terms), guillemservera, debashis74017, pavankrishnanarne.
- CC BY 4.0: ahaanverma00. Apache-2.0: avtnshm, rushil999. DbCL-1.0: henryhan117.
- **CC BY-SA 4.0: sudalairajkumar** (used for NIFTY50 2005-2007 rows). Derived rows keep the share-alike term if the
  file is shared.
- **"copyright-authors" (no reuse grant): mattiuzc.** Used for NYA, IXIC, HSI and SHCOMP. Personal use only.
- **"other": code1110 (Yahoo data), saurabhbadole** (India VIX 2026 extension and the not-used NIFTY/SENSEX files).

This file is for personal analysis. Redistribution is not cleared for any of these series.
