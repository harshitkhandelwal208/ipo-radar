# ipo-radar

ipo-radar is a personal command-line tool for Indian IPOs. It shows live subscription, grey-market premium (GMP) and dates, estimates listing-day gains with a model trained on past IPOs from several markets, and can send phone alerts. It is one command, `ipo`.

## What it is

- **Live tracker (India only).** Subscription by category (QIB, NII/HNI with sNII and bNII where reported, retail, employee, shareholder, total), GMP with its source and age, dates, price band and lot size.
- **Listing-gain model.** A LightGBM quantile model trained on the history table: 4,933 IPOs (IN_MAIN 1,029, IN_SME 1,758, US 1,721, HK 323, OTHER 102). It gives a 10th, 50th and 90th percentile estimate at five stages, from before the issue opens to just before listing, as subscription data arrives.
- **Phone alerts.** Sent through ntfy.sh (a free app, no account) or a Telegram bot. Alerts are de-duplicated locally.
- **One CLI.** `ipo`. The subcommands are listed under Quick start.

## Status

Version 0.1.0, experimental. The data layout, the CLI and the model may change before 1.0. The project is built for one user on a personal machine, with free tools only and no paid APIs.

## Not investment advice

This is not investment advice. The output is a statistical estimate from a small history of IPOs, and on recent data the model does not beat a simple median at every stage (see Model). Do not apply for, buy or sell an IPO on the strength of this tool alone.

## Install

Requires Python 3.12 or newer.

```
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

The `dev` extra adds pytest, optuna and huggingface_hub, used by the tests, the model search and the HuggingFace data download. It also lists shap, which the code does not import.

## Quick start

Every command below is taken from `ipo --help` and the help of each subcommand.

Dashboard and analysis:

```
.venv/bin/ipo fetch --no-alerts       # refresh every live source, predict, show the dashboard
.venv/bin/ipo live                    # the dashboard from the cache (goes to the network if the cache is over 5 minutes old)
.venv/bin/ipo show <name-or-symbol>   # detailed analysis of one IPO
.venv/bin/ipo history --market in_main --sort gain --limit 20
.venv/bin/ipo doctor                  # check packages, data files, model, live sources, alerts and config
```

Phone alerts:

```
.venv/bin/ipo alerts setup            # create an ntfy topic and print how to subscribe on the phone
.venv/bin/ipo alerts test             # one labelled test notification to every configured channel
.venv/bin/ipo alerts status
.venv/bin/ipo alerts run-once         # one alert cycle (what the systemd timer runs)
.venv/bin/ipo alerts install-timer    # write systemd user units, enable a timer every 15 minutes
```

Data, model and reports (these write files):

```
.venv/bin/ipo data build              # build data/processed/ from the raw downloads in data/raw/
.venv/bin/ipo train                   # train locally into models/ (--quick for a fast sanity check, --out-dir for a scratch folder)
.venv/bin/ipo eval                    # walk-forward evaluation; rewrites docs/MODEL_REPORT.md
.venv/bin/ipo research                # rewrites docs/RESEARCH.md
```

`ipo data build` needs the raw downloads under `data/raw/`. They are not in the repository; `docs/data_sources/` says what to download and from where.

`ipo train --where kaggle` uploads the history table as a private Kaggle dataset and runs the search on a Kaggle kernel. Use it only if you accept that upload.

## Data sources

| Source | Used for | Licence | Notes |
|---|---|---|---|
| HuggingFace `sohomghosh/Indian_IPO_datasets` | India IPO anchor rows, NSE listing prices, subscriptions, GMP | CC BY-NC-SA 4.0 | Derived tables carry the same non-commercial, share-alike restriction. See `docs/data_sources/india.md`. |
| Kaggle, India IPO datasets (8 used) | Lot size, dates, subscription categories, listing prices, GMP (one dataset, when it matches HuggingFace) | Varies: CC0-1.0, CC BY-SA 4.0, MIT, DbCL-1.0, and "other" (two datasets) | Licence per dataset in `docs/data_sources/kaggle.md`. Several catalogued datasets have licence "unknown"; they are rejected or not used. |
| Kaggle, global IPO and price datasets | US first-day data (2010-2021), HK (2020-2024), Indonesia (2022-2023), price joins | CC0-1.0, MIT, Apache-2.0, CC BY-NC-SA 4.0 (Indonesia) | The Apache-2.0 HK price dataset carries a description copied from another dataset; only its numbers are used. See `docs/data_sources/global.md`. |
| Kaggle, index history | NIFTY 50, SENSEX, India VIX, S&P 500, NASDAQ, NYSE composite, US VIX, Hang Seng, Shanghai composite | CC0-1.0, CC BY 4.0, Apache-2.0, DbCL-1.0, CC BY-SA 4.0, "copyright-authors" (no reuse grant), "other" | Series derived from Yahoo Finance may be subject to Yahoo's terms. See `docs/data_sources/indices.md`. |
| IPO Scoop (iposcoop.com) priced-IPO tables | US listings 2024-2026 | Not stated; no terms page found | robots.txt allows the pages. |
| Live sites: NSE, Chittorgarh, investorgain, ipowatch, ipopremium, ipocentral | Live subscription, dates, GMP, index values | Not licensed | Scraped for personal use. Their terms have not been cleared. |

The repository redistributes copies of some of this data: small fixtures under `tests/fixtures/`, and derived tables and model files under `models/`. Some of those copies are not trimmed (for example the full processed table in `models/kaggle-20261015-131100/upload/`). They keep the licences above.

## Model

The model estimates the listing-day open gain (the figures below are for that target). It is trained on the history table, with one model per segment: IN_MAIN (India mainboard), IN_SME (India SME) and GLOBAL (US, HK and OTHER). GMP is used only by the pre-listing model, and only for IPOs that have a GMP reading.

Walk-forward evaluation: every listing block is predicted by a model trained only on IPOs listed before it. The numbers below come from `docs/MODEL_REPORT.md`.

**2022-2024 walk-forward, pooled over segments.** "p50 - median" is the model's median-guess error minus the segment's training-median error, in percentage points of the issue price, with a 95% bootstrap interval. Negative means the model is closer.

| stage | n | p50 - median (95% CI) | worse than median | beats median |
|---|---|---|---|---|
| pre_open | 882 | -1.0 [-1.4, -0.6] | no | yes |
| day1 | 882 | -3.2 [-4.1, -2.3] | no | yes |
| day2 | 882 | -3.6 [-4.6, -2.7] | no | yes |
| close | 882 | -4.7 [-5.7, -3.7] | no | yes |
| pre_listing | 882 | -6.5 [-7.9, -5.3] | no | yes |

**2025-2026, locked.** Recorded once. Units are percentage points of the issue price.

| stage | n | MAE p50 | MAE median | p50 - median (95% CI) | coverage p10-p90 % | AUC |
|---|---|---|---|---|---|---|
| pre_open | 470 | 20.5 | 18.4 | +2.1 [+1.4, +2.8] | 94 | 0.463 |
| day1 | 470 | 19.8 | 18.4 | +1.4 [+0.8, +1.9] | 94 | 0.475 |
| day2 | 470 | 19.9 | 18.4 | +1.4 [+0.9, +1.9] | 94 | 0.494 |
| close | 470 | 18.4 | 18.4 | -0.1 [-0.9, +0.8] | 95 | 0.641 |
| pre_listing | 470 | 18.3 | 18.4 | -0.1 [-0.9, +0.6] | 95 | 0.644 |

What this means:

- **2022-2024:** the model beats the historical median, pooled over segments, at all five stages. Per segment, not every stage passes; the per-stage flags are in `docs/MODEL_REPORT.md`.
- **2025-2026:** the model is worse than the median at pre-open, day 1 and day 2, and not clearly different at close and pre-listing. It is not better than the median at any stage.
- **Dashboard:** at the stages where the model is worse on the latest blocks (pre-open, day 1 and day 2), the dashboard shows the segment's training median in place of the model, marked "model not used".
- **2025-2026 is no longer a holdout.** It has been used for serving caveats and reviewed repeatedly. It is reported once, for the record, and no choice was made from it. No clean holdout remains until new IPOs are tracked after release.
- **In-sample numbers are not evidence.** The released bundle is trained on all IPOs, so its fit to the data it was trained on says nothing about accuracy.

Release settings: five seeds (7 to 11), default LightGBM parameters (a tuned set was searched and not adopted), conformal interval widening, monotone constraints on subscription and GMP in the classifier, and no shrinkage toward the median (weight 1.0 at every stage).

Descriptive statistics on the same table, with no correction for multiple testing, are in `docs/RESEARCH.md`.

## Limitations

- **No US IPOs from 2022 or 2023** in the table.
- **Hong Kong** rows end in 2024; the price source for the joined HK rows ends in May 2024.
- **SME category subscription** is almost entirely missing after 2023 (2024: 2% of rows have retail subscription; 2025-2026: none).
- **sNII and bNII** exist only from 2022 onward.
- **GMP** readings stop in 2024, so no 2025-2026 listing can be checked against GMP. GMP is unofficial grey-market data and is untimed. It is used only by the pre-listing model, on the 310 training rows that have a reading (IN_MAIN 118, IN_SME 192).
- **Open-gain targets** are missing for US IPOs from 2024 and for all OTHER (Indonesia) IPOs, so GLOBAL has no open-gain outcome in 2025-2026.
- **Survivorship bias.** The table holds only IPOs that listed, and the joined price rows (US 2019-2021, HK) are a selected subset. Every estimate is optimistic for the average issue.
- **Small samples.** In 2022-2024 each segment has 121 to 598 IPOs per stage (IN_MAIN 163, IN_SME 598, GLOBAL 121). The 2025-2026 block has 470 per stage.
- **Sources disagree.** When two sources give different values for a field, the build blanks the field rather than choosing one. Many cells are therefore missing.
- **No valuation or financial inputs at serving time.** The live feed carries no valuation or financial ratios, so those model inputs are missing when the tool runs.
- **The interval is too wide on recent data.** On 2025-2026 the p10-p90 band held 94-95% of outcomes against a target of 80%. The width was fitted on 2022-2024 and does not transfer.
- **The probability of opening above the issue price is weakly informative.** AUC on 2025-2026 runs from 0.46 to 0.64.
- **Use it for the size of the typical move, not its direction.**
- **No live US or HK source.** The live tracker covers India only.

## Security note

Model bundles (`models/*.joblib`) are Python pickles. Loading a pickle can run code, so load only bundles you made or trust. `ipo train --where kaggle` downloads a bundle and checks it before replacing the local one, but it still loads a pickle; do not point it at a bundle from anywhere else.

## Licence

No licence has been chosen yet. Until one is added, the code is all rights reserved. The trained model is derived from CC BY-NC-SA data and should be treated as non-commercial.

Choosing a licence is an open decision.

## More

- [docs/SPEC.md](docs/SPEC.md): the build rules this project follows
- [docs/MODEL_REPORT.md](docs/MODEL_REPORT.md): full model evaluation
- [docs/RESEARCH.md](docs/RESEARCH.md): descriptive research on listing-day gains
- [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md): index of the data-source notes
- [CHANGELOG.md](CHANGELOG.md): release history
