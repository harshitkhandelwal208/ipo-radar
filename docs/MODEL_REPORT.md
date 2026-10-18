# Model report: release evaluation

Generated 2026-10-18 13:53 UTC. Bundle 20261018-14663db9, data hash 14663db9, seeds [7, 8, 9, 10, 11]. Design: `{"conformal": true, "monotone": true, "gmp_model": "blend", "routing": {"IN_MAIN": "IN_MAIN", "IN_SME": "india", "GLOBAL": "pooled"}}`. Shrink weights: `{"pre_open": 1.0, "day1": 1.0, "day2": 1.0, "close": 1.0, "pre_listing": 1.0}`. Parameters: `{}`.

## Data change: placeholder zeros blanked

After the locked blocks 2025-2026 were first evaluated, the build started blanking exact-zero listing gains and GMP readings as placeholders, because sources write 0 where no value exists. Cells blanked: open 389, high 45, low 75, close 128, gmp_last 50, GMP % 50, sub_qib 15, sub_nii 1, sub_retail 15, sub_employee 2, sub_qib_d1 158, sub_nii_d1 45, sub_retail_d1 59, sub_total_d1 20, sub_qib_d2 59, sub_nii_d2 18, sub_retail_d2 15, sub_total_d2 3, sub_qib_d3 14, sub_nii_d3 6, sub_retail_d3 2. The locked blocks were evaluated again on the changed table, so they are no longer blind to it. The earlier report is kept at models/superseded-zero-fix/MODEL_REPORT.before-zero-fix.md.

| stage | open rows | MAE p50 | MAE median | p50 - median (95% CI) | coverage % | AUC |
|---|---|---|---|---|---|---|
| pre_open | 521 to 470 | 18.6 to 20.5 | 17.1 to 18.4 | +1.5 [+0.9, +2.0] to +2.1 [+1.4, +2.8] | 95 to 94 | 0.487 to 0.463 |
| day1 | 521 to 470 | 18.3 to 19.8 | 17.1 to 18.4 | +1.2 [+0.7, +1.7] to +1.4 [+0.8, +1.9] | 94 to 94 | 0.486 to 0.475 |
| day2 | 521 to 470 | 18.1 to 19.9 | 17.1 to 18.4 | +1.0 [+0.6, +1.4] to +1.4 [+0.9, +1.9] | 93 to 94 | 0.477 to 0.494 |
| close | 521 to 470 | 16.8 to 18.4 | 17.1 to 18.4 | -0.3 [-1.0, +0.4] to -0.1 [-0.9, +0.8] | 95 to 95 | 0.629 to 0.641 |
| pre_listing | 521 to 470 | 16.8 to 18.3 | 17.1 to 18.4 | -0.3 [-1.0, +0.4] to -0.1 [-0.9, +0.6] | 95 to 95 | 0.624 to 0.644 |


## Protocol

- Choices (conformal, monotone, GMP model, segment routing) were made on walk-forward blocks 2022-2024 only. Blocks 2025-2026 are locked: evaluated once for the chosen design. Skill flags use 2022-2024. Serving caveats use the latest blocks (2025-2026) for the 'worse than the historical median' warning only. This warns more, never less; it does not change the model or any choice.
- Selection blocks: listing years 2022, 2023, 2024. Every choice in this release (shrink weights, the Optuna search, the parameter decision) was made on those blocks only. Every block is predicted by models trained only on IPOs listed before it.
- 2025-2026 is no longer a holdout: it has been used for serving caveats and reviewed repeatedly. It is reported once, for the record, and no choice was made from it. No clean holdout remains until new IPOs are tracked after release.

## Release: 2022-2024 before and after

Pinball is the mean of the p10, p50 and p90 losses. 'vs median' is the model's p50 MAE minus the segment's training median MAE on the same IPOs, with a 95% bootstrap interval (negative is better).

| stage | previous config, previous data (1 seed) | previous config, current data (1 seed) | tuned params, current data (1 seed) | release: tuned if chosen, 5 seeds, shrink |
|---|---|---|---|---|
| pre_open | MAE 36.7; pinball 13.29; coverage 73%; vs median -1.0 [-1.4, -0.7] | MAE 36.7; pinball 13.29; coverage 73%; vs median -1.0 [-1.4, -0.6] | MAE 36.8; pinball 13.15; coverage 72%; vs median -0.8 [-1.1, -0.6] | MAE 36.7; pinball 13.28; coverage 72%; vs median -1.0 [-1.4, -0.6] |
| day1 | MAE 34.5; pinball 12.39; coverage 73%; vs median -3.2 [-4.1, -2.3] | MAE 34.4; pinball 12.35; coverage 72%; vs median -3.3 [-4.2, -2.4] | MAE 34.9; pinball 12.46; coverage 71%; vs median -2.8 [-3.5, -2.2] | MAE 34.5; pinball 12.35; coverage 72%; vs median -3.2 [-4.1, -2.4] |
| day2 | MAE 34.1; pinball 12.29; coverage 72%; vs median -3.6 [-4.6, -2.6] | MAE 34.1; pinball 12.27; coverage 72%; vs median -3.6 [-4.6, -2.7] | MAE 34.2; pinball 12.28; coverage 71%; vs median -3.5 [-4.3, -2.7] | MAE 34.1; pinball 12.27; coverage 72%; vs median -3.6 [-4.6, -2.7] |
| close | MAE 33.0; pinball 11.93; coverage 72%; vs median -4.7 [-5.7, -3.7] | MAE 33.0; pinball 11.94; coverage 72%; vs median -4.7 [-5.8, -3.7] | MAE 33.3; pinball 12.01; coverage 71%; vs median -4.4 [-5.3, -3.6] | MAE 33.0; pinball 11.94; coverage 72%; vs median -4.7 [-5.7, -3.7] |
| pre_listing | MAE 31.2; pinball 11.55; coverage 72%; vs median -6.5 [-7.7, -5.2] | MAE 31.1; pinball 11.58; coverage 72%; vs median -6.5 [-7.9, -5.3] | MAE 31.2; pinball 11.55; coverage 72%; vs median -6.5 [-7.8, -5.3] | MAE 31.2; pinball 11.59; coverage 71%; vs median -6.5 [-7.9, -5.3] |
| ALL | MAE 33.9; pinball 12.29; coverage 72%; vs median -3.8 [-4.3, -3.4] | MAE 33.9; pinball 12.29; coverage 72%; vs median -3.8 [-4.3, -3.4] | MAE 34.1; pinball 12.29; coverage 71%; vs median -3.6 [-4.0, -3.2] | MAE 33.9; pinball 12.29; coverage 72%; vs median -3.8 [-4.3, -3.4] |

Decision: **default** parameters (the tuned set is used only if its 2022-2024 single-seed pinball loss is lower than the default's). Chosen overrides: `{}`.
Optuna search: 40 trials on 2022-2024. Default settings loss 12.9685, best 12.6402. Best settings: `{"learning_rate": 0.0113, "num_leaves": 30.0, "min_child_samples": 11.0, "feature_fraction": 0.8216, "bagging_fraction": 0.9336, "reg_lambda": 0.2388}`.

Shrink weights were chosen per stage by the pinball loss of the blended p50 on 2022-2024:

| stage | chosen w | pinball at w=0.00 | pinball at w=0.25 | pinball at w=0.50 | pinball at w=0.75 | pinball at w=1.00 |
|---|---|---|---|---|---|---|
| pre_open | 1.0 | 18.843 | 18.622 | 18.482 | 18.392 | 18.347 |
| day1 | 1.0 | 18.843 | 18.197 | 17.700 | 17.371 | 17.229 |
| day2 | 1.0 | 18.843 | 18.104 | 17.538 | 17.175 | 17.031 |
| close | 1.0 | 18.843 | 17.944 | 17.245 | 16.767 | 16.498 |
| pre_listing | 1.0 | 18.843 | 17.735 | 16.794 | 16.085 | 15.581 |

## Headline: 2025-2026 (recorded once, not used for any choice)

Units: percentage points (pp) of the issue price. Intervals are 95% bootstrap intervals over IPOs. The per-stage skill flags come from 2022-2024 only.

| stage | n | MAE p50 | MAE median | p50 - median (95% CI) | MAE zero | n with GMP | MAE model (GMP rows) | MAE GMP | model - GMP (CI) | coverage p10-p90 % | AUC | Brier | Brier base rate | 2022-24 skill: p50 beats median (segments) | 2022-24 skill: prob beats base (segments) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pre_open | 470 | 20.5 | 18.4 | +2.1 [+1.4, +2.8] | 19.3 | 0 | n/a | n/a | n/a | 94 | 0.463 | 0.251 | 0.237 | IN_SME, GLOBAL | GLOBAL |
| day1 | 470 | 19.8 | 18.4 | +1.4 [+0.8, +1.9] | 19.3 | 0 | n/a | n/a | n/a | 94 | 0.475 | 0.249 | 0.237 | IN_MAIN, IN_SME, GLOBAL | GLOBAL |
| day2 | 470 | 19.9 | 18.4 | +1.4 [+0.9, +1.9] | 19.3 | 0 | n/a | n/a | n/a | 94 | 0.494 | 0.246 | 0.237 | IN_MAIN, IN_SME, GLOBAL | GLOBAL |
| close | 470 | 18.4 | 18.4 | -0.1 [-0.9, +0.8] | 19.3 | 0 | n/a | n/a | n/a | 95 | 0.641 | 0.226 | 0.237 | IN_MAIN, IN_SME, GLOBAL | GLOBAL |
| pre_listing | 470 | 18.3 | 18.4 | -0.1 [-0.9, +0.6] | 19.3 | 0 | n/a | n/a | n/a | 95 | 0.644 | 0.225 | 0.237 | IN_MAIN, IN_SME, GLOBAL | GLOBAL |
| ALL | 2350 | 19.4 | 18.4 | +0.9 [+0.6, +1.3] | 19.3 | 0 | n/a | n/a | n/a | 95 | 0.548 | 0.240 | 0.237 | n/a | n/a |

## Same measures, 2023-2024

| stage | n | MAE p50 | MAE median | p50 - median (95% CI) | MAE zero | n with GMP | MAE model (GMP rows) | MAE GMP | model - GMP (CI) | coverage p10-p90 % | AUC | Brier | Brier base rate | 2022-24 skill: p50 beats median (segments) | 2022-24 skill: prob beats base (segments) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pre_open | 669 | 39.3 | 40.6 | -1.3 [-1.8, -0.8] | 43.4 | 199 | 34.1 | 16.7 | +17.4 [+13.1, +21.7] | 76 | 0.691 | 0.116 | 0.123 | IN_SME, GLOBAL | GLOBAL |
| day1 | 669 | 37.1 | 40.6 | -3.5 [-4.6, -2.4] | 43.4 | 199 | 27.2 | 16.7 | +10.5 [+7.7, +13.5] | 73 | 0.717 | 0.118 | 0.123 | IN_MAIN, IN_SME, GLOBAL | GLOBAL |
| day2 | 669 | 36.9 | 40.6 | -3.7 [-4.9, -2.4] | 43.4 | 199 | 26.8 | 16.7 | +10.1 [+7.6, +12.8] | 73 | 0.687 | 0.124 | 0.123 | IN_MAIN, IN_SME, GLOBAL | GLOBAL |
| close | 669 | 36.2 | 40.6 | -4.4 [-5.7, -3.1] | 43.4 | 199 | 25.8 | 16.7 | +9.1 [+6.5, +11.6] | 71 | 0.727 | 0.115 | 0.123 | IN_MAIN, IN_SME, GLOBAL | GLOBAL |
| pre_listing | 669 | 34.3 | 40.6 | -6.3 [-7.8, -4.7] | 43.4 | 199 | 19.2 | 16.7 | +2.4 [+1.3, +3.8] | 71 | 0.727 | 0.115 | 0.123 | IN_MAIN, IN_SME, GLOBAL | GLOBAL |
| ALL | 3345 | 36.8 | 40.6 | -3.8 [-4.3, -3.3] | 43.4 | 995 | 26.6 | 16.7 | +9.9 [+8.5, +11.3] | 73 | 0.705 | 0.118 | 0.123 | n/a | n/a |

## What this model can and cannot tell you

- **Against the historical median, the model did not beat it on the locked blocks.** Per stage and segment, the 2022-2024 flags (95% CI of the p50 minus median difference entirely below zero) passed at: pre_open (IN_SME, GLOBAL); day1 (IN_MAIN, IN_SME, GLOBAL); day2 (IN_MAIN, IN_SME, GLOBAL); close (IN_MAIN, IN_SME, GLOBAL); pre_listing (IN_MAIN, IN_SME, GLOBAL). Pooled over 2022-2024, the p50 is worse than the median at none (see the selection table). On 2025-2026 it is worse at pre_open, day1, day2, better at none, and not clearly different at close, pre_listing.
- **The p10-p90 band is too wide on the locked blocks.** It held 94-95% of locked outcomes (target 80%), against 71-76% on 2023-2024. The conformal widening was fitted on 2022-2024 and does not transfer to the newer blocks; treat the band as a loose guide.
- **The probability of opening above the issue price is only weakly informative.** Its Brier score beats the base rate at close, pre_listing on the locked blocks; the 2022-2024 flags said it did at pre_open, day1, day2, close, pre_listing. AUC on the locked blocks runs from 0.46 to 0.64. Stages without a passing flag are caveated and reported as not reliable.
- **GMP cannot be checked on the locked blocks.** GMP readings stop in 2024 and there are none in 2025-2026. On 2023-2024 the pre-listing p50 was 19.2 pp off on the 199 GMP-bearing IPOs, against 16.7 pp for the GMP reading itself, so GMP remains the stronger signal where it exists.
- **GLOBAL has no open-gain outcome on the locked blocks.** US IPOs from 2024 carry only close-gain targets and OTHER carries none, so the open-gain numbers for US, HK and OTHER come from 2022-2023 only. The close-gain table above includes the 2024-2026 US IPOs.
- **Use it for the size of the typical move at most, not for direction.** The missing context (no open dates for US rows, about half of SME rows) and the missing category subscription (mostly pre-2023 mainboard) limit what it can see at serving time.

## Per-stage skill flags (2022-2024, per segment, saved in the bundle)

| stage | segment | n | model - median MAE (CI) | p50 beats median | Brier - base rate (CI) | prob beats base rate | AUC |
|---|---|---|---|---|---|---|---|
| pre_open | IN_MAIN | 163 | -0.5 [-2.0, +1.0] | no | +0.0 [-0.0, +0.0] | no | 0.481 |
| pre_open | IN_SME | 598 | -1.0 [-1.3, -0.6] | yes | +0.0 [-0.0, +0.0] | no | 0.522 |
| pre_open | GLOBAL | 121 | -1.9 [-2.8, -1.0] | yes | -0.0 [-0.0, -0.0] | yes | 0.503 |
| day1 | IN_MAIN | 163 | -2.9 [-5.2, -0.7] | yes | +0.0 [-0.0, +0.0] | no | 0.582 |
| day1 | IN_SME | 598 | -3.6 [-4.8, -2.4] | yes | +0.0 [-0.0, +0.0] | no | 0.578 |
| day1 | GLOBAL | 121 | -2.0 [-2.9, -1.1] | yes | -0.0 [-0.1, -0.0] | yes | 0.507 |
| day2 | IN_MAIN | 163 | -3.7 [-6.0, -1.4] | yes | +0.0 [-0.0, +0.0] | no | 0.612 |
| day2 | IN_SME | 598 | -3.9 [-5.3, -2.7] | yes | +0.0 [-0.0, +0.0] | no | 0.545 |
| day2 | GLOBAL | 121 | -2.0 [-2.8, -1.1] | yes | -0.0 [-0.1, -0.0] | yes | 0.518 |
| close | IN_MAIN | 163 | -5.6 [-8.1, -3.1] | yes | -0.0 [-0.0, +0.0] | no | 0.684 |
| close | IN_SME | 598 | -4.9 [-6.3, -3.6] | yes | -0.0 [-0.0, +0.0] | no | 0.559 |
| close | GLOBAL | 121 | -2.6 [-3.5, -1.6] | yes | -0.0 [-0.1, -0.0] | yes | 0.561 |
| pre_listing | IN_MAIN | 163 | -7.6 [-10.5, -4.8] | yes | -0.0 [-0.0, +0.0] | no | 0.702 |
| pre_listing | IN_SME | 598 | -7.0 [-8.8, -5.3] | yes | -0.0 [-0.0, +0.0] | no | 0.588 |
| pre_listing | GLOBAL | 121 | -2.6 [-3.5, -1.6] | yes | -0.0 [-0.1, -0.0] | yes | 0.562 |

## Selection blocks 2022-2024, pooled over segments

| stage | n | p50 - median (95% CI), pooled over segments | worse than median | beats median |
|---|---|---|---|---|
| pre_open | 882 | -1.0 [-1.4, -0.6] | no | yes |
| day1 | 882 | -3.2 [-4.1, -2.3] | no | yes |
| day2 | 882 | -3.6 [-4.6, -2.7] | no | yes |
| close | 882 | -4.7 [-5.7, -3.7] | no | yes |
| pre_listing | 882 | -6.5 [-7.9, -5.3] | no | yes |

## Experiments: every option, kept or dropped (earlier design search, 2022-2024)

Selection blocks 2022, 2023, 2024. Every option is listed, kept or dropped. Numbers are pooled over the selection blocks unless stated.

### 1. interval calibration (conformal, CQR)

Criterion: keep if |coverage - 80| shrinks and mean pinball does not rise (2022-2024, all stages). **Kept: on**.

| option | n | MAE p50 | coverage % | mean pinball | Brier | AUC |
|---|---|---|---|---|---|---|
| off | 4830 | 32.3 | 61 | 12.184 | 0.175 | 0.680 |
| on | 4830 | 32.3 | 72 | 11.753 | 0.175 | 0.680 |

### 2. monotone constraints (increasing in log subscription and GMP%; free elsewhere)

Criterion: keep if the classifier's Brier score falls (2022-2024). Applies to the classifier only: LightGBM rejects monotone constraints with the quantile objective. **Kept: monotone**.

| option | n | MAE p50 | coverage % | mean pinball | Brier | AUC |
|---|---|---|---|---|---|---|
| unconstrained | 4830 | 32.3 | 72 | 11.753 | 0.175 | 0.680 |
| monotone | 4830 | 32.3 | 72 | 11.753 | 0.174 | 0.673 |

### 3. GMP-aware pre_listing model (GBM, linear quantile regression, or their average)

Criterion: keep the option with the lowest mean pinball on GMP-bearing rows (2022-2024). **Kept: blend**.

| option | GMP rows | MAE p50 | coverage % | mean pinball |
|---|---|---|---|---|
| gbm | 331 | 23.0 | 87 | 7.635 |
| linear | 331 | 20.8 | 82 | 8.185 |
| blend | 331 | 21.2 | 85 | 7.515 |

### 4. segment ablation (pooled / India-only / one model per segment)

Criterion: per segment, keep the option with the lowest mean pinball (2022-2024, all stages). **Kept: {'IN_MAIN': 'separate', 'IN_SME': 'india', 'GLOBAL': 'pooled'}**.

| segment | routing option | n | MAE p50 | coverage % | mean pinball |
|---|---|---|---|---|---|
| IN_MAIN | pooled | 855 | 21.9 | 81 | 7.350 |
| IN_MAIN | india | 855 | 21.7 | 79 | 7.478 |
| IN_MAIN | separate | 855 | 21.5 | 85 | 7.136 |
| IN_SME | pooled | 3265 | 38.5 | 66 | 14.141 |
| IN_SME | india | 3265 | 38.2 | 65 | 14.078 |
| IN_SME | separate | 3265 | 39.3 | 66 | 14.491 |
| GLOBAL | pooled | 710 | 15.6 | 89 | 6.015 |
| GLOBAL | separate | 710 | 15.8 | 87 | 6.169 |

### 6. band-only price for pre_open (leak fix)

Criterion: always kept: pre_open may use the price band only; recorded before and after. **Kept: always (fix)**.

| option | n | MAE p50 | coverage % | mean pinball | Brier | AUC |
|---|---|---|---|---|---|---|
| before | 4830 | 31.9 | 72 | 11.664 | 0.172 | 0.686 |
| after | 4830 | 32.0 | 72 | 11.752 | 0.171 | 0.685 |

## Close-gain median, locked blocks

| stage | n (close) | MAE close model | MAE close median | model - median (CI) | verdict |
|---|---|---|---|---|---|
| pre_open | 819 | 22.2 | 21.3 | +0.9 [+0.4, +1.4] | worse than it (CI entirely above zero) |
| day1 | 819 | 22.0 | 21.3 | +0.7 [+0.3, +1.2] | worse than it (CI entirely above zero) |
| day2 | 819 | 22.0 | 21.3 | +0.7 [+0.2, +1.1] | worse than it (CI entirely above zero) |
| close | 819 | 20.8 | 21.3 | -0.5 [-1.1, +0.0] | no clear difference (CI spans zero) |
| pre_listing | 819 | 20.8 | 21.3 | -0.5 [-1.0, +0.0] | no clear difference (CI spans zero) |
| ALL | 4095 | 21.5 | 21.3 | +0.3 [+0.0, +0.5] | worse than it (CI entirely above zero) |

## Locked blocks by market

| stage | market | n | MAE p50 | MAE median | p50 - median (CI) | pinball p50 | coverage % | AUC |
|---|---|---|---|---|---|---|---|---|
| pre_open | IN_MAIN | 178 | 18.0 | 15.4 | +2.7 [+1.2, +4.1] | 9.0 | 87 | 0.554 |
| pre_open | IN_SME | 292 | 22.1 | 20.3 | +1.7 [+0.9, +2.6] | 11.0 | 99 | 0.480 |
| day1 | IN_MAIN | 178 | 16.9 | 15.4 | +1.5 [+0.6, +2.4] | 8.4 | 86 | 0.591 |
| day1 | IN_SME | 292 | 21.6 | 20.3 | +1.3 [+0.6, +2.0] | 10.8 | 99 | 0.498 |
| day2 | IN_MAIN | 178 | 17.1 | 15.4 | +1.8 [+0.9, +2.6] | 8.6 | 86 | 0.586 |
| day2 | IN_SME | 292 | 21.5 | 20.3 | +1.2 [+0.6, +1.9] | 10.8 | 99 | 0.530 |
| close | IN_MAIN | 178 | 13.5 | 15.4 | -1.9 [-3.7, -0.0] | 6.7 | 88 | 0.807 |
| close | IN_SME | 292 | 21.4 | 20.3 | +1.1 [+0.5, +1.7] | 10.7 | 99 | 0.500 |
| pre_listing | IN_MAIN | 178 | 13.4 | 15.4 | -2.0 [-3.8, -0.2] | 6.7 | 89 | 0.809 |
| pre_listing | IN_SME | 292 | 21.3 | 20.3 | +1.0 [+0.4, +1.6] | 10.6 | 99 | 0.501 |

## Calibration of P(opens above issue price), locked blocks

### pre_open

| predicted bin | n | mean predicted % | observed opened above % |
|---|---|---|---|
| 0-10% | 0 | n/a | n/a |
| 10-20% | 0 | n/a | n/a |
| 20-30% | 0 | n/a | n/a |
| 30-40% | 0 | n/a | n/a |
| 40-50% | 0 | n/a | n/a |
| 50-60% | 7 | 51.2 | 42.9 |
| 60-70% | 10 | 65.7 | 100.0 |
| 70-80% | 14 | 74.8 | 57.1 |
| 80-90% | 417 | 86.5 | 69.1 |
| 90-100% | 22 | 98.4 | 59.1 |

### day1

| predicted bin | n | mean predicted % | observed opened above % |
|---|---|---|---|
| 0-10% | 0 | n/a | n/a |
| 10-20% | 0 | n/a | n/a |
| 20-30% | 0 | n/a | n/a |
| 30-40% | 0 | n/a | n/a |
| 40-50% | 0 | n/a | n/a |
| 50-60% | 0 | n/a | n/a |
| 60-70% | 31 | 68.8 | 61.3 |
| 70-80% | 85 | 77.2 | 69.4 |
| 80-90% | 342 | 88.7 | 68.7 |
| 90-100% | 12 | 97.3 | 75.0 |

### day2

| predicted bin | n | mean predicted % | observed opened above % |
|---|---|---|---|
| 0-10% | 0 | n/a | n/a |
| 10-20% | 0 | n/a | n/a |
| 20-30% | 0 | n/a | n/a |
| 30-40% | 0 | n/a | n/a |
| 40-50% | 0 | n/a | n/a |
| 50-60% | 0 | n/a | n/a |
| 60-70% | 17 | 65.6 | 58.8 |
| 70-80% | 66 | 77.1 | 60.6 |
| 80-90% | 380 | 87.6 | 70.3 |
| 90-100% | 7 | 92.5 | 71.4 |

### close

| predicted bin | n | mean predicted % | observed opened above % |
|---|---|---|---|
| 0-10% | 0 | n/a | n/a |
| 10-20% | 0 | n/a | n/a |
| 20-30% | 0 | n/a | n/a |
| 30-40% | 16 | 37.6 | 25.0 |
| 40-50% | 12 | 45.0 | 41.7 |
| 50-60% | 1 | 60.0 | 100.0 |
| 60-70% | 18 | 62.8 | 38.9 |
| 70-80% | 62 | 78.9 | 66.1 |
| 80-90% | 260 | 87.9 | 66.9 |
| 90-100% | 101 | 94.4 | 89.1 |

### pre_listing

| predicted bin | n | mean predicted % | observed opened above % |
|---|---|---|---|
| 0-10% | 0 | n/a | n/a |
| 10-20% | 0 | n/a | n/a |
| 20-30% | 0 | n/a | n/a |
| 30-40% | 20 | 34.1 | 20.0 |
| 40-50% | 8 | 47.5 | 62.5 |
| 50-60% | 13 | 56.8 | 38.5 |
| 60-70% | 5 | 70.0 | 40.0 |
| 70-80% | 55 | 79.0 | 63.6 |
| 80-90% | 263 | 87.9 | 67.3 |
| 90-100% | 106 | 93.7 | 88.7 |


## Every block (2022-2026)

| listing year | stage | n | MAE p50 | MAE median | coverage % | AUC | Brier | Brier base rate |
|---|---|---|---|---|---|---|---|---|
| 2022 | pre_open | 213 | 28.5 | 28.5 | 60 | 0.596 | 0.183 | 0.165 |
| 2022 | day1 | 213 | 26.1 | 28.5 | 68 | 0.631 | 0.185 | 0.165 |
| 2022 | day2 | 213 | 25.1 | 28.5 | 69 | 0.702 | 0.163 | 0.165 |
| 2022 | close | 213 | 23.0 | 28.5 | 72 | 0.716 | 0.159 | 0.165 |
| 2022 | pre_listing | 213 | 21.2 | 28.5 | 72 | 0.737 | 0.155 | 0.165 |
| 2023 | pre_open | 327 | 29.7 | 32.1 | 86 | 0.725 | 0.124 | 0.134 |
| 2023 | day1 | 327 | 25.4 | 32.1 | 86 | 0.746 | 0.123 | 0.134 |
| 2023 | day2 | 327 | 25.1 | 32.1 | 84 | 0.713 | 0.130 | 0.134 |
| 2023 | close | 327 | 24.4 | 32.1 | 83 | 0.740 | 0.123 | 0.134 |
| 2023 | pre_listing | 327 | 20.5 | 32.1 | 86 | 0.756 | 0.122 | 0.134 |
| 2024 | pre_open | 342 | 48.5 | 48.8 | 67 | 0.638 | 0.107 | 0.113 |
| 2024 | day1 | 342 | 48.3 | 48.8 | 61 | 0.652 | 0.115 | 0.113 |
| 2024 | day2 | 342 | 48.2 | 48.8 | 61 | 0.647 | 0.119 | 0.113 |
| 2024 | close | 342 | 47.5 | 48.8 | 61 | 0.696 | 0.108 | 0.113 |
| 2024 | pre_listing | 342 | 47.6 | 48.8 | 57 | 0.687 | 0.108 | 0.113 |
| 2025 | pre_open | 340 | 21.8 | 18.9 | 97 | 0.427 | 0.263 | 0.240 |
| 2025 | day1 | 340 | 20.8 | 18.9 | 97 | 0.443 | 0.260 | 0.240 |
| 2025 | day2 | 340 | 20.8 | 18.9 | 97 | 0.466 | 0.257 | 0.240 |
| 2025 | close | 340 | 20.3 | 18.9 | 97 | 0.585 | 0.245 | 0.240 |
| 2025 | pre_listing | 340 | 20.1 | 18.9 | 97 | 0.581 | 0.245 | 0.240 |
| 2026 | pre_open | 130 | 17.3 | 17.3 | 88 | 0.566 | 0.220 | 0.229 |
| 2026 | day1 | 130 | 17.3 | 17.3 | 87 | 0.560 | 0.218 | 0.229 |
| 2026 | day2 | 130 | 17.5 | 17.3 | 88 | 0.673 | 0.218 | 0.229 |
| 2026 | close | 130 | 13.3 | 17.3 | 89 | 0.785 | 0.178 | 0.229 |
| 2026 | pre_listing | 130 | 13.5 | 17.3 | 90 | 0.805 | 0.172 | 0.229 |

## GMP-bearing training rows

| segment | GMP-bearing rows (all years) | before 2025 | rows per listing year |
|---|---|---|---|
| IN_MAIN | 118 | 118 | 2020: 6, 2021: 39, 2022: 23, 2023: 50 |
| IN_SME | 192 | 192 | 2021: 2, 2022: 41, 2023: 147, 2024: 2 |
| GLOBAL | 0 | 0 | none |

## Feature-group importance (final bundle, in-sample gain)

| group (gain %) | pre_open | day1 | day2 | close | pre_listing |
|---|---|---|---|---|---|
| issue | 31.7 | 17.8 | 16.3 | 10.9 | 8.7 |
| financials | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| subscription | 0.0 | 33.6 | 41.9 | 62.9 | 55.1 |
| gmp | 0.0 | 0.0 | 0.0 | 0.0 | 12.2 |
| market_context | 24.4 | 16.5 | 13.3 | 9.1 | 8.3 |
| calendar | 16.9 | 11.1 | 8.4 | 4.3 | 4.5 |
| identity | 26.9 | 21.0 | 20.1 | 12.8 | 11.2 |

## Rows per market

| market | IPOs | open-gain target | close-gain target | GMP reading | QIB subscription | open date known | listing years |
|---|---|---|---|---|---|---|---|
| IN_MAIN | 1029 | 608 | 682 | 156 | 848 | 739 | 2009-2026 |
| IN_SME | 1758 | 1585 | 1560 | 211 | 97 | 825 | 2012-2026 |
| US | 1721 | 1093 | 1257 | 0 | 0 | 0 | 2010-2026 |
| HK | 323 | 283 | 293 | 0 | 0 | 0 | 2020-2024 |
| OTHER | 102 | 0 | 101 | 0 | 0 | 0 | 2022-2023 |

## Data warnings

- Survivorship: the table holds only IPOs that listed; every estimate is optimistic for the average issue.
- Sources joined by date: India and US tables come from different sites; date or price mismatches show up as noise.
- GMP is unofficial grey-market data, taken shortly before listing, and exists only for India from 2020.
- Exact-zero subscription multiples and listing gains are treated as missing placeholders (see the build notes).
- The live feed carries no valuation or financial ratios, so those inputs are missing at serving time.
- 2026 is a partial year.
- Monotone constraints apply to the classifier only: LightGBM rejects them for quantile objectives.
- The release bundle is trained on all IPOs, so its in-sample numbers are not evidence of accuracy.
