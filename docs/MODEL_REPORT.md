# Model report: walk-forward evaluation

Generated 2026-10-18 11:38 UTC. Data hash 1e4485a4, seed 7. Design: `{"conformal": true, "monotone": true, "gmp_model": "blend", "routing": {"IN_MAIN": "IN_MAIN", "IN_SME": "india", "GLOBAL": "pooled"}}`.

## Data change: placeholder zeros blanked

After the locked blocks 2025-2026 were first evaluated, the build started blanking exact-zero listing gains and GMP readings as placeholders, because sources write 0 where no value exists. Cells blanked: open 389, high 45, low 75, close 128, gmp_last 50, GMP % 50. The locked blocks were evaluated again on the changed table, so they are no longer blind to it. The earlier report is kept at models/superseded-zero-fix/MODEL_REPORT.before-zero-fix.md.

| stage | open rows | MAE p50 | MAE median | p50 - median (95% CI) | coverage % | AUC |
|---|---|---|---|---|---|---|
| pre_open | 521 to 470 | 18.6 to 20.7 | 17.1 to 18.4 | +1.5 [+0.9, +2.0] to +2.3 [+1.5, +3.0] | 95 to 95 | 0.487 to 0.478 |
| day1 | 521 to 470 | 18.3 to 20.1 | 17.1 to 18.4 | +1.2 [+0.7, +1.7] to +1.6 [+1.0, +2.2] | 94 to 94 | 0.486 to 0.472 |
| day2 | 521 to 470 | 18.1 to 19.8 | 17.1 to 18.4 | +1.0 [+0.6, +1.4] to +1.3 [+0.8, +1.9] | 93 to 94 | 0.477 to 0.483 |
| close | 521 to 470 | 16.8 to 18.4 | 17.1 to 18.4 | -0.3 [-1.0, +0.4] to -0.0 [-0.9, +0.8] | 95 to 96 | 0.629 to 0.642 |
| pre_listing | 521 to 470 | 16.8 to 18.4 | 17.1 to 18.4 | -0.3 [-1.0, +0.4] to -0.0 [-0.8, +0.7] | 95 to 96 | 0.624 to 0.633 |


## Protocol

- Choices (conformal, monotone, GMP model, segment routing) were made on walk-forward blocks 2022-2024 only. Blocks 2025-2026 are locked: evaluated once for the chosen design. Skill flags use 2022-2024. Serving caveats use the latest blocks (2025-2026) for the 'worse than the historical median' warning only. This warns more, never less; it does not change the model or any choice.
- Selection blocks: listing years 2022, 2023, 2024. Locked blocks: 2025, 2026. Every block is predicted by models trained only on IPOs listed before it.
- 2025-2026 were seen once in the earlier report (before the protocol), and again after the placeholder blanking (see Data change). Nothing was chosen from them, but they are no longer blind.
- The design above is fixed in code (model/design.py) before this report is generated; the locked numbers below are the one evaluation of that design.

## Headline: locked blocks 2025-2026

Units: percentage points (pp) of the issue price. 'p50 - median' is the model's MAE minus the MAE of the segment's training median, on the same IPOs; a negative value is better. Intervals are 95% bootstrap intervals over IPOs. Skill flags come from 2022-2024 only.

| stage | n | MAE p50 | MAE median | p50 - median (95% CI) | MAE zero | n with GMP | MAE model (GMP rows) | MAE GMP | model - GMP (CI) | coverage p10-p90 % | AUC | Brier | Brier base rate | 2022-24 skill: p50 beats median (segments) | 2022-24 skill: prob beats base (segments) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pre_open | 470 | 20.7 | 18.4 | +2.3 [+1.5, +3.0] | 19.3 | 0 | n/a | n/a | n/a | 95 | 0.478 | 0.249 | 0.237 | IN_SME, GLOBAL | none |
| day1 | 470 | 20.1 | 18.4 | +1.6 [+1.0, +2.2] | 19.3 | 0 | n/a | n/a | n/a | 94 | 0.472 | 0.250 | 0.237 | IN_MAIN, IN_SME, GLOBAL | GLOBAL |
| day2 | 470 | 19.8 | 18.4 | +1.3 [+0.8, +1.9] | 19.3 | 0 | n/a | n/a | n/a | 94 | 0.483 | 0.248 | 0.237 | IN_MAIN, IN_SME, GLOBAL | GLOBAL |
| close | 470 | 18.4 | 18.4 | -0.0 [-0.9, +0.8] | 19.3 | 0 | n/a | n/a | n/a | 96 | 0.642 | 0.229 | 0.237 | IN_MAIN, IN_SME, GLOBAL | GLOBAL |
| pre_listing | 470 | 18.4 | 18.4 | -0.0 [-0.8, +0.7] | 19.3 | 0 | n/a | n/a | n/a | 96 | 0.633 | 0.226 | 0.237 | IN_MAIN, IN_SME, GLOBAL | none |
| ALL | 2350 | 19.5 | 18.4 | +1.0 [+0.7, +1.3] | 19.3 | 0 | n/a | n/a | n/a | 95 | 0.551 | 0.240 | 0.237 | n/a | n/a |

## Same measures, 2023-2024 (selection era, shown for comparison)

| stage | n | MAE p50 | MAE median | p50 - median (95% CI) | MAE zero | n with GMP | MAE model (GMP rows) | MAE GMP | model - GMP (CI) | coverage p10-p90 % | AUC | Brier | Brier base rate | 2022-24 skill: p50 beats median (segments) | 2022-24 skill: prob beats base (segments) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pre_open | 669 | 39.2 | 40.6 | -1.4 [-1.9, -0.8] | 43.4 | 199 | 34.2 | 16.7 | +17.5 [+13.2, +21.8] | 76 | 0.693 | 0.116 | 0.123 | IN_SME, GLOBAL | none |
| day1 | 669 | 37.0 | 40.6 | -3.6 [-4.7, -2.5] | 43.4 | 199 | 27.0 | 16.7 | +10.3 [+7.5, +13.3] | 74 | 0.713 | 0.117 | 0.123 | IN_MAIN, IN_SME, GLOBAL | GLOBAL |
| day2 | 669 | 37.0 | 40.6 | -3.6 [-4.8, -2.3] | 43.4 | 199 | 27.1 | 16.7 | +10.4 [+7.8, +13.2] | 72 | 0.706 | 0.122 | 0.123 | IN_MAIN, IN_SME, GLOBAL | GLOBAL |
| close | 669 | 36.2 | 40.6 | -4.4 [-5.7, -3.1] | 43.4 | 199 | 25.8 | 16.7 | +9.1 [+6.6, +11.6] | 71 | 0.722 | 0.116 | 0.123 | IN_MAIN, IN_SME, GLOBAL | GLOBAL |
| pre_listing | 669 | 34.4 | 40.6 | -6.2 [-7.8, -4.7] | 43.4 | 199 | 19.1 | 16.7 | +2.4 [+1.2, +3.6] | 72 | 0.722 | 0.117 | 0.123 | IN_MAIN, IN_SME, GLOBAL | none |
| ALL | 3345 | 36.8 | 40.6 | -3.8 [-4.3, -3.3] | 43.4 | 995 | 26.6 | 16.7 | +9.9 [+8.5, +11.3] | 73 | 0.707 | 0.118 | 0.123 | n/a | n/a |

## What this model can and cannot tell you

- **Against the historical median, the model did not beat it on the locked blocks.** Per stage and segment, the 2022-2024 flags (95% CI of the p50 minus median difference entirely below zero) passed at: pre_open (IN_SME, GLOBAL); day1 (IN_MAIN, IN_SME, GLOBAL); day2 (IN_MAIN, IN_SME, GLOBAL); close (IN_MAIN, IN_SME, GLOBAL); pre_listing (IN_MAIN, IN_SME, GLOBAL). Pooled over 2022-2024, the p50 is worse than the median at none (see the selection table). On 2025-2026 it is worse at pre_open, day1, day2, better at none, and not clearly different at close, pre_listing.
- **The p10-p90 band is too wide on the locked blocks.** It held 94-96% of locked outcomes (target 80%), against 71-76% on 2023-2024. The conformal widening was fitted on 2022-2024 and does not transfer to the newer blocks; treat the band as a loose guide.
- **The probability of opening above the issue price is only weakly informative.** Its Brier score beats the base rate at close, pre_listing on the locked blocks; the 2022-2024 flags said it did at day1, day2, close. AUC on the locked blocks runs from 0.47 to 0.64. Stages without a passing flag are caveated and reported as not reliable.
- **GMP cannot be checked on the locked blocks.** GMP readings stop in 2024 and there are none in 2025-2026. On 2023-2024 the pre-listing p50 was 19.1 pp off on the 199 GMP-bearing IPOs, against 16.7 pp for the GMP reading itself, so GMP remains the stronger signal where it exists.
- **GLOBAL has no open-gain outcome on the locked blocks.** US IPOs from 2024 carry only close-gain targets and OTHER carries none, so the open-gain numbers for US, HK and OTHER come from 2022-2023 only. The close-gain table above includes the 2024-2026 US IPOs.
- **Use it for the size of the typical move at most, not for direction.** The missing context (no open dates for US rows, about half of SME rows) and the missing category subscription (mostly pre-2023 mainboard) limit what it can see at serving time.

## Selection blocks 2022-2024, pooled over segments

| stage | n | p50 - median (95% CI), pooled over segments | worse than median | beats median |
|---|---|---|---|---|
| pre_open | 882 | -1.0 [-1.4, -0.6] | no | yes |
| day1 | 882 | -3.2 [-4.1, -2.4] | no | yes |
| day2 | 882 | -3.6 [-4.6, -2.6] | no | yes |
| close | 882 | -4.7 [-5.8, -3.7] | no | yes |
| pre_listing | 882 | -6.5 [-7.8, -5.3] | no | yes |

## Per-stage skill flags (2022-2024, saved in the bundle)

| stage | segment | n | model - median MAE (CI) | p50 beats median | Brier - base rate (CI) | prob beats base rate | AUC |
|---|---|---|---|---|---|---|---|
| pre_open | IN_MAIN | 163 | -0.6 [-2.3, +1.0] | no | +0.0 [-0.0, +0.0] | no | 0.472 |
| pre_open | IN_SME | 598 | -1.0 [-1.3, -0.6] | yes | -0.0 [-0.0, +0.0] | no | 0.523 |
| pre_open | GLOBAL | 121 | -1.9 [-2.8, -0.9] | yes | -0.0 [-0.0, +0.0] | no | 0.489 |
| day1 | IN_MAIN | 163 | -3.0 [-5.2, -0.7] | yes | +0.0 [-0.0, +0.0] | no | 0.578 |
| day1 | IN_SME | 598 | -3.5 [-4.7, -2.4] | yes | -0.0 [-0.0, +0.0] | no | 0.564 |
| day1 | GLOBAL | 121 | -2.0 [-2.9, -1.1] | yes | -0.0 [-0.1, -0.0] | yes | 0.509 |
| day2 | IN_MAIN | 163 | -3.9 [-6.2, -1.5] | yes | +0.0 [-0.0, +0.0] | no | 0.643 |
| day2 | IN_SME | 598 | -3.8 [-5.2, -2.6] | yes | +0.0 [-0.0, +0.0] | no | 0.568 |
| day2 | GLOBAL | 121 | -2.0 [-2.9, -1.1] | yes | -0.0 [-0.1, -0.0] | yes | 0.513 |
| close | IN_MAIN | 163 | -5.8 [-8.5, -3.1] | yes | -0.0 [-0.0, +0.0] | no | 0.705 |
| close | IN_SME | 598 | -4.8 [-6.3, -3.5] | yes | -0.0 [-0.0, +0.0] | no | 0.568 |
| close | GLOBAL | 121 | -2.6 [-3.6, -1.6] | yes | -0.0 [-0.1, -0.0] | yes | 0.534 |
| pre_listing | IN_MAIN | 163 | -7.6 [-10.5, -4.8] | yes | -0.0 [-0.0, +0.0] | no | 0.698 |
| pre_listing | IN_SME | 598 | -7.0 [-8.8, -5.3] | yes | -0.0 [-0.0, +0.0] | no | 0.588 |
| pre_listing | GLOBAL | 121 | -2.5 [-3.5, -1.6] | yes | -0.0 [-0.1, +0.0] | no | 0.496 |

## Experiments: every option, kept or dropped (2022-2024)

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
| day1 | 819 | 22.2 | 21.3 | +0.9 [+0.5, +1.4] | worse than it (CI entirely above zero) |
| day2 | 819 | 22.0 | 21.3 | +0.7 [+0.3, +1.1] | worse than it (CI entirely above zero) |
| close | 819 | 20.8 | 21.3 | -0.5 [-1.0, +0.1] | no clear difference (CI spans zero) |
| pre_listing | 819 | 20.8 | 21.3 | -0.5 [-1.0, +0.0] | no clear difference (CI spans zero) |
| ALL | 4095 | 21.6 | 21.3 | +0.3 [+0.1, +0.5] | worse than it (CI entirely above zero) |

## Locked blocks by market

| stage | market | n | MAE p50 | MAE median | p50 - median (CI) | pinball p50 | coverage % | AUC |
|---|---|---|---|---|---|---|---|---|
| pre_open | IN_MAIN | 178 | 18.2 | 15.4 | +2.8 [+1.3, +4.3] | 9.1 | 87 | 0.599 |
| pre_open | IN_SME | 292 | 22.3 | 20.3 | +2.0 [+1.1, +2.9] | 11.1 | 99 | 0.491 |
| day1 | IN_MAIN | 178 | 17.1 | 15.4 | +1.7 [+0.8, +2.7] | 8.6 | 84 | 0.581 |
| day1 | IN_SME | 292 | 21.9 | 20.3 | +1.5 [+0.8, +2.3] | 10.9 | 99 | 0.492 |
| day2 | IN_MAIN | 178 | 16.9 | 15.4 | +1.6 [+0.7, +2.4] | 8.5 | 84 | 0.576 |
| day2 | IN_SME | 292 | 21.5 | 20.3 | +1.2 [+0.5, +1.9] | 10.7 | 99 | 0.504 |
| close | IN_MAIN | 178 | 13.6 | 15.4 | -1.8 [-3.6, +0.2] | 6.8 | 90 | 0.785 |
| close | IN_SME | 292 | 21.3 | 20.3 | +1.0 [+0.4, +1.6] | 10.7 | 99 | 0.499 |
| pre_listing | IN_MAIN | 178 | 13.5 | 15.4 | -1.9 [-3.6, -0.1] | 6.7 | 89 | 0.791 |
| pre_listing | IN_SME | 292 | 21.4 | 20.3 | +1.1 [+0.5, +1.7] | 10.7 | 99 | 0.502 |

## Calibration of P(opens above issue price), locked blocks

### pre_open

| predicted bin | n | mean predicted % | observed opened above % |
|---|---|---|---|
| 0-10% | 0 | n/a | n/a |
| 10-20% | 0 | n/a | n/a |
| 20-30% | 0 | n/a | n/a |
| 30-40% | 0 | n/a | n/a |
| 40-50% | 0 | n/a | n/a |
| 50-60% | 0 | n/a | n/a |
| 60-70% | 19 | 67.7 | 68.4 |
| 70-80% | 22 | 73.6 | 59.1 |
| 80-90% | 409 | 86.9 | 68.9 |
| 90-100% | 20 | 99.1 | 70.0 |

### day1

| predicted bin | n | mean predicted % | observed opened above % |
|---|---|---|---|
| 0-10% | 0 | n/a | n/a |
| 10-20% | 0 | n/a | n/a |
| 20-30% | 0 | n/a | n/a |
| 30-40% | 0 | n/a | n/a |
| 40-50% | 0 | n/a | n/a |
| 50-60% | 0 | n/a | n/a |
| 60-70% | 47 | 68.2 | 68.1 |
| 70-80% | 59 | 77.0 | 67.8 |
| 80-90% | 288 | 87.7 | 70.5 |
| 90-100% | 76 | 93.2 | 61.8 |

### day2

| predicted bin | n | mean predicted % | observed opened above % |
|---|---|---|---|
| 0-10% | 0 | n/a | n/a |
| 10-20% | 0 | n/a | n/a |
| 20-30% | 0 | n/a | n/a |
| 30-40% | 0 | n/a | n/a |
| 40-50% | 0 | n/a | n/a |
| 50-60% | 0 | n/a | n/a |
| 60-70% | 16 | 66.7 | 62.5 |
| 70-80% | 78 | 78.1 | 61.5 |
| 80-90% | 366 | 87.8 | 69.9 |
| 90-100% | 10 | 91.9 | 80.0 |

### close

| predicted bin | n | mean predicted % | observed opened above % |
|---|---|---|---|
| 0-10% | 0 | n/a | n/a |
| 10-20% | 0 | n/a | n/a |
| 20-30% | 0 | n/a | n/a |
| 30-40% | 13 | 37.7 | 23.1 |
| 40-50% | 17 | 47.4 | 47.1 |
| 50-60% | 0 | n/a | n/a |
| 60-70% | 23 | 62.7 | 47.8 |
| 70-80% | 62 | 79.4 | 67.7 |
| 80-90% | 253 | 88.8 | 65.2 |
| 90-100% | 102 | 94.5 | 91.2 |

### pre_listing

| predicted bin | n | mean predicted % | observed opened above % |
|---|---|---|---|
| 0-10% | 0 | n/a | n/a |
| 10-20% | 0 | n/a | n/a |
| 20-30% | 1 | 25.0 | 100.0 |
| 30-40% | 18 | 33.3 | 16.7 |
| 40-50% | 11 | 47.4 | 54.5 |
| 50-60% | 1 | 58.3 | 0.0 |
| 60-70% | 19 | 64.5 | 47.4 |
| 70-80% | 53 | 79.2 | 66.0 |
| 80-90% | 272 | 87.9 | 67.3 |
| 90-100% | 95 | 94.6 | 89.5 |

## Every block (2022-2026)

| listing year | stage | n | MAE p50 | MAE median | coverage % | AUC | Brier | Brier base rate |
|---|---|---|---|---|---|---|---|---|
| 2022 | pre_open | 213 | 28.5 | 28.5 | 62 | 0.600 | 0.182 | 0.165 |
| 2022 | day1 | 213 | 26.4 | 28.5 | 70 | 0.577 | 0.173 | 0.165 |
| 2022 | day2 | 213 | 25.0 | 28.5 | 70 | 0.617 | 0.167 | 0.165 |
| 2022 | close | 213 | 22.8 | 28.5 | 72 | 0.729 | 0.156 | 0.165 |
| 2022 | pre_listing | 213 | 21.3 | 28.5 | 72 | 0.742 | 0.155 | 0.165 |
| 2023 | pre_open | 327 | 29.8 | 32.1 | 87 | 0.719 | 0.124 | 0.134 |
| 2023 | day1 | 327 | 25.3 | 32.1 | 86 | 0.743 | 0.123 | 0.134 |
| 2023 | day2 | 327 | 25.3 | 32.1 | 83 | 0.726 | 0.128 | 0.134 |
| 2023 | close | 327 | 24.4 | 32.1 | 83 | 0.741 | 0.127 | 0.134 |
| 2023 | pre_listing | 327 | 20.4 | 32.1 | 86 | 0.749 | 0.125 | 0.134 |
| 2024 | pre_open | 342 | 48.3 | 48.8 | 67 | 0.663 | 0.109 | 0.113 |
| 2024 | day1 | 342 | 48.3 | 48.8 | 62 | 0.665 | 0.111 | 0.113 |
| 2024 | day2 | 342 | 48.2 | 48.8 | 62 | 0.669 | 0.116 | 0.113 |
| 2024 | close | 342 | 47.5 | 48.8 | 61 | 0.692 | 0.107 | 0.113 |
| 2024 | pre_listing | 342 | 47.7 | 48.8 | 59 | 0.674 | 0.109 | 0.113 |
| 2025 | pre_open | 340 | 21.9 | 18.9 | 97 | 0.451 | 0.260 | 0.240 |
| 2025 | day1 | 340 | 21.1 | 18.9 | 96 | 0.440 | 0.262 | 0.240 |
| 2025 | day2 | 340 | 20.6 | 18.9 | 96 | 0.449 | 0.258 | 0.240 |
| 2025 | close | 340 | 20.3 | 18.9 | 96 | 0.585 | 0.248 | 0.240 |
| 2025 | pre_listing | 340 | 20.3 | 18.9 | 97 | 0.564 | 0.247 | 0.240 |
| 2026 | pre_open | 130 | 17.5 | 17.3 | 88 | 0.558 | 0.221 | 0.229 |
| 2026 | day1 | 130 | 17.4 | 17.3 | 87 | 0.545 | 0.220 | 0.229 |
| 2026 | day2 | 130 | 17.5 | 17.3 | 87 | 0.655 | 0.221 | 0.229 |
| 2026 | close | 130 | 13.4 | 17.3 | 94 | 0.754 | 0.181 | 0.229 |
| 2026 | pre_listing | 130 | 13.5 | 17.3 | 91 | 0.766 | 0.170 | 0.229 |

## GMP-bearing training rows

| segment | GMP-bearing rows (all years) | before 2025 | rows per listing year |
|---|---|---|---|
| IN_MAIN | 118 | 118 | 2020: 6, 2021: 39, 2022: 23, 2023: 50 |
| IN_SME | 192 | 192 | 2021: 2, 2022: 41, 2023: 147, 2024: 2 |
| GLOBAL | 0 | 0 | none |

## Feature-group importance (last locked block, pooled model)

Gain share of the median model; permutation is the increase in open-gain MAE (pp) when the group's inputs are shuffled on the block.

| group (gain %) | pre_open | day1 | day2 | close | pre_listing |
|---|---|---|---|---|---|
| issue | 32.2 | 17.5 | 16.2 | 10.4 | 9.2 |
| financials | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| subscription | 0.0 | 35.8 | 44.0 | 61.5 | 51.7 |
| gmp | 0.0 | 0.0 | 0.0 | 0.0 | 13.4 |
| market_context | 22.8 | 16.3 | 13.1 | 9.2 | 9.1 |
| calendar | 17.7 | 9.8 | 7.8 | 5.0 | 4.4 |
| identity | 27.4 | 20.6 | 18.9 | 13.9 | 12.2 |

| group (permutation, pp) | pre_open | day1 | day2 | close | pre_listing |
|---|---|---|---|---|---|
| issue | +0.36 | -0.01 | +0.14 | +0.04 | +0.12 |
| financials | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| subscription | n/a | +0.00 | +0.00 | +5.42 | +5.74 |
| gmp | n/a | n/a | n/a | n/a | +0.00 |
| market_context | +0.11 | -0.27 | -0.31 | +0.07 | -0.19 |
| calendar | +1.63 | +1.10 | +0.98 | +0.81 | +0.61 |
| identity | -0.00 | -0.02 | -0.02 | -0.01 | -0.01 |

## Rows per market

| market | IPOs | open-gain target | close-gain target | GMP reading | QIB subscription | open date known | listing years |
|---|---|---|---|---|---|---|---|
| IN_MAIN | 1029 | 608 | 682 | 156 | 863 | 739 | 2009-2026 |
| IN_SME | 1758 | 1585 | 1560 | 211 | 97 | 825 | 2012-2026 |
| US | 1721 | 1093 | 1257 | 0 | 0 | 0 | 2010-2026 |
| HK | 323 | 283 | 293 | 0 | 0 | 0 | 2020-2024 |
| OTHER | 102 | 0 | 101 | 0 | 0 | 0 | 2022-2023 |

## Data warnings

- Survivorship: the table holds only IPOs that listed; every estimate is optimistic for the average issue.
- Sources joined by date: India and US tables come from different sites; date or price mismatches show up as noise.
- GMP is unofficial grey-market data, taken shortly before listing, and exists only for India from 2020.
- The live feed carries no valuation or financial ratios, so those inputs are missing at serving time.
- The walk-forward blocks are years of listing; 2026 is a partial year.
- Monotone constraints apply to the classifier only: LightGBM rejects them for quantile objectives.
