# Changelog

## Unreleased

- Licences: MIT for the software, CC BY-NC-SA 4.0 for the data and trained model, and
  third-party material keeps its own terms. Scope in NOTICE.md.

## 0.1.0 - 2026-10-18

- Initial public release.
- Data build fixes: exact-zero listing gains are blanked as placeholders, and exact-zero GMP and subscription multiples (final and day-wise) are blanked too. A missing value stays missing and is never read as zero.
- Model: five-seed release fit (seeds 7 to 11).
- The hyperparameter search (40 trials on 2022-2024) ran, but its settings were not adopted: the confirmation check on 2022-2024 showed the defaults no worse.
- Shrinkage weights are 1.0 at every stage (no blending toward the median).
- Dashboard: the segment's training median is shown, marked "model not used", at any stage where the model is worse than the historical median on the latest blocks.
