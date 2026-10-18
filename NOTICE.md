# Licences and notices

This project has two licences. Which one applies depends on the file.

## 1. Software: MIT (see `LICENSE`)

Applies to everything the project authored: the source code under `src/`, the tests under `tests/` (except the
fixtures), `pyproject.toml`, `README.md`, `CHANGELOG.md`, `.gitignore`, and `docs/SPEC.md`, `docs/DATA_SOURCES.md` and `docs/data_sources/`.

## 2. Data and trained model: CC BY-NC-SA 4.0 (see `LICENSE-DATA`)

Applies to material derived from the HuggingFace `sohomghosh/Indian_IPO_datasets` dataset, which is CC BY-NC-SA 4.0,
and so to anything trained or aggregated from it:

- `models/` in full: the trained model `bundle.joblib`, `train_report.json`, `release_config.json`,
  `release_comparison.json`, `experiments.json`, the archived records, and the Kaggle run folders with their
  derived `ipos.parquet` and `indices.parquet`.
- `docs/MODEL_REPORT.md` and `docs/RESEARCH.md`: analyses computed from that data.

Because the source is non-commercial and share-alike, these files may not be used commercially, and derivatives
must carry the same licence. The software licence (MIT) does not grant commercial use of the model.

## 3. Third-party material (not licensed by this project)

`tests/fixtures/` contains excerpts and copies from third-party sources. These keep their original terms, which
this project does not change:

- Kaggle datasets (`tests/fixtures/*/kaggle/`, `tests/fixtures/global/kaggle/`): licences vary; see
  `docs/data_sources/kaggle.md`. Several are unknown.
- The HuggingFace workbooks (`tests/fixtures/india/*.xlsx`): CC BY-NC-SA 4.0.
- Saved web pages and JSON from NSE, Chittorgarh, investorgain, ipowatch, ipopremium, ipocentral and IPO Scoop
  (`tests/fixtures/live/`, `tests/fixtures/gmp/`, `tests/fixtures/global/iposcoop_*`): copyright of the site
  owners. Their terms were not cleared for redistribution.

If a rights holder asks for material to be removed, the project will remove it.

## Not legal advice

This notice is the author's reading of the sources' licences. It is not legal advice. Check with the rights holders
before any commercial use.
