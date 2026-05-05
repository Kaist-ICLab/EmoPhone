# Changelog

All notable changes to this repository and its accompanying Harvard Dataverse release are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for its schema and metadata artefacts (see [`MAINTENANCE.md`](./MAINTENANCE.md) § 4).

---

## [Unreleased]

### Added
- Repository scaffolding aligned with NeurIPS Datasets & Benchmarks Track artefact requirements: `LICENSE`, `LICENSE-DATA.md`, `CITATION.cff`, `AUTHORS.md`, `RESPONSIBILITY.md`, `MAINTENANCE.md`, `CHANGELOG.md`, `CONTRIBUTING.md`.
- Three-setting benchmark documentation under [`benchmark/`](./benchmark/), including per-setting READMEs (`setting_a`, `setting_b`, `setting_c`), a shared loader/metric contract (`utils/`), and placeholder result CSVs under `benchmark/results/`.
- Cross-wave feature-alignment report ([`docs/feature_alignment.md`](./docs/feature_alignment.md), mirrored as [`preprocessing/feature_alignment.md`](./preprocessing/feature_alignment.md)).
- Per-wave QC thresholds and alias map in [`preprocessing/pipeline_decisions.md`](./preprocessing/pipeline_decisions.md).
- Long-form docs under [`docs/`](./docs/): `dataset_overview.md`, `ethics.md`, `consent_form_en.md`, `neurips_db_checklist.md`.
- Top-level `requirements.txt` and optional `environment.yml` for reproducibility.
- Orphan-code pointers: READMEs in [`basemodel-benchmarking/`](./basemodel-benchmarking/) and [`domain_adaptation/`](./domain_adaptation/) now cross-reference the `benchmark/` protocol docs.

### Changed
- `README.md` rewritten to remove `[DATASET_NAME]` / `[TODO]` placeholders and to reflect the three-setting benchmark, full model inventory, reproducibility policy, and repository layout.
- `DATASHEET.md` expanded with per-wave QC numbers, ABC Logger reference, 8-stage pipeline summary, and explicit `[TBD on acceptance]` markers for fields held for peer review.
- `data/README.md` and `data/schema.md` reconciled: pkl files are now consistently documented as 5-element tuples (`features, y, groups, t, datetimes`), label filenames are lowercase (`valence.pkl`, `mental.pkl`, `valenceChange.pkl`, …), and feature-column counts are corrected (`8,037 / 10,122 / 10,581` total; `7,961 / 10,102 / 10,550` after `PIF#` removal).
- `EDA/utils.py` and `EDA/README.md` updated to reflect the released `D1 / D2 / D3` folder naming on disk while keeping internal dict keys as `D-1 / D-2 / D-3` for notebook compatibility.
- `.gitignore` patched to whitelist documentation assets (`images/*.png`, `metadata/checksums.md5`, `benchmark/results/*.csv|*.json`).
- License, data-access, and citation metadata now keep code license, data license, DUA terms, authors, and final dataset/paper names explicitly marked as TBD until public release.
- README visuals expanded with preserved study-design, coverage, temporal-label, and embedding figures under [`images/`](./images/).
- EDA notebook directory casing normalised from `eda/` to `EDA/` to match documentation links.

### Deprecated
- The placeholder `[DATASET_NAME]` strings across older drafts are no longer used; the working name is `CrossUserDataset` pending final naming on acceptance.

### Removed
- Stale `.gitkeep` placeholders under `benchmark/setting_a|b|c|utils/`, now replaced by real README files.

### Fixed
- Contradictions between `data/README.md` (5-tuple) and `data/schema.md` (DataFrame). The tuple description matches `EDA.utils.load_and_attach` in the code and is the canonical format.
- Inconsistent label filenames across README and data folder.
- Feature-column counts in the schema (previously "~10,000 for D1; ~20,000 for D2/D3"; now corrected to `8,037 / 10,122 / 10,581` total).

### Security / Privacy
- No data-level changes. Privacy transformations (MD5-hashed contacts, UUID-replaced MAC addresses, GPS longitude displacement) remain as previously documented.

---

## [1.0.0] — TBD on acceptance

Initial public release of the dataset on Harvard Dataverse, accompanying the NeurIPS Datasets and Benchmarks Track paper.

### Added
- D1 archive (Feb–Apr 2020, 92 participants post-QC, 10,259 ESM responses).
- D2 archive (Dec 2020–Jan 2021, 99 participants post-QC, 21,042 ESM responses).
- D3 archive (Nov 2021–Jan 2022, 106 participants post-QC, 21,838 ESM responses).
- Per-wave pre-extracted feature matrices (`{label}.pkl`), `EsmResponse.csv`, `UserInfo.csv`.
- Three-setting benchmark definitions (Setting A / B / C) with fixed splits, eligibility filters, and result templates.
- Full Datasheet for Datasets and Croissant metadata.
