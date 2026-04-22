---
name: Align Repo with Paper
overview: Review of `Dataset_Doc_Personalization (1).md` against the current repo, and a concrete, NeurIPS Datasets & Benchmarks Track-oriented plan to bring the repo to release quality. Scope is docs + structural skeleton only (no code migration).
todos:
  - id: readme-rewrite
    content: "Rewrite README.md: remove all [DATASET_NAME]/[TODO] placeholders, fix bibtex + venue, add repo-layout diagram, model inventory, reproducibility pointers, and responsibility/maintenance links"
    status: completed
  - id: license-citation
    content: Add MIT LICENSE full text, LICENSE-DATA.md (CC BY-NC 4.0 pointer), and fill CITATION.cff with author list + DOI placeholder pattern
    status: completed
  - id: datasheet-fill
    content: Resolve placeholders in DATASHEET.md (authors/IRB/funding/contact) or mark [TBD on acceptance]; add ABC Logger reference, per-wave QC and prompt frequency, 8-stage pipeline summary
    status: completed
  - id: data-docs-dedup
    content: "Reconcile data/README.md and data/schema.md: one canonical pkl format, unified label filenames (MentalLoad vs mental, ChangedValence vs valenceChange), corrected column counts (total vs after-PIF)"
    status: completed
  - id: preprocessing-fill
    content: Fill preprocessing/README.md and preprocessing/pipeline_decisions.md with QC thresholds, per-wave exclusion rates, Stress/Disturbance scale normalization, and alias/feature-alignment map lifted from paper Appendix A.3 / C.2 and the FeatureAlignAcrossDatasets section
    status: completed
  - id: benchmark-skeleton
    content: Write benchmark/README.md + per-tier READMEs (tier_a/b/c) + utils/README + results/ templates; document the three-tier ladder, per-tier splits, and full model inventory; link into basemodel-benchmarking/ and domain_adaptation/
    status: completed
  - id: orphan-code-readmes
    content: Add README.md to basemodel-benchmarking/ and domain_adaptation/ describing their role and pointing back to /benchmark/tier_* docs; note TLL/DomainBed provenance
    status: completed
  - id: eda-path-fix
    content: Update EDA/README.md and EDA/utils.py wave path constants from D-1|D-2|D-3 to D1|D2|D3 to match the rest of the repo
    status: completed
  - id: neurips-db-artifacts
    content: Add RESPONSIBILITY.md, MAINTENANCE.md, CHANGELOG.md, CONTRIBUTING.md, AUTHORS.md, and docs/neurips_db_checklist.md covering the D&B track artifact requirements
    status: completed
  - id: docs-folder
    content: Create docs/ with dataset_overview.md, feature_alignment.md (lifted from paper), ethics.md, consent_form_en.md (placeholder referencing supplementary)
    status: completed
  - id: metadata-croissant
    content: Update metadata/croissant.json to fill creator/funder/datePublished/DOI and add metadata/checksums.md5 placeholder; do not wire CI validation here
    status: completed
  - id: gitignore-carveouts
    content: Patch .gitignore to whitelist images/*.png, metadata/checksums.md5, benchmark/results/*.csv|*.json (so documentation assets and result tables are trackable)
    status: completed
  - id: requirements
    content: Add top-level requirements.txt (and optional environment.yml) pinning Python + the libraries referenced by EDA/preprocessing/benchmark (pandas, numpy, scikit-learn, xgboost, lightgbm, optuna, torch, DomainBed-equivalent, transfer-learning-library)
    status: completed
isProject: false
---

# Align Repo with Paper for NeurIPS D&B Release

## 1. What the paper doc actually is

`Dataset_Doc_Personalization (1).md` is a paper-writing scratchpad containing (in this order):
- An overview outline (Sections 1–9)
- A `FeatureAlignAcrossDatasets` report comparing internal `D-2 / D-3 / D-4` pickles (these map to the released **D1 / D2 / D3** — the report is about column differences after `PIF#` removal).
- A `Review4Globem` block with prior GLOBEM reviewer feedback (lessons to internalize, not content to ship).
- A "Draft" block with the full Abstract → Conclusion + References.
- Appendices A (dataset details), B (extended characterization), C (benchmark details & full result tables), plus a Legacy outline.

That means the repo is the **companion artifact** for this paper. The paper defines a three-wave resource (D1, D2, D3; 297 retained / 328 recruited; 53,139 ESM responses) and a three-tier benchmark (A: personal-history, B: within-wave cross-user, C: cross-wave transfer), with a large model inventory (baseline, tabular-NN, DG, DA) and Optuna-tuned unified training.

## 2. Venue clarification

The user mentioned "NIPS journal" — NeurIPS is a conference. The paper intent in the doc is the **NeurIPS 2025/2026 Datasets and Benchmarks Track** (README badge already says "Paper-NeurIPS 2026"). The plan below targets the **D&B track's documentation/artifact requirements**: Gebru datasheet, Croissant metadata, persistent DOI, public license, reproducible benchmarks, author responsibility statement, maintenance plan. Confirm the exact venue when the DOI/URL is known.

## 3. Concrete discrepancies to fix

### 3.1 Hard contradictions between existing repo docs
- **Pickle structure**: [data/README.md](data/README.md) (line ~150) says "pkl files are **5-element tuples** (features, binary labels, pcodes, raw labels, timestamps)". [data/schema.md](data/schema.md) (line 33) says "Each pkl file is a **pandas DataFrame** serialized with pickle." These directly contradict each other. Must pick the real format and normalize everywhere (including the loading example at lines 174–214).
- **Feature-column counts**: [README.md](README.md), [DATASHEET.md](DATASHEET.md) say D1≈8,037 / D2≈10,122 / D3≈10,581. [data/schema.md](data/schema.md) line 43 says "~10,000 for D1; ~20,000 for D2 and D3". The paper's feature-alignment report says 7,961 / 10,102 / 10,550 **after `PIF#` removal** (so the ~8k/10.1k/10.6k numbers are the totals). Schema is wrong.
- **Label filenames**: [README.md](README.md) uses `Attention.pkl, MentalLoad.pkl, ChangedValence.pkl, ChangedArousal.pkl`; [data/README.md](data/README.md) uses `Attention.pkl, mental.pkl, valenceChange.pkl, arousalChange.pkl`. Pick one canonical casing and propagate to README, schema, DATASHEET, croissant, and EDA code.
- **Wave dir naming**: [eda/README.md](EDA/README.md) expects `${DATA_ROOT}/D-1|D-2|D-3` (hyphenated); everywhere else in repo is `D1|D2|D3`. User chose to keep repo style → EDA paths must switch to `D1|D2|D3`.

### 3.2 Missing-content gaps vs. paper
- **LICENSE** is literally `TODO: Add license text.` — blocker for any public release. Paper says **MIT** for code, **CC BY-NC 4.0** for data (CC text lives on Dataverse, but the repo needs the MIT text for the code).
- **CITATION.cff** is all `TODO` — required by Dataverse/GitHub citation widget.
- **Main README** still has `[DATASET_NAME]`, `[TODO: author email]`, `[TODO: IRB name and approval number]`, TODO DOI/URL, and an incorrect bibtex block.
- **DATASHEET.md** placeholders: `[AUTHOR_NAMES]`, `[INSTITUTION]`, `[FUNDING]`, `[IRB_NUMBER]`, `[CONTACT_EMAIL]`, `[IMPACT_ASSESSMENT]`, `[FUTURE_WAVES]`, `[VERSIONING_POLICY]`, `[RELEASE_DATE]`.
- **preprocessing/README.md** and **preprocessing/pipeline_decisions.md** are one-liner TODOs — the paper (Section C.2, Appendix A.3, and the PIF/feature alignment section) has enough material to fill these with QC thresholds (<50% wearable, <50% smartphone, <30% ESM; Polar H10 subgroup rotation; stress/disturbance scale normalization from 0/+6 ↔ −3/+3; alias unification; common-feature intersection for Tier C).
- **benchmark/** subfolders are empty `.gitkeep`s. The real code appears to live in `basemodel-benchmarking/` and `domain_adaptation/models/` at the repo root — these are orphaned from the documented structure and have no READMEs.
- **metadata/croissant.json** is structurally fine but riddled with TODOs (DOI, file IDs, sha256, authors, funder, citation).

### 3.3 NeurIPS D&B Track artifact requirements not yet present
- **Author Responsibility Statement** (explicit "we bear all responsibility in case of violation of rights" text). The paper's Appendix A.9 lists this as a separate section but the repo has no file for it.
- **Maintenance plan** at repo level (DATASHEET has a stub). D&B track expects a versioning + long-term support statement.
- **Reproducibility**: no top-level `requirements.txt` / `environment.yml`, no Python version pin, no seed policy file, no benchmark replication script.
- **Ethics**: consent form translation (Appendix A.1 says "IRB-approved consent form, translated version" is prepared) is not in the repo.
- **Feature alignment** report: the paper's `FeatureAlignAcrossDatasets` section is a key transparency artifact for Tier C. There is no corresponding file in the repo.
- **Benchmark results** (tables C22 / C26–C28 / C30 / C33–C35, and Figures B1–B9) are referenced in the paper but there are no results artifacts (csv/tsv/json) committed alongside code.
- **.gitignore is too aggressive**: lines 27–31 ignore `*.csv`, `*.pkl`, `*.png`, `*.html`, `*.xlsx` globally. This silently hides the `images/table.png` used by the README, any committed benchmark result CSVs, and any figure assets. Needs carve-outs.

## 4. Proposed target repo layout (keeps `docs_plus_skeleton` scope)

```
CrossUserDataset/
├── README.md                       (rewritten, no TODOs)
├── DATASHEET.md                    (all placeholders resolved or clearly deferred with [TBD on acceptance])
├── LICENSE                         (MIT text for code)
├── LICENSE-DATA.md                 (pointer + verbatim CC BY-NC 4.0 DUA summary)
├── CITATION.cff                    (filled)
├── AUTHORS.md                      (new — contributors list)
├── RESPONSIBILITY.md               (new — NeurIPS D&B author responsibility statement)
├── MAINTENANCE.md                  (new — versioning, issue SLA, wave-update policy)
├── CHANGELOG.md                    (new — v1.0.0 entry)
├── CONTRIBUTING.md                 (new — PR / issue guidance)
├── requirements.txt                (new — top-level pinned deps)
├── environment.yml                 (new — optional conda mirror)
├── .gitignore                      (patched; whitelist images/, metadata/, results/)
│
├── docs/                           (new — long-form narrative docs)
│   ├── dataset_overview.md         (paper Sections 3 / 4 condensed)
│   ├── feature_alignment.md        (paper's FeatureAlignAcrossDatasets report)
│   ├── ethics.md                   (paper Section 7.5/7.6 + DATASHEET cross-links)
│   ├── consent_form_en.md          (translated consent form placeholder)
│   └── neurips_db_checklist.md     (track-specific compliance checklist)
│
├── data/                           (unchanged structure; fix contradictions)
│   ├── README.md                   (pick 5-tuple OR DataFrame; fix label filenames; fix col counts)
│   └── schema.md                   (match README on pkl format + column counts)
│
├── preprocessing/                  (fill real content)
│   ├── README.md                   (Zhang 2024 ref + per-wave QC/alignment)
│   ├── pipeline_decisions.md       (QC thresholds, scale normalization, alias map)
│   └── feature_alignment.md        (mirrors docs/feature_alignment.md or symlink)
│
├── benchmark/                      (skeleton only; point to real code)
│   ├── README.md                   (three-tier ladder + model inventory + replication pointers)
│   ├── tier_a/README.md            (what the tier is + link into basemodel-benchmarking/)
│   ├── tier_b/README.md            (ditto + link into domain_adaptation/)
│   ├── tier_c/README.md            (ditto + shared-feature alignment)
│   ├── utils/README.md             (shared loader/metric conventions)
│   └── results/                    (new — committed per-task CSV/JSON summary tables)
│
├── basemodel-benchmarking/         (add README pointing to /benchmark docs)
│   └── README.md
├── domain_adaptation/              (add README pointing to /benchmark docs)
│   └── README.md
│
├── EDA/                            (fix hyphen→non-hyphen wave paths in README + utils)
├── images/                         (committed; add figures referenced by READMEs)
└── metadata/
    ├── croissant.json              (fill authors, funder, DOI when ready; keep TBDs explicit)
    └── checksums.md5               (new — placeholder for Dataverse-provided hashes)
```

## 5. Content decisions that must be made (noted here, not resolved in the plan itself)

- Final dataset name (paper lists: COPE, KAIST-MHLD, GEN-MH, Cross-K, DAMP, CrossUser, CrossShift).
- Canonical pkl format: 5-element tuple vs pandas DataFrame (ground truth is whichever format the code in `basemodel-benchmarking/` / `domain_adaptation/` actually reads).
- Canonical label filenames (e.g., `MentalLoad.pkl` vs `mental.pkl`).
- Final author list, IRB number, contact email, institution, funding.
- Final Dataverse DOI + per-file IDs + sha256s.

## 6. Section-by-section change map

### 6.1 `README.md`
- Replace `[DATASET_NAME]`, `[TODO...]` with chosen name or explicit `[TBD: finalized on acceptance]`.
- Fix bibtex (currently has `note = {NeurIPS 2025 Datasets and Benchmarks Track}` but paper badge says 2026 — align with doc and user confirmation).
- Remove stale commented-out HTML table (lines 29–38) or replace `images/table.png` reference with both a markdown version and the image (since `*.png` is currently gitignored, carve an exception).
- Add a "Benchmark Model Inventory" mini-section summarizing the 20+ models from paper Appendix C.3.
- Add "Reproducibility" section pointing at `requirements.txt`, seed policy, Optuna config (30 trials, validation AUROC selection).
- Add "Repository Layout" diagram matching §4 above.
- Add badges for Croissant validation and DUA link.
- Link to new `RESPONSIBILITY.md`, `MAINTENANCE.md`, `CHANGELOG.md`.

### 6.2 `DATASHEET.md`
- Resolve placeholders or replace with explicit `[TBD on acceptance]` where values are still under embargo.
- Section 2 composition: fix the wearable-loss inconsistency (Polar H10 discontinued in D3 due to participant compliance — paper B.4 says so; DATASHEET currently mentions it more tersely).
- Section 3 collection: add the ABC Logger app reference (paper cites `\cite{Abclogger}`), and the 15 prompts/day (D1) → reduced frequency (D2/D3) detail from paper B.3.
- Section 4 preprocessing: lift the 8-stage pipeline summary and scale-normalization note from paper.
- Section 7 maintenance: fill with content from new `MAINTENANCE.md`.

### 6.3 `data/README.md` and `data/schema.md`
- **Pick one pkl format** and make both files agree. Likely answer: the file is a **DataFrame** where columns include all features + the binary label + metadata cols; the 5-tuple description in README appears to be stale. Verify against actual code before merging.
- Correct column counts (use the paper's authoritative "after PIF# removal" vs "total" numbers; document both).
- Unify label filenames.
- Add a "Feature groups by wave" subsection that pulls the Sensor Feature Alignment matrix (paper Section 3).
- Add note on `keyevent_*` being D2/D3 only, Polar ECG being D1/D2 partial, WiFi similarity being D1/D2 only (paper's alignment report).

### 6.4 `preprocessing/README.md` + `pipeline_decisions.md`
Fill both with real content lifted from paper Appendix A.3, B.1, C.2:
- QC thresholds: wearable coverage ≥ 50%, smartphone coverage ≥ 50%, ESM response rate ≥ 30%.
- Per-wave exclusion rates (D1: 9.8%, D2: 11.6%, D3: 7.0%).
- ESM-level QC: expired prompt removal, >10-min latency removal.
- Scale normalization: Stress/Disturbance D3 (0 to +6) ↔ D1/D2 (−3 to +3).
- Alias unification list (see paper feature-alignment to-dos: CALL_CNT Korean/legacy labels, WLS expansion in D2/D3, Notification_CAT NAVIGATION added, BAT_PLG `UNKNOWN` vs `UNDEFINED`, RSSI bucket absence in D3/D4).
- Known deviations: no fitness data in D1 (paper calls it D-2), WiFi similarity absent in D3-equivalent, Polar H10 partial.

### 6.5 `benchmark/`
- Rewrite `README.md` with the three-tier ladder spec (Section 5) and full model inventory (C.3.1.1–C.3.1.4).
- Add per-tier `README.md`:
  - Tier A: temporal 60/20/20 per user, 30-day truncation, minimum-viability filter, baseline/tabular models only.
  - Tier B: stratified group 5-fold by Pcode, per-user normalization, baseline + DG + DA families.
  - Tier C: leave-one-dataset-out (1→1 and 2→1), common-feature intersection, shared labels only, baseline + DG + DA families.
- Add `utils/README.md` documenting the shared data-loader contract (since real code lives elsewhere).
- Add `results/README.md` + empty CSV/JSON templates for Tier A / B / C summary tables.
- Add cross-references to `basemodel-benchmarking/` and `domain_adaptation/` so reviewers can find the actual code.

### 6.6 Orphan code folders
- `basemodel-benchmarking/README.md`: one-paragraph description + pointer to `/benchmark`.
- `domain_adaptation/README.md`: one-paragraph description + pointer to `/benchmark/tier_b` and `/benchmark/tier_c`, plus TLL/DomainBed provenance (paper C.3.2).

### 6.7 `EDA/`
- Change `utils.py` path constants + `README.md` "Expected wave directories" block from `D-1|D-2|D-3` to `D1|D2|D3` to match the rest of the repo (readonly-safe — we only update strings; notebook outputs don't need rerunning in this plan).

### 6.8 Metadata / licensing / governance
- `LICENSE`: paste full MIT text.
- Add `LICENSE-DATA.md` summarizing CC BY-NC 4.0 + pointing to the Dataverse DUA.
- `CITATION.cff`: fill with the same author list that will land in the paper; include DOI placeholder pattern.
- `metadata/croissant.json`: resolve `creator`, `funder`, `datePublished`, file-level `contentUrl`, `sha256`. Add CI validation later (out of scope here).

### 6.9 NeurIPS D&B-specific artifacts (new files)
- `RESPONSIBILITY.md`: verbatim author responsibility statement per NeurIPS D&B track wording ("The authors bear all responsibility in case of violation of rights…").
- `MAINTENANCE.md`: who maintains, issue response SLA, versioning policy (semver for schema, date-stamped for data revisions), deprecation policy for old waves.
- `CHANGELOG.md`: v1.0.0 entry.
- `docs/neurips_db_checklist.md`: a reviewer-facing checklist pre-answered for every D&B track requirement (datasheet present, Croissant present, license stated, DOI stated, code license stated, author responsibility statement present, ethics review noted, maintenance plan noted, anonymization noted, gated access noted). Useful during rebuttal.

### 6.10 `.gitignore`
- Whitelist specific folders that must be tracked despite the wildcard rules:
  - `!images/*.png`
  - `!metadata/checksums.md5`
  - `!benchmark/results/*.csv`
  - `!benchmark/results/*.json`

## 7. Out of scope for this plan (explicitly deferred)

- Moving or refactoring code under `basemodel-benchmarking/` and `domain_adaptation/` into `/benchmark/tier_*/`.
- Writing new preprocessing or benchmark code.
- Running notebooks / generating figures.
- Validating Croissant JSON in CI.
- Setting up GitHub Actions, issue templates, or PR templates.

All of the above are natural follow-ups under a "full_restructure" scope if/when wanted.
