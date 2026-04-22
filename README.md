# CrossUserDataset (D1–D3)

> A three-wave in-the-wild smartphone and wearable sensing dataset for moment-level affect modeling in everyday life.

[![License: MIT (code)](https://img.shields.io/badge/Code-MIT-blue)](./LICENSE)
[![License: CC BY-NC 4.0 (data)](https://img.shields.io/badge/Data-CC%20BY--NC%204.0-blue)](./LICENSE-DATA.md)
[![DOI](https://img.shields.io/badge/DOI-TBD%20on%20acceptance-lightgrey)](#citation)
[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202026%20D%26B-green)](#citation)
[![Datasheet](https://img.shields.io/badge/Docs-Datasheet-orange)](./DATASHEET.md)
[![Croissant](https://img.shields.io/badge/Metadata-Croissant-7f52b5)](./metadata/croissant.json)
[![Harvard Dataverse](https://img.shields.io/badge/Data-Harvard%20Dataverse-red)](#data-access)

> **Dataset name and DOI are finalised on acceptance.** This repository uses the working name `CrossUserDataset`; the published dataset name will be substituted throughout before release. See the paper's proposed names for context (e.g., COPE, Cross-K, DAMP).

---

## Overview

This repository contains the documentation, metadata, preprocessing notes, and benchmark scaffolding for the **D1, D2, and D3** dataset — three consecutive annual waves of an in-the-wild affective sensing study. Each wave pairs passive sensor data from participants' own Android smartphones and Fitbit wearables with dense in-situ affective state labels collected via the Experience Sampling Method (ESM).

Together, the three waves provide **53,139 ESM responses** from **297 participants** (328 recruited; 9.8% / 11.6% / 7.0% excluded by QC in D1 / D2 / D3), collected between **2020-02-07 and 2022-01-11** in South Korea.

**The dataset is designed to support:**
- Moment-level affect prediction (valence, arousal, stress, task disturbance)
- Within-person (personalized) vs. cross-person generalization benchmarks
- Cross-dataset (cross-wave) transfer-learning experiments
- Longitudinal and individual-difference analyses in mobile affective computing

---

## Dataset at a Glance

| Property | D1 | D2 | D3 |
|---|---|---|---|
| **Collection period** | Feb 7 – Apr 2, 2020 (~30 d) | Dec 7, 2020 – Jan 27, 2021 (~30 d) | Nov 23, 2021 – Jan 11, 2022 (~28 d) |
| **Recruited / retained (QC)** | 102 / 92 | 112 / 99 | 114 / 106 |
| **ESM responses (post-QC)** | 10,259 | 21,042 | 21,838 |
| **Mean responses / participant** | 111.5 (SD 51.1) | 212.5 (SD 44.1) | 206.0 (SD 24.7) |
| **Smartphone** | Android ≥ 7.0 | Android ≥ 8.0 | Android ≥ 8.0 |
| **Wearable** | Fitbit Inspire HR + Polar H10 (sub-period) | Fitbit Inspire HR + Polar H10 (sub-period) | Fitbit Inspire HR only |
| **Feature columns (total)** | 8,037 | 10,122 | 10,581 |
| **Shared labels** | Valence, Arousal, Stress, Disturbance | Valence, Arousal, Stress, Disturbance | Valence, Arousal, Stress, Disturbance |
| **Wave-specific labels** | Attention, Mental, Duration, Change | Attention, Mental, Duration, ValenceChange, ArousalChange | 8 PANAS-style words (Happy, Relaxed, Cheerful, Content, Sad, Anxious, Depressed, Angry) |

![Summary of Dataset Characteristics](images/table.png)

**Shared core labels** (7-point scale, −3 to +3; Stress/Disturbance are 0 to +6 in D3 and normalised to −3/+3 in the benchmark):
Valence, Arousal, Stress, Task Disturbance.

**D3 affect-word labels** (0 to +6): Happy, Relaxed, Cheerful, Content, Sad, Anxious, Depressed, Angry.

See [`data/schema.md`](./data/schema.md) for the full label reference and the column-naming convention.

---

## Benchmark Ladder

| Tier | Setting | Evaluation | Method families |
|---|---|---|---|
| **Tier A** | Personal-history predictability | 60/20/20 chronological split per user (first 30 days), concatenated across users | Baseline + tabular-NN |
| **Tier B** | Within-dataset cross-user transfer | Stratified group 5-fold by `Pcode`, evaluated per wave | Baseline + tabular-NN + DG + DA |
| **Tier C** | Cross-dataset transfer across waves | Leave-one-dataset-out (1→1 and 2→1), shared labels only, common-feature intersection | Baseline + tabular-NN + DG + DA |

All tiers use **AUROC** as the primary metric with Accuracy / Macro-F1 / Precision / Recall reported for diagnostics. Hyperparameters are tuned with **Optuna (30 trials, validation-AUROC selection)**; training uses a unified loop (≤ 50 epochs, patience-based early stopping, fixed seed).

See [`benchmark/README.md`](./benchmark/README.md) for tier-level details and reproduction pointers.

### Benchmark Model Inventory

**Baselines**: XGBoost, LightGBM, MLP, ResNet.
**Tabular neural networks**: TabNet, SAINT, TabTransformer, FTTransformer, DCN.
**Domain generalization (DG)**: IRM, VREx, GroupDRO, MixStyle, MLDG, MASF, Fish, CSD, SagNet.
**Domain adaptation (DA)**: DANN, CDAN, DAN, DeepCORAL, MCC, ADDA, MCD, JAN, SHOT, CBST, CGDM.

DG and DA models share an MLP backbone so that family differences reflect the objective, not the backbone. DG methods follow DomainBed protocols; DA methods follow the Transfer-Learning-Library (TLL); tabular NNs follow their respective upstream repositories.

---

## Preprocessing Pipeline

All three waves were processed using the reproducible mobile-sensing pipeline introduced in:

> Zhang, P., Jung, G., Alikhanov, J., Ahmed, U., & Lee, U. (2024). **A Reproducible Stress Prediction Pipeline with Mobile Sensor Data.** *Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.* 8(3). https://doi.org/10.1145/3678578

Per-wave QC thresholds, feature-alignment decisions, and scale-normalisation rules are documented in [`preprocessing/pipeline_decisions.md`](./preprocessing/pipeline_decisions.md) and [`docs/feature_alignment.md`](./docs/feature_alignment.md).

---

## Repository Layout

```
CrossUserDataset/
│
├── README.md                         ← this file
├── DATASHEET.md                      ← Gebru-style datasheet (NeurIPS D&B requirement)
├── LICENSE                           ← MIT for code
├── LICENSE-DATA.md                   ← CC BY-NC 4.0 summary + link to Dataverse DUA
├── CITATION.cff                      ← machine-readable citation
├── AUTHORS.md                        ← contributors
├── RESPONSIBILITY.md                 ← NeurIPS D&B author responsibility statement
├── MAINTENANCE.md                    ← hosting, versioning, SLA
├── CHANGELOG.md                      ← release history
├── CONTRIBUTING.md                   ← PR / issue guidance
├── requirements.txt                  ← top-level pinned Python deps
├── environment.yml                   ← optional conda mirror
│
├── docs/                             ← long-form docs
│   ├── dataset_overview.md
│   ├── feature_alignment.md          ← cross-wave feature schema & alias map
│   ├── ethics.md
│   ├── consent_form_en.md            ← translated consent form (placeholder)
│   └── neurips_db_checklist.md       ← reviewer-facing compliance checklist
│
├── data/
│   ├── README.md                     ← Dataverse access + how to load files
│   └── schema.md                     ← column-by-column reference
│
├── preprocessing/
│   ├── README.md                     ← pipeline overview & Zhang 2024 reference
│   ├── pipeline_decisions.md         ← per-wave QC, scale normalisation, alias map
│   └── feature_alignment.md          ← mirrors docs/feature_alignment.md
│
├── benchmark/
│   ├── README.md                     ← three-tier ladder + model inventory
│   ├── tier_a/README.md              ← personal-history predictability
│   ├── tier_b/README.md              ← within-wave cross-user transfer
│   ├── tier_c/README.md              ← cross-wave transfer
│   ├── utils/README.md               ← shared loader / metric contract
│   └── results/                      ← committed per-task CSV/JSON summaries
│
├── basemodel-benchmarking/           ← Tier A/B/C baseline + tabular-NN runs (code + outputs)
├── domain_adaptation/                ← Tier B/C DG + DA runs (code)
│
├── EDA/                              ← dataset-characterisation notebooks
├── images/                           ← figures referenced by READMEs
└── metadata/
    ├── croissant.json                ← ML Commons Croissant metadata
    └── checksums.md5                 ← per-archive SHA/MD5 (populated on release)
```

---

## Data Access

The dataset is hosted on **Harvard Dataverse** with gated access. Users must log in, agree to the Data Use Agreement (DUA), and then download. No manual approval step: agreement to terms grants immediate access.

**To download:**
1. Visit the dataset page at **[TBD Harvard Dataverse URL — populated on acceptance]**.
2. Log in or create a free Harvard Dataverse account.
3. Read and agree to the Data Use Agreement.
4. Download individual wave archives (`D1.zip`, `D2.zip`, `D3.zip`) or the full dataset.

**Download via the Dataverse API** (after agreeing to terms on the website):

```bash
# D1 as an example
curl -L "https://dataverse.harvard.edu/api/access/datafile/TBD_FILE_ID_D1" \
     -H "X-Dataverse-key: YOUR_API_TOKEN" \
     -o D1.zip
```

**Python equivalent**:

```python
import requests

api_token = "YOUR_API_TOKEN"  # Dataverse account → API Token
file_id   = "TBD_FILE_ID_D1"  # from the Dataverse dataset page

r = requests.get(
    f"https://dataverse.harvard.edu/api/access/datafile/{file_id}",
    headers={"X-Dataverse-key": api_token},
)
with open("D1.zip", "wb") as f:
    f.write(r.content)
```

See [`data/README.md`](./data/README.md) for the full data-folder structure and loading examples.

**Verify integrity after download:**
```bash
md5sum -c metadata/checksums.md5
```

---

## Reproducibility

- **Python**: 3.10 recommended; 3.9–3.11 supported.
- **Install**: `pip install -r requirements.txt` (or `conda env create -f environment.yml`).
- **Seeds**: all benchmark runs use a fixed seed (documented per tier in [`benchmark/tier_*/README.md`](./benchmark/)).
- **HPO**: Optuna with **30 trials**, validation-AUROC selection, tier-specific validation split.
- **Training loop**: maximum 50 epochs, patience-based early stopping, unified across baseline / DG / DA families.
- **Split definitions**, preprocessing policy, and model-selection rule are held fixed across families within each tier.

---

## Citation

```bibtex
@inproceedings{crossuser2026,
  title     = {[Dataset Title — TBD on acceptance]},
  author    = {[Author list — TBD on acceptance]},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  year      = {2026},
  doi       = {TBD},
  url       = {TBD}
}

@article{zhang2024reproducible,
  title     = {A Reproducible Stress Prediction Pipeline with Mobile Sensor Data},
  author    = {Zhang, Panyu and Jung, Gyuwon and Alikhanov, Jumabek and Ahmed, Uzair and Lee, Uichin},
  journal   = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
  volume    = {8},
  number    = {3},
  year      = {2024},
  doi       = {10.1145/3678578}
}
```

See [`CITATION.cff`](./CITATION.cff) for the machine-readable form.

---

## License and Ethics

- **Code** in this repository is released under the [MIT License](./LICENSE).
- **Data** on Harvard Dataverse is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); see [`LICENSE-DATA.md`](./LICENSE-DATA.md) and the Dataverse DUA.

All data collection was approved by the Institutional Review Board at **[TBD institution]** (approval number **[TBD]**). Participants provided written informed consent and received ~100 USD compensation. Anonymisation applied prior to release: MD5-hashed contact numbers, UUID-replaced Wi-Fi/Bluetooth MAC addresses, per-participant random displacement of GPS longitude. See [`docs/ethics.md`](./docs/ethics.md) and [`DATASHEET.md`](./DATASHEET.md) § 2.

The authors accept full responsibility for any rights violations arising from this release; see [`RESPONSIBILITY.md`](./RESPONSIBILITY.md).

Maintenance plan and versioning policy: [`MAINTENANCE.md`](./MAINTENANCE.md). Change history: [`CHANGELOG.md`](./CHANGELOG.md).

---

## Contact

For questions about the dataset or code, contact **[TBD contact email — populated on acceptance]** or open a GitHub issue. Contribution guidance: [`CONTRIBUTING.md`](./CONTRIBUTING.md).
