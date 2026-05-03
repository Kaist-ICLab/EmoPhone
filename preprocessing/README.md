# Preprocessing

This folder documents the preprocessing and quality-control (QC) pipeline applied to the **D1 / D2 / D3** dataset before release. The pipeline itself is the one introduced in Zhang et al. (2024); this folder records the **dataset-specific parameter settings, per-wave deviations, and cross-wave harmonisation decisions** used for the accompanying benchmark.

> **Upstream pipeline reference.**
> Zhang, P., Jung, G., Alikhanov, J., Ahmed, U., & Lee, U. (2024). *A Reproducible Stress Prediction Pipeline with Mobile Sensor Data.* Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 8(3). https://doi.org/10.1145/3678578

---

## Overview

The Zhang et al. (2024) pipeline organises preprocessing and modelling into **eight stages**:

1. **ESM QC** — remove expired and late responses, remove instances that fail minimum-activity checks.
2. **Feature extraction** — derive per-sensor statistical features across fixed time windows (current, 15-min/30-min immediate past, yesterday/today epoch-level daily).
3. **Feature preparation** — merge sensor streams, attach participant-info (`PIF#`) columns, one-hot categorical values.
4. **Feature selection** — filter by variance / missingness / correlation as appropriate.
5. **Data splitting** — temporal / cross-user / cross-dataset (tier-specific).
6. **Resampling** — class-balancing (oversampling / undersampling) only on the training split if needed.
7. **Model training** — unified training loop (Optuna HPO; see `benchmark/`).
8. **Model evaluation** — AUROC primary, with Accuracy / Macro-F1 / Precision / Recall reported.

The pre-extracted `*.pkl` files in the public release capture the output of stages 1–3 (and optionally 4) for each wave × label. The benchmark performs stages 5–8 on top of those files.

For the concrete parameter values used per wave and the known deviations, see [`pipeline_decisions.md`](./pipeline_decisions.md).

For cross-wave feature alignment (Tier C), see [`feature_alignment.md`](./feature_alignment.md) (mirror of [`../docs/feature_alignment.md`](../docs/feature_alignment.md)).

---

## Where the code lives

- **Baseline + tabular-NN runs** (Tier A / Tier B): [`../basemodel-benchmarking/`](../basemodel-benchmarking/).
- **DG + DA runs** (Tier B / Tier C): [`../domain_adaptation/`](../domain_adaptation/).
- **Tier-level documentation**: [`../benchmark/`](../benchmark/).
- **EDA and characterisation**: [`../EDA/`](../EDA/).

The preprocessing step that produces the `*.pkl` files from raw sensor streams is part of the Zhang et al. (2024) repository (linked above). Because raw sensor CSVs are not part of the public release, the public entry point into the pipeline is at the pkl boundary.

---

## Reproducing the benchmark from the released pkls

1. Download `D1.zip`, `D2.zip`, `D3.zip` from Harvard Dataverse (see [`../data/README.md`](../data/README.md)).
2. Unpack into `data/D1`, `data/D2`, `data/D3` under the repo root (or set `DATA_ROOT` to point to your copy).
3. Install dependencies from [`../requirements.txt`](../requirements.txt).
4. Follow the tier-specific instructions in [`../benchmark/tier_a/README.md`](../benchmark/tier_a/README.md), [`../benchmark/tier_b/README.md`](../benchmark/tier_b/README.md), [`../benchmark/tier_c/README.md`](../benchmark/tier_c/README.md).
