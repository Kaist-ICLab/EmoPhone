# Tier C — Cross-Dataset Transfer Across Waves

Tier C is the benchmark's **strictest transfer setting**. Models train on one or more waves and are evaluated on a **held-out wave**, combining the difficulty of unseen users with wave-level distributional shift in study conditions, sensor coverage, and label distributions.

---

## Research question

*Do models trained on one wave (or two waves) generalise to a different wave with different participants, different study conditions, and different sensor coverage?*

This is the benchmark's **primary cross-dataset transfer** setting.

---

## Split specification

**Protocol:** Leave-one-dataset-out. The wave designated as the **target** is held out entirely for evaluation; the remaining wave(s) form the **source**.

Two transfer structures are evaluated:

- **1 → 1 (single-source):** train on wave A, evaluate on wave B. Six ordered pairs total: D1→D2, D1→D3, D2→D1, D2→D3, D3→D1, D3→D2.
- **2 → 1 (multi-source):** train on the two remaining waves jointly, evaluate on the held-out third wave. Three settings total: {D1 ∪ D2} → D3, {D1 ∪ D3} → D2, {D2 ∪ D3} → D1.

A source-only validation split (~20% of the source data, per-user grouped) is reserved for Optuna model selection.

### Eligibility filter

A (source, target, task) tuple is included only if:

1. Both source and target have the task available (→ restricts Tier C to the **shared core labels**: Valence, Arousal, Stress, Disturbance).
2. After alias unification + common-feature intersection, ≥ 500 aligned feature columns remain.
3. Each split has both classes present.

---

## Tasks covered

**Shared core labels only.** D3-only rich affect-word labels are **not** used in Tier C because they cannot be aligned with D1 / D2.

---

## Feature alignment

Tier C applies, in order:

1. **Alias unification** of categorical vocabularies so that equivalent semantic values collapse to the same one-hot column (Korean CALL_CNT labels, `UNKNOWN` vs. `UNDEFINED` battery plug, etc. — see [`../../preprocessing/pipeline_decisions.md`](../../preprocessing/pipeline_decisions.md) § 6).
2. **Common-feature intersection** across source and target schemas: columns not present in every wave of the current setting are dropped.
3. **Split-consistent scaling** fit on the concatenated source training data, applied unchanged to validation and target.

Full details of which feature blocks survive which transfer setting are in [`../../docs/feature_alignment.md`](../../docs/feature_alignment.md).

---

## Method families evaluated

1. **Baselines + tabular NNs** — XGBoost, LightGBM, MLP, ResNet, TabNet, SAINT, TabTransformer, FTTransformer, DCN.
2. **Domain generalisation (DG)** — IRM, VREx, GroupDRO, MixStyle, MLDG, MASF, Fish, CSD, SagNet. Source domains are the individual source waves (or user-clusters within a single source for 1→1).
3. **Domain adaptation (DA)** — DANN, CDAN, DAN, DeepCORAL, MCC, ADDA, MCD, JAN, SHOT, CBST, CGDM. Target = held-out wave; adaptation uses only unlabelled target features during training.

---

## Evaluation

- Primary metric: **AUROC** on the target wave's labelled data.
- Diagnostics: Accuracy, Macro-F1, Precision, Recall.
- Model selection on **source-only validation AUROC**, Optuna with 30 trials.
- Each (source, target, task) tuple reported separately; averaged tables also produced.

---

## Where the code lives

- **Baselines + tabular NNs** → [`../../basemodel-benchmarking/`](../../basemodel-benchmarking/).
- **DG + DA methods** → [`../../domain_adaptation/`](../../domain_adaptation/).

Summary CSVs are written to [`../results/tier_c_*.csv`](../results/).

---

## Expected results (reference)

From the paper (shared-label family averages across all transfer settings):

| Family | Mean AUROC |
|---|---|
| **Baseline + tabular** | **0.550** |
| DG | 0.544 |
| DA | 0.532 |

Paper findings:

- **Single-source (1→1)**: DG is typically strongest.
- **Multi-source (2→1)**: strong baselines (especially gradient-boosted trees) are typically best.
- DA's advantage from Tier B does **not** carry through to Tier C — it is not consistently the strongest family under realistic cross-wave shift.
- Tier C is clearly the hardest of the three tiers; performance drops materially relative to Tier A and Tier B.

Stress is the shared label most sensitive to cross-wave shift; Disturbance is the most stable.
