# Tier B — Within-Dataset Cross-User Transfer

Tier B evaluates user-independent generalisation **within each wave**: models train on some users and must predict for unseen users from the same wave. This isolates the challenge of person-level heterogeneity from the additional complication of cross-wave dataset shift (Tier C).

---

## Research question

*Do models trained on a subset of users in a wave generalise to users held out from the same wave?*

This is the benchmark's **primary within-dataset transfer** setting.

---

## Split specification

- **Scheme:** Stratified group K-fold cross-validation.
- **K = 5.**
- **Group variable:** `Pcode` (participant identity).
- All samples from a given participant are assigned to a single fold; no participant appears in more than one fold.
- Stratification is performed on the binary label so that HIGH / LOW ratios remain comparable across folds.
- Each wave is evaluated **independently**; results are reported per wave (D1, D2, D3) rather than pooled.

### Eligibility filter

A (wave, task) pair is included only if, after per-fold construction:

1. Each fold has samples from ≥ 2 participants (so LOSO-style degenerate folds are excluded).
2. Each fold has both classes present.
3. The task-specific minimum-sample floor is met.

---

## Tasks covered

- **Shared core (all three waves):** Valence, Arousal, Stress, Task Disturbance.
- **D3 rich affect-word labels (D3 only):** Happy, Relaxed, Cheerful, Content, Sad, Anxious, Depressed, Angry.

---

## Method families evaluated

Tier B compares three families:

1. **Baselines + tabular NNs** — XGBoost, LightGBM, MLP, ResNet, TabNet, SAINT, TabTransformer, FTTransformer, DCN.
2. **Domain generalisation (DG)** — IRM, VREx, GroupDRO, MixStyle, MLDG, MASF, Fish, CSD, SagNet (DomainBed protocols; shared MLP backbone). "Domains" within a wave are groups of users defined by clustering participant-level covariate statistics.
3. **Domain adaptation (DA)** — DANN, CDAN, DAN, DeepCORAL, MCC, ADDA, MCD, JAN, SHOT, CBST, CGDM (TLL protocols; shared MLP backbone). Target = held-out fold's users; adaptation uses only unlabelled target features during training.

---

## Per-user normalisation

Tier B applies **per-user feature normalisation** (z-score within each user) before training to reduce inter-user scale effects and emphasise within-user deviations from each user's own baseline. Normaliser statistics are fit on each user's train-fold samples only.

---

## Evaluation

- Primary metric: **AUROC** across the 5 grouped folds (report mean ± standard deviation).
- Diagnostics: Accuracy, Macro-F1, Precision, Recall.
- Model selection on **fold-level validation AUROC**, Optuna with 30 trials.
- Each wave's results reported separately.

---

## Where the code lives

- **Baselines + tabular NNs** → [`../../basemodel-benchmarking/`](../../basemodel-benchmarking/).
- **DG + DA methods** → [`../../domain_adaptation/`](../../domain_adaptation/).

Summary CSVs are written to [`../results/tier_b_*.csv`](../results/).

---

## Expected results (reference)

From the paper (shared-label family averages across D1 / D2 / D3):

| Family | Mean AUROC |
|---|---|
| Baseline + tabular | 0.566 |
| DG | 0.559 |
| **DA** | **0.587** |

Paper finding: DA (notably MCC, CDAN, JAN) is the strongest family on Tier B shared labels, with the pattern extending to most D3 rich-label tasks.
