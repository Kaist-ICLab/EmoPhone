# Setting A — Personal-History Predictability

Setting A evaluates whether a user's future affective states can be predicted from their **own earlier observations**. It is the benchmark's personal-history scenario and establishes a lower-difficulty reference point for the harder transfer settings.

---

## Research question

*Given the first portion of a user's own timeline, how well can we predict their later affective states?*

This setting is meant as a learnability check and a target-difficulty comparator — not as the primary transfer benchmark. Strong performance here is necessary but not sufficient for claiming a method generalises.

---

## Split specification

For each participant:

- Order samples **chronologically** by response timestamp.
- Restrict to the first **30 days** of the participant's timeline.
- Partition into **train / validation / test = 60 / 20 / 20** chronologically (no shuffling).
- Concatenate the resulting per-user partitions across users to form pooled train / val / test splits while keeping each user's temporal ordering intact.

### Eligibility filter

A (participant, task) pair is included only if **all** of the following hold after the split:

1. Every split (train, val, test) contains **≥ 1 sample** for that task.
2. Every split contains **both classes** (HIGH and LOW) after binarisation.
3. The task-specific minimum-sample floor (from the pipeline) is met.

Participants or tasks failing any filter are dropped from the Setting A leaderboard for that task.

---

## Tasks covered

- **Shared core (all three waves):** Valence, Arousal, Stress, Task Disturbance.
- **D3 rich affect-word labels (D3 only):** Happy, Relaxed, Cheerful, Content, Sad, Anxious, Depressed, Angry.

Setting A is the only setting where **richer D3 labels are exercised inside Setting A's temporal scenario**, giving a comparison of target difficulty across semantically specific vs. dimensional labels.

---

## Method families evaluated

Setting A uses **standard supervised ML/DL predictors only**:

- Baselines: XGBoost, LightGBM, MLP, ResNet.
- Tabular neural networks: TabNet, SAINT, TabTransformer, FTTransformer, DCN.

DG and DA methods are **not** run on Setting A — by construction the source and target are the same user, so no domain shift is present to align.

---

## Evaluation

- Primary metric: **AUROC** on the per-task pooled test split.
- Diagnostics: Accuracy, Macro-F1, Precision, Recall.
- Model selection on **validation AUROC**, Optuna with 30 trials.

---

## Where the code lives

Runnable scripts for Setting A are in [`../../basemodel-benchmarking/`](../../basemodel-benchmarking/). Outputs are written to `basemodel-benchmarking/basemodel_benchmark_outputs/` and summarised into [`../results/tier_a_full.csv`](../results/) once runs complete.

---

## Expected results (reference)

From the paper (averaged across D1 / D2 / D3 where applicable):

| Label | Best test AUROC (reference) |
|---|---|
| Arousal | 0.609 ± 0.035 |
| Valence | 0.601 ± 0.031 |
| Stress | 0.580 ± 0.021 |
| Disturbance | 0.555 ± 0.005 |
| Angry (D3) | 0.673 |
| Depressed (D3) | 0.548 |

Treat these as benchmark baselines, not pass/fail thresholds; per-task variance is non-trivial.
