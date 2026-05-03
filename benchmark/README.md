# Benchmark

The CrossUserDataset ships with a **three-tier benchmark framework** that progressively increases in difficulty and evaluates different facets of affect prediction under realistic mobile-sensing conditions.


| Tier  | Setting                             | What it asks                                                                                                                          | Method families                 |
| ----- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| **A** | Personal-history predictability     | Can a user's **own** recent history predict their near-future affect state?                                                           | Baseline + tabular-NN           |
| **B** | Within-dataset cross-user transfer  | Do models trained on some users generalise to **unseen users** from the same wave?                                                    | Baseline + tabular-NN + DG + DA |
| **C** | Cross-dataset transfer across waves | Do models trained on one wave transfer to a **held-out wave** (different study conditions, label distributions, and sensor coverage)? | Baseline + tabular-NN + DG + DA |


Each tier has its own README with the exact split definition, eligibility filters, and replication entry points:

- [tier_a/README.md](./tier_a/README.md) — personal-history predictability (temporal split).
- [tier_b/README.md](./tier_b/README.md) — stratified group 5-fold across users.
- [tier_c/README.md](./tier_c/README.md) — leave-one-dataset-out across waves.
- [utils/README.md](./utils/README.md) — shared loader / metric contract.

---

## Where the runnable code lives

The `benchmark/` folder holds the **contract and documentation** for each tier. The runnable experiment code sits in two top-level folders:

- [../basemodel-benchmarking/](../basemodel-benchmarking/) — baseline models (XGBoost, LightGBM, MLP, ResNet) and tabular neural networks (TabNet, SAINT, TabTransformer, FTTransformer, DCN). Used across Tier A, Tier B, and Tier C.
- [../domain_adaptation/](../domain_adaptation/) — domain-generalisation (DG) and domain-adaptation (DA) baselines. Used in Tier B and Tier C.

Each tier README points into the appropriate subdirectory for replication.

---

## Prediction targets

**Shared core (all three waves):** Valence, Arousal, Stress, Task Disturbance — binarised HIGH / LOW at 0 on the harmonised [−3, +3] scale.

**D3 rich affect-word labels (Tier A and Tier B only, within D3):** Happy, Relaxed, Cheerful, Content, Sad, Anxious, Depressed, Angry — binarised HIGH / LOW at 3 on the 0-to-6 scale.

**Tier C** is restricted to the shared core so that source and target waves share the same label space.

---

## Model inventory

All models share a unified training loop (see [Shared implementation details](#shared-implementation-details) below) so that differences across methods reflect the method itself, not the training procedure.

### Baselines


| Model        | Family                             | Role                                               |
| ------------ | ---------------------------------- | -------------------------------------------------- |
| **XGBoost**  | Gradient-boosted trees             | Primary tree baseline for high-dim tabular inputs. |
| **LightGBM** | Gradient-boosted trees (histogram) | Second tree baseline; complements XGBoost.         |
| **MLP**      | Feedforward NN                     | Simplest neural baseline for fixed tabular inputs. |
| **ResNet**   | Residual NN                        | Tests whether residual blocks help over plain MLP. |


### Tabular neural networks


| Model              | Mechanism                                                       |
| ------------------ | --------------------------------------------------------------- |
| **TabNet**         | Attentive feature-selection masks over multiple decision steps. |
| **SAINT**          | Transformer-style tabular model with richer attention.          |
| **TabTransformer** | Transformer layers applied to embedded tabular inputs.          |
| **FTTransformer**  | Feature-token transformer with lighter tokenisation.            |
| **DCN**            | Explicit cross layers + deep nonlinear tower.                   |


### Domain generalisation (DG, DomainBed protocols; shared MLP backbone)


| Model        | Mechanism                                                              |
| ------------ | ---------------------------------------------------------------------- |
| **IRM**      | Invariant-classifier constraint across source domains.                 |
| **VREx**     | Risk-variance penalty across source domains.                           |
| **GroupDRO** | Worst-group loss (domains as groups).                                  |
| **MixStyle** | Mix feature statistics across training domains.                        |
| **MLDG**     | Meta-learning for domain generalisation.                               |
| **MASF**     | Task generalisation + feature-space alignment + class-metric learning. |
| **Fish**     | Gradient-matching meta-updates across domains.                         |
| **CSD**      | Common-Specific Decomposition of representation.                       |
| **SagNet**   | Style-Agnostic Network (content / style separation).                   |


### Domain adaptation (DA, TLL protocols; shared MLP backbone)


| Model         | Mechanism                                              |
| ------------- | ------------------------------------------------------ |
| **DANN**      | Adversarial domain discriminator (standard baseline).  |
| **CDAN**      | Conditional adversarial alignment on task predictions. |
| **DAN**       | Multi-kernel MMD feature alignment.                    |
| **DeepCORAL** | Covariance matching between source / target features.  |
| **MCC**       | Minimum class confusion on target predictions.         |
| **ADDA**      | Two-stage adversarial adaptation.                      |
| **MCD**       | Maximum classifier discrepancy on target.              |
| **JAN**       | Joint-distribution MMD (features + predictions).       |
| **SHOT**      | Source-free target adaptation.                         |
| **CBST**      | Class-balanced self-training on pseudo-labels.         |
| **CGDM**      | Cross-domain gradient-discrepancy minimisation.        |


For upstream provenance of each family, see [../domain_adaptation/README.md](../domain_adaptation/README.md) and [../basemodel-benchmarking/README.md](../basemodel-benchmarking/README.md).

---

## Shared implementation details

All tiers follow a unified three-stage evaluation pipeline:

1. **Split construction.** Train / validation / test splits are built per tier (see per-tier READMEs). Splits are **fixed across all compared methods**.
2. **Hyperparameter tuning with Optuna.** 30 trials per (model, task, tier). Selection on **validation AUROC**. Training uses only train + validation; test is held out.
3. **Evaluation on the predefined test split.** Report Accuracy, Macro-F1, Precision, Recall, and AUROC (primary).

**Training loop (all model families):**

- Max 50 epochs with patience-based early stopping.
- Fixed global seed for deterministic comparison.
- Split-consistent preprocessing: normaliser fit on train only, applied to val/test unchanged.
- Leakage-prone fields (`Pcode`, timestamps, raw label) removed before modelling.

**Metric primary:** AUROC. Chosen because (a) many affect tasks have imbalanced classes, and (b) it is threshold-independent.

---

## Replicating the benchmark

1. Download the data (see [../data/README.md](../data/README.md)) and unpack into `data/D1/`, `data/D2/`, `data/D3/`.
2. Install dependencies: `pip install -r ../requirements.txt`.
3. Pick a tier and run its entry script (per-tier READMEs).
4. Summary results will be written into [results/](./results/) (or the tier's own output folder) and aggregated by the scripts referenced in each tier README.

---

## Results

Per-task summary CSVs (AUROC, Accuracy, Macro-F1, Precision, Recall) are committed to [results/](./results/) once populated. They mirror the tables reported in Appendix C of the paper:

- Tier A → `results/tier_a_full.csv`
- Tier B → `results/tier_b_baseline_tabular.csv`, `results/tier_b_dg.csv`, `results/tier_b_da.csv`, `results/tier_b_category_best.csv`
- Tier C → `results/tier_c_baseline_tabular.csv`, `results/tier_c_dg.csv`, `results/tier_c_da.csv`
