# Domain Generalization & Domain Adaptation

This folder hosts the runnable code for the **domain generalisation (DG)** and **domain adaptation (DA)** baselines used in Tier B and Tier C.

> **Conceptual documentation lives in [`../benchmark/`](../benchmark/).** This folder is the implementation side; the `benchmark/` folder describes the protocol each tier expects.

---

## Scope

Runs in this folder produce results for:

- **Tier B** — within-dataset cross-user transfer, DG and DA families ([`../benchmark/tier_b/README.md`](../benchmark/tier_b/README.md)). "Domains" are user-clusters within a single wave.
- **Tier C** — cross-dataset transfer, DG and DA families ([`../benchmark/tier_c/README.md`](../benchmark/tier_c/README.md)). Source / target domains are the waves themselves (1→1 and 2→1 settings).

Baseline and tabular-NN runs live in [`../basemodel-benchmarking/`](../basemodel-benchmarking/).

---

## Model inventory

All DG and DA methods use a **shared MLP backbone** so that family differences reflect the objective, not the architecture (Appendix C.3.1.3–C.3.1.4).

### Domain generalisation (DG) — DomainBed protocols

| Model | Mechanism |
|---|---|
| IRM | Invariant-classifier constraint across source domains. |
| VREx | Risk-variance penalty across source domains. |
| GroupDRO | Worst-group loss (domains as groups). |
| MixStyle | Mix feature statistics across training domains. |
| MLDG | Meta-learning for domain generalisation. |
| MASF | Task generalisation + feature-space alignment + class-metric learning (ported from the original author release). |
| Fish | Gradient-matching meta-updates across domains. |
| CSD | Common-Specific Decomposition of representation. |
| SagNet | Style-Agnostic Network (content / style separation). |

Upstream reference: [DomainBed](https://github.com/facebookresearch/DomainBed) protocols for search space and evaluation.

### Domain adaptation (DA) — Transfer-Learning-Library (TLL) protocols

| Model | Mechanism |
|---|---|
| DANN | Adversarial domain discriminator. |
| CDAN | Conditional adversarial alignment (features + predictions). |
| DAN | Multi-kernel MMD feature alignment. |
| DeepCORAL | Covariance matching (ported from original release). |
| MCC | Minimum class confusion on target predictions. |
| ADDA | Two-stage adversarial adaptation (ported from original release). |
| MCD | Maximum classifier discrepancy on target (ported from original release). |
| JAN | Joint-distribution MMD (features + predictions). |
| SHOT | Source-free target adaptation (ported from original release). |
| CBST | Class-balanced self-training on pseudo-labels (ported from original release). |
| CGDM | Cross-domain gradient-discrepancy minimisation (ported with MLP-style feature extractor + dual classifiers). |

Upstream reference: [Transfer-Learning-Library (TLL)](https://github.com/thuml/Transfer-Learning-Library) for DA protocols and search-space adaptation.

---

## Folder layout

```
domain_adaptation/
├── README.md    # this file
└── models/      # in-repo DG/DA implementations
```

---

## Running

See the tier-level READMEs for the protocol:

- Tier B: [`../benchmark/tier_b/README.md`](../benchmark/tier_b/README.md)
- Tier C: [`../benchmark/tier_c/README.md`](../benchmark/tier_c/README.md)

Dependencies are pinned in the repo root [`requirements.txt`](../requirements.txt).

## Outputs

Summary tables aggregated from runs are written to [`../benchmark/results/tier_b_dg.csv`](../benchmark/results/), [`../benchmark/results/tier_b_da.csv`](../benchmark/results/), [`../benchmark/results/tier_c_dg.csv`](../benchmark/results/), [`../benchmark/results/tier_c_da.csv`](../benchmark/results/) using the output contract in [`../benchmark/utils/README.md`](../benchmark/utils/README.md).
