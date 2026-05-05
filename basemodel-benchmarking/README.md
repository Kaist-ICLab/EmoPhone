# Baseline & Tabular-NN Benchmarking

This folder hosts the runnable code and outputs for the **baseline** and **tabular neural network** models used across Setting A, Setting B, and Setting C.

> **Conceptual documentation lives in [`../benchmark/`](../benchmark/).** This folder is the implementation side; the `benchmark/` folder describes the protocol each setting expects.

---

## Scope

Runs in this folder produce results for:

- **Setting A** — personal-history temporal prediction ([`../benchmark/setting_a/README.md`](../benchmark/setting_a/README.md))
- **Setting B** — within-dataset cross-user transfer (baseline + tabular-NN family only — DG/DA sit in [`../domain_adaptation/`](../domain_adaptation/)) ([`../benchmark/setting_b/README.md`](../benchmark/setting_b/README.md))
- **Setting C** — cross-dataset transfer (baseline + tabular-NN family only) ([`../benchmark/setting_c/README.md`](../benchmark/setting_c/README.md))

## Model inventory

| Model | Family | Upstream library |
|---|---|---|
| XGBoost | Gradient-boosted trees | [`xgboost`](https://xgboost.readthedocs.io) |
| LightGBM | Gradient-boosted trees (histogram) | [`lightgbm`](https://lightgbm.readthedocs.io) |
| MLP | Feedforward NN | PyTorch (in-repo) |
| ResNet | Residual NN | PyTorch (in-repo) |
| TabNet | Attentive tabular | [`pytorch-tabnet`](https://github.com/dreamquark-ai/tabnet) |
| SAINT | Transformer tabular | [`pytorch-widedeep`](https://github.com/jrzaurin/pytorch-widedeep) |
| TabTransformer | Transformer tabular | [`pytorch-widedeep`](https://github.com/jrzaurin/pytorch-widedeep) |
| FTTransformer | Feature-token transformer | [`pytorch-widedeep`](https://github.com/jrzaurin/pytorch-widedeep) |
| DCN | Cross + deep network | [`deepctr-torch`](https://github.com/shenweichen/DeepCTR-Torch) |

Linear-attention patches are applied to `TabTransformer`, `SAINT`, and `FTTransformer` for high-dimensional tabular inputs; see Appendix C.3.2 of the paper.

## Folder layout

```
basemodel-benchmarking/
├── README.md                         # this file
└── basemodel_benchmark_outputs/      # run outputs (gitignored; see .gitignore)
    ├── runs/                         # per-run checkpoints, logs, HPO studies
    ├── latest/                       # convenience symlinks/copies
    ├── cross_domain_tabpfn/          # cross-wave TabPFN exploration
    ├── cross_domain_xgboost/         # cross-wave XGBoost exploration
    └── feature_importances_xgboost/  # XGBoost feature-importance dumps
```

## Running

See the setting-level READMEs for the protocol:

- Setting A: [`../benchmark/setting_a/README.md`](../benchmark/setting_a/README.md)
- Setting B: [`../benchmark/setting_b/README.md`](../benchmark/setting_b/README.md)
- Setting C: [`../benchmark/setting_c/README.md`](../benchmark/setting_c/README.md)

Dependencies are pinned in the repo root [`requirements.txt`](../requirements.txt).

## Outputs

Summary AUROC / Acc / Macro-F1 tables aggregated from the per-run logs are written into [`../benchmark/results/`](../benchmark/results/) using the output contract in [`../benchmark/utils/README.md`](../benchmark/utils/README.md).
