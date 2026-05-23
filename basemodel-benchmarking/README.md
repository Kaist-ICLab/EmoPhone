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
├── README.md                # this file
├── benchmark.py             # Setting A + Setting B runner (within-wave)
├── cross_dataset.py         # Setting C runner (cross-wave 1->1 and 2->1)
├── benchmark_logger.py      # shared logger + metric / output contract
├── data_loader.py           # BenchmarkDataset + per-user / temporal split helpers
├── hparams_registry.py      # per-model Optuna search spaces
├── backbones.py             # shared MLP / ResNet / Transformer feature extractors
├── utility.py               # small helpers reused across runners
├── models/                  # per-wrapper baseline + tabular-NN package
│   ├── __init__.py
│   ├── _helpers.py
│   ├── baselines.py         # MLP, ResNet
│   ├── xgb.py               # XGBoost wrapper
│   ├── lgb.py               # LightGBM wrapper
│   ├── tabnet.py            # pytorch_tabnet wrapper
│   ├── widedeep.py          # SAINT, TabTransformer, FTTransformer (pytorch_widedeep)
│   └── deepctr.py           # DCN / AutoInt (deepctr_torch)
└── scripts/                 # shell drivers for full sweeps (gitignored outputs)
    ├── README.md
    ├── run_setting_a.sh     # Setting A sweep
    ├── run_setting_b.sh     # Setting B sweep
    ├── run_setting_c.sh     # Setting C sweep
    └── run_debug.sh         # 1-trial / 1-fold smoke run
```

Run outputs (`results/`, `results/records/`, per-run checkpoints) are gitignored — see the repo-root [`.gitignore`](../.gitignore).

## Running

Single-model invocation from the repo root:

```bash
# Setting B (within-wave cross-user, group-kfold; default split_strategy)
python basemodel-benchmarking/benchmark.py \
    --dataset D-1 --label valence --model XGB \
    --hpo_trials 30 --hpo_mode fold1

# Setting A (per-user temporal split)
python basemodel-benchmarking/benchmark.py \
    --dataset D-1 --label valence --model XGB \
    --split_strategy temporal --hpo_trials 30

# Setting C (cross-wave)
python basemodel-benchmarking/cross_dataset.py \
    --label arousal --model XGB \
    --run_setting one_to_one --hpo_trials 30
```

Full sweeps that orchestrate every model across labels and waves live in [`scripts/`](./scripts/).

Setting-level protocols:

- Setting A: [`../benchmark/setting_a/README.md`](../benchmark/setting_a/README.md)
- Setting B: [`../benchmark/setting_b/README.md`](../benchmark/setting_b/README.md)
- Setting C: [`../benchmark/setting_c/README.md`](../benchmark/setting_c/README.md)

Dependencies are pinned in the repo root [`requirements.txt`](../requirements.txt).

## Outputs

Summary AUROC / Acc / Macro-F1 tables aggregated from the per-run logs are written into [`../benchmark/results/`](../benchmark/results/) using the output contract in [`../benchmark/utils/README.md`](../benchmark/utils/README.md).
