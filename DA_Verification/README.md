# DA Verification

This folder hosts the **diagnostic and upstream-reference code** that supports the domain-adaptation (DA) family used in the EmoPhone benchmark. It is intentionally separate from the runnable benchmark pipeline at the repo root: nothing in here is required to reproduce the headline Setting B / Setting C numbers, but the scripts here are useful when auditing individual DA methods or rerunning the verification studies referenced in the paper appendix.

## What this folder contains

```
DA_Verification/
├── README.md                       # this file
├── CGDM/                           # CGDM-focused verification scripts and notes
│   ├── EXPERIMENT_NOTES.md         # study log for pair / cluster transfer experiments
│   ├── verify_cgdm_ubicomp.py      # canonical CGDM-on-EmoPhone replay
│   ├── verify_cgdm_only.py         # CGDM-only smoke / regression check
│   ├── verify_all_models.py        # full DA model sweep on a single (source, target) pair
│   ├── verify_loso_feature_tsne.py # leave-one-source-out t-SNE diagnostic
│   ├── scan_pair_transfer_candidates.py
│   ├── run_pair_transfer_experiment.py
│   ├── run_cluster_transfer_experiment.py
│   ├── group_users_by_relation_signature.py
│   └── evaluate_incremental_xgb_features.py
├── upstream/                       # placeholder for upstream reference notes
├── upstream_cbst/                  # vendored reference snippets from the CBST upstream
├── upstream_shot/                  # vendored reference snippets from the SHOT upstream
└── upstream_tllib/                 # vendored reference snippets from Transfer-Learning-Library
```

The `upstream_*/` folders mirror small portions of the original third-party releases (filename prefix indicates the upstream path, e.g. `tllib__alignment__dann.py`). They are kept here for **diff / audit** purposes — the actual training code used by the benchmark lives in [`../src/da_models.py`](../src/da_models.py).

## CGDM verification scripts

| Script | Purpose |
|---|---|
| `verify_cgdm_ubicomp.py` | Replays the CGDM training loop on the EmoPhone source / target split with the same hyper-parameters as the main runner. Used to sanity-check that the in-repo CGDM port reproduces upstream behaviour on this dataset. |
| `verify_cgdm_only.py` | Single-pair CGDM smoke check; lighter than `verify_cgdm_ubicomp.py`. |
| `verify_all_models.py` | Trains every DA / DG model in the inventory on a single source→target pair to produce a head-to-head comparison table for the paper appendix. |
| `verify_loso_feature_tsne.py` | Leave-one-source-out training that dumps target-feature t-SNE projections; used to produce the cross-user / cross-wave embedding figures. |
| `scan_pair_transfer_candidates.py` | Enumerates candidate (source-user, target-user) pairs by relation-signature similarity, then runs a small MLP transfer to rank them. |
| `run_pair_transfer_experiment.py` | Pairwise multi-seed transfer experiment driver used to populate the pair-transfer comparison tables. |
| `run_cluster_transfer_experiment.py` | Multi-source cluster transfer experiment driver (uses the user clusters produced by `group_users_by_relation_signature.py`). |
| `group_users_by_relation_signature.py` | Groups participants by relation-signature cosine similarity to form transferable mini-clusters. |
| `evaluate_incremental_xgb_features.py` | XGBoost feature-importance ranking + incremental-feature-set ablation. |

For the narrative behind these scripts and the experiment log (chosen mini-clusters, observed AUROCs, decisions), see [`CGDM/EXPERIMENT_NOTES.md`](./CGDM/EXPERIMENT_NOTES.md).

## Relationship to the main benchmark

- The main runners ([`../execute_benchmark.py`](../execute_benchmark.py), [`../execute_cross_dataset.py`](../execute_cross_dataset.py)) consume DA implementations from [`../src/da_models.py`](../src/da_models.py).
- Verification scripts here import the same `src/` modules, so any change to the DA training loop is automatically reflected in both pipelines.
- Verification outputs (`*.json`, `*.pt`, `*.jpg`, per-pair / per-cluster artefact directories) are **gitignored** to keep the public release lean; rerun the scripts locally if you need to regenerate them.

## Reproducing a verification run

Example — run the full DA model sweep on a single (source, target) pair from the repo root:

```bash
python DA_Verification/CGDM/verify_all_models.py
```

See the top of each script for its CLI / config block.
