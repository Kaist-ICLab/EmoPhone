# Orchestration Scripts

Shell drivers that orchestrate full or partial sweeps over the EmoPhone three-setting benchmark. They invoke the top-level Python runners ([`../benchmark.py`](../benchmark.py), [`../cross_dataset.py`](../cross_dataset.py)) with the per-setting hyper-parameter settings documented in [`../benchmark/`](../benchmark/).

Run every script from the **repo root** so that relative paths to `data/`, `src/`, and `results/` resolve correctly:

```bash
bash basemodel-benchmarking/scripts/<script>.sh
```

| Script | Setting | Scope |
|---|---|---|
| `run_benchmark_all.sh` | Setting A + Setting B | Full within-wave sweep: baselines + tabular NNs + DG + DA across `D-1 / D-2 / D-3` and every label that exists for that wave. HPO 30 trials, fold1 mode. |
| `run_benchmark_resume.sh` | Setting A + Setting B | Resume-aware version of `run_benchmark_all.sh` — reads the progress CSV and skips completed (model, dataset, label) tuples. |
| `run_benchmark_temporal_resume.sh` | Setting A | Per-user temporal split sweep with first-30-days windowing, 60/20/20 chronological partitioning, label-diversity filter. |
| `run_cross_dataset.sh` | Setting C | Resume-aware full cross-wave sweep covering 1→1 and 2→1 leave-one-dataset-out settings across the shared core labels. |
| `run_cross_dataset_resume.sh` | Setting C | Thin wrapper that calls `run_cross_dataset.sh` with `--resume`-friendly flags. |
| `run_cross_dataset_debug.sh` | Setting C | Fast smoke driver: small HPO trial budget, few folds, low epoch override; intended for code-path verification, not final numbers. |
| `run_fast_cross_dataset.sh` | Setting C | Reduced-budget sweep for CGDM (all labels) and JAN / FTTransformer on `disturbance`, `stress`, `valence`. |
| `run_fast_cross_dataset_extra.sh` | Setting C | Reduced-budget sweep for the remaining DA family on `stress_binary` (DeepCORAL, MCC, MCD, SHOT, CBST). |
| `run_fast_cross_dataset_tabular_stress.sh` | Setting C | Reduced-budget sweep for the tabular-NN family on `stress_binary` (FTTransformer is skipped because it is already covered by `run_fast_cross_dataset.sh`). |
| `run_kfold_transformers_only.sh` | Setting B | Transformer-only Setting B sweep (TabTransformer, SAINT, FTTransformer); useful when reproducing only the tabular-Transformer rows of the leaderboard. |
| `run_debug.sh` | Setting A / Setting B | Quick local sanity run: 1 HPO trial, 1 fold, ~3 epochs. Usage: `bash scripts/run_debug.sh <dataset> <label>`. |

## Output locations

Each script writes:

- Row-level results: `results/*.csv` (per-setting filename baked into the driver).
- Per-experiment metadata: `results/records/*.json` (one file per run).
- Aggregated benchmark tables: `benchmark/results/*.csv` (after the post-hoc aggregator is run; see [`../benchmark/utils/README.md`](../benchmark/utils/README.md)).

## Setting mapping

For the protocol behind each setting, see:

- Setting A: [`../benchmark/setting_a/README.md`](../benchmark/setting_a/README.md)
- Setting B: [`../benchmark/setting_b/README.md`](../benchmark/setting_b/README.md)
- Setting C: [`../benchmark/setting_c/README.md`](../benchmark/setting_c/README.md)
- Shared loader / metric contract: [`../benchmark/utils/README.md`](../benchmark/utils/README.md)
