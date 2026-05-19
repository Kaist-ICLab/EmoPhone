# Orchestration Scripts

Shell drivers that orchestrate full or partial sweeps over the EmoPhone three-setting benchmark. They invoke the top-level Python runners ([`../benchmark.py`](../benchmark.py), [`../cross_dataset.py`](../cross_dataset.py)) with the per-setting hyper-parameter settings documented in [`../benchmark/`](../benchmark/).

Run every script from the **repo root** so that relative paths to `data/`, `src/`, and `results/` resolve correctly:

```bash
bash basemodel-benchmarking/scripts/<script>.sh
```

| Script | Setting | Scope |
|---|---|---|
| `run_benchmark_resume.sh` | Setting A + Setting B | Full within-wave sweep: baselines + tabular NNs + DG + DA across `D-1 / D-2 / D-3` and every label available for the wave. Resume-aware (reads the progress CSV and skips completed `(model, dataset, label)` tuples). HPO 30 trials, fold1 mode. |
| `run_benchmark_temporal_resume.sh` | Setting A | Per-user temporal split sweep with first-30-days windowing, 60/20/20 chronological partitioning, and label-diversity filter. Resume-aware. |
| `run_cross_dataset.sh` | Setting C | Full cross-wave sweep over the six 1→1 and three 2→1 leave-one-dataset-out directions on the shared core labels. Resume-aware. |
| `run_debug.sh` | Setting A / Setting B / Setting C | Quick local smoke run (1 HPO trial, 1 fold, ~3 epochs). Usage: `bash basemodel-benchmarking/scripts/run_debug.sh <dataset> <label>`. |

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
