# Benchmark Results

Committed per-task summary CSVs for every tier. These mirror the tables in Appendix C of the paper so that downstream researchers can compare against our numbers without re-running the full benchmark.

| File | Source paper table | Description |
|---|---|---|
| `tier_a_full.csv` | Table C22 | Full Tier A temporal-prediction matrix across waves and labels. |
| `tier_b_baseline_tabular.csv` | Table C26 | Tier B cross-user results for baseline + tabular NN families. |
| `tier_b_dg.csv` | Table C27 | Tier B cross-user results for DG family. |
| `tier_b_da.csv` | Table C28 | Tier B cross-user results for DA family. |
| `tier_b_category_best.csv` | Table C30 | Tier B category-best comparison across families. |
| `tier_c_baseline_tabular.csv` | Table C33 | Tier C cross-wave results for baseline + tabular NN families. |
| `tier_c_dg.csv` | Table C34 | Tier C cross-wave results for DG family. |
| `tier_c_da.csv` | Table C35 | Tier C cross-wave results for DA family. |

## Schema

Every CSV follows the output contract documented in [`../utils/README.md`](../utils/README.md):

```
tier, wave_or_source, target, task, model, family, n_train, n_val, n_test,
acc, macro_f1, precision, recall, auroc, auroc_std, n_features_after_alignment
```

## Status

These files are **placeholders** until the production run is committed. Per-tier READMEs describe how to regenerate them. The `.csv` templates are intentionally empty (header only) so that downstream tooling can be wired up before the numbers are available.

## Regenerating

- Tier A: see [`../tier_a/README.md`](../tier_a/README.md).
- Tier B: see [`../tier_b/README.md`](../tier_b/README.md).
- Tier C: see [`../tier_c/README.md`](../tier_c/README.md).
