# Benchmark Results

Committed per-task summary CSVs for every setting. These mirror the tables in Appendix C of the paper so that downstream researchers can compare against our numbers without re-running the full benchmark.

| File | Source paper table | Description |
|---|---|---|
| `tier_a_full.csv` | Table C22 | Full Setting A temporal-prediction matrix across waves and labels. |
| `tier_b_baseline_tabular.csv` | Table C26 | Setting B cross-user results for baseline + tabular NN families. |
| `tier_b_dg.csv` | Table C27 | Setting B cross-user results for DG family. |
| `tier_b_da.csv` | Table C28 | Setting B cross-user results for DA family. |
| `tier_b_category_best.csv` | Table C30 | Setting B category-best comparison across families. |
| `tier_c_baseline_tabular.csv` | Table C33 | Setting C cross-wave results for baseline + tabular NN families. |
| `tier_c_dg.csv` | Table C34 | Setting C cross-wave results for DG family. |
| `tier_c_da.csv` | Table C35 | Setting C cross-wave results for DA family. |

## Schema

Every CSV follows the output contract documented in [`../utils/README.md`](../utils/README.md):

```
tier, wave_or_source, target, task, model, family, n_train, n_val, n_test,
acc, macro_f1, precision, recall, auroc, auroc_std, n_features_after_alignment
```

## Status

These files are **placeholders** until the production run is committed. Per-setting READMEs describe how to regenerate them. The `.csv` templates are intentionally empty (header only) so that downstream tooling can be wired up before the numbers are available.

## Regenerating

- Setting A: see [`../setting_a/README.md`](../setting_a/README.md).
- Setting B: see [`../setting_b/README.md`](../setting_b/README.md).
- Setting C: see [`../setting_c/README.md`](../setting_c/README.md).
