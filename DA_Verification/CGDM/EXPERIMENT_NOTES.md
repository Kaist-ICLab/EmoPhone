# CGDM Step Count Experiment Notes

## 2026-04-08: Near-user transfer around `P008-P041`

### Relation-nearest pair
- Source: `relation_signature_groups_top10.json`
- Highest relation-signature cosine similarity pair: `P008-P041` (`0.9934960603713989`)
- Important: this similarity score is not the same metric as transfer AUROC

### Directional pair-transfer scan
- Script: `scan_pair_transfer_candidates.py`
- Output:
  - `pair_transfer_candidate_scan_p008_p041.csv`
  - `pair_transfer_candidate_scan_p008_p041.json`
- Candidate set scanned: `P008, P041, P094, P131, P019, P067, P105`
- Directional MLP transfer was strongest for:
  - `P094 -> P131`: target AUROC `0.9088`
  - `P008 -> P041`: target AUROC `0.9041`
  - `P094 -> P041`: target AUROC `0.8862`
  - `P019 -> P041`: target AUROC `0.8718`
  - `P041 -> P131`: target AUROC `0.8676`

### Chosen mini-cluster
- Selected users for multi-source test: `P008, P041, P094, P131`
- Rationale:
  - `P041` behaves like a hub user
  - `P094` transfers well to both `P041` and `P131`
  - `P131` is compatible with `P041`
  - `P008-P041` was the original highest-similarity seed pair

## 2026-04-08: 4-user mini-cluster transfer

### Setup
- Script: `run_cluster_transfer_experiment.py`
- Run:
  - `source users = other 3 users`
  - `target user = held-out 1 user`
  - Source split: pooled `80/20 stratified random split`
  - Normalization:
    - source train/val: user-wise `StandardScaler`, fit on each source user's train subset and applied to that user's val subset
    - target: independent user-wise normalization on the full target user
  - Feature selection:
    - source-train/source-val only
    - `XGBoost` ranking
    - top `10` features per target fold
  - Models:
    - `MLP`
    - `CGDM`

### Outputs
- Metrics:
  - `cluster_transfer/P008_P041_P094_P131/metrics.csv`
  - `cluster_transfer/P008_P041_P094_P131/summary.json`
- Filtered pickle:
  - `step_count_cluster_P008_P041_P094_P131.pkl`

### Mean performance across 4 target folds
- `MLP` mean source val AUROC: `0.8789`
- `MLP` mean target AUROC: `0.8809`
- `MLP` mean target ACC: `0.8324`
- `CGDM` mean source val AUROC: `0.8579`
- `CGDM` mean target AUROC: `0.8812`
- `CGDM` mean target ACC: `0.8333`

### Fold-level results
- Target `P008`
  - `MLP`: source val AUROC `0.9258`, target AUROC `0.8122`, target ACC `0.7647`
  - `CGDM`: source val AUROC `0.8621`, target AUROC `0.8170`, target ACC `0.7794`
- Target `P041`
  - `MLP`: source val AUROC `0.8010`, target AUROC `0.9242`, target ACC `0.8619`
  - `CGDM`: source val AUROC `0.7915`, target AUROC `0.9223`, target ACC `0.8564`
- Target `P094`
  - `MLP`: source val AUROC `0.9134`, target AUROC `0.8538`, target ACC `0.8252`
  - `CGDM`: source val AUROC `0.9073`, target AUROC `0.8467`, target ACC `0.8350`
- Target `P131`
  - `MLP`: source val AUROC `0.8753`, target AUROC `0.9332`, target ACC `0.8779`
  - `CGDM`: source val AUROC `0.8708`, target AUROC `0.9387`, target ACC `0.8626`

### Repeatedly selected features
- Present in all 4 target folds:
  - `Heartrate#AVG#ImmediatePast_15`
  - `CAL#AVG#ImmediatePast_15`
- Present in 2 folds:
  - `LOC_DST#AVG#ImmediatePast_15`
  - `LOC_DST#TSC#ImmediatePast_15`
  - `Heartrate#MED#ImmediatePast_15`

### Interpretation
- A small relation-consistent cluster works much better than the earlier pooled-LOSO source setting.
- The 4-user cluster stabilizes source validation AUROC around `0.86-0.93`.
- `CGDM` is not uniformly better than `MLP`, but the average target AUROC is slightly higher.
- `P008` remains the hardest target inside this cluster.
- This 4-user cluster is a reasonable reusable source pool for follow-up t-SNE or adaptation experiments.
