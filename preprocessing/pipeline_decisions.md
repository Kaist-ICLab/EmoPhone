# Pipeline Decisions

This file records the concrete QC parameters, harmonisation rules, and per-wave deviations applied during preprocessing. It pairs with [`README.md`](./README.md) (pipeline overview) and [`feature_alignment.md`](./feature_alignment.md) (cross-wave schema alignment).

---

## 1. ESM-level quality control

Applied to `EsmResponse.csv` before feature extraction:

- **Expired prompts removed.** ESM prompts were delivered during waking hours (10:00–22:00) and expired after 10 minutes; unanswered prompts are dropped.
- **Late submissions removed.** Any response submitted more than 10 minutes after the trigger time is dropped (same rule as expiry).
- **Prompt schedule.** D1 used ~15 prompts/day. D2 and D3 used a reduced per-day rate with a participant-facing progress counter and improved monitoring, resulting in tighter distributions of per-participant response counts.

---

## 2. Participant-level quality control

A participant is retained only if **all** of the following coverage thresholds are met across the wave:

- **Wearable coverage ≥ 50%** of eligible waking hours.
- **Smartphone coverage ≥ 50%** of eligible waking hours.
- **ESM response rate ≥ 30%** of delivered prompts.

Per-wave outcomes:

| Wave | Recruited | Retained | Excluded | Exclusion rate |
|---|---|---|---|---|
| D1 | 102 | 92  | 10 | 9.8% |
| D2 | 112 | 99  | 13 | 11.6% |
| D3 | 114 | 106 |  8 | 7.0% |
| **Total** | **328** | **297** | **31** | — |

Excluded participants are not present in the released `UserInfo.csv`, `EsmResponse.csv`, or any `*.pkl`.

---

## 3. Sensor QC

- **Polar H10 ECG** is available only for participants during their assigned sub-period in D1 and D2. ECG-derived features therefore have missing rows outside those sub-periods. D3 does not include ECG at all.
- **Keystroke events** (`keyevent_*`) were added from D2 onward. D1 has no keystroke features.
- **Fitness-derived features** (`FDI`, `FST`, `FCL`, `FAC`) are produced only for D2 and D3; D1 has no fitness block.
- **Wi-Fi similarity** (`WIFI_COS|JAC|EUC|MAN`) is present in D1 and D2; skipped in D3 preprocessing.
- **Bluetooth feature set** differs by wave:
  - D1 → `BT_RSSI` bucketed signal-strength features (113 columns).
  - D2 / D3 → `BT_classType` and `BT_DeviceType` categorical encodings.

These differences are documented empirically in [`feature_alignment.md`](./feature_alignment.md) and handled automatically by the Tier C loader via common-feature intersection (see [`../benchmark/tier_c/README.md`](../benchmark/tier_c/README.md)).

---

## 4. Label harmonisation

### 4.1 Stress and Disturbance scale

The ESM scale differs between waves:

| Wave | Stress | Disturbance |
|---|---|---|
| D1 | −3 to +3 | −3 to +3 |
| D2 | −3 to +3 | −3 to +3 |
| D3 | 0 to +6 | 0 to +6 |

For cross-wave tasks (Tier C), D3 Stress and Disturbance are shifted by −3 so that all waves share the [−3, +3] range. The binary label in `pkl[1]` is already computed on the harmonised scale; the raw-value array `pkl[3]` preserves each wave's original scale and must be shifted manually if used.

The EDA helper `normalize_label_series` in [`../EDA/utils.py`](../EDA/utils.py) applies the same shift automatically:

```python
def normalize_label_series(series):
    # shift [0, 6] to [-3, +3]
    smin, smax = series.min(), series.max()
    if smin >= 0 and smax > 3:
        return series - 3
    return series
```

### 4.2 Binary labelling convention

Each raw ESM label is binarised (`pkl[1]`) for classification:

- Shared labels (Valence, Arousal, Stress, Disturbance): `y = 1` when the harmonised raw value is > 0 (HIGH), else 0 (LOW).
- D1 Attention / Mental / Duration / Change, D2 Attention / Mental / Duration / ValenceChange / ArousalChange: same HIGH / LOW threshold as shared labels.
- D3 PANAS-style affect words (0 to +6): `y = 1` when raw value > 3 (i.e., "moderate or above"); else 0.

---

## 5. Feature extraction time windows

The following windows are emitted per sensor (where applicable):

- **Current**: value at the ESM trigger timestamp.
- **Immediate past**: 15-minute and 30-minute windows ending at the trigger.
- **Yesterday epochs**: dawn / morning / afternoon / late afternoon / evening / night of the previous calendar day.
- **Today-so-far epochs**: same six epochs of the current day up to the trigger.

See [`../data/schema.md`](../data/schema.md) § "Time Window Suffixes" for the exact suffix naming.

---

## 6. Categorical-value alias map (Tier C harmonisation)

Resolved during Tier C feature alignment so that one-hot categorical columns line up across waves. Applied to the **source wave** so that it emits the target-wave's vocabulary; documented for each sensor family.

### 6.1 `CALL_CNT` (call contact type)

| Wave | Categories |
|---|---|
| D1 | `HOME`, `MAIN`, `MOBILE`, `OTHER`, `PAGER`, `VOICE`, `WORK`, + Korean labels `기타`, `휴대전화`, `휴대폰` |
| D2 | `HOME`, `MOBILE`, `OTHER`, `UNDEFINED`, `UNKNOWN`, `WORK` (no `MAIN`; no Korean) |
| D3 | `HOME`, `MAIN`, `MOBILE`, `OTHER`, `UNDEFINED`, `UNKNOWN`, `WORK` |

**Alias map (all → canonical):**
- `기타` → `OTHER`
- `휴대전화`, `휴대폰` → `MOBILE`
- `PAGER`, `VOICE` → `OTHER` (legacy; drop or alias)

### 6.2 `BAT_PLG#` (battery plug state)

- D1 emits `UNKNOWN`; D2 / D3 emit `UNDEFINED` for the same semantic category.
- **Alias**: `UNKNOWN` → `UNDEFINED`.

### 6.3 `WLS#` (wireless state)

- D1: `BT_OFF`, `BT_ON` only (2 categories).
- D2 / D3: 8 categories including transition states (`BT_TURNING_OFF`, `BT_TURNING_ON`, `WIFI_DISABLED`, `WIFI_DISABLING`, `WIFI_ENABLED`, `WIFI_ENABLING`).
- **Action**: drop the extra D2/D3-only categories for D1-inclusive Tier C transfers; keep them for D2↔D3 transfers.

### 6.4 `Notification_CAT#` (notification category)

- D1: 16 categories (no `NAVIGATION`).
- D2 / D3: 17 categories (`NAVIGATION` added).
- **Action**: drop `NAVIGATION` when D1 is in the mix; keep for D2↔D3 transfers.

### 6.5 Location clusters (`LOC_CLS#`)

- Cluster IDs are cohort-specific (per-participant k-means / DBSCAN hashes). They do not align across waves and are not harmonisable via aliasing.
- **Action**: keep within-wave; drop from Tier C common-feature intersection.

### 6.6 Feature-group presence matrix (summary)

| Feature group | D1 | D2 | D3 | Tier C treatment |
|---|---|---|---|---|
| `keyevent_*` | ✗ | ✓ | ✓ | Drop for D1-inclusive transfers |
| `WIFI_COS|JAC|EUC|MAN` | ✓ | ✓ | ✗ | Drop for D3-inclusive transfers |
| `FDI|FST|FCL|FAC` | ✗ | ✓ | ✓ | Drop for D1-inclusive transfers |
| `BT_RSSI` buckets | ✓ | ✗ | ✗ | Drop for cross-wave transfers; D1-only |
| `BT_classType`, `BT_DeviceType` | ✗ | ✓ | ✓ | Drop for D1-inclusive transfers |
| `PIF#` participant info | ✓ | ✓ | ✓ | Retained but not the focus of this benchmark; removed in some analyses (see paper FeatureAlign report: 7,961 / 10,102 / 10,550 cols after `PIF#` removal) |

The Tier C loader intersects the remaining columns and drops any column not present in every source/target wave.

---

## 7. Reproducibility notes

- All benchmarks use a **fixed global seed** (documented in [`../benchmark/README.md`](../benchmark/README.md)).
- Fit statistics for preprocessing transformers (means, scales, imputation values) are **fit on the training split only** and applied to validation/test unchanged.
- No test-split information leaks into feature selection, normalisation, or model selection.
- Optuna runs 30 trials per model × task × tier, selected on validation AUROC.

---

## 8. Known limitations

- D1 / D2 Polar H10 coverage is not complete within those waves (sub-period rotation).
- `LOC_CLS#` cluster IDs are not aligned across waves; cross-wave location analyses must re-cluster.
- `BAT_LEV` and similar numeric features are device-dependent; some Android manufacturers report at different granularities.
- The binary labelling threshold is a design choice; researchers preferring regression should use the raw values in `pkl[3]` and re-derive their own targets.
