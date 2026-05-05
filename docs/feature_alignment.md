# Cross-Wave Feature Alignment

This document is the **authoritative record** of how the sensor-derived feature schema differs across D1, D2, and D3, and of the harmonisation policy used for cross-wave (Setting C) transfer.

It mirrors [`../preprocessing/feature_alignment.md`](../preprocessing/feature_alignment.md); both are kept in sync so that readers discovering the repo via `preprocessing/` or `docs/` find the same content.

---

## 1. Column counts

| Wave | Total columns | After `PIF#` removal |
|---|---|---|
| D1 | 8,037 | 7,961 |
| D2 | 10,122 | 10,102 |
| D3 | 10,581 | 10,550 |
| Shared across all three (after `PIF#` removal) | — | 5,005 |

Removing `PIF#` (participant-info) narrows the diff but does **not** eliminate it. The remaining gap is driven by sensor-pipeline evolution across waves.

---

## 2. Main feature differences

| Feature block | D1 | D2 | D3 | Notes |
|---|---|---|---|---|
| Location clusters (`LOC_CLS#*`) | 2,265 | 3,011 | 3,879 | Cluster IDs are cohort-specific; one-hot encodings diverge. Not harmonisable by aliasing. |
| Key events (`keyevent_*`) | *absent* | 367 | 367 | Introduced from D2 onward. |
| Bluetooth | `BT_RSSI` buckets (113) | `BT_classType` (461), `BT_DeviceType` (93) | `BT_classType` (493), `BT_DeviceType` (93) | D1 uses RSSI buckets; D2/D3 use categorical device types. |
| Wi-Fi similarity (`WIFI_*`) | COS/JAC/EUC/MAN (4 × 113) | COS/JAC/EUC/MAN (4 × 113) | *absent* | Dropped in D3. |
| Fitness (`FDI`, `FST`, `FCL`, `FAC`) | *absent* | 93–113 each | 93–113 each | Extra preprocessing step applied only from D2 onward. |
| Battery plug (`BAT_PLG#*`) | `UNKNOWN` | `UNDEFINED` | `UNDEFINED` | Non-overlapping one-hot categories; aliased. |
| `CALL_CNT` | 189 cols, 10 cats (incl. Korean, `MAIN`, `PAGER`, `VOICE`) | 125 cols, 6 cats | 141 cols, 7 cats | Alias Korean labels → English canonical forms; alias legacy `PAGER` / `VOICE` → `OTHER`. |
| `WLS#` | 47 cols (2 cats: `BT_OFF`, `BT_ON`) | 157 cols (8 cats) | 157 cols (8 cats) | D2/D3 add Wi-Fi state and BT/WiFi transition events. |
| `Notification_CAT#` | 285 cols (16 cats) | 301 cols (17 cats with `NAVIGATION`) | 301 cols (17 cats) | `NAVIGATION` added from D2 onward. |

---

## 3. Harmonisation strategy

Setting C applies the following strategy in order:

1. **Alias unification** — rewrite categorical columns using a canonical vocabulary so that equivalent semantic values collapse to the same one-hot column (see [`../preprocessing/pipeline_decisions.md`](../preprocessing/pipeline_decisions.md) § 6).
2. **Common-feature intersection** — after alias unification, restrict to the set of columns present in every source and target wave for the current transfer scenario.
3. **Split-consistent scaling** — mean/variance computed on the (concatenated) training source; applied unchanged to validation and target.

The expected outcome:

- Cohort-specific artefacts (location clusters, RSSI buckets, wave-only notification categories) are dropped.
- Truly shared behavioural signals (Fitbit streams, core app-usage categories, battery state, GPS distance, most screen-event features) survive intersection and carry the transfer.

---

## 4. How this interacts with the benchmark settings

- **Setting A** (within-user): no alignment needed; uses the wave's native schema.
- **Setting B** (within-wave cross-user): no alignment needed; per-wave schema.
- **Setting C** (cross-wave): alias-unified + common-feature intersection, producing a wave-pair-specific schema. Documented per transfer scenario in [`../benchmark/setting_c/README.md`](../benchmark/setting_c/README.md).

---

## 5. Open notes

- `LOC_CLS#*` cluster IDs are not aligned across waves. Researchers wanting cross-wave location features should re-cluster raw GPS (not included in the public release).
- `PIF#` features (age, gender, BFI, PSS, GHQ, PHQ) are semantically identical across waves but only D1 and D2 carry PHQ-9. PHQ columns are dropped for D3-inclusive Setting C intersections.
- The baseline for this alignment report originally compared internal pickles named `D-2 / D-3 / D-4`; those map to the released **D1 / D2 / D3** respectively. Column counts in this file use the released naming.
