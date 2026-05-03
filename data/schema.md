# Data Schema Reference

Column-by-column reference for every file type in the D1, D2, and D3 release.
All timestamps are Unix epoch time in **milliseconds, UTC** unless noted otherwise.
Wave availability is noted per file where it differs across D1 / D2 / D3.

---

## Table of Contents

1. [pkl Files (Label Feature Matrices)](#pkl-files-label-feature-matrices)
2. [EsmResponse.csv](#esmresponsecsv)
3. [UserInfo.csv](#userinfocsv)

> **Raw sensor CSVs are not part of the public release.** Features derived from the raw sensor streams are already packaged into the pkl feature matrices. Researchers requiring raw sensor data for a specific use case may contact the authors at **[TBD contact email on acceptance]**.

---

## Root-Level Files (per wave)

Each wave folder (`D1/`, `D2/`, `D3/`) contains the following files:

| File | Description |
|---|---|
| `EsmResponse.csv` | All completed ESM responses for the wave; one row per response |
| `UserInfo.csv` | Participant demographics and questionnaire scores; one row per participant |
| `{label}.pkl` | Pre-extracted feature matrix for one label dimension; one pkl per label available in that wave (see [`README.md`](./README.md)) |

---

## pkl Files (Label Feature Matrices)

Each pkl file is a **5-element tuple** serialised with `pickle`. This format is produced by the Zhang et al. (2024) pipeline and is directly consumed by the benchmark code.

**Load with:**
```python
import pandas as pd

features, y, groups, t, datetimes = pd.read_pickle("D3/stress.pkl")
```

| Index | Name | Type | Description |
|---|---|---|---|
| `[0]` | `features` | `pd.DataFrame` | Feature matrix (see columns below). Rows = ESM instances for this wave × label. |
| `[1]` | `y` | `np.ndarray` (int) | Binary label (0 / 1). |
| `[2]` | `groups` | `np.ndarray` (str) | Participant codes, aligned row-wise with `features`. |
| `[3]` | `t` | `np.ndarray` (float) | Raw (pre-binarisation) label values. |
| `[4]` | `datetimes` | `np.ndarray` | Timestamps per row (datetime-like or ms-epoch). |

**Feature-matrix shape (element `[0]`):**

| Wave | Rows | Columns (total) | Columns (after `PIF#` removal) |
|---|---|---|---|
| D1 | per task, varies by QC | 8,037 | 7,961 |
| D2 | per task, varies by QC | 10,122 | 10,102 |
| D3 | per task, varies by QC | 10,581 | 10,550 |

Rows within a wave are consistent across label pkls from that wave — only the label columns (`[1]`, `[3]`) differ. The feature DataFrame is identical in schema across all `{label}.pkl` within the same wave.

---

### Column Naming Convention

Columns follow one of four patterns:

**Pattern 1 — Numeric sensor feature (current window):**
```
{SENSOR_PREFIX}#VAL
```
Example: `FitbitHeartrate#VAL`, `LOC_DST#VAL`, `CALL_DUR#VAL`.

**Pattern 2 — Categorical sensor feature (one-hot encoded):**
```
{SENSOR_PREFIX}#VAL={CATEGORY}
```
Example: `ACT#VAL=WALKING`, `BAT_STA#VAL=CHARGING`, `APP_CAT#VAL=SOCIAL`.

**Pattern 3 — Time-windowed aggregate:**
```
{SENSOR_PREFIX}#{FEATURE}#{TIME_WINDOW}
```
Example: `Sleep#Duration#ImmediatePast_15`, `Sleep#Onset#YesterdayMorning`.

**Pattern 4 — Participant-info feature (static; same value for all rows of a participant):**
```
PIF#{ATTRIBUTE}
```
Example: `PIF#age`, `PIF#gender=F`, `PIF#BFI_OPENNESS`, `PIF#PSS10`.

---

### Feature Groups

| Prefix | Source | Type | Example columns |
|---|---|---|---|
| `PIF#` | Participant info (static) | Static per participant | `PIF#age`, `PIF#gender=F`, `PIF#BFI_OPENNESS`, `PIF#PSS10`, `PIF#GHQ12`, `PIF#PHQ9` (D1/D2 only) |
| `FitbitHeartrate#` | Fitbit heart rate | Numeric | `FitbitHeartrate#VAL` |
| `FitbitStepcount#` | Fitbit steps | Numeric | `FitbitStepcount#VAL` |
| `Fitbitcalorie#` | Fitbit calories | Numeric | `Fitbitcalorie#VAL` |
| `Fitbitdistance#` | Fitbit distance | Numeric | `Fitbitdistance#VAL` |
| `FCL_VAL#`, `FDI_VAL#`, `FST_VAL#`, `FAC_VAL#` | Fitness activity (alt.) | Numeric / Categorical | `FAC_VAL#VAL=walking` — **D2, D3 only** |
| `ACT#`, `ACE_*#` | Activity recognition | Categorical / Numeric | `ACT#VAL=WALKING`, `ACE_WLK#VAL` |
| `APP_CAT#`, `APP_DUR_*#` | App usage | Categorical / Numeric | `APP_CAT#VAL=SOCIAL`, `APP_DUR_WORK#VAL` |
| `BAT_*#` | Battery state | Numeric / Categorical | `BAT_LEV#VAL`, `BAT_STA#VAL=CHARGING`, `BAT_TMP#VAL`, `BAT_PLG#VAL=UNDEFINED` (D2/D3) / `=UNKNOWN` (D1) |
| `CALL_*#` | Call events | Numeric / Categorical | `CALL_DUR#VAL`, `CALL_CNT#VAL=MOBILE` |
| `MSG_*#` | Message events | Numeric | `MSG_SNT#VAL`, `MSG_RCV#VAL` |
| `DATA_*#` | Network traffic | Numeric | `DATA_RCV#VAL`, `DATA_SNT#VAL` |
| `LOC_*#`, `LOC_CLS#*` | GPS / location clusters | Numeric / Categorical | `LOC_DST#VAL`, `LOC_LABEL#VAL=home` |
| `WIFI_*#` | Wi-Fi similarity | Numeric | `WIFI_COS#VAL`, `WIFI_JAC#VAL` — **D1, D2 only** |
| `WLS#` | Wi-Fi / Bluetooth state | Categorical | `WLS#VAL=WIFI_ENABLED`, `WLS#VAL=BT_ON` — D2/D3 add transition states (`BT_TURNING_ON`, `WIFI_DISABLING`, …) |
| `BT_*#` | Bluetooth device info | Categorical / Numeric | D1: `BT_RSSI` buckets. D2/D3: `BT_classType#VAL=PHONE_SMART`, `BT_DeviceType#VAL=…` |
| `SCR#`, `SCR_*#` | Screen events | Numeric / Categorical | `SCR#VAL=SCREEN_ON`, `SCR_DUR#VAL` |
| `RING#`, `CHG#`, `ONOFF#`, `PWR#`, `Dozemode#` | Phone state events | Categorical | `RING#VAL=SILENT`, `CHG#VAL=CONNECTED` |
| `INST_JAC#` | Installed-app similarity | Numeric | `INST_JAC#VAL` |
| `Notification_*#` | Notifications | Categorical | `Notification_CAT#VAL=MESSAGE`, `Notification_CAT#VAL=NAVIGATION` (D2/D3 only), `Notification_VIS#VAL=PUBLIC` |
| `Sleep#` | Fitbit-derived sleep | Numeric | `Sleep#Duration`, `Sleep#Onset`, `Sleep#Duration#ImmediatePast_15` |
| `keyevent_*#` | Keystroke interactions | Numeric / Categorical | `keyevent_DIST#VAL`, `keyevent_CAT#VAL=SOCIAL` — **D2, D3 only** |

For the complete cross-wave presence/absence and category matrix, see [`../docs/feature_alignment.md`](../docs/feature_alignment.md).

---

### Time Window Suffixes

Features without a time-window suffix represent the **current** value at the ESM trigger timestamp. Features with a suffix represent aggregations over a specific past window:

| Suffix | Description |
|---|---|
| *(none)* | Current value at ESM trigger time |
| `#ImmediatePast_15` | Aggregated over 15 minutes before the ESM trigger |
| `#ImmediatePast_30` | Aggregated over 30 minutes before the ESM trigger |
| `#YesterdayDawn` | Yesterday, dawn epoch (~00:00–06:00) |
| `#YesterdayMorning` | Yesterday, morning epoch |
| `#YesterdayAfternoon` | Yesterday, afternoon epoch |
| `#YesterdayLateAfternoon` | Yesterday, late afternoon epoch |
| `#YesterdayEvening` | Yesterday, evening epoch |
| `#YesterdayNight` | Yesterday, night epoch |
| `_TodayDawn` | Today so far, dawn epoch |
| `_TodayMorning` | Today so far, morning epoch |
| `_TodayAfternoon` | Today so far, afternoon epoch |
| `_TodayLateAfternoon` | Today so far, late afternoon epoch |
| `_TodayEvening` | Today so far, evening epoch |
| `_TodayNight` | Today so far, night epoch |

> Yesterday epochs use `#` as separator; Today epochs use `_`. This reflects the original feature-extraction implementation.

---

## EsmResponse.csv

One row per completed ESM response. Expired / unanswered prompts are excluded.

| Column | Type | Description |
|---|---|---|
| `Pcode` | string | Anonymised participant identifier (e.g., `P001`) |
| `TriggerTime` | int64 | Unix timestamp (ms, UTC) when the ESM prompt was delivered |
| `ResponseTime` | int64 | Unix timestamp (ms, UTC) when the participant submitted the response |
| `Valence` | int | Self-reported valence. Scale −3 (very negative) to +3 (very positive). Available in D1, D2, D3. |
| `Arousal` | int | Self-reported arousal. Scale −3 (very calm) to +3 (very activated). Available in D1, D2, D3. |
| `Stress` | int | Self-reported stress. Scale −3 to +3 in D1/D2; 0 to +6 in D3. |
| `Disturbance` | int | Self-reported task disturbance. Scale −3 to +3 in D1/D2; 0 to +6 in D3. |
| `Attention` | int | Self-reported attention. Scale −3 to +3. **D1 and D2 only.** |
| `Mental` | int | Self-reported mental load / cognitive demand. Scale −3 to +3. **D1 and D2 only.** |
| `Duration` | int | Self-reported duration of the current affective state. **D1 and D2 only.** |
| `Change` | int | Self-reported affective change since previous prompt. Scale −3 to +3. **D1 only.** |
| `ValenceChange` | int | Change in valence since previous prompt. Scale −3 to +3. **D2 only.** |
| `ArousalChange` | int | Change in arousal since previous prompt. Scale −3 to +3. **D2 only.** |
| `Happy` | int | PANAS-style affect word. Scale 0 to +6. **D3 only.** |
| `Relaxed` | int | PANAS-style affect word. Scale 0 to +6. **D3 only.** |
| `Cheerful` | int | PANAS-style affect word. Scale 0 to +6. **D3 only.** |
| `Content` | int | PANAS-style affect word. Scale 0 to +6. **D3 only.** |
| `Sad` | int | PANAS-style affect word. Scale 0 to +6. **D3 only.** |
| `Anxious` | int | PANAS-style affect word. Scale 0 to +6. **D3 only.** |
| `Depressed` | int | PANAS-style affect word. Scale 0 to +6. **D3 only.** |
| `Angry` | int | PANAS-style affect word. Scale 0 to +6. **D3 only.** |

> **Column-casing note.** The raw CSV column names are in lowercase / camelCase (`pcode`, `valence`, `arousal`, `stress`, `disturbance`, `attention`, `mental`, `duration`, `change`, `valenceChange`, `arousalChange`, `happy`, …). The EDA loader [`../EDA/utils.py`](../EDA/utils.py) renames them to PascalCase / human-readable form on import. The table above shows the post-import names; the pkl filenames use the raw lowercase / camelCase form.

**Response latency** = `ResponseTime − TriggerTime` (ms).

**Scale normalisation**: D3 Stress and Disturbance are shifted from [0, +6] to [−3, +3] for cross-wave analysis. The binary label already reflects this harmonisation (see [`../preprocessing/pipeline_decisions.md`](../preprocessing/pipeline_decisions.md)).

**Label availability summary by wave:**

| Label | D1 | D2 | D3 |
|---|---|---|---|
| Valence, Arousal, Stress, Disturbance | ✓ | ✓ | ✓ |
| Attention, Mental, Duration | ✓ | ✓ | ✗ |
| Change | ✓ | ✗ | ✗ |
| ValenceChange, ArousalChange | ✗ | ✓ | ✗ |
| Happy, Relaxed, Cheerful, Content, Sad, Anxious, Depressed, Angry | ✗ | ✗ | ✓ |

---

## UserInfo.csv

One row per participant. Collected at onboarding and/or exit via questionnaire.

| Column | Type | Description |
|---|---|---|
| `Pcode` | string | Anonymised participant identifier |
| `Age` | int | Age (years) at time of study |
| `Gender` | string | `M` (male) or `F` (female) |
| `BFI_Openness` | float | Big Five Inventory — Openness subscale (K-BFI-15) |
| `BFI_Conscientiousness` | float | Big Five Inventory — Conscientiousness subscale (K-BFI-15) |
| `BFI_Extraversion` | float | Big Five Inventory — Extraversion subscale (K-BFI-15) |
| `BFI_Agreeableness` | float | Big Five Inventory — Agreeableness subscale (K-BFI-15) |
| `BFI_Neuroticism` | float | Big Five Inventory — Neuroticism subscale (K-BFI-15) |
| `PSS_Total` | float | Perceived Stress Scale (PSS-10) total score. Available in D1, D2, D3. |
| `GHQ_Total` | float | General Health Questionnaire (GHQ-12) total score. Available in D1, D2, D3. |
| `PHQ_Total` | float | Patient Health Questionnaire depression score (PHQ-9). **D1 and D2 only.** |
| `Wave` | string | Wave identifier: `D1`, `D2`, or `D3` |

---

*For wave-specific data-quality issues and preprocessing decisions, see [`../preprocessing/pipeline_decisions.md`](../preprocessing/pipeline_decisions.md). For cross-wave feature alignment (Tier C), see [`../docs/feature_alignment.md`](../docs/feature_alignment.md).*
