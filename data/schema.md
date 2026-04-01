# Data Schema Reference

Column-by-column reference for every file type in the D1, D2, and D3 dataset.  
All timestamps are Unix epoch time in **milliseconds, UTC+0** unless noted otherwise.  
Wave availability is noted per file where it differs across D1, D2, D3.

---

## Table of Contents

1. [pkl Files (Label Feature Matrices)](#pkl-files-label-feature-matrices)
2. [EsmResponse.csv](#2-esmresponsecsv)
3. [UserInfo.csv](#3-userinfocsv)

> **Note on raw sensor CSVs:** The raw per-participant sensor CSV files (Fitbit, AppUsageEvent, Location, etc.) are not part of the public release. Features derived from those files are already included in the pkl feature matrices. If you require the raw sensor data for your research, contact the authors at [CONTACT_EMAIL].

---

## Root-Level Files (per wave)

Each wave folder (D1/, D2/, D3/) contains the following files. Raw sensor CSVs are not released publicly — only the pre-extracted feature matrices (pkl files), the ESM label file, and the participant info file.

| File | Description |
|---|---|
| `EsmResponse.csv` | All ESM responses for the wave; one row per completed response |
| `UserInfo.csv` | Participant demographics and questionnaire scores; one row per participant |
| `{LabelName}.pkl` | Pre-extracted feature matrix for one label dimension; one pkl per label available in that wave |

---

## pkl Files (Label Feature Matrices)

Each pkl file is a **pandas DataFrame** serialized with pickle. Every pkl in a wave folder shares the same structure — rows, index, and feature columns are identical across all label pkls within the same wave. Only the target label column differs.

**Load with:**
```python
import pandas as pd
df = pd.read_pickle("D3/Stress.pkl")
```

**Index:** `RangeIndex` (integer, starting from 0). Each row corresponds to one ESM response instance, aligned in the same order as `EsmResponse.csv`.

**Shape:** ~10,000 columns for D1; ~20,000 columns for D2 and D3 (keystroke features added in D2/D3).

---

### Column Naming Convention

Columns follow one of three patterns depending on whether the feature is numeric, categorical, or participant-level:

**Pattern 1 — Numeric sensor feature (current window):**
```
{SENSOR_PREFIX}#VAL
```
Example: `FitbitHeartrate#VAL`, `LOC_DST#VAL`, `CALL_DUR#VAL`

**Pattern 2 — Categorical sensor feature (one-hot encoded):**
```
{SENSOR_PREFIX}#VAL={CATEGORY}
```
Example: `ACT#VAL=WALKING`, `BAT_STA#VAL=CHARGING`, `APP_CAT#VAL=SOCIAL`

**Pattern 3 — Time-windowed feature:**
```
{SENSOR_PREFIX}#{FEATURE}#{TIME_WINDOW}
```
Example: `Sleep#Duration#ImmediatePast_15`, `Sleep#Onset#YesterdayMorning`

**Pattern 4 — Participant info feature (static, same value for all rows of a participant):**
```
PIF#{ATTRIBUTE}
```
Example: `PIF#age`, `PIF#gender=F`, `PIF#BFI_OPENNESS`, `PIF#PSS10`

---

### Feature Groups

| Prefix | Source | Type | Example columns |
|---|---|---|---|
| `PIF#` | Participant Info | Static per participant | `PIF#age`, `PIF#gender=F`, `PIF#BFI_OPENNESS`, `PIF#PSS10`, `PIF#GHQ12`, `PIF#PHQ9` |
| `FitbitHeartrate#` | Fitbit HR | Numeric | `FitbitHeartrate#VAL` |
| `FitbitStepcount#` | Fitbit steps | Numeric | `FitbitStepcount#VAL` |
| `FitbitcaloriE#` / `Fitbitcalorie#` | Fitbit calories | Numeric | `Fitbitcalorie#VAL` |
| `FitbitdistanCE#` / `Fitbitdistance#` | Fitbit distance | Numeric | `Fitbitdistance#VAL` |
| `FCL_VAL#` | Fitbit calorie (alternative) | Numeric | `FCL_VAL#VAL` |
| `FDI_VAL#` | Fitbit distance (alternative) | Numeric | `FDI_VAL#VAL` |
| `FST_VAL#` | Fitbit steps (alternative) | Numeric | `FST_VAL#VAL` |
| `FAC_VAL#` | Fitbit activity confidence | Categorical | `FAC_VAL#VAL=still`, `FAC_VAL#VAL=walking` |
| `ACT#` | Activity type (current) | Categorical | `ACT#VAL=WALKING`, `ACT#VAL=STILL`, `ACT#VAL=IN_VEHICLE` |
| `ACE_*#` | Activity confidence per type | Numeric | `ACE_WLK#VAL`, `ACE_VHC#VAL`, `ACE_RUN#VAL` |
| `APP_CAT#` | App usage category | Categorical | `APP_CAT#VAL=SOCIAL`, `APP_CAT#VAL=WORK` |
| `APP_DUR_*#` | App usage duration by category | Numeric | `APP_DUR_SOCIAL#VAL`, `APP_DUR_WORK#VAL` |
| `BAT_*#` | Battery state | Numeric/Categorical | `BAT_LEV#VAL`, `BAT_STA#VAL=CHARGING`, `BAT_TMP#VAL` |
| `CALL_*#` | Call events | Numeric/Categorical | `CALL_DUR#VAL`, `CALL_CNT#VAL=MOBILE` |
| `MSG_*#` | Message events | Numeric | `MSG_SNT#VAL`, `MSG_RCV#VAL`, `MSG_ALL#VAL` |
| `DATA_*#` | Network data traffic | Numeric | `DATA_RCV#VAL`, `DATA_SNT#VAL`, `DATA_MRCV#VAL` |
| `LOC_*#` | GPS location | Numeric/Categorical | `LOC_DST#VAL`, `LOC_LABEL#VAL=home`, `LOC_LABEL#VAL=work` |
| `WIFI_*#` | Wi-Fi scan similarity | Numeric | `WIFI_COS#VAL`, `WIFI_EUC#VAL`, `WIFI_JAC#VAL`, `WIFI_MAN#VAL` |
| `WLS#` | Wi-Fi/Bluetooth state | Categorical | `WLS#VAL=WIFI_ENABLED`, `WLS#VAL=BT_ON` |
| `BT_*#` | Bluetooth device info | Categorical | `BT_classType#VAL=PHONE_SMART`, `BT_BondState#VAL=BONDED` |
| `SCR#` / `SCR_*#` | Screen events/duration | Numeric/Categorical | `SCR#VAL=SCREEN_ON`, `SCR_DUR#VAL`, `SCR_EVENT#VAL=USER_PRESENT` |
| `RING#` | Ringer mode | Categorical | `RING#VAL=SILENT`, `RING#VAL=VIBRATE`, `RING#VAL=NORMAL` |
| `CHG#` | Charger state | Categorical | `CHG#VAL=CONNECTED`, `CHG#VAL=DISCONNECTED` |
| `ONOFF#` | Power events | Categorical | `ONOFF#VAL=SHUTDOWN` |
| `PWR#` | Power save mode | Categorical | `PWR#VAL=ACTIVATED`, `PWR#VAL=DEACTIVATED` |
| `Dozemode#` | Doze mode state | Categorical | `Dozemode#VAL=ACTIVATED` |
| `INST_JAC#` | Installed app similarity | Numeric | `INST_JAC#VAL` |
| `Notification_*#` | Notification events | Categorical | `Notification_CAT#VAL=MESSAGE`, `Notification_VIS#VAL=PUBLIC` |
| `Sleep#` | Sleep features | Numeric | `Sleep#Duration`, `Sleep#Onset`, `Sleep#Duration#ImmediatePast_15`, `Sleep#Duration#YesterdayMorning` |
| `keyevent_*#` | Keystroke interactions | Numeric/Categorical | `keyevent_DIST#VAL`, `keyevent_TIME#VAL`, `keyevent_CAT#VAL=SOCIAL` — **D2 and D3 only** |

---

### Time Window Suffixes

Features without a time window suffix represent the **current** value at the ESM trigger timestamp. Features with a suffix represent aggregations over a specific past window:

| Suffix | Description |
|---|---|
| *(no suffix)* | Current value at ESM trigger time |
| `#ImmediatePast_15` | Aggregated over 15 minutes before ESM trigger |
| `#ImmediatePast_30` | Aggregated over 30 minutes before ESM trigger |
| `#YesterdayDawn` | Yesterday, dawn epoch (approx. 00:00–06:00) |
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

Note: Yesterday epochs use `#` as separator; Today epochs use `_` as separator. This is consistent with the original feature extraction implementation.

---

---

## 2. EsmResponse.csv

One row per completed ESM response. Expired (unanswered) prompts are excluded.

| Column | Type | Description |
|---|---|---|
| `Pcode` | string | Anonymized participant identifier (e.g., `P001`) |
| `TriggerTime` | int64 | Unix timestamp (ms, UTC) when the ESM prompt was delivered to the device |
| `ResponseTime` | int64 | Unix timestamp (ms, UTC) when the participant submitted the response |
| `Valence` | int | Self-reported valence. Scale: −3 (very negative) to +3 (very positive). Available in D1, D2, D3. |
| `Arousal` | int | Self-reported arousal. Scale: −3 (very calm) to +3 (very activated). Available in D1, D2, D3. |
| `Stress` | int | Self-reported stress. Scale: −3 to +3 in D1 and D2; 0 to +6 in D3. |
| `Disturbance` | int | Self-reported task disturbance. Scale: −3 to +3 in D1 and D2; 0 to +6 in D3. |
| `Happy` | int | PANAS-style affect word: Happy. Scale: 0 to +6. **D3 only.** |
| `Relaxed` | int | PANAS-style affect word: Relaxed. Scale: 0 to +6. **D3 only.** |
| `Cheerful` | int | PANAS-style affect word: Cheerful. Scale: 0 to +6. **D3 only.** |
| `Content` | int | PANAS-style affect word: Content. Scale: 0 to +6. **D3 only.** |
| `Sad` | int | PANAS-style affect word: Sad. Scale: 0 to +6. **D3 only.** |
| `Anxious` | int | PANAS-style affect word: Anxious. Scale: 0 to +6. **D3 only.** |
| `Depressed` | int | PANAS-style affect word: Depressed. Scale: 0 to +6. **D3 only.** |
| `Angry` | int | PANAS-style affect word: Angry. Scale: 0 to +6. **D3 only.** |
| `Attention` | int | Self-reported attention level. Scale: −3 to +3. **D1 and D2 only.** |
| `MentalLoad` | int | Self-reported mental load / cognitive demand. Scale: −3 to +3. **D1 and D2 only.** |
| `Duration` | int | Self-reported duration [DURATION_DESCRIPTION]. **D1 and D2 only.** |
| `Change` | int | Self-reported affective change since the previous ESM prompt. Scale: −3 to +3. **D1 only.** |
| `ChangedValence` | int | Self-reported change in valence since the previous ESM prompt. Scale: −3 to +3. **D2 only.** |
| `ChangedArousal` | int | Self-reported change in arousal since the previous ESM prompt. Scale: −3 to +3. **D2 only.** |

**Notes:**
- Response latency (time to respond) = `ResponseTime − TriggerTime`
- Scale difference for Stress and Disturbance between waves: −3 to +3 in D1 and D2; 0 to +6 in D3. Must be normalized before cross-wave modeling; see `preprocessing/pipeline_decisions.md`
- Label availability summary by wave:

| Label | D1 | D2 | D3 |
|---|---|---|---|
| Valence, Arousal, Stress, Disturbance | ✓ | ✓ | ✓ |
| Attention, MentalLoad, Duration | ✓ | ✓ | ✗ |
| Change | ✓ | ✗ | ✗ |
| ChangedValence, ChangedArousal | ✗ | ✓ | ✗ |
| Happy, Relaxed, Cheerful, Content, Sad, Anxious, Depressed, Angry | ✗ | ✗ | ✓ |

---

## 3. UserInfo.csv

One row per participant. Collected once per wave via post-study questionnaire.

| Column | Type | Description |
|---|---|---|
| `Pcode` | string | Anonymized participant identifier |
| `Age` | int | Age in years at time of study |
| `Gender` | string | `M` (male) or `F` (female) |
| `BFI_Openness` | float | Big Five Inventory — Openness subscale (K-BFI-15) |
| `BFI_Conscientiousness` | float | Big Five Inventory — Conscientiousness subscale (K-BFI-15) |
| `BFI_Extraversion` | float | Big Five Inventory — Extraversion subscale (K-BFI-15) |
| `BFI_Agreeableness` | float | Big Five Inventory — Agreeableness subscale (K-BFI-15) |
| `BFI_Neuroticism` | float | Big Five Inventory — Neuroticism subscale (K-BFI-15) |
| `PSS_Total` | float | Perceived Stress Scale total score (PSS-10). Available in D1, D2, D3. |
| `GHQ_Total` | float | General Health Questionnaire total score (GHQ-12). Available in D1, D2, D3. |
| `PHQ_Total` | float | Patient Health Questionnaire depression score (PHQ-9). **D1 and D2 only.** |
| `Wave` | string | Wave identifier: `D1`, `D2`, or `D3` |

---

*For wave-specific data quality issues and preprocessing decisions, see [`preprocessing/pipeline_decisions.md`](../preprocessing/pipeline_decisions.md).*
