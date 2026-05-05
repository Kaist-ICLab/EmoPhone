# Data

This folder documents the structure and format of the **CrossUserDataset** (D1, D2, D3) release.
The actual data files are hosted on **Harvard Dataverse** at **[TBD Dataverse URL on acceptance]**.

For a full column-by-column reference of every file type, see [`schema.md`](./schema.md).

> **Access note.** This dataset uses gated access. You must log in to Harvard Dataverse and agree to the Data Use Agreement before downloading. No manual approval is required — agreement to terms grants immediate access. See [`../LICENSE-DATA.md`](../LICENSE-DATA.md).

---

## Download

**Step 1 — Agree to terms.**
Visit **[TBD Dataverse URL on acceptance]**, log in or create a free account, and agree to the Data Use Agreement. This only needs to be done once.

**Step 2 — Get your API token.**
In your Dataverse account, go to **Account → API Token** and copy your token. This is required only for programmatic download.

**Step 3 — Download.**

*Option 1 — Browser (simplest):*
On the Dataverse dataset page, click the download button next to each wave archive (`D1.zip`, `D2.zip`, `D3.zip`).

*Option 2 — Command line:*
```bash
# Replace TBD_FILE_ID_D1 with the file ID shown on the Dataverse page
# Replace YOUR_API_TOKEN with your token from Step 2

curl -L "https://dataverse.harvard.edu/api/access/datafile/TBD_FILE_ID_D1" \
     -H "X-Dataverse-key: YOUR_API_TOKEN" \
     -o D1.zip

# Repeat for D2 and D3
```

*Option 3 — Python:*
```python
import requests

api_token = "YOUR_API_TOKEN"

wave_files = {
    "D1": "TBD_FILE_ID_D1",
    "D2": "TBD_FILE_ID_D2",
    "D3": "TBD_FILE_ID_D3",
}

for wave, file_id in wave_files.items():
    r = requests.get(
        f"https://dataverse.harvard.edu/api/access/datafile/{file_id}",
        headers={"X-Dataverse-key": api_token},
    )
    with open(f"{wave}.zip", "wb") as f:
        f.write(r.content)
    print(f"Downloaded {wave}.zip")
```

**Verify integrity after download:**
```bash
md5sum -c ../metadata/checksums.md5
```

---

## Folder Structure

Each wave folder contains three file types: participant metadata, raw ESM labels, and one pre-extracted feature matrix per label dimension.

```
D1/
├── UserInfo.csv              # one row per participant; demographics + questionnaires
├── EsmResponse.csv           # one row per completed ESM response; all raw self-report labels
│
│   # Label feature matrices (one .pkl per label; 5-tuple; see "File Formats")
├── valence.pkl
├── arousal.pkl
├── stress.pkl
├── disturbance.pkl
├── attention.pkl             # D1, D2 only
├── mental.pkl                # D1, D2 only
├── duration.pkl              # D1, D2 only
└── change.pkl                # D1 only

D2/
├── UserInfo.csv
├── EsmResponse.csv
├── valence.pkl
├── arousal.pkl
├── stress.pkl
├── disturbance.pkl
├── attention.pkl
├── mental.pkl
├── duration.pkl
├── valenceChange.pkl         # D2 only
└── arousalChange.pkl         # D2 only

D3/
├── UserInfo.csv
├── EsmResponse.csv
├── valence.pkl
├── arousal.pkl
├── stress.pkl
├── disturbance.pkl
├── happy.pkl
├── relaxed.pkl
├── cheerful.pkl
├── content.pkl
├── sad.pkl
├── anxious.pkl
├── depressed.pkl
└── angry.pkl
```

**pkl availability by wave:**

| Label | D1 | D2 | D3 |
|---|---|---|---|
| `valence.pkl` | ✓ | ✓ | ✓ |
| `arousal.pkl` | ✓ | ✓ | ✓ |
| `stress.pkl` | ✓ | ✓ | ✓ |
| `disturbance.pkl` | ✓ | ✓ | ✓ |
| `attention.pkl` | ✓ | ✓ | ✗ |
| `mental.pkl` | ✓ | ✓ | ✗ |
| `duration.pkl` | ✓ | ✓ | ✗ |
| `change.pkl` | ✓ | ✗ | ✗ |
| `valenceChange.pkl` | ✗ | ✓ | ✗ |
| `arousalChange.pkl` | ✗ | ✓ | ✗ |
| `happy.pkl` | ✗ | ✗ | ✓ |
| `relaxed.pkl` | ✗ | ✗ | ✓ |
| `cheerful.pkl` | ✗ | ✗ | ✓ |
| `content.pkl` | ✗ | ✗ | ✓ |
| `sad.pkl` | ✗ | ✗ | ✓ |
| `anxious.pkl` | ✗ | ✗ | ✓ |
| `depressed.pkl` | ✗ | ✗ | ✓ |
| `angry.pkl` | ✗ | ✗ | ✓ |

**Feature column counts** (see [`../docs/feature_alignment.md`](../docs/feature_alignment.md) for cross-wave differences):

| Wave | Total columns | After `PIF#` removal |
|---|---|---|
| D1 | 8,037 | 7,961 |
| D2 | 10,122 | 10,102 |
| D3 | 10,581 | 10,550 |

> Filenames are **lowercase** to match the raw ESM-column naming used by the preprocessing pipeline (`valenceChange`, `arousalChange`, `mental`, etc.). The human-readable label names (`Valence`, `Arousal`, `Mental`, `Changed Valence`, …) appear inside `EsmResponse.csv` and in loader output.

---

## File Formats

### CSV files (`UserInfo.csv`, `EsmResponse.csv`)

Use a header row. Key conventions:

- **Timestamps** — Unix epoch time in milliseconds, UTC. Convert with `pd.to_datetime(col, unit="ms", utc=True)`.
- **Participant IDs** — anonymised string codes (e.g., `P001`). Consistent within a wave; not guaranteed consistent across waves (no participant appears in more than one wave).
- **Missing values** — represented as empty cells.

### pkl files (`{label}.pkl`)

Each pkl file is a **5-element tuple** serialised with `pickle` — **not a plain DataFrame**. This format is produced by the Zhang et al. (2024) pipeline and is directly consumed by the benchmark code and the EDA helper `load_and_attach` in [`../EDA/utils.py`](../EDA/utils.py).

Unpacking order:

| Index | Name | Type | Description |
|---|---|---|---|
| `[0]` | `features` | `pd.DataFrame` | Feature matrix. Rows = ESM instances; columns = features (see [`schema.md`](./schema.md)). Includes static `PIF#*` participant-info columns broadcast to every row of that participant. |
| `[1]` | `y` | `np.ndarray` | Binary label (0 / 1), derived from the raw label by median-split or task-specific thresholding. |
| `[2]` | `groups` | `np.ndarray` | Participant codes (`Pcode`), one per row — used as the grouping variable in Setting B LOSO / group-K-fold splits. |
| `[3]` | `t` | `np.ndarray` | Raw-label values (float) before binarisation, retained for regression or re-binarisation. |
| `[4]` | `datetimes` | `np.ndarray` | Timestamps per row; datetime-like or Unix-ms — parse with `EDA.utils.parse_timestamp`. |

**Canonical loader** (see [`../EDA/utils.py`](../EDA/utils.py)):

```python
import pandas as pd

df, y, groups, t, datetimes = pd.read_pickle("D1/valence.pkl")

print(df.shape)        # (n_rows, n_features)
print(y.shape)         # (n_rows,)
print(set(groups))     # participant codes for this wave
```

---

## Loading Examples

### Load ESM labels and participant info

```python
import pandas as pd

wave = "D1"  # or "D2", "D3"

user_info = pd.read_csv(f"{wave}/UserInfo.csv")

esm = pd.read_csv(f"{wave}/EsmResponse.csv")
esm["TriggerTime"]  = pd.to_datetime(esm["TriggerTime"],  unit="ms", utc=True)
esm["ResponseTime"] = pd.to_datetime(esm["ResponseTime"], unit="ms", utc=True)

print(f"Participants: {esm['Pcode'].nunique()}")
print(f"ESM responses: {len(esm)}")
print(esm[["Valence", "Arousal", "Stress", "Disturbance"]].describe())
```

### Load a single label feature matrix

```python
import pandas as pd

wave  = "D1"
label = "valence"  # lowercase; must be available in this wave

features, y, groups, t, datetimes = pd.read_pickle(f"{wave}/{label}.pkl")

print(f"Feature matrix: {features.shape}")
print(f"Unique participants: {len(set(groups))}")
print(f"Positive class rate: {y.mean():.3f}")
```

### Load all shared-core labels for one wave

```python
import pandas as pd

wave        = "D1"
core_labels = ["valence", "arousal", "stress", "disturbance"]

label_data = {}
for label in core_labels:
    features, y, groups, t, datetimes = pd.read_pickle(f"{wave}/{label}.pkl")
    label_data[label] = {
        "features":   features,
        "y":          y,
        "groups":     groups,
        "t":          t,
        "datetimes":  datetimes,
    }
    print(f"{label}: {features.shape}")
```

### Load the same label across all three waves (for Setting C)

```python
import pandas as pd
import numpy as np

label = "valence"  # must be in all three waves; lowercase
waves = ["D1", "D2", "D3"]

features_all, y_all, groups_all, wave_all = [], [], [], []

for wave in waves:
    features, y, groups, t, datetimes = pd.read_pickle(f"{wave}/{label}.pkl")
    features_all.append(features)
    y_all.append(y)
    groups_all.append(groups)
    wave_all.extend([wave] * len(y))

# For Setting C, intersect columns across waves before concatenating:
common_cols = set.intersection(*(set(df.columns) for df in features_all))
features_combined = pd.concat([df[list(common_cols)] for df in features_all],
                              ignore_index=True)
y_combined      = np.concatenate(y_all)
groups_combined = np.concatenate(groups_all)
print(f"Common features: {len(common_cols)}; total instances: {len(y_combined)}")
```

**Note on cross-wave label compatibility.** Stress and Disturbance use different scales (−3 to +3 in D1/D2; 0 to +6 in D3). The binary label in `[1]` is already harmonised by the pipeline; the raw-value array in `[3]` preserves each wave's original scale and must be shifted (subtract 3 from D3) before combining across waves. See [`../preprocessing/pipeline_decisions.md`](../preprocessing/pipeline_decisions.md).

---

## Known Data Quality Issues

| Issue | Affected wave | Detail | Action |
|---|---|---|---|
| Polar H10 partial coverage | D1, D2 | ECG sensor rotated across participant subgroups in sub-periods | ECG-derived features reflect partial coverage; no ECG features in D3 |
| Stress / Disturbance scale difference | D3 vs. D1/D2 | D1/D2: −3 to +3; D3: 0 to +6 | Normalise before cross-wave use (subtract 3 from D3) |
| Keystroke events absent in D1 | D1 | `keyevent_*` block added from D2 onward | Setting C drops keyevent columns automatically via common-feature intersection |
| Wi-Fi similarity absent in D3 | D3 | `WIFI_COS|JAC|EUC|MAN` skipped in D3 preprocessing | Same — dropped by Setting C intersection |
| Fitness features absent in D1 | D1 | `FDI|FST|FCL|FAC` only produced for D2/D3 | Same — dropped by Setting C intersection |
| `BAT_PLG#` categorical mismatch | D1 vs. D2/D3 | D1 emits `UNKNOWN`; D2/D3 emit `UNDEFINED` | Aliased in the Setting C alias map |
| `CALL_CNT` legacy / Korean vocabulary | D1 | Includes `MAIN`, `PAGER`, `VOICE`, `기타`, `휴대전화`, `휴대폰` | Aliased to standardised English labels in the Setting C alias map |
| `Notification_CAT` missing `NAVIGATION` | D1 | Added to `Notification_CAT` from D2 onward | Dropped by Setting C intersection |

For the full feature-alignment story and the alias map, see [`../docs/feature_alignment.md`](../docs/feature_alignment.md). For QC specifics, see [`../preprocessing/pipeline_decisions.md`](../preprocessing/pipeline_decisions.md).
