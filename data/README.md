# Data

This folder documents the structure and format of the [DATASET_NAME] dataset.  
The actual data files are hosted on **Harvard Dataverse** at **[DATAVERSE_URL]**.

For a full column-by-column reference of every file type, see [`schema.md`](./schema.md).

> **Access note:** This dataset uses gated access. You must log in to Harvard Dataverse and agree to the Data Use Agreement before downloading. No manual approval is required — agreement to terms grants immediate access.

---

## Download

**Step 1 — Agree to terms:**  
Visit **[DATAVERSE_URL]**, log in or create a free account, and agree to the Data Use Agreement. This only needs to be done once.

**Step 2 — Get your API token:**  
In your Dataverse account, go to **Account → API Token** and copy your token. This is required for programmatic access.

**Step 3 — Download**

**Option 1 — Direct download via browser (simplest):**  
On the Dataverse dataset page, click the download button next to each wave archive (D1.zip, D2.zip, D3.zip).

**Option 2 — Command line:**
```bash
# Replace [DATAVERSE_FILE_ID_D1] with the file ID shown on the Dataverse page
# Replace YOUR_API_TOKEN with your token from Step 2

curl -L "https://dataverse.harvard.edu/api/access/datafile/[DATAVERSE_FILE_ID_D1]" \
     -H "X-Dataverse-key: YOUR_API_TOKEN" \
     -o D1.zip

# Repeat for D2 and D3
```

**Option 3 — Python:**
```python
import requests

api_token = "YOUR_API_TOKEN"

# File IDs for each wave (find these on the Dataverse dataset page)
wave_files = {
    "D1": "[DATAVERSE_FILE_ID_D1]",
    "D2": "[DATAVERSE_FILE_ID_D2]",
    "D3": "[DATAVERSE_FILE_ID_D3]",
}

for wave, file_id in wave_files.items():
    r = requests.get(
        f"https://dataverse.harvard.edu/api/access/datafile/{file_id}",
        headers={"X-Dataverse-key": api_token}
    )
    with open(f"{wave}.zip", "wb") as f:
        f.write(r.content)
    print(f"Downloaded {wave}.zip")
```

**After downloading, verify file integrity:**
```bash
md5sum -c ../metadata/checksums.md5
```

---

## Folder Structure

Each wave folder contains three file types: participant metadata, raw ESM labels, and one pre-extracted feature matrix per label dimension.

```
D1/
├── UserInfo.csv        ← One row per participant; demographics + questionnaires
├── EsmResponse.csv     ← One row per ESM response; all raw self-report labels
│
│   ── Label feature matrices (one .pkl per label, pre-extracted) ──
├── Valence.pkl         
├── Arousal.pkl         
├── Stress.pkl          
├── Disturbance.pkl     
├── Attention.pkl        ←              [D1, D2 only]
├── mental.pkl           ←              [D1, D2 only]
├── Duration.pkl         ←              [D1, D2 only]
└── Change.pkl           ←              [D1 only]

D2/
├── UserInfo.csv
├── EsmResponse.csv
├── Valence.pkl
├── Arousal.pkl
├── Stress.pkl
├── Disturbance.pkl
├── Attention.pkl
├── mental.pkl
├── Duration.pkl
├── valenceChange.pkl     ←             [D2 only]
└── arousalChange.pkl     ←             [D2 only]

D3/
├── UserInfo.csv
├── EsmResponse.csv
├── Valence.pkl
├── Arousal.pkl
├── Stress.pkl
├── Disturbance.pkl
├── Happy.pkl        
├── Relaxed.pkl
├── Cheerful.pkl
├── Content.pkl
├── Sad.pkl
├── Anxious.pkl
├── Depressed.pkl
└── Angry.pkl
```

**pkl file availability by wave:**

| Label | D1 | D2 | D3 |
|---|---|---|---|
| `Valence.pkl` | ✓ | ✓ | ✓ |
| `Arousal.pkl` | ✓ | ✓ | ✓ |
| `Stress.pkl` | ✓ | ✓ | ✓ |
| `Disturbance.pkl` | ✓ | ✓ | ✓ |
| `Attention.pkl` | ✓ | ✓ | ✗ |
| `mental.pkl` | ✓ | ✓ | ✗ |
| `Duration.pkl` | ✓ | ✓ | ✗ |
| `Change.pkl` | ✓ | ✗ | ✗ |
| `valenceChange.pkl` | ✗ | ✓ | ✗ |
| `arousalChange.pkl` | ✗ | ✓ | ✗ |
| `Happy.pkl` | ✗ | ✗ | ✓ |
| `Relaxed.pkl` | ✗ | ✗ | ✓ |
| `Cheerful.pkl` | ✗ | ✗ | ✓ |
| `Content.pkl` | ✗ | ✗ | ✓ |
| `Sad.pkl` | ✗ | ✗ | ✓ |
| `Anxious.pkl` | ✗ | ✗ | ✓ |
| `Depressed.pkl` | ✗ | ✗ | ✓ |
| `Angry.pkl` | ✗ | ✗ | ✓ |

**Column counts per wave:** D1 has ~8,000 feature columns (exact: 8,037); D2 has ~10,100 (exact: 10,122); D3 has ~10,600 (exact: 10,581). The increase from D1 to D2/D3 reflects the addition of keystroke features. See [`schema.md`](./schema.md) for the feature naming convention.

---

## File Formats

**CSV files** (`UserInfo.csv`, `EsmResponse.csv`) use a header row. Key conventions:
- **Timestamps** — Unix epoch time in milliseconds, UTC+0. Convert with `pd.to_datetime(col, unit="ms", utc=True)`.
- **Participant IDs** — anonymized string codes (e.g., `P001`). Consistent within a wave; not guaranteed consistent across waves.
- **Missing values** — represented as empty cells.

**pkl files** are 5-element tuples serialized with `pickle` — **not plain DataFrames**. Element `[0]` is the feature DataFrame; elements `[1]`–`[4]` are binary labels, participant codes, raw float labels, and timestamps respectively. See [`schema.md`](./schema.md) for full format details and the feature column naming convention.

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

### Load a label feature matrix

Each pkl is a **5-element tuple**. Unpack it as follows:

```python
import pandas as pd

wave  = "D1"    # or "D2", "D3"
label = "valence"  # any label available for that wave (use lowercase)

data = pd.read_pickle(f"{wave}/{label}.pkl")

features    = data[0]   # pd.DataFrame  — feature matrix (rows=ESM instances, cols=features)
labels_bin  = data[1]   # np.ndarray   — binary label (0/1)
pcodes      = data[2]   # np.ndarray   — participant codes
labels_raw  = data[3]   # np.ndarray   — raw float label values
timestamps  = data[4]   # np.ndarray   — timestamps

print(f"Feature matrix shape: {features.shape}")
print(f"Unique participants: {len(set(pcodes))}")
```

### Load all shared-core label pkls for one wave

```python
import pandas as pd

wave        = "D1"
core_labels = ["valence", "arousal", "stress", "disturbance"]  # lowercase

label_data = {}
for label in core_labels:
    data = pd.read_pickle(f"{wave}/{label}.pkl")
    label_data[label] = {
        "features":   data[0],
        "labels_bin": data[1],
        "pcodes":     data[2],
        "labels_raw": data[3],
        "timestamps": data[4],
    }
    print(f"{label}: {data[0].shape}")
```

### Load the same label across all three waves

```python
import pandas as pd
import numpy as np

label = "valence"  # must be available in all three waves; use lowercase
waves = ["D1", "D2", "D3"]

features_all, labels_all, pcodes_all, wave_all = [], [], [], []

for wave in waves:
    data = pd.read_pickle(f"{wave}/{label}.pkl")
    features_all.append(data[0])
    labels_all.append(data[1])
    pcodes_all.append(data[2])
    wave_all.extend([wave] * len(data[1]))

features_combined = pd.concat(features_all, ignore_index=True)
labels_combined   = np.concatenate(labels_all)
pcodes_combined   = np.concatenate(pcodes_all)
print(f"Total instances: {len(labels_combined)}")
```

**Note on cross-wave label compatibility:** Stress and Disturbance use different scales across waves (−3 to +3 in D1/D2; 0 to +6 in D3). Normalize the label column before combining waves for modeling. See [`../preprocessing/pipeline_decisions.md`](../preprocessing/pipeline_decisions.md) for the normalization approach used in the benchmark.

---

## Known Data Quality Issues

| Issue | Affected wave | Detail | Action |
|---|---|---|---|
| Polar H10 partial coverage | D1, D2 | ECG sensor assigned to participant subgroups in rotating sub-periods | ECG-derived features reflect partial coverage; no ECG features in D3 |
| Stress/Disturbance scale difference | D3 vs D1/D2 | D1/D2: −3 to +3; D3: 0 to +6 | Normalize before cross-wave use |

For full QC details, see [`../preprocessing/pipeline_decisions.md`](../preprocessing/pipeline_decisions.md).