# Dataset Overview

A condensed narrative description of the **CrossUserDataset** (D1, D2, D3). For the formal Datasheet for Datasets, see [`../DATASHEET.md`](../DATASHEET.md); for the column-level reference, see [`../data/schema.md`](../data/schema.md).

---

## 1. Motivation

Affective states are **dynamic, short-horizon, and person-dependent**. Longitudinal mobile-sensing benchmarks (e.g., GLOBEM) have shown the value of multi-year collection for behaviour and mental-health modelling, but their typical supervision is **sparse and weekly** rather than dense and moment-level. The CrossUserDataset fills this gap: it pairs **dense in-situ ESM labels** with **multimodal passive sensing** across three consecutive annual waves, on a relatively homogeneous cohort so that cross-user and cross-wave heterogeneity can be studied cleanly.

---

## 2. Three-wave design

| Wave | Year | Collection period | Participants (retained / recruited) | ESM responses (post-QC) | Mean responses / participant |
|---|---|---|---|---|---|
| D1 | 2020 | Feb 7 – Apr 2 (~30 d) | 92 / 102 | 10,259 | 111.5 (SD 51.1) |
| D2 | 2020–21 | Dec 7 – Jan 27 (~30 d) | 99 / 112 | 21,042 | 212.5 (SD 44.1) |
| D3 | 2021–22 | Nov 23 – Jan 11 (~28 d) | 106 / 114 | 21,838 | 206.0 (SD 24.7) |
| **Total** | 2020–22 | — | **297 / 328** | **53,139** | — |

Each wave spans roughly four weeks and yields approximately **6.7–7.0 ESM responses per participant per day**.

### Cross-wave evolution

- **D1** establishes the base protocol (Android + Fitbit Inspire HR + Polar H10 sub-period, 15 prompts/day).
- **D2** refines the questionnaire (splits the single Change item into ValenceChange and ArousalChange) and introduces the participant-facing progress counter.
- **D3** expands the label space (adds 8 PANAS-style affect-word items: Happy, Relaxed, Cheerful, Content, Sad, Anxious, Depressed, Angry) and drops Polar H10 due to participant discomfort.

This gives a **shared core label space** across all three waves (Valence, Arousal, Stress, Task Disturbance) plus **wave-specific semantic enrichment**, which is central to the benchmark design.

---

## 3. What is released

| Layer | Contents |
|---|---|
| **ESM labels** | `EsmResponse.csv` per wave: `Pcode`, `TriggerTime`, `ResponseTime`, and one column per label available in that wave (see [`../data/schema.md`](../data/schema.md)). |
| **Participant metadata** | `UserInfo.csv` per wave: demographics (Age, Gender), Big Five (K-BFI-15), PSS-10, GHQ-12, PHQ-9 (D1/D2 only). |
| **Feature matrices** | One `{label}.pkl` per label per wave, containing a 5-tuple `(features, y, groups, t, datetimes)` — see [`../data/schema.md`](../data/schema.md). |
| **Documentation** | This GitHub repository. |

Raw per-participant sensor CSVs are **not** part of the public release; contact the authors for raw-data requests.

---

## 4. Sensing modalities

### Smartphone (ABC Logger, Android)

App usage (category + duration), notifications, call/message events, screen events, GPS / location clusters, activity recognition, Wi-Fi scan, Bluetooth scan, battery events, network-data traffic, keystroke events (D2/D3 only), ambient light, doze / power-save state, ringer state, charger state.

### Wearable

| Stream | D1 | D2 | D3 |
|---|---|---|---|
| Fitbit Inspire HR — heart rate | ✓ | ✓ | ✓ |
| Fitbit Inspire HR — step count | ✓ | ✓ | ✓ |
| Fitbit Inspire HR — calories | ✓ | ✓ | ✓ |
| Fitbit Inspire HR — distance | ✓ | ✓ | ✓ |
| Fitbit — sleep | ✓ | ✓ | ✓ |
| Polar H10 — ECG (sub-period subgroups) | ✓ | ✓ | ✗ |

For the cross-wave feature-level schema differences (categorical vocabularies, added / removed blocks, alignment map), see [`feature_alignment.md`](./feature_alignment.md).

---

## 5. Labels

### Shared core (all three waves, 7-point scale)

| Label | D1/D2 scale | D3 scale |
|---|---|---|
| Valence | −3 to +3 | −3 to +3 |
| Arousal | −3 to +3 | −3 to +3 |
| Stress | −3 to +3 | 0 to +6 (shifted to −3/+3 for cross-wave) |
| Task Disturbance | −3 to +3 | 0 to +6 (shifted to −3/+3 for cross-wave) |

### Wave-specific auxiliary labels

- **D1**: Attention, Mental, Duration, Change (all −3 / +3).
- **D2**: Attention, Mental, Duration, ValenceChange, ArousalChange (all −3 / +3).
- **D3 (PANAS-style, 0 / +6)**: Happy, Relaxed, Cheerful, Content, Sad, Anxious, Depressed, Angry.

See [`../data/schema.md`](../data/schema.md) for the raw column names and post-import aliases.

---

## 6. Benchmark framework

The dataset ships with a **three-setting benchmark** (Setting A → Setting B → Setting C, progressively harder). See [`../benchmark/README.md`](../benchmark/README.md) for the full specification and [`../benchmark/setting_a/README.md`](../benchmark/setting_a/README.md), [`../benchmark/setting_b/README.md`](../benchmark/setting_b/README.md), [`../benchmark/setting_c/README.md`](../benchmark/setting_c/README.md) for setting-level details.

---

## 7. Cohort design

The cohort is deliberately homogeneous (university students in South Korea) to reduce broad demographic confounds and let cross-user / cross-wave heterogeneity be studied without large background-population differences dominating. This is a design choice, not a coincidence: the paper's central empirical result is that **substantial user-level heterogeneity persists even within this relatively homogeneous cohort**, making user-aware evaluation essential.

---

## 8. What this release does not attempt

- It is **not** a population-representative sample.
- It does **not** include free-text communications or media content.
- It does **not** include iOS-based usage patterns.
- It does **not** provide a symmetric PANAS-style label layer across all three waves (only D3).
- It does **not** provide raw sensor streams publicly; only pre-extracted features, ESM labels, and participant metadata.
