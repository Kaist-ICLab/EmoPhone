# Datasheet for CrossUserDataset (D1–D3)

*Following the Datasheet for Datasets framework (Gebru et al., 2021).*
*Last updated: [TBD release date]*

> Fields marked **[TBD on acceptance]** will be populated at the time of public release so that the authors and institution remain anonymous during peer review.

---

## 1. Motivation

**For what purpose was the dataset created?**
The dataset supports research in mobile affective computing — specifically, dense moment-level modelling of affective states (valence, arousal, stress, task disturbance) as they unfold in everyday life. Existing longitudinal mobile-sensing benchmarks (e.g., GLOBEM) target broader behavioural or mental-health outcomes at weekly intervals. This dataset fills that gap by combining repeated in-situ self-reports with ecologically valid passive sensing across three consecutive annual waves, enabling both within-wave and cross-wave benchmarking.

**Who created the dataset and on behalf of which entity?**
[TBD on acceptance]

**Who funded the creation of the dataset?**
[TBD on acceptance]

---

## 2. Composition

**What do the instances represent?**
Each instance is a time-stamped ESM (Experience Sampling Method) response from a study participant, linked to a window of passively collected sensor data from that participant's smartphone and wearable device. A single instance captures a participant's momentary self-reported affective state together with the sensor context surrounding that moment.

**How many instances are there in total (across all waves)?**

| Wave | Recruited | Retained (QC) | QC exclusion rate | ESM responses (post-QC) |
|---|---|---|---|---|
| D1 | 102 | 92 | 9.8% | 10,259 |
| D2 | 112 | 99 | 11.6% | 21,042 |
| D3 | 114 | 106 | 7.0% | 21,838 |
| **Total** | **328** | **297** | — | **53,139** |

Quality-control thresholds: wearable coverage ≥ 50%, smartphone coverage ≥ 50%, ESM response rate ≥ 30%. See [`preprocessing/pipeline_decisions.md`](./preprocessing/pipeline_decisions.md) for the full QC specification.

Participants were recruited independently for each wave; no participant appears in more than one wave, so 297 retained participants equals 297 unique individuals.

**Does the dataset contain all possible instances, or is it a sample?**
The dataset is a sample of university students in South Korea recruited via campus bulletin boards. It is not a random population sample. The cohort is intentionally relatively homogeneous to reduce broad demographic confounds in cross-user and cross-dataset analyses.

**What data does each instance consist of?**
Each instance consists of:

- **Self-report labels** (shared core): Valence, Arousal, Stress, Task Disturbance (7-point scale, −3 to +3 in D1/D2; 0 to +6 for Stress and Disturbance in D3, normalised to −3/+3 for benchmarking).
- **Wave-specific labels**:
  - D1: Attention, Mental, Duration, Change.
  - D2: Attention, Mental, Duration, ValenceChange, ArousalChange.
  - D3: 8 PANAS-style affect-word intensity items — Happy, Relaxed, Cheerful, Content, Sad, Anxious, Depressed, Angry (0 to +6).
- **Pre-extracted feature matrices** (one pkl per label per wave):
  - D1: 8,037 feature columns total (7,961 after `PIF#` removal).
  - D2: 10,122 feature columns total (10,102 after `PIF#` removal).
  - D3: 10,581 feature columns total (10,550 after `PIF#` removal).
  - Features are derived from passive smartphone sensing (app usage, notifications, call/message events, screen events, GPS, activity recognition, Wi-Fi scan, Bluetooth scan, battery, data traffic, keystroke events in D2/D3) and wearable sensing (Fitbit Inspire HR heart rate, step count, calories, distance; Polar H10 ECG-derived features for D1/D2 sub-periods only). See [`data/schema.md`](./data/schema.md) for the feature-naming convention and [`docs/feature_alignment.md`](./docs/feature_alignment.md) for cross-wave differences.
- **Participant characteristics** (collected once per wave, propagated as `PIF#` columns in every pkl): Age, Gender, Big Five personality (K-BFI-15), Perceived Stress (PSS-10), General Health (GHQ-12), and Depressive symptoms (PHQ-9; D1 and D2 only).

> **Note on raw sensor data.** Raw per-participant sensor CSVs are not part of the public release. The Dataverse release contains the pre-extracted pkl feature matrices, `EsmResponse.csv`, and `UserInfo.csv` per wave. Researchers requiring raw sensor data for a specific use case may contact the authors at **[TBD contact email on acceptance]**.

**Is there a label or target associated with each instance?**
Yes. Primary prediction targets are the four shared core labels. D3 additionally includes eight affect-word intensity labels.

**Is any information missing from individual instances?**
Yes. Sensor-data availability varies across participants due to:

- Device heterogeneity (Android version, manufacturer).
- **Polar H10** chest ECG: assigned to participant subgroups in rotating sub-periods for D1 and D2; not collected in D3 (discontinued due to participant discomfort).
- **Keystroke events** (`keyevent_*`): introduced from D2 onward; absent in D1.
- **Wi-Fi similarity** features (`WIFI_COS|JAC|EUC|MAN`): present in D1/D2, skipped in D3 preprocessing.
- **Fitness data** (`FDI|FST|FCL|FAC`): derived only for D2/D3 due to a preprocessing step that was not applied to D1.
- **Battery plug** (`BAT_PLG#`) categorical domain: `UNKNOWN` in D1 vs. `UNDEFINED` in D2/D3 (to be harmonised on import — see alias map).

**Are relationships between instances made explicit?**
Yes. Each ESM response is linked to a participant identifier (`Pcode`) and trigger/response timestamps, enabling alignment with passive-sensor time series. Temporal ordering within and across waves is preserved via Unix timestamps (milliseconds, UTC).

**Are there recommended data splits?**
Yes. We provide a three-tier benchmark framework:

- **Tier A** (personal predictability): per-user 60/20/20 chronological split over the first 30 days.
- **Tier B** (cross-user transfer): stratified group 5-fold cross-validation with `Pcode` as the group variable, per wave.
- **Tier C** (cross-dataset transfer): leave-one-dataset-out (1→1 and 2→1) across waves, using the shared-label intersection and a harmonised common feature space.

See [`benchmark/README.md`](./benchmark/README.md).

**Does the dataset contain data that might be considered confidential?**
The dataset contains passively collected behavioural data (app usage, location, communication metadata) from real people. All data were collected under IRB approval and written informed consent. Prior to release, the following anonymisation steps were applied:

- Contact numbers (calls, messages): MD5-hashed.
- Bluetooth and Wi-Fi MAC addresses: replaced with UUIDs.
- GPS longitude: offset by a per-participant random constant (latitude is retained).

**Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?**
No. The dataset does not contain free-text communications or media content.

**Does the dataset relate to people?**
Yes. All data were collected from human participants (university students).

**Might the dataset expose people to harm or legal action?**
Risk is low given the applied anonymisation. Residual re-identification risk via location patterns is mitigated by restricting use to non-commercial academic research under the Data Use Agreement (see [`LICENSE-DATA.md`](./LICENSE-DATA.md)).

---

## 3. Collection Process

**How was the data associated with each instance acquired?**

- **ESM labels**: self-reported via the **ABC Logger** Android application, prompted at random intervals during waking hours (10 AM – 10 PM). D1 used ~15 prompts/day; D2 and D3 used a reduced, more concentrated schedule designed to improve compliance.
- **Smartphone sensor data**: passively collected in the background by ABC Logger and uploaded to a secure server. Modalities include GPS/location, Bluetooth and Wi-Fi scans, app usage, screen events, notifications, battery events, accelerometer, ambient light, activity transitions, network traffic, call/message metadata, and (D2/D3) keystroke events.
- **Wearable data**: Fitbit Inspire HR synced via the Fitbit API across all three waves; Polar H10 ECG synced via Bluetooth to the Android app for sub-period subgroups in D1 and D2.

**What mechanisms or procedures were used to collect the data?**
A custom Android application (ABC Logger) managed ESM-prompt delivery, sensor collection, and upload. ESM prompts expired after 10 minutes if unanswered. Participants were asked to keep the app and wearables active for ≥ 12 hours per day. Research staff monitored participation every 2–3 days. From D2 onward, a participant-facing progress counter was introduced to improve compliance.

**Who was involved in the data collection process?**
Research team members at **[TBD institution on acceptance]**. Participants were university students recruited via online campus bulletin boards. Participation was voluntary and compensated (~100 USD on completion).

**Over what timeframe was the data collected?**

| Wave | Collection period |
|---|---|
| D1 | 2020-02-07 – 2020-04-02 (~30 days) |
| D2 | 2020-12-07 – 2021-01-27 (~30 days) |
| D3 | 2021-11-23 – 2022-01-11 (~28 days) |

D1 overlaps with the onset of the COVID-19 pandemic in South Korea. D2 and D3 occurred during the pandemic. This contextual difference may influence behavioural patterns and should be considered in analyses.

**Were any ethical review processes conducted?**
Yes. All procedures were approved by the Institutional Review Board (IRB) at **[TBD institution on acceptance]**, approval number **[TBD on acceptance]**. All participants provided written informed consent prior to participation.

**Did the individuals in question consent to the collection and use of their data?**
Yes. Written informed consent was obtained at a one-hour in-person briefing before the study began. Participants were informed of the study's purpose, procedures, data-sharing plan, and their right to withdraw at any time. An English translation of the consent form is provided at [`docs/consent_form_en.md`](./docs/consent_form_en.md).

**Has an analysis of the potential impact of the dataset and its use on data subjects been conducted?**
An informal privacy-risk review was conducted prior to release (see Section 2 anonymisation steps). A formal external impact assessment has not been conducted. See [`docs/ethics.md`](./docs/ethics.md) for the current risk register.

---

## 4. Preprocessing / Cleaning / Labeling

**Was any preprocessing/cleaning/labeling of the data done?**
Yes. Data preprocessing follows the reproducible 8-stage pipeline introduced in Zhang et al. (2024): ESM quality control, feature extraction, feature preparation, feature selection, data splitting, resampling, model training, and model evaluation.

Specific QC and harmonisation steps applied to this dataset:

- **ESM-level QC**: expired (unanswered) prompts removed; responses submitted after the 10-minute expiry window removed.
- **Participant-level QC**: wearable coverage ≥ 50%, smartphone coverage ≥ 50%, ESM response rate ≥ 30% (exclusion rates per wave: D1 9.8% / D2 11.6% / D3 7.0%).
- **Sensor QC**: Polar H10 ECG is available only for participants during their assigned sub-period in D1 and D2; ECG-derived features reflect this partial coverage.
- **Label harmonisation**: Stress and Disturbance scales differ between D1/D2 (−3 to +3) and D3 (0 to +6). For cross-wave tasks, D3 Stress/Disturbance is shifted by −3. See [`preprocessing/pipeline_decisions.md`](./preprocessing/pipeline_decisions.md).
- **Feature extraction**: statistical features computed across multiple time windows (current value, 15-min and 30-min immediate past, daily epochs yesterday/today), following Zhang et al. (2024).
- **Cross-wave feature alignment** (Tier C only): alias-cleanup of inconsistent categorical label vocabularies (e.g., Korean / legacy CALL_CNT values in D1; `UNKNOWN` → `UNDEFINED` for `BAT_PLG#`; `NAVIGATION` added to `Notification_CAT` in D2/D3), and projection to the common-feature intersection between source and target waves. See [`docs/feature_alignment.md`](./docs/feature_alignment.md).

**Full reference:**
Zhang, P., Jung, G., Alikhanov, J., Ahmed, U., & Lee, U. (2024). A Reproducible Stress Prediction Pipeline with Mobile Sensor Data. *Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.* 8(3). https://doi.org/10.1145/3678578.

**Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data?**
The public release on Harvard Dataverse contains only the pre-extracted feature matrices (pkl files), `EsmResponse.csv` (raw ESM labels), and `UserInfo.csv` (participant info). Raw per-participant sensor CSVs are retained internally but are not publicly released. Researchers requiring raw sensor data may contact the authors directly at **[TBD contact email on acceptance]**.

**Is the software used to preprocess/clean/label the data available?**
Yes. The preprocessing conventions, QC thresholds, and alias/feature-alignment decisions are documented in [`preprocessing/`](./preprocessing/). The upstream pipeline code from Zhang et al. (2024) is available at https://doi.org/10.1145/3678578.

---

## 5. Uses

**Has the dataset been used for any tasks already?**
Yes. The dataset was used in the accompanying NeurIPS paper to evaluate the three benchmark tiers (A: personal-history predictability, B: within-wave cross-user transfer, C: cross-wave transfer) across a model inventory comprising tree-based baselines (XGBoost, LightGBM), tabular neural networks (MLP, ResNet, TabNet, SAINT, TabTransformer, FTTransformer, DCN), domain-generalisation methods (IRM, VREx, GroupDRO, MixStyle, MLDG, MASF, Fish, CSD, SagNet), and domain-adaptation methods (DANN, CDAN, DAN, DeepCORAL, MCC, ADDA, MCD, JAN, SHOT, CBST, CGDM).

**What (other) tasks could the dataset be used for?**

- Personalised affect modelling and adaptive user interfaces.
- Transfer learning and domain adaptation in mobile health.
- Individual-difference research (personality, mental-health screening, affect dynamics).
- Circadian-rhythm and temporal-pattern analysis.
- Passive-sensing feature engineering for affective states.
- Fairness and bias analysis in affect-recognition systems.

**Is there anything about the composition or collection that might impact future uses?**

- Cohort: university students in South Korea, limiting broad demographic generalizability.
- D1 collection overlaps with the onset of the COVID-19 pandemic, which may have distorted behavioural baselines.
- GPS longitude displacement and MAC-address anonymisation limit certain location- and proximity-based analyses.
- Polar H10 ECG is only partially available (sub-period subgroups in D1/D2); no ECG in D3.
- PANAS-style affect-word labels are available only in D3; fine-grained affect-word analyses cannot yet be conducted symmetrically across all three waves.

**Are there tasks for which the dataset should not be used?**

- Commercial use of any kind.
- Re-identification of individual participants.
- Surveillance or profiling of individuals.
- Any use outside the scope of the signed Data Use Agreement.

---

## 6. Distribution

**Will the dataset be distributed to third parties outside of the entity on behalf of which the dataset was created?**
Yes. The dataset is released for non-commercial academic research via Harvard Dataverse with gated access.

**How will the dataset be distributed?**
Data files are hosted on **Harvard Dataverse** at **[TBD Dataverse URL on acceptance]**. Users must create a free Dataverse account and agree to the Data Use Agreement before downloading. No manual approval is required — agreement grants immediate access. This GitHub repository hosts documentation, benchmark scaffolding, and metadata separately from the data files.

**When will the dataset be distributed?**
Upon acceptance of the accompanying NeurIPS paper. The dataset will remain accessible on Harvard Dataverse in perpetuity; Dataverse provides long-term preservation guarantees and a persistent DOI.

**Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use?**
Yes, but the final dataset license, repository code license, and Data Use Agreement text are **[TBD on acceptance/public release]**. The full Data Use Agreement will be presented to users on the Harvard Dataverse page prior to download. See [`LICENSE`](./LICENSE) and [`LICENSE-DATA.md`](./LICENSE-DATA.md).

**Have any third parties imposed IP-based or other restrictions on the data?**
No, aside from the IRB-mandated privacy protections already applied (contact-number hashing, MAC-address anonymisation, GPS longitude displacement).

---

## 7. Maintenance

**Who is supporting/hosting/maintaining the dataset?**
**[TBD institution on acceptance]**. See [`MAINTENANCE.md`](./MAINTENANCE.md) for the maintenance plan (contact, issue SLA, versioning).

**How can the owner/curator/manager of the dataset be reached?**
Open a GitHub issue (preferred) or email **[TBD contact email on acceptance]**.

**Will the dataset be updated?**
The D1/D2/D3 release is intended as a stable archival release. Corrective patches (e.g., documentation fixes, additional alias-alignment mappings) may be issued as minor versions on the same Dataverse entry. Any future waves would be released as a separate Dataverse entry and announced via the GitHub repository. See [`CHANGELOG.md`](./CHANGELOG.md) and [`MAINTENANCE.md`](./MAINTENANCE.md).

**Will older versions of the dataset continue to be supported/hosted?**
Yes. Harvard Dataverse retains all published versions of the dataset permanently. Version-specific DOIs are issued so downstream research can cite an exact revision.

**If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?**
Contributions to the code repository are welcome via pull request; see [`CONTRIBUTING.md`](./CONTRIBUTING.md). Requests to contribute supplementary annotations or derived features should be directed to the maintainers via email.

---

## References

- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for Datasets. *Communications of the ACM*, 64(12), 86–92.
- Zhang, P., Jung, G., Alikhanov, J., Ahmed, U., & Lee, U. (2024). A Reproducible Stress Prediction Pipeline with Mobile Sensor Data. *Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.* 8(3). https://doi.org/10.1145/3678578.
