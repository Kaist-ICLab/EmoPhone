# Datasheet for [DATASET_NAME] (D1–D3)

*Following the Datasheet for Datasets framework (Gebru et al., 2021).*  
*Last updated: [RELEASE_DATE]*

---

## 1. Motivation

**For what purpose was the dataset created?**  
The dataset was created to support research in mobile affective computing — specifically, the dense moment-level modeling of affective states (valence, arousal, stress, task disturbance) as they unfold in everyday life. Existing longitudinal mobile sensing benchmarks target broader behavioral or mental-health outcomes at weekly intervals. This dataset fills the gap by combining repeated in-situ self-reports with ecologically valid passive sensing across three consecutive annual waves, enabling both within-wave and cross-wave benchmarking.

**Who created the dataset and on behalf of which entity?**  
[AUTHOR_NAMES], [INSTITUTION]

**Who funded the creation of the dataset?**  
[FUNDING]

---

## 2. Composition

**What do the instances represent?**  
Each instance is a time-stamped ESM (Experience Sampling Method) response from a study participant, linked to a window of passively collected sensor data from their smartphone and wearable device. A single instance captures a participant's momentary affective state (self-reported labels) and the sensor context surrounding that moment.

**How many instances are there in total (across all waves)?**

| Wave | Participants (after QC) | ESM responses (after QC) |
|---|---|---|
| D1 | 92 | 10,259 |
| D2 | 99 | 21,042 |
| D3 | 106 | 21,838 |
| **Total** | **297 unique** | **53,139** |

Note: All participants were recruited independently for each wave. No participant appears in more than one wave, so the total of 297 individuals is also the unique individual count.

**Does the dataset contain all possible instances, or is it a sample?**  
The dataset represents a sample of university students in South Korea recruited via campus bulletin boards. It is not a random population sample. The cohort is intentionally relatively homogeneous to reduce broad demographic confounds in cross-user and cross-dataset analyses.

**What data does each instance consist of?**  
Each instance consists of:
- **Self-report labels** (shared core): Valence, Arousal, Stress, Task Disturbance (7-point scale, −3 to +3 for D1/D2; 0 to +6 for Stress/Disturbance in D3)
- **D3 additional labels**: 8 PANAS-style affect-word intensity items (Happy, Relaxed, Cheerful, Content, Sad, Anxious, Depressed, Angry; 0 to +6)
- **Pre-extracted feature matrices** (one pkl file per label per wave): ~8,000 features for D1 (exact: 8,037); ~10,100 for D2 (exact: 10,122); ~10,600 for D3 (exact: 10,581). Features are derived from passive smartphone sensing (app usage, notifications, call/message events, screen events, GPS location, activity recognition, Wi-Fi scan, Bluetooth scan, battery, data traffic, keystroke interactions for D2/D3) and wearable sensing (Fitbit Inspire HR heart rate, step count, calories; Polar H10 ECG-derived features for D1/D2 subsets). See `data/schema.md` for the full feature naming convention.
- **Individual characteristics** (collected once per wave): age, gender, Big Five personality (K-BFI-15), perceived stress (PSS-10), general health (GHQ-12), depressive symptoms (PHQ-9; D1 and D2 only). These are included as static columns in every pkl file under the `PIF#` prefix.

> **Note on raw sensor data:** Raw per-participant sensor CSV files are not included in the public release. Only the pre-extracted pkl feature matrices, `EsmResponse.csv`, and `UserInfo.csv` are released on Harvard Dataverse. Researchers requiring raw sensor data for their specific use case may contact the authors at [CONTACT_EMAIL].

**Is there a label or target associated with each instance?**  
Yes. The primary prediction targets are the shared core labels: valence, arousal, stress, and disturbance. D3 additionally includes 8 affective-word intensity labels.

**Is any information missing from individual instances?**  
Yes. Sensor data availability varies across participants due to:
- Device differences (Android version, manufacturer)

- Polar H10 chest sensor was assigned to participant subgroups in rotating sub-periods (D1, D2) and not collected in D3

**Are relationships between instances made explicit?**  
Yes. Each ESM response is linked to a participant identifier (`pcode`) and a trigger timestamp, enabling alignment with sensor time series. Temporal ordering within and across waves is preserved via Unix timestamps (UTC milliseconds).

**Are there recommended data splits?**  
Yes. We provide a three-tier benchmark structure following the cross-validation conventions established in Zhang et al. (2024):
- **Tier A** (personal predictability): temporal split within each user's history
- **Tier B** (cross-user transfer): leave-one-subject-out (LOSO) split within a wave
- **Tier C** (cross-dataset transfer): train on one wave, test on another

**Does the dataset contain data that might be considered confidential?**  
The dataset contains passively collected behavioral data (app usage, location, communication patterns) from real people. All data were collected under IRB approval with written informed consent. Prior to release, the following anonymization steps were applied:
- Contact numbers (calls, messages): MD5-hashed
- Bluetooth and Wi-Fi MAC addresses: replaced with UUID
- GPS longitude: offset by a fixed random constant per participant

**Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?**  
No. The dataset does not contain free-text communications or media content.

**Does the dataset relate to people?**  
Yes. All data were collected from human participants (university students).

**Might the dataset expose people to harm or legal action?**  
Risk is low given the anonymization applied. GPS latitude is unmodified; re-identification of participants via location patterns is a residual risk addressed by restricting use to non-commercial academic research in the Data Use Agreement.

---

## 3. Collection Process

**How was the data associated with each instance acquired?**  
- ESM labels: self-reported via a custom Android application, prompted at random intervals during waking hours (10 AM–10 PM)
- Smartphone sensor data: passively collected in the background by the same Android application, uploaded to a secure server
- Wearable data: Fitbit Inspire HR synced via the Fitbit API; Polar H10 ECG synced via Bluetooth to the Android app

**What mechanisms or procedures were used to collect the data?**  
A custom Android application managed ESM prompt delivery, sensor data collection, and data upload. ESM prompts expired after 10 minutes if unanswered. Participants were required to keep the app and wearables active for at least 12 hours per day. The research team monitored participation every 2–3 days.

**Who was involved in the data collection process?**  
Research team members at [INSTITUTION]. Participants were university students recruited via online campus bulletin boards. Participation was voluntary and compensated (~100 USD upon completion).

**Over what timeframe was the data collected?**

| Wave | Collection Period |
|---|---|
| D1 | February 7 – April 2, 2020 (~30 days) |
| D2 | December 7, 2020 – January 27, 2021 (~30 days) |
| D3 | November 23, 2021 – January 11, 2022 (~28 days) |

Note: D1 overlaps with the onset of the COVID-19 pandemic in South Korea. D2 and D3 were collected during the pandemic. This contextual difference may influence behavioral patterns and should be considered in analyses.

**Were any ethical review processes conducted?**  
Yes. All procedures were approved by the Institutional Review Board (IRB) at [INSTITUTION], approval number [IRB_NUMBER]. All participants provided written informed consent before participation.

**Did the individuals in question consent to the collection and use of their data?**  
Yes. Written informed consent was obtained at a one-hour in-person briefing session before the study began. Participants were informed of the study's purpose, procedures, and their right to withdraw at any time.

**Has an analysis of the potential impact of the dataset and its use on data subjects been conducted?**  
[IMPACT_ASSESSMENT]

---

## 4. Preprocessing / Cleaning / Labeling

**Was any preprocessing/cleaning/labeling of the data done?**  
Yes. Data preprocessing follows the reproducible pipeline introduced in Zhang et al. (2024), which defines eight stages covering ESM quality control, feature extraction, feature preparation, feature selection, data splitting, oversampling/undersampling, model training, and model evaluation.

Specific QC steps applied to this dataset:

- **ESM-level QC**: Expired (unanswered) prompts excluded. Responses submitted after the 10-minute expiry window excluded. [QC_ADDITIONAL_FILTERS]
- **Participant-level QC**: Participants below the minimum valid ESM response threshold excluded (thresholds vary per wave; see `docs/quality_control.md`).
- **Sensor QC**: Polar H10 ECG data is available only for participants during their assigned sub-period in D1 and D2; ECG-derived features reflect this partial coverage.
- **Label harmonization**: Stress and Disturbance scales differ between D1/D2 (−3 to +3) and D3 (0 to +6). Normalization procedures are described in `docs/dataset_documentation.md`.
- **Feature extraction**: Statistical features extracted across multiple time windows (current value, 15-min immediate past, epoch-level daily, full daily), following the feature extraction conventions in Zhang et al. (2024).

**Full reference:** Zhang, P., Jung, G., Alikhanov, J., Ahmed, U., & Lee, U. (2024). A Reproducible Stress Prediction Pipeline with Mobile Sensor Data. *Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.*, 8(3). https://doi.org/10.1145/3678578

**Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data?**  
The public release on Harvard Dataverse contains only the pre-extracted feature matrices (pkl files), `EsmResponse.csv` (raw ESM labels), and `UserInfo.csv` (participant info). Raw per-participant sensor CSV files are not publicly released. Researchers requiring raw sensor data may contact the authors directly at [CONTACT_EMAIL].

**Is the software used to preprocess/clean/label the data available?**  
Yes. Preprocessing code is available in [`/code/preprocessing`](./code/preprocessing/). The upstream pipeline code from Zhang et al. (2024) is available at https://doi.org/10.1145/3678578.

---

## 5. Uses

**Has the dataset been used for any tasks already?**  
Yes. The dataset was used in the associated NeurIPS paper to evaluate three benchmark tiers: personal-history predictability (Tier A), within-dataset cross-user transfer (Tier B), and cross-dataset transfer across waves (Tier C). Preprocessing and baseline modeling follows the pipeline of Zhang et al. (2024), with baseline models including Random Forest, XGBoost, LightGBM, and SVM.

**What (other) tasks could the dataset be used for?**  
- Personalized affect modeling and adaptive user interfaces
- Transfer learning and domain adaptation in mobile health
- Individual difference research (personality, mental health, affect dynamics)
- Circadian rhythm and temporal pattern analysis
- Passive sensing feature engineering for affective states
- Fairness and bias analysis in affect recognition systems

**Is there anything about the composition or collection that might impact future uses?**  
- The cohort is university students in South Korea, limiting demographic generalizability
- D1 was collected during a major societal disruption (COVID-19 onset), which may have distorted behavioral baselines
- GPS longitude displacement and MAC address anonymization limit certain location- and proximity-based analyses
- The Polar H10 ECG data is only available for subsets of D1 and D2 participants; analyses relying on it are not reproducible for D3

**Are there tasks for which the dataset should not be used?**  
- Commercial use of any kind
- Re-identification of individual participants
- Surveillance or profiling of individuals
- Any use outside the scope of the signed Data Use Agreement

---

## 6. Distribution

**Will the dataset be distributed to third parties outside of the entity on behalf of which the dataset was created?**  
Yes. The dataset is released for non-commercial academic research via Harvard Dataverse with gated access.

**How will the dataset be distributed?**  
Data files are hosted on **Harvard Dataverse** at [DATAVERSE_URL]. Users must create a free Dataverse account and agree to the Data Use Agreement before downloading. No manual approval is required — agreement to terms grants immediate access. This GitHub repository hosts documentation, benchmark code, and metadata separately from the data files.

**When will the dataset be distributed?**  
Upon acceptance of the associated NeurIPS paper. [RELEASE_DATE]. The dataset will remain accessible on Harvard Dataverse in perpetuity; Harvard Dataverse provides long-term preservation guarantees and a persistent DOI.

**Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use?**  
Yes. The dataset is released under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**, restricting use to non-commercial academic research purposes. The full Data Use Agreement is presented to users on the Harvard Dataverse page prior to download. Code and documentation in this GitHub repository are released under the **MIT License**.

**Have any third parties imposed IP-based or other restrictions on the data?**  
No, aside from the IRB-mandated privacy protections already applied (contact number hashing, MAC address anonymization, GPS longitude displacement).

---

## 7. Maintenance

**Who is supporting/hosting/maintaining the dataset?**  
[INSTITUTION]. Contact: [CONTACT_EMAIL]

**How can the owner/curator/manager of the dataset be reached?**  
Open a GitHub issue or email [CONTACT_EMAIL].

**Will the dataset be updated?**  
[FUTURE_WAVES]

**Will older versions of the dataset continue to be supported/hosted?**  
[VERSIONING_POLICY]

**If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?**  
Contributions to the code repository are welcome via pull request. Requests to contribute supplementary annotations or derived features should be directed to the maintainers via email.

---

## References

- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for Datasets. *Communications of the ACM*, 64(12), 86–92.
- Zhang, P., Jung, G., Alikhanov, J., Ahmed, U., & Lee, U. (2024). A Reproducible Stress Prediction Pipeline with Mobile Sensor Data. *Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.*, 8(3). https://doi.org/10.1145/3678578
