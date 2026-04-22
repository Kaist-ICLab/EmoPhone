# Ethics, Governance, and Privacy

This document consolidates the ethics story behind the CrossUserDataset release. For the formal datasheet equivalents, see [`../DATASHEET.md`](../DATASHEET.md) §§ 2–3 and § 7. For the author responsibility statement, see [`../RESPONSIBILITY.md`](../RESPONSIBILITY.md).

---

## 1. Ethics and governance

- **IRB approval.** All data-collection procedures were reviewed and approved by the Institutional Review Board at **[TBD institution on acceptance]** under approval number **[TBD on acceptance]**.
- **Informed consent.** Written informed consent was obtained from every participant in a one-hour in-person briefing before the study began. The consent form covered:
  - Purpose of the study and high-level research goals.
  - What sensors and labels would be collected and at what frequency.
  - Data-sharing plan (academic release via a gated Dataverse repository).
  - Anonymisation steps that would be applied.
  - The participant's right to withdraw at any time.
  - Compensation (~100 USD) on study completion.
  An English translation of the consent form is provided at [`consent_form_en.md`](./consent_form_en.md) (placeholder — final text populated on release).
- **Right to withdraw.** Participants could withdraw during the study; data from withdrawn participants was deleted from the active collection.
- **Voluntariness.** Participation was fully voluntary and compensated. No participant's academic standing or coursework was linked to their participation.

---

## 2. Privacy protection

Prior to public release, the following anonymisation transformations were applied to the data:

| Transformation | Applied to | Purpose |
|---|---|---|
| **MD5 hashing** | Contact numbers in call / message logs | Prevent direct identification of communication partners |
| **UUID replacement** | Wi-Fi BSSIDs and Bluetooth MAC addresses | Prevent cross-referencing with external device databases |
| **Per-participant longitude offset** | GPS longitude | Preserve relative-movement structure while removing absolute location |
| **QC filtering** | Structurally invalid rows; streams below coverage thresholds | Remove records unsuitable for modelling |

Latitude is retained unmodified. Residual re-identification risk via location patterns is addressed by:

- Restricting distribution to Harvard Dataverse with mandatory DUA acceptance before download.
- Prohibiting re-identification attempts and commercial use in the DUA.
- Not releasing raw sensor CSVs (only pre-extracted features, ESM labels, and participant metadata).

---

## 3. Data Use Agreement

Access to the data requires agreement to the DUA presented on the Harvard Dataverse landing page. The DUA restricts use to non-commercial academic research and imposes additional conditions (no re-identification, no surveillance, no redistribution of raw data, citation requirements). See [`../LICENSE-DATA.md`](../LICENSE-DATA.md) for the summary and the Dataverse page for the authoritative text.

---

## 4. Risk register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Re-identification via location pattern | Low | High | Longitude displacement; restrict to academic research via DUA; no raw GPS released |
| Re-identification via device cross-reference | Very low | Medium | MAC address UUID replacement; hashed contact numbers |
| Re-identification via rare demographics | Low | Medium | Homogeneous cohort; no free-text; `PIF#` features are coarse (BFI subscales, age in years, gender M/F) |
| Unauthorised commercial use | Low | Medium | Final license / DUA terms TBD; intended DUA will prohibit misuse |
| Surveillance / profiling | Low | High | DUA prohibits; incident reporting channel in [`../RESPONSIBILITY.md`](../RESPONSIBILITY.md) |
| Inference of mental-health status | Medium | Medium | No individual-level predictions shipped; paper discusses limitations of the underlying labels; no direct PHI leakage |

### Residual risks we acknowledge

- **Re-identification is not zero.** Even with anonymisation, a determined adversary with auxiliary data could attempt re-identification. The DUA is the legal mechanism for deterring this; technical mitigations reduce but do not eliminate it.
- **Behavioural inference.** Derived features can in principle be used to infer sensitive attributes the participants did not explicitly share. This is a general concern with passive-sensing data; the dataset is not unusual in this regard, but we flag it explicitly.
- **D3 PANAS labels are self-reported.** Self-report measures of depressive / anxious states are clinical proxies, not diagnoses. Do not use the dataset to label individuals as clinically affected.

---

## 5. Broader impact

- **Positive impact.** The release supports research into personalised affect modelling, generalisable mobile-health analytics, and methods for cross-user / cross-wave transfer — areas where open benchmarks are scarce.
- **Potential for misuse.** Affect prediction from passive sensing could be misused for surveillance, coercive labour management, or discriminatory profiling. The DUA prohibits such uses; users bear responsibility for compliance. We encourage researchers to publish fairness analyses alongside affect models and to reflect on deployment contexts.

---

## 6. Incident response

Concerns about ethics or privacy should be reported per [`../RESPONSIBILITY.md`](../RESPONSIBILITY.md) § Incident response. The authors will acknowledge within 10 business days and, if warranted, temporarily or permanently restrict access to the affected data.
