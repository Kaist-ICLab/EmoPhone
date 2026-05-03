# Authors and Contributors

This file lists the authors and contributors responsible for the **CrossUserDataset** release, aligned with the byline of the accompanying NeurIPS paper.

> **Note.** Names, affiliations, and contact information are held as `[TBD on acceptance]` during peer review. They will be populated on paper acceptance.

---

## Dataset authors

| Name | Affiliation | Role |
|---|---|---|
| [TBD on acceptance — Author 1] | [TBD institution] | Lead author; dataset design and curation |
| [TBD on acceptance — Author 2] | [TBD institution] | Data collection; ESM protocol |
| [TBD on acceptance — Author 3] | [TBD institution] | Benchmark design; Tier A/B/C implementation |
| [TBD on acceptance — Author 4] | [TBD institution] | Preprocessing; feature alignment |
| [TBD on acceptance — Author 5] | [TBD institution] | Ethics oversight; IRB coordination |
| [TBD on acceptance — Principal Investigator] | [TBD institution] | Corresponding author |

Add one entry per author before publication. Author ordering on this file must match the paper byline.

---

## Acknowledgements

### Upstream pipeline

This dataset's preprocessing is built on the reproducible pipeline introduced in:

> Zhang, P., Jung, G., Alikhanov, J., Ahmed, U., & Lee, U. (2024). *A Reproducible Stress Prediction Pipeline with Mobile Sensor Data.* Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 8(3). https://doi.org/10.1145/3678578.

### Benchmark baselines

The benchmark baselines draw on the following open-source libraries and repositories:

- [DomainBed](https://github.com/facebookresearch/DomainBed) — DG protocols.
- [Transfer-Learning-Library (TLL)](https://github.com/thuml/Transfer-Learning-Library) — DA protocols.
- [pytorch-widedeep](https://github.com/jrzaurin/pytorch-widedeep) — SAINT / TabTransformer / FTTransformer implementations.
- [pytorch-tabnet](https://github.com/dreamquark-ai/tabnet) — TabNet.
- [deepctr-torch](https://github.com/shenweichen/DeepCTR-Torch) — DCN.
- Original author releases for MASF, CGDM, DeepCORAL, ADDA, MCD, SHOT, and CBST (see [`domain_adaptation/README.md`](./domain_adaptation/README.md)).

### Participants

We thank the ~328 participants who contributed their time, attention, and passive-sensing data across the three collection waves. Data collection under their informed consent made this release possible.

---

## How to contribute

External contributors are welcome. See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for the contribution workflow. Substantive contributions may be acknowledged in subsequent releases via a dedicated "Contributors" subsection here.
