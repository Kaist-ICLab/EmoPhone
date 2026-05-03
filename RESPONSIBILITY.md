# Author Responsibility Statement

*Required by the NeurIPS Datasets and Benchmarks Track.*

---

## Statement of responsibility

The authors of this dataset and accompanying codebase confirm the following:

1. **Rights and licensing.**
   - We have the right to release the dataset and the accompanying code under the final licenses declared before public release; the current license files are placeholders (see [`LICENSE`](./LICENSE) and [`LICENSE-DATA.md`](./LICENSE-DATA.md)).
   - Participants gave written informed consent covering research use, third-party sharing for academic research under a Data Use Agreement, and long-term archival on Harvard Dataverse.

2. **Responsibility for violations.**
   - **The authors bear all responsibility in case of violation of rights**, including but not limited to privacy rights, data-protection regulations, intellectual-property rights, or the terms of the Data Use Agreement. We accept liability for any claims arising from the release of this dataset.
   - Users of the dataset are bound by the Data Use Agreement accepted at download time; user-side violations do not transfer liability back to the authors provided the authors have complied with their disclosure obligations.

3. **License confirmation.**
   - Dataset license, repository code license, and project-specific DUA terms are **TBD** and will be finalized before public release.
   - Until the final license files are populated, this repository should be treated as pre-release research material.

4. **Ethics and IRB oversight.**
   - All data collection was approved by the Institutional Review Board at **[TBD institution on acceptance]** under approval number **[TBD on acceptance]**.
   - All participants provided written informed consent. Compensation (~100 USD) was provided on study completion.
   - Anonymisation transformations (MD5 hashing of contact identifiers, UUID replacement of Wi-Fi/Bluetooth MAC addresses, per-participant random displacement of GPS longitude) were applied before release. Residual re-identification risk is mitigated by restricting use to non-commercial academic research under the DUA.

5. **Hosting and long-term availability.**
   - Dataset files are hosted on **Harvard Dataverse**, which provides long-term preservation guarantees and a persistent DOI.
   - Code, documentation, and metadata are hosted on GitHub under this repository. Mirrors may be maintained by the authors' institution.

6. **Maintenance commitment.**
   - The authors commit to maintaining the dataset and code for at least **3 years** following the initial publication. See [`MAINTENANCE.md`](./MAINTENANCE.md) for the full maintenance plan, including issue-response service-level expectations and versioning policy.

7. **Attribution and citation.**
   - The dataset builds on the reproducible preprocessing pipeline introduced by Zhang et al. (2024), DOI: 10.1145/3678578. We cite this work prominently in the accompanying NeurIPS paper and in [`CITATION.cff`](./CITATION.cff).
   - All third-party models, libraries, and algorithms used in the benchmark (DomainBed, Transfer-Learning-Library, pytorch-widedeep, pytorch-tabnet, deepctr-torch, XGBoost, LightGBM, individual DG/DA baseline repositories) are cited in [`benchmark/README.md`](./benchmark/README.md) and the paper's Appendix C.

8. **Incident response.**
   - If a participant or third party notifies us of a privacy concern or rights violation, we will:
     - Acknowledge the report within 10 business days.
     - Investigate and, if warranted, temporarily or permanently restrict access to the affected data via the Dataverse page.
     - Publish a post-mortem and update the [`CHANGELOG.md`](./CHANGELOG.md) when resolution is complete.

---

## Contact

Reports, questions, or disputes should be directed to **[TBD contact email on acceptance]** or filed as a GitHub issue on this repository.
