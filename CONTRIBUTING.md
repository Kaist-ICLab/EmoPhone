# Contributing

Thank you for your interest in contributing to the **CrossUserDataset** repository. This guide describes how to propose changes, what kinds of contributions are welcome, and what review standards apply.

---

## Scope of contributions

### Welcome

- **Documentation improvements** — clarifications, typo fixes, additional examples, broken-link repairs.
- **Benchmark additions** — new baseline / DG / DA methods, new tier variants, new metrics. These must follow the loader / metric contract in [`benchmark/utils/README.md`](./benchmark/utils/README.md) and must not require re-releasing the dataset itself.
- **Analysis notebooks** — additional EDA, domain-specific analyses, reproducibility sanity checks. Add under [`EDA/`](./EDA/) with a short README entry.
- **Bug fixes** — in preprocessing documentation, benchmark code, or EDA utilities.
- **Metadata updates** — Croissant schema tweaks, CITATION.cff updates, checksum regeneration.

### Requires author coordination (email first)

- **Dataset-level changes** — new labels, new waves, new participant recruitment, label corrections, new anonymisation steps. These require IRB review and a new Dataverse revision.
- **Licensing changes** — code and data licenses are TBD until public release. Any license-related change requires author discussion.
- **Schema-breaking changes** — renaming columns, changing pkl tuple structure, dropping labels.

Email **[TBD contact email on acceptance]** before submitting PRs in this category.

---

## Workflow

1. **Open an issue first** describing the change, especially for anything non-trivial. This avoids duplicate work and lets maintainers weigh in on direction.
2. **Fork the repository** and create a topic branch from `main`.
3. **Make the change** in small, reviewable commits. Follow the style conventions below.
4. **Open a pull request** referencing the issue. Fill in the PR template with:
   - Summary of the change.
   - Motivation / related issue.
   - Test plan (what you ran, what you verified).
   - Impact (does it affect schema? benchmark results? reproducibility?).
5. **Respond to review**. We aim to respond within 10 business days; see [`MAINTENANCE.md`](./MAINTENANCE.md).

---

## Style conventions

### Documentation

- Markdown for all prose docs.
- File paths as relative links wherever possible (e.g., `[utils.py](./EDA/utils.py)`).
- Wave IDs: `D1 / D2 / D3` on disk and in public docs; `D-1 / D-2 / D-3` is acceptable as a dict-key convention inside EDA code.
- Label filenames: lowercase (`valence.pkl`, `stress.pkl`, `valenceChange.pkl`, `mental.pkl`). See [`data/README.md`](./data/README.md).

### Code

- Python 3.10 recommended; code must remain compatible with 3.9–3.11.
- Follow PEP 8; run `ruff check` (if added) before committing.
- Type hints for new function signatures.
- No breaking changes to `EDA.utils` public API (`load_esm`, `load_userinfo`, `load_df_X_combined`, `load_and_attach`, etc.) without a matching version bump.

### Commits

- Use clear, short commit subjects (≤ 72 characters). The body should explain the *why*, not the *what*.
- One logical change per commit. Squash trivial fixups before merging.

---

## Benchmark additions — acceptance criteria

To be merged, a new benchmark method must:

1. **Conform to the loader contract** (5-tuple pickle input; see [`benchmark/utils/README.md`](./benchmark/utils/README.md)).
2. **Use the unified training loop** (≤ 50 epochs, patience-based early stopping, fixed seed, 30 Optuna trials on validation AUROC).
3. **Emit output rows** in the canonical schema: `tier, wave_or_source, target, task, model, family, n_train, n_val, n_test, acc, macro_f1, precision, recall, auroc, auroc_std, n_features_after_alignment`.
4. **Include upstream attribution** — original paper citation, upstream implementation link, licence compatibility note.
5. **Reproduce on D1 / D2 / D3** at least for one task on one tier. Summary AUROC should be committed to the matching CSV in `benchmark/results/`.

---

## Reporting security or privacy issues

Do **not** open a public issue for privacy or security concerns. Email **[TBD contact email on acceptance]** directly. See [`RESPONSIBILITY.md`](./RESPONSIBILITY.md) for the incident-response process.

---

## Code of conduct

Contributors are expected to maintain a respectful, inclusive environment. Personal attacks, harassment, and discriminatory language are not tolerated. Project maintainers reserve the right to moderate or close contributions that violate this standard.
