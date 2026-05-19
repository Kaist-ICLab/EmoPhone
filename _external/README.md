# External Reference Code (`_external/`)

This folder contains **vendored third-party reference implementations** that the EmoPhone benchmark references but does not import at run time. The code is preserved here for two reasons:

1. **Audit trail.** When porting an algorithm (e.g. CGDM, ADDA) into [`../src/da_models.py`](../src/da_models.py) we keep a frozen copy of the upstream so that reviewers can diff the in-repo port against the original.
2. **Reproducibility.** The benchmark pins specific commit-level behaviour from these upstreams; the vendored snapshot guards against upstream-side drift breaking comparisons.

## Contents

```
_external/
└── pytorch-domain-adaptation/      # reference adversarial-DA implementation
```

### pytorch-domain-adaptation

- Upstream: <https://github.com/jvanvugt/pytorch-domain-adaptation>
- License: MIT (see [`pytorch-domain-adaptation/LICENSE`](./pytorch-domain-adaptation/LICENSE)).
- Why vendored: provides a minimal reference for the gradient-reversal (`revgrad`) and Wasserstein-distance (`wdgrl`) implementations that informed the design of `DANN` and related adversarial-DA wrappers in [`../src/da_models.py`](../src/da_models.py). Not imported at run time.

### Adding a new vendored upstream

If you need to add another reference implementation:

1. Place it under a subfolder named after the upstream repo (`_external/<repo_name>/`).
2. Keep the upstream's `LICENSE` file intact.
3. Note the commit hash you snapshotted in a short `_external/<repo_name>/SOURCE.md`.
4. Do **not** import from `_external/` in any production module — copy / adapt the relevant code into [`../src/`](../src/) and document the port in [`../DA_Verification/`](../DA_Verification/).
