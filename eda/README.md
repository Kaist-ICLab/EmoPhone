# Exploratory Data Analysis (EDA)

This folder contains the current EDA notebooks and shared helpers used to characterize the three-wave dataset.

> **Naming convention.** On-disk folders use the released form `D1 / D2 / D3` (as documented in [`../data/README.md`](../data/README.md)). Inside the notebooks and `utils.py`, the internal keys `D-1 / D-2 / D-3` are used as dict keys and for figure labels. The `WAVE_TO_DIR` mapping in [`utils.py`](./utils.py) bridges the two. Keep the internal form when writing dict keys in new notebooks; keep the released form when writing filesystem paths.

These notebooks are for dataset characterization and figure generation, not benchmark training/evaluation. For modeling benchmarks, see [`../benchmark/`](../benchmark/).

## Files

| File | Description |
|---|---|
| `utils.py` | Shared constants, loaders, and helpers used by all notebooks. Exports: wave color constants, label category lists (`SHARED_LABELS`, `W1_ONLY`, `W2_ONLY`, `W3_ONLY`, `ALL_LABELS`, `COVERAGE_BY_WAVE`), `load_esm()`, `load_userinfo()`, `load_wave_esm_userinfo()`, `load_df_X_combined()`, `normalize_label_series()`, `add_study_day()`, `get_data_root()`, `set_paper_style()` |
| `overview.ipynb` | Dataset overview: participant flow diagram, age/gender demographics, ESM time-of-day distributions, stress drift over participation time, and Big Five trait profiles for low vs. high affect groups |
| `labels.ipynb` | Label characterization: availability matrix, shared/wave-specific label distributions, stress distribution by wave, shared-label and full correlation matrices, high-label ratios, and class balance |
| `density.ipynb` | Participation density: per-participant ESM response counts, daily response rate over study days, per-wave and combined activity heatmaps (hour × study day), and prompt-to-response delay |
| `sensors.ipynb` | Sensor coverage: binary availability matrix per wave, sensor modality matrix, and sensor/questionnaire coverage summary by category |
| `psych_affect.ipynb` | Psychological baselines and dataset shift: PSS-10/GHQ-12 distributions, Big Five radar chart, PCA/UMAP/t-SNE projections of per-user feature profiles, feature distribution drift, concept shift line charts (feature tertile vs. % high label), and conditional shift boxplots |
| `variability.ipynb` | Individual variability: variance decomposition by person/day, ICC per label, representative four-week stress trajectories, and cross-user covariate/conditional/label/concept shift magnitudes per wave |

## Data Path Resolution

All notebooks use `get_data_root()` from `utils.py`. Resolution order:

1. `DATA_ROOT` environment variable (if set)
2. Nearest ancestor directory containing `data/`
3. Fallback path relative to this folder (`../../data`)

Expected wave directories:

```text
${DATA_ROOT}/D1
${DATA_ROOT}/D2
${DATA_ROOT}/D3
```

Core files expected by loaders:

- `${DATA_ROOT}/D*/EsmResponse.csv`
- `${DATA_ROOT}/D*/UserInfo.csv`

Legacy sensor-feature pickle files expected by `load_df_X_combined()`:

- `${DATA_ROOT}/D1/stress_binary_personal-full_D#2.pkl`
- `${DATA_ROOT}/D2/stress_binary_personal-full_D#3.pkl`
- `${DATA_ROOT}/D3/stress_binary_personal-full.pkl`

> These `stress_binary_personal-full*.pkl` files coexist with the per-label pickles (`{label}.pkl`) documented in [`../data/README.md`](../data/README.md). They predate the release layout and are retained because a few EDA notebooks still consume them. See `utils.py` for the authoritative path list.

Additional files used by trait-profile cells in `overview.ipynb`:

- `${DATA_ROOT}/D1/full/{valence|arousal|stress|disturbance}_personal-full*.pkl`
- `${DATA_ROOT}/D2/full/{valence|arousal|stress|disturbance}_personal-full*.pkl`
- `${DATA_ROOT}/D3/full/{valence|arousal|stress|disturbance}_personal-full*.pkl`

## Setup

```bash
cd eda
pip install -r requirements.txt
```

Optional explicit data-root override:

```bash
export DATA_ROOT="/absolute/path/to/data"
```

## Run

Launch from the `eda` directory so `import utils` works as expected:

```bash
cd eda
jupyter lab
```

Open and run any notebook:

- `overview.ipynb`
- `labels.ipynb`
- `density.ipynb`
- `sensors.ipynb`
- `psych_affect.ipynb`
- `variability.ipynb`

## Notebook Visual Inventory

### `overview.ipynb`
| Visual | Insight |
|---|---|
| Participant flow diagram | Recruited → final sample per wave, with excluded counts |
| Age distribution + gender composition | Combined age histogram and per-wave gender breakdown |
| Per-wave demographics table | Mean age, SD, and M/F counts per wave |
| ESM time-of-day distribution | When participants respond (10 AM–10 PM window) |
| Hourly ESM activity heatmap | Normalized hourly proportions across waves |
| Stress drift — per-user aligned | Stress label proportions over participation time, split by gender, for each wave and combined |
| Trait profile (low vs. high groups) | Big Five radar plot comparing low vs. high affect groups for valence, arousal, disturbance, stress |
| Trait profile (z-normalized) | Same comparison with per-wave z-score standardization |

### `labels.ipynb`
| Visual | Insight |
|---|---|
| Label coverage matrix | Which labels are available per wave; shared labels highlighted |
| Shared-label distributions | Histograms of valence, arousal, stress, disturbance across waves (D3 scale normalized) |
| Stress distribution by wave | Focused line plot of stress score distributions |
| Non-shared label distributions | Distributions for D1-only (Change), D2-only (Changed Valence/Arousal), D3-only (PANAS words) |
| Shared label distribution profiles | Side-by-side panel for all four shared labels |
| Shared-label correlation matrices | Pearson inter-label correlations per wave for the four core labels |
| Full label correlation heatmaps | Expanded per-wave correlation including wave-specific labels |
| High-label ratio by shared label | Proportion of HIGH (>0) responses per label per wave |
| High vs. low class balance (stacked bars) | HIGH/LOW split proportions per shared label per wave |

### `density.ipynb`
| Visual | Insight |
|---|---|
| Per-participant ESM count | Response count dispersion across participants per wave |
| Daily response rate by study day | Per-participant ESM rate over the study period with week markers |
| Per-wave activity heatmaps | ESM response intensity by hour-of-day × study day, one heatmap per wave |
| Combined activity heatmap | Same aggregated across all three waves |
| Prompt-to-response delay | Distribution of time between prompt delivery and response |

### `sensors.ipynb`
| Visual | Insight |
|---|---|
| Sensor availability matrix | Which smartphone and wearable sensors are present in D1/D2/D3 |
| Sensor modality matrix | Key hardware differences; highlights Polar H10 removal in D3 and keystroke addition in D2 |
| Sensor and questionnaire coverage by category | Modality availability and granularity summary per sensor group |

### `psych_affect.ipynb`
| Visual | Insight |
|---|---|
| PSS-10 distribution | Perceived stress baseline scores per wave |
| GHQ-12 distribution | General mental health baseline scores per wave |
| Big Five radar chart | Mean personality trait profiles per wave |
| PCA scatter (per-user means) | Per-user feature profiles in 2D with 95% confidence ellipses per wave |
| UMAP projection | Non-linear 2D projection of participant profiles; reveals wave clustering |
| t-SNE projection | Alternative non-linear projection of participant profiles |
| Feature distribution drift | KDE plots of representative sensor features across waves |
| Concept shift line charts | For three sensor features: feature tertile (low/mid/high) vs. % high-stress label per wave — shows P(Y\|X) changes across waves |
| Conditional shift boxplots | P(X\|Y) shift — feature distribution split by stress class per wave |

### `variability.ipynb`
| Visual | Insight |
|---|---|
| Variance decomposition (bar chart) | Between-person, within-person between-day, and residual variance per label |
| ICC per label | Intra-class correlation showing how much variance is attributable to person identity |
| Representative four-week trajectories | Individual daily mean stress ± 1 SD for archetypical users |
| Cross-user shift magnitude (2×2 panel) | Covariate, conditional, label, and concept shift magnitudes within each wave, quantifying how much users differ from each other |

---