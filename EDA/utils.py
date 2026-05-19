"""Common helpers shared across EDA notebooks.

This is the single helper module imported by every notebook in this folder.
It defines:

* Constants used for plotting (`COLORS`, `WAVE_LABELS`, `WAVE_KEYS`, ...).
* Loaders for ESM/UserInfo CSVs and the per-wave sensor pickles.
* Higher-level loaders that bundle the common "load → 28-day window →
  split-by-wave" preparation that every notebook performs.
* Small analytical helpers (label correlation, per-user means).

Wave-key conventions:

* Internal dict keys / display labels: ``D-1 / D-2 / D-3``.
* On-disk folder names (per ``../data/README.md``): ``D1 / D2 / D3``.
* Bridge: :data:`WAVE_TO_DIR`.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────
GENDER_COLORS: dict[str, str] = {"M": "#4A90E2", "F": "#FF69B4"}
COLORS: dict[str, str] = {"D-1": "#2C7BB6", "D-2": "#D7191C", "D-3": "#1A9641"}
WAVE_LABELS: list[str] = ["D-1 (2020)", "D-2 (2021)", "D-3 (2022)"]
WAVE_KEYS: list[str] = ["D-1", "D-2", "D-3"]
WAVE_TO_DIR: dict[str, str] = {"D-1": "D1", "D-2": "D2", "D-3": "D3"}

TRAIT_COLS: list[str] = [
    "age",
    "openness",
    "conscientiousness",
    "neuroticism",
    "extraversion",
    "agreeableness",
    "pss10",
    "ghq12",
]
LABEL_NAMES: list[str] = ["valence", "arousal", "stress", "disturbance"]

SHARED_LABELS: list[str] = ["Valence", "Arousal", "Stress", "Task Disturbance"]
W1_W2_ONLY: list[str] = ["Attention", "Mental", "Duration"]
W1_ONLY: list[str] = ["Change"]
W2_ONLY: list[str] = ["Changed Valence", "Changed Arousal"]
W3_ONLY: list[str] = [
    "Happy",
    "Relaxed",
    "Cheerful",
    "Content",
    "Sad",
    "Angry",
    "Anxious",
    "Depressed",
]
ALL_LABELS: list[str] = SHARED_LABELS + W1_W2_ONLY + W1_ONLY + W2_ONLY + W3_ONLY

COVERAGE_BY_WAVE: dict[str, list[str]] = {
    "D-1": SHARED_LABELS + W1_W2_ONLY + W1_ONLY,
    "D-2": SHARED_LABELS + W1_W2_ONLY + W2_ONLY,
    "D-3": SHARED_LABELS + W3_ONLY,
}

# Default ESM/sensor metadata column names — used by the high-level loaders.
ESM_DATASET_COL = "Wave"
ESM_PID_COL = "Pcode"
ESM_TS_COL = "ResponseTime"
SENSOR_DATASET_COL = "META#dataset"
SENSOR_PID_COL = "PIF#participantID"
SENSOR_TS_COL = "PIF#timestamp"


# ── Style ──────────────────────────────────────────────────────────────────────
def set_paper_style() -> None:
    """Set matplotlib/seaborn publication-style theme."""
    import seaborn as sns

    sns.set_theme(
        context="paper",
        style="white",
        font="serif",
        rc={
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


# ── Timestamp parsing ──────────────────────────────────────────────────────────
def parse_timestamp(series: pd.Series | pd.DataFrame) -> pd.Series:
    """Auto-detect epoch (s or ms) vs string timestamps and convert to datetime.

    Mixed-type or all-null inputs return an empty datetime series.
    """
    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 0:
            return pd.Series(dtype="datetime64[ns]")
        series = series.iloc[:, 0]

    non_null = series.dropna()
    if non_null.empty:
        return pd.Series(dtype="datetime64[ns]", index=series.index)

    if pd.api.types.is_numeric_dtype(non_null):
        unit = "ms" if non_null.iloc[0] > 1e10 else "s"
        return pd.to_datetime(series, unit=unit, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def to_utc_local(series: pd.Series, tz: str = "Asia/Seoul") -> pd.Series:
    """Parse timestamps to UTC and convert to a target timezone."""
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(tz)


# ── Path helpers ───────────────────────────────────────────────────────────────
def get_data_root() -> str:
    """Return the project ``data/`` directory.

    Resolution order:
        1. ``DATA_ROOT`` environment variable (if set).
        2. Nearest ancestor directory containing ``./data``.
        3. Fallback to ``../../data`` relative to this file.
    """
    env_root = os.environ.get("DATA_ROOT")
    if env_root:
        return os.path.abspath(env_root)

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data"
        if candidate.is_dir():
            return str(candidate)

    return str((here.parent / ".." / ".." / "data").resolve())


_PKL_RELATIVE: dict[str, str] = {
    "D-1": os.path.join("D1", "stress_binary_personal-full_D#2.pkl"),
    "D-2": os.path.join("D2", "stress_binary_personal-full_D#3.pkl"),
    "D-3": os.path.join("D3", "stress_binary_personal-full.pkl"),
}


def get_pkl_paths(data_root: str | None = None) -> dict[str, str]:
    """Return wave key → legacy sensor-pickle file path."""
    root = data_root or get_data_root()
    return {k: os.path.join(root, v) for k, v in _PKL_RELATIVE.items()}


# ── ESM / UserInfo loaders ─────────────────────────────────────────────────────
_ESM_RENAME_MAP: dict[str, str] = {
    "pcode": "Pcode",
    "responseTime": "ResponseTime",
    "actualTriggerTime": "TriggerTime",
    "intendedTriggerTime": "TriggerTime",
    "reactionTime": "ReactionTime",
    "valence": "Valence",
    "arousal": "Arousal",
    "stress": "Stress",
    "disturbance": "Task Disturbance",
    "duration": "Duration",
    "attention": "Attention",
    "mental": "Mental",
    "valenceChange": "Changed Valence",
    "arousalChange": "Changed Arousal",
    "happy": "Happy",
    "relaxed": "Relaxed",
    "cheerful": "Cheerful",
    "content": "Content",
    "sad": "Sad",
    "anxious": "Anxious",
    "depressed": "Depressed",
    "angry": "Angry",
    "change": "Change",
}

_USERINFO_RENAME_MAP: dict[str, str] = {
    "pcode": "Pcode",
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "neuroticism": "Neuroticism",
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "age": "Age",
    "gender": "Gender",
}


def _resolve_data_root(data_root: str | None) -> str:
    return data_root if data_root is not None else os.environ.get("DATA_ROOT", "")


def load_esm(
    wave_key: str,
    data_root: str | None = None,
    wave_to_dir: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Load and normalise ``EsmResponse.csv`` for one wave."""
    root = _resolve_data_root(data_root)
    dirs = wave_to_dir or WAVE_TO_DIR

    fpath = os.path.join(root, dirs[wave_key], "EsmResponse.csv")
    df = pd.read_csv(fpath)
    if "actualTriggerTime" in df.columns and "intendedTriggerTime" in df.columns:
        df = df.drop(columns=["intendedTriggerTime"])
    df = df.rename(columns={k: v for k, v in _ESM_RENAME_MAP.items() if k in df.columns})
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    for col in ("ResponseTime", "TriggerTime", "ReactionTime"):
        if col in df.columns:
            df[col] = parse_timestamp(df[col])
    for col in SHARED_LABELS:
        if col in df.columns:
            df[col] = normalize_label_series(df[col])
    df["Wave"] = wave_key
    return df


def load_userinfo(
    wave_key: str,
    data_root: str | None = None,
    wave_to_dir: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Load and normalise ``UserInfo.csv`` for one wave."""
    root = _resolve_data_root(data_root)
    dirs = wave_to_dir or WAVE_TO_DIR

    fpath = os.path.join(root, dirs[wave_key], "UserInfo.csv")
    df = pd.read_csv(fpath)
    df = df.rename(columns={k: v for k, v in _USERINFO_RENAME_MAP.items() if k in df.columns})
    df["Wave"] = wave_key
    return df


def load_wave_esm_userinfo(
    wave_keys: Sequence[str] = WAVE_KEYS,
    data_root: str | None = None,
    wave_to_dir: Mapping[str, str] | None = None,
    include_userinfo: bool = True,
    include_study_day: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """Load ESM/UserInfo tables for each wave and build a combined ESM DataFrame.

    Returns
    -------
    (esm_by_wave, userinfo_by_wave, esm_all)
    """
    root = _resolve_data_root(data_root)
    dirs = wave_to_dir or WAVE_TO_DIR

    esm_by_wave = {k: load_esm(k, root, dirs) for k in wave_keys}
    userinfo_by_wave = (
        {k: load_userinfo(k, root, dirs) for k in wave_keys} if include_userinfo else {}
    )

    non_empty = [df for df in esm_by_wave.values() if not df.empty]
    esm_all = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    if include_study_day and not esm_all.empty and "ResponseTime" in esm_all.columns:
        esm_all = add_study_day(esm_all)
    return esm_by_wave, userinfo_by_wave, esm_all


# ── Date alignment ─────────────────────────────────────────────────────────────
def align_dates_relative(
    df: pd.DataFrame,
    group_cols: str | list[str],
    date_col: str = "ResponseTime",
    output_col: str = "study_day",
) -> pd.DataFrame:
    """Add a 1-indexed day column relative to each group's first date."""
    out = df.copy()
    out["_date"] = out[date_col].dt.normalize()
    first_day = out.groupby(group_cols)["_date"].transform("min")
    out[output_col] = (out["_date"] - first_day).dt.days + 1
    return out.drop(columns=["_date"])


def add_study_day(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 1-indexed ``study_day`` column (convenience wrapper)."""
    return align_dates_relative(df, ["Wave", "Pcode"])


# ── Label helpers ─────────────────────────────────────────────────────────────
def normalize_label_series(series: pd.Series | Iterable[float]) -> pd.Series:
    """Shift a [0, 6] Likert range to [-3, +3] when detected."""
    s = pd.Series(series)
    smin, smax = s.min(), s.max()
    if pd.notna(smin) and pd.notna(smax) and smin >= 0 and smax > 3:
        return s - 3
    return s


def get_label_series(df: pd.DataFrame, label: str) -> pd.Series:
    """Extract and normalise a single ESM label column.

    Returns an empty series if the column is missing or the frame is empty.
    """
    if df.empty or label not in df.columns:
        return pd.Series(dtype=float)
    return normalize_label_series(df[label].dropna())


def esm_counts_per_person(df: pd.DataFrame) -> np.ndarray:
    """Return an array of per-participant ESM response counts."""
    if df.empty:
        return np.array([])
    return df.groupby("Pcode").size().to_numpy()


def compute_label_corr(
    df: pd.DataFrame,
    labels: Sequence[str],
    method: str = "pearson",
    min_rows: int = 5,
) -> pd.DataFrame:
    """Return a label-by-label correlation matrix (empty if too few rows)."""
    present = [c for c in labels if c in df.columns]
    if not present:
        return pd.DataFrame()
    frame = pd.DataFrame({c: get_label_series(df, c) for c in present}).dropna(how="any")
    if frame.shape[0] < min_rows:
        return pd.DataFrame()
    corr = frame.corr(method=method, numeric_only=True).astype(float)
    return corr.reindex(index=present, columns=present)


def user_means(
    df: pd.DataFrame,
    group_cols: str | list[str],
    value_col: str,
) -> pd.Series:
    """Per-group mean of ``value_col`` (drops NaNs first)."""
    if df.empty or value_col not in df.columns:
        return pd.Series(dtype=float)
    sub = df[[*([group_cols] if isinstance(group_cols, str) else group_cols), value_col]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby(group_cols)[value_col].mean()


# ── Wave splitting ─────────────────────────────────────────────────────────────
def split_by_wave(
    df: pd.DataFrame,
    wave_col: str = "Wave",
    keys: Sequence[str] = WAVE_KEYS,
) -> dict[str, pd.DataFrame]:
    """Split a long-form DataFrame into ``{wave_key: sub_df}`` dictionary."""
    if df.empty:
        return {k: df.iloc[0:0].copy() for k in keys}
    return {k: df.loc[df[wave_col] == k].copy() for k in keys}


# ── Gender extraction ──────────────────────────────────────────────────────────
def extract_gender(df: pd.DataFrame) -> pd.Series:
    """Extract gender as 'M'/'F'/NaN, handling single-column and one-hot encodings."""
    gender_col = next((c for c in df.columns if c.lower() == "pif#gender"), None)
    if gender_col:
        return df[gender_col].astype(str).str.upper().where(lambda x: x.isin(["M", "F"]))

    def _flag(suffix: str) -> pd.Series:
        col = next((c for c in df.columns if c.upper() == f"PIF#GENDER={suffix}"), None)
        return df[col].astype(bool) if col is not None else pd.Series(False, index=df.index)

    is_m, is_f = _flag("M"), _flag("F")
    return pd.Series(np.where(is_m, "M", np.where(is_f, "F", np.nan)), index=df.index)


# ── Pickle data loaders ────────────────────────────────────────────────────────
_META_COLS_ORDER: list[str] = [
    "META#dataset",
    "PIF#participantID",
    "PIF#stress_label",
    "PIF#time_offset",
    "PIF#timestamp",
]


def load_and_attach(path: str, dataset_tag: str) -> pd.DataFrame:
    """Load a pickle of ``(features, y, groups, t, datetimes)`` and attach metadata."""
    df, y, groups, t, datetimes = pd.read_pickle(path)
    meta = pd.DataFrame(
        {
            "PIF#participantID": groups,
            "PIF#stress_label": y,
            "PIF#time_offset": t,
            "PIF#timestamp": datetimes,
            "META#dataset": dataset_tag,
        }
    )
    out = pd.concat([df.reset_index(drop=True), meta.reset_index(drop=True)], axis=1)
    return out[_META_COLS_ORDER + [c for c in out.columns if c not in _META_COLS_ORDER]]


def load_label_pickle_with_meta(
    pkl_path: str | os.PathLike,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Load a per-label pickle and return ``(y, pcode, timestamps)`` arrays.

    Used by the trait-profile cells in ``overview.ipynb`` to align label files
    that don't follow the 5-tuple shape consumed by :func:`load_and_attach`.
    """
    with open(pkl_path, "rb") as f:
        _x, y, pcode, _t, timestamp = pickle.load(f)
    return (
        np.asarray(y),
        np.asarray(pcode),
        pd.to_datetime(np.asarray(timestamp)),
    )


def load_df_X_combined(data_root: str | None = None) -> pd.DataFrame:
    """Load and stack sensor pickles from all waves into a single DataFrame.

    Adds ``PIF#gender`` (via :func:`extract_gender`) and a normalised
    ``PIF#age`` column. Returns an empty DataFrame if no pickles are found.
    """
    root = data_root or get_data_root()
    parts: list[pd.DataFrame] = []
    for wk, path in get_pkl_paths(root).items():
        if not os.path.isfile(path):
            continue
        df = load_and_attach(path, wk)
        df["PIF#gender"] = extract_gender(df)
        if "PIF#age" in df.columns:
            df["PIF#age"] = df.filter(regex="(?i)PIF#age").bfill(axis=1).iloc[:, 0]
        parts.append(df.assign(__src=WAVE_TO_DIR[wk]))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=0, join="inner", ignore_index=True)


# ── 28-day window ──────────────────────────────────────────────────────────────
def apply_28d_window(
    df: pd.DataFrame,
    dataset_col: str,
    pid_col: str,
    ts_col: str,
    tz: str = "Asia/Seoul",
    dataset_order: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align each participant's first day to a per-dataset anchor and keep a 28-day window.

    Returns ``(filtered_df, summary_df)``. ``summary_df`` reports
    ``rows_before / rows_after / participants_after`` per dataset.
    """
    summary_cols = [dataset_col, "rows_before", "rows_after", "participants_after"]

    required = (ts_col, dataset_col, pid_col)
    if df is None or df.empty or any(c not in df.columns for c in required):
        empty = df.copy() if df is not None else pd.DataFrame()
        return empty, pd.DataFrame(columns=summary_cols)

    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    if tz:
        ts = ts.dt.tz_convert(tz)

    ok = ts.notna()
    base = df.loc[ok].copy()
    base["_ts"] = ts.loc[ok]

    datasets = list(dataset_order) if dataset_order else list(pd.unique(base[dataset_col].dropna()))
    rows: list[dict] = []
    kept: list[pd.DataFrame] = []

    for ds in datasets:
        dsub = base[base[dataset_col] == ds].copy()
        before = len(dsub)
        if dsub.empty:
            rows.append(
                {dataset_col: ds, "rows_before": 0, "rows_after": 0, "participants_after": 0}
            )
            continue

        anchor_date = dsub["_ts"].dt.normalize().min()
        user_start = dsub.groupby(pid_col)["_ts"].transform(lambda s: s.dt.normalize().min())
        day_offset = (user_start - anchor_date).dt.days
        aligned_ts = dsub["_ts"] - pd.to_timedelta(day_offset, unit="D")

        window_end = anchor_date + pd.Timedelta(days=27)
        aligned_dates = aligned_ts.dt.normalize()
        in_window = (aligned_dates >= anchor_date) & (aligned_dates <= window_end)

        kept_sub = dsub.loc[in_window].drop(columns=["_ts"])
        kept.append(kept_sub)
        rows.append(
            {
                dataset_col: ds,
                "rows_before": before,
                "rows_after": len(kept_sub),
                "participants_after": kept_sub[pid_col].nunique(),
            }
        )

    filtered = pd.concat(kept, ignore_index=True) if kept else df.iloc[0:0].copy()
    return filtered, pd.DataFrame(rows)


# ── High-level "load + 28-day window + split" bundles ──────────────────────────
def load_esm_28d(
    wave_keys: Sequence[str] = WAVE_KEYS,
    data_root: str | None = None,
    wave_to_dir: Mapping[str, str] | None = None,
    include_userinfo: bool = True,
    include_study_day: bool = True,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    pd.DataFrame,
    dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    """Load ESM/UserInfo, apply the 28-day window, and split the windowed frame.

    Returns
    -------
    (esm_by_wave, userinfo_by_wave, esm_all_28d, esm_by_wave_28d, window_stats)
    """
    esm_by_wave, userinfo_by_wave, esm_all = load_wave_esm_userinfo(
        wave_keys=wave_keys,
        data_root=data_root,
        wave_to_dir=wave_to_dir,
        include_userinfo=include_userinfo,
        include_study_day=include_study_day,
    )
    esm_all_28d, stats = apply_28d_window(
        esm_all,
        dataset_col=ESM_DATASET_COL,
        pid_col=ESM_PID_COL,
        ts_col=ESM_TS_COL,
        dataset_order=wave_keys,
    )
    esm_by_wave_28d = split_by_wave(esm_all_28d, ESM_DATASET_COL, wave_keys)
    return esm_by_wave, userinfo_by_wave, esm_all_28d, esm_by_wave_28d, stats


def load_sensor_28d(
    data_root: str | None = None,
    wave_keys: Sequence[str] = WAVE_KEYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load combined sensor DataFrame and apply the 28-day window.

    Returns ``(df_X_28d, window_stats)``.
    """
    df_X_combined = load_df_X_combined(data_root)
    return apply_28d_window(
        df_X_combined,
        dataset_col=SENSOR_DATASET_COL,
        pid_col=SENSOR_PID_COL,
        ts_col=SENSOR_TS_COL,
        dataset_order=wave_keys,
    )


__all__ = [
    # constants
    "GENDER_COLORS",
    "COLORS",
    "WAVE_LABELS",
    "WAVE_KEYS",
    "WAVE_TO_DIR",
    "TRAIT_COLS",
    "LABEL_NAMES",
    "SHARED_LABELS",
    "W1_W2_ONLY",
    "W1_ONLY",
    "W2_ONLY",
    "W3_ONLY",
    "ALL_LABELS",
    "COVERAGE_BY_WAVE",
    "ESM_DATASET_COL",
    "ESM_PID_COL",
    "ESM_TS_COL",
    "SENSOR_DATASET_COL",
    "SENSOR_PID_COL",
    "SENSOR_TS_COL",
    # style
    "set_paper_style",
    # parsing / paths
    "parse_timestamp",
    "to_utc_local",
    "get_data_root",
    "get_pkl_paths",
    # loaders
    "load_esm",
    "load_userinfo",
    "load_wave_esm_userinfo",
    "load_and_attach",
    "load_label_pickle_with_meta",
    "load_df_X_combined",
    "load_esm_28d",
    "load_sensor_28d",
    # analytical helpers
    "align_dates_relative",
    "add_study_day",
    "normalize_label_series",
    "get_label_series",
    "esm_counts_per_person",
    "compute_label_corr",
    "user_means",
    "split_by_wave",
    "extract_gender",
    "apply_28d_window",
]
