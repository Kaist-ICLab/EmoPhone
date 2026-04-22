"""
Common helper functions shared across EDA notebooks.
This is the primary helper module used by notebooks.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────
# Internal wave keys are kept as "D-1/D-2/D-3" because the notebooks use them
# as dict keys and display labels. The on-disk folder structure (as documented
# in ../data/README.md) is the non-hyphenated form "D1/D2/D3", mapped via
# WAVE_TO_DIR below. Do not mix the two forms.
GENDER_COLORS = {"M": "#4A90E2", "F": "#FF69B4"}
COLORS = {"D-1": "#2C7BB6", "D-2": "#D7191C", "D-3": "#1A9641"}
WAVE_LABELS = ["D-1 (2020)", "D-2 (2021)", "D-3 (2022)"]
WAVE_KEYS = ["D-1", "D-2", "D-3"]
WAVE_TO_DIR = {"D-1": "D1", "D-2": "D2", "D-3": "D3"}

TRAIT_COLS = ["age", "openness", "conscientiousness", "neuroticism", "extraversion", "agreeableness", "pss10", "ghq12"]
LABEL_NAMES = ["valence", "arousal", "stress", "disturbance"]
# Backward compatibility for existing notebooks.


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
    """Auto-detect epoch (s or ms) vs string timestamps and convert to datetime."""
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0] if series.shape[1] > 0 else pd.Series(dtype="datetime64[ns]")
    non_null = series.dropna()
    sample = non_null.iloc[0] if not non_null.empty else None
    if sample is None:
        return pd.Series(dtype="datetime64[ns]")
    try:
        unit = "ms" if float(sample) > 1e10 else "s"
        return pd.to_datetime(series, unit=unit, errors="coerce")
    except (ValueError, TypeError):
        return pd.to_datetime(series, errors="coerce")


# ── Data loaders ───────────────────────────────────────────────────────────────
def load_esm(
    wave_key: str,
    data_root: str | None = None,
    wave_to_dir: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Load and normalise EsmResponse.csv for one wave."""
    if data_root is None:
        data_root = os.environ.get("DATA_ROOT", "")
    if wave_to_dir is None:
        wave_to_dir = WAVE_TO_DIR

    fpath = os.path.join(data_root, wave_to_dir[wave_key], "EsmResponse.csv")
    df = pd.read_csv(fpath)
    if "actualTriggerTime" in df.columns and "intendedTriggerTime" in df.columns:
        df = df.drop(columns=["intendedTriggerTime"])
    rename_map = {
        "pcode": "Pcode", "responseTime": "ResponseTime",
        "actualTriggerTime": "TriggerTime", "intendedTriggerTime": "TriggerTime",
        "reactionTime": "ReactionTime", "valence": "Valence", "arousal": "Arousal",
        "stress": "Stress", "disturbance": "Task Disturbance", "duration": "Duration",
        "attention": "Attention", "mental": "Mental",
        "valenceChange": "Changed Valence", "arousalChange": "Changed Arousal",
        "happy": "Happy", "relaxed": "Relaxed", "cheerful": "Cheerful",
        "content": "Content", "sad": "Sad", "anxious": "Anxious",
        "depressed": "Depressed", "angry": "Angry", "change": "Change",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    for col in ["ResponseTime", "TriggerTime", "ReactionTime"]:
        if col in df.columns:
            df[col] = parse_timestamp(df[col])
    for _col in ["Valence", "Arousal", "Stress", "Task Disturbance"]:
        if _col in df.columns:
            df[_col] = normalize_label_series(df[_col])
    df["Wave"] = wave_key
    return df


def load_userinfo(
    wave_key: str,
    data_root: str | None = None,
    wave_to_dir: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Load and normalise UserInfo.csv for one wave."""
    if data_root is None:
        data_root = os.environ.get("DATA_ROOT", "")
    if wave_to_dir is None:
        wave_to_dir = WAVE_TO_DIR

    fpath = os.path.join(data_root, wave_to_dir[wave_key], "UserInfo.csv")
    df = pd.read_csv(fpath)
    rename_map = {
        "pcode": "Pcode", "openness": "Openness",
        "conscientiousness": "Conscientiousness", "neuroticism": "Neuroticism",
        "extraversion": "Extraversion", "agreeableness": "Agreeableness",
        "age": "Age", "gender": "Gender",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["Wave"] = wave_key
    return df


def load_wave_esm_userinfo(
    wave_keys: Sequence[str] = WAVE_KEYS,
    data_root: str | None = None,
    wave_to_dir: Mapping[str, str] | None = None,
    include_userinfo: bool = True,
    include_study_day: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load ESM/UserInfo tables for each wave and build a combined ESM DataFrame.

    Returns (esm_by_wave, userinfo_by_wave, esm_all).
    """
    root = data_root or os.environ.get("DATA_ROOT", "")
    wave_dirs = wave_to_dir or WAVE_TO_DIR

    esm_by_wave = {k: load_esm(k, root, wave_dirs) for k in wave_keys}
    userinfo_by_wave = (
        {k: load_userinfo(k, root, wave_dirs) for k in wave_keys}
        if include_userinfo
        else {}
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
    output_col: str = "study_day"
) -> pd.DataFrame:
    """Add a 1-indexed day column relative to each group's first date."""
    out = df.copy()
    out["_date"] = out[date_col].dt.normalize()
    first_day = out.groupby(group_cols)["_date"].transform("min")
    out[output_col] = (out["_date"] - first_day).dt.days + 1
    return out.drop(columns=["_date"])


def add_study_day(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 1-indexed study_day column (convenience wrapper)."""
    return align_dates_relative(df, ["Wave", "Pcode"])


# ── Label helpers ─────────────────────────────────────────────────────────────
def normalize_label_series(series) -> pd.Series:
    """Shift a [0, 6] Likert range to [-3, +3] when detected."""
    series = pd.Series(series)
    smin, smax = series.min(), series.max()
    if pd.notna(smin) and pd.notna(smax) and smin >= 0 and smax > 3:
        return series - 3
    return series


def get_label_series(df: pd.DataFrame, label: str) -> pd.Series:
    """Extract and normalise a single ESM label column. Returns empty series if column missing."""
    if df.empty or label not in df.columns:
        return pd.Series(dtype=float)
    return normalize_label_series(df[label].dropna())


def esm_counts_per_person(df: pd.DataFrame) -> np.ndarray:
    """Return an array of per-participant ESM response counts."""
    return df.groupby("Pcode").size().to_numpy() if not df.empty else np.array([])


# ── Timezone helpers ───────────────────────────────────────────────────────────
def to_local_time(series: pd.Series, tz: str = "Asia/Seoul") -> pd.Series:
    """Convert timezone-aware datetime, handling string or numeric timestamps."""
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(tz)


# ── Gender extraction ──────────────────────────────────────────────────────────
def extract_gender(df: pd.DataFrame) -> pd.Series:
    """Extract gender as 'M'/'F'/NaN, handling single-column and one-hot encodings."""
    # Single column (case-insensitive match)
    gender_col = next((c for c in df.columns if c.lower() == "pif#gender"), None)
    if gender_col:
        return df[gender_col].astype(str).str.upper().where(lambda x: x.isin(["M", "F"]))

    # One-hot encoded — handles pif#gender=M/F and PIF#GENDER=M/F
    def _flag(suffix: str) -> pd.Series:
        col = next((c for c in df.columns if c.upper() == f"PIF#GENDER={suffix}"), None)
        return df[col].astype(bool) if col is not None else pd.Series(False, index=df.index)

    is_m, is_f = _flag("M"), _flag("F")
    return pd.Series(np.where(is_m, "M", np.where(is_f, "F", np.nan)), index=df.index)


# ── Pickle data loader ─────────────────────────────────────────────────────────
def load_and_attach(path: str, dataset_tag: str) -> pd.DataFrame:
    """Load pickle data and attach metadata columns."""
    df, y, groups, t, datetimes = pd.read_pickle(path)
    meta = pd.DataFrame({
        'PIF#participantID': groups,
        'PIF#stress_label': y,
        'PIF#time_offset': t,
        'PIF#timestamp': datetimes,
        'META#dataset': dataset_tag,
    })
    out = pd.concat([df.reset_index(drop=True), meta.reset_index(drop=True)], axis=1)
    meta_cols = ['META#dataset', 'PIF#participantID', 'PIF#stress_label',
                 'PIF#time_offset', 'PIF#timestamp']
    return out[meta_cols + [c for c in out.columns if c not in meta_cols]]


# ── Label category constants ────────────────────────────────────────────────────
SHARED_LABELS = ["Valence", "Arousal", "Stress", "Task Disturbance"]
W1_W2_ONLY    = ["Attention", "Mental", "Duration"]
W1_ONLY       = ["Change"]
W2_ONLY       = ["Changed Valence", "Changed Arousal"]
W3_ONLY       = ["Happy", "Relaxed", "Cheerful", "Content", "Sad", "Angry", "Anxious", "Depressed"]
ALL_LABELS    = SHARED_LABELS + W1_W2_ONLY + W1_ONLY + W2_ONLY + W3_ONLY

COVERAGE_BY_WAVE: dict[str, list[str]] = {
    "D-1": SHARED_LABELS + W1_W2_ONLY + W1_ONLY,
    "D-2": SHARED_LABELS + W1_W2_ONLY + W2_ONLY,
    "D-3": SHARED_LABELS + W3_ONLY,
}


# ── Path helpers ────────────────────────────────────────────────────────────────
# These paths reference legacy sensor-only pickle files used by a few EDA
# notebooks (stress_binary_personal-full). They live alongside the per-label
# release pickles (`{label}.pkl`) documented in ../data/README.md. Disk paths
# use the released D1/D2/D3 folder names while the dict keys use the internal
# D-1/D-2/D-3 form to match WAVE_KEYS.
_PKL_RELATIVE: dict[str, str] = {
    "D-1": os.path.join("D1", "stress_binary_personal-full_D#2.pkl"),
    "D-2": os.path.join("D2", "stress_binary_personal-full_D#3.pkl"),
    "D-3": os.path.join("D3", "stress_binary_personal-full.pkl"),
}


def get_data_root() -> str:
    """
    Return the project data directory.

    Resolution order:
    1) DATA_ROOT environment variable (if set),
    2) nearest ancestor directory that contains ./data,
    3) fallback to ../../data relative to this file.
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


def get_pkl_paths(data_root: str | None = None) -> dict[str, str]:
    """Return wave -> pickle file path dict."""
    root = data_root or get_data_root()
    return {k: os.path.join(root, v) for k, v in _PKL_RELATIVE.items()}


# ── Combined sensor dataframe loader ────────────────────────────────────────────
def load_df_X_combined(data_root: str | None = None) -> pd.DataFrame:
    """
    Load and stack sensor pickle files from all waves into a single DataFrame.
    Adds PIF#gender (via extract_gender) and normalises PIF#age.
    Returns an empty DataFrame if no pickle files are found.
    """
    root = data_root or get_data_root()
    _src_tags = {"D-1": "D1", "D-2": "D2", "D-3": "D3"}
    parts = []
    for wk, path in get_pkl_paths(root).items():
        if not os.path.isfile(path):
            continue
        df = load_and_attach(path, wk)
        df["PIF#gender"] = extract_gender(df)
        if "PIF#age" in df.columns:
            df["PIF#age"] = df.filter(regex="(?i)PIF#age").bfill(axis=1).iloc[:, 0]
        parts.append(df.assign(__src=_src_tags[wk]))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=0, join="inner", ignore_index=True)


__all__ = [
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
    "set_paper_style",
    "parse_timestamp",
    "load_esm",
    "load_userinfo",
    "load_wave_esm_userinfo",
    "align_dates_relative",
    "add_study_day",
    "normalize_label_series",
    "get_label_series",
    "esm_counts_per_person",
    "to_local_time",
    "extract_gender",
    "load_and_attach",
    "get_data_root",
    "get_pkl_paths",
    "load_df_X_combined",
]
