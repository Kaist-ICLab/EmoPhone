import pickle
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
import io
import os
import zipfile
import logging

logger = logging.getLogger(__name__)



class BenchmarkDataset:
    def __init__(self, dataset_name: str, file_path: str, min_samples: int = 100, min_class_samples: int = 10):
        self.dataset_name = dataset_name
        self.file_path = file_path
        self.min_samples = min_samples
        self.min_class_samples = min_class_samples

        self.X = None
        self.y = None
        self.users = None
        self.timestamps = None
        self.feature_names = None

        self.load_data()

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        logger.info(f"Loading {self.dataset_name} from {self.file_path}...")
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, (tuple, list)) and len(data) >= 5:
            self.X = data[0]
            self.y = data[1]
            self.users = data[2]
            self.timestamps = data[4]

            if isinstance(self.X, pd.DataFrame):
                self.feature_names = list(self.X.columns)
                self.X = self.X.values
            else:
                if self.feature_names is None or len(self.feature_names) != self.X.shape[1]:
                    self.feature_names = [f"feature_{i}" for i in range(self.X.shape[1])]

            logger.info(f"Feature Names Sample: {self.feature_names[:5]}")

            cols_to_drop = []
            for i, name in enumerate(self.feature_names):
                name_lower = str(name).lower()
                if 'timestamp' in name_lower or 'participant' in name_lower or 'label' in name_lower:
                    cols_to_drop.append(i)

            if cols_to_drop:
                logger.info(f"Dropping {len(cols_to_drop)} non-feature columns: {[self.feature_names[i] for i in cols_to_drop[:5]]}...")
                keep_mask = np.ones(self.X.shape[1], dtype=bool)
                keep_mask[cols_to_drop] = False
                self.X = self.X[:, keep_mask]
                self.feature_names = [name for i, name in enumerate(self.feature_names) if keep_mask[i]]

            self.X = np.array(self.X, dtype=np.float32)
            self.y = np.array(self.y, dtype=np.int64)
            self.users = np.array(self.users)
            self.timestamps = np.array(self.timestamps)

            logger.info(f"Loaded {self.X.shape[0]} samples, {self.X.shape[1]} features.")
        else:
            raise ValueError(f"Unexpected data format in {self.file_path}")

    def filter_one_month(self):
        logger.info("Filtering usage data to first 1 month per user...")
        unique_users = np.unique(self.users)
        valid_indices = []

        for user in unique_users:
            user_mask = (self.users == user)
            user_global_indices = np.where(user_mask)[0]

            if len(user_global_indices) == 0:
                continue

            user_timestamps = self.timestamps[user_global_indices]
            start_date = user_timestamps.min()
            cutoff_date = start_date + pd.DateOffset(months=1)
            keep_mask = (user_timestamps <= cutoff_date)
            valid_indices.extend(user_global_indices[keep_mask])

        valid_indices = np.array(valid_indices)
        valid_indices.sort()

        original_count = len(self.X)
        self.X = self.X[valid_indices]
        self.y = self.y[valid_indices]
        self.users = self.users[valid_indices]
        self.timestamps = self.timestamps[valid_indices]

        logger.info(f"Filtered data to 1 month. Samples: {original_count} -> {len(self.X)} (Removed {original_count - len(self.X)})")

    def filter_users(self):
        unique_users = np.unique(self.users)
        valid_indices = []
        dropped_users = []

        for user in unique_users:
            user_mask = (self.users == user)
            user_y = self.y[user_mask]

            if len(user_y) < self.min_samples:
                dropped_users.append(user)
                continue

            classes, counts = np.unique(user_y, return_counts=True)
            if len(classes) < 2 or np.min(counts) < self.min_class_samples:
                dropped_users.append(user)
                continue

            valid_indices.extend(np.where(user_mask)[0])

        if dropped_users:
            logger.info(f"Dropping {len(dropped_users)} users due to insufficient data: {dropped_users}")

        valid_indices = np.array(valid_indices)
        self.X = self.X[valid_indices]
        self.y = self.y[valid_indices]
        self.users = self.users[valid_indices]
        self.timestamps = self.timestamps[valid_indices]
        logger.info(f"Taking {len(valid_indices)} samples after filtering.")

    def normalize_features(self, train_idx, val_idx=None, test_idx=None):
        unique_users = np.unique(self.users)

        for user in unique_users:
            user_mask = (self.users == user)
            user_indices = np.where(user_mask)[0]
            user_train_indices = np.intersect1d(user_indices, train_idx)

            if len(user_train_indices) == 0:
                user_X_all = self.X[user_indices]
                mean = np.mean(user_X_all, axis=0)
                std = np.std(user_X_all, axis=0)
            else:
                user_X_train = self.X[user_train_indices]
                mean = np.mean(user_X_train, axis=0)
                std = np.std(user_X_train, axis=0)

            std[std < 1e-6] = 1.0
            self.X[user_mask] = ((self.X[user_mask] - mean) / std).astype(np.float32)

        clip_percentile = 99.9
        clip_min = 10.0
        train_vals = self.X[train_idx].reshape(-1)
        sample_size = min(1_000_000, train_vals.size)
        rng = np.random.default_rng(0)
        if train_vals.size > sample_size:
            sample_idx = rng.choice(train_vals.size, size=sample_size, replace=False)
            sample = np.abs(train_vals[sample_idx])
        else:
            sample = np.abs(train_vals)
        clip_value = float(np.percentile(sample, clip_percentile))
        if clip_value < clip_min:
            clip_value = clip_min
        self.X = np.clip(self.X, -clip_value, clip_value).astype(np.float32)

    def get_temporal_splits(self, train_ratio: float = 0.6, val_ratio: float = 0.2):
        train_indices = []
        val_indices = []
        test_indices = []

        unique_users = np.unique(self.users)

        for user in unique_users:
            user_global_indices = np.where(self.users == user)[0]
            user_timestamps = self.timestamps[user_global_indices]
            sorted_order = np.argsort(user_timestamps)
            sorted_indices = user_global_indices[sorted_order]

            n_samples = len(sorted_indices)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)

            train_idx = sorted_indices[:n_train]
            val_idx = sorted_indices[n_train:n_train + n_val]
            test_idx = sorted_indices[n_train + n_val:]

            train_indices.extend(train_idx)
            val_indices.extend(val_idx)
            test_indices.extend(test_idx)

        return np.array(train_indices), np.array(val_indices), np.array(test_indices)

    def get_raw_label_aligned(
        self,
        label_name: str,
        esm_response_root: str = '/var/nfs_share/harvard_dataverse',
        labels_zip_path: Optional[str] = None,
    ) -> np.ndarray:
        """Return raw <label_name> per current row, joined on (pcode, timestamp).

        For `stress_binary` the raw column is `stress`; for any other label the raw
        column has the same name. D-1/D-2 EsmResponse.csv exposes valence, arousal,
        stress, disturbance; D-3 additionally exposes happy/relaxed/cheerful/content/
        sad/anxious/depressed/angry. responseTime is epoch ms (UTC).
        """
        raw_col = 'stress' if label_name == 'stress_binary' else label_name
        ds_key = self.dataset_name.replace('-', '').replace('_', '')
        if not ds_key.upper().startswith('D'):
            raise ValueError(f"Unsupported dataset for raw label join: {self.dataset_name}")
        wave = ds_key[1:]
        esm_path = os.path.join(esm_response_root, f"D{wave}", "EsmResponse.csv")
        if not os.path.exists(esm_path):
            raise FileNotFoundError(f"EsmResponse.csv not found: {esm_path}")

        df = pd.read_csv(esm_path)
        if raw_col not in df.columns:
            raise ValueError(
                f"Raw column '{raw_col}' not found in {esm_path}. "
                f"Available: {[c for c in df.columns if c not in ('pcode', 'responseTime', 'actualTriggerTime', 'reactionTime', 'intendedTriggerTime')]}"
            )
        df = df[['pcode', 'responseTime', raw_col]].copy()
        df['responseTime'] = pd.to_numeric(df['responseTime'], errors='coerce')
        df[raw_col] = pd.to_numeric(df[raw_col], errors='coerce')
        df = df.dropna(subset=['responseTime', raw_col])
        df['timestamp'] = pd.to_datetime(df['responseTime'].astype('int64'), unit='ms', utc=True)
        df = df[['pcode', 'timestamp', raw_col]].rename(columns={raw_col: '_raw'})
        df = df.drop_duplicates(subset=['pcode', 'timestamp'])

        ds_ts = pd.to_datetime(pd.Series(self.timestamps), utc=True, errors='coerce')
        if ds_ts.isna().any():
            bad = int(ds_ts.isna().sum())
            raise ValueError(f"Failed to parse {bad} dataset timestamps for raw-label join.")
        ds_df = pd.DataFrame({
            '_idx': np.arange(len(self.users)),
            'pcode': np.asarray(self.users),
            'timestamp': ds_ts.to_numpy(),
        })
        merged = ds_df.merge(df, on=['pcode', 'timestamp'], how='left')
        missing = int(merged['_raw'].isna().sum())
        if missing > 0:
            raise ValueError(
                f"{missing}/{len(merged)} rows could not be matched to {esm_path} "
                f"on (pcode, timestamp). label={label_name}, raw_col={raw_col}."
            )
        return merged.sort_values('_idx')['_raw'].to_numpy(dtype=np.float64)

    def get_raw_stress_aligned(self, labels_zip_path: Optional[str] = None,
                                labels_dir: Optional[str] = None) -> np.ndarray:
        """Return raw `stress` per current row, joined on (pcode, timestamp) from labels.zip.

        Required for Tier A train-only personal binarization: the precomputed pickles
        already contain a leaky per-user binary label (threshold computed over the full
        user history). To rebuild the threshold from train rows only, we need the raw
        Likert stress value for every row.
        """
        ds_key = self.dataset_name.replace('-', '').replace('_', '')
        if not ds_key.upper().startswith('D'):
            raise ValueError(f"Unsupported dataset for raw stress join: {self.dataset_name}")
        member = f"labels/labels_1h_esmsyn_D{ds_key[1:]}"

        labels_df: Optional[pd.DataFrame] = None
        if labels_dir is not None:
            csv_path = os.path.join(labels_dir, os.path.basename(member))
            if os.path.exists(csv_path):
                labels_df = pd.read_csv(csv_path)
        if labels_df is None:
            zip_path = labels_zip_path or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'labels.zip'
            )
            if not os.path.exists(zip_path):
                raise FileNotFoundError(
                    f"labels.zip not found at {zip_path}; pass labels_zip_path or labels_dir."
                )
            with zipfile.ZipFile(zip_path, 'r') as zf:
                with zf.open(member) as fh:
                    labels_df = pd.read_csv(io.BytesIO(fh.read()))

        labels_df = labels_df[['pcode', 'timestamp', 'stress']].copy()
        labels_df['timestamp'] = pd.to_datetime(
            labels_df['timestamp'], utc=True, format='ISO8601', errors='coerce'
        )
        bad_label_ts = int(labels_df['timestamp'].isna().sum())
        if bad_label_ts > 0:
            raise ValueError(
                f"Failed to parse {bad_label_ts} timestamps in labels.zip ({member})."
            )

        ds_ts = pd.to_datetime(pd.Series(self.timestamps), utc=True, errors='coerce')
        if ds_ts.isna().any():
            bad = int(ds_ts.isna().sum())
            raise ValueError(f"Failed to parse {bad} dataset timestamps for raw-stress join.")

        ds_df = pd.DataFrame({
            '_idx': np.arange(len(self.users)),
            'pcode': np.asarray(self.users),
            'timestamp': ds_ts.to_numpy(),
        })
        merged = ds_df.merge(labels_df, on=['pcode', 'timestamp'], how='left')

        missing = int(merged['stress'].isna().sum())
        if missing > 0:
            raise ValueError(
                f"{missing}/{len(merged)} rows could not be matched to labels.zip "
                f"on (pcode, timestamp). Check that the pkl and labels.zip come from the same wave."
            )
        return merged.sort_values('_idx')['stress'].to_numpy(dtype=np.float64)


def rebinarize_personal_train_only(
    users: np.ndarray,
    raw_stress: np.ndarray,
    train_idx: np.ndarray,
    agg: str = 'median',
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Re-derive per-user binary stress labels using only training-set raw stress.

    threshold[u] = median (or mean) of raw_stress[i] over i in train_idx with users[i] == u.
    For users with no training rows, falls back to the global train threshold.
    Binarization rule matches the original notebook: 1 if raw_stress > threshold else 0.
    """
    if agg not in ('median', 'mean'):
        raise ValueError(f"agg must be 'median' or 'mean', got {agg!r}")

    users = np.asarray(users)
    raw_stress = np.asarray(raw_stress, dtype=np.float64)
    train_idx = np.asarray(train_idx, dtype=int)

    train_users = users[train_idx]
    train_stress = raw_stress[train_idx]

    if train_stress.size == 0:
        raise ValueError("rebinarize_personal_train_only: train_idx is empty.")

    global_threshold = float(np.median(train_stress) if agg == 'median' else np.mean(train_stress))

    thresholds: Dict[str, float] = {}
    unique_users = np.unique(users)
    for user in unique_users:
        user_train_mask = (train_users == user)
        if user_train_mask.any():
            vals = train_stress[user_train_mask]
            thresholds[str(user)] = float(np.median(vals) if agg == 'median' else np.mean(vals))
        else:
            thresholds[str(user)] = global_threshold

    threshold_per_row = np.array([thresholds[str(u)] for u in users], dtype=np.float64)
    y_new = (raw_stress > threshold_per_row).astype(np.int64)
    return y_new, thresholds


def filter_split_for_label_diversity(
    y: np.ndarray,
    users: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    min_classes: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """After re-binarization, drop users whose train/val/test slice lost label diversity."""
    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    test_set = set(test_idx.tolist())

    new_train, new_val, new_test = [], [], []
    stats = {"kept_users": 0, "dropped_no_diversity_after_rebin": 0}

    for user in np.unique(users):
        u_idx = np.where(users == user)[0]
        tr = [i for i in u_idx if i in train_set]
        va = [i for i in u_idx if i in val_set]
        te = [i for i in u_idx if i in test_set]
        if not tr or not va or not te:
            continue
        if (
            np.unique(y[tr]).size < min_classes
            or np.unique(y[va]).size < min_classes
            or np.unique(y[te]).size < min_classes
        ):
            stats["dropped_no_diversity_after_rebin"] += 1
            continue
        new_train.extend(tr)
        new_val.extend(va)
        new_test.extend(te)
        stats["kept_users"] += 1

    return (
        np.asarray(new_train, dtype=int),
        np.asarray(new_val, dtype=int),
        np.asarray(new_test, dtype=int),
        stats,
    )


