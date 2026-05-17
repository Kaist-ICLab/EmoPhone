import argparse
import json
import pickle
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.da_models import DAModel  # noqa: E402


def load_pickle(path: Path):
    with path.open("rb") as handle:
        x_df, y, groups, *rest = pickle.load(handle)
    return x_df.copy(), np.asarray(y), np.asarray(groups)


def normalize_single_user(x_train: pd.DataFrame, x_test: pd.DataFrame):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled


def normalize_groups_independently(x_df: pd.DataFrame, groups: np.ndarray, features: list[str]) -> pd.DataFrame:
    parts = []
    for user in np.unique(groups):
        mask = groups == user
        x_user = x_df.loc[mask, features]
        scaler = StandardScaler()
        x_scaled = pd.DataFrame(
            scaler.fit_transform(x_user),
            columns=features,
            index=x_user.index,
        )
        parts.append(x_scaled)
    return pd.concat(parts).sort_index()


def train_mlp_with_early_stopping(x_train, y_train, x_val, y_val, epochs, patience, batch_size, device):
    model = DAModel(
        input_dim=x_train.shape[1],
        num_classes=2,
        hparams={
            "backbone": "MLP",
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.0,
        },
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )

    best_val_auroc = -float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        probs = []
        targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred = torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy()
                probs.extend(pred)
                targets.extend(yb.numpy())
        val_auroc = float(roc_auc_score(targets, probs))

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_auroc, best_epoch


def compute_global_xgb_ranking(x_df, y, groups, seed, top_n):
    features = x_df.columns.tolist()
    importances = []
    users = sorted(np.unique(groups))

    for user in users:
        mask = groups == user
        x_user = x_df.loc[mask, features]
        y_user = y[mask]
        if len(np.unique(y_user)) < 2:
            continue
        x_train, x_test, y_train, y_test = train_test_split(
            x_user, y_user, test_size=0.2, random_state=seed, stratify=y_user
        )
        x_train_s, x_test_s = normalize_single_user(x_train, x_test)
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=seed,
            eval_metric="auc",
            early_stopping_rounds=20,
            n_jobs=1,
            tree_method="hist",
        )
        model.fit(x_train_s, y_train, eval_set=[(x_test_s, y_test)], verbose=False)
        importances.append(model.feature_importances_)

    mean_importance = np.mean(np.vstack(importances), axis=0)
    ranking_idx = np.argsort(mean_importance)[::-1]
    ranked_features = [features[idx] for idx in ranking_idx[:top_n]]
    ranked_scores = [float(mean_importance[idx]) for idx in ranking_idx[:top_n]]
    return ranked_features, ranked_scores


def evaluate_incremental_features(x_df, y, groups, ranked_features, epochs, patience, batch_size, seed, device):
    users = sorted(np.unique(groups))
    records = []

    for k in range(1, len(ranked_features) + 1):
        selected = ranked_features[:k]
        fold_aurocs = []
        fold_epochs = []
        for target_user in users:
            target_mask = groups == target_user
            source_mask = ~target_mask
            x_source = x_df.loc[source_mask, selected].copy()
            y_source = y[source_mask]
            groups_source = groups[source_mask]

            x_train_df, x_val_df, y_train, y_val, groups_train, groups_val = train_test_split(
                x_source, y_source, groups_source,
                test_size=0.2,
                random_state=seed,
                stratify=y_source,
            )
            x_train_norm = normalize_groups_independently(x_train_df, groups_train, selected)
            x_val_norm = normalize_groups_independently(x_val_df, groups_val, selected)
            x_train = x_train_norm[selected].to_numpy(dtype=np.float32)
            x_val = x_val_norm[selected].to_numpy(dtype=np.float32)
            y_train = y_train.astype(np.int64)
            y_val = y_val.astype(np.int64)

            val_auroc, best_epoch = train_mlp_with_early_stopping(
                x_train, y_train, x_val, y_val, epochs, patience, batch_size, device
            )
            fold_aurocs.append(val_auroc)
            fold_epochs.append(best_epoch)

        records.append({
            "k": k,
            "added_feature": ranked_features[k - 1],
            "mean_source_val_auroc": float(np.mean(fold_aurocs)),
            "std_source_val_auroc": float(np.std(fold_aurocs)),
            "mean_best_epoch": float(np.mean(fold_epochs)),
        })
        print(
            f"k={k:02d} feature={ranked_features[k-1]} "
            f"mean_val_auroc={records[-1]['mean_source_val_auroc']:.4f}"
        )
    return records


def main():
    parser = argparse.ArgumentParser(description="Incremental XGB-ranked feature evaluation for source MLP")
    parser.add_argument("--data", type=Path, default=Path("DA_Verification/CGDM/step_count_binary_personal-15min_mlp-auroc85.pkl"))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", type=Path, default=Path("DA_Verification/CGDM/incremental_xgb_features"))
    args = parser.parse_args()

    x_df, y, groups = load_pickle(args.data)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ranked_features, ranked_scores = compute_global_xgb_ranking(x_df, y, groups, args.seed, args.top_n)
    records = evaluate_incremental_features(
        x_df, y, groups, ranked_features, args.epochs, args.patience, args.batch_size, args.seed, device
    )

    ranking_path = args.out_prefix.with_name(args.out_prefix.name + "_ranking.json")
    metrics_path = args.out_prefix.with_name(args.out_prefix.name + "_metrics.csv")
    summary_path = args.out_prefix.with_name(args.out_prefix.name + "_summary.json")

    ranking_path.write_text(
        json.dumps(
            {
                "data": str(args.data),
                "ranked_features": ranked_features,
                "ranked_scores": ranked_scores,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(records).to_csv(metrics_path, index=False)
    best_record = max(records, key=lambda row: row["mean_source_val_auroc"])
    summary_path.write_text(
        json.dumps(
            {
                "device": device,
                "top_n": args.top_n,
                "best_k": best_record["k"],
                "best_mean_source_val_auroc": best_record["mean_source_val_auroc"],
                "best_features": ranked_features[: best_record["k"]],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved ranking to {ranking_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
