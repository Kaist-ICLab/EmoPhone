import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.da_models import CGDM, DAModel, train_cgdm, train_standard  # noqa: E402


INVALID_COL_PATTERN = re.compile(r"[\[\]<>{}(),]")


def load_pickle(path: Path):
    with path.open("rb") as handle:
        x_df, y, groups, *rest = pickle.load(handle)
    return x_df.copy(), np.asarray(y), np.asarray(groups)


def get_candidate_columns(x_df: pd.DataFrame) -> list[str]:
    numeric_cols = x_df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if not INVALID_COL_PATTERN.search(col)]


def normalize_source_train_val_by_user(
    x_train_df: pd.DataFrame,
    x_val_df: pd.DataFrame,
    groups_train: np.ndarray,
    groups_val: np.ndarray,
    features: list[str],
):
    train_out = pd.DataFrame(index=x_train_df.index, columns=features, dtype=np.float32)
    val_out = pd.DataFrame(index=x_val_df.index, columns=features, dtype=np.float32)

    for user in np.unique(groups_train):
        train_mask = groups_train == user
        val_mask = groups_val == user
        x_user_train = x_train_df.loc[train_mask, features]
        if x_user_train.empty:
            continue
        scaler = StandardScaler()
        scaler.fit(x_user_train)
        train_out.loc[x_train_df.index[train_mask], features] = scaler.transform(x_user_train).astype(np.float32)
        if np.any(val_mask):
            x_user_val = x_val_df.loc[val_mask, features]
            val_out.loc[x_val_df.index[val_mask], features] = scaler.transform(x_user_val).astype(np.float32)

    return train_out, val_out


def normalize_target_independently(x_target_df: pd.DataFrame, features: list[str]):
    scaler = StandardScaler()
    return scaler.fit_transform(x_target_df[features]).astype(np.float32)


def fit_xgb_and_rank_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    features: list[str],
    top_k: int,
    seed: int,
):
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
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    importances = model.feature_importances_
    ranking_idx = np.argsort(importances)[::-1]
    ranked = [features[idx] for idx in ranking_idx if importances[idx] > 0]
    if len(ranked) < top_k:
        extra = [features[idx] for idx in ranking_idx if features[idx] not in ranked]
        ranked.extend(extra)
    return ranked[:top_k], [float(importances[features.index(feature)]) for feature in ranked[:top_k]]


def build_mlp(input_dim: int):
    return DAModel(
        input_dim=input_dim,
        num_classes=2,
        hparams={
            "backbone": "MLP",
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.0,
        },
    )


def metric_dict(model, x_np, y_np, device, prefix: str):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_np, dtype=torch.float32, device=device)
        logits = model.predict(x_tensor) if hasattr(model, "predict") else model(x_tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return {
        f"{prefix}_acc": float(accuracy_score(y_np, preds)),
        f"{prefix}_auroc": float(roc_auc_score(y_np, probs[:, 1])),
        f"{prefix}_prauc": float(average_precision_score(y_np, probs[:, 1])),
    }


def parse_user_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def summarize_records(records: list[dict]):
    if not records:
        return {}
    df = pd.DataFrame(records)
    summary = {
        "num_folds": int(len(df)),
        "mean_source_val_auroc_mlp": float(df["mlp_source_val_auroc"].mean()),
        "mean_target_auroc_mlp": float(df["mlp_target_auroc"].mean()),
        "mean_target_acc_mlp": float(df["mlp_target_acc"].mean()),
    }
    if "cgdm_target_auroc" in df.columns:
        summary.update(
            {
                "mean_source_val_auroc_cgdm": float(df["cgdm_source_val_auroc"].mean()),
                "mean_target_auroc_cgdm": float(df["cgdm_target_auroc"].mean()),
                "mean_target_acc_cgdm": float(df["cgdm_target_acc"].mean()),
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Mini-cluster transfer experiment")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("DA_Verification/CGDM/step_count_binary_personal-15min_mlp-auroc85.pkl"),
    )
    parser.add_argument("--users", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", type=str, default="MLP,CGDM")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("DA_Verification/CGDM/cluster_transfer"),
    )
    args = parser.parse_args()

    selected_users = parse_user_list(args.users)
    run_models = {item.strip().upper() for item in args.models.split(",") if item.strip()}

    x_df, y, groups = load_pickle(args.data)
    candidate_cols = get_candidate_columns(x_df)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = args.out_dir / "_".join(selected_users)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for target_user in selected_users:
        source_users = [user for user in selected_users if user != target_user]
        source_mask = np.isin(groups, source_users)
        target_mask = groups == target_user

        x_source = x_df.loc[source_mask, candidate_cols].copy()
        y_source = y[source_mask].astype(np.int64)
        groups_source = groups[source_mask]
        x_target = x_df.loc[target_mask, candidate_cols].copy()
        y_target = y[target_mask].astype(np.int64)

        x_train_df, x_val_df, y_train, y_val, groups_train, groups_val = train_test_split(
            x_source,
            y_source,
            groups_source,
            test_size=0.2,
            random_state=args.seed,
            stratify=y_source,
        )

        x_train_norm_df, x_val_norm_df = normalize_source_train_val_by_user(
            x_train_df, x_val_df, groups_train, groups_val, candidate_cols
        )
        x_train_norm = x_train_norm_df[candidate_cols].to_numpy(dtype=np.float32)
        x_val_norm = x_val_norm_df[candidate_cols].to_numpy(dtype=np.float32)

        top_features, top_scores = fit_xgb_and_rank_features(
            x_train_norm,
            y_train,
            x_val_norm,
            y_val,
            candidate_cols,
            args.top_k,
            args.seed,
        )

        x_train = x_train_norm_df[top_features].to_numpy(dtype=np.float32)
        x_val = x_val_norm_df[top_features].to_numpy(dtype=np.float32)
        x_target_np = normalize_target_independently(x_target, top_features)

        record = {
            "target_user": target_user,
            "source_users": source_users,
            "n_source": int(len(y_source)),
            "n_target": int(len(y_target)),
            "top_features": top_features,
            "top_feature_scores": top_scores,
        }

        if "MLP" in run_models:
            mlp = build_mlp(len(top_features))
            mlp = train_standard(
                mlp,
                x_train,
                y_train,
                x_val,
                y_val,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=1e-4,
                patience=args.patience,
                device=device,
            )
            mlp_source = metric_dict(mlp, x_val, y_val, device, "source_val")
            mlp_target = metric_dict(mlp, x_target_np, y_target, device, "target")
            for key, value in {**mlp_source, **mlp_target}.items():
                record[f"mlp_{key}"] = value
            torch.save(mlp.state_dict(), out_dir / f"{target_user}_mlp_checkpoint.pt")

        if "CGDM" in run_models:
            cgdm = CGDM(input_dim=len(top_features), num_classes=2)
            cgdm = train_cgdm(
                cgdm,
                x_train,
                y_train,
                x_target_np,
                y_target=y_target,
                X_val=x_val,
                y_val=y_val,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=1e-4,
                patience=args.patience,
                device=device,
            )
            cgdm_source = metric_dict(cgdm, x_val, y_val, device, "source_val")
            cgdm_target = metric_dict(cgdm, x_target_np, y_target, device, "target")
            for key, value in {**cgdm_source, **cgdm_target}.items():
                record[f"cgdm_{key}"] = value
            torch.save(cgdm.state_dict(), out_dir / f"{target_user}_cgdm_checkpoint.pt")

        records.append(record)
        print(
            f"target={target_user} source={source_users} "
            f"mlp_target_auc={record.get('mlp_target_auroc', float('nan')):.4f} "
            f"cgdm_target_auc={record.get('cgdm_target_auroc', float('nan')):.4f}"
        )

    summary = summarize_records(records)
    summary["users"] = selected_users
    summary["top_k"] = args.top_k
    summary["epochs"] = args.epochs
    summary["patience"] = args.patience

    csv_path = out_dir / "metrics.csv"
    json_path = out_dir / "summary.json"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2),
        encoding="utf-8",
    )
    print(f"saved {csv_path}")
    print(f"saved {json_path}")


if __name__ == "__main__":
    main()
