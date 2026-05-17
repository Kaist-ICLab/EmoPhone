import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.da_models import DAModel, train_standard  # noqa: E402


def load_pickle(path: Path):
    with path.open("rb") as handle:
        x_df, y, groups, *rest = pickle.load(handle)
    return x_df.copy(), np.asarray(y), np.asarray(groups)


def load_pair_features(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {frozenset(row["users"]): row["top_features"] for row in rows}


def load_candidates_from_neighbors(path: Path, seed_users: list[str], neighbor_k: int):
    data = json.loads(path.read_text(encoding="utf-8"))
    nearest = data["nearest_neighbors"]
    users = set(seed_users)
    for user in seed_users:
        for item in nearest.get(user, [])[:neighbor_k]:
            users.add(item["user"])
    return sorted(users)


def normalize_source_train_val(x_train: pd.DataFrame, x_val: pd.DataFrame):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    return x_train_scaled.astype(np.float32), x_val_scaled.astype(np.float32)


def normalize_target_independently(x_target: pd.DataFrame):
    scaler = StandardScaler()
    return scaler.fit_transform(x_target).astype(np.float32)


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


def parse_csv_arg(value: str | None):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Scan directional pair-transfer candidates")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("DA_Verification/CGDM/step_count_binary_personal-15min_mlp-auroc85.pkl"),
    )
    parser.add_argument(
        "--pair-search-json",
        type=Path,
        default=Path("DA_Verification/CGDM/pair_cluster_search.json"),
    )
    parser.add_argument(
        "--relation-json",
        type=Path,
        default=Path("DA_Verification/CGDM/relation_signature_groups_top10.json"),
    )
    parser.add_argument("--seed-users", type=str, default="P008,P041")
    parser.add_argument("--users", type=str, default="")
    parser.add_argument("--neighbor-k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("DA_Verification/CGDM/pair_transfer_candidate_scan"),
    )
    args = parser.parse_args()

    seed_users = parse_csv_arg(args.seed_users)
    explicit_users = parse_csv_arg(args.users)
    if explicit_users:
        candidate_users = sorted(set(explicit_users))
    else:
        candidate_users = load_candidates_from_neighbors(args.relation_json, seed_users, args.neighbor_k)

    x_df, y, groups = load_pickle(args.data)
    pair_features = load_pair_features(args.pair_search_json)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = []
    for source_user in candidate_users:
        for target_user in candidate_users:
            if source_user == target_user:
                continue
            key = frozenset((source_user, target_user))
            if key not in pair_features:
                continue

            features = pair_features[key]
            source_mask = groups == source_user
            target_mask = groups == target_user
            x_source = x_df.loc[source_mask, features].copy()
            y_source = y[source_mask].astype(np.int64)
            x_target = x_df.loc[target_mask, features].copy()
            y_target = y[target_mask].astype(np.int64)

            if len(np.unique(y_source)) < 2 or len(np.unique(y_target)) < 2:
                continue

            x_train_df, x_val_df, y_train, y_val = train_test_split(
                x_source, y_source, test_size=0.2, random_state=42, stratify=y_source
            )
            x_train, x_val = normalize_source_train_val(x_train_df, x_val_df)
            x_target_np = normalize_target_independently(x_target)

            model = build_mlp(len(features))
            model = train_standard(
                model,
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

            record = {
                "source_user": source_user,
                "target_user": target_user,
                "n_source": int(len(y_source)),
                "n_target": int(len(y_target)),
                "n_features": int(len(features)),
                "features": features,
            }
            record.update(metric_dict(model, x_val, y_val, device, "source_val"))
            record.update(metric_dict(model, x_target_np, y_target, device, "target"))
            records.append(record)
            print(
                f"{source_user}->{target_user} "
                f"source_val_auc={record['source_val_auroc']:.4f} "
                f"target_auc={record['target_auroc']:.4f}"
            )

    out_csv = args.out_prefix.with_suffix(".csv")
    out_json = args.out_prefix.with_suffix(".json")
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["target_auroc", "source_val_auroc"], ascending=[False, False])
    df.to_csv(out_csv, index=False)

    user_summary = []
    if not df.empty:
        for user in candidate_users:
            incoming = df[df["target_user"] == user]
            outgoing = df[df["source_user"] == user]
            user_summary.append(
                {
                    "user": user,
                    "incoming_mean_target_auroc": float(incoming["target_auroc"].mean()) if not incoming.empty else None,
                    "outgoing_mean_target_auroc": float(outgoing["target_auroc"].mean()) if not outgoing.empty else None,
                    "incoming_max_target_auroc": float(incoming["target_auroc"].max()) if not incoming.empty else None,
                    "outgoing_max_target_auroc": float(outgoing["target_auroc"].max()) if not outgoing.empty else None,
                    "num_incoming_pairs": int(len(incoming)),
                    "num_outgoing_pairs": int(len(outgoing)),
                }
            )

    payload = {
        "seed_users": seed_users,
        "candidate_users": candidate_users,
        "epochs": args.epochs,
        "patience": args.patience,
        "records": records,
        "user_summary": user_summary,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {out_csv}")
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
