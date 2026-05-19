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

from src.da_models import CGDM, DAModel, train_cgdm, train_standard  # noqa: E402


def load_pickle(path: Path):
    with path.open("rb") as handle:
        x_df, y, groups, *rest = pickle.load(handle)
    return x_df.copy(), np.asarray(y), np.asarray(groups)


def get_pair_features(pair_search_json: Path, source_user: str, target_user: str):
    with pair_search_json.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    wanted = {source_user, target_user}
    for row in rows:
        if set(row["users"]) == wanted:
            return row["top_features"]
    raise ValueError(f"No pair feature set found for {source_user}, {target_user}")


def normalize_source_train_val(x_train: pd.DataFrame, x_val: pd.DataFrame):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    return x_train_scaled.astype(np.float32), x_val_scaled.astype(np.float32)


def normalize_target_independently(x_target: pd.DataFrame):
    scaler = StandardScaler()
    return scaler.fit_transform(x_target).astype(np.float32)


def target_metrics(model, x_target, y_target, device):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_target, dtype=torch.float32, device=device)
        logits = model.predict(x_tensor) if hasattr(model, "predict") else model(x_tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return {
        "target_acc": float(accuracy_score(y_target, preds)),
        "target_auroc": float(roc_auc_score(y_target, probs[:, 1])),
        "target_prauc": float(average_precision_score(y_target, probs[:, 1])),
    }


def source_val_metrics(model, x_val, y_val, device):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)
        logits = model.predict(x_tensor) if hasattr(model, "predict") else model(x_tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return {
        "source_val_acc": float(accuracy_score(y_val, preds)),
        "source_val_auroc": float(roc_auc_score(y_val, probs[:, 1])),
        "source_val_prauc": float(average_precision_score(y_val, probs[:, 1])),
    }


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


def main():
    parser = argparse.ArgumentParser(description="Pairwise source-target transfer experiment")
    parser.add_argument("--data", type=Path, default=Path("DA_Verification/CGDM/step_count_binary_personal-15min_mlp-auroc85.pkl"))
    parser.add_argument("--pair-search-json", type=Path, default=Path("DA_Verification/CGDM/pair_cluster_search.json"))
    parser.add_argument("--source-user", type=str, required=True)
    parser.add_argument("--target-user", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, default=Path("DA_Verification/CGDM/pair_transfer"))
    args = parser.parse_args()

    x_df, y, groups = load_pickle(args.data)
    features = get_pair_features(args.pair_search_json, args.source_user, args.target_user)

    source_mask = groups == args.source_user
    target_mask = groups == args.target_user
    x_source = x_df.loc[source_mask, features].copy()
    y_source = y[source_mask].astype(np.int64)
    x_target = x_df.loc[target_mask, features].copy()
    y_target = y[target_mask].astype(np.int64)

    x_train_df, x_val_df, y_train, y_val = train_test_split(
        x_source, y_source, test_size=0.2, random_state=42, stratify=y_source
    )
    x_train, x_val = normalize_source_train_val(x_train_df, x_val_df)
    x_target_np = normalize_target_independently(x_target)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.out_dir / f"{args.source_user}_to_{args.target_user}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "source_user": args.source_user,
        "target_user": args.target_user,
        "n_source": int(len(y_source)),
        "n_target": int(len(y_target)),
        "features": features,
    }

    mlp = build_mlp(len(features))
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
    results["MLP"] = {
        **source_val_metrics(mlp, x_val, y_val, device),
        **target_metrics(mlp, x_target_np, y_target, device),
    }
    torch.save(mlp.state_dict(), out_dir / "mlp_checkpoint.pt")

    cgdm = CGDM(input_dim=len(features), num_classes=2)
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
    results["CGDM"] = {
        **source_val_metrics(cgdm, x_val, y_val, device),
        **target_metrics(cgdm, x_target_np, y_target, device),
    }
    torch.save(cgdm.state_dict(), out_dir / "cgdm_checkpoint.pt")

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"saved {metrics_path}")


if __name__ == "__main__":
    main()
