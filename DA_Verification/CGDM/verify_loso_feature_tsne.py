import argparse
import json
import pickle
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.da_models import CGDM, DAModel, DANN, train_cgdm, train_dann, train_standard  # noqa: E402


FEATURES = [
    "Heartrate#STD#ImmediatePast_15",
    "CAL#AVG#ImmediatePast_15",
    "Heartrate#AVG#ImmediatePast_15",
]
PROBLEMATIC_CHARS = ["[", "]", "<", ">", "{", "}", "(", ")", ","]


def load_pickle(path: Path):
    with path.open("rb") as handle:
        x_df, y, groups, *rest = pickle.load(handle)
    return x_df.copy(), np.asarray(y), np.asarray(groups), rest


def normalize_data(
    x_train: pd.DataFrame,
    groups_train: np.ndarray,
    x_test: pd.DataFrame,
    groups_test: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = x_train.select_dtypes(exclude=[np.number])

    x_numeric_train = x_train[numeric_cols]
    train_vars = x_numeric_train.var()
    valid_features = train_vars[train_vars > 0].index.tolist()

    user_ids = np.unique(groups_train)
    train_normalized_parts = []
    test_normalized_parts = []

    for user in user_ids:
        user_train_mask = groups_train == user
        x_user_train = x_train.loc[user_train_mask, valid_features]
        if len(x_user_train) == 0:
            continue

        scaler = StandardScaler()
        x_user_train_scaled = pd.DataFrame(
            scaler.fit_transform(x_user_train),
            columns=x_user_train.columns,
            index=x_user_train.index,
        )
        train_normalized_parts.append(x_user_train_scaled)

        user_test_mask = groups_test == user
        x_user_test = x_test.loc[user_test_mask]
        if len(x_user_test) == 0:
            continue

        common_features = [feature for feature in valid_features if feature in x_user_test.columns]
        if not common_features:
            continue

        x_user_test_common = x_user_test[common_features].copy()
        x_user_test_common[common_features] = scaler.transform(x_user_test_common[common_features])
        test_normalized_parts.append(x_user_test_common)

    x_train_normalized = pd.concat(train_normalized_parts).sort_index()
    x_test_normalized = (
        pd.concat(test_normalized_parts).sort_index()
        if test_normalized_parts
        else pd.DataFrame(index=x_test.index)
    )

    if not categorical_cols.empty:
        x_train_final = pd.concat(
            [x_train_normalized, categorical_cols.loc[x_train_normalized.index]],
            axis=1,
        )
        x_test_final = pd.concat(
            [x_test_normalized, categorical_cols.reindex(x_test_normalized.index)],
            axis=1,
        )
    else:
        x_train_final = x_train_normalized
        x_test_final = x_test_normalized

    drop_train = [
        col for col in x_train_final.columns
        if any(char in str(col) for char in PROBLEMATIC_CHARS)
    ]
    if drop_train:
        x_train_final = x_train_final.drop(columns=drop_train)
    drop_test = [
        col for col in x_test_final.columns
        if any(char in str(col) for char in PROBLEMATIC_CHARS)
    ]
    if drop_test:
        x_test_final = x_test_final.drop(columns=drop_test)

    common_cols = [col for col in x_train_final.columns if col in x_test_final.columns]
    return x_train_final[common_cols], x_test_final[common_cols]


def normalize_groups_independently(x_df: pd.DataFrame, groups: np.ndarray) -> pd.DataFrame:
    numeric_cols = x_df.select_dtypes(include=[np.number]).columns.tolist()
    train_vars = x_df[numeric_cols].var()
    valid_features = train_vars[train_vars > 0].index.tolist()
    parts = []

    for user in np.unique(groups):
        user_mask = groups == user
        x_user = x_df.loc[user_mask, valid_features]
        if len(x_user) == 0:
            continue
        scaler = StandardScaler()
        x_user_scaled = pd.DataFrame(
            scaler.fit_transform(x_user),
            columns=x_user.columns,
            index=x_user.index,
        )
        parts.append(x_user_scaled)

    x_norm = pd.concat(parts).sort_index()
    drop_cols = [
        col for col in x_norm.columns
        if any(char in str(col) for char in PROBLEMATIC_CHARS)
    ]
    if drop_cols:
        x_norm = x_norm.drop(columns=drop_cols)
    return x_norm


def sample_for_plot(x, y, max_points, seed):
    if max_points is None or len(x) <= max_points:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_points, replace=False)
    return x[idx], y[idx]


def plot_tsne(
    source_x,
    target_x,
    source_y,
    target_y,
    out_path,
    title,
    perplexity,
    max_points,
    seed,
    max_iter,
):
    source_x, source_y = sample_for_plot(source_x, source_y, max_points, seed)
    target_x, target_y = sample_for_plot(target_x, target_y, max_points, seed + 1)

    all_x = np.concatenate([source_x, target_x], axis=0)
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(perplexity, max(5, len(all_x) - 1)),
        max_iter=max_iter,
    )
    coords = tsne.fit_transform(all_x)

    ns = len(source_x)
    s_coords = coords[:ns]
    t_coords = coords[ns:]

    plt.figure(figsize=(8, 6))
    for label, color in [(0, "#d55e00"), (1, "#0072b2")]:
        mask = source_y == label
        if np.any(mask):
            plt.scatter(
                s_coords[mask, 0], s_coords[mask, 1],
                c=color, alpha=0.35, marker="o", label=f"Source class {label}",
            )
        mask = target_y == label
        if np.any(mask):
            plt.scatter(
                t_coords[mask, 0], t_coords[mask, 1],
                c=color, alpha=0.35, marker="^", label=f"Target class {label}",
            )
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def extract_features(model, x, device):
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(x, dtype=torch.float32, device=device)
        if hasattr(model, "feature_extractor"):
            feats = model.feature_extractor(tensor)
        elif hasattr(model, "featurizer"):
            feats = model.featurizer(tensor)
        else:
            feats = tensor
    return feats.cpu().numpy()


def target_metrics(model, x_target, y_target, device):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_target, dtype=torch.float32, device=device)
        logits = model.predict(x_tensor) if hasattr(model, "predict") else model(x_tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    acc = float((preds == y_target).mean())
    try:
        auroc = float(roc_auc_score(y_target, probs[:, 1]))
    except Exception:
        auroc = float("nan")
    return acc, auroc


def build_model(model_name: str, input_dim: int):
    hparams = {
        "backbone": "MLP",
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.0,
    }
    if model_name == "MLP":
        return DAModel(input_dim=input_dim, num_classes=2, hparams=hparams)
    if model_name == "DANN":
        return DANN(input_dim=input_dim, num_classes=2, num_domains=2, hparams=hparams)
    if model_name == "CGDM":
        return CGDM(input_dim=input_dim, num_classes=2)
    raise ValueError(f"Unsupported model: {model_name}")


def train_weighted_mlp(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    epochs,
    batch_size,
    patience,
    device,
    lr=1e-4,
):
    model = model.to(device)
    class_counts = np.bincount(y_train.astype(np.int64), minlength=2)
    class_weights = len(y_train) / (2.0 * np.maximum(class_counts, 1))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    best_val_score = -float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                logits = model(x_batch)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend(y_batch.numpy())

        try:
            val_auroc = roc_auc_score(val_targets, val_probs)
        except Exception:
            val_auroc = 0.5

        if val_auroc > best_val_score:
            best_val_score = val_auroc
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def train_focal_mlp(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    epochs,
    batch_size,
    patience,
    device,
    lr=1e-4,
    gamma=2.0,
):
    model = model.to(device)
    class_counts = np.bincount(y_train.astype(np.int64), minlength=2)
    alpha = len(y_train) / (2.0 * np.maximum(class_counts, 1))
    criterion = FocalLoss(
        alpha=torch.tensor(alpha, dtype=torch.float32, device=device),
        gamma=gamma,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    best_val_score = -float("inf")
    best_model_state = None
    patience_counter = 0

    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                logits = model(x_batch)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend(y_batch.numpy())

        try:
            val_auroc = roc_auc_score(val_targets, val_probs)
        except Exception:
            val_auroc = 0.5

        if val_auroc > best_val_score:
            best_val_score = val_auroc
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


def train_model(
    model_name,
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    x_target,
    y_target,
    device,
    epochs,
    batch_size,
    patience,
    weighted_loss,
    focal_loss,
):
    if model_name == "MLP":
        if focal_loss:
            return train_focal_mlp(
                model,
                x_train,
                y_train,
                x_val,
                y_val,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                device=device,
            )
        if weighted_loss:
            return train_weighted_mlp(
                model,
                x_train,
                y_train,
                x_val,
                y_val,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                device=device,
            )
        return train_standard(
            model, x_train, y_train, x_val, y_val,
            epochs=epochs, batch_size=batch_size, lr=1e-4, patience=patience, device=device,
        )
    if model_name == "DANN":
        d_train = np.zeros(len(y_train), dtype=np.int64)
        d_val = np.zeros(len(y_val), dtype=np.int64)
        return train_dann(
            model, x_train, y_train, d_train, x_val, y_val, d_val,
            epochs=epochs, batch_size=batch_size, lr=1e-4, patience=patience, device=device,
            X_target=x_target,
        )
    if model_name == "CGDM":
        return train_cgdm(
            model, x_train, y_train, x_target, y_target=y_target, X_val=x_val, y_val=y_val,
            epochs=epochs, batch_size=batch_size, lr=1e-4, patience=patience, device=device,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def prepare_fold_data(x_df, y, groups, target_user, seed, val_ratio):
    target_mask = groups == target_user
    source_mask = ~target_mask

    x_source = x_df.loc[source_mask, FEATURES].copy()
    y_source = y[source_mask]
    groups_source = groups[source_mask]
    x_target = x_df.loc[target_mask, FEATURES].copy()
    y_target = y[target_mask]
    groups_target = groups[target_mask]

    x_source_train, x_source_val, y_source_train, y_source_val, groups_train, groups_val = train_test_split(
        x_source,
        y_source,
        groups_source,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_source,
    )

    x_train_norm, x_val_norm = normalize_data(x_source_train, groups_train, x_source_val, groups_val)
    x_train_full_norm = normalize_groups_independently(x_source, groups_source)
    x_target_norm = normalize_groups_independently(x_target, groups_target)

    cols = [col for col in x_train_norm.columns if col in x_val_norm.columns and col in x_train_full_norm.columns and col in x_target_norm.columns]
    return {
        "x_source_raw": x_source.to_numpy(dtype=np.float32),
        "y_source": y_source.astype(np.int64),
        "x_target_raw": x_target.to_numpy(dtype=np.float32),
        "y_target": y_target.astype(np.int64),
        "x_train": x_train_norm[cols].to_numpy(dtype=np.float32),
        "y_train": y_source_train.astype(np.int64),
        "x_val": x_val_norm[cols].to_numpy(dtype=np.float32),
        "y_val": y_source_val.astype(np.int64),
        "x_source_full": x_train_full_norm[cols].to_numpy(dtype=np.float32),
        "x_target_full": x_target_norm[cols].to_numpy(dtype=np.float32),
        "feature_names": cols,
    }


def run_fold(
    target_user,
    fold_data,
    model_names,
    out_dir,
    device,
    epochs,
    batch_size,
    perplexity,
    max_tsne,
    seed,
    tsne_iter,
    patience,
    weighted_loss,
    focal_loss,
):
    fold_dir = out_dir / f"target_{target_user}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    plot_tsne(
        fold_data["x_source_full"],
        fold_data["x_target_full"],
        fold_data["y_source"],
        fold_data["y_target"],
        fold_dir / "tsne_raw_normalized.png",
        f"Raw normalized t-SNE: source vs target ({target_user})",
        perplexity,
        max_tsne,
        seed,
        tsne_iter,
    )

    results = []
    baseline_features = None

    for model_name in model_names:
        model = build_model(model_name, input_dim=fold_data["x_train"].shape[1])
        start = perf_counter()
        trained = train_model(
            model_name,
            model,
            fold_data["x_train"],
            fold_data["y_train"],
            fold_data["x_val"],
            fold_data["y_val"],
            fold_data["x_target_full"],
            fold_data["y_target"],
            device,
            epochs,
            batch_size,
            patience,
            weighted_loss,
            focal_loss,
        )
        elapsed = perf_counter() - start
        acc, auroc = target_metrics(trained, fold_data["x_target_full"], fold_data["y_target"], device)

        feature_source = extract_features(trained, fold_data["x_source_full"], device)
        feature_target = extract_features(trained, fold_data["x_target_full"], device)
        plot_tsne(
            feature_source,
            feature_target,
            fold_data["y_source"],
            fold_data["y_target"],
            fold_dir / f"tsne_{model_name.lower()}.png",
            f"{model_name} feature t-SNE ({target_user})",
            perplexity,
            max_tsne,
            seed,
            tsne_iter,
        )

        if model_name == "MLP":
            baseline_features = {
                "source": feature_source,
                "target": feature_target,
                "state_dict": deepcopy(trained.state_dict()),
            }
            torch.save(trained.state_dict(), fold_dir / "mlp_checkpoint.pt")
        if model_name == "CGDM":
            torch.save(trained.state_dict(), fold_dir / "cgdm_checkpoint.pt")

        results.append({
            "target_user": target_user,
            "model": model_name,
            "target_acc": acc,
            "target_auroc": auroc,
            "seconds": round(elapsed, 2),
            "num_source": int(len(fold_data["y_source"])),
            "num_target": int(len(fold_data["y_target"])),
            "num_features": int(fold_data["x_train"].shape[1]),
            "features": fold_data["feature_names"],
        })

    if baseline_features is not None and "CGDM" in model_names:
        plot_tsne(
            baseline_features["source"],
            baseline_features["target"],
            fold_data["y_source"],
            fold_data["y_target"],
            fold_dir / "tsne_before_adaptation_mlp_features.png",
            f"Before adaptation MLP features ({target_user})",
            perplexity,
            max_tsne,
            seed,
            tsne_iter,
        )

    with (fold_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="LOSO feature verification on Ubicomp step-count pickle")
    parser.add_argument("--data", type=Path, default=Path("DA_Verification/CGDM/step_count_binary_personal-15min.pkl"))
    parser.add_argument("--target-user", type=str, default=None, help="Run a single LOSO fold for this user")
    parser.add_argument("--all-users", action="store_true", help="Run LOSO for all users")
    parser.add_argument("--models", type=str, default="MLP,CGDM", help="Comma-separated subset of MLP,DANN,CGDM")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weighted-loss", action="store_true", help="Use class-weighted CE for MLP baseline")
    parser.add_argument("--focal-loss", action="store_true", help="Use focal loss for MLP baseline")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--tsne-iter", type=int, default=3000)
    parser.add_argument("--max-tsne", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("DA_Verification/CGDM/loso_feature_tsne"))
    args = parser.parse_args()

    x_df, y, groups, _ = load_pickle(args.data)
    missing = [feature for feature in FEATURES if feature not in x_df.columns]
    if missing:
        raise ValueError(f"Missing requested features: {missing}")

    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    valid_models = {"MLP", "DANN", "CGDM"}
    invalid_models = [name for name in model_names if name not in valid_models]
    if invalid_models:
        raise ValueError(f"Unsupported models: {invalid_models}")

    users = sorted(np.unique(groups).tolist())
    if args.all_users:
        target_users = users
    elif args.target_user:
        target_users = [args.target_user]
    else:
        target_users = [users[0]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results = []
    for target_user in target_users:
        fold_data = prepare_fold_data(x_df, y, groups, target_user, args.seed, args.val_ratio)
        fold_results = run_fold(
            target_user,
            fold_data,
            model_names,
            args.out_dir,
            device,
            args.epochs,
            args.batch_size,
            args.perplexity,
            args.max_tsne,
            args.seed,
            args.tsne_iter,
            args.patience,
            args.weighted_loss,
            args.focal_loss,
        )
        all_results.extend(fold_results)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "summary_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)

    print(f"Finished {len(target_users)} LOSO fold(s). Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
