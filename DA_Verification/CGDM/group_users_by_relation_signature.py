import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


TOP10_FEATURES = [
    "LOC_DST#ASC#ImmediatePast_15",
    "Heartrate#AVG#ImmediatePast_15",
    "LOC_DST#AVG#ImmediatePast_15",
    "CAL#AVG#ImmediatePast_15",
    "LOC_DST#STD#ImmediatePast_15",
    "Heartrate#STD#ImmediatePast_15",
    "LOC_LABEL#ETP##ImmediatePast_15",
    "CAL#MED#ImmediatePast_15",
    "ESM#LastLabel",
    "LOC_DST#MED#ImmediatePast_15",
]


def load_pickle(path: Path):
    with path.open("rb") as handle:
        x_df, y, groups, *rest = pickle.load(handle)
    return x_df.copy(), np.asarray(y), np.asarray(groups)


def normalize_single_user(x_user: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    return pd.DataFrame(
        scaler.fit_transform(x_user),
        columns=x_user.columns,
        index=x_user.index,
    )


def signed_univariate_auroc(feature_values: np.ndarray, labels: np.ndarray) -> float:
    pos = feature_values[labels == 1]
    neg = feature_values[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    total = 0.0
    for p in pos:
        total += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    auroc = total / (len(pos) * len(neg))
    sign = 1.0 if np.mean(pos) >= np.mean(neg) else -1.0
    return sign * abs(auroc - 0.5) * 2.0


def compute_user_signature(x_df: pd.DataFrame, y: np.ndarray, groups: np.ndarray, features: list[str]):
    signatures = {}
    rows = []
    for user in sorted(np.unique(groups)):
        mask = groups == user
        x_user = x_df.loc[mask, features]
        y_user = y[mask]
        x_user_norm = normalize_single_user(x_user)

        signature = []
        for feature in features:
            values = x_user_norm[feature].to_numpy()
            signed_auc = signed_univariate_auroc(values, y_user)
            mean_diff = float(values[y_user == 1].mean() - values[y_user == 0].mean())
            signature.extend([signed_auc, mean_diff])
        signatures[user] = np.asarray(signature, dtype=np.float32)
        rows.append(
            {
                "user": user,
                "n_samples": int(mask.sum()),
                "pos_rate": float(np.mean(y_user)),
            }
        )
    return signatures, pd.DataFrame(rows)


def cluster_signatures(signatures: dict[str, np.ndarray], min_k: int = 2, max_k: int = 6):
    users = list(signatures.keys())
    matrix = np.vstack([signatures[user] for user in users])

    best = None
    for k in range(min_k, min(max_k, len(users) - 1) + 1):
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(matrix)
        score = silhouette_score(matrix, labels)
        if best is None or score > best["silhouette"]:
            best = {"k": k, "labels": labels, "silhouette": float(score)}

    clusters = {}
    for user, label in zip(users, best["labels"]):
        clusters.setdefault(int(label), []).append(user)
    return best, clusters, users, matrix


def cosine_similarity_matrix(matrix: np.ndarray):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = matrix / norms
    return normalized @ normalized.T


def nearest_neighbors(users, sim_matrix, top_k=5):
    neighbors = {}
    for i, user in enumerate(users):
        order = np.argsort(sim_matrix[i])[::-1]
        ranked = []
        for j in order:
            if i == j:
                continue
            ranked.append({"user": users[j], "similarity": float(sim_matrix[i, j])})
            if len(ranked) >= top_k:
                break
        neighbors[user] = ranked
    return neighbors


def main():
    parser = argparse.ArgumentParser(description="Group users by feature-label relation signatures")
    parser.add_argument("--data", type=Path, default=Path("DA_Verification/CGDM/step_count_binary_personal-15min_mlp-auroc85.pkl"))
    parser.add_argument("--out-prefix", type=Path, default=Path("DA_Verification/CGDM/relation_signature_groups"))
    args = parser.parse_args()

    x_df, y, groups = load_pickle(args.data)
    signatures, stats_df = compute_user_signature(x_df, y, groups, TOP10_FEATURES)
    best, clusters, users, matrix = cluster_signatures(signatures)
    sim_matrix = cosine_similarity_matrix(matrix)
    neighbors = nearest_neighbors(users, sim_matrix, top_k=5)

    cluster_rows = []
    for cluster_id, members in sorted(clusters.items()):
        for user in members:
            cluster_rows.append({"cluster": cluster_id, "user": user})

    csv_path = args.out_prefix.with_name(args.out_prefix.name + ".csv")
    json_path = args.out_prefix.with_name(args.out_prefix.name + ".json")

    pd.DataFrame(cluster_rows).sort_values(["cluster", "user"]).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "data": str(args.data),
                "features": TOP10_FEATURES,
                "best_n_clusters": best["k"],
                "silhouette": best["silhouette"],
                "clusters": clusters,
                "nearest_neighbors": neighbors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"best_k={best['k']} silhouette={best['silhouette']:.4f}")
    for cluster_id, members in sorted(clusters.items()):
        print(f"cluster {cluster_id}: {members}")
    print(f"saved {csv_path}")
    print(f"saved {json_path}")


if __name__ == "__main__":
    main()
