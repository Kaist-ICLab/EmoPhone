"""CDTrans training pipeline integrating pretraining and domain adaptation."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from .common import ArrayDataset, safe_accuracy, safe_auc, safe_auprc





DEFAULT_PRIORITY_PREFIXES: Tuple[str, ...] = (
    "DATA_MRCV",
    "DATA_RCV",
    "DATA_SNT",
    "DATA_MSNT",
    "LOC_CLS",
    "PIF",
    "keyevent_TIME",
    "WLS",
    "Sleep",
    "APP_DUR_SYSTEM",
    "ONOFF",
    "Dozemode",
    "PWR",
)


# ---------------------------------------------------------------------------
# Model building blocks (adapted from scripts/experiments/loso_cdtrans_vs_lgbm.py)
# ---------------------------------------------------------------------------


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.feature_embedding = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_embedding(x)


class FeatureBlockTokenizer(nn.Module):
    """
    TokenLearner-style feature pooling (Ryoo et al., NeurIPS'21) adapted for tabular sensors.
    Supports grouping by prefix or prefix+time slot to preserve circadian patterns.
    """

    SLOT_PATTERN = re.compile(r"(Today|Yesterday)([A-Za-z]+)")

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        *,
        feature_names: Optional[Sequence[str]] = None,
        max_block_size: int = 128,
        grouping: str = "prefix",
        priority_prefixes: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.max_block_size = max(1, int(max_block_size))
        self.priority_prefixes: Optional[Set[str]] = (
            set(priority_prefixes) if priority_prefixes else None
        )
        self.block_indices: List[torch.Tensor] = []
        self.projectors = nn.ModuleList()
        blocks, slot_ids, slot_vocab_size = self._build_blocks(
            input_dim,
            feature_names,
            grouping,
            self.priority_prefixes,
        )
        if not blocks:
            raise ValueError("Unable to construct feature blocks")
        for block in blocks:
            indices = torch.tensor(block, dtype=torch.long)
            self.block_indices.append(indices)
            self.projectors.append(nn.Linear(len(block), d_model))

        self.block_slot_ids: List[int] = slot_ids
        self.token_embedding = nn.Embedding(len(self.block_indices), d_model)
        self.slot_embedding = (
            nn.Embedding(slot_vocab_size, d_model) if slot_vocab_size > 0 else None
        )

    def _extract_slot(self, name: str) -> Optional[str]:
        match = self.SLOT_PATTERN.search(name)
        if match:
            return match.group(2)
        if "Morning" in name or "Night" in name or "Afternoon" in name or "Evening" in name or "Dawn" in name:
            return name.split("Today")[-1] if "Today" in name else name
        return None

    def _build_blocks(
        self,
        input_dim: int,
        feature_names: Optional[Sequence[str]],
        grouping: str,
        priority_prefixes: Optional[Set[str]],
    ) -> Tuple[List[List[int]], List[int], int]:
        if feature_names is None or grouping not in {"prefix", "prefix_slot"}:
            blocks = [
                list(range(start, min(start + self.max_block_size, input_dim)))
                for start in range(0, input_dim, self.max_block_size)
            ]
            return blocks, [-1] * len(blocks), 0
        buckets: Dict[str, List[int]] = defaultdict(list)
        slot_vocab: Dict[str, int] = {}
        block_slot_ids: List[int] = []
        for idx, name in enumerate(feature_names):
            prefix = name.split("#", 1)[0] if "#" in name else name
            slot = self._extract_slot(name) if grouping == "prefix_slot" else None
            prefix_key = prefix
            if priority_prefixes and prefix not in priority_prefixes:
                prefix_key = "__OTHER__"
            key = f"{prefix_key}|{slot}" if slot else prefix_key
            buckets[key].append(idx)
        blocks: List[List[int]] = []
        slot_per_block: List[int] = []
        for key, indices in buckets.items():
            slot = key.split("|", 1)[1] if "|" in key else None
            slot_id = -1
            if slot:
                slot_id = slot_vocab.setdefault(slot, len(slot_vocab))
            for start in range(0, len(indices), self.max_block_size):
                block = indices[start:start + self.max_block_size]
                blocks.append(block)
                slot_per_block.append(slot_id)
        covered = {idx for block in blocks for idx in block}
        if len(covered) < input_dim:
            remaining = [idx for idx in range(input_dim) if idx not in covered]
            for start in range(0, len(remaining), self.max_block_size):
                block = remaining[start:start + self.max_block_size]
                blocks.append(block)
                slot_per_block.append(-1)
        return blocks, slot_per_block, len(slot_vocab)

    @property
    def num_tokens(self) -> int:
        return len(self.block_indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens: List[torch.Tensor] = []
        for block_idx, (idx, projector) in enumerate(zip(self.block_indices, self.projectors)):
            gathered = torch.index_select(x, dim=1, index=idx.to(x.device))
            token = projector(gathered)
            if self.slot_embedding is not None:
                slot_id = self.block_slot_ids[block_idx]
                if slot_id >= 0:
                    slot_vec = self.slot_embedding(
                        torch.tensor(slot_id, device=x.device)
                    )
                    token = token + slot_vec
            tokens.append(token)
        stacked = torch.stack(tokens, dim=1)
        ids = torch.arange(
            self.num_tokens,
            device=x.device,
            dtype=torch.long,
        )
        return stacked + self.token_embedding(ids).unsqueeze(0)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with optional Performer-style linearisation (ELU feature map).
    """

    def __init__(
        self,
        d_model: int,
        h: int,
        dropout: float,
        attention_type: str = "linear",
        nb_features: int = 64,
    ):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention_type = attention_type
        self.eps = 1e-6
        self.nb_features = max(8, nb_features)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0 + self.eps

    def _scaled_dot(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        scores = scores.softmax(dim=-1)
        scores = self.dropout(scores)
        return scores @ v

    def _linear_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_prime = self._feature_map(q)
        k_prime = self._feature_map(k)
        kv = torch.einsum("bhln,bhld->bhnd", k_prime, v)
        z = 1.0 / (torch.einsum("bhln,bhn->bhl", q_prime, k_prime.sum(dim=2)) + self.eps)
        out = torch.einsum("bhln,bhnd->bhld", q_prime, kv)
        return out * z.unsqueeze(-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch = q.size(0)
        q_proj = self.w_q(q).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        k_proj = self.w_k(k).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        v_proj = self.w_v(v).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        if self.attention_type == "softmax":
            attn = self._scaled_dot(q_proj, k_proj, v_proj)
        else:
            attn = self._linear_attention(q_proj, k_proj, v_proj)
        x = attn.transpose(1, 2).contiguous().view(batch, -1, self.h * self.d_k)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class CDTransLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, attention_type: str, nb_features: int):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout, attention_type=attention_type, nb_features=nb_features)
        self.residual1 = ResidualConnection(dropout)
        self.residual2 = ResidualConnection(dropout)
        self.ff = FeedForwardBlock(d_model, d_ff=4 * d_model, dropout=dropout)

    def forward(self,
                h_s_prev: torch.Tensor,
                h_t_prev: torch.Tensor,
                h_st_prev: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_s = self.residual1(h_s_prev, lambda x: self.mha(x, x, x))
        h_s = self.residual2(h_s, self.ff)
        h_t = self.residual1(h_t_prev, lambda x: self.mha(x, x, x))
        h_t = self.residual2(h_t, self.ff)
        h_st = self.residual1(h_s_prev, lambda x: self.mha(x, h_t_prev, h_t_prev))
        h_st = self.residual2(h_st, self.ff)
        if h_st_prev is not None:
            h_st = h_st + h_st_prev
        return h_s, h_t, h_st

    def train_source(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual1(x, lambda x: self.mha(x, x, x))
        x = self.residual2(x, self.ff)
        return x


class CDTransEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_layers: int, dropout: float, attention_type: str, nb_features: int):
        super().__init__()
        self.layers = nn.ModuleList([
            CDTransLayer(d_model, n_heads, dropout, attention_type, nb_features)
            for _ in range(num_layers)
        ])

    def forward(self, h_s: torch.Tensor, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_st = None
        for layer in self.layers:
            h_s, h_t, h_st = layer(h_s, h_t, h_st)
        return h_s, h_t, h_st

    def train_source(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer.train_source(x)
        return x


class CDTransModel(nn.Module):
    def __init__(self,
                 f_dim: int,
                 d_model: int = 256,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 feature_names: Optional[Sequence[str]] = None,
                 max_block_size: int = 128,
                 grouping: str = "prefix_slot",
                 attention_type: str = "linear",
                 linear_attn_features: int = 64,
                 priority_prefixes: Optional[Sequence[str]] = None):
        super().__init__()
        self.tokenizer = FeatureBlockTokenizer(
            f_dim,
            d_model,
            feature_names=feature_names,
            max_block_size=max_block_size,
            grouping=grouping,
            priority_prefixes=priority_prefixes,
        )
        self.seq_len = self.tokenizer.num_tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.encoder = CDTransEncoder(d_model, n_heads, n_layers, dropout, attention_type, linear_attn_features)
        self.classifier = nn.Linear(d_model, num_classes)
        self.attention_type = attention_type

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        h = self.tokenizer(x)
        cls = self.cls_token.expand(batch, -1, -1)
        return torch.cat([cls, h], dim=1)

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor):
        h_s = self.embed(x_s)
        h_t = self.embed(x_t)
        H_s, H_t, H_st = self.encoder(h_s, h_t)
        feat_s = H_s[:, 0, :]
        feat_t = H_t[:, 0, :]
        feat_st = H_st[:, 0, :]
        logit_s = self.classifier(feat_s)
        logit_t = self.classifier(feat_t)
        logit_st = self.classifier(feat_st)
        return logit_s, logit_t, logit_st, H_s, H_t, H_st

    def train_source(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.embed(x)
        H = self.encoder.train_source(h)
        feat = H[:, 0, :]
        logits = self.classifier(feat)
        return logits, feat


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


class EarlyStopper:
    def __init__(self, patience: int, mode: str = "max", min_delta: float = 0.0):
        self.patience = max(0, patience)
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.bad_epochs = 0

    @staticmethod
    def _is_nan(value: Optional[float]) -> bool:
        if value is None:
            return False
        try:
            return bool(np.isnan(value))
        except TypeError:
            return False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None or self._is_nan(self.best_score):
            return True
        if self.mode == "max":
            return score > (self.best_score + self.min_delta)
        return score < (self.best_score - self.min_delta)

    def step(self, score: float, model: nn.Module) -> Tuple[bool, bool]:
        if self._is_nan(score):
            self.bad_epochs += 1
            stop = self.bad_epochs >= self.patience
            return stop, False

        if self.best_score is None or self._is_improvement(score):
            self.best_score = score
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            self.bad_epochs = 0
            return False, True

        self.bad_epochs += 1
        stop = self.bad_epochs >= self.patience
        return stop, False

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _to_dataset(array_dataset: ArrayDataset) -> TensorDataset:
    features = torch.from_numpy(array_dataset.X.astype(np.float32))
    labels = torch.from_numpy(array_dataset.y.astype(np.int64))
    return TensorDataset(features, labels)


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "auroc": safe_auc(y_true, y_prob),
        "accuracy": safe_accuracy(y_true, y_prob),
        "auprc": safe_auprc(y_true, y_prob),
    }


def _evaluate(model: CDTransModel, dataset: ArrayDataset, device: torch.device, batch_size: int) -> Dict[str, float]:
    if dataset.X.size == 0:
        return {"auroc": float("nan"), "accuracy": float("nan"), "auprc": float("nan")}
    loader = DataLoader(_to_dataset(dataset), batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, _ = model.train_source(xb)
            if logits.size(1) == 2:
                prob = torch.softmax(logits, dim=1)[:, 1]
            else:
                prob = torch.sigmoid(logits.squeeze(-1))
            probs.append(prob.cpu().numpy())
            labels.append(yb.cpu().numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(labels)
    return _compute_metrics(y_true, y_prob)


def _default_metrics() -> Dict[str, float]:
    return {"auroc": float("nan"), "accuracy": float("nan"), "auprc": float("nan")}


def _fmt_metric(value: float) -> str:
    if value is None or np.isnan(value):
        return "nan"
    return f"{value:.3f}"


def _dataset_with_indices(dataset: ArrayDataset) -> TensorDataset:
    features = torch.from_numpy(dataset.X.astype(np.float32))
    labels = torch.from_numpy(dataset.y.astype(np.int64))
    indices = torch.arange(len(dataset.y), dtype=torch.long)
    return TensorDataset(features, labels, indices)


def _extract_features(
    model: CDTransModel,
    dataset: ArrayDataset,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if dataset.X.size == 0:
        return np.zeros((0, model.classifier.in_features), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    loader = DataLoader(_dataset_with_indices(dataset), batch_size=batch_size, shuffle=False, drop_last=False)
    feats, labels, indices = [], [], []
    model.eval()
    with torch.no_grad():
        for xb, yb, idx in loader:
            xb = xb.to(device)
            logits, feat = model.train_source(xb)
            feats.append(feat.cpu())
            labels.append(yb.long())
            indices.append(idx)
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    order = torch.cat(indices, dim=0).numpy().argsort()
    feats = feats.numpy()[order]
    labels = labels.numpy()[order]
    return feats, labels


def _update_pseudo_labels(
    model: CDTransModel,
    dataset: ArrayDataset,
    device: torch.device,
    batch_size: int,
    prev_centers: Optional[np.ndarray],
    momentum: float,
    temperature: float,
    confidence_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    if dataset.X.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, model.classifier.in_features), dtype=np.float32),
            prev_centers,
            np.zeros((0,), dtype=bool),
        )
    loader = DataLoader(_dataset_with_indices(dataset), batch_size=batch_size, shuffle=False, drop_last=False)
    feats, probs, indices = [], [], []
    model.eval()
    with torch.no_grad():
        for xb, _, idx in loader:
            xb = xb.to(device)
            logits, feat = model.train_source(xb)
            probs.append(F.softmax(logits / max(temperature, 1e-6), dim=1).cpu())
            feats.append(feat.cpu())
            indices.append(idx)
    feats = torch.cat(feats, dim=0)
    probs = torch.cat(probs, dim=0)
    order = torch.cat(indices, dim=0).numpy().argsort()
    feats = feats.numpy()[order]
    probs = probs.numpy()[order]

    num_classes = probs.shape[1]
    centers = []
    for c in range(num_classes):
        weights = probs[:, c:c + 1]
        denom = weights.sum() + 1e-6
        centers.append((weights * feats).sum(axis=0) / denom)
    centers = np.stack(centers, axis=0)
    if prev_centers is not None and prev_centers.shape == centers.shape:
        centers = momentum * prev_centers + (1.0 - momentum) * centers

    dists = np.linalg.norm(feats[:, None, :] - centers[None, :, :], axis=2)
    init_labels = np.argmin(dists, axis=1)

    refined_centers = []
    for c in range(num_classes):
        mask = init_labels == c
        if np.any(mask):
            refined_centers.append(feats[mask].mean(axis=0))
        else:
            refined_centers.append(centers[c])
    refined_centers = np.stack(refined_centers, axis=0)

    dists = np.linalg.norm(feats[:, None, :] - refined_centers[None, :, :], axis=2)
    pseudo = np.argmin(dists, axis=1).astype(np.int64)
    confidence = probs.max(axis=1)
    valid_mask = confidence >= confidence_threshold
    pseudo_filtered = pseudo.copy()
    pseudo_filtered[~valid_mask] = -1
    return pseudo_filtered, feats, refined_centers, valid_mask


class _PairDataset(Dataset):
    def __init__(
        self,
        source_dataset: ArrayDataset,
        target_dataset: ArrayDataset,
        pseudo: np.ndarray,
        pairs: List[Tuple[int, int]],
    ):
        self.source_X = source_dataset.X.astype(np.float32)
        self.source_y = source_dataset.y.astype(np.int64)
        self.target_X = target_dataset.X.astype(np.float32)
        self.pseudo = pseudo.astype(np.int64)
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        s_idx, t_idx = self.pairs[idx]
        x_s = torch.from_numpy(self.source_X[s_idx])
        y_s = torch.tensor(int(self.source_y[s_idx]), dtype=torch.long)
        x_t = torch.from_numpy(self.target_X[t_idx])
        y_t = torch.tensor(int(self.pseudo[t_idx]), dtype=torch.long)
        return (x_s, y_s, torch.tensor(s_idx, dtype=torch.long)), (x_t, y_t, torch.tensor(t_idx, dtype=torch.long))


def _create_matched_pair_loader(
    model: CDTransModel,
    source_dataset: ArrayDataset,
    target_dataset: ArrayDataset,
    source_feats: np.ndarray,
    source_labels: np.ndarray,
    target_feats: np.ndarray,
    pseudo_labels: np.ndarray,
    valid_mask: np.ndarray,
    batch_size: int,
    top_k: int,
) -> Optional[DataLoader]:
    if source_feats.size == 0 or target_feats.size == 0:
        return None
    valid_indices = np.where(valid_mask)[0]
    if valid_indices.size == 0:
        return None
    num_classes = int(max(np.max(source_labels), np.max(pseudo_labels[valid_mask]))) + 1
    pairs: set[Tuple[int, int]] = set()
    for c in range(num_classes):
        src_idx = np.where(source_labels == c)[0]
        tgt_idx = np.where((pseudo_labels == c) & valid_mask)[0]
        if src_idx.size == 0 or tgt_idx.size == 0:
            continue
        tgt_feats_c = target_feats[tgt_idx]
        for s in src_idx:
            dists = np.linalg.norm(tgt_feats_c - source_feats[s], axis=1)
            order = np.argsort(dists)[: max(1, top_k)]
            for idx in order:
                pairs.add((int(s), int(tgt_idx[idx])))
        src_feats_c = source_feats[src_idx]
        for t in tgt_idx:
            dists = np.linalg.norm(src_feats_c - target_feats[t], axis=1)
            order = np.argsort(dists)[: max(1, top_k)]
            for idx in order:
                pairs.add((int(src_idx[idx]), int(t)))
    if not pairs:
        return None
    dataset = _PairDataset(source_dataset, target_dataset, pseudo_labels, list(pairs))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class CDTransConfig:
    input_dim: int
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    feature_names: Optional[Tuple[str, ...]] = None
    max_block_size: int = 128
    block_grouping: str = "prefix_slot"
    attention_type: str = "linear"
    linear_attn_features: int = 64
    priority_prefixes: Tuple[str, ...] = DEFAULT_PRIORITY_PREFIXES
    lr: float = 5e-4
    weight_decay: float = 1e-5
    batch_size: int = 256
    pretrain_epochs: int = 100
    adapt_epochs: int = 100
    pretrain_patience: int = 100
    adapt_patience: int = 100
    early_stop_metric: str = "auroc"
    early_stop_min_delta: float = 0.0
    lambda_src: float = 0.3
    lambda_tgt: float = 1.0
    lambda_distill: float = 1.0
    lambda_align: float = 0.1
    temperature: float = 0.5
    pseudo_threshold: float = 0.7
    center_momentum: float = 0.9
    pair_top_k: int = 3


@dataclass
class CDTransRunResult:
    pretrain_val_metrics: Dict[str, float]
    pretrain_test_metrics: Dict[str, float]
    adapt_train_metrics: Optional[Dict[str, float]]
    adapt_val_metrics: Optional[Dict[str, float]]
    adapt_test_metrics: Dict[str, float]
    stage_durations: Dict[str, float]
    stage_epochs: Dict[str, int]
    best_pretrain_epoch: Optional[int]
    best_adapt_epoch: Optional[int]


class CDTransPipeline:
    def __init__(self, config: CDTransConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self._target_centers: Optional[np.ndarray] = None

    def _monitor_value(self, metrics: Dict[str, float]) -> float:
        key = self.config.early_stop_metric
        value = metrics.get(key)
        if value is None or np.isnan(value):
            fallback = metrics.get("auroc")
            if fallback is None or np.isnan(fallback):
                return float("nan")
            return fallback
        return value

    def run(
        self,
        *,
        seed: int,
        source_train: ArrayDataset,
        source_val: Optional[ArrayDataset],
        target_train: ArrayDataset,
        target_val: ArrayDataset,
        target_eval: ArrayDataset,
    ) -> CDTransRunResult:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = CDTransModel(
            f_dim=self.config.input_dim,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout,
            feature_names=self.config.feature_names,
            max_block_size=self.config.max_block_size,
            grouping=self.config.block_grouping,
            attention_type=self.config.attention_type,
            linear_attn_features=self.config.linear_attn_features,
            priority_prefixes=self.config.priority_prefixes,
        ).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        stage_durations: Dict[str, float] = {}
        stage_epochs: Dict[str, int] = {"pretrain_epochs": 0, "adapt_epochs": 0}

        source_loader = DataLoader(
            _to_dataset(source_train),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        ) if source_train.X.size > 0 else None
        source_val_metrics = _default_metrics()

        pretrain_stopper: Optional[EarlyStopper] = None
        if self.config.pretrain_patience >= 0:
            mode = "min" if self.config.early_stop_metric.endswith("loss") else "max"
            pretrain_stopper = EarlyStopper(self.config.pretrain_patience, mode=mode, min_delta=self.config.early_stop_min_delta)
        best_pretrain_epoch: Optional[int] = None

        if source_loader is not None:
            start = perf_counter()
            iterator = tqdm(
                range(1, self.config.pretrain_epochs + 1),
                desc="[CDTrans][Pretrain]",
                leave=False,
                dynamic_ncols=True,
                disable=False,
            )
            for epoch in iterator:
                model.train()
                running_loss = 0.0
                sample_count = 0
                for xb, yb in source_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits, _ = model.train_source(xb)
                    loss = self.criterion(logits, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * xb.size(0)
                    sample_count += xb.size(0)
                stage_epochs["pretrain_epochs"] = epoch

                avg_loss = running_loss / sample_count if sample_count else float("nan")
                train_metrics = _evaluate(model, source_train, self.device, self.config.batch_size)
                if source_val is not None and source_val.X.size > 0:
                    source_val_metrics = _evaluate(model, source_val, self.device, self.config.batch_size)
                    if pretrain_stopper:
                        monitor = self._monitor_value(source_val_metrics)
                        stop, is_best = pretrain_stopper.step(monitor, model)
                        if is_best:
                            best_pretrain_epoch = epoch
                        if stop:
                            break
                    iterator.set_postfix({
                        "epoch": epoch,
                        "loss": f"{avg_loss:.4f}",
                        "train_auroc": _fmt_metric(train_metrics["auroc"]),
                        "val_auroc": _fmt_metric(source_val_metrics["auroc"]),
                    })
                else:
                    iterator.set_postfix({
                        "epoch": epoch,
                        "loss": f"{avg_loss:.4f}",
                        "train_auroc": _fmt_metric(train_metrics["auroc"]),
                    })
            stage_durations["pretrain_seconds"] = perf_counter() - start

            if pretrain_stopper:
                pretrain_stopper.restore_best(model)
                if source_val is not None and source_val.X.size > 0:
                    source_val_metrics = _evaluate(model, source_val, self.device, self.config.batch_size)

        pretrain_test_metrics = (
            _evaluate(model, target_eval, self.device, self.config.batch_size)
            if target_eval.X.size > 0
            else _default_metrics()
        )
        print(
            "[CDTrans][Pretrain] Completed"
            f" best_epoch={best_pretrain_epoch if best_pretrain_epoch else stage_epochs['pretrain_epochs']}"
            f" test_auroc={_fmt_metric(pretrain_test_metrics['auroc'])}"
        )

        adapt_train_metrics: Optional[Dict[str, float]] = None
        adapt_val_metrics: Optional[Dict[str, float]] = None
        adapt_test_metrics = pretrain_test_metrics
        best_adapt_epoch: Optional[int] = None

        if target_train.X.size > 0 and source_train.X.size > 0:
            adapt_stopper: Optional[EarlyStopper] = None
            if self.config.adapt_patience >= 0:
                mode = "min" if self.config.early_stop_metric.endswith("loss") else "max"
                adapt_stopper = EarlyStopper(self.config.adapt_patience, mode=mode, min_delta=self.config.early_stop_min_delta)

            start = perf_counter()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.adapt_epochs)
            iterator = tqdm(
                range(1, self.config.adapt_epochs + 1),
                desc="[CDTrans][Adapt]",
                leave=False,
                dynamic_ncols=True,
                disable=False,
            )
            for epoch in iterator:
                source_feats, source_labels = _extract_features(model, source_train, self.device, self.config.batch_size)
                pseudo_labels, target_feats, updated_centers, valid_mask = _update_pseudo_labels(
                    model,
                    target_train,
                    self.device,
                    self.config.batch_size,
                    self._target_centers,
                    self.config.center_momentum,
                    self.config.temperature,
                    self.config.pseudo_threshold,
                )
                self._target_centers = updated_centers
                pair_loader = _create_matched_pair_loader(
                    model,
                    source_train,
                    target_train,
                    source_feats,
                    source_labels,
                    target_feats,
                    pseudo_labels,
                    valid_mask,
                    self.config.batch_size,
                    self.config.pair_top_k,
                )
                if pair_loader is None:
                    print("[CDTrans][Adapt] Skipping epoch due to empty matched pairs")
                    break
                model.train()
                running_loss = 0.0
                sample_count = 0
                for src_item, tgt_item in pair_loader:
                    x_src, y_src, _ = src_item
                    x_tgt, y_pseudo, _ = tgt_item
                    x_src = x_src.to(self.device)
                    y_src = y_src.to(self.device)
                    x_tgt = x_tgt.to(self.device)
                    y_pseudo = y_pseudo.to(self.device)
                    try:
                        logits_src, logits_tgt, logits_cross, H_s, H_t, _ = model(x_src, x_tgt)
                    except RuntimeError as exc:
                        print(
                            "[CDTrans][Adapt] RuntimeError",
                            f"x_src_shape={tuple(x_src.shape)}",
                            f"x_tgt_shape={tuple(x_tgt.shape)}",
                            f"d_model={self.config.d_model}",
                            f"seq_len={self.config.input_dim}",
                            flush=True,
                        )
                        raise
                    loss_src = self.criterion(logits_src, y_src)
                    teacher = torch.clamp(F.softmax(logits_cross.detach(), dim=1), min=1e-6)
                    loss_tgt = self.criterion(logits_tgt, y_pseudo)
                    loss_distill = F.kl_div(
                        F.log_softmax(logits_tgt, dim=1),
                        teacher,
                        reduction="batchmean",
                    )
                    feat_src = H_s[:, 0, :]
                    feat_tgt = H_t[:, 0, :]
                    loss_align = F.mse_loss(feat_src.mean(dim=0), feat_tgt.mean(dim=0))
                    loss = (
                        self.config.lambda_src * loss_src
                        + self.config.lambda_tgt * loss_tgt
                        + self.config.lambda_distill * loss_distill
                        + self.config.lambda_align * loss_align
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * x_src.size(0)
                    sample_count += x_src.size(0)

                stage_epochs["adapt_epochs"] = epoch

                adapt_train_metrics = _evaluate(model, target_train, self.device, self.config.batch_size)
                adapt_val_metrics = _evaluate(model, target_val, self.device, self.config.batch_size)
                adapt_test_metrics = _evaluate(model, target_eval, self.device, self.config.batch_size)
                avg_loss = running_loss / sample_count if sample_count else float("nan")
                iterator.set_postfix({
                    "epoch": epoch,
                    "loss": f"{avg_loss:.4f}",
                    "train_auroc": _fmt_metric(adapt_train_metrics["auroc"]),
                    "val_auroc": _fmt_metric(adapt_val_metrics["auroc"]),
                    "test_auroc": _fmt_metric(adapt_test_metrics["auroc"]),
                })

                if adapt_stopper:
                    monitor = self._monitor_value(adapt_val_metrics)
                    stop, is_best = adapt_stopper.step(monitor, model)
                    if is_best:
                        best_adapt_epoch = epoch
                    if stop:
                        break
                scheduler.step()

            stage_durations["adapt_seconds"] = perf_counter() - start

            if adapt_stopper:
                adapt_stopper.restore_best(model)
                adapt_train_metrics = _evaluate(model, target_train, self.device, self.config.batch_size)
                adapt_val_metrics = _evaluate(model, target_val, self.device, self.config.batch_size)
                adapt_test_metrics = _evaluate(model, target_eval, self.device, self.config.batch_size)
            print(
                "[CDTrans][Adapt] Completed"
                f" best_epoch={best_adapt_epoch if best_adapt_epoch else stage_epochs['adapt_epochs']}"
                f" test_auroc={_fmt_metric(adapt_test_metrics['auroc'])}"
            )

        return CDTransRunResult(
            pretrain_val_metrics=source_val_metrics,
            pretrain_test_metrics=pretrain_test_metrics,
            adapt_train_metrics=adapt_train_metrics,
            adapt_val_metrics=adapt_val_metrics,
            adapt_test_metrics=adapt_test_metrics,
            stage_durations=stage_durations,
            stage_epochs=stage_epochs,
            best_pretrain_epoch=best_pretrain_epoch,
            best_adapt_epoch=best_adapt_epoch,
        )
