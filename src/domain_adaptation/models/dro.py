"""DRO-style MLP training pipeline with Optuna tuning."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .common import ArrayDataset, safe_accuracy, safe_auc, safe_auprc





try:  # pragma: no cover - optional dependency for tuning
    import optuna
    from optuna.exceptions import TrialPruned
except Exception:  # pragma: no cover
    optuna = None
    TrialPruned = Exception



@dataclass
class DROConfig:
    input_dim: int
    hidden_dim_choices: Tuple[int, ...] = (64, 128, 256)
    num_layer_choices: Tuple[int, ...] = (1, 2, 3)
    finetune_epoch_choices: Tuple[int, ...] = (100, 200, 300)
    pretrain_epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    grad_clip: float = 1.0
    tuning_trials: int = 8
    tuning_patience: int = 20
    device: Optional[str] = None


@dataclass
class DROSingleRunResult:
    mode: str
    train_auroc: float
    val_auroc: float
    test_auroc: float
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    train_auprc: float
    val_auprc: float
    test_auprc: float
    pretrain_val_auroc: float
    pretrain_val_accuracy: float
    pretrain_val_auprc: float
    stage_durations: Dict[str, float]
    freeze_backbone: bool
    scratch: bool
    hyperparams: Dict[str, object]
    trial_score: float


class DROBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(1, num_layers)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x)
        return self.head(features).squeeze(-1)

    def freeze_backbone(self) -> None:
        for param in self.net.parameters():
            param.requires_grad = False


class DROPipeline:
    def __init__(self, config: DROConfig):
        if optuna is None:
            raise ImportError("optuna is required to run the DRO pipeline")
        self.config = config
        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.BCEWithLogitsLoss()

    def _seed(self, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _to_tensor_dataset(self, dataset: ArrayDataset) -> Optional[TensorDataset]:
        if dataset.X.size == 0:
            return None
        X = torch.from_numpy(dataset.X.astype(np.float32))
        y = torch.from_numpy(dataset.y.astype(np.float32))
        return TensorDataset(X, y)

    def _make_loader(self, dataset: Optional[TensorDataset], shuffle: bool) -> Optional[DataLoader]:
        if dataset is None:
            return None
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle, drop_last=False)

    def _evaluate_dataset(self, model: DROBackbone, dataset: Optional[TensorDataset]) -> Dict[str, float]:
        if dataset is None:
            return {"auroc": float("nan"), "accuracy": float("nan"), "auprc": float("nan")}
        loader = self._make_loader(dataset, shuffle=False)
        if loader is None:
            return {"auroc": float("nan"), "accuracy": float("nan"), "auprc": float("nan")}
        model.eval()
        preds: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                logits = model(xb)
                preds.append(torch.sigmoid(logits).cpu().numpy())
                labels.append(yb.numpy())
        if not preds:
            return {"auroc": float("nan"), "accuracy": float("nan"), "auprc": float("nan")}
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(labels)
        return {
            "auroc": safe_auc(y_true, y_pred),
            "accuracy": safe_accuracy(y_true, y_pred),
            "auprc": safe_auprc(y_true, y_pred),
        }

    def _train_epochs(
        self,
        model: DROBackbone,
        train_loader: Optional[DataLoader],
        val_dataset: Optional[TensorDataset],
        epochs: int,
        seed: int,
    ) -> Tuple[Optional[Dict[str, float]], float]:
        if train_loader is None or epochs <= 0:
            return None, 0.0
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        best_state = copy.deepcopy(model.state_dict())
        best_score = float("-inf")
        start = perf_counter()
        self._seed(seed)
        for _ in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                if self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                optimizer.step()
            # Track best based on validation AUROC (fallback to accuracy if NaN)
            val_metrics = self._evaluate_dataset(model, val_dataset)
            score = val_metrics["auroc"]
            if np.isnan(score):
                score = val_metrics["accuracy"]
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        duration = perf_counter() - start
        final_metrics = self._evaluate_dataset(model, val_dataset)
        final_metrics.setdefault("auroc", float("nan"))
        final_metrics.setdefault("accuracy", float("nan"))
        final_metrics.setdefault("auprc", float("nan"))
        return final_metrics, duration

    def _build_model(self, hidden_dim: int, num_layers: int) -> DROBackbone:
        model = DROBackbone(
            input_dim=self.config.input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=self.config.dropout,
        )
        return model.to(self.device)

    def _tune_hyperparams(
        self,
        seed: int,
        train_ds: ArrayDataset,
        val_ds: ArrayDataset,
    ) -> Tuple[int, int, int, float]:
        if optuna is None or not train_ds.is_valid() or not val_ds.is_valid():
            hidden = self.config.hidden_dim_choices[0]
            layers = self.config.num_layer_choices[0]
            epochs = self.config.finetune_epoch_choices[0]
            return hidden, layers, epochs, float("nan")

        train_tensor = self._to_tensor_dataset(train_ds)
        val_tensor = self._to_tensor_dataset(val_ds)

        def objective(trial: "optuna.Trial") -> float:
            hidden_dim = trial.suggest_categorical("hidden_dim", list(self.config.hidden_dim_choices))
            num_layers = trial.suggest_categorical("num_layers", list(self.config.num_layer_choices))
            epochs = trial.suggest_categorical("finetune_epochs", list(self.config.finetune_epoch_choices))
            model = self._build_model(hidden_dim, num_layers)
            loader = self._make_loader(train_tensor, shuffle=True)
            val_metrics, _ = self._train_epochs(model, loader, val_tensor, int(epochs), seed + trial.number)
            if val_metrics is None:
                raise TrialPruned()
            score = val_metrics.get("auroc")
            if score is None or np.isnan(score):
                raise TrialPruned()
            return score

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        try:
            study.optimize(lambda trial: objective(trial), n_trials=self.config.tuning_trials, n_jobs=1)
        except TrialPruned:
            pass
        if not study.best_trials:
            hidden = self.config.hidden_dim_choices[0]
            layers = self.config.num_layer_choices[0]
            epochs = self.config.finetune_epoch_choices[0]
            return hidden, layers, epochs, float("nan")
        best_params = study.best_trial.params
        return (
            int(best_params.get("hidden_dim", self.config.hidden_dim_choices[0])),
            int(best_params.get("num_layers", self.config.num_layer_choices[0])),
            int(best_params.get("finetune_epochs", self.config.finetune_epoch_choices[0])),
            float(study.best_trial.value),
        )

    def _finalize_metrics(
        self,
        model: DROBackbone,
        train_tensor: Optional[TensorDataset],
        val_tensor: Optional[TensorDataset],
        eval_tensor: Optional[TensorDataset],
    ) -> Dict[str, float]:
        train_metrics = self._evaluate_dataset(model, train_tensor)
        val_metrics = self._evaluate_dataset(model, val_tensor)
        test_metrics = self._evaluate_dataset(model, eval_tensor)
        return {
            "train_auroc": train_metrics["auroc"],
            "val_auroc": val_metrics["auroc"],
            "test_auroc": test_metrics["auroc"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "train_auprc": train_metrics["auprc"],
            "val_auprc": val_metrics["auprc"],
            "test_auprc": test_metrics["auprc"],
        }

    def run(
        self,
        *,
        seed: int,
        pretrain: Optional[ArrayDataset],
        pretrain_val: Optional[ArrayDataset],
        train: ArrayDataset,
        val: ArrayDataset,
        evaluation: ArrayDataset,
    ) -> List[DROSingleRunResult]:
        tuning_hidden, tuning_layers, tuning_epochs, tuning_score = self._tune_hyperparams(seed, train, val)

        train_tensor = self._to_tensor_dataset(train)
        val_tensor = self._to_tensor_dataset(val)
        eval_tensor = self._to_tensor_dataset(evaluation)
        pretrain_tensor = self._to_tensor_dataset(pretrain) if pretrain is not None else None
        pretrain_val_tensor = self._to_tensor_dataset(pretrain_val) if pretrain_val is not None else None

        stage_durations_common: Dict[str, float] = {}
        pretrain_val_metrics = {"auroc": float("nan"), "accuracy": float("nan"), "auprc": float("nan")}
        pretrained_state: Optional[Dict[str, torch.Tensor]] = None
        if pretrain_tensor is not None:
            model = self._build_model(tuning_hidden, tuning_layers)
            loader = self._make_loader(pretrain_tensor, shuffle=True)
            pretrain_metrics, elapsed = self._train_epochs(model, loader, pretrain_val_tensor, self.config.pretrain_epochs, seed)
            stage_durations_common["pretrain_seconds"] = elapsed
            if pretrain_metrics is not None:
                pretrain_val_metrics = pretrain_metrics
            pretrained_state = copy.deepcopy(model.state_dict())
        results: List[DROSingleRunResult] = []

        def _run_variant(mode: str, freeze: bool, scratch: bool, seed_offset: int) -> None:
            variant_model = self._build_model(tuning_hidden, tuning_layers)
            if not scratch and pretrained_state is not None:
                variant_model.load_state_dict(pretrained_state)
            if freeze:
                variant_model.freeze_backbone()
            loader = self._make_loader(train_tensor, shuffle=True)
            val_metrics, elapsed = self._train_epochs(
                variant_model,
                loader,
                val_tensor,
                tuning_epochs,
                seed + seed_offset,
            )
            stage_times = dict(stage_durations_common) if not scratch else {}
            if freeze:
                stage_times["finetune_seconds_freeze"] = elapsed
            elif scratch:
                stage_times["target_only_seconds"] = elapsed
            else:
                stage_times["finetune_seconds_unfreeze"] = elapsed
            metric_bundle = self._finalize_metrics(
                variant_model,
                train_tensor,
                val_tensor,
                eval_tensor,
            )
            result = DROSingleRunResult(
                mode=mode,
                train_auroc=metric_bundle["train_auroc"],
                val_auroc=metric_bundle["val_auroc"],
                test_auroc=metric_bundle["test_auroc"],
                train_accuracy=metric_bundle["train_accuracy"],
                val_accuracy=metric_bundle["val_accuracy"],
                test_accuracy=metric_bundle["test_accuracy"],
                train_auprc=metric_bundle["train_auprc"],
                val_auprc=metric_bundle["val_auprc"],
                test_auprc=metric_bundle["test_auprc"],
                pretrain_val_auroc=pretrain_val_metrics["auroc"],
                pretrain_val_accuracy=pretrain_val_metrics["accuracy"],
                pretrain_val_auprc=pretrain_val_metrics["auprc"],
                stage_durations=stage_times,
                freeze_backbone=freeze,
                scratch=scratch,
                hyperparams={
                    "dro_hidden_dim": tuning_hidden,
                    "dro_num_layers": tuning_layers,
                    "dro_epochs": tuning_epochs,
                },
                trial_score=tuning_score,
            )
            results.append(result)

        if pretrained_state is not None:
            _run_variant("pretrain_finetune_freeze", freeze=True, scratch=False, seed_offset=0)
            _run_variant("pretrain_finetune_unfreeze", freeze=False, scratch=False, seed_offset=1)
        _run_variant("target_only", freeze=False, scratch=True, seed_offset=2)
        return results
