"""SHOT: Source Hypothesis Transfer (Liang et al., 2020)."""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from models import train_torch_model

from .._da_helpers import (
    DAModel,
    EarlyStopTracker,
    _build_loaders,
    _evaluate_val,
    _finalize_training_metadata,
    _infinite_iterator,
)
from ..da_tllib_losses import entropy as tllib_entropy


class SHOT(DAModel):
    """
    SHOT: Source Hypothesis Transfer (Liang et al., 2020)
    Source-free Domain Adaptation.
    """

    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(SHOT, self).__init__(input_dim, num_classes, hparams)

    def forward(self, x):
        return self.predict(x)


def _shot_op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]
    return optimizer


def _shot_lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay
        param_group["weight_decay"] = 1e-3
        param_group["momentum"] = 0.9
        param_group["nesterov"] = True
    return optimizer


def _shot_obtain_label(
    loader, model, num_classes, distance="cosine", threshold=0, epsilon=1e-5, device="cuda"
):
    start = True
    with torch.no_grad():
        for X_batch, idx in loader:
            X_batch = X_batch.to(device)
            feat = model.feature_extractor(X_batch)
            outputs = model.classifier(feat)
            if start:
                all_fea = feat.float().cpu()
                all_output = outputs.float().cpu()
                start = False
            else:
                all_fea = torch.cat((all_fea, feat.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)

    all_output = F.softmax(all_output, dim=1)
    ent = torch.sum(-all_output * torch.log(all_output + epsilon), dim=1)
    _unknown_weight = 1 - ent / np.log(num_classes)
    _, predict = torch.max(all_output, 1)

    if distance == "cosine":
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        cls_count = np.eye(num_classes)[predict].sum(axis=0)
        labelset = np.where(cls_count > threshold)[0]
        dd = cdist(all_fea, initc[labelset], distance)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]
        aff = np.eye(num_classes)[predict]

    return predict.astype("int")


def train_shot(
    model,
    X_train,
    y_train,
    d_train,
    X_val,
    y_val,
    d_val,
    epochs=50,
    batch_size=64,
    lr=1e-3,
    patience=5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    X_target=None,
):
    """
    SHOT (official-style):
      1) Train source model on labeled source.
      2) Freeze classifier, adapt feature extractor with pseudo-labels + InfoMax on target.
    Requires unlabeled target samples (X_target).
    """
    if X_target is None:
        raise ValueError("SHOT requires X_target for UDA. Use --uda to provide target samples.")

    model = model.to(device)

    # Phase 1: Source pretraining
    model = train_torch_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        device=device,
    )
    source_pretrain_info = dict(getattr(model, "_training_info", {}))

    # Freeze classifier
    for p in model.classifier.parameters():
        p.requires_grad = False

    # Hyperparams (from official defaults)
    cls_par = model.hparams.get("shot_cls_par", 0.3)
    ent_par = model.hparams.get("shot_ent_par", 1.0)
    gent = model.hparams.get("shot_gent", True)
    ent = model.hparams.get("shot_ent", True)
    threshold = model.hparams.get("shot_threshold", 0)
    distance = model.hparams.get("shot_distance", "cosine")
    epsilon = model.hparams.get("shot_epsilon", 1e-5)
    adapt_epochs = model.hparams.get("shot_epochs", max(1, epochs // 2))
    interval = model.hparams.get("shot_interval", 15)
    lr_decay = model.hparams.get("shot_lr_decay", 0.1)

    # Target loaders with indices
    target_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_target, dtype=torch.float32), torch.arange(len(X_target), dtype=torch.long)
    )
    target_loader = torch.utils.data.DataLoader(
        target_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    target_loader_eval = torch.utils.data.DataLoader(
        target_dataset, batch_size=batch_size * 3, shuffle=False, drop_last=False
    )

    # Validation loader (source)
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer (SGD + official LR schedule)
    optimizer = torch.optim.SGD(model.feature_extractor.parameters(), lr=lr * lr_decay)
    optimizer = _shot_op_copy(optimizer)

    max_iter = max(1, adapt_epochs * len(target_loader))
    interval_iter = max(1, max_iter // interval) if interval > 0 else max_iter
    iter_num = 0

    best_val_score = -float("inf")
    best_model_state = None
    patience_counter = 0
    best_epoch = None
    early_stopped = False
    early_stop_epoch = None
    epochs_ran = 0
    epoch_history = []

    epoch_iterator = tqdm(range(adapt_epochs), desc="SHOT Adaptation")
    target_iter = iter(target_loader)
    mem_label = None

    for epoch in epoch_iterator:
        model.train()
        model.feature_extractor.train()
        model.classifier.eval()

        train_loss = 0.0
        for _ in range(len(target_loader)):
            try:
                X_t, idx = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                X_t, idx = next(target_iter)

            if X_t.size(0) == 1:
                continue

            if iter_num % interval_iter == 0 and cls_par > 0:
                model.eval()
                mem_label = _shot_obtain_label(
                    target_loader_eval,
                    model,
                    model.classifier[-1].out_features,
                    distance=distance,
                    threshold=threshold,
                    epsilon=epsilon,
                    device=device,
                )
                mem_label = torch.from_numpy(mem_label).to(device)
                model.train()
                model.classifier.eval()

            X_t = X_t.to(device)
            idx = idx.to(device)

            iter_num += 1
            _shot_lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

            features_t = model.feature_extractor(X_t)
            outputs_t = model.classifier(features_t)

            loss = torch.tensor(0.0, device=device)
            if cls_par > 0 and mem_label is not None:
                pseudo = mem_label[idx]
                loss = loss + cls_par * nn.CrossEntropyLoss()(outputs_t, pseudo)

            if ent:
                softmax_out = F.softmax(outputs_t, dim=1)
                entropy_loss = torch.mean(tllib_entropy(softmax_out))
                if gent:
                    msoftmax = softmax_out.mean(dim=0)
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + epsilon))
                    entropy_loss -= gentropy_loss
                loss = loss + ent_par * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(target_loader))
        val_loss, val_auroc = _evaluate_val(model, val_loader, device)
        epoch_num = epoch + 1
        epochs_ran = epoch_num

        if val_auroc > best_val_score:
            best_val_score = val_auroc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch_num
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_iterator.write(
                    f"Early stopping at epoch {epoch} (Best AUROC: {best_val_score:.4f})"
                )
                early_stopped = True
                early_stop_epoch = epoch_num
                break

        postfix = {
            "Loss": f"{train_loss:.4f}",
            "Val Loss": f"{val_loss:.4f}",
            "Val AUC": f"{val_auroc:.4f}",
        }
        epoch_iterator.set_postfix(postfix)
        epoch_history.append(
            {
                "epoch": epoch_num,
                "train_loss": round(float(train_loss), 6),
                "val_loss": round(float(val_loss), 6),
                "val_auroc": round(float(val_auroc), 6),
            }
        )

    if best_model_state:
        model.load_state_dict(best_model_state)

    _finalize_training_metadata(
        model,
        optimizer=optimizer.__class__.__name__,
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        early_stop_epoch=early_stop_epoch,
        epochs_ran=epochs_ran,
        max_epochs=adapt_epochs,
        batch_size=batch_size,
        patience=patience,
        lr=lr * lr_decay,
        weight_decay=1e-3,
        best_metric_value=round(float(best_val_score), 6) if best_epoch is not None else None,
        epoch_history=epoch_history,
        extra={
            "phases": {
                "source_pretrain": source_pretrain_info,
                "target_adaptation": {"epochs_ran": epochs_ran},
            },
            "shot_interval": interval,
        },
    )
    return model


# CBST (Class-Balanced Self-Training)
