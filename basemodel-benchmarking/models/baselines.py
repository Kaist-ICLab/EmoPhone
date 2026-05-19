"""Plain torch baselines (MLP and ResNet) and the shared training/eval loop."""

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

from ._helpers import FIXED_BATCH_SIZE, attach_training_metadata


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2)) # Binary classification
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ResNetBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super(ResNetBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_blocks=2, dropout=0.3):
        super(ResNet, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        out = self.input_proj(x)
        for block in self.blocks:
            out = block(out)
        return self.output(out)

# --- Training Loop for DL ---


def train_torch_model(model, X_train, y_train, X_val, y_val, 
                      epochs=50, batch_size=FIXED_BATCH_SIZE, lr=1e-3, weight_decay=0.0, patience=5,
                      device='cuda' if torch.cuda.is_available() else 'cpu'):
    
    model = model.to(device)
    batch_size = int(batch_size or FIXED_BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    pin = torch.cuda.is_available()
    train_drop_last = len(train_dataset) > batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=train_drop_last, pin_memory=pin, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin, num_workers=2, persistent_workers=True)
    
    criterion = nn.CrossEntropyLoss()
    
    epoch_iterator = tqdm(range(epochs), desc="Training Epochs")

    
    best_val_score = -float('inf')  # Changed from best_val_loss (inf)
    best_model_state = None
    patience_counter = 0
    best_epoch = None
    early_stopped = False
    early_stop_epoch = None
    epochs_ran = 0
    epoch_history = []

    for epoch in epoch_iterator:
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= max(1, len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_probs = []
        val_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                # Collect probs for AUROC
                probs = torch.softmax(outputs, dim=1)[:, 1] # Binary classification assumption (pos class)
                val_probs.extend(probs.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())
        
        val_loss /= max(1, len(val_loader))
        try:
            val_auroc = roc_auc_score(val_targets, val_probs)
        except ValueError:
            val_auroc = 0.5
        epoch_num = epoch + 1
        epochs_ran = epoch_num

        # Early stopping (Maximize AUROC)
        if val_auroc > best_val_score:
            best_val_score = val_auroc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch_num
            # epoch_iterator.write(f"New Best AUROC: {val_auroc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_iterator.write(f"Early stopping at epoch {epoch} (Best AUROC: {best_val_score:.4f})")
                early_stopped = True
                early_stop_epoch = epoch_num
                break
        postfix = {'Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}', 'Val AUC': f'{val_auroc:.4f}'}
        epoch_iterator.set_postfix(postfix)
        epoch_history.append({
            "epoch": epoch_num,
            "train_loss": round(float(train_loss), 6),
            "val_loss": round(float(val_loss), 6),
            "val_auroc": round(float(val_auroc), 6),
        })

    if best_model_state:
        model.load_state_dict(best_model_state)

    attach_training_metadata(
        model,
        optimizer=optimizer.__class__.__name__,
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        early_stop_epoch=early_stop_epoch,
        epochs_ran=epochs_ran,
        max_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        best_metric_value=round(float(best_val_score), 6) if best_epoch is not None else None,
        model_selection_metric="val_auroc",
        epoch_history=epoch_history,
    )
    return model

def evaluate_model(model, X_test, y_test, device='cuda' if torch.cuda.is_available() else 'cpu'):
    is_torch = isinstance(model, nn.Module)

    if is_torch:
        model.eval()
        model.to(device)
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            if hasattr(model, 'predict'):
                # DG/DA models (DGModel, DAModel subclasses) expose predict() not forward()
                logits = model.predict(X_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]  # some models return (class_logits, domain_logits)
            else:
                logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    else:
        probs = model.predict_proba(X_test)
        preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    try:
        if probs.shape[1] == 2:
            auroc = roc_auc_score(y_test, probs[:, 1])
        else:
            auroc = roc_auc_score(y_test, probs, multi_class='ovr')
    except Exception:
        auroc = 0.5

    return {"Accuracy": acc, "F1": f1, "AUROC": auroc}
