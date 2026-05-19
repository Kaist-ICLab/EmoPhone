"""LightGBM classifier wrapper with built-in early stopping."""

import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin

from ._helpers import attach_training_metadata


class LightGBMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_jobs=-1, patience=20, **kwargs):
        self.patience = patience
        self.model = lgb.LGBMClassifier(n_jobs=n_jobs, **kwargs)

    def fit(self, X, y, X_val=None, y_val=None):
        X_train = X.values if hasattr(X, "values") else X
        y_train = y.values if hasattr(y, "values") else y
        X_valid = X_val.values if (X_val is not None and hasattr(X_val, "values")) else X_val
        y_valid = y_val.values if (y_val is not None and hasattr(y_val, "values")) else y_val

        eval_set = [(X_valid, y_valid)] if X_valid is not None else None
        callbacks = [lgb.early_stopping(self.patience, verbose=True)] if eval_set else None
        self.model.fit(X_train, y_train, eval_set=eval_set, eval_metric="auc", callbacks=callbacks)
        best_iteration = getattr(self.model, "best_iteration_", None)
        n_estimators = getattr(self.model, "n_estimators", None)
        attach_training_metadata(
            self,
            optimizer="lightgbm",
            best_epoch=best_iteration,
            epochs_ran=best_iteration or n_estimators,
            max_epochs=n_estimators,
            early_stopped=bool(
                best_iteration is not None
                and n_estimators is not None
                and best_iteration < n_estimators
            ),
            model_selection_metric="val_auroc",
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
