"""XGBoost classifier wrapper with built-in early stopping."""

import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin

from ._helpers import attach_training_metadata


class XGBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_jobs=-1, patience=20, **kwargs):
        self.patience = patience
        # Force CPU by default to avoid CUDA errors in some environments.
        # Users can still override by passing tree_method explicitly.
        kwargs.setdefault("tree_method", "hist")
        # XGBoost 2.0+ requires early_stopping_rounds in constructor
        # use_label_encoder is deprecated/removed in 3.0+
        self.model = xgb.XGBClassifier(eval_metric='auc', n_jobs=n_jobs, early_stopping_rounds=patience, **kwargs)

    def fit(self, X, y, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        # XGBoost scikit-learn API handles early stopping if early_stopping_rounds is passed to constructor
        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        best_iteration = getattr(self.model, "best_iteration", None)
        n_estimators = getattr(self.model, "n_estimators", None)
        attach_training_metadata(
            self,
            optimizer="xgboost",
            best_epoch=(best_iteration + 1) if best_iteration is not None else None,
            epochs_ran=(best_iteration + 1) if best_iteration is not None else n_estimators,
            max_epochs=n_estimators,
            early_stopped=bool(best_iteration is not None and n_estimators is not None and best_iteration + 1 < n_estimators),
            model_selection_metric="val_auroc",
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

