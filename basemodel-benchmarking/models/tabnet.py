"""TabNet classifier wrapper."""

import logging

from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)

from ._helpers import (
    FIXED_BATCH_SIZE,
    _drop_optuna_helper_params,
    _filter_supported_kwargs,
    attach_training_metadata,
)


class TabNetWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, batch_size=FIXED_BATCH_SIZE, epochs=50, patience=20, **kwargs):
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        params = _drop_optuna_helper_params(self.kwargs)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.batch_size = int(self.batch_size or FIXED_BATCH_SIZE)

        # Remove batch_size from kwargs if it accidentally got in there
        if "batch_size" in params:
            self.batch_size = params.pop("batch_size")
        if "epochs" in params:
            self.epochs = params.pop("epochs")
        if "patience" in params:
            self.patience = params.pop("patience")

        if "verbose" in params:
            verbose = params.pop("verbose")
        else:
            verbose = 1

        if "n_d" in params and int(params["n_d"]) > 32:
            logger.info(f"[INFO] TabNet: capped n_d from {params['n_d']} to 32.")
            params["n_d"] = 32
        if "n_a" in params and int(params["n_a"]) > 32:
            logger.info(f"[INFO] TabNet: capped n_a from {params['n_a']} to 32.")
            params["n_a"] = 32
        if "n_steps" in params and int(params["n_steps"]) > 6:
            logger.info(f"[INFO] TabNet: capped n_steps from {params['n_steps']} to 6.")
            params["n_steps"] = 6

        tabnet_params = _filter_supported_kwargs(TabNetClassifier.__init__, params)
        self.model = TabNetClassifier(verbose=verbose, **tabnet_params)
        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            eval_metric=["auc"],
            patience=self.patience,
            max_epochs=self.epochs,
            batch_size=self.batch_size,
            num_workers=0,
        )
        best_epoch = getattr(self.model, "best_epoch", None)
        attach_training_metadata(
            self,
            optimizer="Adam",
            best_epoch=(best_epoch + 1) if isinstance(best_epoch, int) else best_epoch,
            epochs_ran=(best_epoch + 1) if isinstance(best_epoch, int) else self.epochs,
            max_epochs=self.epochs,
            batch_size=self.batch_size,
            early_stopped=bool(
                best_epoch is not None
                and isinstance(best_epoch, int)
                and best_epoch + 1 < self.epochs
            ),
            model_selection_metric="val_auroc",
            architecture_params=tabnet_params,
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# --- Wrappers for Advanced Tabular DL ---
