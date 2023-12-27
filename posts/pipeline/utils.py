import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.linear_model import Ridge

class Estimator2TransformerMixin(TransformerMixin):
    def __init__(self, estimator_target):
        self.estimator_target = estimator_target

    def _set_estimator_target_indx(self, X):
        if isinstance(self.estimator_target, str) or isinstance(
            self.estimator_target, int
        ):
            self.estimator_target = [self.estimator_target]

        if not hasattr(self, "estimator_target_indx") or not hasattr(
            self, "est_vars_indx"
        ):
            if isinstance(X, pd.DataFrame):
                self.estimator_target_indx = [
                    i
                    for i, name in enumerate(X.columns)
                    if name in self.estimator_target
                ]
                self.est_vars_indx = [
                    i
                    for i, name in enumerate(X.columns)
                    if name not in self.estimator_target
                ]
            else:
                self.estimator_target_indx = [
                    i for i in np.arange(X.shape[1]) if i in self.estimator_target
                ]
                self.est_vars_indx = [
                    i for i in range(X.shape[1]) if i not in self.estimator_target_indx
                ]

        assert len(self.estimator_target) == len(self.estimator_target_indx), (
            "Couldn't find estimator target in `X`."
            " If `X` is a `pandas.DataFrame`, `estimator_target` should"
            " be a string. If `X` is a `numpy.array`, "
            " `estimator_target` should be an int."
        )

    def _check_X_target(self, X):

        if not hasattr(self, "estimator_target_indx") or not hasattr(
            self, "est_vars_indx"
        ):
            self._set_estimator_target_indx(X)

        if isinstance(X, pd.DataFrame):
            y = X.values[:, self.estimator_target_indx]
            X_est = X.values[:, self.est_vars_indx]
        else:
            y = X[:, self.estimator_target_indx]
            X_est = X[:, self.est_vars_indx]
        return X_est, y


class RidgeRegTransformer(Ridge, Estimator2TransformerMixin):
    """
    Ridge regression estimator class with different target from overall pipeline

    Added scaler support internal to class.
    """    
    def __init__(
        self,
        estimator_target="target",
        fit_intercept=True,
        alpha=1.0,
        random_state=None,
        scaler=None,
    ):
        self.scaler = scaler
        self.estimator_target = estimator_target
        super().__init__(
            fit_intercept=fit_intercept,
            alpha=alpha, random_state=random_state
        )

    def fit(self, X, y=None):
        X_reg, y = self._check_X_target(X)
        if self.scaler:
            X_reg = self.scaler.fit_transform(X_reg)
        sup_fit = super().fit(X_reg, y)
        return sup_fit

    def predict(self, X):
        X_reg, _ = self._check_X_target(X)
        if self.scaler:
            X_reg = self.scaler.transform(X_reg)
        return super().predict(X_reg)

    def transform(self, X):
        preds = self.predict(X)
        return preds

    def fit_transform(self, X, y=None):
        X_reg, y = self._check_X_target(X)
        if self.scaler:
            X_reg = self.scaler.fit_transform(X_reg)
        super().fit(X_reg, y)

        preds = self.predict(X)
        return preds


def map_idx(data, col_names):
    return list(np.where(data.columns.isin(col_names))[0])
