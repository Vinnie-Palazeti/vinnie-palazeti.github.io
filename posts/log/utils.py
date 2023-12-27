from scipy.special import expit
from sklearn.base import ClassifierMixin, BaseEstimator



def contrast_and_logistic_score(X, coefs, scale=1.0):
    contrasts = X.dot(coefs) / scale
    return expit(contrasts.astype(float)).astype(float)

class LogisticDifference(BaseEstimator, ClassifierMixin):
    def __init__(self, scale=1.0):
        self.coef_ = np.array([1.0, -1])
        self.scale = scale

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return (X.dot(self.coef_) / self.scale > 0).astype(int)

    def predict_proba(self, X):
        preds = contrast_and_logistic_score(X, self.coef_, self.scale)
        return np.vstack((1.0 - preds, preds)).T
