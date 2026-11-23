"""PCA via economy SVD (hand-written). Centres only unless standardize=True (zero-variance guarded)."""
import numpy as np

import config


class PCA:
    def __init__(self, var=None, n_components=None, standardize=None):
        self.var = config.PCA_VAR if var is None else var
        self.n_components = n_components
        self.standardize = config.PCA_STANDARDIZE if standardize is None else standardize

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        self.scale_ = np.ones(X.shape[1])
        if self.standardize:
            sd = Xc.std(axis=0, ddof=1)
            self.scale_ = np.where(sd > 1e-12, sd, 1.0)
            Xc = Xc / self.scale_
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.singular_values_ = s
        self.explained_variance_ = s ** 2 / max(len(X) - 1, 1)
        tot = self.explained_variance_.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / tot if tot > 0 else np.zeros_like(s)
        if self.n_components is not None:
            k = min(int(self.n_components), len(s))
        else:
            k = int(np.searchsorted(np.cumsum(self.explained_variance_ratio_), self.var - 1e-12) + 1)
            k = min(k, len(s))
        self.k_ = k
        self.components_ = Vt[:k]
        return self

    def transform(self, X):
        return ((np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return (Z @ self.components_) * self.scale_ + self.mean_

    def k_for_variance(self, var):
        return int(np.searchsorted(np.cumsum(self.explained_variance_ratio_), var - 1e-12) + 1)
