import numpy as np
from sklearn.decomposition import PCA as SKPCA

from src.pca import PCA


def _data(n=60, d=40, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)) @ np.diag(np.linspace(5, 0.1, d)) + 3.0


def test_matches_sklearn():
    X = _data()
    k = 8
    ours = PCA(n_components=k).fit(X)
    sk = SKPCA(n_components=k, svd_solver="full").fit(X)
    assert np.abs(ours.singular_values_[:k] - sk.singular_values_).max() / sk.singular_values_[0] < 1e-12
    cos = np.linalg.svd(ours.components_ @ sk.components_.T, compute_uv=False)
    assert cos.min() >= 1 - 1e-8
    Xt = _data(seed=1)
    err_o = np.linalg.norm(Xt - ours.inverse_transform(ours.transform(Xt)))
    err_s = np.linalg.norm(Xt - sk.inverse_transform(sk.transform(Xt)))
    assert abs(err_o - err_s) < 1e-9 * err_s
    assert np.abs(ours.explained_variance_[:k] - sk.explained_variance_).max() < 1e-10


def test_threshold_k_and_train_only_fit():
    X = _data()
    p = PCA(var=0.95).fit(X)
    cum = np.cumsum(p.explained_variance_ratio_)
    assert cum[p.k_ - 1] >= 0.95 and (p.k_ == 1 or cum[p.k_ - 2] < 0.95)
    assert p.k_for_variance(0.5) < p.k_for_variance(0.95)
    c1 = p.components_.copy()
    _ = p.transform(_data(seed=9) * 100)                     # transforming other data must not alter the fit
    assert np.array_equal(c1, PCA(var=0.95).fit(X).components_)
    assert np.abs(p.mean_ - X.mean(0)).max() < 1e-14         # centres with the training mean only


def test_standardize_zero_variance_guard():
    X = _data()
    X[:, 3] = 7.0
    p = PCA(n_components=5, standardize=True).fit(X)
    assert np.all(np.isfinite(p.transform(X))) and p.scale_[3] == 1.0
