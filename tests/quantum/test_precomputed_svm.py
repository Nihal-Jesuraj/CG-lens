"""Stage-2 tests: ``kernel="precomputed"`` in ``SVM_QP`` / ``SVM_SMO`` (T4, T5).

T4 checks the precomputed path against libsvm on the same Gram matrix.
T5 checks that the precomputed path is numerically the same model as the
explicit-feature path, and exercises it with the quantum kernel of stage 1.
"""
import numpy as np
import pytest

import config
from src import svm
from src.quantum import feature_maps as fm
from src.quantum import kernels as kn


def _blobs(n=120, seed=0, sep=1.2, imb=0.3):
    rng = np.random.default_rng(seed)
    npos = int(n * imb)
    X = np.vstack([rng.standard_normal((npos, 3)) + sep, rng.standard_normal((n - npos, 3)) - sep])
    y = np.r_[np.ones(npos), -np.ones(n - npos)]
    return X, y


@pytest.fixture(scope="module")
def problem():
    X, y = _blobs(seed=config.GLOBAL_SEED)
    gamma = svm.gamma_from_g(1.0, X)
    K = svm.rbf_kernel(X, X, gamma)
    return X, y, K, gamma


# ---------------------------------------------------------------- T4
@pytest.mark.parametrize("cls,tol", [(svm.SVM_QP, 1e-3), (svm.SVM_SMO, 1e-2)])
def test_t4_precomputed_matches_libsvm(problem, cls, tol):
    X, y, K, gamma = problem
    m = cls(C=1.0, kernel="precomputed").fit(K, y)
    a, b, sk = svm.svc_reference(K, y, 1.0, "precomputed", None)
    d_svc = svm.dual_objective(a, y, K)
    rel = abs(m.dual_ - d_svc) / abs(d_svc)
    agree = float(np.mean(m.predict(K) == sk.predict(K)))
    print(f"T4 {cls.__name__}: rel dual vs libsvm {rel:.3e}; prediction agreement {agree:.4f}; "
          f"kkt {m.kkt_gap_:.2e}")
    assert rel < tol
    assert agree >= 0.97
    assert abs(np.dot(m.alpha_, y)) < 1e-6
    assert np.all(m.alpha_ >= -1e-12) and np.all(m.alpha_ <= m.Cvec_ + 1e-12)
    if cls is svm.SVM_SMO:
        assert m.final_gap_ <= config.SMO_TOL and not m.hit_max_iter_


def test_t4_fit_and_fit_kernel_agree(problem):
    X, y, K, gamma = problem
    for cls in (svm.SVM_QP, svm.SVM_SMO):
        a = cls(C=1.0, kernel="precomputed").fit(K, y)
        b = cls(C=1.0, kernel="precomputed").fit_kernel(K, y)
        assert np.array_equal(a.alpha_, b.alpha_)
        assert a.b_ == b.b_ and a.dual_ == b.dual_


def test_t4_precomputed_decision_is_support_vector_expansion(problem):
    X, y, K, gamma = problem
    m = svm.SVM_QP(C=1.0, kernel="precomputed").fit(K, y)
    Kt = K[:17]
    dec = m.decision_function(Kt)
    manual = Kt[:, m.sv_idx_] @ m.coef_ + m.b_
    assert dec.shape == (17,)
    assert np.allclose(dec, manual, atol=1e-12)
    assert np.array_equal(m.predict(Kt), np.where(manual >= 0, 1, -1))


# ---------------------------------------------------------------- T5
@pytest.mark.parametrize("cls", [svm.SVM_QP, svm.SVM_SMO])
def test_t5_precomputed_rbf_is_the_same_model(problem, cls):
    X, y, K, gamma = problem
    Xt, _ = _blobs(seed=7)
    m_feat = cls(C=1.0, kernel="rbf", gamma=gamma).fit(X, y)
    m_pre = cls(C=1.0, kernel="precomputed").fit(K, y)
    assert np.allclose(m_feat.alpha_, m_pre.alpha_, atol=1e-12, rtol=0.0)
    assert abs(m_feat.b_ - m_pre.b_) <= 1e-12
    dec_feat = m_feat.decision_function(Xt)
    dec_pre = m_pre.decision_function(svm.rbf_kernel(Xt, X, gamma))
    print(f"T5 {cls.__name__}: max decision diff {np.abs(dec_feat - dec_pre).max():.3e}")
    assert np.allclose(dec_feat, dec_pre, atol=1e-10)
    assert np.array_equal(m_feat.predict(Xt), m_pre.predict(svm.rbf_kernel(Xt, X, gamma)))


def test_t5_quantum_kernel_through_precomputed_path():
    rng = np.random.default_rng(config.GLOBAL_SEED)
    q = 5
    theta_tr = rng.uniform(0.0, np.pi, size=(70, q))
    theta_te = rng.uniform(0.0, np.pi, size=(25, q))
    y = np.r_[np.ones(35), -np.ones(35)]
    K = kn.gram(fm.zz_map(theta_tr, L=2))
    Kt = kn.gram(fm.zz_map(theta_te, L=2), fm.zz_map(theta_tr, L=2))
    m = svm.SVM_QP(C=1.0, kernel="precomputed").fit(K, y)
    dec = m.decision_function(Kt)
    manual = Kt[:, m.sv_idx_] @ m.coef_ + m.b_
    print(f"T5 ZZ kernel: {len(m.sv_idx_)} support vectors; max diff {np.abs(dec - manual).max():.3e}")
    assert np.allclose(dec, manual, atol=1e-12)
    assert np.array_equal(m.predict(Kt), np.where(dec >= 0, 1, -1))
    assert np.allclose(np.diag(K), 1.0)
    assert float(np.linalg.eigvalsh(0.5 * (K + K.T))[0]) >= -1e-10


def test_t5_precomputed_rejects_feature_matrices(problem):
    X, y, K, gamma = problem
    with pytest.raises(ValueError):
        svm.SVM_QP(kernel="precomputed").fit(X, y)          # 120 x 3, not square
    m = svm.SVM_QP(kernel="precomputed").fit(K, y)
    with pytest.raises(ValueError):
        m.decision_function(np.zeros((5, 3)))              # wrong number of columns
    with pytest.raises(RuntimeError):
        m._K(X, X)                                         # kernel functions are bypassed
    with pytest.raises(ValueError):
        svm.SVM_QP(kernel="polynomial")


def test_t5_linear_precomputed_is_the_same_model():
    X, y = _blobs(seed=11)
    Xt, _ = _blobs(seed=12)
    K, Kt = svm.linear_kernel(X, X), svm.linear_kernel(Xt, X)
    m_feat = svm.SVM_QP(C=1.0, kernel="linear").fit(X, y)
    m_pre = svm.SVM_QP(C=1.0, kernel="precomputed").fit(K, y)
    assert np.allclose(m_feat.decision_function(Xt), m_pre.decision_function(Kt), atol=1e-10)
