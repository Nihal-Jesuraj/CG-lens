"""T7: the depolarizing channel is a uniform rescaling of C.

Proposition (E9 prereg, P1).  For ``K_p = (1 - p) K + (p / 2**q) 1 1^T`` and
any ``C > 0``, the SVM trained on ``K_p`` with box ``C`` and the SVM trained on
``K`` with box ``(1 - p) C`` have the same decision function.

Why.  On the equality constraint ``y^T alpha = 0`` the rank-one term vanishes,
so ``alpha^T [(y y^T) o K_p] alpha = (1 - p) alpha^T [(y y^T) o K] alpha``.
Substituting ``alpha = beta / (1 - p)`` maps the box ``[0, C]`` onto
``[0, (1 - p) C]`` and multiplies the dual objective by ``1 / (1 - p)``, so the
two duals have the same maximiser up to this scaling; the bias is unchanged
because the KKT condition for a free ``alpha_i`` carries the same ``(1 - p)``
factor.  Both the training Gram and the test-train cross Gram must be
depolarized for the decision function to agree.

``SVM_QP`` adds ``config.SVM_RIDGE * I`` to the dual quadratic form, which
breaks the identity by ``O(ridge)``.  The test therefore sets the ridge to zero
to check the mathematical statement, and uses the exact class-balanced
``C_i`` scaling (``C_i -> (1 - p) C_i``).
"""
import cvxopt
import numpy as np
import pytest

import config
from src import svm
from src.quantum import feature_maps as fm
from src.quantum import kernels as kn


@pytest.fixture
def tight_qp(monkeypatch):
    """CVXOPT's default 1e-7 tolerance is too loose for a 1e-10 assertion."""
    for key, value in dict(abstol=1e-12, reltol=1e-12, feastol=1e-12, maxiters=1000).items():
        monkeypatch.setitem(cvxopt.solvers.options, key, value)


def _problem(seed=0, q=5, n_tr=24, n_te=12):
    """Unbalanced two-class problem with a genuine quantum kernel."""
    rng = np.random.default_rng(seed)
    theta_tr = rng.uniform(0.0, np.pi, size=(n_tr, q))
    theta_te = rng.uniform(0.0, np.pi, size=(n_te, q))
    n_pos = n_tr // 3
    y = np.r_[np.ones(n_pos), -np.ones(n_tr - n_pos)]
    Phi_tr = fm.zz_map(theta_tr, L=2)
    Phi_te = fm.zz_map(theta_te, L=2)
    return kn.gram(Phi_tr), kn.gram(Phi_te, Phi_tr), y


def _depolarize_cross(Kt, p, q):
    """Same channel on a rectangular test-train Gram: (1-p) Kt + p / 2**q."""
    return (1.0 - p) * np.asarray(Kt, dtype=np.float64) + float(p) / float(2 ** int(q))


@pytest.mark.parametrize("cls", [svm.SVM_QP, svm.SVM_SMO])
def test_t7_depolarizing_is_a_uniform_rescaling_of_C(cls, monkeypatch, tight_qp):
    # The proposition is about the ridge-free dual; the production ridge is 1e-8.
    monkeypatch.setattr(config, "SVM_RIDGE", 0.0)

    K, Kt, y = _problem(seed=71)
    p, q, C = 0.2, 5, 1.0
    Kp = kn.depolarize(K, p, q)
    Ktp = _depolarize_cross(Kt, p, q)

    extra = dict(tol=1e-12, max_iter=200000) if cls is svm.SVM_SMO else {}
    m_p = cls(C=C, kernel="precomputed", **extra).fit(Kp, y)
    m_s = cls(C=(1.0 - p) * C, kernel="precomputed", **extra).fit(K, y)

    dec_p = m_p.decision_function(Ktp)
    dec_s = m_s.decision_function(Kt)
    dec_diff = float(np.abs(dec_p - dec_s).max())
    alpha_diff = float(np.abs(m_p.alpha_ - m_s.alpha_ / (1.0 - p)).max())
    bias_diff = abs(m_p.b_ - m_s.b_)
    print(f"T7 {cls.__name__}: max decision diff {dec_diff:.3e}; "
          f"max alpha diff {alpha_diff:.3e}; bias diff {bias_diff:.3e}")

    assert dec_diff <= 1e-10
    assert alpha_diff <= 1e-10
    assert bias_diff <= 1e-10
    assert abs(np.dot(m_p.alpha_, y)) <= 1e-10


def test_t7_depolarized_training_gram_alone_is_not_enough():
    """The cross Gram must be depolarized too: the two decision functions differ."""
    K, Kt, y = _problem(seed=73)
    p, q, C = 0.2, 5, 1.0
    m_p = svm.SVM_QP(C=C, kernel="precomputed").fit(kn.depolarize(K, p, q), y)
    m_s = svm.SVM_QP(C=(1.0 - p) * C, kernel="precomputed").fit(K, y)
    # Same model, but the wrong cross kernel: Kt instead of Ktp.
    wrong = m_p.decision_function(Kt)
    right = m_s.decision_function(Kt)
    assert float(np.abs(wrong - right).max()) > 1e-3
