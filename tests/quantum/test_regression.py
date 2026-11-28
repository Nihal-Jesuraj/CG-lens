"""T6: seed-2026 regression for the pre-existing SVM paths.

Stage 2 only adds a ``kernel="precomputed"`` branch.  The explicit-feature
path must stay bit-for-bit the same, so its dual objectives, biases and
support-vector counts are frozen here as golden values captured from the
code *before* the branch was added.  E1-E5 and E8 are covered by running the
existing suite unchanged (``python -m pytest -q``).
"""
import numpy as np
import pandas as pd
import pytest

import config
from src import svm
from src.pca import PCA

# Captured on the pre-change code with config.GLOBAL_SEED == 2026.
GOLDEN = {
    "gamma": 0.05588640361117562,
    "qp_dual": -4.84070864502536,
    "smo_dual": -4.840704331918999,
    "qp_bias": -0.0924009689304537,
    "smo_bias": -0.09260010258094989,
    "qp_nsv": 25,
    "smo_nsv": 24,
}


def _problem():
    rng = np.random.default_rng(config.GLOBAL_SEED)
    npos, n = 40, 120
    X = np.vstack([rng.standard_normal((npos, 8)) + 1.2,
                   rng.standard_normal((n - npos, 8)) - 1.2])
    y = np.r_[np.ones(npos), -np.ones(n - npos)]
    Z = PCA(n_components=6).fit(X).transform(X)
    return Z, y


def test_t6_explicit_rbf_path_is_unchanged():
    Z, y = _problem()
    gamma = svm.gamma_from_g(1.0, Z)
    assert abs(gamma - GOLDEN["gamma"]) <= 1e-15 * GOLDEN["gamma"]
    qp = svm.SVM_QP(C=1.0, kernel="rbf", gamma=gamma).fit(Z, y)
    smo = svm.SVM_SMO(C=1.0, kernel="rbf", gamma=gamma).fit(Z, y)
    print(f"T6 qp dual {qp.dual_:.12f} bias {qp.b_:.12f} nsv {qp.sv_mask_.sum()}; "
          f"smo dual {smo.dual_:.12f} bias {smo.b_:.12f} nsv {smo.sv_mask_.sum()}")
    assert abs(qp.dual_ - GOLDEN["qp_dual"]) <= 1e-12 * abs(GOLDEN["qp_dual"])
    assert abs(smo.dual_ - GOLDEN["smo_dual"]) <= 1e-12 * abs(GOLDEN["smo_dual"])
    assert abs(qp.b_ - GOLDEN["qp_bias"]) <= 1e-12
    assert abs(smo.b_ - GOLDEN["smo_bias"]) <= 1e-12
    assert int(qp.sv_mask_.sum()) == GOLDEN["qp_nsv"]
    assert int(smo.sv_mask_.sum()) == GOLDEN["smo_nsv"]
    # fit_kernel(K, y, X) keeps the same meaning as before (X -> X_sv_).
    K = svm.rbf_kernel(Z, Z, gamma)
    qp2 = svm.SVM_QP(C=1.0, kernel="rbf", gamma=gamma).fit_kernel(K, y, Z)
    assert np.array_equal(qp2.alpha_, qp.alpha_) and qp2.X_sv_.shape == qp.X_sv_.shape
    assert np.array_equal(qp2.coef_, qp.coef_)


def test_t6_solver_agreement_matches_the_declared_thresholds():
    Z, y = _problem()
    gamma = svm.gamma_from_g(1.0, Z)
    K = svm.rbf_kernel(Z, Z, gamma)
    qp = svm.SVM_QP(C=1.0, kernel="rbf", gamma=gamma).fit(Z, y)
    smo = svm.SVM_SMO(C=1.0, kernel="rbf", gamma=gamma).fit(Z, y)
    a, b, sk = svm.svc_reference(Z, y, 1.0, "rbf", gamma)
    d_svc = svm.dual_objective(a, y, K)
    rel_qp = abs(qp.dual_ - d_svc) / abs(d_svc)
    rel_smo = abs(smo.dual_ - d_svc) / abs(d_svc)
    print(f"T6 rel dual vs libsvm: qp {rel_qp:.3e}, smo {rel_smo:.3e}")
    assert rel_qp < 1e-3 and rel_smo < 1e-2
    assert np.mean(qp.predict(Z) == sk.predict(Z)) >= 0.999
    assert np.mean(smo.predict(Z) == sk.predict(Z)) >= 0.999


AGREEMENT_CSV = config.RESULTS_DIR / "e4_solver_agreement.csv"


@pytest.mark.skipif(not AGREEMENT_CSV.exists(), reason="E4 outputs not present")
def test_t6_e4_solver_agreement_artifact():
    """The frozen E4 artifact must still satisfy the report's thresholds."""
    ag = pd.read_csv(AGREEMENT_CSV)
    print(f"T6 E4 artifact: {len(ag)} rows; median rel dual qp "
          f"{ag.rel_dual_qp_svc.median():.2e}, smo {ag.rel_dual_smo_svc.median():.2e}")
    assert ag.rel_dual_qp_svc.median() < 1e-6
    assert ag.rel_dual_smo_svc.median() < 1e-6
    # Report Table: mean agreement 0.9999 / 1.0000 against the declared 0.98 floor.
    assert ag.agree_qp_svc.mean() >= 0.999 and ag.agree_qp_svc.min() >= 0.98
    assert ag.agree_smo_svc.mean() >= 0.999 and ag.agree_smo_svc.min() >= 0.98
