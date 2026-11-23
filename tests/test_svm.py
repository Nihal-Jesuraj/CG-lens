import numpy as np
import pytest

import config
from src import svm


def _blobs(n=120, seed=0, sep=1.2, imb=0.3):
    rng = np.random.default_rng(seed)
    npos = int(n * imb)
    X = np.vstack([rng.standard_normal((npos, 3)) + sep, rng.standard_normal((n - npos, 3)) - sep])
    y = np.r_[np.ones(npos), -np.ones(n - npos)]
    return X, y


@pytest.fixture(scope="module")
def fitted():
    X, y = _blobs()
    out = {}
    for kern in ("rbf", "linear"):
        gamma = svm.gamma_from_g(1.0, X)
        qp = svm.SVM_QP(C=1.0, kernel=kern, gamma=gamma).fit(X, y)
        smo = svm.SVM_SMO(C=1.0, kernel=kern, gamma=gamma).fit(X, y)
        a, b, m = svm.svc_reference(X, y, 1.0, kern, gamma)
        K = qp._K(X, X)
        out[kern] = dict(qp=qp, smo=smo, svc_alpha=a, svc=m, K=K, dual_svc=svm.dual_objective(a, y, K))
    return X, y, out


def test_balanced_C():
    y = np.r_[np.ones(10), -np.ones(30)]
    c = svm.balanced_C(y, 2.0)
    assert np.allclose(c[y == 1], 2.0 * 40 / 20) and np.allclose(c[y == -1], 2.0 * 40 / 60)
    from sklearn.utils.class_weight import compute_class_weight
    w = compute_class_weight("balanced", classes=np.array([-1.0, 1.0]), y=y)
    assert np.allclose([c[y == -1][0], c[y == 1][0]], 2.0 * w)


def test_constraints_gap_and_dual_vs_svc(fitted):
    X, y, out = fitted
    for kern, o in out.items():
        for name in ("qp", "smo"):
            m = o[name]
            assert abs(np.dot(m.alpha_, y)) < 1e-6 and np.all(m.alpha_ >= -1e-12) and np.all(m.alpha_ <= m.Cvec_ + 1e-12)
        assert o["smo"].final_gap_ <= config.SMO_TOL and not o["smo"].hit_max_iter_
        rel_qp = abs(o["qp"].dual_ - o["dual_svc"]) / abs(o["dual_svc"])
        rel_smo = abs(o["smo"].dual_ - o["dual_svc"]) / abs(o["dual_svc"])
        print(f"{kern}: dual svc={o['dual_svc']:.8f} qp={o['qp'].dual_:.8f} (rel {rel_qp:.2e}) smo={o['smo'].dual_:.8f} (rel {rel_smo:.2e})"
              f" kkt qp={o['qp'].kkt_gap_:.2e} smo={o['smo'].kkt_gap_:.2e} svc={svm.kkt_gap(o['svc_alpha'], y, o['K'], o['qp'].Cvec_):.2e}")
        assert rel_qp < 1e-3 and rel_smo < 1e-2


def test_predictions_agree_with_svc(fitted):
    X, y, out = fitted
    Xt, _ = _blobs(seed=5)
    for o in out.values():
        ref = o["svc"].predict(Xt)
        assert np.mean(o["qp"].predict(Xt) == ref) >= 0.97 and np.mean(o["smo"].predict(Xt) == ref) >= 0.97


def test_kernel_psd():
    X, _ = _blobs()
    for K in (svm.rbf_kernel(X, X, 0.3), svm.linear_kernel(X, X)):
        assert np.linalg.eigvalsh((K + K.T) / 2).min() > -1e-9


def test_hard_margin_toy():
    X = np.array([[-1.0], [1.0]])
    y = np.array([-1.0, 1.0])
    for cls in (svm.SVM_QP, svm.SVM_SMO):
        m = cls(C=1e4, kernel="linear", class_balanced=True).fit(X, y)
        assert np.allclose(m.alpha_, 0.5, atol=1e-4) and abs(m.b_) < 1e-4 and abs(m.dual_ + 0.5) < 1e-4
    X2 = np.array([[2.0, 2], [3, 3], [0, 0], [-1, 0.5]])
    y2 = np.array([1.0, 1, -1, -1])
    for cls in (svm.SVM_QP, svm.SVM_SMO):
        m = cls(C=1e4, kernel="linear").fit(X2, y2)
        assert np.all(m.predict(X2) == y2)
        w = (m.coef_[:, None] * m.X_sv_).sum(0)
        assert np.min(y2 * (X2 @ w + m.b_)) > 1 - 1e-2          # active margin constraints are tight at 1


def test_duplicate_rows_zero_curvature():
    rng = np.random.default_rng(3)
    X, y = _blobs(n=40, seed=3)
    X = np.vstack([X, X[:5], X[:3]])                           # identical rows, same labels
    y = np.r_[y, y[:5], y[:3]]
    X = np.vstack([X, X[2:4]])                                 # identical rows, opposite labels
    y = np.r_[y, -y[2:4]]
    gamma = svm.gamma_from_g(1.0, X)
    for kern in ("rbf", "linear"):
        smo = svm.SVM_SMO(C=1.0, kernel=kern, gamma=gamma, track=True).fit(X, y)
        qp = svm.SVM_QP(C=1.0, kernel=kern, gamma=gamma).fit(X, y)
        h = np.array(smo.dual_hist_)
        assert np.all(np.diff(h) <= 1e-12) and h[-1] <= 0         # never increases the dual objective
        assert smo.final_gap_ <= config.SMO_TOL and qp.dual_ <= 1e-9
        a, _, _ = svm.svc_reference(X, y, 1.0, kern, gamma)
        d_svc = svm.dual_objective(a, y, smo._K(X, X))
        assert abs(smo.dual_ - d_svc) < 1e-2 * abs(d_svc) and abs(qp.dual_ - d_svc) < 1e-3 * abs(d_svc)
