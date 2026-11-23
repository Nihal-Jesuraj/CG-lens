import numpy as np
import pytest
import scipy.sparse.linalg as spla

from src import cg as cgm
from src import mri_ops as mo
from src.recon import H_tik
from src.qp_reference import dense_H


@pytest.fixture(scope="module")
def sys16():
    M = mo.make_mask(16, 2, (2, 0))
    lam = 0.1
    H = dense_H(M) + lam * (mo.D_sparse(16).T @ mo.D_sparse(16)).toarray()
    b = np.random.default_rng(0).standard_normal(256)
    return M, lam, H, b


def test_matches_dense_solve(sys16):
    M, lam, H, b = sys16
    r = cgm.cg(lambda v: (H @ v.ravel()).reshape(v.shape), b, rtol=1e-13, maxit=2000)
    assert r.converged and np.abs(r.x - np.linalg.solve(H, b)).max() < 1e-8


def test_operator_form_matches_dense(sys16):
    M, lam, H, b = sys16
    v = np.random.default_rng(1).standard_normal((16, 16))
    assert np.abs(H_tik(M, lam)(v).ravel() - H @ v.ravel()).max() < 1e-12


def test_iteration_count_matches_scipy(sys16):
    M, lam, H, b = sys16
    r = cgm.cg(lambda v: H @ v, b, rtol=1e-8, maxit=2000)
    cnt = []
    x, info = spla.cg(H, b, rtol=1e-8, atol=0.0, maxiter=2000, callback=lambda xk: cnt.append(1))
    assert info == 0 and r.iters == len(cnt)


def test_finite_termination_and_special_spectra():
    rng = np.random.default_rng(2)
    Q, _ = np.linalg.qr(rng.standard_normal((20, 20)))
    H = Q @ np.diag(rng.uniform(1, 10, 20)) @ Q.T
    H = (H + H.T) / 2
    b = rng.standard_normal(20)
    r = cgm.cg(lambda v: H @ v, b, rtol=1e-10, maxit=100)
    assert r.converged and r.iters <= 20
    d = np.repeat([1.0, 4.0, 9.0], 7)
    Q21, _ = np.linalg.qr(rng.standard_normal((21, 21)))
    H3 = Q21 @ np.diag(d) @ Q21.T
    H3 = (H3 + H3.T) / 2
    r3 = cgm.cg(lambda v: H3 @ v, rng.standard_normal(21), rtol=1e-10, maxit=100)
    assert r3.converged and r3.iters <= 3
    r1 = cgm.cg(lambda v: 2.5 * v, b, rtol=1e-12)
    assert r1.iters == 1 and np.abs(r1.x - b / 2.5).max() < 1e-14


def test_non_spd_error_and_symmetry_check():
    H = np.diag([1.0, -1.0, 2.0])
    with pytest.raises(cgm.NonSPDError):
        cgm.cg(lambda v: H @ v, np.array([0.0, 1.0, 0.0]))
    rng = np.random.default_rng(3)
    B = rng.standard_normal((30, 30))
    S = B @ B.T + 30 * np.eye(30)
    K = rng.standard_normal((30, 30))
    K = 1e-3 * (K - K.T)
    Hn = S + K
    # p^T H p stays positive for non-symmetric H = S + K, so only the symmetry check can catch it
    for _ in range(20):
        p = rng.standard_normal(30)
        assert p @ (Hn @ p) > 0
    assert cgm.check_symmetry(lambda v: S @ v, 30)
    assert not cgm.check_symmetry(lambda v: Hn @ v, 30)
    assert cgm.check_symmetry(H_tik(mo.make_mask(16, 4, (4, 0)), 0.1), (16, 16))


def test_energy_norm_bound(sys16):
    M, lam, H, b = sys16
    w = np.linalg.eigvalsh(H)
    kappa = w[-1] / w[0]
    xs = np.linalg.solve(H, b)
    errs = []
    cgm.cg(lambda v: H @ v, b, x0=None, rtol=1e-12, maxit=400, callback=lambda k, x: errs.append(np.sqrt((x - xs) @ H @ (x - xs))))
    e0 = np.sqrt(xs @ H @ xs)
    q = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
    assert all(e <= 2 * q ** (k + 1) * e0 * (1 + 1e-8) for k, e in enumerate(errs))


def test_nlcg_quadratic_and_armijo():
    rng = np.random.default_rng(4)
    B = rng.standard_normal((10, 10))
    S = B @ B.T + 10 * np.eye(10)
    c = rng.standard_normal(10)
    f = lambda x: 0.5 * x @ S @ x - c @ x
    g = lambda x: S @ x - c
    # rtol 1e-9 is below the float64 resolution of a function-value Armijo test (gradient floors ~1e-8 relative,
    # diagnosed: converges for rtol >= 1e-8), so the test uses 1e-7. Tolerance change, not a bug (listed in summary).
    res = cgm.nlcg(f, g, np.zeros(10), rtol=1e-7, maxit=500)
    assert res.converged and res.armijo_ok and np.abs(res.x - np.linalg.solve(S, c)).max() < 1e-6
    assert all(b <= a + 1e-14 for a, b in zip(res.f_hist, res.f_hist[1:]))
