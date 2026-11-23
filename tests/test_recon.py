import numpy as np
import pytest

import config
from src import cg as cgm
from src import mri_ops as mo
from src import recon as rc
from src.qp_reference import dense_H


def _problem(N=16, R=4, snr=np.inf, seed=0):
    x = np.random.default_rng(seed).random((N, N))
    from scipy.ndimage import gaussian_filter
    x = gaussian_filter(x, 1.5)
    x = (x - x.min()) / (x.max() - x.min())
    M, y = mo.simulate(x, R, snr, seed)
    return x, M, y


def test_zero_filled_is_adjoint():
    x, M, y = _problem()
    assert np.array_equal(rc.zero_filled(y, M), mo.AH(y, M))
    Mfull = np.ones((16, 16))
    assert np.abs(rc.zero_filled(mo.A(x, Mfull), Mfull) - x).max() < 1e-12


def test_tikhonov_solves_normal_equations():
    x, M, y = _problem(R=2)
    lam = 1e-2
    xh, info = rc.tikhonov_cg(y, M, lam, rtol=1e-12, maxit=2000)
    Dm = mo.D_sparse(16)
    H = dense_H(M) + lam * (Dm.T @ Dm).toarray()
    assert info["converged"] and np.abs(H @ xh.ravel() - mo.AH(y, M).ravel()).max() < 1e-9
    assert np.all(np.linalg.eigvalsh(H) > 0)                      # SPD because DC is sampled


def test_m1_operator_is_symmetric():
    M = mo.make_mask(16, 2, (2, 0))
    lam = 0.1
    cgm.assert_symmetric(rc.H_tik(M, lam), (16, 16))


def test_irls_weighted_operator_is_symmetric():
    M = mo.make_mask(16, 2, (2, 0))
    lam, mu = 0.1, 1e-2
    x = np.random.default_rng(18).standard_normal((16, 16))
    weights = 1.0 / np.sqrt(mo.D(x) ** 2 + mu ** 2)
    cgm.assert_symmetric(rc.H_irls(M, lam, weights), (16, 16))


def test_tikhonov_needs_dc():
    M = mo.make_mask(16, 4, (4, 0)).copy()
    M[0, 0] = 0
    H = dense_H(M) + 1e-2 * (mo.D_sparse(16).T @ mo.D_sparse(16)).toarray()
    assert np.linalg.eigvalsh(H).min() < 1e-12                    # constants fall in the joint kernel


def test_gradient_check():
    x, M, y = _problem(snr=25.0)
    rng = np.random.default_rng(5)
    lam, mu = 1e-2, 1e-2
    x0 = rng.random((16, 16))
    g = rc.gradient(x0, y, M, lam, mu)
    for _ in range(5):
        v = rng.standard_normal((16, 16))
        h = 1e-6
        fd = (rc.objective(x0 + h * v, y, M, lam, mu) - rc.objective(x0 - h * v, y, M, lam, mu)) / (2 * h)
        assert abs(fd - np.sum(g * v)) / abs(fd) < 1e-5


def test_gap_bound_psi():
    rng = np.random.default_rng(6)
    p = mo.n_diffs(16)
    for mu in (1e-1, 1e-3):
        for _ in range(5):
            x = rng.random((16, 16))
            gap = rc.psi(x, mu) - rc.psi0(x)
            assert -1e-12 <= gap <= p * mu + 1e-12


def test_majorization_and_monotone_descent():
    x, M, y = _problem(snr=25.0)
    lam, mu = 1e-2, 1e-3
    rng = np.random.default_rng(7)
    xk = rng.random((16, 16))
    wk = 1 / np.sqrt(mo.D(xk) ** 2 + mu ** 2)

    def Q(z):
        return 0.5 * np.sum(np.abs(mo.A(z, M) - y) ** 2) + 0.5 * lam * np.sum(wk * mo.D(z) ** 2) + lam * np.sum(0.5 * mu ** 2 * wk + 0.5 / wk)
    assert abs(Q(xk) - rc.objective(xk, y, M, lam, mu)) < 1e-9
    for _ in range(10):
        z = xk + 0.3 * rng.standard_normal((16, 16))
        assert Q(z) >= rc.objective(z, y, M, lam, mu) - 1e-9
    xh, info = rc.tv_irls_cg(y, M, lam, mu)
    assert info["n_violations"] == 0
    assert all(b <= a + 1e-12 * max(1, abs(a)) for a, b in zip(info["J_hist"], info["J_hist"][1:]))
    assert info["J_hist"][-1] < info["J_hist"][0]
    assert len(info["inner_iters"]) == info["outer_iters"] and info["matvecs"] >= info["iters"]


def test_irls_converges_to_stationary_point():
    x, M, y = _problem(snr=25.0)
    lam, mu = 1e-2, 1e-2
    xh, info = rc.tv_irls_cg(y, M, lam, mu, inner_rtol=config.VAL_INNER_RTOL, outer_tol=config.VAL_OUTER_TOL, outer_maxit=config.VAL_OUTER_MAX)
    g = rc.gradient(xh, y, M, lam, mu)
    assert info["converged"] and np.linalg.norm(g) < 1e-6 * max(1, np.linalg.norm(mo.AH(y, M)))
    # warm-started IRLS beats the zero-filled image in objective
    assert rc.objective(xh, y, M, lam, mu) < rc.objective(mo.AH(y, M), y, M, lam, mu)


def test_nlcg_matches_irls_objective():
    x, M, y = _problem(snr=25.0)
    lam, mu = 1e-2, 1e-1
    xi, _ = rc.tv_irls_cg(y, M, lam, mu, inner_rtol=1e-10, outer_tol=1e-10, outer_maxit=500)
    xn, info = rc.tv_nlcg(y, M, lam, mu, rtol=1e-8, maxit=3000)
    assert info["armijo_ok"]
    assert abs(rc.objective(xn, y, M, lam, mu) - rc.objective(xi, y, M, lam, mu)) < 1e-6
