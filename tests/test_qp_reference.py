import numpy as np
import pytest

import config
from src import mri_ops as mo
from src import qp_reference as qr
from src import recon as rc
from scipy.ndimage import gaussian_filter


def _prob(R=2, snr=25.0, seed=3):
    x = gaussian_filter(np.random.default_rng(seed).random((16, 16)), 1.2)
    x = (x - x.min()) / (x.max() - x.min())
    M, y = mo.simulate(x, R, snr, seed)
    return x, M, y


def test_real_stacked_matches_H():
    for R in (2, 4):
        M = mo.make_mask(16, R, (R, 0))
        Ar, sel = qr.real_stacked_A(M)
        assert Ar.shape == (2 * int(M.sum()), 256)
        assert np.linalg.norm(Ar.T @ Ar - qr.dense_H(M), "fro") < 1e-10


def test_qp_optimal_and_gap_bound():
    x, M, y = _prob()
    lam = 1e-2
    xq, info = qr.tv_qp(M, y, lam)
    assert info["status"] == "optimal" and info["rungs"][0]["status"] == "optimal"
    p = mo.n_diffs(16)
    for mu in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        xi, _ = rc.tv_irls_cg(y, M, lam, mu, inner_rtol=config.VAL_INNER_RTOL, outer_tol=config.VAL_OUTER_TOL, outer_maxit=config.VAL_OUTER_MAX)
        gap = rc.objective(xi, y, M, lam, mu) - qr.j0(xi, y, M, lam)
        assert -1e-12 <= gap <= lam * p * mu + 1e-12
        assert qr.j0(xi, y, M, lam) >= info["J0"] - 1e-6          # QP is the exact minimiser of J0 (up to tolerance)
    assert info["ridge_bound"] < 1e-6


def test_irls_approaches_qp_as_mu_shrinks():
    x, M, y = _prob()
    lam = 1e-2
    xq, info = qr.tv_qp(M, y, lam)
    errs = []
    for mu in (1e-1, 1e-2, 1e-3, 1e-4):
        xi, _ = rc.tv_irls_cg(y, M, lam, mu, inner_rtol=config.VAL_INNER_RTOL, outer_tol=config.VAL_OUTER_TOL, outer_maxit=config.VAL_OUTER_MAX)
        errs.append(np.linalg.norm(xi - xq) / np.linalg.norm(xq))
    print("relative IRLS-QP differences for mu=1e-1..1e-4:", errs)
    assert errs[-1] < errs[0] and errs[-1] < 1e-2
