"""M0 zero-filled, M1 Tikhonov-H1 (CG), M2 TV lagged-diffusivity IRLS (CG), M3 TV nonlinear CG (P1)."""
import numpy as np

import config
from src import cg as cgm
from src import mri_ops as mo


def zero_filled(y, M):
    return mo.AH(y, M)


def H_tik(M, lam):
    N = M.shape[0]
    return lambda x: mo.AHA(x, M) + lam * mo.DT(mo.D(x), N)


def tikhonov_cg(y, M, lam, rtol=None, maxit=None, x0=None, callback=None):
    """M1: solve (A^H A + lam D^T D) x = A^H y by CG, started at the zero-filled image."""
    b = mo.AH(y, M)
    H = H_tik(M, lam)
    cgm.assert_symmetric(H, b.shape)
    res = cgm.cg(H, b, x0=b if x0 is None else x0, rtol=rtol, maxit=maxit, callback=callback)
    return res.x, dict(method="M1", lam=lam, iters=res.iters, matvecs=res.matvecs, converged=res.converged, resnorms=res.resnorms)


# ------------------------------------------------------------------ TV objective
def psi(x, mu):
    d = mo.D(x)
    return float(np.sum(np.sqrt(d * d + mu * mu)))


def psi0(x):
    return float(np.sum(np.abs(mo.D(x))))


def objective(x, y, M, lam, mu):
    r = mo.A(x, M) - y
    return 0.5 * float(np.sum(np.abs(r) ** 2)) + lam * psi(x, mu)


def gradient(x, y, M, lam, mu):
    d = mo.D(x)
    w = 1.0 / np.sqrt(d * d + mu * mu)
    return mo.AH(mo.A(x, M) - y, M) + lam * mo.DT(w * d, x.shape[0])


def H_irls(M, lam, weights):
    """Lagged-diffusivity normal operator for one IRLS weight update."""
    N = M.shape[0]
    return lambda v: mo.AHA(v, M) + lam * mo.DT(weights * mo.D(v), N)


def tv_irls_cg(y, M, lam, mu, inner_rtol=None, inner_maxit=None, outer_tol=None, outer_maxit=None, x0=None, store_iterates=False):
    """M2: lagged diffusivity. (A^H A + lam D^T W_k D) x_{k+1} = A^H y, CG warm-started at x_k, mu fixed."""
    inner_rtol = config.CG_RTOL if inner_rtol is None else inner_rtol
    inner_maxit = config.CG_MAXIT if inner_maxit is None else inner_maxit
    outer_tol = config.IRLS_OUTER_TOL if outer_tol is None else outer_tol
    outer_maxit = config.IRLS_OUTER_MAX if outer_maxit is None else outer_maxit
    N = M.shape[0]
    b = mo.AH(y, M)
    x = b.copy() if x0 is None else np.array(x0, dtype=np.float64)
    J = [objective(x, y, M, lam, mu)]
    inner_iters, hit, matvecs, viol, conv = [], 0, 0, 0, False
    iterates = [x.copy()] if store_iterates else None
    for k in range(outer_maxit):
        d = mo.D(x)
        w = 1.0 / np.sqrt(d * d + mu * mu)
        H = H_irls(M, lam, w)
        if k == 0:
            cgm.assert_symmetric(H, (N, N))
        r = cgm.cg(H, b, x0=x, rtol=inner_rtol, maxit=inner_maxit)
        inner_iters.append(r.iters)
        matvecs += r.matvecs
        hit += int(not r.converged)
        rel = np.linalg.norm(r.x - x) / max(np.linalg.norm(x), 1e-300)
        x = r.x
        J.append(objective(x, y, M, lam, mu))
        viol += int(J[-1] - J[-2] > config.J_VIOLATION_TOL * max(1.0, abs(J[-2])))
        if store_iterates:
            iterates.append(x.copy())
        if rel < outer_tol:
            conv = True
            break
    info = dict(method="M2", lam=lam, mu=mu, outer_iters=len(inner_iters), inner_iters=inner_iters, inner_hit_maxit=hit, matvecs=matvecs,
                J_hist=J, n_violations=viol, converged=conv, iters=int(sum(inner_iters)))
    if store_iterates:
        info["iterates"] = iterates
    return x, info


def tv_nlcg(y, M, lam, mu, x0=None, **kw):
    """M3 (P1): nonlinear CG on the smoothed TV objective."""
    f = lambda x: objective(x, y, M, lam, mu)
    g = lambda x: gradient(x, y, M, lam, mu)
    res = cgm.nlcg(f, g, mo.AH(y, M) if x0 is None else x0, **kw)
    return res.x, dict(method="M3", lam=lam, mu=mu, iters=res.iters, matvecs=res.matvecs, converged=res.converged, f_hist=res.f_hist, armijo_ok=res.armijo_ok)
