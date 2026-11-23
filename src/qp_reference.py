"""CVXOPT anisotropic-TV QP reference on tiny (16x16) problems (Section 4.6)."""
import time

import cvxopt
import numpy as np
import scipy.sparse as sp

from src import mri_ops as mo
from src.recon import objective, psi0

RIDGE = 1e-10
LADDER = [None, 1e-8, 1e-9, 1e-10]    # None = CVXOPT defaults (abstol 1e-7, reltol 1e-6, feastol 1e-7)


def dense_H(M):
    """Dense A^H A built from unit vectors."""
    N = M.shape[0]
    H = np.zeros((N * N, N * N))
    for j in range(N * N):
        e = np.zeros(N * N)
        e[j] = 1.0
        H[:, j] = mo.AHA(e.reshape(N, N), M).ravel()
    return H


def real_stacked_A(M):
    """A_r = [Re(A); Im(A)] restricted to sampled rows (2m x n), via the Kronecker DFT."""
    N = M.shape[0]
    F1 = np.fft.fft(np.eye(N), norm="ortho")
    F2 = np.kron(F1, F1)
    sel = np.flatnonzero(M.ravel() > 0)
    S = F2[sel]
    return np.vstack([S.real, S.imag]), sel


def real_stacked_y(y, sel):
    v = y.ravel()[sel]
    return np.concatenate([v.real, v.imag])


def _spm(S):
    c = sp.coo_matrix(S)
    return cvxopt.spmatrix(c.data.astype(float), c.row.tolist(), c.col.tolist(), c.shape)


def j0(x, y, M, lam):
    r = mo.A(x, M) - y
    return 0.5 * float(np.sum(np.abs(r) ** 2)) + lam * psi0(x)


def tv_qp(M, y, lam, ladder=None):
    """min 0.5||A_r x - y_r||^2 + lam 1^T t  s.t.  -t <= Dx <= t. Returns x (N x N) and a diagnostics dict."""
    ladder = LADDER if ladder is None else ladder
    N = M.shape[0]
    n, p = N * N, mo.n_diffs(N)
    Ar, sel = real_stacked_A(M)
    yr = real_stacked_y(y, sel)
    Dm = mo.D_sparse(N)
    I = sp.identity(p)
    P = np.zeros((n + p, n + p))
    P[:n, :n] = Ar.T @ Ar
    P += RIDGE * np.eye(n + p)
    q = np.concatenate([-Ar.T @ yr, lam * np.ones(p)])
    G = sp.vstack([sp.hstack([Dm, -I]), sp.hstack([-Dm, -I])]).tocsr()
    Pc, qc, Gc, hc = cvxopt.matrix(P), cvxopt.matrix(q), _spm(G), cvxopt.matrix(np.zeros(2 * p))
    best, rungs = None, []
    cvxopt.solvers.options["show_progress"] = False
    for tol in ladder:
        opts = {} if tol is None else dict(abstol=tol, reltol=tol, feastol=tol)
        cvxopt.solvers.options.update({"abstol": 1e-7, "reltol": 1e-6, "feastol": 1e-7, "maxiters": 100})
        cvxopt.solvers.options.update(opts)
        t0 = time.perf_counter()
        try:
            sol = cvxopt.solvers.qp(Pc, qc, Gc, hc)
        except (ValueError, ArithmeticError) as e:
            rungs.append(dict(tol=tol, status=f"error: {e}", iterations=None, time=time.perf_counter() - t0))
            break
        rungs.append(dict(tol=tol, status=sol["status"], iterations=int(sol["iterations"]), time=time.perf_counter() - t0))
        if sol["status"] != "optimal":
            break
        best = (tol, sol)
    if best is None:
        raise RuntimeError(f"CVXOPT never reached 'optimal': {rungs}")
    tol, sol = best
    z = np.array(sol["x"]).ravel()
    x = z[:n].reshape(N, N)
    info = dict(status=sol["status"], tolerance=tol, iterations=int(sol["iterations"]), gap=float(sol["gap"]),
                relative_gap=None if sol["relative gap"] is None else float(sol["relative gap"]),
                rungs=rungs, time=float(sum(r["time"] for r in rungs)), J0=j0(x, y, M, lam),
                ridge_bound=0.5 * RIDGE * float(z @ z))   # <= 0.5*ridge*||z*||^2 deviation of the objective
    return x, info
