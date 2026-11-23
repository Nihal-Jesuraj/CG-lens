"""Hand-written linear CG, separate symmetry check, nonlinear CG (P1) with Armijo line search."""
from dataclasses import dataclass, field

import numpy as np

import config


class NonSPDError(RuntimeError):
    """Raised when p^T H p <= 0 (H is not positive definite along p)."""


@dataclass
class CGResult:
    x: np.ndarray
    iters: int
    matvecs: int            # applications of H (each = one A + one A^H); forming A^H y is not counted
    resnorms: list = field(default_factory=list)
    converged: bool = False


def cg(matvec, b, x0=None, rtol=None, maxit=None, callback=None):
    """Solve H x = b for SPD H; stop when ||r|| <= rtol*||b||. callback(k, x_k) after each step."""
    rtol = config.CG_RTOL if rtol is None else rtol
    maxit = config.CG_MAXIT if maxit is None else maxit
    b = np.asarray(b, dtype=np.float64)
    bn = float(np.linalg.norm(b))
    if bn == 0.0:
        return CGResult(np.zeros_like(b), 0, 0, [0.0], True)
    mv = 0
    if x0 is None or not np.any(x0):
        x, r = np.zeros_like(b), b.copy()
    else:
        x = np.array(x0, dtype=np.float64)
        r = b - matvec(x)
        mv += 1
    p = r.copy()
    rs = float(np.vdot(r, r))
    res = [rs ** 0.5]
    if res[0] <= rtol * bn:
        return CGResult(x, 0, mv, res, True)
    converged = False
    k = 0
    for k in range(1, maxit + 1):
        Hp = matvec(p)
        mv += 1
        pHp = float(np.vdot(p, Hp))
        if pHp <= 0.0:
            raise NonSPDError(f"p^T H p = {pHp:.3e} <= 0 at iteration {k}")
        alpha = rs / pHp
        x += alpha * p
        r -= alpha * Hp
        rs_new = float(np.vdot(r, r))
        res.append(rs_new ** 0.5)
        if callback is not None:
            callback(k, x)
        if res[-1] <= rtol * bn:
            converged = True
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return CGResult(x, k, mv, res, converged)


def check_symmetry(matvec, shape, trials=5, seed=0, tol=1e-10):
    """True if |u^T(Hv) - v^T(Hu)| <= tol*(|u^T(Hv)| + |v^T(Hu)|) for random real u, v. Independent of SPD."""
    rng = np.random.default_rng(seed)
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    for _ in range(trials):
        u, v = rng.standard_normal(shape), rng.standard_normal(shape)
        a, b = float(np.vdot(u, matvec(v))), float(np.vdot(v, matvec(u)))
        if abs(a - b) > tol * (abs(a) + abs(b)):
            return False
    return True


def assert_symmetric(matvec, shape, **kw):
    if not check_symmetry(matvec, shape, **kw):
        raise ValueError("operator failed the symmetry check")


@dataclass
class NLCGResult:
    x: np.ndarray
    iters: int
    nfun: int
    ngrad: int
    f_hist: list
    converged: bool
    armijo_ok: bool = True

    @property
    def matvecs(self):
        return self.ngrad + 0.5 * self.nfun   # gradient = A + A^H, function = A only


def nlcg(f, grad, x0, c1=None, shrink=None, restart=None, rtol=None, maxit=None, max_backtracks=60, callback=None):
    """Polak-Ribiere+ NLCG, Armijo backtracking (c1, shrink), restart every `restart` iterations."""
    P = config.NLCG
    c1, shrink = P["c1"] if c1 is None else c1, P["shrink"] if shrink is None else shrink
    restart, rtol, maxit = P["restart"] if restart is None else restart, P["rtol"] if rtol is None else rtol, P["maxit"] if maxit is None else maxit
    x = np.array(x0, dtype=np.float64)
    fx, g = f(x), grad(x)
    nfun, ngrad = 1, 1
    g0 = float(np.linalg.norm(g))
    d, hist, ok, conv = -g, [fx], True, False
    if g0 == 0:
        return NLCGResult(x, 0, nfun, ngrad, hist, True)
    k = 0
    for k in range(1, maxit + 1):
        slope = float(np.vdot(g, d))
        if slope >= 0:
            d, slope = -g, -float(np.vdot(g, g))
        t = 1.0
        for _ in range(max_backtracks):
            fn = f(x + t * d)
            nfun += 1
            if fn <= fx + c1 * t * slope:
                break
            t *= shrink
        else:
            break                                   # line search failed: stop, report not converged
        ok &= bool(fn <= fx + c1 * t * slope)
        x = x + t * d
        gn = grad(x)
        ngrad += 1
        fx = fn
        hist.append(fx)
        if callback is not None:
            callback(k, x)
        if float(np.linalg.norm(gn)) <= rtol * g0:
            conv = True
            break
        beta = max(0.0, float(np.vdot(gn, gn - g)) / float(np.vdot(g, g)))
        d = -gn + (0.0 if k % restart == 0 else beta) * d
        g = gn
    return NLCGResult(x, k, nfun, ngrad, hist, conv, ok)
