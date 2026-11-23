"""Class-balanced kernel SVMs: dual QP via CVXOPT and deterministic max-violating-pair SMO (Section 4.8)."""
import time

import cvxopt
import numpy as np
import scipy.sparse as sp

import config

_UP_LOW_EPS = config.SV_THRESH


def rbf_kernel(X, Y, gamma):
    sq = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2.0 * X @ Y.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


def linear_kernel(X, Y):
    return X @ Y.T


def gamma_from_g(g, Xp):
    """gamma = g / (n_features * Var(X_train_pca))."""
    return g / (Xp.shape[1] * Xp.var())


def balanced_C(y, C):
    """C_i = C * n / (2 * n_class(i))."""
    n = len(y)
    npos, nneg = int((y == 1).sum()), int((y == -1).sum())
    if npos == 0 or nneg == 0:
        raise ValueError("training set must contain both classes")
    return np.where(y == 1, C * n / (2.0 * npos), C * n / (2.0 * nneg))


def dual_objective(alpha, y, K):
    """D(alpha) = 1/2 alpha^T Q alpha - sum(alpha), Q = (y y^T) o K."""
    v = alpha * y
    return float(0.5 * v @ K @ v - alpha.sum())


def _sets(alpha, y, Cvec):
    lo, hi = _UP_LOW_EPS * Cvec, (1.0 - _UP_LOW_EPS) * Cvec
    up = ((y == 1) & (alpha < hi)) | ((y == -1) & (alpha > lo))
    low = ((y == -1) & (alpha < hi)) | ((y == 1) & (alpha > lo))
    return up, low


def kkt_gap(alpha, y, K, Cvec):
    """max_{I_up}(-y g) - min_{I_low}(-y g), g = Q alpha - 1 (same for every solver)."""
    s = -y * (y * (K @ (y * alpha)) - 1.0)
    up, low = _sets(alpha, y, Cvec)
    if not up.any() or not low.any():
        return 0.0
    return float(s[up].max() - s[low].min())


def _bias(alpha, y, K, Cvec):
    f0 = K @ (alpha * y)
    free = (alpha > _UP_LOW_EPS * Cvec) & (alpha < (1 - _UP_LOW_EPS) * Cvec)
    if free.any():
        return float(np.mean(y[free] - f0[free]))
    s = -y * (y * f0 - 1.0)
    up, low = _sets(alpha, y, Cvec)
    if up.any() and low.any():
        return float(0.5 * (s[up].max() + s[low].min()))
    return 0.0


class _SVM:
    solver = "base"

    def __init__(self, C=1.0, kernel="rbf", gamma=1.0, class_balanced=True):
        self.C, self.kernel, self.gamma, self.class_balanced = C, kernel, gamma, class_balanced

    def _K(self, X, Y):
        return rbf_kernel(X, Y, self.gamma) if self.kernel == "rbf" else linear_kernel(X, Y)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        t0 = time.perf_counter()
        K = self._K(X, X)
        self.kernel_time = time.perf_counter() - t0
        return self.fit_kernel(K, y, X)

    def fit_kernel(self, K, y, X):
        y = np.asarray(y, dtype=np.float64)
        Cvec = balanced_C(y, self.C) if self.class_balanced else np.full(len(y), float(self.C))
        t0 = time.perf_counter()
        alpha = self._solve(K, y, Cvec)
        self.solve_time = time.perf_counter() - t0
        self.alpha_, self.y_, self.Cvec_ = alpha, y, Cvec
        self.b_ = _bias(alpha, y, K, Cvec)
        self.sv_mask_ = alpha > _UP_LOW_EPS * Cvec
        self.X_sv_, self.coef_ = np.asarray(X)[self.sv_mask_], (alpha * y)[self.sv_mask_]
        self.dual_ = dual_objective(alpha, y, K)
        self.kkt_gap_ = kkt_gap(alpha, y, K, Cvec)
        return self

    def decision_function(self, X):
        return self._K(np.asarray(X, dtype=np.float64), self.X_sv_) @ self.coef_ + self.b_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


class SVM_QP(_SVM):
    """Solver A: cvxopt.solvers.qp on the dual (dense P, sparse box G, equality y^T alpha = 0)."""
    solver = "qp"

    def _solve(self, K, y, Cvec):
        n = len(y)
        P = cvxopt.matrix((y[:, None] * y[None, :]) * K + config.SVM_RIDGE * np.eye(n))
        q = cvxopt.matrix(-np.ones(n))
        rows = np.concatenate([np.arange(n), n + np.arange(n)])
        G = cvxopt.spmatrix(np.concatenate([-np.ones(n), np.ones(n)]), rows.tolist(), np.concatenate([np.arange(n), np.arange(n)]).tolist(), (2 * n, n))
        h = cvxopt.matrix(np.concatenate([np.zeros(n), Cvec]))
        Aeq, b0 = cvxopt.matrix(y.reshape(1, -1).astype(float), tc="d"), cvxopt.matrix(0.0)
        cvxopt.solvers.options["show_progress"] = False
        sol = cvxopt.solvers.qp(P, q, G, h, Aeq, b0)
        self.status_, self.iterations_ = sol["status"], int(sol["iterations"])
        return np.clip(np.array(sol["x"]).ravel(), 0.0, Cvec)


class SVM_SMO(_SVM):
    """Solver B: deterministic maximal-violating-pair SMO, libsvm stopping gap, zero-curvature endpoint rule."""
    solver = "smo"

    def __init__(self, C=1.0, kernel="rbf", gamma=1.0, class_balanced=True, tol=None, max_iter=None, track=False):
        super().__init__(C, kernel, gamma, class_balanced)
        self.tol = config.SMO_TOL if tol is None else tol
        self.max_iter, self.track = max_iter, track

    def _solve(self, K, y, Cvec):
        n = len(y)
        max_iter = self.max_iter or (100 * n + 10000)
        alpha, g, kd = np.zeros(n), -np.ones(n), np.diag(K).copy()
        self.dual_hist_ = [0.0] if self.track else None
        gap, it = np.inf, 0
        for it in range(max_iter):
            s = -y * g
            up = ((y == 1) & (alpha < Cvec)) | ((y == -1) & (alpha > 0))
            low = ((y == -1) & (alpha < Cvec)) | ((y == 1) & (alpha > 0))
            if not up.any() or not low.any():
                gap = 0.0
                break
            i = int(np.argmax(np.where(up, s, -np.inf)))
            j = int(np.argmin(np.where(low, s, np.inf)))
            gap = float(s[i] - s[j])
            if gap <= self.tol:
                break
            eta = kd[i] + kd[j] - 2.0 * K[i, j]
            room_i = Cvec[i] - alpha[i] if y[i] == 1 else alpha[i]
            room_j = alpha[j] if y[j] == 1 else Cvec[j] - alpha[j]
            dmax = min(room_i, room_j)
            if eta > 1e-12 * (kd[i] + kd[j]):
                delta = min(gap / eta, dmax)
            else:                                   # dual is linear along the pair: compare the two endpoints
                f_end = -gap * dmax + 0.5 * eta * dmax * dmax
                delta = dmax if f_end < 0.0 else 0.0
            if delta <= 0.0:
                break
            alpha[i] += y[i] * delta
            alpha[j] -= y[j] * delta
            if delta >= room_i:
                alpha[i] = Cvec[i] if y[i] == 1 else 0.0
            if delta >= room_j:
                alpha[j] = 0.0 if y[j] == 1 else Cvec[j]
            g += delta * y * (K[:, i] - K[:, j])
            if self.track:
                self.dual_hist_.append(dual_objective(alpha, y, K))
        else:
            gap = kkt_gap(alpha, y, K, Cvec)
        self.iterations_, self.final_gap_, self.hit_max_iter_ = it, gap, it == max_iter - 1 and gap > self.tol
        return alpha


def svc_reference(X, y, C, kernel, gamma):
    """scikit-learn SVC (libsvm) with the same class-balanced C; returns (alpha vector, intercept, model)."""
    from sklearn.svm import SVC
    m = SVC(C=C, kernel=kernel, gamma=gamma if kernel == "rbf" else "scale", class_weight="balanced", tol=config.SMO_TOL, cache_size=1000).fit(X, y)
    a = np.zeros(len(y))
    a[m.support_] = np.abs(m.dual_coef_.ravel())
    return a, float(m.intercept_[0]), m


def jaccard(a, b):
    a, b = set(np.flatnonzero(a)), set(np.flatnonzero(b))
    return len(a & b) / max(len(a | b), 1)
