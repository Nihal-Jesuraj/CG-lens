"""Kernel utilities for the E9 quantum-kernel SVM.

All kernels here are Gram matrices of statevectors: ``K_ij = |<phi_i|phi_j>|**2``.
They are positive semidefinite by construction, have unit diagonal, and lie in
``[0, 1]``.  The finite-shot and depolarizing models below break exact PSD in
controlled ways, so the diagnostics report what actually happens instead of
assuming it.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "gram",
    "psd_report",
    "kernel_target_alignment",
    "shot_noise",
    "psd_project",
    "depolarize",
]


def _sym(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("kernel matrix must be square")
    return 0.5 * (K + K.T)


def gram(Phi_a: np.ndarray, Phi_b: np.ndarray | None = None) -> np.ndarray:
    """``K_ij = |<phi_i|psi_j>|**2`` for rows of two statevector matrices."""
    A = np.atleast_2d(np.asarray(Phi_a, dtype=np.complex128))
    B = A if Phi_b is None else np.atleast_2d(np.asarray(Phi_b, dtype=np.complex128))
    if A.shape[1] != B.shape[1]:
        raise ValueError("statevectors must have the same dimension")
    return np.abs(A @ B.conj().T) ** 2


def psd_report(K: np.ndarray, eps: float = 1e-10) -> dict:
    """Spectrum summary of a kernel matrix.

    Returns the minimum and maximum eigenvalue, the numerical rank at
    ``eps * max(1, lambda_max)``, the stable rank
    ``||K||_F**2 / ||K||_2**2`` (reported as ``effective_rank``), and the
    condition number of ``K + eps I`` (``inf`` if that matrix is still not
    positive definite).
    """
    K = _sym(K)
    w = np.linalg.eigvalsh(K)
    lmax, lmin = float(w[-1]), float(w[0])
    absmax = max(abs(lmax), abs(lmin))
    fro2 = float(np.sum(K * K))
    stable_rank = fro2 / (absmax ** 2) if absmax > 0.0 else 0.0
    tol = float(eps) * max(1.0, abs(lmax))
    rank = int(np.sum(w > tol))
    denom = lmin + float(eps)
    cond = float((lmax + float(eps)) / denom) if denom > 0.0 else math.inf
    return {
        "min_eig": lmin,
        "max_eig": lmax,
        "rank": rank,
        "effective_rank": float(stable_rank),
        "cond_eps": cond,
        "eps": float(eps),
    }


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """``<K, y y^T>_F / (||K||_F ||y y^T||_F)`` for labels in {-1, +1}."""
    K = _sym(K)
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) != K.shape[0]:
        raise ValueError("y must have one entry per kernel row")
    T = np.outer(y, y)
    denom = float(np.linalg.norm(K) * np.linalg.norm(T))
    if denom == 0.0:
        return 0.0
    return float(np.sum(K * T) / denom)


def shot_noise(K: np.ndarray, S: int | None, rng: np.random.Generator) -> np.ndarray:
    """Binomial shot-noise estimate of ``K`` with ``S`` shots per entry.

    Only the strict upper triangle is sampled; the matrix is mirrored and the
    diagonal is set to exactly 1.  ``S=None`` or ``S=inf`` returns a copy.
    """
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("kernel matrix must be square")
    if S is None or (isinstance(S, float) and math.isinf(S)):
        # Exact path: return the caller's matrix untouched (no symmetrisation),
        # so an infinite-shot estimate is bit-identical to the exact kernel.
        return K.copy()
    K = _sym(K)
    S = int(S)
    if S < 1:
        raise ValueError("S must be a positive integer")
    n = K.shape[0]
    iu = np.triu_indices(n, k=1)
    p = np.clip(K[iu], 0.0, 1.0)
    draws = rng.binomial(S, p) / float(S)
    Khat = np.eye(n, dtype=np.float64)
    Khat[iu] = draws
    Khat[(iu[1], iu[0])] = draws
    return Khat


def psd_project(K: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone by clipping eigenvalues.

    The diagonal is not renormalised; callers that need unit diagonal should
    decide explicitly how to restore it.
    """
    K = _sym(K)
    w, V = np.linalg.eigh(K)
    w = np.clip(w, 0.0, None)
    return _sym((V * w) @ V.T)


def depolarize(K: np.ndarray, p: float, q: int) -> np.ndarray:
    """Global depolarizing model ``(1 - p) K + p / 2**q``."""
    if not 0.0 <= float(p) <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    if int(q) < 1:
        raise ValueError("q must be >= 1")
    K = _sym(K)
    return (1.0 - float(p)) * K + float(p) / float(2 ** int(q))
