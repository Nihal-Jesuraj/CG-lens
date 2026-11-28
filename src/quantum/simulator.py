"""Exact statevector simulator for the CG Lens quantum extensions.

This is a classical simulator only: every state is a dense ``complex128``
vector and every gate is applied exactly.  No quantum speedup is claimed.

Conventions
-----------
* ``q`` qubits, statevector length ``N = 2**q``.
* Basis index ``k`` stores qubit ``i`` in bit ``(k >> i) & 1``; qubit 0 is the
  least-significant bit (little-endian).
* ``H`` is the Hadamard gate ``[[1, 1], [1, -1]] / sqrt(2)``.  ``H`` on every
  qubit is applied with a fast Walsh-Hadamard transform followed by a single
  ``1 / sqrt(N)`` rescaling.
* ``Z = diag(1, -1)``; the eigenvalue of ``Z_i`` on basis state ``k`` is
  ``1 - 2 * bit_i(k)``.

These conventions are fixed here and reused by every feature map and test.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = [
    "zero_state",
    "basis_bits",
    "pauli_z_signs",
    "walsh_hadamard",
    "apply_phase",
    "zz_phase",
    "zz_circuit",
    "ry_product_state",
    "state_fidelity",
]


def zero_state(q: int, batch: int = 1) -> np.ndarray:
    """Return ``batch`` copies of ``|0...0>`` as rows of a complex matrix."""
    if q < 1:
        raise ValueError("q must be >= 1")
    psi = np.zeros((int(batch), 1 << int(q)), dtype=np.complex128)
    psi[:, 0] = 1.0
    return psi


@lru_cache(maxsize=None)
def basis_bits(q: int) -> np.ndarray:
    """``(2**q, q)`` int8 matrix; entry ``[k, i]`` is bit ``i`` of ``k``."""
    if q < 1:
        raise ValueError("q must be >= 1")
    k = np.arange(1 << int(q), dtype=np.int64)[:, None]
    i = np.arange(int(q), dtype=np.int64)[None, :]
    return ((k >> i) & 1).astype(np.int8)


def pauli_z_signs(q: int) -> np.ndarray:
    """``(2**q, q)`` float matrix of ``Z_i`` eigenvalues (+1 for bit 0)."""
    return 1.0 - 2.0 * basis_bits(q).astype(np.float64)


def walsh_hadamard(v: np.ndarray) -> np.ndarray:
    """Apply ``H`` to every qubit of a statevector (last axis).

    ``v`` may have any number of leading axes; the last axis must have length
    ``2**q``.  The transform is exact up to floating-point round-off.
    """
    arr = np.asarray(v, dtype=np.complex128)
    if arr.ndim == 0:
        raise ValueError("walsh_hadamard expects an array")
    n = arr.shape[-1]
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError("last axis must have power-of-two length >= 2")
    orig = arr.shape
    flat = arr.reshape(-1, n).copy()
    h = 1
    while h < n:
        flat = flat.reshape(-1, n // (2 * h), 2 * h)
        a = flat[..., :h].copy()
        b = flat[..., h:2 * h].copy()
        flat[..., :h] = a + b
        flat[..., h:2 * h] = a - b
        flat = flat.reshape(-1, n)
        h *= 2
    return (flat / np.sqrt(n)).reshape(orig)


def apply_phase(psi: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Multiply a statevector by a diagonal phase vector."""
    return np.asarray(psi, dtype=np.complex128) * np.asarray(phases, dtype=np.complex128)


def zz_phase(theta: np.ndarray) -> np.ndarray:
    """Diagonal phase vector of the IQP/ZZ feature map.

    For one sample ``theta`` of length ``q``,

    ``D(theta) = exp(i [sum_i theta_i Z_i
                        + sum_{i<j} (pi - theta_i)(pi - theta_j) Z_i Z_j])``.

    ``theta`` may be ``(q,)`` or ``(n, q)``; the result is ``(2**q,)`` or
    ``(n, 2**q)`` respectively.  This is the Havlicek et al. (2019)
    convention, applied consistently in every test.
    """
    th = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    if th.ndim != 2:
        raise ValueError("theta must be (q,) or (n, q)")
    n, q = th.shape
    signs = pauli_z_signs(q)                       # (N, q)
    lin = th @ signs.T                             # (n, N)
    w = np.pi - th                                 # (n, q)
    pair_w = np.triu(w[:, :, None] * w[:, None, :], k=1)
    pair_signs = signs[:, :, None] * signs[:, None, :]   # (N, q, q)
    quad = np.einsum("kij,nij->nk", pair_signs, pair_w, optimize=True)  # (n, N)
    out = np.exp(1j * (lin + quad))
    return out[0] if np.ndim(theta) == 1 else out


def zz_circuit(theta: np.ndarray, L: int = 2) -> np.ndarray:
    """Statevectors of ``U(theta) = [D(theta) H^{otimes q}]**L |0...0>``."""
    if int(L) < 1:
        raise ValueError("L must be >= 1")
    th = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    q = th.shape[1]
    psi = zero_state(q, batch=len(th))
    phases = zz_phase(th)                          # D(theta) is the same in every layer
    for _ in range(int(L)):
        # U = D H: the Hadamard layer acts first, then the diagonal phase.
        psi = walsh_hadamard(psi)
        psi = psi * phases
    return psi


def ry_product_state(theta: np.ndarray) -> np.ndarray:
    """``otimes_i RY(theta_i) |0>`` with qubit 0 as the least-significant bit."""
    th = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    n, q = th.shape
    a, b = np.cos(th / 2.0), np.sin(th / 2.0)
    state = np.ones((n, 1), dtype=np.complex128)
    # Build from the most-significant qubit down so that the last appended
    # factor is qubit 0, matching the little-endian basis convention.
    for i in range(q - 1, -1, -1):
        state = np.stack([state * a[:, i, None], state * b[:, i, None]], axis=-1)
        state = state.reshape(n, -1)
    return state


def state_fidelity(phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """``|<phi|psi>|**2`` for matching rows of two statevector matrices."""
    phi = np.atleast_2d(np.asarray(phi, dtype=np.complex128))
    psi = np.atleast_2d(np.asarray(psi, dtype=np.complex128))
    if phi.shape != psi.shape:
        raise ValueError("state matrices must have the same shape")
    overlap = np.sum(np.conj(phi) * psi, axis=-1)
    return np.abs(overlap) ** 2
