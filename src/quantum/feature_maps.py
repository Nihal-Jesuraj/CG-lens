"""Quantum feature maps for the E9 quantum-kernel SVM.

A feature map sends a classical vector ``z`` to a statevector
``|phi(z)> = U(z)|0...0>`` of ``q`` qubits.  The quantum kernel is

    K(z, z') = |<phi(z)|phi(z')>|**2,

which is the Hilbert-Schmidt inner product of the density matrices
``rho(z) = |phi(z)><phi(z)|`` and is therefore positive semidefinite with unit
diagonal and entries in ``[0, 1]``.

Features are PCA components (fitted on the training fold only), min-max
scaled with training-fold extremes, and mapped to angles ``theta = s * pi * u``.
``s`` is the bandwidth analogue of the RBF ``gamma``.  Test values are not
clipped; a value outside the training range simply gives ``u`` outside
``[0, 1]``.
"""
from __future__ import annotations

import numpy as np

from src.quantum import simulator

__all__ = [
    "AngleScaler",
    "features_to_angles",
    "product_map",
    "zz_map",
    "feature_map",
]


def features_to_angles(
    Z: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    s: float = 1.0,
) -> np.ndarray:
    """Min-max scale with training extremes and map to angles.

    ``u = (Z - mins) / (maxs - mins)`` where the training range is non-zero,
    and ``u = 0.5`` for a constant component.  No clipping is applied.
    """
    Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
    mins = np.asarray(mins, dtype=np.float64)
    maxs = np.asarray(maxs, dtype=np.float64)
    span = maxs - mins
    safe = np.where(span > 0.0, span, 1.0)
    u = np.where(span > 0.0, (Z - mins) / safe, 0.5)
    return float(s) * np.pi * u


class AngleScaler:
    """Training-fold min/max scaler followed by ``theta = s * pi * u``.

    Fit on training features only; ``transform`` never clips, so test values
    outside the training range are preserved (and can leave ``[0, s*pi]``).
    """

    def __init__(self, s: float = 1.0):
        self.s = float(s)

    def fit(self, Z: np.ndarray) -> "AngleScaler":
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        self.min_ = Z.min(axis=0)
        self.max_ = Z.max(axis=0)
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        if not hasattr(self, "min_"):
            raise RuntimeError("AngleScaler.transform called before fit")
        return features_to_angles(Z, self.min_, self.max_, self.s)

    def fit_transform(self, Z: np.ndarray) -> np.ndarray:
        return self.fit(Z).transform(Z)


def product_map(theta: np.ndarray) -> np.ndarray:
    """Product (non-entangling) map ``otimes_i RY(theta_i)|0>``.

    Closed form: ``K_ij = prod_k cos**2((theta_ik - theta_jk) / 2)``.
    """
    return simulator.ry_product_state(theta)


def zz_map(theta: np.ndarray, L: int = 2) -> np.ndarray:
    """IQP/ZZ map ``[D(theta) H^{otimes q}]**L |0...0>``.

    ``D`` is the diagonal phase of :func:`simulator.zz_phase`, following the
    Havlicek et al. (2019) convention.  ``L`` is the number of repetitions.
    """
    return simulator.zz_circuit(theta, L=L)


def feature_map(name: str, L: int = 2):
    """Return a callable ``theta -> Phi`` for ``name`` in {product, zz}."""
    key = name.lower()
    if key == "product":
        return product_map
    if key == "zz":
        return lambda theta: zz_map(theta, L=L)
    raise ValueError(f"unknown feature map {name!r}; expected 'product' or 'zz'")
