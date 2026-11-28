"""Stage-1 tests: feature maps (T1, T2) and the training-fold angle scaler.

The T1/T2 references here are deliberately *self-contained*: they import only
NumPy, SciPy's ``expm`` and the module under test (``feature_maps``).  In
particular nothing is imported from ``src.quantum.simulator`` or
``src.quantum.kernels``, so a bug in those modules cannot mask a bug in a
feature map.
"""
import numpy as np
import pytest
from scipy.linalg import expm

from src.quantum import feature_maps as fm


# --------------------------------------------------------------------------
# Independent dense reference (NumPy/SciPy only)
# --------------------------------------------------------------------------
def _dense_hadamard(q):
    H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    out = np.array([[1.0]])
    for _ in range(q):
        out = np.kron(out, H)
    return out


def _dense_z(q, i):
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    return np.kron(np.kron(np.eye(2 ** (q - 1 - i)), Z), np.eye(2 ** i))


def _dense_zz_state(theta, L):
    """Independent dense-matrix ZZ state ``(D H)**L |0>``, used only by T2."""
    q = len(theta)
    Hq = _dense_hadamard(q)
    Zs = [_dense_z(q, i) for i in range(q)]
    D = np.zeros((2 ** q, 2 ** q), dtype=np.complex128)
    for i in range(q):
        D += theta[i] * Zs[i]
    for i in range(q):
        for j in range(i + 1, q):
            D += (np.pi - theta[i]) * (np.pi - theta[j]) * (Zs[i] @ Zs[j])
    U = expm(1j * D) @ Hq
    psi = np.zeros(2 ** q, dtype=np.complex128)
    psi[0] = 1.0
    for _ in range(L):
        psi = U @ psi
    return psi


def _gram_local(Phi):
    """``K_ij = |<phi_i|phi_j>|**2`` without importing ``src.quantum.kernels``."""
    return np.abs(Phi @ Phi.conj().T) ** 2


def _fidelity_local(a, b):
    """Row-wise ``|<a|b>|**2`` without importing ``src.quantum.simulator``."""
    return np.abs(np.sum(np.conj(a) * b, axis=-1)) ** 2


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_t1_product_map_matches_closed_form():
    rng = np.random.default_rng(2026)
    theta = rng.uniform(0.0, np.pi, size=(9, 5))
    K = _gram_local(fm.product_map(theta))
    diff = theta[:, None, :] - theta[None, :, :]
    K_closed = np.prod(np.cos(diff / 2.0) ** 2, axis=-1)
    err = np.abs(K - K_closed).max()
    print(f"T1 product-map max abs error: {err:.3e}")
    assert err <= 1e-12


@pytest.mark.parametrize("q", [3, 4, 5, 6])
@pytest.mark.parametrize("L", [1, 2])
def test_t2_zz_map_matches_dense_kronecker(q, L):
    rng = np.random.default_rng(100 + q + L)
    theta = rng.uniform(0.0, np.pi, size=(6, q))
    Phi_sim = fm.zz_map(theta, L=L)
    Phi_dense = np.stack([_dense_zz_state(t, L) for t in theta])
    err = np.abs(_gram_local(Phi_sim) - _gram_local(Phi_dense)).max()
    fid = _fidelity_local(Phi_sim, Phi_dense)
    print(f"T2 q={q} L={L} kernel max abs error: {err:.3e}; min fidelity {fid.min():.15f}")
    assert err <= 1e-10
    assert np.all(fid >= 1.0 - 1e-10)


def test_angle_scaler_uses_training_extremes_without_clipping():
    rng = np.random.default_rng(5)
    Z_train = rng.normal(size=(40, 4))
    Z_test = rng.normal(size=(7, 4))
    scaler = fm.AngleScaler(s=0.5).fit(Z_train)
    theta = scaler.transform(Z_test)
    u = (Z_test - Z_train.min(axis=0)) / (Z_train.max(axis=0) - Z_train.min(axis=0))
    assert np.allclose(theta, 0.5 * np.pi * u)
    outside = scaler.transform((Z_train.max(axis=0) + 1.0)[None, :])
    assert np.all(outside > 0.5 * np.pi)          # no clipping


def test_constant_component_maps_to_midpoint():
    Z = np.ones((10, 3))
    Z[:, 1] = np.linspace(-2.0, 2.0, 10)
    scaler = fm.AngleScaler(s=1.0).fit(Z)
    theta = scaler.transform(Z)
    assert np.allclose(theta[:, 0], 0.5 * np.pi)
    assert np.allclose(theta[:, 2], 0.5 * np.pi)


def test_feature_map_factory():
    theta = np.zeros((2, 3))
    assert np.allclose(fm.feature_map("product")(theta), fm.product_map(theta))
    assert np.allclose(fm.feature_map("zz", L=1)(theta), fm.zz_map(theta, L=1))
    with pytest.raises(ValueError):
        fm.feature_map("nope")
