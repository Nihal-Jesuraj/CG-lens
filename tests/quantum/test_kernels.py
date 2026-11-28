"""Stage-1 tests: kernel utilities and PSD properties (T3)."""
import numpy as np
import pytest

from src.quantum import feature_maps as fm
from src.quantum import kernels as kn


@pytest.mark.parametrize("name,L", [("product", 1), ("zz", 1), ("zz", 2)])
def test_t3_kernel_is_symmetric_unit_diagonal_bounded_and_psd(name, L):
    rng = np.random.default_rng(31 + L)
    theta = rng.uniform(0.0, np.pi, size=(12, 6))
    Phi = fm.feature_map(name, L=L)(theta)
    K = kn.gram(Phi)
    sym_err = np.abs(K - K.T).max()
    diag_err = np.abs(np.diag(K) - 1.0).max()
    min_eig = float(np.linalg.eigvalsh(0.5 * (K + K.T))[0])
    print(f"T3 {name} L={L}: sym {sym_err:.2e} diag {diag_err:.2e} "
          f"min eig {min_eig:.2e} range [{K.min():.6f}, {K.max():.6f}]")
    assert sym_err <= 1e-12
    assert diag_err <= 1e-12
    assert K.min() >= -1e-12 and K.max() <= 1.0 + 1e-12
    assert min_eig >= -1e-10


def test_shot_noise_keeps_unit_diagonal_and_bounds():
    rng = np.random.default_rng(41)
    theta = rng.uniform(0.0, np.pi, size=(10, 5))
    K = kn.gram(fm.zz_map(theta, L=2))
    Khat = kn.shot_noise(K, 1000, rng)
    assert np.abs(Khat - Khat.T).max() == 0.0
    assert np.allclose(np.diag(Khat), 1.0)
    assert Khat.min() >= 0.0 and Khat.max() <= 1.0
    K_inf = kn.shot_noise(K, None, rng)
    assert np.array_equal(K_inf, K)


def test_psd_project_clips_negative_eigenvalues():
    rng = np.random.default_rng(43)
    theta = rng.uniform(0.0, np.pi, size=(8, 4))
    K = kn.gram(fm.zz_map(theta, L=2))
    Khat = kn.shot_noise(K, 100, rng)
    Kp = kn.psd_project(Khat)
    assert np.abs(Kp - Kp.T).max() <= 1e-12
    assert float(np.linalg.eigvalsh(Kp)[0]) >= -1e-10
    assert np.allclose(Kp, kn.psd_project(Khat))


def test_depolarize_formula_and_psd():
    rng = np.random.default_rng(47)
    q = 5
    theta = rng.uniform(0.0, np.pi, size=(8, q))
    K = kn.gram(fm.zz_map(theta, L=2))
    Kd = kn.depolarize(K, 0.05, q)
    assert np.allclose(Kd, 0.95 * K + 0.05 / (2 ** q))
    assert float(np.linalg.eigvalsh(Kd)[0]) >= -1e-10


def test_psd_report_and_alignment():
    rng = np.random.default_rng(53)
    theta = rng.uniform(0.0, np.pi, size=(9, 5))
    K = kn.gram(fm.zz_map(theta, L=2))
    rep = kn.psd_report(K)
    assert set(rep) >= {"min_eig", "max_eig", "rank", "effective_rank", "cond_eps"}
    # Unit diagonal only bounds the entries; the spectrum obeys 0 <= lambda <= n.
    assert rep["min_eig"] >= -1e-10
    assert rep["max_eig"] <= len(K) + 1e-10
    assert rep["rank"] >= 1 and rep["effective_rank"] >= 1.0
    assert rep["cond_eps"] >= 1.0
    y = np.array([1, -1, 1, 1, -1, -1, 1, -1, 1], dtype=float)
    kta = kn.kernel_target_alignment(K, y)
    assert -1.0 <= kta <= 1.0
    assert abs(kn.kernel_target_alignment(np.outer(y, y), y) - 1.0) <= 1e-12
