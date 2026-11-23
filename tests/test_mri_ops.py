import numpy as np
import pytest

import config
from src import mri_ops as mo


def _rand_complex(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


@pytest.mark.parametrize("R", [2, 4, 6, 8])
def test_mask_symmetric_and_rate(R):
    for s in range(20):
        M = mo.make_mask(128, R, (R, s))
        assert np.array_equal(M, mo.partner(M))
        assert abs(mo.achieved_R(M) - R) / R < 0.03
        assert M[0, 0] == 1
    # central disc always sampled
    k = np.fft.fftfreq(128, 1 / 128)
    disc = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2) <= 0.05 * 128
    assert M[disc].all()


def test_adjoint_dot_test_and_realness():
    rng = np.random.default_rng(0)
    M = mo.make_mask(16, 4, (4, 0))
    for _ in range(5):
        x = rng.standard_normal((16, 16))
        y = _rand_complex(rng, (16, 16))
        assert abs(mo.rdot(mo.A(x, M), y) - mo.rdot(x, mo.AH(y, M))) < 1e-10
        raw = mo.ifft2c(M * mo.fft2c(x))
        assert np.abs(raw.imag).max() < 1e-12


def test_projection_property_n16():
    N = 16
    M = mo.make_mask(N, 4, (4, 1))
    H = np.zeros((N * N, N * N))
    for j in range(N * N):
        e = np.zeros(N * N)
        e[j] = 1.0
        H[:, j] = mo.AHA(e.reshape(N, N), M).ravel()
    assert np.abs(H @ H - H).max() < 1e-10
    assert abs(np.trace(H) - M.sum()) < 1e-10
    assert np.abs(H - H.T).max() < 1e-12


def test_asymmetric_mask_demonstration(capsys):
    N = 16
    Ms = mo.make_mask(N, 4, (4, 2), symmetric=True)
    Ma = mo.make_mask(N, 4, (4, 2), symmetric=False)

    def eig(M):
        H = np.zeros((N * N, N * N))
        for j in range(N * N):
            e = np.zeros(N * N)
            e[j] = 1.0
            H[:, j] = mo.AHA(e.reshape(N, N), M).ravel()
        return np.round(np.unique(np.round(np.linalg.eigvalsh((H + H.T) / 2), 6)), 4), H
    es, _ = eig(Ms)
    ea, Ha = eig(Ma)
    print("symmetric mask eigenvalues:", es)
    print("asymmetric mask eigenvalues:", ea)
    assert set(es) <= {0.0, 1.0}
    assert np.any(np.isclose(ea, 0.5))          # half-weight eigenvalues appear
    assert np.abs(Ha @ Ha - Ha).max() > 1e-3    # not a projection


def test_noise_properties():
    rng = np.random.default_rng(1)
    x = rng.random((32, 32))
    M = mo.make_mask(32, 4, (4, 5))
    Ax = mo.A(x, M)
    eta = mo.make_noise(Ax, M, 25.0, (4, 25, 5))
    assert np.abs(mo.ifft2c(eta).imag).max() < 1e-12
    assert abs(np.linalg.norm(eta) / (np.linalg.norm(Ax) * 10 ** (-25 / 20)) - 1) < 1e-12
    assert np.all(mo.make_noise(Ax, M, np.inf, (4, 999, 5)) == 0)


def test_seed_determinism():
    x = np.random.default_rng(2).random((32, 32))
    M1, y1 = mo.simulate(x, 4, 25.0, 7)
    M2, y2 = mo.simulate(x, 4, 25.0, 7)
    assert np.array_equal(M1, M2) and np.array_equal(y1, y2)
    M3, y3 = mo.simulate(x, 4, 25.0, 8)
    assert not np.array_equal(M1, M3) and not np.array_equal(y1, y3)
    M4, y4 = mo.simulate(x, 4, np.inf, 7)       # same mask across SNRs, different noise
    assert np.array_equal(M1, M4) and not np.array_equal(y1, y4)
    assert config.snr_code(np.inf) != config.snr_code(25.0)


def test_D_dot_test_shape_and_sparse():
    N = 16
    rng = np.random.default_rng(3)
    x = rng.standard_normal((N, N))
    g = rng.standard_normal(mo.n_diffs(N))
    assert mo.n_diffs(N) == 2 * N * (N - 1) == mo.D(x).size
    assert abs(np.dot(mo.D(x), g) - np.sum(x * mo.DT(g, N))) < 1e-10
    Ds = mo.D_sparse(N)
    assert Ds.shape == (2 * N * (N - 1), N * N)
    assert np.abs(Ds @ x.ravel() - mo.D(x)).max() < 1e-12
    assert np.abs(Ds.T @ g - mo.DT(g, N).ravel()).max() < 1e-12
    assert np.abs(mo.D(np.ones((N, N)))).max() == 0  # Neumann: constants in the kernel


def test_mask_1d_symmetric():
    M = mo.make_mask_1d(64, 4, (4, 0))
    assert np.array_equal(M, mo.partner(M))
    assert M[0, :].all()
