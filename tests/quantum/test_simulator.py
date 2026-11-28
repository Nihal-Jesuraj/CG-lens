"""Stage-1 tests: statevector primitives (T1-T3 support)."""
import numpy as np
from scipy.linalg import expm

from src.quantum import simulator as sim


def _dense_z(q, i):
    """Dense Z on qubit i with qubit 0 as the least-significant bit."""
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    return np.kron(np.kron(np.eye(2 ** (q - 1 - i)), Z), np.eye(2 ** i))


def _dense_hadamard(q):
    H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    out = np.array([[1.0]])
    for _ in range(q):
        out = np.kron(out, H)
    return out


def test_walsh_hadamard_matches_dense():
    rng = np.random.default_rng(2026)
    worst = 0.0
    for q in (1, 2, 3, 5):
        Hq = _dense_hadamard(q)
        v = rng.standard_normal((3, 2 ** q)) + 1j * rng.standard_normal((3, 2 ** q))
        err = np.abs(sim.walsh_hadamard(v) - v @ Hq.T).max()
        worst = max(worst, float(err))
        assert err <= 1e-12
    print(f"WHT max abs error: {worst:.3e}")


def test_zz_phase_matches_dense_pauli_sum():
    rng = np.random.default_rng(7)
    worst = 0.0
    for q in (1, 2, 3, 4):
        theta = rng.uniform(0.0, np.pi, size=q)
        D = np.zeros((2 ** q, 2 ** q), dtype=np.complex128)
        Zs = [_dense_z(q, i) for i in range(q)]
        for i in range(q):
            D += theta[i] * Zs[i]
        for i in range(q):
            for j in range(i + 1, q):
                D += (np.pi - theta[i]) * (np.pi - theta[j]) * (Zs[i] @ Zs[j])
        phases_dense = np.exp(1j * np.diag(D))
        err = np.abs(sim.zz_phase(theta) - phases_dense).max()
        worst = max(worst, float(err))
        assert err <= 1e-12
    print(f"ZZ phase max abs error: {worst:.3e}")


def test_zz_circuit_matches_dense_unitary():
    rng = np.random.default_rng(11)
    q = 3
    theta = rng.uniform(0.0, np.pi, size=q)
    Hq = _dense_hadamard(q)
    Zs = [_dense_z(q, i) for i in range(q)]
    D = np.zeros((2 ** q, 2 ** q), dtype=np.complex128)
    for i in range(q):
        D += theta[i] * Zs[i]
    for i in range(q):
        for j in range(i + 1, q):
            D += (np.pi - theta[i]) * (np.pi - theta[j]) * (Zs[i] @ Zs[j])
    U = expm(1j * D) @ Hq
    psi_dense = U[:, 0]
    psi = sim.zz_circuit(theta, L=1)[0]
    fid = float(sim.state_fidelity(psi, psi_dense)[0])
    assert abs(fid - 1.0) <= 1e-12
    print(f"ZZ state fidelity vs dense: {fid:.15f}")


def test_zero_state_and_fidelity():
    psi = sim.zero_state(4, batch=2)
    assert psi.shape == (2, 16) and np.allclose(psi[:, 0], 1.0)
    assert np.allclose(sim.state_fidelity(psi, psi), 1.0)
