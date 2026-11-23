"""Masks, forward operator A, adjoint, noise, and finite differences D (Conventions 2-6, 11)."""
import numpy as np
import scipy.sparse as sp

import config


def seed_rng(*keys):
    """Generator from SeedSequence([GLOBAL_SEED, *integer keys]); never uses hash()."""
    return np.random.default_rng(np.random.SeedSequence([config.GLOBAL_SEED, *[int(k) for k in keys]]))


def fft2c(x):
    return np.fft.fft2(x, norm="ortho")


def ifft2c(y):
    return np.fft.ifft2(y, norm="ortho")


def rdot(a, b):
    """Real inner product Re sum conj(a)*b."""
    return float(np.real(np.vdot(a, b)))


def A(x, M):
    return M * fft2c(x)


def AH(y, M):
    return np.real(ifft2c(M * y))


def AHA(x, M):
    return np.real(ifft2c(M * fft2c(x)))


# ---------------------------------------------------------------- masks
def partner(M):
    """Conjugate partner: M[(-i) mod N, (-j) mod N]."""
    return np.roll(np.flip(M, axis=(0, 1)), shift=(1, 1), axis=(0, 1))


def _freq(N):
    return np.fft.fftfreq(N, d=1.0 / N)


def _radius(N):
    k = _freq(N)
    return np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)


def _density(N, a, disc):
    r = np.minimum(_radius(N) / (N / 2), 1.0)
    p = np.clip(a * (1.0 - r) ** 3, 0.0, 1.0)
    p[disc] = 1.0
    return p


def _solve_a(N, target_mean, disc):
    lo, hi = 0.0, 1.0
    while _density(N, hi, disc).mean() < target_mean and hi < 1e12:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _density(N, mid, disc).mean() < target_mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_mask(N, R, seed_keys, symmetric=None):
    """2-D variable-density conjugate-symmetric mask with exact pair-corrected rate (Sec 4.2)."""
    if symmetric is None:
        symmetric = config.SYMMETRIC_MASK
    if R == 1:
        return np.ones((N, N))
    rng = seed_rng(*seed_keys)
    disc = _radius(N) <= config.CENTER_DISC_FRAC * N
    disc[0, 0] = True
    a = _solve_a(N, 1.0 / R, disc)
    p = _density(N, a, disc)
    idx = np.arange(N * N).reshape(N, N)
    u = rng.random(N * N)
    if not symmetric:
        return (u.reshape(N, N) < p).astype(np.float64)
    pid = np.minimum(idx, partner(idx))
    M = (u[pid.ravel()].reshape(N, N) < p)
    size = np.where(idx == partner(idx), 1, 2)
    target = int(round(N * N / R))
    m = int(M.sum())
    reps = (idx == pid) & ~disc
    for want_on in ((False,) if m < target else (True,)):
        cand = np.flatnonzero((reps & (M == want_on)).ravel())
        cand = cand[rng.permutation(len(cand))]
        flat = M.ravel()
        szf = size.ravel()
        idxf = idx.ravel()
        for c in cand:
            if abs(m - target) <= 1:
                break
            s = int(szf[c])
            new_m = m + s if not want_on else m - s
            if abs(new_m - target) < abs(m - target):
                flat[c] = not want_on
                flat[int(partner(idx).ravel()[c])] = not want_on
                m = new_m
        M = flat.reshape(N, N)
    return M.astype(np.float64)


def make_mask_1d(N, R, seed_keys):
    """P1 secondary: 1-D Cartesian phase-encode lines, central 8% full, conjugate-symmetric line pairs."""
    rng = seed_rng(*seed_keys)
    k = np.rint(_freq(N)).astype(int)
    lines = np.zeros(N, dtype=bool)
    lines[np.abs(k) <= config.CAR_FRAC * N / 2] = True
    target = max(int(round(N / R)), int(lines.sum()))
    ids = np.arange(N)
    partner_id = (-ids) % N
    cand = [i for i in ids if i <= partner_id[i] and not lines[i]]
    cand = [cand[j] for j in rng.permutation(len(cand))]
    for i in cand:
        if lines.sum() >= target:
            break
        lines[i] = True
        lines[partner_id[i]] = True
    return np.tile(lines[:, None], (1, N)).astype(np.float64)


def achieved_R(M):
    return M.size / M.sum()


# ---------------------------------------------------------------- noise and simulation
def make_noise(Ax, M, snr_db, seed_keys):
    """eta = M * F(w), w real Gaussian image, rescaled so ||eta|| = ||Ax|| 10^(-SNR/20)."""
    if not np.isfinite(snr_db):
        return np.zeros_like(Ax)
    w = seed_rng(*seed_keys).standard_normal(Ax.shape)
    eta = M * fft2c(w)
    return eta * (np.linalg.norm(Ax) * 10.0 ** (-snr_db / 20.0) / np.linalg.norm(eta))


def simulate(x, R, snr_db, slice_index, mask_kind="2d"):
    """Mask keys (R, slice); noise keys (R, SNR code, slice): identical for every method."""
    N = x.shape[0]
    mk = make_mask if mask_kind == "2d" else make_mask_1d
    M = mk(N, R, (R, slice_index))
    Ax = A(x, M)
    return M, Ax + make_noise(Ax, M, snr_db, (R, config.snr_code(snr_db), slice_index))


# ---------------------------------------------------------------- finite differences
def D(x):
    """Stacked valid forward differences [Dx; Dy], p = 2N(N-1), Neumann boundary."""
    dx = x[1:, :] - x[:-1, :]
    dy = x[:, 1:] - x[:, :-1]
    return np.concatenate([dx.ravel(), dy.ravel()])


def DT(g, N):
    """Transpose of D (negative divergence)."""
    nx = (N - 1) * N
    gx = g[:nx].reshape(N - 1, N)
    gy = g[nx:].reshape(N, N - 1)
    out = np.zeros((N, N))
    out[1:, :] += gx
    out[:-1, :] -= gx
    out[:, 1:] += gy
    out[:, :-1] -= gy
    return out


def D_sparse(N):
    d1 = sp.diags([-np.ones(N - 1), np.ones(N - 1)], [0, 1], shape=(N - 1, N))
    I = sp.identity(N)
    return sp.vstack([sp.kron(d1, I), sp.kron(I, d1)]).tocsr()


def n_diffs(N):
    return 2 * N * (N - 1)
