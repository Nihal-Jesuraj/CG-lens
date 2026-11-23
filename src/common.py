"""Shared experiment helpers: data access, cached reconstruction, hyperparameter lookup, features."""
import json
import hashlib
import math
import os
import time

import numpy as np

import config
from src import cache, data, metrics
from src import mri_ops as mo
from src import recon as rc

METHODS = ["M0", "M1", "M2"]


def cell_key(R, snr):
    return f"R{R}_SNR{'inf' if not math.isfinite(snr) else int(snr)}"


def cond_name(method, R, snr):
    return "original" if method == "original" else f"{method}_{cell_key(R, snr)}"


def load_all():
    df, img, msk = data.load_slices()
    return df, img, msk, data.load_splits()


def load_hyper(required=True):
    p = config.RESULTS_DIR / "hyperparams.json"
    if not p.exists():
        if required:
            raise FileNotFoundError("results/hyperparams.json missing: run `python -m experiments.tune` first (freeze rule)")
        return None
    return json.loads(p.read_text())


def params_for(hyper, method, R, snr):
    if method == "M0":
        return {}
    return hyper[method][cell_key(R, snr)]


def solver_settings(method, validation=False):
    if method == "M1":
        return dict(rtol=config.CG_RTOL, maxit=config.CG_MAXIT)
    if method == "M2":
        if validation:
            return dict(inner_rtol=config.VAL_INNER_RTOL, inner_maxit=config.CG_MAXIT, outer_tol=config.VAL_OUTER_TOL, outer_maxit=config.VAL_OUTER_MAX)
        return dict(inner_rtol=config.CG_RTOL, inner_maxit=config.CG_MAXIT, outer_tol=config.IRLS_OUTER_TOL, outer_maxit=config.IRLS_OUTER_MAX)
    return {}


def observe(x_true, slice_index, R, snr):
    return mo.simulate(x_true, R, snr, slice_index)


def reconstruct(x_true, slice_index, method, R, snr, params=None, settings=None, use_cache=True, tag="recon"):
    """Cached reconstruction. Returns (x_hat, meta). Cache key = slice, method, R, SNR, lam, mu, solver settings, seed keys, config hash."""
    params = params or {}
    settings = solver_settings(method) if settings is None else settings
    input_sha = hashlib.sha256(np.ascontiguousarray(x_true, dtype=np.float64).tobytes()).hexdigest()
    key = dict(kind=tag, slice=int(slice_index), method=method, R=R, snr=snr, lam=params.get("lam"), mu=params.get("mu"), settings=settings,
               mask_keys=[R, int(slice_index)], noise_keys=[R, config.snr_code(snr), int(slice_index)], N=int(x_true.shape[0]),
               input_sha256=input_sha)

    def compute():
        M, y = observe(x_true, slice_index, R, snr)
        t0 = time.perf_counter()
        if method == "M0":
            xh, info = rc.zero_filled(y, M), dict(matvecs=0, iters=0, converged=True)
        elif method == "M1":
            xh, info = rc.tikhonov_cg(y, M, params["lam"], **settings)
        elif method == "M2":
            xh, info = rc.tv_irls_cg(y, M, params["lam"], params["mu"], **settings)
        else:
            raise ValueError(method)
        wall = time.perf_counter() - t0
        meta = dict(matvecs=int(info["matvecs"]), iters=int(info["iters"]), converged=bool(info["converged"]), wall_s=wall,
                    n_violations=int(info.get("n_violations", 0)), inner_hit_maxit=int(info.get("inner_hit_maxit", 0)),
                    outer_iters=int(info.get("outer_iters", 0)))
        return xh, meta
    return cache.get_or_compute(key, compute) if use_cache else compute()


def feature(x):
    """128x128 -> 64x64 area average -> flatten (Convention 9)."""
    return data.area_resize(x, (config.N_FEAT, config.N_FEAT)).ravel()


def slice_metrics(xh, x, mask):
    return metrics.all_metrics(xh, x, mask)


def serial_threads():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def pick_quick(df_slices, indices, n=None, per_class=True):
    """QUICK subset: n slices from distinct patients, class-balanced."""
    n = n or config.QUICK_SLICES
    sub = df_slices[df_slices["index"].isin(indices)].drop_duplicates("pid")
    out = []
    for lab in (1, 2):
        out += list(sub[sub.label == lab]["index"][: n // 2])
    return sorted(int(i) for i in out)
