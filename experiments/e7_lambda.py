"""E7 (P1): noise-adaptive lambda by the discrepancy principle (bisection on log lambda) vs tuned and oracle lambda."""
import argparse
import math

import numpy as np
import pandas as pd

import config
from src import common, metrics
from src import mri_ops as mo
from src import recon as rc


def estimate_noise_norm(y, M, band=0.9):
    """eps_hat = sigma_hat * sqrt(m); sigma_hat^2 = mean |y|^2 over sampled entries with normalised radius >= band."""
    N = M.shape[0]
    k = np.fft.fftfreq(N, d=1.0 / N)
    r = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2) / (N / 2)
    sel = (M > 0) & (r >= band)
    if not sel.any():                                   # fall back: the 10% of sampled entries with the largest radius
        rr = np.where(M > 0, r, -1)
        sel = rr >= np.quantile(rr[M > 0], 0.9)
    sigma2 = float(np.mean(np.abs(y[sel]) ** 2))
    return float(np.sqrt(sigma2 * M.sum()))


def discrepancy_lambda(residual_fn, eps, lo=-6.0, hi=1.0, tol=1e-3, max_iter=30):
    """Bisection on log10(lambda) so that residual_fn(lambda) = eps (residual increases with lambda)."""
    r_lo, r_hi = residual_fn(10 ** lo), residual_fn(10 ** hi)
    if r_lo >= eps:
        return dict(lam=10 ** lo, bracketed=False, iters=0, residual=r_lo)
    if r_hi <= eps:
        return dict(lam=10 ** hi, bracketed=False, iters=0, residual=r_hi)
    for it in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        r = residual_fn(10 ** mid)
        if abs(r - eps) <= tol * eps:
            return dict(lam=10 ** mid, bracketed=True, iters=it, residual=r)
        lo, hi = (mid, hi) if r < eps else (lo, mid)
    return dict(lam=10 ** (0.5 * (lo + hi)), bracketed=True, iters=max_iter, residual=r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-slices", type=int, default=6)
    a = ap.parse_args()
    config.ensure_dirs()
    df, img, msk, splits = common.load_all()
    hyper = common.load_hyper()
    idxs = splits["recon_eval"]["slice_index"][: 2 if a.quick else a.n_slices]
    R, snr = config.R_PRIMARY, config.SNR_PRIMARY
    rows = []
    for i in idxs:
        x = img[i]
        M, y = mo.simulate(x, R, snr, i)
        eps_hat = estimate_noise_norm(y, M)
        eps_true = float(np.linalg.norm(y - mo.A(x, M)))
        for method in ("M1", "M2"):
            tuned = common.params_for(hyper, method, R, snr)

            def solve(lam):
                p = dict(tuned, lam=lam)
                return rc.tikhonov_cg(y, M, lam, rtol=1e-8, maxit=500)[0] if method == "M1" else rc.tv_irls_cg(y, M, lam, p["mu"])[0]
            res_fn = lambda lam: float(np.linalg.norm(mo.A(solve(lam), M) - y))
            d = discrepancy_lambda(res_fn, eps_hat, tol=1e-2, max_iter=12 if a.quick else 25)
            grid = config.M1_LAMBDAS if method == "M1" else config.M2_LAMBDAS
            ps = {lam: metrics.psnr(solve(lam), x) for lam in grid}
            oracle = max(ps, key=ps.get)
            rows.append(dict(slice_index=i, method=method, eps_hat=eps_hat, eps_true=eps_true, lam_tuned=tuned["lam"], lam_discrepancy=d["lam"], lam_oracle=oracle,
                             bracketed=d["bracketed"], bisect_iters=d["iters"], psnr_tuned=metrics.psnr(solve(tuned["lam"]), x), psnr_discrepancy=metrics.psnr(solve(d["lam"]), x),
                             psnr_oracle=ps[oracle]))
        print("slice", i, "done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(config.RESULTS_DIR / "e7_lambda.csv", index=False)
    print(r.groupby("method")[["psnr_tuned", "psnr_discrepancy", "psnr_oracle"]].mean())


if __name__ == "__main__":
    main()
