"""E3: IRLS (validation settings) vs the exact CVXOPT anisotropic-TV QP on 16x16 slices from recon_val."""
import argparse
import itertools
import math
import time

import numpy as np
import pandas as pd

import config
from src import common, data
from src import mri_ops as mo
from src import qp_reference as qr
from src import recon as rc

MUS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
LAMS = [1e-3, 1e-2, 1e-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--snr", type=float, default=25.0, help="assumption: 25 dB unless set (use inf for noise-free)")
    a = ap.parse_args()
    config.ensure_dirs()
    df, img, msk, splits = common.load_all()
    idxs = splits["recon_val"]["slice_index"][:10]
    Rs, lams, mus = [2, 4], LAMS, MUS
    if a.quick:
        idxs, Rs, lams = idxs[:3], [4], [1e-2]
    p = mo.n_diffs(16)
    rows = []
    for i in idxs:
        x16 = data.area_resize(img[i], (16, 16))       # no re-scaling: area average of a [0,1] image stays in [0,1]
        for R in Rs:
            M, y = mo.simulate(x16, R, a.snr, i)
            for lam in lams:
                xq, qi = qr.tv_qp(M, y, lam)
                for mu in mus:
                    t0 = time.perf_counter()
                    xi, info = rc.tv_irls_cg(y, M, lam, mu, inner_rtol=config.VAL_INNER_RTOL, inner_maxit=config.CG_MAXIT * 5,
                                             outer_tol=config.VAL_OUTER_TOL, outer_maxit=config.VAL_OUTER_MAX)
                    ti = time.perf_counter() - t0
                    Jm, J0i = rc.objective(xi, y, M, lam, mu), qr.j0(xi, y, M, lam)
                    gap = Jm - J0i
                    rows.append(dict(slice_index=i, R=R, lam=lam, mu=mu, rel_diff=float(np.linalg.norm(xi - xq) / np.linalg.norm(xq)),
                                     J0_gap=float(J0i - qi["J0"]), gap_lhs=float(gap), gap_rhs=float(lam * p * mu), gap_bound_ok=bool(-1e-12 <= gap <= lam * p * mu + 1e-12),
                                     qp_status=qi["status"], qp_tol=qi["tolerance"] if qi["tolerance"] is not None else "default", qp_iters=qi["iterations"],
                                     qp_duality_gap=qi["gap"], qp_rel_gap=qi["relative_gap"], qp_time=qi["time"], ridge_bound=qi["ridge_bound"],
                                     irls_time=ti, irls_outer=info["outer_iters"], irls_converged=info["converged"],
                                     irls_inner_hit_maxit=info["inner_hit_maxit"], irls_inner_iters_total=info["iters"],
                                     irls_violations=info["n_violations"]))
        print(f"slice {i} done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(config.RESULTS_DIR / "e3_results.csv", index=False)
    s = r.groupby("mu").agg(rel_diff_median=("rel_diff", "median"), rel_diff_max=("rel_diff", "max"), J0_gap_median=("J0_gap", "median"),
                            gap_bound_all_ok=("gap_bound_ok", "all"), irls_converged_frac=("irls_converged", "mean"),
                            irls_inner_hit_maxit_mean=("irls_inner_hit_maxit", "mean"), irls_outer_mean=("irls_outer", "mean")).reset_index()
    s.to_csv(config.RESULTS_DIR / "e3_summary.csv", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
