"""E2: CG convergence vs conditioning (a1 32x32 with kappa bound, a2 128x128), IRLS descent (b); P1: NLCG, NMSE vs matvecs, sensitivity (c-e)."""
import argparse

import numpy as np
import pandas as pd
import scipy.sparse.linalg as spla

import config
from src import cg as cgm
from src import common, data, metrics
from src import mri_ops as mo
from src import qp_reference as qr
from src import recon as rc

E2_LAMBDA_32 = 1e-2
MUS = [1e-1, 1e-2, 1e-3, 1e-4]


def a1(img, idx):
    N = 32
    x = data.area_resize(img[idx], (N, N))
    M, y = mo.simulate(x, config.R_PRIMARY, config.SNR_PRIMARY, idx)
    b = mo.AH(y, M).ravel()
    Dm = mo.D_sparse(N)
    H0 = qr.dense_H(M)
    d = mo.D(x)
    curves, summ, eigs = [], [], {}
    for mu in MUS:
        w = 1 / np.sqrt(d * d + mu * mu)
        H = H0 + E2_LAMBDA_32 * (Dm.T @ Dm.multiply(w[:, None])).toarray()
        assert cgm.check_symmetry(lambda v: H @ v, N * N)
        ev = np.linalg.eigvalsh((H + H.T) / 2)
        kappa = float(ev[-1] / ev[0])
        xs = np.linalg.solve(H, b)
        e0 = float(np.sqrt(xs @ H @ xs))
        errs, res_hist = [], []
        r = cgm.cg(lambda v: H @ v, b, rtol=1e-12, maxit=N * N, callback=lambda k, xk: errs.append(float(np.sqrt((xk - xs) @ H @ (xk - xs)))))
        q = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
        for k, e in enumerate(errs, start=1):
            curves.append(dict(mu=mu, k=k, anorm_rel=e / e0, bound=2 * q ** k, resid_rel=r.resnorms[k] / r.resnorms[0]))
        viol = sum(e / e0 > 2 * q ** k * (1 + 1e-8) for k, e in enumerate(errs, start=1))
        summ.append(dict(mu=mu, lam=E2_LAMBDA_32, kappa=kappa, lam_min=float(ev[0]), lam_max=float(ev[-1]), iters=r.iters, converged=r.converged, bound_violations=int(viol)))
        eigs[f"mu_{mu:g}"] = ev
    pd.DataFrame(curves).to_csv(config.RESULTS_DIR / "e2a1_curves.csv", index=False)
    pd.DataFrame(summ).to_csv(config.RESULTS_DIR / "e2a1_summary.csv", index=False)
    np.savez(config.RESULTS_DIR / "e2a1_eigs.npz", **eigs)
    print(pd.DataFrame(summ).to_string(index=False))


def a2(img, idx, hyper):
    N = config.N
    x = img[idx]
    R, snr = config.R_PRIMARY, config.SNR_PRIMARY
    p = common.params_for(hyper, "M2", R, snr)
    xh, _ = common.reconstruct(x, idx, "M2", R, snr, p)
    M, y = mo.simulate(x, R, snr, idx)
    d = mo.D(xh)
    w = 1 / np.sqrt(d * d + p["mu"] ** 2)
    H = lambda v: mo.AHA(v, M) + p["lam"] * mo.DT(w * mo.D(v), N)
    assert cgm.check_symmetry(H, (N, N))
    b = mo.AH(y, M)
    ref = cgm.cg(H, b, rtol=1e-12, maxit=3000)
    e0 = np.sqrt(mo.rdot(ref.x, H(ref.x)))
    errs = []

    def cb(k, xk):
        e = xk - ref.x
        errs.append(float(np.sqrt(mo.rdot(e, H(e)))))
    r = cgm.cg(H, b, rtol=1e-10, maxit=1000, callback=cb)
    kappa = np.nan
    try:
        op = spla.LinearOperator((N * N, N * N), matvec=lambda v: H(v.reshape(N, N)).ravel(), dtype=np.float64)
        lmax = spla.eigsh(op, k=1, which="LA", return_eigenvectors=False, tol=1e-6)[0]
        lmin = spla.eigsh(op, k=1, which="SA", return_eigenvectors=False, tol=1e-4, maxiter=3000)[0]
        kappa = float(lmax / lmin)
    except Exception as e:  # diagnostics only
        print("eigsh diagnostic failed:", e)
    out = pd.DataFrame(dict(k=np.arange(1, len(errs) + 1), anorm_rel=np.array(errs) / e0, resid_rel=np.array(r.resnorms[1:]) / r.resnorms[0]))
    out["kappa_est"] = kappa
    out.to_csv(config.RESULTS_DIR / "e2a2_cg.csv", index=False)
    print("a2: iters", r.iters, "kappa_est", kappa)


def b(img, idxs, hyper, quick):
    rows = []
    R, snr = config.R_PRIMARY, config.SNR_PRIMARY
    for i in idxs:
        p = common.params_for(hyper, "M2", R, snr)
        M, y = mo.simulate(img[i], R, snr, i)
        _, info = rc.tv_irls_cg(y, M, p["lam"], p["mu"], inner_rtol=config.VAL_INNER_RTOL, outer_tol=config.VAL_OUTER_TOL,
                                outer_maxit=40 if quick else config.VAL_OUTER_MAX)
        J = info["J_hist"]
        for k in range(len(info["inner_iters"])):
            rows.append(dict(slice_index=i, outer=k + 1, J=J[k + 1], J_prev=J[k], inner_iters=info["inner_iters"][k], violations_total=info["n_violations"],
                             inner_hit_maxit_total=info["inner_hit_maxit"], lam=p["lam"], mu=p["mu"]))
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "e2b_irls.csv", index=False)
    print("b: monotonicity violations per slice:", pd.DataFrame(rows).groupby("slice_index").violations_total.first().to_dict())


def p1(img, idx, hyper, quick):
    R, snr = config.R_PRIMARY, config.SNR_PRIMARY
    x = img[idx]
    M, y = mo.simulate(x, R, snr, idx)
    rows = []
    p2 = common.params_for(hyper, "M2", R, snr)
    # (c, d) NLCG descent and NMSE vs matvecs, IRLS vs NLCG vs Tikhonov-CG
    cnt = dict(f=0, g=0)

    def f(z):
        cnt["f"] += 1
        return rc.objective(z, y, M, p2["lam"], p2["mu"])

    def g(z):
        cnt["g"] += 1
        return rc.gradient(z, y, M, p2["lam"], p2["mu"])
    res = cgm.nlcg(f, g, mo.AH(y, M), callback=lambda k, z: rows.append(dict(method="M3_NLCG", matvecs=cnt["g"] + 0.5 * cnt["f"], nmse=metrics.nmse(z, x), J=rc.objective(z, y, M, p2["lam"], p2["mu"]))))
    _, info = rc.tv_irls_cg(y, M, p2["lam"], p2["mu"], store_iterates=True, outer_maxit=15 if quick else config.IRLS_OUTER_MAX)
    mv = 0
    for k, z in enumerate(info["iterates"]):
        mv += 0 if k == 0 else info["inner_iters"][k - 1] + 1
        rows.append(dict(method="M2_IRLS", matvecs=mv, nmse=metrics.nmse(z, x), J=info["J_hist"][k]))
    p1_ = common.params_for(hyper, "M1", R, snr)
    mv = [0]
    from src.recon import H_tik
    b = mo.AH(y, M)
    cgm.cg(H_tik(M, p1_["lam"]), b, x0=b, rtol=1e-8, maxit=300, callback=lambda k, z: rows.append(dict(method="M1_CG", matvecs=k, nmse=metrics.nmse(z, x), J=np.nan)))
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "e2cd_nlcg_nmse.csv", index=False)
    common.write_json(config.RESULTS_DIR / "e2c_nlcg_flags.json", dict(armijo_ok=res.armijo_ok, converged=res.converged, iters=res.iters))
    # (e) (lambda, mu) sensitivity from the tuning tables
    t = pd.read_csv(config.RESULTS_DIR / "tuning_table.csv")
    t.to_csv(config.RESULTS_DIR / "e2e_sensitivity.csv", index=False)
    print("p1: NLCG converged", res.converged, "armijo_ok", res.armijo_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--p1", action="store_true")
    a = ap.parse_args()
    config.ensure_dirs()
    df, img, msk, splits = common.load_all()
    hyper = common.load_hyper()
    ev = splits["recon_eval"]["slice_index"]
    a1(img, splits["recon_val"]["slice_index"][0])
    a2(img, ev[0], hyper)
    b(img, ev[:1] if a.quick else ev[:2], hyper, a.quick)
    if a.p1:
        p1(img, ev[0], hyper, a.quick)


if __name__ == "__main__":
    main()
