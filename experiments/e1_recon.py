"""E1: reconstruction grid M0/M1/M2 x R x SNR on recon_eval with frozen per-cell parameters; patient-level paired statistics."""
import sys

if "--timing" in sys.argv:                    # serial pass: single-threaded BLAS/FFT before numpy is imported
    import os
    for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_k] = "1"

import argparse
import itertools
import math
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats as sps

import config
from src import common, metrics, stats, mri_ops

PAIRS = [("M1", "M0"), ("M2", "M0"), ("M2", "M1")]
HIGHER_BETTER = {"psnr": True, "ssim": True, "nmse": False, "roi_nmse": False}
PRIMARY_METRICS = ["psnr", "roi_nmse"]


def _one(df_row, x, mask, idx, method, R, snr, hyper, use_cache=True):
    p = common.params_for(hyper, method, R, snr)
    xh, meta = common.reconstruct(x, idx, method, R, snr, p, use_cache=use_cache)
    m = metrics.all_metrics(xh, x, mask)
    return dict(slice_index=idx, pid=df_row.pid, label=int(df_row.label), method=method, R=R, snr=snr if math.isfinite(snr) else "inf",
                lam=p.get("lam"), mu=p.get("mu"), matvecs=meta["matvecs"], iters=meta["iters"], converged=meta["converged"],
                n_violations=meta["n_violations"], inner_hit_maxit=meta["inner_hit_maxit"], wall_s=meta["wall_s"], **m), xh


def wilcoxon_p(d):
    d = np.asarray(d)
    if np.allclose(d, 0):
        return 1.0
    return float(sps.wilcoxon(d).pvalue)


def analyse(pat):
    rows = []
    for (R, snr, metric), g in pat.groupby(["R", "snr", "metric"]):
        for a, b in PAIRS:
            pa = g[g.method == a].set_index("pid").value
            pb = g[g.method == b].set_index("pid").value
            pids = pa.index.intersection(pb.index)
            d = (pa[pids] - pb[pids]).to_numpy()
            ci = stats.mean_diff_ci(d, seed_keys=(21, int(R), config.snr_code(float(snr)) if snr != "inf" else config.SNR_INF_CODE, len(rows)))
            primary = R == config.R_PRIMARY and snr == str(int(config.SNR_PRIMARY)) and metric in PRIMARY_METRICS or (R == config.R_PRIMARY and str(snr) in (str(config.SNR_PRIMARY), "25") and metric in PRIMARY_METRICS)
            rows.append(dict(R=R, snr=snr, metric=metric, A=a, B=b, n_patients=len(d), mean_diff=float(d.mean()), ci_lo=ci["lo"], ci_hi=ci["hi"],
                             wilcoxon_p=wilcoxon_p(d), family="primary" if primary else "exploratory", improves=bool((d.mean() > 0) == HIGHER_BETTER[metric])))
    st = pd.DataFrame(rows)
    st["p_holm"] = np.nan
    prim = st.family == "primary"
    if prim.any():
        st.loc[prim, "p_holm"] = stats.holm(st.loc[prim, "wilcoxon_p"].to_numpy())
    for _, idx in st[~prim].groupby(["R", "snr", "metric"]).groups.items():
        st.loc[idx, "p_holm"] = stats.holm(st.loc[idx, "wilcoxon_p"].to_numpy())   # exploratory: Holm within (R, SNR, metric)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--timing", action="store_true", help="serial wall-clock pass, no cache")
    a = ap.parse_args()
    config.ensure_dirs()
    df, img, msk, splits = common.load_all()
    hyper = common.load_hyper()
    idxs = splits["recon_eval"]["slice_index"]
    Rs, snrs = config.R_LIST, config.SNR_LIST
    if a.quick:
        idxs, Rs = common.pick_quick(df, idxs), [config.R_PRIMARY]
    rows_by_idx = df.set_index("index")
    jobs = [(i, m, R, s) for i in idxs for m in common.METHODS for R in Rs for s in snrs]
    if a.timing:
        out = []
        for i, m, R, s in jobs:
            r, _ = _one(rows_by_idx.loc[i].rename_axis(None), img[i], msk[i], i, m, R, s, hyper, use_cache=False)
            out.append({k: r[k] for k in ("slice_index", "method", "R", "snr", "wall_s", "matvecs")})
        pd.DataFrame(out).to_csv(config.RESULTS_DIR / "e1_timing.csv", index=False)
        print("serial timing written")
        return
    res = Parallel(n_jobs=a.n_jobs)(delayed(_one)(rows_by_idx.loc[i], img[i], msk[i], i, m, R, s, hyper) for i, m, R, s in jobs)
    met = pd.DataFrame([r for r, _ in res])
    met["wall_s_parallel_pass"] = met.pop("wall_s")      # not a valid timing (parallel); see e1_timing.csv
    met.to_csv(config.RESULTS_DIR / "e1_metrics.csv", index=False)
    long = met.melt(id_vars=["pid", "method", "R", "snr", "slice_index"], value_vars=["psnr", "ssim", "nmse", "roi_nmse", "matvecs"], var_name="metric", value_name="value")
    pat = long.groupby(["pid", "method", "R", "snr", "metric"], as_index=False).value.mean()   # per-patient first (<=2 slices)
    pat.to_csv(config.RESULTS_DIR / "e1_patient.csv", index=False)
    summ = []
    for (m, R, s, metric), g in pat.groupby(["method", "R", "snr", "metric"]):
        ci = stats.clustered_bootstrap(np.stack([g.value.to_numpy(), np.ones(len(g))], 1), lambda S: S[:, 0] / S[:, 1], seed_keys=(22, int(R), len(summ)))
        summ.append(dict(method=m, R=R, snr=s, metric=metric, mean=float(g.value.mean()), std=float(g.value.std(ddof=1)) if len(g) > 1 else np.nan,
                         ci_lo=ci["lo"], ci_hi=ci["hi"], n_patients=len(g)))
    summ = pd.DataFrame(summ)
    tp = config.RESULTS_DIR / "e1_timing.csv"
    if tp.exists():
        t = pd.read_csv(tp).groupby(["method", "R", "snr"], as_index=False).wall_s.mean().rename(columns={"wall_s": "serial_wall_s_mean"})
        t["snr"] = t["snr"].astype(str)
        summ["snr"] = summ["snr"].astype(str)
        summ = summ.merge(t, on=["method", "R", "snr"], how="left")
    summ.to_csv(config.RESULTS_DIR / "e1_summary.csv", index=False)
    pat["snr"] = pat["snr"].astype(str)
    analyse(pat.query("metric != 'matvecs'")).to_csv(config.RESULTS_DIR / "e1_stats.csv", index=False)
    # examples for figure 3 (first two eval slices, R = primary, SNR = primary)
    ex = {}
    pixel_frames = []
    for k, i in enumerate(idxs[:2]):
        ex[f"x_true_{k}"], ex[f"mask_{k}"] = img[i], msk[i]
        yy, xx = np.indices(img[i].shape)
        frame = pd.DataFrame(dict(sample=k, slice_index=int(i), pixel_y=yy.ravel(), pixel_x=xx.ravel(),
                                  truth=img[i].ravel(), mask=msk[i].ravel()))
        for m in common.METHODS:
            ex[f"{m}_{k}"], _ = common.reconstruct(img[i], i, m, config.R_PRIMARY, config.SNR_PRIMARY, common.params_for(hyper, m, config.R_PRIMARY, config.SNR_PRIMARY))
            frame[m] = ex[f"{m}_{k}"].ravel()
        pixel_frames.append(frame)
    np.savez(config.RESULTS_DIR / "e1_examples.npz", **ex)
    pd.concat(pixel_frames, ignore_index=True).to_csv(config.RESULTS_DIR / "e1_examples.csv", index=False)
    mask_rows = []
    mask_slice = int(idxs[0])
    for R in config.R_LIST:
        M = mri_ops.make_mask(config.N, R, (R, mask_slice))
        yy, xx = np.indices(M.shape)
        mask_rows.append(pd.DataFrame(dict(R=R, slice_index=mask_slice, pixel_y=yy.ravel(), pixel_x=xx.ravel(), sampled=M.ravel())))
    pd.concat(mask_rows, ignore_index=True).to_csv(config.RESULTS_DIR / "mask_patterns.csv", index=False)
    print(summ.query("R == %d and metric == 'psnr'" % config.R_PRIMARY).to_string(index=False))


if __name__ == "__main__":
    main()
