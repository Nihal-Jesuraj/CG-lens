"""G-BENCH: time 10 slices x {M0, M1, M2} serially, project total serial hours, write results/benchmark.json."""
import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import config
from src import common

CUT_ORDER = ["M10 (E7, E8)", "M9 (NLCG, E2 c-e, 1-D mask)", "E2(c-e)", "QP runs at n >= 1000 in E5", "secondary SNR in E4", "R = 2 and R = 6 in E4", "E4 repeats 3 -> 1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=config.BENCH_SLICES)
    a = ap.parse_args()
    config.ensure_dirs()
    df, img, msk, splits = common.load_all()
    idxs = splits["recon_val"]["slice_index"][: a.n]
    R, snr = config.R_PRIMARY, config.SNR_PRIMARY
    hyper = common.load_hyper(required=False)
    t = {}
    for m in common.METHODS:
        if m == "M2":
            pts = [dict(lam=1e-2, mu=1e-2), dict(lam=1e-3, mu=1e-3), dict(lam=1e-4, mu=1e-4)]    # representative grid points
        elif m == "M1":
            pts = [dict(lam=1e-3), dict(lam=1e-2), dict(lam=1e-1)]
        else:
            pts = [{}]
        per = []
        for p in pts:
            ts = []
            for i in idxs:
                t0 = time.perf_counter()
                common.reconstruct(img[i], i, m, R, snr, p, use_cache=False)
                ts.append(time.perf_counter() - t0)
            per.append(float(np.mean(ts)))
        t[m] = dict(per_point=per, mean=float(np.mean(per)), middle=per[len(per) // 2])
    cells = len(config.R_LIST) * len(config.SNR_LIST)
    nval, neval, npool = config.RECON_VAL_PATIENTS * 2, config.RECON_EVAL_PATIENTS * 2, config.CLF_POOL_PATIENTS * 2
    if splits:
        nval, neval, npool = splits["recon_val"]["n_slices"], splits["recon_eval"]["n_slices"], splits["clf_pool"]["n_slices"]
    ext = 1.3   # allowance for edge-rule grid extensions
    tune = nval * cells * ext * (len(config.M1_LAMBDAS) * t["M1"]["mean"] + len(config.M2_LAMBDAS) * len(config.M2_MUS) * t["M2"]["mean"])
    e1 = neval * cells * sum(t[m]["middle"] for m in common.METHODS)
    e4 = npool * (cells * sum(t[m]["middle"] for m in common.METHODS) + t["M0"]["middle"])
    total = tune + e1 + e4
    out = dict(per_method_seconds=t, slices_timed=len(idxs), n_val=nval, n_eval=neval, n_pool=npool, projected_seconds=dict(tuning=tune, e1=e1, e4_recon=e4),
               projected_serial_hours=total / 3600.0, budget_hours=config.BENCH_BUDGET_HOURS, over_budget=bool(total / 3600.0 > config.BENCH_BUDGET_HOURS),
               cut_order=CUT_ORDER, note="timing at R=4, SNR=25 dB; M2 at three representative (lam, mu); excludes classification (minutes) and E5")
    common.write_json(config.RESULTS_DIR / "benchmark.json", out)
    print(f"projected serial reconstruction time: {total / 3600:.2f} h (budget {config.BENCH_BUDGET_HOURS} h)")
    if out["over_budget"]:
        print("OVER BUDGET: apply the cut order from Section 5.3 before continuing:", " -> ".join(CUT_ORDER))


if __name__ == "__main__":
    main()
