"""E5: SVM_QP vs SVM_SMO vs libsvm (SVC) on E5_pool features. Serial timing; QP time cap with hard-timeout subprocess."""
import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

import config
from src import common
from src import mri_ops as mo
from src import pca as pcam
from src import svm


def _qp_worker(q, Z, y, C, kernel, gamma):
    from src import svm as s
    m = s.SVM_QP(C=C, kernel=kernel, gamma=gamma).fit(Z, y)
    q.put((m.alpha_, m.b_, m.solve_time, m.kernel_time, m.status_))


def qp_subprocess(Z, y, C, kernel, gamma, timeout):
    """Run the dense QP in a separate process with a hard timeout. Returns tuple or None on timeout."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_qp_worker, args=(q, Z, y, C, kernel, gamma))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None
    return q.get() if not q.empty() else None


def balanced_acc(y, pred):
    return 0.5 * (np.mean(pred[y == 1] == 1) + np.mean(pred[y == -1] == -1))


def fit_all(Ztr, ytr, Zte, yte, C, kernel, g, n, t500_qp, cap):
    gamma = svm.gamma_from_g(g, Ztr) if kernel == "rbf" else 1.0
    K = svm.rbf_kernel(Ztr, Ztr, gamma) if kernel == "rbf" else svm.linear_kernel(Ztr, Ztr)
    Cvec = svm.balanced_C(ytr, C)
    out, alphas, preds = [], {}, {}
    # QP
    qp_row = dict(solver="qp")
    if n <= config.QP_DENSE_MAX_N:
        m = svm.SVM_QP(C=C, kernel=kernel, gamma=gamma).fit(Ztr, ytr)
        alpha, solve_t, kern_t, status = m.alpha_, m.solve_time, m.kernel_time, m.status_
        preds["qp"] = m.predict(Zte)
    else:
        pred_t = None if t500_qp is None else t500_qp * (n / config.QP_DENSE_MAX_N) ** 3
        if pred_t is None or pred_t > cap:
            alpha, status, solve_t, kern_t = None, f"skipped: predicted {pred_t if pred_t is None else round(pred_t, 1)} s above cap {cap} s", np.nan, np.nan
        else:
            r = qp_subprocess(Ztr, ytr, C, kernel, gamma, cap)
            if r is None:
                alpha, status, solve_t, kern_t = None, f"skipped: exceeded hard timeout {cap} s", np.nan, np.nan
            else:
                alpha, b, solve_t, kern_t, status = r
                # Use the timed subprocess solution directly. Refitting here
                # would repeat the dense QP without the promised hard timeout.
                Kte = svm.rbf_kernel(Zte, Ztr, gamma) if kernel == "rbf" else svm.linear_kernel(Zte, Ztr)
                sv = alpha > config.SV_THRESH * Cvec
                decision = Kte[:, sv] @ (alpha[sv] * ytr[sv]) + b
                preds["qp"] = np.where(decision >= 0, 1, -1)
    alphas["qp"] = alpha
    out.append(dict(solver="qp", solve_time=solve_t, kernel_time=kern_t, status=status))
    # SMO
    smo = svm.SVM_SMO(C=C, kernel=kernel, gamma=gamma).fit(Ztr, ytr)
    alphas["smo"], preds["smo"] = smo.alpha_, smo.predict(Zte)
    out.append(dict(solver="smo", solve_time=smo.solve_time, kernel_time=smo.kernel_time, status=f"gap={smo.final_gap_:.2e} iters={smo.iterations_}"))
    # libsvm
    t0 = time.perf_counter()
    a_svc, b_svc, sk = svm.svc_reference(Ztr, ytr, C, kernel, gamma)
    t_svc = time.perf_counter() - t0
    alphas["svc"], preds["svc"] = a_svc, sk.predict(Zte)
    out.append(dict(solver="svc", solve_time=t_svc, kernel_time=np.nan, status="fit time includes kernel evaluation"))
    rows = []
    for o in out:
        s = o["solver"]
        a = alphas[s]
        row = dict(o, n=n, kernel=kernel, C=C, g=g if kernel == "rbf" else np.nan, n_features=Ztr.shape[1])
        if a is not None:
            row.update(dual=svm.dual_objective(a, ytr, K), n_sv=int((a > config.SV_THRESH * Cvec).sum()),
                       libsvm_stop_criterion_b_interval_width=svm.kkt_gap(a, ytr, K, Cvec),
                       accuracy=float(np.mean(preds[s] == yte)), balanced_accuracy=balanced_acc(yte, preds[s]))
        rows.append(row)
    sv = {s: (a > config.SV_THRESH * Cvec) if a is not None else None for s, a in alphas.items()}
    for row in rows:
        for other in ("qp", "smo", "svc"):
            row[f"jaccard_vs_{other}"] = svm.jaccard(sv[row["solver"]], sv[other]) if sv[row["solver"]] is not None and sv[other] is not None else np.nan
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    config.ensure_dirs()
    df, img, msk, splits = common.load_all()
    e5 = splits["E5_pool"]
    tr_idx, te_idx = np.array(e5["train_index"]), np.array(e5["test_index"])
    y = df.y.to_numpy().astype(int)
    feats = np.stack([common.feature(img[i]) for i in df["index"]])
    ns, seeds, cs = config.E5_NS, config.E5_SEEDS, config.E5_C_GRID
    cap = config.QP_TIME_CAP_S
    if a.quick:
        ns, seeds, cs, cap = [40, 80], [0], [1.0, 10.0], 60
    rows, sweep, log = [], [], []
    for seed in seeds:
        t500 = {}
        for n in ns:
            if n > len(tr_idx):
                log.append(f"skipped n={n}: only {len(tr_idx)} training slices")
                continue
            rng = mo.seed_rng(31, seed, n)
            sel = np.sort(rng.choice(tr_idx, n, replace=False))
            if len(set(y[sel])) < 2:
                log.append(f"skipped n={n}, seed={seed}: sampled training subset contains one class")
                continue
            pca = pcam.PCA().fit(feats[sel])
            Ztr, Zte = pca.transform(feats[sel]), pca.transform(feats[te_idx])
            for kernel in ("rbf", "linear"):
                nref = max(x for x in ns if x <= config.QP_DENSE_MAX_N) if any(x <= config.QP_DENSE_MAX_N for x in ns) else None
                r = fit_all(Ztr, y[sel], Zte, y[te_idx], 1.0, kernel, 1.0, n, t500.get(kernel), cap)
                for row in r:
                    row.update(seed=seed, k_pca=pca.k_)
                    if row["solver"] == "qp" and str(row["status"]).startswith("skipped:"):
                        log.append(f"QP skipped n={n}, seed={seed}, kernel={kernel}: {row['status']}")
                    if row["solver"] == "qp" and n == nref and np.isfinite(row["solve_time"]):
                        t500[kernel] = row["solve_time"] * (config.QP_DENSE_MAX_N / n) ** 3   # extrapolate to n = 500 (~ n^3)
                rows += r
                print(f"seed {seed} n {n} {kernel}: " + ", ".join(f"{x['solver']} {x['solve_time']:.3f}s" for x in r), flush=True)
        nsw = 500 if 500 in ns else ns[len(ns) // 2] if a.quick else 500
        nsw = min(nsw, len(tr_idx))
        rng = mo.seed_rng(32, seed)
        sel = np.sort(rng.choice(tr_idx, nsw, replace=False))
        pca = pcam.PCA().fit(feats[sel])
        Ztr, Zte = pca.transform(feats[sel]), pca.transform(feats[te_idx])
        for C in cs:
            sweep_rows = fit_all(Ztr, y[sel], Zte, y[te_idx], C, "rbf", 1.0, nsw, None, cap)
            for row in sweep_rows:
                if row["solver"] == "qp" and str(row["status"]).startswith("skipped:"):
                    log.append(f"QP skipped n={nsw}, seed={seed}, C={C}, kernel=rbf: {row['status']}")
                sweep.append(dict(row, seed=seed))
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "e5_scaling.csv", index=False)
    pd.DataFrame(sweep).to_csv(config.RESULTS_DIR / "e5_c_sweep.csv", index=False)
    (config.RESULTS_DIR / "e5_log.txt").write_text("\n".join(log) if log else "no skips or timeouts\n")
    r = pd.DataFrame(rows)
    print(r.groupby(["n", "kernel", "solver"])[["solve_time", "dual", "n_sv", "balanced_accuracy"]].mean().to_string())


if __name__ == "__main__":
    main()
