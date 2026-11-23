"""Solver-validation summary table: CG vs dense and scipy, PCA vs sklearn, QP/SMO vs libsvm, IRLS vs CVXOPT (from E3)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import scipy.sparse.linalg as spla
from sklearn.decomposition import PCA as SKPCA

import config
from src import cg as cgm
from src import mri_ops as mo
from src import pca as pcam
from src import qp_reference as qr
from src import recon as rc
from src import svm


def main():
    config.ensure_dirs()
    rows = []
    def add(check, value, thr, ok, **extra):
        rows.append(dict(check=check, value=value, threshold=thr, passed=ok, **extra))
    M = mo.make_mask(16, 2, (2, 0))
    lam = 0.1
    H = qr.dense_H(M) + lam * (mo.D_sparse(16).T @ mo.D_sparse(16)).toarray()
    b = np.random.default_rng(0).standard_normal(256)
    r = cgm.cg(lambda v: H @ v, b, rtol=1e-13, maxit=2000)
    err = float(np.abs(r.x - np.linalg.solve(H, b)).max())
    add("CG vs np.linalg.solve (16x16, max abs error)", err, 1e-8, err < 1e-8)
    cnt = []
    spla.cg(H, b, rtol=1e-8, atol=0.0, maxiter=2000, callback=lambda xk: cnt.append(1))
    it = cgm.cg(lambda v: H @ v, b, rtol=1e-8, maxit=2000).iters
    add("CG iteration count vs scipy.sparse.linalg.cg", it - len(cnt), 0, it == len(cnt))
    p = config.RESULTS_DIR
    if (p / "e2a1_summary.csv").exists():
        v = int(pd.read_csv(p / "e2a1_summary.csv").bound_violations.sum())
        add("CG A-norm bound violations (32x32, all mu)", v, 0, v == 0)
    if (p / "e3_summary.csv").exists():
        s = pd.read_csv(p / "e3_summary.csv").sort_values("mu")
        trend = np.diff(s.rel_diff_median.to_numpy())
        add("IRLS vs CVXOPT: median relative difference decreases as mu shrinks",
            bool(np.all(trend >= -1e-12)), "monotone across all adjacent mu values", bool(np.all(trend >= -1e-12)))
        add("Gap bound 0 <= J_mu - J_0 <= lam p mu at every mu", bool(s.gap_bound_all_ok.all()), True, bool(s.gap_bound_all_ok.all()))
        for r in s.itertuples(index=False):
            frac, outer = float(r.irls_converged_frac), float(r.irls_outer_mean)
            add(f"E3 IRLS convergence fraction at mu={r.mu:g}", frac, "1.0; inspect outer mean against 500 cap",
                bool(frac >= 1.0 - 1e-12), mu=float(r.mu),
                irls_converged_frac=frac, irls_outer_mean=outer)
    X = np.random.default_rng(1).standard_normal((60, 40)) @ np.diag(np.linspace(5, 0.1, 40))
    ours, sk = pcam.PCA(n_components=8).fit(X), SKPCA(n_components=8, svd_solver="full").fit(X)
    e = float(np.abs(ours.singular_values_[:8] - sk.singular_values_).max() / sk.singular_values_[0])
    cos = float(np.linalg.svd(ours.components_ @ sk.components_.T, compute_uv=False).min())
    add("PCA singular values vs sklearn (relative)", e, 1e-10, e < 1e-10)
    add("PCA subspace min cos(theta) vs sklearn", cos, ">= 1 - 1e-8", cos >= 1 - 1e-8)
    rng = np.random.default_rng(2)
    Xs = np.vstack([rng.standard_normal((40, 3)) + 1.2, rng.standard_normal((100, 3)) - 1.2])
    ys = np.r_[np.ones(40), -np.ones(100)]
    gam = svm.gamma_from_g(1.0, Xs)
    qp, smo = svm.SVM_QP(C=1.0, gamma=gam).fit(Xs, ys), svm.SVM_SMO(C=1.0, gamma=gam).fit(Xs, ys)
    a, _, m = svm.svc_reference(Xs, ys, 1.0, "rbf", gam)
    K = qp._K(Xs, Xs)
    ds = svm.dual_objective(a, ys, K)
    rq, rs = abs(qp.dual_ - ds) / abs(ds), abs(smo.dual_ - ds) / abs(ds)
    add("SVM_QP dual objective vs SVC (relative)", float(rq), 1e-3, rq < 1e-3)
    add("SVM_SMO dual objective vs SVC (relative)", float(rs), 1e-2, rs < 1e-2)
    add("SVM_SMO final KKT gap", float(smo.final_gap_), config.SMO_TOL, smo.final_gap_ <= config.SMO_TOL)
    if (p / "e4_solver_agreement.csv").exists():
        ag = pd.read_csv(p / "e4_solver_agreement.csv")
        for col, name in (("agree_qp_svc", "QP vs SVC"), ("agree_smo_svc", "SMO vs SVC")):
            v = float(ag[col].mean())
            add(f"E4 held-out prediction agreement {name} (R=4)", v, 0.98, v >= 0.98)
        for solver, name, threshold in (("qp", "SVM_QP", 1e-3), ("smo", "SVM_SMO", 1e-2)):
            rel = np.abs(ag[f"dual_{solver}"] - ag["dual_svc"]) / np.abs(ag["dual_svc"])
            v = float(rel.mean())
            add(f"E4 relative dual-objective difference |dual - dual_svc| / |dual_svc| ({name})",
                v, threshold, bool(np.isfinite(v) and v < threshold))
    out = pd.DataFrame(rows)
    out.to_csv(config.RESULTS_DIR / "validation_summary.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
