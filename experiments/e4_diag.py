"""E4: patient-disjoint reconstruction-to-diagnosis. Order: folds.json -> tuning-consistency check -> grid (+ solver-agreement check)."""
import argparse
import itertools
import json
import math

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import config
from src import common
from src import pca as pcam
from src import stats, svm

CONDS_FIXED = ["original", "original_noisy"]
PRIMARY_CONDS = ["original"] + [f"{m}_{common.cell_key(config.R_PRIMARY, config.SNR_PRIMARY)}" for m in common.METHODS]
KERNELS = ["rbf", "linear"]


# ----------------------------------------------------------------- folds
def make_folds(y, groups, pids, P):
    """Outer StratifiedGroupKFold x repeats, inner folds per outer training set. Identical for every condition."""
    out = []
    for r in range(P["repeats"]):
        sgk = StratifiedGroupKFold(n_splits=P["outer"], shuffle=True, random_state=config.GLOBAL_SEED + r)
        for f, (tr, te) in enumerate(sgk.split(np.zeros(len(y)), y, groups)):
            assert not set(pids[tr]) & set(pids[te]) and not set(groups[tr]) & set(groups[te]), "patient overlap"
            inner = StratifiedGroupKFold(n_splits=P["inner"], shuffle=True, random_state=config.GLOBAL_SEED + 100 * r + f)
            inn = [(tr[a].tolist(), tr[b].tolist()) for a, b in inner.split(np.zeros(len(tr)), y[tr], groups[tr])]
            out.append(dict(repeat=r, fold=f, train=tr.tolist(), test=te.tolist(), inner=inn))
    for r in range(P["repeats"]):
        seen = sorted(i for fo in out if fo["repeat"] == r for i in fo["test"])
        assert seen == list(range(len(y))), "each slice must be tested exactly once per repeat"
    return out


# ----------------------------------------------------------------- tuning
def _ba(y, pred):
    pos, neg = y == 1, y == -1
    if not pos.any() or not neg.any():
        return np.nan
    return 0.5 * (np.mean(pred[pos] == 1) + np.mean(pred[neg] == -1))


def tune(X, y, tr, inner, kernel, use_sklearn, P):
    """Inner-CV selection of (C, g) on training folds only; PCA refitted inside every inner training fold."""
    from sklearn.svm import SVC
    gs = P["g"] if kernel == "rbf" else [None]
    grid = list(itertools.product(P["C"], gs))
    scores = np.full((len(grid), len(inner)), np.nan)
    for k, (itr, iva) in enumerate(inner):
        ytr, yva = y[itr], y[iva]
        if len(set(ytr)) < 2:
            continue
        pca = pcam.PCA().fit(X[itr])
        Ztr, Zva = pca.transform(X[itr]), pca.transform(X[iva])
        for gi, g in enumerate(gs):
            gamma = svm.gamma_from_g(g, Ztr) if kernel == "rbf" else None
            Ktr = svm.rbf_kernel(Ztr, Ztr, gamma) if kernel == "rbf" else svm.linear_kernel(Ztr, Ztr)
            Kva = svm.rbf_kernel(Zva, Ztr, gamma) if kernel == "rbf" else svm.linear_kernel(Zva, Ztr)
            for ci, C in enumerate(P["C"]):
                row = ci * len(gs) + gi
                if use_sklearn:
                    pred = SVC(C=C, kernel="precomputed", class_weight="balanced", tol=config.SMO_TOL).fit(Ktr, ytr).predict(Kva)
                else:
                    m = svm.SVM_QP(C=C, kernel=kernel, gamma=gamma or 1.0).fit_kernel(Ktr, ytr, Ztr)
                    pred = np.where(Kva[:, m.sv_mask_] @ m.coef_ + m.b_ >= 0, 1, -1)
                scores[row, k] = _ba(yva, pred)
    mean = np.nanmean(np.where(np.isnan(scores), np.nan, scores), axis=1) if np.isfinite(scores).any() else np.zeros(len(grid))
    mean = np.where(np.isnan(mean), -1.0, mean)
    C, g = grid[int(np.argmax(mean))]
    return C, (g if g is not None else 1.0)


def clf_metrics(y, dec):
    pred = np.where(dec >= 0, 1, -1)
    tp, fp = int(np.sum((pred == 1) & (y == 1))), int(np.sum((pred == 1) & (y == -1)))
    fn, tn = int(np.sum((pred == -1) & (y == 1))), int(np.sum((pred == -1) & (y == -1)))
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    f1p, f1n = 2 * tp / max(2 * tp + fp + fn, 1), 2 * tn / max(2 * tn + fn + fp, 1)
    auc = float(roc_auc_score(y, dec)) if len(set(y)) == 2 else np.nan
    return dict(balanced_accuracy=0.5 * (sens + spec), macro_f1=0.5 * (f1p + f1n), sensitivity=sens, specificity=spec, auc=auc, tp=tp, fp=fp, fn=fn, tn=tn)


def bias_diagnostic(alpha, y, K, Cvec):
    """Summarize free support vectors and the KKT-admissible bias interval."""
    alpha, y = np.asarray(alpha), np.asarray(y)
    f0 = K @ (alpha * y)
    free = (alpha > config.SV_THRESH * Cvec) & (alpha < (1.0 - config.SV_THRESH) * Cvec)
    score = -y * (y * f0 - 1.0)
    up, low = svm._sets(alpha, y, Cvec)
    lo = float(score[up].max()) if up.any() else np.nan
    hi = float(score[low].min()) if low.any() else np.nan
    finite_interval = np.isfinite(lo) and np.isfinite(hi)
    return dict(n_free=int(free.sum()), bias_lo=lo, bias_hi=hi,
                bias_width=float(hi - lo) if finite_interval and lo <= hi else np.nan,
                kkt_violation=float(lo - hi) if finite_interval else np.nan)


# ----------------------------------------------------------------- one outer fold
def run_fold(fo, feats, y, pids, sidx, conds, use_sk, P, agree_conds):
    tr, te = np.array(fo["train"]), np.array(fo["test"])
    inner = fo["inner"]
    ytr, yte = y[tr], y[te]
    oof, mets, agree, pcav = [], [], [], []

    def record(cond, prot, kernel, dec, C, g, k):
        m = clf_metrics(yte, dec)
        mets.append(dict(repeat=fo["repeat"], fold=fo["fold"], condition=cond, protocol=prot, model=kernel, n_train=len(tr), n_test=len(te), C=C, g=g, k_pca=k, **m))
        for s, d in zip(te, dec):
            oof.append((fo["repeat"], fo["fold"], int(sidx[s]), pids[s], cond, prot, kernel, int(y[s]), float(d), float(y[s] * d), 1 if d >= 0 else -1))

    maj = 1 if (ytr == 1).sum() >= (ytr == -1).sum() else -1
    mm = clf_metrics(yte, np.full(len(te), float(maj)))
    mets.append(dict(repeat=fo["repeat"], fold=fo["fold"], condition="majority", protocol="A", model="reference", n_train=len(tr), n_test=len(te), C=np.nan, g=np.nan, k_pca=0, **mm))
    for kernel in KERNELS:
        mod = {}
        for cond in conds:
            X = feats[cond]
            C, g = tune(X, y, tr, inner, kernel, use_sk, P)
            pca = pcam.PCA().fit(X[tr])
            Z = pca.transform(X[tr])
            gamma = svm.gamma_from_g(g, Z) if kernel == "rbf" else 1.0
            m = svm.SVM_QP(C=C, kernel=kernel, gamma=gamma).fit(Z, ytr)
            mod[cond] = (pca, m, C, g, gamma)
            record(cond, "A", kernel, m.decision_function(pca.transform(X[te])), C, g, pca.k_)
            if cond == "original" and kernel == "rbf" and fo["repeat"] == 0 and fo["fold"] == 0:
                cum = np.cumsum(pca.explained_variance_ratio_)
                pcav = [dict(k=i + 1, cum_var=float(c)) for i, c in enumerate(cum[:100])]
                pcav += [dict(k=-kk, cum_var=float(cum[min(kk, len(cum)) - 1])) for kk in config.PCA_SENS_K]   # negative k marks sensitivity rows
        pca_o, m_o, C_o, g_o, gam_o = mod["original"]
        for cond in conds:
            record(cond, "B", kernel, m_o.decision_function(pca_o.transform(feats[cond][te])), C_o, g_o, pca_o.k_)
        if kernel == "rbf":
            for cond in agree_conds:
                for prot in ("A", "B"):
                    pca, m_qp, C, g, gamma = mod[cond] if prot == "A" else mod["original"]
                    Z = pca.transform(feats["original" if prot == "B" else cond][tr])
                    Zte = pca.transform(feats[cond][te])
                    smo = svm.SVM_SMO(C=C, kernel="rbf", gamma=gamma).fit(Z, ytr)
                    a_svc, b_svc, sk = svm.svc_reference(Z, ytr, C, "rbf", gamma)
                    p_qp, p_smo, p_sk = m_qp.predict(Zte), smo.predict(Zte), sk.predict(Zte)
                    ba = lambda p: _ba(yte, p)
                    Ktrain = svm.rbf_kernel(Z, Z, gamma)
                    Cvec = svm.balanced_C(ytr, C)
                    qdiag = bias_diagnostic(m_qp.alpha_, ytr, Ktrain, Cvec)
                    sdiag = bias_diagnostic(smo.alpha_, ytr, Ktrain, Cvec)
                    cdiag = bias_diagnostic(a_svc, ytr, Ktrain, Cvec)
                    dual_qp = float(m_qp.dual_)
                    dual_smo = float(smo.dual_)
                    dual_svc = float(svm.dual_objective(a_svc, ytr, Ktrain))
                    agree.append(dict(repeat=fo["repeat"], fold=fo["fold"], condition=cond, protocol=prot, C=C, g=g, agree_qp_svc=float(np.mean(p_qp == p_sk)),
                                      agree_smo_svc=float(np.mean(p_smo == p_sk)), agree_qp_smo=float(np.mean(p_qp == p_smo)), ba_qp=ba(p_qp), ba_smo=ba(p_smo), ba_svc=ba(p_sk),
                                      dual_qp=dual_qp, dual_smo=dual_smo, dual_svc=dual_svc,
                                      rel_dual_qp_svc=abs(dual_qp - dual_svc) / abs(dual_svc),
                                      rel_dual_smo_svc=abs(dual_smo - dual_svc) / abs(dual_svc),
                                      smo_final_gap=smo.final_gap_,
                                      qp_status=m_qp.status_, svc_fit_status=int(sk.fit_status_),
                                      qp_stop_criterion_b_interval_width=m_qp.kkt_gap_,
                                      smo_stop_criterion_b_interval_width=smo.final_gap_,
                                      svc_stop_criterion_b_interval_width=svm.kkt_gap(a_svc, ytr, Ktrain, Cvec),
                                      qp_bias=m_qp.b_, smo_bias=smo.b_, svc_bias=b_svc,
                                      qp_n_free=qdiag["n_free"], qp_bias_lo=qdiag["bias_lo"], qp_bias_hi=qdiag["bias_hi"], qp_bias_width=qdiag["bias_width"], qp_kkt_violation=qdiag["kkt_violation"],
                                      smo_n_free=sdiag["n_free"], smo_bias_lo=sdiag["bias_lo"], smo_bias_hi=sdiag["bias_hi"], smo_bias_width=sdiag["bias_width"], smo_kkt_violation=sdiag["kkt_violation"],
                                      svc_n_free=cdiag["n_free"], svc_bias_lo=cdiag["bias_lo"], svc_bias_hi=cdiag["bias_hi"], svc_bias_width=cdiag["bias_width"], svc_kkt_violation=cdiag["kkt_violation"]))
    return oof, mets, agree, pcav


# ----------------------------------------------------------------- analysis
def soft_counts(oof, cond, prot, kernel):
    d = oof[(oof.condition == cond) & (oof.protocol == prot) & (oof.model == kernel)]
    g = d.groupby("slice_index").agg(pid=("pid", "first"), y=("y_true", "first"), pp=("pred", lambda p: float(np.mean(p == 1))))
    g["tp"], g["fn"] = (g.y == 1) * g.pp, (g.y == 1) * (1 - g.pp)
    g["fp"], g["tn"] = (g.y == -1) * g.pp, (g.y == -1) * (1 - g.pp)
    pc = g.groupby("pid")[["tp", "fp", "fn", "tn"]].sum()
    return pc


def primary_analysis(oof, fm, kernel="rbf"):
    rows = []
    for prot in ("A", "B"):
        for metric in ("balanced_accuracy", "macro_f1"):
            fam = []
            for a, b in itertools.combinations(PRIMARY_CONDS, 2):
                fa = fm[(fm.condition == a) & (fm.protocol == prot) & (fm.model == kernel)].sort_values(["repeat", "fold"])
                fb = fm[(fm.condition == b) & (fm.protocol == prot) & (fm.model == kernel)].sort_values(["repeat", "fold"])
                d = fa[metric].to_numpy() - fb[metric].to_numpy()
                nb = stats.nadeau_bengio_ttest(d, fa.n_train.mean(), fa.n_test.mean())
                ca, cb = soft_counts(oof, a, prot, kernel), soft_counts(oof, b, prot, kernel)
                pids = ca.index.intersection(cb.index)
                ci = stats.paired_diff_ci(ca.loc[pids].to_numpy(), cb.loc[pids].to_numpy(), metric, seed_keys=(41, len(rows)))
                fam.append(dict(protocol=prot, metric=metric, A=a, B=b, mean_diff_folds=nb["mean"], t=nb["t"], df=nb["df"], p_nb=nb["p"],
                                boot_point=ci["point"], ci_lo=ci["lo"], ci_hi=ci["hi"], n_patients=len(pids), J=len(d)))
            adj = stats.holm([r["p_nb"] for r in fam])
            for r, p in zip(fam, adj):
                r["p_holm"] = float(p)
                sig_p, sig_ci = p < config.ALPHA, not (r["ci_lo"] <= 0 <= r["ci_hi"])
                r["claim"] = "significant" if sig_p and sig_ci else ("inconclusive" if sig_p != sig_ci else "not significant")
            rows += fam
    return pd.DataFrame(rows)


def cell_table(oof, fm, conds, kernels):
    rows = []
    for cond in conds:
        for prot in ("A", "B"):
            for kernel in kernels:
                sub = fm[(fm.condition == cond) & (fm.protocol == prot) & (fm.model == kernel)]
                pc = soft_counts(oof, cond, prot, kernel)
                if sub.empty or pc.empty:
                    continue
                r = dict(condition=cond, protocol=prot, model=kernel, J=len(sub))
                for metric in ("balanced_accuracy", "macro_f1", "sensitivity", "specificity", "auc"):
                    r[f"{metric}_mean"], r[f"{metric}_std"] = float(sub[metric].mean()), float(sub[metric].std(ddof=1)) if len(sub) > 1 else np.nan
                ci = stats.cell_ci(pc.to_numpy(), "balanced_accuracy", seed_keys=(42, len(rows)))
                r["ba_boot_lo"], r["ba_boot_hi"] = ci["lo"], ci["hi"]
                rows.append(r)
    return pd.DataFrame(rows)


def association(oof, roi, conds_cells, kernel="rbf"):
    rows, pts = [], []
    for prot in ("A", "B"):
        for (m, R, snr) in conds_cells:
            if m == "M0":
                continue
            c, c0 = f"{m}_{common.cell_key(R, snr)}", f"M0_{common.cell_key(R, snr)}"
            def per_slice(cond):
                d = oof[(oof.condition == cond) & (oof.protocol == prot) & (oof.model == kernel)]
                return d.groupby("slice_index").agg(pid=("pid", "first"), margin=("margin", "mean"), correct=("margin", lambda v: float(np.mean(v > 0))))
            a, b = per_slice(c), per_slice(c0)
            idx = a.index.intersection(b.index)
            gain = np.array([roi[c0][i] - roi[c][i] for i in idx])          # ROI-NMSE improvement vs M0 (positive = better)
            dm, dc = (a.loc[idx].margin - b.loc[idx].margin).to_numpy(), (a.loc[idx].correct - b.loc[idx].correct).to_numpy()
            pid = a.loc[idx].pid.to_numpy()
            if len(idx) < 4 or np.std(gain) == 0:
                continue
            for outcome, v in (("margin_change", dm), ("correctness_change", dc)):
                if np.std(v) == 0:
                    continue
                r = stats.clustered_spearman(gain, v, pid, B=config.BOOT_B_EXPLORATORY, seed_keys=(43, len(rows)))
                rows.append(dict(method=m, R=R, snr=snr if math.isfinite(snr) else "inf", protocol=prot, outcome=outcome, family="exploratory", **r))
            pts += [dict(method=m, R=R, snr=snr if math.isfinite(snr) else "inf", protocol=prot, slice_index=int(i), roi_gain=float(g_), margin_change=float(x), correctness_change=float(z))
                    for i, g_, x, z in zip(idx, gain, dm, dc)]
    return pd.DataFrame(rows), pd.DataFrame(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=1)
    a = ap.parse_args()
    config.ensure_dirs()
    df, img, msk, splits = common.load_all()
    hyper = common.load_hyper()
    pool = splits["clf_pool"]["slice_index"]
    P = dict(outer=config.OUTER_FOLDS, repeats=config.OUTER_REPEATS, inner=config.INNER_FOLDS, C=config.C_GRID, g=config.G_GRID)
    Rs, snrs = config.R_LIST, config.SNR_LIST
    if a.quick:
        pool = common.pick_quick(df, pool)
        P.update(outer=2, repeats=1, inner=2, C=[1.0, 10.0], g=[1.0])
        Rs = [config.R_PRIMARY]
    sub = df.set_index("index").loc[pool]
    y, pids, groups, sidx = sub.y.to_numpy().astype(int), sub.pid.to_numpy(), sub.group.to_numpy(), np.array(pool)

    # 1. folds.json first
    folds = make_folds(y, groups, pids, P)
    common.write_json(config.RESULTS_DIR / "folds.json", dict(params={k: v for k, v in P.items()}, slice_index=[int(s) for s in sidx], folds=folds))

    # 2. reconstruct all pool slices once (cached) and build features / ROI-NMSE
    from src import metrics
    cells = [(m, R, s) for m in common.METHODS for R in Rs for s in snrs]
    jobs = [(i, "M0", 1, config.SNR_PRIMARY, {}) for i in pool] + [(i, m, R, s, common.params_for(hyper, m, R, s)) for i in pool for (m, R, s) in cells]
    res = Parallel(n_jobs=a.n_jobs)(delayed(lambda i, m, R, s, p: (common.reconstruct(img[i], i, m, R, s, p)[0]))(i, m, R, s, p) for i, m, R, s, p in jobs)
    feats, roi = {"original": np.stack([common.feature(img[i]) for i in pool])}, {}
    it = iter(res)
    feats["original_noisy"] = np.stack([common.feature(next(it)) for _ in pool])
    # per-condition lists in job order
    idx_res = {}
    for (i, m, R, s, p), x in zip(jobs[len(pool):], res[len(pool):]):
        idx_res[(i, common.cond_name(m, R, s))] = x
    for (m, R, s) in cells:
        cond = common.cond_name(m, R, s)
        feats[cond] = np.stack([common.feature(idx_res[(i, cond)]) for i in pool])
        roi[cond] = {i: metrics.roi_nmse(idx_res[(i, cond)], img[i], msk[i]) for i in pool}
    conds = list(feats)

    # 3. mandatory check 1: tuning consistency (original, repeat 0)
    cons, flag = [], config.TUNE_WITH_SKLEARN
    for fo in [f for f in folds if f["repeat"] == 0]:
        tr = np.array(fo["train"])
        c_sk = tune(feats["original"], y, tr, fo["inner"], "rbf", True, P)
        c_qp = tune(feats["original"], y, tr, fo["inner"], "rbf", False, P)
        gi = lambda v, grid: grid.index(v)
        cons.append(dict(fold=fo["fold"], sk=list(c_sk), qp=list(c_qp), exact=c_sk == c_qp,
                         within_one=abs(gi(c_sk[0], P["C"]) - gi(c_qp[0], P["C"])) <= 1 and abs(gi(c_sk[1], P["g"]) - gi(c_qp[1], P["g"])) <= 1))
    n_exact, n_one = sum(c["exact"] for c in cons), sum(c["within_one"] for c in cons)
    if n_one < len(cons) / 2:       # assumption: "most disagree" == fewer than half of the folds within one grid step
        flag = False
    common.write_json(config.RESULTS_DIR / "e4_tuning_consistency.json", dict(folds=cons, n_folds=len(cons), n_exact=n_exact, n_within_one_step=n_one,
                                                                              TUNE_WITH_SKLEARN_used=flag, rule="flag False if fewer than half of the folds are within one grid step"))
    print(f"tuning consistency: exact {n_exact}/{len(cons)}, within one step {n_one}/{len(cons)}; TUNE_WITH_SKLEARN={flag}", flush=True)

    # 4. the grid (+ mandatory check 2, solver agreement, at the primary R for both SNRs)
    agree_conds = [c for c in conds if c.startswith(("M0_", "M1_", "M2_")) and f"R{config.R_PRIMARY}_" in c]
    out = Parallel(n_jobs=a.n_jobs)(delayed(run_fold)(fo, feats, y, pids, sidx, conds, flag, P, agree_conds) for fo in folds)
    oof = pd.DataFrame([r for o in out for r in o[0]], columns=["repeat", "fold", "slice_index", "pid", "condition", "protocol", "model", "y_true", "decision", "margin", "pred"])
    raw_meta = pd.read_csv(config.PROC_DIR / "raw_meta.csv", dtype={"id": str})
    raw_meta["raw_pos"] = np.arange(len(raw_meta))
    index_to_raw_pos = df.set_index("index")["raw_pos"]
    raw_pos_to_id = raw_meta.set_index("raw_pos")["id"]
    oof.insert(2, "id", oof.slice_index.map(index_to_raw_pos).map(raw_pos_to_id))
    if oof.id.isna().any():
        raise RuntimeError("could not map every E4 OOF slice to its raw .mat filename stem")
    fm = pd.DataFrame([r for o in out for r in o[1]])
    ag = pd.DataFrame([r for o in out for r in o[2]])
    pv = [r for o in out for r in o[3]]
    oof.to_csv(config.RESULTS_DIR / "e4_oof.csv", index=False)
    fm.to_csv(config.RESULTS_DIR / "e4_fold_metrics.csv", index=False)
    ag.to_csv(config.RESULTS_DIR / "e4_solver_agreement.csv", index=False)
    pd.DataFrame(pv).to_csv(config.RESULTS_DIR / "e4_pca_variance.csv", index=False)
    cell = cell_table(oof, fm, conds, KERNELS)
    cell.to_csv(config.RESULTS_DIR / "e4_summary.csv", index=False)
    prim = primary_analysis(oof, fm)
    prim.to_csv(config.RESULTS_DIR / "e4_primary.csv", index=False)
    conf = fm[fm.model.isin(["rbf", "reference"])].groupby(["condition", "protocol", "model"], as_index=False)[["tp", "fp", "fn", "tn"]].sum()
    conf.to_csv(config.RESULTS_DIR / "e4_confusion.csv", index=False)
    asso, pts = association(oof, roi, cells)
    asso.to_csv(config.RESULTS_DIR / "e4_association.csv", index=False)
    pts.to_csv(config.RESULTS_DIR / "e4_association_points.csv", index=False)
    print(prim.to_string(index=False))
    if len(ag):
        print(ag.groupby("protocol")[["agree_qp_svc", "agree_smo_svc", "agree_qp_smo"]].mean())
    print("majority-class reference BA:", fm[fm.condition == "majority"].balanced_accuracy.mean())


if __name__ == "__main__":
    main()
