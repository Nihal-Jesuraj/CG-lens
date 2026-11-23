"""Hand-written inference: Nadeau-Bengio corrected resampled t, patient-clustered bootstrap, Holm."""
import warnings

import numpy as np
from scipy import stats as sps

import config
from src import mri_ops


def nadeau_bengio_ttest(d, n_train, n_test):
    """Paired fold differences d_j (J of them); var = (1/J + n_test/n_train) s_d^2; t with J-1 df."""
    d = np.asarray(d, dtype=np.float64)
    J = len(d)
    mean, s2 = float(d.mean()), float(d.var(ddof=1))
    var = (1.0 / J + n_test / n_train) * s2
    if var == 0.0:
        t = 0.0 if mean == 0 else float(np.sign(mean) * np.inf)
    else:
        t = mean / np.sqrt(var)
    p = 1.0 if t == 0 else float(2.0 * sps.t.sf(abs(t), J - 1))
    return dict(mean=mean, t=float(t), df=J - 1, p=p, var=var)


def holm(p):
    """Holm-Bonferroni adjusted p-values (monotone step-down), original order."""
    p = np.asarray(p, dtype=np.float64)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def clustered_bootstrap(per_cluster, stat_fn, B=None, seed_keys=(1,), alpha=None, chunk=1000):
    """Resample clusters with replacement. per_cluster (P, k); stat_fn maps summed (b, k) -> (b,). Percentile CI."""
    B = config.BOOT_B if B is None else B
    alpha = config.ALPHA if alpha is None else alpha
    per_cluster = np.asarray(per_cluster, dtype=np.float64)
    P = len(per_cluster)
    rng = mri_ops.seed_rng(*seed_keys)
    out = []
    for s in range(0, B, chunk):
        idx = rng.integers(0, P, size=(min(chunk, B - s), P))
        out.append(stat_fn(per_cluster[idx].sum(axis=1)))
    dist = np.concatenate(out)
    point = float(stat_fn(per_cluster.sum(axis=0, keepdims=True))[0])
    lo, hi = np.percentile(dist, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(point=point, lo=float(lo), hi=float(hi), dist=dist)


def mean_diff_ci(d, B=None, seed_keys=(2,)):
    """Patient-level paired mean difference with bootstrap CI (E1)."""
    d = np.asarray(d, dtype=np.float64)
    pc = np.stack([d, np.ones_like(d)], axis=1)
    return clustered_bootstrap(pc, lambda s: s[:, 0] / s[:, 1], B, seed_keys)


# ---- classification statistics from per-cluster soft confusion counts [tp, fp, fn, tn] ----
def ba_from_counts(S):
    S = np.atleast_2d(S)
    tp, fp, fn, tn = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
    return 0.5 * (tp / np.maximum(tp + fn, 1e-300) + tn / np.maximum(tn + fp, 1e-300))


def macro_f1_from_counts(S):
    S = np.atleast_2d(S)
    tp, fp, fn, tn = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
    f1p = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-300)
    f1n = 2 * tn / np.maximum(2 * tn + fn + fp, 1e-300)
    return 0.5 * (f1p + f1n)


METRIC_FNS = {"balanced_accuracy": ba_from_counts, "macro_f1": macro_f1_from_counts}


def paired_diff_ci(counts_a, counts_b, metric, B=None, seed_keys=(3,)):
    """Clustered-bootstrap CI for metric(a) - metric(b); both (P, 4) per-patient soft counts, same resamples."""
    fn = METRIC_FNS[metric]
    return clustered_bootstrap(np.hstack([counts_a, counts_b]), lambda S: fn(S[:, :4]) - fn(S[:, 4:]), B, seed_keys)


def cell_ci(counts, metric, B=None, seed_keys=(4,)):
    return clustered_bootstrap(counts, METRIC_FNS[metric], B, seed_keys)


def clustered_spearman(x, y, clusters, B=None, seed_keys=(5,), alpha=None):
    """Spearman rho with a patient-clustered bootstrap percentile CI."""
    B = config.BOOT_B if B is None else B
    alpha = config.ALPHA if alpha is None else alpha
    x, y, clusters = np.asarray(x, float), np.asarray(y, float), np.asarray(clusters)
    uniq = np.unique(clusters)
    members = [np.flatnonzero(clusters == c) for c in uniq]
    rho = float(sps.spearmanr(x, y).statistic)
    rng = mri_ops.seed_rng(*seed_keys)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        ix = np.concatenate([members[k] for k in pick])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # constant resamples give NaN and are skipped
            r = sps.spearmanr(x[ix], y[ix]).statistic
        if np.isfinite(r):
            vals.append(r)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if vals else (np.nan, np.nan)
    return dict(rho=rho, lo=float(lo), hi=float(hi), n=len(x), n_clusters=len(uniq))
