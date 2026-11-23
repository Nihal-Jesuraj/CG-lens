import numpy as np
from scipy.special import betainc

from src import stats as st


def test_nadeau_bengio_hand_example():
    d = [1, 2, 3, 4, 5]
    r = st.nadeau_bengio_ttest(d, n_train=80, n_test=20)
    # mean 3, s^2 = 2.5, var = (1/5 + 20/80) * 2.5 = 1.125, t = 3/sqrt(1.125)
    assert abs(r["mean"] - 3.0) < 1e-12 and abs(r["var"] - 1.125) < 1e-12
    assert abs(r["t"] - 3.0 / np.sqrt(1.125)) < 1e-10 and r["df"] == 4
    p_beta = betainc(2.0, 0.5, 4.0 / (4.0 + r["t"] ** 2))      # independent route: two-sided t tail via incomplete beta
    assert abs(r["p"] - p_beta) < 1e-10
    assert st.nadeau_bengio_ttest([0.1] * 5, 80, 20)["p"] == 0.0     # zero variance, nonzero mean
    assert st.nadeau_bengio_ttest([0, 0, 0], 8, 2)["p"] == 1.0


def test_holm_hand_example():
    adj = st.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06], atol=1e-12)
    assert np.allclose(st.holm([0.5, 0.5]), [1.0, 1.0])
    assert np.allclose(st.holm([0.001]), [0.001])
    a = st.holm([0.2, 0.01, 0.9, 0.04])        # sorted 0.01*4, 0.04*3, 0.2*2, 0.9*1 -> 0.04, 0.12, 0.4, 0.9
    assert np.allclose(a, [0.4, 0.04, 0.9, 0.12], atol=1e-12)


def test_bootstrap_two_clusters_exact():
    # clusters with values 1 and 3: resampled means are 1, 2, 2, 3 with prob 1/4, 1/2, 1/4
    pc = np.array([[1.0, 1.0], [3.0, 1.0]])
    r = st.clustered_bootstrap(pc, lambda s: s[:, 0] / s[:, 1], B=10000, seed_keys=(1,))
    assert abs(r["point"] - 2.0) < 1e-10 and abs(r["lo"] - 1.0) < 1e-10 and abs(r["hi"] - 3.0) < 1e-10
    ident = st.clustered_bootstrap(np.tile([[2.0, 1.0]], (6, 1)), lambda s: s[:, 0] / s[:, 1], B=500)
    assert ident["lo"] == ident["hi"] == ident["point"] == 2.0


def test_bootstrap_vectorised_equals_loop():
    rng = np.random.default_rng(0)
    pc = rng.random((12, 3))
    fn = lambda s: s[:, 0] / s[:, 1] - s[:, 2]
    r = st.clustered_bootstrap(pc, fn, B=2500, seed_keys=(9,), chunk=1000)
    gen = st.mri_ops.seed_rng(9)
    ref = []
    for s in range(0, 2500, 1000):
        idx = gen.integers(0, 12, size=(min(1000, 2500 - s), 12))
        ref += [fn(pc[i].sum(axis=0, keepdims=True))[0] for i in idx]
    assert np.abs(np.array(ref) - r["dist"]).max() < 1e-12


def test_classification_stats_and_paired_ci():
    S = np.array([[8, 1, 2, 9.0]])                                 # tp, fp, fn, tn
    assert abs(st.ba_from_counts(S)[0] - 0.5 * (8 / 10 + 9 / 10)) < 1e-12
    f1p, f1n = 2 * 8 / (16 + 1 + 2), 2 * 9 / (18 + 2 + 1)
    assert abs(st.macro_f1_from_counts(S)[0] - 0.5 * (f1p + f1n)) < 1e-12
    a = np.tile([[4.0, 0, 1, 5]], (10, 1))
    b = np.tile([[3.0, 1, 2, 4]], (10, 1))
    r = st.paired_diff_ci(a, b, "balanced_accuracy", B=300)
    assert abs(r["point"] - (st.ba_from_counts(a.sum(0))[0] - st.ba_from_counts(b.sum(0))[0])) < 1e-12 and r["lo"] == r["hi"]


def test_clustered_spearman_perfect():
    x = np.arange(20.0)
    r = st.clustered_spearman(x, 2 * x + 1, np.repeat(np.arange(10), 2), B=200)
    assert abs(r["rho"] - 1) < 1e-12 and r["lo"] > 0.9


def test_mean_diff_ci():
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r = st.mean_diff_ci(d, B=2000)
    assert abs(r["point"] - 3.0) < 1e-12 and r["lo"] < 3 < r["hi"]
