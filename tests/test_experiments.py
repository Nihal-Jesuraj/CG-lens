import numpy as np
import pandas as pd

import config
from experiments import e1_recon, e4_diag, e7_lambda, e8_crime
from src import data
from src import mri_ops as mo


def test_discrepancy_bisection_root():
    res = lambda lam: 2.0 * np.sqrt(lam)            # increasing in lambda
    eps = 0.37
    out = e7_lambda.discrepancy_lambda(res, eps, tol=1e-6)
    assert out["bracketed"] and abs(res(out["lam"]) - eps) <= 1e-6 * eps
    assert not e7_lambda.discrepancy_lambda(res, 1e9)["bracketed"]


def test_noise_estimate_scale():
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    x = gaussian_filter(rng.random((64, 64)), 4)
    x = (x - x.min()) / (x.max() - x.min())
    M, y = mo.simulate(x, 2, 25.0, 3)
    eps_true = np.linalg.norm(y - mo.A(x, M))
    assert 0.5 < e7_lambda.estimate_noise_norm(y, M) / eps_true < 2.0


def test_e8_mean_intensity_and_symmetry():
    rng = np.random.default_rng(1)
    from scipy.ndimage import gaussian_filter
    x256 = gaussian_filter(rng.random((256, 256)), 8)
    x256 = (x256 - x256.min()) / (x256.max() - x256.min())
    x128 = data.area_resize(x256, (128, 128))
    c = e8_crime.hires_spectrum(x256)
    assert np.array_equal(c, np.conj(mo.partner(c)))
    rec = mo.ifft2c(c)                                   # noise-free, fully sampled
    assert np.abs(rec.imag).max() < 1e-12 and abs(rec.real.mean() - x128.mean()) < 1e-12
    assert 10 * np.log10(1 / np.mean((rec.real - x128) ** 2)) > 30


def test_folds_patient_disjoint_and_complete():
    rng = np.random.default_rng(2)
    n = 60
    pids = np.repeat(np.arange(30), 2).astype(str)
    groups = np.repeat(np.arange(30) // 2, 2)           # pairs of patients share a group
    y = np.repeat(np.where(rng.random(30) < 0.5, 1, -1), 2)
    P = dict(outer=5, repeats=2, inner=3)
    folds = e4_diag.make_folds(y, groups, pids, P)
    assert len(folds) == 10
    for fo in folds:
        assert not set(pids[fo["train"]]) & set(pids[fo["test"]])
        for itr, iva in fo["inner"]:
            assert not set(groups[itr]) & set(groups[iva]) and set(itr) | set(iva) == set(fo["train"])
    assert folds == e4_diag.make_folds(y, groups, pids, P)         # deterministic => identical for every condition


def test_bias_diagnostic_reports_empty_interval_as_violation(monkeypatch):
    y = np.array([1.0, -1.0])
    alpha = np.zeros(2)
    K = np.zeros((2, 2))
    Cvec = np.ones(2)
    monkeypatch.setattr(e4_diag.svm, "_sets", lambda a, yy, C: (np.array([True, False]), np.array([False, True])))
    empty = e4_diag.bias_diagnostic(alpha, y, K, Cvec)
    assert empty["bias_lo"] > empty["bias_hi"]
    assert np.isnan(empty["bias_width"])
    assert empty["kkt_violation"] == 2.0

    monkeypatch.setattr(e4_diag.svm, "_sets", lambda a, yy, C: (np.array([False, True]), np.array([True, False])))
    feasible = e4_diag.bias_diagnostic(alpha, y, K, Cvec)
    assert feasible["bias_width"] == 2.0
    assert feasible["kkt_violation"] == -2.0


def test_e1_primary_family_is_six_tests_with_holm():
    rng = np.random.default_rng(3)
    rows = []
    for pid in range(12):
        for m, off in (("M0", 0.0), ("M1", 1.0), ("M2", 2.0)):
            for R in (4, 8):
                for snr in ("25.0", "inf"):
                    for metric, sign in (("psnr", 1), ("ssim", 1), ("nmse", -1), ("roi_nmse", -1)):
                        rows.append(dict(pid=str(pid), method=m, R=R, snr=snr, metric=metric, value=30 + sign * off + rng.normal(0, 0.3)))
    st = e1_recon.analyse(pd.DataFrame(rows))
    prim = st[st.family == "primary"]
    assert len(prim) == 6 and set(prim.metric) == {"psnr", "roi_nmse"} and set(prim.R) == {4} and set(prim.snr) == {"25.0"}
    assert (prim.p_holm >= prim.wilcoxon_p - 1e-15).all()
    assert prim.improves.all()
