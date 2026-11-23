"""Every table as CSV + LaTeX, and results/headline.csv, from results/*.csv and hyperparams.json only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config

R = config.RESULTS_DIR


def esc(s):
    return str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else (f"{v:.4g}")
    return esc(v)


def to_latex(df):
    cols = "l" * len(df.columns)
    lines = [f"\\begin{{tabular}}{{{cols}}}", "\\hline", " & ".join(esc(c) for c in df.columns) + " \\\\", "\\hline"]
    lines += [" & ".join(fmt(v) for v in row) + " \\\\" for row in df.itertuples(index=False)]
    return "\n".join(lines + ["\\hline", "\\end{tabular}"]) + "\n"


def save(name, df):
    d = R / "tables"
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / f"{name}.csv", index=False)
    (d / f"{name}.tex").write_text(to_latex(df))
    print("table", name, df.shape)


def have(name):
    return (R / name).exists()


def solver_agreement_metrics(a):
    """Normalize legacy solver diagnostics and add relative dual-objective differences."""
    a = a.copy()
    for solver in ("qp", "smo", "svc"):
        old, new = f"{solver}_kkt_gap", f"{solver}_stop_criterion_b_interval_width"
        if old in a and new not in a:
            a = a.rename(columns={old: new})
        lo, hi = f"{solver}_bias_lo", f"{solver}_bias_hi"
        if lo in a and hi in a:
            valid = np.isfinite(a[lo]) & np.isfinite(a[hi])
            a[f"{solver}_bias_width"] = np.where(valid & (a[lo] <= a[hi]), a[hi] - a[lo], np.nan)
            a[f"{solver}_kkt_violation"] = np.where(valid, a[lo] - a[hi], np.nan)
    for solver in ("qp", "smo"):
        col = f"rel_dual_{solver}_svc"
        if col not in a and {f"dual_{solver}", "dual_svc"} <= set(a.columns):
            a[col] = np.abs(a[f"dual_{solver}"] - a.dual_svc) / np.abs(a.dual_svc)
    return a


def main():
    config.ensure_dirs()
    head = []
    if have("attrition.csv"):
        save("table_attrition", pd.read_csv(R / "attrition.csv"))
    if have("roi_filter_audit.csv"):
        save("table_roi_filter", pd.read_csv(R / "roi_filter_audit.csv"))
    if have("e1_summary.csv"):
        s = pd.read_csv(R / "e1_summary.csv")
        s["value"] = [f"{m:.4g} [{lo:.4g}, {hi:.4g}]" for m, lo, hi in zip(s["mean"], s.ci_lo, s.ci_hi)]
        wide = s.pivot_table(index=["method", "R", "snr"], columns="metric", values="value", aggfunc="first").reset_index()
        save("table_recon_metrics", wide)
    if have("e1_stats.csv"):
        st = pd.read_csv(R / "e1_stats.csv")
        save("table_recon_stats_primary", st[st.family == "primary"])
        save("table_recon_stats_exploratory", st[st.family == "exploratory"])
        for r in st[st.family == "primary"].itertuples():
            head.append(dict(endpoint=f"E1 {r.metric} {r.A} - {r.B} (R={r.R}, SNR={r.snr})", value=r.mean_diff, lo=r.ci_lo, hi=r.ci_hi, p_adjusted=r.p_holm, note="Wilcoxon, Holm (primary family)"))
    if have("validation_summary.csv"):
        save("table_validation", pd.read_csv(R / "validation_summary.csv"))
    if have("e3_summary.csv"):
        save("table_e3_summary", pd.read_csv(R / "e3_summary.csv"))
    if have("e2a1_summary.csv"):
        save("table_e2a1_kappa", pd.read_csv(R / "e2a1_summary.csv"))
    if have("e4_primary.csv"):
        p = pd.read_csv(R / "e4_primary.csv")
        save("table_e4_primary", p)
        for r in p.itertuples():
            head.append(dict(endpoint=f"E4 {r.metric} {r.A} - {r.B} (protocol {r.protocol}, RBF)", value=r.mean_diff_folds, lo=r.ci_lo, hi=r.ci_hi, p_adjusted=r.p_holm, note=f"claim: {r.claim}"))
    if have("e4_summary.csv"):
        s = pd.read_csv(R / "e4_summary.csv")
        save("table_e4_cells", s)
        for r in s[(s.model == "rbf") & s.condition.isin(["original", "original_noisy"] + [f"{m}_R{config.R_PRIMARY}_SNR{int(config.SNR_PRIMARY)}" for m in ("M0", "M1", "M2")])].itertuples():
            head.append(dict(endpoint=f"E4 balanced accuracy {r.condition} (protocol {r.protocol})", value=r.balanced_accuracy_mean, lo=r.ba_boot_lo, hi=r.ba_boot_hi, p_adjusted=np.nan, note="mean over folds; patient-clustered bootstrap CI"))
    if have("e4_solver_agreement.csv"):
        a = solver_agreement_metrics(pd.read_csv(R / "e4_solver_agreement.csv"))
        metrics = ["agree_qp_svc", "agree_smo_svc", "agree_qp_smo", "rel_dual_qp_svc", "rel_dual_smo_svc"]
        metrics += [f"{solver}_{name}" for solver in ("qp", "smo", "svc")
                    for name in ("stop_criterion_b_interval_width", "bias_width", "kkt_violation")]
        metrics = [col for col in metrics if col in a]
        save("table_e4_solver_agreement", a.groupby(["protocol"], as_index=False)[metrics].mean())
    if have("e5_scaling.csv"):
        e = pd.read_csv(R / "e5_scaling.csv")
        if "kkt_gap" in e and "libsvm_stop_criterion_b_interval_width" not in e:
            e = e.rename(columns={"kkt_gap": "libsvm_stop_criterion_b_interval_width"})
        cols = ["solve_time", "dual", "n_sv", "libsvm_stop_criterion_b_interval_width", "accuracy", "balanced_accuracy"]
        save("table_e5_scaling", e.groupby(["n", "kernel", "solver"], as_index=False)[[c for c in cols if c in e]].mean())
    if have("hyperparameters.csv"):
        save("table_hyperparameters", pd.read_csv(R / "hyperparameters.csv"))
    if have("limitations.csv"):
        save("table_limitations", pd.read_csv(R / "limitations.csv"))
    pd.DataFrame(head).to_csv(R / "headline.csv", index=False)
    print("headline.csv rows:", len(head))


if __name__ == "__main__":
    main()
