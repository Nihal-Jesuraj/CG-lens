"""Figures 1-12 as vector PDF + 300-dpi PNG, drawn only from results/ CSV/NPZ files (and deterministic masks)."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src import common

R_ = config.RESULTS_DIR
FIG = R_ / "figures"
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 300, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
COL = {"M0": "#7f7f7f", "M1": "#1f77b4", "M2": "#d62728", "original": "#2ca02c", "original_noisy": "#9467bd"}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("figure", name)


def need(*files):
    miss = [f for f in files if not (R_ / f).exists()]
    if miss:
        print("  skipped (missing):", miss)
    return not miss


def fig1():
    pipeline = pd.read_csv(R_ / "pipeline.csv").sort_values("order")
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.axis("off")
    boxes = pipeline.stage.tolist()
    for i, b in enumerate(boxes):
        x = i / len(boxes)
        ax.text(x + 0.06, 0.5, b, ha="center", va="center", bbox=dict(boxstyle="round", fc="#eef3fa", ec="#335"), transform=ax.transAxes, fontsize=8)
        if i:
            ax.annotate("", xy=(x + 0.005, 0.5), xytext=(x - 0.03, 0.5), xycoords="axes fraction", arrowprops=dict(arrowstyle="->"))
    save(fig, "fig01_pipeline")


def fig2():
    masks = pd.read_csv(R_ / "mask_patterns.csv")
    fig, axs = plt.subplots(1, len(config.R_LIST), figsize=(10, 2.6))
    for ax, R in zip(axs, config.R_LIST):
        d = masks[masks.R == R].sort_values(["pixel_y", "pixel_x"])
        M = d.sampled.to_numpy().reshape(config.N, config.N)
        ax.imshow(np.fft.fftshift(M), cmap="gray")
        ax.set_title(f"R = {R} (achieved {config.N ** 2 / M.sum():.2f})")
        ax.axis("off")
    save(fig, "fig02_masks")


def fig3():
    examples = pd.read_csv(R_ / "e1_examples.csv")
    ks = sorted(examples["sample"].unique())
    fig, axs = plt.subplots(2 * len(ks), 4, figsize=(8, 4 * len(ks)))
    axs = np.atleast_2d(axs)
    for r, k in enumerate(ks):
        d = examples[examples["sample"] == k].sort_values(["pixel_y", "pixel_x"])
        x = d.truth.to_numpy().reshape(config.N, config.N)
        mk = d["mask"].to_numpy().reshape(config.N, config.N)
        axs[2 * r, 0].imshow(x, cmap="gray", vmin=0, vmax=1)
        axs[2 * r, 0].contour(mk, levels=[0.5], colors="y", linewidths=0.6)
        axs[2 * r, 0].set_title("truth + tumor mask")
        axs[2 * r + 1, 0].axis("off")
        for c, m in enumerate(common.METHODS, start=1):
            xh = d[m].to_numpy().reshape(config.N, config.N)
            axs[2 * r, c].imshow(xh, cmap="gray", vmin=0, vmax=1)
            axs[2 * r, c].set_title(m)
            e = axs[2 * r + 1, c].imshow(np.abs(xh - x), cmap="magma", vmin=0, vmax=0.3)
            axs[2 * r + 1, c].contour(mk, levels=[0.5], colors="c", linewidths=0.6)
            axs[2 * r + 1, c].set_title(f"|error|  ROI NMSE={np.sum((xh - x)[mk > 0] ** 2) / np.sum(x[mk > 0] ** 2):.3f}", fontsize=7)
    for a in axs.ravel():
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    save(fig, "fig03_examples")


def fig4():
    s = pd.read_csv(R_ / "e1_summary.csv")
    s["snr"] = s.snr.astype(str)
    snrs = sorted(s.snr.unique())
    fig, axs = plt.subplots(len(snrs), 3, figsize=(10, 3 * len(snrs)), squeeze=False)
    for r, snr in enumerate(snrs):
        for c, metric in enumerate(["psnr", "ssim", "roi_nmse"]):
            for m in common.METHODS:
                d = s[(s.snr == snr) & (s.metric == metric) & (s.method == m)].sort_values("R")
                axs[r, c].errorbar(d.R, d["mean"], yerr=[d["mean"] - d.ci_lo, d.ci_hi - d["mean"]], marker="o", ms=3, capsize=2, color=COL[m], label=m)
            axs[r, c].set_title(f"{metric} (SNR {snr})")
            axs[r, c].set_xlabel("R")
    axs[0, 0].legend()
    save(fig, "fig04_metrics_vs_R")


def fig5():
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    c = pd.read_csv(R_ / "e2a1_curves.csv")
    for k, (mu, g) in enumerate(c.groupby("mu")):
        axs[0].semilogy(g.k, g.anorm_rel, color=f"C{k}", label=f"mu={mu:g}")
        axs[0].semilogy(g.k, g.bound, "--", color=f"C{k}", lw=0.8)
    axs[0].set_title("32x32: A-norm error (solid) and kappa bound (dashed)"); axs[0].set_xlabel("iteration"); axs[0].legend(fontsize=6)
    if need("e2a2_cg.csv"):
        d = pd.read_csv(R_ / "e2a2_cg.csv")
        axs[1].semilogy(d.k, d.anorm_rel); axs[1].set_title("128x128: A-norm error (no bound drawn)")
        axs[2].semilogy(d.k, d.resid_rel); axs[2].set_title("128x128: 2-norm residual")
    for a in axs:
        a.set_xlabel("iteration")
    save(fig, "fig05_cg_convergence")


def fig6():
    d = pd.read_csv(R_ / "e2b_irls.csv")
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    for i, g in d.groupby("slice_index"):
        axs[0].plot(g.outer, g.J, label=f"slice {i} (violations={int(g.violations_total.iloc[0])})")
        axs[1].plot(g.outer, g.inner_iters)
    axs[0].set_title("J_mu(x_k) per outer iteration"); axs[1].set_title("warm-started inner CG iterations")
    axs[0].legend(fontsize=6)
    for a in axs:
        a.set_xlabel("outer iteration")
    save(fig, "fig06_irls_descent")


def fig7():
    d = pd.read_csv(R_ / "e3_results.csv")
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    for lam, g in d.groupby("lam"):
        m = g.groupby("mu")[["rel_diff", "J0_gap"]].median()
        axs[0].loglog(m.index, m.rel_diff, "o-", label=f"lam={lam:g}")
        axs[1].loglog(m.index, np.maximum(m.J0_gap, 1e-16), "o-")
    axs[0].set_title("relative difference IRLS vs QP"); axs[1].set_title("J0 gap")
    axs[0].legend(fontsize=6)
    for a in axs:
        a.set_xlabel("mu")
    save(fig, "fig07_irls_vs_qp")


def fig8():
    d = pd.read_csv(R_ / "e4_pca_variance.csv")
    main_ = d[d.k > 0]
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(main_.k, main_.cum_var)
    ax.axhline(config.PCA_VAR, ls="--", color="k", lw=0.7)
    for _, r in d[d.k < 0].iterrows():
        ax.plot(-r.k, r.cum_var, "ro", ms=3)
    ax.set_xlabel("components"); ax.set_ylabel("cumulative explained variance"); ax.set_title("PCA, original, fold (0,0); red = k in {10,25,50}")
    save(fig, "fig08_pca_variance")


def fig9():
    s = pd.read_csv(R_ / "e4_summary.csv")
    s = s[s.model == "rbf"]
    rx = re.compile(r"^(M\d)_R(\d+)_SNR(\w+)$")
    parsed = s.condition.map(lambda c: rx.match(c).groups() if rx.match(c) else None)
    s = s.assign(method=parsed.map(lambda g: g[0] if g else None), R=parsed.map(lambda g: int(g[1]) if g else np.nan), snr=parsed.map(lambda g: g[2] if g else None))
    snrs = sorted(s.dropna(subset=["snr"]).snr.unique())
    fig, axs = plt.subplots(2, len(snrs), figsize=(4.5 * len(snrs), 6), squeeze=False)
    for r, prot in enumerate(["A", "B"]):
        for c, snr in enumerate(snrs):
            for m in common.METHODS:
                d = s[(s.method == m) & (s.snr == snr) & (s.protocol == prot)].sort_values("R")
                axs[r, c].errorbar(d.R, d.balanced_accuracy_mean, yerr=[d.balanced_accuracy_mean - d.ba_boot_lo, d.ba_boot_hi - d.balanced_accuracy_mean], marker="o", ms=3, capsize=2, color=COL[m], label=m)
            o = s[(s.condition == "original") & (s.protocol == prot)]
            if len(o):
                axs[r, c].axhline(o.balanced_accuracy_mean.iloc[0], color=COL["original"], ls="--", label="original")
            axs[r, c].set_title(f"Protocol {prot}, SNR {snr}"); axs[r, c].set_xlabel("R"); axs[r, c].set_ylabel("balanced accuracy")
    axs[0, 0].legend(fontsize=7)
    save(fig, "fig09_ba_vs_R")


def fig10():
    e = pd.read_csv(R_ / "e5_scaling.csv")
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    for solver, col in (("qp", "C0"), ("smo", "C1"), ("svc", "C2")):
        g = e[(e.solver == solver) & (e.kernel == "rbf")].groupby("n")[["solve_time", "n_sv"]].mean()
        axs[0].loglog(g.index, g.solve_time, "o-", color=col, label=solver)
        axs[1].plot(g.index, g.n_sv, "o-", color=col, label=solver)
    axs[0].set_title("RBF solve time (serial)"); axs[1].set_title("support vectors")
    axs[0].legend()
    for a in axs:
        a.set_xlabel("n")
    save(fig, "fig10_svm_scaling")


def fig11():
    c = pd.read_csv(R_ / "e4_confusion.csv")
    conds = ["original"] + [f"{m}_R{config.R_PRIMARY}_SNR{int(config.SNR_PRIMARY)}" for m in common.METHODS]
    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    for r, prot in enumerate(["A", "B"]):
        for k, cond in enumerate(conds):
            row = c[(c.condition == cond) & (c.protocol == prot) & (c.model == "rbf")]
            if row.empty:
                axs[r, k].axis("off"); continue
            row = row.iloc[0]
            cm = np.array([[row.tp, row.fn], [row.fp, row.tn]])
            axs[r, k].imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                axs[r, k].text(j, i, int(v), ha="center", va="center")
            axs[r, k].set_xticks([0, 1], ["pred +", "pred -"]); axs[r, k].set_yticks([0, 1], ["true +", "true -"]); axs[r, k].grid(False)
            axs[r, k].set_title(f"{cond}\nprotocol {prot}", fontsize=7)
    save(fig, "fig11_confusion")


def fig12():
    p = pd.read_csv(R_ / "e4_association_points.csv")
    p = p[(p.R == config.R_PRIMARY) & (p.snr.astype(str).isin([str(config.SNR_PRIMARY), "25", "25.0"])) & (p.protocol == "A")]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for m, g in p.groupby("method"):
        ax.scatter(g.roi_gain, g.margin_change, s=8, alpha=0.6, color=COL[m], label=m)
    ax.set_xlabel("ROI-NMSE improvement vs M0"); ax.set_ylabel("signed-margin change vs M0"); ax.set_title("exploratory (R=4, 25 dB, protocol A)")
    ax.legend()
    save(fig, "fig12_gain_vs_margin")


def main():
    config.ensure_dirs()
    fig1()
    if need("mask_patterns.csv"):
        fig2()
    for fn, files in ((fig3, ["e1_examples.csv"]), (fig4, ["e1_summary.csv"]), (fig5, ["e2a1_curves.csv"]), (fig6, ["e2b_irls.csv"]), (fig7, ["e3_results.csv"]),
                      (fig8, ["e4_pca_variance.csv"]), (fig9, ["e4_summary.csv"]), (fig10, ["e5_scaling.csv"]), (fig11, ["e4_confusion.csv"]), (fig12, ["e4_association_points.csv"])):
        if need(*files):
            fn()


if __name__ == "__main__":
    main()
