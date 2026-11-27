"""Paper-specific versions of figures 1, 4, 5, 7 and 8 used in report/CG_Lens_Project_Report.tex.

Only presentation is changed (labels, legend placement, axis limits, pipeline
layout); every number comes from the same results/ CSV files that
scripts/make_figures.py reads.  Output goes to report/figures/ so the source
figures in results/figures/ stay untouched.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import config
from src import common

R_ = config.RESULTS_DIR
OUT = Path(__file__).resolve().parent / "figures"
plt.rcParams.update(
    {
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.grid": False,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "font.family": "DejaVu Sans",
    }
)
COL = {"M0": "#7f7f7f", "M1": "#1f77b4", "M2": "#d62728", "original": "#2ca02c"}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("figure", name)


def fig1():
    labels = [
        "T1-CE slice\n+ tumor mask",
        "undersampled\nnoisy $k$-space",
        "M0 / M1 / M2\nreconstruction",
        "PSNR, SSIM\nROI-NMSE",
        "64$\\times$64 features\n+ PCA",
        "SVM QP / SMO\npatient-disjoint",
        "fidelity vs\ndiagnosis",
    ]
    fig, ax = plt.subplots(figsize=(7.16, 1.10))
    ax.set_xlim(0, 7.16)
    ax.set_ylim(0, 1.10)
    ax.axis("off")
    w, gap, h, y = 0.90, 0.105, 0.70, 0.20
    total = len(labels) * w + (len(labels) - 1) * gap
    x = (7.16 - total) / 2
    boxes = []
    for label in labels:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.06",
            linewidth=0.8,
            edgecolor="#33415c",
            facecolor="#eef3fa",
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=5.8,
            color="#111827",
            linespacing=1.25,
        )
        boxes.append(x)
        x += w + gap
    for left, right in zip(boxes[:-1], boxes[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (left + w + 0.018, y + h / 2),
                (right - 0.018, y + h / 2),
                arrowstyle="-|>",
                mutation_scale=7,
                linewidth=0.8,
                color="#33415c",
                shrinkA=0,
                shrinkB=0,
            )
        )
    save(fig, "fig01_pipeline")


def fig2():
    masks = pd.read_csv(R_ / "mask_patterns.csv")
    pca = pd.read_csv(R_ / "e4_pca_variance.csv")
    fig = plt.figure(figsize=(7.16, 1.32))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.40], wspace=0.28)
    for i, R in enumerate(config.R_LIST):
        ax = fig.add_subplot(gs[0, i])
        d = masks[masks.R == R].sort_values(["pixel_y", "pixel_x"])
        M = d.sampled.to_numpy().reshape(config.N, config.N)
        ax.imshow(np.fft.fftshift(M), cmap="gray", interpolation="nearest")
        ax.set_title(f"$R$ = {R}", fontsize=8, pad=2)
        ax.axis("off")
    axp = fig.add_subplot(gs[0, 4])
    main_ = pca[pca.k > 0]
    axp.plot(main_.k, main_.cum_var, lw=1.1)
    axp.axhline(config.PCA_VAR, ls="--", color="k", lw=0.6)
    ref = pca[pca.k < 0].copy()
    ref["k"] = -ref.k
    axp.plot(ref.k, ref.cum_var, "o", color="#d62728", ms=3.5, label="$k$ = 10, 25, 50")
    axp.set_xlabel("components", fontsize=6, labelpad=1)
    axp.set_title("PCA (fold 0,0)", fontsize=8, pad=2)
    axp.set_ylim(0.1, 0.97)
    axp.tick_params(labelsize=5.6, length=2)
    axp.legend(fontsize=5.6, loc="lower right", frameon=False)
    save(fig, "fig02_masks_pca")


def fig4():
    s = pd.read_csv(R_ / "e1_summary.csv")
    s["snr"] = s.snr.astype(str)
    rows = [("25.0", "25 dB"), ("inf", "noise-free")]
    cols = [("psnr", "PSNR"), ("ssim", "SSIM"), ("roi_nmse", "ROI-NMSE")]
    fig, axs = plt.subplots(2, 3, figsize=(10, 5.9), squeeze=False)
    for r, (snr, snr_label) in enumerate(rows):
        for c, (metric, metric_label) in enumerate(cols):
            ax = axs[r, c]
            for m in common.METHODS:
                d = s[(s.snr == snr) & (s.metric == metric) & (s.method == m)].sort_values("R")
                ax.errorbar(
                    d.R,
                    d["mean"],
                    yerr=[d["mean"] - d.ci_lo, d.ci_hi - d["mean"]],
                    marker="o",
                    ms=3,
                    lw=1.1,
                    capsize=2,
                    color=COL[m],
                    label=m,
                )
            ax.set_title(f"{metric_label} ({snr_label})")
            ax.set_xlabel("R")
            ax.set_xticks([2, 4, 6, 8])
            if c == 0:
                ax.set_ylabel("PSNR (dB)" if metric == "psnr" else "score")
    axs[0, 0].legend(fontsize=7, framealpha=0.9)
    fig.tight_layout(h_pad=1.6)
    save(fig, "fig04_metrics_vs_R")


def fig5():
    c = pd.read_csv(R_ / "e2a1_curves.csv")
    d2 = pd.read_csv(R_ / "e2a2_cg.csv")
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.0))
    for k, (mu, g) in enumerate(c.groupby("mu")):
        axs[0].semilogy(g.k, g.anorm_rel, color=f"C{k}", lw=1.1, label=f"$\\mu$ = {mu:g}")
        axs[0].semilogy(g.k, g.bound, "--", color=f"C{k}", lw=0.7)
    axs[0].set_title("$32\\times32$: energy-norm error")
    axs[0].set_xlabel("iteration")
    axs[0].set_ylabel("relative energy-norm error")
    axs[0].legend(fontsize=6, loc="upper right", framealpha=0.9)
    axs[1].semilogy(d2.k, d2.anorm_rel, lw=1.1)
    axs[1].set_title("$128\\times128$: energy-norm error")
    axs[1].set_xlabel("iteration")
    axs[1].set_ylabel("relative energy-norm error")
    axs[2].semilogy(d2.k, d2.resid_rel, lw=1.1)
    axs[2].set_title("$128\\times128$: residual")
    axs[2].set_xlabel("iteration")
    axs[2].set_ylabel("relative residual")
    fig.tight_layout()
    save(fig, "fig05_cg_convergence")


def fig6():
    d = pd.read_csv(R_ / "e2b_irls.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.30))
    for sl, col in [(15, "#1f77b4"), (19, "#ff7f0e")]:
        g = d[d.slice_index == sl].sort_values("outer")
        axs[0].plot(g.outer, g.J, color=col, lw=1.0, label=f"slice {sl}")
        axs[1].plot(g.outer, g.inner_iters, color=col, lw=1.0, label=f"slice {sl}")
    axs[0].set_xlabel("outer iteration")
    axs[0].set_ylabel(r"$J_\mu(x_k)$")
    axs[0].set_title("smoothed objective", fontsize=8.5, pad=3)
    axs[1].set_xlabel("outer iteration")
    axs[1].set_ylabel("inner CG iterations")
    axs[1].set_title("warm-started inner CG", fontsize=8.5, pad=3)
    for ax in axs:
        ax.legend(fontsize=6.5, frameon=False, loc="best")
        ax.tick_params(labelsize=6.5)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    save(fig, "fig06_irls_descent")


def fig7():
    d = pd.read_csv(R_ / "e3_results.csv")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.0))
    for k, (lam, g) in enumerate(d.groupby("lam")):
        m = g.groupby("mu")[["rel_diff", "J0_gap"]].median()
        axs[0].loglog(m.index, m.rel_diff, "o-", ms=3, color=f"C{k}", label=f"$\\lambda$ = {lam:g}")
        axs[1].loglog(m.index, np.maximum(m.J0_gap, 1e-16), "o-", ms=3, color=f"C{k}")
    for ax, title in zip(axs, ["relative image difference (IRLS vs QP)", "$J_0$ gap"]):
        ax.set_title(title)
        ax.set_xlabel("$\\mu$")
    axs[0].set_ylabel("relative difference")
    axs[1].set_ylabel("gap")
    axs[0].legend(fontsize=7, framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig07_irls_vs_qp")


def fig9():
    s = pd.read_csv(R_ / "e4_summary.csv")
    s = s[s.model == "rbf"]
    rx = re.compile(r"^(M\d)_R(\d+)_SNR(\w+)$")

    def parse(c):
        m = rx.match(c)
        return m.groups() if m else (None, None, None)

    parsed = s.condition.map(parse)
    s = s.assign(
        method=parsed.map(lambda g: g[0]),
        R=pd.to_numeric(parsed.map(lambda g: g[1])),
        snr=parsed.map(lambda g: g[2]),
    )
    panels = [("A", "25", "25 dB"), ("A", "inf", "noise-free"), ("B", "25", "25 dB"), ("B", "inf", "noise-free")]
    fig, axs = plt.subplots(2, 2, figsize=(9, 5.8), squeeze=False)
    for idx, (prot, snr, snr_label) in enumerate(panels):
        ax = axs[idx // 2, idx % 2]
        lo_all, hi_all = [], []
        for m in common.METHODS:
            d = s[(s.method == m) & (s.snr == snr) & (s.protocol == prot)].sort_values("R")
            yerr = [d.balanced_accuracy_mean - d.ba_boot_lo, d.ba_boot_hi - d.balanced_accuracy_mean]
            ax.errorbar(
                d.R,
                d.balanced_accuracy_mean,
                yerr=yerr,
                marker="o",
                ms=3,
                lw=1.1,
                capsize=1.8,
                color=COL[m],
                label=m,
                alpha=0.95,
            )
            lo_all.extend(d.ba_boot_lo)
            hi_all.extend(d.ba_boot_hi)
        o = s[(s.condition == "original") & (s.protocol == prot)]
        if len(o):
            ax.axhline(o.balanced_accuracy_mean.iloc[0], color=COL["original"], ls="--", lw=0.9, label="original")
        pad = 0.006
        ax.set_ylim(min(lo_all) - pad, max(hi_all) + pad)
        ax.set_title(f"Protocol {prot}, {snr_label}")
        ax.set_xlabel("R")
        ax.set_xticks([2, 4, 6, 8])
        if idx % 2 == 0:
            ax.set_ylabel("balanced accuracy")
    axs[0, 0].legend(fontsize=6.5, loc="upper left", framealpha=0.9)
    fig.tight_layout(h_pad=1.6)
    save(fig, "fig09_ba_vs_R")


def fig10():
    e = pd.read_csv(R_ / "e5_scaling.csv")
    fig, axs = plt.subplots(2, 1, figsize=(3.4, 2.95))
    for solver, col in (("qp", "#1f77b4"), ("smo", "#ff7f0e"), ("svc", "#2ca02c")):
        g = e[(e.solver == solver) & (e.kernel == "rbf")].groupby("n")[["solve_time", "n_sv"]].mean()
        axs[0].loglog(g.index, g.solve_time, "o-", ms=3.0, lw=1.0, color=col, label=solver)
        axs[1].plot(g.index, g.n_sv, "o-", ms=3.0, lw=1.0, color=col, label=solver)
    axs[0].set_title("RBF solve time (serial)", fontsize=8, pad=2)
    axs[1].set_title("support-vector count", fontsize=8, pad=2)
    axs[0].set_ylabel("time (s)", fontsize=7)
    axs[1].set_ylabel("support vectors", fontsize=7)
    for a in axs:
        a.set_xlabel("$n$", fontsize=7)
        a.tick_params(labelsize=6)
    axs[0].legend(fontsize=6, frameon=False)
    fig.tight_layout(pad=0.3, h_pad=0.8)
    save(fig, "fig10_svm_scaling")


def fig12():
    p = pd.read_csv(R_ / "e4_association_points.csv")
    p = p[(p.R == config.R_PRIMARY) & (p.snr.astype(str).isin([str(config.SNR_PRIMARY), "25", "25.0"])) & (p.protocol == "A")]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for m, g in p.groupby("method"):
        ax.scatter(g.roi_gain, g.margin_change, s=6, alpha=0.6, color=COL[m], label=m)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("ROI-NMSE improvement vs M0", fontsize=7.5)
    ax.set_ylabel("signed-margin change vs M0", fontsize=7.5)
    ax.set_title("$R$ = 4, 25 dB, Protocol A", fontsize=8.5, pad=3)
    ax.legend(fontsize=6.5, frameon=False)
    ax.tick_params(labelsize=6.5)
    fig.tight_layout(pad=0.3)
    save(fig, "fig12_gain_vs_margin")


def figconf():
    c = pd.read_csv(R_ / "e4_confusion.csv")
    conds = [("original", "original")] + [
        (f"{m}_R{config.R_PRIMARY}_SNR{int(config.SNR_PRIMARY)}", m) for m in common.METHODS
    ]
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 2.9))
    for r, prot in enumerate(["A", "B"]):
        for k, (cond, short) in enumerate(conds):
            ax = axs[r, k]
            row = c[(c.condition == cond) & (c.protocol == prot) & (c.model == "rbf")]
            if row.empty:
                ax.axis("off")
                continue
            row = row.iloc[0]
            cm = np.array([[row.tp, row.fn], [row.fp, row.tn]])
            ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, int(v), ha="center", va="center", fontsize=7)
            ax.set_xticks([0, 1], ["pred +", "pred -"])
            ax.set_yticks([0, 1], ["true +", "true -"])
            ax.tick_params(labelsize=5.4, length=1.5, pad=1)
            ax.grid(False)
            ax.set_title(f"{short} ({prot})", fontsize=7, pad=2)
    fig.tight_layout(pad=0.5, w_pad=0.4, h_pad=1.0)
    save(fig, "fig11_confusion")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "4", "5", "6", "7", "9", "10", "12", "conf"]
    for name in which:
        if name == "conf":
            figconf()
        else:
            globals()[f"fig{name}"]()
