"""
Where do MINT and Boltz-2 work, and where do they fail?

Cross-joins the existing per-entry data:
  - MINT 3khmvobe rerun (analysis/mint_3khmvobe_rerun.csv) — refreshed numbers
  - Boltz-2 with-MSA + Boltz-2 no-MSA P@K (from msa_depth_vs_pak.csv)
  - Per-chain MSA depth, length, homo/hetero (from msa_depth_vs_pak.csv)
  - Inter-chain contact density (computed on the fly from each entry's GT .pt)

Produces (under analysis/figures/):
  fig01_pak_vs_depth_2dhist.png    2D histograms of P@K vs log10 MSA depth per model
  fig02_cdfs.png                   CDFs of P@K per model (overlay)
  fig03_stratified_by_length.png   P@K vs L_min, faceted by model
  fig04_stratified_by_homo.png     homo vs hetero comparison per model
  fig05_stratified_by_density.png  P@K vs interface contact density
  fig06_head2head_mint_vs_boltz.png  joint scatter w/ marginals, colored by depth
  fig07_winloss_per_feature.png    bar chart: who wins by feature stratum
  fig08_difficulty_profile.png     entries sorted by mean P@K; feature heatmap
  fig09_msa_gain_drivers.png       what features predict Boltz-2's MSA gain?
  fig10_quadrant_analysis.png      MINT/Boltz quadrants: when do they agree/disagree?

Also writes analysis/figures/findings.md with key takeaways.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

GT_ROOT = Path("/projects/u6jv/public/MINT/DATA/pdb/processed/data")


def load_data() -> pd.DataFrame:
    base = pd.read_csv(HERE / "msa_depth_vs_pak.csv")
    rerun = pd.read_csv(HERE / "mint_3khmvobe_rerun.csv")
    rerun = rerun.rename(
        columns={
            "AUC": "mint_auc",
            "P@K": "mint_pak",
            "P@K/2": "mint_pak2",
            "P@K/5": "mint_pak5",
            "L": "L_pred",
            "K": "n_true_inter",
        }
    )
    df = base.merge(
        rerun[["entry_id", "mint_auc", "mint_pak", "mint_pak2", "mint_pak5", "n_true_inter"]],
        on="entry_id",
        how="inner",
    )
    df["boltz_msa_pak"] = df["pak_boltz_msa"]
    df["boltz_nomsa_pak"] = df["pak_boltz_nomsa"]
    df["msa_gain"] = df["boltz_msa_pak"] - df["boltz_nomsa_pak"]

    df["L_a"] = df["L_total"] - df["L_min"]  # the longer chain
    df["L_max"] = df.apply(lambda r: max(r["L_min"], r["L_total"] - r["L_min"]), axis=1)
    df["L_asym"] = (df["L_max"] - df["L_min"]) / df["L_max"]
    df["interface_density"] = df["n_true_inter"] / (df["L_min"] * df["L_a"]).clip(lower=1)
    df["log_density"] = np.log10(df["interface_density"].clip(lower=1e-6))
    return df


# ---------------------------------------------------------------------------
# Figure 1: 2D histograms of P@K vs MSA depth
# ---------------------------------------------------------------------------

def fig_2dhist(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    x = df["log_depth_min"].values
    models = [
        ("mint_pak", "MINT (3khmvobe, no MSA)", "viridis"),
        ("boltz_msa_pak", "Boltz-2 + MSA", "viridis"),
        ("boltz_nomsa_pak", "Boltz-2, single-seq", "viridis"),
    ]
    for ax, (col, lbl, cmap) in zip(axes, models):
        y = df[col].values
        # hexbin gives smoother density than a square histogram
        hb = ax.hexbin(x, y, gridsize=(28, 22), cmap=cmap, mincnt=1, bins="log")
        ax.set_xlabel("log10 (MSA depth, min over chains)")
        ax.set_title(lbl)
        cb = plt.colorbar(hb, ax=ax, label="entries (log scale)")
        # rolling median overlay
        order = np.argsort(x)
        win = max(31, len(df) // 25) | 1
        med = pd.Series(y[order]).rolling(win, center=True, min_periods=10).median().values
        ax.plot(x[order], med, color="white", lw=2.0, alpha=0.9, label="rolling median")
        ax.plot(x[order], med, color="red", lw=1.2, alpha=0.9)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("inter-chain P@K")
    fig.suptitle(f"P@K vs MSA depth (2-D histograms, n={len(df)} dimers)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig01_pak_vs_depth_2dhist.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: CDFs
# ---------------------------------------------------------------------------

def fig_cdfs(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for col, lbl, color in [
        ("mint_pak", "MINT (no MSA)", "tab:green"),
        ("boltz_nomsa_pak", "Boltz-2 single-seq", "tab:orange"),
        ("boltz_msa_pak", "Boltz-2 +MSA", "tab:blue"),
    ]:
        v = np.sort(df[col].values)
        axes[0].step(v, np.arange(len(v)) / len(v), where="post", lw=2, color=color, label=lbl)
    axes[0].set_xlabel("inter-chain P@K")
    axes[0].set_ylabel("CDF (fraction of entries ≤ x)")
    axes[0].set_title("CDFs across all 1473 dimers")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower right")
    axes[0].set_xlim(-0.02, 1.02)

    # Right panel: same CDFs but log scale on probability axis -> tail visibility
    for col, lbl, color in [
        ("mint_pak", "MINT", "tab:green"),
        ("boltz_nomsa_pak", "Boltz-2 -MSA", "tab:orange"),
        ("boltz_msa_pak", "Boltz-2 +MSA", "tab:blue"),
    ]:
        v = np.sort(df[col].values)
        # plot survival (1-CDF) on log-y
        surv = 1 - np.arange(len(v)) / len(v)
        axes[1].step(v, surv, where="post", lw=2, color=color, label=lbl)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("inter-chain P@K")
    axes[1].set_ylabel("survival = P(P@K ≥ x)  (log scale)")
    axes[1].set_title("Tail: how often does each model do well?")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(loc="upper right")
    axes[1].set_xlim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig02_cdfs.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Stratified by chain length
# ---------------------------------------------------------------------------

def _boxstrip(ax, groups, labels, colors, ylabel, title):
    positions = np.arange(len(groups))
    bp = ax.boxplot(groups, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color("black")
    # overlay strip
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        if len(g) == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(g))
        ax.scatter(i + jitter, g, s=3, alpha=0.15, color=colors[i], edgecolors="none")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(-0.02, 1.02)


def fig_by_length(df: pd.DataFrame) -> None:
    df = df.copy()
    df["L_bin"] = pd.cut(
        df["L_min"],
        bins=[0, 100, 150, 200, 300, 1000],
        labels=["<100", "100-150", "150-200", "200-300", ">300"],
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (col, lbl, color) in zip(
        axes,
        [
            ("mint_pak", "MINT (no MSA)", "tab:green"),
            ("boltz_msa_pak", "Boltz-2 + MSA", "tab:blue"),
            ("boltz_nomsa_pak", "Boltz-2 single-seq", "tab:orange"),
        ],
    ):
        groups = [df.loc[df["L_bin"] == b, col].values for b in df["L_bin"].cat.categories]
        n_per_bin = [len(g) for g in groups]
        _boxstrip(
            ax, groups,
            [f"{b}\n(n={n})" for b, n in zip(df['L_bin'].cat.categories, n_per_bin)],
            [color] * 5, "inter-chain P@K", lbl,
        )
    axes[0].set_xlabel("min-chain length (residues)")
    fig.suptitle("P@K stratified by chain length (shorter chain of the dimer)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig03_stratified_by_length.png", dpi=130)
    plt.close(fig)


def fig_by_homo(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    models = [
        ("mint_pak", "MINT", "tab:green"),
        ("boltz_msa_pak", "Boltz-2 +MSA", "tab:blue"),
        ("boltz_nomsa_pak", "Boltz-2 -MSA", "tab:orange"),
    ]
    positions = np.arange(len(models))
    width = 0.36
    for j, (homo_val, offset, label) in enumerate(
        [(True, -width / 2, "homo"), (False, +width / 2, "hetero")]
    ):
        sub = df[df["is_homo"] == homo_val]
        means = [sub[col].mean() for col, _, _ in models]
        ses = [sub[col].std(ddof=1) / np.sqrt(len(sub)) for col, _, _ in models]
        bars = ax.bar(positions + offset, means, width=width, yerr=ses, capsize=4,
                      label=f"{label} (n={len(sub)})", alpha=0.85,
                      color=["#666"] * len(models))
        for bar, color in zip(bars, [c for _, _, c in models]):
            bar.set_color(color)
            bar.set_alpha(0.5 if j == 0 else 0.85)
    ax.set_xticks(positions)
    ax.set_xticklabels([lbl for _, lbl, _ in models])
    ax.set_ylabel("mean inter-chain P@K (± SE)")
    ax.set_title("Homo- vs hetero-dimer performance")
    ax.legend(title="")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig04_stratified_by_homo.png", dpi=130)
    plt.close(fig)


def fig_by_density(df: pd.DataFrame) -> None:
    df = df.copy()
    # Build density quintiles
    df["dens_bin"] = pd.qcut(
        df["interface_density"], q=5,
        labels=["Q1 (sparse iface)", "Q2", "Q3", "Q4", "Q5 (dense iface)"],
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (col, lbl, color) in zip(
        axes,
        [
            ("mint_pak", "MINT", "tab:green"),
            ("boltz_msa_pak", "Boltz-2 +MSA", "tab:blue"),
            ("boltz_nomsa_pak", "Boltz-2 -MSA", "tab:orange"),
        ],
    ):
        groups = [df.loc[df["dens_bin"] == b, col].values for b in df["dens_bin"].cat.categories]
        ns = [len(g) for g in groups]
        _boxstrip(
            ax, groups,
            [f"{b}\n(n={n})" for b, n in zip(df['dens_bin'].cat.categories, ns)],
            [color] * 5, "inter-chain P@K", lbl,
        )
    axes[0].set_xlabel("interface contact density (true inter-contacts / (L_min × L_max))")
    fig.suptitle("P@K vs interface size (dense interfaces should be easier)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig05_stratified_by_density.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: MINT vs Boltz head-to-head with marginals
# ---------------------------------------------------------------------------

def fig_head2head(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(11, 9))
    gs = GridSpec(4, 4, figure=fig)
    ax_main = fig.add_subplot(gs[1:, :3])
    ax_top = fig.add_subplot(gs[0, :3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

    color_by = df["log_depth_min"].values
    sc = ax_main.scatter(
        df["mint_pak"], df["boltz_msa_pak"],
        c=color_by, s=12, alpha=0.6, cmap="viridis", edgecolors="none",
    )
    ax_main.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax_main.set_xlabel("MINT inter P@K")
    ax_main.set_ylabel("Boltz-2 +MSA inter P@K")
    ax_main.set_xlim(-0.02, 1.02)
    ax_main.set_ylim(-0.02, 1.02)
    ax_main.grid(alpha=0.3)
    cax = fig.add_axes([0.92, 0.12, 0.015, 0.6])
    cbar = plt.colorbar(sc, cax=cax, label="log10 (MSA depth, min chain)")

    # Marginals
    bins = np.linspace(0, 1, 41)
    ax_top.hist(df["mint_pak"], bins=bins, color="tab:green", alpha=0.7)
    ax_top.set_ylabel("MINT count")
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(alpha=0.3)
    ax_right.hist(df["boltz_msa_pak"], bins=bins, color="tab:blue", alpha=0.7,
                  orientation="horizontal")
    ax_right.set_xlabel("Boltz count")
    ax_right.tick_params(labelleft=False)
    ax_right.grid(alpha=0.3)

    # Quadrant counts (split at 0.3)
    thr = 0.3
    q1 = ((df["mint_pak"] >= thr) & (df["boltz_msa_pak"] >= thr)).sum()
    q2 = ((df["mint_pak"] < thr) & (df["boltz_msa_pak"] >= thr)).sum()
    q3 = ((df["mint_pak"] < thr) & (df["boltz_msa_pak"] < thr)).sum()
    q4 = ((df["mint_pak"] >= thr) & (df["boltz_msa_pak"] < thr)).sum()
    ax_main.axvline(thr, color="grey", lw=0.5, ls=":")
    ax_main.axhline(thr, color="grey", lw=0.5, ls=":")
    for x, y, txt in [
        (0.65, 0.95, f"both good\n{q1} ({q1/len(df):.0%})"),
        (0.05, 0.95, f"Boltz only\n{q2} ({q2/len(df):.0%})"),
        (0.05, 0.05, f"both fail\n{q3} ({q3/len(df):.0%})"),
        (0.65, 0.05, f"MINT only\n{q4} ({q4/len(df):.0%})"),
    ]:
        ax_main.text(x, y, txt, fontsize=10, ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.9))
    ax_main.set_title(f"Head-to-head: MINT vs Boltz-2 +MSA  (threshold P@K = {thr})")
    fig.tight_layout(rect=(0, 0, 0.91, 1))
    fig.savefig(OUT / "fig06_head2head_mint_vs_boltz.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: Win/loss bar chart per feature stratum
# ---------------------------------------------------------------------------

def fig_winloss(df: pd.DataFrame) -> None:
    df = df.copy()
    df["depth_bin"] = pd.qcut(df["n_seqs_min"], q=4, labels=["depth Q1", "Q2", "Q3", "Q4"])
    df["len_bin"] = pd.cut(df["L_min"], bins=[0, 150, 250, 1000], labels=["L<150", "L 150-250", "L>250"])
    df["dens_bin"] = pd.qcut(df["interface_density"], q=3, labels=["sparse iface", "mid", "dense iface"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (group_col, title) in zip(
        axes.ravel(),
        [
            ("depth_bin", "by MSA depth quartile"),
            ("len_bin", "by min-chain length"),
            ("dens_bin", "by interface density"),
            ("is_homo", "homo vs hetero"),
        ],
    ):
        if group_col == "is_homo":
            groups = [(False, "hetero"), (True, "homo")]
            labels = [g[1] for g in groups]
            subs = [df[df["is_homo"] == g[0]] for g in groups]
        else:
            cats = df[group_col].cat.categories
            labels = list(cats)
            subs = [df[df[group_col] == c] for c in cats]
        x = np.arange(len(labels))
        w = 0.27
        for j, (col, lbl, color) in enumerate(
            [
                ("mint_pak", "MINT", "tab:green"),
                ("boltz_msa_pak", "Boltz-2 +MSA", "tab:blue"),
                ("boltz_nomsa_pak", "Boltz-2 -MSA", "tab:orange"),
            ]
        ):
            means = [s[col].mean() for s in subs]
            ses = [s[col].std(ddof=1) / np.sqrt(max(1, len(s))) for s in subs]
            ax.bar(x + (j - 1) * w, means, width=w, yerr=ses, capsize=3, color=color, alpha=0.8, label=lbl)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l}\n(n={len(s)})" for l, s in zip(labels, subs)])
        ax.set_title(title)
        ax.set_ylabel("mean P@K")
        ax.set_ylim(0, 0.9)
        ax.grid(alpha=0.3, axis="y")
        if ax is axes[0, 0]:
            ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("Mean P@K stratified by features", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig07_winloss_per_feature.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8: Difficulty profile (entries sorted by model-averaged P@K)
# ---------------------------------------------------------------------------

def fig_difficulty(df: pd.DataFrame) -> None:
    df = df.copy()
    df["mean_pak"] = df[["mint_pak", "boltz_msa_pak", "boltz_nomsa_pak"]].mean(axis=1)
    df_sorted = df.sort_values("mean_pak").reset_index(drop=True)
    x = np.arange(len(df_sorted))

    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.15)
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(x, df_sorted["mint_pak"], lw=0.8, color="tab:green", label="MINT")
    ax0.plot(x, df_sorted["boltz_nomsa_pak"], lw=0.8, color="tab:orange", label="Boltz-2 -MSA")
    ax0.plot(x, df_sorted["boltz_msa_pak"], lw=0.8, color="tab:blue", label="Boltz-2 +MSA")
    ax0.set_ylabel("inter P@K")
    ax0.set_title("Entries sorted by mean P@K across all three models (easy → hard, left → right)")
    ax0.set_ylim(-0.02, 1.02)
    ax0.grid(alpha=0.3)
    ax0.legend(loc="upper left")
    ax0.set_xlim(0, len(df_sorted))

    # Feature strips along the same x
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(x, np.log10(df_sorted["n_seqs_min"].clip(lower=1)), lw=0.6, color="navy")
    ax1.set_ylabel("log10 MSA depth")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax2.plot(x, df_sorted["L_min"], lw=0.6, color="darkred")
    ax2.set_ylabel("L_min")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax3.plot(x, df_sorted["interface_density"], lw=0.6, color="purple")
    ax3.set_ylabel("iface density")
    ax3.set_xlabel(f"entry rank by mean P@K  (n={len(df_sorted)})")
    ax3.grid(alpha=0.3)

    fig.savefig(OUT / "fig08_difficulty_profile.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9: What features predict Boltz-2's MSA gain?
# ---------------------------------------------------------------------------

def fig_msa_gain(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plots = [
        ("log_depth_min", "log10 MSA depth (min chain)"),
        ("L_min", "L_min (shorter chain)"),
        ("interface_density", "interface contact density"),
        ("log_density", "log10 interface density"),
    ]
    for ax, (col, lbl) in zip(axes.ravel(), plots):
        x = df[col].values
        y = df["msa_gain"].values
        hb = ax.hexbin(x, y, gridsize=(28, 22), cmap="magma", mincnt=1, bins="log")
        order = np.argsort(x)
        win = max(31, len(df) // 25) | 1
        med = pd.Series(y[order]).rolling(win, center=True, min_periods=10).median().values
        ax.plot(x[order], med, color="cyan", lw=2.0, label="rolling median")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(lbl)
        ax.set_ylabel("ΔP@K (Boltz +MSA − -MSA)")
        ax.legend(loc="lower right", fontsize=9)
        plt.colorbar(hb, ax=ax, label="entries (log)")
        ax.grid(alpha=0.2)
        # spearman
        r, p = stats.spearmanr(x, y)
        ax.set_title(f"{lbl}    ρ={r:+.2f}, p={p:.0e}")
    fig.suptitle("What predicts Boltz-2's MSA gain?", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig09_msa_gain_drivers.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 10: Quadrant analysis with feature heatmap
# ---------------------------------------------------------------------------

def fig_quadrants(df: pd.DataFrame) -> None:
    thr = 0.3
    df = df.copy()
    df["q_boltz"] = (df["boltz_msa_pak"] >= thr)
    df["q_mint"] = (df["mint_pak"] >= thr)
    df["quadrant"] = pd.Categorical(
        df.apply(
            lambda r: (
                "both good" if r["q_mint"] and r["q_boltz"]
                else "Boltz only" if r["q_boltz"]
                else "MINT only" if r["q_mint"]
                else "both fail"
            ),
            axis=1,
        ),
        categories=["both good", "Boltz only", "MINT only", "both fail"],
        ordered=True,
    )

    features = [
        ("log_depth_min", "log10 MSA depth"),
        ("L_min", "L_min (res)"),
        ("L_asym", "chain asymmetry"),
        ("interface_density", "iface density"),
        ("is_homo", "homo fraction"),
    ]
    quadrants = list(df["quadrant"].cat.categories)
    # build matrix: median per quadrant per feature; standardize within row for heatmap
    rows = []
    for col, _ in features:
        if col == "is_homo":
            vals = [df.loc[df["quadrant"] == q, col].astype(float).mean() for q in quadrants]
        else:
            vals = [df.loc[df["quadrant"] == q, col].median() for q in quadrants]
        rows.append(vals)
    rows = np.array(rows)
    # row-z-score for the heatmap colormap
    row_mean = rows.mean(axis=1, keepdims=True)
    row_std = rows.std(axis=1, keepdims=True) + 1e-9
    z = (rows - row_mean) / row_std

    fig = plt.figure(figsize=(13, 6))
    gs = GridSpec(1, 2, width_ratios=[1, 1.4], wspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    counts = df["quadrant"].value_counts().reindex(quadrants)
    colors = {"both good": "#2ca02c", "Boltz only": "#1f77b4", "MINT only": "#9467bd", "both fail": "#7f7f7f"}
    ax1.bar(range(len(quadrants)), counts.values, color=[colors[q] for q in quadrants])
    ax1.set_xticks(range(len(quadrants)))
    ax1.set_xticklabels(quadrants)
    for i, v in enumerate(counts.values):
        ax1.text(i, v + 5, f"{v} ({v/len(df):.0%})", ha="center", fontsize=10)
    ax1.set_ylabel("entries")
    ax1.set_title(f"Quadrants by P@K ≥ {thr}")
    ax1.grid(alpha=0.3, axis="y")

    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(z, cmap="RdBu_r", aspect="auto", vmin=-1.5, vmax=1.5)
    ax2.set_yticks(range(len(features)))
    ax2.set_yticklabels([f for _, f in features])
    ax2.set_xticks(range(len(quadrants)))
    ax2.set_xticklabels(quadrants)
    ax2.set_title("Feature profile per quadrant (row z-score; raw values labeled)")
    for i in range(len(features)):
        for j in range(len(quadrants)):
            raw = rows[i, j]
            txt = f"{raw:.2f}" if abs(raw) < 100 else f"{raw:.0f}"
            ax2.text(j, i, txt, ha="center", va="center", fontsize=10,
                     color="white" if abs(z[i, j]) > 0.8 else "black")
    plt.colorbar(im, ax=ax2, label="row z-score")
    fig.tight_layout()
    fig.savefig(OUT / "fig10_quadrant_analysis.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Findings markdown
# ---------------------------------------------------------------------------

def write_findings(df: pd.DataFrame) -> None:
    lines = []
    lines.append("# Where MINT and Boltz-2 work, where they don't")
    lines.append("")
    lines.append(f"n = {len(df)} dimers from mint_seqid30 val split, after joining MINT 3khmvobe rerun + Boltz-2 ±MSA per-entry P@K + chain MSA depth.")
    lines.append("")
    lines.append("## Mean P@K headline")
    lines.append("")
    lines.append("| model | mean | median | P@K ≥ 0.3 | P@K ≥ 0.6 |")
    lines.append("|---|---|---|---|---|")
    for col, lbl in [
        ("mint_pak", "MINT (3khmvobe, no MSA)"),
        ("boltz_nomsa_pak", "Boltz-2 single-seq"),
        ("boltz_msa_pak", "Boltz-2 + MSA"),
    ]:
        v = df[col].values
        lines.append(
            f"| {lbl} | {v.mean():.3f} | {np.median(v):.3f} | {(v >= 0.3).mean():.0%} | {(v >= 0.6).mean():.0%} |"
        )
    lines.append("")
    thr = 0.3
    q1 = ((df["mint_pak"] >= thr) & (df["boltz_msa_pak"] >= thr)).sum()
    q2 = ((df["mint_pak"] < thr) & (df["boltz_msa_pak"] >= thr)).sum()
    q3 = ((df["mint_pak"] < thr) & (df["boltz_msa_pak"] < thr)).sum()
    q4 = ((df["mint_pak"] >= thr) & (df["boltz_msa_pak"] < thr)).sum()
    lines.append(f"## Quadrants at P@K ≥ {thr}")
    lines.append("")
    lines.append(f"- **Both good**: {q1} ({q1/len(df):.0%})")
    lines.append(f"- **Boltz only**: {q2} ({q2/len(df):.0%})")
    lines.append(f"- **MINT only**: {q4} ({q4/len(df):.0%})")
    lines.append(f"- **Both fail**: {q3} ({q3/len(df):.0%})")
    lines.append("")
    lines.append("## Correlation table (Spearman ρ vs inter P@K)")
    lines.append("")
    lines.append("| feature | MINT | Boltz +MSA | Boltz -MSA | MSA gain |")
    lines.append("|---|---|---|---|---|")
    for feat, lbl in [
        ("log_depth_min", "log10 MSA depth"),
        ("L_min", "L_min"),
        ("interface_density", "iface density"),
        ("L_asym", "chain asymmetry"),
    ]:
        row = []
        for col in ["mint_pak", "boltz_msa_pak", "boltz_nomsa_pak", "msa_gain"]:
            r, _ = stats.spearmanr(df[feat].values, df[col].values)
            row.append(f"{r:+.2f}")
        lines.append(f"| {lbl} | " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for fn in sorted(OUT.glob("fig*.png")):
        lines.append(f"- `{fn.name}`")
    (OUT / "findings.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_data()
    print(f"Loaded n={len(df)} dimers with all three predictions + MSA depth.")

    fig_2dhist(df)
    print("  -> fig01_pak_vs_depth_2dhist.png")
    fig_cdfs(df)
    print("  -> fig02_cdfs.png")
    fig_by_length(df)
    print("  -> fig03_stratified_by_length.png")
    fig_by_homo(df)
    print("  -> fig04_stratified_by_homo.png")
    fig_by_density(df)
    print("  -> fig05_stratified_by_density.png")
    fig_head2head(df)
    print("  -> fig06_head2head_mint_vs_boltz.png")
    fig_winloss(df)
    print("  -> fig07_winloss_per_feature.png")
    fig_difficulty(df)
    print("  -> fig08_difficulty_profile.png")
    fig_msa_gain(df)
    print("  -> fig09_msa_gain_drivers.png")
    fig_quadrants(df)
    print("  -> fig10_quadrant_analysis.png")

    df.to_csv(OUT / "joined_table.csv", index=False)
    write_findings(df)
    print(f"  -> {OUT / 'findings.md'}")


if __name__ == "__main__":
    main()
