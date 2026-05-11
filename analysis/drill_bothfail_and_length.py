"""Drill into the both-fail bucket and chain-length effects.

Both-fail = MINT P@K < 0.3 AND Boltz-2 +MSA P@K < 0.3. Of 1473 dimers, 549 (37%)
land here. The question is: what makes them hard, and is it structural,
evolutionary, or an artifact of the val split / metric?

Reads analysis/figures/joined_table.csv produced by visualize_mint_vs_boltz.py.

Outputs to analysis/figures/:
  fig11_bothfail_feature_distributions.png  per-feature histogram, both-fail vs others
  fig12_chain_length_detail.png             4-panel: L_min, L_max, L_total, L_asym vs P@K
  fig13_length_x_depth_heatmap.png          2D bin grid: L_min × log_depth -> mean P@K
  fig14_bothfail_vs_bothgood.png            KS / effect-size: which features separate fail vs good
  fig15_length_thresholds.png               cumulative P(P@K ≥ τ) curves binned by L_min
  fig16_failure_clusters.png                k-means over feature space; per-cluster mean P@K
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)


def load() -> pd.DataFrame:
    df = pd.read_csv(OUT / "joined_table.csv")
    thr = 0.3
    df["bucket"] = pd.Categorical(
        df.apply(
            lambda r: (
                "both_good" if r["mint_pak"] >= thr and r["boltz_msa_pak"] >= thr
                else "boltz_only" if r["boltz_msa_pak"] >= thr
                else "mint_only" if r["mint_pak"] >= thr
                else "both_fail"
            ),
            axis=1,
        ),
        categories=["both_good", "boltz_only", "mint_only", "both_fail"],
        ordered=True,
    )
    df["log_n_true"] = np.log10(df["n_true_inter"].clip(lower=1))
    return df


# ---------------------------------------------------------------------------
# Figure 11: feature distributions, both-fail vs not-both-fail
# ---------------------------------------------------------------------------

def fig_bothfail_feature_distributions(df: pd.DataFrame) -> None:
    feats = [
        ("L_min", "L_min (shorter chain)", False),
        ("L_max", "L_max (longer chain)", False),
        ("L_total", "L_total (dimer length)", False),
        ("L_asym", "chain asymmetry (L_max-L_min)/L_max", False),
        ("n_seqs_min", "MSA depth (min over chains)", True),
        ("interface_density", "interface contact density", True),
        ("n_true_inter", "# true inter-chain contacts", True),
        ("is_homo", "homodimer fraction", False),
    ]
    fail = df[df["bucket"] == "both_fail"]
    other = df[df["bucket"] != "both_fail"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, (col, lbl, logx) in zip(axes.ravel(), feats):
        if col == "is_homo":
            f = fail[col].astype(float).mean()
            o = other[col].astype(float).mean()
            ax.bar([0, 1], [f, o], color=["tab:red", "tab:green"], alpha=0.75)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([f"both_fail (n={len(fail)})", f"other (n={len(other)})"])
            ax.set_ylabel("fraction homodimer")
            ax.set_title(lbl)
            ax.grid(alpha=0.3, axis="y")
            continue
        f_vals = fail[col].values
        o_vals = other[col].values
        bins = np.geomspace(max(1e-6, min(f_vals.min(), o_vals.min())), max(f_vals.max(), o_vals.max()), 35) if logx else np.linspace(min(f_vals.min(), o_vals.min()), max(f_vals.max(), o_vals.max()), 35)
        ax.hist(o_vals, bins=bins, alpha=0.55, density=True, color="tab:green", label=f"other  (n={len(other)})")
        ax.hist(f_vals, bins=bins, alpha=0.55, density=True, color="tab:red", label=f"both_fail  (n={len(fail)})")
        ks_stat, ks_p = stats.ks_2samp(f_vals, o_vals)
        ax.set_xlabel(lbl)
        ax.set_ylabel("density")
        ax.set_title(f"{lbl}    KS={ks_stat:.2f}, p={ks_p:.0e}")
        if logx:
            ax.set_xscale("log")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle("Feature distributions: both-fail vs everyone else", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "fig11_bothfail_feature_distributions.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 12: chain length detail, 4 panels (L_min / L_max / L_total / L_asym)
# ---------------------------------------------------------------------------

def _scatter_with_rolling(ax, x, y, color, label, log_x=False):
    ax.scatter(x, y, s=6, alpha=0.25, color=color, edgecolors="none", label=label)
    order = np.argsort(x)
    win = max(31, len(x) // 25) | 1
    med = pd.Series(y[order]).rolling(win, center=True, min_periods=10).median().values
    ax.plot(x[order], med, color=color, lw=2.0, alpha=0.9)
    if log_x:
        ax.set_xscale("log")


def fig_chain_length_detail(df: pd.DataFrame) -> None:
    metrics = [
        ("L_min", "L_min (shorter chain, residues)", False),
        ("L_max", "L_max (longer chain, residues)", False),
        ("L_total", "L_total (dimer length, residues)", False),
        ("L_asym", "L asymmetry (L_max-L_min)/L_max", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for ax, (col, lbl, logx) in zip(axes.ravel(), metrics):
        x = df[col].values
        for ycol, mlbl, color in [
            ("mint_pak", "MINT", "tab:green"),
            ("boltz_nomsa_pak", "Boltz-2 -MSA", "tab:orange"),
            ("boltz_msa_pak", "Boltz-2 +MSA", "tab:blue"),
        ]:
            _scatter_with_rolling(ax, x, df[ycol].values, color, mlbl, log_x=logx)
        # spearman per model
        rs = {
            m: f"{stats.spearmanr(x, df[c].values)[0]:+.2f}"
            for m, c in [("MINT", "mint_pak"), ("Boltz+MSA", "boltz_msa_pak"), ("Boltz-MSA", "boltz_nomsa_pak")]
        }
        ax.set_xlabel(lbl)
        ax.set_ylabel("inter-chain P@K")
        ax.set_title(f"{lbl}    ρ: MINT={rs['MINT']}, Boltz+MSA={rs['Boltz+MSA']}, Boltz-MSA={rs['Boltz-MSA']}")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("Chain-length detail: how does each length metric affect P@K?", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "fig12_chain_length_detail.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 13: 2D heatmap of mean P@K over (L_min, log_depth)
# ---------------------------------------------------------------------------

def fig_length_x_depth_heatmap(df: pd.DataFrame) -> None:
    df = df.copy()
    df["L_bin"] = pd.cut(
        df["L_min"], bins=[0, 100, 150, 200, 250, 300, 400, 1000],
        labels=["<100", "100-150", "150-200", "200-250", "250-300", "300-400", ">400"],
    )
    df["D_bin"] = pd.qcut(
        df["n_seqs_min"], q=5,
        labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"],
    )

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (col, lbl, cmap) in zip(
        axes,
        [
            ("mint_pak", "MINT (no MSA)", "Greens"),
            ("boltz_msa_pak", "Boltz-2 + MSA", "Blues"),
            ("boltz_nomsa_pak", "Boltz-2 single-seq", "Oranges"),
        ],
    ):
        piv = df.groupby(["L_bin", "D_bin"], observed=True)[col].mean().unstack("D_bin")
        cnt = df.groupby(["L_bin", "D_bin"], observed=True)[col].count().unstack("D_bin")
        im = ax.imshow(piv.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels(piv.index)
        ax.set_xlabel("MSA depth quintile (min chain)")
        ax.set_ylabel("L_min bin")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                n = cnt.values[i, j]
                if np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="grey")
                else:
                    txt = f"{v:.2f}\n(n={n})"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                            color="white" if v > 0.55 else "black")
        ax.set_title(lbl)
        plt.colorbar(im, ax=ax, label="mean P@K")
    fig.suptitle("Mean P@K over (L_min × MSA depth) grid", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig13_length_x_depth_heatmap.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 14: effect-size summary, which features separate bothfail vs bothgood
# ---------------------------------------------------------------------------

def fig_bothfail_vs_bothgood(df: pd.DataFrame) -> None:
    fail = df[df["bucket"] == "both_fail"]
    good = df[df["bucket"] == "both_good"]
    feats = [
        ("L_min", "L_min"),
        ("L_max", "L_max"),
        ("L_total", "L_total"),
        ("L_asym", "L asymmetry"),
        ("log_depth_min", "log10 MSA depth (min)"),
        ("interface_density", "iface density"),
        ("log_n_true", "log10 # true inter contacts"),
    ]
    rows = []
    for col, lbl in feats:
        f, g = fail[col].values, good[col].values
        u, p = stats.mannwhitneyu(f, g, alternative="two-sided")
        # Cliff's delta as effect-size (robust, scale-free)
        cliff = 2 * u / (len(f) * len(g)) - 1
        rows.append(dict(feature=lbl, fail_median=np.median(f), good_median=np.median(g),
                         cliff_delta=float(cliff), mwu_p=float(p)))
    stats_df = pd.DataFrame(rows).sort_values("cliff_delta")

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["tab:red" if d < 0 else "tab:green" for d in stats_df["cliff_delta"]]
    bars = ax.barh(range(len(stats_df)), stats_df["cliff_delta"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(stats_df)))
    ax.set_yticklabels(stats_df["feature"])
    ax.axvline(0, color="k", lw=0.5)
    for i, (d, p, fm, gm) in enumerate(
        zip(stats_df["cliff_delta"], stats_df["mwu_p"], stats_df["fail_median"], stats_df["good_median"])
    ):
        sgn = "+" if d > 0 else ""
        ax.text(d + (0.01 if d >= 0 else -0.01), i, f"{sgn}{d:.2f}  p={p:.0e}\nfail={fm:.2g}, good={gm:.2g}",
                va="center", ha="left" if d >= 0 else "right", fontsize=9)
    ax.set_xlabel("Cliff's δ   ( <0 → feature lower in both_fail than both_good )")
    ax.set_xlim(-1, 1)
    ax.set_title(f"Effect size: both_fail (n={len(fail)}) vs both_good (n={len(good)})")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "fig14_bothfail_vs_bothgood.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 15: length-bin success curves
# ---------------------------------------------------------------------------

def fig_length_thresholds(df: pd.DataFrame) -> None:
    df = df.copy()
    df["L_bin"] = pd.cut(
        df["L_min"], bins=[0, 100, 150, 200, 300, 1000],
        labels=["L<100", "100-150", "150-200", "200-300", ">300"],
    )
    taus = np.linspace(0, 1, 101)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    bins = list(df["L_bin"].cat.categories)
    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(bins)))
    for ax, (col, lbl) in zip(
        axes,
        [
            ("mint_pak", "MINT (no MSA)"),
            ("boltz_msa_pak", "Boltz-2 + MSA"),
            ("boltz_nomsa_pak", "Boltz-2 single-seq"),
        ],
    ):
        for i, b in enumerate(bins):
            sub = df[df["L_bin"] == b][col].values
            if len(sub) == 0: continue
            surv = np.array([(sub >= t).mean() for t in taus])
            ax.plot(taus, surv, color=cmap[i], lw=2, label=f"{b} (n={len(sub)})")
        ax.set_xlabel("P@K threshold τ")
        ax.set_title(lbl)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
    axes[0].set_ylabel("P(P@K ≥ τ)")
    fig.suptitle("Success rate vs P@K threshold, stratified by L_min", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig15_length_thresholds.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 16: k-means clustering on the both-fail set
# ---------------------------------------------------------------------------

def fig_failure_clusters(df: pd.DataFrame) -> None:
    fail = df[df["bucket"] == "both_fail"].copy()
    feats = ["L_min", "L_max", "L_asym", "log_depth_min", "interface_density", "log_n_true"]
    X = fail[feats].values
    # z-score
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    from scipy.cluster.vq import kmeans2
    K = 4
    np.random.seed(0)
    centers, labels = kmeans2(Xz, K, seed=0, minit="++")
    km = type("KM", (), {"cluster_centers_": centers, "labels_": labels})
    fail["cluster"] = labels

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # heatmap: cluster centroids in z-space
    centroids = pd.DataFrame(km.cluster_centers_, columns=feats)
    im = axes[0].imshow(centroids.values, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    axes[0].set_xticks(range(len(feats)))
    axes[0].set_xticklabels(feats, rotation=30, ha="right")
    axes[0].set_yticks(range(K))
    axes[0].set_yticklabels([f"cluster {i} (n={(km.labels_==i).sum()})" for i in range(K)])
    axes[0].set_title("Both-fail clusters (z-scored centroids)")
    for i in range(K):
        for j, f in enumerate(feats):
            raw = fail.loc[km.labels_ == i, f].median()
            txt = f"{raw:.2g}"
            axes[0].text(j, i, txt, ha="center", va="center", fontsize=9,
                         color="white" if abs(centroids.values[i, j]) > 0.8 else "black")
    plt.colorbar(im, ax=axes[0], label="z-score")

    # right panel: counts + per-cluster median features as a "profile"
    profile = fail.groupby("cluster")[feats + ["mint_pak", "boltz_msa_pak"]].median().round(3)
    counts = fail["cluster"].value_counts().sort_index()
    axes[1].bar(counts.index, counts.values, color="tab:red", alpha=0.7)
    axes[1].set_xticks(range(K))
    axes[1].set_xticklabels([f"cluster {i}" for i in range(K)])
    axes[1].set_ylabel("count")
    axes[1].set_title("Both-fail cluster sizes")
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + 4, str(v), ha="center")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT / "fig16_failure_clusters.png", dpi=130)
    plt.close(fig)

    # Save the clustered table
    fail[["entry_id", "cluster"] + feats + ["mint_pak", "boltz_msa_pak", "boltz_nomsa_pak"]].to_csv(
        OUT / "bothfail_clusters.csv", index=False
    )


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def write_drilldown(df: pd.DataFrame) -> None:
    fail = df[df["bucket"] == "both_fail"]
    good = df[df["bucket"] == "both_good"]
    other = df[df["bucket"] != "both_fail"]
    lines = []
    lines.append("# Drill-down: both-fail entries and chain length")
    lines.append("")
    lines.append(f"both_fail = MINT P@K < 0.3 AND Boltz-2 +MSA P@K < 0.3  →  n = {len(fail)} ({len(fail)/len(df):.0%} of {len(df)})")
    lines.append("")
    lines.append("## Median feature comparison (both_fail vs everyone else)")
    lines.append("")
    lines.append("| feature | both_fail (n=549) | other (n=924) | Δ (fail − other) |")
    lines.append("|---|---|---|---|")
    for col, lbl in [
        ("L_min", "L_min"), ("L_max", "L_max"), ("L_total", "L_total"),
        ("L_asym", "L asymmetry"), ("n_seqs_min", "MSA depth (min)"),
        ("interface_density", "iface density"), ("n_true_inter", "# true inter contacts"),
        ("is_homo", "homo fraction"),
    ]:
        if col == "is_homo":
            f, o = fail[col].astype(float).mean(), other[col].astype(float).mean()
            lines.append(f"| {lbl} | {f:.2f} | {o:.2f} | {f - o:+.2f} |")
        else:
            f, o = fail[col].median(), other[col].median()
            lines.append(f"| {lbl} | {f:.4g} | {o:.4g} | {f - o:+.4g} |")
    lines.append("")
    lines.append("## Top 20 both-fail entries by interface size (highest n_true_inter)")
    lines.append("")
    top_iface = fail.nlargest(20, "n_true_inter")[
        ["entry_id", "L_min", "L_max", "n_seqs_min", "interface_density", "n_true_inter", "mint_pak", "boltz_msa_pak"]
    ]
    lines.append("| " + " | ".join(top_iface.columns) + " |")
    lines.append("|" + "|".join(["---"] * len(top_iface.columns)) + "|")
    for _, row in top_iface.iterrows():
        lines.append("| " + " | ".join(f"{row[c]:.3g}" if isinstance(row[c], float) else str(row[c]) for c in top_iface.columns) + " |")
    lines.append("")
    lines.append("## Chain-length spearman ρ vs P@K")
    lines.append("")
    lines.append("| feature | MINT | Boltz +MSA | Boltz -MSA |")
    lines.append("|---|---|---|---|")
    for col, lbl in [("L_min", "L_min"), ("L_max", "L_max"), ("L_total", "L_total"), ("L_asym", "L asymmetry")]:
        rs = []
        for m, ycol in [("MINT", "mint_pak"), ("Boltz+MSA", "boltz_msa_pak"), ("Boltz-MSA", "boltz_nomsa_pak")]:
            r, _ = stats.spearmanr(df[col].values, df[ycol].values)
            rs.append(f"{r:+.2f}")
        lines.append(f"| {lbl} | " + " | ".join(rs) + " |")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for fn in [
        "fig11_bothfail_feature_distributions.png",
        "fig12_chain_length_detail.png",
        "fig13_length_x_depth_heatmap.png",
        "fig14_bothfail_vs_bothgood.png",
        "fig15_length_thresholds.png",
        "fig16_failure_clusters.png",
    ]:
        lines.append(f"- `{fn}`")
    (OUT / "drilldown.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    df = load()
    print(f"Loaded n={len(df)} dimers. buckets: {df['bucket'].value_counts().to_dict()}")
    fig_bothfail_feature_distributions(df)
    print("  -> fig11_bothfail_feature_distributions.png")
    fig_chain_length_detail(df)
    print("  -> fig12_chain_length_detail.png")
    fig_length_x_depth_heatmap(df)
    print("  -> fig13_length_x_depth_heatmap.png")
    fig_bothfail_vs_bothgood(df)
    print("  -> fig14_bothfail_vs_bothgood.png")
    fig_length_thresholds(df)
    print("  -> fig15_length_thresholds.png")
    fig_failure_clusters(df)
    print("  -> fig16_failure_clusters.png")
    write_drilldown(df)
    print(f"  -> {OUT / 'drilldown.md'}")


if __name__ == "__main__":
    main()
