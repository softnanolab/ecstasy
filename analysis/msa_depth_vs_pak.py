"""
P@K vs MSA depth on the mint_seqid30 val split.

Uses pre-computed predictions:
  - MINT inter-chain P@K from run 3khmvobe (results.json)
  - Boltz-2 with MSA      (boltz2_pak.json)
  - Boltz-2 without MSA   (boltz2_pak_nomsa.json)

MSA depth is computed from the per-chain a3m files in
/projects/u6jv/boltz_benchmarking/DATA/benchmarks/mint_val_seqid30/msas/, then
aggregated per entry as min/mean across the two chains.

Outputs (in this script's directory):
  msa_depth_vs_pak.csv  – per-entry joined table
  msa_depth_vs_pak.png  – 4-panel figure
  msa_depth_vs_pak.md   – summary stats + Spearman correlations
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/projects/u6jv/boltz_benchmarking/DATA/benchmarks/mint_val_seqid30")
MSA_DIR = ROOT / "msas"
MANIFEST = ROOT / "inputs" / "val_manifest.json"
BOLTZ_MSA = ROOT / "results" / "boltz2_pak.json"
BOLTZ_NOMSA = ROOT / "results" / "boltz2_pak_nomsa.json"
MINT_RESULTS = Path(
    "/projects/u6jv/harsh/MINT_META/LOGS/MINT_AFDD_PRETRAIN_8M_35M/3khmvobe/checkpoints/results.json"
)
PARQUET = Path(
    "/projects/u6jv/public/MINT/DATA/pdb/processed/splits/seq_id_30/index.parquet"
)
OUT = Path(__file__).parent


def count_a3m_seqs(a3m_path: Path) -> int:
    """Count records in an a3m. Returns N including the query (>=1)."""
    n = 0
    with a3m_path.open() as f:
        for line in f:
            if line.startswith(">"):
                n += 1
    return n


def build_entry_to_chains(manifest: dict) -> dict[str, list[str]]:
    """Invert manifest: {pdb_id: [chain_hash, ...]} preserving chain_idx order."""
    out: dict[str, dict[int, str]] = {}
    for chain_hash, info in manifest.items():
        for occ in info["occurrences"]:
            pdb_id = occ["pdb_id"]
            out.setdefault(pdb_id, {})[occ["chain_idx"]] = chain_hash
    return {k: [v[i] for i in sorted(v)] for k, v in out.items()}


def main() -> None:
    print("Loading manifest + predictions...")
    manifest = json.loads(MANIFEST.read_text())
    boltz_msa = json.loads(BOLTZ_MSA.read_text())["strict_19"]["per_protein"]
    boltz_nomsa = json.loads(BOLTZ_NOMSA.read_text())["strict_19"]["per_protein"]
    mint = json.loads(MINT_RESULTS.read_text())["pdb"]["seq_id_30"]["val"]["per_protein"]

    df_val = pd.read_parquet(PARQUET)
    df_val = df_val[df_val["split"] == "val"].copy()
    df_val["L_total"] = df_val["total_sequence_length"].astype(int)
    df_val["L_min"] = df_val["sequences"].apply(lambda s: min(len(x) for x in s))
    df_val["is_homo"] = df_val["sequences"].apply(lambda s: len(s) == 2 and s[0] == s[1])

    e2c = build_entry_to_chains(manifest)

    # Cache MSA depth per chain (each chain hash counted once even if shared).
    print("Counting MSA depth per chain...")
    depth_cache: dict[str, int] = {}
    for chain_hash in manifest.keys():
        a3m = MSA_DIR / f"{chain_hash}.a3m"
        depth_cache[chain_hash] = count_a3m_seqs(a3m) if a3m.exists() else 0

    rows = []
    for entry_id in sorted(set(boltz_msa) & set(boltz_nomsa) & set(mint)):
        chains = e2c.get(entry_id, [])
        depths = [depth_cache.get(h, 0) for h in chains]
        if len(chains) != 2 or any(d <= 0 for d in depths):
            continue  # need both chains present with a real a3m
        rows.append(
            dict(
                entry_id=entry_id,
                n_seqs_min=min(depths),
                n_seqs_mean=float(np.mean(depths)),
                n_seqs_a=depths[0],
                n_seqs_b=depths[1],
                pak_mint=mint[entry_id]["inter_P@K"],
                pak_mint_5=mint[entry_id]["inter_P@K/5"],
                pak_boltz_msa=boltz_msa[entry_id]["P@K"],
                pak_boltz_msa_5=boltz_msa[entry_id]["P@K/5"],
                pak_boltz_nomsa=boltz_nomsa[entry_id]["P@K"],
                pak_boltz_nomsa_5=boltz_nomsa[entry_id]["P@K/5"],
            )
        )

    df = pd.DataFrame(rows)
    df_val_idx = df_val.set_index("id")
    df["L_total"] = df["entry_id"].map(df_val_idx["L_total"])
    df["L_min"] = df["entry_id"].map(df_val_idx["L_min"])
    df["is_homo"] = df["entry_id"].map(df_val_idx["is_homo"])

    # Common normalized depth: log10(Neff_proxy) = log10(N / sqrt(L_min))
    df["log_depth_min"] = np.log10(df["n_seqs_min"].clip(lower=1.0))
    df["log_depth_norm"] = np.log10(
        (df["n_seqs_min"] / np.sqrt(df["L_min"].clip(lower=1))).clip(lower=1e-3)
    )
    df["boltz_msa_gain"] = df["pak_boltz_msa"] - df["pak_boltz_nomsa"]

    df.to_csv(OUT / "msa_depth_vs_pak.csv", index=False)
    print(f"  -> {OUT / 'msa_depth_vs_pak.csv'}  (n={len(df)})")

    # Stats
    def spearman(x, y):
        m = ~(np.isnan(x) | np.isnan(y))
        r, p = stats.spearmanr(x[m], y[m])
        return float(r), float(p)

    s_lines = []
    s_lines.append("# P@K vs MSA depth — mint_seqid30 val split")
    s_lines.append("")
    s_lines.append(f"N entries (intersection of MINT ∩ Boltz-MSA ∩ Boltz-noMSA, both chains have a3m): **{len(df)}**")
    s_lines.append("")
    s_lines.append(
        "MSA depth = number of sequences in the a3m for each chain (incl. the "
        "query). Per-entry depth aggregated as `min` (bottleneck chain) and "
        "`mean` across the two chains. `Neff_proxy = N_seqs_min / sqrt(L_min)`."
    )
    s_lines.append("")
    s_lines.append("## Spearman ρ (P@K vs log10 depth)")
    s_lines.append("")
    s_lines.append("| model | depth metric | ρ | p |")
    s_lines.append("|---|---|---|---|")
    for depth_col in ["log_depth_min", "log_depth_norm"]:
        for model_col, lbl in [
            ("pak_mint", "MINT (no MSA)"),
            ("pak_boltz_msa", "Boltz-2 (with MSA)"),
            ("pak_boltz_nomsa", "Boltz-2 (no MSA)"),
            ("boltz_msa_gain", "Boltz-2 MSA gain"),
        ]:
            r, p = spearman(df[depth_col].values, df[model_col].values)
            s_lines.append(f"| {lbl} | {depth_col} | {r:+.3f} | {p:.2e} |")
    s_lines.append("")
    s_lines.append("## P@K binned by MSA-depth quintile (median ± IQR)")
    s_lines.append("")
    df["depth_bin"] = pd.qcut(
        df["n_seqs_min"], q=5, labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]
    )
    summary = df.groupby("depth_bin", observed=True).agg(
        n=("entry_id", "count"),
        depth_min=("n_seqs_min", "median"),
        L_min=("L_min", "median"),
        mint=("pak_mint", "median"),
        boltz_msa=("pak_boltz_msa", "median"),
        boltz_nomsa=("pak_boltz_nomsa", "median"),
        gain=("boltz_msa_gain", "median"),
    )
    cols = list(summary.columns)
    s_lines.append("| bin | " + " | ".join(cols) + " |")
    s_lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for idx, row in summary.iterrows():
        vals = [f"{row[c]:.3g}" if isinstance(row[c], float) else str(row[c]) for c in cols]
        s_lines.append(f"| {idx} | " + " | ".join(vals) + " |")
    s_lines.append("")
    s_lines.append("## Headline numbers (full split, mean P@K)")
    s_lines.append("")
    for col, lbl in [
        ("pak_mint", "MINT"),
        ("pak_boltz_msa", "Boltz-2 (with MSA)"),
        ("pak_boltz_nomsa", "Boltz-2 (no MSA)"),
    ]:
        s_lines.append(f"- **{lbl}** mean inter P@K = {df[col].mean():.3f}  (median {df[col].median():.3f})")
    s_lines.append(
        f"- **Boltz-2 MSA gain** mean = {df['boltz_msa_gain'].mean():+.3f}  (median {df['boltz_msa_gain'].median():+.3f})"
    )
    s_lines.append("")
    win_vs_nomsa = (df["pak_mint"] > df["pak_boltz_nomsa"]).mean()
    win_vs_msa = (df["pak_mint"] > df["pak_boltz_msa"]).mean()
    s_lines.append(f"- MINT > Boltz-2 (no MSA): **{win_vs_nomsa:.1%}** of entries")
    s_lines.append(f"- MINT > Boltz-2 (with MSA): **{win_vs_msa:.1%}** of entries")
    s_lines.append("")
    s_lines.append("## Takeaways")
    s_lines.append("")
    s_lines.append(
        "1. **Boltz-2 benefits massively from MSAs on this set.** Mean inter-chain "
        "P@K goes from 0.084 (single-seq) to 0.503 (with MSA), a +0.42 absolute gain."
    )
    s_lines.append(
        "2. **The MSA benefit grows with MSA depth.** Median ΔP@K rises monotonically "
        "from +0.28 in the lowest-depth quintile (median 88 seqs) to +0.58 in the "
        "highest-depth quintile (median 11k seqs). The gain roughly doubles."
    )
    s_lines.append(
        "3. **Per-model raw correlations of P@K vs depth are weak (ρ≈0.06–0.10).** "
        "Easy/hard intrinsic difficulty dominates within a single model. The cleanest "
        "depth signal is the *Boltz-2 with-MSA minus no-MSA delta*, since it "
        "controls for entry difficulty."
    )
    s_lines.append(
        "4. **MINT inter-chain P@K is low across the board (median 0.006).** "
        "It beats Boltz-no-MSA on 33% of entries but loses to Boltz-with-MSA on 93.5%."
    )
    s_lines.append("")
    s_lines.append("## Important caveats")
    s_lines.append("")
    s_lines.append(
        "- **MINT checkpoint is the local 8M/35M training run** "
        "(`MINT_AFDD_PRETRAIN_8M_35M/3khmvobe`), not the published 650M MINT. "
        "Numbers should not be read as 'MINT vs Boltz-2'."
    )
    s_lines.append(
        "- **This val split is MINT's own training-time validation set.** Model "
        "selection pressure on this set exists for the local MINT run; Boltz-2 "
        "and the MSA pipeline have not seen it as a held-out test."
    )
    s_lines.append(
        "- **Boltz-2 (training cutoff 2023-06-01) likely saw many of these "
        "dimers during training.** The 'with-MSA' P@K is therefore partly an "
        "upper bound; a recent-PDB temporal holdout would be needed to "
        "disentangle MSA contribution from memorization."
    )
    s_lines.append(
        "- **Depth metric is raw a3m line count, not Neff.** Higher counts also "
        "correlate with intrinsically easier (well-studied) proteins, so the "
        "depth↔P@K trend conflates evolutionary signal with entry difficulty."
    )
    s_lines.append("")
    (OUT / "msa_depth_vs_pak.md").write_text("\n".join(s_lines) + "\n")
    print(f"  -> {OUT / 'msa_depth_vs_pak.md'}")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    log_x = df["log_depth_min"].values

    # (a) Scatter MINT vs Boltz-MSA vs Boltz-noMSA
    ax = axes[0, 0]
    for col, lbl, color in [
        ("pak_mint", "MINT (no MSA)", "tab:green"),
        ("pak_boltz_msa", "Boltz-2 + MSA", "tab:blue"),
        ("pak_boltz_nomsa", "Boltz-2 single-seq", "tab:orange"),
    ]:
        ax.scatter(log_x, df[col], s=6, alpha=0.18, label=lbl, color=color, edgecolors="none")
    # rolling medians over sorted depth
    order = np.argsort(log_x)
    win = max(31, len(df) // 30) | 1
    for col, color in [("pak_mint", "tab:green"), ("pak_boltz_msa", "tab:blue"), ("pak_boltz_nomsa", "tab:orange")]:
        y = df[col].values[order]
        med = pd.Series(y).rolling(win, center=True, min_periods=10).median().values
        ax.plot(log_x[order], med, color=color, lw=2.0)
    ax.set_xlabel("log10 (MSA depth, min over chains)")
    ax.set_ylabel("inter-chain P@K")
    ax.set_title("P@K vs MSA depth (per entry, rolling median)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)

    # (b) Quintile boxplots
    ax = axes[0, 1]
    bin_labels = ["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"]
    bin_codes = df["depth_bin"].cat.codes.values
    positions = np.arange(5)
    width = 0.25
    for i, (col, lbl, color) in enumerate([
        ("pak_mint", "MINT", "tab:green"),
        ("pak_boltz_msa", "Boltz-2 +MSA", "tab:blue"),
        ("pak_boltz_nomsa", "Boltz-2 -MSA", "tab:orange"),
    ]):
        data = [df[col].values[bin_codes == b] for b in range(5)]
        bp = ax.boxplot(
            data, positions=positions + (i - 1) * width, widths=width * 0.95,
            patch_artist=True, showfliers=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for median in bp["medians"]:
            median.set_color("black")
        ax.plot([], [], color=color, lw=6, alpha=0.7, label=lbl)
    ax.set_xticks(positions)
    ax.set_xticklabels(bin_labels)
    ax.set_ylabel("inter-chain P@K")
    ax.set_title("P@K by MSA-depth quintile (min-chain depth)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3, axis="y")

    # (c) Boltz MSA gain vs depth
    ax = axes[1, 0]
    ax.scatter(log_x, df["boltz_msa_gain"], s=6, alpha=0.25, color="tab:blue", edgecolors="none")
    order = np.argsort(log_x)
    y = df["boltz_msa_gain"].values[order]
    med = pd.Series(y).rolling(win, center=True, min_periods=10).median().values
    ax.plot(log_x[order], med, color="darkblue", lw=2.0, label="rolling median")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("log10 (MSA depth, min over chains)")
    ax.set_ylabel("ΔP@K  (Boltz-2 with MSA  −  no MSA)")
    ax.set_title("Boltz-2 gain from MSA vs MSA depth")
    ax.legend()
    ax.grid(alpha=0.3)

    # (d) Head-to-head MINT vs Boltz-MSA, colored by depth
    ax = axes[1, 1]
    sc = ax.scatter(
        df["pak_mint"], df["pak_boltz_msa"],
        c=log_x, s=10, alpha=0.6, cmap="viridis", edgecolors="none",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("MINT inter P@K")
    ax.set_ylabel("Boltz-2 +MSA inter P@K")
    ax.set_title("Head-to-head, colored by log10 MSA depth")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.colorbar(sc, ax=ax, label="log10 (MSA depth, min chain)")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"P@K vs MSA depth — mint_seqid30 val (n={len(df)} dimers)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "msa_depth_vs_pak.png", dpi=130)
    print(f"  -> {OUT / 'msa_depth_vs_pak.png'}")
    plt.close(fig)

    # Also dump quintile summary as CSV
    summary.to_csv(OUT / "msa_depth_vs_pak_quintiles.csv")


if __name__ == "__main__":
    main()
