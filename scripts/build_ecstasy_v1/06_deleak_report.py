"""Generate the ecstasy_v1 leakage analysis report.

Inputs:
  - candidates/dimers.parquet            (319 candidate dimers)
  - all_edges_with_coverage.parquet      (every Foldseek hit with computed
                                          interface coverage, no threshold yet)

Outputs:
  - deleak_report.md         (numbers + threshold sweep)
  - deleak_report.png        (4-panel figure)
  - per_dimer_leakage.parquet (one row per dimer: leakage stats)

The "drop count" sweep simulates Pinder Level-2: a dimer is dropped if
*either* of its two chains has any Foldseek hit to a MINT-train chain that
passes both the coverage and LDDT thresholds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1")
DIMERS_PATH = ROOT / "candidates" / "dimers.parquet"
ALL_EDGES_PATH = ROOT / "all_edges_with_coverage.parquet"
REPORT_MD = ROOT / "deleak_report.md"
REPORT_PNG = ROOT / "deleak_report.png"
PER_DIMER_PATH = ROOT / "per_dimer_leakage.parquet"

LDDT_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]
COVERAGE_GRID = [0.3, 0.5, 0.7]


def main() -> int:
    dimers = pd.read_parquet(DIMERS_PATH)
    edges = pd.read_parquet(ALL_EDGES_PATH)
    print(f"Loaded {len(dimers)} dimers, {len(edges)} edges (any coverage)")

    # Per-dimer leakage summary
    per_dimer = []
    for di, d in dimers.iterrows():
        e = edges[edges["dimer_idx"] == di]
        # split by side
        e_a = e[e["side"] == "a"]
        e_b = e[e["side"] == "b"]
        per_dimer.append(
            {
                "dimer_idx": di,
                "pdb_id": d["pdb_id"],
                "chain_a": d["chain_a"],
                "chain_b": d["chain_b"],
                "len_a": d["len_a"],
                "len_b": d["len_b"],
                "is_homodimer": d["is_homodimer"],
                "n_hits_a": len(e_a),
                "n_hits_b": len(e_b),
                "n_hits_a_cov50": int((e_a["coverage"] >= 0.5).sum()),
                "n_hits_b_cov50": int((e_b["coverage"] >= 0.5).sum()),
                "max_lddt_a_cov50": (
                    float(e_a.loc[e_a["coverage"] >= 0.5, "lddt"].max())
                    if (e_a["coverage"] >= 0.5).any()
                    else 0.0
                ),
                "max_lddt_b_cov50": (
                    float(e_b.loc[e_b["coverage"] >= 0.5, "lddt"].max())
                    if (e_b["coverage"] >= 0.5).any()
                    else 0.0
                ),
            }
        )
    per_dimer_df = pd.DataFrame(per_dimer)
    per_dimer_df.to_parquet(PER_DIMER_PATH, index=False)
    print(f"  wrote {PER_DIMER_PATH}")

    # Threshold-sweep drop matrix (Level-2: either chain triggers => drop)
    sweep_rows = []
    for lddt_th in LDDT_GRID:
        for cov_th in COVERAGE_GRID:
            mask = (edges["coverage"] >= cov_th) & (edges["lddt"] >= lddt_th)
            leaky_dimers = set(edges.loc[mask, "dimer_idx"].unique())
            n_drop = len(leaky_dimers)
            n_keep = len(dimers) - n_drop
            # subsplits
            homo_drop = sum(
                1 for di in leaky_dimers if dimers.iloc[di]["is_homodimer"]
            )
            sweep_rows.append(
                {
                    "lddt_th": lddt_th,
                    "cov_th": cov_th,
                    "n_dropped": n_drop,
                    "n_kept": n_keep,
                    "n_dropped_homo": homo_drop,
                    "n_dropped_hetero": n_drop - homo_drop,
                }
            )
    sweep = pd.DataFrame(sweep_rows)

    # === figure ===
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) Per-dimer max LDDT histogram, at coverage>=0.5
    ax = axes[0, 0]
    max_lddt_per_dimer = np.maximum(
        per_dimer_df["max_lddt_a_cov50"], per_dimer_df["max_lddt_b_cov50"]
    )
    ax.hist(max_lddt_per_dimer, bins=40, edgecolor="black", color="steelblue")
    ax.set_xlabel("max LDDT to any MINT-train chain (coverage ≥ 0.5)")
    ax.set_ylabel("# candidate dimers")
    ax.set_title("Per-dimer max LDDT (Level-2 leak signal)")
    ax.axvline(0.7, color="red", linestyle="--", label="Pinder LDDT 0.7 cutoff")
    ax.legend()

    # (b) #hits per chain (cov >= 0.5) CDF — how many MINT-train chains overlap
    ax = axes[0, 1]
    n_hits_per_chain = pd.concat(
        [per_dimer_df["n_hits_a_cov50"], per_dimer_df["n_hits_b_cov50"]]
    )
    sorted_hits = np.sort(n_hits_per_chain)
    cdf = np.arange(1, len(sorted_hits) + 1) / len(sorted_hits)
    ax.semilogx(np.maximum(sorted_hits, 0.5), cdf, color="darkorange")
    ax.set_xlabel("# MINT-train chains with cov ≥ 0.5 (per candidate chain)")
    ax.set_ylabel("CDF over candidate chains")
    ax.set_title("Neighborhood density: chain-level coverage hits")
    ax.grid(True, alpha=0.3)

    # (c) Drop count vs LDDT for cov=0.5
    ax = axes[1, 0]
    for cov_th, color in zip([0.3, 0.5, 0.7], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        sub = sweep[sweep["cov_th"] == cov_th].sort_values("lddt_th")
        ax.plot(
            sub["lddt_th"],
            sub["n_dropped"],
            marker="o",
            color=color,
            label=f"cov ≥ {cov_th}",
        )
    ax.set_xlabel("LDDT threshold")
    ax.set_ylabel("# candidate dimers dropped (Level-2)")
    ax.set_title(f"Drop-count sweep (out of {len(dimers)} candidates)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) homo vs hetero breakdown at default (cov=0.5, lddt=0.7)
    ax = axes[1, 1]
    n_homo = int(dimers["is_homodimer"].sum())
    n_het = len(dimers) - n_homo
    default = sweep[(sweep["cov_th"] == 0.5) & (sweep["lddt_th"] == 0.7)].iloc[0]
    bars = ax.bar(
        ["homo", "hetero"],
        [n_homo, n_het],
        color=["#888888", "#888888"],
        label="total",
    )
    ax.bar(
        ["homo", "hetero"],
        [int(default["n_dropped_homo"]), int(default["n_dropped_hetero"])],
        color=["red", "red"],
        alpha=0.7,
        label="dropped @ cov≥0.5, LDDT≥0.7",
    )
    ax.set_ylabel("# dimers")
    ax.set_title("Homo vs hetero leakage (Pinder defaults)")
    ax.legend()
    for i, b in enumerate(bars):
        n_drop_i = int(default["n_dropped_homo"]) if i == 0 else int(default["n_dropped_hetero"])
        n_tot_i = n_homo if i == 0 else n_het
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 2,
            f"{n_drop_i}/{n_tot_i} drop",
            ha="center",
            fontsize=10,
        )

    fig.suptitle(
        f"ecstasy_v1 leakage analysis — {len(dimers)} candidate dimers vs MINT-train",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(REPORT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {REPORT_PNG}")

    # === markdown report ===
    n_dimers = len(dimers)
    n_homo = int(dimers["is_homodimer"].sum())
    n_het = n_dimers - n_homo
    default_row = sweep[(sweep["cov_th"] == 0.5) & (sweep["lddt_th"] == 0.7)].iloc[0]
    sweep_table = sweep.pivot(
        index="lddt_th", columns="cov_th", values="n_dropped"
    ).round(0).astype(int)

    md = []
    md.append("# ecstasy_v1 leakage analysis report")
    md.append("")
    md.append(
        "Candidate pool: dimers enumerated from Boltz-2 `validation_ids_v2.txt` "
        "(bio-assembly 1, all chain pairs ≤ 10 Å backbone contact, X-ray ≤ 3.5 Å, "
        "min chain ≥ 40 res, total ≤ 1200 res).\n"
    )
    md.append("Leak target: MINT-softnano train chains (PDB ≤ 2021-09-30, seq_id_30 split).\n")
    md.append("Method: Foldseek `easy-search --alignment-type 2 -s 11.0 -e 0.05`, then "
              "per-hit interface coverage = |I_candidate ∩ [qstart..qend]| / |I_candidate|. "
              "Level-2 drop rule: a dimer is dropped if *either* chain has any qualifying hit.\n")
    md.append("")
    md.append(f"## Candidate pool")
    md.append(f"- Total candidate dimers: **{n_dimers}**")
    md.append(f"  - homodimers: {n_homo}  ({100*n_homo/n_dimers:.1f}%)")
    md.append(f"  - heterodimers: {n_het}  ({100*n_het/n_dimers:.1f}%)")
    md.append(f"- Total length range: "
              f"[{(dimers['len_a']+dimers['len_b']).min()}, "
              f"{(dimers['len_a']+dimers['len_b']).max()}] residues")
    md.append("")
    md.append("## Threshold sweep — # candidates dropped under Level-2 rule")
    md.append("")
    md.append("Rows = LDDT threshold; columns = coverage threshold. "
              f"Value = # candidate dimers ({n_dimers} total) whose either chain has "
              "at least one MINT-train Foldseek hit passing both thresholds → dropped.")
    md.append("")
    md.append("| LDDT \\ cov | " + " | ".join(f"≥{c}" for c in COVERAGE_GRID) + " |")
    md.append("|---" + "|---" * len(COVERAGE_GRID) + "|")
    for lddt_th in LDDT_GRID:
        row = [f"≥{lddt_th}"]
        for cov_th in COVERAGE_GRID:
            v = int(sweep_table.loc[lddt_th, cov_th])
            row.append(f"{v} ({100*v/n_dimers:.0f}%)")
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    md.append(f"## At Pinder defaults (cov ≥ 0.5, LDDT ≥ 0.7)")
    md.append(f"- dropped: **{int(default_row['n_dropped'])}** of {n_dimers}  "
              f"({100*default_row['n_dropped']/n_dimers:.1f}%)")
    md.append(f"- kept (master set): **{int(default_row['n_kept'])}** dimers")
    md.append(f"  - of which homodimers dropped: {int(default_row['n_dropped_homo'])} / {n_homo}")
    md.append(f"  - of which heterodimers dropped: {int(default_row['n_dropped_hetero'])} / {n_het}")
    md.append("")
    md.append("## Per-dimer details")
    md.append(f"See `per_dimer_leakage.parquet` for the full per-dimer leakage table "
              f"(max LDDT per side, # qualifying hits per side, etc.) — {len(per_dimer_df)} rows.")
    md.append("")
    md.append("## Files in this report")
    md.append(f"- `{REPORT_PNG.name}` — 4-panel figure")
    md.append(f"- `{PER_DIMER_PATH.name}` — per-dimer leakage stats")
    md.append(f"- `all_edges_with_coverage.parquet` — every Foldseek hit + computed coverage (no threshold)")
    md.append(f"- `interface_edges.parquet` — kept edges (coverage ≥ 0.5)")

    REPORT_MD.write_text("\n".join(md) + "\n")
    print(f"  wrote {REPORT_MD}")

    print("\n=== headline ===")
    print(f"  candidates: {n_dimers} dimers")
    print(f"  default-cut drop: {int(default_row['n_dropped'])} ({100*default_row['n_dropped']/n_dimers:.1f}%)")
    print(f"  default-cut keep: {int(default_row['n_kept'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
