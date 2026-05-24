"""Compute Pinder-style interface-coverage edges from Foldseek hits.

For each Foldseek hit (candidate_chain, mentos_train_chain):
  - look up the candidate chain's interface residues from dimers.parquet
  - coverage = |I_candidate ∩ [qstart..qend]| / |I_candidate|
  - keep the hit if coverage >= 0.5 (Pinder GraphConfig default)

A candidate chain can appear in multiple dimers (if its PDB ID had >2 chains
in bio-assembly 1 and so it contributed to >1 contacting pair). The interface
residue set differs per partner, so we expand each hit into one row per
(candidate dimer, candidate chain) and recompute coverage with the right
interface set.

Output: interface_edges.parquet with one row per kept edge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATES_DIR = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/candidates")
HITS_PATH = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/foldseek_hits.m8")
OUT_PATH = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/interface_edges.parquet")
ALL_EDGES_PATH = Path(
    "/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/all_edges_with_coverage.parquet"
)

COVERAGE_THRESHOLD = 0.5  # Pinder GraphConfig default


def _strip_ext(name: str) -> str:
    if name.endswith(".pdb"):
        return name[:-4]
    return name


def main() -> int:
    print(f"Loading Foldseek hits from {HITS_PATH} ...")
    cols = [
        "query", "target", "lddt",
        "qstart", "qend", "qlen",
        "tstart", "tend", "tlen",
        "alnlen", "evalue",
    ]
    hits = pd.read_csv(HITS_PATH, sep="\t", header=None, names=cols)
    hits["query"] = hits["query"].map(_strip_ext)
    hits["target"] = hits["target"].map(_strip_ext)
    print(f"  {len(hits)} raw hits")
    hits = hits[hits["query"] != hits["target"]].reset_index(drop=True)
    print(f"  {len(hits)} hits after dropping any accidental self-hits")

    print(f"Loading candidate dimers from {CANDIDATES_DIR}/dimers.parquet ...")
    dimers = pd.read_parquet(CANDIDATES_DIR / "dimers.parquet")
    print(f"  {len(dimers)} candidate dimers")

    cand_long: list[dict] = []
    for di, d in dimers.iterrows():
        key_a = f"{d['pdb_id']}_{d['assembly_id']}_{d['chain_a']}"
        key_b = f"{d['pdb_id']}_{d['assembly_id']}_{d['chain_b']}"
        cand_long.append(
            {
                "dimer_idx": di,
                "pdb_id": d["pdb_id"],
                "chain_label": d["chain_a"],
                "partner_label": d["chain_b"],
                "side": "a",
                "query_key": key_a,
                "interface_idx": np.array(d["interface_idx_a"], dtype=np.int32),
                "n_interface": int(d["n_interface_residues_a"]),
                "chain_len": int(d["len_a"]),
                "is_homodimer": bool(d["is_homodimer"]),
            }
        )
        cand_long.append(
            {
                "dimer_idx": di,
                "pdb_id": d["pdb_id"],
                "chain_label": d["chain_b"],
                "partner_label": d["chain_a"],
                "side": "b",
                "query_key": key_b,
                "interface_idx": np.array(d["interface_idx_b"], dtype=np.int32),
                "n_interface": int(d["n_interface_residues_b"]),
                "chain_len": int(d["len_b"]),
                "is_homodimer": bool(d["is_homodimer"]),
            }
        )
    cand_long_df = pd.DataFrame(cand_long)
    print(f"  expanded to {len(cand_long_df)} (dimer, chain-side) rows")

    print("Joining + computing coverage ...")
    merged = hits.merge(
        cand_long_df, left_on="query", right_on="query_key", how="inner"
    )
    print(f"  {len(merged)} (hit, candidate-side) rows after join")

    def _coverage_row(row) -> float:
        iface = row["interface_idx"]
        if len(iface) == 0:
            return 0.0
        qs, qe = int(row["qstart"]), int(row["qend"])
        in_range = (iface >= qs - 1) & (iface <= qe - 1)
        return float(in_range.sum()) / float(len(iface))

    merged["coverage"] = merged.apply(_coverage_row, axis=1)

    keep_cols = [
        "dimer_idx", "pdb_id", "chain_label", "partner_label", "side",
        "query_key", "target", "lddt", "qstart", "qend", "qlen",
        "tstart", "tend", "tlen", "alnlen", "evalue",
        "n_interface", "chain_len", "is_homodimer", "coverage",
    ]
    full = merged[keep_cols].copy()
    full.to_parquet(ALL_EDGES_PATH, index=False)
    print(f"  wrote {ALL_EDGES_PATH}  ({len(full)} rows)")

    kept = full[full["coverage"] >= COVERAGE_THRESHOLD].copy()
    kept.to_parquet(OUT_PATH, index=False)
    print(f"  wrote {OUT_PATH}  ({len(kept)} rows above coverage {COVERAGE_THRESHOLD})")

    print()
    print("=== summary ===")
    print(f"raw hits:                    {len(hits)}")
    print(f"hits joined to candidates:   {len(merged)}")
    print(f"with coverage >= {COVERAGE_THRESHOLD}:        {len(kept)}")
    if len(kept):
        print(f"LDDT distribution of kept edges:")
        print(kept["lddt"].describe().to_string())
        print(f"unique candidate dimers w/ >=1 kept edge: {kept['dimer_idx'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
