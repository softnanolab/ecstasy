"""Compute Pinder-style interface-coverage edges from BOTH Foldseek hit files.

Same per-edge math as v1's 05 (interface ∩ aligned-query-range / interface),
applied to each (candidate_chain, train_chain) hit, but expanded into one
row per (dimer, chain-side, source DB).

The `source` column tags each edge as "mentos" or "boltz2".

Outputs:
  all_edges_with_coverage.parquet   # everything, both sources
  interface_edges.parquet           # filtered to coverage >= 0.5 (Pinder default)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v2")
CANDIDATES_DIR = ROOT / "candidates"
HITS_MENTOS = ROOT / "foldseek_hits_mentos.m8"
HITS_BOLTZ2 = ROOT / "foldseek_hits_boltz2.m8"
OUT_PATH = ROOT / "interface_edges.parquet"
ALL_EDGES_PATH = ROOT / "all_edges_with_coverage.parquet"

COVERAGE_THRESHOLD = 0.5

HIT_COLS = [
    "query", "target", "lddt",
    "qstart", "qend", "qlen",
    "tstart", "tend", "tlen",
    "alnlen", "evalue",
]


def _strip_ext(name: str) -> str:
    return name[:-4] if name.endswith(".pdb") else name


def load_hits(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        print(f"  WARNING: {path} missing — skipping {source} source")
        return pd.DataFrame(columns=[*HIT_COLS, "source"])
    df = pd.read_csv(path, sep="\t", header=None, names=HIT_COLS)
    df["query"] = df["query"].map(_strip_ext)
    df["target"] = df["target"].map(_strip_ext)
    df = df[df["query"] != df["target"]].reset_index(drop=True)
    df["source"] = source
    print(f"  {source}: {len(df)} hits after self-hit drop")
    return df


def main() -> int:
    print("Loading Foldseek hits ...")
    hits_m = load_hits(HITS_MENTOS, "mentos")
    hits_b = load_hits(HITS_BOLTZ2, "boltz2")
    hits = pd.concat([hits_m, hits_b], ignore_index=True)
    print(f"  total hits across sources: {len(hits)}")

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
    merged = hits.merge(cand_long_df, left_on="query", right_on="query_key", how="inner")
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
        "query_key", "target", "source", "lddt",
        "qstart", "qend", "qlen", "tstart", "tend", "tlen", "alnlen", "evalue",
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
    print(f"raw hits (both sources):     {len(hits)}")
    print(f"hits joined to candidates:   {len(merged)}")
    print(f"with coverage >= {COVERAGE_THRESHOLD}:        {len(kept)}")
    if len(kept):
        print(f"  per-source breakdown:      {kept['source'].value_counts().to_dict()}")
        print(f"unique candidate dimers w/ >=1 kept edge: {kept['dimer_idx'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
