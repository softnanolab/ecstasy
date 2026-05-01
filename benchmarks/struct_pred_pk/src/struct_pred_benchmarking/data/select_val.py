"""Select N proteins from the MINT PDB validation parquet → manifest.json."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pyarrow.parquet as pq

from struct_pred_benchmarking.config import BenchmarkConfig


def _read_val_dimers(parquet_path: Path) -> list[dict]:
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    val_dimers = df[(df["split"] == "val") & (df["num_chains"] == 2)]
    rows: list[dict] = []
    for _, row in val_dimers.iterrows():
        seqs = tuple(row["sequences"])
        if len(seqs) != 2:
            continue
        rows.append(
            {
                "id": str(row["id"]),
                "sequences": [str(seqs[0]), str(seqs[1])],
                "relative_path": str(row["relative_path"]),
                "total_sequence_length": int(row["total_sequence_length"]),
            }
        )
    return rows


def select(cfg: BenchmarkConfig) -> list[dict]:
    rows = _read_val_dimers(cfg.split_parquet)
    rows = [
        r
        for r in rows
        if cfg.length_filter.min_total <= r["total_sequence_length"] <= cfg.length_filter.max_total
    ]
    if len(rows) < cfg.n_proteins:
        raise RuntimeError(
            f"Only {len(rows)} val dimers match length filter "
            f"[{cfg.length_filter.min_total}, {cfg.length_filter.max_total}]; "
            f"need {cfg.n_proteins}"
        )
    rng = random.Random(cfg.seed)
    rows.sort(key=lambda r: r["id"])  # deterministic order before sampling
    chosen = rng.sample(rows, cfg.n_proteins)
    chosen.sort(key=lambda r: r["total_sequence_length"])  # smallest first
    return chosen


def write_manifest(cfg: BenchmarkConfig) -> Path:
    chosen = select(cfg)
    manifest_path = cfg.run_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(chosen, indent=2))
    return manifest_path


def load_manifest(cfg: BenchmarkConfig) -> list[dict]:
    return json.loads((cfg.run_dir / "manifest.json").read_text())
