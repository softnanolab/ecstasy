"""Manifest entries → Boltz YAML input files.

Layout follows boltz docs (https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md):
    sequences:
      - protein:
          id: A
          sequence: <seqA>
      - protein:
          id: B
          sequence: <seqB>

No `msa:` field is set, so `--use_msa_server` triggers the ColabFold MSA
server for both chains.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from struct_pred_benchmarking.config import BenchmarkConfig
from struct_pred_benchmarking.data.select_val import load_manifest


_CHAIN_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def build_yaml(entry: dict) -> dict:
    seqs = entry["sequences"]
    return {
        "sequences": [
            {"protein": {"id": _CHAIN_LETTERS[i], "sequence": seq}}
            for i, seq in enumerate(seqs)
        ]
    }


def write_one(cfg: BenchmarkConfig, entry: dict) -> Path:
    out_path = cfg.run_dir / "inputs" / f"{entry['id']}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        yaml.safe_dump(build_yaml(entry), fh, sort_keys=False)
    return out_path


def write_all(cfg: BenchmarkConfig) -> list[Path]:
    return [write_one(cfg, e) for e in load_manifest(cfg)]
