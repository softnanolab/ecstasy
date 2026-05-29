"""Self-contained AlphaFold-2 / ColabFold-batch runner.

Reads a JSON bundle from stdin:
  { entry_id, sequences[], chain_ids[], msa_paths{} (optional), out_dir, config }

Bundle's "params":
  msa_mode      — "single_sequence" | "mmseqs2_uniref_env" | "mmseqs2_uniref"
                  Default: single_sequence (smoke); for the with-MSA column use
                  mmseqs2_uniref_env which hits api.colabfold.com unless the
                  env's MMSEQS2_LOCAL/local DB is set up.
  num_recycle   — recycling iterations (default 3)
  num_models    — number of model parameter sets to predict (default 1; AF2 has 5)
  num_relax     — Amber relaxation rounds (default 0; not needed for contact P@K)

Writes:
  <out_dir>/raw/                 — colabfold_batch native outputs (predicted PDB,
                                   PAE json, etc.)
  <out_dir>/contact.npz          — probs (L, L) float16, length int32

Contact prob is computed from the predicted Cβ-Cβ distance (Cα for Gly) via
``exp(-d / 8 Å)``: monotonic, in (0, 1], bounded so float16 storage is exact in
the contact range. Score-based ranking only — not a calibrated probability.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _cb_coords_from_pdb(pdb_path: Path) -> tuple[np.ndarray, list[str]]:
    """Extract one Cβ coord per residue (Cα for Gly). Returns ((L, 3), [chain_id...]).

    Uses biopython since biotite isn't available in this env.
    """
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("p", str(pdb_path))
    coords: list[tuple[float, float, float]] = []
    chain_ids: list[str] = []
    for model in s:
        for chain in model:
            for res in chain:
                # Skip hetero (waters, ligands)
                if res.id[0].strip():
                    continue
                if "CB" in res:
                    a = res["CB"]
                elif "CA" in res:
                    a = res["CA"]
                else:
                    continue
                coords.append(tuple(a.coord))
                chain_ids.append(chain.id)
        break  # only first model
    return np.array(coords, dtype=np.float64), chain_ids


def main():
    bundle = json.loads(sys.stdin.read())
    entry_id: str = bundle["entry_id"]
    sequences: list[str] = bundle["sequences"]
    out_dir = Path(bundle["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    cfg = bundle.get("params") or {}

    msa_mode: str = cfg.get("msa_mode", "single_sequence")
    num_recycle: int = int(cfg.get("num_recycle", 3))
    num_models: int = int(cfg.get("num_models", 1))
    num_relax: int = int(cfg.get("num_relax", 0))

    # ColabFold expects multimer chains separated by ':'.
    seq = ":".join(sequences)
    fasta = out_dir / "input.fasta"
    fasta.write_text(f">{entry_id}\n{seq}\n")

    # Locate colabfold_batch in this venv.
    colabfold_bin = Path(sys.executable).parent / "colabfold_batch"
    if not colabfold_bin.exists():
        raise FileNotFoundError(f"colabfold_batch not found at {colabfold_bin}")

    cmd = [
        str(colabfold_bin),
        str(fasta),
        str(raw_dir),
        "--msa-mode", msa_mode,
        "--num-recycle", str(num_recycle),
        "--num-models", str(num_models),
        "--num-relax", str(num_relax),
        "--model-type", cfg.get("model_type", "alphafold2_multimer_v3"),
    ]
    if cfg.get("zip"):
        cmd.append("--zip")
    print(f"[colabfold] RUN: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    pdb_paths = sorted(raw_dir.glob(f"{entry_id}_unrelaxed_rank_001_*.pdb"))
    if not pdb_paths:
        # fallback: any rank_001 pdb
        pdb_paths = sorted(raw_dir.glob("*_rank_001_*.pdb"))
    if not pdb_paths:
        raise FileNotFoundError(f"no rank_001 .pdb under {raw_dir}")
    pdb = pdb_paths[0]
    print(f"[colabfold] parsing {pdb}", flush=True)

    cb, chain_ids = _cb_coords_from_pdb(pdb)
    print(f"[colabfold] L={len(cb)}  chains={set(chain_ids)}", flush=True)

    # Sanity-check shape against input
    L_expected = sum(len(s) for s in sequences)
    if len(cb) != L_expected:
        print(
            f"[colabfold] WARN: parsed {len(cb)} residues, expected {L_expected}; "
            "this typically indicates missing Cα/Cβ in the predicted structure.",
            flush=True,
        )

    diff = cb[:, None, :] - cb[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))                     # (L, L)
    contact = np.exp(-dist / 8.0).astype(np.float16)          # (L, L)

    np.savez_compressed(
        out_dir / "contact.npz",
        probs=contact,
        length=np.int32(contact.shape[0]),
    )
    print(f"[colabfold] WROTE {out_dir / 'contact.npz'}  shape={contact.shape}",
          flush=True)


if __name__ == "__main__":
    main()
