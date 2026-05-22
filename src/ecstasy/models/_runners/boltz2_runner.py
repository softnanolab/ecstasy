"""Self-contained Boltz-2 runner — invoked via env's python from the outer adapter.

Reads a JSON bundle from stdin:
  { entry_id, sequences[], chain_ids[], msa_paths{chain_id: path}, out_dir, config }

Writes:
  <out_dir>/raw/                                — boltz native outputs (mmcif + distogram_<id>.npz)
  <out_dir>/contact.npz                         — probs (L,L) float16, length int32

This script is intentionally dependency-light: stdlib + boltz CLI subprocess + numpy.
It does not import other ecstasy modules so it works inside envs/boltz without
installing ecstasy there.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def _emit_yaml(entry_id: str, sequences: list[str], chain_ids: list[str],
               msa_paths: dict[str, str]) -> str:
    """Emit a Boltz YAML, collapsing chains that share a sequence into one entry.

    Boltz YAML format: a chain group with id ['A', 'B'] (homodimer) shares its msa.
    Hetero chains get separate entries.
    """
    by_seq: dict[str, list[str]] = {}
    seq_to_msa: dict[str, str] = {}
    for cid, seq in zip(chain_ids, sequences):
        by_seq.setdefault(seq, []).append(cid)
        if cid in msa_paths:
            seq_to_msa[seq] = msa_paths[cid]

    lines = ["version: 1", "sequences:"]
    for seq, cids in by_seq.items():
        id_field = cids[0] if len(cids) == 1 else "[" + ", ".join(cids) + "]"
        lines.append("  - protein:")
        lines.append(f"      id: {id_field}")
        lines.append(f"      sequence: {seq}")
        msa = seq_to_msa.get(seq, "empty")
        lines.append(f"      msa: {msa}")
    return "\n".join(lines) + "\n"


def main():
    bundle = json.loads(sys.stdin.read())
    entry_id: str = bundle["entry_id"]
    sequences: list[str] = bundle["sequences"]
    chain_ids: list[str] = bundle["chain_ids"]
    msa_paths: dict[str, str] = bundle.get("msa_paths") or {}
    out_dir = Path(bundle["out_dir"])
    cfg = (bundle.get("config") or {}).get("model_config", {}) or {}
    cutoff_bin: int = int(cfg.get("contact_cutoff_bin", 19))

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    yaml_dir = out_dir / "_yaml"
    yaml_dir.mkdir(exist_ok=True)
    yaml_path = yaml_dir / f"{entry_id}.yaml"
    yaml_path.write_text(_emit_yaml(entry_id, sequences, chain_ids, msa_paths))

    boltz_bin = Path(sys.executable).parent / "boltz"
    cmd = [
        str(boltz_bin), "predict", str(yaml_dir),
        "--out_dir", str(raw_dir),
        "--model", "boltz2",
        "--devices", str(cfg.get("devices", 1)),
        "--recycling_steps", str(cfg.get("recycling_steps", 3)),
        "--sampling_steps", str(cfg.get("sampling_steps", 25)),
        "--diffusion_samples", str(cfg.get("diffusion_samples", 1)),
        "--num_workers", str(cfg.get("num_workers", 0)),
        "--output_format", "mmcif",
        "--override",
        "--dump_distogram",
    ]
    if cfg.get("no_kernels", True):
        cmd.append("--no_kernels")
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    distogram_paths = list(raw_dir.glob(f"**/distogram_{entry_id}.npz"))
    if not distogram_paths:
        raise FileNotFoundError(f"no distogram_{entry_id}.npz under {raw_dir}")
    d = np.load(distogram_paths[0])
    probs64 = d["probs"]  # (L, L, 64) float16
    contact = probs64[..., :cutoff_bin].sum(-1).astype(np.float16)  # (L, L) float16
    np.savez_compressed(
        out_dir / "contact.npz",
        probs=contact,
        length=np.int32(int(d["length"])),
    )
    shutil.rmtree(yaml_dir, ignore_errors=True)
    print(f"WROTE {out_dir / 'contact.npz'}  shape={contact.shape}  cutoff_bin={cutoff_bin}", flush=True)


if __name__ == "__main__":
    main()
