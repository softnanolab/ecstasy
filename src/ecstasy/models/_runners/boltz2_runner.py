"""Self-contained Boltz-2 runner — invoked via env's python from the outer adapter.

Reads a JSON bundle from stdin:
  { entry_id, sequences[], chain_ids[], msa_paths{chain_id: path}, out_dir,
    params{recycling_steps, sampling_steps, diffusion_samples, contact_cutoff_bin},
    infra{devices, num_workers, no_kernels} }

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
    cfg = bundle.get("params") or {}        # output-affecting params (preset + overrides)
    infra = bundle.get("infra") or {}       # machine knobs (devices/num_workers/no_kernels)
    cutoff_bin: int = int(cfg.get("contact_cutoff_bin", 19))
    profile = bool(bundle.get("profile"))

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    yaml_dir = out_dir / "_yaml"
    yaml_dir.mkdir(exist_ok=True)
    yaml_path = yaml_dir / f"{entry_id}.yaml"
    yaml_path.write_text(_emit_yaml(entry_id, sequences, chain_ids, msa_paths))

    recycling_steps = int(cfg.get("recycling_steps", 3))
    no_kernels = bool(infra.get("no_kernels", True))
    if profile:
        # Trunk-only FLOP profiling: run boltz in-process with structure skipped
        # (diffusion never runs; the distogram, computed before diffusion, is
        # identical). Produces distogram_<id>.npz + flops_<id>.json under raw_dir.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import _boltz_profile
        print(f"[boltz2] PROFILE trunk-only forward (recycling_steps={recycling_steps}, "
              f"no_kernels={no_kernels})", flush=True)
        _boltz_profile.run_profiled_predict(
            yaml_dir, raw_dir,
            recycling_steps=recycling_steps,
            sampling_steps=int(cfg.get("sampling_steps", 25)),
            diffusion_samples=int(cfg.get("diffusion_samples", 1)),
            no_kernels=no_kernels,
            devices=int(infra.get("devices", 1)),
            num_workers=int(infra.get("num_workers", 0)),
        )
    else:
        boltz_bin = Path(sys.executable).parent / "boltz"
        cmd = [
            str(boltz_bin), "predict", str(yaml_dir),
            "--out_dir", str(raw_dir),
            "--model", "boltz2",
            "--devices", str(infra.get("devices", 1)),
            "--recycling_steps", str(recycling_steps),
            "--sampling_steps", str(cfg.get("sampling_steps", 25)),
            "--diffusion_samples", str(cfg.get("diffusion_samples", 1)),
            "--num_workers", str(infra.get("num_workers", 0)),
            "--output_format", "mmcif",
            "--override",
            "--dump_distogram",
        ]
        if no_kernels:
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

    if profile:
        flops_paths = list(raw_dir.glob(f"**/flops_{entry_id}.json"))
        if not flops_paths:
            raise FileNotFoundError(f"profile mode but no flops_{entry_id}.json under {raw_dir}")
        payload = json.loads(flops_paths[0].read_text())
        by_module = payload.get("by_module") or {}
        # Hard sanity: the diffusion sampler must NOT run under skip_run_structure.
        diffusion = sum(v for k, v in by_module.items()
                        if k.split(".")[-1] in ("structure_module", "diffusion_conditioning"))
        if diffusion:
            raise RuntimeError(f"diffusion subtree counted {diffusion} FLOPs — skip_run_structure "
                               "failed; the trunk-only count would be wrong")
        # Confidence/bfactor heads still run but were subtracted in _boltz_profile (off-path).
        if payload.get("off_path_flops"):
            print(f"[boltz2] off-path heads subtracted: {payload['off_path_flops']:.3e} FLOPs "
                  f"(confidence/bfactor)", flush=True)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import _flops
        sidecar = _flops.write_flops_sidecar(
            out_dir,
            # "debug" is only present under ECSTASY_FLOPS_DEBUG; without it in this
            # whitelist the block is silently dropped here and the flag looks broken.
            {k: payload[k] for k in ("flops", "macs", "flops_total", "off_path_flops",
                                     "by_module", "debug")
             if k in payload},
            L=int(contact.shape[0]), msa_depth=None, recycles=recycling_steps, model="boltz2",
        )
        print(f"WROTE {sidecar}  flops={payload['flops']:.3e}", flush=True)


if __name__ == "__main__":
    main()
