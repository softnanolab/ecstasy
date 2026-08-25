"""Fold PDB val dimers with single-chain MiniFold using an ESMFold-style chain break.

MiniFold is monomer-only. Two chains are presented as one sequence joined by a
poly-glycine linker, and the residue index is jumped by --index_jump at the break so
the trunk's RelativePosition embedding (clamped at +/-32) saturates and reads the two
halves as unrelated chains rather than one continuous polymer.

Usage: python minifold_dimer_predict.py --natives_dir DIR --out_dir DIR --checkpoint CKPT
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def native_sequences(path):
    """Per-chain one-letter sequence, read from CA records in file order."""
    chains = {}
    order = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("ATOM") and line[12:16] == " CA ":
            ch = line[21]
            if ch not in chains:
                chains[ch] = []
                order.append(ch)
            chains[ch].append(THREE2ONE.get(line[17:20].strip(), "X"))
    return [(c, "".join(chains[c])) for c in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--natives_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--minifold_src", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--linker", type=int, default=25)
    ap.add_argument("--index_jump", type=int, default=512)
    ap.add_argument("--num_recycling", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, args.minifold_src)
    torch.hub.set_dir(args.cache)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")

    import minifold

    # The published pyproject installs only the top-level package, so the subpackages
    # exist solely in the source tree -- which is also where the residx patch lives.
    assert Path(minifold.__file__).parent.parent == Path(args.minifold_src).resolve(), (
        f"minifold resolved to {minifold.__file__}, not {args.minifold_src}"
    )

    import predict as mf
    from minifold.data.config import model_config
    from minifold.utils.protein import Protein, to_pdb
    from minifold.utils.residue_constants import restype_order_with_x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    alphabet, model = mf.create_model(args.checkpoint, device)
    config = model_config("initial_training", train=False, low_prec=False,
                          long_sequence_inference=False).data

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    natives = sorted(Path(args.natives_dir).glob("*_native.pdb"))
    if args.limit:
        natives = natives[: args.limit]

    manifest = []
    n_failed = 0
    for n, npath in enumerate(natives, 1):
        pid = npath.name.replace("_native.pdb", "")
        chains = native_sequences(npath)
        if len(chains) != 2:
            print(f"[{n}/{len(natives)}] {pid} SKIP: {len(chains)} chains", flush=True)
            continue
        (ca, sa), (cb, sb) = chains
        la, lb = len(sa), len(sb)
        seq = sa + "G" * args.linker + sb

        residx = np.concatenate([
            np.arange(la + args.linker),
            np.arange(lb) + la + args.linker + args.index_jump,
        ])

        try:
            enc, mask, of = mf.prepare_input(seq, config, alphabet)
            # The OpenFold feature pipeline may pad; keep every tensor on one length.
            n_tok = len(enc)
            assert n_tok == len(seq), f"{pid}: encoded {n_tok} != seq {len(seq)}"
            mask = mask[:n_tok]
            of = {k: v[:n_tok] for k, v in of.items()}
            batch = {
                "seq": enc[None].to(device),
                "mask": mask[None].to(device),
                "residx": torch.tensor(residx, dtype=torch.long)[None].to(device),
                "batch_of": {k: v[None].to(device) for k, v in of.items()},
            }
            with torch.autocast("cuda" if device.type == "cuda" else "cpu",
                                dtype=torch.bfloat16):
                out = model(batch, num_recycling=args.num_recycling)
            pos = out["final_atom_positions"].float().cpu().numpy()[0]
            amask = out["final_atom_mask"].float().cpu().numpy()[0]
            plddt = out["plddt"].float().cpu().numpy()[0]
        except Exception as e:  # noqa: BLE001
            print(f"[{n}/{len(natives)}] {pid} FAIL: {type(e).__name__}: {e}", flush=True)
            n_failed += 1
            # A systematic error (bad weights, OOM on every target, shape mismatch) shows
            # up immediately; abort rather than burn the whole GPU allocation on it.
            if n_failed >= 3 and not manifest:
                raise SystemExit(f"aborting: first {n_failed} targets all failed")
            continue

        keep = np.concatenate([np.arange(la), np.arange(lb) + la + args.linker])
        kept_seq = sa + sb
        aatype = np.array([restype_order_with_x[r] for r in kept_seq])
        chain_index = np.array([0] * la + [1] * lb)
        residue_index = np.concatenate([np.arange(1, la + 1), np.arange(1, lb + 1)])
        b = plddt[keep][:, None].repeat(pos.shape[1], axis=1)

        pdb = to_pdb(Protein(
            aatype=aatype,
            atom_positions=pos[keep],
            atom_mask=amask[keep],
            residue_index=residue_index,
            chain_index=chain_index,
            b_factors=b,
        ))
        (out_dir / f"{pid}_pred.pdb").write_text(pdb)
        mean_plddt = float(plddt[keep].mean())
        manifest.append({"id": pid, "len_a": la, "len_b": lb,
                         "homodimer": sa == sb, "mean_plddt": mean_plddt})
        print(f"[{n}/{len(natives)}] {pid} La={la} Lb={lb} plddt={mean_plddt:.1f}", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(manifest)}/{len(natives)} predictions to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
