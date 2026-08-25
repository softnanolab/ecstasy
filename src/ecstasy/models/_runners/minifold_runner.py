"""Self-contained MiniFold runner — invoked via env's python from the outer adapter.

MiniFold (Wohlwend et al., TMLR 2025) is a **single-chain** folder that has never seen a
complex. It is run here on dimers through the same chain-break hack the ``esmfold`` row
uses, to supply a third-party reference point for MENTOS's docking: a strong monomer
folder given no docking ability, scored identically. Whatever DockQ it posts is what the
task gives away to a model that cannot dock.

Reads a JSON bundle from stdin:
  { entry_id, sequences[], chain_ids[], msa_paths{} (ignored), out_dir, params, profile }

Bundle's "params":
  minifold_src         — path to the PATCHED minifold source tree (required, see below)
  checkpoint           — MiniFold .ckpt (48L or 12L)
  torch_hub_dir        — torch.hub cache holding the ESM2 backbone
  num_recycles         — trunk recycling iterations (MiniFold's default 3)
  chain_linker_length  — poly-G linker residues between chains (25, MiniFold's default)
  residue_index_offset — positional-index jump across the break, in addition to the linker
  contact_cutoff_bin   — distogram bin index below which a distance counts as a contact

Writes:
  <out_dir>/contact.npz     — probs (L, L) float16, length int32
  <out_dir>/structure.npz   — atom37 bundle for DockQ (see ecstasy.structure.pdb)

**contact_cutoff_bin is 17 here, not the 19 used everywhere else in ecstasy.** It is not a
free parameter: bins are edges, and the two models bin differently. MiniFold's 48L
checkpoint has ``no_bins=64, max_dist=25``, so its boundaries are ``linspace(2, 25, 63)``
and ``probs[..., :k].sum(-1)`` is ``P(d <= boundaries[k-1])``:

    ESMFold  linspace(2.3125, 21.6875, 63)  bin 19 -> 7.9375 A   <- the ecstasy convention
    MiniFold linspace(2, 25, 63)            bin 19 -> 8.6774 A   <- 0.74 A too loose
    MiniFold linspace(2, 25, 63)            bin 17 -> 7.9355 A   <- matches, to 0.002 A

Copying 19 across would have quietly inflated MiniFold's P@K against every other model.

Caveat that no bin index can fix: MiniFold's distogram is trained on **CA-CA** distances
(``train/data.py`` builds coords as ``all_atom_positions[:, 0:3]`` = N/CA/C and
``train/model.py`` selects index 1), whereas ecstasy's ground truth — and ESMFold's and
Boltz-2's heads — are **Cb-Cb**. That is a property of the pretrained head. It will
modestly depress MiniFold's P@K relative to the Cb models and must be stated wherever the
comparison is published. It does not affect the structure/DockQ numbers, which are
coordinate-based.

The chain break is linker AND index jump, both. ``FoldingTrunk.forward`` hardcoded
``residx = arange(L)`` upstream; the patched fork accepts an injected ``residx``, and
``RelativePosition`` clamps the pairwise difference at 32 bins, so a +512 jump saturates
the clamp many times over and the trunk reads two unrelated chains. Verified numerically
on 9zdi: the residue-index step from the last linker position to the first residue of
chain B is 513, and from the last real residue of chain A it is 513 + linker_length.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


def _linker_trim_index(sequences: list[str], linker_len: int) -> np.ndarray:
    """Token indices of real residues in the linker-joined sequence, chains concatenated."""
    keep: list[int] = []
    pos = 0
    for i, s in enumerate(sequences):
        keep.extend(range(pos, pos + len(s)))
        pos += len(s)
        if i < len(sequences) - 1:
            pos += linker_len
    return np.asarray(keep, dtype=int)


def _residx(sequences: list[str], linker_len: int, offset: int) -> np.ndarray:
    """Residue index over the linker-joined sequence, jumping `offset` at each break.

    The linker itself continues the preceding chain's numbering — it is inert padding,
    and the jump is what the trunk actually reads.
    """
    out: list[np.ndarray] = []
    base = 0
    for i, s in enumerate(sequences):
        n = len(s) + (linker_len if i < len(sequences) - 1 else 0)
        out.append(np.arange(n) + base)
        base += n + offset
    return np.concatenate(out)


def main():
    bundle = json.loads(sys.stdin.read())
    entry_id: str = bundle["entry_id"]
    sequences: list[str] = list(bundle["sequences"])
    out_dir = Path(bundle["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = bundle.get("params") or {}

    minifold_src = str(cfg["minifold_src"])
    checkpoint = str(cfg["checkpoint"])
    cutoff_bin = int(cfg.get("contact_cutoff_bin", 17))
    linker_len = int(cfg.get("chain_linker_length", 25))
    residue_index_offset = int(cfg.get("residue_index_offset", 512))
    num_recycles = int(cfg.get("num_recycles", 3))
    profile = bool(bundle.get("profile"))
    if profile:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import _flops

    sys.path.insert(0, minifold_src)
    if cfg.get("torch_hub_dir"):
        torch.hub.set_dir(str(cfg["torch_hub_dir"]))
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")

    import minifold

    # MiniFold's published pyproject declares `packages = ["minifold"]`, so `pip install .`
    # installs ONLY the top-level package: `import minifold` succeeds while
    # `minifold.utils`/`.model`/`.data` are absent. The source tree is therefore the only
    # working copy — and it is also where the residx patch lives. Without this assert a
    # stale installed copy silently gives unpatched behaviour, i.e. the linker-only
    # variant, and a quietly wrong baseline.
    resolved = Path(minifold.__file__).resolve().parent.parent
    if resolved != Path(minifold_src).resolve():
        raise RuntimeError(
            f"minifold resolved to {minifold.__file__} (tree {resolved}), not "
            f"{minifold_src}. The installed copy is incomplete and unpatched; "
            f"the source tree must win on sys.path.")

    import predict as mf
    from minifold.data.config import model_config
    from minifold.model.model import FoldingTrunk
    from minifold.utils.residue_constants import restype_order_with_x

    # The residx patch must be present, or `FoldingTrunk.forward` falls back to
    # `arange(L)`, the index jump never reaches the trunk, and what gets measured is the
    # linker-only variant — silently, and scored as though it were the real thing.
    import inspect
    if "residx" not in inspect.signature(FoldingTrunk.forward).parameters:
        raise RuntimeError(
            f"{minifold_src}: FoldingTrunk.forward takes no `residx` — the residx patch "
            f"is not applied. Run scripts/install/minifold.sh, which applies it.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[minifold] {entry_id} device={device} recycles={num_recycles} "
          f"linker={linker_len}G offset={residue_index_offset} cutoff_bin={cutoff_bin}",
          flush=True)

    alphabet, model = mf.create_model(checkpoint, device)
    of_config = model_config("initial_training", train=False, low_prec=False,
                             long_sequence_inference=False).data

    seq = ("G" * linker_len).join(sequences)
    residx = _residx(sequences, linker_len, residue_index_offset)

    enc, mask, of = mf.prepare_input(seq, of_config, alphabet)
    # The OpenFold feature pipeline may pad; hold every tensor at one length.
    n_tok = len(enc)
    if n_tok != len(seq):
        raise RuntimeError(f"{entry_id}: encoded {n_tok} tokens != sequence {len(seq)}")
    mask = mask[:n_tok]
    of = {k: v[:n_tok] for k, v in of.items()}
    batch = {
        "seq": enc[None].to(device),
        "mask": mask[None].to(device),
        "residx": torch.tensor(residx[:n_tok], dtype=torch.long)[None].to(device),
        "batch_of": {k: v[None].to(device) for k, v in of.items()},
    }

    flops_payload = None
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.autocast(autocast_device, dtype=torch.bfloat16):
        if profile:
            out, flops_payload = _flops.profile_call(model, batch,
                                                     num_recycling=num_recycles)
        else:
            out = model(batch, num_recycling=num_recycles)

    keep = _linker_trim_index(sequences, linker_len)

    # `preds` is the trunk distogram: (B, L_total, L_total, 64) logits.
    logits = out["preds"].float()[0]
    probs = torch.softmax(logits, dim=-1)
    contact_full = probs[..., :cutoff_bin].sum(-1).cpu().numpy()
    contact = contact_full[keep][:, keep].astype(np.float16)
    np.savez_compressed(out_dir / "contact.npz",
                        probs=contact, length=np.int32(contact.shape[0]))
    print(f"[minifold] WROTE {out_dir / 'contact.npz'}  shape={contact.shape}  "
          f"cutoff_bin={cutoff_bin}", flush=True)

    # Structure: atom37 over real residues only, with the chain break expressed as
    # asym_id + a per-chain residue_index. ecstasy renders it to PDB with the same
    # writer it uses for the natives, so DockQ compares structures, not serialisations.
    pos = out["final_atom_positions"].float().cpu().numpy()[0][keep]
    amask = out["final_atom_mask"].float().cpu().numpy()[0][keep].astype(bool)
    # aatype comes from the sequence, NOT from `of["aatype"]`: the OpenFold features
    # carry a trailing recycling axis (they are (L, n_recycle)), so indexing that here
    # would yield (L_keep, n_recycle) and corrupt every residue name in the PDB.
    # `restype_order_with_x` is AF2's 20+X ordering, which is what the writer expects.
    aatype = np.array([restype_order_with_x[r] for r in "".join(sequences)])
    asym_id = np.concatenate([np.full(len(s), i) for i, s in enumerate(sequences)])
    residue_index = np.concatenate([np.arange(len(s)) for s in sequences])
    np.savez_compressed(
        out_dir / "structure.npz",
        atom37_positions=pos.astype(np.float32),
        atom37_mask=amask,
        aatype=aatype.astype(np.int8),
        asym_id=asym_id.astype(np.int8),
        residue_index=residue_index.astype(np.int32),
    )
    print(f"[minifold] WROTE {out_dir / 'structure.npz'}  n_res={len(asym_id)}  "
          f"chains={len(sequences)}", flush=True)

    if profile and flops_payload is not None:
        # Scope note (FLOPS_BENCHMARK_PLAN.md): unlike ESMFold, MiniFold's structure
        # module is NOT on the contact-map dependency path — the distogram `preds` comes
        # out of `self.fold` before the structure module runs. Profiling the whole model
        # is right for the DockQ axis and an OVER-count for the P@K-vs-FLOPs axis, so the
        # scope is recorded rather than left for a reader to assume.
        sidecar = _flops.write_flops_sidecar(
            out_dir, flops_payload, model="minifold",
            length=int(contact.shape[0]), recycles=num_recycles,
            scope="full_model_incl_structure_module")
        print(f"[minifold] WROTE {sidecar}", flush=True)


if __name__ == "__main__":
    main()
