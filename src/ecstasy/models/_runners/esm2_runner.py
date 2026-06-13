"""Self-contained ESM2 contact-prediction runner — invoked via env's python from the
outer adapter. Runs in its own .venv-esm2 (torch + fair-esm only; see
scripts/install/esm2.sh) — fair-esm ships every ESM2 size + the contact regression head.

Reads a JSON bundle from stdin:
  { entry_id, sequences[], chain_ids[], msa_paths{} (ignored), out_dir, params }

Bundle's "params" (all optional):
  model_name           — fair-esm ESM2 checkpoint name (default esm2_t33_650M_UR50D);
                         one of esm2_t6_8M / t12_35M / t30_150M / t33_650M / t36_3B _UR50D
  chain_linker_length  — poly-G linker residues inserted between chains (default 25).

Writes:
  <out_dir>/contact.npz   — probs (L, L) float16, length int32

ESM2's `predict_contacts` is the supervised contact head from the original ESM repo
(https://github.com/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb):
symmetrised + APC-corrected attention maps fed through a logistic-regression head, giving
a per-residue-pair **contact probability in [0, 1]** (not a distogram, no recycles).

Multimer hack (mirrors ESMFold): the chains are concatenated into one sequence separated
by a "GGGG…" poly-G linker so the model sees a single L_total = ΣL_i + (k-1)·linker token
construct. ESM2 uses **rotary (relative) position embeddings**, so unlike ESMFold there is
no separate absolute residue-index offset to apply — the linker length *is* the positional
skip between chains. The runner strips the linker positions from the (L_total, L_total)
contact map to recover an (L, L) matrix over residues only.

We keep the head's continuous probability (the natural contact score) rather than a hard
0/1: interchain P@K ranks pairs by score and takes the top-K, so a thresholded map would
make the ranking — and hence P@K — ill-defined among ties.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    bundle = json.loads(sys.stdin.read())
    sequences: list[str] = bundle["sequences"]
    out_dir = Path(bundle["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = bundle.get("params") or {}

    model_name: str = str(cfg.get("model_name", "esm2_t33_650M_UR50D"))
    chain_linker_len: int = int(cfg.get("chain_linker_length", 25))
    profile = bool(bundle.get("profile"))
    if profile:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import _flops

    import esm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[esm2] device={device}  model={model_name}  linker={chain_linker_len}G", flush=True)

    model, alphabet = getattr(esm.pretrained, model_name)()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    linker = "G" * chain_linker_len
    seq = linker.join(sequences)
    _, _, tokens = batch_converter([("complex", seq)])
    tokens = tokens.to(device)

    flops_payload = None
    with torch.no_grad():
        if profile:
            # Single forward, no recycles: the whole contact-head forward IS the
            # contact-dependency subgraph (see FLOPS_BENCHMARK_PLAN.md §3.5).
            out, flops_payload = _flops.profile_call(
                model, tokens, return_contacts=True
            )
            contacts = out["contacts"]
        else:
            contacts = model(tokens, return_contacts=True)["contacts"]

    # The contact head returns (B, L_total, L_total) already stripped of BOS/EOS,
    # so positions index directly into `seq`.
    contact_full = contacts[0].float().cpu().numpy()

    keep: list[int] = []
    pos = 0
    for i, s in enumerate(sequences):
        keep.extend(range(pos, pos + len(s)))
        pos += len(s)
        if i < len(sequences) - 1:
            pos += chain_linker_len
    keep_arr = np.asarray(keep)
    contact = contact_full[keep_arr][:, keep_arr].astype(np.float16)

    np.savez_compressed(
        out_dir / "contact.npz",
        probs=contact,
        length=np.int32(contact.shape[0]),
    )
    print(f"[esm2] WROTE {out_dir / 'contact.npz'}  shape={contact.shape}", flush=True)

    if profile and flops_payload is not None:
        sidecar = _flops.write_flops_sidecar(
            out_dir, flops_payload,
            L=int(contact.shape[0]), msa_depth=0, recycles=0, model=model_name,
        )
        print(f"[esm2] WROTE {sidecar}  flops={flops_payload['flops']:.3e}", flush=True)


if __name__ == "__main__":
    main()
