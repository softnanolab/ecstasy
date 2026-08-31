"""Self-contained ESMFold runner — invoked via env's python from the outer adapter.

Reads a JSON bundle from stdin:
  { entry_id, sequences[], chain_ids[], msa_paths{} (ignored), out_dir, config }

Bundle's "params" (all optional):
  num_recycles         — recycle iterations (default ESMFold's 4)
  chain_linker_length  — poly-G linker residues between chains (default 25, ESMFold default)
  residue_index_offset — positional-index jump between chains so the trunk treats them as
                         separate chains (ESMFold default 512); applied in addition to the linker
  chunk_size           — axial-attention chunk; reduces memory at small wallclock cost
  contact_cutoff_bin   — distogram bin index below which Cβ-Cβ counts as contact.
                        ESMFold uses linspace(2.3125, 21.6875, 63) of squared distance
                        boundaries; bin 19 ≈ 8 Å (matches MENTOS's threshold).

Writes:
  <out_dir>/contact.npz   — probs (L, L) float16, length int32

ESMFold concatenates chains with a "GGGG…" linker so the trunk sees a single L_total =
ΣL_i + (k-1)·linker_len token sequence. The runner softmaxes the distogram, sums bins
0..cutoff-1, and trims the linker positions to recover an (L, L) over residues only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    bundle = json.loads(sys.stdin.read())
    entry_id: str = bundle["entry_id"]
    sequences: list[str] = bundle["sequences"]
    out_dir = Path(bundle["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = bundle.get("params") or {}

    cutoff_bin: int = int(cfg.get("contact_cutoff_bin", 19))
    chain_linker_len: int = int(cfg.get("chain_linker_length", 25))
    residue_index_offset: int = int(cfg.get("residue_index_offset", 512))
    num_recycles = cfg.get("num_recycles")
    chunk_size = cfg.get("chunk_size")
    profile = bool(bundle.get("profile"))
    if profile:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import _flops

    import esm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[esmfold] device={device}  recycles={num_recycles}  chunk={chunk_size}", flush=True)

    model = esm.pretrained.esmfold_v1()
    model = model.eval().to(device)
    if chunk_size is not None:
        model.set_chunk_size(int(chunk_size))

    seq = ":".join(sequences)
    chain_linker = "G" * chain_linker_len
    print(f"[esmfold] linker={chain_linker_len}G  residue_index_offset={residue_index_offset}", flush=True)

    def _is_oom(exc: BaseException) -> bool:
        # torch 1.12 has no torch.cuda.OutOfMemoryError (added in 1.13), so match the
        # message. Keep it narrow — a non-OOM RuntimeError must still propagate.
        return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

    def _infer():
        if profile:
            # Count the whole infer: the structure module is ON the contact path
            # (it runs inside the recycle loop and feeds the next iteration's pair
            # rep → final distogram), so it must be included. lddt_head/ptm_head are
            # terminal and negligible (<0.01%). See FLOPS_BENCHMARK_PLAN.md §3.5.
            (out, _), payload = _flops.profile_call(
                model.infer,
                seq,
                num_recycles=num_recycles,
                chain_linker=chain_linker,
                residue_index_offset=residue_index_offset,
            )
            return out, payload
        out, _ = model.infer(
            seq,
            num_recycles=num_recycles,
            chain_linker=chain_linker,
            residue_index_offset=residue_index_offset,
        )
        return out, None

    # Axial-attention chunking is a pure memory/speed trade — same arithmetic, so the
    # contact map and the FLOP count are unchanged by it. Rather than chunk everything
    # (which would slow the ~100 short entries for the benefit of the long tail) or guess
    # a length threshold, start unchunked and step down only on OOM. recent_pp reaches
    # L=1006, and ESMFold OOMs around L~970 on a 40GB A100: a softmax there wants a
    # single 14.99 GiB allocation.
    flops_payload = None
    attempts = [chunk_size] if chunk_size is not None else [None, 128, 64, 32]
    with torch.no_grad():
        for i, chunk in enumerate(attempts):
            if chunk is not None:
                model.set_chunk_size(int(chunk))
            try:
                output, flops_payload = _infer()
                break
            except Exception as exc:  # noqa: BLE001
                if not _is_oom(exc) or i == len(attempts) - 1:
                    raise
                print(f"[esmfold] OOM at chunk={chunk}; retrying with "
                      f"chunk={attempts[i + 1]}", flush=True)
                torch.cuda.empty_cache()

    logits = output["distogram_logits"][0]                 # (L_total, L_total, 64)
    probs = torch.softmax(logits.float(), dim=-1)
    contact_full = probs[..., :cutoff_bin].sum(-1).cpu().numpy()

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
    print(f"[esmfold] WROTE {out_dir / 'contact.npz'}  shape={contact.shape}  cutoff_bin={cutoff_bin}",
          flush=True)

    if profile and flops_payload is not None:
        eff_recycles = int(num_recycles) if num_recycles is not None else 4  # esm default
        sidecar = _flops.write_flops_sidecar(
            out_dir, flops_payload,
            L=int(contact.shape[0]), msa_depth=0, recycles=eff_recycles, model="esmfold",
        )
        print(f"[esmfold] WROTE {sidecar}  flops={flops_payload['flops']:.3e}", flush=True)


if __name__ == "__main__":
    main()
