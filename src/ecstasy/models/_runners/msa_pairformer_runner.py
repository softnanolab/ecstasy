"""Self-contained MSA Pairformer runner.

Reads a JSON bundle from stdin:
  { entry_id, sequences[], chain_ids[], msa_paths{} (per-chain a3m, optional),
    out_dir, config }

Bundle's "params":
  complex_a3m_dir   — optional. Directory of pre-generated complex a3ms in
                      ColabFold format (header `#L1,L2<TAB>copies1,copies2`).
                      If present and `<entry_id>.a3m` exists there, uses it.
  max_msa_depth     — passed to MSAPairformer.dataset.MSA (default 512)
  weights_dir       — optional local weights dir; otherwise downloaded from HF
  hhfilter_bin      — path to hhfilter; if None, uses greedy diversity selection

Writes:
  <out_dir>/contact.npz   — probs (L, L) float16, length int32

If no `complex_a3m_dir` is provided, this runner constructs a single-sequence
complex a3m on the fly (just the query, no homologs). That's a degenerate
single-sequence baseline for MSA Pairformer; for the real with-MSA column,
generate proper complex MSAs via softnanolab/colabfold-local first and pass the
directory via `complex_a3m_dir`.
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _parse_chain_breaks(a3m_path: Path) -> list[int]:
    """Chain-break indices from a ColabFold complex header `#L1,L2<TAB>c1,c2`.

    Returns the cumulative residue offsets at each chain boundary (excluding the
    end), expanding lengths by copy count. `#315,316\t1,1` -> [315];
    `#60\t2` (homodimer) -> [60]. Empty for a monomer.
    """
    header = None
    for ln in a3m_path.read_text().splitlines():
        if ln.startswith("#"):
            header = ln
            break
    if not header:
        return []
    fields = header.lstrip("#").split("\t")
    lens = [int(x) for x in fields[0].split(",")]
    copies = [int(x) for x in fields[1].split(",")] if len(fields) > 1 else [1] * len(lens)
    expanded = [L for L, c in zip(lens, copies) for _ in range(c)]
    breaks, pos = [], 0
    for L in expanded[:-1]:
        pos += L
        breaks.append(pos)
    return breaks


def _build_singleseq_a3m(entry_id: str, sequences: list[str], dest: Path) -> Path:
    """Write a minimal ColabFold-format complex a3m with just the query.

    Always emits one length-per-chain in the header and the concatenated body
    sequence, even for homodimers — colabfold-local's parse_colabfold_header
    treats `lengths <= 1` as a monomer and skips chain-break tracking. To get an
    (L_A + L_B, L_A + L_B) prediction we always look like a heterodimer with
    `copies = 1,1,...`.
    """
    lengths = ",".join(str(len(s)) for s in sequences)
    copies = ",".join("1" for _ in sequences)
    header = f"#{lengths}\t{copies}"
    body_seq = "".join(sequences)
    dest.write_text(f"{header}\n>{entry_id}\n{body_seq}\n")
    return dest


def main():
    bundle = json.loads(sys.stdin.read())
    entry_id: str = bundle["entry_id"]
    sequences: list[str] = bundle["sequences"]
    out_dir = Path(bundle["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = bundle.get("params") or {}

    max_msa_depth: int = int(cfg.get("max_msa_depth", 512))
    weights_dir = cfg.get("weights_dir")
    hhfilter_bin = cfg.get("hhfilter_bin")
    # complex paired a3m: the pipeline resolves it from the shared MSA store and
    # passes the path directly; fall back to a config dir, then single-sequence.
    complex_a3m = bundle.get("complex_a3m") or cfg.get("complex_a3m")
    complex_a3m_dir = cfg.get("complex_a3m_dir")

    a3m_path: Path
    if complex_a3m:
        a3m_path = Path(complex_a3m)
        if not a3m_path.exists():
            raise FileNotFoundError(f"complex_a3m {a3m_path} missing — generate complex MSAs first.")
    elif complex_a3m_dir:
        candidate = Path(complex_a3m_dir) / f"{entry_id}.a3m"
        if not candidate.exists():
            raise FileNotFoundError(
                f"complex_a3m_dir set but {candidate} missing — generate complex MSAs first "
                "(see softnanolab/colabfold-local pipeline)."
            )
        a3m_path = candidate
    else:
        a3m_path = _build_singleseq_a3m(entry_id, sequences, out_dir / f"{entry_id}.a3m")
    print(f"[msa_pairformer] a3m: {a3m_path}", flush=True)

    from torch.nn.functional import one_hot
    from MSA_Pairformer.model import MSAPairformer
    from MSA_Pairformer.dataset import MSA, prepare_msa_masks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[msa_pairformer] device={device}  max_msa_depth={max_msa_depth}", flush=True)

    chain_breaks = _parse_chain_breaks(a3m_path)

    # Biopython 1.85+ SeqIO.parse("fasta") rejects leading `#` comments that
    # ColabFold writes as its complex header. MSA_Pairformer's dataset.parse_a3m_file
    # uses the strict "fasta" parser. We've already parsed chain_breaks above; now
    # strip the `#` line into a clean a3m for the MSA loader.
    raw = a3m_path.read_text()
    stripped_lines = [ln for ln in raw.splitlines() if not ln.startswith("#")]
    stripped_a3m = out_dir / f"_stripped_{a3m_path.name}"
    stripped_a3m.write_text("\n".join(stripped_lines) + "\n")
    a3m_path = stripped_a3m

    with open(a3m_path) as f:
        for line in f:
            if line.startswith("#") or line.startswith(">"):
                continue
            total_length = len(
                line.strip().translate(str.maketrans("", "", "abcdefghijklmnopqrstuvwxyz.-"))
            )
            break

    # Notebook fidelity: hhfilter is mandatory (no greedy fallback), plain `-diff 512`
    # selection (NO ×4 inflation, NO chain-aware reselection — those change the kept
    # subset and thus the coevolution signal), and np.random.seed(42) before MSA build.
    if not hhfilter_bin:
        raise RuntimeError("hhfilter_bin is required for faithful MSA Pairformer reproduction "
                           "(the notebook always uses hhfilter -diff 512); none provided.")
    np.random.seed(42)
    msa_obj = MSA(
        msa_file_path=str(a3m_path),
        max_seqs=max_msa_depth,
        max_length=total_length,
        max_tokens=int(1e12),
        diverse_select_method="hhfilter",
        hhfilter_kwargs={"binary": hhfilter_bin},
    )
    msa_tokenized = msa_obj.diverse_tokenized_msa
    n_seqs = msa_obj.n_diverse_seqs
    print(f"[msa_pairformer] MSA depth: {n_seqs}", flush=True)

    msa_tensor = msa_tokenized.long() if isinstance(msa_tokenized, torch.Tensor) else torch.from_numpy(msa_tokenized).long()
    msa_onehot = one_hot(msa_tensor, num_classes=28).float().unsqueeze(0).to(device)
    masks = prepare_msa_masks(msa_tensor.unsqueeze(0))
    mask, msa_mask, full_mask, pairwise_mask = masks

    model = MSAPairformer.from_pretrained(device=device, weights_dir=weights_dir)
    model.eval()
    # Notebook runs with query biasing ON (cell 4 use_query_biasing=True); make it explicit.
    if hasattr(model, "turn_on_query_biasing"):
        model.turn_on_query_biasing()

    # autocast is CUDA-only here; on the CPU fallback path use a no-op context
    # so the runner still produces contact.npz without raising on CPU nodes.
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    mk = dict(
        msa=msa_onehot,
        mask=mask.to(device),
        msa_mask=msa_mask.to(device),
        full_mask=full_mask.to(device),
        pairwise_mask=pairwise_mask.to(device),
        complex_chain_break_indices=[chain_breaks] if chain_breaks else None,
        return_seq_weights=True,
    )
    with torch.no_grad(), autocast_ctx:
        # Cb-Cb head (layer 15) — the definition-matched, scored map (vs MENTOS Cb<8A GT).
        res = model.predict_cb_contacts(**mk)
        # ConFind head (layer 18) — free secondary column (side-chain-contact definition).
        confind = None
        if hasattr(model, "predict_confind_contacts"):
            try:
                cres = model.predict_confind_contacts(**mk)
                confind = cres["predicted_confind_contacts"][0].cpu().float().numpy().astype(np.float16)
            except Exception as e:  # noqa: BLE001
                print(f"[msa_pairformer] confind head skipped: {e}", flush=True)
    contact = res["predicted_cb_contacts"][0].cpu().float().numpy().astype(np.float16)
    print(f"[msa_pairformer] contact shape: {contact.shape}", flush=True)

    extra = {"probs_confind": confind} if confind is not None else {}
    np.savez_compressed(
        out_dir / "contact.npz",
        probs=contact,
        length=np.int32(contact.shape[0]),
        **extra,
    )
    print(f"[msa_pairformer] WROTE {out_dir / 'contact.npz'}", flush=True)


if __name__ == "__main__":
    main()
