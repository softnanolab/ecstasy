"""Self-contained MSA Pairformer runner.

Reads a JSON bundle from stdin:
  { entry_id, sequences[], chain_ids[], msa_paths{} (per-chain a3m, optional),
    out_dir, config }

Bundle's config["model_config"]:
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

import json
import sys
from pathlib import Path

import numpy as np
import torch


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
    cfg = (bundle.get("config") or {}).get("model_config", {}) or {}

    max_msa_depth: int = int(cfg.get("max_msa_depth", 512))
    weights_dir = cfg.get("weights_dir")
    hhfilter_bin = cfg.get("hhfilter_bin")
    complex_a3m_dir = cfg.get("complex_a3m_dir")

    a3m_path: Path
    if complex_a3m_dir:
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

    sys.path.insert(0, str(Path("/home/u6jv/harsh.u6jv/colabfold-local/src")))
    from run_pairformer import parse_colabfold_header, chain_aware_select

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[msa_pairformer] device={device}  max_msa_depth={max_msa_depth}", flush=True)

    chain_breaks = parse_colabfold_header(a3m_path)

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

    hhfilter_kwargs = {"binary": hhfilter_bin} if hhfilter_bin else {}
    use_chain_aware = bool(chain_breaks)
    candidate_depth = max_msa_depth * 4 if use_chain_aware else max_msa_depth
    msa_obj = MSA(
        msa_file_path=str(a3m_path),
        max_seqs=candidate_depth,
        max_length=total_length,
        max_tokens=int(1e12),
        diverse_select_method="hhfilter" if hhfilter_bin else "greedy",
        hhfilter_kwargs=hhfilter_kwargs,
    )
    msa_tokenized = msa_obj.diverse_tokenized_msa
    n_seqs = msa_obj.n_diverse_seqs

    if use_chain_aware and n_seqs > max_msa_depth:
        tokens_np = msa_tokenized.numpy() if isinstance(msa_tokenized, torch.Tensor) else msa_tokenized
        indices = chain_aware_select(tokens_np, chain_breaks[0], n_select=max_msa_depth)
        msa_tokenized = msa_tokenized[indices] if isinstance(msa_tokenized, torch.Tensor) else msa_tokenized[indices]
        n_seqs = len(indices)
    print(f"[msa_pairformer] MSA depth: {n_seqs}", flush=True)

    msa_tensor = msa_tokenized.long() if isinstance(msa_tokenized, torch.Tensor) else torch.from_numpy(msa_tokenized).long()
    msa_onehot = one_hot(msa_tensor, num_classes=28).float().unsqueeze(0).to(device)
    masks = prepare_msa_masks(msa_tensor.unsqueeze(0))
    mask, msa_mask, full_mask, pairwise_mask = masks

    model = MSAPairformer.from_pretrained(device=device, weights_dir=weights_dir)
    model.eval()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        res = model.predict_cb_contacts(
            msa=msa_onehot,
            mask=mask.to(device),
            msa_mask=msa_mask.to(device),
            full_mask=full_mask.to(device),
            pairwise_mask=pairwise_mask.to(device),
            complex_chain_break_indices=[chain_breaks] if chain_breaks else None,
            return_seq_weights=True,
        )
    contact = res["predicted_cb_contacts"][0].cpu().float().numpy().astype(np.float16)
    print(f"[msa_pairformer] contact shape: {contact.shape}", flush=True)

    np.savez_compressed(
        out_dir / "contact.npz",
        probs=contact,
        length=np.int32(contact.shape[0]),
    )
    print(f"[msa_pairformer] WROTE {out_dir / 'contact.npz'}", flush=True)


if __name__ == "__main__":
    main()
