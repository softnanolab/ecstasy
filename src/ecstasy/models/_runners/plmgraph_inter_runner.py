"""Self-contained PLMGraph-Inter runner — invoked via its env's python by the adapter.

PLMGraph-Inter (Si & Yan, eLife 2024) is DRN-1D2D_Inter's successor: the same
MSA/coevolution pipeline (CCMpred + alnstats + hh-suite + ESM-1b/ESM-MSA-1b) PLUS
ESM-IF1 inverse-folding embeddings and GVP structure graphs. It therefore needs a
per-chain PDB **structure** in addition to the per-chain MSA.

This runner reproduces modules/plmgraph_inter/predict.py, but:
  * MSAs come from the Boltz per-chain CSVs (``msa: boltz_csv``) — we reuse Boltz's
    taxonomy pairing (CSV ``key`` column) to build the paired complex a3m, bypassing
    PLMGraph's header-parsed pair_msa (zero pairing on ColabFold-style a3ms);
  * per-chain PDBs are resolved from ``cfg["struct_dir"]/<entry_id>_<chain_id>.pdb``
    (ESMFold monomers for sequence-only splits — generated as a prerequisite);
  * tool/weight paths come from ``params`` (registry preset), nothing hardcoded;
  * the model emits the (lenA, lenB) inter-chain block; we embed it into the full
    square (L, L) contact-probability map ecstasy's scorer consumes.

PORT (torch>=2.6): torch.load is wrapped weights_only=False so fair-esm's local
checkpoints load; per-chain a3m homolog headers are ``>h<i>`` so PLMGraph's LoadHHM
(which prefix-matches '>'+name) picks the query, not a homolog.

Reads a JSON bundle on stdin: { entry_id, sequences[2], chain_ids[2],
msa_paths{chain_id: boltz_csv}, out_dir, params, infra }.  Writes <out_dir>/contact.npz.

NOTE: PLMGraph is intentionally OUTSIDE the FLOPs-profiling scope (ignores
``bundle["profile"]``).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def _run(*args) -> None:
    """Run an external tool list-form, aborting loudly on non-zero exit."""
    subprocess.run([str(a) for a in args], check=True)


def _write_fasta(path: Path, name: str, seq: str) -> None:
    path.write_text(f">{name}\n{seq}\n")


# --- Boltz CSV -> DRN/PLMGraph MSA conversion (shared with the DRN pipeline) -----
def _read_boltz_csv(path: str | Path) -> list[tuple[int, str]]:
    """Parse a Boltz per-chain MSA CSV into ``(key, sequence)`` rows (key>=0 paired,
    -1 unpaired; row 0 is the query)."""
    rows: list[tuple[int, str]] = []
    for line in Path(path).read_text().splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        key, _, seq = line.partition(",")
        rows.append((int(key), seq))
    return rows


def _chain_a3m_from_rows(rows: list[tuple[int, str]], query_name: str) -> str:
    """Per-chain a3m (query first). Homolog headers MUST NOT start with query_name —
    LoadHHM prefix-matches '>'+name to find the query, so use ``>h<i>``."""
    out: list[str] = []
    for i, (key, seq) in enumerate(rows):
        out.append(f">{query_name}" if i == 0 else f">h{i}_k{key}")
        out.append(seq)
    return "\n".join(out) + "\n"


def _paired_a3m_from_csvs(rowsA, rowsB, query_name: str) -> str:
    """Join the two chains' boltz CSV rows on shared key>=0 -> paired complex a3m,
    reusing Boltz's taxonomy pairing instead of PLMGraph's pair_msa."""
    byB = {k: s for k, s in rowsB if k >= 0}
    out: list[str] = []
    for k, sA in rowsA:
        if k < 0 or k not in byB:
            continue
        out.append(f">{query_name}" if k == 0 else f">pair_k{k}")
        out.append(sA + byB[k])
    return "\n".join(out) + "\n"


_A3M_INSERTION = str.maketrans("", "", "abcdefghijklmnopqrstuvwxyz.")


def _a3m_to_aln(a3m_text: str) -> str:
    """a3m -> CCMpred .aln (drop headers, strip insertions; keep '-' match gaps)."""
    out: list[str] = []
    seq: list[str] = []
    for line in a3m_text.splitlines():
        if line.startswith(">"):
            if seq:
                out.append("".join(seq).translate(_A3M_INSERTION))
                seq = []
        elif line:
            seq.append(line.strip())
    if seq:
        out.append("".join(seq).translate(_A3M_INSERTION))
    return "\n".join(out) + "\n"


def _embed_block(block: np.ndarray, lenA: int, lenB: int) -> np.ndarray:
    """Embed the (lenA, lenB) inter-chain block into the symmetric (L, L) map."""
    if block.shape != (lenA, lenB):
        raise RuntimeError(f"PLMGraph block {block.shape} != expected ({lenA}, {lenB})")
    L = lenA + lenB
    probs = np.zeros((L, L), dtype=np.float32)
    probs[:lenA, lenA:] = block
    probs[lenA:, :lenA] = block.T
    return probs.astype(np.float16)


def main() -> None:
    import torch

    # PORT SHIM: fair-esm local checkpoints were pickled under torch<2; torch>=2.6
    # defaults weights_only=True and refuses them. Restore the old default.
    _orig = torch.load
    torch.load = lambda *a, **k: (k.setdefault("weights_only", False), _orig(*a, **k))[1]

    bundle = json.loads(sys.stdin.read())
    entry_id = bundle["entry_id"]
    sequences = bundle["sequences"]
    chain_ids = bundle["chain_ids"]
    msa_paths = bundle.get("msa_paths") or {}
    out_dir = Path(bundle["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = bundle.get("params") or {}

    if len(sequences) != 2:
        raise ValueError(f"PLMGraph handles exactly 2 chains, got {len(sequences)} for {entry_id}")
    if not msa_paths.get(chain_ids[0]) or not msa_paths.get(chain_ids[1]):
        raise ValueError(f"PLMGraph needs a Boltz MSA CSV for both chains; have {sorted(msa_paths)}")

    # Per-chain PDB structures (ESMFold monomers for sequence-only splits).
    struct_dir = Path(cfg["struct_dir"])
    pdbA = struct_dir / f"{entry_id}_{chain_ids[0]}.pdb"
    pdbB = struct_dir / f"{entry_id}_{chain_ids[1]}.pdb"
    for p in (pdbA, pdbB):
        if not p.exists():
            raise FileNotFoundError(f"PLMGraph needs a per-chain PDB at {p} "
                                    "(generate ESMFold monomers first)")

    plmg_root = Path(cfg.get("plmg_root") or os.environ.get("PLMG_ROOT")
                     or (Path(__file__).resolve().parents[4] / "modules" / "plmgraph_inter"))
    sys.path.insert(0, str(plmg_root))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    py = sys.executable

    ccmpred = cfg["ccmpred_bin"]; alnstats = cfg["alnstats_bin"]
    hhmake = cfg["hhmake_bin"]; hhfilter = cfg["hhfilter_bin"]
    esm1b_w = cfg["esm1b_weights"]; esm_msa1b_w = cfg["esm_msa1b_weights"]; esmif_w = cfg["esmif_weights"]
    model_dir = Path(cfg.get("model_dir") or (plmg_root / "model"))
    loadhhm = str(plmg_root / "plm" / "LoadHHM.py")

    import load_feature
    import pdb_graph
    import plm.esm1b_repr as esm1b_repr
    import plm.esmif_repr as esmif_repr
    import plm.msa1b_attn as msa1b_attn
    import plm.msa1b_repr as msa1b_repr
    from model import resnet18

    print(f"[plmgraph] {entry_id}  device={device}", flush=True)
    rp = out_dir
    seqA, seqB = sequences[0], sequences[1]
    lenA, lenB = len(seqA), len(seqB)
    fasA, fasB = rp / "A.fasta", rp / "B.fasta"
    _write_fasta(fasA, "A", seqA)
    _write_fasta(fasB, "B", seqB)

    # MSAs from the Boltz CSVs (reuse Boltz pairing; bypass pair_msa).
    rowsA = _read_boltz_csv(msa_paths[chain_ids[0]])
    rowsB = _read_boltz_csv(msa_paths[chain_ids[1]])
    a3mA, a3mB = rp / "A.a3m", rp / "B.a3m"
    a3mA.write_text(_chain_a3m_from_rows(rowsA, "A"))
    a3mB.write_text(_chain_a3m_from_rows(rowsB, "B"))
    a3mA, a3mB = str(a3mA), str(a3mB)
    paired_a3m = rp / "paired.a3m"
    paired_a3m.write_text(_paired_a3m_from_csvs(rowsA, rowsB, "paired"))

    # reformat / filter
    filter_paired_a3m = rp / "filtered_paired.a3m"
    paired_aln = rp / "paired.aln"
    filter_a3mA = rp / "filteredA.a3m"
    filter_a3mB = rp / "filteredB.a3m"
    _run(hhfilter, "-i", paired_a3m, "-o", filter_paired_a3m, "-diff", "256")
    paired_aln.write_text(_a3m_to_aln(paired_a3m.read_text()))
    _run(hhfilter, "-i", a3mA, "-o", filter_a3mA, "-diff", "256")
    _run(hhfilter, "-i", a3mB, "-o", filter_a3mB, "-diff", "256")

    # CCMpred + alnstats (GPU CCMpred uses device 0 by default — no -t)
    _run(ccmpred, "-R", paired_aln, rp / "paired.ccmpred")
    _run(alnstats, paired_aln, rp / "paired.singout", rp / "paired.pairout")

    # ESM-MSA-1b attention
    msa1b_attn.main(esm_msa1b_w, str(filter_paired_a3m), str(fasA),
                    str(rp / "msa1b_rt.attn"), str(rp / "msa1b_sw.attn"), device)
    # PSSM (hhmake -> LoadHHM)
    _run(hhmake, "-i", a3mA, "-o", rp / "A.hhm"); _run(py, loadhhm, rp / "A.hhm")
    _run(hhmake, "-i", a3mB, "-o", rp / "B.hhm"); _run(py, loadhhm, rp / "B.hhm")
    # ESM-1b / ESM-MSA-1b representations
    esm1b_repr.main(esm1b_w, str(fasA), str(rp / "A_esm1b.repr"), device)
    esm1b_repr.main(esm1b_w, str(fasB), str(rp / "B_esm1b.repr"), device)
    msa1b_repr.main(esm_msa1b_w, str(filter_a3mA), str(rp / "A_msa1b.repr"), device)
    msa1b_repr.main(esm_msa1b_w, str(filter_a3mB), str(rp / "B_msa1b.repr"), device)
    # ESM-IF1 representations (from structure)
    esmif_repr.main(esmif_w, str(pdbA), str(rp / "A_esmif.repr"), device)
    esmif_repr.main(esmif_w, str(pdbB), str(rp / "B_esmif.repr"), device)
    # GVP graphs (from structure)
    pdb_graph.main(str(pdbA), str(rp / "graphA.pkl"))
    pdb_graph.main(str(pdbB), str(rp / "graphB.pkl"))

    # load features + run the 7-member graph ResNet ensemble
    featureA, featureB = load_feature.graph_feature(str(rp))
    rt_p2d, sw_p2d = load_feature.paired_feature(str(rp))

    def _to_dev(f):
        return ((f["nodes_scat"].to(device), f["nodes_vec"].to(device)),
                (f["edge_scat"].to(device), f["edge_vec"].to(device)),
                f["edge_index"].to(device))
    nodeA, edgeA, eidxA = _to_dev(featureA)
    nodeB, edgeB, eidxB = _to_dev(featureB)
    rt_p2d = rt_p2d.to(device).float()
    sw_p2d = sw_p2d.to(device).float()

    model = resnet18()
    torch.set_grad_enabled(False)
    _, lx, ly = rt_p2d.shape
    acc = torch.zeros(lx, ly)
    for i in range(1, 8):
        model.load_state_dict(torch.load(str(model_dir / str(i)), map_location=device))
        model.to(device).eval()
        acc += model(nodeA, edgeA, eidxA, nodeB, edgeB, eidxB, rt_p2d).detach().cpu()
        acc += model(nodeB, edgeB, eidxB, nodeA, edgeA, eidxA, sw_p2d).T.detach().cpu()
    block = (acc / 14.0).numpy()

    probs = _embed_block(block, lenA, lenB)
    L = lenA + lenB
    np.savez_compressed(out_dir / "contact.npz", probs=probs, length=np.int32(L))
    print(f"[plmgraph] WROTE {out_dir/'contact.npz'} shape={probs.shape} block={block.shape}", flush=True)


if __name__ == "__main__":
    main()
