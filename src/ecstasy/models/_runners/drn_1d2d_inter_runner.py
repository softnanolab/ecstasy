"""Self-contained DRN-1D2D_Inter runner — invoked via its env's python by the adapter.

DRN-1D2D_Inter (Si & Yan, Brief. Bioinform. 2023) is SEQUENCE-ONLY inter-protein
contact prediction for a 2-chain complex. It needs a per-chain MSA per chain and a
PAIRED complex MSA, no monomer structure, so the registry sets ``msa: boltz_csv``.

This runner reproduces modules/drn_1d2d_inter/predict.py step-for-step, but:
  * inputs come from the ecstasy stdin bundle (sequences + per-chain MSA), not argv;
  * every external-tool / weight path comes from ``params`` (registry preset), so
    nothing is hardcoded the way upstream predict.py is;
  * MSAs come from the Boltz per-chain CSVs (``msa: boltz_csv``); we reuse Boltz's
    taxonomy pairing (the CSV ``key`` column) to build the paired complex a3m,
    REPLACING DRN's ``pair_msa`` (its header-parsed pairing yields zero pairs on
    ColabFold-style a3ms — see _paired_a3m_from_csvs);
  * DRN emits only the inter-chain block (shape lenA x lenB); we embed it into the
    full square (L, L) probability map that ecstasy's scorer consumes.

Reads a JSON bundle from stdin:
  { entry_id, sequences[2], chain_ids[2], msa_paths{chain_id: boltz_csv}, out_dir, params, infra }

params (resolved ${...} paths from the registry preset):
  drn_root          repo dir of the DRN submodule (default: computed from __file__)
  ccmpred_bin, alnstats_bin, hhmake_bin, hhfilter_bin
  esm1b_weights, esm_msa1b_weights      ESM-1b / ESM-MSA-1b .pt (regression .pt alongside)
  model_dir         dir holding the 7 trained ResNet weights named 1..7

Writes:
  <out_dir>/contact.npz   — probs (L, L) float16, length int32

DRN is intentionally OUTSIDE the FLOPs-profiling scope (CLAUDE.md lists only
boltz2/esmfold), so this runner ignores ``bundle["profile"]`` and emits no FLOPs
sidecar by design.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# NB: torch is imported lazily inside main() (it only runs in the DRN env), so the
# pure helpers below — _embed_block in particular — stay importable for unit tests
# in the torch-less orchestrator env.


def _write_fasta(path: Path, name: str, seq: str) -> None:
    path.write_text(f">{name}\n{seq}\n")


def _run(*args) -> None:
    """Run an external tool in list form, aborting loudly on non-zero exit.

    Replaces ``os.system`` (which swallows the exit code): a failed CCMpred /
    hhfilter / alnstats stage would otherwise leave empty feature files that only
    surface as a cryptic shape error deep in ``load_feature.*``. Matches the
    list-form ``subprocess.run([...], check=True)`` house style of the other runners.
    """
    subprocess.run([str(a) for a in args], check=True)


def _embed_block(block: np.ndarray, lenA: int, lenB: int) -> np.ndarray:
    """Embed DRN's ``(lenA, lenB)`` inter-chain block into the full symmetric
    ``(L, L)`` contact-probability map the scorer consumes.

    Chain A occupies the first ``lenA`` rows/cols; chain B the rest. The scorer
    (``metrics/contact.py: pak_inter_chain``) reads only the strict upper-triangle
    inter-chain pairs, so the intra-chain blocks stay 0; the block is mirrored into
    the lower triangle for a symmetric map. Returns float16, shape ``(L, L)``.
    """
    if block.shape != (lenA, lenB):
        raise RuntimeError(f"DRN block {block.shape} != expected ({lenA}, {lenB})")
    L = lenA + lenB
    probs = np.zeros((L, L), dtype=np.float32)
    probs[:lenA, lenA:] = block
    probs[lenA:, :lenA] = block.T
    return probs.astype(np.float16)


def _read_boltz_csv(path: str | Path) -> list[tuple[int, str]]:
    """Parse a Boltz per-chain MSA CSV into ``(key, sequence)`` rows.

    Boltz CSV schema (see ``msa/boltz_csv.py``): a ``key,sequence`` header then one
    row per sequence. ``key`` is the row index of the taxonomy-paired alignment row
    (so the *same* key in chain A's and chain B's CSV = a boltz-paired pair); ``-1``
    marks an unpaired row. Row 0 (key 0) is the query.
    """
    rows: list[tuple[int, str]] = []
    for line in Path(path).read_text().splitlines()[1:]:  # skip header
        line = line.strip()
        if not line:
            continue
        key, _, seq = line.partition(",")
        rows.append((int(key), seq))
    return rows


def _chain_a3m_from_rows(rows: list[tuple[int, str]], query_name: str) -> str:
    """Render a chain's boltz CSV rows as a per-chain a3m (query first).

    Used for DRN's per-chain features (PSSM, ESM-MSA-1b repr), which need depth but
    NOT pairing — so synthetic headers are fine. Homolog headers MUST NOT start with
    ``query_name``: DRN's LoadHHM picks the query out of hhmake's .hhm SEQ block by
    prefix-matching ``'>' + name``, so ``>A_1`` would shadow the query ``>A`` and the
    PSSM would load the wrong sequence (use ``>h<i>`` instead).
    """
    out: list[str] = []
    for i, (key, seq) in enumerate(rows):
        out.append(f">{query_name}" if i == 0 else f">h{i}_k{key}")
        out.append(seq)
    return "\n".join(out) + "\n"


def _paired_a3m_from_csvs(rowsA: list[tuple[int, str]], rowsB: list[tuple[int, str]],
                          query_name: str) -> str:
    """Build DRN's ``paired.a3m`` by joining the two chains' boltz CSV rows on their
    shared pairing key (key >= 0), concatenating ``seqA + seqB`` per matched pair.

    This REPLACES DRN's ``pair_msa.main`` (Si & Yan's taxonomy-header pairing): boltz
    CSVs carry no taxonomy headers, but their ``key`` column already encodes boltz's
    taxonomy pairing, so reusing it gives a faithful paired complex MSA without the
    zero-pairing failure ColabFold-style a3ms hit. Row 0 (key 0) is the query complex.
    """
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
    """Convert an a3m alignment to CCMpred's ``.aln`` format: one sequence per line,
    headers dropped, a3m insertions removed (lowercase letters + ``.``), so every row
    is the query-length match-state alignment CCMpred/alnstats consume.

    Replaces the upstream ``fasta2aln`` binary, which is shipped x86-only and won't
    run on aarch64 — and is just this insertion-stripping conversion.
    """
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


def main() -> None:
    import torch  # lazy: only present in the DRN env (keeps helpers import-clean elsewhere)

    # PORT SHIM (torch>=2.6): fair-esm's load_model_and_alphabet_local and DRN's
    # ensemble weights were pickled under torch<2 and call torch.load WITHOUT
    # weights_only; torch>=2.6 defaults that to True and refuses the (trusted, local)
    # ESM argparse checkpoints with an UnpicklingError. Restore the old default for
    # this process. See scripts/install/drn_1d2d_inter.sh PORT NOTE.
    _orig_torch_load = torch.load
    def _torch_load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig_torch_load(*a, **k)
    torch.load = _torch_load

    bundle = json.loads(sys.stdin.read())
    entry_id: str = bundle["entry_id"]
    sequences: list[str] = bundle["sequences"]
    chain_ids: list[str] = bundle["chain_ids"]
    msa_paths: dict[str, str] = bundle.get("msa_paths") or {}
    out_dir = Path(bundle["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = bundle.get("params") or {}

    if len(sequences) != 2:
        raise ValueError(f"DRN-1D2D_Inter handles exactly 2 chains, got {len(sequences)} for {entry_id}")
    if not msa_paths.get(chain_ids[0]) or not msa_paths.get(chain_ids[1]):
        raise ValueError(f"DRN needs a Boltz MSA CSV for both chains; have {sorted(msa_paths)}")

    # Make DRN's library modules importable (paired.*, plm.*, load_feature, model).
    drn_root = Path(cfg.get("drn_root") or os.environ.get("DRN_ROOT")
                    or (Path(__file__).resolve().parents[4] / "modules" / "drn_1d2d_inter"))
    sys.path.insert(0, str(drn_root))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    py = sys.executable

    # Tool / weight paths (registry preset resolves the ${...} placeholders).
    ccmpred = cfg["ccmpred_bin"]
    alnstats = cfg["alnstats_bin"]
    hhmake = cfg["hhmake_bin"]
    hhfilter = cfg["hhfilter_bin"]
    esm1b_w = cfg["esm1b_weights"]
    esm_msa1b_w = cfg["esm_msa1b_weights"]
    # Trained weights + LoadHHM live inside the submodule (no scratch placeholder).
    model_dir = Path(cfg.get("model_dir") or (drn_root / "model"))
    loadhhm = str(drn_root / "plm" / "LoadHHM.py")

    import load_feature
    import plm.esm1b_attn as esm1b_attn
    import plm.esm1b_repr as esm1b_repr
    import plm.msa1b_attn as msa1b_attn
    import plm.msa1b_repr as msa1b_repr
    from model import resnet18

    print(f"[drn] {entry_id}  device={device}  drn_root={drn_root}", flush=True)

    rp = out_dir  # DRN writes all intermediate features into the result_path
    seqA, seqB = sequences[0], sequences[1]
    lenA, lenB = len(seqA), len(seqB)
    fasA, fasB = rp / "A.fasta", rp / "B.fasta"
    _write_fasta(fasA, "A", seqA)
    _write_fasta(fasB, "B", seqB)

    # --- MSAs from the Boltz per-chain CSVs (msa: boltz_csv) ----------------
    # We reuse Boltz's taxonomy pairing (CSV `key` column) instead of DRN's
    # header-based pair_msa, which gives zero pairing on ColabFold-style a3ms whose
    # headers lack the species ids it parses. Build the two per-chain a3ms (depth-
    # only features) + the paired complex a3m (key-joined) that pair_msa used to emit.
    rowsA = _read_boltz_csv(msa_paths[chain_ids[0]])
    rowsB = _read_boltz_csv(msa_paths[chain_ids[1]])
    a3mA, a3mB = rp / "A.a3m", rp / "B.a3m"
    a3mA.write_text(_chain_a3m_from_rows(rowsA, "A"))
    a3mB.write_text(_chain_a3m_from_rows(rowsB, "B"))
    a3mA, a3mB = str(a3mA), str(a3mB)

    # --- feature pipeline (mirrors predict.py from here on) ----------------
    # 1. paired MSA (key-joined complex a3m; replaces pair_msa.main output)
    paired_a3m = rp / "paired.a3m"
    paired_a3m.write_text(_paired_a3m_from_csvs(rowsA, rowsB, "paired"))

    # 2. reformat / filter
    filter_paired_a3m = rp / "filtered_paired.a3m"
    paired_aln = rp / "paired.aln"
    filter_a3mA = rp / "filteredA.a3m"
    filter_a3mB = rp / "filteredB.a3m"
    _run(hhfilter, "-i", paired_a3m, "-o", filter_paired_a3m, "-diff", "256")
    paired_aln.write_text(_a3m_to_aln(paired_a3m.read_text()))  # was: fasta2aln (x86-only)
    _run(hhfilter, "-i", a3mA, "-o", filter_a3mA, "-diff", "256")
    _run(hhfilter, "-i", a3mB, "-o", filter_a3mB, "-diff", "256")

    # 3. paired seq
    paired_seq = rp / "paired.fasta"
    _write_fasta(paired_seq, "paired", seqA + seqB)

    # 4. CCMpred + alnstats
    _run(ccmpred, "-R", paired_aln, rp / "paired.ccmpred")
    _run(alnstats, paired_aln, rp / "paired.singout", rp / "paired.pairout")

    # 5. ESM-1b attention
    esm1b_attn.main(esm1b_w, str(paired_seq), str(fasA),
                    str(rp / "esm1b_rt.attn"), str(rp / "esm1b_sw.attn"), device)
    # 6. ESM-MSA-1b attention
    msa1b_attn.main(esm_msa1b_w, str(filter_paired_a3m), str(fasA),
                    str(rp / "msa1b_rt.attn"), str(rp / "msa1b_sw.attn"), device)

    # 7. PSSM (hhm -> pkl)
    _run(hhmake, "-i", a3mA, "-o", rp / "A.hhm")
    _run(py, loadhhm, rp / "A.hhm")
    _run(hhmake, "-i", a3mB, "-o", rp / "B.hhm")
    _run(py, loadhhm, rp / "B.hhm")

    # 8. ESM-1b representations
    esm1b_repr.main(esm1b_w, str(fasA), str(rp / "A_esm1b.repr"), device)
    esm1b_repr.main(esm1b_w, str(fasB), str(rp / "B_esm1b.repr"), device)
    # 9. ESM-MSA-1b representations
    msa1b_repr.main(esm_msa1b_w, str(filter_a3mA), str(rp / "A_msa1b.repr"), device)
    msa1b_repr.main(esm_msa1b_w, str(filter_a3mB), str(rp / "B_msa1b.repr"), device)

    # --- load features + run the 7-member ResNet ensemble ------------------
    featureA, featureB = load_feature.chain_feature(str(rp))
    rt_p2d, sw_p2d = load_feature.paired_feature(str(rp))
    featureA = featureA.to(device).float()
    featureB = featureB.to(device).float()
    rt_p2d = rt_p2d.to(device).float()
    sw_p2d = sw_p2d.to(device).float()
    rt_input = load_feature.concat(featureA, featureB, rt_p2d)
    sw_input = load_feature.concat(featureB, featureA, sw_p2d)

    model = resnet18()
    torch.set_grad_enabled(False)
    _, _, lx, ly = rt_input.shape
    acc = torch.zeros(lx, ly)
    for i in range(1, 8):
        model.load_state_dict(torch.load(str(model_dir / str(i)), map_location=device))
        model.to(device).eval()
        acc += model(rt_input).detach().cpu()
        acc += model(sw_input).T.detach().cpu()
    block = (acc / 14.0).numpy()  # (lenA, lenB) inter-chain contact probability

    # Chain order in the bundle == token order in the GT, so chain A occupies the
    # first lenA rows/cols of the embedded square (see _embed_block).
    probs = _embed_block(block, lenA, lenB)
    L = lenA + lenB

    np.savez_compressed(out_dir / "contact.npz", probs=probs, length=np.int32(L))
    print(f"[drn] WROTE {out_dir/'contact.npz'}  shape={probs.shape}  block={block.shape}", flush=True)


if __name__ == "__main__":
    main()
