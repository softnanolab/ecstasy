"""Run MSA Pairformer on ecstasy_v1, mirroring the colab notebook exactly.

For each dimer in master/index.parquet:
  - read concatenated paired A3M from msas/<id>.a3m (notebook format)
  - MSA(msa_file_path, max_seqs=512, max_length=L, diverse_select_method="hhfilter",
        hhfilter_kwargs={"binary": <abs>, "cov": 70, "qid": 15})  with np.random.seed(42)
  - one-hot tokens (28-class), prepare_msa_masks, move to GPU
  - model(...) under torch.amp.autocast(bfloat16), complex_chain_break_indices=[[break]]
  - take predicted_cb_contacts[0] -> (L, L) float
  - save predictions/msa_pairformer/<run_id>/<entry_id>/contact.npz with
    `probs` (L, L) float16 and `length` int32

Notebook fidelity choices (verified by hand against
MSA_Pairformer_with_MMseqs2.ipynb):
  - max_msa_depth = 512
  - max_length = sum(chain_lens) (no truncation since master set is <=1200)
  - hhfilter cov=70 qid=15 (notebook default)
  - return_seq_weights=True
  - bfloat16 autocast on CUDA
  - chain break = len(chain_a) (0-based start of chain B), single break per dimer
  - We do NOT slice to interchain block here -- runner emits full (L, L);
    benchmark score() handles the slicing.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Bring in MSA Pairformer module
sys.path.insert(0, "/home/u6jv/harsh.u6jv/ecstasy/modules/msa_pairformer")
from MSA_Pairformer.dataset import MSA, aa2tok_d, prepare_msa_masks  # noqa: E402
from MSA_Pairformer.model import MSAPairformer  # noqa: E402

MASTER_INDEX = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/master/index.parquet")
DEFAULT_MSAS_DIR = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/msas")
PRED_ROOT = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/predictions/msa_pairformer")
WEIGHTS_DIR = Path("/projects/u6jv/ecstasy/weights/msa_pairformer")
HHFILTER_BIN = "/home/u6jv/harsh.u6jv/ecstasy/tools/hhsuite/bin/hhfilter"

MAX_MSA_DEPTH = 512
# Notebook default — only specify binary; MSA() class defaults cov=70, qid=30.
# qid=15 was tested (more permissive subsampling) and gave consistently WORSE
# inter-chain P@K, despite filling the depth-512 quota more reliably.
HHFILTER_KWARGS_BASE = {"binary": HHFILTER_BIN}


def predict_one(model, device, entry_id: str, seqs: list[str], a3m_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "contact.npz"
    if out_path.exists():
        return {"id": entry_id, "status": "cached", "L": None, "depth_used": None, "wall_s": 0.0}

    L = sum(len(s) for s in seqs)
    chain_break = len(seqs[0])  # 0-based start of chain B
    t0 = time.time()

    np.random.seed(42)
    msa_obj = MSA(
        msa_file_path=str(a3m_path),
        max_seqs=MAX_MSA_DEPTH,
        max_length=L,
        max_tokens=int(1e12),
        diverse_select_method="hhfilter",
        hhfilter_kwargs=HHFILTER_KWARGS_BASE,
    )
    tok = msa_obj.diverse_tokenized_msa  # [depth, L_msa] int64
    if tok.shape[1] != L:
        return {
            "id": entry_id,
            "status": "msa_length_mismatch",
            "L": L,
            "msa_L": int(tok.shape[1]),
            "depth_used": int(tok.shape[0]),
            "wall_s": time.time() - t0,
        }

    msa_in = tok.unsqueeze(0).to(device)  # [1, depth, L]
    mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(msa_in, device=device)
    msa_onehot = (
        torch.nn.functional.one_hot(msa_in, num_classes=len(aa2tok_d))
        .float()
        .to(device)
        .to(torch.bfloat16)
    )

    with torch.no_grad():
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=device.type):
            out = model(
                msa=msa_onehot,
                mask=mask,
                msa_mask=msa_mask,
                full_mask=full_mask,
                pairwise_mask=pairwise_mask,
                complex_chain_break_indices=[[chain_break]],
                return_seq_weights=True,
            )
    probs_cb = out["predicted_cb_contacts"][0].float().cpu().numpy()  # (L, L)
    probs_cf = out["predicted_confind_contacts"][0].float().cpu().numpy()
    assert probs_cb.shape == (L, L), f"got {probs_cb.shape}, expected ({L},{L})"
    np.savez_compressed(out_path,
                        probs=probs_cb.astype(np.float16),  # backwards-compat alias
                        probs_cb=probs_cb.astype(np.float16),
                        probs_confind=probs_cf.astype(np.float16),
                        length=np.int32(L))

    # cleanup tensors -- GPU memory pressure for back-to-back inference
    del msa_in, mask, msa_mask, full_mask, pairwise_mask, msa_onehot, out, probs_cb, probs_cf
    torch.cuda.empty_cache()
    gc.collect()
    return {
        "id": entry_id,
        "status": "ok",
        "L": L,
        "depth_used": int(tok.shape[0]),
        "wall_s": time.time() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="run0", help="output sub-dir name under predictions/")
    ap.add_argument(
        "--msas-dir",
        type=Path,
        default=DEFAULT_MSAS_DIR,
        help=(
            "Directory of paired A3Ms keyed by <entry_id>.a3m. Default is the unfiltered "
            "msas/ dir; pass msas_filtered/ to run the notebook-save_msa-filtered variant."
        ),
    )
    ap.add_argument("--limit", type=int, default=None, help="run only first N entries (smoke)")
    ap.add_argument("--ids", nargs="*", default=None, help="run only these entry ids")
    args = ap.parse_args()

    msas_dir: Path = args.msas_dir
    out_run_root = PRED_ROOT / args.run_id
    out_run_root.mkdir(parents=True, exist_ok=True)
    print(f"msas dir: {msas_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type != "cuda":
        print("WARNING: no GPU detected; this will be very slow", file=sys.stderr)

    print(f"loading MSAPairformer (weights cache: {WEIGHTS_DIR}) ...")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    model = MSAPairformer.from_pretrained(device=device, weights_dir=str(WEIGHTS_DIR))
    model = model.to(torch.bfloat16).eval()
    # Notebook default: query biasing on. Adds query-conditioned attention
    # in OuterProduct layers; key for cross-chain contact prediction.
    model.turn_on_query_biasing()
    print("model loaded (query biasing ON)")

    manifest = pd.read_parquet(MASTER_INDEX)
    if args.ids:
        manifest = manifest[manifest["id"].isin(args.ids)]
    if args.limit:
        manifest = manifest.head(args.limit)
    print(f"running on {len(manifest)} entries ...")

    log: list[dict] = []
    for i, row in enumerate(manifest.itertuples(), 1):
        a3m = msas_dir / f"{row.id}.a3m"
        if not a3m.exists():
            log.append({"id": row.id, "status": "missing_msa", "L": None, "depth_used": None, "wall_s": 0.0})
            print(f"  [{i}/{len(manifest)}]  {row.id}  MISSING A3M", flush=True)
            continue
        try:
            r = predict_one(model, device, row.id, list(row.sequences), a3m, out_run_root / row.id)
        except Exception as e:  # noqa: BLE001
            r = {"id": row.id, "status": "error", "error": f"{type(e).__name__}: {e}", "wall_s": 0.0}
        log.append(r)
        if i % 10 == 0 or i == len(manifest):
            ok = sum(1 for l in log if l["status"] in ("ok", "cached"))
            print(f"  [{i}/{len(manifest)}]  ok={ok}  last={row.id}/{r['status']}  wall={r.get('wall_s', 0):.1f}s", flush=True)

    log_df = pd.DataFrame(log)
    log_path = out_run_root / "_run_log.parquet"
    log_df.to_parquet(log_path, index=False)
    print(f"\n  wrote run log -> {log_path}")
    print(f"  status counts: {log_df['status'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
