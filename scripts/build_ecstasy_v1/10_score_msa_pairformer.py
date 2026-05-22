"""Score MSA Pairformer predictions on ecstasy_v1.

For each entry in master/index.parquet, loads:
  - predictions/msa_pairformer/<run_id>/<id>/contact.npz   (full L x L probs)
  - master/data/<id[:2]>/<id>.pt                            (rectangular Na x Nb GT)
and computes Pinder/MINT-style interchain Precision@K via the benchmark's
score() method.

Outputs:
  results/ecstasy_v1__msa_pairformer__<run_id>.json (per-entry + aggregates)
  results/ecstasy_v1__msa_pairformer__<run_id>.csv  (per-entry table)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/u6jv/harsh.u6jv/ecstasy/src")
from ecstasy.benchmarks.ecstasy_v1 import EcstasyV1Bench  # noqa: E402
from ecstasy.benchmarks.base import Entry  # noqa: E402

PRED_ROOT = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/predictions/msa_pairformer")
RESULTS_DIR = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/results")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="run0")
    args = ap.parse_args()
    pred_dir = PRED_ROOT / args.run_id
    if not pred_dir.exists():
        print(f"ERROR: predictions dir {pred_dir} does not exist", file=sys.stderr)
        return 1

    bench = EcstasyV1Bench(data_root="/projects/u6jv/ecstasy")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    n_scored = 0
    for entry in bench.entries():
        contact_path = pred_dir / entry.id / "contact.npz"
        if not contact_path.exists():
            rows.append({"id": entry.id, "status": "missing_prediction"})
            continue
        try:
            # Cb head (default)
            m = bench.score(entry, contact_path)
            if "_error" in m:
                rows.append({"id": entry.id, "status": "error", "error": m["_error"]})
                continue
            row = {
                "id": entry.id,
                "status": "ok",
                "AUC": m["AUC"],
                "P@K": m["P@K"],
                "P@K/2": m["P@K/2"],
                "P@K/5": m["P@K/5"],
                "K": m["K"],
            }
            # Optional: confind head (if present in npz)
            try:
                import numpy as _np
                npz = _np.load(contact_path)
                if "probs_confind" in npz.files:
                    # score the confind matrix using the same Cb GT (interchain block)
                    import sys as _sys
                    _sys.path.insert(0, "/home/u6jv/harsh.u6jv/ecstasy/src")
                    from ecstasy.benchmarks.ecstasy_v1 import _pak_from_rect as _pak
                    probs = npz["probs_confind"].astype("float32")
                    gt = bench.gt_for(entry.id)
                    seqs = gt["sequences"]
                    la, lb = len(seqs[0]), len(seqs[1])
                    inter_ab = probs[:la, la:la+lb]
                    inter_ba = probs[la:la+lb, :la].T
                    inter = 0.5 * (inter_ab + inter_ba)
                    mc = _pak(inter, gt["contact_map"])
                    row["P@K_confind"] = mc["P@K"]
                    row["AUC_confind"] = mc["AUC"]
            except Exception as _e:
                row["confind_err"] = f"{type(_e).__name__}: {_e}"
            rows.append(row)
            n_scored += 1
        except Exception as e:  # noqa: BLE001
            rows.append({"id": entry.id, "status": "score_error", "error": f"{type(e).__name__}: {e}"})

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / f"ecstasy_v1__msa_pairformer__{args.run_id}.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote per-entry CSV -> {csv_path}")

    # Aggregates over scored entries
    ok = df[df["status"] == "ok"]
    agg = {
        "benchmark": "ecstasy_v1",
        "model": "msa_pairformer",
        "run_id": args.run_id,
        "n_entries_total": int(len(df)),
        "n_scored": int(len(ok)),
        "n_missing_pred": int((df.status == "missing_prediction").sum()),
        "n_error": int((df.status.isin(["error", "score_error"])).sum()),
        "mean_AUC": float(ok["AUC"].mean()) if len(ok) else None,
        "mean_P@K": float(ok["P@K"].mean()) if len(ok) else None,
        "mean_P@K/2": float(ok["P@K/2"].mean()) if len(ok) else None,
        "mean_P@K/5": float(ok["P@K/5"].mean()) if len(ok) else None,
        "median_P@K": float(ok["P@K"].median()) if len(ok) else None,
    }
    json_path = RESULTS_DIR / f"ecstasy_v1__msa_pairformer__{args.run_id}.json"
    json_path.write_text(json.dumps(agg, indent=2) + "\n")
    print(f"wrote aggregates JSON -> {json_path}")
    print()
    print("=== aggregates ===")
    for k, v in agg.items():
        print(f"  {k:20s}  {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
