"""Standalone driver — runs INSIDE the colabfold-local venv (no ecstasy imports).

Invoked by the ``complex`` backend's SLURM job. Reads a manifest written by
``complex.prepare`` and generates each complex MSA via colabfold-local's
``get_paired_msa_local`` (local colabfold_search), writing each a3m straight to its
final store path.

  python _complex_local_driver.py <manifest.json>

Requires env: COLABFOLD_LOCAL_DIR (the colabfold-local checkout, so its ``src`` is
importable), DATA_DIR (local ColabFold DBs), MMSEQS_BIN. Manifest entries are
``{"seqs": [...], "dst": "<store a3m path>", "header": "<pair_hash>"}``.
"""
import json
import os
import sys
from pathlib import Path


def main() -> int:
    manifest_path = sys.argv[1]
    cl_dir = os.environ.get("COLABFOLD_LOCAL_DIR")
    if not cl_dir:
        print("[complex-local] COLABFOLD_LOCAL_DIR not set", file=sys.stderr)
        return 2
    sys.path.insert(0, str(Path(cl_dir) / "src"))
    from local_msa_adapter import get_paired_msa_local  # noqa: E402  (colabfold-local)

    items = json.loads(Path(manifest_path).read_text())
    ok = skip = err = 0
    for it in items:
        dst = Path(it["dst"])
        if dst.exists():
            skip += 1
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            get_paired_msa_local(sequences=it["seqs"], output_path=dst, use_cache=True)
            ok += 1
        except Exception as e:  # noqa: BLE001 — one bad complex must not kill the batch
            err += 1
            print(f"[complex-local] ERROR {it.get('header')}: {e}", file=sys.stderr, flush=True)
        n = ok + skip + err
        if n % 25 == 0 or n == len(items):
            print(f"[complex-local] {n}/{len(items)} (wrote={ok} skip={skip} err={err})", flush=True)
    print(f"[complex-local] DONE wrote={ok} skipped={skip} errors={err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
