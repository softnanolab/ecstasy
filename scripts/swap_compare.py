"""Chain-permutation experiment analysis: original val_seq_pair vs val_seq_pair_swapped.

Per (model, recycle) reports mean P@K original vs swapped, the mean delta, the
per-protein mean |delta| (order-sensitivity magnitude), Pearson r, and the fraction
of dimers whose P@K moves by >0.05 when chains are flipped. Writes a paired-scatter
PNG per model. mentos/msa_pairformer excluded (not part of the experiment)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ecstasy.config import settings  # noqa: E402

R = settings().runs_root
MODELS = ["boltz2", "boltz2_nomsa", "esmfold"]
PRESETS = ["r0", "r1", "r3", "r5"]
ORIG, SWAP = "val_seq_pair", "val_seq_pair_swapped"


def per_protein(split: str, model: str, preset: str) -> dict[str, float]:
    p = R / split / model / preset / "result.json"
    if not p.exists():
        return {}
    per = json.loads(p.read_text()).get("per_protein", {})
    return {k: v["P@K"] for k, v in per.items()
            if isinstance(v.get("P@K"), (int, float)) and v["P@K"] == v["P@K"]}


def main() -> None:
    print(f"{'model':14}{'rec':4}{'n':>5}{'origP@K':>9}{'swapP@K':>9}{'Δmean':>8}"
          f"{'mean|Δ|':>9}{'r':>7}{'|Δ|>.05':>9}")
    rows = []
    for m in MODELS:
        fig, ax = plt.subplots(figsize=(5, 5))
        for pi, pr in enumerate(PRESETS):
            o = per_protein(ORIG, m, pr)
            s = per_protein(SWAP, m, pr)
            ids = sorted(set(o) & set(s))
            if not ids:
                print(f"{m:14}{pr:4}{'--  (missing results)':>40}")
                continue
            oa = np.array([o[i] for i in ids])
            sa = np.array([s[i] for i in ids])
            d = sa - oa
            r = float(np.corrcoef(oa, sa)[0, 1]) if len(ids) > 2 else float("nan")
            frac = float(np.mean(np.abs(d) > 0.05))
            print(f"{m:14}{pr:4}{len(ids):>5}{oa.mean():>9.3f}{sa.mean():>9.3f}"
                  f"{d.mean():>+8.3f}{np.abs(d).mean():>9.3f}{r:>7.3f}{frac:>9.1%}")
            rows.append({"model": m, "recycle": pr, "n": len(ids),
                         "orig": oa.mean(), "swap": sa.mean(), "dmean": float(d.mean()),
                         "mad": float(np.abs(d).mean()), "r": r, "frac_gt05": frac})
            ax.scatter(oa, sa, s=6, alpha=0.35, label=f"{pr} (|Δ|={np.abs(d).mean():.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
        ax.set_xlabel("P@K original (A,B)"); ax.set_ylabel("P@K swapped (B,A)")
        ax.set_title(f"{m} — chain-order sensitivity (val_seq_pair)")
        ax.legend(fontsize=7, loc="best"); ax.set_aspect("equal")
        out = R / SWAP / f"swap_scatter_{m}.png"
        fig.tight_layout(); fig.savefig(out, dpi=150)
        fig.savefig(out.with_suffix(".pdf"))   # vector copy for the FYP report
        plt.close(fig)
        print(f"  wrote {out} (+pdf)")
    (R / SWAP / "swap_compare.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {R / SWAP / 'swap_compare.json'}")


if __name__ == "__main__":
    main()
