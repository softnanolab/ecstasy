"""Faithful rendering of the MENTOS distogram-evolution npz dumps.

WHY THIS EXISTS — the original ad-hoc figure for this experiment was MISLEADING:
it imshow'd each panel with matplotlib's default per-panel auto-normalization
(vmin/vmax = that panel's own min/max) AND overlaid ground-truth contacts on every
panel. At step 0 the distogram head is random, so the distogram is ~uniform (cprob
≈ 19/64 = 0.30 everywhere, entropy = ln 64, no information). Auto-normalization
stretches that razor-thin numerical noise across the full colormap, and the GT
overlay paints true contacts onto every column — together turning a content-free
map into vivid apparent "structure that matches GT at t=0". It does not.

What this script does instead:
  * Plots the ARGMAX distogram (most-likely Cβ-Cβ distance bin) on a FIXED scale
    ``vmin=0, vmax=VMAX`` so panels are comparable across steps and predicted
    contacts (short distances) read sharply. NO ground-truth overlay.
  * Ground truth is its own column on the FAR LEFT (separate, never overlaid).
  * Each training-step column is headed with that protein's Inter and Intra P@K so
    the eye is anchored to the scored quantities — e.g. a distogram whose diagonal
    (intra) blocks "look like GT" while the off-diagonal interface is empty shows up
    as high Intra P@K but Inter P@K = 0.

Input npz keys (see run_distevo.py): argmax_bin (L,L) int, cprob (L,L) float,
gt_bins (L,L) int (-1 unresolved), la, lb.

  python plot_distogram_evolution.py --npz-dir <dir> --ids 9uc5,8pdc \
      --steps 0,20000,40000,60000,90000 --out fig.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

# Project house style (CMU Concrete). Lives next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plotstyle  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from ecstasy.metrics.contact import pak_from_pairs, pak_inter_chain  # noqa: E402

CONTACT_BIN = 19   # bins 0..18 = Cβ-Cβ ≤ 7.9375 Å (mentos CONTACT_LABEL_THRESHOLD_BIN)
VMAX = 40          # distance-bin colour cap: far pixels saturate, contacts dominate
MIN_SEP = 6        # intra-chain P@K excludes trivially-close pairs (mentos DEFAULT_MIN_SEP)


def _load(npz_dir: Path, pid: str, step: int):
    z = np.load(npz_dir / f"{pid}_step{step}.npz")
    return (z["argmax_bin"].astype(float), z["cprob"].astype(np.float32),
            z["gt_bins"].astype(float), int(z["la"]), int(z["lb"]))


def _paks(cprob: np.ndarray, gt_bins: np.ndarray, la: int, lb: int) -> tuple[float, float]:
    """(inter P@K, intra P@K) from the predicted contact-prob map + GT bins.

    Inter uses the official ``pak_inter_chain`` (matches the eval pipeline). Intra is
    the mean over the two chains of per-chain P@K on the strict upper triangle with
    ``min_sep >= MIN_SEP``, gated to resolved (defined) pairs.
    """
    cgt = (gt_bins >= 0) & (gt_bins < CONTACT_BIN)
    valid = gt_bins >= 0
    L = la + lb
    inter = pak_inter_chain(cprob, cgt, np.array([0] * la + [1] * lb), valid=valid)["P@K"]
    intra_vals = []
    for s in (slice(0, la), slice(la, L)):
        p, g, v = cprob[s, s], cgt[s, s], valid[s, s]
        n = p.shape[0]
        ii, jj = np.indices((n, n))
        m = v & np.triu(np.ones((n, n), bool), 1) & (np.abs(ii - jj) >= MIN_SEP)
        if (g & m).sum() > 0:
            intra_vals.append(pak_from_pairs(p[m], g[m])["P@K"])
    return inter, (float(np.mean(intra_vals)) if intra_vals else float("nan"))


def _overlay_rgb(gt_bins: np.ndarray, argmax: np.ndarray | None = None, blend: float = 0.70) -> np.ndarray:
    """White-background RGB image: GT contacts in black, predicted (if given) in red.

    white = neither, black = GT-only, light red = predicted-only (FP), dark red = both.
    Pass ``argmax=None`` for a GT-only panel.
    """
    L = gt_bins.shape[0]
    img = np.ones((L, L, 3))
    img[(gt_bins >= 0) & (gt_bins < CONTACT_BIN)] = (0.0, 0.0, 0.0)
    if argmax is not None:
        pred = argmax < CONTACT_BIN
        img[pred] = (1 - blend) * img[pred] + blend * np.array([1.0, 0.0, 0.0])
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--steps", default="0,20000,40000,60000,90000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--binary", action="store_true",
                    help="render predicted contacts as binary (argmax bin < CONTACT_BIN, i.e. < ~8 Å) "
                         "instead of the continuous argmax distance gradient")
    ap.add_argument("--overlay", action="store_true",
                    help="overlay GT (black) and predicted (bright red) contacts on a white background "
                         "(white=none, black=GT-only, light red=FP, dark red=correct); overrides --binary")
    args = ap.parse_args()

    if not _plotstyle.use_cmu_concrete():
        print("warning: CMU Concrete not found; falling back to default font", file=sys.stderr)
    npz_dir = Path(args.npz_dir)
    ids = [s.strip() for s in args.ids.split(",")]
    steps = [int(s) for s in args.steps.split(",")]
    overlay = args.overlay
    line_c = "0.55" if overlay else ("tab:red" if args.binary else "0.4")  # boundary line readable per mode
    if not overlay:
        if args.binary:
            cmap = plt.get_cmap("gray_r").copy(); vmax = 1   # contact=black, none=white
        else:
            cmap = plt.get_cmap("magma_r").copy(); vmax = VMAX  # distance gradient, bright=close
        cmap.set_bad("0.82")  # unresolved GT pixels (-1) → grey

    def gt_disp(gt_bins):
        m = np.ma.masked_less(gt_bins, 0)
        return np.ma.masked_array((m < CONTACT_BIN).astype(float), mask=np.ma.getmaskarray(m)) if args.binary else m

    def pred_disp(am):
        return (am < CONTACT_BIN).astype(float) if args.binary else am

    ncol = len(steps) + 1  # GT (far left) + one per step
    # aspect="equal" keeps the L×L maps square (no horizontal squeeze);
    # constrained_layout packs the wider grid + shared colorbar cleanly.
    fig, ax = plt.subplots(len(ids), ncol, figsize=(3.0 * ncol, 3.5 * len(ids)),
                           squeeze=False, constrained_layout=True)

    im = None
    for r, pid in enumerate(ids):
        am, cprob, gt_bins, la, lb = _load(npz_dir, pid, steps[-1])
        # GT column, far LEFT (3-line title pad keeps its panel height aligned with the
        # step columns, whose titles carry two P@K lines below the step label).
        if overlay:
            ax[r, 0].imshow(_overlay_rgb(gt_bins), aspect="equal", interpolation="nearest")
        else:
            ax[r, 0].imshow(gt_disp(gt_bins), cmap=cmap, vmin=0, vmax=vmax, aspect="equal", interpolation="nearest")
        ax[r, 0].axhline(la - 0.5, c=line_c, lw=0.5); ax[r, 0].axvline(la - 0.5, c=line_c, lw=0.5)
        ax[r, 0].set_title("Ground Truth\n \n ", fontsize=11)
        ax[r, 0].set_ylabel(pid, fontsize=14)
        for j, step in enumerate(steps, start=1):
            am, cprob, gt_bins, la, lb = _load(npz_dir, pid, step)
            if overlay:
                ax[r, j].imshow(_overlay_rgb(gt_bins, am), aspect="equal", interpolation="nearest")
            else:
                im = ax[r, j].imshow(pred_disp(am), cmap=cmap, vmin=0, vmax=vmax, aspect="equal", interpolation="nearest")
            ax[r, j].axhline(la - 0.5, c=line_c, lw=0.5); ax[r, j].axvline(la - 0.5, c=line_c, lw=0.5)
            inter, intra = _paks(cprob, gt_bins, la, lb)
            lab = "0" if step == 0 else f"{step // 1000}K"
            ax[r, j].set_title(f"step {lab}\nInter P@K = {inter:.2f}\nIntra P@K = {intra:.2f}", fontsize=10)
        for c in range(ncol):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])

    if overlay:
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(facecolor="black", label="GT contact"),
                            Patch(facecolor=(1.0, 0.30, 0.30), label="Predicted only (FP)"),
                            Patch(facecolor=(0.30, 0.0, 0.0), label="Both (correct)")],
                   loc="lower center", ncol=3, frameon=False, fontsize=10)
        fig.suptitle("MENTOS distogram evolution — GT (black) vs predicted (bright red) contacts "
                     "[white = none; dark red = correct; grey lines = chain boundary]", fontsize=13)
    elif args.binary:
        fig.suptitle("MENTOS distogram evolution — predicted contacts (binary: argmax bin < ~8 Å; "
                     "black = contact; boundary lines = chain split)", fontsize=13)
    else:
        cb = fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.015, pad=0.01)
        cb.set_label("predicted C$\\beta$–C$\\beta$ distance bin  (bright = close; bins 0–18 $\\approx$ contact)", fontsize=10)
        fig.suptitle("MENTOS distogram evolution over training  (argmax; grey lines = chain boundary)", fontsize=13)
    fig.savefig(args.out, dpi=150)  # constrained_layout handles spacing; no bbox_inches
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
