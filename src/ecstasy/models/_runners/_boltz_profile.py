"""In-process Boltz-2 trunk-only FLOP profiling, without editing the boltz fork.

Boltz's contact map is the trunk distogram (``boltz2.py``: ``pdistogram =
distogram_module(z)``), computed *before* the diffusion sampler. So the
contact-dependency-subgraph FLOP count (decision (ii), FLOPS_BENCHMARK_PLAN.md
§3.5) is just the forward run with ``skip_run_structure=True`` — diffusion never
executes, the distogram is identical, and ``FlopCounterMode`` over that forward
gives the trunk-only true FLOPs with no module attribution needed.

We achieve this by monkeypatching two boltz methods at runtime (so the change
lives in ecstasy, not in the boltz fork) and then calling boltz's own
``predict`` so all of its real featurization/data-module/trainer setup is reused:

* ``Boltz2.predict_step`` → set ``skip_run_structure``, wrap the forward in
  ``FlopCounterMode``, return ``{pdistogram, token_masks, flops}`` (no coords).
* ``BoltzWriter.write_on_batch_end`` → dump ``distogram_<id>.npz`` +
  ``flops_<id>.json`` per record, and skip all structure/mmcif writing.

Runs inside ``.venv-boltz`` (has boltz). Imports ``_flops`` from this directory.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _flops  # noqa: E402  (sibling helper; FlopCounterMode breakdown + 2*MACs convention)


def _install_patches():
    import numpy as np
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    from boltz.model.models.boltz2 import Boltz2
    from boltz.data.write.writer import BoltzWriter

    # Off the contact-map dependency path (plan §3.5). `skip_run_structure` keeps the
    # diffusion sampler (structure_module) + diffusion_conditioning from running at all,
    # but boltz still computes the confidence/bfactor heads — they are downstream of the
    # discarded coordinates, not of the distogram, so subtract whatever they cost.
    OFF_PATH = ("structure_module", "diffusion_conditioning", "confidence_module", "bfactor_module")

    def profiled_predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Trunk-only: structure/diffusion never runs; the distogram (computed before
        # diffusion) is unchanged. The (ii) count = total minus any off-path head that
        # still ran (confidence). These run via __call__, so subtree attribution is exact.
        self.skip_run_structure = True
        fc = FlopCounterMode(display=False)
        with fc:
            out = self(
                batch,
                recycling_steps=self.predict_args["recycling_steps"],
                num_sampling_steps=self.predict_args["sampling_steps"],
                diffusion_samples=self.predict_args["diffusion_samples"],
                max_parallel_samples=self.predict_args["max_parallel_samples"],
                run_confidence_sequentially=True,
            )
        total = int(fc.get_total_flops())
        raw_counts = fc.get_flop_counts()
        by_module = _flops._top_level_breakdown(raw_counts)
        off_path = int(sum(v for k, v in by_module.items()
                           if k.split(".")[-1] in OFF_PATH))
        flops = total - off_path
        payload = {
            "flops": flops,                 # contact-dependency subgraph (true FLOPs)
            "macs": flops // 2,
            "flops_total": total,           # whole profiled forward (audit)
            "off_path_flops": off_path,     # confidence/bfactor subtracted
            "by_module": by_module,
            "recycling_steps": int(self.predict_args["recycling_steps"]),
        }
        if os.environ.get("ECSTASY_FLOPS_DEBUG"):
            # Full un-filtered attribution + the model flags that decide whether the
            # trunk runs eagerly (and so whether FlopCounterMode can observe it).
            payload["debug"] = {
                "all_modules": {k: int(sum(v.values())) for k, v in raw_counts.items()},
                "op_types": sorted({str(op) for v in raw_counts.values() for op in v}),
                "n_module_paths": len(raw_counts),
                "flags": {
                    "use_kernels": getattr(self, "use_kernels", None),
                    "run_trunk_and_structure": getattr(self, "run_trunk_and_structure", None),
                    "skip_run_structure": getattr(self, "skip_run_structure", None),
                    "is_msa_compiled": getattr(self, "is_msa_compiled", None),
                    "is_pairformer_compiled": getattr(self, "is_pairformer_compiled", None),
                    "training": bool(self.training),
                },
                # If the trunk really ran, the distogram must differ between recycle
                # counts; a constant checksum here would mean z never left zeros.
                "pdistogram_sum": float(out["pdistogram"].float().abs().sum().item()),
            }
        return {
            "exception": False,
            "pdistogram": out["pdistogram"],
            "token_masks": batch["token_pad_mask"],
            "flops": payload,
        }

    def profiled_write_on_batch_end(self, trainer, pl_module, prediction,
                                    batch_indices, batch, batch_idx, dataloader_idx):
        if prediction.get("exception"):
            self.failed += 1
            return
        records = batch["record"]
        for i, record in enumerate(records):
            rdir = self.output_dir / record.id
            rdir.mkdir(parents=True, exist_ok=True)
            disto = prediction["pdistogram"][i].float()
            if disto.dim() == 4:                       # (L,L,1,B) -> (L,L,B)
                disto = disto.squeeze(-2)
            probs = torch.softmax(disto, dim=-1)
            n = int(prediction["token_masks"][i].cpu().bool().sum().item())
            probs = probs[:n, :n]
            np.savez_compressed(
                rdir / f"distogram_{record.id}.npz",
                probs=probs.cpu().numpy().astype(np.float16),
                length=np.int32(probs.shape[0]),
            )
            (rdir / f"flops_{record.id}.json").write_text(json.dumps(prediction["flops"]))

    Boltz2.predict_step = profiled_predict_step
    BoltzWriter.write_on_batch_end = profiled_write_on_batch_end
    # Assert the structure subtree is absent from the trunk-only count (it should
    # never run); checked per-record downstream from by_module in the sidecar.


def run_profiled_predict(yaml_dir: Path, raw_out_dir: Path, *, recycling_steps: int,
                         sampling_steps: int, diffusion_samples: int, no_kernels: bool,
                         devices: int, num_workers: int) -> None:
    """Patch boltz for trunk-only FLOP profiling, then run its real predict.

    Writes ``<raw_out_dir>/predictions/<id>/distogram_<id>.npz`` and
    ``flops_<id>.json`` (no structure/mmcif). Mirrors the CLI args the
    non-profile path passes to ``boltz predict``.
    """
    _install_patches()
    import boltz.main as bmain

    # Call boltz's own predict (click command's underlying function) so all of its
    # preprocessing / data module / trainer wiring is reused. Non-essential options
    # keep their signature defaults.
    bmain.predict.callback(
        data=str(yaml_dir),
        out_dir=str(raw_out_dir),
        model="boltz2",
        devices=int(devices),
        recycling_steps=int(recycling_steps),
        sampling_steps=int(sampling_steps),
        diffusion_samples=int(diffusion_samples),
        num_workers=int(num_workers),
        output_format="mmcif",
        override=True,
        no_kernels=bool(no_kernels),
    )
