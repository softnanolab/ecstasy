"""Standalone FLOP-profiling helper shared by the model runners.

Imported by each ``_runners/<x>.py`` via its own directory (the runners are run
as scripts, ``python runner.py``, so they add ``Path(__file__).parent`` to
``sys.path`` and ``import _flops``). Depends only on ``torch`` — never on
ecstasy — so it works inside every model venv.

What it measures (see FLOPS_BENCHMARK_PLAN.md):

* **True FLOPs = 2 x MACs.** ``FlopCounterMode.get_total_flops()`` already returns
  this on torch >= 2 (verified: a pure ``Linear`` reports ``2*m*n*k``). We report
  it directly and never double again.
* **Whole-call counting, no module attribution.** Each caller wraps exactly the
  contact-map-producing call; the off-path structure/diffusion compute is avoided
  upstream (Boltz ``skip_run_structure``) or negligible (ESMFold terminal heads),
  so the counted total *is* the contact-dependency-subgraph FLOP count.
* A small top-level **per-module breakdown** is kept for sanity checks
  (affine-in-recycles; verifying a skipped subtree is ~0).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.flop_counter import FlopCounterMode


def _top_level_breakdown(flop_counts: dict[str, dict[Any, int]]) -> dict[str, int]:
    """Per-module subtree FLOPs for the root's **direct children**.

    ``FlopCounterMode.get_flop_counts()`` keys are dotted module paths
    (``Boltz2``, ``Boltz2.structure_module``, ``Boltz2.structure_module.x`` ...)
    and already aggregate each module's whole subtree. We keep only the root's
    direct children (one level down), which partition the total without the
    parent/child double counting that summing every row would cause. Useful for
    the affine-in-recycles check and to confirm a skipped subtree (e.g. Boltz
    ``structure_module`` under ``skip_run_structure``) reports ~0 FLOPs.
    ``Global`` is dropped (it duplicates ``get_total_flops``).
    """
    paths = [p for p in flop_counts if p != "Global"]
    if not paths:
        return {}
    root_len = min(len(p.split(".")) for p in paths)  # the root module's depth (==1)
    child_len = root_len + 1
    return {
        p: int(sum(flop_counts[p].values()))
        for p in paths
        if len(p.split(".")) == child_len
    }


def profile_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """Run ``fn(*args, **kwargs)`` under ``FlopCounterMode`` and return
    ``(result, payload)``.

    ``payload`` keys:
      ``flops``      true FLOPs (2*MACs) of the whole counted call
      ``macs``       ``flops // 2``
      ``by_module``  top-level per-module subtree FLOPs (audit/sanity)
    """
    fc = FlopCounterMode(display=False)
    with fc:
        result = fn(*args, **kwargs)
    total = int(fc.get_total_flops())
    by_module = _top_level_breakdown(fc.get_flop_counts())
    payload = {"flops": total, "macs": total // 2, "by_module": by_module}
    return result, payload


def write_flops_sidecar(out_dir: str | Path, payload: dict[str, Any], **meta: Any) -> Path:
    """Write ``<out_dir>/flops.json`` merging ``payload`` with provenance ``meta``
    (e.g. ``L``, ``msa_depth``, ``recycles``, ``model``). Returns the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {**payload, **{k: v for k, v in meta.items() if v is not None}}
    path = out_dir / "flops.json"
    path.write_text(json.dumps(record, indent=1))
    return path
