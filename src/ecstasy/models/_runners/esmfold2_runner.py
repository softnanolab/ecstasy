"""Self-contained ESMFold2 runner — invoked via env's python from the outer adapter.

Reads a JSON bundle from stdin:
  { entry_id, sequences[], chain_ids[], msa_paths{} (ignored), out_dir, config }

Bundle's "params" (all optional):
  num_loops             — recycle iterations; ESMFold2's analogue of ESMFold's
                          num_recycles / Boltz's recycling_steps. This is the compute
                          knob for the r0/r1/r3/r5 ladder.
  contact_threshold_a   — Cb-Cb distance in ANGSTROM below which a pair counts as a
                          contact (default 7.9375, the MENTOS GT threshold). Deliberately
                          a distance, not a bin index — see "Binning" below.
  checkpoint            — HF repo id (default "biohub/ESMFold2").
  max_train_date        — refuse checkpoints trained past this date (default 2025-01-01).
  seed                  — forwarded to input prep and the model.

Writes:
  <out_dir>/contact.npz   — probs (L, L) float16, length int32

Binning — the one thing that does NOT carry over from the other models
----------------------------------------------------------------------
Every other model here (Boltz-2, and the MENTOS ground truth itself) uses a 64-bin
distogram over 2-22 A, so a contact is "bins below index 19" == 7.9375 A. ESMFold2's
*output* head does not: it is 128 bins with midpoints ~1.5-54.5 A (Algorithm 12). The
64-bin 2-22 A grid that appears in the paper is the *input conditioning* distogram, a
different object. Reusing contact_cutoff_bin=19 here would score contacts at ~8.9 A and
silently inflate P@K against every other row of the benchmark.

So this runner takes an ANGSTROM threshold and derives the bin count from the model's
own grid, and asserts the grid is the size it expects. See ESMFOLD2_INTEGRATION.md.

Determinism
-----------
`ESMFold2InputBuilder.fold()` is NOT used: it runs the full diffusion sampler and
defaults to lm_dropout=0.3, which is deliberately stochastic (fresh dropout mask per
loop, for ensemble diversity). A benchmark number has to be reproducible, so this runs
the forward directly with LM dropout disabled, one diffusion sample, and a single
sampling step — which also keeps the measured FLOPs on the contact-map dependency
subgraph rather than paying for structure generation we discard.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

# 128 output bins, Algorithm 12. Kept as the expected grid; the real one is read off the
# model's logits and cross-checked against this so a checkpoint change fails loudly.
_N_BINS = 128


def _bin_midpoints() -> torch.Tensor:
    """Distance-bin midpoints of the ESMFold2 distogram head (1.5 .. 54.5 A)."""
    boundaries = torch.linspace(2, 52.0, _N_BINS - 1)
    lower = torch.tensor([1.0])
    upper = torch.tensor([52.0 + 5.0])
    edges = torch.cat((lower, boundaries, upper))
    return (edges[:-1] + edges[1:]) / 2


def _assert_clean_checkpoint(model, repo_id: str, max_train_date: str) -> None:
    """Refuse a checkpoint whose training data overlaps the eval window.

    The public list includes ESMFold2-Experimental-Cutoff2025 variants alongside the
    plain ones. recent_pp is a temporal holdout over 2025-06-30 -> 2026-01-21, so a
    Cutoff2025 checkpoint would be contaminated. Check the loaded config rather than the
    repo id: a typo or an upstream default change would be invisible in the results
    otherwise, and a contaminated headline number is worse than a crash.
    """
    cfg = getattr(model, "config", None)
    found = None
    for attr in ("training_data_cutoff", "data_cutoff", "cutoff_date", "train_cutoff"):
        val = getattr(cfg, attr, None)
        if val:
            found = f"{attr}={val}"
            if str(val) > max_train_date:
                raise RuntimeError(
                    f"checkpoint {repo_id!r} reports {attr}={val}, past max_train_date="
                    f"{max_train_date}; this would contaminate the recent_pp holdout"
                )
            break
    if "cutoff2025" in repo_id.lower():
        raise RuntimeError(
            f"checkpoint {repo_id!r} is a Cutoff2025 variant — its training data overlaps "
            f"the recent_pp eval window (2025-06-30 -> 2026-01-21). Use biohub/ESMFold2."
        )
    print(f"[esmfold2] checkpoint={repo_id} cutoff={found or 'not declared in config'}",
          flush=True)


def main():
    bundle = json.loads(sys.stdin.read())
    entry_id: str = bundle["entry_id"]
    sequences: list[str] = bundle["sequences"]
    chain_ids: list[str] = bundle.get("chain_ids") or [str(i) for i in range(len(sequences))]
    out_dir = Path(bundle["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = bundle.get("params") or {}

    num_loops = int(cfg.get("num_loops", 0))
    threshold_a = float(cfg.get("contact_threshold_a", 7.9375))
    repo_id = str(cfg.get("checkpoint", "biohub/ESMFold2"))
    max_train_date = str(cfg.get("max_train_date", "2025-01-01"))
    seed = cfg.get("seed")
    seed = int(seed) if seed is not None else 0
    profile = bool(bundle.get("profile"))
    if profile:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import _flops

    from transformers import AutoConfig
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    from transformers.models.esmfold2.modeling_esmfold2_common import (
        BACKEND_CUEQ, BACKEND_FUSED, CUE_AVAILABLE, TRITON_KERNELS_AVAILABLE,
    )
    from transformers.models.esmfold2.modeling_esmfold2_experimental import (
        ESMFold2ExperimentalModel,
    )

    from esm.models.esmfold2.processor import ESMFold2InputBuilder
    from esm.utils.structure.input_builder import ProteinInput, StructurePredictionInput

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[esmfold2] device={device}  num_loops={num_loops}  "
          f"threshold={threshold_a} A", flush=True)

    # Two model classes ship side by side and they are NOT interchangeable: the released
    # checkpoints declare config.type == "release" and want ESMFold2Model, while the
    # -Experimental checkpoints want ESMFold2ExperimentalModel. The packaged code
    # disambiguates on exactly this field (see _lm_dropout_context, which reads
    # config.lm_encoder.lm_dropout for release and top-level config.lm_dropout for
    # experimental). The binder-design cookbook uses the experimental class only because
    # it loads experimental checkpoints — copying that blindly against biohub/ESMFold2
    # mismatches the weights. Pick from the config rather than hardcoding either.
    _cfg = AutoConfig.from_pretrained(repo_id)
    ckpt_type = getattr(_cfg, "type", None)
    model_cls = ESMFold2ExperimentalModel if ckpt_type == "experimental" else ESMFold2Model
    print(f"[esmfold2] config.type={ckpt_type!r} -> {model_cls.__name__}", flush=True)

    model = model_cls.from_pretrained(repo_id)
    _assert_clean_checkpoint(model, repo_id, max_train_date)

    # Triton / cuequivariance kernels are optional accelerations with an explicit None
    # fallback; an A100 without them is supported. Released code is bfloat16 throughout,
    # so the paper's float8 production setup is not required here.
    kernel_backend = None
    if TRITON_KERNELS_AVAILABLE:
        kernel_backend = BACKEND_FUSED
    elif CUE_AVAILABLE:
        kernel_backend = BACKEND_CUEQ
    model.set_kernel_backend(kernel_backend)
    print(f"[esmfold2] kernel_backend={kernel_backend}", flush=True)

    # Determinism. This is NOT belt-and-braces — the release checkpoint is stochastic at
    # inference out of the box. biohub/ESMFold2 ships lm_encoder.lm_dropout=0.25 with
    # per_loop_lm_dropout=True, and modeling_esmfold2.py applies it with a hardcoded
    # `F.dropout(..., training=True)`:
    #
    #     # training=True forces dropout under eval(), matching the per-loop
    #     # dropout strategy used at train time.
    #
    # so .eval() does not switch it off, and the gate never consults the config's own
    # force_lm_dropout_during_inference=False. Left alone, every run returns a different
    # contact map. configure_lm_dropout() only exists on the experimental class, so zero
    # it on the config, which works for both.
    #
    # Note this is a deliberate methodological choice, not just a bug workaround: the
    # packaged fold() defaults to lm_dropout=0.3 and describes it as the paper's
    # folding-eval value, i.e. the published protocol *ensembles* several stochastic
    # folds. A single dropout-free pass is what makes ESMFold2 comparable with ESMFold
    # and Boltz-2 here (one deterministic forward, one FLOPs number), so expect this to
    # read lower than an ensembled result. `lm_dropout` is exposed as a param for anyone
    # who wants the ensemble behaviour back.
    lm_dropout = float(cfg.get("lm_dropout", 0.0))
    lm_cfg = getattr(model.config, "lm_encoder", None)
    before = getattr(lm_cfg, "lm_dropout", None) if lm_cfg is not None else None
    if lm_cfg is not None and hasattr(lm_cfg, "lm_dropout"):
        lm_cfg.lm_dropout = lm_dropout
        if hasattr(lm_cfg, "per_loop_lm_dropout"):
            lm_cfg.per_loop_lm_dropout = lm_dropout > 0.0
    if hasattr(model.config, "lm_dropout"):
        model.config.lm_dropout = lm_dropout
    if hasattr(model, "configure_lm_dropout"):
        model.configure_lm_dropout(lm_dropout, force_lm_dropout_during_inference=False)
    print(f"[esmfold2] lm_dropout {before} -> {lm_dropout} "
          f"({'deterministic' if lm_dropout == 0.0 else 'STOCHASTIC'})", flush=True)

    model = model.to(device=device).eval().requires_grad_(False)

    spi = StructurePredictionInput(sequences=[
        ProteinInput(id=str(cid), sequence=seq, msa=None)     # single-sequence: msa=None
        for cid, seq in zip(chain_ids, sequences)
    ])
    builder = ESMFold2InputBuilder()
    features, chain_infos = builder.prepare_input(spi, seed=seed, device=device)

    flops_payload = None

    def _forward():
        # bfloat16 autocast is required, not an optimisation. from_pretrained keeps the
        # checkpoint's bf16 weights while prepare_input hands back fp32 features, so an
        # un-autocast forward dies inside the pair transition with
        #   "self and mat2 must have the same dtype, but got BFloat16 and Float".
        # The cookbook does not hit this only because its design path happens to build
        # its own tensors at the model dtype. bf16 is also the released model's native
        # inference precision, so this matches intended usage rather than working around
        # it. Dtype does not affect the FLOP count — the counter keys off matmul shapes.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            return model(
                **features,
                num_diffusion_samples=1,
                num_sampling_steps=1,    # off the contact path; keeps FLOPs on the trunk
                num_loops=num_loops,
                calculate_confidence=False,
                seed=seed,
            )

    with torch.no_grad():
        if profile:
            output, flops_payload = _flops.profile_call(_forward)
        else:
            output = _forward()

    logits = output["distogram_logits"]
    if logits.dim() == 4:                      # (B, L, L, bins) -> drop batch
        logits = logits[0]
    n_bins = logits.shape[-1]
    if n_bins != _N_BINS:
        raise RuntimeError(
            f"expected a {_N_BINS}-bin distogram head, got {n_bins}. The Angstrom->bin "
            f"mapping in this runner is calibrated to Algorithm 12's grid; re-derive it "
            f"before trusting any contact map from this checkpoint."
        )

    mids = _bin_midpoints()
    cutoff_bin = int((mids < threshold_a).sum())
    if not 0 < cutoff_bin < n_bins:
        raise RuntimeError(f"threshold {threshold_a} A maps to bin {cutoff_bin}, out of range")
    print(f"[esmfold2] threshold {threshold_a} A -> summing bins 0..{cutoff_bin - 1} "
          f"of {n_bins} (last included midpoint {mids[cutoff_bin - 1]:.4f} A)", flush=True)

    probs = torch.softmax(logits.float(), dim=-1)
    contact = probs[..., :cutoff_bin].sum(-1).cpu().numpy()

    # Token order/identity: for pure-protein chains tokens are 1:1 with residues, but
    # assert it rather than assume — a silent off-by-one here would look like a merely
    # mediocre model. asym_id is the model's own per-token chain label.
    expected = [len(s) for s in sequences]
    if "asym_id" in features:
        asym = features["asym_id"]
        asym = asym[0] if asym.dim() > 1 else asym
        asym = asym.detach().cpu().numpy()
        _, counts = np.unique(asym, return_counts=True)
        order = [int(c) for c in counts[np.argsort(np.unique(asym))]]
        if order != expected:
            raise RuntimeError(
                f"per-chain token counts {order} != sequence lengths {expected}; the "
                f"contact map would not line up with the ground truth"
            )
    if contact.shape[0] != sum(expected):
        raise RuntimeError(
            f"contact map is {contact.shape[0]} tokens, expected {sum(expected)} residues"
        )

    contact = contact.astype(np.float16)
    np.savez_compressed(out_dir / "contact.npz",
                        probs=contact, length=np.int32(contact.shape[0]))
    print(f"[esmfold2] WROTE {out_dir / 'contact.npz'}  shape={contact.shape}  "
          f"cutoff_bin={cutoff_bin}", flush=True)

    if profile and flops_payload is not None:
        sidecar = _flops.write_flops_sidecar(
            out_dir, flops_payload,
            L=int(contact.shape[0]), msa_depth=0, recycles=num_loops, model="esmfold2",
        )
        print(f"[esmfold2] WROTE {sidecar}  flops={flops_payload['flops']:.3e}", flush=True)


if __name__ == "__main__":
    main()
