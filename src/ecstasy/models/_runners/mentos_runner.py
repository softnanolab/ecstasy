"""Self-contained MENTOS runner — invoked via env's python from the outer adapter.

Reads a JSON bundle from stdin:
  { entry_id, sequences[], chain_ids[], msa_paths{} (ignored), out_dir, params{},
    infra{}, profile(bool) }

Bundle's "params":
  model_config_path   — path to a MENTOS training config YAML. May be a flat Hydra
                        config (top-level ``model``/``data``/...) OR a W&B-wrapped
                        ``{key: {value: ...}}`` config.yaml. Both are handled.
  model_weights_path  — path to a Lightning ``.ckpt`` of trained weights.
  run_id (optional)   — if set and no usable ``model_config_path``, the config is
                        fetched from ``$LOGS_DIR`` W&B run files via the eval helper.
  contact_cutoff_bin (optional, ignored by default) — see note below.

Writes:
  <out_dir>/contact.npz   — probs (L, L) float16 inter-chain contact probability over
                            residues (<cls>/<eos> stripped), length int32
  <out_dir>/flops.json    — (only when profile=True) inference-FLOPs sidecar

Design (mirrors ``scripts/evals/evaluate_from_wandb.py`` — the current eval ref):
  * The CURRENT MENTOS API lives in ``scripts/pretrain/pretrain_mentos.py`` (class
    ``MENTOS``). The old ``scripts.finetune.contact_prediction.ContactPrediction``
    no longer exists. We instantiate ``MENTOS(cfg)`` and load the checkpoint with
    ``strict=False`` (eval.load_model).
  * Tokenization mirrors ``mentos.data.collate_fn.CollateFn`` exactly: each chain is
    encoded as ``<cls>{seq.replace('J','L')}<eos>`` with the ESM-1b alphabet, chains
    are concatenated into one token sequence, and ``chain_ids`` carries the chain
    index per token (specials inherit their chain's index — CollateFn does the same).
  * Inference uses ``model(batch, mask_inputs=False)`` — the forward DEFAULTS to
    masking 15% of residues (MLM), which would deflate contacts; ``mask_inputs=False``
    feeds the unmasked tokens.
  * Contacts come from ``distogram_to_contacts(out.predicted_distogram)`` (64-bin
    aware, default 8 Å threshold). We do NOT hardcode a bin range or a softmax — the
    metric owns the threshold. ``contact_cutoff_bin`` in params is accepted but
    intentionally ignored so the runner stays faithful to the eval's contact head.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def _tokenize_chains(sequences, alphabet):
    """Mirror ``CollateFn._encode_chains`` / ``_build_chain_ids`` exactly.

    Each chain -> ``<cls>{seq.replace('J','L')}<eos>``; concatenate; chain_ids is
    the chain index per token (specials inherit the chain's index). Returns
    ``(tokens (1,T) long, chain_ids (1,T) long, chain_lengths list[int])``.
    """
    chain_token_ids = []
    chain_lengths = []
    for s in sequences:
        encoded = alphabet.encode(f"<cls>{s.replace('J', 'L')}<eos>")
        t = torch.tensor(encoded, dtype=torch.int64)
        chain_token_ids.append(t)
        chain_lengths.append(int(t.numel()))
    tokens = torch.cat(chain_token_ids, dim=0).unsqueeze(0)
    chain_ids = torch.cat(
        [torch.full((length,), i, dtype=torch.int64) for i, length in enumerate(chain_lengths)]
    ).unsqueeze(0)
    return tokens, chain_ids, chain_lengths


def _residue_token_indices(chain_lengths):
    """Token positions of actual residues (strip the per-chain <cls>/<eos>).

    For chain c spanning tokens ``[pos, pos+L)``, residues occupy ``[pos+1, pos+L-1)``.
    Mirrors CollateFn's residue mask (``start=pos+1, end=pos+L-1``).
    """
    idx = []
    pos = 0
    for length in chain_lengths:
        idx.extend(range(pos + 1, pos + length - 1))
        pos += length
    return torch.tensor(idx, dtype=torch.long)


def _load_cfg(model_config_path, run_id):
    """Return a flat Hydra DictConfig for ``MENTOS(cfg)``.

    Handles three shapes via the eval's own helpers:
      1. flat Hydra config (top-level ``model``/``data``/...) -> used as-is
      2. W&B-wrapped ``{key: {value: ...}}`` config.yaml -> reconstruct_config
      3. no config path but a run_id -> fetch_wandb_config($LOGS_DIR, run_id) then
         reconstruct_config
    """
    from omegaconf import DictConfig, OmegaConf

    from scripts.evals.evaluate_from_wandb import fetch_wandb_config, reconstruct_config

    if model_config_path:
        raw = OmegaConf.load(model_config_path)
        if not isinstance(raw, DictConfig):
            raise TypeError(f"expected DictConfig from {model_config_path}, got {type(raw)}")
        # W&B config.yaml wraps every top-level key as {key: {value: ...}}. Detect
        # by checking whether the top-level entries are all {value: ...} dicts.
        keys = [str(k) for k in raw if not str(k).startswith("_")]
        is_wandb_wrapped = bool(keys) and all(
            isinstance(raw[k], DictConfig) and "value" in raw[k] for k in keys
        )
        return reconstruct_config(raw) if is_wandb_wrapped else raw

    if run_id:
        logs_dir = Path(os.environ["LOGS_DIR"])
        return reconstruct_config(fetch_wandb_config(logs_dir, run_id))

    raise ValueError("mentos preset must set model_config_path or run_id")


def main():
    bundle = json.loads(sys.stdin.read())
    entry_id: str = bundle["entry_id"]
    sequences: list[str] = bundle["sequences"]
    out_dir = Path(bundle["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_params = bundle.get("params") or {}
    profile = bool(bundle.get("profile"))

    model_config_path = cfg_params.get("model_config_path")
    model_weights_path = cfg_params.get("model_weights_path")
    run_id = cfg_params.get("run_id")
    if not model_weights_path:
        raise ValueError("mentos preset must set model_weights_path")

    import mentos
    from mentos.data.esm import Alphabet
    from mentos.metrics.contact_prediction import distogram_to_contacts

    # The mentos `scripts.*` tree is co-located with the repo but not part of the
    # installed `mentos` package. Inject the repo root onto sys.path so
    # `import scripts.…` (eval helpers, MENTOS class) resolves regardless of cwd.
    # mentos.__path__[0] == <repo>/src/mentos  ->  .parent.parent == <repo>.
    mentos_repo = Path(mentos.__path__[0]).parent.parent
    if str(mentos_repo) not in sys.path:
        sys.path.insert(0, str(mentos_repo))

    from scripts.evals.evaluate_from_wandb import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[mentos] device={device}  cfg={model_config_path}  run_id={run_id}  "
        f"weights={model_weights_path}",
        flush=True,
    )

    cfg = _load_cfg(model_config_path, run_id)
    # MENTOS recycles `num_recycles + 1` pair-stack passes (pair_stack.num_recycles;
    # a5sgd6ul trains at 1). A preset may override it (e.g. 0 = single pass, no
    # distogram recycle) to sweep the recycle knob like boltz2/esmfold.
    from omegaconf import OmegaConf
    if cfg_params.get("num_recycles") is not None:
        OmegaConf.update(cfg, "model.pair_stack.num_recycles", int(cfg_params["num_recycles"]))
    eff_recycles = int(OmegaConf.select(cfg, "model.pair_stack.num_recycles") or 0)
    print(f"[mentos] num_recycles={eff_recycles}", flush=True)
    # Reuse the eval's loader verbatim: instantiates MENTOS(cfg), sets mlm mask_prob 0
    # and loss.mlm 0, loads state_dict(strict=False), .to(device), .eval().
    model = load_model(cfg, Path(model_weights_path), str(device))

    alphabet = Alphabet.from_architecture("ESM-1b")
    tokens, chain_ids_t, chain_lengths = _tokenize_chains(sequences, alphabet)
    tokens, chain_ids_t = tokens.to(device), chain_ids_t.to(device)
    T = int(tokens.shape[1])

    # Construct a valid ContactPredictionBatch. The forward (mask_inputs=False) reads
    # ONLY `tokens` and `chain_ids`; every other field is consumed by loss/metrics,
    # not the prediction path. We still fill all required fields with correctly
    # shaped/typed dummies so the dataclass is valid and `to(device)`-able.
    from mentos.dataclasses import ContactPredictionBatch

    batch = ContactPredictionBatch(
        ids=[entry_id],
        tokens=tokens,  # (1, T) int64
        chain_ids=chain_ids_t,  # (1, T) int64, chain index per token
        true_contacts=torch.full((1, T, T), -1, dtype=torch.int64, device=device),
        seq_lengths=torch.tensor([T], dtype=torch.int64, device=device),
        is_homodimer=torch.zeros(1, dtype=torch.bool, device=device),
        residue_map=torch.full((1, T), -1, dtype=torch.long, device=device),
        distance_map=None,
    )

    flops_payload = None
    with torch.no_grad():
        if profile:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import _flops

            # The WHOLE forward (incl. its num_recycles pair-stack passes) is the
            # contact-dependency subgraph — no structure module to exclude. Count all.
            out, flops_payload = _flops.profile_call(model, batch, mask_inputs=False)
        else:
            out = model(batch, mask_inputs=False)

    # distogram_to_contacts: 64-bin aware, default 8 Å threshold; returns (B, T, T).
    contact_full = distogram_to_contacts(out.predicted_distogram)[0].float().cpu()  # (T, T)

    keep = _residue_token_indices(chain_lengths)
    contact = contact_full[keep][:, keep].numpy().astype(np.float16)
    np.savez_compressed(
        out_dir / "contact.npz",
        probs=contact,
        length=np.int32(contact.shape[0]),
    )
    print(
        f"[mentos] WROTE {out_dir / 'contact.npz'}  shape={contact.shape}",
        flush=True,
    )

    if profile and flops_payload is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import _flops

        sidecar = _flops.write_flops_sidecar(
            out_dir,
            flops_payload,
            L=int(contact.shape[0]),
            msa_depth=0,
            recycles=eff_recycles,
            model="mentos",
        )
        print(f"[mentos] WROTE {sidecar}  flops={flops_payload['flops']:.3e}", flush=True)


if __name__ == "__main__":
    main()
