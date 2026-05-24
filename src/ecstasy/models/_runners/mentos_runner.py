"""Self-contained MENTOS runner — invoked via env's python from the outer adapter.

Reads a JSON bundle from stdin:
  { entry_id, sequences[], chain_ids[], msa_paths{} (ignored), out_dir, config }

Bundle's config["model_config"] must contain:
  model_config_path  — path to a MENTOS (Hydra/OmegaConf) YAML config used to
                       instantiate the ContactPrediction LightningModule.
  model_weights_path — path to a Lightning .ckpt (or .pt) of trained weights.

Writes:
  <out_dir>/contact.npz   — probs (L, L) float16, length int32
  <out_dir>/raw/raw_inter_logits_<id>.npz (optional debug)

MENTOS is single-sequence (no MSA needed), takes per-chain residues, encodes
each chain as ``<cls>{seq}<eos>``, concatenates into a single token sequence
with chain_ids per token, and runs the dual distogram head. Inter-chain
logits are softmaxed and the bins below ``contact_threshold_bin`` (default 5,
< 8 Å Cβ-Cβ) are summed to give the (L, L) contact probability over residues
(``<cls>``/``<eos>`` positions are dropped before saving).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _tokenize_chains(sequences, alphabet):
    chain_token_ids = []
    chain_lengths = []
    for s in sequences:
        encoded = alphabet.encode(f"<cls>{s.replace('J', 'L')}<eos>")
        t = torch.tensor(encoded, dtype=torch.int64)
        chain_token_ids.append(t)
        chain_lengths.append(int(t.numel()))
    tokens = torch.cat(chain_token_ids, dim=0).unsqueeze(0)
    chain_ids = torch.cat(
        [torch.full((L,), i, dtype=torch.int64) for i, L in enumerate(chain_lengths)]
    ).unsqueeze(0)
    return tokens, chain_ids, chain_lengths


def _residue_token_indices(chain_lengths):
    """Token positions corresponding to actual residues (strip <cls>/<eos>)."""
    idx = []
    pos = 0
    for L in chain_lengths:
        idx.extend(range(pos + 1, pos + L - 1))
        pos += L
    return torch.tensor(idx, dtype=torch.long)


def main():
    bundle = json.loads(sys.stdin.read())
    entry_id: str = bundle["entry_id"]
    sequences: list[str] = bundle["sequences"]
    out_dir = Path(bundle["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = (bundle.get("config") or {}).get("model_config", {}) or {}

    model_config_path = cfg.get("model_config_path")
    model_weights_path = cfg.get("model_weights_path")
    if not model_config_path or not model_weights_path:
        raise ValueError(
            "mentos adapter requires --model_config <yaml> and --model_weights <ckpt>"
        )
    cutoff_bin: int = int(cfg.get("contact_threshold_bin", 5))

    from omegaconf import OmegaConf
    import mentos
    from mentos.data.esm import Alphabet

    # `scripts.finetune.contact_prediction` is co-located with the mentos repo but
    # not a Python package (the repo intends scripts to be run as `python -m`
    # from its root). Inject the repo root onto sys.path so `import scripts.…`
    # resolves regardless of cwd.
    mentos_repo = Path(mentos.__path__[0]).parent.parent
    if str(mentos_repo) not in sys.path:
        sys.path.insert(0, str(mentos_repo))
    from scripts.finetune.contact_prediction import ContactPrediction

    mentos_cfg = OmegaConf.load(model_config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mentos] device={device}  cfg={model_config_path}  weights={model_weights_path}", flush=True)

    model = ContactPrediction(mentos_cfg)
    ckpt = torch.load(model_weights_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        print(f"[mentos] {len(missing)} missing keys (first 3): {missing[:3]}", flush=True)
    if unexpected:
        print(f"[mentos] {len(unexpected)} unexpected keys (first 3): {unexpected[:3]}", flush=True)
    model.to(device)
    model.eval()

    alphabet = Alphabet.from_architecture("ESM-1b")
    tokens, chain_ids_t, chain_lengths = _tokenize_chains(sequences, alphabet)
    tokens, chain_ids_t = tokens.to(device), chain_ids_t.to(device)
    batch = SimpleNamespace(tokens=tokens, chain_ids=chain_ids_t)

    with torch.no_grad():
        intra_logits, inter_logits = model.forward(batch)
    probs = torch.softmax(inter_logits[0].float(), dim=-1)         # (T, T, num_bins)
    contact_full = probs[..., :cutoff_bin].sum(-1).cpu()            # (T, T)

    keep = _residue_token_indices(chain_lengths)
    contact = contact_full[keep][:, keep].numpy().astype(np.float16)
    np.savez_compressed(
        out_dir / "contact.npz",
        probs=contact,
        length=np.int32(contact.shape[0]),
    )
    print(f"[mentos] WROTE {out_dir / 'contact.npz'}  shape={contact.shape}  cutoff_bin={cutoff_bin}",
          flush=True)


if __name__ == "__main__":
    main()
