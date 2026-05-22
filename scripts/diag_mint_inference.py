"""Diagnostic — run MINT inference exactly like evaluate_from_wandb.load_model and
compare both the token-grid metric path (baseline) and the residue-grid + float16
round-trip path (what my ecstasy runner does).

Goal: localize whether the mismatch vs the wandb-eval baseline numbers comes from
the inference (model output) or from the post-processing (residue trim + f16 cast).
"""
import sys, numpy as np, torch
sys.path.insert(0, '/home/u6jv/harsh.u6jv/mint')

from omegaconf import OmegaConf
from mint.data.collate_fn import CollateFn
from mint.metrics.contact_prediction import distogram_to_contacts, metrics_inter_chain
from mint.trainer import freeze_and_configure_model
from scripts.finetune.contact_prediction import ContactPrediction


CFG_PATH = '/projects/u6jv/harsh/MINT_META/LOGS/MINT_AFDD_PRETRAIN_8M_35M/3khmvobe/config.yaml'
CKPT_PATH = '/projects/u6jv/harsh/MINT_META/LOGS/MINT_AFDD_PRETRAIN_8M_35M/3khmvobe/checkpoints/last.ckpt'
PT_PATH = '/projects/u6jv/public/MINT/DATA/pdb/processed/data/10/10jy.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

cfg = OmegaConf.load(CFG_PATH)

# Mirror evaluate_from_wandb.load_model exactly
model = ContactPrediction(cfg)
freeze_and_configure_model(model, cfg)
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
res = model.load_state_dict(ckpt['state_dict'], strict=False)
print('missing[:3]:', list(res.missing_keys)[:3])
print('unexpected[:3]:', list(res.unexpected_keys)[:3])
model.to(device); model.eval()

sample = torch.load(PT_PATH, weights_only=False)
seqs = sample.sequences
la, lb = len(seqs[0]), len(seqs[1])
print('seqs lens:', la, lb)

collate = CollateFn(truncation_seq_length=cfg.data.max_len)
batch = collate([sample])
batch.tokens = batch.tokens.to(device)
batch.chain_ids = batch.chain_ids.to(device)
batch.true_contacts = batch.true_contacts.to(device)

with torch.no_grad():
    esm2_out = model.esm2(batch.tokens, batch.chain_ids,
                          need_head_weights=False, repr_layers=None)
    intra_logits, inter_logits = model.contact_head(esm2_out, batch.chain_ids)

inter_contacts_token = distogram_to_contacts(inter_logits)   # (1, T, T)
print('token grid:', inter_contacts_token.shape, 'dtype:', inter_contacts_token.dtype)

# Path A: baseline path (token grid)
mA = metrics_inter_chain(inter_contacts_token, batch.true_contacts, batch.chain_ids)
print('\n[A] BASELINE PATH (token-grid metric, what evaluate_from_wandb does):')
for k in ['AUC', 'P@K', 'P@K/2', 'P@K/5']:
    print(f'  {k}: {float(mA[k]):.6f}')

# Path B: residue grid f32
keep = list(range(1, la + 1)) + list(range(la + 3, la + lb + 3))
keep_t = torch.tensor(keep, device=device)
contact_res = inter_contacts_token[0][keep_t][:, keep_t]
gt_res = sample.contact_map.long()[None].to(device)
chain_ids_res = torch.tensor([0]*la + [1]*lb, device=device)[None]
mB = metrics_inter_chain(contact_res[None], gt_res, chain_ids_res)
print('\n[B] residue-grid metric, float32:')
for k in ['AUC', 'P@K', 'P@K/2', 'P@K/5']:
    print(f'  {k}: {float(mB[k]):.6f}')

# Path C: residue grid + float16 round-trip (what my runner produces)
contact_f16 = contact_res.cpu().numpy().astype(np.float16).astype(np.float32)
mC = metrics_inter_chain(torch.from_numpy(contact_f16)[None].to(device), gt_res, chain_ids_res)
print('\n[C] residue-grid metric, float16 round-trip (matches my ecstasy runner):')
for k in ['AUC', 'P@K', 'P@K/2', 'P@K/5']:
    print(f'  {k}: {float(mC[k]):.6f}')
