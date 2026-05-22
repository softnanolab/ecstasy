"""Run our exact pipeline on the notebook's `test/msa.a3m` as MONOMER.

If our resulting (110, 110) ConFind matrix matches the notebook's
`predicted_confind_contacts.txt`, our inference path is bit-identical
(modulo bf16 sampling noise from hhfilter). Any remaining difference
isolates to the hhfilter subsample (random rows kept).
"""
import os, sys, numpy as np, torch
sys.path.insert(0, "/home/u6jv/harsh.u6jv/ecstasy/modules/msa_pairformer")
from MSA_Pairformer.dataset import MSA, aa2tok_d, prepare_msa_masks
from MSA_Pairformer.model import MSAPairformer

os.environ["PATH"] = "/home/u6jv/harsh.u6jv/ecstasy/tools/hhsuite/bin:" + os.environ.get("PATH", "")

device = torch.device("cuda:0")
print(f"device: {device}")
model = MSAPairformer.from_pretrained(
    device=device, weights_dir="/projects/u6jv/ecstasy/weights/msa_pairformer"
).to(torch.bfloat16).eval()
model.turn_on_query_biasing()
print("model loaded, query biasing ON")

A3M = "/projects/u6jv/ecstasy/tmp/notebook_verification/test/msa.a3m"
np.random.seed(42)
msa = MSA(
    msa_file_path=A3M,
    max_seqs=512,
    max_length=110,
    max_tokens=int(1e12),
    diverse_select_method="hhfilter",
    hhfilter_kwargs={"binary": "hhfilter"},
)
tok = msa.diverse_tokenized_msa
print(f"MSA loaded: shape {tok.shape}  (depth, L)")

msa_in = tok.unsqueeze(0).to(device)
mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(msa_in, device=device)
onehot = torch.nn.functional.one_hot(msa_in, num_classes=len(aa2tok_d)).float().to(device).to(torch.bfloat16)

with torch.no_grad():
    with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
        out = model(
            msa=onehot,
            mask=mask, msa_mask=msa_mask,
            full_mask=full_mask, pairwise_mask=pairwise_mask,
            complex_chain_break_indices=None,  # MONOMER mode
            return_seq_weights=True,
        )
ours_cf = out["predicted_confind_contacts"][0].float().cpu().numpy()
ours_cb = out["predicted_cb_contacts"][0].float().cpu().numpy()
print(f"our output shape: {ours_cf.shape}")
print(f"ConFind range: [{ours_cf.min():.6f}, {ours_cf.max():.6f}]")

# Load notebook output
nb = np.loadtxt("/projects/u6jv/ecstasy/tmp/notebook_verification/test/predicted_confind_contacts.txt")
print(f"notebook output shape: {nb.shape}")

from scipy.stats import spearmanr
print()
print("=== Bit-by-bit comparison: notebook vs our pipeline (same MSA, monomer) ===")
print(f"  Max |diff|:   {np.abs(nb - ours_cf).max():.6f}")
print(f"  Mean |diff|:  {np.abs(nb - ours_cf).mean():.6f}")
print(f"  RMSE:         {np.sqrt(((nb - ours_cf)**2).mean()):.6f}")
print(f"  Pearson:      {np.corrcoef(nb.flatten(), ours_cf.flatten())[0,1]:.6f}")
print(f"  Spearman:     {spearmanr(nb.flatten(), ours_cf.flatten())[0]:.6f}")

# Top-10 contacts in each
def topk(m, k=10):
    n = m.shape[0]
    tu = np.triu(np.ones_like(m, dtype=bool), k=1)
    idx = np.argwhere(tu)
    order = np.argsort(-m[tu])[:k]
    return [(int(idx[o,0]), int(idx[o,1]), float(m[idx[o,0], idx[o,1]])) for o in order]
print()
print("Top-10 notebook | Top-10 ours:")
nb_top = topk(nb, 10)
ours_top = topk(ours_cf, 10)
for (i1, j1, p1), (i2, j2, p2) in zip(nb_top, ours_top):
    marker = "*" if (i1, j1) == (i2, j2) else " "
    print(f"  {marker} ({i1:>3d},{j1:>3d}) p={p1:.4f}  |  ({i2:>3d},{j2:>3d}) p={p2:.4f}")
nb_set50 = set((i,j) for i,j,_ in topk(nb, 50))
our_set50 = set((i,j) for i,j,_ in topk(ours_cf, 50))
print(f"\nTop-50 overlap: {len(nb_set50 & our_set50)} / 50")

# Save our output for reference
np.savetxt("/projects/u6jv/ecstasy/tmp/notebook_verification/our_predicted_confind_contacts.txt", ours_cf, fmt='%.18e')
print(f"\nWrote our output to /projects/u6jv/ecstasy/tmp/notebook_verification/our_predicted_confind_contacts.txt")
