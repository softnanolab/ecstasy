"""Run MSA Pairformer on the notebook's reference 1B70 to verify pipeline."""
import os, sys, numpy as np, torch
from urllib.request import urlopen
sys.path.insert(0, "/home/u6jv/harsh.u6jv/ecstasy/modules/msa_pairformer")
from MSA_Pairformer.dataset import MSA, aa2tok_d, prepare_msa_masks
from MSA_Pairformer.model import MSAPairformer

os.environ["PATH"] = "/home/u6jv/harsh.u6jv/ecstasy/tools/hhsuite/bin:" + os.environ.get("PATH", "")

device = torch.device("cuda:0")
print(f"device: {device}", flush=True)
WEIGHTS = "/projects/u6jv/ecstasy/weights/msa_pairformer"
model = MSAPairformer.from_pretrained(device=device, weights_dir=WEIGHTS).to(torch.bfloat16).eval()
print("model loaded", flush=True)


def run(msa_file, chain_break_idx, label, hhfilter_kwargs=None):
    if hhfilter_kwargs is None:
        hhfilter_kwargs = {"binary": "hhfilter"}
    np.random.seed(42)
    msa = MSA(
        msa_file_path=msa_file,
        max_seqs=512,
        max_length=10240,
        max_tokens=int(1e12),
        diverse_select_method="hhfilter",
        hhfilter_kwargs=hhfilter_kwargs,
    )
    tok = msa.diverse_tokenized_msa
    msa_in = tok.unsqueeze(0).to(device)
    mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(msa_in, device=device)
    onehot = torch.nn.functional.one_hot(msa_in, num_classes=len(aa2tok_d)).float().to(device).to(torch.bfloat16)
    with torch.no_grad():
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            out = model(
                msa=onehot,
                mask=mask, msa_mask=msa_mask,
                full_mask=full_mask, pairwise_mask=pairwise_mask,
                complex_chain_break_indices=[[chain_break_idx]],
                return_seq_weights=True,
            )
    probs = out["predicted_cb_contacts"][0].float().cpu().numpy()
    L = probs.shape[0]
    la, lb = chain_break_idx, L - chain_break_idx
    inter = probs[:la, la:la+lb]
    inter_avg = 0.5 * (inter + probs[la:la+lb, :la].T)
    print(f"\n=== {label} ===")
    print(f"  L={L}  la={la}  lb={lb}  MSA depth (post-hhfilter): {tok.shape[0]}")
    print(f"  inter shape={inter_avg.shape}  max={inter_avg.max():.4f}  mean={inter_avg.mean():.4f}")
    top = np.unravel_index(np.argsort(-inter_avg.flatten())[:10], inter_avg.shape)
    print(f"  Top-10 interchain predicted contacts (residue indices on each chain, 0-based):")
    for a, b in zip(top[0], top[1]):
        print(f"    A[{a:>3}] - B[{b:>3}]  prob={inter_avg[a,b]:.4f}")
    return inter_avg, probs


# Run 1B70 reference (notebook example)
inter_1b70, _ = run(
    "/home/u6jv/harsh.u6jv/ecstasy/modules/msa_pairformer/data/1B70_A_1B70_B.fas",
    chain_break_idx=265,
    label="1B70 (notebook reference)",
)

# Run 8tp8 with notebook-default hhfilter_kwargs (no cov/qid override)
inter_8tp8_def, _ = run(
    "/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/msas/8tp8_A_B.a3m",
    chain_break_idx=325,
    label="8tp8 with hhfilter defaults (qid=30)",
)

# Run 8tp8 with cov=70 qid=15 (what we did)
inter_8tp8_q15, _ = run(
    "/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/msas/8tp8_A_B.a3m",
    chain_break_idx=325,
    label="8tp8 with cov=70 qid=15 (our setting)",
    hhfilter_kwargs={"binary": "hhfilter", "cov": 70, "qid": 15},
)

# Compute 1B70 GT (download structure)
print("\n=== 1B70 GT verification ===")
try:
    url = "https://files.rcsb.org/download/1b70.pdb"
    r = urlopen(url, timeout=30).read().decode("utf-8")
    pdb_text = r
    # Parse with biotite to compute Cβ-Cβ for chains A and B
    from io import StringIO
    from biotite.structure.io.pdb import PDBFile
    pdb = PDBFile.read(StringIO(pdb_text))
    s = pdb.get_structure(model=1)
    from biotite.structure import filter_amino_acids
    s = s[filter_amino_acids(s)]
    chains = sorted(set(s.chain_id))
    print(f"  chains in 1B70 model 1: {chains}")
    # Map chain A = pheS, chain B = pheT — paper notation
    cA = s[s.chain_id == chains[0]]
    cB = s[s.chain_id == chains[1]]
    print(f"  chain {chains[0]}: {len(set(cA.res_id))} res; chain {chains[1]}: {len(set(cB.res_id))} res")
except Exception as e:
    print(f"  GT fetch failed: {e}")
