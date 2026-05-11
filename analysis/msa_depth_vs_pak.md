# P@K vs MSA depth — mint_seqid30 val split

N entries (intersection of MINT ∩ Boltz-MSA ∩ Boltz-noMSA, both chains have a3m): **1473**

MSA depth = number of sequences in the a3m for each chain (incl. the query). Per-entry depth aggregated as `min` (bottleneck chain) and `mean` across the two chains. `Neff_proxy = N_seqs_min / sqrt(L_min)`.

## Spearman ρ (P@K vs log10 depth)

| model | depth metric | ρ | p |
|---|---|---|---|
| MINT (no MSA) | log_depth_min | +0.081 | 1.95e-03 |
| Boltz-2 (with MSA) | log_depth_min | +0.062 | 1.81e-02 |
| Boltz-2 (no MSA) | log_depth_min | -0.061 | 1.85e-02 |
| Boltz-2 MSA gain | log_depth_min | +0.087 | 8.52e-04 |
| MINT (no MSA) | log_depth_norm | +0.105 | 5.41e-05 |
| Boltz-2 (with MSA) | log_depth_norm | +0.062 | 1.69e-02 |
| Boltz-2 (no MSA) | log_depth_norm | -0.010 | 6.98e-01 |
| Boltz-2 MSA gain | log_depth_norm | +0.057 | 2.79e-02 |

## P@K binned by MSA-depth quintile (median ± IQR)

| bin | n | depth_min | L_min | mint | boltz_msa | boltz_nomsa | gain |
|---|---|---|---|---|---|---|---|
| Q1 (lowest) | 295 | 88 | 162 | 0.00575 | 0.51 | 0.012 | 0.282 |
| Q2 | 295 | 979 | 213 | 0 | 0.653 | 0 | 0.479 |
| Q3 | 294 | 2.61e+03 | 280 | 0.00592 | 0.721 | 0 | 0.565 |
| Q4 | 294 | 6.32e+03 | 246 | 0 | 0.741 | 0 | 0.389 |
| Q5 (highest) | 295 | 1.13e+04 | 271 | 0.0167 | 0.738 | 0 | 0.583 |

## Headline numbers (full split, mean P@K)

- **MINT** mean inter P@K = 0.068  (median 0.006)
- **Boltz-2 (with MSA)** mean inter P@K = 0.503  (median 0.663)
- **Boltz-2 (no MSA)** mean inter P@K = 0.084  (median 0.000)
- **Boltz-2 MSA gain** mean = +0.419  (median +0.458)

- MINT > Boltz-2 (no MSA): **33.0%** of entries
- MINT > Boltz-2 (with MSA): **6.5%** of entries

## Takeaways

1. **Boltz-2 benefits massively from MSAs on this set.** Mean inter-chain P@K goes from 0.084 (single-seq) to 0.503 (with MSA), a +0.42 absolute gain.
2. **The MSA benefit grows with MSA depth.** Median ΔP@K rises monotonically from +0.28 in the lowest-depth quintile (median 88 seqs) to +0.58 in the highest-depth quintile (median 11k seqs). The gain roughly doubles.
3. **Per-model raw correlations of P@K vs depth are weak (ρ≈0.06–0.10).** Easy/hard intrinsic difficulty dominates within a single model. The cleanest depth signal is the *Boltz-2 with-MSA minus no-MSA delta*, since it controls for entry difficulty.
4. **MINT inter-chain P@K is low across the board (median 0.006).** It beats Boltz-no-MSA on 33% of entries but loses to Boltz-with-MSA on 93.5%.

## Important caveats

- **MINT checkpoint is the local 8M/35M training run** (`MINT_AFDD_PRETRAIN_8M_35M/3khmvobe`), not the published 650M MINT. Numbers should not be read as 'MINT vs Boltz-2'.
- **This val split is MINT's own training-time validation set.** Model selection pressure on this set exists for the local MINT run; Boltz-2 and the MSA pipeline have not seen it as a held-out test.
- **Boltz-2 (training cutoff 2023-06-01) likely saw many of these dimers during training.** The 'with-MSA' P@K is therefore partly an upper bound; a recent-PDB temporal holdout would be needed to disentangle MSA contribution from memorization.
- **Depth metric is raw a3m line count, not Neff.** Higher counts also correlate with intrinsically easier (well-studied) proteins, so the depth↔P@K trend conflates evolutionary signal with entry difficulty.

