# Where MINT and Boltz-2 work, where they don't

n = 1473 dimers from mint_seqid30 val split, after joining MINT 3khmvobe rerun + Boltz-2 ±MSA per-entry P@K + chain MSA depth.

## Mean P@K headline

| model | mean | median | P@K ≥ 0.3 | P@K ≥ 0.6 |
|---|---|---|---|---|
| MINT (3khmvobe, no MSA) | 0.076 | 0.005 | 8% | 2% |
| Boltz-2 single-seq | 0.084 | 0.000 | 9% | 4% |
| Boltz-2 + MSA | 0.503 | 0.663 | 62% | 53% |

## Quadrants at P@K ≥ 0.3

- **Both good**: 118 (8%)
- **Boltz only**: 802 (54%)
- **MINT only**: 4 (0%)
- **Both fail**: 549 (37%)

## Correlation table (Spearman ρ vs inter P@K)

| feature | MINT | Boltz +MSA | Boltz -MSA | MSA gain |
|---|---|---|---|---|
| log10 MSA depth | +0.10 | +0.06 | -0.06 | +0.09 |
| L_min | -0.14 | +0.04 | -0.29 | +0.20 |
| iface density | +0.48 | +0.24 | +0.49 | +0.05 |
| chain asymmetry | -0.05 | -0.03 | +0.06 | -0.04 |

## Figures

- `fig01_pak_vs_depth_2dhist.png`
- `fig02_cdfs.png`
- `fig03_stratified_by_length.png`
- `fig04_stratified_by_homo.png`
- `fig05_stratified_by_density.png`
- `fig06_head2head_mint_vs_boltz.png`
- `fig07_winloss_per_feature.png`
- `fig08_difficulty_profile.png`
- `fig09_msa_gain_drivers.png`
- `fig10_quadrant_analysis.png`
