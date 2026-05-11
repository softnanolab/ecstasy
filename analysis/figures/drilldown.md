# Drill-down: both-fail entries and chain length

both_fail = MINT P@K < 0.3 AND Boltz-2 +MSA P@K < 0.3  →  n = 549 (37% of 1473)

## Median feature comparison (both_fail vs everyone else)

| feature | both_fail (n=549) | other (n=924) | Δ (fail − other) |
|---|---|---|---|
| L_min | 222 | 240.5 | -18.5 |
| L_max | 253 | 284 | -31 |
| L_total | 483 | 543.5 | -60.5 |
| L asymmetry | 0.0102 | 0.01074 | -0.0005343 |
| MSA depth (min) | 3049 | 2435 | +614 |
| iface density | 0.0009357 | 0.001598 | -0.0006623 |
| # true inter contacts | 51 | 98 | -47 |
| homo fraction | 0.27 | 0.24 | +0.02 |

## Top 20 both-fail entries by interface size (highest n_true_inter)

| entry_id | L_min | L_max | n_seqs_min | interface_density | n_true_inter | mint_pak | boltz_msa_pak |
|---|---|---|---|---|---|---|---|
| 8uwm | 277 | 277 | 3989 | 0.00529 | 406 | 0.0123 | 0.158 |
| 8t87 | 277 | 277 | 3989 | 0.00519 | 398 | 0.0126 | 0.113 |
| 8tfw | 277 | 277 | 3989 | 0.00515 | 395 | 0.0101 | 0.122 |
| 8uix | 277 | 279 | 3751 | 0.00511 | 395 | 0.0101 | 0.0608 |
| 8t88 | 277 | 277 | 3989 | 0.00511 | 392 | 0.0128 | 0.117 |
| 8ugm | 277 | 277 | 3989 | 0.00511 | 392 | 0.0128 | 0.107 |
| 8sbq | 277 | 277 | 3989 | 0.00507 | 389 | 0.0129 | 0.116 |
| 9dro | 277 | 277 | 3989 | 0.005 | 384 | 0.0104 | 0.104 |
| 9edj | 276 | 278 | 3875 | 0.00485 | 372 | 0.00806 | 0.159 |
| 8i6j | 170 | 500 | 5219 | 0.00422 | 359 | 0 | 0 |
| 7xt2 | 388 | 388 | 7804 | 0.00186 | 280 | 0 | 0.0571 |
| 7sku | 299 | 301 | 153 | 0.0029 | 261 | 0 | 0.295 |
| 8qlo | 437 | 455 | 2489 | 0.0013 | 259 | 0.00386 | 0 |
| 9cys | 229 | 277 | 2188 | 0.00388 | 246 | 0.00813 | 0 |
| 7ula | 316 | 430 | 1072 | 0.00179 | 243 | 0 | 0 |
| 8gp6 | 261 | 285 | 105 | 0.00323 | 240 | 0.00833 | 0 |
| 9dsb | 146 | 147 | 5776 | 0.0109 | 234 | 0.0513 | 0 |
| 9ckx | 222 | 222 | 8607 | 0.00457 | 225 | 0.00444 | 0.12 |
| 9oa9 | 270 | 280 | 5648 | 0.00291 | 220 | 0.0273 | 0.0136 |
| 8cm0 | 114 | 159 | 185 | 0.0118 | 214 | 0.028 | 0.014 |

## Chain-length spearman ρ vs P@K

| feature | MINT | Boltz +MSA | Boltz -MSA |
|---|---|---|---|
| L_min | -0.14 | +0.04 | -0.29 |
| L_max | -0.27 | +0.02 | -0.30 |
| L_total | -0.24 | +0.03 | -0.31 |
| L asymmetry | -0.05 | -0.03 | +0.06 |

## Figures

- `fig11_bothfail_feature_distributions.png`
- `fig12_chain_length_detail.png`
- `fig13_length_x_depth_heatmap.png`
- `fig14_bothfail_vs_bothgood.png`
- `fig15_length_thresholds.png`
- `fig16_failure_clusters.png`
