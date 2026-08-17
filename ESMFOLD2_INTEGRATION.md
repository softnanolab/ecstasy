# ESMFold2 integration spec

Design notes for adding ESMFold2 as an ecstasy contact-prediction model, to be built
after the ESMFold / Boltz-2 no-MSA ladders land. Everything below was read off the
released source (`github.com/Biohub/esm`, `esm` v3.3.0) rather than the paper — where the
two disagree, the code wins and the disagreement is called out.

Scope for this round: **single-sequence only** (`msa=None`). No MSA pipeline work.

---

## 1. Why it needs its own runner

ESMFold2 is not a drop-in variant of ESMFold. It is ESMC-6B (frozen) plus a recurrent
pair-only folding trunk plus a diffusion head, and it differs from every other model in
the matrix in two ways that touch scoring directly:

- Its distogram head is **128 bins over ~1.5–54.5 Å**, not the 64-bin 2–22 Å grid the
  rest of the matrix uses. See §4 — this is the single highest-risk detail.
- It takes **genuine multi-chain input**, so it needs no poly-G linker and no
  `residue_index_offset` hack. Chain boundaries come back as metadata instead of being
  reconstructed by the caller.

---

## 2. Environment

Hard constraint from `pyproject.toml`:

```toml
requires-python = ">=3.12,<3.13"
"transformers @ git+https://github.com/Biohub/transformers.git@main",
```

Two consequences:

- **A new `${ENVS_ROOT}/.venv-esmfold2` is mandatory.** The existing `.venv-esmfold` is
  Python **3.7** (fair-esm 2.0.1) and cannot host this. They must not be merged.
- `transformers` must come from the **Biohub fork**, not PyPI. Upstream transformers does
  not carry `ESMFold2ExperimentalModel`. Pin the commit rather than tracking `@main`, so
  the benchmark stays reproducible.

License is MIT, and inference runs locally — no API dependency.

## 3. Checkpoint selection — contamination hazard

The cookbook enumerates four public checkpoints:

```
ESMFold2-Experimental                  <- use this one
ESMFold2-Experimental-Cutoff2025
ESMFold2-Experimental-Fast
ESMFold2-Experimental-Fast-Cutoff2025
```

`recent_pp` is a temporal holdout over **2025-06-30 → 2026-01-21**. The plain checkpoints
have a **September 2021** training cutoff, which is cleanly before that window. The
`-Cutoff2025` variants do not, and loading one would silently contaminate the headline
comparison with structures the model may have trained on.

**Requirement:** the runner must read the cutoff off the loaded config and fail loudly if
it is not the expected one. Do not trust the repo id string alone — a typo or a default
change upstream would be undetectable in the results otherwise.

## 4. Contact extraction — do NOT reuse `contact_cutoff_bin: 19`

This corrects an earlier assumption in the benchmark plan.

The paper's §A.2.8 training-loss text describes a 64-bin 2–22 Å distogram, which matches
MENTOS's GT grid and made it look as though `contact_bin: 19` transferred directly. It
does not. In the released code, the 64-bin 2–22 Å grid is the **input conditioning**
distogram (`compute_distogram_conditioning(..., min_dist=2.0, max_dist=22.0,
num_bins=64)`), a different object from the **output** head. The output head follows
Algorithm 12:

```python
def get_mid_points() -> torch.Tensor:
    """128 distance bin midpoints (2p-52 Angstrom range)."""
    boundaries = torch.linspace(2, 52.0, 127)
    lower = torch.tensor([1.0]); upper = torch.tensor([52.0 + 5.0])
    exp_boundaries = torch.cat((lower, boundaries, upper))
    return (exp_boundaries[:-1] + exp_boundaries[1:]) / 2
```

That yields **128 bins**, midpoints **1.5 … 54.5 Å**, inner spacing **0.3968 Å**.

MENTOS's GT contact threshold is bin 19 of a 64-bin 2–22 Å grid, i.e. an upper edge of
`2.0 + 19*0.3125 = 7.9375 Å`. Mapping that onto ESMFold2's grid:

| | value |
|---|---|
| threshold | 7.9375 Å |
| bins strictly below it | **16** (indices 0..15) |
| last included midpoint | 7.7540 Å |
| first excluded midpoint | 8.1508 Å |

So contact probability is the softmax mass over **bins 0..15**. Reusing
`contact_cutoff_bin: 19` would instead score at roughly 8.9 Å and quietly inflate P@K
relative to every other model in the matrix.

**Design requirement:** the ESMFold2 preset must carry a **distance threshold in
Ångström** and derive the bin index from the model's own bin edges, rather than hardcoding
an integer. ESMFold2 is the second distinct binning scheme in the matrix; a third would
otherwise repeat this bug. The existing integer `contact_cutoff_bin` stays as-is for the
models whose grids already match.

Representative atom is **Cβ, with CA for glycine** (`prepare_input.py`,
`rep_idx = ad.get("CB", ad.get("CA", fallback_idx))`), which matches the MENTOS GT
convention — no adjustment needed there.

## 5. Input construction and chain boundaries

Multi-chain is native — one `ProteinInput` per chain, no linker:

```python
StructurePredictionInput(sequences=[
    ProteinInput(id="A", sequence=seq_a, msa=None),
    ProteinInput(id="B", sequence=seq_b, msa=None),
])
```

`ESMFold2InputBuilder.prepare_input(...) -> (features, chain_infos)`. Recover the
inter-chain block from **`features["asym_id"]`** (per-token chain identity) rather than
assuming `[:LA, LA:]` ordering; cross-check against `len(chain_infos[i].tokens)` and fail
if they disagree. For pure-protein dimers tokens are 1:1 with residues, so the resulting
map is directly comparable with the other models' `(L, L)` output — but that 1:1 property
is an assumption worth asserting, not relying on.

## 6. Runner shape

Model the runner on the cookbook's `fold_and_get_distogram`, which is the reference for
reading the distogram without paying for structure generation:

```python
output = model(**inputs, num_diffusion_samples=1, num_sampling_steps=1,
               num_loops=N, calculate_confidence=False, seed=seed)
logits = output["distogram_logits"]      # (B, L, L, 128)
```

- `num_sampling_steps=1`, `calculate_confidence=False` keep the call on the
  contact-map dependency subgraph, consistent with the FLOPs scope in
  `FLOPS_BENCHMARK_PLAN.md`.
- **`num_loops` is the recycle-ladder knob** — the ESMFold2 analogue of
  `num_recycles` / `recycling_steps`, so the r0/r1/r3/r5 ladder carries over.

Loading, with the documented kernel fallback:

```python
model = ESMFold2ExperimentalModel.from_pretrained("biohub/ESMFold2", ...)
kernel_backend = None
if TRITON_KERNELS_AVAILABLE:  kernel_backend = BACKEND_FUSED
elif CUE_AVAILABLE:           kernel_backend = BACKEND_CUEQ
model.set_kernel_backend(kernel_backend)
```

Triton / cuequivariance kernels are **optional accelerations with an explicit `None`
fallback**, so an A100 without them is supported. The released code runs **bfloat16**
throughout; there is no fp8 anywhere in it, so the paper's float8 production setup is not
a requirement for us.

Runner I/O contract stays identical to the others: JSON bundle on stdin, writes
`<out_dir>/contact.npz` with `probs` (L, L) float16 + `length`, plus the `_flops.py`
sidecar when `--profile` is set.

## 7. Operational notes

- **Memory.** ESMC-6B is far larger than ESMFold's 3B. Budget CPU RAM for it explicitly in
  the sbatch; the 8G default already livelocked ESMFold's 3B load (see the comment block
  in `scripts/run_experiment.sbatch`). Expect this to need more than the 64G set there,
  and to need checking against A100-40GB GPU capacity at L≈1000, the `recent_pp` max.
- **Sequence lengths.** `recent_pp` is 151 two-chain complexes, median L=618, max L=1006.

## 8. Validation gates before trusting any number

1. Bin arithmetic reproduced from the loaded model's own edges, asserting 16 bins below
   7.9375 Å — not copied from this document.
2. Loaded checkpoint's training cutoff asserted to be pre-2025.
3. `contact.npz` shape equals `len(chain_A) + len(chain_B)`.
4. Chain split from `asym_id` agrees with `chain_infos` token counts.
5. Non-degenerate P@K on the smoke entry (a uniform or saturated map indicates the
   softmax or the bin slice is wrong).
