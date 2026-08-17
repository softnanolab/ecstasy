# P@K vs. inference FLOPs — benchmarking plan

Replace the **params** x-axis with **empirically-measured neural inference FLOPs**, so the
compute cost of recycles and diffusion is visible. Each decision below was resolved in a
design review; the rationale is kept so the plan is self-justifying.

---

## 0. The question this plot answers

> *"For each available tool, how much arithmetic does it take to produce **this contact map**,
> and what P@K does that buy?"*

Not "which architecture is most FLOP-efficient at contact prediction in principle" — the
structure predictors are charged only for the compute their **contact output** depends on
(see §3). x = FLOPs (log), y = P@K, on the **same proteins** for both axes.

---

## 1. Scope of the FLOP count — decisions

| # | Decision | Choice | Why |
|---|----------|--------|-----|
| 1 | What's in the box | **Neural forward FLOPs only.** MSA *search* (mmseqs/colabfold, CPU DB op) excluded. | FLOPs measures arithmetic in the differentiable model; MSA search is I/O-bound and ill-defined as FLOPs. MSA *dependency* is shown as a marker style (§4), not a number. |
| 2 | How measured | **Empirical, `torch.utils.flop_counter.FlopCounterMode`.** Not analytic `6ND`, not fvcore (trace-based, misses recycle loops), not DeepSpeed. | Dispatch-based → counts real aten ops *as executed*, so recycles, diffusion steps, and per-protein MSA depth are captured natively. Has an SDPA formula. |
| 2b| Fused-kernel blind spot | Profile the **exact `no_kernels=true` eager path** the P@K eval already runs. | Op-counters see opaque fused kernels as 0 FLOPs. The eval path is already unfused/pure-aten → fully visible **and** faithful to the P@K run. |
| 2c| Unit convention | **True FLOPs = 2 × MACs — and `FlopCounterMode.get_total_flops()` ALREADY returns this** (empirically verified on torch 2.9.0: a pure `Linear` gives `total = 2·m·n·k`). **Report `get_total_flops()` directly; do NOT double again** (would be a 2× overcount). Store `macs = flops/2`. | Matches AF2/ESM convention; torch's counter uses the `2·MACs` formula natively. SDPA attention is counted (decomposes to `bmm`, or via the registered flash-attention formula). |
| 3 | Per-model = a distribution, not a scalar | **Profile per-protein on the real eval set; plot the dataset-mean FLOPs** (paired with mean-P@K). Keep per-protein spread for whiskers. | FLOPs ∝ L² (and MSA depth N) → 474 different values on `val_pinder_pair`. x and y must describe the **same population**. |
| 7 | Structure predictors: what subgraph | **(ii) Contact-map dependency subgraph only** — count exactly the ops the contact map depends on. The boundary is **per-model**, verified in source (§3.5) — it is **not** "trunk only" for everyone. | Boltz/ESMFold read contacts from a **distogram head**, but whether the 3D-structure module is on the dependency path differs by architecture (Boltz: post-hoc decoder → off-path; ESMFold: recurrent in the recycle loop → on-path). Scope (1) literally says "work to turn sequence→contact map." |

**Headline x for each model = mean over `val_pinder_pair` proteins of the contact-dependency-subgraph
FLOPs (2×MACs), `no_kernels` eager path.** The subgraph boundary per model is §3.5.

### 3.5 Per-model dependency-subgraph boundaries (VERIFIED IN SOURCE)

The naïve "trunk + distogram head, exclude the structure module" is **only correct for Boltz.**
Verified against the softnano forks:

| Model | Contact source | Structure/diffusion on contact path? | Count | Exclude |
|-------|----------------|--------------------------------------|-------|---------|
| **Boltz-2** | `distogram_module(z)` — `boltz2.py:508`, computed from trunk `z` **before** diffusion | **No.** `structure_module.sample()` (AtomDiffusion, `boltz2.py:515-562`) runs *after* the distogram, conditions on `s,z`, never feeds back into the recycle loop (`455-506`). | input_embedder + msa_module + pairformer×recycles + distogram_module | diffusion (AtomDiffusion sampling), diffusion_conditioning, confidence, bfactor |
| **ESMFold** | `distogram_head(structure["s_z"])` — `esmfold.py:259`, from trunk pair rep | **Yes (for num_recycles ≥ 1).** The structure module runs *inside* the recycle loop (`trunk.py:203`); its predicted positions compute `recycle_bins` (`212`) that feed the **next** iteration's pair rep (`198`) → final `s_z` → distogram. | ESM2 LM + FoldingTrunk blocks + **structure_module (×recycles)** + distogram_head | only `lddt_head`, `ptm_head` (terminal, tiny linears off-path) |
| **MENTOS** | distogram head (single forward) | n/a (no structure module) | whole forward | — (mask_inputs=False) |
| **MSA-Pairformer** | `predict_cb_contacts` (Cβ head, layer 15) | n/a (no structure module) | trunk through layer-15 Cβ head | the `predict_confind_contacts` layer-18 secondary head (separate call in the runner) |

**Consequence:** Boltz and ESMFold are deliberately **asymmetric** — diffusion excluded for Boltz,
structure module included for ESMFold — because that asymmetry is the *truth* of decision (ii):
count what the contact map depends on. Boltz's decoder is post-hoc; ESMFold's is recurrent.

**Boltz simplification (verified):** because the distogram is computed *before* diffusion
(`boltz2.py:508`), running Boltz with the existing `skip_run_structure=True` knob produces an
**identical distogram → identical P@K** with no diffusion. So the (ii) count for Boltz needs **no
module attribution** — just profile the skip-structure forward.

**No exclusion machinery needed anywhere (key robustness win).** Attribution-by-module turned out
to be *fragile*: `FlopCounterMode` only builds attribute-qualified paths (`Boltz2.structure_module`)
when a module is entered via `__call__`, but Boltz's diffusion runs via `structure_module.sample(...)`
(a method) — so its FLOPs would *not* sit under the `structure_module` subtree and a name-based
subtraction would silently miss them. We sidestep this entirely:

| Model | How (ii) is achieved | Exclusion needed? |
|-------|----------------------|-------------------|
| Boltz | `skip_run_structure=True` → diffusion never executes; counted total = trunk-only | none |
| ESMFold | count whole `model.infer` — SM is on-path; `lddt_head`+`ptm_head` are ~0.004 TFLOP (<0.01% of the trunk) | none (document the negligible inclusion) |
| MENTOS | whole forward `model(batch, mask_inputs=False)` | none |
| MSA-Pairformer | whole `predict_cb_contacts(...)` call (NOT the confind head) | none |

A top-level per-module breakdown is still recorded in each sidecar for the affine-in-recycles
sanity check (§8c) and to verify (e.g.) Boltz's `structure_module` subtree FLOPs == 0 in profile
mode (proof that diffusion was skipped).

---

## 2. How the count is obtained — decisions

| # | Decision | Choice |
|---|----------|--------|
| 4a | Instrumentation point | **In-runner `profile: true` flag.** When set, the runner wraps its real forward in `FlopCounterMode` and writes a per-prediction sidecar. No standalone profiling script (would drift from the eval path). |
| 4b | Which proteins | **Full split** (`val_pinder_pair`, 474) for the headline; fixed **seeded 40-protein** subsample only if an MSA-model's walltime forces it (recorded in the experiment YAML, never a runtime random draw). |
| 4c | Sidecar | Per-protein `flops.json` next to `contact.npz`: `{flops, flops_macs, L, msa_depth, recycles, module_breakdown}`. An aggregator folds the mean into `comparison.csv`. |
| 8 | Extract the (ii) count | **Per-model, per §3.5.** Boltz: profile the **skip-structure** forward (diffusion simply does not run → no attribution needed). ESMFold/MENTOS/MSA-Pairformer: wrap the contact-producing call in `FlopCounterMode`; for ESMFold use the per-module breakdown only to subtract the two terminal heads (`lddt_head`,`ptm_head`). |
| 8b | In-process requirement | **We may edit the softnano forks** (user-confirmed). So Boltz gets a `--profile_flops` flag added to **its own predict CLI** (sets `skip_run_structure=True`, wraps the forward in `FlopCounterMode`, dumps `flops.json` into the boltz out dir) — reusing boltz's real featurization instead of reimplementing it. The ecstasy boltz_runner keeps shelling out, now passing `--profile_flops`. MENTOS/ESMFold/MSA-Pairformer already run in-process in their runners → wrap inline. |
| 8c | Free correctness check | Trunk FLOPs must be **affine in recycles**: `base + k·per_recycle`. Fit {0,1,3,5}; non-clean fit ⇒ wrong module attribution. Intercept (k=0 single pass) must match a hand `~2·params·L` order bound. Record the contact-dependency module-name set as an **asserted list** so a future refactor that renames modules fails loudly. |
| 4d | Counter backend | **Two, because the model venvs do not share a torch version.** `FlopCounterMode` (torch ≥ 2.1) for `.venv-boltz` / `.venv-mentos`; `torch.profiler(with_flops=True)` (torch ≥ 1.8) for `.venv-esmfold`, pinned to py3.7 / torch 1.12 because openfold's `structure_module` imports a cp37 CUDA extension **at module import time** — moving that env to py3.12 would mean rebuilding CUDA 11.3-era kernels against 12.4. The two are comparable: both charge 2·MACs over the same matmul family (mm/addmm/bmm/baddbmm/conv) and ignore elementwise work. Verified on torch 1.12.1 that the profiler returns exactly `2·m·k·n` for a Linear (two shapes) and that `key_averages()` **sums** over repeated calls rather than averaging — the latter is what makes the count scale with recycles. Each sidecar records which backend produced it. |

### 2.1 Status of check 8c (2026-08-17)

Check 8c **fired on its first real use, and it was right.** Boltz-2 reported an identical
`2.929e9` for every rung of the r0/r1/r3/r5 ladder — slope zero, the most degenerate
possible violation of "affine in recycles" — while P@K moved 0.053 → 0.348 → 0.462 →
0.492 over the same rungs, so the trunk was demonstrably running and recycling.

The magnitude is wrong too: ~2.9 GFLOP for an L=689 trunk is 3–4 orders of magnitude
below the `~L³·c` order bound the same decision calls for. The counted total is *exactly*
`input_embedder + confidence_module`; every module between them — including plain
`Linear` children such as `s_init`, `z_init_1/2` and `distogram_module`, which must run
because `pdistogram` depends on them — contributes zero. So this is not mis-attribution
(`get_total_flops()` is a global sum, independent of module naming): the dispatch mode
stops observing ops partway through the forward.

Ruled out so far: `torch.compile` (inference unwraps to `._orig_mod`, so the trunk is
eager), custom kernels (`no_kernels=True` ⇒ `use_kernels=False`), `no_grad`
(`input_embedder` is under the same `set_grad_enabled` block and *is* counted), and the
trunk not running at all. `FlopCounterMode` itself is exact on a plain `Linear` in that
same venv. Diagnosis continues under `ECSTASY_FLOPS_DEBUG=1`, which dumps the unfiltered
attribution, the aten ops actually observed, and a `pdistogram` checksum.

**No Boltz-2 FLOPs number should be published until this resolves.** The invalid sidecars
have been deleted rather than left on disk.

---

## 3. The recycle sweep — decision 6

Lines mean "compute→quality at **fixed architecture**" only if the **sole** knob changing is a
compute knob. The current presets do **not** satisfy this (`esmfold:glinker32` vs `full` change
the *linker*, both at `num_recycles=4` — a modeling ablation, not a compute sweep).

**Add dedicated single-knob sweep presets** holding everything else fixed:

- **recycles ∈ {0, 1, 3, 5}**, diffusion/sampling held at **default** (Boltz `sampling_steps=25`,
  `diffusion_samples=1`).
- Boltz: `recycling_steps`. ESMFold: `num_recycles` (fixed linker, e.g. 32). colabfold: `num_recycle`.
- **MENTOS, MSA-Pairformer**: no recycle knob → **single point each** (informative: fixed-x).

Each sweep value = one marker; same-model markers joined by a faint line.

---

## 4. Presentation — decision 5

Single scatter:

- **x = log₁₀(FLOPs)** (models span 2–3 orders of magnitude; linear-x smears everyone but Boltz).
- **y = P@K** (linear), the `val_pinder_pair` `inter_P@K` headline.
- **One marker per (model × preset)**; **same-model presets joined by a faint line** = that
  architecture's own compute→quality curve. *This is the payoff over the params plot — params are
  constant across recycles, FLOPs are not.*
- **Encoding:** color = model family; **marker fill = MSA dependency** (filled = boltz2,
  msa_pairformer; hollow = mentos, esmfold, colabfold-singleseq).
- **Error bars:** vertical = **95% bootstrap CI of P@K** over proteins (per-protein P@K already
  exists); horizontal = faint **per-protein FLOP IQR** whisker (exposes that MSA-models have wide
  compute spread, MENTOS is a near-vertical line).
- **Pareto staircase:** dashed upper-left envelope (max P@K achievable at ≤ given FLOPs) = the
  efficiency frontier, the actual takeaway.
- **Annotations:** recycle/step count next to each point so the line is legible.
- **Optional faint secondary marker** per structure model: the **full as-shipped** FLOPs
  (incl. diffusion/structure module) to make the discarded compute visible without polluting the
  architecture comparison. Caption states the headline excludes it.

**Caption must state:** FLOPs = 2×MACs, `no_kernels` eager path, MSA search excluded, structure
predictors counted on the contact-dependency subgraph only (diffusion/structure-module FLOPs they
*also* spend for coordinates are excluded because the contact map does not depend on them).

---

## 5. Implementation checklist

1. **Profiler util** (`metrics/flops.py`): wrap a callable in `FlopCounterMode`, return
   `{total_macs, total_flops=2×, module_breakdown}`. Helper to sum a given subtree name-set and
   assert it's present.
2. **Per-runner `profile` flag** writing `flops.json` sidecar (`flops`, `flops_macs`, `L`,
   `msa_depth`, `recycles`, `module_breakdown`). Easy for the in-process models (mentos,
   esmfold, msa_pairformer) first.
3. **Boltz `--profile_flops` CLI flag** (decision 8b, edit the fork): in boltz's predict path, when
   set, force `skip_run_structure=True` and wrap the model forward in `FlopCounterMode`, dumping
   `flops.json` next to the distogram. Verify the skip-structure distogram matches the full-pipeline
   `distogram_<id>.npz` bit-for-bit (so the profiled path == the P@K path). ecstasy boltz_runner
   passes `--profile_flops` through when `profile` is set.
4. **Per-model subgraph boundary is §3.5** (verified in source). ESMFold: subtract `lddt_head` +
   `ptm_head` via the per-module breakdown; **structure module stays IN** (recurrent, on-path).
   Record the excluded-module name set as an asserted list so a rename fails loudly.
5. **Sweep presets** in `models.yaml`: `r0/r1/r3/r5` per recycle-capable model, single-knob.
6. **Aggregator**: per-protein sidecars → dataset-mean FLOPs + IQR into `comparison.csv`.
7. **Validation**: affine-in-recycles fit per structure model (decision 8c); FlopCount
   non-trivial vs `6ND` lower bound (catches silent fused-kernel zero-count).
8. **Plot script**: log-x scatter per §4 (lines, fill=MSA, bootstrap CI, FLOP IQR whisker, Pareto
   staircase, optional as-shipped secondary marker).

## 6. Order of work

MENTOS (single forward, trivial) → ESMFold / MSA-Pairformer (in-process already) →
Boltz (needs the in-process path, §5.3, the only real surgery) → aggregator → plot.
Land MENTOS+ESMFold first to validate the sidecar/aggregator/plot end-to-end before the Boltz work.
