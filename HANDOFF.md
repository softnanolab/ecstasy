# HANDOFF: MiniFold as a multimer baseline in ecstasy

Written for a Claude agent picking this up in **softnanolab/ecstasy**. Read every section before running anything.

The scientific design was settled with the user through a full `/softnano:grill-me` pass; §2 records each decision and the evidence behind it. **Do not re-litigate those.** What is *not* settled is how the task maps onto ecstasy's architecture — that is §3, and it is the one thing you must resolve with the user before writing code.

## 0. At a glance

| Field | Value |
|---|---|
| Task | Benchmark pretrained MiniFold 48L on 151 PDB val dimers as an independent baseline for MENTOS |
| Target repo | `softnanolab/ecstasy` @ branch `handoff/minifold-multimer-dockq` (base `main` @ `0ed56a7`) |
| Prior work | Built and staged on Imperial CX3 against the `mentos` repo; **nothing has been run** |
| Written (UTC) | 2026-08-25 13:02 |
| Compute | Imperial CX3, PBS Pro. Two jobs submitted then **held** — see §7 |
| Blocked on user? | **Yes** — one architectural question in §3 |

## 1. Objective

MENTOS folds monomers well and cannot dock. At step 46000 it scores TM median 0.701 with 76.5% of chains above TM 0.5, while across all 23 evaluated checkpoints its DockQ never exceeds mean 0.162 and its median has sat under 0.08 since step 14000. The open question is whether that is *bad* or merely *unremarkable* — there is no third-party reference on this exact 151-dimer split.

MiniFold (Wohlwend et al., TMLR 2025) is a strong single-chain folder that has never seen a complex. Running it on the same dimers via a chain-break hack, scored identically, supplies that reference. A random-placement null supplies the floor.

**Done when:** MiniFold appears as a first-class ecstasy model with results on the 151-dimer split, alongside the random-placement null and the MENTOS reference points, and a standalone comparison artifact is published.

## 2. Settled design — reproduce, do not revisit

The user chose each of these explicitly:

1. **External pretrained MiniFold 48L.** Not MENTOS's own vendored structure module — that *is* the run being benchmarked. The `mentos` repo vendors only minifold's geometry utils (`src/mentos/structure/`) plus a port of its structure module; the model itself is third-party.
2. **Chain break = 25×G linker *plus* a `residx` +512 jump** (ESMFold's approach). Not linker-alone, not jump-alone.
3. **Report per-chain monomer metrics and a random-placement null alongside DockQ.** A clash audit was offered and explicitly declined.
4. **Deliverable is a new standalone comparison artifact**, not extra lines on the existing MENTOS DockQ curves page.

**Ecstasy already agrees with decision 2.** The `esmfold` row in `src/ecstasy/registry/models.yaml` carries `chain_linker_length: 32, residue_index_offset: 512`, with the comment "32-residue poly-G linker + 512 positional-index skip between chains (multimer hack)". So MiniFold's presets should follow that shape. Note the user picked linker **25** (MiniFold's own README/ESMFold default) where ecstasy's `glinker32` preset uses 32; `esmfold`'s `full` preset already uses 25. Mirror the preset naming rather than inventing new keys.

### Verified facts (each checked with a command on CX3)

- **MiniFold is single-chain.** Its README says so; `mentos/src/mentos/model/structure.py` calls its port "single-chain minifold".
- **The chain-break mechanism is sound.** `FoldingTrunk.forward` (upstream `minifold/model/model.py:147`) hardcodes `residx = torch.arange(L)`, and `RelativePosition` clamps the pairwise difference at `bins=32`. A +512 jump saturates that clamp, so the trunk reads two unrelated chains. Verified numerically on `9zdi`: the difference across the break is **513**.
- **The 151 natives already exist** at `$LOGS_DIR/dockq_stage3_cap50_full/*_native.pdb` on CX3 — chains A and B, renumbered `1..L`. Reading sequences from their CA records means MiniFold predicts exactly the residues MENTOS was graded on.
- **Composition:** 40 true homodimers (chain A sequence == chain B sequence), 111 heterodimers.
- **Sizes:** `La+25+Lb` ranges 129–1018, mean 613. **No target exceeds 1024 tokens.**
- **Two-chain PDB writing round-trips.** `minifold.utils.protein.Protein` accepts `chain_index`; verified on `9zdi` → chains `['A','B']`, 2 `TER`, CA counts `{A: 52, B: 52}` matching the native.
- **The null control runs.** Smoke-tested on two MENTOS predictions: `21ie` mean 0.106 / max 0.233, `9zdi` mean 0.062 / max 0.124 — against MENTOS's actual mean of 0.150. `n=2` proves nothing, but it is why the null was extended to the full MENTOS set.
- **All four reference scripts compile** (`python -m py_compile`).
- **The residx patch is 3 hunks** — `scripts/minifold-baseline/reference/minifold_residx.patch`.

## 3. THE OPEN QUESTION — resolve before writing code

**Ecstasy benchmarks contact maps. This task asks for DockQ, a structure metric.**

Evidence:
- Every runner writes `<out_dir>/contact.npz`. `src/ecstasy/models/adapter.py` hard-fails if it is absent.
- `src/ecstasy/metrics/` contains **only** `contact.py`. There is no DockQ metric anywhere in `src/`.
- ESMFold — a structure predictor — is already reduced to contacts via `contact_cutoff_bin: 19`.
- But `DockQ` **is** a declared pip dependency in `scripts/install/ecstasy.yaml`, and `src/ecstasy/structure/` exists with Pinder-style chain and interface utilities. So the plumbing is half there and someone intended this.

Three ways forward. **Ask the user which.**

| Option | What it means | Cost |
|---|---|---|
| **A. Contact-map row only** | Add `minifold` to `models.yaml` + `_runners/minifold_runner.py` emitting `contact.npz`. Perfectly idiomatic; gives P@K comparable to every other model. **Does not answer the DockQ question.** | Small |
| **B. Extend ecstasy with a structure/DockQ path** | Runners additionally emit a PDB; add `metrics/structure.py` with DockQ + monomer metrics. Answers the question and pays off for ESMFold, Boltz2 and MENTOS too. | Medium — touches the adapter contract |
| **C. Both** | Ship A now for the contact axis, then B. | Largest |

My recommendation is **B**, because the user's question is specifically about docking and a contact map cannot answer it — but this is a repo-architecture decision and it is theirs, not yours. Do not quietly do A and report it as done.

## 4. Integration map

Whatever option is chosen, MiniFold becomes a normal ecstasy model:

| Piece | Path | Model after |
|---|---|---|
| Install script | `scripts/install/minifold.sh` | `scripts/install/esm2.sh` — a small dedicated venv |
| Runner | `src/ecstasy/models/_runners/minifold_runner.py` | `esmfold_runner.py` — closest analogue (folder + linker hack) |
| Registry row | `src/ecstasy/registry/models.yaml` | the `esmfold` row, presets and all |

Runner contract (`src/ecstasy/models/adapter.py`): reads a JSON bundle on **stdin** with `entry_id`, `sequences`, `chain_ids`, `out_dir`, `params`, `infra`, `profile`; writes `out_dir/contact.npz`. **Runners import no ecstasy code** — they run inside their own venv. The reference scripts in `scripts/minifold-baseline/reference/` are standalone CX3 scripts, **not** runners; port the logic, do not copy the files.

Suggested presets, mirroring `esmfold`:

```yaml
minifold:
  runner: minifold_runner.py
  env: ${ENVS_ROOT}/.venv-minifold
  msa: none
  default_preset: full
  presets:
    full:      {num_recycles: 3, chain_linker_length: 25, residue_index_offset: 512, contact_cutoff_bin: 19, model_size: "48L"}
    glinker32: {num_recycles: 3, chain_linker_length: 32, residue_index_offset: 512, contact_cutoff_bin: 19, model_size: "48L"}
    small:     {num_recycles: 3, chain_linker_length: 25, residue_index_offset: 512, contact_cutoff_bin: 19, model_size: "12L"}
```

`UNVERIFIED`: I did not check what `contact_cutoff_bin: 19` means against MiniFold's 64-bin distogram. MiniFold's trunk emits `no_bins: 64` logits (`hyper_parameters` in the checkpoint) — confirm the bin edges match ESMFold's before reusing 19.

## 5. Environment

MiniFold's venv is independent of everything else in ecstasy.

```bash
uv venv --python 3.12 "$ENVS_ROOT/.venv-minifold"
git clone --depth 1 https://github.com/jwohlwend/minifold.git <src>
cd <src> && VIRTUAL_ENV="$ENVS_ROOT/.venv-minifold" uv pip install .
patch -p1 -d <src> < scripts/minifold-baseline/reference/minifold_residx.patch
```

Verified on CX3: `torch 2.13.0+cu130`; the model is ESM2-3B (`esm2_t36_3B_UR50D`) + a 48-block miniformer.

Weights (both already cached on CX3 at `$LOGS_DIR/minifold/cache/`, but **re-download rather than transfer**):

| Weight | Size | Source |
|---|---|---|
| MiniFold 48L | 2.6 GB | `https://huggingface.co/jwohlwend/minifold/resolve/main/minifold_48L_final.ckpt` |
| ESM2-3B | 5.3 GB | `torch.hub` on first model load — set `torch.hub.set_dir(cache)` |

CX3 compute nodes may lack outbound network; both were pre-fetched on the login node. Do the same.

## 6. Data

| Artifact | Location on CX3 | Size | Needed |
|---|---|---|---|
| 151 native dimer PDBs | `$LOGS_DIR/dockq_stage3_cap50_full/*_native.pdb` | ~55 MB | **required** |
| MENTOS step-50000 predictions | same dir, `*_pred.pdb` | ~55 MB | for the MENTOS null reference |
| MENTOS step-14000 predictions | `$LOGS_DIR/dockq_stage3_cap50_step14000/` | 111 MB | optional — best interface geometry |

There is no rebuild path short of re-running the MENTOS eval, so these must be copied or read in place. Same machine, so reading in place is fine.

Whether this split is already a registered ecstasy dataset is `UNVERIFIED` — check `src/ecstasy/registry/datasets.yaml` and the Notion **Datasets** table before adding a duplicate. Per the README, benchmark scripts take dataset **names**, never paths.

## 7. Live jobs on CX3 — DO NOT DUPLICATE

| Job ID | Script | State | Action |
|---|---|---|---|
| `3908962.pbs-7` | `$JOBS_DIR/mentos/minifold_predict.pbs` | **Held** (`qhold`), never started | Leave held. Ask the user before `qdel`. `qrls` resumes it if ecstasy stalls. |
| `3908963.pbs-7` | `$JOBS_DIR/mentos/minifold_score.pbs` | **Held**, also `depend=afterok:3908962` | Same. Releasing it alone does nothing. |

Neither ever entered R state, so **no partial output exists**; `$LOGS_DIR/minifold_eval/` does not exist. Nothing is consuming allocation.

No W&B run is involved — this is an offline evaluation throughout.

Separately: a 10-minute DockQ watcher (`$JOBS_DIR/mentos/dockq_watch.sh`) is a standing task returning NEEDS_AUTH for Isambard. Unrelated. Leave it alone.

## 8. Verification checklist

1. `git log --oneline -1` → the commit carrying this file
2. `python -c "import sys; sys.path.insert(0,'<src>'); import minifold; print(minifold.__file__)"` → a path under `<src>`, **not** site-packages (§9)
3. `grep -n "residx=None" <src>/minifold/model/model.py` → one hit, on the `FoldingTrunk.forward` signature
4. `ls $LOGS_DIR/dockq_stage3_cap50_full/*_native.pdb | wc -l` → `151`
5. Both weight files present
6. Smoke: predict 2 targets. Expect chains A and B with CA counts matching the natives.

## 9. Traps

- **MiniFold's packaging is broken.** `pyproject.toml` declares `packages = ["minifold"]`, so `pip install .` installs *only* the top-level package — `minifold.utils`, `minifold.model`, `minifold.data` are all absent. `import minifold` succeeds while `import minifold.utils.esm` raises `ModuleNotFoundError`. This cost real time. The fix is `sys.path.insert(0, <src>)`, which is also what makes the patched fork win over the installed copy. The reference driver asserts on this — **keep the assert when you port it.** Without it a stale installed copy silently gives unpatched behaviour and a quietly wrong baseline.
- **`residx` must actually reach the trunk.** If `batch["residx"]` is missing, `FoldingTrunk.forward` falls back to `arange` and you have silently measured the linker-only variant the user rejected. Checklist step 3 guards this.
- **DockQ collapses toward `fnat/3`.** When the backbone has not formed, both RMSD terms give near-zero credit and fnat carries the score. MENTOS step 2000 posts its *highest* median DockQ (0.123) on its *worst* iRMSD (21.2 Å) for exactly this reason. Always report iRMSD and LRMSD beside DockQ; never read a DockQ rise as improvement without them.
- **Do not change the DockQ invocation.** Comparability to the 23-checkpoint MENTOS series depends on byte-identical scoring: `DockQ <pred> <native>`, no flags. The regex `DockQ[:\s]+([0-9.]+)` with `.search()` skips the banner and legend and lands on the real value — it works, but by accident of ordering. Do not "tidy" it.
- **40 of 151 are homodimers**, so the input is one sequence duplicated around a glycine linker — something ESM2 has never seen. Report homo/hetero split separately; a collapsed or domain-swapped prediction there is a property of the hack, not of MiniFold's docking ability.
- **Never run compute on a login node** (standing rule in both repos' CLAUDE.md). That includes the ~8 GB of model loading.
- The reference null control seeds from `hash(pid)`, which is **not stable across processes** unless `PYTHONHASHSEED` is set. Replace with a stable hash when porting. Flagged, not fixed.

## 10. Open questions for the user

1. **§3 — contact-map row, structure/DockQ path, or both?** Blocking.
2. Is the 151-dimer PDB val split already a registered ecstasy dataset, or does it need adding to `datasets.yaml` and the Notion Datasets table?
3. Should the two held CX3 jobs be deleted once ecstasy reproduces the run, or kept as a fallback?

## 11. Provenance

Generated by `/softnano:handoff` on Imperial CX3 at 2026-08-25 13:02 UTC.

Facts gathered from: direct reads of upstream `minifold/model/model.py`, `predict.py`, `pyproject.toml` and README; reads of ecstasy's `models/registry.py`, `models/adapter.py`, `registry/models.yaml`, `metrics/`, `README.md`; a CA-record parse of all 151 natives (lengths, homodimer count); an executed two-chain PDB round-trip on `9zdi`; an executed null-control smoke test on two MENTOS predictions; `python -m py_compile` on all four reference scripts; `qstat -u $USER`; `du -sh` on every artifact path.

Marked `UNVERIFIED`: the meaning of `contact_cutoff_bin: 19` against MiniFold's 64-bin distogram (§4), and whether the 151-dimer split is already a registered ecstasy dataset (§6). Nothing else above is unchecked.

---

## 12. Resume-side verification (CX3, 2026-08-25, receiving agent)

Checklist from §8, plus both `UNVERIFIED` items from §4 and §6 — now resolved.

### §8 checklist

| # | Check | Result |
|---|---|---|
| 1 | `git log --oneline -1` | PASS — `6474f5a` |
| 2 | `import minifold` resolves to `<src>` | N/A — ecstasy's `.venv-minifold` does not exist yet (§5 is still to run) |
| 3 | `grep -n "residx=None" <src>/minifold/model/model.py` | PASS — one hit, `model.py:133`, `FoldingTrunk.forward` |
| 4 | `ls $LOGS_DIR/dockq_stage3_cap50_full/*_native.pdb \| wc -l` | PASS — `151` |
| 5 | Both weight files present | PASS — `minifold_48L.ckpt` 2.6 GB + `checkpoints/` (7.9 GB total) under `$LOGS_DIR/minifold/cache/` |
| 6 | Smoke: predict 2 targets | NOT RUN — blocked on §3 |
| — | `qstat -u $USER` | PASS — `3908962.pbs-7` (`mf_predict`) and `3908963.pbs-7` (`mf_score`) both `H`, `--` elapsed. Untouched. |

### §4 UNVERIFIED — `contact_cutoff_bin: 19` does NOT transfer to MiniFold

**Use `contact_cutoff_bin: 17`, not 19.** MiniFold's distogram is 64 bins over
`torch.linspace(2, 25, 63)`, not ESMFold's `linspace(2.3125, 21.6875, 63)`.
`probs[..., :k].sum(-1)` is `P(d <= boundaries[k-1])`, so:

| Model | Boundaries | `cutoff_bin` | Implied cutoff |
|---|---|---|---|
| ESMFold | `linspace(2.3125, 21.6875, 63)` | 19 | 7.9375 Å |
| MiniFold | `linspace(2, 25, 63)` | 19 | **8.6774 Å** — wrong, 0.74 Å too loose |
| MiniFold | `linspace(2, 25, 63)` | **17** | **7.9355 Å** — matches ESMFold to 0.002 Å |

Evidence (commands, all on CX3):
- `unzip -p $LOGS_DIR/minifold/cache/minifold_48L.ckpt archive/data.pkl` → `pickletools.dis`
  → `hyper_parameters` carries `no_bins=64`, `max_dist=25`, `num_blocks=48`,
  `esm_model_name='esm2_t36_3B_UR50D'`, `num_recycling=3`.
- `<src>/minifold/train/model.py:64` — `boundaries = torch.linspace(2, max_dist, self.no_bins - 1)`,
  registered as a buffer and used directly for the distogram label at `model.py:111`
  (`(dists.unsqueeze(-1) > self.boundaries).sum(-1)`). The OpenFold-default
  `2.3125/21.6875` in `train/loss.py:498` belongs to the vendored `distogram_loss`,
  which this LightningModule does not call — do not read it as MiniFold's convention.
- `src/ecstasy/models/_runners/esmfold_runner.py:13-14` states ESMFold's boundaries
  and that bin 19 ≈ 8 Å "matches MENTOS's threshold".

**Second, unrelated mismatch — reference atom.** MiniFold's distogram is trained on
**CA–CA** distances: `train/data.py:46` builds `coords` as `all_atom_positions[:, 0:3]`
(N, CA, C) and `train/model.py:107` selects index `1` = CA. Ecstasy's ground truth is
**Cβ–Cβ** (`registry/datasets.yaml` `contact_bin`; `metrics/contact.py` `valid` doc).
ESMFold/Boltz-2 both predict Cβ. This cannot be fixed by a bin index — it is a property
of the pretrained head. It will modestly depress MiniFold's P@K relative to the Cβ
models and must be stated wherever the comparison is published.

### §6 UNVERIFIED — the 151-dimer split is NOT registered, but needs no ID list

The 151 natives are **exactly** the `val` split index, verified by set equality:

```
$LOGS_DIR/dockq_stage3_cap50_full/*_native.pdb   -> 151 ids
$MENTOS_ROOT/pdb/processed/splits/val/index.parquet (split == "val") -> 151 ids
overlap = 151 / 151
```

Overlap with every registered ecstasy dataset is partial, so none of them is a
substitute: `val_seq_pair` 76/151, `val_seq_chain` 75/151, `val_pinder_pair` 37/151,
`val_pinder_chain` 17/151. So this is a **new `datasets.yaml` row**, not a duplicate —
and a 3-line one, since the parquet already exists:

```yaml
# MENTOS PDB val dimers — the 151-entry DockQ evaluation set
mentos_val151:
  <<: *mentos
  index: ${MENTOS_ROOT}/pdb/processed/splits/val/index.parquet
```

Still to confirm before publishing: whether this split is in the Notion **Datasets**
table under another name.

### New finding — ecstasy is not yet runnable on CX3

Independent of §3, this is prerequisite work:

- **No `.env` at the ecstasy repo root.** `.env.example` requires `DATA_ROOT`,
  `MENTOS_ROOT`, `ECSTASY_ROOT`, `ENVS_ROOT`, `TOOLS_ROOT`; `ecstasy.config` errors on
  empty. The only `.env` on this machine belongs to the `mentos` repo and uses the
  unrelated `DATA_DIR`/`LOGS_DIR`/`JOBS_DIR` names.
- `MENTOS_ROOT` resolves to `/rds/general/user/ha1822/ephemeral/MENTOS/DATA` —
  confirmed to hold `pdb/processed/splits/{val,val_master,val_union,val_seq_*,val_pinder_*}`.
  **It is on `ephemeral`**, which is purge-eligible.
- `${MENTOS_ROOT}/pdb/processed/splits/seq_id_30` does **not** exist, so the
  `mentos_seqid30` row in `datasets.yaml` is currently unresolvable here. Unrelated to
  this task; noted so it is not mistaken for damage done by it.
- No per-model venvs exist under any plausible `ENVS_ROOT` (`find -name '.venv-*'` → none).

### Correction to §3's cost estimate for option B

§3 rates B as "touches the adapter contract". It need not. `predict_one`
(`models/adapter.py:47`) hard-fails only on a **missing** `contact.npz`; it ignores
anything else in `out_dir`. Two runners already write coordinates as a side effect —
`boltz2_runner.py:102` (`--output_format mmcif` into `out_dir/raw/`) and
`colabfold_runner.py:102` (`*_rank_001_*.pdb`). So a structure path can be added as an
**optional** second artifact under an agreed name, with `metrics/structure.py` skipping
models that do not emit it. No existing runner or contract changes.
