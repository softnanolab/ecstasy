# ecstasy design of record

Why this document exists: ecstasy grew as a contact-prediction benchmark and is being
turned into a general, modular benchmarking tool — every dataset first-class, every metric
reusable, every result traceable. This records the decisions and, more importantly, the
**evidence** behind them. Where a decision looks fussy, it is because something concrete
was already broken.

Status: Phase 1 is implemented (commit `07c3c19`). Phases 2-5 are designed, not built.

---

## 1. What was actually wrong

Five defects found by inspection, not speculation. Each one motivates a decision below.

| # | Finding | Evidence |
|---|---|---|
| 1 | **A tolerant metric existed that nothing could call.** Tolerant inter-chain P@K lived inside a plotting script; `ecstasy score` had no way to reach it. | `plot_pak_vs_flops.py::_tol_inter_pak` |
| 2 | **Dataset row comments had already drifted from reality.** | Comments claimed `val_pinder_chain`=98, `val_pinder_pair`=474; the parquets hold **106** and **454** |
| 3 | **Two different experiments serialised identically.** The MiniFold runner takes `minifold_src` as a *path*; whether the `residx` patch is applied inside that tree is the whole difference between the intended chain break and the linker-only variant. `params.json` recorded no commit at all. | `params.json` keys: dataset/model/preset/variant/msa/params/infra |
| 4 | **No registered dataset can be fully scored on CX3.** `gt_root` holds only the 151 `val` entries. | `val_seq_pair` 76/930 present · `val_seq_chain` 75/632 · `val_pinder_pair` 37/454 · `val_pinder_chain` 17/106 |
| 5 | **Submodule SHAs describe nothing.** All eight submodules are uninitialised (`-` prefix), `modules/mentos` is empty, and the MENTOS that actually runs is an editable install from `~/code/nanolab/mentos` @ `2cc5309`. | `git submodule status`; `scripts/install/mentos.sh` says so in its own comment |

A sixth, found while designing: **MSA sets are keyed by sequence hash and a bare `kind`
string** — nothing about search engine, database versions, pairing mode or depth. Changing
how MSAs are generated silently overwrites or reuses the old ones.

---

## 2. Decisions

### D1 — Metrics are registered by name (built)
One registry; a name maps to exactly one implementation; re-registering a name is refused.
The tolerant P@K is hoisted in and **verified to reproduce the original exactly**: 68 real
targets, 612 comparisons across tolerance × divisor, max |diff| `0.000e+00`.

### D2 — Datasets carry identity (built)
`version`, `description`, `expected_entries` required on every row;
`ecstasy datasets --verify` asserts them. Counts are measured, never copied from comments.

### D3 — A result names the code that produced it (built)
`provenance.json` beside and inside `result.json`: ecstasy commit + dirty, source-tree git
state for any path in params, and byte identity of weight files **following symlinks**.

### D4 — Cached predictions are fingerprint-gated
`out_dir` is `<dataset>/<model>/<variant>` and contains nothing about code, while
`run_predict` skips any entry that already has a `contact.npz`. So bumping MENTOS reuses old
predictions under a *new* provenance record — a confidently false claim, which is worse than
none. **Fix:** compute a fingerprint and refuse reuse when it differs, offering `--force`.

### D5 — Two fingerprints, not one
* **prediction fingerprint** — model code, weights, resolved params, MSA recipe id, entry sequences
* **scoring fingerprint** — ground truth, metric implementations, metric set, `contact_bin`

Predictions never see GT (`predict_one` receives only sequences and params), so a GT
regeneration or a metric fix must re-score in minutes, not re-run a GPU sweep.

### D6 — Code identity comes from the venv, not from submodules
Each venv's `dist-info/direct_url.json` records the exact source path a package was installed
from; that path resolves to a git SHA. This correctly reports MENTOS as
`~/code/nanolab/mentos@2cc5309` where submodule state would have reported an empty pin.
Submodule capture is demoted, and the `at_pin: true`-for-uninitialised bug is fixed.

### D7 — Partial ground truth cannot produce a silent headline number
`verify` reports GT coverage per split; `run_score` records `coverage` and refuses a headline
mean below 100% without `--allow_partial`, which stamps the result as partial.

### D8 — ecstasy derives its own ground truth
Built from mmCIF using the existing `src/ecstasy/structure/` primitives, **gated on
reproducing the 151 existing MENTOS `.pt` files exactly** before being trusted for the rest.
Conventions to match are documented and simple: 64 bins, `linspace(2.3125, 21.6875, 63)`,
threshold bin 19 (≤ 7.9375 Å), `-1` undefined.

*Chain selection.* The split index records `id`, `sequences`, `num_chains` and
`bsa_per_side` — but **not chain IDs**; its `relative_path` points straight at the `.pt`, so
today the index depends on the GT file for anything beyond sequences. Derivation therefore
sequence-matches the index's two sequences against the chains of the mmCIF assembly, using
`bsa_per_side` to disambiguate when several pairs match.

*Gate failure is not negotiable.* A 99% match is the dangerous outcome — it looks fine and
quietly shifts a handful of numbers. If 151/151 is not achieved, the derived GT is a
**different** ground truth: ship it as a separately versioned `ecstasy-gt-v1`, label every
result with which GT produced it, and never average across the two.

### D9 — GT is stored as pickle-free per-entry `.npz`
Consequence worth stating plainly: `gt_for` currently unpickles a `mentos.dataclasses.Sample`,
and **that is the only reason a scoring env needs MENTOS at all.** Owning the format removes
MENTOS from scoring entirely.

### D10 — A dataset is one self-contained folder holding everything any model needs
Duplication across splits is explicitly acceptable (the validation sets are small).

```
$DATA_ROOT/
  datasets/<name>/
    dataset.yaml              identity, provenance, source, coverage
    index.parquet
    gt/<xx>/<id>.npz          contacts, valid, atom37, asym_id, residue_index
    natives/<id>.pdb          complex, for DockQ
    assets/<asset>/           native_chains, esmfold_monomers, … each provenanced
    msa/<recipe_id>/          recipe.yaml + per-entry MSAs
  experiments/<dataset>/<model>/<variant>/
    params.json provenance.json result.json predictions/<id>/
```

Models request assets **by name** (`needs_assets: [esmfold_monomers]`), never by hardcoded
path. This fixes `plmgraph_inter` and `deepinteract`, which today read
`${DATA_ROOT}/structures/esmfold` — a loose global directory, produced by a different model,
with no provenance, **absent on this machine**.

### D11 — MSA sets are identified by their generation recipe
`msa/<recipe_id>/` with `recipe.yaml` recording kind, engine, databases + versions, pairing
mode and depth. A changed recipe writes a **new folder beside the old**, never over it, so
Boltz-2's paired per-chain CSVs and MSA-Pairformer's stitched complex a3m coexist for the same
dataset. A model declares only the *kind* it can consume; the recipe resolves at run time
(`--msa_recipe`) and its id is part of the prediction fingerprint — so switching recipe
invalidates cached predictions automatically, with no special-casing.

> Cost accepted deliberately: the global dedup store is dropped, so a chain shared by several
> splits is searched more than once. Mitigation that does not reintroduce a global store:
> `ecstasy msa` checks other dataset folders for an identical sequence hash **and recipe** and
> copies rather than re-searching.

### D12 — MENTOS is pinned for benchmarking, separate from your working tree
`.venv-mentos` installs at a pinned rev declared in `models.yaml`.
`ecstasy deps bump mentos [--rev X]` moves the pin, reinstalls, records the SHA.
`ecstasy deps use-local mentos` installs editable from your checkout when you *want* to
benchmark work in progress — and provenance reports it as local, so the two can never be
confused.

### D13 — Results are published deliberately
`ecstasy publish` appends one JSONL line per scored run, keyed by both fingerprints, carrying
metrics, n, coverage, FLOPs and provenance. It refuses partial or dirty runs unless overridden.
A MENTOS bump therefore writes a **new line**, so `git log` on that file shows a number moving
and why. Per-protein detail stays in `$DATA_ROOT`; the repo keeps summaries, not blobs. Notion
becomes a generated view rather than the source of truth.

### D14 — Weights are first-class folders, addressed by name
`$DATA_ROOT/weights/<model>/<name>/` with a manifest recording source URL, sha256, size and
what produced it — the same shape as a dataset folder, so one mental model covers both.
`--checkpoint <name>` then works for **any** model, not just mentos, and the name plus content
hash lands in provenance. Removes Notion from the resolution path and fixes MiniFold's weights
currently being a bare symlink into a MENTOS log directory.

### D15 — Results are displayed from the committed data
`ecstasy report` renders the JSONL into a leaderboard: markdown in-repo for agents and diffs,
plus a shareable page. `--to notion` pushes the same data. Nothing needs a token or a network
to **read** results — an agent picking up a task must be able to see what has already been
benchmarked from the repo alone. Notion keeps what it is genuinely better at: narrative
campaign entries with figures.

### D16 — The layout change is a clean break
There is no historical corpus to migrate: `$DATA_ROOT` holds 72 MB, all of it one in-flight
run. The old `runs/` tree is left untouched and ignored rather than converted.

### D17 — The first published record is a re-run, not an import
The current MiniFold sweep predates provenance capture and has only a hand-written record. It
will be re-run once fingerprinting lands, so the store has no half-trusted first row and no
special case. The existing numbers are not wasted: **the re-run must reproduce them exactly**,
which makes it a regression test that the refactor did not change the science.

---

## 3. Sequencing

| Phase | Contents | State |
|---|---|---|
| 1 | Metric registry, dataset identity, provenance | **done** — `07c3c19` |
| 2 | Fingerprints D4/D5/D6 + GT coverage D7 | **done** — `d9c57b3` |
| 2b | Structure metrics in the registry (DockQ/RMSD/TM) | **done** — `9e244b3` |
| 3a | GT geometry D8 + pickle-free format D9 + importer | **done** — `7b33f33`, gates 151/151 |
| 3b | mmCIF derivation for entries with no `.pt`; assets; MSA recipes D11 | open |
| 4 | `ecstasy deps` D12, weights folders D14 | open |
| 5 | `ecstasy publish` D13, `ecstasy report` D15, MiniFold re-run D17 | open |

PR #28 is **not** being merged, so the structure metrics were built natively in the
registry shape rather than ported from a pre-registry design. What that leaves outstanding
is tracked as issues, not assumed.

**Known gap, stated plainly:** structure metrics are registered, verified and reachable by
name, but `run_score` does not yet invoke them — a model emitting `structure.npz` is
currently ignored by the scoring pipeline. Likewise the `minifold` model row, its runner,
and the `mentos_val151` dataset row live only on the unmerged branch and are absent here.

---

## 4. How each phase is verified

* **D1** — hoisted metric reproduces the original on real predictions (done: 612 comparisons, max diff 0.0)
* **D4/D5** — bump a dependency, re-run, assert the run refuses to reuse and says what changed
* **D6** — assert the recorded SHA equals `git -C $(direct_url path) rev-parse HEAD` for every model venv
* **D7** — `ecstasy datasets --verify` reproduces the coverage table in §1 finding 4
* **D8** — derived GT must equal all 151 existing `.pt` files exactly; anything less blocks the phase
* **D9** — score a run in a venv with **no mentos installed**; it must succeed
* **D11** — generate two recipes for one dataset; assert both persist and that switching invalidates predictions
* **D13** — publish twice with identical fingerprints (duplicate refused); bump a dep and publish (new line appended)
* **D14** — `--checkpoint <name>` resolves for a non-mentos model, and the recorded sha256 matches the file on disk
* **D17** — the re-run's per-target numbers must equal the current sweep's exactly; any drift is a refactor regression, not a new result
