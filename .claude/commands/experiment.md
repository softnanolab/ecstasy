---
description: Fully vet and materialize an ecstasy benchmarking run before executing it — checks the model, dataset, metric, and prior-run history against the registries before proposing the exact command.
---

# /experiment — grill-me before running a benchmark

You are helping run an **ecstasy** benchmark (dataset × model → metrics). The user will
give you something like:

> "Can you run mentos: `<checkpoint>` on the recent PP val. I want to know its DockQ score."

Do **not** materialize or execute a run from a single message like that. Your job is to
walk through the checklist below, asking the user only what is genuinely unresolved —
never assume a default silently. This command trades a few questions up front for never
producing a mislabeled or duplicate result later.

Ground everything in the actual registries — `ecstasy list`, `ecstasy datasets`,
`ecstasy metrics`, and the `data/runs/` tree — never in memory of a prior session. Run
these read-only commands liberally; they are cheap and this whole command exists because
memory drifts from what is actually registered.

## Step 0 — parse the request

Extract three things from the user's message, best-effort:
- **model** name (and, if pasted, a checkpoint identifier / weights reference)
- **dataset** name or description (e.g. "recent PP val")
- **metric(s)** requested (e.g. "DockQ score")

If any of the three is simply absent from the message (not ambiguous, just not
mentioned), ask for it directly before proceeding — don't guess a dataset or metric the
user never named.

## Step 1 — resolve the model

Run `ecstasy list` to see every registered model, its `msa` kind, and its presets.

- **Model name matches a registered row exactly** → continue to preset/checkpoint
  resolution below.
- **Model name is close but not exact** (typo, alias, casing) → confirm with the user
  which registered model they mean; do not silently pick the closest match.
- **Model name does not match anything registered** → this is a new model. Stop and ask
  the user for what `models.yaml` needs for a new row: the runner script name (under
  `src/ecstasy/models/_runners/`), which venv it runs in (`env:`), its `msa` kind
  (`none` / `per_chain` / `boltz_csv` / `complex` / `complex_api`), and either its
  presets (name → param dict) or, if it is checkpoint-based like mentos, confirm it has
  no committed presets and will resolve checkpoints from `checkpoints.yaml` instead.
  Do not add the row yourself without the user confirming the values — read back what
  you're about to add and get a yes before writing to `models.yaml`.

### Checkpoint-based models (no committed presets, e.g. mentos)

If `ecstasy list` shows the model has no presets (or `presets_for(model)` is empty),
it resolves by checkpoint name against `src/ecstasy/registry/checkpoints.yaml`, not by
preset.

1. Take the checkpoint identifier the user pasted (a name, a path, a run id — whatever
   they gave you) and check whether a matching entry already exists in
   `checkpoints.yaml`.
2. **If it matches an existing entry** → confirm with the user this is the checkpoint
   they mean (name, abs_path, run_id, num_recycles) and proceed.
3. **If it does not match anything in `checkpoints.yaml`** → this is a new checkpoint.
   Ask the user for whatever of the following they haven't already given you:
   - `abs_path` (the weights file — required to be runnable)
   - `run_id` (optional but useful for provenance)
   - `num_recycles` (optional)
   - `model_config_path` (optional)
   - a short name to register it under (suggest `<run_id>_s<step>` per the file's own
     convention if a run_id/step is available, otherwise ask the user to name it)
   Then propose the exact YAML row you're about to append to `checkpoints.yaml` and get
   explicit confirmation before writing it. Never invent a path — if the user doesn't
   know the abs_path, stop and ask; do not guess a plausible-looking location.

### Preset-based models

If the model has committed presets:

- **No preset named in the request, and the model has exactly one preset** → do not
  silently use it. Ask explicitly: *"`<model>` only has one preset, `<preset>` — is
  that the one you mean?"*
- **No preset named, and the model has a `default_preset` among several** → tell the
  user what the default is and what the alternatives are, and ask them to confirm the
  default is what they want rather than silently applying it.
- **A preset is named** → check it against `presets_for(model)`; if it doesn't exist,
  show the valid list and ask again.

## Step 2 — resolve the dataset

Run `ecstasy datasets` (and `ecstasy datasets --verify` if you need to confirm it's
actually built on disk) to see every registered split: name, version, size, tags,
description.

- Match the user's description (e.g. "recent PP val") against the registered names and
  descriptions — several rows may plausibly match a loose phrase like that
  (`recent_pp`, `foldbench_pp`, `foldbench_pp_post2024`). If more than one registered
  dataset is a plausible match, list the candidates with their one-line descriptions and
  ask the user to pick, rather than guessing the most likely one.
- Once resolved to an exact registered name, confirm it is **built on disk** for this
  machine (`ecstasy datasets --verify` reports "not built yet" if not). If it isn't
  built, tell the user and give them the exact `ecstasy import_dataset --dataset <name>`
  command rather than running it yourself unasked.
- If the phrase doesn't match anything registered at all, say so plainly and show the
  full list from `ecstasy datasets` — do not propose building a new dataset without the
  user explicitly asking for that.

## Step 3 — resolve the metric

Run `ecstasy metrics` to see every registered metric, its kind (`contact` vs
`structure`), and whether higher or lower is better.

- Match the user's requested metric (e.g. "DockQ score") to a registered name.
- **Check the metric's kind against what the model can produce.** A `structure` metric
  (DockQ, Fnat, iRMSD, LRMSD, TM_mean, ...) requires a model whose runner emits a
  structure, not just a contact map — check `CLAUDE.md` / the model's row for whether it
  scores structure (e.g. `minifold`, `mentos`, `boltz2` can emit structures; a
  contact-only model cannot produce DockQ at all). If the requested metric cannot be
  computed for this model, say so and ask whether the user wants a different metric or
  a different model.
- If the requested metric isn't registered under any name close to what the user asked
  for, say so and show the full list from `ecstasy metrics` — do not silently substitute
  a different metric.
- Remind the user of `CLAUDE.md`'s DockQ rule if DockQ is requested: **never report
  DockQ without iRMSD and LRMSD alongside it**, and flag if an early/unconverged
  checkpoint is being scored (a converged-model comparison is very different from a
  step-2000 comparison — see `CLAUDE.md` for the documented failure mode).

## Step 4 — check for prior runs (exact and related)

Once dataset and model+preset/checkpoint are both resolved to their exact registered
identifiers, compute the run's expected output directory:
`$DATA_ROOT/runs/<dataset>/<model>/<variant>/` (variant = preset name, or the checkpoint
name for checkpoint-based models, or `<preset>+<sha8>` if overrides were given).

- **Exact match exists** (that directory has a `result.json`) → this is a duplicate.
  Show the user the existing `result.json` summary (mean/median for the requested
  metric, n_evaluated, when it was produced if inferable) and ask whether they want to:
  (a) just read the existing result, (b) re-run with `--force` because inputs changed,
  or (c) proceed anyway for some other reason. Do not silently re-run.
- **No exact match, but `$DATA_ROOT/runs/<dataset>/<model>/` has other variant
  subdirectories** (same model + dataset, different preset/checkpoint) → this is
  related, not a duplicate. Tell the user what other variants of this model have already
  been run on this dataset, with their headline metric values, as useful comparison
  context — then proceed to Step 5. Do not block on this, just surface it.
- **No exact or related match** → say so plainly ("no prior run of `<model>` on
  `<dataset>` found") and proceed to Step 5.

## Step 5 — propose the run

Once model, preset/checkpoint, dataset, and metric are all fully resolved (and any new
model/checkpoint rows have been confirmed and written), summarize the fully-resolved
experiment in one block:

```
dataset:   <name>  (v<version>, n=<expected_entries>)
model:     <name>
preset/checkpoint: <preset or checkpoint name>
metrics:   <requested metric(s)>, plus defaults ecstasy will compute alongside them
```

Then show the **exact CLI invocation** you intend to run, e.g.:

```
ecstasy run --dataset recent_pp --model mentos --checkpoint a5sgd6ul_s90k
```

(or `ecstasy score ...` if predictions already exist and only scoring is needed —
check `predictions/` under the run dir first). Ask for explicit confirmation before
executing. Do not run it in the same turn you propose it unless the user has already
said "go ahead" for exactly this configuration.

## Ground rules throughout

- Never fabricate a registry value (a path, a run_id, a preset name) — if you don't
  have it from the registries or from the user, ask.
- Never silently pick among multiple plausible matches (model, dataset, metric,
  checkpoint) — always surface the candidates and ask.
- Never write to `models.yaml` or `checkpoints.yaml` without reading back the exact
  new row to the user and getting explicit confirmation first.
- Prefer the registries' own machine-readable output (`--json_out` where available)
  over parsing prose you remember from earlier in the conversation — registries change.
