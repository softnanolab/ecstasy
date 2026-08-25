# MiniFold multimer baseline

MiniFold (Wohlwend et al., TMLR 2025) is a single-chain folder. This campaign runs it on
dimers via the same poly-G-linker + positional-index-skip hack the `esmfold` row already
uses, to get a third-party reference point for MENTOS's docking performance.

`reference/` holds the standalone CX3 scripts the design was prototyped and verified with.
They are **not** ecstasy runners — they predate the port and read/write plain PDBs. Port the
logic into `src/ecstasy/models/_runners/minifold_runner.py`; do not wire these in directly.

| File | Role |
|---|---|
| `minifold_dimer_predict.py` | Builds the linker + `residx` input, folds, splits chains, writes `*_pred.pdb` |
| `score_dockq_dir.py` | DockQ over a directory of pred/native pairs |
| `minifold_null_control.py` | Random-placement DockQ floor, 10 draws per target |
| `minifold_predict.pbs`, `minifold_score.pbs` | CX3 PBS wrappers |
| `minifold_residx.patch` | 3-hunk patch letting upstream MiniFold accept an injected `residx` |

See `HANDOFF.md` at the repo root for the full state, the settled design decisions, and the
open architectural question that must be resolved before writing the runner.
