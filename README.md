# Ecstasy

Benchmarking different models in protein design.

## Setup
### 1. Setting up the environment
Clone the repository with submodules:
```bash
git submodule update --init --recursive
```
#### 3.1. Installing Ecstasy Base Environment
```bash
bash scripts/install/ecstasy.sh
```

#### 3.2. Installing ESMFold
```bash
bash scripts/install/esmfold.sh
```

#### 3.3. Installing Boltz
```bash
bash scripts/install/boltz.sh
```

#### 3.4. Installing ColabFold
```bash
bash scripts/install/colabfold.sh
```

## Configuration & the benchmarking registry

Ecstasy commits **no machine-specific paths**. Everything concrete lives in one of two places:

1. **Filesystem roots** — a gitignored repo-root `.env` (copy `.env.example`). `ecstasy.config`
   resolves them `env > .env`, and the registries (`registry/datasets.yaml`,
   `registry/models.yaml`) reference them with `${VAR}` placeholders.

2. **The Notion benchmarking registry** — the source of truth for everything concrete. It is the
   **Ecstasy** page (set `ECSTASY_NOTION_PAGE` in `.env`), laid out as a project hub:

   - **Registry** — two small reference tables that scripts resolve by name:

     | Table | Holds |
     |-------|-------|
     | **Checkpoints** | `key` (run name) → absolute path, model, run_id, step, num_recycles, status |
     | **Datasets** | name → index path, gt_root, n, split, contact_bin |

   - **Benchmarking Log** — a database of dated, narrative campaign entries (the project's
     evolution, MENTOS-Sprints style): goal → TL;DR → what we ran → results with figures embedded
     inline → decision → artifacts. Raw results and charts live here, in context.

   Benchmark scripts (under `scripts/mentos-perf-benchmarking/`) take checkpoint and dataset
   **names**, never paths — they resolve names → paths from the Registry tables. Nothing in git
   pins where a checkpoint or split physically lives; that lives in Notion (a local copy is kept
   under `$DATA_ROOT`, as today). Each new benchmarking campaign gets its own `scripts/<project>/`
   and its own Benchmarking Log entries.

## [DEV] Management of Repository
### Maintaning dependencies
To add a package to primary dependencies in UV pyproject.toml, run:
`uv add <package-name>`

To add an optional dependency to the `dev` group in UV pyproject.toml, run:
`uv add <package-name> --optional dev`

### Managing submodules
To add a new submodule, run:
```bash
git submodule add <repository-url> <path-to-submodule>
```

### Managing Tests
To only run `ecstasy` tests, run (from the root of the repository):
```bash
uv run pytest tests/
```

### UV for Submodules
Create venv using:
```bash
uv venv envs/boltz && source envs/boltz/bin/activate
```

Install dependencies using:
```bash
uv pip install -e modules/boltz
```