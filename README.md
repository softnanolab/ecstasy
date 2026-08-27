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

Ecstasy commits **no machine-specific paths** directly — everything concrete lives behind
`${VAR}` placeholders in one of the committed registry files:

- **Filesystem roots** — a gitignored repo-root `.env` (copy `.env.example`). `ecstasy.config`
  resolves them `env > .env`, and the registries (`registry/datasets.yaml`,
  `registry/models.yaml`, `registry/checkpoints.yaml`) reference them with `${VAR}` placeholders.

- **Checkpoints** (`registry/checkpoints.yaml`) — MENTOS checkpoint *names* resolve to concrete
  `abs_path` / `run_id` / `num_recycles` / `model_config_path` rows, committed like any other
  registry file. Add a row by hand (or via the `/experiment` command) for each checkpoint you
  want to run; see the file's header comment for the schema.

- **Datasets** (`registry/datasets.yaml`) — every evaluation split, its identity fields, and
  its `built_from` rebuild recipe.

Benchmark scripts (under `scripts/mentos-perf-benchmarking/`) take checkpoint and dataset
**names**, never paths — they resolve names → paths from these committed registry files. Each
new benchmarking campaign gets its own `scripts/<project>/`.

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