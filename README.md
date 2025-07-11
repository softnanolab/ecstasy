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