# Ecstasy

Benchmarking different models in protein design.

## Setup
### 1. Setting up the environment
We use [uv](https://docs.astral.sh/uv/) to manage the environment.

Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Once installed, simply run:
```bash
uv sync
``` 
This should install the dependencies in the `pyproject.toml` file. To install all the dev dependencies as well, run: 

(OPTIONAL)
```bash
uv sync --all-extras
```
### 2. Install Ecstacy Module 
To install the Ecstacy module, first activate the environment:
```bash
source .venv/bin/activate
```

Then run:
```bash
uv pip install -e .
```

### 3. Installing different submodules
Clone the repository with submodules:
```bash
git submodule update --init --recursive
```

#### 3.1. Installing ESMFold
```bash
bash scripts/install/esmfold.sh
```

#### 3.2. Installing Boltz
```bash
bash scripts/install/boltz.sh
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