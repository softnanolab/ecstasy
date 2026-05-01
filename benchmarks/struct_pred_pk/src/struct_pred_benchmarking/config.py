from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class LengthFilter:
    min_total: int = 0
    max_total: int = 10_000


@dataclass
class PBSConfig:
    queue: str = "v1_a100"
    walltime: str = "04:00:00"
    ncpus: int = 8
    mem_gb: int = 64
    ngpus: int = 1
    max_queued: int = 50


@dataclass
class BenchmarkConfig:
    run_name: str
    split_parquet: Path
    mint_data_root: Path
    boltz_env: Path
    conda_base: Path = Path("~/miniconda3").expanduser()
    n_proteins: int = 1
    seed: int = 42
    length_filter: LengthFilter = field(default_factory=LengthFilter)
    model: str = "boltz2"
    diffusion_samples: int = 20
    contact_threshold_angstrom: float = 8.0
    intra_min_sep: int = 24
    pbs: PBSConfig = field(default_factory=PBSConfig)

    repo_root: Path = field(init=False)

    @property
    def run_dir(self) -> Path:
        return self.repo_root / "runs" / self.run_name


def load_config(path: str | Path) -> BenchmarkConfig:
    path = Path(path).resolve()
    with path.open() as fh:
        data = yaml.safe_load(fh)

    length_raw = data.pop("length_filter", {}) or {}
    pbs_raw = data.pop("pbs", {}) or {}

    cfg = BenchmarkConfig(
        run_name=data["run_name"],
        split_parquet=Path(data["split_parquet"]),
        mint_data_root=Path(data["mint_data_root"]),
        boltz_env=Path(data["boltz_env"]),
        conda_base=Path(data.get("conda_base", "~/miniconda3")).expanduser(),
        n_proteins=int(data.get("n_proteins", 1)),
        seed=int(data.get("seed", 42)),
        length_filter=LengthFilter(**length_raw),
        model=data.get("model", "boltz2"),
        diffusion_samples=int(data.get("diffusion_samples", 20)),
        contact_threshold_angstrom=float(data.get("contact_threshold_angstrom", 8.0)),
        intra_min_sep=int(data.get("intra_min_sep", 24)),
        pbs=PBSConfig(**pbs_raw),
    )

    # Resolve the repo root to the parent of the config file's parent (configs/ -> repo)
    cfg.repo_root = path.parent.parent.resolve()
    return cfg
