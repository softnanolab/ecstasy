"""ecstasy CLI entry point: `ecstasy bench <subcmd> --config <yaml>`."""
from __future__ import annotations

import os
from pathlib import Path

import fire
import yaml

from ecstasy.benchmarks import load_benchmark, BENCHMARKS
from ecstasy.models import load_model, MODELS


def _read_config(config: str) -> dict:
    cfg = yaml.safe_load(Path(config).read_text())
    cfg.setdefault("data_root", os.environ.get("DATA_ROOT"))
    if not cfg.get("data_root"):
        raise ValueError("data_root not set: either set DATA_ROOT env or add data_root: to the config")
    return cfg


def _materialize(config: str, overrides: dict | None = None):
    cfg = _read_config(config)
    if overrides:
        cfg.setdefault("model_config", {}).update(overrides)
    bench = load_benchmark(cfg["benchmark"], data_root=Path(cfg["data_root"]))
    model = load_model(cfg["model"], config=cfg)
    return cfg, bench, model


def _adapter_overrides(model_config: str | None, model_weights: str | None) -> dict | None:
    """Build adapter-config overrides from generic CLI flags.

    --model_config / --model_weights are accepted for adapters whose model definition
    is data-driven (e.g., MINT, where weights and config vary per experiment). They
    land under cfg["model_config"]["model_config_path"] and cfg["model_config"]["model_weights_path"].
    Adapters that ignore them are unaffected.
    """
    out: dict = {}
    if model_config:
        out["model_config_path"] = model_config
    if model_weights:
        out["model_weights_path"] = model_weights
    return out or None


class _Bench:
    """`ecstasy bench <subcmd>` — run benchmark stages."""

    def list(self):
        print("benchmarks:", sorted(BENCHMARKS))
        print("models:    ", sorted(MODELS))

    def msa(self, config: str, model_config: str | None = None, model_weights: str | None = None):
        cfg, bench, model = _materialize(config, _adapter_overrides(model_config, model_weights))
        if not model.needs_msa:
            print(f"{model.name}: needs_msa=False, skipping")
            return
        from ecstasy.pipelines import contact_prediction
        contact_prediction.run_msa(cfg, bench, model)

    def predict(self, config: str, submit: bool = False,
                model_config: str | None = None, model_weights: str | None = None):
        cfg, bench, model = _materialize(config, _adapter_overrides(model_config, model_weights))
        from ecstasy.pipelines import contact_prediction
        contact_prediction.run_predict(cfg, bench, model, submit=submit)

    def score(self, config: str,
              model_config: str | None = None, model_weights: str | None = None):
        cfg, bench, model = _materialize(config, _adapter_overrides(model_config, model_weights))
        from ecstasy.pipelines import contact_prediction
        contact_prediction.run_score(cfg, bench, model)

    def all(self, config: str, submit: bool = False,
            model_config: str | None = None, model_weights: str | None = None):
        self.msa(config, model_config=model_config, model_weights=model_weights)
        self.predict(config, submit=submit, model_config=model_config, model_weights=model_weights)
        self.score(config, model_config=model_config, model_weights=model_weights)

    def compare(self, task: str, data_root: str | None = None):
        from ecstasy.pipelines import contact_prediction
        contact_prediction.run_compare(task=task, data_root=Path(data_root or os.environ["DATA_ROOT"]))


def main():
    fire.Fire({"bench": _Bench()})


if __name__ == "__main__":
    main()
