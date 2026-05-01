"""End-to-end benchmark orchestrator.

Stages (run in order; each is independently re-runnable):

    select   parquet -> manifest.json
    prepare  manifest -> boltz YAMLs
    gt       MINT .pt -> ground_truth/<id>.npz
    submit   render PBS template, qsub each protein
    parse    20 CIFs -> boltz_contacts/<id>.npz (empirical P(i,j))
    score    -> metrics/results.csv via mint.metrics.contact_prediction
    all      runs every stage; blocks on the PBS queue between submit & parse

Each stage call updates runs/<run_name>/TODO.md so progress is visible at a
glance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from struct_pred_benchmarking.config import BenchmarkConfig, load_config
from struct_pred_benchmarking.data import ground_truth, select_val
from struct_pred_benchmarking.hpc import scheduler
from struct_pred_benchmarking.metrics import pk
from struct_pred_benchmarking.models.boltz import parse_outputs, prepare_inputs


STAGES = ("select", "prepare", "gt", "submit", "parse", "score", "all")


def _state(cfg: BenchmarkConfig) -> dict[str, bool | str]:
    """Inspect the run dir and return per-stage completion booleans."""
    rd = cfg.run_dir
    manifest = rd / "manifest.json"
    if not manifest.exists():
        return {s: False for s in STAGES if s != "all"}
    import json

    entries = json.loads(manifest.read_text())
    ids = [e["id"] for e in entries]
    n = cfg.diffusion_samples

    inputs_done = all((rd / "inputs" / f"{i}.yaml").exists() for i in ids)
    gt_done = all((rd / "ground_truth" / f"{i}.npz").exists() for i in ids)

    def _pred_done(pid: str) -> bool:
        d = rd / "boltz_predictions" / pid
        if not d.exists():
            return False
        cifs = list(d.glob("boltz_results_*/predictions/*/*_model_*.cif")) or list(d.rglob("*_model_*.cif"))
        return len(cifs) >= n

    submit_done = all(_pred_done(i) for i in ids)
    parse_done = all((rd / "boltz_contacts" / f"{i}.npz").exists() for i in ids)
    score_done = (rd / "metrics" / "results.csv").exists()

    return {
        "select": True,
        "prepare": inputs_done,
        "gt": gt_done,
        "submit": submit_done,
        "parse": parse_done,
        "score": score_done,
        "ids": ids,
    }


def _write_todo(cfg: BenchmarkConfig) -> None:
    s = _state(cfg)
    ids = s.get("ids", [])
    rd = cfg.run_dir
    lines = [f"# TODO — {cfg.run_name}", ""]
    for stage in ("select", "prepare", "gt", "submit", "parse", "score"):
        done = s.get(stage, False)
        mark = "x" if done else " "
        lines.append(f"- [{mark}] {stage}")
    if ids:
        lines += ["", "## per-protein status"]
        for pid in ids:
            yaml_ok = (rd / "inputs" / f"{pid}.yaml").exists()
            gt_ok = (rd / "ground_truth" / f"{pid}.npz").exists()
            n_cifs = 0
            d = rd / "boltz_predictions" / pid
            if d.exists():
                cifs = list(d.glob("boltz_results_*/predictions/*/*_model_*.cif")) or list(d.rglob("*_model_*.cif"))
                n_cifs = len(cifs)
            parsed = (rd / "boltz_contacts" / f"{pid}.npz").exists()
            lines.append(
                f"- {pid}: yaml={yaml_ok} gt={gt_ok} cifs={n_cifs}/{cfg.diffusion_samples} "
                f"parsed={parsed}"
            )
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "TODO.md").write_text("\n".join(lines) + "\n")


def _stage_select(cfg: BenchmarkConfig) -> None:
    path = select_val.write_manifest(cfg)
    entries = select_val.load_manifest(cfg)
    print(f"[select] wrote {path} with {len(entries)} entries")
    for e in entries:
        print(f"          - {e['id']} (len={e['total_sequence_length']})")


def _stage_prepare(cfg: BenchmarkConfig) -> None:
    paths = prepare_inputs.write_all(cfg)
    print(f"[prepare] wrote {len(paths)} boltz YAML files under {cfg.run_dir / 'inputs'}")


def _stage_gt(cfg: BenchmarkConfig) -> None:
    paths = ground_truth.extract_all(cfg)
    print(f"[gt] extracted {len(paths)} ground-truth npz files")


def _stage_submit(cfg: BenchmarkConfig) -> None:
    job_ids = scheduler.submit_all(cfg)
    print(f"[submit] qsub'd {len(job_ids)} jobs")


def _stage_parse(cfg: BenchmarkConfig) -> None:
    paths = parse_outputs.parse_all(cfg)
    print(f"[parse] wrote {len(paths)} contact-prob npz files")


def _stage_score(cfg: BenchmarkConfig) -> None:
    out_path = pk.score_all(cfg)
    print(f"[score] wrote {out_path}")
    print(out_path.read_text())


def _stage_all(cfg: BenchmarkConfig) -> None:
    _stage_select(cfg); _write_todo(cfg)
    _stage_prepare(cfg); _write_todo(cfg)
    _stage_gt(cfg); _write_todo(cfg)
    _stage_submit(cfg); _write_todo(cfg)
    scheduler.wait_for_all(cfg); _write_todo(cfg)
    _stage_parse(cfg); _write_todo(cfg)
    _stage_score(cfg); _write_todo(cfg)


_STAGE_FNS = {
    "select": _stage_select,
    "prepare": _stage_prepare,
    "gt": _stage_gt,
    "submit": _stage_submit,
    "parse": _stage_parse,
    "score": _stage_score,
    "all": _stage_all,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--stage", choices=STAGES, default="all")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    print(f"[run] {cfg.run_name} — stage={args.stage} — run_dir={cfg.run_dir}")
    _STAGE_FNS[args.stage](cfg)
    _write_todo(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
