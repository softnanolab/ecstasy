# benchmarks/struct_pred_pk

P@K contact-prediction benchmark — **MINT vs Boltz-2** (extensible to other
structure-prediction models) on the MINT PDB validation set.

## Approach

For each protein:
- **MINT**: distogram head emits a `(T, T)` per-pair contact probability directly.
- **Boltz-2**: run `--diffusion_samples 20`, parse the 20 CIF outputs, compute
  pairwise Cβ-Cβ distances, and define
  `P(i, j) = (#samples with Cβ-Cβ < 8 Å) / 20`.

Both models are then ranked against the same MINT ground-truth distogram
labels using `mint.metrics.contact_prediction.metrics_inter_chain` /
`granular_metrics_intra_chain` so the numbers are apples-to-apples with what
MINT itself reports.

## Layout

```
benchmarks/struct_pred_pk/
├── pyproject.toml              # uv project, py3.12; deps incl. mint editable
├── configs/
│   └── pilot_1protein_cx3.yaml # 1-protein pilot — edit absolute paths to match host
├── src/struct_pred_benchmarking/
│   ├── config.py               # YAML → dataclass
│   ├── data/                   # val-set selection + ground-truth extraction
│   ├── models/boltz/           # boltz YAML prep + CIF parsing
│   ├── metrics/                # P@K wrapper around mint.metrics
│   ├── hpc/                    # PBS template + qstat-capped scheduler
│   └── cli/run_benchmark.py    # stage orchestrator
├── runs/                       # per-run artefacts (gitignored)
└── tests/
```

## External setup (one-time, per machine)

1. **Boltz-2** (this repo's install script):
   ```bash
   cd /path/to/ecstasy
   bash scripts/install/boltz.sh
   ```
   Creates `./envs/boltz` (conda, py3.12) with the `boltz` CLI.

2. **CUDA-driver-matched torch wheels** — the default install pulls cu13
   wheels which require driver ≥ ~12.8. If the cluster's NVIDIA driver
   reports an older CUDA Driver API (e.g. 12.6), reinstall torch with
   matching wheels:
   ```bash
   conda run -p ./envs/boltz \
     pip install --upgrade --force-reinstall torch==2.11.0 \
     --index-url https://download.pytorch.org/whl/cu126
   ```

3. **MINT** — clone alongside this repo (the default `mint` editable source
   in `pyproject.toml` resolves to `../../../mint` relative to this benchmark
   directory):
   ```bash
   cd /path/to/parent_of_ecstasy
   git clone https://github.com/VarunUllanat/mint.git
   ```

4. **Benchmark venv**:
   ```bash
   cd /path/to/ecstasy/benchmarks/struct_pred_pk
   uv venv --python 3.12
   uv pip install -e .
   ```

## Running the pilot

Edit `configs/pilot_1protein_cx3.yaml` so all `/CHANGE_ME/...` paths point at
the absolute paths on the target machine.

```bash
cd /path/to/ecstasy/benchmarks/struct_pred_pk
source .venv/bin/activate

# stage-by-stage:
python -m struct_pred_benchmarking.cli.run_benchmark --config configs/pilot_1protein_cx3.yaml --stage select
python -m struct_pred_benchmarking.cli.run_benchmark --config configs/pilot_1protein_cx3.yaml --stage prepare
python -m struct_pred_benchmarking.cli.run_benchmark --config configs/pilot_1protein_cx3.yaml --stage gt
python -m struct_pred_benchmarking.cli.run_benchmark --config configs/pilot_1protein_cx3.yaml --stage submit
# wait for PBS (qstat -u $USER), then:
python -m struct_pred_benchmarking.cli.run_benchmark --config configs/pilot_1protein_cx3.yaml --stage parse
python -m struct_pred_benchmarking.cli.run_benchmark --config configs/pilot_1protein_cx3.yaml --stage score

# or all in one (blocks on the queue between submit & parse):
python -m struct_pred_benchmarking.cli.run_benchmark --config configs/pilot_1protein_cx3.yaml --stage all
```

Per-run artefacts land under `runs/<run_name>/`:

```
runs/<run_name>/
  manifest.json        list of selected proteins
  inputs/              one boltz YAML per protein
  ground_truth/        *.npz with (T, T) distogram bins + chain_ids
  boltz_predictions/   boltz --out_dir per protein (20 CIFs each)
  boltz_contacts/      *.npz with (T, T) empirical P(i, j)
  metrics/results.csv  one row per protein
  logs/                PBS .out / .err
  TODO.md              live progress checklist
```

## Scaling up

Same code paths work for N proteins — copy the pilot config, set
`n_proteins: 10`, drop the `length_filter`, rerun. The PBS scheduler caps
queued jobs at `pbs.max_queued` (default 50).
