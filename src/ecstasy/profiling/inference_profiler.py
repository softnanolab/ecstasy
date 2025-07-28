"""
This script profiles the basic performance of ESMFold and Boltz predictions.
"""

import json
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np

# Suppress specific biotite warning
warnings.filterwarnings(
    "ignore",
    message="Attribute 'auth_atom_id' not found within 'atom_site' category. The fallback attribute 'label_atom_id' will be used instead",
)


class InferenceProfiler:
    """Analyzes and visualizes inference performance for protein folding models.

    This class processes benchmark data from either ESMFold or Boltz to generate
    plots of memory usage and inference time against protein sequence length.

    Args:
        model_type (str): The model type, either "esmfold" or "boltz".
        pdb_features_json_path (str | Path): Path to the JSON file containing
            PDB features, including sequence lengths.
        predictions_dir (str | Path): Path to the directory containing
            the benchmark results.
        outputs_dir (str | Path, optional): Directory to save the generated
            plots. Defaults to "inference_plots".
        output_prefix (str, optional): Prefix for the output plot
            filenames. Defaults to "inference_profile".

    Raises:
        ValueError: If an unsupported `model_type` is provided.
    """

    _BOLTZ_TIME_KEY = "time"
    _BOLTZ_MEM_KEY = "peak memory"

    def __init__(
        self,
        model_type: str,
        pdb_features_json_path: str | Path,
        predictions_dir: str | Path,
        outputs_dir: str | Path = "inference_plots",
        output_prefix: str = "inference_profile",
    ) -> None:
        self.model_type = model_type.lower()
        if self.model_type not in {"esmfold", "boltz"}:
            raise ValueError("model_type must be either 'esmfold' or 'boltz'.")

        self.pdb_features_json_path = Path(pdb_features_json_path)
        self.predictions_dir = Path(predictions_dir)
        self.outputs_dir = Path(outputs_dir)
        self.output_prefix = output_prefix

        if self.model_type == "esmfold":
            self.benchmark_path = self.predictions_dir / "benchmark.json"
        else:  # boltz
            self.benchmark_path = self.predictions_dir

        with open(self.pdb_features_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._pdbid_to_len: Dict[str, int] = {
            str(d["pdb_id"]).upper(): d["total_residues"]
            for d in data
            if isinstance(d, dict) and d.get("pdb_id") and d.get("total_residues")
        }

        self._metrics: dict = {}

    def plot_memory_vs_seq_len(self) -> None:
        """Generates and saves a plot of peak memory vs. sequence length."""
        self._collect_metrics_if_needed()
        self._plot_memory_line_chart()

    def plot_time_vs_seq_len(self) -> None:
        """Generates and saves both a stacked-bar and CI line chart for inference time."""
        self._collect_metrics_if_needed()
        # Stacked bar (incremental contributions)
        self._plot_time_stacked_bar_chart()
        # Line chart with 95 % CI
        self._plot_time_line_chart()

    def _get_sorted_submodules(self) -> list[str]:
        """Returns a chronologically sorted list of all submodules found in the metrics."""
        all_submodules = list(
            {
                submodule
                for seq_len_data in self._metrics.values()
                for submodule in seq_len_data.keys()
            }
        )

        if self.model_type == "esmfold":
            # Sort based on a predefined, fixed order
            fixed_order = ["after_esm_lm", "before_trunk", "after_trunk", "after_secondary_heads"]
            return sorted(all_submodules, key=lambda s: fixed_order.index(s) if s in fixed_order else float('inf'))
        
        # For boltz, sort numerically by recycle number
        return self._sort_boltz_submodules(all_submodules)

    def _collect_metrics_if_needed(self) -> None:
        """Parses benchmark files if metric data has not yet been collected."""
        if self._metrics:
            return

        if self.model_type == "esmfold":
            self._metrics = self._parse_esmfold_benchmark()
        else:
            self._metrics = self._parse_boltz_benchmarks()

    def _parse_esmfold_benchmark(self) -> dict:
        """Parses an ESMFold `benchmark.json` file to populate metric dictionaries."""
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        submodule_order = ["after_esm_lm", "before_trunk", "after_trunk", "after_secondary_heads"]

        for key, entry in data.items():
            if not key.startswith("n"):
                continue

            try:
                pdb_id = key.split("_")[1].upper()
            except IndexError:
                continue

            seq_len = self._pdbid_to_len.get(pdb_id)
            if seq_len is None:
                continue

            for submodule_name in submodule_order:
                submodule_data = entry.get(submodule_name)
                if not isinstance(submodule_data, dict):
                    continue

                # Cumulative time
                current_time = submodule_data.get("time", 0.0)
                metrics[seq_len][submodule_name]["time"].append(current_time)

                # Peak memory
                peak_mem = submodule_data.get("peak_memory")
                if peak_mem is not None:
                    metrics[seq_len][submodule_name]["peak_memory"].append(peak_mem)
        return metrics

    def _sort_boltz_submodules(self, submodules: list[str]) -> list[str]:
        """Sorts boltz submodules by recycle number, with specific bookends."""
        prefix = "after_pairformer_recycle_"
        
        # Start with a baseline
        ordered_list = ["before_pairformers"] if "before_pairformers" in submodules else []
        
        # Sort recycles numerically
        ordered_list.extend(sorted(
            [s for s in submodules if s.startswith(prefix)],
            key=lambda s: int(s[len(prefix) :]),
        ))
        
        # Add final stages
        if "after_structure_module" in submodules:
            ordered_list.append("after_structure_module")
        if "after_confidence_module" in submodules:
            ordered_list.append("after_confidence_module")
            
        return [s for s in ordered_list if s in submodules]

    def _parse_boltz_benchmarks(self) -> dict:
        """Parses all Boltz benchmark files in a directory to populate metric dictionaries."""
        metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        pattern = re.compile(r"n\d+_([A-Za-z0-9]{4})_p\d+")

        def _iter_bench_files(path: Path) -> Iterable[Path]:
            """Yield all candidate Boltz benchmark JSONs under *path*."""
            if path.is_file():
                if path.name.startswith("benchmark_stats") and path.suffix == ".json":
                    yield path
            else:
                # match both with and without explicit model index
                patterns = ["**/benchmark_stats_*_model_*.json", "**/benchmark_stats_*.json"]
                for pat in patterns:
                    yield from path.glob(pat)

        for file in _iter_bench_files(self.benchmark_path):
            name = file.stem
            try:
                trimmed = name[len("benchmark_stats_") :].split("_model_")[0]
            except Exception:
                continue
            m = pattern.match(trimmed)
            if m is None:
                continue

            pdb_id = m.group(1).upper()
            seq_len = self._pdbid_to_len.get(pdb_id)
            if seq_len is None:
                continue

            try:
                with open(file, "r", encoding="utf-8") as f:
                    bench = json.load(f)
            except Exception:
                continue
            
            # Add a zero-time entry point for consistent delta calculation
            bench["before_pairformers"] = {"time": 0.0, "peak memory": bench.get("after_preprocessing", {}).get("peak memory", 0.0)}

            submodule_order = self._sort_boltz_submodules(list(bench.keys()))

            for submodule_name in submodule_order:
                step = bench[submodule_name]
                if not isinstance(step, dict):
                    continue

                t = step.get(self._BOLTZ_TIME_KEY) or step.get(self._BOLTZ_TIME_KEY.capitalize())
                if t is not None:
                    metrics[seq_len][submodule_name]["time"].append(float(t))

                pm = step.get(self._BOLTZ_MEM_KEY) or step.get(
                    self._BOLTZ_MEM_KEY.replace(" ", "_")
                )
                if pm is not None:
                    peak_mem_gb = float(pm)
                    if peak_mem_gb > 1024:  # Heuristic for MB to GB
                        peak_mem_gb /= 1024
                    metrics[seq_len][submodule_name]["peak_memory"].append(peak_mem_gb)
        return metrics

    def _plot_time_stacked_bar_chart(self) -> None:
        """Creates a stacked bar chart for incremental time usage per submodule."""
        if not self._metrics:
            raise RuntimeError("No metric data available – did parsing succeed?")

        plt.figure(figsize=(12, 7))
        all_seq_lens = sorted(self._metrics.keys())
        all_submodules = self._get_sorted_submodules()

        bottoms = np.zeros(len(all_seq_lens))
        last_means = np.zeros(len(all_seq_lens))
        submodule_index = {subm: idx for idx, subm in enumerate(all_submodules)}

        for idx, submodule in enumerate(all_submodules):
            current_means = []
            for seq_len in all_seq_lens:
                values = self._metrics[seq_len][submodule].get("time")
                if values:
                    current_means.append(np.mean(values))
                else:
                    # fallback to previous submodule's cumulative time
                    if idx > 0:
                        prev_sub = all_submodules[idx - 1]
                        prev_vals = self._metrics[seq_len][prev_sub].get("time")
                        current_means.append(np.mean(prev_vals) if prev_vals else 0)
                    else:
                        current_means.append(0)

            current_means = np.array(current_means)
            time_deltas = np.maximum(0, current_means - last_means)

            plt.bar(
                range(len(all_seq_lens)),
                time_deltas,
                bottom=bottoms,
                label=submodule,
            )
            bottoms += time_deltas
            last_means = current_means

        plt.xlabel("Sequence Length (total residues)")
        plt.ylabel("Inference Time (s)")
        plt.title(f"Inference Time vs. Sequence Length for {self.model_type.capitalize()}")
        plt.xticks(range(len(all_seq_lens)), all_seq_lens, rotation=45)
        plt.legend()
        plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
        plt.tight_layout()

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{self.output_prefix}_{self.model_type}_time.png"
        plt.savefig(self.outputs_dir / file_name, dpi=200)
        plt.close()

    def _plot_memory_line_chart(self) -> None:
        """Creates and saves a line plot for peak memory with confidence intervals."""
        self._plot_line_metric(metric_type="peak_memory", ylabel="Peak Memory Usage (GB)")

    def _plot_line_metric(self, metric_type: str, ylabel: str, *, file_suffix: str | None = None) -> None:
        """Creates and saves a line plot for a given metric with confidence intervals.

        Args:
            metric_type (str): Key inside ``self._metrics`` (``"time"`` or ``"peak_memory"``).
            ylabel (str): Y-axis label.
            file_suffix (str, optional): Filename suffix. Defaults to ``metric_type``.
        """
        if not self._metrics:
            raise RuntimeError("No metric data available – did parsing succeed?")

        suffix = file_suffix or metric_type

        plt.figure(figsize=(12, 7))
        all_seq_lens = sorted(self._metrics.keys())
        all_submodules = self._get_sorted_submodules()

        for submodule in all_submodules:
            seq_lens, means, CIs = [], [], []
            for seq_len in all_seq_lens:
                values = self._metrics[seq_len][submodule].get(metric_type)
                if values:
                    seq_lens.append(seq_len)
                    mean = np.mean(values)
                    means.append(mean)
                    if len(values) > 1:
                        std_err = np.std(values, ddof=1) / np.sqrt(len(values))
                        CIs.append(1.96 * std_err)
                    else:
                        CIs.append(0)

            if seq_lens:
                p = plt.plot(seq_lens, means, label=submodule)
                plt.fill_between(
                    seq_lens,
                    np.array(means) - np.array(CIs),
                    np.array(means) + np.array(CIs),
                    color=p[0].get_color(),
                    alpha=0.2,
                )

        plt.xlabel("Sequence Length (total residues)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs. Sequence Length for {self.model_type.capitalize()}")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{self.output_prefix}_{self.model_type}_{suffix}.png"
        plt.savefig(self.outputs_dir / file_name, dpi=200)
        plt.close()


def plot_benchmark_stacked_bars(
    benchmark_json_path: str,
    pdb_features_json_path: str,
    output_prefix: str = "benchmark_plot",
):
    """Plot stacked bar charts for time, peak memory, and current memory usage by submodule.

    Args:
        benchmark_json_path (str): Path to benchmark JSON file.
        pdb_features_json_path (str): Path to PDB features JSON file.
        output_prefix (str): Prefix for output plot files.
    """
    # Load benchmark data
    with open(benchmark_json_path, "r") as f:
        benchmark = json.load(f)

    # Load pdb_features data
    with open(pdb_features_json_path, "r") as f:
        pdb_features = json.load(f)

    # Map PDB IDs to sequence lengths
    pdbid_to_length = {}
    for entry in pdb_features:
        if isinstance(entry, dict):
            pdbid = entry.get("pdb_id")
            length = entry.get("total_residues")
            if pdbid and length:
                pdbid_to_length[pdbid] = length

    # Collect all submodule names
    submodules = set()
    for k, v in benchmark.items():
        if not k.startswith("n"):  # skip non-prediction keys
            continue
        for subk in v:
            if subk.startswith("after_") or subk.startswith("before_"):
                submodules.add(subk)
    submodules = sorted(list(submodules))

    # Aggregate data: {seq_len: {submodule: [metric values]}}
    time_data = {}
    peak_mem_data = {}
    curr_mem_data = {}
    for k, v in benchmark.items():
        if not k.startswith("n"):  # skip non-prediction keys
            continue
        # Example key: n2_4ZHY_p1
        try:
            parts = k.split("_")
            pdb_id = parts[1]
        except Exception:
            continue
        seq_len = pdbid_to_length.get(pdb_id)
        if seq_len is None:
            continue
        for subm in submodules:
            if subm in v:
                t = v[subm].get("time", 0)
                pm = v[subm].get("peak_memory", 0)
                cm = v[subm].get("current_memory", 0)
                time_data.setdefault(seq_len, {}).setdefault(subm, []).append(t)
                peak_mem_data.setdefault(seq_len, {}).setdefault(subm, []).append(pm)
                curr_mem_data.setdefault(seq_len, {}).setdefault(subm, []).append(cm)

    # For each metric, plot stacked bar chart
    def plot_stacked(metric_data, ylabel, outname):
        # Sort sequence lengths
        seq_lens = sorted(metric_data.keys())
        x = np.arange(len(seq_lens))
        # For each submodule, get mean value per seq_len
        bottoms = np.zeros(len(seq_lens))
        plt.figure(figsize=(12, 7))
        for subm in submodules:
            values = [np.mean(metric_data[sl].get(subm, [0])) for sl in seq_lens]
            plt.bar(x, values, bottom=bottoms, label=subm)
            bottoms += np.array(values)
        plt.xlabel("Sequence Length (total residues)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} per Submodule (Stacked) vs Sequence Length")
        plt.xticks(x, seq_lens, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_{outname}.png", dpi=200)
        plt.close()

    plot_stacked(time_data, "Time (s)", "time")
    plot_stacked(peak_mem_data, "Peak Memory Usage (GB)", "peak_memory")
    plot_stacked(curr_mem_data, "Current Memory Usage (GB)", "current_memory")
