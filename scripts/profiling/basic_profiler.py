"""
This script is used to profile the basic performance of the ESMFold and Boltz
predictions.
"""
from pathlib import Path
from collections import defaultdict



import warnings
import json
import matplotlib.pyplot as plt
import numpy as np


# ignore this warning from biotite
warnings.filterwarnings(
    "ignore",
    message="Attribute 'auth_atom_id' not found within 'atom_site' category. The fallback attribute 'label_atom_id' will be used instead",
)


def plot_benchmark_stacked_bars(
    benchmark_json_path: str,
    pdb_features_json_path: str,
    output_prefix: str = "benchmark_plot"
):
    """
    Plots three separate stacked bar charts for time, peak memory, and current memory usage
    for each submodule across sequence lengths.
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

