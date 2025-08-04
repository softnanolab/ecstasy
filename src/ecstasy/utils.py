# Standardized imports
import os
import shutil
from pathlib import Path
from typing import Dict
import json
from glob import glob

import biotite.structure as structure
import biotite.structure.io as io
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdbx as pdbx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from biotite.sequence import ProteinSequence
from DockQ import DockQ

SRC_DIR = Path(__file__).parent.parent
BASE_DIR = SRC_DIR.parent


# 1.Utils for playing around with structures / Sequences
def load_structure(input_str: str, hetero: bool = False) -> structure.AtomArray | None:
    """Load a protein structure from a PDB ID or local CIF file path.

    Args:
        input_str (str): PDB ID or CIF file path.
        hetero (bool): If True, load hetero atoms; otherwise, only protein atoms.

    Returns:
        structure.AtomArray | None: Loaded structure or None if loading fails.
    """
    try:
        # If input = PDB ID
        if len(input_str) == 4 and input_str.isalnum():
            cif_file_object = rcsb.fetch(input_str, "cif", target_path=None)
            cif_file = pdbx.CIFFile.read(cif_file_object)
            atom_array = pdbx.get_structure(cif_file, model=1)

        # If input = CIF file path
        else:
            if not os.path.exists(input_str):
                print(f"Error: CIF file not found at {input_str}.")
                return None
            atom_array = io.load_structure(input_str)

        if atom_array is None or atom_array.array_length == 0:
            print(f"Error: No atoms loaded for structure from {input_str}.")
            return None
        return atom_array[atom_array.hetero == hetero]

    except Exception as e:
        print(f"Error loading structure {input_str}: {e}")
        return None


def get_sequence(structure: structure.AtomArray) -> dict:
    """Extracts chain sequences from a Biotite AtomArray.

    Args:
        structure (structure.AtomArray): Structure to extract sequences from.

    Returns:
        dict: Dictionary mapping chain IDs to sequences.
    """
    sequences = {}
    for chain_id in np.unique(structure.chain_id):
        chain_mask = (structure.chain_id == chain_id) & (structure.atom_name == "CA")
        chain_structure = structure[chain_mask]
        sequence = ""
        for n in range(len(chain_structure)):
            sequence += ProteinSequence.convert_letter_3to1(chain_structure[n].res_name)
        sequences[str(chain_id)] = sequence
    return sequences


def filter_structure_by_chain_id(
    structure: structure.AtomArray, chain_id: str
) -> structure.AtomArray:
    """
    Filter a structure by chain ID.
    Args:
        structure (AtomArray): The structure to filter.
        chain_id (str): The chain ID to filter by.
    Returns:
        AtomArray: The filtered structure.
    """
    filtered_structure = structure[structure.chain_id == chain_id]
    if len(filtered_structure) == 0:
        print(f"Chain {chain_id} not found in structure")
    return filtered_structure


def match_chains(structure_a: structure.AtomArray, structure_b: structure.AtomArray):
    """
    Match chains between two structures.

    Args:
        structure_a (AtomArray): The first structure.
        structure_b (AtomArray): The second structure.

    Returns:
        dict: A dictionary of matched chains.
    """

    seq_a = get_sequence(structure_a)
    seq_b = get_sequence(structure_b)

    matched_chains = {}
    for chain_id_a in seq_a.keys():
        for chain_id_b in seq_b.keys():
            if seq_a[chain_id_a] == seq_b[chain_id_b]:
                matched_chains[chain_id_a] = chain_id_b
                break
        if chain_id_a not in matched_chains:
            print(f"Chain {chain_id_a} not found in {structure_b}")
    return matched_chains


# 2.Utils for scoring structures: TM Score, DockQ Score
def tm_score(
    ref_structure: structure.AtomArray, sub_structure: structure.AtomArray
) -> float:
    """
    Calculate the TM-score between two structures.
    Args:
        ref_structure (AtomArray): The reference structure.
        sub_structure (AtomArray): The structure to superimpose onto the reference.

    Returns:
        float: The TM-score between the two structures.
    """
    try:
        superimposed, _, ref_indices, sub_indices = (
            structure.superimpose_structural_homologs(
                ref_structure, sub_structure, max_iterations=5
            )
        )
        return structure.tm_score(ref_structure, superimposed, ref_indices, sub_indices)
    except ValueError:
        print(f"Error calculating TM-score! Returning 0.")
        return 0

def generate_tm_confusion_matrix(
    proteins, figsize=(10, 8), show_plot=True, return_matrix=False
):
    """
    Generate a confusion matrix of TM scores for all pairs of proteins.

    Args:
        proteins (list): List of protein structures (biotite AtomArray objects)
        figsize (tuple): Figure size for the plot (width, height)
        show_plot (bool): Whether to display the plot
        return_matrix (bool): Whether to return the TM score matrix

    Returns:
        numpy.ndarray: TM score matrix if return_matrix=True, otherwise None
    """

    n = len(proteins)

    # Create a matrix to store TM scores
    tm_matrix = np.zeros((n, n))
    print(f"Running {n*n} pairs")

    # Calculate TM scores for all pairs
    for i in range(n):
        for j in range(n):
            if i == j:
                tm_matrix[i, j] = 1.0  # Self-comparison
            elif j < i:
                # Already calculated for (j, i)
                continue
            else:
                tm = tm_score(proteins[i], proteins[j])
                tm_matrix[i, j] = tm
                tm_matrix[j, i] = tm  # Symmetric matrix

    if show_plot:
        # Create the confusion matrix plot
        plt.figure(figsize=figsize)
        sns.heatmap(
            tm_matrix,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            cbar_kws={"label": "TM Score"},
        )
        plt.title("TM Score Confusion Matrix")
        plt.xlabel("Structure 1")
        plt.ylabel("Structure 2")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print(f"Mean TM Score: {np.mean(tm_matrix):.4f}")
        print(f"Std TM Score: {np.std(tm_matrix):.4f}")
        print(f"Min TM Score: {np.min(tm_matrix):.4f}")
        print(f"Max TM Score: {np.max(tm_matrix):.4f}")

    if return_matrix:
        return tm_matrix


def compute_tm_scores_against_gd(out: dict):
    """
    Computes TM-scores for all permutations in 'out' against ground truth structures.

    Args:
        out (dict): dict mapping n_chains -> protein_id -> list of permuted pdb file paths.
            can be obtained from scripts.generate_permutations.ChainPermutationGenerator.organize_pdb_files_by_protein

    Returns:
        dict: nested dict of TM-scores
    """
    tm_scores_against_gd = {}
    for n_chains in out.keys():
        print("Processing n_chains: ", n_chains)
        tm_scores_against_gd[n_chains] = {}
        for protein_id in out[n_chains].keys():
            print("Processing protein: ", protein_id)
            tm_scores_against_gd[n_chains][protein_id] = {}
            # get the true reference structure
            gd_truth_structure = load_structure(protein_id)
            gd_truth_seq = get_sequence(gd_truth_structure)

            for perm_path in out[n_chains][protein_id]:
                # load the predicted structure
                pred_structure = load_structure(perm_path)
                perm = int(perm_path.split("/")[-1].split("_")[2].split(".")[0][1:])
                print("Processing permutation: ", perm)

                try:
                    tm_scores_against_gd[n_chains][protein_id][perm] = {
                        "overall": tm_score(gd_truth_structure, pred_structure),
                    }
                except Exception as e:
                    print(f"Error processing permutation {perm}: {e}")
                    tm_scores_against_gd[n_chains][protein_id][perm] = {"overall": 0}
                    continue

                matched_chains = match_chains(gd_truth_structure, pred_structure)
                for chain_id in gd_truth_seq.keys():
                    gd_truth_chain = filter_structure_by_chain_id(
                        gd_truth_structure, chain_id
                    )
                    pred_chain = filter_structure_by_chain_id(
                        pred_structure, matched_chains[chain_id]
                    )

                    try:
                        tm_scores_against_gd[n_chains][protein_id][perm][chain_id] = (
                            tm_score(gd_truth_chain, pred_chain)
                        )
                    except Exception as e:
                        print(f"Error processing chain {chain_id}: {e}")
                        tm_scores_against_gd[n_chains][protein_id][perm][chain_id] = 0
                        continue
    return tm_scores_against_gd


def plot_tm_scores_for_proteins_against_gd(tm_scores_against_gd: dict):
    """
    Plots TM-scores (chain and overall) for all protein_ids in tm_scores_against_gd.
    For each protein, two boxplots are shown: one for all chain TM-scores, one for overall TM-scores.
    All proteins are shown together in one plot.

    Args:
        tm_scores_against_gd (dict): dict as produced, with protein_ids as top-level keys.
    """
    protein_ids = list(tm_scores_against_gd.keys())
    boxplot_data = []
    scatter_x = []
    scatter_y = []
    xtick_labels = []
    stat_texts = []
    x_positions = []
    pos = 0

    for protein_id in protein_ids:
        if protein_id not in tm_scores_against_gd:
            print(f"Warning: {protein_id} not found in tm_scores_against_gd")
            # Add empty data to keep positions aligned
            boxplot_data.extend([[], []])
            xtick_labels.extend([f"{protein_id} (chains)", f"{protein_id} (overall)"])
            stat_texts.extend([None, None])
            x_positions.extend([pos, pos + 1])
            pos += 2
            continue

        perms = sorted(tm_scores_against_gd[protein_id].keys())
        chain_scores = []
        overall_scores = []
        for perm in perms:
            entry = tm_scores_against_gd[protein_id][perm]
            overall_scores.append(entry["overall"])
            for chain_id in entry:
                if chain_id == "overall":
                    continue
                chain_scores.append(entry[chain_id])
        # Save for boxplot
        boxplot_data.append(chain_scores)
        boxplot_data.append(overall_scores)
        xtick_labels.append(f"{protein_id} (chains)")
        xtick_labels.append(f"{protein_id} (overall)")
        x_positions.extend([pos, pos + 1])

        # Stats for annotation
        if len(chain_scores) > 0:
            chain_min, chain_max, chain_mean = (
                np.min(chain_scores),
                np.max(chain_scores),
                np.mean(chain_scores),
            )
        else:
            chain_min, chain_max, chain_mean = np.nan, np.nan, np.nan
        if len(overall_scores) > 0:
            overall_min, overall_max, overall_mean = (
                np.min(overall_scores),
                np.max(overall_scores),
                np.mean(overall_scores),
            )
        else:
            overall_min, overall_max, overall_mean = np.nan, np.nan, np.nan
        stat_texts.append((chain_min, chain_max, chain_mean))
        stat_texts.append((overall_min, overall_max, overall_mean))

        # For scatter
        np.random.seed(0)
        jitter_chain = np.random.normal(0, 0.04, len(chain_scores))
        jitter_overall = np.random.normal(0, 0.04, len(overall_scores))
        scatter_x.extend([pos + jitter_chain, (pos + 1) + jitter_overall])
        scatter_y.extend([chain_scores, overall_scores])

        pos += 2

        # Print stats
        print(
            f"{protein_id} Chain TM-scores: min={chain_min:.4f}, max={chain_max:.4f}, mean={chain_mean:.4f}"
        )
        print(
            f"{protein_id} Overall TM-scores: min={overall_min:.4f}, max={overall_max:.4f}, mean={overall_mean:.4f}"
        )

    # Flatten scatter data
    scatter_x_flat = (
        np.concatenate([np.array(xx) for xx in scatter_x])
        if scatter_x
        else np.array([])
    )
    scatter_y_flat = (
        np.concatenate([np.array(yy) for yy in scatter_y])
        if scatter_y
        else np.array([])
    )

    fig, ax = plt.subplots(figsize=(2.5 * len(protein_ids) * 2, 8))

    # Boxplot
    bp = ax.boxplot(
        boxplot_data,
        positions=x_positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
    )

    # Scatter
    ax.scatter(
        scatter_x_flat,
        scatter_y_flat,
        alpha=0.7,
        color="black",
        zorder=3,
        label="Permutations",
    )

    # X-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xtick_labels, rotation=20, ha="right")

    ax.set_ylabel("TM-score")
    ax.set_title(f"TM-scores: Chains vs Overall (per permutation)")

    # Annotate min, max, mean on the chart near each box
    def stat_text(minv, maxv, meanv):
        if np.isnan(minv):
            return "No data"
        return f"min={minv:.3f}\nmax={maxv:.3f}\nmean={meanv:.3f}"

    # Get y-limits for placement
    ymin, ymax = ax.get_ylim()
    ypos = ymax - (ymax - ymin) * 0.15

    for i, stats in enumerate(stat_texts):
        if stats is not None:
            minv, maxv, meanv = stats
            ax.text(
                x_positions[i],
                ypos,
                stat_text(minv, maxv, meanv),
                ha="center",
                va="top",
                fontsize=11,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
            )

    plt.tight_layout()
    plt.show()


def dockq_score(protein_a_path: str, protein_b_path: str, chain_map: dict) -> float:
    """
    Calculate the DockQ score for a given protein pair.
    """
    protein_a = DockQ.load_PDB(protein_a_path)
    protein_b = DockQ.load_PDB(protein_b_path)
    dockq_scores = DockQ.run_on_all_native_interfaces(
        model_structure=protein_a, native_structure=protein_b, chain_map=chain_map
    )
    # return the average dockq score for the multimer comparision
    return np.mean([_item["DockQ"] for _item in list(dockq_scores[0].values())])


# 3.Utils for writing files: FASTA, PDB, etc.
def write_fasta_esmfold(
    sequences: Dict[str, Dict[str, str]],
    output_dir: str,
    file_name: str = "batch_esmfold.fasta",
):
    """
    Write a dictionary of sequences to a FASTA file.

    Args:
        sequences (Dict[str, Dict[str, str]]): A dictionary of protein names and their sequences.
        output_dir (str): The directory to write the FASTA file to.
        file_name (str): The name of the FASTA file.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    to_write = []
    for protein_name, protein_sequences in sequences.items():
        comb_seq = ":".join(protein_sequences.values())
        to_write.append(f">{protein_name}\n{comb_seq}")

    with open(os.path.join(output_dir, file_name), "w") as f:
        f.write("\n".join(to_write))


def combine_fastas_esmfold(input_dir: str, combined_name: str = "combined.fasta"):
    """
    Combine all FASTA files in a directory into a single FASTA file.

    Args:
        input_dir (str): The directory containing FASTA files to combine.
        combined_name (str): The name of the combined FASTA file.

    Returns:
        None
    """

    # Get all FASTA files in the directory
    fasta_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".fasta")])

    if not fasta_files:
        print(f"No FASTA files found in {input_dir}")
        return

    # Read and combine all FASTA files
    combined_content = []
    for fasta_file in fasta_files:
        file_path = os.path.join(input_dir, fasta_file)
        with open(file_path, "r") as f:
            content = f.read().strip()
            if content:  # Only add non-empty content
                combined_content.append(content)

    # Write combined content to output file
    output_path = os.path.join(input_dir, combined_name)
    with open(output_path, "w") as f:
        f.write("\n".join(combined_content))

    print(f"Combined {len(fasta_files)} FASTA files into {output_path}")


def write_fasta_boltz(
    sequences: Dict[str, Dict[str, str]], output_dir: str, use_msas: bool = False
):
    """
    Write a dictionary of sequences to a FASTA file.

    Args:
        sequences (Dict[str, Dict[str, str]]): A dictionary of protein names and their sequences.
        output_dir (str): The directory to write the FASTA file to.
        use_msas (bool): Whether to use MSAs. (default: False)

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    for protein_name, protein_sequences in sequences.items():
        to_write = [
            f">{_chain_id}|protein{"|empty" if not use_msas else ""}\n{_seq}"
            for _chain_id, _seq in protein_sequences.items()
        ]

        with open(os.path.join(output_dir, f"{protein_name}.fasta"), "w") as f:
            f.write("\n".join(to_write))


def get_confidence_file_path_boltz(cif_file_path: str) -> str:
    """
    Get the confidence file path for a given CIF file path.
    """
    dir_path = os.path.dirname(cif_file_path)
    model_file = os.path.basename(cif_file_path)
    confidence_file = "confidence_" + model_file.replace(".cif", ".json")
    return os.path.join(dir_path, confidence_file)


def get_confidence_file_path_colabfold(pdb_file_path: str) -> str:
    """Return the associated *scores* JSON path for a ColabFold PDB file.

    ColabFold writes both *unrelaxed* and *relaxed* PDBs alongside JSON files
    that contain per-model score metrics such as pTM and ipTM.  The filename
    pattern is::

        <key>_(un)relaxed_rank_XXX_... .pdb
        <key>_scores_rank_XXX_... .json

    We derive the JSON path by replacing the substring ``unrelaxed`` or
    ``relaxed`` with ``scores`` and switching the extension to ``.json``.
    """

    import re

    dir_path = os.path.dirname(pdb_file_path)
    filename = os.path.basename(pdb_file_path)

    # Replace the first occurrence of 'unrelaxed' or 'relaxed' with 'scores'
    filename = re.sub(r"unrelaxed|relaxed", "scores", filename, count=1)
    confidence_file = filename.replace(".pdb", ".json")
    return os.path.join(dir_path, confidence_file)


def clean_up_colabfold_predictions(predictions_dir: str):
    """Organise ColabFold output files by prediction.

    ColabFold multimer runs often dump many files with long names into a single
    directory.  For permutation-invariance experiments we follow the naming
    scheme

        n{chains}_{protein_id}_p{perm}<whatever>.<ext>

    for example::

        n4_4YTP_p12_unrelaxed_rank_001_model_1.pdb
        n4_4YTP_p12_unrelaxed_rank_001_model_1_scores.json

    All files (or directories) that share the *prediction key* – the part of the
    name that matches the regular expression ``n\\d+_[A-Za-z0-9]+_p\\d+`` – belong
    to the same prediction.  This helper collects every such item and moves it
    into a dedicated sub-folder with that key as its name, giving a much
    cleaner directory structure::

        predictions_dir/
        ├── n4_4YTP_p12/
        │   ├── n4_4YTP_p12_unrelaxed_rank_001_model_1.pdb
        │   ├── n4_4YTP_p12_unrelaxed_rank_001_model_1_scores.json
        │   └── …
        ├── n4_4YTP_p34/
        │   └── …
        └── …

    Parameters
    ----------
    predictions_dir : str
        Path to the directory produced by ColabFold (e.g.
        ``/path/to/outputs/colabfold``).
    """

    predictions_path = Path(predictions_dir).expanduser()
    if not predictions_path.is_dir():
        raise FileNotFoundError(f"{predictions_path} is not a valid directory")

    # ColabFold writes a small <prefix>.done.txt file per prediction – we use its
    # *stem* (filename without the ``.done.txt`` suffix) as the prediction key.
    done_files = list(predictions_path.glob("*.done.txt"))
    if not done_files:
        print(f"No *.done.txt files found in {predictions_path}; nothing to organise.")
        return

    # Create a list of prediction keys, processed longest-first to avoid
    # scenarios like key "..._p1" matching files that actually belong to
    # "..._p10".
    keys = sorted({f.stem for f in done_files}, key=len, reverse=True)

    # Pre-create destination directories
    for key in keys:
        (predictions_path / key).mkdir(exist_ok=True)

    moved_items = 0
    for item in predictions_path.iterdir():
        # Skip hidden files and the *.done.txt marker files themselves
        if item.name.startswith(".") or item.suffixes[-2:] == [".done", ".txt"]:
            continue

        # Skip items that are already organised (their parent name equals a key)
        if item.parent != predictions_path and item.parent.name in keys:
            continue

        for key in keys:
            if item.name.startswith(key):
                # Ensure the key match is exact and not a prefix of a longer key
                remainder_idx = len(key)
                if len(item.name) == remainder_idx or item.name[remainder_idx] in {
                    "_",
                    ".",
                }:
                    dest_dir = predictions_path / key
                    try:
                        shutil.move(str(item), dest_dir)
                        moved_items += 1
                    except shutil.Error as e:
                        print(f"[WARNING] Could not move {item}: {e}")
                    break  # Done with this item

    print(
        f"Moved {moved_items} items into organised sub-folders inside {predictions_path}."
    )


def find_max_protein_iptm_boltz(
    folder_path: str,
) -> tuple[str, float] | tuple[None, None]:
    """
    Finds the confidence_*.json file in the given folder with the highest 'protein_iptm' value.

    Args:
        folder_path (str): Path to the folder containing confidence_*.json files.

    Returns:
        tuple: (filename, protein_iptm) of the file with the highest protein_iptm, or (None, None) if not found.
    """

    confidence_files = glob(os.path.join(folder_path, "confidence_*.json"))

    max_iptm = float("-inf")
    max_file = None
    max_score = None

    for file_path in confidence_files:
        with open(file_path, "r") as f:
            data = json.load(f)
            iptm = data.get("protein_iptm", None)
            if iptm is not None and iptm > max_iptm:
                max_iptm = iptm
                max_file = file_path
                max_score = iptm

    if max_file is not None:
        print(
            f"File with highest protein_iptm: {os.path.basename(max_file)} (protein_iptm={max_score})"
        )
        return os.path.basename(max_file), max_score
    else:
        print("No confidence_*.json files with 'protein_iptm' found.")
        return None, None
