import biotite.structure as structure
import biotite.structure.io as io
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdbx as pdbx
import os
from pathlib import Path
import numpy as np
from biotite.sequence import ProteinSequence
from typing import List, Dict

import matplotlib.pyplot as plt
import seaborn as sns

SRC_DIR = Path(__file__).parent.parent
BASE_DIR = SRC_DIR.parent


def load_structure(input_str: str, hetero: bool = False) -> structure.AtomArray | None:
    """
    Loads a protein structure from a PDB ID (fetching CIF) or a local CIF file path.
    Automatically detects input type based on string length.

    Args:
        input_str (str): The PDB ID or CIF file path.
        hetero (bool): Whether to load hetero atoms. If False, only load protein atoms.

    Returns:
        AtomArray: The loaded structure.
        None: If the structure cannot be loaded.
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
    superimposed, _, ref_indices, sub_indices = (
        structure.superimpose_structural_homologs(
            ref_structure, sub_structure, max_iterations=1
        )
    )
    return structure.tm_score(ref_structure, superimposed, ref_indices, sub_indices)


def get_sequence(structure: structure.AtomArray) -> str:
    """
    Get the sequence of a structure.

    Args:
        structure (AtomArray): The structure to get the sequence from.

    Returns:
        dict: A dictionary of chain IDs and their corresponding sequences.
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
