# Standardized imports
import json
from typing import Optional
from typing_extensions import deprecated
import warnings
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from math import inf
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ecstasy import utils
from ecstasy.utils import generate_tm_confusion_matrix

# Suppress specific biotite warning
warnings.filterwarnings(
    "ignore",
    message="Attribute 'auth_atom_id' not found within 'atom_site' category. The fallback attribute 'label_atom_id' will be used instead",
)


@deprecated("Function is deprecated and will be removed in the future.")
def process_seeds_for_a_single_permutation(
    path_to_model_seeds: list[str], num_cpus: int = 128
) -> list:
    """Processes seeds for a single prediction.

    Args:
        path_to_model_seeds (list[str]): List of paths to model seeds.
        num_cpus (int, optional): Number of CPUs to use. Defaults to 128.

    Returns:
        list: List of results from pairwise comparisons.
    """

    num_model_seeds = len(path_to_model_seeds)

    # Prepare all (i, j) pairs for i > j
    pairs = [
        # (file i path, file j path)
        (path_to_model_seeds[i], path_to_model_seeds[j])
        for i in range(num_model_seeds)
        for j in range(num_model_seeds)
        if i > j
    ]

    # Use multiprocessing Pool
    with Pool(processes=num_cpus) as pool:
        results = list(
            tqdm(
                pool.imap(do_monomer_and_multimer_comparision, pairs),
                total=len(pairs),
                desc="Pairwise comparisons",
            )
        )

    return results


@deprecated("Function is deprecated and will be removed in the future.")
def calculate_tm_statistics(organized_files: dict) -> dict:
    """Calculates mean and std TM scores for each protein and chain count.

    Args:
        organized_files (dict): Organized PDB file structure.

    Returns:
        dict: Statistics for each chain count and protein.
    """
    statistics = {}

    for n_chains, proteins in organized_files.items():
        print(f"Processing {n_chains}-chain proteins...")

        chain_statistics = {}

        for protein_id, file_paths in proteins.items():
            print(
                f"  Processing protein {protein_id} with {len(file_paths)} permutations"
            )

            # Load structures
            structures = [utils.load_structure(fp) for fp in file_paths]

            # Calculate total chain length for this protein (using first structure)
            protein_sequences = utils.get_sequence(structures[0])
            total_length = sum(len(seq) for seq in protein_sequences.values())

            # Calculate TM confusion matrix
            tm_matrix = generate_tm_confusion_matrix(structures, return_matrix=True)

            # Extract upper triangle (excluding diagonal) for pairwise comparisons
            upper_triangle = tm_matrix[np.triu_indices_from(tm_matrix, k=1)]

            # Calculate statistics for this protein
            if len(upper_triangle) > 0:
                chain_statistics[protein_id] = {
                    "mean": np.mean(upper_triangle),
                    "std": np.std(upper_triangle),
                    "count": len(upper_triangle),
                    "scores": upper_triangle,
                    "total_length": total_length,
                }
                print(
                    f"    {protein_id}: mean={chain_statistics[protein_id]['mean']:.3f}, "
                    f"std={chain_statistics[protein_id]['std']:.3f}, n={chain_statistics[protein_id]['count']}, "
                    f"total_length={total_length}"
                )

        statistics[n_chains] = chain_statistics

    return statistics


@deprecated("Function is deprecated and will be removed in the future.")
def plot_tm_statistics(statistics: dict, output_dir: str = None):
    """Plots mean and standard deviation TM scores for each protein, separated by chain count.

    Args:
        statistics (dict): TM score statistics for each chain count and protein.
        output_dir (str, optional): Directory to save plots. Defaults to None.
    """
    if not statistics:
        print("No statistics to plot")
        return

    # Create separate plots for each chain count
    for n_chains, protein_stats in statistics.items():
        if not protein_stats:
            continue

        plt.figure(figsize=(14, 6))

        # Get protein IDs and their statistics
        protein_ids = list(protein_stats.keys())
        means = [protein_stats[pid]["mean"] for pid in protein_ids]
        stds = [protein_stats[pid]["std"] for pid in protein_ids]
        total_lengths = [protein_stats[pid]["total_length"] for pid in protein_ids]

        # Create bar plot with error bars
        x_pos = np.arange(len(protein_ids))
        bars = plt.bar(
            x_pos,
            means,
            yerr=stds,
            capsize=5,
            alpha=0.7,
            color="skyblue",
            edgecolor="navy",
        )

        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 0.01,
                f"{mean:.3f}\n±{std:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.xlabel("Protein ID")
        plt.ylabel("TM Score")
        plt.title(f"TM Scores for {n_chains}-Chain Proteins")

        # Create labels with protein ID, total length, and sample size
        labels = [
            f'{pid}\n({total_lengths[i]} aa, n={protein_stats[pid]["count"]})'
            for i, pid in enumerate(protein_ids)
        ]
        plt.xticks(x_pos, labels, rotation=45)
        plt.grid(axis="y", alpha=0.3)
        plt.ylim(0, 1.1)

        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                output_path / f"tm_scores_{n_chains}_chains.png",
                dpi=300,
                bbox_inches="tight",
            )
            print(f"Plot saved to {output_path / f'tm_scores_{n_chains}_chains.png'}")

        plt.show()

        # Also create a box plot for each chain count
        plt.figure(figsize=(14, 6))

        # Prepare data for box plot
        box_data = [protein_stats[pid]["scores"] for pid in protein_ids]
        labels = [
            f"{pid}\n({total_lengths[i]} aa, n={len(scores)})"
            for i, (pid, scores) in enumerate(zip(protein_ids, box_data))
        ]

        # Create box plot
        bp = plt.boxplot(box_data, tick_labels=labels)

        # Add individual data points on top
        for i, scores in enumerate(box_data):
            # Add jitter to x-coordinates to spread out points
            x_jittered = np.random.normal(i + 1, 0.04, len(scores))
            plt.scatter(x_jittered, scores, alpha=0.6, s=20, color="red", zorder=10)

        plt.xlabel("Protein ID")
        plt.ylabel("TM Score")
        plt.title(f"TM Score Distribution for {n_chains}-Chain Proteins")
        plt.grid(axis="y", alpha=0.3)
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)

        plt.tight_layout()

        if output_dir:
            plt.savefig(
                output_path / f"tm_scores_distribution_{n_chains}_chains.png",
                dpi=300,
                bbox_inches="tight",
            )
            print(
                f"Distribution plot saved to {output_path / f'tm_scores_distribution_{n_chains}_chains.png'}"
            )

        plt.show()


class AnalyzePermutations:
    """
    Primary class for analyzing permutations and seeds.

    This class provides methods for organizing predictions, comparing permutations and seeds,
    and plotting various statistics.

    Attributes:
        model_name (str): One of ``{"boltz", "colabfold", "esmfold"}`` - determines which
            organize_* function to use and how to interpret confidence JSONs
        predictions_dir (str): Directory containing the prediction files (Boltz `.cif`, ColabFold
            `.pdb`, or ESMFold `.pdb`).
        output_dir (str): Path where JSON result files and plots will be written. For convenience,
            the same path is passed to the lower-level functions; they decide
            whether it is interpreted as a directory or full file path.
        relaxed (bool, optional): Only relevant for ColabFold predictions - choose *relaxed* (default) or
            *unrelaxed* models.
    """

    def __init__(
        self,
        model_name: str,
        predictions_dir: str,
        output_dir: str,
        *,
        relaxed: bool = True,
        esmfold_logs_filepath: Optional[str] = None,
    ):
        self.predictions_dir = predictions_dir

        assert model_name in [
            "boltz",
            "colabfold",
            "esmfold",
        ], f"Invalid model name: {model_name}. Must be one of 'boltz', 'colabfold', or 'esmfold'."
        self.model_name = model_name.lower()
        self.output_dir = output_dir
        self.relaxed = relaxed

        if model_name == "esmfold":
            assert (
                esmfold_logs_filepath is not None
            ), "ESMFold logs file path is required."
            assert Path(
                esmfold_logs_filepath
            ).is_file(), "ESMFold logs file does not exist."
            esmfold_logs_path = Path(esmfold_logs_filepath)
            self.esmfold_confidence_scores = self.parse_esmfold_logs(esmfold_logs_path)

    @staticmethod
    def parse_esmfold_logs(log_path: str) -> dict:
        """
        Parses an esmfold log file to extract pLDDT and pTM scores into a nested dict.

        Args:
            log_path (str): Path to the esmfold.txt log file.

        Returns:
            dict: Nested dictionary {n_chains: {protein_id: {permutation: {"plddt": float, "ptm": float}}}}
        """
        # Nested dict: n_chains -> protein_id -> permutation -> scores
        result = defaultdict(lambda: defaultdict(dict))

        # Regex to match prediction lines
        pattern = re.compile(
            r"Predicted structure for n(\d+)_(\w+)_p(\d+) with length \d+, pLDDT ([\d.]+), pTM ([\d.]+)"
        )

        with open(log_path, "r") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    n_chains = int(match.group(1))
                    protein_id = match.group(2)
                    permutation = int(match.group(3))
                    plddt = float(match.group(4))
                    ptm = float(match.group(5))
                    result[n_chains][protein_id][permutation] = {
                        "plddt": plddt,
                        "ptm": ptm,
                    }

        return dict(result)

    def organize_esmfold_predictions(self) -> dict:
        """Organizes PDB files by protein ID and chain count.

        Returns:
            dict: {n_chains: {protein_id: [file_paths]}}
        """
        pdb_path = Path(self.predictions_dir)
        organized_files = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # Find all PDB files
        pdb_files = list(pdb_path.glob("*.pdb"))

        for pdb_file in pdb_files:
            # Parse filename to extract n_chains and protein_id
            # Expected format: n{num_chains}_{protein_id}_p{permutation}.pdb
            filename = pdb_file.stem
            parts = filename.split("_")

            if len(parts) >= 3 and parts[0].startswith("n"):
                try:
                    n_chains = int(parts[0][1:])  # Extract number after 'n'
                    protein_id = parts[1]
                    permutation = int(parts[2][1:])

                    organized_files[n_chains][protein_id][permutation].append(
                        str(pdb_file)
                    )
                except ValueError:
                    print(f"Could not parse filename: {filename}")
                    continue

        return dict(organized_files)

    def organize_boltz_predictions(self) -> dict:
        """Organizes Boltz prediction files by protein ID and chain count.

        Returns:
            dict: {n_chains: {protein_id: {permutation: [file_paths]}}}
        """
        predictions_path = Path(self.predictions_dir)
        organized_files = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # Find all subdirectories (each represents a permutation)
        prediction_dirs = [d for d in predictions_path.iterdir() if d.is_dir()]

        for pred_dir in prediction_dirs:
            # Parse directory name to extract n_chains and protein_id
            # Expected format: n{num_chains}_{protein_id}_p{permutation}
            dirname = pred_dir.name
            parts = dirname.split("_")

            if len(parts) >= 3 and parts[0].startswith("n"):
                try:
                    n_chains = int(parts[0][1:])  # Extract number after 'n'
                    protein_id = parts[1]
                    permutation_number = int(parts[2][1:])

                    # List all model_*.cif files for a given prediction
                    cif_files = sorted(list(pred_dir.glob("*model_*.cif")))
                    if cif_files:
                        organized_files[n_chains][protein_id][permutation_number] = [
                            str(cif_file) for cif_file in cif_files
                        ]
                    else:
                        print(f"No cif files found in directory: {dirname}")

                except ValueError:
                    print(f"Could not parse directory name: {dirname}")
                    continue

        return dict(organized_files)

    def organize_colabfold_predictions(self) -> dict:
        """
        Organize ColabFold prediction files hierarchically by protein ID and chain count.
        """
        from collections import defaultdict
        from pathlib import Path

        predictions_path = Path(self.predictions_dir)
        organized_files = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # Each permutation is stored in its own directory e.g. n4_6DFS_p1/
        prediction_dirs = [d for d in predictions_path.iterdir() if d.is_dir()]

        for pred_dir in prediction_dirs:
            dirname = pred_dir.name
            parts = dirname.split("_")

            # Expected format: n{num_chains}_{protein_id}_p{permutation}
            if (
                len(parts) >= 3
                and parts[0].startswith("n")
                and parts[2].startswith("p")
            ):
                try:
                    n_chains = int(parts[0][1:])
                    protein_id = parts[1]
                    permutation_number = int(parts[2][1:])
                except ValueError:
                    continue

                # Prefer unrelaxed models (smaller) but fall back to relaxed models
                if self.relaxed:
                    pdb_files = sorted(pred_dir.glob("*_relaxed_*.pdb"))
                else:
                    pdb_files = sorted(pred_dir.glob("*_unrelaxed_*.pdb"))

                organized_files[n_chains][protein_id][permutation_number] = [
                    str(pdb) for pdb in pdb_files
                ]

        # Convert to plain dicts
        return {
            k: {kk: dict(vv) for kk, vv in v.items()}
            for k, v in organized_files.items()
        }

    def organize_predictions(self) -> dict:
        """Organize predictions based on the model name.

        Returns:
            dict: Organized predictions.
        """
        match self.model_name:
            case "boltz":
                return self.organize_boltz_predictions()
            case "colabfold":
                return self.organize_colabfold_predictions()
            case "esmfold":
                return self.organize_esmfold_predictions()

    def get_iptm_score(self, file_path: str) -> float | None:
        """Get the iPTM score for a prediction.

        Args:
            file_path (str): Path to the file.

        Returns:
            float | None: The iPTM score.
        """
        try:
            if self.model_name == "boltz":
                # Boltz format
                conf_path = utils.get_confidence_file_path_boltz(file_path)
                with open(conf_path, "r") as f:
                    iptm_val = json.load(f)["protein_iptm"]
            elif self.model_name == "colabfold":
                # ColabFold format (PDB)
                conf_path = utils.get_confidence_file_path_colabfold(file_path)
                with open(conf_path, "r") as f:
                    iptm_val = json.load(f)["iptm"]
            # Using pTM scores as iPTM is not available for ESMFold
            elif self.model_name == "esmfold":
                filename_contents = Path(file_path).stem.split("_")
                n_chains = int(filename_contents[0][1:])
                pdb_id = filename_contents[1]
                permutation = int(filename_contents[2][1:])
                iptm_val = self.esmfold_confidence_scores[n_chains][pdb_id][
                    permutation
                ]["ptm"]
            return iptm_val
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            # Skip files with missing/invalid confidence JSON
            return None

    def _wrapper_compare_seeds(self, job: tuple) -> tuple:
        """Wrapper for monomer and multimer comparison for multiprocessing.

        Args:
            job (tuple): Job tuple containing indices, file paths, and metadata.

        Returns:
            tuple: Results including scores and metadata.
        """
        i, j, file_i_path, file_j_path, n_chain, protein_id, permutation_number = job
        dockq, tm, iptm = self.do_monomer_and_multimer_comparision(
            (file_i_path, file_j_path)
        )
        return (
            dockq,
            tm,
            iptm,
            n_chain,
            protein_id,
            permutation_number,
            i,
            j,
        )

    def _wrapper_compare_permutations(self, job: tuple) -> tuple:
        """Wrapper used by multiprocessing to compare best seeds across permutations.

        Args:
            job (tuple): Job tuple containing indices, file paths, and metadata.

        Returns:
            tuple: Results including scores and metadata.
        """
        perm_i, perm_j, file_i_path, file_j_path, n_chain, protein_id = job

        dockq, tm, iptm_prod = self.do_monomer_and_multimer_comparision(
            (file_i_path, file_j_path)
        )

        return (
            dockq,
            tm,
            iptm_prod,
            n_chain,
            protein_id,
            perm_i,
            perm_j,
        )

    def do_monomer_and_multimer_comparision(
        self,
        file_path_pair: tuple[str, str],
    ) -> tuple[float, float, float]:
        """Performs monomer and multimer comparison for a given pair of files.

        Args:
            file_path_pair (tuple[str, str]): Tuple of file paths.

        Returns:
            tuple[float, float, float]: DockQ score, TM score, iPTM product.
        """
        # protein structures for TM Score comparison
        file_i_path, file_j_path = file_path_pair
        file_i = utils.load_structure(file_i_path)
        file_j = utils.load_structure(file_j_path)

        # obtain chain map from i -> j
        chain_map = utils.match_chains(file_i, file_j)

        # calculate the DockQ for the multimer comparision
        dockq_score = utils.dockq_score(file_i_path, file_j_path, chain_map)

        # calculate the TM Score for the monomer comparision
        tm_score = 0
        for chain_id_i, chain_id_j in chain_map.items():
            chain_i = utils.filter_structure_by_chain_id(file_i, chain_id_i)
            chain_j = utils.filter_structure_by_chain_id(file_j, chain_id_j)
            tm_score += utils.tm_score(chain_i, chain_j)

        # calculate the average TM Score for all monomer pair comparisions
        tm_score /= len(chain_map)

        # get the iptm scores for the two proteins
        file_i_iptm = self.get_iptm_score(file_i_path)
        file_j_iptm = self.get_iptm_score(file_j_path)

        # return dockq score, tm score, and the product of the two iptm scores
        return dockq_score, tm_score, file_i_iptm * file_j_iptm

    def compare_permutations(
        self,
        organized_files: dict,
        *,
        n_cpus: int = 128,
    ):
        """Compare the *best* model seed of every permutation against all other permutations.

        The procedure is:
            1. For every protein and every permutation, load the confidence JSON of
            each model seed and select the file with the highest iPTM score.
            2. Create pairwise comparison jobs **between permutations** (not seeds).
            3. Run the comparisons in parallel and collect DockQ, TM and product
            iPTM scores.

        The resulting nested dictionary structure is::

            results[n_chains][protein_id]["{perm_i}_{perm_j}"] = (dockq, tm, iptm)

        where *perm_i* and *perm_j* are permutation indices with ``perm_i > perm_j``.

        Args:
            organized_files (dict): Mapping ``{n_chains: {protein_id: {permutation: [model_files]}}}`` as
                produced by :func:`organize_boltz_predictions` or
                :func:`organize_colabfold_predictions`.
            output_dir (str): JSON file path to write the results to.
            n_cpus (int, optional): Number of worker processes for multiprocessing. Default is 128.

        Returns:
            dict: Nested results dictionary as described above.
        """

        jobs: list[tuple] = []
        results: dict = {}

        # Get total number of proteins for progress bar
        total_proteins = sum(len(proteins) for proteins in organized_files.values())

        # Create progress bar for proteins
        with tqdm(total=total_proteins, desc="Processing proteins") as pbar:
            for n_chain, proteins in organized_files.items():
                results[n_chain] = {}
                for protein_id, permutations in proteins.items():
                    results[n_chain][protein_id] = {}

                    # Determine best seed (highest iptm) for every permutation
                    best_seed_paths: dict[int, str] = {}
                    for permutation_number, file_paths in permutations.items():
                        best_iptm = -inf
                        best_path = None

                        for file_path in file_paths:
                            iptm_val = self.get_iptm_score(file_path)
                            if iptm_val is None:
                                continue

                            if iptm_val > best_iptm:
                                best_iptm = iptm_val
                                best_path = file_path

                        if best_path is not None:
                            best_seed_paths[permutation_number] = best_path

                    # Create pairwise jobs between permutations using the best seeds
                    perm_numbers = sorted(list(best_seed_paths.keys()))
                    for i in perm_numbers:
                        for j in perm_numbers:
                            if i > j:
                                fp_i = best_seed_paths[i]
                                fp_j = best_seed_paths[j]

                                # Reserve dictionary entry; will be filled after multiprocessing
                                results[n_chain][protein_id][f"{i}_{j}"] = ()

                                jobs.append(
                                    (
                                        i,
                                        j,
                                        fp_i,
                                        fp_j,
                                        n_chain,
                                        protein_id,
                                    )
                                )

                    # Update progress bar after each protein
                    pbar.update(1)

        # Execute comparisons in parallel
        if jobs:
            with Pool(processes=n_cpus) as pool:
                multiprocessing_results = list(
                    tqdm(
                        pool.imap(self._wrapper_compare_permutations, jobs),
                        total=len(jobs),
                        desc="Pairwise comparisons (across permutations)",
                    )
                )

            # Populate results dictionary
            for (
                dockq,
                tm,
                iptm,
                n_chain_res,
                protein_id_res,
                i_res,
                j_res,
            ) in multiprocessing_results:
                results[n_chain_res][protein_id_res][f"{i_res}_{j_res}"] = (
                    dockq,
                    tm,
                    iptm,
                )

        # Write results to JSON
        with open(Path(self.output_dir) / "permutations_comparisons.json", "w") as f:
            json.dump(results, f, indent=4)

        return results

    def compare_seeds(
        self,
        organized_files: dict,
        *,
        n_cpus: int = 128,
    ):
        """Processes all permutations for all proteins using multiprocessing.

        Args:
            organized_files (dict): Organized file structure.
            n_cpus (int, optional): Number of CPUs to use. Defaults to 128.

        Returns:
            dict: Results of all pairwise comparisons.
        """
        # generate a list of comparision jobs for multiprocessing

        jobs: list[tuple] = []
        results: dict = {}
        for n_chain, proteins in organized_files.items():
            results[n_chain] = {}
            for protein_id, permutations in proteins.items():
                results[n_chain][protein_id] = {}
                for permutation_number, file_paths in permutations.items():
                    results[n_chain][protein_id][permutation_number] = {}
                    jobs.extend(
                        [
                            # (file i path, file j path)
                            (
                                i,
                                j,
                                file_paths[i],
                                file_paths[j],
                                n_chain,
                                protein_id,
                                permutation_number,
                            )
                            for i in range(len(file_paths))
                            for j in range(len(file_paths))
                            if i > j
                        ]
                    )

        # Use multiprocessing Pool
        with Pool(processes=n_cpus) as pool:
            multiprocessing_results = list(
                tqdm(
                    pool.imap(self._wrapper_compare_seeds, jobs),
                    total=len(jobs),
                    desc="Pairwise comparisons",
                )
            )
        # save results to a json file
        for (
            _dockq,
            _tm,
            _iptm,
            _n_chain,
            _protein_id,
            _permutation_number,
            _i,
            _j,
        ) in multiprocessing_results:
            results[_n_chain][_protein_id][_permutation_number][f"{_i}_{_j}"] = (
                _dockq,
                _tm,
                _iptm,
            )

        with open(Path(self.output_dir) / "seeds_comparisons.json", "w") as f:
            json.dump(results, f, indent=4)

        return results

    def plot_iptm_vs_scores_heatmaps(self, comparision_type: str) -> None:
        """Plots heatmaps and histograms for iPTM, DockQ, and TM scores from a JSON file.

        Args:
            comparision_type (str): Type of comparision to plot. Must be one of "seeds", "permutations".
        """

        assert comparision_type in [
            "seeds",
            "permutations",
        ], f"Invalid comparision type: {comparision_type}. Must be one of 'seeds', 'permutations'."

        # Load comparison results
        with open(
            Path(self.output_dir) / f"{comparision_type}_comparisons.json", "r"
        ) as f:
            data = json.load(f)

        dockq_scores = []
        tm_scores = []
        iptm_scores = []

        # Traverse the hierarchy and collect scores
        for n_chains in data:
            for pdb_id in data[n_chains]:
                if comparision_type == "seeds":
                    for perm in data[n_chains][pdb_id]:
                        for pair in data[n_chains][pdb_id][perm]:
                            scores = data[n_chains][pdb_id][perm][pair]
                        if not isinstance(scores, list) or len(scores) != 3:
                            continue
                        dockq, tm, iptm = scores
                        dockq_scores.append(dockq)
                        tm_scores.append(tm)
                        iptm_scores.append(iptm)

                elif comparision_type == "permutations":
                    for pair in data[n_chains][pdb_id]:
                        scores = data[n_chains][pdb_id][pair]
                        if not isinstance(scores, list) or len(scores) != 3:
                            continue
                        dockq, tm, iptm = scores
                        dockq_scores.append(dockq)
                        tm_scores.append(tm)
                        iptm_scores.append(iptm)

        # Convert to numpy arrays for convenience
        dockq_scores = np.asarray(dockq_scores)
        tm_scores = np.asarray(tm_scores)
        iptm_scores = np.asarray(iptm_scores)

        # Plot iPTM vs DockQ (probability-normalized counts)
        n_samples = len(dockq_scores)
        weights = np.full(n_samples, 1 / n_samples)
        plt.figure(figsize=(7, 5))
        plt.hist2d(
            iptm_scores,
            dockq_scores,
            bins=40,
            weights=weights,
            cmap="viridis",
        )
        plt.xlabel("iPTM Product")
        plt.ylabel("DockQ Score")
        plt.title("iPTM vs DockQ (Probability Mass)")
        plt.colorbar(label="Probability")
        plt.tight_layout()
        plt.savefig(
            Path(self.output_dir) / f"{comparision_type}_iptm_vs_dockq_heatmap.png",
            dpi=300,
        )
        plt.show()

        # Plot iPTM vs TM (probability-normalized counts)
        n_samples_tm = len(tm_scores)
        weights_tm = np.full(n_samples_tm, 1 / n_samples_tm)
        plt.figure(figsize=(7, 5))
        plt.hist2d(
            iptm_scores,
            tm_scores,
            bins=40,
            weights=weights_tm,
            cmap="viridis",
        )
        plt.xlabel("iPTM Product")
        plt.ylabel("TM Score")
        plt.title("iPTM vs TM (Probability Mass)")
        plt.colorbar(label="Probability")
        plt.tight_layout()
        plt.savefig(
            Path(self.output_dir) / f"{comparision_type}_iptm_vs_tm_heatmap.png",
            dpi=300,
        )
        plt.show()

        # 1D Histogram: DockQ Score (normalized counts)
        plt.figure(figsize=(7, 4))
        plt.hist(
            dockq_scores,
            bins=40,
            color="dodgerblue",
            edgecolor="black",
            alpha=0.7,
            weights=np.ones_like(dockq_scores) / len(dockq_scores),
        )
        plt.xlabel("DockQ Score")
        plt.ylabel("Probability")
        plt.title("Histogram of DockQ Scores (Probability)")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(
            Path(self.output_dir) / f"{comparision_type}_dockq_score_histogram.png",
            dpi=300,
        )
        plt.show()

        # 1D Histogram: TM Score (normalized counts)
        plt.figure(figsize=(7, 4))
        plt.hist(
            tm_scores,
            bins=40,
            color="seagreen",
            edgecolor="black",
            alpha=0.7,
            weights=np.ones_like(tm_scores) / len(tm_scores),
        )
        plt.xlabel("TM Score")
        plt.ylabel("Probability")
        plt.title("Histogram of TM Scores (Probability)")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(
            Path(self.output_dir) / f"{comparision_type}_tm_score_histogram.png",
            dpi=300,
        )
        plt.show()

        # 1D Histogram: iPTM Score (normalized counts)
        plt.figure(figsize=(7, 4))
        plt.hist(
            iptm_scores,
            bins=40,
            color="orange",
            edgecolor="black",
            alpha=0.7,
            weights=np.ones_like(iptm_scores) / len(iptm_scores),
        )
        plt.xlabel("iPTM Product")
        plt.ylabel("Probability")
        plt.title("Histogram of iPTM Scores (Probability)")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(
            Path(self.output_dir) / f"{comparision_type}_iptm_score_histogram.png",
            dpi=300,
        )
        plt.show()

        # CDF: TM Score
        tm_scores_sorted = np.sort(tm_scores)
        tm_cdf = np.arange(1, len(tm_scores_sorted) + 1) / len(tm_scores_sorted)
        plt.figure(figsize=(7, 4))
        plt.plot(tm_scores_sorted, tm_cdf, color="seagreen", lw=2)
        plt.xlabel("TM Score")
        plt.ylabel("Cumulative Probability")
        plt.title("Cumulative Distribution Function of TM Scores")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            Path(self.output_dir) / f"{comparision_type}_tm_score_cdf.png",
            dpi=300,
        )
        plt.show()

        # CDF: DockQ Score
        dockq_scores_sorted = np.sort(dockq_scores)
        dockq_cdf = np.arange(1, len(dockq_scores_sorted) + 1) / len(
            dockq_scores_sorted
        )
        plt.figure(figsize=(7, 4))
        plt.plot(dockq_scores_sorted, dockq_cdf, color="dodgerblue", lw=2)
        plt.xlabel("DockQ Score")
        plt.ylabel("Cumulative Probability")
        plt.title("Cumulative Distribution Function of DockQ Scores")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            Path(self.output_dir) / f"{comparision_type}_dockq_score_cdf.png",
            dpi=300,
        )
        plt.show()

        # CDF: iPTM Score
        iptm_scores_sorted = np.sort(iptm_scores)
        iptm_cdf = np.arange(1, len(iptm_scores_sorted) + 1) / len(iptm_scores_sorted)
        plt.figure(figsize=(7, 4))
        plt.plot(iptm_scores_sorted, iptm_cdf, color="orange", lw=2)
        plt.xlabel("iPTM Product")
        plt.ylabel("Cumulative Probability")
        plt.title("Cumulative Distribution Function of iPTM Scores")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            Path(self.output_dir) / f"{comparision_type}_iptm_score_cdf.png",
            dpi=300,
        )
        plt.show()
