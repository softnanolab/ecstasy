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

# Suppress specific biotite warning
warnings.filterwarnings(
    "ignore",
    message="Attribute 'auth_atom_id' not found within 'atom_site' category. The fallback attribute 'label_atom_id' will be used instead",
)

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

        # ------------------------------------------------------------------
        # Create a single figure containing 8 sub-plots arranged in a 4×2 grid
        # ------------------------------------------------------------------

        fig, axs = plt.subplots(4, 2, figsize=(14, 20))

        # Row 1 — Heatmaps
        # iPTM vs TM (left)
        im_tm = axs[0, 0].hist2d(
            iptm_scores,
            tm_scores,
            bins=40,
            weights=np.full(tm_scores.shape, 1 / len(tm_scores)),
            cmap="viridis",
        )
        axs[0, 0].set_xlabel("iPTM Product")
        axs[0, 0].set_ylabel("TM Score")
        axs[0, 0].set_title("iPTM vs TM (Probability Mass)")
        fig.colorbar(im_tm[3], ax=axs[0, 0], label="Probability")

        # iPTM vs DockQ (right)
        im_dq = axs[0, 1].hist2d(
            iptm_scores,
            dockq_scores,
            bins=40,
            weights=np.full(dockq_scores.shape, 1 / len(dockq_scores)),
            cmap="viridis",
        )
        axs[0, 1].set_xlabel("iPTM Product")
        axs[0, 1].set_ylabel("DockQ Score")
        axs[0, 1].set_title("iPTM vs DockQ (Probability Mass)")
        fig.colorbar(im_dq[3], ax=axs[0, 1], label="Probability")

        # Row 2 — Histograms
        # TM histogram (left)
        axs[1, 0].hist(
            tm_scores,
            bins=40,
            color="seagreen",
            edgecolor="black",
            alpha=0.7,
            weights=np.ones_like(tm_scores) / len(tm_scores),
        )
        axs[1, 0].set_xlabel("TM Score (normalized)")
        axs[1, 0].set_ylabel("Probability")
        axs[1, 0].set_title("Histogram of TM Scores (Probability)")
        axs[1, 0].set_ylim(0, 1)

        # DockQ histogram (right)
        axs[1, 1].hist(
            dockq_scores,
            bins=40,
            color="dodgerblue",
            edgecolor="black",
            alpha=0.7,
            weights=np.ones_like(dockq_scores) / len(dockq_scores),
        )
        axs[1, 1].set_xlabel("DockQ Score (normalized)")
        axs[1, 1].set_ylabel("Probability")
        axs[1, 1].set_title("Histogram of DockQ Scores (Probability)")
        axs[1, 1].set_ylim(0, 1)

        # Row 3 — CDFs
        tm_sorted = np.sort(tm_scores)
        tm_cdf = np.arange(1, len(tm_sorted) + 1) / len(tm_sorted)
        axs[2, 0].plot(tm_sorted, tm_cdf, color="seagreen", lw=2)
        axs[2, 0].set_xlabel("TM Score")
        axs[2, 0].set_ylabel("Cumulative Probability")
        axs[2, 0].set_title("CDF of TM Scores")
        axs[2, 0].grid(True, alpha=0.3)

        dockq_sorted = np.sort(dockq_scores)
        dockq_cdf = np.arange(1, len(dockq_sorted) + 1) / len(dockq_sorted)
        axs[2, 1].plot(dockq_sorted, dockq_cdf, color="dodgerblue", lw=2)
        axs[2, 1].set_xlabel("DockQ Score")
        axs[2, 1].set_ylabel("Cumulative Probability")
        axs[2, 1].set_title("CDF of DockQ Scores")
        axs[2, 1].grid(True, alpha=0.3)

        # Row 4 — iPTM histogram & CDF
        axs[3, 0].hist(
            iptm_scores,
            bins=40,
            color="orange",
            edgecolor="black",
            alpha=0.7,
            weights=np.ones_like(iptm_scores) / len(iptm_scores),
        )
        axs[3, 0].set_xlabel("iPTM Product (normalized)")
        axs[3, 0].set_ylabel("Probability")
        axs[3, 0].set_title("Histogram of iPTM Scores (Probability)")
        axs[3, 0].set_ylim(0, 1)

        iptm_sorted = np.sort(iptm_scores)
        iptm_cdf = np.arange(1, len(iptm_sorted) + 1) / len(iptm_sorted)
        axs[3, 1].plot(iptm_sorted, iptm_cdf, color="orange", lw=2)
        axs[3, 1].set_xlabel("iPTM Product")
        axs[3, 1].set_ylabel("Cumulative Probability")
        axs[3, 1].set_title("CDF of iPTM Scores")
        axs[3, 1].grid(True, alpha=0.3)

        # Adjust spacing and save
        plt.tight_layout()
        save_path = Path(self.output_dir) / f"{comparision_type}_summary_8plots.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    def plot_permutation_score_boxplots(
        self,
        comparision_type: str,
    ) -> None:
        """Create one consolidated 3×2 grid of TM & DockQ box-plots coloured by iPTM.

        Layout (rows = chain count):
            Row 1 → TM (n=2) | DockQ (n=2)
            Row 2 → TM (n=3) | DockQ (n=3)
            Row 3 → TM (n=4) | DockQ (n=4)

        The figure is saved to ``<output_dir>/<comparision_type>_score_boxplots_grid.png``.
        """

        assert comparision_type in [
            "seeds",
            "permutations",
        ], f"Invalid comparision type: {comparision_type}. Must be one of 'seeds', 'permutations'."

        # Mapping from score label to the index inside the tuple returned by
        # *compare_permutations*
        score_map = {"tm": 1, "dockq": 0}  # left col uses tm, right col dockq

        # Load the comparison JSON
        with open(
            Path(self.output_dir) / f"{comparision_type}_comparisons.json", "r"
        ) as f:
            comparison_results = json.load(f)

        # Prepare a fixed grid for n_chains = 2,3,4
        chain_order = [2, 3, 4]
        fig, axs = plt.subplots(3, 2, figsize=(16, 18))

        for row_idx, n_chains in enumerate(chain_order):
            proteins = (
                comparison_results.get(str(n_chains), {})
                if isinstance(list(comparison_results.keys())[0], str)
                else comparison_results.get(n_chains, {})
            )

            # Collect per-protein data once, reuse for both TM and DockQ axes
            protein_ids: list[str] = []
            values_by_score: dict[str, list[list[float]]] = {"tm": [], "dockq": []}
            iptm_by_protein: list[list[float]] = []

            for protein_id, pair_dict in proteins.items():
                tm_vals: list[float] = []
                dockq_vals: list[float] = []
                iptms: list[float] = []
                for triplet in pair_dict.values():
                    if not isinstance(triplet, (list, tuple)) or len(triplet) != 3:
                        continue
                    dockq_val, tm_val, iptm_val = triplet
                    if None in (dockq_val, tm_val, iptm_val):
                        continue
                    tm_vals.append(tm_val)
                    dockq_vals.append(dockq_val)
                    iptms.append(iptm_val)

                if tm_vals:  # Only if data collected
                    protein_ids.append(protein_id)
                    values_by_score["tm"].append(tm_vals)
                    values_by_score["dockq"].append(dockq_vals)
                    iptm_by_protein.append(iptms)

            # Iterate over the two columns (0 → TM  | 1 → DockQ)
            for col_idx, score_label in enumerate(["tm", "dockq"]):
                ax = axs[row_idx, col_idx]

                if not protein_ids:
                    ax.axis("off")
                    continue

                # Box plot
                ax.boxplot(values_by_score[score_label], patch_artist=True)

                # Scatter overlay with iPTM colouring (use same iptm list for either TM/DockQ)
                scatter_handle = None
                for prot_idx, score_list in enumerate(values_by_score[score_label]):
                    iptm_list = iptm_by_protein[prot_idx]
                    x_center = prot_idx + 1
                    x_jitter = np.random.normal(x_center, 0.06, len(score_list))
                    scatter_handle = ax.scatter(
                        x_jitter,
                        score_list,
                        c=iptm_list,
                        cmap="viridis",
                        edgecolors="black",
                        alpha=0.8,
                        s=30,
                        zorder=10,
                    )

                ax.set_xlabel("Protein ID")
                ax.set_ylabel("TM Score" if score_label == "tm" else "DockQ Score")
                ax.set_title(f"{score_label.upper()} Score (n={n_chains})")
                ax.set_xticks(np.arange(1, len(protein_ids) + 1))
                ax.set_xticklabels(protein_ids, rotation=45, ha="right")
                ax.set_ylim(0, 1.05)
                ax.grid(axis="y", alpha=0.3)

                # Colour-bar only once per axis
                if scatter_handle is not None:
                    cbar = fig.colorbar(scatter_handle, ax=ax)
                    cbar.set_label("iPTM Product")

        plt.tight_layout()
        out_path = Path(self.output_dir) / f"{comparision_type}_score_boxplots_grid.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
