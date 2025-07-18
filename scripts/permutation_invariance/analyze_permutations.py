import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from ecstasy import utils
from ecstasy.utils import generate_tm_confusion_matrix

import warnings

# ignore this warning from biotite
warnings.filterwarnings(
    "ignore",
    message="Attribute 'auth_atom_id' not found within 'atom_site' category. The fallback attribute 'label_atom_id' will be used instead",
)


def organize_esmfold_predictions(pdb_dir: str) -> dict:
    """
    Organize PDB files hierarchically by protein ID and chain count.

    Args:
        pdb_dir: Directory containing PDB files

    Returns:
        dict: Hierarchical organization {n_chains: {protein_id: [file_paths]}}
    """
    pdb_path = Path(pdb_dir)
    organized_files = defaultdict(lambda: defaultdict(list))

    # Find all PDB files
    pdb_files = list(pdb_path.glob("*.pdb"))

    for pdb_file in pdb_files:
        # Parse filename to extract n_chains and protein_id
        # Expected format: n{num_chains}_{protein_id}_p{permutation}.pdb
        filename = pdb_file.stem
        parts = filename.split('_')

        if len(parts) >= 3 and parts[0].startswith('n'):
            try:
                n_chains = int(parts[0][1:])  # Extract number after 'n'
                protein_id = parts[1]

                organized_files[n_chains][protein_id].append(str(pdb_file))
            except ValueError:
                print(f"Could not parse filename: {filename}")
                continue

    # Sort file paths for each protein
    for n_chains in organized_files:
        for protein_id in organized_files[n_chains]:
            organized_files[n_chains][protein_id].sort()

    return dict(organized_files)


def organize_boltz_predictions(predictions_dir: str) -> dict:
    """
    Organize Boltz prediction files hierarchically by protein ID and chain count.

    Args:
        predictions_dir: Directory containing Boltz prediction subdirectories

    Returns:
        dict: Hierarchical organization {n_chains: {protein_id: {permutation: [file_paths]}}}
    """
    predictions_path = Path(predictions_dir)
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


def process_permutation_invariance(
    organized_files: dict, output_dir: str, n_cpus: int = 128
):
    """
    Process permutation invariance for a given set of files.
    """
    pass


def process_seeds_for_a_single_permutation(
    path_to_model_seeds: list[str], num_cpus: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """
    Process seeds for a single prediction.

    Args:
        path_to_model_seeds (list[str]): List of paths to model seeds
        num_cpus (int): Number of CPUs to use

    Returns:
        tuple[np.ndarray, np.ndarray]: TM scores and DockQ scores
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


def process_seeds_for_all_permutations(
    organized_files: dict,
    output_dir: str,
    n_cpus: int = 128,
):
    # generate a list of comparision jobs for multiprocessing

    jobs = []
    results = {}
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
                pool.imap(_wrapper_do_monomer_and_multimer_comparision, jobs),
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

    with open(output_dir, "w") as f:
        json.dump(results, f, indent=4)

    return results


def _wrapper_do_monomer_and_multimer_comparision(job: tuple):
    i, j, file_i_path, file_j_path, n_chain, protein_id, permutation_number = job
    dockq, tm, iptm = do_monomer_and_multimer_comparision((file_i_path, file_j_path))
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


def do_monomer_and_multimer_comparision(
    file_path_pair: tuple[str, str],
) -> tuple[float, float, float]:
    """
    Do monomer and multimer comparision for a given pair of files.

    Args:
        file_i_path (str): Path to the first file
        file_j_path (str): Path to the second file
    Returns:
        tuple[float, float]: DockQ score, TM score
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
    with open(utils.get_confidence_file_path_boltz(file_i_path), "r") as f:
        file_i_iptm = json.load(f)["protein_iptm"]

    with open(utils.get_confidence_file_path_boltz(file_j_path), "r") as f:
        file_j_iptm = json.load(f)["protein_iptm"]

    # return dockq score, tm score, and the product of the two iptm scores
    return dockq_score, tm_score, file_i_iptm * file_j_iptm


def calculate_tm_statistics(organized_files: dict) -> dict:
    """
    Calculate mean and std TM scores for each protein and chain count.
    
    Args:
        organized_files: Hierarchical organization of PDB files
        
    Returns:
        dict: Statistics for each chain count and protein
    """
    statistics = {}
    
    for n_chains, proteins in organized_files.items():
        print(f"Processing {n_chains}-chain proteins...")
        
        chain_statistics = {}
        
        for protein_id, file_paths in proteins.items():
            print(f"  Processing protein {protein_id} with {len(file_paths)} permutations")
            
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
                    'mean': np.mean(upper_triangle),
                    'std': np.std(upper_triangle),
                    'count': len(upper_triangle),
                    'scores': upper_triangle,
                    'total_length': total_length
                }
                print(f"    {protein_id}: mean={chain_statistics[protein_id]['mean']:.3f}, "
                      f"std={chain_statistics[protein_id]['std']:.3f}, n={chain_statistics[protein_id]['count']}, "
                      f"total_length={total_length}")
        
        statistics[n_chains] = chain_statistics
    
    return statistics


def plot_tm_statistics(statistics: dict, output_dir: str = None):
    """
    Plot mean and standard deviation TM scores for each protein, separated by chain count.
    
    Args:
        statistics: TM score statistics for each chain count and protein
        output_dir: Directory to save plots
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
        means = [protein_stats[pid]['mean'] for pid in protein_ids]
        stds = [protein_stats[pid]['std'] for pid in protein_ids]
        total_lengths = [protein_stats[pid]['total_length'] for pid in protein_ids]

        # Create bar plot with error bars
        x_pos = np.arange(len(protein_ids))
        bars = plt.bar(x_pos, means, yerr=stds, capsize=5, 
                       alpha=0.7, color='skyblue', edgecolor='navy')

        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=9)

        plt.xlabel('Protein ID')
        plt.ylabel('TM Score')
        plt.title(f'TM Scores for {n_chains}-Chain Proteins')

        # Create labels with protein ID, total length, and sample size
        labels = [f'{pid}\n({total_lengths[i]} aa, n={protein_stats[pid]["count"]})' 
                 for i, pid in enumerate(protein_ids)]
        plt.xticks(x_pos, labels, rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 1.1)

        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path / f'tm_scores_{n_chains}_chains.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_path / f'tm_scores_{n_chains}_chains.png'}")

        plt.show()

        # Also create a box plot for each chain count
        plt.figure(figsize=(14, 6))

        # Prepare data for box plot
        box_data = [protein_stats[pid]['scores'] for pid in protein_ids]
        labels = [f'{pid}\n({total_lengths[i]} aa, n={len(scores)})' 
                 for i, (pid, scores) in enumerate(zip(protein_ids, box_data))]

        # Create box plot
        bp = plt.boxplot(box_data, tick_labels=labels)

        # Add individual data points on top
        for i, scores in enumerate(box_data):
            # Add jitter to x-coordinates to spread out points
            x_jittered = np.random.normal(i+1, 0.04, len(scores))
            plt.scatter(x_jittered, scores, alpha=0.6, s=20, color='red', zorder=10)

        plt.xlabel('Protein ID')
        plt.ylabel('TM Score')
        plt.title(f'TM Score Distribution for {n_chains}-Chain Proteins')
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)

        plt.tight_layout()

        if output_dir:
            plt.savefig(output_path / f'tm_scores_distribution_{n_chains}_chains.png', dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to {output_path / f'tm_scores_distribution_{n_chains}_chains.png'}")

        plt.show()


def plot_iptm_vs_scores_heatmaps(json_path: str, output_dir: str = None):
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    dockq_scores = []
    tm_scores = []
    iptm_scores = []

    # Traverse the hierarchy and collect scores
    for n_chains in data:
        for pdb_id in data[n_chains]:
            for perm in data[n_chains][pdb_id]:
                for pair in data[n_chains][pdb_id][perm]:
                    scores = data[n_chains][pdb_id][perm][pair]
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
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(output_dir) / "iptm_vs_dockq_heatmap.png", dpi=300)
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
    if output_dir:
        plt.savefig(Path(output_dir) / "iptm_vs_tm_heatmap.png", dpi=300)
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
    if output_dir:
        plt.savefig(Path(output_dir) / "dockq_score_histogram.png", dpi=300)
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
    if output_dir:
        plt.savefig(Path(output_dir) / "tm_score_histogram.png", dpi=300)
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
    if output_dir:
        plt.savefig(Path(output_dir) / "iptm_score_histogram.png", dpi=300)
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
    if output_dir:
        plt.savefig(Path(output_dir) / "tm_score_cdf.png", dpi=300)
    plt.show()

    # CDF: DockQ Score
    dockq_scores_sorted = np.sort(dockq_scores)
    dockq_cdf = np.arange(1, len(dockq_scores_sorted) + 1) / len(dockq_scores_sorted)
    plt.figure(figsize=(7, 4))
    plt.plot(dockq_scores_sorted, dockq_cdf, color="dodgerblue", lw=2)
    plt.xlabel("DockQ Score")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distribution Function of DockQ Scores")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_dir:
        plt.savefig(Path(output_dir) / "dockq_score_cdf.png", dpi=300)
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
    if output_dir:
        plt.savefig(Path(output_dir) / "iptm_score_cdf.png", dpi=300)
    plt.show()


def main(pdb_dir: str, output_dir: str, model_name: str):
    """
    Main function to analyze permutation results.

    Args:
        pdb_dir: Directory containing PDB files
        output_dir: Directory to save plots (optional)
        model_name: Name of the model to analyze
    """
    assert model_name in [
        "esmfold",
        "boltz",
    ], "Model name must be either 'esmfold' or 'boltz'"
    print(f"Analyzing PDB files in: {pdb_dir}")

    # Organize files hierarchically
    if model_name == "esmfold":
        organized_files = organize_esmfold_predictions(pdb_dir)
    elif model_name == "boltz":
        organized_files = organize_boltz_predictions(pdb_dir)

    print("\nFile organization:")
    for n_chains, proteins in organized_files.items():
        print(f"{n_chains}-chain proteins: {len(proteins)} proteins")
        for protein_id, file_paths in proteins.items():
            print(f"{protein_id}: {len(file_paths)} permutations")

    # Calculate TM statistics
    print("\nCalculating TM statistics...")
    statistics = calculate_tm_statistics(organized_files)

    # Plot results
    print("\nGenerating plots...")
    plot_tm_statistics(statistics, output_dir)

    # Save statistics to JSON
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to native Python types for JSON serialization
        json_stats = {}
        for n_chains, protein_stats in statistics.items():
            json_stats[n_chains] = {}
            for protein_id, stats in protein_stats.items():
                json_stats[n_chains][protein_id] = {
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'count': int(stats['count']),
                    'total_length': int(stats['total_length'])
                }

        with open(output_path / 'tm_statistics.json', 'w') as f:
            json.dump(json_stats, f, indent=2)
        print(f"Statistics saved to {output_path / 'tm_statistics.json'}")


if __name__ == "__main__":
    # import fire
    # fire.Fire(main)
    organized_boltz = organize_boltz_predictions(
        "/home/jovyan/workspace/ecstasy/predictions/permutation_test_2/outputs/boltz_no_msa/predictions"
    )

    process_seeds_for_all_permutations(
        organized_boltz,
        output_dir="/home/jovyan/workspace/ecstasy/predictions/permutation_test_2/benchmarks/boltz_no_msas_seeds_comparision.json",
        n_cpus=156,
    )
