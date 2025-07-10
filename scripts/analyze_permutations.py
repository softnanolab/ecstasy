import os
import json
import itertools
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from ecstasy import utils
from ecstasy.utils import generate_tm_confusion_matrix


def organize_pdb_files_by_protein(pdb_dir: str) -> dict:
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


def organize_boltz_predictions_by_protein(predictions_dir: str) -> dict:
    """
    Organize Boltz prediction files hierarchically by protein ID and chain count.

    Args:
        predictions_dir: Directory containing Boltz prediction subdirectories

    Returns:
        dict: Hierarchical organization {n_chains: {protein_id: [file_paths]}}
    """
    predictions_path = Path(predictions_dir)
    organized_files = defaultdict(lambda: defaultdict(list))

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

                # Find the CIF file inside the directory (model_0.cif)
                cif_files = list(pred_dir.glob("*model_0.cif"))
                if cif_files:
                    organized_files[n_chains][protein_id].append(str(cif_files[0]))
                else:
                    print(f"No model_0.cif found in directory: {dirname}")

            except ValueError:
                print(f"Could not parse directory name: {dirname}")
                continue

    # Sort file paths for each protein
    for n_chains in organized_files:
        for protein_id in organized_files[n_chains]:
            organized_files[n_chains][protein_id].sort()

    return dict(organized_files)


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
        organized_files = organize_pdb_files_by_protein(pdb_dir)
    elif model_name == "boltz":
        organized_files = organize_boltz_predictions_by_protein(pdb_dir)

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
    import fire
    fire.Fire(main)
