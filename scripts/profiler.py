import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import re
from ecstasy import utils


def parse_benchmark_data(benchmark_file: str, pdb_dir: str = None) -> dict:
    """
    Parse benchmark.json and organize data by protein and submodule.
    
    Args:
        benchmark_file: Path to benchmark.json file
        pdb_dir: Directory containing PDB files to calculate total chain length
        
    Returns:
        dict: Organized data {protein_id: {submodule: {total_chain_length, time_data, memory_data}}}
    """
    with open(benchmark_file, 'r') as f:
        data = json.load(f)
    
    # Remove the global 'after_load_model' entry
    if 'after_load_model' in data:
        del data['after_load_model']
    
    organized_data = {}
    
    for key, value in data.items():
        # Parse key format: n{chain_count}_{protein_id}_p{permutation}
        match = re.match(r'n(\d+)_([A-Z0-9]+)_p(\d+)', key)
        if match:
            chain_count = int(match.group(1))
            protein_id = match.group(2)
            permutation = int(match.group(3))
            
            if protein_id not in organized_data:
                # Calculate total chain length if PDB directory is provided
                total_chain_length = None
                if pdb_dir:
                    try:
                        # Find the first PDB file for this protein to get total length
                        pdb_path = Path(pdb_dir)
                        pdb_files = list(pdb_path.glob(f"*{protein_id}*.pdb"))
                        if pdb_files:
                            structure = utils.load_structure(str(pdb_files[0]))
                            protein_sequences = utils.get_sequence(structure)
                            total_chain_length = sum(len(seq) for seq in protein_sequences.values())
                    except Exception as e:
                        print(f"Warning: Could not calculate total chain length for {protein_id}: {e}")
                        total_chain_length = None
                
                organized_data[protein_id] = {
                    'chain_count': chain_count,
                    'total_chain_length': total_chain_length,
                    'submodules': defaultdict(list)
                }
            
            # Organize data by submodule
            for submodule, metrics in value.items():
                organized_data[protein_id]['submodules'][submodule].append({
                    'time': metrics['time'],
                    'current_memory': metrics['current memory'],
                    'peak_memory': metrics['peak memory'],
                    'permutation': permutation
                })
    
    return organized_data


def calculate_averages(organized_data: dict) -> dict:
    """
    Calculate average time and memory across permutations for each protein and submodule.
    
    Args:
        organized_data: Organized benchmark data
        
    Returns:
        dict: Averaged data {submodule: {total_chain_lengths, avg_times, avg_current_memory, avg_peak_memory}}
    """
    submodule_stats = defaultdict(lambda: {
        'total_chain_lengths': [],
        'avg_times': [],
        'avg_current_memory': [],
        'avg_peak_memory': [],
        'std_times': [],
        'std_current_memory': [],
        'std_peak_memory': []
    })
    
    for protein_id, protein_data in organized_data.items():
        total_chain_length = protein_data['total_chain_length']
        
        # Skip if we couldn't calculate total chain length
        if total_chain_length is None:
            print(f"Warning: Skipping {protein_id} - no total chain length available")
            continue
        
        for submodule, measurements in protein_data['submodules'].items():
            # Calculate averages across permutations
            times = [m['time'] for m in measurements]
            current_memory = [m['current_memory'] for m in measurements]
            peak_memory = [m['peak_memory'] for m in measurements]
            
            submodule_stats[submodule]['total_chain_lengths'].append(total_chain_length)
            submodule_stats[submodule]['avg_times'].append(np.mean(times))
            submodule_stats[submodule]['avg_current_memory'].append(np.mean(current_memory))
            submodule_stats[submodule]['avg_peak_memory'].append(np.mean(peak_memory))
            submodule_stats[submodule]['std_times'].append(np.std(times))
            submodule_stats[submodule]['std_current_memory'].append(np.std(current_memory))
            submodule_stats[submodule]['std_peak_memory'].append(np.std(peak_memory))
    
    return dict(submodule_stats)


def plot_profiling_results(submodule_stats: dict, output_dir: str = None):
    """
    Create plots for time and memory profiling across submodules.
    
    Args:
        submodule_stats: Statistics for each submodule
        output_dir: Directory to save plots
    """
    if not submodule_stats:
        print("No data to plot")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create time profiling plot
    plt.figure(figsize=(12, 8))
    
    for submodule, stats in submodule_stats.items():
        total_chain_lengths = np.array(stats['total_chain_lengths'])
        avg_times = np.array(stats['avg_times'])
        std_times = np.array(stats['std_times'])
        
        # Sort by total chain length for proper plotting
        sort_idx = np.argsort(total_chain_lengths)
        total_chain_lengths_sorted = total_chain_lengths[sort_idx]
        avg_times_sorted = avg_times[sort_idx]
        std_times_sorted = std_times[sort_idx]
        
        plt.errorbar(total_chain_lengths_sorted, avg_times_sorted, yerr=std_times_sorted, 
                    marker='o', label=submodule, capsize=5, capthick=2, linewidth=2, markersize=8)
    
    plt.xlabel('Total Chain Length (amino acids)', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.title('ESMFold Submodule Time Profiling', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / 'esmfold_time_profiling.png', dpi=300, bbox_inches='tight')
        print(f"Time profiling plot saved to {output_path / 'esmfold_time_profiling.png'}")
    
    plt.show()
    
    # Create current memory profiling plot
    plt.figure(figsize=(12, 8))
    
    for submodule, stats in submodule_stats.items():
        total_chain_lengths = np.array(stats['total_chain_lengths'])
        avg_current_memory = np.array(stats['avg_current_memory'])
        std_current_memory = np.array(stats['std_current_memory'])
        
        # Sort by total chain length for proper plotting
        sort_idx = np.argsort(total_chain_lengths)
        total_chain_lengths_sorted = total_chain_lengths[sort_idx]
        avg_current_memory_sorted = avg_current_memory[sort_idx]
        std_current_memory_sorted = std_current_memory[sort_idx]
        
        plt.errorbar(total_chain_lengths_sorted, avg_current_memory_sorted, yerr=std_current_memory_sorted, 
                    marker='s', label=submodule, capsize=5, capthick=2, linewidth=2, markersize=8)
    
    plt.xlabel('Total Chain Length (amino acids)', fontsize=14)
    plt.ylabel('Current Memory (GB)', fontsize=14)
    plt.title('ESMFold Submodule Current Memory Profiling', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_path / 'esmfold_current_memory_profiling.png', dpi=300, bbox_inches='tight')
        print(f"Current memory profiling plot saved to {output_path / 'esmfold_current_memory_profiling.png'}")
    
    plt.show()
    
    # Create peak memory profiling plot
    plt.figure(figsize=(12, 8))
    
    for submodule, stats in submodule_stats.items():
        total_chain_lengths = np.array(stats['total_chain_lengths'])
        avg_peak_memory = np.array(stats['avg_peak_memory'])
        std_peak_memory = np.array(stats['std_peak_memory'])
        
        # Sort by total chain length for proper plotting
        sort_idx = np.argsort(total_chain_lengths)
        total_chain_lengths_sorted = total_chain_lengths[sort_idx]
        avg_peak_memory_sorted = avg_peak_memory[sort_idx]
        std_peak_memory_sorted = std_peak_memory[sort_idx]
        
        plt.errorbar(total_chain_lengths_sorted, avg_peak_memory_sorted, yerr=std_peak_memory_sorted, 
                    marker='^', label=submodule, capsize=5, capthick=2, linewidth=2, markersize=8)
    
    plt.xlabel('Total Chain Length (amino acids)', fontsize=14)
    plt.ylabel('Peak Memory (GB)', fontsize=14)
    plt.title('ESMFold Submodule Peak Memory Profiling', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_path / 'esmfold_peak_memory_profiling.png', dpi=300, bbox_inches='tight')
        print(f"Peak memory profiling plot saved to {output_path / 'esmfold_peak_memory_profiling.png'}")
    
    plt.show()


def print_summary_statistics(submodule_stats: dict):
    """
    Print summary statistics for each submodule.
    
    Args:
        submodule_stats: Statistics for each submodule
    """
    print("\n" + "="*80)
    print("ESMFOLD PROFILING SUMMARY")
    print("="*80)
    
    for submodule, stats in submodule_stats.items():
        print(f"\n{submodule.upper()}:")
        print("-" * len(submodule) + "-")
        
        # Group by total chain length
        total_chain_lengths = np.array(stats['total_chain_lengths'])
        avg_times = np.array(stats['avg_times'])
        avg_current_memory = np.array(stats['avg_current_memory'])
        avg_peak_memory = np.array(stats['avg_peak_memory'])
        
        unique_total_lengths = sorted(set(total_chain_lengths))
        
        for total_len in unique_total_lengths:
            mask = total_chain_lengths == total_len
            times_for_length = avg_times[mask]
            current_mem_for_length = avg_current_memory[mask]
            peak_mem_for_length = avg_peak_memory[mask]
            
            print(f"  {total_len} total amino acids:")
            print(f"    Avg Time: {np.mean(times_for_length):.3f} ± {np.std(times_for_length):.3f} seconds")
            print(f"    Avg Current Memory: {np.mean(current_mem_for_length):.3f} ± {np.std(current_mem_for_length):.3f} GB")
            print(f"    Avg Peak Memory: {np.mean(peak_mem_for_length):.3f} ± {np.std(peak_mem_for_length):.3f} GB")


def main(benchmark_file: str, pdb_dir: str = None, output_dir: str = None):
    """
    Main function to analyze ESMFold profiling data.
    
    Args:
        benchmark_file: Path to benchmark.json file
        pdb_dir: Directory containing PDB files to calculate total chain length
        output_dir: Directory to save plots (optional)
    """
    print(f"Analyzing benchmark data from: {benchmark_file}")
    if pdb_dir:
        print(f"Using PDB directory: {pdb_dir}")
    
    # Parse and organize data
    organized_data = parse_benchmark_data(benchmark_file, pdb_dir)
    
    print(f"\nFound data for {len(organized_data)} proteins:")
    for protein_id, data in organized_data.items():
        chain_count = data['chain_count']
        total_length = data['total_chain_length']
        if total_length:
            print(f"  {protein_id}: {chain_count} chains, {total_length} total amino acids, {len(data['submodules'])} submodules")
        else:
            print(f"  {protein_id}: {chain_count} chains, total length unknown, {len(data['submodules'])} submodules")
    
    # Calculate averages
    submodule_stats = calculate_averages(organized_data)
    
    print(f"\nSubmodules found: {list(submodule_stats.keys())}")
    
    # Print summary statistics
    print_summary_statistics(submodule_stats)
    
    # Create plots
    print("\nGenerating plots...")
    plot_profiling_results(submodule_stats, output_dir)
    
    # Save statistics to JSON
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        json_stats = {}
        for submodule, stats in submodule_stats.items():
            json_stats[submodule] = {
                'total_chain_lengths': [int(x) for x in stats['total_chain_lengths']],
                'avg_times': [float(x) for x in stats['avg_times']],
                'avg_current_memory': [float(x) for x in stats['avg_current_memory']],
                'avg_peak_memory': [float(x) for x in stats['avg_peak_memory']],
                'std_times': [float(x) for x in stats['std_times']],
                'std_current_memory': [float(x) for x in stats['std_current_memory']],
                'std_peak_memory': [float(x) for x in stats['std_peak_memory']]
            }
        
        with open(output_path / 'profiling_statistics.json', 'w') as f:
            json.dump(json_stats, f, indent=2)
        print(f"Statistics saved to {output_path / 'profiling_statistics.json'}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
