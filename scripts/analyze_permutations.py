#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tmtools import tm_align
import fire
from typing import Dict
from collections import defaultdict
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1
import numpy as np

class PermutationAnalyzer:
    def __init__(self, predictions_dir: str = "predictions"):
        self.predictions_dir = Path(predictions_dir)
        self.results_dir = self.predictions_dir / "analysis"
        self.results_dir.mkdir(exist_ok=True)
        
        # Store TM-scores for each PDB ID
        self.tm_scores: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
        
        # Initialize PDB parser
        self.parser = MMCIFParser(QUIET=True)

    def get_structure_data(self, cif_file: str) -> tuple:
        """Extract coordinates and sequence from a CIF file."""
        try:
            structure = self.parser.get_structure('', cif_file)
            # Get all CA atoms from standard amino acids
            ca_atoms = []
            sequence = ""
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue and is_aa(residue, standard=True):
                            ca_atoms.append(residue['CA'].get_coord())
                            try:
                                aa = seq1(residue.get_resname(), custom_map={"MSE": "M"})
                            except Exception:
                                aa = "X"  # Unknown amino acid
                            sequence += aa
            
            return np.array(ca_atoms), sequence
        except Exception as e:
            print(f"Error parsing {cif_file}: {e}")
            return None, None

    def calculate_tm_score(self, cif1: str, cif2: str) -> float:
        """Calculate TM-score between two CIF structures using tmtools."""
        try:
            # Get coordinates and sequences
            coords1, seq1 = self.get_structure_data(cif1)
            coords2, seq2 = self.get_structure_data(cif2)
            
            if coords1 is None or coords2 is None:
                return 0.0
                
            # Calculate TM-score
            result = tm_align(coords1, coords2, seq1, seq2)
            return result.tm_norm_chain1  # Normalized TM-score
        except Exception as e:
            print(f"Error calculating TM-score between {cif1} and {cif2}: {e}")
            return 0.0

    def analyze_permutations(self):
        """Analyze permutations for each unique PDB ID."""
        # Group folders by PDB ID
        pdb_groups = defaultdict(list)
        for folder in self.predictions_dir.glob("n*_*_p*"):
            if not folder.is_dir():
                continue
            
            # Extract PDB ID from folder name (e.g., n4_6C09_p1 -> 6C09)
            parts = folder.name.split('_')
            if len(parts) != 3:
                continue
            pdb_id = parts[1]
            pdb_groups[pdb_id].append(folder)

        # For each PDB ID, compare all permutations
        for pdb_id, folders in pdb_groups.items():
            print(f"\nAnalyzing permutations for PDB ID: {pdb_id}")
            
            # Get all CIF files for this PDB ID
            cif_files = {}
            for folder in folders:
                cif_file = next(folder.glob("*.cif"), None)
                if cif_file:
                    cif_files[folder.name] = str(cif_file)

            # Calculate TM-scores between all pairs
            for perm1, file1 in cif_files.items():
                self.tm_scores[pdb_id][perm1] = {}
                for perm2, file2 in cif_files.items():
                    if perm1 != perm2:
                        tm_score = self.calculate_tm_score(file1, file2)
                        self.tm_scores[pdb_id][perm1][perm2] = tm_score
                        print(f"TM-score between {perm1} and {perm2}: {tm_score:.3f}")

    def generate_heatmaps(self):
        """Generate heatmaps of TM-scores for each PDB ID."""
        for pdb_id, scores in self.tm_scores.items():
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(scores, orient='index')
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(df, annot=True, cmap='YlOrRd', vmin=0, vmax=1, fmt='.3f')
            plt.title(f'TM-scores between Permutations for {pdb_id}')
            plt.tight_layout()
            plt.savefig(self.results_dir / f'tm_score_heatmap_{pdb_id}.png')
            plt.close()
            print(f"Generated heatmap for {pdb_id}")

    def run_analysis(self):
        """Run the complete analysis pipeline."""
        # Analyze permutations
        self.analyze_permutations()
        
        # Generate visualizations
        self.generate_heatmaps()

def main(predictions_dir: str = "predictions"):
    """Run the permutation analysis.
    
    Args:
        predictions_dir: Path to the directory containing the predictions
    """
    analyzer = PermutationAnalyzer(predictions_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    fire.Fire(main) 
