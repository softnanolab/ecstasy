import biotite.structure as structure
import biotite.structure.io as io
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdbx as pdbx
import os
from pathlib import Path
import numpy as np
from biotite.sequence import ProteinSequence
from typing import List, Dict

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

def tm_score(ref_structure: structure.AtomArray, sub_structure: structure.AtomArray) -> float:
    """
    Calculate the TM-score between two structures.
    Args:
        ref_structure (AtomArray): The reference structure.
        sub_structure (AtomArray): The structure to superimpose onto the reference.
    
    Returns:
        float: The TM-score between the two structures.
    """
    superimposed, _, ref_indices, sub_indices = structure.superimpose_structural_homologs(
        ref_structure, sub_structure, max_iterations=1
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

def write_fasta_esmfold(sequences: Dict[str, Dict[str, str]], output_dir: str, file_name: str = "batch_esmfold.fasta"):
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
    """
    """
    Combine all FASTA files in a directory into a single FASTA file.
    
    Args:
        input_dir (str): The directory containing FASTA files to combine.
        combined_name (str): The name of the combined FASTA file.
    
    Returns:
        None
    """
    
    # Get all FASTA files in the directory
    fasta_files = [f for f in os.listdir(input_dir) if f.endswith('.fasta')]
    
    if not fasta_files:
        print(f"No FASTA files found in {input_dir}")
        return
    
    # Read and combine all FASTA files
    combined_content = []
    for fasta_file in fasta_files:
        file_path = os.path.join(input_dir, fasta_file)
        with open(file_path, 'r') as f:
            content = f.read().strip()
            if content:  # Only add non-empty content
                combined_content.append(content)
    
    # Write combined content to output file
    output_path = os.path.join(input_dir, combined_name)
    with open(output_path, 'w') as f:
        f.write('\n'.join(combined_content))
    
    print(f"Combined {len(fasta_files)} FASTA files into {output_path}")
    
def write_fasta_boltz(sequences: Dict[str, Dict[str, str]], output_dir: str, use_msas=None):
    """
    Write a dictionary of sequences to a FASTA file.
    
    Args:
        sequences (Dict[str, Dict[str, str]]): A dictionary of protein names and their sequences.
        output_dir (str): The directory to write the FASTA file to.
        use_msas (bool): Whether to use MSAs.
    
    Returns:
        None
    """
    assert use_msas is None, "MSAs are not supported for Bolt at the moment"
    os.makedirs(output_dir, exist_ok=True)
    for protein_name, protein_sequences in sequences.items():
        to_write = [
            f">{_chain_id}|protein|empty\n{_seq}"
            for _chain_id, _seq in protein_sequences.items()
        ]
        
        with open(os.path.join(output_dir, f"{protein_name}.fasta"), "w") as f:
           f.write("\n".join(to_write))