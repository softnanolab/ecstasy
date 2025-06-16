import biotite.structure as structure
import biotite.structure.io as io
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdbx as pdbx
import os
from pathlib import Path

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

