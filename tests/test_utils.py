from pathlib import Path
import biotite.structure as structure
from ecstasy.utils import load_structure, tm_score, get_sequence

# Get the path to the test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"

def test_load_structure_from_pdb_id():
    # Test loading structure from PDB ID
    atom_array = load_structure("1ubq", hetero=False)
    assert atom_array is not None
    assert isinstance(atom_array, structure.AtomArray)
    assert atom_array.array_length() > 0
    assert not any(atom_array.hetero)  # Check no hetero atoms by default

def test_load_structure_from_file():
    # Test loading structure from local file
    test_file = TEST_DATA_DIR / "1UBQ.cif"
    atom_array = load_structure(str(test_file), hetero=False)
    assert atom_array is not None
    assert isinstance(atom_array, structure.AtomArray)
    assert atom_array.array_length() > 0
    assert not any(atom_array.hetero)  # Check no hetero atoms by default

def test_load_structure_with_hetero():
    # Test loading structure with hetero atoms
    atom_array = load_structure("1ubq", hetero=True)
    assert atom_array is not None
    assert isinstance(atom_array, structure.AtomArray)
    assert atom_array.array_length() > 0
    assert any(atom_array.hetero)  # Should include hetero atoms

def test_load_structure_invalid_pdb():
    # Test loading invalid PDB ID
    atom_array = load_structure("xxxx")
    assert atom_array is None

def test_load_structure_nonexistent_file():
    # Test loading non-existent file
    atom_array = load_structure("nonexistent.cif")
    assert atom_array is None

def test_tm_score():
    # Load the same structure twice for perfect alignment
    ref = load_structure("1ubq")
    sub = load_structure("1ubq")
    assert ref is not None and sub is not None
    
    score = tm_score(ref, sub)
    assert isinstance(score, float)
    assert score > 0.99  # Should be nearly perfect alignment with itself

def test_tm_score_different_structures():
    # Load two different structures
    # Using same structure but selecting different parts to simulate different structures
    ref = load_structure("1ubq")
    sub = load_structure("1ubq")
    assert ref is not None and sub is not None
    
    # Take first half of structure
    half_ref = ref[:len(ref)//2]
    score = tm_score(ref, half_ref)
    assert isinstance(score, float)
    assert 0 <= score <= 1  # TM-score should be between 0 and 1

def test_get_sequence():
    # Test getting sequence from structure
    atom_array = load_structure("1ubq")
    assert atom_array is not None
    
    sequences = get_sequence(atom_array)
    assert isinstance(sequences, dict)
    assert len(sequences) > 0
    
    # Ubiquitin sequence should be present
    # Get first chain's sequence (ubiquitin is single chain)
    first_chain_seq = next(iter(sequences.values()))
    assert len(first_chain_seq) > 0
    assert isinstance(first_chain_seq, str)
    # Check if it contains valid amino acid letters
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    assert all(aa in valid_aa for aa in first_chain_seq)

def test_get_sequence_multiple_chains():
    # Test with a multi-chain structure (using 1ubq but artificially split)
    atom_array = load_structure("1ubq")
    assert atom_array is not None
    
    # Artificially create multiple chains by modifying chain IDs
    half_point = len(atom_array) // 2
    atom_array.chain_id[:half_point] = "A"
    atom_array.chain_id[half_point:] = "B"
    
    sequences = get_sequence(atom_array)
    assert isinstance(sequences, dict)
    assert len(sequences) == 2  # Should have two chains
    assert "A" in sequences
    assert "B" in sequences
    assert len(sequences["A"]) > 0
    assert len(sequences["B"]) > 0 