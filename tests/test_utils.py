import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import pytest
import numpy as np
import biotite.structure as structure
from biotite.sequence import ProteinSequence

from ecstasy.utils import (
    load_structure, 
    get_sequence, 
    filter_structure_by_chain_id,
    match_chains,
    tm_score,
    generate_tm_confusion_matrix,
    compute_tm_scores_against_gd,
    plot_tm_scores_for_proteins_against_gd,
    dockq_score,
    write_fasta_esmfold,
    combine_fastas_esmfold,
    write_fasta_boltz,
    get_confidence_file_path_boltz,
    get_confidence_file_path_colabfold,
    clean_up_colabfold_predictions,
    find_max_protein_iptm_boltz
)

# Get the path to the test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"

# Fixtures for test data
@pytest.fixture
def sample_structure():
    """Create a sample structure for testing."""
    return load_structure("1ubq", hetero=False)

@pytest.fixture
def multi_chain_structure():
    """Create a multi-chain structure for testing."""
    atom_array = load_structure("1ubq")
    if atom_array is not None:
        # Artificially create multiple chains
        half_point = len(atom_array) // 2
        atom_array.chain_id[:half_point] = "A"
        atom_array.chain_id[half_point:] = "B"
    return atom_array

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing file operations."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_sequences():
    """Sample sequence data for testing."""
    return {
        "protein1": {"A": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG", "B": "MKTTLFVKGLHGDLAKVAGGVAALASLLSQPKQRGLLGRNLSLLDLVVFGRAAEGDLQVGDIVVSQPQVLGIAQ"},
        "protein2": {"A": "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS"}
    }

# Tests for structure loading and manipulation functions
class TestStructureLoading:
    def test_load_structure_from_pdb_id(self):
        """Test loading structure from PDB ID."""
        atom_array = load_structure("1ubq", hetero=False)
        assert atom_array is not None
        assert isinstance(atom_array, structure.AtomArray)
        assert atom_array.array_length() > 0
        assert not any(atom_array.hetero)

    def test_load_structure_from_file(self):
        """Test loading structure from local file."""
        test_file = TEST_DATA_DIR / "1UBQ.cif"
        atom_array = load_structure(str(test_file), hetero=False)
        assert atom_array is not None
        assert isinstance(atom_array, structure.AtomArray)
        assert atom_array.array_length() > 0
        assert not any(atom_array.hetero)

    def test_load_structure_with_hetero(self):
        """Test loading structure with hetero atoms."""
        atom_array = load_structure("1ubq", hetero=True)
        assert atom_array is not None
        assert isinstance(atom_array, structure.AtomArray)
        assert atom_array.array_length() > 0
        assert any(atom_array.hetero)

    def test_load_structure_invalid_pdb(self):
        """Test loading invalid PDB ID."""
        atom_array = load_structure("xxxx")
        assert atom_array is None

    def test_load_structure_nonexistent_file(self):
        """Test loading non-existent file."""
        atom_array = load_structure("nonexistent.cif")
        assert atom_array is None

    def test_load_structure_empty_input(self):
        """Test loading with empty input."""
        atom_array = load_structure("")
        assert atom_array is None

    @patch('biotite.structure.io.load_structure')
    def test_load_structure_exception_handling(self, mock_load):
        """Test exception handling in load_structure."""
        mock_load.side_effect = Exception("Test exception")
        atom_array = load_structure("test.cif")
        assert atom_array is None

class TestSequenceExtraction:
    def test_get_sequence(self, sample_structure):
        """Test getting sequence from structure."""
        if sample_structure is None:
            pytest.skip("Could not load sample structure")
        
        sequences = get_sequence(sample_structure)
        assert isinstance(sequences, dict)
        assert len(sequences) > 0
        
        first_chain_seq = next(iter(sequences.values()))
        assert len(first_chain_seq) > 0
        assert isinstance(first_chain_seq, str)
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        assert all(aa in valid_aa for aa in first_chain_seq)

    def test_get_sequence_multiple_chains(self, multi_chain_structure):
        """Test sequence extraction from multi-chain structure."""
        if multi_chain_structure is None:
            pytest.skip("Could not create multi-chain structure")
        
        sequences = get_sequence(multi_chain_structure)
        assert isinstance(sequences, dict)
        assert len(sequences) == 2
        assert "A" in sequences
        assert "B" in sequences
        assert len(sequences["A"]) > 0
        assert len(sequences["B"]) > 0

    def test_get_sequence_empty_structure(self):
        """Test sequence extraction from empty structure."""
        # Create empty structure
        empty_structure = structure.AtomArray(0)
        sequences = get_sequence(empty_structure)
        assert isinstance(sequences, dict)
        assert len(sequences) == 0

class TestStructureFiltering:
    def test_filter_structure_by_chain_id(self, multi_chain_structure):
        """Test filtering structure by chain ID."""
        if multi_chain_structure is None:
            pytest.skip("Could not create multi-chain structure")
        
        chain_a = filter_structure_by_chain_id(multi_chain_structure, "A")
        assert isinstance(chain_a, structure.AtomArray)
        assert len(chain_a) > 0
        assert all(chain_a.chain_id == "A")

    def test_filter_structure_nonexistent_chain(self, sample_structure):
        """Test filtering with non-existent chain ID."""
        if sample_structure is None:
            pytest.skip("Could not load sample structure")
        
        filtered = filter_structure_by_chain_id(sample_structure, "Z")
        assert isinstance(filtered, structure.AtomArray)
        assert len(filtered) == 0

class TestChainMatching:
    def test_match_chains_identical_structures(self, sample_structure):
        """Test matching chains between identical structures."""
        if sample_structure is None:
            pytest.skip("Could not load sample structure")
        
        matches = match_chains(sample_structure, sample_structure)
        assert isinstance(matches, dict)
        assert len(matches) > 0
        
        # All chains should match themselves
        for chain_a, chain_b in matches.items():
            assert chain_a == chain_b

    def test_match_chains_different_structures(self, multi_chain_structure):
        """Test matching chains between different structures."""
        if multi_chain_structure is None:
            pytest.skip("Could not create multi-chain structure")
        
        # Create a second structure with swapped chain IDs
        structure_b = multi_chain_structure.copy()
        # Swap chain IDs
        mask_a = structure_b.chain_id == "A"
        mask_b = structure_b.chain_id == "B"
        structure_b.chain_id[mask_a] = "C"
        structure_b.chain_id[mask_b] = "A"
        structure_b.chain_id[structure_b.chain_id == "C"] = "B"
        
        matches = match_chains(multi_chain_structure, structure_b)
        assert isinstance(matches, dict)

# Tests for scoring functions
class TestTMScore:
    def test_tm_score_identical_structures(self, sample_structure):
        """Test TM-score between identical structures."""
        if sample_structure is None:
            pytest.skip("Could not load sample structure")
        
        score = tm_score(sample_structure, sample_structure)
        assert isinstance(score, float)
        assert score > 0.99

    def test_tm_score_different_structures(self, sample_structure):
        """Test TM-score between different structures."""
        if sample_structure is None:
            pytest.skip("Could not load sample structure")
        
        half_ref = sample_structure[:len(sample_structure)//2]
        score = tm_score(sample_structure, half_ref)
        assert isinstance(score, float)
        assert 0 <= score <= 1

    @patch('biotite.structure.superimpose_structural_homologs')
    def test_tm_score_exception_handling(self, mock_superimpose, sample_structure):
        """Test TM-score exception handling."""
        if sample_structure is None:
            pytest.skip("Could not load sample structure")
        
        mock_superimpose.side_effect = ValueError("Test error")
        score = tm_score(sample_structure, sample_structure)
        assert score == 0

class TestTMConfusionMatrix:
    def test_generate_tm_confusion_matrix(self, sample_structure):
        """Test TM confusion matrix generation."""
        if sample_structure is None:
            pytest.skip("Could not load sample structure")
        
        proteins = [sample_structure, sample_structure]
        
        with patch('matplotlib.pyplot.show'):
            matrix = generate_tm_confusion_matrix(
                proteins, show_plot=False, return_matrix=True
            )
        
        assert matrix is not None
        assert matrix.shape == (2, 2)
        assert np.allclose(np.diag(matrix), 1.0)  # Diagonal should be 1.0

    def test_generate_tm_confusion_matrix_no_return(self, sample_structure):
        """Test TM confusion matrix without returning matrix."""
        if sample_structure is None:
            pytest.skip("Could not load sample structure")
        
        proteins = [sample_structure]
        
        with patch('matplotlib.pyplot.show'):
            result = generate_tm_confusion_matrix(
                proteins, show_plot=False, return_matrix=False
            )
        
        assert result is None

class TestComputeTMScores:
    @patch('ecstasy.utils.load_structure')
    @patch('ecstasy.utils.get_sequence')
    @patch('ecstasy.utils.tm_score')
    @patch('ecstasy.utils.match_chains')
    @patch('ecstasy.utils.filter_structure_by_chain_id')
    def test_compute_tm_scores_against_gd(self, mock_filter, mock_match, mock_tm, mock_seq, mock_load):
        """Test computing TM scores against ground truth."""
        # Mock return values
        mock_structure = MagicMock()
        mock_load.return_value = mock_structure
        mock_seq.return_value = {"A": "SEQUENCE"}
        mock_tm.return_value = 0.8
        mock_match.return_value = {"A": "A"}
        mock_filter.return_value = mock_structure
        
        test_input = {
            "2": {
                "1ABC": ["/path/to/n2_1ABC_p1.pdb", "/path/to/n2_1ABC_p2.pdb"]
            }
        }
        
        result = compute_tm_scores_against_gd(test_input)
        
        assert isinstance(result, dict)
        assert "2" in result
        assert "1ABC" in result["2"]
        assert 1 in result["2"]["1ABC"]
        assert 2 in result["2"]["1ABC"]

class TestPlotTMScores:
    def test_plot_tm_scores_for_proteins_against_gd(self):
        """Test plotting TM scores."""
        test_data = {
            "protein1": {
                1: {"overall": 0.8, "A": 0.7, "B": 0.9},
                2: {"overall": 0.6, "A": 0.5, "B": 0.7}
            }
        }
        
        with patch('matplotlib.pyplot.show'):
            with patch('matplotlib.pyplot.subplots') as mock_subplots:
                with patch('matplotlib.pyplot.tight_layout'):
                    # Mock the return value to be a figure and axis
                    mock_fig = MagicMock()
                    mock_ax = MagicMock()
                    mock_ax.get_ylim.return_value = (0.0, 1.0)  # Mock y limits
                    mock_ax.boxplot.return_value = MagicMock()
                    mock_subplots.return_value = (mock_fig, mock_ax)
                    
                    # Should not raise an exception
                    plot_tm_scores_for_proteins_against_gd(test_data)

class TestDockQScore:
    @patch('DockQ.DockQ.load_PDB')
    @patch('DockQ.DockQ.run_on_all_native_interfaces')
    def test_dockq_score(self, mock_run, mock_load):
        """Test DockQ score calculation."""
        mock_load.return_value = MagicMock()
        # Fix the mock return value to match the actual DockQ output format
        mock_run.return_value = ({"interface1": {"DockQ": 0.8}, "interface2": {"DockQ": 0.6}}, None)
        
        score = dockq_score("protein_a.pdb", "protein_b.pdb", {"A": "A", "B": "B"})
        
        assert isinstance(score, float)
        assert score == 0.7  # Average of 0.8 and 0.6

# Tests for file writing utilities
class TestFASTAWriting:
    def test_write_fasta_esmfold(self, sample_sequences, temp_dir):
        """Test writing FASTA for ESMFold."""
        write_fasta_esmfold(sample_sequences, temp_dir, "test.fasta")
        
        output_file = os.path.join(temp_dir, "test.fasta")
        assert os.path.exists(output_file)
        
        with open(output_file, 'r') as f:
            content = f.read()
            assert ">protein1" in content
            assert ">protein2" in content
            assert ":" in content  # Chain separator

    def test_write_fasta_boltz(self, sample_sequences, temp_dir):
        """Test writing FASTA for Boltz."""
        write_fasta_boltz(sample_sequences, temp_dir, use_msas=False)
        
        protein1_file = os.path.join(temp_dir, "protein1.fasta")
        protein2_file = os.path.join(temp_dir, "protein2.fasta")
        
        assert os.path.exists(protein1_file)
        assert os.path.exists(protein2_file)
        
        with open(protein1_file, 'r') as f:
            content = f.read()
            assert ">A|protein|empty" in content
            assert ">B|protein|empty" in content

    def test_write_fasta_boltz_with_msas(self, sample_sequences, temp_dir):
        """Test writing FASTA for Boltz with MSAs."""
        write_fasta_boltz(sample_sequences, temp_dir, use_msas=True)
        
        protein1_file = os.path.join(temp_dir, "protein1.fasta")
        
        with open(protein1_file, 'r') as f:
            content = f.read()
            assert ">A|protein" in content
            assert "empty" not in content

class TestFASTACombining:
    def test_combine_fastas_esmfold(self, temp_dir):
        """Test combining FASTA files."""
        # Create test FASTA files
        fasta1_content = ">protein1\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
        fasta2_content = ">protein2\nMTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS"
        
        with open(os.path.join(temp_dir, "file1.fasta"), 'w') as f:
            f.write(fasta1_content)
        with open(os.path.join(temp_dir, "file2.fasta"), 'w') as f:
            f.write(fasta2_content)
        
        combine_fastas_esmfold(temp_dir, "combined.fasta")
        
        combined_file = os.path.join(temp_dir, "combined.fasta")
        assert os.path.exists(combined_file)
        
        with open(combined_file, 'r') as f:
            content = f.read()
            assert ">protein1" in content
            assert ">protein2" in content

    def test_combine_fastas_no_files(self, temp_dir):
        """Test combining when no FASTA files exist."""
        combine_fastas_esmfold(temp_dir)
        # Should not create combined file
        combined_file = os.path.join(temp_dir, "combined.fasta")
        assert not os.path.exists(combined_file)

# Tests for confidence file utilities
class TestConfidenceFiles:
    def test_get_confidence_file_path_boltz(self):
        """Test getting Boltz confidence file path."""
        cif_path = "/path/to/model.cif"
        confidence_path = get_confidence_file_path_boltz(cif_path)
        
        expected = "/path/to/confidence_model.json"
        assert confidence_path == expected

    def test_get_confidence_file_path_colabfold_unrelaxed(self):
        """Test getting ColabFold confidence file path for unrelaxed structure."""
        pdb_path = "/path/to/protein_unrelaxed_rank_001_model_1.pdb"
        confidence_path = get_confidence_file_path_colabfold(pdb_path)
        
        expected = "/path/to/protein_scores_rank_001_model_1.json"
        assert confidence_path == expected

    def test_get_confidence_file_path_colabfold_relaxed(self):
        """Test getting ColabFold confidence file path for relaxed structure."""
        pdb_path = "/path/to/protein_relaxed_rank_001_model_1.pdb"
        confidence_path = get_confidence_file_path_colabfold(pdb_path)
        
        expected = "/path/to/protein_scores_rank_001_model_1.json"
        assert confidence_path == expected

class TestColabFoldCleanup:
    def test_clean_up_colabfold_predictions(self, temp_dir):
        """Test ColabFold predictions cleanup."""
        # Create test files
        test_files = [
            "n4_4YTP_p12.done.txt",
            "n4_4YTP_p12_unrelaxed_rank_001_model_1.pdb",
            "n4_4YTP_p12_scores_rank_001_model_1.json",
            "n4_4YTP_p34.done.txt",
            "n4_4YTP_p34_unrelaxed_rank_001_model_1.pdb"
        ]
        
        for filename in test_files:
            with open(os.path.join(temp_dir, filename), 'w') as f:
                f.write("test content")
        
        clean_up_colabfold_predictions(temp_dir)
        
        # The function creates directories based on the .done.txt file stems
        # .stem on "n4_4YTP_p12.done.txt" gives "n4_4YTP_p12.done"
        assert os.path.exists(os.path.join(temp_dir, "n4_4YTP_p12.done"))
        assert os.path.exists(os.path.join(temp_dir, "n4_4YTP_p34.done"))
        
        # Check that .done.txt files remain in the original directory (they are not moved)
        assert os.path.exists(os.path.join(temp_dir, "n4_4YTP_p12.done.txt"))
        assert os.path.exists(os.path.join(temp_dir, "n4_4YTP_p34.done.txt"))

    def test_clean_up_colabfold_predictions_no_done_files(self, temp_dir):
        """Test cleanup when no .done.txt files exist."""
        # Create some random files
        with open(os.path.join(temp_dir, "random_file.pdb"), 'w') as f:
            f.write("test")
        
        # Should not raise an exception
        clean_up_colabfold_predictions(temp_dir)

    def test_clean_up_colabfold_predictions_invalid_dir(self):
        """Test cleanup with invalid directory."""
        with pytest.raises(FileNotFoundError):
            clean_up_colabfold_predictions("/nonexistent/directory")

class TestProteinIPTMFinder:
    def test_find_max_protein_iptm_boltz(self, temp_dir):
        """Test finding max protein iPTM from Boltz confidence files."""
        # Create test confidence files
        confidence_data1 = {"protein_iptm": 0.8, "other_metric": 0.5}
        confidence_data2 = {"protein_iptm": 0.9, "other_metric": 0.7}
        confidence_data3 = {"other_metric": 0.6}  # No protein_iptm
        
        with open(os.path.join(temp_dir, "confidence_model1.json"), 'w') as f:
            json.dump(confidence_data1, f)
        with open(os.path.join(temp_dir, "confidence_model2.json"), 'w') as f:
            json.dump(confidence_data2, f)
        with open(os.path.join(temp_dir, "confidence_model3.json"), 'w') as f:
            json.dump(confidence_data3, f)
        
        filename, score = find_max_protein_iptm_boltz(temp_dir)
        
        assert filename == "confidence_model2.json"
        assert score == 0.9

    def test_find_max_protein_iptm_boltz_no_files(self, temp_dir):
        """Test finding max protein iPTM when no confidence files exist."""
        filename, score = find_max_protein_iptm_boltz(temp_dir)
        
        assert filename is None
        assert score is None

    def test_find_max_protein_iptm_boltz_no_iptm_values(self, temp_dir):
        """Test finding max protein iPTM when no files have iPTM values."""
        confidence_data = {"other_metric": 0.6}
        
        with open(os.path.join(temp_dir, "confidence_model1.json"), 'w') as f:
            json.dump(confidence_data, f)
        
        filename, score = find_max_protein_iptm_boltz(temp_dir)
        
        assert filename is None
        assert score is None

# Integration tests
class TestIntegration:
    def test_full_workflow_with_sample_structure(self, sample_structure, temp_dir):
        """Test a full workflow using sample structure."""
        if sample_structure is None:
            pytest.skip("Could not load sample structure")
        
        # Extract sequences
        sequences = get_sequence(sample_structure)
        assert len(sequences) > 0
        
        # Write FASTA
        seq_dict = {"test_protein": sequences}
        write_fasta_esmfold(seq_dict, temp_dir, "test.fasta")
        
        # Check file was created
        fasta_file = os.path.join(temp_dir, "test.fasta")
        assert os.path.exists(fasta_file)
        
        # Calculate TM score with itself
        tm = tm_score(sample_structure, sample_structure)
        assert tm > 0.99

if __name__ == "__main__":
    pytest.main([__file__]) 