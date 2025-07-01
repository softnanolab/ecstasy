import os
import json
import itertools
from pathlib import Path
import requests
import fire
import random
from ecstasy import utils


SRC_DIR = Path(__file__).parent.parent
BASE_DIR = SRC_DIR.parent


class ChainPermutationGenerator:
    def __init__(
        self,
        pdb_features: str,
        output_dir: str,
    ):
        """Initialize the generator with paths for input and output.

        Args:
            pdb_features: Path to the JSON file containing PDB features
            output_dir: Directory where the permutation FASTA files will be saved
        """
        self.pdb_features = pdb_features
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load PDB features
        with open(self.pdb_features, "r") as f:
            self.all_info = json.load(f)

    def generate_permutations(self, pdb_id: str) -> bool:
        """Generate FASTA files with chain permutations for a PDB structure.

        Args:
            pdb_id: PDB ID or CIF file path
        """
        # obtain a dict of chain id and sequences
        protein_sequences = utils.get_sequence(utils.load_structure(pdb_id))

        chain_ids = list(protein_sequences.keys())
        permutations = list(itertools.permutations(chain_ids))

        sequences = {}
        for i, perm in enumerate(permutations, 1):
            sequences[f"n{len(chain_ids)}_{pdb_id}_p{i}"] = {
                chr(j+1 + 64): protein_sequences[chain_id] for j, chain_id in enumerate(perm)
            }
            
            utils.write_fasta_esmfold(sequences, os.path.join(self.output_dir, "esmfold"), file_name=f"{pdb_id}.fasta") 
            utils.write_fasta_boltz(sequences, os.path.join(self.output_dir, "boltz"))

        print(f"{i}. Generated {len(permutations)} permutations for {pdb_id}")
        return True

    def main(
        self,
        total_num_chains: list[int],
        num_samples: int,
        min_chain_length: int,
    ):
        """
        Generate permutations for proteins with unique chain lengths and lengths greater than min_chain_length.
        Args:
            total_num_chains: List of total number of chains to generate permutations for
            num_samples: Number of proteins to sample for each chain number
            min_chain_length: Minimum chain length to generate permutations for
        """
        # Loop through each chain number and sample proteins
        for num_chains in total_num_chains:

            num_success = 0

            while num_success < num_samples:
                protein = random.choice(self.all_info)
                # check if all chains have unique lengths > min_chain_length
                if (
                    protein["num_chains"] == num_chains
                    and len(set(protein["chain_lengths"].values()))
                    == len(protein["chain_lengths"])
                    and all(
                        length > min_chain_length
                        for length in protein["chain_lengths"].values()
                    )
                ):
                    if self.generate_permutations(protein["pdb_id"]):
                        num_success += 1

            print(f"Generated {num_success} permutations for {num_chains} chains")


def run_generate_permutations(
    pdb_features: str,
    output_dir: str,
    total_num_chains: list[int] = [2, 3, 4],
    num_samples: int = 10,
    min_chain_length: int = 100,
):
    """
    Main function to generate chain permutations.
    
    Args:
        pdb_features: Path to the JSON file containing PDB features
        output_dir: Directory where the permutation FASTA files will be saved
        total_num_chains: List of total number of chains to generate permutations for
        num_samples: Number of proteins to sample for each chain number
        min_chain_length: Minimum chain length to generate permutations for
    """
    generator = ChainPermutationGenerator(pdb_features, output_dir)
    generator.main(total_num_chains, num_samples, min_chain_length)


if __name__ == "__main__":
    fire.Fire(run_generate_permutations)
