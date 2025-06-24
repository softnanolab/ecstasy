import json
import itertools
from pathlib import Path
import requests
import fire
import random


SRC_DIR = Path(__file__).parent.parent
BASE_DIR = SRC_DIR.parent


class ChainPermutationGenerator:
    def __init__(
        self,
        output_dir: str,
        pdb_features_path: str = f"{BASE_DIR}/resources/pdb_features.json",
    ):
        """Initialize the generator with paths for input and output.

        Args:
            pdb_features_path: Path to the JSON file containing PDB features
            output_dir: Directory where the permutation FASTA files will be saved
        """
        self.pdb_features_path = pdb_features_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load PDB features
        with open(self.pdb_features_path, "r") as f:
            self.all_info = json.load(f)

    def get_chain_sequence(self, pdb_id: str, chain_id: str) -> str:
        """Get the sequence for a specific chain from the PDB API."""
        url = f"https://www.rcsb.org/fasta/entry/{pdb_id}/download"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch FASTA for {pdb_id}")

        sequences = {}
        for block in response.text.split(">"):
            if not block.strip():
                continue
            lines = block.strip().split("\n")
            header = lines[0]
            seq = "".join(lines[1:])

            # Extract chain ID from header (format: pdb_id|Chain X|...)
            chain = header.split("|")[1].split()[1] if "|" in header else None
            if chain:
                sequences[chain] = seq

        if chain_id not in sequences:
            raise Exception(f"Chain {chain_id} not found in {pdb_id}")
        return sequences[chain_id]

    def generate_permutations(self, protein_info: dict) -> None:
        """Generate FASTA files with chain permutations for a PDB structure.

        Args:
            protein_info: Dictionary containing protein information including PDB ID and chain lengths
        """
        pdb_id = protein_info["pdb_id"]
        chain_ids = list(protein_info["chain_lengths"].keys())

        permutations = list(itertools.permutations(chain_ids))

        try:
            for i, perm in enumerate(permutations, 1):
                fasta_content = [
                    f">{chain_id}|protein|empty\n{self.get_chain_sequence(pdb_id, chain_id)}"
                    for chain_id in perm
                ]

                output_file = self.output_dir / f"n{len(chain_ids)}_{pdb_id}_p{i}.fasta"
                with open(output_file, "w") as f:
                    f.write("\n".join(fasta_content))

            print(f"{i}. Generated {len(permutations)} permutations for {pdb_id}")
            return True
        except Exception as e:
            print(f"Error processing {pdb_id}: {str(e)}")
            return False

    def main(
        self,
        total_num_chains: list[int] = [2, 3, 4],
        num_samples: int = 10,
        min_chain_length: int = 100,
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
                    if self.generate_permutations(protein):
                        num_success += 1

            print(f"Generated {num_success} permutations for {num_chains} chains")


if __name__ == "__main__":
    fire.Fire(ChainPermutationGenerator)
