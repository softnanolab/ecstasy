import os
import json
import itertools
from pathlib import Path
import fire
import random
from ecstasy import utils
import pandas as pd
from typing import Optional
import datetime

SRC_DIR = Path(__file__).parent.parent.parent


class ChainPermutationGenerator:

    def __init__(
        self,
        pdb_features_path: str,
        output_dir: str,
    ):
        """Initialize the generator with paths for input and output.

        Args:
            pdb_features: Path to the JSON file containing PDB features
            output_dir: Directory where the permutation FASTA files will be saved
        """

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load PDB features
        with open(pdb_features_path, "r") as f:
            self.pdb_features = json.load(f)

        # Load SAbDab database
        with open(SRC_DIR / "resources/sabdab_summary_all.tsv", "r") as f:
            self.sabdab_db = pd.read_csv(f, sep="\t")

    def _filter_unique_chain_lengths(self, protein: dict) -> bool:
        """Filter to ensure all chains have unique lengths.

        Args:
            protein (dict): Protein dictionary containing chain lengths

        Returns:
            bool: True if all chains have unique lengths, False otherwise
        """
        chain_lengths = protein["chain_lengths"].values()
        return len(set(chain_lengths)) == len(chain_lengths)

    def _filter_not_in_sabdab(self, protein: dict) -> bool:
        """Filter to exclude proteins in SAbDab database.

        Args:
            protein (dict): Protein dictionary containing PDB ID

        Returns:
            bool: True if protein is not in SAbDab database, False otherwise
        """
        pdb_id = protein["pdb_id"].lower()
        is_in_sabdab = pdb_id in self.sabdab_db["pdb"].str.lower().values

        return not is_in_sabdab

    def _filter_min_chain_length(self, protein: dict, min_chain_length: int) -> bool:
        """Filter to ensure all chains have lengths greater than min_chain_length.

        Args:
            protein (dict): Protein dictionary containing chain lengths
            min_chain_length (int): Minimum chain length to filter by

        Returns:
            bool: True if all chains have lengths greater than min_chain_length, False otherwise
        """
        return all(
            length > min_chain_length for length in protein["chain_lengths"].values()
        )

    def _filter_max_total_length(self, protein: dict, max_total_length: int) -> bool:
        """Filter to ensure the protein has a total chain length less than max_total_length.

        Args:
            protein (dict): Protein dictionary containing chain lengths
            max_total_length (int): Maximum total chain length to filter by

        Returns:
            bool: True if protein has a total chain length less than max_total_length, False otherwise
        """
        return sum(protein["chain_lengths"].values()) <= max_total_length

    def _filter_num_chains(self, protein: dict, num_chains: int) -> bool:
        """Filter to ensure the protein has the correct number of chains.

        Args:
            protein (dict): Protein dictionary containing chain lengths
            num_chains (int): Number of chains to filter by

        Returns:
            bool: True if protein has the correct number of chains, False otherwise
        """
        return protein["num_chains"] == num_chains

    def _filter_date_cutoff(
        self, protein: dict, date_cutoff_before: str, date_cutoff_after: str
    ) -> bool:
        """Filter to ensure the protein was deposited after date_cutoff_after and before date_cutoff_before.

        Args:
            protein (dict): Protein dictionary containing PDB ID
            date_cutoff_before (str): Date cutoff before
            date_cutoff_after (str): Date cutoff after

        Returns:
            bool: True if protein was deposited after date_cutoff_after and before date_cutoff_before, False otherwise
        """
        protein_date = datetime.datetime.fromtimestamp(protein["date"])
        if date_cutoff_after:
            date_cutoff_after = datetime.datetime.strptime(
                date_cutoff_after, "%Y-%m-%d"
            )
        else:
            # set an arbitrary date before the first protein was deposited
            date_cutoff_after = datetime.datetime.strptime("1900-01-01", "%Y-%m-%d")
        if date_cutoff_before:
            date_cutoff_before = datetime.datetime.strptime(
                date_cutoff_before, "%Y-%m-%d"
            )
        else:
            # set an arbitrary date after the last protein was deposited: today
            date_cutoff_before = datetime.datetime.now()
        return date_cutoff_after < protein_date < date_cutoff_before

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
                chr(j + 65): protein_sequences[chain_id]
                for j, chain_id in enumerate(perm)
            }

            # Write FASTA files
            utils.write_fasta_esmfold(
                sequences,
                os.path.join(self.output_dir, "esmfold"),
                file_name=f"{pdb_id}.fasta",
            )
            utils.write_fasta_boltz(
                sequences, os.path.join(self.output_dir, "boltz"), use_msas=False
            )
            utils.write_fasta_boltz(
                sequences, os.path.join(self.output_dir, "boltz_msas"), use_msas=True
            )

        print(f"Generated {len(permutations)} permutations for {pdb_id}")

    def main(
        self,
        total_num_chains: list[int],
        num_samples: int,
        filter_abs: Optional[bool],
        filter_min_chain_length: Optional[int],
        filter_max_total_length: Optional[int],
        filter_unique_chain_lengths: Optional[bool],
        filter_date_cutoff_before: Optional[str],
        filter_date_cutoff_after: Optional[str],
    ):
        """
        Generate permutations for proteins with unique chain lengths and lengths greater than min_chain_length.
        Args:
            total_num_chains (list[int]): List of total number of chains to generate permutations for
            num_samples (int): Number of proteins to sample for each chain number
            filter_abs (bool): Whether to filter Antibodies from SAbDab. (default: False)
            filter_min_chain_length (int): Minimum chain length to generate permutations for
            filter_max_total_length (int): Maximum total chain length to generate permutations for
            filter_unique_chain_lengths (bool): Whether to filter proteins with non-unique chain lengths. (default: True)
            filter_date_cutoff_before (str): Date cutoff before to generate permutations for. In format "2025-01-01"
            filter_date_cutoff_after (str): Date cutoff after to generate permutations for. In format "2025-01-01"
        """

        # Loop through each chain number and sample proteins
        for num_chains in total_num_chains:

            num_success = 0

            while num_success < num_samples:
                protein = random.choice(self.pdb_features)

                # Apply all filters to the protein
                if all(
                    [
                        # Filter: Remove Proteins with incorrect number of chains
                        self._filter_num_chains(protein, num_chains),
                        # Filter: Remove Proteins in SAbDab
                        self._filter_not_in_sabdab(protein) if filter_abs else True,
                        # Filter: Remove Proteins with total chain length greater than max_total_length
                        (
                            self._filter_max_total_length(
                                protein, max_total_length=filter_max_total_length
                            )
                            if filter_max_total_length is not None
                            else True
                        ),
                        # Filter: Remove Proteins with non-unique chain lengths
                        (
                            self._filter_unique_chain_lengths(protein)
                            if filter_unique_chain_lengths
                            else True
                        ),
                        # Filter: Remove Proteins with every chain length less than min_chain_length
                        (
                            self._filter_min_chain_length(
                                protein, min_chain_length=filter_min_chain_length
                            )
                            if filter_min_chain_length is not None
                            else True
                        ),
                        # Filter: Date Cutoff
                        self._filter_date_cutoff(
                            protein,
                            date_cutoff_before=filter_date_cutoff_before,
                            date_cutoff_after=filter_date_cutoff_after,
                        ),
                    ]
                ):
                    self.generate_permutations(protein["pdb_id"])
                    num_success += 1

            print(
                "-" * 25
                + f"Generated {num_success} permutations for {num_chains} chains"
                + "-" * 25
            )


def run_generate_permutations(
    pdb_features: str,
    output_dir: str,
    num_samples: int,
    total_num_chains: list[int] = [2, 3, 4],
    filter_min_chain_length: Optional[int] = None,
    filter_max_total_length: Optional[int] = None,
    filter_unique_chain_lengths: Optional[bool] = None,
    filter_date_cutoff_before: Optional[str] = None,
    filter_date_cutoff_after: Optional[str] = None,
    filter_abs: Optional[bool] = None,
):
    """
    Main function to generate chain permutations.

    Args:
        pdb_features (str): Path to the JSON file containing PDB features
        output_dir (str): Directory where the permutation FASTA files will be saved
        total_num_chains (list[int]): List of total number of chains to generate permutations for
            (default: [2, 3, 4])
        num_samples (int): Number of proteins to sample for each chain number
            (default: 10)
        filter_min_chain_length (int): Minimum chain length to generate permutations for
            (default: None)
        filter_max_total_length (int): Maximum total chain length to generate permutations for
            (default: None)
        filter_unique_chain_lengths (bool): Whether to filter proteins with non-unique chain lengths.
            (default: None)
        filter_date_cutoff_before (str): Date cutoff before to generate permutations for. In format "2025-01-01"
            (default: None)
        filter_date_cutoff_after (str): Date cutoff after to generate permutations for. In format "2025-01-01"
            (default: None)
        filter_abs (bool): Whether to filter Antibodies from SAbDab
            (default: None)
    """
    generator = ChainPermutationGenerator(pdb_features, output_dir)

    # Log the filters
    print("Using the following filters:")
    print(f"filter_min_chain_length: {filter_min_chain_length}")
    print(f"filter_max_total_length: {filter_max_total_length}")
    print(f"filter_unique_chain_lengths: {filter_unique_chain_lengths}")
    print(f"filter_date_cutoff_before: {filter_date_cutoff_before}")
    print(f"filter_date_cutoff_after: {filter_date_cutoff_after}")
    print(f"filter_abs: {filter_abs}")

    generator.main(
        total_num_chains=total_num_chains,
        num_samples=num_samples,
        filter_min_chain_length=filter_min_chain_length,
        filter_max_total_length=filter_max_total_length,
        filter_unique_chain_lengths=filter_unique_chain_lengths,
        filter_date_cutoff_before=filter_date_cutoff_before,
        filter_date_cutoff_after=filter_date_cutoff_after,
        filter_abs=filter_abs,
    )


if __name__ == "__main__":
    fire.Fire(run_generate_permutations)
