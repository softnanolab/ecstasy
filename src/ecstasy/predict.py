# Standardized imports
import os
import subprocess

import fire


def _validate_args(model: str, input_path: str, output_dir: str):
    """Validates input arguments for prediction.

    Args:
        model (str): Model name.
        input_path (str): Path to input fasta file or directory.
        output_dir (str): Output directory.

    Raises:
        FileNotFoundError: If input path does not exist.
        ValueError: If input file/dir is not valid or model is unsupported.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path {input_path} does not exist.")
    if os.path.isfile(input_path):
        if not (input_path.endswith(".fasta") or input_path.endswith(".fa")):
            raise ValueError("Input file must end with .fasta or .fa")
    elif os.path.isdir(input_path):
        if not any(
            f.endswith(".fasta") or f.endswith(".fa") for f in os.listdir(input_path)
        ):
            raise ValueError(
                "Input directory must contain at least one .fasta or .fa file"
            )
    else:
        raise ValueError("Input path must be a file or directory")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if model != "boltz":
        raise ValueError("Currently only 'boltz' model is supported.")


def predict(model: str, input_path: str, output_dir: str):
    """Predicts the structure of a protein using the specified model.

    Args:
        model (str): Model to use for prediction. Only 'boltz' is supported.
        input_path (str): Path to input fasta file or directory.
        output_dir (str): Directory to write output files.
    """
    _validate_args(model, input_path, output_dir)
    if model == "boltz":
        _run_prediction_boltz(input_path, output_dir)
    else:
        raise ValueError(f"Model {model} is not supported.")


def _run_prediction_boltz(input_path: str, output_dir: str):
    """
    Run the prediction using Boltz and stream the output in real-time.

    Args:
        input_path (str): The path to the input fasta file or directory.
        output_dir (str): The directory to write the output files.
    """
    cmd = (
        'source "$(conda info --base)/etc/profile.d/conda.sh" && '
        f"conda activate ./envs/boltz && boltz predict "
        f"{input_path} --out_dir {output_dir}"
    )
    print(f"Running: {cmd}")

    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        executable="/bin/bash",
    )

    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            print(line, end="")

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"Prediction failed with exit code {return_code}")


if __name__ == "__main__":
    # To run in the boltz conda environment:
    # python src/ecstasy/predict.py --model=boltz --input_path=your.fasta --output_dir=outdir
    fire.Fire(predict)
