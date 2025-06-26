import fire
import os
import sys
import subprocess

def _validate_args(model: str, input_path: str, output_dir: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path {input_path} does not exist.")
    if not os.path.isdir(output_dir):
        print(f"Output directory {output_dir} does not exist. Creating it.")
        os.makedirs(output_dir, exist_ok=True)
    if model != "boltz":
        raise ValueError("Currently only 'boltz' model is supported.")

def predict(model: str, input_path: str, output_dir: str):
    """
    Predict the structure of a protein using the specified model.

    Args:
        model (str): The model to use for prediction. Currently only 'boltz' is supported.
        input_path (str): The path to the input fasta file or directory.
        output_dir (str): The directory to write the output files.
    """
    _validate_args(model, input_path, output_dir)
    if model == "boltz":
        _run_prediction_boltz(input_path, output_dir)
    else:
        raise ValueError(f"Model {model} is not supported.")

def _run_prediction_boltz(input_path: str, output_dir: str):
    """
    Run the prediction using Boltz.

    Args:
        input_path (str): The path to the input fasta file or directory.
        output_dir (str): The directory to write the output files.
    """
    cmd = [
        f"conda run -p ./envs/boltz boltz predict {input_path} --out_dir {output_dir}",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Prediction failed with exit code {result.returncode}")

if __name__ == "__main__":
    # To run in the boltz conda environment:
    # python src/ecstasy/predict.py --model=boltz --input_path=your.fasta --output_dir=outdir
    fire.Fire(predict)
