# Initialize conda in this shell session
source "$(conda info --base)/etc/profile.d/conda.sh"

# Define absolute path for the environment
ENV_PATH="./envs/esmfold"

# Create the environment if it doesn't exist
conda env create -f ./scripts/install/esmfold.yaml -p "$ENV_PATH"

# Build the openfold extension
conda run -p "$ENV_PATH" python ./modules/openfold build_extension.py