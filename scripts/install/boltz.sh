#!/bin/bash

# Initialize conda in this shell session
source "$(conda info --base)/etc/profile.d/conda.sh"

# Define absolute path for the environment
ENV_PATH="./envs/boltz"

# Create the environment if it doesn't exist
conda create -p "$ENV_PATH" --no-default-packages -y

# Install the package using conda run
conda run -p "$ENV_PATH" pip install -e ./modules/boltz
