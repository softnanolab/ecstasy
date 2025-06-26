#!/bin/bash

# Print step messages in blue
BLUE='\033[1;34m'
NC='\033[0m' # No Color

# Initialize conda in this shell session
echo -e "${BLUE}Initializing conda in this shell session...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"

# Define absolute path for the environment
echo -e "${BLUE}Setting environment path...${NC}"
ENV_PATH="./envs/esmfold_py39"

# Create the environment if it doesn't exist
echo -e "${BLUE}Creating conda environment at $ENV_PATH...${NC}"
conda create -y -p "$ENV_PATH"

# Activate the environment
echo -e "${BLUE}Activating environment at $ENV_PATH...${NC}"
conda activate "$ENV_PATH"

# Step 2: Install PyTorch and CUDA
echo -e "${BLUE}Installing PyTorch, torchvision, torchaudio, and CUDA...${NC}"
conda install -y python==3.9 pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# Step 3: Install additional dependencies
echo -e "${BLUE}Installing cudatoolkit-dev, gxx, cudnn, R, and devtools...${NC}"
conda install -y cudatoolkit-dev 'gxx>=6.0.0,<12.0' cudnn r-base r-devtools -c defaults -c conda-forge -c pytorch

# Step 4: Update environment with YAML
echo -e "${BLUE}Updating environment with esmfold_py39.yaml...${NC}"
conda env update -p "$ENV_PATH" --file /home/jovyan/workspace/ecstasy/scripts/install/esmfold_py39.yaml

# Step 5: Install fair-esm[esmfold]
echo -e "${BLUE}Installing fair-esm[esmfold]...${NC}"
pip install "fair-esm[esmfold]"

# Step 6: Install OpenFold from specific commit
echo -e "${BLUE}Installing OpenFold from aqlaboratory's repo...${NC}"
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

# Step 7: Install pip 24.x
echo -e "${BLUE}Upgrading pip to 24.x...${NC}"
pip install 'pip==24.0.*'

# Step 8: Install pytorch_lightning 1.7.7
echo -e "${BLUE}Installing pytorch_lightning==1.7.7...${NC}"
pip install pytorch_lightning==1.7.7

# Step 9: Install torchmetrics 0.11.4
echo -e "${BLUE}Installing torchmetrics==0.11.4...${NC}"
pip install torchmetrics==0.11.4

# Step 10: Install biotite
echo -e "${BLUE}Installing biotite...${NC}"
pip install biotite

echo -e "${BLUE}esmfold_py39 environment setup complete!${NC}" 