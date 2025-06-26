# Print step messages in blue
BLUE='\033[1;34m'
NC='\033[0m' # No Color

# Initialize conda in this shell session
echo -e "${BLUE}Initializing conda in this shell session...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"

# Define absolute path for the environment
echo -e "${BLUE}Setting environment path...${NC}"
ENV_PATH="./envs/esmfold"

echo -e "${BLUE}Checking if envs directory exists...${NC}"
if [ ! -d "./envs" ]; then
  echo -e "${BLUE}envs directory does not exist. Creating...${NC}"
  mkdir -p "./envs"
fi

# Create the environment if it doesn't exist
echo -e "${BLUE}Creating conda environment if it does not exist...${NC}"
conda env create -f ./scripts/install/esmfold.yaml -p "$ENV_PATH"

# Install ESMFold from Softnanolab's fork
echo -e "${BLUE}Installing ESMFold from Softnanolab's fork...${NC}"
conda run -p "$ENV_PATH" pip install -e ./modules/esm

# Build the openfold extension
echo -e "${BLUE}Setting compiler environment variables...${NC}"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

echo -e "${BLUE}Installing OpenFold from softnanolab's esmfold_compat branch...${NC}"
conda run -p "$ENV_PATH" pip install --no-cache-dir -e ./modules/openfold