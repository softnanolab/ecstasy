# Print step messages in blue
BLUE='\033[1;34m'
NC='\033[0m' # No Color

# Initialize conda in this shell session
echo -e "${BLUE}Initializing conda in this shell session...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"

# Define absolute path for the environment
echo -e "${BLUE}Setting environment path...${NC}"
ENV_PATH="./envs/ecstasy"

echo -e "${BLUE}Checking if envs directory exists...${NC}"
if [ ! -d "./envs" ]; then
  echo -e "${BLUE}envs directory does not exist. Creating...${NC}"
  mkdir -p "./envs"
fi

# Create the environment if it doesn't exist
echo -e "${BLUE}Creating conda environment if it does not exist...${NC}"
conda create -p "$ENV_PATH" python=3.12 -y

conda activate "$ENV_PATH" && pip install -e .